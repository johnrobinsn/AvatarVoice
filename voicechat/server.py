"""
Voice-chat demo WebRTC server — full session wiring (Step 9f/9g).

One session at a time (v1). On POST /offer:
  - Accept browser mic+webcam tracks → ring buffers
  - Start VAD/turn-detect task on mic ring
  - Start avatar worker thread (uses the FaceCropper & AvatarForcing singletons)
  - Mount outbound tracks: avatar video (from worker queue) + avatar audio (from pcm ring)
  - On turn-complete: run STT → reply_pipeline(LLM→Phonon→wav2vec→rings)
  - On start-of-speech while reply in flight: cancel reply_task (barge-in)

Usage:
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=1 python voicechat/server.py --port 8082
Then open http://<host>:8082 in a browser.
"""
from __future__ import annotations

import os, sys, asyncio, time, logging, argparse, threading, queue, ctypes, json
from pathlib import Path
from fractions import Fraction

ROOT = Path(__file__).resolve().parent.parent
# Upstream AvatarForcing code is vendored as a git submodule. We insert its
# path so the `inference` and `models.avatarforcing.*` imports below resolve
# against the submodule's checkout.
AF_ROOT = ROOT / "avatarforcing"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".inductor_cache"))
# Expandable segments — tolerates the per-specialization VRAM allocations that
# dynamo retains across recompiles. Without this we OOM after ~15 blocks.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Preload CUDA 12 libs for faster-whisper (ctranslate2 is CUDA-12-built)
_venv = ROOT / ".venv/lib/python3.11/site-packages"
for _p in [_venv / "nvidia/cublas/lib/libcublas.so.12",
           _venv / "nvidia/cudnn/lib/libcudnn.so.9"]:
    if _p.exists():
        try: ctypes.CDLL(str(_p), mode=ctypes.RTLD_GLOBAL)
        except OSError: pass

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AF_ROOT))   # upstream AvatarForcing (inference.py, models/)

import av
import numpy as np
import torch
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, VideoStreamTrack, RTCConfiguration
from aiortc.mediastreams import MediaStreamError
from scipy import signal as _scipy_signal
from omegaconf import OmegaConf

from inference import InferenceAgent
from voicechat.rings import AudioRing, VideoRing, FeatureRing
from voicechat.face_crop import FaceCropper
from voicechat.vad import VADWorker, TurnDetectorConfig
from voicechat.stt import WhisperSTT, ParakeetSTT
from voicechat.llm import VLLMClient
from voicechat.tts import PhononTTS, resample_24k_to_48k
from voicechat.reply_pipeline import reply_pipeline
from voicechat.avatar_worker import AvatarWorker

# Log to logs/server.log, truncated fresh on every server start. We set up
# the FileHandler explicitly (no basicConfig StreamHandler) so that launchers
# which do "2>&1 > logs/server.log" don't double-log each line.
_LOG_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_file_handler = logging.FileHandler(_LOG_DIR / "server.log", mode="w")
_file_handler.setFormatter(logging.Formatter(_LOG_FMT))
_root = logging.getLogger()
_root.setLevel(logging.INFO)
_root.addHandler(_file_handler)
log = logging.getLogger("af-voice")


# ---------- Shared singletons (loaded once per process) ----------
AGENT: "InferenceAgent" = None          # AvatarForcing
STT: WhisperSTT = None
LLM: VLLMClient = None
TTS: PhononTTS = None
FACE_CROPPER: FaceCropper = None
AVATAR_REF_TENSOR: torch.Tensor = None  # [1, 3, 512, 512], preprocessed
AVATAR_WORKER: AvatarWorker = None      # persistent worker thread


def _extend_rope_tables(flow_transformer, new_max_len: int = 65536):
    """Extend RoPE freqs_cis tables so long continuous sessions don't overflow."""
    from models.avatarforcing.flow_transformer import precompute_freqs_cis
    replaced = 0
    for name, mod in flow_transformer.named_modules():
        if "freqs_cis" in getattr(mod, "_buffers", {}):
            old = mod._buffers["freqs_cis"]
            # Post real-valued rewrite: freqs_cis is [end, head_dim]. Previously
            # it was complex [end, head_dim/2]; handle either for safety.
            head_dim = old.shape[-1] * 2 if old.is_complex() else old.shape[-1]
            new_table = precompute_freqs_cis(head_dim, new_max_len).to(old.device)
            mod.register_buffer("freqs_cis", new_table, persistent=False)
            replaced += 1
    return replaced


def init_singletons(avatar_ref_path: str = None, single_gpu: bool = False):
    global AGENT, STT, LLM, TTS, FACE_CROPPER, AVATAR_REF_TENSOR
    if AGENT is not None:
        return

    log.info("loading AvatarForcing")
    class Opt: pass
    opt = Opt()
    cfg = OmegaConf.load(str(AF_ROOT / "configs/inference.yaml"))
    for k, v in cfg.items(): setattr(opt, k, v)
    opt.infer_config = str(AF_ROOT / "configs/inference.yaml")
    opt.mae_ckpt_path = str(AF_ROOT / "pretrained_dir/motion_autoencoder.pth")
    opt.ckpt_path = str(AF_ROOT / "pretrained_dir/flow_transformer.pth")
    # Config file has these as relative paths assuming CWD == AvatarForcing
    # repo root; the submodule layout breaks that assumption, so re-root them.
    opt.wav2vec_model_path = str(AF_ROOT / opt.wav2vec_model_path)
    opt.pretrained_dir = str(AF_ROOT / opt.pretrained_dir)
    # GPU layout:
    #   single (default — or if only 1 card is visible, or AF_SINGLE_GPU=1):
    #     - cuda:0 = AvatarForcing + STT + FaceCropper (shares with vLLM)
    #   dual (opt-in via --dual-gpu or AF_SINGLE_GPU=0):
    #     - cuda:1 = AvatarForcing (GPU-heavy, isolated for throughput)
    #     - cuda:0 = STT, FaceCropper, vLLM (external process)
    import torch as _torch
    force_single = single_gpu or os.environ.get("AF_SINGLE_GPU", "0") == "1"
    if force_single or _torch.cuda.device_count() < 2:
        if force_single and _torch.cuda.device_count() >= 2:
            log.info("single-GPU mode forced — avatar + aux on cuda:0 (second card unused)")
        elif _torch.cuda.device_count() < 2:
            log.warning(f"only {_torch.cuda.device_count()} GPU visible — avatar + aux will share it")
        opt.rank = 0
        aux_device = "cuda:0" if _torch.cuda.device_count() >= 2 else "cuda"
    else:
        opt.rank = 1
        aux_device = "cuda:0"
    opt.ngpus = 1
    opt.result_dir = str(ROOT / "results/voicechat")
    opt.compile = True
    opt.warmup = 0
    os.makedirs(opt.result_dir, exist_ok=True)
    log.info(f"AvatarForcing on cuda:{opt.rank}")
    AGENT = InferenceAgent(opt)
    n = _extend_rope_tables(AGENT.G.flow_transformer, 65536)
    log.info(f"extended {n} RoPE tables to max_seq_len=65536")

    # FaceCropper intentionally disabled: with u_cfg_scale=0 the user-motion
    # CFG branch is zeroed out, so the webcam signal never affects the
    # avatar. Keeping cuda:0 free of concurrent GPU work with the cuda:1
    # worker also seems to eliminate the lingering CUDA illegal-memory-
    # access race. Re-enable if we switch to a dialog/user-mirroring mode.
    FACE_CROPPER = None

    # AF_STT: "parakeet" (default) or "whisper".
    # Parakeet (via onnx-asr) loads in ~1 s vs Whisper's ~15 s and matches
    # the Mac/MLX backend's STT weights (same TDT-0.6b-v2). English-only.
    #
    # Parakeet is loaded LAZILY here (after the avatar worker warmup) because
    # onnxruntime-gpu's CUDA context interferes with torch's AOTI/compile CUDA
    # graph capture — co-loading them adds ~11 s to avatar warmup. Whisper
    # uses ctranslate2 which doesn't have the same contention.
    stt_backend = os.environ.get("AF_STT", "parakeet").lower()
    _load_parakeet_later = False
    if stt_backend == "parakeet":
        _load_parakeet_later = True
    else:
        log.info(f"loading Whisper STT on {aux_device}")
        STT = WhisperSTT(device=aux_device)

    log.info("loading Phonon TTS")
    TTS = PhononTTS()

    log.info("creating vLLM client")
    LLM = VLLMClient(model=os.environ.get("AF_LLM_MODEL", "cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit"))

    # Preprocess the avatar reference portrait once
    ref_path = avatar_ref_path or str(ROOT / "data/rumi.jpg")
    log.info(f"preprocessing avatar ref: {ref_path}")
    sample = AGENT.data_processor.preprocess(
        avatar_ref_path   = ref_path,
        avatar_audio_path = str(AF_ROOT / "data/avatar.wav"),
        user_audio_path   = str(AF_ROOT / "data/user.wav"),
        user_video_path   = str(AF_ROOT / "data/user/"),
    )
    AVATAR_REF_TENSOR = sample['avatar_ref']
    log.info(f"singletons ready (avatar_ref shape={tuple(AVATAR_REF_TENSOR.shape)})")

    # Start the persistent avatar worker (runs torch.compile warmup once).
    global AVATAR_WORKER
    log.info("starting persistent avatar worker (warmup will run now)")
    AVATAR_WORKER = AvatarWorker(af_model=AGENT.G, avatar_ref_tensor=AVATAR_REF_TENSOR,
                                 device=f"cuda:{opt.rank}")
    AVATAR_WORKER.start()
    AVATAR_WORKER._warmup_done.wait()   # block until warmup finishes
    log.info("avatar worker ready")

    # Lazy Parakeet load — after avatar warmup to avoid CUDA context contention.
    if _load_parakeet_later:
        log.info(f"loading Parakeet STT on {aux_device} (post-warmup)")
        STT = ParakeetSTT(device=aux_device)

    # Pre-preprocess all catalog portraits so /set-avatar is instant.
    _preload_avatar_tensors()


def _playback_delay(worker) -> float:
    """Adaptive audio+video playback delay.

    Reads the avatar worker's EWMA of block wall-time and adds a small
    safety margin. Both the audio and video tracks shift their playback
    cursor back by this amount so the worker has enough head-room to
    produce matching video for whatever audio is about to be played.
    Clamped to a sensible floor/ceiling so adverse readings can't make
    the UX bizarre.
    """
    base = 0.3
    if worker is not None:
        base = max(base, getattr(worker, "block_time_ewma", 0.3) * 1.2)
    return max(0.15, min(0.6, base))


# ---------- Session (per browser connection) ----------
class Session:
    def __init__(self, session_id: str):
        self.id = session_id
        self.t0 = time.monotonic()
        self.user_audio_ring = AudioRing(capacity_seconds=30.0, sample_rate=16000)
        self.user_video_ring = VideoRing(capacity_seconds=30.0, fps=25)
        self.avatar_audio_feat_ring = FeatureRing(capacity_seconds=30.0, fps=25, dim=512)
        self.avatar_audio_pcm_ring  = AudioRing(capacity_seconds=30.0, sample_rate=16000)
        # Tiny jitter buffer (~200 ms). Worker's blocking put + wall-clock
        # pacing keeps the pipeline synced to realtime; the queue just
        # absorbs typical per-block variance (we see 100-200 ms spikes).
        # Larger queue = more steady-state audio delay.
        self.out_video_q: queue.Queue = queue.Queue(maxsize=5)
        self._tasks: list[asyncio.Task] = []
        self.reply_task: asyncio.Task | None = None
        self.worker: AvatarWorker | None = None
        self.vad: VADWorker | None = None
        self.chat_ch = None   # RTCDataChannel; set from pc.on("datachannel")
        self.barge_in_enabled: bool = True   # client can toggle via data channel
        # When muted the server drops mic audio before it reaches the ring,
        # so VAD can't fire regardless of whatever the browser does with
        # track.enabled. Toggled via data channel.
        self.mic_muted: bool = False
        # Live-tunable lip-sync offset (seconds). Positive = delay audio
        # relative to video (mouth leads sound). Negative = inverse.
        # Exposed to the client via /config lipsync messages.
        self.lipsync_offset_s: float = 0.0
        # Logical video playhead: advanced by AvatarVideoTrack each time it
        # emits a real (non-prebuffer) frame. AvatarAudioTrack reads pcm_ring
        # at this cursor so audio and video stay lip-synced even when the
        # worker falls behind wall-clock.
        self.video_playhead: float = 0.0

    def chat(self, who: str, text: str):
        """Push a chat message to the browser if the data channel is open."""
        ch = self.chat_ch
        if ch is None:
            return
        try:
            if getattr(ch, "readyState", "open") != "open":
                return
            ch.send(json.dumps({"who": who, "text": text}))
        except Exception as e:
            log.warning(f"[{self.id}] chat send failed: {e}")

    def now(self) -> float:
        return time.monotonic() - self.t0

    def add_task(self, task: asyncio.Task):
        self._tasks.append(task)

    async def close(self):
        if self.reply_task and not self.reply_task.done():
            self.reply_task.cancel()
        if self.worker:
            self.worker.end_session()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try: await t
            except: pass
        # Drop LLM conversation history so the next session starts fresh.
        if LLM is not None:
            LLM.reset()


# ---------- Inbound track consumers ----------
MIC_JITTER_S = 0.05


def mic_start_ts(ring: AudioRing, start_ts: float) -> float | None:
    """Ring start_ts for a mic frame. Frames arrive with a few ms of jitter;
    passing their arrival time straight through makes the ring zero-fill every
    late frame's jitter, chopping silence into the user's speech (it broke VAD
    start-of-speech on the first words of a turn). Gaps under MIC_JITTER_S are
    treated as jitter (append contiguously); real dropouts still zero-fill."""
    start_ts = max(0.0, start_ts)
    return start_ts if start_ts - ring.write_ts > MIC_JITTER_S else None


async def consume_mic(track: MediaStreamTrack, session: Session):
    resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
    log.info(f"[{session.id}] mic consumer started")
    while True:
        try:
            frame = await track.recv()
        except Exception:
            log.info(f"[{session.id}] mic ended")
            return
        for out_frame in resampler.resample(frame):
            samples = out_frame.to_ndarray().reshape(-1).astype(np.float32)
            if samples.size == 0: continue
            ts = session.now()
            dur = samples.size / 16000.0
            # When muted, write zeros rather than dropping the frame — keeps
            # ring.write_ts advancing so VAD doesn't stall on the wait loop.
            if session.mic_muted:
                samples = np.zeros_like(samples)
            session.user_audio_ring.write(samples, start_ts=mic_start_ts(session.user_audio_ring, ts - dur))


async def consume_webcam(track: MediaStreamTrack, session: Session):
    """Rate-limited webcam consumer. Drops extra frames so we never do more than
    one face-crop per 40 ms (25 fps) even if the browser sends faster. Also
    skips a crop if the previous one is still running — don't backlog."""
    log.info(f"[{session.id}] webcam consumer started")
    loop = asyncio.get_running_loop()
    MIN_INTERVAL = 1.0 / 25
    last_crop_ts = 0.0
    cropping = False
    while True:
        try:
            frame = await track.recv()
        except Exception:
            log.info(f"[{session.id}] webcam ended"); return
        now = session.now()
        if cropping or (now - last_crop_ts) < MIN_INTERVAL:
            continue
        cropping = True
        try:
            img = frame.to_ndarray(format="rgb24")
            crop = await loop.run_in_executor(FACE_CROPPER._executor, FACE_CROPPER.crop, img)
            if crop is None:
                continue
            session.user_video_ring.write(crop, ts=session.now())
            last_crop_ts = session.now()
        finally:
            cropping = False


# ---------- Outbound tracks ----------
class AvatarVideoTrack(VideoStreamTrack):
    """Reads frames from the worker's output queue, paces to 25 fps using next_timestamp()."""
    kind = "video"

    def __init__(self, session: Session, fps: int = 25, prebuffer_frames: int = 3):
        super().__init__()
        self.session = session
        self.fps = fps
        self._time_base = Fraction(1, fps)
        self._blank = np.zeros((512, 512, 3), dtype=np.uint8)
        self._prebuffer_frames = prebuffer_frames
        self._prebuffered = False
        self._ended = False

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        if self._ended:
            await asyncio.sleep(1.0)
            raise asyncio.CancelledError()

        loop = asyncio.get_running_loop()
        if not self._prebuffered:
            if self.session.out_video_q.qsize() < self._prebuffer_frames:
                frame = av.VideoFrame.from_ndarray(self._blank, format="rgb24")
                frame.pts = pts; frame.time_base = time_base
                return frame
            self._prebuffered = True
            log.info(f"[{self.session.id}] video: prebuffer met (q={self.session.out_video_q.qsize()})")

        # Video adapts: drop frames whose logical ts has already passed by
        # more than one video-frame interval, emit the freshest frame.
        # If the queue is momentarily empty (bursty worker vs. steady 25 fps
        # consumer), wait briefly for a fresh frame instead of immediately
        # repeating last — otherwise WebRTC stats show low fps because we
        # keep sending the same frame.
        w = getattr(self.session, "worker", None)
        delay = _playback_delay(w)
        now = self.session.now() - delay
        tolerance = 1.0 / self.fps
        picked_frame = None
        picked_ts = None
        while True:
            try:
                item = self.session.out_video_q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                log.info(f"[{self.session.id}] avatar video: end of stream")
                self._ended = True
                await asyncio.sleep(0.1)
                raise asyncio.CancelledError()
            ts, img = item
            if ts < now - tolerance:
                # Stale — a newer frame has already caught up to wall-clock.
                continue
            picked_frame = img
            picked_ts = ts
            break
        if picked_frame is None:
            # Nothing fresh; repeat last emitted frame (or blank).
            picked_frame = getattr(self, "_last_frame", None)
            picked_ts = getattr(self, "_last_ts", 0.0)
        else:
            self._last_frame = picked_frame
            self._last_ts = picked_ts
        if picked_frame is None:
            picked_frame = self._blank
        self.session.video_playhead = picked_ts
        # Periodic lag indicator to the browser. lag = how far the emitted
        # frame is behind wall-clock.
        self._lag_tick = getattr(self, "_lag_tick", 0) + 1
        if self._lag_tick % 25 == 0:
            lag = now - (picked_ts or now)
            try: self.session.chat_ch and self.session.chat_ch.send(json.dumps({"type":"lag","value":lag}))
            except Exception: pass
        frame = av.VideoFrame.from_ndarray(picked_frame, format="rgb24")
        frame.pts = pts; frame.time_base = time_base
        return frame


class AvatarAudioTrack(MediaStreamTrack):
    """Audio-pacing in a DEDICATED OS THREAD.

    A background thread wakes every 20 ms (monotonic clock), reads one
    chunk from pcm_ring, resamples 16k→48k s16, and pushes it into a
    bounded queue. `recv()` just pops — no asyncio.sleep timing, no
    cursor math on the event loop. This insulates audio from event-loop
    starvation that happens during big LLM/TTS bursts, which otherwise
    caused the cursor to lag by multiple seconds (phrases "arriving
    late") OR skip content (fragmented audio), depending on the strategy.
    """
    kind = "audio"
    OUT_SR = 48000
    FRAME_MS = 20
    FRAME_SAMPLES_48K = OUT_SR * FRAME_MS // 1000   # 960
    N_SAMPLES_16K = 16000 * FRAME_MS // 1000        # 320

    def __init__(self, session: Session):
        super().__init__()
        self.session = session
        self._pts = 0
        self._time_base = Fraction(1, self.OUT_SR)
        # Small queue — keep end-to-end latency low. Pacer drops oldest
        # if aiortc's consumer momentarily falls behind.
        self._out_q: queue.Queue = queue.Queue(maxsize=8)
        self._thread_stop = threading.Event()
        self._thread = threading.Thread(
            target=self._pacer_loop, daemon=True,
            name=f"af-audio-{session.id}")
        self._thread.start()

    def _pacer_loop(self):
        chunk_s = self.FRAME_MS / 1000.0
        last_end = None
        tick = 0
        t_start = time.monotonic()
        while not self._thread_stop.is_set():
            # Precise wait until next tick boundary. The monotonic
            # anchor keeps us from accumulating drift from sleep
            # inaccuracy — always sleep to an absolute target time.
            next_wall = t_start + tick * chunk_s
            dt = next_wall - time.monotonic()
            if dt > 0:
                if self._thread_stop.wait(dt):
                    break
            tick += 1

            w = getattr(self.session, "worker", None)
            playback_delay = _playback_delay(w) + self.session.lipsync_offset_s
            target = self.session.now() - playback_delay
            if last_end is None:
                last_end = max(0.0, target - chunk_s)
            drift = last_end - target
            if abs(drift) > 0.15:
                ring_write_ts = self.session.avatar_audio_pcm_ring.write_ts
                if last_end + chunk_s >= ring_write_ts:
                    log.info(f"audio resync (thread): cursor {last_end:.2f} → {target - chunk_s:.2f} (drift {drift*1000:.0f}ms, ring_ts {ring_write_ts:.2f})")
                    last_end = max(0.0, target - chunk_s)
            start = last_end
            end = start + chunk_s
            last_end = end
            samples_16k = self.session.avatar_audio_pcm_ring.read_window(start, end)
            # Resample 16k → 48k
            samples_48k = _scipy_signal.resample_poly(
                samples_16k, up=3, down=1).astype(np.float32)
            if samples_48k.size < self.FRAME_SAMPLES_48K:
                samples_48k = np.pad(samples_48k, (0, self.FRAME_SAMPLES_48K - samples_48k.size))
            else:
                samples_48k = samples_48k[:self.FRAME_SAMPLES_48K]
            samples_s16 = np.clip(samples_48k * 32767, -32768, 32767).astype(np.int16)
            try:
                self._out_q.put_nowait(samples_s16)
            except queue.Full:
                # Consumer is slow — drop the OLDEST chunk (so what we
                # emit stays current), then put the new one.
                try: self._out_q.get_nowait()
                except queue.Empty: pass
                try: self._out_q.put_nowait(samples_s16)
                except queue.Full: pass

    def _next_chunk(self):
        """Blocking pop for the executor. Polls the stop flag: a plain get()
        pending when the track stops would block its default-executor thread
        forever — one leaked thread per session until the pool (cpu_count+4)
        is exhausted, after which everything on the default executor hangs
        (httpx's getaddrinfo for the LLM, aiortc's encoders) and the process
        can't exit on SIGTERM."""
        while not self._thread_stop.is_set():
            try:
                return self._out_q.get(timeout=0.1)
            except queue.Empty:
                continue
        return None

    async def recv(self):
        loop = asyncio.get_running_loop()
        # Pop the next pre-paced chunk; the pacer thread produces at precise 20 ms.
        samples_s16 = await loop.run_in_executor(None, self._next_chunk)
        if samples_s16 is None:
            raise MediaStreamError
        frame = av.AudioFrame.from_ndarray(samples_s16.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self.OUT_SR
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += self.FRAME_SAMPLES_48K
        return frame

    def stop(self):
        self._thread_stop.set()
        try:
            super().stop()
        except Exception:
            pass


# ---------- VAD event wiring ----------
def make_vad(session: Session) -> VADWorker:
    async def on_sos(ts):
        log.info(f"[{session.id}] start-of-speech @ {ts:.2f}")
        if not session.barge_in_enabled:
            log.info(f"[{session.id}] SOS ignored: barge-in disabled")
            return
        # Barge-in semantics: stop producing new output, but let any
        # already-buffered audio play through to natural phrase end. Hard
        # cutting the ring at playhead was slicing off the final word of
        # whatever the avatar was in the middle of saying (false VAD
        # trigger from mic-leaked avatar audio was the common cause).
        #   (1) cancel the in-flight reply so no more LLM/TTS phrases arrive
        #   (2) drop any TTS already queued for wav2vec on the worker
        barged = session.reply_task and not session.reply_task.done()
        if barged:
            log.info(f"[{session.id}] barge-in: cancelling reply_task")
            session.reply_task.cancel()
        if AVATAR_WORKER is not None:
            AVATAR_WORKER.cancel_pending_tts()

    async def on_done(audio: np.ndarray, start_ts: float, end_ts: float):
        log.info(f"[{session.id}] turn complete: {end_ts - start_ts:.2f}s audio")
        # STT → reply_pipeline
        try:
            text = await STT.transcribe(audio)
        except Exception:
            log.exception("stt failed")
            return
        if not text.strip():
            return
        session.chat("you", text)
        # Stream avatar text per-phrase as it's queued for speech; keeps the
        # transcript in sync even if barge-in cancels mid-reply.
        session.reply_task = asyncio.create_task(reply_pipeline(
            text, LLM, TTS, AGENT.G,
            feat_ring=session.avatar_audio_feat_ring,
            pcm_ring=session.avatar_audio_pcm_ring,
            now_fn=session.now,
            worker=AVATAR_WORKER,
            on_phrase=lambda phrase: session.chat("avatar", phrase),
        ))

    return VADWorker(
        audio_ring=session.user_audio_ring,
        smart_turn_onnx_path=str(ROOT / "voicechat/models/smart-turn-v3.2-cpu.onnx"),
        on_start_of_speech=on_sos,
        on_turn_complete=on_done,
        config=TurnDetectorConfig(),
        logger_tag=f"vad-{session.id}",
    )


# ---------- aiohttp ----------
PCS: set[RTCPeerConnection] = set()
SESSIONS: dict[str, Session] = {}


async def index(request):
    html = (ROOT / "voicechat/client.html").read_text()
    return web.Response(text=html, content_type="text/html")


async def memory_stats(request):
    import torch as _torch
    out = {}
    for i in range(_torch.cuda.device_count()):
        allocated = _torch.cuda.memory_allocated(i)
        reserved = _torch.cuda.memory_reserved(i)
        max_allocated = _torch.cuda.max_memory_allocated(i)
        max_reserved = _torch.cuda.max_memory_reserved(i)
        total = _torch.cuda.get_device_properties(i).total_memory
        out[f"cuda:{i}"] = {
            "device_name": _torch.cuda.get_device_name(i),
            "allocated_gb": round(allocated / 1e9, 3),
            "reserved_gb": round(reserved / 1e9, 3),
            "max_allocated_gb": round(max_allocated / 1e9, 3),
            "max_reserved_gb": round(max_reserved / 1e9, 3),
            "total_gb": round(total / 1e9, 3),
            "reserved_pct": round(100 * reserved / total, 1),
        }
    # Enable torch's per-category breakdown if asked
    if request.query.get("full") == "1":
        for i in range(_torch.cuda.device_count()):
            out[f"cuda:{i}"]["summary"] = _torch.cuda.memory_summary(i, abbreviated=True)
    return web.json_response(out)


AVATAR_CATALOG = [
    {"id": "rumi",  "file": "rumi.jpg",       "name": "Rumi",  "gender": "female"},
    {"id": "kelly", "file": "kelly.jpg",      "name": "Kelly", "gender": "female"},
    {"id": "ken",  "file": "ken.jpg", "name": "Ken",  "gender": "male"},
]
# Currently selected (updated by /set-avatar).
CURRENT_AVATAR = {"id": "rumi", "file": "rumi.jpg", "name": "Rumi", "gender": "female"}
# id → preprocessed [1,3,512,512] tensor, built once at startup so /set-avatar
# is a hashmap lookup instead of a 2-second face-detect+crop.
AVATAR_TENSORS: dict[str, torch.Tensor] = {}


def _preload_avatar_tensors():
    """Preprocess every catalog portrait once so session start is instant."""
    global AVATAR_TENSORS
    for entry in AVATAR_CATALOG:
        path = ROOT / "data" / entry["file"]
        if not path.exists():
            log.warning(f"avatar '{entry['id']}' file missing: {path} — skipping")
            continue
        try:
            t0 = time.perf_counter()
            sample = AGENT.data_processor.preprocess(
                avatar_ref_path   = str(path),
                avatar_audio_path = str(AF_ROOT / "data/avatar.wav"),
                user_audio_path   = str(AF_ROOT / "data/user.wav"),
                user_video_path   = str(AF_ROOT / "data/user/"),
            )
            AVATAR_TENSORS[entry["id"]] = sample["avatar_ref"]
            log.info(f"preloaded avatar '{entry['id']}' in {(time.perf_counter()-t0)*1000:.0f}ms")
        except Exception as e:
            log.exception(f"preloading avatar '{entry['id']}' failed: {e}")


async def avatar_ref(request):
    """Serve the CURRENT avatar's reference portrait (for the spinner
    overlay). Updates when /set-avatar is called."""
    path = ROOT / "data" / CURRENT_AVATAR["file"]
    return web.FileResponse(path)


async def list_avatars(request):
    """Return the catalog so the client can render thumbnails."""
    return web.json_response({"avatars": AVATAR_CATALOG,
                              "current": CURRENT_AVATAR["id"]})


async def thumb(request):
    """Serve a thumbnail for an avatar id."""
    avatar_id = request.match_info["id"]
    entry = next((a for a in AVATAR_CATALOG if a["id"] == avatar_id), None)
    if entry is None:
        return web.Response(status=404)
    return web.FileResponse(ROOT / "data" / entry["file"])


async def set_avatar(request):
    """Switch the active avatar and set the LLM persona name.
    Body: {"id": "...", "name": "..."}.
    Takes effect on the NEXT session. If a session is active it is not
    interrupted here — the client ends and restarts the session itself."""
    global CURRENT_AVATAR
    data = await request.json()
    avatar_id = data.get("id")
    entry = next((a for a in AVATAR_CATALOG if a["id"] == avatar_id), None)
    if entry is None:
        return web.json_response({"error": f"unknown avatar id {avatar_id!r}"}, status=400)
    display_name = (data.get("name") or entry["name"]).strip() or entry["name"]
    # Fast path: hit the preloaded cache. Falls back to on-demand preprocess
    # (slow) only if the cache entry is missing for some reason.
    new_tensor = AVATAR_TENSORS.get(entry["id"])
    if new_tensor is None:
        ref_path = ROOT / "data" / entry["file"]
        log.warning(f"avatar '{entry['id']}' not pre-cached, preprocessing now (slow)")
        sample = AGENT.data_processor.preprocess(
            avatar_ref_path   = str(ref_path),
            avatar_audio_path = str(AF_ROOT / "data/avatar.wav"),
            user_audio_path   = str(AF_ROOT / "data/user.wav"),
            user_video_path   = str(AF_ROOT / "data/user/"),
        )
        new_tensor = sample["avatar_ref"]
        AVATAR_TENSORS[entry["id"]] = new_tensor
    if AVATAR_WORKER is not None:
        AVATAR_WORKER.avatar_ref = new_tensor
    # Update LLM persona — tell the model its presenting gender so it stays
    # consistent if the user asks.
    gender = entry.get("gender", "female")
    article = "an" if gender[0].lower() in "aeiou" else "a"
    LLM.system_prompt = (f"You are {display_name}, {article} {gender} voice-chat "
                        f"assistant. Reply briefly and conversationally.")
    LLM.reset()   # fresh conversation with the new persona
    # Match TTS voice to gender so the avatar sounds plausible.
    if TTS is not None:
        voice = (PhononTTS.DEFAULT_VOICE_MALE if gender == "male"
                 else PhononTTS.DEFAULT_VOICE_FEMALE)
        TTS.set_voice(voice)
    CURRENT_AVATAR = {"id": entry["id"], "file": entry["file"],
                      "name": display_name, "gender": gender}
    log.info(f"switched to avatar id={entry['id']} name={display_name!r} gender={gender}")
    return web.json_response({"ok": True, "current": CURRENT_AVATAR})


async def offer(request):
    init_singletons()

    # Belt-and-suspenders: also clear LLM history here. If the previous
    # session didn't close cleanly (crash, abandoned tab, etc.) sess.close()
    # may not have run; this guarantees fresh conversation per /offer.
    if LLM is not None:
        LLM.reset()

    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Empty ice servers list = skip STUN. For local/LAN delivery the host
    # candidates alone are enough and we don't pay ~1 s waiting for STUN.
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    PCS.add(pc)
    session_id = f"s{len(SESSIONS)+1}"
    sess = Session(session_id)
    SESSIONS[session_id] = sess
    log.info(f"[{session_id}] new peer connection")

    @pc.on("track")
    def on_track(track):
        log.info(f"[{session_id}] inbound track: {track.kind}")
        if track.kind == "audio":
            sess.add_task(asyncio.create_task(consume_mic(track, sess)))
        # Video track from browser is accepted (needed for WebRTC SDP
        # negotiation) but not consumed — FaceCropper is disabled, user
        # motion is zeroed, and nothing needs the frames.

    @pc.on("datachannel")
    def on_datachannel(channel):
        log.info(f"[{session_id}] data channel: {channel.label}")
        if channel.label == "chat":
            sess.chat_ch = channel
            @channel.on("message")
            def on_chat_msg(msg):
                try:
                    data = json.loads(msg)
                except Exception:
                    return
                t = data.get("type")
                if t == "barge-in":
                    sess.barge_in_enabled = bool(data.get("enabled", True))
                    log.info(f"[{session_id}] barge-in {'enabled' if sess.barge_in_enabled else 'disabled'}")
                elif t == "mute-mic":
                    sess.mic_muted = bool(data.get("muted", False))
                    log.info(f"[{session_id}] mic {'muted' if sess.mic_muted else 'unmuted'}")
                elif t == "config":
                    key = data.get("key"); value = data.get("value")
                    if key == "lipsync":
                        # milliseconds in from the client; store as seconds
                        sess.lipsync_offset_s = float(value) / 1000.0
                        log.info(f"[{session_id}] lipsync offset = {value:.0f} ms")
                        return
                    if sess.vad is None: return
                    cfg = sess.vad.cfg
                    if key == "rms":
                        cfg.rms_gate = float(value)
                    elif key == "prob":
                        cfg.speech_prob_threshold = float(value)
                    elif key == "enter":
                        # seconds → chunks (32 ms each)
                        cfg.enter_speech_frames = max(1, int(round(float(value) * 1000 / 32)))
                    log.info(f"[{session_id}] config {key}={value} → "
                             f"rms={cfg.rms_gate} prob={cfg.speech_prob_threshold} "
                             f"enter={cfg.enter_speech_frames}")

    @pc.on("connectionstatechange")
    async def on_state():
        log.info(f"[{session_id}] pc state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await sess.close()
            SESSIONS.pop(session_id, None)
            PCS.discard(pc)
            try: await pc.close()
            except: pass

    # Hand this session's rings to the persistent avatar worker
    sess.worker = AVATAR_WORKER   # reference so sess.close() can call end_session()
    def _on_reanchor():
        try: sess.chat_ch and sess.chat_ch.send(json.dumps({"type": "log", "text": "reanchor"}))
        except Exception: pass
    AVATAR_WORKER.start_session(
        uv_ring=sess.user_video_ring,
        af_ring=sess.avatar_audio_feat_ring,
        pcm_ring=sess.avatar_audio_pcm_ring,
        out_q=sess.out_video_q,
        now_fn=sess.now,
        playhead_fn=lambda: sess.video_playhead,
        on_reanchor=_on_reanchor,
    )

    # Start VAD/turn detect
    sess.vad = make_vad(sess)
    sess.add_task(asyncio.create_task(sess.vad.run()))

    # Mount outbound tracks
    pc.addTrack(AvatarVideoTrack(sess))
    pc.addTrack(AvatarAudioTrack(sess))

    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    # Prune the candidate pool before replying. Removes IPv6 link-local
    # (fe80:/fd*:), Docker bridge (172.17.*, 172.18.*), Tailscale (100.64/10),
    # and the loopback-stun placeholder. Dramatically cuts the connectivity-
    # check time because the browser isn't trying every path.
    sdp = _trim_ice_candidates(pc.localDescription.sdp)
    return web.json_response({"sdp": sdp, "type": pc.localDescription.type})


def _trim_ice_candidates(sdp: str) -> str:
    """Drop candidate lines we don't want the browser to bother with."""
    kept = []
    dropped = 0
    for line in sdp.splitlines():
        if line.startswith("a=candidate:"):
            lower = line.lower()
            drop = (
                " fe80:" in lower or                # IPv6 link-local
                " fd" in lower.split(" typ ")[0] or  # IPv6 ULA (Tailscale, etc.)
                " 172.17." in line or " 172.18." in line or   # Docker
                " 100." in line or                 # Tailscale CGNAT / carrier-NAT
                " 169.254." in line                # IPv4 link-local
            )
            if drop:
                dropped += 1
                continue
        kept.append(line)
    if dropped:
        log.info(f"ICE: trimmed {dropped} candidate(s) from answer SDP")
    return "\r\n".join(kept) + "\r\n"


async def on_shutdown(app):
    await asyncio.gather(*[s.close() for s in SESSIONS.values()], return_exceptions=True)
    await asyncio.gather(*[pc.close() for pc in PCS], return_exceptions=True)
    SESSIONS.clear(); PCS.clear()
    if LLM: await LLM.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("AF_PORT", 8082)))
    ap.add_argument("--host", default="0.0.0.0")
    # Default: single-GPU mode — avatar + aux share cuda:0, even when 2 cards
    # are visible. Pass --dual-gpu to split (avatar on cuda:1, aux on cuda:0).
    # Also settable via AF_SINGLE_GPU=0 / AF_SINGLE_GPU=1.
    ap.add_argument("--dual-gpu", action="store_true",
                    help="Split onto 2 GPUs: avatar on cuda:1, aux on cuda:0. Default is single-GPU on cuda:0.")
    args = ap.parse_args()
    # Single-gpu defaults to True unless --dual-gpu is passed or AF_SINGLE_GPU=0 is set.
    env_single = os.environ.get("AF_SINGLE_GPU")
    if env_single is not None:
        single_gpu = env_single == "1"
    else:
        single_gpu = not args.dual_gpu

    # Eager singleton load so first browser doesn't eat warmup
    init_singletons(single_gpu=single_gpu)

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/avatar-ref.jpg", avatar_ref)
    app.router.add_get("/avatars", list_avatars)
    app.router.add_get("/memory", memory_stats)
    app.router.add_get("/thumb/{id}", thumb)
    app.router.add_post("/set-avatar", set_avatar)
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)

    log.info(f"voice-chat server listening on http://<host>:{args.port}")
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
