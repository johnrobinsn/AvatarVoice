"""
Closed-loop integration test — no browser, no WebRTC.

Drives the full voice-chat pipeline (STT/LLM/TTS/wav2vec/avatar_worker) using
`data/user.wav` as the "mic" stream and `data/user/*.jpg` as the "webcam",
played into the session's rings at wall-clock pace.

Run with CUDA_LAUNCH_BLOCKING=1 to get synchronous (truthful) tracebacks:

    cd <repo-root> && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python -u voicechat/tests/test_closed_loop.py
"""
from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import queue
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Preload CUDA 12 libs for faster-whisper (same hack as server.py)
ROOT = Path(__file__).resolve().parent.parent.parent
AF_ROOT = ROOT / "avatarforcing"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AF_ROOT))
_venv = ROOT / ".venv/lib/python3.11/site-packages"
for _p in [_venv / "nvidia/cublas/lib/libcublas.so.12",
           _venv / "nvidia/cudnn/lib/libcudnn.so.9"]:
    if _p.exists():
        try: ctypes.CDLL(str(_p), mode=ctypes.RTLD_GLOBAL)
        except OSError: pass

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".inductor_cache"))

import cv2
import librosa
import torch
from omegaconf import OmegaConf

from inference import InferenceAgent
from voicechat.rings import AudioRing, VideoRing, FeatureRing
from voicechat.vad import VADWorker, TurnDetectorConfig
from voicechat.stt import WhisperSTT
from voicechat.llm import VLLMClient
from voicechat.tts import PhononTTS
from voicechat.reply_pipeline import reply_pipeline
from voicechat.avatar_worker import AvatarWorker

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("af-test")


def build_agent():
    class Opt: pass
    opt = Opt()
    cfg = OmegaConf.load(str(ROOT / "configs/inference.yaml"))
    for k, v in cfg.items(): setattr(opt, k, v)
    opt.infer_config = str(ROOT / "configs/inference.yaml")
    opt.mae_ckpt_path = str(ROOT / "pretrained_dir/motion_autoencoder.pth")
    opt.ckpt_path = str(ROOT / "pretrained_dir/flow_transformer.pth")
    import torch as _torch
    if _torch.cuda.device_count() >= 2:
        opt.rank = 1  # avatar on physical GPU 1, aux on GPU 0
    else:
        opt.rank = 0
    opt.ngpus = 1
    opt.result_dir = str(ROOT / "results/closed_loop")
    opt.compile = True
    opt.warmup = 0
    os.makedirs(opt.result_dir, exist_ok=True)
    return InferenceAgent(opt)


def extend_rope(ft, new_max=65536):
    from models.avatarforcing.flow_transformer import precompute_freqs_cis
    n = 0
    for name, mod in ft.named_modules():
        if "freqs_cis" in getattr(mod, "_buffers", {}):
            old = mod._buffers["freqs_cis"]
            head_dim = old.shape[-1] * 2
            new_table = precompute_freqs_cis(head_dim, new_max).to(old.device)
            mod.register_buffer("freqs_cis", new_table, persistent=False)
            n += 1
    return n


# ---- Fake "session" — same ring layout as the real one ----
class MockSession:
    def __init__(self):
        self.t0 = time.monotonic()
        self.user_audio_ring = AudioRing(capacity_seconds=60.0, sample_rate=16000)
        self.user_video_ring = VideoRing(capacity_seconds=60.0, fps=25)
        self.avatar_audio_feat_ring = FeatureRing(capacity_seconds=60.0, fps=25, dim=512)
        self.avatar_audio_pcm_ring  = AudioRing(capacity_seconds=60.0, sample_rate=16000)
        self.out_video_q: queue.Queue = queue.Queue(maxsize=400)

    def now(self):
        return time.monotonic() - self.t0


# ---- "Mic" and "webcam" feeders ----
async def feed_mic(session: MockSession, wav_path: Path):
    """Replay a WAV at 16kHz into user_audio_ring at wall-clock pace."""
    wav, _ = librosa.load(str(wav_path), sr=16000, mono=True)
    log.info(f"mic feeder: {wav.shape[0]/16000:.2f}s audio from {wav_path.name}")
    chunk_s = 0.02   # 20 ms
    n = int(16000 * chunk_s)
    for i in range(0, wav.shape[0] - n, n):
        ts = session.now()
        session.user_audio_ring.write(wav[i:i+n].astype(np.float32), start_ts=max(0.0, ts - chunk_s))
        await asyncio.sleep(chunk_s)
    log.info("mic feeder: done")


async def feed_webcam(session: MockSession, frame_dir: Path):
    """Replay JPEG frames at 25 fps into user_video_ring."""
    frames = sorted(frame_dir.glob("*.jpg"))
    log.info(f"webcam feeder: {len(frames)} frames from {frame_dir}")
    dt = 1.0 / 25
    for p in frames:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        session.user_video_ring.write(img, ts=session.now())
        await asyncio.sleep(dt)
    log.info("webcam feeder: done")


# ---- Drain avatar worker output + write to MP4 ----
async def drain_video(session: MockSession, out_path: Path, timeout_s: float = 30.0):
    loop = asyncio.get_running_loop()
    frames_u8 = []
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < timeout_s:
        try:
            item = await asyncio.wait_for(
                loop.run_in_executor(None, session.out_video_q.get), timeout=5.0)
        except asyncio.TimeoutError:
            log.warning(f"drain_video: 5s idle, abandoning (got {len(frames_u8)} frames)")
            break
        if item is None:
            log.info("drain_video: got None sentinel")
            break
        _ts, img = item
        frames_u8.append(img)
        if len(frames_u8) % 25 == 0:
            log.info(f"drain_video: {len(frames_u8)} frames so far")

    if not frames_u8:
        log.error("drain_video: ZERO frames produced")
        return

    arr = np.stack(frames_u8)
    H, W = arr.shape[1], arr.shape[2]
    log.info(f"drain_video: writing {arr.shape[0]} frames @ {W}x{H} to {out_path}")
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", "25", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ff.stdin.write(arr.tobytes())
    ff.stdin.close()
    ff.wait()
    log.info(f"drain_video: wrote {out_path}")


# ---- Trigger a single reply (STT→LLM→TTS→wav2vec→rings) ----
async def trigger_reply(session: MockSession, text: str,
                        llm: VLLMClient, tts: PhononTTS, af_model, worker):
    log.info(f"trigger_reply: user_text={text!r}")
    await reply_pipeline(
        text, llm, tts, af_model,
        feat_ring=session.avatar_audio_feat_ring,
        pcm_ring=session.avatar_audio_pcm_ring,
        now_fn=session.now,
        worker=worker,
    )


async def main():
    log.info("=== closed-loop test starting ===")

    # 1. Build all singletons
    log.info("building AvatarForcing agent")
    agent = build_agent()
    extend_rope(agent.G.flow_transformer, 65536)

    log.info("building STT / TTS / LLM")
    import torch as _torch
    stt_dev = "cuda:0" if _torch.cuda.device_count() >= 2 else "cuda"
    stt = WhisperSTT(device=stt_dev)
    tts = PhononTTS()
    llm = VLLMClient(model=os.environ.get(
        "AF_LLM_MODEL", "cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit"))

    # 2. Preprocess avatar ref
    log.info("preprocessing avatar ref")
    sample = agent.data_processor.preprocess(
        avatar_ref_path   = str(ROOT / "data/rumi.jpg"),
        avatar_audio_path = str(ROOT / "data/avatar.wav"),
        user_audio_path   = str(ROOT / "data/user.wav"),
        user_video_path   = str(ROOT / "data/user/"),
    )
    avatar_ref_tensor = sample['avatar_ref']

    # 3. Start persistent avatar worker + warmup
    log.info("starting avatar worker (will warmup)")
    worker = AvatarWorker(af_model=agent.G, avatar_ref_tensor=avatar_ref_tensor,
                          device=f"cuda:{agent.G.rank}",
                          nfe=int(os.environ.get("AF_NFE", "6")))
    worker.start()
    worker._warmup_done.wait()
    log.info("avatar worker warmup complete")

    # 4. Mock session + hand to worker
    session = MockSession()
    worker.start_session(
        uv_ring=session.user_video_ring,
        af_ring=session.avatar_audio_feat_ring,
        pcm_ring=session.avatar_audio_pcm_ring,
        out_q=session.out_video_q,
        now_fn=session.now,
    )

    # 5. Start mic + webcam feeders (play in background at realtime)
    mic_task = asyncio.create_task(feed_mic(session, ROOT / "data/user.wav"))
    cam_task = asyncio.create_task(feed_webcam(session, ROOT / "data/user/"))

    # Give the rings ~2 s to fill before triggering reply
    await asyncio.sleep(2.5)

    # 6. Drain avatar video in a background task
    out_mp4 = ROOT / "results/closed_loop/out.mp4"
    drain_task = asyncio.create_task(drain_video(session, out_mp4, timeout_s=25.0))

    # 7. Trigger a scripted reply
    await trigger_reply(session, "Say hello and introduce yourself briefly.", llm, tts, agent.G, worker)

    # Wait for drain to finish
    await drain_task

    # Teardown
    worker.end_session()
    worker.shutdown()
    mic_task.cancel(); cam_task.cancel()
    try: await mic_task
    except: pass
    try: await cam_task
    except: pass
    await llm.close()
    log.info("=== closed-loop test complete ===")


if __name__ == "__main__":
    asyncio.run(main())
