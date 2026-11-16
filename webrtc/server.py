"""
AvatarForcing WebRTC streaming demo (Phase C).

Serves a single page at http://HOST:$PORT (default 8081) that opens a WebRTC
peer connection. Server generates avatar video block-by-block via
inference_stream() and streams to the browser alongside the pre-recorded
avatar_audio.

Usage:
    cd <repo-root>
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=1 python webrtc/server.py [--port 8081]

Persistent compile cache (reduces subsequent-startup warmup):
    export TORCHINDUCTOR_CACHE_DIR=<repo-root>/.inductor_cache
"""
import os, sys, asyncio, json, queue, threading, time, logging, argparse

# Enable a persistent compile cache if the user hasn't set one — next startup will reuse
# compiled kernels and skip most of the warmup (expected: 135s → ~30s on 2nd boot).
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".inductor_cache")),
)
from fractions import Fraction
from pathlib import Path

import av
import numpy as np  # noqa: F401 — used in VideoTrack placeholder
import torch
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRelay
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
AF_ROOT = ROOT / "avatarforcing"   # upstream AvatarForcing (submodule)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AF_ROOT))

from inference import InferenceAgent, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("af-webrtc")

# ---------- Model agent + single long-lived generation worker ----------
# All AvatarForcing GPU work happens on ONE dedicated thread.  torch.compile /
# dynamo guards are thread-local — if we spawn a new thread per browser session
# we pay the ~2-minute re-trace cost every time.  Keeping a single worker thread
# means warmup compiles once in that thread and every subsequent session reuses
# the compiled graphs.
AGENT = None
GEN_READY = threading.Event()
GEN_WORK_Q: "queue.Queue" = queue.Queue()   # items: (data, cfg_a, cfg_u, nfe, seed, output_q, stop_event)


def _extend_rope_tables(flow_transformer, new_max_len: int):
    """Replace all precomputed RoPE freqs_cis buffers with an extended table.

    The shipped model precomputes freqs_cis at max_seq_len=1024. That corresponds
    to ~41s of video at 25fps and 10-frame blocks (block 100 => start_pos~1000).
    Past that, `freqs_cis[start_pos:end_pos]` goes out of bounds and dynamo's
    shape math fails. Extending the table lets us stream far longer.

    No retraining involved — RoPE is a positional formula, we just precompute
    more of it.
    """
    from models.avatarforcing.flow_transformer import precompute_freqs_cis
    replaced = 0
    for name, mod in flow_transformer.named_modules():
        if "freqs_cis" in getattr(mod, "_buffers", {}):
            old = mod._buffers["freqs_cis"]
            # old has shape [cur_max_len, head_dim/2] complex; we need head_dim
            head_dim = old.shape[-1] * 2
            new_table = precompute_freqs_cis(head_dim, new_max_len).to(old.device)
            mod.register_buffer("freqs_cis", new_table, persistent=False)
            replaced += 1
    return replaced


def _build_agent():
    class Opt: pass
    opt = Opt()
    cfg = OmegaConf.load(str(AF_ROOT / "configs/inference.yaml"))
    for k, v in cfg.items():
        setattr(opt, k, v)
    opt.infer_config = str(AF_ROOT / "configs/inference.yaml")
    opt.mae_ckpt_path = str(AF_ROOT / "pretrained_dir/motion_autoencoder.pth")
    opt.ckpt_path = str(AF_ROOT / "pretrained_dir/flow_transformer.pth")
    opt.rank, opt.ngpus = 0, 1
    opt.result_dir = str(ROOT / "results/webrtc")
    opt.compile = True
    opt.warmup = 0
    os.makedirs(opt.result_dir, exist_ok=True)
    agent = InferenceAgent(opt)
    # Extend RoPE tables BEFORE torch.compile wraps anything else — the compile
    # happens inside InferenceAgent.init_network() already; however compile
    # captures module parameters by reference, so swapping a buffer after
    # compile should also be seen on next call. We reset here to be safe.
    rope_max = int(os.environ.get("AF_ROPE_MAX", 65536))  # 65536 frames = ~43 minutes at 25fps
    n = _extend_rope_tables(agent.G.flow_transformer, rope_max)
    log.info(f"extended {n} RoPE freqs_cis buffers to max_seq_len={rope_max}")
    return agent


def _generation_worker():
    """Long-lived worker: builds agent, warms up, then serves jobs from GEN_WORK_Q."""
    global AGENT
    try:
        log.info("[worker] building AvatarForcing agent")
        AGENT = _build_agent()

        log.info("[worker] warming up torch.compile via streaming path (two full passes)")
        t0 = time.perf_counter()
        warm_data = AGENT.data_processor.preprocess(
            avatar_ref_path=str(ROOT / "data/rumi.jpg"),
            avatar_audio_path=str(AF_ROOT / "data/avatar.wav"),
            user_audio_path=str(AF_ROOT / "data/user.wav"),
            user_video_path=str(AF_ROOT / "data/user/"),
        )
        for pass_idx in (1, 2):
            tp = time.perf_counter()
            n = 0
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for block_idx, pixels, is_last in AGENT.G.inference_stream(
                    data=warm_data, a_cfg_scale=2.0, u_cfg_scale=1.0, nfe=10, seed=25
                ):
                    n += 1
                    if is_last:
                        break
            torch.cuda.synchronize()
            log.info(f"[worker]   warmup pass {pass_idx}: {time.perf_counter()-tp:.1f}s ({n} blocks)")
        log.info(f"[worker] warmup done in {time.perf_counter()-t0:.1f}s")
        GEN_READY.set()
    except Exception:
        log.exception("[worker] warmup failed")
        GEN_READY.set()  # unblock main even on failure, server will bail
        return

    # Serve jobs
    while True:
        job = GEN_WORK_Q.get()
        if job is None:
            log.info("[worker] shutdown signal, exiting")
            return
        data, cfg_a, cfg_u, nfe, seed, output_q, stop_event = job
        try:
            log.info("[worker] starting inference_stream for new session")
            t_start = time.perf_counter()
            t_prev = t_start
            n = 0
            first_t = None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for block_idx, pixels, is_last in AGENT.G.inference_stream(
                    data=data, a_cfg_scale=cfg_a, u_cfg_scale=cfg_u, nfe=nfe, seed=seed,
                ):
                    if stop_event.is_set():
                        log.info("[worker] stop requested")
                        break
                    torch.cuda.synchronize()
                    t_now = time.perf_counter()
                    block_ms = (t_now - t_prev) * 1000
                    t_prev = t_now
                    if first_t is None:
                        first_t = t_now - t_start

                    vid = pixels.squeeze(0).detach().clamp(-1, 1).cpu()
                    vid = ((vid - vid.min()) / (vid.max() - vid.min() + 1e-8) * 255.0)
                    vid = vid.permute(0, 2, 3, 1).contiguous().to(torch.uint8).numpy()
                    for frame in vid:
                        output_q.put(frame)
                    n += 1
                    if n <= 10 or n % 10 == 0 or is_last:
                        log.info(f"[worker]   block {n:3d} gen={block_ms:7.1f}ms  q_depth={output_q.qsize()}")
                    if is_last:
                        break
            dt = time.perf_counter() - t_start
            log.info(f"[worker] session done — {n} blocks in {dt:.2f}s (first-block {first_t*1000:.0f}ms)")
        except Exception:
            log.exception("[worker] session crashed")
        finally:
            output_q.put(None)


def get_agent():
    """Block until the worker has built the agent and finished warmup."""
    GEN_READY.wait()
    return AGENT


# ---------- Streaming video track ----------
class StreamingAvatarTrack(VideoStreamTrack):
    """Video track that pulls RGB frames from a thread-safe queue.

    Waits for a prebuffer_frames threshold before emitting the first frame so a
    mid-stream generator hiccup (e.g. torch.compile re-trace on a rare shape)
    doesn't immediately drain the queue and stall the browser.
    """
    kind = "video"

    def __init__(self, frame_queue: queue.Queue, fps: int = 25, prebuffer_frames: int = 40):
        super().__init__()
        self.frame_queue = frame_queue
        self.fps = fps
        self._time_base = Fraction(1, fps)
        self._ended = False
        self._prebuffer_frames = prebuffer_frames
        self._prebuffered = False
        self._blank = np.zeros((512, 512, 3), dtype=np.uint8)  # black placeholder while prebuffering

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        if self._ended:
            await asyncio.sleep(1.0)
            raise asyncio.CancelledError()

        loop = asyncio.get_running_loop()

        # Prebuffer gate: hold back emitting real frames until we've built up
        # enough queue depth that a mid-stream spike won't drain us.
        if not self._prebuffered:
            qd = self.frame_queue.qsize()
            if qd < self._prebuffer_frames:
                # Emit a black frame to keep the WebRTC stream alive
                frame = av.VideoFrame.from_ndarray(self._blank, format="rgb24")
                frame.pts = pts
                frame.time_base = time_base
                return frame
            self._prebuffered = True
            log.info(f"video track: prebuffer ready (q_depth={qd}) — start emitting")

        try:
            frame_rgb = self.frame_queue.get_nowait()
        except queue.Empty:
            frame_rgb = await loop.run_in_executor(None, self.frame_queue.get)

        if frame_rgb is None:
            log.info("video track: reached end of stream")
            self._ended = True
            await asyncio.sleep(0.1)
            raise asyncio.CancelledError()

        frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


# (old per-session generation_thread removed — all generation now dispatched
#  to the single long-lived _generation_worker via GEN_WORK_Q)


# ---------- aiohttp app ----------
PCS = set()
RELAY = MediaRelay()


async def index(request):
    html = (ROOT / "webrtc/client.html").read_text()
    return web.Response(text=html, content_type="text/html")


async def offer(request):
    agent = get_agent()
    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    PCS.add(pc)

    # Preprocess fixed inputs (Phase C uses pre-recorded user side)
    data = agent.data_processor.preprocess(
        avatar_ref_path=str(ROOT / "data/rumi.jpg"),
        avatar_audio_path=str(AF_ROOT / "data/avatar.wav"),
        user_audio_path=str(AF_ROOT / "data/user.wav"),
        user_video_path=str(AF_ROOT / "data/user/"),
    )

    # Loop the inputs N times for an "infinite" demo. Expect identity drift
    # to show up past ~30-60s per upstream issue #10 — that's a model/attention-
    # sink problem, independent of this streaming plumbing. 60x of a 16s clip
    # is ~16 minutes.
    loop_factor = int(os.environ.get("AF_LOOP_FACTOR", 60))
    data['avatar_a'] = data['avatar_a'].repeat(1, loop_factor)
    data['user_a']   = data['user_a'].repeat(1, loop_factor)
    data['user_frame'] = data['user_frame'] * loop_factor
    log.info(f"input looped {loop_factor}x → ~{loop_factor * 15.88:.0f}s of generation")

    # Frame queue; generation runs on the single long-lived worker thread
    frame_q: queue.Queue = queue.Queue(maxsize=200)  # ~8s buffer at 25fps
    stop_event = threading.Event()
    GEN_WORK_Q.put((data, 2.0, 1.0, 10, 25, frame_q, stop_event))

    # Video track from our queue
    video_track = StreamingAvatarTrack(frame_q, fps=agent.opt.fps)
    pc.addTrack(video_track)

    # Audio track from avatar.wav, looped so audio doesn't cut out while video
    # continues from the looped input tensors above.
    audio_player = MediaPlayer(str(AF_ROOT / "data/avatar.wav"), loop=True)
    if audio_player.audio:
        pc.addTrack(audio_player.audio)

    @pc.on("connectionstatechange")
    async def on_state():
        log.info(f"pc state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            stop_event.set()
            PCS.discard(pc)
            try:
                await pc.close()
            except Exception:
                pass

    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


async def on_shutdown(app):
    log.info("Shutting down, closing %d peer connections", len(PCS))
    await asyncio.gather(*[pc.close() for pc in PCS], return_exceptions=True)
    PCS.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("AF_PORT", 8081)))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the eager warmup (first connection pays ~2min)")
    args = ap.parse_args()

    log.info(f"TORCHINDUCTOR_CACHE_DIR = {os.environ.get('TORCHINDUCTOR_CACHE_DIR')}")

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)

    # Start the single generation worker. It builds the agent + warms up
    # inside its own thread so dynamo's thread-local state is already primed
    # for every subsequent /offer dispatched to the same thread.
    log.info("Starting generation worker (build + warmup)")
    worker = threading.Thread(target=_generation_worker, daemon=True, name="af-gen-worker")
    worker.start()

    if not args.no_warmup:
        GEN_READY.wait()
    log.info(f"Ready. Open http://<host>:{args.port} in a browser.")
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
