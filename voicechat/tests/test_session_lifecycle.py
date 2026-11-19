"""
Regression tests for session-lifecycle bugs (no GPU or models needed).

  1. AvatarAudioTrack.recv() left a default-executor thread blocked forever in
     queue.get() when its track stopped — one per session. After ~cpu_count+4
     sessions the default executor was exhausted: httpx's getaddrinfo for the LLM
     hung (avatar stops replying), aiortc encoders stalled, and the process could
     not exit on SIGTERM.
  2. AvatarWorker._emit_pixel_block blocked forever in out_q.put() after a session
     ended mid-block (end_session drains the queue once), so the worker never went
     back to idle for the next session.
  3. VideoRing allocated ~590 MB per session up front, though the webcam ring is
     never written.
  4. Mic frames arriving with a few ms of jitter were zero-filled by the ring,
     chopping the start of speech (VAD missed the first words of a turn).

    python -u voicechat/tests/test_session_lifecycle.py
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "avatarforcing"))

from voicechat.rings import AudioRing, VideoRing


def test_audio_track_releases_executor_thread():
    from aiortc.mediastreams import MediaStreamError
    from voicechat.server import AvatarAudioTrack

    async def run():
        loop = asyncio.get_running_loop()
        sess = types.SimpleNamespace(id="t", now=lambda: 0.0, worker=None, lipsync_offset_s=0.0,
                                     avatar_audio_pcm_ring=AudioRing(5.0, 16000))
        for _ in range(5):                       # more sessions than executor threads
            track = AvatarAudioTrack(sess)
            track._thread_stop.set()             # pacer stops producing
            track._thread.join(timeout=1)
            while not track._out_q.empty():
                track._out_q.get_nowait()
            pending = asyncio.ensure_future(track.recv())   # blocks in the executor
            await asyncio.sleep(0.05)
            track.stop()
            try:
                await asyncio.wait_for(pending, timeout=2.0)
                raise AssertionError("recv() returned a frame after stop()")
            except MediaStreamError:
                pass
        # the executor must still be usable
        assert await asyncio.wait_for(loop.run_in_executor(None, lambda: 42), timeout=2.0) == 42

    # A private loop + 2-thread default executor, closed without waiting: with the bug,
    # the leaked threads never finish, and asyncio.run() would hang joining them.
    loop = asyncio.new_event_loop()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    loop.set_default_executor(pool)
    try:
        loop.run_until_complete(run())
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        loop.close()


def test_emit_gives_up_when_session_ends():
    import torch
    from voicechat.avatar_worker import AvatarWorker
    w = AvatarWorker(af_model=None, avatar_ref_tensor=torch.zeros(1, 3, 8, 8), device="cpu")
    w.out_q = queue.Queue(maxsize=2)            # nobody consumes
    pixels = torch.zeros(1, 10, 3, 8, 8)
    t = threading.Thread(target=w._emit_pixel_block, args=(pixels, 0.0), daemon=True)
    t.start()
    time.sleep(0.3)
    assert t.is_alive(), "should be blocked on the full queue while the session is live"
    w._session_end.set()
    t.join(timeout=1.0)
    assert not t.is_alive(), "_emit_pixel_block still blocked after the session ended"


def test_video_ring_allocates_lazily():
    r = VideoRing(capacity_seconds=30.0, fps=25)
    assert r._buf is None
    assert r.read_window(0.0, 0.2).shape == (5, 512, 512, 3)      # reads before any write: zeros
    r.write(np.full((512, 512, 3), 7, np.uint8), ts=0.0)
    assert r._buf is not None and r.read_window(0.0, 0.04)[0, 0, 0, 0] == 7


def test_mic_jitter_is_not_zero_filled():
    from voicechat.server import mic_start_ts
    ring = AudioRing(capacity_seconds=10.0, sample_rate=16000)
    rng = np.random.default_rng(0)
    chunk = np.ones(320, np.float32)               # 20 ms frames of non-silence
    for n in range(1, 101):
        arrival = n * 0.02 + rng.uniform(0, 0.008)  # realtime cadence + arrival jitter
        ring.write(chunk, start_ts=mic_start_ts(ring, arrival - 0.02))
    a = ring.read_window(0.0, ring.write_ts)
    assert (a == 0).sum() == 0, f"{(a == 0).sum()} zero samples inserted by jitter"
    ring.write(chunk, start_ts=mic_start_ts(ring, ring.write_ts + 0.5))   # a real 500 ms dropout
    assert abs(ring.write_ts - (2.0 + 0.5 + 0.02)) < 0.2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed", flush=True)
    # os._exit: with the executor bug present, leaked threads would block interpreter exit.
    import os
    os._exit(1 if failed else 0)
