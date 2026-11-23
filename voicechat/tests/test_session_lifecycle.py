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
  5. Barge-in only stopped phrases not yet scheduled; LLM + TTS run faster than
     realtime, so the whole reply was usually already in the rings and kept playing.
  6. The avatar's audio rings were fixed 30 s circular buffers, but a whole reply is
     scheduled ahead of playback: a long one overwrote audio not yet played, and the
     avatar went silent mid-reply.

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


def test_barge_in_cuts_scheduled_audio():
    import torch
    from voicechat.avatar_worker import AvatarWorker
    from voicechat.rings import FeatureRing
    w = AvatarWorker(af_model=None, avatar_ref_tensor=torch.zeros(1, 3, 8, 8), device="cpu")
    w.pcm_ring, w.af_ring = AudioRing(30.0, 16000), FeatureRing(30.0, 25, 512)
    w.playhead = lambda: 1.0
    w.gen_horizon = 1.0
    def phrase(seconds):
        return np.ones((int(seconds * 25), 512), np.float32), np.ones(int(seconds * 16000), np.float32)
    w._schedule_phrase(w._tts_epoch, "a", *phrase(4.0), 0.0, "test")   # reply already in the rings
    w._schedule_phrase(w._tts_epoch, "b", *phrase(4.0), 0.0, "test")
    epoch = w._tts_epoch                                                # wav2vec of "c" in flight
    w.submit_tts("d", np.ones(16000, np.float32))                       # queued, not yet processed
    w.barge_in(2.5)
    w._schedule_phrase(epoch, "c", *phrase(2.0), 0.0, "test")           # "c" finishes after the cut
    assert abs(w.pcm_ring.write_ts - 2.5) < 1e-6 and abs(w.af_ring.write_ts - 2.5) < 0.04
    assert w.tts_in_q.empty(), "queued phrase survived barge-in"
    assert not w.pcm_ring.read_window(2.5, 9.0).any(), "audio after the cut still plays"
    w._schedule_phrase(w._tts_epoch, "next reply", *phrase(1.0), 0.0, "test")
    assert w.pcm_ring.write_ts > 2.5, "the next reply after a barge-in isn't scheduled"


def test_avatar_rings_hold_a_long_reply():
    from voicechat.rings import FeatureRing
    pcm, feats = AudioRing(None, 16000), FeatureRing(None, 25, 512)
    for k in range(40):                                  # a 120 s reply, written 1 s into the session
        pcm.write(np.full(3 * 16000, k + 1, np.float32), start_ts=1.0 + 3 * k)
        feats.write(np.full((75, 512), k + 1, np.float32), start_ts=1.0 + 3 * k)
    assert (pcm.read_window(1.0, 4.0) == 1).all(), "start of the reply was overwritten"
    assert (pcm.read_window(118.0, 121.0) == 40).all() and (feats.read_window(118.0, 121.0) == 40).all()
    pcm.release_before(100.0)                            # played: freed; the rest still plays
    assert not pcm.read_window(1.0, 4.0).any() and (pcm.read_window(100.0, 103.0) == 34).all()
    pcm.release_before(500.0)                            # idle a long time, then the next reply
    pcm.write(np.ones(16000, np.float32), start_ts=600.0)
    assert pcm._buf.shape[0] < 60 * 16000, "idle gap was zero-filled into the buffer"
    assert (pcm.read_window(600.0, 601.0) == 1).all()


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
