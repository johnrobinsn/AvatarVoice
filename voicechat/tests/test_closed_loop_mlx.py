"""
Closed-loop integration test for the Apple-Silicon backend — no browser, no WebRTC.

Drives the full voice-chat chain with the MLX worker:

    synthetic "mic" (a Phonon-spoken question, played at wall-clock pace)
      -> Silero VAD + Smart Turn -> Parakeet STT -> LLM -> Phonon TTS
      -> MLXAvatarWorker (wav2vec, MLX flow transformer, MPS decoder) -> frames

then writes results/closed_loop_mlx.mp4 with the avatar video and the avatar
audio ring aligned by their shared timeline, and prints latency + timing stats.

Needs an OpenAI-compatible LLM server (default http://localhost:8000/v1), e.g.

    .venv-llm/bin/mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8000

Run:
    .venv/bin/python -u voicechat/tests/test_closed_loop_mlx.py ["question text"]
"""
from __future__ import annotations

import asyncio
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "avatarforcing"))
sys.path.insert(0, str(ROOT / "mlx_port"))
os.environ.setdefault("AF_BACKEND", "mlx")

from voicechat import server as S
from voicechat.rings import AudioRing, VideoRing, FeatureRing
from voicechat.vad import VADWorker, TurnDetectorConfig
from voicechat.reply_pipeline import reply_pipeline
from voicechat.tts import resample_24k_to_16k

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
log = logging.getLogger("af-test")

QUESTION = "Hey there! What's a fun thing to do on a rainy Saturday afternoon?"


class MockSession:
    def __init__(self):
        self.t0 = time.monotonic()
        self.user_audio_ring = AudioRing(capacity_seconds=60.0, sample_rate=16000)
        self.user_video_ring = VideoRing(capacity_seconds=60.0, fps=25)
        self.avatar_audio_feat_ring = FeatureRing(capacity_seconds=60.0, fps=25, dim=512)
        self.avatar_audio_pcm_ring = AudioRing(capacity_seconds=60.0, sample_rate=16000)
        self.out_video_q: queue.Queue = queue.Queue(maxsize=5)

    def now(self):
        return time.monotonic() - self.t0


def speak_question(text: str) -> np.ndarray:
    """The question as 16 kHz mic audio (a different voice from the avatar's),
    at a typical mic level, with lead-in and trailing silence."""
    chunks = list(S.TTS._blocking_stream(text))
    a = resample_24k_to_16k(np.concatenate(chunks))
    a = a / (np.abs(a).max() + 1e-8) * 0.5
    sil = lambda s: np.zeros(int(s * 16000), np.float32)
    return np.concatenate([sil(1.0), a.astype(np.float32), sil(3.0)])


async def feed_mic(session: MockSession, pcm: np.ndarray, marks: dict):
    n = 320   # 20 ms
    t_start = time.perf_counter()
    for i in range(0, len(pcm) - n + 1, n):
        target = t_start + i / 16000
        await asyncio.sleep(max(0.0, target - time.perf_counter()))
        ring = session.user_audio_ring
        ring.write(pcm[i:i + n], start_ts=S.mic_start_ts(ring, session.now() - 0.02))
    marks["mic_done"] = session.now()


def drain_video(session: MockSession, frames: list, stop: threading.Event):
    while not stop.is_set():
        try:
            item = session.out_video_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if item is None:
            break
        frames.append((item[0], item[1], session.now()))


async def main():
    question = sys.argv[1] if len(sys.argv) > 1 else QUESTION
    t0 = time.perf_counter()
    S.init_singletons()
    log.info(f"=== singletons ready in {time.perf_counter() - t0:.1f}s ===")
    worker = S.AVATAR_WORKER

    session = MockSession()
    frames: list = []
    stop = threading.Event()
    drainer = threading.Thread(target=drain_video, args=(session, frames, stop), daemon=True)
    drainer.start()
    worker.start_session(session.user_video_ring, session.avatar_audio_feat_ring,
                         session.avatar_audio_pcm_ring, session.out_video_q, session.now)

    marks: dict = {}
    reply_done = asyncio.Event()

    async def on_sos(ts):
        marks.setdefault("sos", ts)
        log.info(f"[test] start of speech @ {ts:.2f}")

    async def on_done(audio, start_ts, end_ts):
        marks["turn_end"] = session.now()
        log.info(f"[test] turn complete: {start_ts:.2f}-{end_ts:.2f} (detected @ {marks['turn_end']:.2f})")
        text = await S.STT.transcribe(audio)
        marks["stt_done"] = session.now()
        marks["transcript"] = text

        def on_phrase(p):
            marks.setdefault("first_phrase", session.now())
        await reply_pipeline(text, S.LLM, S.TTS, None,
                             feat_ring=session.avatar_audio_feat_ring,
                             pcm_ring=session.avatar_audio_pcm_ring,
                             now_fn=session.now, worker=worker, on_phrase=on_phrase)
        marks["reply_done"] = session.now()
        reply_done.set()

    vad = VADWorker(audio_ring=session.user_audio_ring,
                    smart_turn_onnx_path=str(ROOT / "voicechat/models/smart-turn-v3.2-cpu.onnx"),
                    on_start_of_speech=on_sos, on_turn_complete=on_done,
                    # the synthetic mic voice is quieter than a close mic with AGC; use a
                    # lower energy gate than the server default (see test_webrtc_client.py)
                    config=TurnDetectorConfig(rms_gate=0.02))
    vad_task = asyncio.create_task(vad.run())

    mic = speak_question(question)
    log.info(f"[test] mic: {len(mic) / 16000:.1f}s question {question!r}")
    await asyncio.sleep(2.5)     # let the worker's primer + first blocks run
    await feed_mic(session, mic, marks)
    try:
        await asyncio.wait_for(reply_done.wait(), timeout=60)
    except asyncio.TimeoutError:
        log.error("[test] no reply within 60 s")
    # Let the avatar finish speaking what's scheduled, plus a little idle.
    while session.now() < session.avatar_audio_pcm_ring.write_ts + 1.5:
        await asyncio.sleep(0.2)
    vad_task.cancel()
    worker.end_session()
    stop.set()
    drainer.join(timeout=2)

    # ---- report ----
    pcm_ring = session.avatar_audio_pcm_ring
    first_audio_ts = None
    if pcm_ring.write_ts > 0:
        pcm_all = pcm_ring.read_window(0.0, pcm_ring.write_ts)
        voiced = np.nonzero(np.abs(pcm_all) > 0.01)[0]
        if voiced.size:
            first_audio_ts = voiced[0] / 16000
    ts = np.array([f[0] for f in frames])
    gaps = np.diff(ts) if len(ts) > 1 else np.array([0.04])
    print("\n=== closed loop (mlx) ===")
    print(f"transcript: {marks.get('transcript')!r}")
    if "turn_end" in marks:
        te = marks["turn_end"]
        print(f"turn detected  +{(te - marks['mic_done'] + 3.0):.2f}s after the user stopped talking")
        print(f"STT            {(marks['stt_done'] - te) * 1000:.0f} ms")
        if "first_phrase" in marks:
            print(f"first phrase   {(marks['first_phrase'] - te) * 1000:.0f} ms after turn end (LLM + TTS + wav2vec)")
        if first_audio_ts is not None:
            print(f"avatar speaks  @ {first_audio_ts:.2f}s session time = "
                  f"{(first_audio_ts - te) * 1000:.0f} ms after turn end")
    print(f"frames: {len(frames)}  logical span {ts[-1] - ts[0]:.1f}s  "
          f"max ts gap {gaps.max() * 1000:.0f} ms  (40 ms = no drops)")
    print(f"worker EWMA {worker.block_time_ewma * 1000:.0f} ms/block  (400 ms = realtime)")

    out = ROOT / "results" / "closed_loop_mlx.mp4"
    out.parent.mkdir(exist_ok=True)
    import run_stream
    video = np.stack([f[1] for f in frames])
    audio = pcm_ring.read_window(ts[0], ts[0] + len(frames) / 25)
    run_stream.write_video(str(out), video, audio)
    print(f"wrote {out}")
    await S.LLM.close()
    worker.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
