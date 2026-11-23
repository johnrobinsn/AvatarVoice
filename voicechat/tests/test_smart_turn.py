"""
Turn detection: VAD start-of-speech and the Smart Turn end-of-turn check (no GPU
needed; Phonon TTS makes the speech).

Start of speech must fire for choppy speech too ("wait, hold on, please stop…"):
it used to need 15 consecutive speech chunks, which the gaps between words broke,
so barge-ins were missed.

VADWorker._smart_turn_check must say "done" for a finished question and "not done"
for one cut off mid-sentence. Two bugs made it say "done" for everything:
  1. the ONNX graph already ends in a Sigmoid (its output is named "logits"), and
     vad.py applied a second one, squashing every probability into ~[0.5, 0.73];
  2. the log-mel features didn't match the model's training recipe (Whisper's
     feature extractor, with do_normalize, speech at the end of the window), so the
     model scored everything ~0.98 regardless.

    python -u voicechat/tests/test_smart_turn.py
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from voicechat.rings import AudioRing
from voicechat.vad import VADWorker

SR = 16000
QUESTION = "Can you tell me a quick fun fact about octopuses and how they change color?"


async def _noop(*_):
    pass


@lru_cache
def _worker() -> VADWorker:
    return VADWorker(AudioRing(30.0, SR), str(ROOT / "voicechat/models/smart-turn-v3.2-cpu.onnx"),
                     on_start_of_speech=_noop, on_turn_complete=_noop)


@lru_cache
def _say(text: str) -> np.ndarray:
    import librosa
    import ptts
    a = np.concatenate([np.asarray(c, np.float32).reshape(-1)
                        for c in ptts.TTS(lang="en").stream(text, voice="Miles")])
    a = librosa.resample(a, orig_sr=24000, target_sr=SR)
    return a / (np.abs(a).max() + 1e-8) * 0.9


def _question() -> np.ndarray:
    return _say(QUESTION)


def test_choppy_interrupt_starts_speech():
    import asyncio
    speech = _say("Wait, hold on, please stop talking for a moment.")
    ring = AudioRing(30.0, SR)
    ring.write(np.concatenate([np.zeros(SR, np.float32), speech, np.zeros(SR, np.float32)]), start_ts=0.0)
    starts = []

    async def run():
        async def on_sos(ts):
            starts.append(ts)
        w = VADWorker(ring, str(ROOT / "voicechat/models/smart-turn-v3.2-cpu.onnx"),
                      on_start_of_speech=on_sos, on_turn_complete=_noop)
        task = asyncio.create_task(w.run())
        for _ in range(100):
            await asyncio.sleep(0.02)
            if starts or w._read_ts >= ring.write_ts - 0.1:
                break
        task.cancel()
    asyncio.run(run())
    assert starts, "start-of-speech never fired"
    assert starts[0] <= 1.0, f"turn audio starts at {starts[0]:.2f}s, after the speech began (1.0s)"


def _turn_done(speech: np.ndarray, trailing_silence_s: float = 0.1) -> bool:
    """Feed the mic ring like a live session (quiet room, the speech, a short pause) and ask Smart Turn."""
    rng = np.random.default_rng(0)
    quiet = lambda s: (rng.standard_normal(int(s * SR)) * 0.002).astype(np.float32)
    w = _worker()
    w.audio_ring = ring = AudioRing(30.0, SR)
    ring.write(np.concatenate([quiet(8.0), speech, quiet(trailing_silence_s)]), start_ts=0.0)
    return w._smart_turn_check(ring.write_ts)


def test_complete_question_ends_turn():
    assert _turn_done(_question()), "finished question was judged incomplete"


def test_question_cut_mid_sentence_does_not_end_turn():
    q = _question()
    assert not _turn_done(q[: q.size // 2]), "question cut off mid-sentence was judged complete"


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
    sys.exit(1 if failed else 0)
