"""
TTS worker: Phonon 31m-en wrapped as an asyncio-callable streaming synth.

Phonon's native output is 24 kHz f32 mono. For the avatar's wav2vec we need
16 kHz, and for the browser's WebRTC audio track we want 48 kHz s16 stereo.
We resample lazily on demand at the consumption points.

Interface:
    tts = PhononTTS()
    async for chunk_24k in tts.stream("Hello there."):
        # chunk_24k is a 1-D np.float32 at 24 kHz
        ...
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from typing import AsyncIterator, Iterator

import numpy as np
import ptts
import re as _re

log = logging.getLogger("af-tts")


# Phonon's digit pronunciation drags syllables and sounds unnatural
# ("13," synthesizes at 1.84 s vs. "thirteen," at 1.52 s and clearly mangled).
# Expand integers in [0, 9999] to their word form before handing text to the
# synth. Outside that range we leave alone — Phonon handles very large
# numbers reasonably if the caller has already spelled them out.
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]

def _int_to_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, u = divmod(n, 10)
        return _TENS[t] if u == 0 else f"{_TENS[t]}-{_ONES[u]}"
    if n < 1000:
        h, r = divmod(n, 100)
        return f"{_ONES[h]} hundred" + (f" {_int_to_words(r)}" if r else "")
    if n < 10000:
        th, r = divmod(n, 1000)
        return f"{_ONES[th]} thousand" + (f" {_int_to_words(r)}" if r else "")
    return str(n)

_INT_RE = _re.compile(r"\b\d+\b")

def _expand_numbers(text: str) -> str:
    def sub(m: _re.Match) -> str:
        n = int(m.group(0))
        return _int_to_words(n) if n < 10000 else m.group(0)
    return _INT_RE.sub(sub, text)


class PhononTTS:
    # Named voices from the default Phonon catalog
    # (hf://gradium/phonon-voices). "Miles" is resonant/decisive, "Harper" is
    # the stock friendly American female. Changeable at runtime via set_voice.
    DEFAULT_VOICE_FEMALE = "Harper"
    DEFAULT_VOICE_MALE   = "Miles"

    def __init__(self, lang: str = "en", voice: str = None):
        import os
        self.voice = voice or os.environ.get("AF_TTS_VOICE", self.DEFAULT_VOICE_FEMALE)
        log.info(f"loading Phonon ({lang}, voice={self.voice})")
        t0 = time.perf_counter()
        self._tts = ptts.TTS(lang=lang)
        self.sample_rate = self._tts.sample_rate   # 24000
        log.info(f"loaded in {time.perf_counter()-t0:.2f}s sr={self.sample_rate} voice={self.voice}")
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="af-tts")

    def set_voice(self, voice: str) -> None:
        """Change the active voice. New phrases submitted after this call use
        the new voice; in-flight generations are unaffected."""
        if voice and voice != self.voice:
            log.info(f"tts voice: {self.voice!r} → {voice!r}")
            self.voice = voice

    def _blocking_stream(self, text: str) -> Iterator[np.ndarray]:
        """Produce chunks of f32 PCM at self.sample_rate. Called from the worker thread."""
        text = _expand_numbers(text)
        for chunk in self._tts.stream(text, voice=self.voice):
            arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if arr.size:
                yield arr

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        """Async wrapper: yields chunks of f32 PCM at Phonon's native sample rate."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=32)

        def producer():
            try:
                t0 = time.perf_counter()
                first_t = None
                total_samples = 0
                for chunk in self._blocking_stream(text):
                    if first_t is None:
                        first_t = time.perf_counter() - t0
                    total_samples += chunk.shape[0]
                    asyncio.run_coroutine_threadsafe(q.put(chunk), loop).result()
                asyncio.run_coroutine_threadsafe(q.put(None), loop).result()
                total_dt = time.perf_counter() - t0
                audio_dur = total_samples / self.sample_rate
                log.info(f"tts: text={text[:80]!r}  TTFA={first_t*1000 if first_t else -1:.0f}ms  "
                         f"produced {audio_dur:.2f}s audio in {total_dt*1000:.0f}ms (RTF={total_dt/max(audio_dur,1e-6):.2f})")
            except Exception:
                log.exception("tts producer crashed")
                asyncio.run_coroutine_threadsafe(q.put(None), loop).result()

        self._executor.submit(producer)
        while True:
            chunk = await q.get()
            if chunk is None:
                break
            yield chunk


# ---- Fast resampler (polyphase via scipy or np) -------------------------------
def resample_24k_to_16k(audio_24k: np.ndarray) -> np.ndarray:
    """Phonon 24k → wav2vec 16k. Ratio 2:3."""
    from scipy import signal
    return signal.resample_poly(audio_24k, up=2, down=3).astype(np.float32)


def resample_24k_to_48k(audio_24k: np.ndarray) -> np.ndarray:
    """Phonon 24k → WebRTC 48k. Ratio 2:1."""
    from scipy import signal
    return signal.resample_poly(audio_24k, up=2, down=1).astype(np.float32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    async def main():
        tts = PhononTTS()
        # Warm
        async for _ in tts.stream("Hello."): pass
        # Measured
        for text in ["Nice to meet you.", "The quick brown fox jumps over the lazy dog."]:
            async for _ in tts.stream(text): pass
    asyncio.run(main())
