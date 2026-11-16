"""
STT worker — faster-whisper distil-large-v3 wrapped as an asyncio-callable.

Runs on a dedicated single-threaded executor so ctranslate2's state is not
shared across threads (CT2 is not thread-safe for concurrent `encode` calls).

Also sets LD_LIBRARY_PATH to our venv's pip-installed CUDA 12 libs because
ctranslate2 is built against CUDA 12 and we have CUDA 13 on the system.

Usage:
    stt = WhisperSTT()
    text = await stt.transcribe(audio_f32_16k)

NOTE: we tried swapping to NVIDIA Parakeet-TDT-0.6b-v2 via NeMo for parity
with macvoice, but NeMo #16325 (open PR #16330 as of 11/2026) crashes the
CUDA-graph decoder with illegal-memory-access on the second transcribe() call
after any empty_cache() runs in the process. Reverted. Revisit once that PR
lands, or try the ONNX Runtime port of Parakeet.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# Preload CUDA 12 cuBLAS + cuDNN via ctypes so ctranslate2's dlopen finds them
# in-process (setting os.environ['LD_LIBRARY_PATH'] here is a no-op — ld.so
# only reads the env at process start).
_ROOT = Path(__file__).resolve().parent.parent
_venv = _ROOT / ".venv/lib/python3.11/site-packages"
import ctypes
for _p in [_venv / "nvidia/cublas/lib/libcublas.so.12",
           _venv / "nvidia/cudnn/lib/libcudnn.so.9"]:
    if _p.exists():
        ctypes.CDLL(str(_p), mode=ctypes.RTLD_GLOBAL)

log = logging.getLogger("af-stt")


class WhisperSTT:
    def __init__(self, model_id: str = "distil-large-v3", device: str = "cuda:0",
                 compute_type: str = "int8_float16"):
        from faster_whisper import WhisperModel
        log.info(f"loading {model_id} on {device}/{compute_type}")
        t0 = time.perf_counter()
        # ctranslate2 wants device='cuda' and device_index for the GPU number
        if device.startswith("cuda:"):
            device_index = int(device.split(":")[1])
            self._model = WhisperModel(model_id, device="cuda",
                                        device_index=device_index,
                                        compute_type=compute_type)
        else:
            self._model = WhisperModel(model_id, device=device, compute_type=compute_type)
        log.info(f"loaded in {time.perf_counter()-t0:.2f}s")
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="af-stt")

    def _blocking_transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        t0 = time.perf_counter()
        segs, info = self._model.transcribe(
            audio, language=language, beam_size=1, vad_filter=False,
        )
        text = "".join(s.text for s in segs).strip()
        dt = (time.perf_counter() - t0) * 1000
        log.info(f"stt: {audio.shape[0]/16000:.2f}s audio -> {dt:.1f}ms  text={text!r}")
        return text

    async def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        """Audio: f32 PCM mono at 16 kHz."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._blocking_transcribe, audio, language)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import librosa
    async def main():
        stt = WhisperSTT()
        wav, _ = librosa.load(str(_ROOT / "data/user.wav"), sr=16000, mono=True)
        for i, (s, e) in enumerate([(0, 6.78), (6.9, 11.2)]):
            seg = wav[int(s*16000):int(e*16000)]
            t = await stt.transcribe(seg)
            print(f"turn {i}: {t!r}")
    asyncio.run(main())
