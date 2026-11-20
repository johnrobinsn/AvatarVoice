"""
STT worker — faster-whisper distil-large-v3 wrapped as an asyncio-callable.

Runs on a dedicated single-threaded executor so ctranslate2's state is not
shared across threads (CT2 is not thread-safe for concurrent `encode` calls).

Also sets LD_LIBRARY_PATH to our venv's pip-installed CUDA 12 libs because
ctranslate2 is built against CUDA 12 and we have CUDA 13 on the system.

Usage:
    stt = WhisperSTT()
    text = await stt.transcribe(audio_f32_16k)

For parity with the Mac/MLX backend (which runs Parakeet-TDT-0.6b-v2 via
parakeet-mlx), the CUDA server can opt into the same weights via the
onnxruntime-based `ParakeetSTT` below. Enable with `AF_STT=parakeet`.
Parakeet loads in ~2 s (vs Whisper's ~15 s) and emits streaming-friendly
RNN-T output — it's a startup + latency win.

Historical: an earlier attempt to use NVIDIA Parakeet via NeMo crashed the
CUDA-graph decoder with an illegal-memory-access on the second transcribe()
call after any empty_cache() (NeMo issue #16325, open PR #16330). The
onnx-asr + onnxruntime-gpu path sidesteps that entirely.
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

_ROOT = Path(__file__).resolve().parent.parent
_venv = _ROOT / ".venv/lib/python3.11/site-packages"
import ctypes


def _preload_whisper_cuda_libs():
    """Preload the venv's CUDA 12 cuBLAS + cuDNN via ctypes so ctranslate2's
    dlopen finds them in-process (setting os.environ['LD_LIBRARY_PATH'] here
    is a no-op — ld.so only reads the env at process start). Called from
    WhisperSTT.__init__ rather than module-load time so the cuBLAS 12 symbol
    isn't force-loaded for every process — onnxruntime-gpu (ParakeetSTT)
    also needs libcublas.so.12 but sets up its init differently."""
    for _p in [_venv / "nvidia/cublas/lib/libcublas.so.12",
               _venv / "nvidia/cudnn/lib/libcudnn.so.9"]:
        if _p.exists():
            ctypes.CDLL(str(_p), mode=ctypes.RTLD_GLOBAL)


def _preload_parakeet_cuda_libs():
    """Preload the venv's CUDA 12 cuBLAS, cuBLAS-Lt, and cuDNN 9 via ctypes
    so onnxruntime-gpu's dlopen finds them in-process. Without this, the
    first MatMul inside the Parakeet preprocessor session fails with
    `CUBLAS failure 1: the library was not initialized` because
    onnxruntime 1.32 was built against CUDA 12 cuBLAS but torch 2.11 ships
    CUDA 13 cuBLAS as its default — the dynamic loader resolves to the
    wrong version. Preloading the venv's cuBLAS-12 (torch-installed as a
    dependency) via RTLD_GLOBAL puts its symbols in the global namespace
    before ORT dlopen-s its CUDA execution provider library."""
    for _p in [_venv / "nvidia/cublas/lib/libcublasLt.so.12",
               _venv / "nvidia/cublas/lib/libcublas.so.12",
               _venv / "nvidia/cudnn/lib/libcudnn.so.9"]:
        if _p.exists():
            ctypes.CDLL(str(_p), mode=ctypes.RTLD_GLOBAL)

log = logging.getLogger("af-stt")


class WhisperSTT:
    def __init__(self, model_id: str = "distil-large-v3", device: str = "cuda:0",
                 compute_type: str = "int8_float16"):
        _preload_whisper_cuda_libs()
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


class ParakeetSTT:
    """Parakeet-TDT-0.6b-v2 via onnx-asr + onnxruntime-gpu.

    Same interface as WhisperSTT (`await stt.transcribe(audio_f32_16k)`).
    Loads in ~2 s (vs Whisper's ~15 s). Transcription latency on a 2 s
    clip is ~100-150 ms. Uses the same weights as the Mac/MLX backend.

    Requires: onnx-asr[gpu,hub]. Required CUDA libraries (cuBLAS 12 +
    cuDNN 9) are preloaded from the project venv at init time — no
    LD_LIBRARY_PATH gymnastics needed by the user.
    """
    DEFAULT_MODEL_ID = "nemo-parakeet-tdt-0.6b-v2"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cuda:0"):
        _preload_parakeet_cuda_libs()
        import onnx_asr
        log.info(f"loading {model_id} (onnx-asr) on {device}")
        t0 = time.perf_counter()
        # Let onnx-asr pick the provider — it enumerates installed onnxruntime
        # providers and uses CUDA when onnxruntime-gpu is installed. Passing
        # providers=[("CUDAExecutionProvider", {"device_id": N}), ...]
        # explicitly caused onnxruntime 1.32 to reject CUDA and silently fall
        # back to CPU. Control the GPU with CUDA_VISIBLE_DEVICES instead.
        self._model = onnx_asr.load_model(model_id)
        log.info(f"loaded in {time.perf_counter() - t0:.2f}s")
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="af-stt-parakeet")

    def _blocking_transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        t0 = time.perf_counter()
        # onnx-asr accepts a float32 waveform at 16 kHz directly; language is
        # ignored (the TDT v2 model is English-only; v3 is multilingual).
        text = self._model.recognize(audio, sample_rate=16000).strip()
        dt = (time.perf_counter() - t0) * 1000
        log.info(f"stt: {audio.shape[0] / 16000:.2f}s audio -> {dt:.1f}ms  text={text!r}")
        return text

    async def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        """Audio: f32 PCM mono at 16 kHz."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._blocking_transcribe, audio, language)


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
