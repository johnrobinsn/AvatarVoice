"""
VAD + end-of-turn detection on the user_audio_ring.

Silero VAD runs on each 32 ms chunk (very cheap, ~0.5 ms).
Smart Turn v3.2 ONNX runs on the trailing 8 s of audio whenever VAD transitions
from speech→silence, to decide whether the user is actually done talking.

The worker emits:
  on_start_of_speech(ts)   - called the moment VAD first detects speech
  on_turn_complete(audio)  - called when Smart Turn says the turn is done
                             (audio = the full speech segment as f32 PCM 16kHz)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

import numpy as np
import onnxruntime as ort
import librosa  # for mel features

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voicechat.rings import AudioRing

log = logging.getLogger("af-vad")


@dataclass
class TurnDetectorConfig:
    chunk_samples: int = 512          # 32 ms at 16 kHz; Silero native
    # Energy gate ahead of Silero. Chunks with RMS below this are forced
    # to prob=0 regardless of Silero output — kills low-volume mic-leaked
    # avatar audio that AEC doesn't fully cancel. Was 0.015 (quiet room,
    # headphones); 0.14 rejects much more speaker bleed on a laptop but needs
    # a reasonably loud voice at the mic. Tunable live in the client's gear panel.
    rms_gate: float = 0.14
    speech_prob_threshold: float = 0.6      # per-chunk confidence cutoff (was 0.7; originally 0.5, raised so low-volume mic-leaked avatar audio is below the bar)
    # Transition thresholds (hysteresis):
    enter_speech_frames: int = 15     # 15 × 32 ms ≈ 480 ms of sustained (speech_prob ≥ threshold) before we fire SOS
    exit_speech_frames: int = 20      # need N consecutive silence chunks to maybe end turn
    # Smart Turn:
    smart_turn_window_s: float = 8.0  # how much trailing audio to feed Smart Turn
    smart_turn_prob_threshold: float = 0.5
    # How often to re-check smart turn once we're in "trailing silence"
    smart_turn_recheck_interval_s: float = 0.25


@dataclass
class _State:
    in_speech: bool = False
    n_speech_frames_in_window: int = 0
    n_silence_frames_in_window: int = 0
    speech_start_ts: Optional[float] = None
    last_smart_turn_check_ts: float = 0.0


class VADWorker:
    """Background asyncio task; polls `user_audio_ring`, calls event handlers."""

    def __init__(
        self,
        audio_ring: AudioRing,
        smart_turn_onnx_path: str,
        on_start_of_speech: Callable[[float], Awaitable[None]],
        on_turn_complete:  Callable[[np.ndarray, float, float], Awaitable[None]],
        config: TurnDetectorConfig = TurnDetectorConfig(),
        logger_tag: str = "vad",
    ):
        self.audio_ring = audio_ring
        self.on_start_of_speech = on_start_of_speech
        self.on_turn_complete = on_turn_complete
        self.cfg = config
        self.tag = logger_tag
        self.state = _State()

        # Load Silero VAD (via its pip package which wraps the ONNX)
        from silero_vad import load_silero_vad
        self._silero = load_silero_vad()
        self._silero.reset_states()

        # Load Smart Turn ONNX
        self._smart = ort.InferenceSession(smart_turn_onnx_path, providers=["CPUExecutionProvider"])

        # Sliding read cursor through the audio ring
        self._read_ts: float = 0.0

    async def run(self) -> None:
        log.info(f"[{self.tag}] VAD worker started (chunk={self.cfg.chunk_samples} samples)")
        import torch  # silero expects torch tensor
        sr = self.audio_ring.sample_rate
        chunk_s = self.cfg.chunk_samples / sr
        while True:
            # Wait until the ring has at least one chunk past our read cursor
            while self.audio_ring.write_ts < self._read_ts + chunk_s:
                await asyncio.sleep(chunk_s / 2)
            # Read exactly one chunk
            chunk = self.audio_ring.read_window(self._read_ts, self._read_ts + chunk_s)
            if chunk.size < self.cfg.chunk_samples:
                # shouldn't happen past the initial frames
                await asyncio.sleep(chunk_s / 2)
                continue
            chunk = chunk[:self.cfg.chunk_samples]
            self._read_ts += chunk_s

            # RMS gate ahead of Silero. If the mic chunk is too quiet,
            # force prob to 0 — avoids Silero mistaking low-volume avatar
            # audio bleeding through AEC as speech.
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            if rms < self.cfg.rms_gate:
                prob = 0.0
            else:
                prob = float(self._silero(torch.from_numpy(chunk), sr).item())
            is_speech = prob > self.cfg.speech_prob_threshold

            if is_speech:
                self.state.n_speech_frames_in_window += 1
                self.state.n_silence_frames_in_window = 0
            else:
                self.state.n_silence_frames_in_window += 1
                self.state.n_speech_frames_in_window = 0

            if not self.state.in_speech:
                # Entering speech?
                if self.state.n_speech_frames_in_window >= self.cfg.enter_speech_frames:
                    self.state.in_speech = True
                    # Rewind start to account for the chunks we already consumed proving speech
                    self.state.speech_start_ts = self._read_ts - chunk_s * self.cfg.enter_speech_frames
                    log.info(f"[{self.tag}] start-of-speech @ {self.state.speech_start_ts:.2f}")
                    asyncio.create_task(self.on_start_of_speech(self.state.speech_start_ts))
            else:
                # In speech; wait for sustained silence then ask Smart Turn
                if self.state.n_silence_frames_in_window >= self.cfg.exit_speech_frames:
                    # Rate-limit Smart Turn checks
                    if (self._read_ts - self.state.last_smart_turn_check_ts
                            >= self.cfg.smart_turn_recheck_interval_s):
                        self.state.last_smart_turn_check_ts = self._read_ts
                        done = self._smart_turn_check(self._read_ts)
                        if done:
                            start_ts = self.state.speech_start_ts
                            end_ts = self._read_ts
                            seg = self.audio_ring.read_window(start_ts, end_ts)
                            log.info(f"[{self.tag}] turn-complete [{start_ts:.2f}, {end_ts:.2f}] "
                                     f"= {end_ts - start_ts:.2f}s audio")
                            asyncio.create_task(self.on_turn_complete(seg, start_ts, end_ts))
                            # Reset
                            self.state = _State()

    def _smart_turn_check(self, end_ts: float) -> bool:
        """Return True if Smart Turn says the turn is complete at `end_ts`."""
        sr = self.audio_ring.sample_rate
        start_ts = max(0.0, end_ts - self.cfg.smart_turn_window_s)
        audio = self.audio_ring.read_window(start_ts, end_ts)
        # Right-pad to the exact 8 s expected by the model
        target = int(self.cfg.smart_turn_window_s * sr)
        if audio.size < target:
            audio = np.pad(audio, (0, target - audio.size))
        else:
            audio = audio[-target:]
        # Whisper-style log-mel: 80 bins, 400 fft, 160 hop → ~800 frames for 8s
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=80, n_fft=400, hop_length=160, win_length=400, center=False,
        )
        log_mel = np.log(np.clip(mel, 1e-10, None)).astype(np.float32)
        log_mel = np.clip(log_mel, log_mel.max() - 8.0, None)
        log_mel = (log_mel + 4.0) / 4.0
        # Pad/trim to exactly 800 frames
        if log_mel.shape[1] < 800:
            log_mel = np.pad(log_mel, ((0, 0), (0, 800 - log_mel.shape[1])))
        log_mel = log_mel[:, :800]
        inp = log_mel[None, ...]  # [1, 80, 800]
        logit = float(self._smart.run(None, {"input_features": inp})[0].flatten()[0])
        prob = 1.0 / (1.0 + np.exp(-logit))
        log.debug(f"[{self.tag}] smart-turn prob={prob:.3f} (threshold={self.cfg.smart_turn_prob_threshold})")
        return prob > self.cfg.smart_turn_prob_threshold


if __name__ == "__main__":
    # Smoke test: stream data/user.wav through a ring, see turn-complete fire.
    import librosa as _librosa
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ROOT = Path(__file__).resolve().parent.parent

    async def main():
        ring = AudioRing(capacity_seconds=30.0, sample_rate=16000)
        events = []
        async def on_sos(ts):
            events.append(("sos", ts))
        async def on_done(audio, s, e):
            events.append(("turn", s, e, audio.shape))
        vad = VADWorker(
            audio_ring=ring,
            smart_turn_onnx_path=str(ROOT / "voicechat/models/smart-turn-v3.2-cpu.onnx"),
            on_start_of_speech=on_sos,
            on_turn_complete=on_done,
        )
        # Feed user.wav into the ring at ~realtime
        wav, _ = _librosa.load(str(ROOT / "data/user.wav"), sr=16000, mono=True)
        async def feeder():
            chunk_s = 0.02   # 20 ms chunks
            n = int(16000 * chunk_s)
            t = 0.0
            for i in range(0, wav.shape[0] - n, n):
                ring.write(wav[i:i+n].astype(np.float32), start_ts=t)
                t += chunk_s
                await asyncio.sleep(chunk_s)
            # trailing silence so VAD can exit speech
            await asyncio.sleep(2.0)
        vad_task = asyncio.create_task(vad.run())
        feed_task = asyncio.create_task(feeder())
        try:
            await asyncio.wait_for(feed_task, timeout=30)
        except asyncio.TimeoutError:
            pass
        await asyncio.sleep(1.0)
        vad_task.cancel()
        print("\nEvents:")
        for e in events:
            print(" ", e)
    asyncio.run(main())
