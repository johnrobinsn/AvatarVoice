"""
Time-indexed ring buffers for the voice-chat demo.

Key invariant: `read_window(t0, t1)` always returns a contiguous tensor of
exactly the requested size. Underflow (reading past write_ts) is handled
in-ring, not by the caller:

    AudioRing  -> zero-pads the missing tail  (silence)
    VideoRing  -> repeats the last-good frame (still face, not a black frame)

This is what makes the event-driven design in ARCHITECTURE.md work without
a state machine: when no task is writing to a ring, the avatar worker just
reads silence / repeated frames and the model naturally transitions mode.

All timestamps are seconds (float64, monotonic wall-clock from time.monotonic()).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


class AudioRing:
    """16 kHz f32 mono audio ring. Thread-safe. Zero-pads underflows."""

    def __init__(self, capacity_seconds: float = 30.0, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.capacity = int(capacity_seconds * sample_rate)
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._write_pos = 0        # index into buffer (mod capacity)
        self._write_ts: float = 0.0  # wall-clock of next unwritten sample
        self._start_ts: float = 0.0  # wall-clock of oldest sample still in buf
        self._lock = threading.Lock()

    @property
    def write_ts(self) -> float:
        with self._lock:
            return self._write_ts

    def write(self, samples: np.ndarray, start_ts: Optional[float] = None) -> None:
        """Append samples. If start_ts is given and > current write_ts we zero-pad
        the gap; if < write_ts we silently overwrite (shouldn't normally happen)."""
        assert samples.dtype == np.float32, f"expected f32, got {samples.dtype}"
        samples = np.ascontiguousarray(samples).ravel()
        with self._lock:
            if start_ts is not None and start_ts > self._write_ts + 0.5 / self.sample_rate:
                # Gap: fill with silence
                gap_n = int(round((start_ts - self._write_ts) * self.sample_rate))
                if gap_n > 0:
                    self._put(np.zeros(gap_n, dtype=np.float32))
            self._put(samples)

    def _put(self, samples: np.ndarray) -> None:
        n = samples.shape[0]
        if n == 0:
            return
        if n >= self.capacity:
            # Overwrite everything; keep only tail that fits. write_ts must
            # advance by the ORIGINAL sample count, not capacity — otherwise a
            # gap-fill larger than the ring loses the extra seconds and all
            # subsequent writes land at the wrong logical ts.
            original_n = n
            samples = samples[-self.capacity:]
            self._buf[:] = samples
            self._write_pos = 0
            self._write_ts += original_n / self.sample_rate
            self._start_ts = self._write_ts - self.capacity / self.sample_rate
            return
        end = self._write_pos + n
        if end <= self.capacity:
            self._buf[self._write_pos:end] = samples
        else:
            k = self.capacity - self._write_pos
            self._buf[self._write_pos:] = samples[:k]
            self._buf[:n - k] = samples[k:]
        self._write_pos = end % self.capacity
        self._write_ts += n / self.sample_rate
        self._start_ts = max(0.0, self._write_ts - self.capacity / self.sample_rate)

    def truncate_at(self, ts: float) -> None:
        """Rewind write_ts to `ts`, discarding any content buffered at ts or
        later. Used for barge-in: future TTS that was pre-scheduled into the
        ring gets cut off at the user's start-of-speech. Reads past the new
        write_ts return zero-padded silence (same as the normal underflow
        path)."""
        with self._lock:
            if ts >= self._write_ts:
                return
            samples_back = int(round((self._write_ts - ts) * self.sample_rate))
            samples_back = min(samples_back, self.capacity)
            self._write_pos = (self._write_pos - samples_back) % self.capacity
            self._write_ts = ts

    def read_window(self, t0: float, t1: float) -> np.ndarray:
        """Return exactly round((t1-t0)*sr) samples covering [t0, t1).
        Missing head (before start_ts) and missing tail (after write_ts)
        are zero-padded."""
        with self._lock:
            n_total = int(round((t1 - t0) * self.sample_rate))
            if n_total <= 0:
                return np.zeros(0, dtype=np.float32)
            out = np.zeros(n_total, dtype=np.float32)

            # Clip the requested window to what the ring actually holds
            avail_t0 = max(t0, self._start_ts)
            avail_t1 = min(t1, self._write_ts)
            if avail_t1 - avail_t0 < 0.5 / self.sample_rate:
                return out  # entirely outside ring; zeros

            # Where in `out` does the available portion go?
            out_start = int(round((avail_t0 - t0) * self.sample_rate))
            out_end = int(round((avail_t1 - t0) * self.sample_rate))

            # Where in the ring is the available portion?
            # Offset from write_pos going backwards.
            offset_from_end = int(round((self._write_ts - avail_t1) * self.sample_rate))
            chunk_n = out_end - out_start
            if chunk_n <= 0:
                return out
            # Compute the start index in buffer
            start_idx = (self._write_pos - offset_from_end - chunk_n) % self.capacity
            end_idx = (start_idx + chunk_n) % self.capacity
            if end_idx > start_idx:
                out[out_start:out_end] = self._buf[start_idx:end_idx]
            else:
                k = self.capacity - start_idx
                out[out_start:out_start + k] = self._buf[start_idx:]
                out[out_start + k:out_end] = self._buf[:end_idx]
            return out


class FeatureRing:
    """Frame-aligned feature ring (e.g. wav2vec features at 25 fps).

    Same zero-pad semantics as AudioRing but a feature vector per frame instead
    of a scalar sample. Reads for a window past write_ts zero-pad the tail (=
    silence equivalent in feature space); reads before the backing buffer
    zero-pad the head.
    """

    def __init__(self, capacity_seconds: float = 30.0, fps: int = 25, dim: int = 512,
                 dtype: np.dtype = np.float32):
        self.fps = fps
        self.dim = dim
        self.dtype = dtype
        self.capacity = int(capacity_seconds * fps)
        self._buf = np.zeros((self.capacity, dim), dtype=dtype)
        self._write_pos = 0
        self._write_ts: float = 0.0
        self._start_ts: float = 0.0
        self._lock = threading.Lock()

    @property
    def write_ts(self) -> float:
        with self._lock:
            return self._write_ts

    def write(self, feats: np.ndarray, start_ts: Optional[float] = None) -> None:
        """Append `feats` of shape [T, dim]. If start_ts > write_ts we zero-pad
        the gap with zero-feature frames."""
        assert feats.ndim == 2 and feats.shape[1] == self.dim, \
            f"expected (T, {self.dim}), got {feats.shape}"
        feats = np.ascontiguousarray(feats, dtype=self.dtype)
        with self._lock:
            if start_ts is not None and start_ts > self._write_ts + 0.5 / self.fps:
                gap_n = int(round((start_ts - self._write_ts) * self.fps))
                if gap_n > 0:
                    self._put(np.zeros((gap_n, self.dim), dtype=self.dtype))
            self._put(feats)

    def _put(self, feats: np.ndarray) -> None:
        n = feats.shape[0]
        if n == 0:
            return
        if n >= self.capacity:
            # Same gap-overflow fix as AudioRing: advance write_ts by the
            # ORIGINAL n, not capacity, so a huge gap-fill doesn't lose time.
            original_n = n
            feats = feats[-self.capacity:]
            self._buf[:] = feats
            self._write_pos = 0
            self._write_ts += original_n / self.fps
            self._start_ts = self._write_ts - self.capacity / self.fps
            return
        end = self._write_pos + n
        if end <= self.capacity:
            self._buf[self._write_pos:end] = feats
        else:
            k = self.capacity - self._write_pos
            self._buf[self._write_pos:] = feats[:k]
            self._buf[:n - k] = feats[k:]
        self._write_pos = end % self.capacity
        self._write_ts += n / self.fps
        self._start_ts = max(0.0, self._write_ts - self.capacity / self.fps)

    def truncate_at(self, ts: float) -> None:
        """Rewind write_ts to `ts`, discarding anything at or after that time.
        Used for barge-in."""
        with self._lock:
            if ts >= self._write_ts:
                return
            frames_back = int(round((self._write_ts - ts) * self.fps))
            frames_back = min(frames_back, self.capacity)
            self._write_pos = (self._write_pos - frames_back) % self.capacity
            self._write_ts = ts

    def read_window(self, t0: float, t1: float) -> np.ndarray:
        with self._lock:
            n_total = int(round((t1 - t0) * self.fps))
            if n_total <= 0:
                return np.zeros((0, self.dim), dtype=self.dtype)
            out = np.zeros((n_total, self.dim), dtype=self.dtype)

            avail_t0 = max(t0, self._start_ts)
            avail_t1 = min(t1, self._write_ts)
            if avail_t1 - avail_t0 < 0.5 / self.fps:
                return out  # effectively empty window; zeros

            out_start = int(round((avail_t0 - t0) * self.fps))
            out_end = int(round((avail_t1 - t0) * self.fps))
            chunk_n = out_end - out_start
            if chunk_n <= 0:
                return out
            offset_from_end = int(round((self._write_ts - avail_t1) * self.fps))
            start_idx = (self._write_pos - offset_from_end - chunk_n) % self.capacity
            end_idx = (start_idx + chunk_n) % self.capacity
            if end_idx > start_idx:
                out[out_start:out_end] = self._buf[start_idx:end_idx]
            else:
                k = self.capacity - start_idx
                out[out_start:out_start + k] = self._buf[start_idx:]
                out[out_start + k:out_end] = self._buf[:end_idx]
            return out


class VideoRing:
    """RGB uint8 face-crop ring at a target fps. Thread-safe. Repeats last-good
    frame on underflow (not black)."""

    def __init__(self, capacity_seconds: float = 30.0, fps: int = 25,
                 height: int = 512, width: int = 512, channels: int = 3):
        self.fps = fps
        self.H, self.W, self.C = height, width, channels
        self.capacity = int(capacity_seconds * fps)
        # Allocated on first write: 30 s at 512x512 is ~590 MB, and the voice
        # chat never writes this ring (webcam frames aren't consumed). Eager
        # np.zeros of it every session is wasted; on macOS it also grew the
        # process footprint by ~one ring per session (the allocator zero-fills
        # reused pages, making them resident). Linux/glibc mmaps buffers this
        # large fresh, so there it only cost address space.
        self._buf: Optional[np.ndarray] = None
        self._times = np.zeros(self.capacity, dtype=np.float64)
        self._write_pos = 0
        self._count = 0  # number of valid frames in buffer (<= capacity)
        self._last_good: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def write(self, frame: np.ndarray, ts: float) -> None:
        assert frame.shape == (self.H, self.W, self.C), \
            f"frame shape {frame.shape} != expected {(self.H, self.W, self.C)}"
        assert frame.dtype == np.uint8
        with self._lock:
            if self._buf is None:
                self._buf = np.zeros((self.capacity, self.H, self.W, self.C), dtype=np.uint8)
            self._buf[self._write_pos] = frame
            self._times[self._write_pos] = ts
            self._write_pos = (self._write_pos + 1) % self.capacity
            self._count = min(self._count + 1, self.capacity)
            self._last_good = frame

    @property
    def write_ts(self) -> float:
        with self._lock:
            if self._count == 0:
                return 0.0
            last_idx = (self._write_pos - 1) % self.capacity
            return float(self._times[last_idx]) + 1.0 / self.fps

    def read_window(self, t0: float, t1: float) -> np.ndarray:
        """Return exactly round((t1-t0)*fps) frames at `fps` spacing starting at t0.
        For each sample time, pick the latest frame with ts <= sample_t. If none
        (reading before the earliest frame or before any writes), use the
        last-good frame if we have one, else zeros."""
        n_frames = int(round((t1 - t0) * self.fps))
        if n_frames <= 0:
            return np.zeros((0, self.H, self.W, self.C), dtype=np.uint8)
        out = np.zeros((n_frames, self.H, self.W, self.C), dtype=np.uint8)
        with self._lock:
            if self._count == 0:
                return out  # nothing yet; zeros
            # Snapshot indices and times of valid frames (in write order)
            if self._count < self.capacity:
                order = np.arange(self._count)
            else:
                order = (np.arange(self.capacity) + self._write_pos) % self.capacity
            times = self._times[order]

            # Target sample times (start-aligned)
            dt = 1.0 / self.fps
            sample_ts = t0 + np.arange(n_frames) * dt

            # For each sample time, find the index into `order` of the latest
            # frame with ts <= sample_t. searchsorted(side='right')-1.
            idx = np.searchsorted(times, sample_ts, side='right') - 1

            # Fill valid picks; underflow → last_good
            for i, picked in enumerate(idx):
                if picked < 0:
                    out[i] = self._last_good if self._last_good is not None else 0
                else:
                    out[i] = self._buf[order[picked]]
            return out


if __name__ == "__main__":
    # Smoke test
    import time
    print("=== AudioRing ===")
    sr = 16000
    r = AudioRing(capacity_seconds=2.0, sample_rate=sr)

    # Write 0.5 s at t=0
    r.write(np.ones(sr // 2, dtype=np.float32), start_ts=0.0)
    print(f"after writing 0.5s at t=0: write_ts={r.write_ts}")

    # Read [0, 1.0) — second half should be zero-padded
    out = r.read_window(0.0, 1.0)
    print(f"read [0,1): shape={out.shape} first={out[0]:.2f} mid_full={out[sr//2-1]:.2f} "
          f"mid_zero={out[sr//2]:.2f} tail={out[-1]:.2f}")
    assert out[0] == 1.0 and out[sr//2 - 1] == 1.0
    assert out[sr//2] == 0.0 and out[-1] == 0.0
    print("  pass: tail zero-padded")

    # Write with a gap: next sample starts at t=1.0 (0.5 s of "gap")
    r.write(np.ones(sr // 2, dtype=np.float32) * 2.0, start_ts=1.0)
    out = r.read_window(0.5, 1.0)
    print(f"gap [0.5,1.0): all zeros? {(out == 0).all()}")
    assert (out == 0).all()
    out = r.read_window(1.0, 1.5)
    print(f"post-gap [1.0,1.5): all 2.0? {(out == 2.0).all()}")
    assert (out == 2.0).all()
    print("  pass: gap zero-filled")

    # Future read beyond write_ts
    out = r.read_window(1.5, 2.0)
    print(f"future [1.5,2.0): all zeros? {(out == 0).all()}")
    assert (out == 0).all()
    print("  pass: future read is silence")

    print("\n=== VideoRing ===")
    v = VideoRing(capacity_seconds=1.0, fps=25)
    f1 = (np.ones((512, 512, 3), dtype=np.uint8) * 100)
    f2 = (np.ones((512, 512, 3), dtype=np.uint8) * 200)
    v.write(f1, ts=0.0)
    v.write(f2, ts=0.04)   # 1 frame later
    print(f"write_ts after 2 frames: {v.write_ts:.3f}")

    # Read 2 frames at correct spacing
    out = v.read_window(0.0, 0.08)
    print(f"read [0,0.08) ({out.shape[0]} frames): "
          f"f0 val={out[0,0,0,0]}  f1 val={out[1,0,0,0]}")
    assert out[0, 0, 0, 0] == 100 and out[1, 0, 0, 0] == 200
    print("  pass: both frames readable")

    # Future read → last-good (frame2)
    out = v.read_window(0.1, 0.2)
    print(f"future read: all equal to last-good (200)? {(out == 200).all()}")
    assert (out == 200).all()
    print("  pass: future read repeats last-good")

    print("\n=== FeatureRing ===")
    fr = FeatureRing(capacity_seconds=1.0, fps=25, dim=4)
    f1 = np.ones((5, 4), dtype=np.float32) * 10
    f2 = np.ones((5, 4), dtype=np.float32) * 20
    fr.write(f1, start_ts=0.0)          # frames 0-4 at t=[0.00, 0.20)
    fr.write(f2, start_ts=0.4)          # gap of 5 frames at t=[0.20, 0.40), then frames at [0.40, 0.60)

    out = fr.read_window(0.0, 0.2)
    print(f"read [0.0, 0.2): shape={out.shape} val={out[0,0]}")
    assert (out == 10).all()

    out = fr.read_window(0.2, 0.4)
    print(f"gap [0.2, 0.4): zeros? {(out == 0).all()}")
    assert (out == 0).all()

    out = fr.read_window(0.4, 0.6)
    print(f"post-gap [0.4, 0.6): val={out[0,0]}")
    assert (out == 20).all()

    # Future read beyond write_ts
    out = fr.read_window(0.6, 1.0)
    print(f"future [0.6, 1.0): zeros? {(out == 0).all()}")
    assert (out == 0).all()
    print("  pass: FeatureRing zero-pad works")

    print("\nAll ring smoke tests passed.")
