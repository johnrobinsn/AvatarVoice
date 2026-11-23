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


class _TimeSeries:
    """Time-indexed buffer of items (audio samples or feature frames) at a fixed
    rate. Thread-safe. Item i covers time i / rate.

    capacity_seconds: keep only that long behind write_ts, like a ring buffer
    (fixed memory; for buffers written in real time, e.g. the mic).
    None: grow as needed and keep everything until release_before() — for buffers
    written AHEAD of their reader (the avatar's TTS is scheduled a whole reply
    ahead of playback), where dropping the oldest data would drop audio that
    hasn't played yet.
    """

    def __init__(self, rate: int, item_shape: tuple, dtype, capacity_seconds: Optional[float]):
        self._rate = rate
        self._item_shape = item_shape
        self._dtype = dtype
        self._keep = None if capacity_seconds is None else int(round(capacity_seconds * rate))
        self._min_len = 2 * self._keep if self._keep else 10 * rate
        self._buf = np.zeros((self._min_len,) + item_shape, dtype)
        self._off = 0      # item index of _buf[0]
        self._begin = 0    # item index of the oldest item held
        self._end = 0      # item index of the next unwritten item (write_ts * rate)
        self._lock = threading.Lock()

    @property
    def write_ts(self) -> float:
        with self._lock:
            return self._end / self._rate

    def _write(self, items: np.ndarray, start_ts: Optional[float]) -> None:
        with self._lock:
            if start_ts is not None:
                idx = int(round(start_ts * self._rate))
                if idx > self._end:
                    if self._begin == self._end:          # empty: no need to store the gap
                        self._begin = self._end = self._off = idx
                    else:                                 # gap: silence
                        self._append(np.zeros((idx - self._end,) + self._item_shape, self._dtype))
                # start_ts < write_ts (shouldn't normally happen): append at write_ts
            self._append(items)

    def _append(self, items: np.ndarray) -> None:
        n = items.shape[0]
        if n == 0:
            return
        if self._keep is not None and n >= self._keep:   # only the tail survives
            self._begin = self._off = self._end + n - self._keep
            self._buf[:self._keep] = items[-self._keep:]
            self._end += n
            return
        need = self._end + n - self._begin
        if self._end + n - self._off > len(self._buf):
            # Out of room at the back: move the held items to the front of a buffer
            # sized for them (grows for a long reply, shrinks back once it has played).
            held = self._buf[self._begin - self._off:self._end - self._off]
            size = len(self._buf) if self._keep else max(2 * need, self._min_len)
            if size != len(self._buf):
                new = np.zeros((size,) + self._item_shape, self._dtype)
                new[:len(held)] = held
                self._buf = new
            else:
                self._buf[:len(held)] = held.copy()
            self._off = self._begin
        self._buf[self._end - self._off:self._end - self._off + n] = items
        self._end += n
        if self._keep is not None:
            self._begin = max(self._begin, self._end - self._keep)

    def truncate_at(self, ts: float) -> None:
        """Rewind write_ts to `ts`, discarding anything at or after it. Used for
        barge-in: TTS scheduled ahead of playback is cut at the playback cursor.
        Reads past the new write_ts return zeros (silence)."""
        with self._lock:
            self._end = max(self._begin, min(self._end, int(round(ts * self._rate))))

    def release_before(self, ts: float) -> None:
        """Drop everything before `ts` (the reader is done with it). If that empties
        the buffer, write_ts jumps to `ts`, so the next write after a long idle
        spell doesn't zero-fill the whole gap."""
        with self._lock:
            idx = int(round(ts * self._rate))
            if idx >= self._end:
                self._begin = self._end = self._off = max(idx, self._end)
                if not self._keep and len(self._buf) > self._min_len:   # a long reply has played
                    self._buf = np.zeros((self._min_len,) + self._item_shape, self._dtype)
            elif idx > self._begin:
                self._begin = idx

    def _read(self, t0: float, t1: float) -> np.ndarray:
        """Exactly round((t1-t0)*rate) items covering [t0, t1); anything not held
        (before the oldest item, in a gap, past write_ts) is zeros."""
        n_total = int(round((t1 - t0) * self._rate))
        out = np.zeros((max(0, n_total),) + self._item_shape, self._dtype)
        with self._lock:
            i0 = int(round(t0 * self._rate))
            a, b = max(i0, self._begin), min(i0 + n_total, self._end)
            if b > a:
                out[a - i0:b - i0] = self._buf[a - self._off:b - self._off]
        return out


class AudioRing(_TimeSeries):
    """16 kHz f32 mono audio. Zero-pads underflows (silence). See _TimeSeries for
    capacity_seconds (None = grow; the reader calls release_before)."""

    def __init__(self, capacity_seconds: Optional[float] = 30.0, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        super().__init__(sample_rate, (), np.float32, capacity_seconds)

    def write(self, samples: np.ndarray, start_ts: Optional[float] = None) -> None:
        """Append samples. If start_ts is given and > write_ts the gap is silence;
        if < write_ts the samples are appended at write_ts."""
        assert samples.dtype == np.float32, f"expected f32, got {samples.dtype}"
        self._write(np.ascontiguousarray(samples).ravel(), start_ts)

    def read_window(self, t0: float, t1: float) -> np.ndarray:
        return self._read(t0, t1)


class FeatureRing(_TimeSeries):
    """Frame-aligned features (e.g. wav2vec at 25 fps). Same semantics as
    AudioRing with a feature vector per frame: a gap, the missing head and the
    unwritten tail read as zero frames."""

    def __init__(self, capacity_seconds: Optional[float] = 30.0, fps: int = 25, dim: int = 512,
                 dtype: np.dtype = np.float32):
        self.fps = fps
        self.dim = dim
        self.dtype = dtype
        super().__init__(fps, (dim,), dtype, capacity_seconds)

    def write(self, feats: np.ndarray, start_ts: Optional[float] = None) -> None:
        """Append `feats` of shape [T, dim]; a gap before start_ts is zero frames."""
        assert feats.ndim == 2 and feats.shape[1] == self.dim, \
            f"expected (T, {self.dim}), got {feats.shape}"
        self._write(np.ascontiguousarray(feats, dtype=self.dtype), start_ts)

    def read_window(self, t0: float, t1: float) -> np.ndarray:
        return self._read(t0, t1)


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
