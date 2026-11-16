"""
Live webcam face-crop pipeline.

Mirrors `preprocess_user_video.py`'s per-frame face detection + bbox crop + resize
to 512², but optimised for a streaming webcam source:

- S3FD face detector runs on every frame (fast enough on CPU, ~10 ms).
- Smoothed bbox: exponentially-moving-average of the detected box so the crop
  doesn't jitter between frames. Face_alignment's detector sometimes misses a
  frame; we hold the last-known box for up to N frames before falling back.
- No landmark model needed (we just need the bbox).

Interface:
    cropper = FaceCropper(pad_ratio=1.0, smoothing=0.7, device='cuda')
    # per frame:
    rgb_512 = cropper.crop(rgb_frame)   # [512, 512, 3] uint8 or None if no face yet

Thread-safety: not thread-safe — intended to run inside a single asyncio
executor task per session.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Optional
import cv2
import numpy as np


class FaceCropper:
    def __init__(self, pad_ratio: float = 1.0, smoothing: float = 0.7,
                 miss_tolerance: int = 10, device: str = "cuda",
                 detection_confidence: float = 0.90, out_size: int = 512):
        """
        pad_ratio: how much extra padding around the detected bbox (same as preprocess script)
        smoothing: EMA weight on the previous box (0=no smoothing, 0.9=very sticky)
        miss_tolerance: number of consecutive detect-misses before we reset the box
        device: 'cuda' or 'cpu' for the face detector
        detection_confidence: S3FD threshold (preprocess script uses 0.95; we loosen
                              since webcam frames are often noisier)
        """
        self._device = device
        # Dedicated single-thread executor. Each cuda:N context is initialized
        # once in this thread; no cross-thread cuda context racing.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="af-facecrop")
        # Pin the thread to the target device before touching torch at all.
        self._executor.submit(self._init_worker, device).result()

        self.pad_ratio = pad_ratio
        self.smoothing = smoothing
        self.miss_tolerance = miss_tolerance
        self.conf = detection_confidence
        self.out_size = out_size

        # Smoothed box state: (cx, cy, half_size)
        self._box: Optional[tuple[float, float, float]] = None
        self._misses: int = 0

    def _init_worker(self, device: str) -> None:
        """Runs on the worker thread: pin device, build face_alignment."""
        if device.startswith("cuda"):
            import torch
            idx = int(device.split(":")[1]) if ":" in device else 0
            torch.cuda.set_device(idx)
        import face_alignment
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device=device,
        )

    def _detect(self, img: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """Return (x1, y1, x2, y2) of highest-confidence box, or None."""
        h, w = img.shape[:2]
        # Fast detect at ~360p for throughput, map back.
        scale = 360.0 / max(h, 1)
        if scale < 1.0:
            small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = img
            scale = 1.0
        boxes = self.fa.face_detector.detect_from_image(small)
        if not boxes:
            return None
        # S3FD returns [x1, y1, x2, y2, score]
        boxes = [b for b in boxes if b[4] > self.conf]
        if not boxes:
            return None
        b = max(boxes, key=lambda x: x[4])
        x1, y1, x2, y2 = (int(b[0] / scale), int(b[1] / scale),
                          int(b[2] / scale), int(b[3] / scale))
        return x1, y1, x2, y2

    def crop(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Return a 512×512 RGB uint8 face crop, or None if no face is locked yet.

        img: HxWx3 uint8 RGB
        """
        H, W = img.shape[:2]
        det = self._detect(img)

        if det is not None:
            x1, y1, x2, y2 = det
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            half = max((y2 - y1), (x2 - x1)) / 2 * (1 + self.pad_ratio)
            new_box = (cx, cy, half)
            # Smooth with previous box
            if self._box is not None:
                a = self.smoothing
                self._box = (
                    a * self._box[0] + (1 - a) * new_box[0],
                    a * self._box[1] + (1 - a) * new_box[1],
                    a * self._box[2] + (1 - a) * new_box[2],
                )
            else:
                self._box = new_box
            self._misses = 0
        else:
            self._misses += 1
            if self._box is None or self._misses > self.miss_tolerance:
                return None  # haven't locked yet / lost for too long

        cx, cy, half = self._box
        x1 = int(max(0, cx - half))
        y1 = int(max(0, cy - half))
        x2 = int(min(W, cx + half))
        y2 = int(min(H, cy + half))

        # Enforce square — use the smaller of the clipped dims
        side = min(x2 - x1, y2 - y1)
        if side <= 0:
            return None
        xm = (x1 + x2) // 2
        ym = (y1 + y2) // 2
        half_side = side // 2
        x1 = max(0, xm - half_side)
        y1 = max(0, ym - half_side)
        x2 = x1 + 2 * half_side
        y2 = y1 + 2 * half_side

        face = img[y1:y2, x1:x2]
        if face.shape[0] == 0 or face.shape[1] == 0:
            return None
        if face.shape[0] != self.out_size:
            face = cv2.resize(face, (self.out_size, self.out_size), interpolation=cv2.INTER_AREA)
        return face


if __name__ == "__main__":
    # Smoke test on an existing frame
    import sys, time
    from pathlib import Path
    frame = cv2.imread(str(Path(__file__).resolve().parent.parent / "data/user/00050.jpg"))
    if frame is None:
        print("no test frame"); sys.exit(1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(f"input frame: {frame.shape}")

    c = FaceCropper(pad_ratio=1.0, device="cuda")
    for i in range(5):
        t0 = time.perf_counter()
        out = c.crop(frame)
        dt = (time.perf_counter() - t0) * 1000
        print(f"crop {i}: {dt:.1f} ms  shape={out.shape if out is not None else None}")
