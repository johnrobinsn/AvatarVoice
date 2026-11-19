"""Motion-decoder (pixel decode) timing on torch MPS, original vs decoder_fast_torch.

Also checks the fast kernels match the original decoder (real weights, real
reference portrait) before timing them.

Usage:
    python bench_decoder.py [--image ../data/rumi.jpg] [--iters 8]
"""
from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

import decoder_fast_torch
from pipeline_mlx import load_generator, load_image


def decode_inputs(g, img, n_frames=10, seed=0):
    with torch.inference_mode():
        s_r, _, feats = g.enc(img, input_target=None)
        r_s = g.dec.direction(g.enc.fc(s_r))
    gen = torch.Generator().manual_seed(seed)
    # Motion offsets around the identity, roughly latent-sized.
    r_d = (torch.randn(1, n_frames, 512, generator=gen) * r_s.abs().mean().cpu()).to(img.device)
    s = (s_r.unsqueeze(1) + r_d).reshape(n_frames, -1)
    fe = [f.repeat_interleave(n_frames, 0) for f in feats]
    return s, fe


def time_decode(dec, s, fe, iters, autocast_dtype=None):
    ts = []
    for _ in range(iters + 2):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("mps", dtype=autocast_dtype or torch.float32,
                                                    enabled=autocast_dtype is not None):
            out, _ = dec(s, alpha=None, feats=fe)
        torch.mps.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = sorted(ts[2:])
    return out, ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.join(HERE, "..", "data", "rumi.jpg"))
    ap.add_argument("--iters", type=int, default=8)
    args = ap.parse_args()

    dev = "mps"
    g = load_generator(dev)
    s, fe = decode_inputs(g, load_image(args.image, dev))
    d = lambda a, b: float((a.float() - b.float()).abs().max())
    rows = []

    decoder_fast_torch.unpatch()
    ref, t = time_decode(g.dec, s, fe, args.iters)
    rows.append(("original, fp32", t, 0.0))
    out, t = time_decode(g.dec, s, fe, args.iters, torch.float16)
    rows.append(("original, fp16 autocast", t, d(ref, out)))

    decoder_fast_torch.patch()
    out, t = time_decode(g.dec, s, fe, args.iters, torch.float16)
    rows.append(("fast kernels, fp16 autocast", t, d(ref, out)))
    dec16 = copy.deepcopy(g.dec).half()
    s16, fe16 = s.half(), [f.half() for f in fe]
    out, t = time_decode(dec16, s16, fe16, args.iters)                 # also warms the grid cache
    rows.append(("fast kernels, pure fp16", t, d(ref, out)))
    out, t = time_decode(torch.compile(dec16, dynamic=False), s16, fe16, args.iters)
    rows.append(("fast kernels, pure fp16, torch.compile", t, d(ref, out)))
    decoder_fast_torch.unpatch()

    # Exactness of the kernel rewrites on CPU fp32 (MPS fp32 conv algorithms differ between paths).
    gc = load_generator("cpu")
    sc, fec = s[:2].cpu(), [f[:2].cpu() for f in fe]
    with torch.inference_mode():
        ref_c, _ = gc.dec(sc, alpha=None, feats=fec)
        decoder_fast_torch.patch()
        out_c, _ = gc.dec(sc, alpha=None, feats=fec)
        decoder_fast_torch.unpatch()

    print(f"fast kernels exactness (CPU fp32, 2 frames): max_abs={d(ref_c, out_c):.2e}  (pixels in [-1, 1])")
    print("decode 10 frames @512 on MPS (median):")
    for name, t, err in rows:
        print(f"  {name:<40s} {t:6.0f} ms   max_abs vs original fp32 = {err:.2e}")


if __name__ == "__main__":
    main()
