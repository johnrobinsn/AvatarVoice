"""Per-block timing benchmark for the MLX flow transformer.

Times real steady-state streaming blocks through FlowSamplerMLX (CFG batch 2,
N=12 tokens, KV cache read + update), with mx.eval() synchronization per block —
without it, MLX's lazy evaluation gives meaningless numbers.

Usage:
    python bench.py                         # converted weights, float32 (default)
    python bench.py --dtype bfloat16
    python bench.py --no-skip-noop          # include the no-op t=0 denoising pass

Target: < 400 ms per block for the full pipeline (flow transformer + motion
decoder + wav2vec).  This script only measures the flow transformer.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

import numpy as np
import mlx.core as mx

from flow_transformer_mlx import FlowTransformerMLX, load_weights
from sampler_mlx import FlowSamplerMLX

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "avatarforcing",
                               "pretrained_dir", "flow_transformer_mlx.safetensors")


class Opt:
    """Match avatarforcing/configs/inference.yaml."""
    dim_w = 512
    dim_h = 1024
    dim_a = 512
    num_heads = 8
    transformer_depth = 8
    mlp_ratio = 4.0
    attention_window = 2
    sec = 2
    fps = 25
    num_frames_per_block = 10


def summarize(label: str, times_ms: list[float]):
    t = sorted(times_ms)
    print(f"{label}: n={len(t)}  min={t[0]:.1f}  median={t[len(t) // 2]:.1f}  "
          f"mean={statistics.mean(t):.1f}  p90={t[int(len(t) * 0.9)]:.1f}  max={t[-1]:.1f} ms")


def bench(sampler: FlowSamplerMLX, iters: int, warmup: int = 3):
    rng = np.random.default_rng(0)
    clip, blk, dim = sampler.clip, sampler.block, sampler.dim_w
    n = warmup + iters
    wa = mx.array(rng.standard_normal((1, clip + n * blk, dim)).astype(np.float32))
    wr = mx.array(rng.standard_normal((1, dim)).astype(np.float32))
    noise = lambda L: mx.random.normal((1, L, dim))

    sampler.reset(1)
    t0 = time.perf_counter()
    x_t = sampler.prime(wa[:, :clip], wr, noise(clip))
    mx.eval(x_t, sampler.kv_cache)
    print(f"primer (50 frames, first call): {(time.perf_counter() - t0) * 1000:.1f} ms")

    times = []
    for b in range(n):
        t_idx = clip + b * blk
        t0 = time.perf_counter()
        x_t = sampler.step(wa[:, t_idx - 2:t_idx + blk], wr, x_t, noise(blk), t_idx - 2)
        mx.eval(x_t, sampler.kv_cache)
        if b >= warmup:
            times.append((time.perf_counter() - t0) * 1000)

    sampler.reset(1)
    t0 = time.perf_counter()
    mx.eval(sampler.prime(wa[:, :clip], wr, noise(clip)), sampler.kv_cache)
    print(f"primer (50 frames, warm): {(time.perf_counter() - t0) * 1000:.1f} ms")
    passes = len(sampler.steps) + 1
    summarize(f"steady-state block ({passes} forward passes, B=2, N=12)", times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--nfe", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--no-skip-noop", action="store_true")
    args = ap.parse_args()

    model = FlowTransformerMLX(Opt())
    load_weights(model, args.weights, dtype=getattr(mx, args.dtype))
    sampler = FlowSamplerMLX(model, nfe=args.nfe, skip_noop_steps=not args.no_skip_noop)
    print(f"device={mx.default_device()}  dtype={args.dtype}  nfe={args.nfe}  "
          f"denoise passes/block={len(sampler.steps)}")
    bench(sampler, args.iters)


if __name__ == "__main__":
    main()
