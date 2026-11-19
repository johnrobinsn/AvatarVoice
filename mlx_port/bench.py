"""Per-block timing benchmark for the MLX flow transformer.

Measures on-device compute time with proper mx.eval() synchronization —
without that, MLX's lazy evaluation gives meaningless numbers.

Usage:
    python bench.py --weights ../avatarforcing/pretrained_dir/flow_transformer_mlx.safetensors

Expected target: < 400 ms per block on M5 Pro (full pipeline incl MAE + wav2vec).
This script only measures flow_transformer — subtract ~30-50 ms for the rest.
"""
from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path

import mlx.core as mx

from flow_transformer_mlx import FlowTransformerMLX, load_weights
from helpers_mlx import precompute_freqs_cis


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


def bench_block(model: FlowTransformerMLX, batch_size: int = 2, nfe: int = 10, iters: int = 20):
    """Time a full block: `nfe` forward_with_precomputed calls + 1 update pass.

    Mimics `AvatarForcing.inference_stream` loop structure.
    """
    # Build synthetic inputs matching steady-state shapes.
    N = model.num_frames_per_block + 2  # 12
    hidden = model.opt.dim_h

    x = mx.random.normal((batch_size, N, model.opt.dim_w), dtype=mx.bfloat16)

    # Precompute c, adaLN, wr once (done once per block in the real code).
    # TODO: replace with real precomputes when implemented.
    #   precomputed_c = model.compute_condition(...)
    #   adaLN_list   = model.precompute_timestep_adaLN(timesteps, batch_size, N)
    #   wr_emb       = model.compute_wr_embedded(wr, N)

    freqs_cis_slice = model.blocks[0].attn.freqs_cis[0:N]  # dummy start_pos=0

    # Initialize KV caches.
    model.init_kv_caches(batch_size)

    # Warmup: force compilation / first-touch caches.
    for _ in range(3):
        # TODO: real call
        # out = model.forward_with_precomputed(x, precomputed_c, adaLN_list[0], wr_emb, freqs_cis_slice, update_kv_cache=False)
        # mx.eval(out)
        pass

    block_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        # nfe denoising steps
        for i in range(nfe):
            # TODO: real call
            # out = model.forward_with_precomputed(x, precomputed_c, adaLN_list[i], wr_emb, freqs_cis_slice, update_kv_cache=False)
            pass
        # One kv-cache update pass
        # TODO: model.forward_with_precomputed(..., update_kv_cache=True)
        mx.eval(x)   # sync — replace with real output when available
        block_times.append((time.perf_counter() - t0) * 1000)

    block_times.sort()
    print(f"\nflow_transformer per-block timing ({iters} blocks, nfe={nfe}, B={batch_size}):")
    print(f"  min:    {block_times[0]:7.1f} ms")
    print(f"  median: {block_times[len(block_times)//2]:7.1f} ms")
    print(f"  mean:   {statistics.mean(block_times):7.1f} ms")
    print(f"  p90:    {block_times[int(len(block_times)*0.9)]:7.1f} ms")
    print(f"  max:    {block_times[-1]:7.1f} ms")
    print(f"\n  target: < 400 ms (full pipeline) → < ~350 ms flow_transformer alone")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=False, default=None,
                    help="MLX safetensors.  If omitted, uses random init (timing only, not correctness).")
    ap.add_argument("--batch", type=int, default=2, help="2 for CFG (cond+uncond); 1 to skip CFG")
    ap.add_argument("--nfe", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    print(f"Building FlowTransformerMLX (depth={Opt.transformer_depth}, hidden={Opt.dim_h}, heads={Opt.num_heads})")
    model = FlowTransformerMLX(Opt())
    mx.eval(model.parameters())

    if args.weights:
        if not Path(args.weights).exists():
            print(f"weights not found: {args.weights}")
            return
        print(f"loading weights from {args.weights}")
        load_weights(model, args.weights)
        mx.eval(model.parameters())

    bench_block(model, batch_size=args.batch, nfe=args.nfe, iters=args.iters)


if __name__ == "__main__":
    main()
