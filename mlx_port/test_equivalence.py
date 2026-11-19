"""Numerical equivalence tests: MLX output ≈ PyTorch output.

Run bottom-up:
  1. Helpers (RoPE, masks) — tolerance 1e-6
  2. Individual modules (Attention, CrossAttention, DFoTBlock) — tolerance 1e-3 bf16
  3. Full forward_with_precomputed — tolerance 1e-2 bf16 end-to-end
  4. Full inference (end of block 0, end of block 5) — SSIM >= 0.995

Usage:
    python test_equivalence.py                    # run all
    python test_equivalence.py helpers            # just phase 1
    python test_equivalence.py attention          # specific module
"""
from __future__ import annotations

import os
import sys
import math

import numpy as np

# Keep PyTorch import gated so this file still parses on Mac without torch.
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import mlx.core as mx
import mlx.nn as nn


# Add submodule to path so we can import the PyTorch reference.
AF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "avatarforcing"))
sys.path.insert(0, AF_PATH)


def _as_np(t) -> np.ndarray:
    if hasattr(t, "detach"):          # torch.Tensor
        return t.detach().cpu().numpy()
    if isinstance(t, mx.array):
        return np.array(t)
    return np.asarray(t)


def _max_abs_diff(a, b) -> float:
    return float(np.max(np.abs(_as_np(a) - _as_np(b))))


# ============================================================
#  Phase 1: helpers
# ============================================================

def test_rope_precompute(tol: float = 1e-5):
    """Compare precompute_freqs_cis MLX vs PyTorch."""
    if not HAS_TORCH:
        print("skip: no torch"); return
    from models.avatarforcing.flow_transformer import precompute_freqs_cis as torch_fn
    from helpers_mlx import precompute_freqs_cis as mlx_fn

    dim, end = 64, 1024
    t_ref = torch_fn(dim, end)                 # complex64 (end, dim/2)
    x_mlx = mlx_fn(dim, end)
    # Compare via real/imag parts (MLX complex -> two real arrays)
    diff_real = _max_abs_diff(t_ref.real, x_mlx.real if hasattr(x_mlx, "real") else mx.real(x_mlx))
    diff_imag = _max_abs_diff(t_ref.imag, x_mlx.imag if hasattr(x_mlx, "imag") else mx.imag(x_mlx))
    print(f"rope precompute: real_diff={diff_real:.2e} imag_diff={diff_imag:.2e}  "
          f"{'PASS' if max(diff_real, diff_imag) < tol else 'FAIL'}")


def test_block_mask(tol: float = 0):
    if not HAS_TORCH:
        print("skip: no torch"); return
    from models.avatarforcing.flow_transformer import make_block_bidir_causal_lookahead_mask_window as torch_fn
    from helpers_mlx import make_block_bidir_causal_lookahead_mask_window as mlx_fn

    for (start_pos, cache_len, n_tokens, local_len, lookahead) in [
        (0, 0, 50, 10, 2),       # warmup
        (38, 38, 12, 10, 2),     # steady state
        (128, 38, 12, 10, 2),
    ]:
        t = torch_fn(start_pos, cache_len, n_tokens, local_len, lookahead).numpy()
        x = _as_np(mlx_fn(start_pos, cache_len, n_tokens, local_len, lookahead))
        ok = np.array_equal(t, x)
        print(f"block_mask ({start_pos=},{cache_len=},{n_tokens=}): {'PASS' if ok else 'FAIL'} "
              f"shape t={t.shape} x={x.shape}")


# ============================================================
#  Phase 2: modules
# ============================================================

def test_attention(tol: float = 1e-3):
    """One self-attention forward on random inputs with fixed seed."""
    if not HAS_TORCH:
        print("skip: no torch"); return
    # TODO:
    # 1. Instantiate torch Attention + mlx Attention with identical hyperparams
    # 2. Copy torch weights into mlx (via state dict -> mx.array)
    # 3. Run same (q, mask, freqs_cis_slice) through both
    # 4. Assert max abs diff < tol
    print("test_attention: TODO")


def test_cross_attention(tol: float = 1e-3):
    print("test_cross_attention: TODO")


def test_dfot_block(tol: float = 1e-3):
    print("test_dfot_block: TODO")


# ============================================================
#  Phase 3: end-to-end
# ============================================================

def test_forward_with_precomputed(tol: float = 1e-2):
    """One full pass of forward_with_precomputed, warmup shape (B=2, N=50)."""
    if not HAS_TORCH:
        print("skip: no torch"); return
    print("test_forward_with_precomputed: TODO")


def test_streaming_block(tol: float = 1e-2):
    """Steady-state streaming shape (B=2, N=12) + KV cache check."""
    print("test_streaming_block: TODO")


ALL_TESTS = [
    test_rope_precompute,
    test_block_mask,
    test_attention,
    test_cross_attention,
    test_dfot_block,
    test_forward_with_precomputed,
    test_streaming_block,
]


def main():
    filt = sys.argv[1] if len(sys.argv) > 1 else None
    for fn in ALL_TESTS:
        if filt and filt not in fn.__name__:
            continue
        fn()


if __name__ == "__main__":
    main()
