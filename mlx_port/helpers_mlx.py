"""MLX counterparts for the small helpers in flow_transformer.py:
  - RoPE precompute + apply
  - Block-bidir-causal lookahead masks
  - adaLN modulate

Keep these simple and well-tested — everything else depends on them.
"""
from __future__ import annotations

import math

import mlx.core as mx


# ---------- Rotary embeddings ----------

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> mx.array:
    """Polar-form complex rotary frequencies.

    Returns: (end, dim/2) complex64 array.

    PyTorch reference (flow_transformer.py:41-47):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        return torch.polar(torch.ones_like(freqs), freqs)  # [end, dim/2]
    """
    if dim % 2 != 0:
        raise ValueError("RoPE dim must be even.")
    # TODO: implement
    # idx = mx.arange(0, dim, 2)[: (dim // 2)].astype(mx.float32)
    # inv_freq = 1.0 / (theta ** (idx / dim))
    # t = mx.arange(end).astype(mx.float32)
    # freqs = mx.outer(t, inv_freq)          # (end, dim/2)
    # Return complex polar form.  MLX: mx.exp(1j * freqs) is one way; or build
    # from cos/sin and view-as-complex.  Dtype complex64.
    raise NotImplementedError


def apply_rotary_emb_q(xq: mx.array, freqs_cis_n: mx.array) -> mx.array:
    """Apply rotary to Q only.

    Shapes:
      xq: (B, H, N, D)  — D must be even
      freqs_cis_n: (N, D/2) complex64 — already sliced for the current window

    PyTorch reference (flow_transformer.py:76-84).  Note that PyTorch uses
    view_as_complex/view_as_real; in MLX, either do the same (if complex dtype
    supports it) or decompose into (cos, sin) pair:
        xq_pair = xq.reshape(B, H, N, D//2, 2)
        real = xq_pair[..., 0] * cos - xq_pair[..., 1] * sin
        imag = xq_pair[..., 0] * sin + xq_pair[..., 1] * cos
        return mx.stack([real, imag], axis=-1).reshape(B, H, N, D)
    """
    # TODO
    raise NotImplementedError


def apply_rotary_emb_k(xk: mx.array, freqs_cis_n: mx.array) -> mx.array:
    """Apply rotary to K only.  Same math as Q, kept separate to match the
    PyTorch-side split (flow_transformer.py:86-93)."""
    # TODO
    raise NotImplementedError


# ---------- Masks ----------

def make_block_bidir_causal_lookahead_mask_window(
    start_pos: int,
    cache_len: int,
    n_tokens: int,
    local_len: int,
    lookahead: int = 2,
) -> mx.array:
    """Self-attention mask used by `Attention`.

    PyTorch reference (flow_transformer.py:33-39):
        q_abs = torch.arange(start_pos, start_pos + n_tokens)[:, None]
        k_abs = torch.arange(start_pos - cache_len, start_pos + n_tokens)[None, :]
        q_blk = q_abs // local_len
        k_blk = k_abs // local_len
        allow = (k_blk < q_blk) | (k_abs <= q_abs + lookahead)
        return allow                  # shape (n_tokens, cache_len + n_tokens)
    """
    # TODO
    raise NotImplementedError


def enc_dec_mask_window(
    start_pos: int,
    cache_len: int,
    n_tokens: int,
    expansion: int,
) -> mx.array:
    """Cross-attention mask: local window around each Q position.

    PyTorch reference (flow_transformer.py:19-23):
        q_abs = torch.arange(start_pos, start_pos + n_tokens)[:, None]
        k_abs = torch.arange(start_pos - cache_len, start_pos + n_tokens)[None, :]
        allow = (k_abs >= q_abs - expansion) & (k_abs <= q_abs + expansion)
        return allow
    """
    # TODO
    raise NotImplementedError


def enc_dec_mask(T: int, S: int, frame_width: int = 1, expansion: int = 2) -> mx.array:
    """Alignment mask used at warmup (non-streaming path).

    PyTorch reference (flow_transformer.py:13-17):
        mask = torch.ones(T, S)
        for i in range(T):
            mask[i, max(0, (i - expansion) * frame_width):(i + expansion + 1) * frame_width] = 0
        return mask == 0
    """
    # TODO
    raise NotImplementedError


# ---------- Modulation ----------

def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    """x * (1 + scale) + shift.  PyTorch: flow_transformer.py:10-11."""
    return x * (1.0 + scale) + shift
