"""MLX port of AvatarForcing flow_transformer.

Module tree (mirrors PyTorch `flow_transformer.py` 1:1):

    FlowTransformerMLX
        .x_embedder        : SequenceEmbed (Linear 1024 -> 1024)
        .t_embedder        : TimestepEmbedder (sinusoidal -> MLP)
        .c_embedder        : AudioMotionAttention (motion MLP + 2× MHA + Linear)
        .adaLN_modulation  : Sequential(SiLU, Linear hidden -> 6*hidden)
        .blocks[0..7]      : DFoTBlock
            .attn          : Attention (self-attn, RoPE, block-bidir mask, 38-slot KV cache)
            .cross_attn    : CrossAttention (windowed cross-attn, 2-slot KV cache)
            .mlp           : Mlp (hidden -> 4*hidden -> hidden, GELU-tanh)
            .norm1/2/3     : LayerNorm (elementwise_affine=False)
        .final_layer       : FinalLayer (adaLN + Linear hidden -> dim_w)

Weight-key convention: dot-separated.  `blocks.0.attn.qkv.weight`, etc.
After conversion by `convert_weights.py`, the `.safetensors` keys match this
structure exactly.

Hyperparameters (from avatarforcing/configs/inference.yaml):
    dim_w = 512   (motion latent)
    dim_h = 1024  (hidden)
    dim_a = 512   (audio feature)
    num_heads = 8
    transformer_depth = 8
    num_frames_for_clip = 50
    num_frames_per_block = 10
    attention_window = 2
    mlp_ratio = 4.0
    nfe = 10
"""
from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from helpers_mlx import (
    precompute_freqs_cis,
    apply_rotary_emb_q,
    apply_rotary_emb_k,
    make_block_bidir_causal_lookahead_mask_window,
    enc_dec_mask_window,
    enc_dec_mask,
    modulate,
)


# =====================================================================
#  Small embedders
# =====================================================================

class SequenceEmbed(nn.Module):
    """nn.Linear wrapped with an identity post-norm.  PyTorch: line 361."""
    def __init__(self, dim_w: int, dim_h: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_w, dim_h, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        # x shape: (B, N, dim_w)
        return self.proj(x)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding + MLP.  PyTorch: line 627.

    Weight keys:
      mlp.0.weight/bias, mlp.2.weight/bias
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
        """Sinusoidal.  PyTorch: lines 640-660."""
        # TODO
        # half = dim // 2
        # freqs = mx.exp(-math.log(max_period) * mx.arange(half).astype(mx.float32) / half)
        # args = t[:, None].astype(mx.float32) * freqs[None]
        # emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        # if dim % 2: emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1))], axis=-1)
        # return emb
        raise NotImplementedError

    def __call__(self, t: mx.array) -> mx.array:
        # t: (B,) float32
        emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(emb)


# =====================================================================
#  AudioMotionAttention (c_embedder)
# =====================================================================

class AudioMotionAttention(nn.Module):
    """PyTorch: line 97.

    Fuses user motion (q), user audio (kv), and agent audio (q2) into a
    hidden-dim condition sequence.  Two nn.MultiheadAttention calls.
    """
    def __init__(self, dim_a: int, dim_h: int):
        super().__init__()
        # Weight keys: motion_embedder.0.weight, .1.weight+bias (LayerNorm), .3.weight, ...
        self.motion_embedder = nn.Sequential(
            nn.Linear(dim_a, dim_a),
            nn.LayerNorm(dim_a),
            nn.SiLU(),
            nn.Linear(dim_a, dim_a),
            nn.LayerNorm(dim_a),
            nn.SiLU(),
        )

        # MLX nn.MultiHeadAttention exists.  PyTorch uses batch_first=True.
        # Weight keys: dual_modal_attn.{in_proj_weight,in_proj_bias,out_proj.weight,out_proj.bias}
        # (same for user_agent_attn)
        self.dual_modal_attn = nn.MultiHeadAttention(dim_a, num_heads=4, bias=True)
        self.ln1 = nn.LayerNorm(dim_a)

        self.user_agent_attn = nn.MultiHeadAttention(dim_a, num_heads=4, bias=True)
        self.ln2 = nn.LayerNorm(dim_a)

        self.to_cond_user = nn.Linear(dim_a, dim_h)

    def __call__(self, x: mx.array, c: mx.array, agent_audio: mx.array) -> mx.array:
        # TODO (follow the PyTorch forward at line 121; mind weight-key convention
        # — PyTorch nn.MultiheadAttention has a single packed in_proj_weight;
        # MLX nn.MultiHeadAttention uses separate q/k/v Linears.  The converter
        # must split the packed weight into three.)
        raise NotImplementedError


# =====================================================================
#  Attention (self-attention)
# =====================================================================

class Attention(nn.Module):
    """PyTorch: line 228.

    Self-attention with:
      - rotary embeddings (applied to Q and K)
      - dense block-bidir-causal lookahead mask
      - 38-slot rolling KV cache

    Weight keys:
      qkv.weight, qkv.bias          # packed Q/K/V
      proj.weight, proj.bias
      freqs_cis                     # buffer, not persistent — regenerated on load

    The qkv Linear maps hidden -> 3*hidden.  Split into Q/K/V after projection.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 num_frames_for_clip: int = 50, num_frames_per_block: int = 10,
                 max_seq_len: int = 65536):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_frames_for_clip = num_frames_for_clip
        self.num_frames_per_block = num_frames_per_block

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Precompute RoPE frequencies.  Stored as regular attribute (MLX has no
        # register_buffer); gets picked up by mx.eval but not serialized as a
        # weight.
        self.freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)

        # KV cache — module-owned, mutable via reassign (functional style).
        # Initialized by `init_kv_cache()` before first streaming call.
        self._k_cache: Optional[mx.array] = None
        self._v_cache: Optional[mx.array] = None

    def init_kv_cache(self, batch_size: int, cache_len: int):
        """Allocate zero-filled rolling KV cache.  cache_len = 38 steady state."""
        self._k_cache = mx.zeros((batch_size, self.num_heads, cache_len, self.head_dim), dtype=mx.bfloat16)
        self._v_cache = mx.zeros((batch_size, self.num_heads, cache_len, self.head_dim), dtype=mx.bfloat16)

    def __call__(self, x: mx.array, mask: mx.array, freqs_cis_slice: mx.array,
                 update_kv_cache: bool = False) -> mx.array:
        """
        x: (B, N, D)  — N is either num_frames_for_clip (50) at warmup or
                        num_frames_per_block + 2 (12) at steady state.
        mask: pre-sliced mask matching the current Q/K lengths.
        freqs_cis_slice: (N, head_dim/2) complex.
        """
        # TODO: replicate PyTorch forward (lines 263-303).
        # 1. qkv projection: (B, N, D) -> (B, N, 3D) -> split to Q, K, V
        #    reshape to (B, H, N, head_dim)
        # 2. apply rotary to Q, K
        # 3. if kv_cache is live and N == num_frames_per_block + 2:
        #      concat k_cache + k_new along axis=-2 (same for v)
        #      trim mask to the last (num_frames_per_block + 2) rows
        #    else:
        #      k_attn, v_attn = k_new, v_new
        # 4. if update_kv_cache: self.update_kv_cache(k_new, v_new)  [below]
        # 5. SDPA.  Prefer mlx.core.fast.scaled_dot_product_attention; if it doesn't
        #    support dense bool mask, do manual: softmax((Q @ K^T) * scale + mask_bias) @ V
        # 6. merge heads, output projection
        raise NotImplementedError

    def update_kv_cache(self, k_new: mx.array, v_new: mx.array):
        """Reassign-and-concat rolling update (no in-place mutation).

        PyTorch reference (lines 305-316):
            if num_new >= cache_size:
                kv_cache["k"][:, :, :, :] = k_new[:, :, -cache_size - 2:-2, :]
            else:
                kv_cache["k"][:, :, :-10, :] = kv_cache["k"][:, :, 10:, :].clone()
                kv_cache["k"][:, :, -10:, :] = k_new[:, :, :-2, :]

        MLX version: use mx.concatenate and reassign self._k_cache.  Make sure
        to call mx.eval on the new cache so it's materialized before the next
        forward.
        """
        # TODO
        raise NotImplementedError


# =====================================================================
#  CrossAttention
# =====================================================================

class CrossAttention(nn.Module):
    """PyTorch: line 138.

    Weight keys:
      kv.weight, kv.bias    # packed K/V
      q.weight, q.bias
      proj.weight, proj.bias

    KV cache is 2 slots (cross_cache_len = attention_window = 2).
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 num_frames_for_clip: int = 50, num_frames_per_block: int = 10):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_frames_per_block = num_frames_per_block

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q  = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self._k_cache: Optional[mx.array] = None
        self._v_cache: Optional[mx.array] = None

    def init_kv_cache(self, batch_size: int, cross_cache_len: int = 2):
        self._k_cache = mx.zeros((batch_size, self.num_heads, cross_cache_len, self.head_dim), dtype=mx.bfloat16)
        self._v_cache = mx.zeros((batch_size, self.num_heads, cross_cache_len, self.head_dim), dtype=mx.bfloat16)

    def __call__(self, x: mx.array, c: mx.array, mask: Optional[mx.array] = None,
                 update_kv_cache: bool = False) -> mx.array:
        # TODO: replicate PyTorch forward (lines 181-218).
        # Cross-attention mask is cross_mask_local(q_len, cache_len=2) at steady state.
        raise NotImplementedError

    def append_kv_to_buffer(self, k_new: mx.array, v_new: mx.array):
        """PyTorch reference (lines 220-225):
            kv_cache["k"] = k_new[:, :, -4:-2, :]
            kv_cache["v"] = v_new[:, :, -4:-2, :]

        Keep the "take 2 slots from the end minus 2" pattern.
        """
        # TODO
        raise NotImplementedError


# =====================================================================
#  DFoTBlock
# =====================================================================

class Mlp(nn.Module):
    """Gated-GELU-tanh MLP.  Matches timm's Mlp (what the PyTorch module uses).

    Weight keys: fc1.weight, fc1.bias, fc2.weight, fc2.bias
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: Optional[int] = None):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def __call__(self, x: mx.array) -> mx.array:
        # tanh-approx GELU to match PyTorch's `nn.GELU(approximate="tanh")`
        return self.fc2(nn.gelu_approx(self.fc1(x)))


class DFoTBlock(nn.Module):
    """PyTorch: line 320.

    Layer norms are elementwise_affine=False (no params — these are pure
    normalizations modulated by adaLN output).
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                 num_frames_for_clip: int = 50, num_frames_per_block: int = 10):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True,
                              num_frames_for_clip=num_frames_for_clip,
                              num_frames_per_block=num_frames_per_block)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                         num_frames_for_clip=num_frames_for_clip,
                                         num_frames_per_block=num_frames_per_block)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        # Note: PyTorch has norm1, norm2, norm3 as elementwise_affine=False.
        # MLX LayerNorm has an affine flag; set affine=False and these become
        # parameterless.  No weight keys for these.

    def __call__(self, x: mx.array, scale_shift_alpha, c: mx.array,
                 mask: mx.array, cross_mask: mx.array, freqs_cis_slice: mx.array,
                 update_kv_cache: bool = False) -> mx.array:
        # TODO: mirror PyTorch forward (lines 335-358).
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = scale_shift_alpha
        # x_attn = self.attn(modulate(layernorm(x), shift_msa, scale_msa), mask=..., freqs_cis_slice=..., update_kv_cache=...)
        # x = x + gate_msa * x_attn
        # x_cross = self.cross_attn(layernorm(x), c=c, mask=cross_mask, update_kv_cache=...)
        # x = x + x_cross
        # x = x + gate_mlp * self.mlp(modulate(layernorm(x), shift_mlp, scale_mlp))
        raise NotImplementedError


class FinalLayer(nn.Module):
    """PyTorch: look for `class FinalLayer` near line 559.  AdaLN + Linear."""
    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        # Weight keys: adaLN_modulation.1.weight/bias (Sequential + SiLU + Linear),
        #              linear.weight/bias
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.linear = nn.Linear(hidden_size, out_dim)

    def __call__(self, x: mx.array, t_embed: mx.array) -> mx.array:
        # TODO: PyTorch computes shift, scale = adaLN.chunk(2); x = modulate(ln(x), shift, scale); return linear(x)
        raise NotImplementedError

    def forward_precomputed(self, x: mx.array, final_adaLN) -> mx.array:
        """Fast path when adaLN has been precomputed.  final_adaLN is (shift, scale)."""
        # TODO
        raise NotImplementedError


# =====================================================================
#  FlowTransformer (top)
# =====================================================================

class FlowTransformerMLX(nn.Module):
    """PyTorch: line 378.

    Entry points to implement:
      - __call__         — full forward (warmup path, non-streaming)
      - forward_with_precomputed  — hot path: 20× per block
      - precompute_timestep_adaLN — one-shot per NFE schedule
      - compute_condition         — flattens the c_embedder once
      - compute_wr_embedded       — reference embedding once per session
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.num_frames_for_clip = int(opt.sec * opt.fps)  # 50
        self.num_frames_per_block = opt.num_frames_per_block  # 10

        self.x_embedder = SequenceEmbed(opt.dim_w * 2, opt.dim_h)
        self.t_embedder = TimestepEmbedder(opt.dim_h)
        self.c_embedder = AudioMotionAttention(opt.dim_a, opt.dim_h)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(opt.dim_h, 6 * opt.dim_h, bias=True),
        )

        self.blocks = [
            DFoTBlock(opt.dim_h, opt.num_heads, mlp_ratio=opt.mlp_ratio,
                      num_frames_for_clip=self.num_frames_for_clip,
                      num_frames_per_block=opt.num_frames_per_block)
            for _ in range(opt.transformer_depth)
        ]

        self.final_layer = FinalLayer(opt.dim_h, opt.dim_w)

        # Static masks — computed once.
        self.alignment_mask = enc_dec_mask(
            self.num_frames_for_clip, self.num_frames_for_clip, 1, opt.attention_window)
        # TODO: non-windowed block-bidir mask (for warmup path).

    def init_kv_caches(self, batch_size: int):
        """Allocate KV caches on every layer before streaming."""
        cache_len = self.num_frames_for_clip - self.num_frames_per_block  # 40 ... but PyTorch uses `cache_len - 2` = 38 for self-attn
        for blk in self.blocks:
            blk.attn.init_kv_cache(batch_size, cache_len - 2)   # 38 (match PyTorch)
            blk.cross_attn.init_kv_cache(batch_size, self.opt.attention_window)  # 2

    def forward_with_precomputed(
        self,
        x: mx.array,
        precomputed_c: mx.array,
        precomputed_adaLN,          # (scale_shift_alpha, t_embed, final_adaLN)
        precomputed_wr: mx.array,
        freqs_cis_slice: mx.array,
        update_kv_cache: bool = False,
    ) -> mx.array:
        # TODO: mirror PyTorch forward_with_precomputed (lines 500-518).
        raise NotImplementedError

    def precompute_timestep_adaLN(self, timestep_list, batch_size, seq_len, context_len=0):
        """Mirror PyTorch precompute_timestep_adaLN (lines 580-623).  Returns a
        list of (scale_shift_alpha_tuple, t_embed, final_adaLN_tuple) one per NFE."""
        # TODO
        raise NotImplementedError

    def compute_condition(self, wa: mx.array, wa_user: mx.array, motion_user: mx.array) -> mx.array:
        """Thin wrapper around c_embedder.  PyTorch: lines 527-531."""
        return self.c_embedder(wa, motion_user, wa_user)   # NB: argument order matches PyTorch

    def compute_wr_embedded(self, wr: mx.array, seq_len: int) -> mx.array:
        """PyTorch: lines 534-537."""
        # wr: (B, dim_a)  -> embed via x_embedder's sibling path? Actually PyTorch does
        # sequence_embedder(wr.unsqueeze(1)) which is identity at inference.
        return mx.broadcast_to(wr[:, None, :], (wr.shape[0], seq_len, wr.shape[-1]))


# =====================================================================
#  Weight loading
# =====================================================================

def load_weights(model: FlowTransformerMLX, safetensors_path: str) -> None:
    """Load converted weights (see convert_weights.py) into the model.

    MLX convention: model.load_weights(list_of_(key,value_pairs)).
    """
    # TODO:
    # from safetensors.numpy import load_file
    # arrays = {k: mx.array(v) for k, v in load_file(safetensors_path).items()}
    # model.update(tree_unflatten(list(arrays.items())))
    raise NotImplementedError
