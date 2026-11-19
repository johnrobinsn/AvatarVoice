"""Diffusion-forcing sample loop for the MLX flow transformer.

Mirrors the latent path of `AvatarForcing.inference_stream` (primer + steady-state
blocks): `prepare_cfg_condition`, `solve_cfg`, `solve_closed`, `update_kv_cache`.
Pixel decoding is not included — the caller hands latents to the motion decoder.

2-branch CFG as in the PyTorch app: [uncond, audio]; user audio/motion are null.

`solve_closed` runs in float64 in PyTorch (unsupported on Apple GPUs).  Its
coefficients depend only on the timestep schedule, so they are computed once in
float64 with numpy, and the per-step update is `x_next = a * x + b * v`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import mlx.core as mx

from flow_transformer_mlx import FlowTransformerMLX


class FlowSamplerMLX:
    def __init__(self, model: FlowTransformerMLX, nfe: int = 10, a_cfg_scale: float = 2.0,
                 num_train_timestep: int = 1000, skip_noop_steps: bool = True):
        self.model = model
        self.a_cfg_scale = a_cfg_scale
        self.clip = model.num_frames_for_clip          # 50
        self.block = model.num_frames_per_block        # 10
        self.ctx = 2                                   # overlap tokens per block
        self.dim_w = model.opt.dim_w

        # AvatarForcing: denoising_step_list = linspace(num_train_timestep, 0, nfe - 1)
        self.timesteps = np.linspace(num_train_timestep, 0, nfe - 1).tolist()
        # FlowMatchScheduler(shift=1, sigma_min=0, extra_one_step=False).set_timesteps(1000)
        self._sched_sigmas = np.linspace(1.0, 0.0, num_train_timestep)
        self._sched_timesteps = self._sched_sigmas * num_train_timestep

        self._coef_cache: dict = {}
        self._adaLN_cache: dict = {}
        self.kv_cache: Optional[list] = None

        # A step whose current and next timestep are both 0 leaves every token
        # unchanged (solve_closed's hist_mask), so its forward pass is wasted.
        self.steps = [i for i in range(len(self.timesteps))
                      if not (skip_noop_steps and self.timesteps[i] == 0 and self._next_t(i) == 0)]

    # ---------------- schedule ----------------

    def _next_t(self, i: int) -> float:
        return self.timesteps[i + 1] if i < len(self.timesteps) - 1 else self.timesteps[i]

    def _sigma(self, t: np.ndarray) -> np.ndarray:
        idx = np.abs(self._sched_timesteps[None, :] - t[:, None]).argmin(axis=-1)
        return np.where(t == 0, 0.0, self._sched_sigmas[idx])

    def _coefs(self, i: int, n_ctx: int, n_new: int):
        """(a, b) of shape (1, n_ctx + n_new, 1) with x_next = a * x + b * v."""
        key = (i, n_ctx, n_new)
        if key not in self._coef_cache:
            t = np.array([0.0] * n_ctx + [self.timesteps[i]] * n_new, dtype=np.float64)
            tn = np.array([0.0] * n_ctx + [self._next_t(i)] * n_new, dtype=np.float64)
            s_t, s_tp = self._sigma(t), self._sigma(tn)
            ratio = np.where(s_t > 0, s_tp / np.maximum(s_t, 1e-8), 0.0)
            c_x0 = (1.0 - s_tp) - ratio * (1.0 - s_t)
            a, b = ratio + c_x0, -c_x0 * s_t
            hist = (t == 0) & (tn == 0)
            a, b = np.where(hist, 1.0, a), np.where(hist, 0.0, b)
            self._coef_cache[key] = (mx.array(a.reshape(1, -1, 1), dtype=mx.float32),
                                     mx.array(b.reshape(1, -1, 1), dtype=mx.float32))
        return self._coef_cache[key]

    # ---------------- conditioning ----------------

    def _adaLN(self, batch: int, seq_len: int, context_len: int):
        key = (batch, seq_len, context_len)
        if key not in self._adaLN_cache:
            m = self.model
            steps = m.precompute_timestep_adaLN(self.timesteps, batch, seq_len, context_len)
            zero = m.precompute_zero_adaLN(batch, seq_len + context_len)
            mx.eval(steps, zero)
            self._adaLN_cache[key] = (steps, zero)
        return self._adaLN_cache[key]

    def prepare_cfg_condition(self, wa: mx.array, wr: mx.array, seq_len: int):
        """AvatarForcing.prepare_cfg_condition (2-branch, null user inputs)."""
        null = mx.zeros_like(wa)
        wa_cat = mx.concatenate([null, wa], axis=0)
        null_cat = mx.concatenate([null, null], axis=0)
        c = self.model.compute_condition(wa_cat, null_cat, null_cat)
        wr_emb = self.model.compute_wr_embedded(mx.concatenate([wr, wr], axis=0), seq_len)
        return c, wr_emb

    # ---------------- loop ----------------

    def reset(self, batch_size: int = 1):
        """AvatarForcing.initialize_kv_cache (2 CFG branches)."""
        self.kv_cache = self.model.init_kv_cache(batch_size * 2)

    def _denoise(self, x_t, c, wr_emb, adaLN_steps, start_pos, kv, n_ctx, n_new):
        for i in self.steps:
            x_cat = mx.concatenate([x_t, x_t], axis=0)
            v = self.model.forward_with_precomputed(x_cat, c, adaLN_steps[i], wr_emb, start_pos,
                                                    kv_cache_list=kv, update_kv_cache=False)
            v_uncond, v_aud = mx.split(v, 2, axis=0)
            v_t = v_uncond + self.a_cfg_scale * (v_aud - v_uncond)
            a, b = self._coefs(i, n_ctx, n_new)
            x_t = a * x_t + b * v_t
        return x_t

    def _update_kv_cache(self, x_t, c, wr_emb, adaLN_zero, start_pos):
        x_cat = mx.concatenate([x_t, x_t], axis=0)
        self.model.forward_with_precomputed(x_cat, c, adaLN_zero, wr_emb, start_pos,
                                            kv_cache_list=self.kv_cache, update_kv_cache=True)

    def prime(self, wa: mx.array, wr: mx.array, noise: mx.array) -> mx.array:
        """First block: 50 frames, no KV cache read, then fill the cache.
        wa: (B, 50, dim_w) audio features; wr: (B, dim_w); noise: (B, 50, dim_w)."""
        B = wa.shape[0]
        if self.kv_cache is None:
            self.reset(B)
        c, wr_emb = self.prepare_cfg_condition(wa, wr, self.clip)
        steps, zero = self._adaLN(2 * B, self.clip, 0)
        x_t = self._denoise(noise.astype(mx.float32), c, wr_emb, steps, 0, None, 0, self.clip)
        self._update_kv_cache(x_t, c, wr_emb, zero, 0)
        return x_t

    def step(self, wa: mx.array, wr: mx.array, prev_x_t: mx.array, noise: mx.array,
             start_pos: int) -> mx.array:
        """Steady-state block.  wa: (B, 12, dim_w) = frames [t-2, t+10);
        prev_x_t: previous block's latents (last 2 tokens used); noise: (B, 10, dim_w);
        start_pos = t - 2.  Returns (B, 12, dim_w); the last 10 are the new frames."""
        B = wa.shape[0]
        c, wr_emb = self.prepare_cfg_condition(wa, wr, self.block + self.ctx)
        steps, zero = self._adaLN(2 * B, self.block, self.ctx)
        x_t = mx.concatenate([prev_x_t[:, -self.ctx:], noise.astype(mx.float32)], axis=1)
        x_t = self._denoise(x_t, c, wr_emb, steps, start_pos, self.kv_cache, self.ctx, self.block)
        self._update_kv_cache(x_t, c, wr_emb, zero, start_pos)
        return x_t
