"""Numerical equivalence tests: MLX output ≈ PyTorch output (real weights).

The PyTorch reference runs on CPU in float32.  Each test runs the MLX port in
two modes:
  strict — float32 on the MLX CPU stream.  Checks the port is mathematically
           identical.  (Not the GPU: fp32 matmuls on the M5 GPU carry ~1e-3
           relative error, which would mask real bugs.)
  bf16   — bfloat16 weights/activations on the GPU, i.e. how it will be deployed.

Run bottom-up:
  1. helpers   — RoPE, masks
  2. modules   — Attention, CrossAttention, c_embedder, adaLN precompute
  3. forward   — full forward_with_precomputed (primer + steady state + KV update)
  4. loop      — full sampling loop vs AvatarForcing's own solve_cfg/update_kv_cache

Usage:
    python test_equivalence.py              # run all
    python test_equivalence.py helpers      # tests whose name contains "helpers"
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import torch
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
AF_PATH = os.path.abspath(os.path.join(HERE, "..", "avatarforcing"))
sys.path.insert(0, AF_PATH)

from omegaconf import OmegaConf

import models.avatarforcing.flow_transformer as tft
from models import FlowMatchScheduler
from models.avatarforcing.AvatarForcing import AvatarForcing

import helpers_mlx as hm
from flow_transformer_mlx import FlowTransformerMLX, load_weights
from sampler_mlx import FlowSamplerMLX

PTH = os.path.join(AF_PATH, "pretrained_dir", "flow_transformer.pth")
SAFETENSORS = os.path.join(AF_PATH, "pretrained_dir", "flow_transformer_mlx.safetensors")

# (strict tol, bf16 tol) per test family.
# strict: max_abs / max(1,|ref|);  bf16: relative RMS error.
TOL_MODULE = (1e-4, 1e-2)
TOL_FORWARD = (1e-3, 2e-2)

RESULTS: list[tuple[str, bool]] = []


# ============================================================
#  Fixtures
# ============================================================

def _opt():
    opt = OmegaConf.load(os.path.join(AF_PATH, "configs", "inference.yaml"))
    opt.rank = "cpu"
    return opt


_CACHE: dict = {}


def torch_model() -> tft.FlowTransformer:
    if "torch" not in _CACHE:
        torch.manual_seed(0)
        m = tft.FlowTransformer(_opt())
        sd = torch.load(PTH, map_location="cpu", weights_only=True)
        sd = {k[len("flow_transformer."):]: v for k, v in sd.items() if k.startswith("flow_transformer.")}
        # Load parameters only, like inference.py:load_ckpt.  The checkpoint's
        # mask buffers are the inverse of the generated ones (a training-time
        # convention) and are never loaded by the app.
        params = dict(m.named_parameters())
        assert set(params) == set(sd) - {"alignment_mask", "block_bidir_causal_lookahead_mask"}
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(sd[k])
        m.eval()
        # Extend RoPE tables like voicechat/server.py does, so high start_pos works.
        for blk in m.blocks:
            blk.attn.freqs_cis = tft.precompute_freqs_cis(blk.attn.head_dim, 65536)
        _CACHE["torch"] = m
    return _CACHE["torch"]


def mlx_model(mode: str) -> FlowTransformerMLX:
    if mode not in _CACHE:
        m = FlowTransformerMLX(_opt())
        load_weights(m, SAFETENSORS, dtype=mx.float32 if mode == "strict" else mx.bfloat16)
        _CACHE[mode] = m
    return _CACHE[mode]


class Mode:
    """Context manager selecting the MLX device for a mode."""
    def __init__(self, mode):
        self.mode = mode
    def __enter__(self):
        self.prev = mx.default_device()
        mx.set_default_device(mx.cpu if self.mode == "strict" else mx.gpu)
        return self
    def __exit__(self, *a):
        mx.set_default_device(self.prev)


def _np(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().float().cpu().numpy()
    if isinstance(a, mx.array):
        return np.array(a.astype(mx.float32))
    return np.asarray(a)


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a))


def _m(a: np.ndarray) -> mx.array:
    return mx.array(a)


def check(name: str, ref, out, tol: float, baseline=None) -> bool:
    """strict: max_abs <= tol * max(1, |ref|).
    bf16 (name starts with "[bf16]"): relative RMS error <= tol, or <= 1.25x the
    error of `baseline` (PyTorch's own bf16-autocast output) when given."""
    r, o = _np(ref), _np(out)
    if r.shape != o.shape:
        print(f"  FAIL {name}: shape ref={r.shape} mlx={o.shape}")
        RESULTS.append((name, False))
        return False
    d = float(np.max(np.abs(r - o)))
    scale = float(np.max(np.abs(r)))
    rel_rms = float(np.sqrt(np.mean((r - o) ** 2)) / max(np.sqrt(np.mean(r ** 2)), 1e-12))
    if name.startswith("[bf16]"):
        ok = rel_rms <= tol
        extra = ""
        if baseline is not None:
            b = _np(baseline)
            base_rms = float(np.sqrt(np.mean((r - b) ** 2)) / max(np.sqrt(np.mean(r ** 2)), 1e-12))
            ok = ok or rel_rms <= 1.25 * base_rms
            extra = f"  torch-bf16={base_rms:.2e}"
        print(f"  {'PASS' if ok else 'FAIL'} {name:<52s} rel_rms={rel_rms:.2e} (tol {tol:.0e}){extra}")
    else:
        ok = d <= tol * max(1.0, scale)      # absolute for |x| <= 1, relative above
        print(f"  {'PASS' if ok else 'FAIL'} {name:<52s} max_abs={d:.2e} rel={d / max(scale, 1e-12):.2e} "
              f"(tol {tol:.0e}·max(1,|ref|))")
    RESULTS.append((name, ok))
    return ok


def _ref_and_baseline(fn):
    """Run a PyTorch reference twice: float32, and under CPU bf16 autocast (how
    the CUDA app runs).  The bf16 run is the yardstick for MLX bf16 error."""
    ref = fn()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        base = fn()
    return ref, base


def rng(seed=0):
    return np.random.default_rng(seed)


def audio_feats(r, B, L, dim):
    """Shaped like AudioEncoder output (Linear -> LayerNorm -> SiLU)."""
    x = r.standard_normal((B, L, dim)).astype(np.float32)
    return (x / (1 + np.exp(-x))).astype(np.float32)


# ============================================================
#  1. helpers
# ============================================================

def test_helpers():
    print("helpers")
    ref = tft.precompute_freqs_cis(128, 2048)
    out = hm.precompute_freqs_cis(128, 2048)
    check("precompute_freqs_cis real", ref.real, mx.real(out), 1e-3)
    check("precompute_freqs_cis imag", ref.imag, mx.imag(out), 1e-3)

    x = rng(1).standard_normal((2, 8, 12, 128)).astype(np.float32)
    for start in (0, 38, 1100):
        fc = ref[start:start + 12]
        r = tft.apply_rotary_emb_q(_t(x), fc)
        check(f"apply_rotary_emb_q (table) start={start}", r,
              hm.apply_rotary_emb_q(_m(x), hm.precompute_freqs_cis(128, 2048)[start:start + 12]), 1e-3)
        with Mode("strict"):
            check(f"mx.fast.rope traditional start={start}", r,
                  mx.fast.rope(_m(x), 128, traditional=True, base=10000.0, scale=1.0, offset=start), 1e-3)

    eq = lambda name, a, b: RESULTS.append((name, bool(np.array_equal(_np(a), _np(b))))) or \
        print(f"  {'PASS' if RESULTS[-1][1] else 'FAIL'} {name}")
    eq("enc_dec_mask(50,50,1,2)", tft.enc_dec_mask(50, 50, 1, 2), hm.enc_dec_mask(50, 50, 1, 2))
    eq("block_bidir_causal_lookahead_mask(50,10,2)",
       tft.make_block_bidir_causal_lookahead_mask(50, 10, 2), hm.make_block_bidir_causal_lookahead_mask(50, 10, 2))
    eq("cross_mask_local(12,2)", tft.CrossAttention.cross_mask_local(12, 2), hm.cross_mask_local(12, 2))
    for args in [(0, 0, 50, 10, 2), (38, 38, 12, 10, 2), (128, 38, 12, 10, 2)]:
        eq(f"block_mask_window{args}", tft.make_block_bidir_causal_lookahead_mask_window(*args),
           hm.make_block_bidir_causal_lookahead_mask_window(*args))
    for args in [(0, 0, 50, 2), (38, 2, 12, 2)]:
        eq(f"enc_dec_mask_window{args}", tft.enc_dec_mask_window(*args), hm.enc_dec_mask_window(*args))


# ============================================================
#  2. modules
# ============================================================

def _kv_np(r, B, L, H=8, D=128):
    return {"k": r.standard_normal((B, H, L, D)).astype(np.float32),
            "v": r.standard_normal((B, H, L, D)).astype(np.float32)}


def _get(d, path):
    for p in path:
        d = d[p]
    return d


def _comparer(R, BASE, mode, tol):
    """check() against R[path], with the torch-bf16 baseline at BASE[path]."""
    def cmp(name, path, out):
        check(f"[{mode}] {name}", _get(R, path), out, tol,
              baseline=_get(BASE, path) if mode == "bf16" else None)
    return cmp


def test_modules():
    print("modules")
    T = torch_model()
    r = rng(2)
    B, H = 4, 1024
    x50 = r.standard_normal((B, 50, H)).astype(np.float32)
    x12 = r.standard_normal((B, 12, H)).astype(np.float32)
    c12 = r.standard_normal((B, 12, H)).astype(np.float32)
    kv_self, kv_cross = _kv_np(r, B, 38), _kv_np(r, B, 2)
    start = 438
    wa = audio_feats(r, B, 12, 512)
    z = np.zeros_like(wa)
    wa_rev = wa[:, ::-1].copy()
    ts = np.linspace(1000, 0, 9).tolist()

    def torch_refs():
        ta, tc = T.blocks[3].attn, T.blocks[3].cross_attn
        fc, bm, am = ta.freqs_cis, T.block_bidir_causal_lookahead_mask, T.alignment_mask
        R = {}
        with torch.no_grad():
            R["attn50"] = ta(_t(x50), mask=bm, freqs_cis_slice=fc[0:50])
            R["kv"] = {k: _t(v.copy()) for k, v in kv_self.items()}
            R["attn12"] = ta(_t(x12), mask=bm, freqs_cis_slice=fc[start:start + 12],
                             kv_cache=R["kv"], update_kv_cache=True)
            R["kv50"] = {k: _t(v.copy()) for k, v in kv_self.items()}
            ta(_t(x50), mask=bm, freqs_cis_slice=fc[0:50], kv_cache=R["kv50"], update_kv_cache=True)
            R["cross50"] = tc(_t(x50), c=_t(x50), mask=am)
            R["ckv"] = {k: _t(v.copy()) for k, v in kv_cross.items()}
            R["cross12"] = tc(_t(x12), c=_t(c12), mask=am, kv_cache=R["ckv"], update_kv_cache=True)
            R["cond"] = T.compute_condition(_t(wa), _t(z), _t(z))
            R["cond_user"] = T.compute_condition(_t(wa), _t(wa * 0.5), _t(wa_rev))
            R["ada"] = T.precompute_timestep_adaLN(ts, B, 10, "cpu", context_len=2)
        return R

    R, BASE = _ref_and_baseline(torch_refs)

    for mode, tol in zip(("strict", "bf16"), TOL_MODULE):
        cmp = _comparer(R, BASE, mode, tol)
        with Mode(mode):
            M = mlx_model(mode)
            ma, mc = M.blocks[3].attn, M.blocks[3].cross_attn
            cast = lambda a: _m(a).astype(M.dtype)
            cmp("Attention primer N=50", ["attn50"], ma(cast(x50), M._block_mask, 0))
            mkv = {k: cast(v) for k, v in kv_self.items()}
            cmp("Attention steady N=12 + cache", ["attn12"],
                ma(cast(x12), M._block_mask, start, kv_cache=mkv, update_kv_cache=True))
            cmp("Attention cache after update (k)", ["kv", "k"], mkv["k"])
            cmp("Attention cache after update (v)", ["kv", "v"], mkv["v"])
            mkv50 = {k: cast(v) for k, v in kv_self.items()}
            ma(cast(x50), M._block_mask, 0, kv_cache=mkv50, update_kv_cache=True)
            cmp("Attention cache after primer update (k)", ["kv50", "k"], mkv50["k"])
            cmp("CrossAttention primer N=50", ["cross50"], mc(cast(x50), cast(x50), mask=M._alignment_mask))
            mckv = {k: cast(v) for k, v in kv_cross.items()}
            cmp("CrossAttention steady N=12 + cache", ["cross12"],
                mc(cast(x12), cast(c12), mask=M._alignment_mask, kv_cache=mckv, update_kv_cache=True))
            cmp("CrossAttention cache after update (k)", ["ckv", "k"], mckv["k"])
            cmp("compute_condition (null user)", ["cond"], M.compute_condition(_m(wa), _m(z), _m(z)))
            cmp("compute_condition (non-null user)", ["cond_user"],
                M.compute_condition(_m(wa), _m(wa * 0.5), _m(wa_rev)))
            ada = M.precompute_timestep_adaLN(ts, B, 10, context_len=2)
            for i in (0, 4, 8):
                cmp(f"adaLN step {i} shift_msa", ["ada", i, 0, 0], ada[i][0][0])
                cmp(f"adaLN step {i} gate_mlp", ["ada", i, 0, 5], ada[i][0][5])
                cmp(f"adaLN step {i} final scale", ["ada", i, 2, 1], ada[i][2][1])


# ============================================================
#  3. forward_with_precomputed
# ============================================================

def test_forward():
    print("forward_with_precomputed")
    T = torch_model()
    r = rng(3)
    B = 2
    ts = np.linspace(1000, 0, 9).tolist()
    x50 = r.standard_normal((B, 50, 512)).astype(np.float32)
    x12 = r.standard_normal((B, 12, 512)).astype(np.float32)
    wa50, wa12 = audio_feats(r, B, 50, 512), audio_feats(r, B, 12, 512)
    wr = r.standard_normal((B, 512)).astype(np.float32)
    kv_np = [(_kv_np(r, B, 38), _kv_np(r, B, 2)) for _ in range(8)]
    start = 98

    def torch_refs():
        R = {}
        with torch.no_grad():
            z50, z12 = torch.zeros(B, 50, 512), torch.zeros(B, 12, 512)
            c50 = T.compute_condition(_t(wa50), z50, z50)
            c12 = T.compute_condition(_t(wa12), z12, z12)
            ada50 = T.precompute_timestep_adaLN(ts, B, 50, "cpu", 0)
            ada12 = T.precompute_timestep_adaLN(ts, B, 10, "cpu", 2)
            fc = T.blocks[0].attn.freqs_cis
            R["primer"] = T.forward_with_precomputed(_t(x50), c50, ada50[2], T.compute_wr_embedded(_t(wr), 50),
                                                     fc[0:50], None, False)
            R["kv"] = [tuple({k: _t(v.copy()) for k, v in d.items()} for d in pair) for pair in kv_np]
            R["steady"] = T.forward_with_precomputed(_t(x12), c12, ada12[5], T.compute_wr_embedded(_t(wr), 12),
                                                     fc[start:start + 12], R["kv"], True)
            # full forward() path at steady state, against the updated cache
            t_tok = torch.cat([torch.zeros(B, 2), torch.full((B, 10), 250.0)], dim=1)
            R["full"] = T(t_tok, _t(x12), _t(wa12), z12, z12, _t(wr), train=False, start_pos=start + 10,
                          kv_cache_list=R["kv"], update_kv_cache=False)
        return R

    R, BASE = _ref_and_baseline(torch_refs)

    for mode, tol in zip(("strict", "bf16"), TOL_FORWARD):
        cmp = _comparer(R, BASE, mode, tol)
        with Mode(mode):
            M = mlx_model(mode)
            mz50, mz12 = mx.zeros((B, 50, 512)), mx.zeros((B, 12, 512))
            mc50 = M.compute_condition(_m(wa50), mz50, mz50)
            mc12 = M.compute_condition(_m(wa12), mz12, mz12)
            ma50 = M.precompute_timestep_adaLN(ts, B, 50, 0)
            ma12 = M.precompute_timestep_adaLN(ts, B, 10, 2)
            cmp("primer N=50 (no cache)", ["primer"],
                M.forward_with_precomputed(_m(x50), mc50, ma50[2], M.compute_wr_embedded(_m(wr), 50), 0))
            mkv = [tuple({k: _m(v).astype(M.dtype) for k, v in d.items()} for d in pair) for pair in kv_np]
            cmp("steady N=12 + cache + update", ["steady"],
                M.forward_with_precomputed(_m(x12), mc12, ma12[5], M.compute_wr_embedded(_m(wr), 12),
                                           start, mkv, True))
            for i in (0, 7):
                cmp(f"block {i} self-cache k after update", ["kv", i, 0, "k"], mkv[i][0]["k"])
                cmp(f"block {i} cross-cache v after update", ["kv", i, 1, "v"], mkv[i][1]["v"])
            t_tok = mx.concatenate([mx.zeros((B, 2)), mx.full((B, 10), 250.0)], axis=1)
            cmp("full forward() t per-token", ["full"],
                M(t_tok, _m(x12), _m(wa12), mz12, mz12, _m(wr), start_pos=start + 10, kv_cache_list=mkv))


# ============================================================
#  4. full sampling loop
# ============================================================

def _torch_af_stub(T: tft.FlowTransformer, nfe: int, a_cfg_scale: float):
    """Bind AvatarForcing's own sampling methods to a minimal object, so the
    reference is the real solve_cfg / solve_closed / update_kv_cache code."""
    opt = T.opt
    s = types.SimpleNamespace()
    s.opt, s.rank, s.flow_transformer = opt, "cpu", T
    s.num_frames_for_clip = int(opt.sec * opt.fps)
    s.num_frames_per_block = opt.num_frames_per_block
    s.scheduler = FlowMatchScheduler(num_steps=1000, shift=1, sigma_min=0.0, extra_one_step=False)
    s.scheduler.set_timesteps(opt.num_train_timestep, training=True)
    s.denoising_step_list = torch.tensor(np.linspace(opt.num_train_timestep, 0, nfe - 1).tolist())
    for name in ("solve_cfg", "solve_closed", "prepare_cfg_condition", "update_kv_cache", "initialize_kv_cache"):
        setattr(s, name, types.MethodType(getattr(AvatarForcing, name), s))
    s.a_cfg_scale = a_cfg_scale
    return s


def torch_stream_latents(af, wa, wr, noises, n_blocks):
    """Latent path of AvatarForcing.inference_stream (user inputs null)."""
    B, clip, blk = 1, af.num_frames_for_clip, af.num_frames_per_block
    af.initialize_kv_cache(batch_size=B, dtype=torch.float32, device="cpu")
    zeros = lambda L: torch.zeros(B, L, af.opt.dim_w)
    out = []
    with torch.inference_mode():
        x_t = _t(noises[0])
        pc = af.prepare_cfg_condition(_t(wa[:, :clip]), zeros(clip), zeros(clip), _t(wr), seq_len=clip, context_len=0)
        for i, ts in enumerate(af.denoising_step_list):
            x_t = af.solve_cfg(B=B, index=i, current_timestep=ts, x_t=x_t, precomputed_c=pc[0],
                               precomputed_wr=pc[1], precomputed_adaLN=pc[2], start_pos=0, context_len=0,
                               use_kv_cache=False, a_cfg_scale=af.a_cfg_scale, u_cfg_scale=1.0)
        af.update_kv_cache(final_latents=x_t, precomputed_c=pc[0], precomputed_wr=pc[1],
                           precomputed_adaLN=pc[2], start_pos=0)
        out.append(x_t.numpy())
        for b in range(n_blocks):
            t_idx = clip + b * blk
            start_pos = t_idx - 2
            wa_t = _t(wa[:, t_idx - 2:t_idx + blk])
            x_t = torch.cat([x_t[:, -2:], _t(noises[b + 1])], dim=1)
            pc = af.prepare_cfg_condition(wa_t, zeros(blk + 2), zeros(blk + 2), _t(wr), seq_len=blk, context_len=2)
            for i, ts in enumerate(af.denoising_step_list):
                x_t = af.solve_cfg(B=B, index=i, current_timestep=ts, x_t=x_t, precomputed_c=pc[0],
                                   precomputed_wr=pc[1], precomputed_adaLN=pc[2], start_pos=start_pos,
                                   context_len=2, use_kv_cache=True, a_cfg_scale=af.a_cfg_scale, u_cfg_scale=1.0)
            af.update_kv_cache(final_latents=x_t, precomputed_c=pc[0], precomputed_wr=pc[1],
                               precomputed_adaLN=pc[2], start_pos=start_pos)
            out.append(x_t.numpy())
    return out


def mlx_stream_latents(sampler: FlowSamplerMLX, wa, wr, noises, n_blocks):
    clip, blk = sampler.clip, sampler.block
    sampler.reset(1)
    out = []
    x_t = sampler.prime(_m(wa[:, :clip]), _m(wr), _m(noises[0]))
    mx.eval(x_t)
    out.append(np.array(x_t))
    for b in range(n_blocks):
        t_idx = clip + b * blk
        x_t = sampler.step(_m(wa[:, t_idx - 2:t_idx + blk]), _m(wr), x_t, _m(noises[b + 1]), t_idx - 2)
        mx.eval(x_t)
        out.append(np.array(x_t))
    return out


def test_loop(n_blocks: int = 6):
    print(f"sampling loop (primer + {n_blocks} blocks, nfe=10, cfg=2.0)")
    T = torch_model()
    r = rng(4)
    clip, blk = 50, 10
    total = clip + n_blocks * blk
    wa = audio_feats(r, 1, total, 512)
    wr = r.standard_normal((1, 512)).astype(np.float32)
    noises = [r.standard_normal((1, clip, 512)).astype(np.float32)] + \
             [r.standard_normal((1, blk, 512)).astype(np.float32) for _ in range(n_blocks)]

    af = _torch_af_stub(T, nfe=10, a_cfg_scale=2.0)
    ref = torch_stream_latents(af, wa, wr, noises, n_blocks)
    # PyTorch's own bf16 trajectory (CPU autocast, as the CUDA app runs): the
    # yardstick for how far bf16 sampling drifts from float32 on its own.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        base = torch_stream_latents(_torch_af_stub(T, nfe=10, a_cfg_scale=2.0), wa, wr, noises, n_blocks)

    for mode, tol in zip(("strict", "bf16"), (1e-3, None)):
        with Mode(mode):
            for skip in ((False, True) if mode == "strict" else (True,)):
                sampler = FlowSamplerMLX(mlx_model(mode), nfe=10, a_cfg_scale=2.0, skip_noop_steps=skip)
                out = mlx_stream_latents(sampler, wa, wr, noises, n_blocks)
                tag = f"[{mode}{' skip-noop' if skip else ''}]"
                for b in range(n_blocks + 1):
                    name = f"{tag} {'primer' if b == 0 else f'block {b}'} latents"
                    if tol is not None:
                        check(name, ref[b], out[b], tol)
                    else:
                        _report_similarity(name, ref[b], out[b], base[b])


def _similarity(ref, out):
    r, o = ref.reshape(-1, ref.shape[-1]), out.reshape(-1, out.shape[-1])
    cos = np.sum(r * o, -1) / (np.linalg.norm(r, axis=-1) * np.linalg.norm(o, axis=-1))
    rel_rms = float(np.sqrt(np.mean((ref - out) ** 2)) / np.sqrt(np.mean(ref ** 2)))
    return rel_rms, float(cos.min())


def _report_similarity(name, ref, out, base):
    """bf16 trajectories: pass if MLX-bf16 stays within 1.5x (+0.01) of the drift
    PyTorch-bf16 itself shows against the float32 reference."""
    rel_rms, cos = _similarity(ref, out)
    b_rms, b_cos = _similarity(ref, base)
    ok = rel_rms <= 1.5 * b_rms + 0.01
    print(f"  {'PASS' if ok else 'FAIL'} {name:<44s} rel_rms={rel_rms:.2e} min_cos={cos:.4f} | "
          f"torch-bf16 rel_rms={b_rms:.2e} min_cos={b_cos:.4f}")
    RESULTS.append((name, ok))


ALL_TESTS = [test_helpers, test_modules, test_forward, test_loop]


def main():
    filt = sys.argv[1] if len(sys.argv) > 1 else None
    for fn in ALL_TESTS:
        if filt and filt not in fn.__name__:
            continue
        fn()
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
