"""Compile the flow_transformer read-only path to an AOTInductor .pt2 archive.

Run once per machine (per GPU + torch version). Writes to:
    compiled_models/flow_transformer_sm{cap}_n12.pt2

At server startup, if a matching .pt2 is present, inference.py loads it
instead of running torch.compile. First run of a session still exercises
the write path (update_kv_cache=True) and the warmup path (N=50) via
eager — those are 1x/block and 1x/session respectively.

Usage:
    python scripts/compile_aot.py                              # build for current GPU
    python scripts/compile_aot.py --force                      # rebuild even if .pt2 exists
    python scripts/compile_aot.py --out /path/to/x.pt2         # custom output path
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AF_ROOT = ROOT / "avatarforcing"
sys.path.insert(0, str(AF_ROOT))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".inductor_cache"))
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("CPATH", str(Path(os.environ["CUDA_HOME"]) / "include"))

import torch
from omegaconf import OmegaConf


# Matches the live flow_transformer: steady-state shapes with
# num_frames_per_block=10 → N = 10 + 2 = 12.
N_STEADY = 12
BATCH = 2       # 2× for cond+uncond CFG batching
NUM_HEADS = 8
HEAD_DIM = 128
DIM_H = 1024
DIM_W = 512
TRANSFORMER_DEPTH = 8
CACHE_LEN = 38  # num_frames_for_clip - num_frames_per_block - 2
CROSS_CACHE_LEN = 2  # attention_window


def build_agent(device: str):
    """Load the real model (same configuration the server uses)."""
    from inference import InferenceAgent

    class Opt: pass
    opt = Opt()
    cfg = OmegaConf.load(str(AF_ROOT / "configs/inference.yaml"))
    for k, v in cfg.items(): setattr(opt, k, v)
    opt.infer_config = str(AF_ROOT / "configs/inference.yaml")
    opt.mae_ckpt_path = str(AF_ROOT / "pretrained_dir/motion_autoencoder.pth")
    opt.ckpt_path = str(AF_ROOT / "pretrained_dir/flow_transformer.pth")
    opt.wav2vec_model_path = str(AF_ROOT / opt.wav2vec_model_path)
    opt.pretrained_dir = str(AF_ROOT / opt.pretrained_dir)
    opt.rank = 0 if device == "cuda:0" else int(device.split(":")[-1])
    opt.ngpus = 1
    opt.compile = False            # don't torch.compile for the export
    opt.warmup = 0
    opt.result_dir = str(ROOT / "results/compile_aot")
    os.makedirs(opt.result_dir, exist_ok=True)
    return InferenceAgent(opt)


class ReadOnlyWrapper(torch.nn.Module):
    """Export-friendly wrapper around FlowTransformer.forward_with_precomputed.

    Bakes in update_kv_cache=False, flattens the adaLN tuple-of-tuples and
    kv_cache list-of-dicts into plain tensor args, and casts fp32 inputs to
    bf16 at the boundary so the exported graph sees consistent dtypes.
    """
    def __init__(self, ft):
        super().__init__()
        self.ft = ft

    def forward(self,
                x, precomputed_c,
                scale_shift_alpha_0, scale_shift_alpha_1, scale_shift_alpha_2,
                scale_shift_alpha_3, scale_shift_alpha_4, scale_shift_alpha_5,
                t_embed, final_adaLN_0, final_adaLN_1,
                precomputed_wr, freqs_cis_slice,
                kv_cache_flat):
        x = x.to(torch.bfloat16)
        precomputed_wr = precomputed_wr.to(torch.bfloat16)
        kv_cache_flat = tuple(t.to(torch.bfloat16) for t in kv_cache_flat)
        scale_shift_alpha = (scale_shift_alpha_0, scale_shift_alpha_1,
                             scale_shift_alpha_2, scale_shift_alpha_3,
                             scale_shift_alpha_4, scale_shift_alpha_5)
        final_adaLN = (final_adaLN_0, final_adaLN_1)
        precomputed_adaLN = (scale_shift_alpha, t_embed, final_adaLN)
        kv_cache_list = []
        for i in range(0, len(kv_cache_flat), 4):
            kv_cache_list.append((
                {"k": kv_cache_flat[i], "v": kv_cache_flat[i + 1]},
                {"k": kv_cache_flat[i + 2], "v": kv_cache_flat[i + 3]},
            ))
        return self.ft.forward_with_precomputed(
            x, precomputed_c, precomputed_adaLN, precomputed_wr,
            freqs_cis_slice, kv_cache_list, False)


def make_synthetic_inputs(device: str):
    """Shapes matching the live steady-state call. Dtypes also match what
    forward_with_precomputed receives in the live pipeline (bf16 for the
    autocast'd tensors, fp32 for the ones that stay in fp32)."""
    gen = torch.Generator(device=device).manual_seed(42)
    def _r(shape, dtype):
        return torch.randn(*shape, device=device, dtype=dtype, generator=gen)

    x = _r((BATCH, N_STEADY, DIM_W), torch.float32)
    precomputed_c = _r((BATCH, N_STEADY, DIM_H), torch.bfloat16)
    scale_shift_alpha = [
        _r((BATCH, N_STEADY, DIM_H), torch.bfloat16) for _ in range(6)]
    t_embed = _r((BATCH, N_STEADY, DIM_H), torch.bfloat16)
    final_adaLN = [_r((BATCH, N_STEADY, DIM_H), torch.bfloat16) for _ in range(2)]
    precomputed_wr = _r((BATCH, N_STEADY, DIM_W), torch.float32)
    # freqs_cis_slice: real-valued [N, head_dim] where first half is cos,
    # second half is sin. The content doesn't matter for compile — torch.export
    # only cares about shapes/dtypes.
    freqs_cis_slice = torch.zeros((N_STEADY, HEAD_DIM), device=device, dtype=torch.float32)
    kv_cache_flat = []
    for _ in range(TRANSFORMER_DEPTH):
        kv_cache_flat += [
            _r((BATCH, NUM_HEADS, CACHE_LEN, HEAD_DIM), torch.float32),         # self K
            _r((BATCH, NUM_HEADS, CACHE_LEN, HEAD_DIM), torch.float32),         # self V
            _r((BATCH, NUM_HEADS, CROSS_CACHE_LEN, HEAD_DIM), torch.float32),   # cross K
            _r((BATCH, NUM_HEADS, CROSS_CACHE_LEN, HEAD_DIM), torch.float32),   # cross V
        ]
    return (x, precomputed_c, *scale_shift_alpha, t_embed, *final_adaLN,
            precomputed_wr, freqs_cis_slice, tuple(kv_cache_flat))


def default_out_path():
    cap = torch.cuda.get_device_capability(0)
    return ROOT / "compiled_models" / f"flow_transformer_sm{cap[0]}{cap[1]}_n12.pt2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None,
                    help="Output .pt2 path (default: compiled_models/flow_transformer_sm{cap}_n12.pt2)")
    ap.add_argument("--device", default="cuda:0", help="CUDA device")
    ap.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = ap.parse_args()

    out_path = args.out or default_out_path()
    if out_path.exists() and not args.force:
        print(f"{out_path} already exists; pass --force to rebuild. done.")
        return

    t_total = time.perf_counter()
    print(f"target: {out_path}")
    print(f"GPU:    {torch.cuda.get_device_name(0)} "
          f"(sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]})")

    t = time.perf_counter()
    agent = build_agent(args.device)
    print(f"  loaded AvatarForcing in {time.perf_counter() - t:.1f}s")

    wrapper = ReadOnlyWrapper(agent.G.flow_transformer).eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)
    wrapper.ft = wrapper.ft.to(torch.bfloat16)

    flat = make_synthetic_inputs(args.device)

    # sanity check — the wrapper should run eagerly on these synthetic inputs
    t = time.perf_counter()
    with torch.inference_mode():
        out_sanity = wrapper(*flat)
    print(f"  eager sanity: out shape {tuple(out_sanity.shape)} in {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    with torch.inference_mode():
        ep = torch.export.export(wrapper, args=flat)
    print(f"  exported ({len(list(ep.graph.nodes))} nodes) in {time.perf_counter() - t:.1f}s")

    from torch._inductor import aoti_compile_and_package
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    pkg = aoti_compile_and_package(ep, package_path=str(out_path))
    print(f"  AOTI compiled -> {pkg} in {time.perf_counter() - t:.1f}s "
          f"({out_path.stat().st_size // (1024*1024)} MB)")

    print(f"\ntotal: {time.perf_counter() - t_total:.1f}s")
    print("done. server will pick this up automatically on next start.")


if __name__ == "__main__":
    main()
