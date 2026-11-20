"""Compile motion_autoencoder.dec (MAE decoder) to an AOTInductor .pt2.

The MAE decoder runs 1x per block (vs flow_transformer's 20x), but torch.compile
on it still contributes ~8-12s to warmup. Pre-compiling it lets us skip that.

Run once per machine (per GPU + torch version):
    python scripts/compile_mae_aot.py                      # build for current GPU
    python scripts/compile_mae_aot.py --force              # rebuild

Shapes are captured by running one real inference block, intercepting the
first call to motion_autoencoder.dec, then exporting at those shapes.
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


def build_agent(device: str):
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
    opt.rank = int(device.split(":")[-1]) if ":" in device else 0
    opt.ngpus = 1
    opt.compile = False
    opt.warmup = 0
    opt.result_dir = str(ROOT / "results/compile_mae_aot")
    os.makedirs(opt.result_dir, exist_ok=True)
    return InferenceAgent(opt)


def capture_mae_call(agent, device):
    """Run a short inference and intercept the first call to
    motion_autoencoder.dec. Returns (wa_tensor, feats_list) captured args."""
    captured = {}
    dec = agent.G.motion_autoencoder.dec
    orig_fwd = dec.forward

    def grab(wa, alpha, feats):
        if "args" not in captured and alpha is None:
            captured["args"] = (
                wa.detach().clone(),
                [f.detach().clone() for f in feats],
            )
        return orig_fwd(wa, alpha, feats)

    dec.forward = grab
    try:
        data = agent.data_processor.preprocess(
            avatar_ref_path   = str(ROOT / "data/rumi.jpg"),
            avatar_audio_path = str(AF_ROOT / "data/avatar.wav"),
            user_audio_path   = str(AF_ROOT / "data/user.wav"),
            user_video_path   = str(AF_ROOT / "data/user/"),
        )
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for block_idx, pixels, is_last in agent.G.inference_stream(
                    data=data, a_cfg_scale=2.0, u_cfg_scale=1.0, nfe=10, seed=25):
                if "args" in captured:
                    break
    finally:
        dec.forward = orig_fwd

    if "args" not in captured:
        raise RuntimeError("did not capture a motion_autoencoder.dec call")
    return captured["args"]


class MAEWrapper(torch.nn.Module):
    """Wrap motion_autoencoder.dec so alpha=None is baked in, feats flattened
    to a tuple positional arg, and only the image tensor is returned
    (dec returns (img, something_else))."""
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, wa, feats_flat):
        feats = list(feats_flat)
        img, _ = self.dec(wa, None, feats)
        return img


def default_out_path():
    cap = torch.cuda.get_device_capability(0)
    return ROOT / "compiled_models" / f"mae_decoder_sm{cap[0]}{cap[1]}.pt2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true")
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

    t = time.perf_counter()
    wa, feats = capture_mae_call(agent, args.device)
    print(f"  captured MAE call in {time.perf_counter() - t:.1f}s")
    print(f"    wa:    {tuple(wa.shape)} {wa.dtype}")
    for i, f in enumerate(feats):
        print(f"    feat[{i}]: {tuple(f.shape)} {f.dtype}")

    wrapper = MAEWrapper(agent.G.motion_autoencoder.dec).eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)
    # Keep MAE in fp32: it has mixed-dtype internals (grid_sample on bf16
    # feat + fp32 flow). bf16 cast of the whole module fails at the flow
    # path. MAE is 1x/block so fp32 cost is acceptable (~15-25 ms vs bf16
    # ~5-10 ms — still comfortably within the 400 ms block budget).
    wa = wa.to(torch.float32)
    feats = [f.to(torch.float32) for f in feats]

    # Eager sanity
    t = time.perf_counter()
    with torch.inference_mode():
        out_ref = wrapper(wa, tuple(feats))
    print(f"  eager sanity: img {tuple(out_ref.shape)} {out_ref.dtype} in {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    with torch.inference_mode():
        ep = torch.export.export(wrapper, args=(wa, tuple(feats)))
    print(f"  exported ({len(list(ep.graph.nodes))} nodes) in {time.perf_counter() - t:.1f}s")

    from torch._inductor import aoti_compile_and_package
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    pkg = aoti_compile_and_package(ep, package_path=str(out_path))
    print(f"  AOTI compiled -> {pkg} in {time.perf_counter() - t:.1f}s "
          f"({out_path.stat().st_size // (1024*1024)} MB)")

    print(f"\ntotal: {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()
