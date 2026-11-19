"""One-shot converter: PyTorch flow_transformer.pth -> MLX safetensors.

Usage:
    python convert_weights.py \
        --src ../avatarforcing/pretrained_dir/flow_transformer.pth \
        --dst ../avatarforcing/pretrained_dir/flow_transformer_mlx.safetensors

What it does:
  1. Loads the PyTorch state_dict on CPU.
  2. Renames any keys that differ between the PyTorch module tree and the
     MLX module tree (see RENAME_MAP below).
  3. Unpacks nn.MultiheadAttention's packed in_proj_weight into separate
     query/key/value Linear weights (MLX's nn.MultiHeadAttention uses split
     Linears, PyTorch uses a packed one).
  4. Saves as safetensors (fp32 arrays, cast at load time on MLX side).

After conversion, inspect with:
    python -c "from safetensors.numpy import load_file; d = load_file('.../flow_transformer_mlx.safetensors'); [print(k, v.shape, v.dtype) for k,v in d.items()]"
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file


# Keys in PyTorch that need renaming before save.  Empty for now — we expect
# most keys to map 1:1 since the MLX module tree mirrors PyTorch's structure.
# Fill in as you discover divergences during Phase 2.
RENAME_MAP: dict[str, str] = {
    # Example (hypothetical):
    # "blocks.0.attn.freqs_cis": DROP,   # buffer, regenerated at init
}

DROP = "__DROP__"


def unpack_mha(packed_weight: np.ndarray, packed_bias: np.ndarray | None, embed_dim: int):
    """Split PyTorch nn.MultiheadAttention's packed in_proj (3*embed_dim, embed_dim)
    into three Linear weights (embed_dim, embed_dim) for MLX nn.MultiHeadAttention.

    Returns dict with "query_proj", "key_proj", "value_proj" (weight + optional bias).
    """
    assert packed_weight.shape[0] == 3 * embed_dim, f"expected {3 * embed_dim}, got {packed_weight.shape}"
    q_w = packed_weight[0:embed_dim]
    k_w = packed_weight[embed_dim:2 * embed_dim]
    v_w = packed_weight[2 * embed_dim:3 * embed_dim]
    out = {
        "query_proj.weight": q_w,
        "key_proj.weight":   k_w,
        "value_proj.weight": v_w,
    }
    if packed_bias is not None:
        out["query_proj.bias"] = packed_bias[0:embed_dim]
        out["key_proj.bias"]   = packed_bias[embed_dim:2 * embed_dim]
        out["value_proj.bias"] = packed_bias[2 * embed_dim:3 * embed_dim]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to flow_transformer.pth")
    ap.add_argument("--dst", required=True, help="Path to write MLX safetensors")
    ap.add_argument("--embed_dim", type=int, default=512,
                    help="embed_dim for nn.MultiheadAttention unpacking (= opt.dim_a; default 512)")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"ERROR: source not found: {src}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading PyTorch weights from {src} ...")
    sd = torch.load(src, map_location="cpu", weights_only=True)
    print(f"  loaded {len(sd)} keys")

    out: dict[str, np.ndarray] = {}
    renamed = dropped = unpacked = 0

    # Match keys like "c_embedder.dual_modal_attn.in_proj_weight"
    MHA_PACKED_RE = re.compile(r"^(.+)\.in_proj_weight$")

    # First pass: handle packed MHA weights
    mha_prefixes: set[str] = set()
    for k in sd:
        m = MHA_PACKED_RE.match(k)
        if m:
            mha_prefixes.add(m.group(1))

    for prefix in mha_prefixes:
        pw = sd[f"{prefix}.in_proj_weight"].numpy()
        pb = sd.get(f"{prefix}.in_proj_bias")
        pb = pb.numpy() if pb is not None else None
        unpacked_dict = unpack_mha(pw, pb, args.embed_dim)
        for suffix, arr in unpacked_dict.items():
            out[f"{prefix}.{suffix}"] = arr
        unpacked += 1

    # Second pass: everything else
    for k, v in sd.items():
        m = MHA_PACKED_RE.match(k)
        if m and m.group(1) in mha_prefixes:
            continue  # handled above
        if k.endswith(".in_proj_bias") and k[:-len(".in_proj_bias")] in mha_prefixes:
            continue  # handled above
        if k.endswith(".out_proj.weight"):
            # PyTorch MHA.out_proj maps to MLX MHA.out_proj — same name, keep
            pass
        # Apply rename map
        new_k = RENAME_MAP.get(k, k)
        if new_k == DROP:
            dropped += 1
            continue
        if new_k != k:
            renamed += 1
        arr = v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        # Cast complex-polar buffers (freqs_cis) are not in the state_dict if
        # register_buffer(persistent=False), so no need to handle here.
        out[new_k] = arr.astype(np.float32)

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(out)} keys to {dst} ...")
    save_file(out, str(dst))
    print(f"  renamed={renamed}  dropped={dropped}  MHA unpacked={unpacked}")
    print(f"  file size: {dst.stat().st_size / 1e6:.1f} MB")
    print("done.")


if __name__ == "__main__":
    main()
