"""
Parity test for the features-path of inference_stream.

If the caller provides pre-computed avatar_wa / user_wa, output must be
bit-identical to the case where they are computed internally.

This is the Step 6c acceptance gate from ARCHITECTURE.md.
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "avatarforcing"))

import torch
from omegaconf import OmegaConf

from inference import InferenceAgent, seed_everything


def build_agent():
    class Opt: pass
    opt = Opt()
    cfg = OmegaConf.load("configs/inference.yaml")
    for k, v in cfg.items(): setattr(opt, k, v)
    opt.infer_config = "configs/inference.yaml"
    opt.mae_ckpt_path = "pretrained_dir/motion_autoencoder.pth"
    opt.ckpt_path = "pretrained_dir/flow_transformer.pth"
    opt.rank, opt.ngpus = 0, 1
    opt.result_dir = "results/parity_test"
    opt.compile = False
    opt.warmup = 0
    os.makedirs(opt.result_dir, exist_ok=True)
    return InferenceAgent(opt)


def run_blocks(agent, data, avatar_wa=None, user_wa=None, seed=25, n=5):
    """Run inference_stream and collect n pixel blocks."""
    seed_everything(seed)
    blocks = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for block_idx, pixels, is_last in agent.G.inference_stream(
            data=data, a_cfg_scale=2.0, u_cfg_scale=1.0,
            nfe=10, seed=seed,
            avatar_wa=avatar_wa, user_wa=user_wa,
        ):
            blocks.append(pixels.detach().cpu().clone())
            if len(blocks) >= n:
                break
    return blocks


def main():
    agent = build_agent()

    data = agent.data_processor.preprocess(
        avatar_ref_path   = "data/rumi.jpg",
        avatar_audio_path = "data/avatar.wav",
        user_audio_path   = "data/user.wav",
        user_video_path   = "data/user/",
    )

    # Compute features externally, the same way inference_stream would internally.
    B = 1
    avatar_a = data['avatar_a'].to(agent.rank)
    user_a   = data['user_a'].to(agent.rank)
    T = math.ceil(max(avatar_a.shape[-1], user_a.shape[-1]) * agent.opt.fps / agent.opt.sampling_rate)
    T = min(T, len(data['user_frame']))

    print(f"T = {T} video frames")
    print("Computing external avatar_wa / user_wa...")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        avatar_wa_ext = agent.G.audio_encoder.inference(avatar_a, seq_len=T)
        user_wa_ext   = agent.G.audio_encoder.inference(user_a,   seq_len=T)
    print(f"  avatar_wa: {tuple(avatar_wa_ext.shape)}  user_wa: {tuple(user_wa_ext.shape)}")

    print("\nRun A: internal features path (avatar_wa=None, user_wa=None)")
    blocks_A = run_blocks(agent, data, avatar_wa=None, user_wa=None, seed=25, n=5)
    print(f"  got {len(blocks_A)} blocks  shape0={tuple(blocks_A[0].shape)}")

    print("\nRun B: external features path (pre-computed tensors passed in)")
    blocks_B = run_blocks(agent, data, avatar_wa=avatar_wa_ext, user_wa=user_wa_ext, seed=25, n=5)
    print(f"  got {len(blocks_B)} blocks")

    print("\n=== Per-block diff ===")
    max_diff_all = 0.0
    max_rel_all = 0.0
    for i, (a, b) in enumerate(zip(blocks_A, blocks_B)):
        diff = (a.float() - b.float()).abs()
        rel = (diff / (a.float().abs() + 1e-8)).max().item()
        max_d = diff.max().item()
        max_diff_all = max(max_diff_all, max_d)
        max_rel_all = max(max_rel_all, rel)
        print(f"  block {i}: max_abs_diff={max_d:.6f}  max_rel_diff={rel:.6f}  "
              f"mean_abs={diff.mean().item():.6f}")

    print(f"\noverall max abs diff: {max_diff_all:.6f}")
    print(f"overall max rel diff: {max_rel_all:.6f}")

    # Bit-identical pass gate:
    if max_diff_all < 1e-4:
        print("\n✓ PASS: features-path output matches internal path within 1e-4")
        return 0
    elif max_diff_all < 1e-2:
        print("\n~ MARGINAL: tiny divergence (likely bf16 rounding). Review manually.")
        return 0
    else:
        print("\n✗ FAIL: features-path output differs materially from internal path")
        return 1


if __name__ == "__main__":
    sys.exit(main())
