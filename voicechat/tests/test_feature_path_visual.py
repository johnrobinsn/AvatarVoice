"""
Visual parity test — generate the same clip two ways and write MP4s for
side-by-side comparison. Also compute aggregate pixel stats in uint8 space
where the comparison is actually meaningful.
"""
import os, sys, math, subprocess, tempfile, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "avatarforcing"))

import torch
import numpy as np
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
    opt.result_dir = "results/parity_visual"
    opt.compile = False
    opt.warmup = 0
    os.makedirs(opt.result_dir, exist_ok=True)
    return InferenceAgent(opt)


def pixels_to_u8(pixels_block: torch.Tensor) -> np.ndarray:
    """[1, k, 3, H, W] float -> [k, H, W, 3] uint8 using the same normalize as save_video."""
    vid = pixels_block.squeeze(0).detach().clamp(-1, 1).cpu()
    vid = ((vid - vid.min()) / (vid.max() - vid.min() + 1e-8) * 255.0)
    vid = vid.permute(0, 2, 3, 1).contiguous().to(torch.uint8).numpy()
    return vid


def write_mp4(frames_u8: np.ndarray, path: str, fps: int = 25, audio_path: str = None):
    """frames_u8: [T, H, W, 3] uint8 RGB."""
    T, H, W, C = frames_u8.shape
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ff.stdin.write(frames_u8.tobytes())
    ff.stdin.close()
    ff.wait()
    if audio_path and os.path.exists(audio_path):
        subprocess.call(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", tmp, "-i", audio_path, "-shortest",
             "-c:v", "copy", "-c:a", "aac", path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(tmp)
    else:
        os.rename(tmp, path)


def run_full(agent, data, avatar_wa=None, user_wa=None, seed=25):
    seed_everything(seed)
    all_pix = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for block_idx, pixels, is_last in agent.G.inference_stream(
            data=data, a_cfg_scale=2.0, u_cfg_scale=1.0,
            nfe=10, seed=seed,
            avatar_wa=avatar_wa, user_wa=user_wa,
        ):
            all_pix.append(pixels_to_u8(pixels))
    return np.concatenate(all_pix, axis=0)


def main():
    agent = build_agent()
    data = agent.data_processor.preprocess(
        avatar_ref_path   = "data/rumi.jpg",
        avatar_audio_path = "data/avatar.wav",
        user_audio_path   = "data/user.wav",
        user_video_path   = "data/user/",
    )
    B = 1
    avatar_a = data['avatar_a'].to(agent.rank)
    user_a   = data['user_a'].to(agent.rank)
    T = math.ceil(max(avatar_a.shape[-1], user_a.shape[-1]) * agent.opt.fps / agent.opt.sampling_rate)
    T = min(T, len(data['user_frame']))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        avatar_wa_ext = agent.G.audio_encoder.inference(avatar_a, seq_len=T)
        user_wa_ext   = agent.G.audio_encoder.inference(user_a,   seq_len=T)

    print("A: internal features path")
    frames_A = run_full(agent, data, avatar_wa=None, user_wa=None, seed=25)
    print(f"  frames_A: {frames_A.shape}")

    print("B: external features path")
    frames_B = run_full(agent, data, avatar_wa=avatar_wa_ext, user_wa=user_wa_ext, seed=25)
    print(f"  frames_B: {frames_B.shape}")

    assert frames_A.shape == frames_B.shape, "shape mismatch"

    # uint8 pixel diff
    diff = np.abs(frames_A.astype(np.int16) - frames_B.astype(np.int16))
    print(f"\nuint8 pixel diff on {frames_A.shape[0]} frames:")
    print(f"  mean: {diff.mean():.3f}")
    print(f"  max:  {diff.max()}")
    print(f"  p99:  {np.percentile(diff, 99):.2f}")
    print(f"  p99.9:{np.percentile(diff, 99.9):.2f}")
    print(f"  fraction of pixels with diff > 5:   {(diff > 5).mean()*100:.3f}%")
    print(f"  fraction of pixels with diff > 10:  {(diff > 10).mean()*100:.3f}%")
    print(f"  fraction of pixels with diff > 20:  {(diff > 20).mean()*100:.3f}%")

    out_A = os.path.join("results/parity_visual", "A_internal.mp4")
    out_B = os.path.join("results/parity_visual", "B_external.mp4")
    out_diff = os.path.join("results/parity_visual", "diff_x4.mp4")

    write_mp4(frames_A, out_A, audio_path="data/avatar.wav")
    write_mp4(frames_B, out_B, audio_path="data/avatar.wav")
    # Amplified diff video
    diff_amp = np.clip(diff.astype(np.int16) * 4, 0, 255).astype(np.uint8)
    write_mp4(diff_amp, out_diff)

    print(f"\nWrote:\n  {out_A}\n  {out_B}\n  {out_diff}")

    # Pass if mean uint8 diff < 1.0 and 99.9th pct diff < 15
    mean_ok = diff.mean() < 1.0
    p999_ok = np.percentile(diff, 99.9) < 15
    if mean_ok and p999_ok:
        print("\n✓ PASS: mean uint8 diff < 1.0 and p99.9 < 15 — visually indistinguishable")
        return 0
    else:
        print("\n✗ FAIL: material pixel divergence")
        return 1


if __name__ == "__main__":
    sys.exit(main())
