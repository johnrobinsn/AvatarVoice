"""
Validation: does `AudioEncoder.rolling_slice` produce features close enough to
the batch path (`inference(full_audio)` + slice) that the downstream avatar
pixels don't visibly diverge?

Sweeps lookback and lookahead sizes, reports cosine similarity + max abs diff
per slice position for the full `avatar.wav` sample. If any config lands in
"features effectively identical" territory (cosine > 0.98 across all slices),
that's the one we'll use in `inference_stream_rolling`.

Run (from repo root):
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=1 python voicechat/tests/test_rolling_wav2vec.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "avatarforcing"))

import torch
import librosa
import numpy as np
from omegaconf import OmegaConf

from inference import InferenceAgent


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float(); b = b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))


def build_agent():
    class Opt: pass
    opt = Opt()
    cfg = OmegaConf.load("configs/inference.yaml")
    for k, v in cfg.items(): setattr(opt, k, v)
    opt.infer_config = "configs/inference.yaml"
    opt.mae_ckpt_path = "pretrained_dir/motion_autoencoder.pth"
    opt.ckpt_path = "pretrained_dir/flow_transformer.pth"
    opt.rank, opt.ngpus = 0, 1
    opt.result_dir = "results/rolling_test"
    opt.compile = False
    opt.warmup = 0
    os.makedirs(opt.result_dir, exist_ok=True)
    return InferenceAgent(opt)


def main():
    agent = build_agent()
    enc = agent.G.audio_encoder
    fps = agent.opt.fps
    spf = agent.opt.sampling_rate // fps

    # Load avatar.wav the same way DataProcessor does.
    speech, _ = librosa.load("data/avatar.wav", sr=agent.opt.sampling_rate)
    audio = enc.wav2vec_preprocessor = None  # not used; we bypass preprocessor since inference() does padding
    audio = torch.from_numpy(speech).unsqueeze(0).to(agent.rank)
    print(f"audio: {audio.shape[1]} samples = {audio.shape[1]/agent.opt.sampling_rate:.2f}s")

    T = audio.shape[1] // spf
    print(f"T = {T} video frames")

    # Full batch reference
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        full = enc.inference(audio, seq_len=T)
    print(f"full batch feats: {tuple(full.shape)}")

    # Slice positions matching inference_stream's subsequent-block loop:
    # for each block at frame t_idx, we use features[ss_idx-2 : ee_idx]
    # where ss_idx = t_idx, ee_idx = t_idx + num_frames_per_block.
    num_frames_per_block = agent.G.num_frames_per_block
    num_frames_for_clip  = agent.G.num_frames_for_clip
    block_starts = list(range(num_frames_for_clip, T - num_frames_per_block + 1, num_frames_per_block))

    configs = [
        ("lb  0, la   0",    0,   0),
        ("lb 20, la   0",   20,   0),
        ("lb 20, la  10",   20,  10),
        ("lb 20, la  25",   20,  25),
        ("lb 20, la  50",   20,  50),
        ("lb 50, la  25",   50,  25),
        ("lb 50, la  50",   50,  50),
        ("lb 50, la 100",   50, 100),
        ("lb 100,la 100",  100, 100),
        ("lb 100,la 200",  100, 200),
        ("lb 200,la 200",  200, 200),
    ]

    print(f"\n{len(block_starts)} block positions × {len(configs)} configs")
    print()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for name, lb, la in configs:
            cos_vals = []
            abs_vals = []
            t0 = time.perf_counter()
            for t_idx in block_starts:
                ss_idx = t_idx
                ee_idx = t_idx + num_frames_per_block
                frame_lo = ss_idx - 2
                frame_hi = ee_idx
                ref  = full[:, frame_lo:frame_hi]
                roll = enc.rolling_slice(audio, frame_lo, frame_hi,
                                         lookback_frames=lb, lookahead_frames=la)
                # Align shapes (rolling may differ by 0 since we match exactly)
                n = min(ref.shape[1], roll.shape[1])
                cos_vals.append(cosine(ref[:, :n], roll[:, :n]))
                abs_vals.append(float((ref[:, :n] - roll[:, :n]).abs().max()))
            dt = time.perf_counter() - t0
            cos_arr = np.array(cos_vals)
            abs_arr = np.array(abs_vals)
            print(f"  {name:28s} cos min/mean={cos_arr.min():.4f}/{cos_arr.mean():.4f}  "
                  f"abs_max max/mean={abs_arr.max():.3f}/{abs_arr.mean():.3f}  "
                  f"({dt*1000:.0f}ms for {len(block_starts)} slices)")


if __name__ == "__main__":
    main()
