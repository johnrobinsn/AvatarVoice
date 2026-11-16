"""Produce a batch of sample videos in one process so torch.compile only pays once."""
import os, sys, torch, uuid, subprocess, time, tempfile
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "avatarforcing"))
from inference import InferenceAgent, seed_everything

SAMPLES = [
    # (avatar_ref, seed, nfe, a_cfg, u_cfg, tag)
    ("data/rumi.jpg",       25, 10, 2.0, 1.0, "rumi_s25_n10_a2"),
    ("data/rumi.jpg",       42, 10, 2.0, 1.0, "rumi_s42_n10_a2"),
    ("data/rumi.jpg",        7, 10, 2.0, 1.0, "rumi_s7_n10_a2"),
    ("data/rumi.jpg",       25, 20, 2.0, 1.0, "rumi_s25_n20_a2_HIGHQ"),
    ("data/rumi.jpg",       25, 10, 3.0, 1.0, "rumi_s25_n10_a3_STRONGLIP"),
    ("data/rumi.jpg",       25, 10, 2.0, 2.0, "rumi_s25_n10_u2_STRONGREACT"),
    ("data/alt_avatar.jpg", 25, 10, 2.0, 1.0, "altavatar_s25_n10_a2"),
    ("data/alt_avatar.jpg", 42, 10, 2.0, 1.0, "altavatar_s42_n10_a2"),
]

def main():
    class Opt: pass
    opt = Opt()
    cfg = OmegaConf.load("configs/inference.yaml")
    for k, v in cfg.items(): setattr(opt, k, v)

    # args that InferenceAgent needs
    opt.infer_config = "configs/inference.yaml"
    opt.mae_ckpt_path = "pretrained_dir/motion_autoencoder.pth"
    opt.ckpt_path = "pretrained_dir/flow_transformer.pth"
    opt.rank, opt.ngpus = 0, 1
    opt.result_dir = "results/samples"
    opt.compile = True
    opt.warmup = 0

    os.makedirs(opt.result_dir, exist_ok=True)

    # Build the agent — compiles once
    print(">>> Initializing agent (torch.compile wrap)")
    agent = InferenceAgent(opt)

    print(">>> Warmup run (paying compile cost)")
    t0 = time.perf_counter()
    agent.run_inference(
        avatar_ref_path   = "data/rumi.jpg",
        avatar_audio_path = "data/avatar.wav",
        user_audio_path   = "data/user.wav",
        user_video_path   = "data/user/",
        a_cfg_scale=2.0, u_cfg_scale=1.0, nfe=10, seed=25,
    )
    print(f">>> warmup done in {time.perf_counter()-t0:.1f}s")

    # Now produce each sample
    for avatar, seed, nfe, a_cfg, u_cfg, tag in SAMPLES:
        print(f"\n>>> {tag}")
        t0 = time.perf_counter()
        seed_everything(seed)
        avatar_name = os.path.splitext(os.path.basename(avatar))[0]
        out_filename = f"{tag}-seed{seed}.mp4"
        # Call agent but with tagged filename
        data = agent.data_processor.preprocess(
            avatar_ref_path=avatar,
            avatar_audio_path="data/avatar.wav",
            user_audio_path="data/user.wav",
            user_video_path="data/user/",
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            d_hat = agent.G.inference(
                data=data,
                a_cfg_scale=a_cfg, u_cfg_scale=u_cfg,
                nfe=nfe, seed=seed, use_kv_cache=True,
            )['d_hat']
        out_path = os.path.join(opt.result_dir, out_filename)
        agent.save_video(d_hat, out_path, "data/avatar.wav")
        dt = time.perf_counter() - t0
        print(f">>> {tag}: {dt:.2f}s")

if __name__ == "__main__":
    main()
