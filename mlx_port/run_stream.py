"""End-to-end streaming run of the hybrid MLX pipeline on real audio + portrait.

Writes an mp4 (with audio), prints per-block timing (EWMA like the live worker),
and with --compare checks frames against the PyTorch reference:

  gold      : AvatarForcing's own sampling code, torch float32 (CPU) +
              original motion decoder, torch float32 (MPS)
  torch-bf16: same, flow transformer under CPU bf16 autocast — how far bf16
              alone moves frames from gold (the CUDA app runs bf16)
  mlx       : this pipeline (MLX flow, default fp32 + fast fp16 decoder)

Usage:
    python run_stream.py                                  # default clip + rumi.jpg
    python run_stream.py --audio speech.wav --image ../data/ken.jpg --seconds 8
    python run_stream.py --compare --seconds 6
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx

import pipeline_mlx as P

DEFAULT_AUDIO = os.path.join(P.AF_PATH, "assets", "rumi-seed25-88942ee804.mp4")
DEFAULT_IMAGE = os.path.join(P.HERE, "..", "data", "rumi.jpg")
RESULTS = os.path.join(P.HERE, "..", "results")


def write_video(path: str, frames: np.ndarray, pcm: np.ndarray, fps: int = 25, sr: int = 16000):
    import av
    from fractions import Fraction
    with av.open(path, "w") as out:
        vs = out.add_stream("libx264", rate=fps)
        vs.width, vs.height, vs.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        vs.options = {"crf": "18"}
        as_ = out.add_stream("aac", rate=sr)
        as_.layout = "mono"
        for f in frames:
            for pkt in vs.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                out.mux(pkt)
        for pkt in vs.encode():
            out.mux(pkt)
        pcm = pcm[: int(len(frames) / fps * sr)]
        step = 1024
        for i in range(0, len(pcm), step):
            af = av.AudioFrame.from_ndarray(pcm[None, i:i + step].astype(np.float32), format="flt", layout="mono")
            af.sample_rate, af.pts, af.time_base = sr, i, Fraction(1, sr)
            for pkt in as_.encode(af):
                out.mux(pkt)
        for pkt in as_.encode():
            out.mux(pkt)


def ssim(a: np.ndarray, b: np.ndarray, device="mps") -> np.ndarray:
    """Per-frame SSIM (11x11 Gaussian, sigma 1.5) on uint8 [T, H, W, 3], channel-averaged."""
    g = torch.exp(-((torch.arange(11) - 5.0) ** 2) / (2 * 1.5 ** 2))
    w = (g[:, None] * g[None, :]) / g.sum() ** 2
    w = w.view(1, 1, 11, 11).repeat(3, 1, 1, 1).to(device)
    out = []
    for i in range(0, len(a), 16):
        x = torch.from_numpy(a[i:i + 16]).permute(0, 3, 1, 2).float().to(device)
        y = torch.from_numpy(b[i:i + 16]).permute(0, 3, 1, 2).float().to(device)
        f = lambda t: F.conv2d(t, w, groups=3)
        mx_, my = f(x), f(y)
        sxx, syy, sxy = f(x * x) - mx_ ** 2, f(y * y) - my ** 2, f(x * y) - mx_ * my
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        s = ((2 * mx_ * my + c1) * (2 * sxy + c2)) / ((mx_ ** 2 + my ** 2 + c1) * (sxx + syy + c2))
        out.append(s.mean(dim=(1, 2, 3)).cpu().numpy())
    return np.concatenate(out)


def make_noises(T: int, clip: int, block: int, seed: int):
    rng = np.random.default_rng(seed)
    n_blocks = len(range(clip, T, block))
    return [rng.standard_normal((1, clip, 512)).astype(np.float32)] + \
           [rng.standard_normal((1, block, 512)).astype(np.float32) for _ in range(n_blocks)]


def torch_reference_frames(image, wa_np, noises, T, bf16: bool) -> np.ndarray:
    """PyTorch reference: AvatarForcing sampling (CPU) + original decoder fp32 (MPS)."""
    import test_equivalence as te
    import decoder_fast_torch
    decoder_fast_torch.unpatch()
    clip, blk = 50, 10
    n_blocks = len(noises) - 1
    total = clip + n_blocks * blk + 2
    wa_pad = np.concatenate([wa_np, np.repeat(wa_np[:, -1:], total - wa_np.shape[1], axis=1)], axis=1)
    af = te._torch_af_stub(te.torch_model(), nfe=10, a_cfg_scale=2.0)
    g = P.load_generator("mps")
    with torch.inference_mode():
        s_r, _, feats = g.enc(image, input_target=None)
        r_s = g.dec.direction(g.enc.fc(s_r)).cpu().numpy()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        lat = te.torch_stream_latents(af, wa_pad, r_s, noises, n_blocks)
    lat = np.concatenate([lat[0]] + [l[:, -blk:] for l in lat[1:]], axis=1)[:, :T]
    feats_e = [f.repeat_interleave(blk, 0) for f in feats]
    frames = []
    with torch.inference_mode():
        for b in range(0, T, blk):
            z = torch.from_numpy(lat[:, b:b + blk]).to("mps")
            k = z.shape[1]
            if k < blk:
                z = torch.cat([z, z[:, -1:].expand(1, blk - k, -1)], dim=1)
            img, _ = g.dec((s_r.unsqueeze(1) + z).reshape(blk, -1), alpha=None, feats=feats_e)
            frames.append(P.to_uint8(img[:k]))
    return np.concatenate(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=DEFAULT_AUDIO)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--seconds", type=float, default=None, help="truncate audio")
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--out", default=os.path.join(RESULTS, "mlx_stream.mp4"))
    ap.add_argument("--compare", action="store_true", help="also run the PyTorch reference and report SSIM")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--flow-dtype", default="float32", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--decoder-dtype", default="float16", choices=["float16", "float32"])
    args = ap.parse_args()

    pcm = P.load_audio(args.audio)
    if args.seconds:
        pcm = pcm[: int(args.seconds * 16000)]
    T = int(np.ceil(len(pcm) * 25 / 16000))
    print(f"audio {len(pcm) / 16000:.2f}s -> {T} frames")

    image = P.load_image(args.image, "mps")
    noises = make_noises(T, 50, 10, args.seed)

    refs = {}
    if args.compare:
        # Reference runs come first: they need the original (unpatched) decoder.
        enc_opt = P.load_opt()
        wa_ref = P.load_audio_encoder(enc_opt, "mps")
        with torch.inference_mode():
            a = torch.from_numpy(P.normalize_audio(pcm)).unsqueeze(0).to("mps")
            wa_np = wa_ref.inference(a, seq_len=T).float().cpu().numpy()
        for name, bf16 in (("gold", False), ("torch-bf16", True)):
            t0 = time.perf_counter()
            refs[name] = torch_reference_frames(image, wa_np, noises, T, bf16)
            print(f"{name} reference: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    pipe = P.HybridAvatarPipeline(flow_dtype=getattr(mx, args.flow_dtype),
                                  decoder_dtype=getattr(torch, args.decoder_dtype),
                                  compile_decoder=not args.no_compile)
    print(f"flow {args.flow_dtype} (MLX), decoder {args.decoder_dtype} (torch MPS)")
    pipe.set_reference(image)
    print(f"pipeline ready in {time.perf_counter() - t0:.1f}s (incl. decoder compile)")

    t0 = time.perf_counter()
    wa = pipe.audio_features(pcm, T)
    mx.eval(wa)
    print(f"wav2vec ({len(pcm) / 16000:.1f}s audio): {(time.perf_counter() - t0) * 1000:.0f} ms")

    stats = P.StreamStats()
    frames = []
    t_start = time.perf_counter()
    for idx, pix, timing in pipe.stream(wa, seed=args.seed, noises=noises, stats=stats):
        frames.append(P.to_uint8(pix))
    wall = time.perf_counter() - t_start
    frames = np.concatenate(frames)[:T]

    st = stats.timings
    tot = sorted(t.total_ms for t in st)
    print(f"\nsteady-state blocks: {len(st)}  (each = 10 frames = 400 ms of video)")
    print(f"  flow    median {np.median([t.flow_ms for t in st]):6.1f} ms")
    print(f"  decode  median {np.median([t.decode_ms for t in st]):6.1f} ms")
    print(f"  total   median {np.median(tot):6.1f}  p90 {tot[int(len(tot) * 0.9)]:6.1f}  max {tot[-1]:6.1f}  "
          f"EWMA {stats.ewma_ms:6.1f} ms   (target EWMA < 400)")
    print(f"  realtime factor {400 / stats.ewma_ms:.2f}x   wall {wall:.1f}s for {T / 25:.1f}s of video")

    print(f"  memory: MLX peak {mx.get_peak_memory() / 1e9:.2f} GB, "
          f"torch MPS driver {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_video(args.out, frames, pcm)
    print(f"wrote {args.out}")

    if refs:
        print("\nper-frame SSIM vs gold (spec target >= 0.995):")
        for name, fr in (("mlx", frames), ("torch-bf16", refs["torch-bf16"])):
            s = ssim(fr, refs["gold"])
            print(f"  {name:<11s} mean {s.mean():.4f}  min {s.min():.4f}  p5 {np.percentile(s, 5):.4f}  "
                  f"frames>=0.995: {(s >= 0.995).mean() * 100:.0f}%")
        s = ssim(frames, refs["torch-bf16"])
        print(f"  mlx vs torch-bf16: mean {s.mean():.4f}  min {s.min():.4f}")
        gold_path = os.path.join(os.path.dirname(args.out), "gold.mp4")
        if not os.path.exists(gold_path):
            write_video(gold_path, refs["gold"], pcm)


if __name__ == "__main__":
    main()
