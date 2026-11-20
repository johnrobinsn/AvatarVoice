"""Hybrid Apple-Silicon avatar pipeline: MLX flow transformer + torch-MPS encoders/decoder.

    reference portrait ─► motion encoder (torch MPS fp32, once) ─► s_r, feats, r_s
    avatar audio       ─► wav2vec2 + audio_projection (torch MPS) ─► wa  [B, T, 512]
    wa, r_s            ─► FlowSamplerMLX (MLX fp32)               ─► motion latents
    latents            ─► motion decoder (torch MPS fp16, decoder_fast_torch, torch.compile)
                                                                   ─► RGB frames

Boundary crossings per block: wa slice torch->MLX, latents MLX->torch (both tiny).

`stream()` mirrors AvatarForcing.inference_stream: a 50-frame primer, then
10-frame blocks with 2 frames of overlap, yielding decoded frames per block.
User audio / user motion are null, as in the voice-chat app.

Precision: the flow transformer defaults to float32.  bf16 is ~25 ms/block
faster, but diffusion sampling amplifies its rounding (as it does for PyTorch's
own bf16): per-frame SSIM vs the fp32 reference drops from ~0.997 to ~0.976.
The decoder runs fp16 (SSIM cost < 0.001).  See MLX_PERF.md.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
AF_PATH = os.path.abspath(os.path.join(HERE, "..", "avatarforcing"))
if AF_PATH not in sys.path:
    sys.path.insert(0, AF_PATH)

from omegaconf import OmegaConf

from models.avatarforcing.generator import Generator
from models.avatarforcing.AvatarForcing import AudioEncoder

import decoder_fast_torch
from flow_transformer_mlx import FlowTransformerMLX, load_weights
from sampler_mlx import FlowSamplerMLX

PRETRAINED = os.path.join(AF_PATH, "pretrained_dir")


def load_opt():
    opt = OmegaConf.load(os.path.join(AF_PATH, "configs", "inference.yaml"))
    opt.wav2vec_model_path = os.path.join(PRETRAINED, "wav2vec2-base-960h")
    return opt


def _load_params(module: torch.nn.Module, ckpt: str, prefix: str):
    """Copy parameters only (like inference.py:load_ckpt — buffers are not loaded)."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    params = dict(module.named_parameters())
    missing = set(params) - set(sd)
    assert not missing, f"missing params: {sorted(missing)[:5]}"
    with torch.no_grad():
        for k, p in params.items():
            p.copy_(sd[k])
    return module


def load_generator(device) -> Generator:
    g = Generator(512, 512, 20)
    _load_params(g, os.path.join(PRETRAINED, "motion_autoencoder.pth"), "motion_autoencoder.")
    return g.to(device).eval()


def load_audio_encoder(opt, device) -> AudioEncoder:
    # Note: transformers warns that `pos_conv_embed.conv.weight_g/weight_v` were
    # "not used".  That's spurious — torch's weight_norm parametrization remaps
    # those legacy keys via a load-state-dict hook (verified: effective conv
    # weight == weight_g * v / ||v||).
    enc = AudioEncoder(opt)
    sd = torch.load(os.path.join(PRETRAINED, "flow_transformer.pth"), map_location="cpu", weights_only=True)
    enc.audio_projection.load_state_dict(
        {k[len("audio_encoder.audio_projection."):]: v for k, v in sd.items()
         if k.startswith("audio_encoder.audio_projection.")})
    return enc.to(device).eval()


def load_image(path: str, device) -> torch.Tensor:
    """512x512 RGB in [-1, 1], shape [1, 3, 512, 512] (A.Normalize(0.5, 0.5))."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if img.size != (512, 512):
        img = img.resize((512, 512), Image.BICUBIC)
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return (x * 2 - 1).unsqueeze(0).to(device)


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Mono float32 PCM at `sr`, from any file PyAV can read."""
    import av
    with av.open(path) as c:
        res = av.AudioResampler(format="flt", layout="mono", rate=sr)
        chunks = []
        for frame in c.decode(audio=0):
            for f in res.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in res.resample(None):
            chunks.append(f.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32)


def normalize_audio(pcm: np.ndarray) -> np.ndarray:
    """Wav2Vec2FeatureExtractor (do_normalize=True): zero mean, unit variance."""
    return ((pcm - pcm.mean()) / np.sqrt(pcm.var() + 1e-7)).astype(np.float32)


def to_uint8(pixels: torch.Tensor) -> np.ndarray:
    """[k, 3, H, W] in [-1, 1] -> [k, H, W, 3] uint8."""
    x = ((pixels.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()


@dataclass
class BlockTiming:
    flow_ms: float
    decode_ms: float
    total_ms: float


@dataclass
class StreamStats:
    timings: list = field(default_factory=list)
    ewma_ms: float | None = None
    alpha: float = 0.1

    def add(self, t: BlockTiming):
        self.timings.append(t)
        self.ewma_ms = t.total_ms if self.ewma_ms is None else \
            (1 - self.alpha) * self.ewma_ms + self.alpha * t.total_ms


class HybridAvatarPipeline:
    def __init__(self, flow_dtype=mx.float32, decoder_dtype=torch.float16, compile_decoder: bool = True,
                 nfe: int = 10, a_cfg_scale: float = 2.0, device: str = "mps"):
        self.opt = opt = load_opt()
        self.device = device
        self.clip = int(opt.sec * opt.fps)
        self.block = opt.num_frames_per_block

        self.flow = FlowTransformerMLX(opt)
        load_weights(self.flow, os.path.join(PRETRAINED, "flow_transformer_mlx.safetensors"), dtype=flow_dtype)
        self.sampler = FlowSamplerMLX(self.flow, nfe=nfe, a_cfg_scale=a_cfg_scale)

        self.generator = load_generator(device)              # encoder stays fp32 (QR has no fp16 MPS kernel)
        self.audio_encoder = load_audio_encoder(opt, device)

        decoder_fast_torch.patch()
        self.decoder_dtype = decoder_dtype
        self.generator.dec.to(decoder_dtype)
        self._decode = self.generator.dec
        self._compile = compile_decoder
        self._compiled = None

    # ---------------- encoders ----------------

    @torch.inference_mode()
    def set_reference(self, image: torch.Tensor):
        """Encode the avatar portrait once.  image: [1, 3, 512, 512] in [-1, 1]."""
        enc = self.generator.enc
        s_r, _, feats = enc(image.to(self.device), input_target=None)
        r_s = self.generator.dec.direction.float()(enc.fc(s_r))
        self.s_r = s_r.to(self.decoder_dtype).unsqueeze(1)                       # [1, 1, 512]
        self.feats = [f.to(self.decoder_dtype).repeat_interleave(self.block, 0) for f in feats]
        self.r_s = mx.array(r_s.float().cpu().numpy())                          # [1, 512]
        self._warm_decoder()

    @torch.inference_mode()
    def audio_features(self, pcm_16k: np.ndarray, n_frames: int | None = None,
                       normalize: bool = True) -> mx.array:
        """wav2vec2 + audio_projection -> MLX [1, T, 512].

        normalize=True matches inference.py (Wav2Vec2FeatureExtractor); the live
        voice-chat worker feeds raw TTS PCM (normalize=False)."""
        if n_frames is None:
            n_frames = int(np.ceil(len(pcm_16k) * self.opt.fps / self.opt.sampling_rate))
        pcm = normalize_audio(pcm_16k) if normalize else np.asarray(pcm_16k, dtype=np.float32)
        a = torch.from_numpy(pcm).unsqueeze(0).to(self.device)
        wa = self.audio_encoder.inference(a, seq_len=n_frames)
        return mx.array(wa.float().cpu().numpy())

    # ---------------- decoder ----------------

    def _warm_decoder(self):
        """First (eager) call fills decoder_fast_torch's grid cache; then compile."""
        z = torch.zeros(1, self.block, 512, device=self.device)
        self.decode(z)
        if self._compile and self._compiled is None:
            self._compiled = torch.compile(self.generator.dec, dynamic=False)
            self._decode = self._compiled
            self.decode(z)

    @torch.inference_mode()
    def decode(self, latents) -> torch.Tensor:
        """latents [1, k, 512] (MLX or torch, k <= 10) -> pixels [k, 3, 512, 512] on device."""
        if isinstance(latents, mx.array):
            latents = torch.from_numpy(np.array(latents.astype(mx.float32)))
        latents = latents.to(self.device, self.decoder_dtype)
        k = latents.shape[1]
        if k < self.block:                       # replicate-pad to the compiled block size
            latents = torch.cat([latents, latents[:, -1:].expand(1, self.block - k, -1)], dim=1)
        s = (self.s_r + latents).reshape(self.block, -1)
        img, _ = self._decode(s, alpha=None, feats=self.feats)
        return img[:k]

    # ---------------- streaming ----------------

    def stream(self, wa: mx.array, seed: int = 25, noises: list | None = None, emit_primer: bool = True,
               stats: StreamStats | None = None):
        """Yield (block_idx, pixels [k, 3, H, W] torch, BlockTiming).

        wa: [1, T, 512] audio features.  `noises` (optional) — list of numpy arrays
        [1, 50, 512] then [1, 10, 512] per block, for reproducible comparisons."""
        T = wa.shape[1]
        rng = np.random.default_rng(seed)
        noise = lambda i, L: mx.array(noises[i] if noises is not None else
                                      rng.standard_normal((1, L, 512)).astype(np.float32))
        pad = lambda a, L: a if a.shape[1] >= L else \
            mx.concatenate([a, mx.repeat(a[:, -1:], L - a.shape[1], axis=1)], axis=1)

        self.sampler.reset(1)
        t0 = time.perf_counter()
        x_t = self.sampler.prime(pad(wa[:, :self.clip], self.clip), self.r_s, noise(0, self.clip))
        mx.eval(x_t, self.sampler.kv_cache)
        t_flow = time.perf_counter()
        if emit_primer:
            for b in range(0, min(self.clip, T), self.block):
                t1 = time.perf_counter()
                pix = self.decode(x_t[:, b:min(b + self.block, T)])
                torch.mps.synchronize()
                yield b // self.block, pix, BlockTiming((t_flow - t0) * 1000 if b == 0 else 0.0,
                                                        (time.perf_counter() - t1) * 1000, 0.0)

        for i, t_idx in enumerate(range(self.clip, T, self.block)):
            t0 = time.perf_counter()
            wa_t = pad(wa[:, t_idx - 2:t_idx + self.block], self.block + 2)
            x_t = self.sampler.step(wa_t, self.r_s, x_t, noise(i + 1, self.block), t_idx - 2)
            latents = x_t[:, -self.block:][:, :min(self.block, T - t_idx)]
            mx.eval(latents, self.sampler.kv_cache)
            t1 = time.perf_counter()
            pix = self.decode(latents)
            torch.mps.synchronize()
            t2 = time.perf_counter()
            timing = BlockTiming((t1 - t0) * 1000, (t2 - t1) * 1000, (t2 - t0) * 1000)
            if stats is not None:
                stats.add(timing)
            yield t_idx // self.block, pix, timing
