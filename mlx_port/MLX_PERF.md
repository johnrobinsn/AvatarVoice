# MLX port — measured results

Machine: MacBook Pro, **Apple M5 (base), 32 GB**, macOS 26.6. MLX 0.32.4, torch 2.15.0 (MPS).
(The spec targets an M5 Pro; this is the smaller chip.)

## Headline

Full avatar pipeline (MLX flow transformer → torch-MPS motion decoder), real speech
(15.8 s clip) + `data/rumi.jpg`, nfe=10, CFG 2.0, 25 fps, 512×512, 10 frames/block:

| | Target | Measured |
|---|---|---|
| Per-block EWMA (steady state) | < 400 ms | **210 ms** (1.91× realtime) |
| Per-block max | < 600 ms | **213 ms** |
| KV re-anchor (50-frame primer, flow only) | — | 68 ms fp32 / 52 ms bf16 |
| Unified memory (avatar pipeline) | < 16 GB total | MLX 0.76 GB + MPS 4.7 GB |
| Per-frame SSIM vs fp32 PyTorch reference | ≥ 0.995 | mean **0.9975**, min 0.9943 (89% of frames ≥ 0.995) |

`python mlx_port/run_stream.py [--compare]` reproduces these.

## Per-component (per 10-frame block)

| Component | Before (torch MPS, as-is) | After | How |
|---|---|---|---|
| Flow transformer | not runnable on MPS (float64 in `solve_closed`) | **58 ms** fp32 / **37.5 ms** bf16 (MLX) | MLX port; 9 forward passes per block (8 denoise + 1 KV update) |
| Motion decoder | 613 ms fp32 / 510 ms fp16-autocast | **150–159 ms** fp16 | `decoder_fast_torch.py` + pure fp16 + `torch.compile` |
| wav2vec2 + projection | — | ~2–4 ms per block of speech | torch MPS, unchanged (runs once per phrase) |

For reference, the CUDA path (2× RTX Pro 6000, `reduce-overhead`): flow transformer
135 ms offline, ~210 ms live EWMA.

### Flow transformer (MLX), `bench.py`

| dtype | steady block | primer (50 frames) |
|---|---|---|
| float32 | 57–62 ms | 68 ms |
| bfloat16 | 37.5 ms | 52 ms |
| float16 | 37.4 ms | — |
| bf16, without skipping the no-op step | 41.7 ms | — |

The last denoising step of every block has t = 0 → next t = 0, which `solve_closed`
treats as "history" for every token (output = input). Its forward pass is discarded,
so `FlowSamplerMLX` skips it (`skip_noop_steps=True`, verified bit-identical).

### Motion decoder (torch MPS), `bench_decoder.py`

| Variant | 10 frames @ 512 | max abs vs original fp32 |
|---|---|---|
| original, fp32 | 613 ms | — |
| original, fp16 autocast | 510 ms | 1.4e-2 |
| fast kernels, fp16 autocast | 458 ms | 1.1e-2 |
| fast kernels, pure fp16 | 333 ms | 1.1e-1 |
| fast kernels, pure fp16, `torch.compile` | **150 ms** | 6.9e-2 |

(pixels in [-1, 1]; isolated max-pixel errors — per-frame SSIM impact < 0.001)

Where the time went: convolutions are only ~60–70 ms of compute at the ~12 TFLOP/s MPS
sustains on these shapes. The rest was (a) grouped per-sample modulated convs,
(b) `upfirdn2d` as a single-channel conv over B·C maps, (c) autocast promoting every
activation back to fp32 at `FusedLeakyReLU`'s fp32 bias, (d) a numpy-built sampling grid
copied from CPU every `ToFlow` call, and (e) un-fused elementwise ops over 170 MB
tensors. The rewrites are exact (max abs diff 2.4e-5 in fp32 on CPU); `torch.compile`
(Inductor's MPS backend) fuses the elementwise chains.

MLX convolutions were evaluated for a decoder port and are *slower* than MPS at high
resolution (64→32 ch @ 512: MLX 57 ms vs MPS 7.6 ms channels-last), so the decoder
stays on torch. Channels-last activations helped eager fp16 slightly (~4%) but nothing
once compiled, so they're not used.

## Precision and quality

Per-frame SSIM vs the gold reference (PyTorch fp32 sampling + original fp32 decoder),
6 s clip, same noise:

| Flow / decoder | EWMA | SSIM mean | SSIM min | frames ≥ 0.995 |
|---|---|---|---|---|
| MLX bf16 / fp16 | 189 ms | 0.976 | 0.916 | 23% |
| **MLX fp32 / fp16 (default)** | **214 ms** | **0.9975** | **0.9943** | **89%** |
| MLX fp32 / fp32 | 372 ms | 0.9982 | 0.9951 | 100% |
| PyTorch bf16 autocast (≈ the CUDA app) | — | 0.981 | 0.952 | 35% |

bf16 rounding is amplified by the diffusion sampler with CFG — PyTorch's own bf16 drifts
by the same amount (latent rel-RMS vs fp32 up to ~27% mid-stream for both). So the
spec's "max abs diff ≤ 1e-2 in bf16" over a sampling run isn't achievable by the
reference implementation either; the default here is fp32 for the transformer, which
is closer to the fp32 reference than the CUDA app is, at +21 ms/block.

## Numerical equivalence (`test_equivalence.py`, 89 checks)

- **strict** (MLX fp32 on the MLX CPU stream vs PyTorch fp32): every module, both
  shapes (primer N=50, steady N=12 + KV cache), KV-cache contents after update, full
  `forward()`, and a primer + 6-block sampling loop driven by AvatarForcing's own
  `solve_cfg` / `solve_closed` / `update_kv_cache` — max relative error ≤ 2e-5.
- **bf16** (MLX GPU): each check within 1.25× of PyTorch's own bf16-autocast error.

The strict tests run on the MLX CPU stream because fp32 matmuls on the M5 GPU carry
~1e-3 relative error (vs 1e-6 on CPU).

## Findings along the way

- The checkpoint's `alignment_mask` / `block_bidir_causal_lookahead_mask` buffers are
  the *inverse* of the masks the code generates. The app never loads them
  (`load_ckpt` copies parameters only), so the generated masks are the right ones;
  `load_state_dict(strict=True)` would silently invert attention.
- Checkpoints are fp32 (not bf16 as the spec assumed).
- `mx.fast.scaled_dot_product_attention` accepts boolean masks, and
  `mx.fast.rope(traditional=True, offset=start_pos)` matches the complex-polar RoPE —
  no table, so no 1024-position limit (`server.py` extends the torch table to 65536).
- transformers' warning that wav2vec2 `pos_conv_embed.conv.weight_g/weight_v` were "not
  used" is spurious: torch's weight-norm parametrization remaps them on load.
