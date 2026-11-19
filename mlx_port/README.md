# MLX port — AvatarForcing on Apple Silicon

Hybrid pipeline: the flow transformer runs in **MLX**; wav2vec2, the motion encoder and
the motion decoder run in **torch MPS** (the decoder with exact fast-path kernel rewrites
and `torch.compile`). See `../SPEC_MLX_PORT.md` for the plan and
[`MLX_PERF.md`](MLX_PERF.md) for measured results.

**Status:** port complete and verified. Full pipeline at **210 ms EWMA per 10-frame
block (1.9× realtime)** on a base M5, per-frame SSIM 0.9975 vs the fp32 PyTorch
reference. Wired into the voice-chat demo via `voicechat/avatar_worker_mlx.py` (`AF_BACKEND=mlx`,
the default on macOS) — see the top-level README, "Running on Apple Silicon".

## Layout

| File | Purpose |
|---|---|
| `helpers_mlx.py` | RoPE helpers, attention masks, `modulate` |
| `flow_transformer_mlx.py` | MLX `FlowTransformerMLX` — mirrors the PyTorch module tree and state_dict keys 1:1 |
| `sampler_mlx.py` | `FlowSamplerMLX` — diffusion-forcing loop (primer + streaming blocks, CFG, `solve_closed`, KV-cache update) |
| `decoder_fast_torch.py` | Exact, faster torch-MPS kernels for the motion decoder (`patch()` / `unpatch()`) |
| `pipeline_mlx.py` | `HybridAvatarPipeline` — portrait + audio → streamed RGB frames |
| `convert_weights.py` | `flow_transformer.pth` → `flow_transformer_mlx.safetensors` |
| `test_equivalence.py` | 89 checks vs the PyTorch reference (modules → full sampling loop) |
| `bench.py` | Flow-transformer per-block timing |
| `bench_decoder.py` | Motion-decoder timing + exactness of the fast kernels |
| `run_stream.py` | End-to-end run on real audio → mp4, timing, optional SSIM vs reference |
| `requirements-mac.txt` | Tested Mac environment |

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r mlx_port/requirements-mac.txt
uv pip install --python .venv/bin/python --no-deps transformers==4.30.2

# weights (avatarforcing/download_weights.sh, using the venv)
cd avatarforcing
../.venv/bin/gdown 1a0ZqX9GMrkb3JFQgOBmUB0V-Bpwtv4Hp -O pretrained_dir/
../.venv/bin/gdown 1lchvHiDka0bBQ29bdY6lHxCyNGbGzvS_ -O pretrained_dir/
../.venv/bin/hf download facebook/wav2vec2-base-960h --local-dir pretrained_dir/wav2vec2-base-960h
cd ..

.venv/bin/python mlx_port/convert_weights.py \
    --src avatarforcing/pretrained_dir/flow_transformer.pth \
    --dst avatarforcing/pretrained_dir/flow_transformer_mlx.safetensors
```

## Use

```bash
cd mlx_port
../.venv/bin/python test_equivalence.py            # all checks; or: helpers|modules|forward|loop
../.venv/bin/python bench.py [--dtype bfloat16]
../.venv/bin/python bench_decoder.py
../.venv/bin/python run_stream.py [--compare] [--seconds 6] [--audio x.wav] [--image ../data/ken.jpg]
```

```python
import pipeline_mlx as P
pipe = P.HybridAvatarPipeline()                       # flow fp32 (MLX), decoder fp16 (MPS)
pipe.set_reference(P.load_image("data/rumi.jpg", "mps"))
wa = pipe.audio_features(pcm_16k)                     # [1, T, 512]
for block_idx, pixels, timing in pipe.stream(wa):     # pixels: [k, 3, 512, 512] in [-1, 1]
    frames = P.to_uint8(pixels)
```

## API differences from the PyTorch model

- `forward_with_precomputed(..., start_pos, kv_cache_list, update_kv_cache)` takes the
  integer `start_pos` instead of a pre-sliced `freqs_cis_slice` (RoPE via
  `mx.fast.rope`, no table / no max length).
- KV cache keeps PyTorch's structure (`[(self_dict, cross_dict)] * depth`, dicts of
  `"k"`/`"v"`); updates rebind dict entries instead of writing in place.
  `init_kv_cache(batch)` takes the CFG-expanded batch (2·B).
- Conditioning modules (`t_embedder`, `c_embedder`, adaLN projections) stay fp32 under
  `set_dtype` — they run once per block on a few dozen tokens.
- Inference only: no condition dropout (`train=False` paths).

Torch-MPS caveats handled in `pipeline_mlx.py`: no float64 on MPS (the sampler computes
`solve_closed` coefficients in numpy float64 once); `torch.linalg.qr` (motion
`Direction`) has no fp16 MPS kernel, so only the decoder's synthesis path runs fp16.
