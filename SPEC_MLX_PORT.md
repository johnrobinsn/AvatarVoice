# MLX port — AvatarForcing on Apple Silicon

## Goal

Port the hot path of AvatarForcing (the flow transformer) to MLX so that a full real-time video-avatar pipeline runs at **< 400 ms EWMA per block** (i.e. 1.0× realtime at 25 fps, 10 frames per block) on a **MacBook Pro M5 Pro, 32 GB**.

## Success criteria

| Metric | Target |
|---|---|
| Per-block EWMA (steady state, active chat) | **< 400 ms** |
| Per-block max under reanchor | < 600 ms |
| VRAM / unified memory (steady state) | < 16 GB total (AvatarVoice + LLM + macvoice) |
| Lip-sync quality (subjective) | Indistinguishable from CUDA reference |
| Numerical equivalence (flow transformer outputs) | Max abs diff vs PyTorch ≤ 1e-2 (bf16), per-frame SSIM ≥ 0.995 |

## Non-goals

- Dropping NFE, CFG, frame rate, or resolution for perf (user has ruled these out).
- Porting MAE decoder or wav2vec2 in Phase 1 — those stay in `torch` on MPS (or move to CoreML later if needed).
- Rewriting STT/TTS — those come from `macvoice`.
- Supporting non-Apple hardware with MLX code; CUDA path remains on torch.

## Target hardware

- MacBook Pro M5 Pro, 32 GB unified memory
- macOS latest (for latest MPS/MLX perf)
- No external accelerator

## Architecture — hybrid (recommended)

```
┌─────────────────────────────────────────────────────────┐
│  avatar_worker.py (Python glue)                         │
│                                                          │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│   │  wav2vec2    │ →  │   MLX flow   │ →  │  MAE dec │  │
│   │  (torch MPS) │    │ transformer  │    │(torch MPS)│  │
│   └──────────────┘    │  (new port)  │    └──────────┘  │
│                       └──────────────┘                   │
│                                                          │
│   LLM:  mlx_lm.Qwen3-4B-4bit  (in-process)              │
│   STT:  mlx-whisper or faster-whisper (CoreML backend)  │
│   TTS:  Kokoro-MLX or similar Mac-compatible            │
└─────────────────────────────────────────────────────────┘
```

Why hybrid:
- The flow transformer is ~1600 kernel launches/block — the launch-overhead king. Biggest win from MLX's lazy eval + fusion.
- The MAE decoder is 1 call/block over a large conv-ish block. Launch overhead is proportionally smaller. torch MPS or CoreML on ANE should be acceptable. Port later if measured bottleneck.
- Wav2vec2 is small and runs 1×/block; torch MPS is fine.
- Boundary crossings (torch tensor ↔ MLX array) happen only 2×/block — cheap because unified memory means no device copy, only a `.numpy() → mx.array()` wrap.

## Source material

### Repos
- **AvatarVoice (parent)**: https://github.com/johnrobinsn/AvatarVoice
- **AvatarForcing fork (submodule)**: https://github.com/johnrobinsn/AvatarForcing — branch `perf/reduce-overhead-compile` has latest compile-mode work
- **Pretrained weights**: `avatarforcing/pretrained_dir/flow_transformer.pth`, `motion_autoencoder.pth`, `wav2vec2-base-960h/`

### Key files to read
| File | Why |
|---|---|
| `avatarforcing/models/avatarforcing/flow_transformer.py` | The module hierarchy to port (classes below) |
| `avatarforcing/models/avatarforcing/AvatarForcing.py` | The inference loop (CFG, diffusion sampling, KV cache orchestration) |
| `avatarforcing/configs/inference.yaml` | Hyperparameters (dim_h=1024, num_heads=8, depth=8, num_frames_for_clip=50, num_frames_per_block=10, attention_window=2) |
| `voicechat/avatar_worker.py` | How the worker currently uses the model; where the MLX path will hook in |

### Modules to port (in `flow_transformer.py`)
| Class | Lines | Notes |
|---|---|---|
| `TimestepEmbedder` | 627 | Sinusoidal + 2-layer MLP. Trivial. |
| `SequenceEmbed` | (search) | `nn.Linear` wrapper. Trivial. |
| `AudioMotionAttention` (`c_embedder`) | 97 | Uses `nn.MultiheadAttention` × 2. Port to MLX attention primitives. |
| `CrossAttention` | 138 | Has rotary + windowed KV cache. |
| `Attention` | 228 | Self-attention with rotary + block-bidir causal mask + 38-slot rolling KV cache. **Hot path**. |
| `DFoTBlock` | 320 | Attention + CrossAttention + MLP + adaLN. 8 of these. |
| `FlowTransformer` | 378 | Top-level — needs both `forward_with_precomputed` and the KV-cache update path. |
| `FinalLayer` | (search) | AdaLN + linear projection back to motion-latent dim. |

### Helpers to port
- `precompute_freqs_cis(dim, end, theta=10000)` — polar form, `torch.polar(...)` returns complex64. MLX has `mx.complex64` dtype.
- `apply_rotary_emb_q` / `apply_rotary_emb_k` — complex multiply, used on Q and K before SDPA.
- `make_block_bidir_causal_lookahead_mask_window(start_pos, cache_len, n_tokens, local_len, lookahead=2)` — produces a dense bool mask. Pure arithmetic, trivial.
- `enc_dec_mask_window(start_pos, cache_len, n_tokens, expansion)` — likewise.

## Phased plan

### Phase 0 — Mac env setup (half day)

- `pip install mlx mlx-lm` into a fresh venv on the Mac
- `pip install torch` (for MPS baseline + the hybrid MAE/wav2vec path)
- `pip install transformers librosa face_alignment aiortc aiohttp` (existing deps)
- Clone AvatarVoice + submodule, confirm it imports and finds weights
- Run the existing CUDA inference eagerly against `device="mps"` for baseline timing. Expected: 600-800 ms EWMA. **Record this number** — it's the "no change" floor.

### Phase 1 — Weight conversion (1 day)

Write `scripts/convert_flow_transformer_to_mlx.py`:
```python
import torch, mlx.core as mx
from safetensors.numpy import save_file

sd = torch.load("pretrained_dir/flow_transformer.pth", map_location="cpu", weights_only=True)
# Rename keys as needed to match the MLX FlowTransformer module tree
out = {k: v.numpy() for k, v in sd.items()}
save_file(out, "pretrained_dir/flow_transformer.safetensors")
```

Deliverable: a `.safetensors` file that loads cleanly into the MLX module tree (Phase 2).

Also: diff the PyTorch state_dict keys against the planned MLX module names and record the rename map. Most keys should map 1:1 (`blocks.0.attn.qkv.weight` → `blocks.0.attn.qkv.weight`), but MLX uses `nn.Module`'s dot-attr convention — verify.

### Phase 2 — Port modules (2 days)

Create `avatarforcing/mlx/flow_transformer.py`. Mirror the PyTorch module tree 1:1. Key implementation notes:

**Rotary embeddings**
```python
import mlx.core as mx
# complex64 is supported in MLX
freqs = mx.polar(mx.ones_like(freqs_abs), freqs_abs).astype(mx.complex64)
```
Keep the pre-slicing pattern: slice `freqs_cis[start_pos:start_pos+N]` outside the attention call so no int crosses a compile boundary (there's no compile in MLX per se, but the pattern is still clean).

**SDPA with dense mask**
MLX has `mlx.core.fast.scaled_dot_product_attention` — check mask support. If dense bool masks are NOT accepted (they weren't in MLX 0.x as of this writing), fall back to manual:
```python
scores = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(head_dim))
scores = mx.where(mask, scores, mx.array(-1e9))
attn = mx.softmax(scores, axis=-1)
out = attn @ v
```
MLX's lazy eval will fuse the softmax + matmul chain. For the small shapes we have (q=12 × k=50), this is fine.

**KV cache**
MLX arrays are nominally immutable (functional). For rolling KV cache, use **module state** — store the cache as an attribute on the attention layer, and on update, slice-and-concat:
```python
# instead of `kv_cache["k"][:, :, -10:, :] = k_new[:, :, :-2, :]`
# do: concatenate tail with slice then assign
new_k = mx.concatenate([self._k_cache[:, :, 10:, :], k_new[:, :, :-2, :]], axis=2)
self._k_cache = new_k
mx.eval(self._k_cache)  # force computation so later calls see finalized buffer
```
This is also how you dodge the in-place-mutation issue that broke CUDA graphs in the torch version.

**AdaLN modulation**
`adaLN_modulation(t_embed).chunk(6, dim=-1)` → MLX: `mx.split(..., 6, axis=-1)`.

**Autocast / dtype**
Model weights are bf16 in `.pth`. MLX supports bf16 on M-series. Use bf16 for all weights + activations in the transformer. Output cast to fp32 for the diffusion step state.

### Phase 3 — Integration (0.5-1 day)

- Add an `AF_BACKEND=mlx|torch` env var to `avatar_worker.py` or `inference.py`
- When `AF_BACKEND=mlx`, replace the flow_transformer path with the MLX version; keep MAE + wav2vec on torch MPS
- Boundary tensors: convert via `mx.array(t.cpu().numpy())` on the way in, `torch.from_numpy(np.array(a))` on the way out. Both free (unified memory on M-series, just metadata).
- Preserve all existing API: `compute_wr_embedded`, `compute_condition`, `forward_with_precomputed`, `update_kv_cache` methods must have equivalent MLX entry points with the same signatures.
- The diffusion-forcing sample loop (in `AvatarForcing.py:solve_cfg`, `inference_stream`) stays in Python; only the forward call changes.

### Phase 4 — Verify + measure (0.5-1 day)

**Numerical check** (`tests/test_mlx_equivalence.py`):
- Load same weights into PyTorch and MLX versions
- Seed both, run same input through `forward_with_precomputed`
- Compare outputs — max abs diff ≤ 1e-2 in bf16. If larger: check rotary, mask, SDPA path.

**Perf check**: Run existing `inference.py --stream` style path, measure per-block EWMA, per-block max. Log CPU-side perf (`time.perf_counter()`) plus `mx.eval(output)` synchronization points so you're measuring real GPU time, not lazy-eval posting.

**Compare**:
- torch MPS eager (Phase 0 baseline)
- MLX flow / torch MPS MAE (Phase 3 result)
- Target: < 400 ms EWMA

## Known gotchas

1. **MLX is lazy.** Timing without `mx.eval(output)` just measures how long it takes to *post* work, not do it. Forget this and you'll "measure" 10 ms per block that's actually 500 ms queued.
2. **Dtype propagation.** Mixing fp32 time embeddings with bf16 activations needs explicit `.astype(mx.bfloat16)` at join points. Autocast doesn't exist.
3. **Complex numbers.** `mx.complex64` works but is less tested than fp dtypes. If rotary gives NaNs, consider the real-valued rotary rewrite (two real multiplies instead of one complex multiply) — works everywhere.
4. **Unified memory.** No `.to(device)`. All arrays are device-visible. Great for perf; bad if you accidentally `.numpy()` the KV cache every block (ruins memory locality).
5. **In-place semantics.** `array[idx] = value` style updates don't exist. Either use `mx.concatenate` + reassign, or use `mlx.core.utils.save_safetensors` for persistent state.
6. **`mlx.fast.scaled_dot_product_attention`** may or may not accept our dense mask. Verify, fall back to manual if not.
7. **Weight key names.** PyTorch's `nn.ModuleList` uses `blocks.0.attn.qkv.weight`; MLX uses the same but *case matters*. Be strict.

## Deliverables

- [ ] `scripts/convert_flow_transformer_to_mlx.py` — one-shot weight converter
- [ ] `avatarforcing/mlx/flow_transformer.py` — MLX implementation of the full module tree
- [ ] `avatarforcing/mlx/__init__.py` — exports for drop-in use
- [ ] `tests/test_mlx_equivalence.py` — numerical check vs PyTorch reference
- [ ] `tests/bench_mlx_vs_mps.py` — per-block timing comparison
- [ ] Minimal glue in `voicechat/avatar_worker.py` behind `AF_BACKEND=mlx`
- [ ] Short `MLX_PERF.md` with the measured numbers

## Risks and off-ramps

| Risk | Signal | Mitigation |
|---|---|---|
| MLX attention is slower than hoped for seq_len=12 | Flow transformer alone > 350 ms | Try `mlx.fast.scaled_dot_product_attention` if a MLX release has added mask support; fall back to manual matmul chain |
| MAE decoder dominates (torch MPS slow for the pixel decoder) | Block total > 400 ms with flow transformer at ~250 ms | Port MAE to MLX (Phase 2.5) or convert MAE to CoreML (Phase 2.5-alt) |
| KV cache cost of reassign-and-concat vs in-place mutation | Reanchor spikes > 1 s | Investigate MLX's in-place indexed assignment API (`mlx.core.inplace`?) or pre-allocate a ring buffer and use `mx.slice` |
| Full port exceeds budget | Timing not close after Phase 3 | Fall back to lower-fps compromise rather than continue |

## Rough time budget

| Phase | Days |
|---|---|
| 0: Env + baseline | 0.5 |
| 1: Weight conversion | 1 |
| 2: Module port | 2 |
| 3: Integration | 0.5-1 |
| 4: Measure | 0.5-1 |
| **Total** | **~4-5 days focused work** |

Add ~1 day buffer for whichever gotcha bites. Call ~5-7 days end-to-end.

## Reference: current CUDA perf (what we're trying to match)

From the current Linux + 2× RTX Pro 6000 setup:

| | mode=default | mode=reduce-overhead (current) |
|---|---|---|
| Offline flow transformer median | 175 ms | 135 ms |
| Live EWMA (full pipeline) | 437 ms | ~210 ms |
| Peak VRAM (avatar, after MAE=default) | — | 24.7 GB (warmup only) |

The MLX target is to match the **live EWMA** range on Mac. Perf per-op won't match CUDA + CUDA graphs, but with fewer sources of launch overhead (unified memory, no process boundary, no HTTP) we should be in the same ballpark.
