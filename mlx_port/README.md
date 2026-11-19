# MLX port scaffold

Starter files for the Mac-side MLX port. See `../SPEC_MLX_PORT.md` for the full plan.

## Layout

| File | Purpose | Status |
|---|---|---|
| `helpers_mlx.py` | Rotary embeddings, masks, modulation — small, reusable | Signatures + TODO |
| `flow_transformer_mlx.py` | All MLX modules (Attention, CrossAttention, DFoTBlock, FlowTransformer, ...) | Module tree + TODO |
| `convert_weights.py` | One-shot PyTorch `.pth` → MLX `.safetensors` converter | Mostly complete |
| `test_equivalence.py` | Compare MLX output to PyTorch reference, per-module and end-to-end | Test harness skeleton |
| `bench.py` | Measure per-block timing on Mac | Timing skeleton |

## How to use

1. Run `python convert_weights.py --src ../avatarforcing/pretrained_dir/flow_transformer.pth --dst ../avatarforcing/pretrained_dir/flow_transformer.safetensors` — produces the MLX weight file.
2. Implement the TODOs in `helpers_mlx.py`, then `flow_transformer_mlx.py`.
3. Run `python test_equivalence.py` to verify each module matches PyTorch (start with helpers, move up).
4. Once end-to-end equivalence passes, run `python bench.py` to measure perf.
5. When ready, move `flow_transformer_mlx.py` + `helpers_mlx.py` into `avatarforcing/mlx/` in the submodule.

## Expected target

< 400 ms EWMA per block on M5 Pro, 32 GB.

## Dependencies

```bash
pip install mlx torch safetensors numpy
# Plus the AvatarVoice stack (torch MPS for MAE + wav2vec during hybrid phase)
```
