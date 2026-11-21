# AvatarVoice

Real-time voice chat with a lip-synced diffusion head-avatar. Browser speaks to it; the avatar speaks and lip-syncs back, over WebRTC.

[![Watch the video](https://img.youtube.com/vi/qaVM5GoHLPA/maxresdefault.jpg)](https://www.youtube.com/watch?v=qaVM5GoHLPA)

Runs on two platforms from the same code:

- **Linux + NVIDIA** — works well on a single NVIDIA Blackwell GPU with 32 GB+ VRAM (hosts the full avatar pipeline + a 4B LLM alongside). Default configuration is single-GPU; a `--dual-gpu` opt-in splits the avatar onto a second card for extra headroom.
- **macOS + Apple Silicon** — runs on a 32 GB M-series MacBook (tested on an M5): the AvatarForcing flow transformer in [MLX](https://github.com/ml-explore/mlx), everything else on the Mac GPU, ~2× realtime.

Wraps [AvatarForcing](https://github.com/TaekyungKi/AvatarForcing) (Ki et al., CVPR 2026) in a voice-chat loop (Silero VAD + Smart Turn + Parakeet STT + LLM + Phonon TTS + wav2vec batching), plus the streaming infrastructure needed to keep the model generating indefinitely on a single GPU.

[Blog Article: https://www.storminthecastle.com/posts/avatar-voice/](https://www.storminthecastle.com/posts/avatar-voice/)

## Architecture

```
mic ──► VAD + Smart Turn ──► STT ──► LLM ──► phrase-level TTS
                                                    │
                                   PCM + wav2vec audio features
                                                    │
                                        ┌───────────┴───────────┐
                                        ▼                       ▼
                                 pcm_ring (16 k)        feat_ring (25 fps)
                                        │                       │
                                        ▼                       ▼
                        AvatarVideoTrack ◄──── avatar worker ◄──┘
                        AvatarAudioTrack ◄──── pcm_ring read
                                        │
                                        ▼
                                WebRTC ──► browser
```

Time-indexed ring buffers are the only shared state between stages. Audio is the master clock; video adapts.

The pipeline, rings, VAD/turn detection, TTS and WebRTC are shared; the backend (`AF_BACKEND`, chosen automatically) picks the rest:

| | Linux + NVIDIA (`torch`) | macOS + Apple Silicon (`mlx`) |
|---|---|---|
| Avatar (AvatarForcing) | PyTorch CUDA, `torch.compile` + CUDA graphs | flow transformer in MLX ([`mlx_port/`](mlx_port/)); motion decoder + wav2vec2 on PyTorch MPS |
| STT | Parakeet TDT 0.6B v2 via onnx-asr (`AF_STT=whisper`: faster-whisper) | Parakeet TDT 0.6B v2 on MLX (as in [macvoice](https://github.com/johnrobinsn/macvoice)) |
| LLM | vLLM | `mlx_lm.server` (any OpenAI-compatible server works for either) |
| TTS · VAD · turn detection | Phonon · Silero · Smart Turn | same |

## Numbers

| Metric | NVIDIA RTX PRO 6000 Blackwell Max-Q (98 GB), single-GPU | MacBook Pro M5, 32 GB |
|---|---|---|
| Steady-state EWMA / block (400 ms realtime budget) | ~210 ms | ~200–235 ms |
| Realtime factor (lower is better; < 1.0× keeps up) | 0.5× | 0.5–0.6× |
| Avatar memory | ~5 GB VRAM resident, ~25 GB warmup peak | ~5.5 GB unified (0.8 GB MLX + 4.7 GB MPS) |
| LLM (Qwen3-4B, 4-bit) | ~13 GB VRAM at `--gpu-memory-utilization 0.15` | ~2.4 GB |
| Combined (steady state) | ~18 GB VRAM | ~11.5 GB process footprint |
| Reply starts after you stop talking | — | ~2.3–2.8 s |

## What's engineered on top of AvatarForcing

Upstream AvatarForcing is a research reference implementation that OOMs within a couple of minutes of streaming. This repo adds the engineering to make it stream indefinitely and realtime on one GPU:

- **VRAM leaks fixed**: dynamo `start_pos` specialization (via pre-sliced `freqs_cis`) and autograd retention (via `inference_mode`) — together these are the difference between OOMing at ~90 s and holding flat at ~5 GB resident.
- **RoPE table extended** to 65 536 so long sessions don't run off the end.
- **`torch.compile(mode="reduce-overhead")` on the flow transformer** — CUDA graphs on the 20× / block read-only path cut EWMA from ~437 ms → ~210 ms (-52% live, -23% offline). The `update_kv_cache=True` write path (1× / block) falls back to inductor-only; this is expected and fine. Controlled by `AF_COMPILE_MODE` env var (default `reduce-overhead`).
- **MAE decoder compile mode = `default`** independently of the flow transformer — the CUDA graph pool the MAE would reserve for pixel outputs is large (~6 GB persistent) for only 1× / block savings; the `default` mode's inductor fusion is the better tradeoff. Controlled by `AF_MAE_COMPILE_MODE` (default `default`).
- **Periodic KV reanchor** every ~45 s (deferred until TTS queue is empty and the audio buffer has drained), with a 25-frame pixel-space crossfade on reconnect. Hard cap at 65 s even mid-speech.
- **CFG collapsed from 3 → 2 branches**: dropped the user-motion branch (zeroed here anyway), 33 % flow-transformer throughput win.
- **KV update folded into the final denoise step**: saves one forward pass per block.
- **Pipelined decode** on a side CUDA stream — motion autoencoder of block N overlaps with denoise of block N+1.
- **Wall-clock-paced worker** + adaptive playback delay keyed on the worker's EWMA block time.
- **Pink-noise idle features**: zero audio features are out-of-distribution for wav2vec2 and cause phantom silent-mouthing. 5 s of shaped pink noise at -50 dBFS, encoded once at startup, cycled during silence.
- **Soft barge-in**: three-layer mic protection (RMS gate, Silero threshold, sustained-speech hysteresis) plus cancel-future-but-finish-in-flight semantics.
- **Wav2vec on a dedicated CUDA side-stream thread** — overlaps audio-feature extraction with the flow transformer so start-of-phrase doesn't stutter.
- **Fresh logs on every start**: `logs/server.log` is truncated by the server process regardless of how it's launched.

The model-side edits (RoPE pre-slice, CFG collapse, KV-update fold, torch.compile mode wiring) live in a fork of AvatarForcing at [johnrobinsn/AvatarForcing](https://github.com/johnrobinsn/AvatarForcing), referenced here as a git submodule.

On Apple Silicon (details in [`mlx_port/MLX_PERF.md`](mlx_port/MLX_PERF.md)):

- **Flow transformer + sampler ported to MLX** — ~40–60 ms per block, verified against the PyTorch code (sampling loop within ~1e-5 in float32; per-frame SSIM 0.9975 vs the fp32 reference).
- **Motion decoder 613 → 150 ms per block on MPS** — exact rewrites of its modulated convolutions, `upfirdn2d` and the warp grid, pure fp16, and `torch.compile` (MLX convolutions turned out slower at 512×512, so it stays on PyTorch).
- **float32 flow transformer by default** — bf16 is faster but the diffusion sampler amplifies its rounding (as it does for PyTorch's bf16 on CUDA).
- No float64 on Apple GPUs: `solve_closed`'s coefficients are precomputed once in float64 on the CPU.

## Layout

```
AvatarVoice/
├── README.md                this file
├── install.sh               one-line installer (macOS + Linux)
├── pyproject.toml, uv.lock  Python environment for both platforms (uv sync)
├── SPEC_MLX_PORT.md         original design doc for the MLX / Apple Silicon port
├── voicechat/               WebRTC server, VAD, STT, LLM, TTS, workers (avatar_worker.py: CUDA,
│                            avatar_worker_mlx.py: Apple Silicon), rings, browser client,
│                            launcher.py (the `avatarvoice` command), tests
├── webrtc/                  earlier single-track streaming demo (kept for reference)
├── scripts/                 download_models.py (used by `avatarvoice download`) and utilities
├── mlx_port/                MLX port of the flow transformer + sampler, fast MPS decoder kernels,
│                            weight converter, equivalence tests, benchmarks (MLX_PERF.md)
├── data/                    avatar portraits (rumi, vicky, ken); sources in data/CREDITS.md
└── avatarforcing/           submodule → johnrobinsn/AvatarForcing (fork of TaekyungKi's)
```

## Quick install

macOS (Apple Silicon) or Linux (NVIDIA GPU):

```bash
curl -LsSf https://raw.githubusercontent.com/johnrobinsn/AvatarVoice/main/install.sh | bash
avatarvoice run          # starts the LLM + voice servers and opens your browser; Ctrl-C stops
```

The installer checks your machine, sets up private Python environments with [uv](https://docs.astral.sh/uv/)
in `~/.avatarvoice` (nothing system-wide), downloads the models (~4–6 GB) and adds an `avatarvoice`
command (`run`, `status`, `stop`, `logs`, `download`, `update`, `uninstall`). It shows progress as it goes and
keeps a log in `~/.avatarvoice/install.log`; re-running it resumes or updates. On a Mac with a fast
connection it takes a few minutes. Options: `AVATARVOICE_HOME` (install location), `AVATARVOICE_REF`
(branch or tag).

Prefer to read it first? `curl -LsSf https://raw.githubusercontent.com/johnrobinsn/AvatarVoice/main/install.sh -o install.sh`,
look it over, then `bash install.sh`. It runs the same steps as the manual install below.

## Manual install

The same steps the installer runs. Needs [uv](https://docs.astral.sh/uv/) and git.

### 1. Clone with submodule

```bash
git clone --recurse-submodules https://github.com/johnrobinsn/AvatarVoice.git
cd AvatarVoice
```

### 2. Python environment

```bash
uv sync
```

`pyproject.toml` / `uv.lock` define both platforms: on **macOS** (Python 3.12) the MLX avatar stack
and Parakeet STT; on **Linux** (Python 3.11) PyTorch for CUDA 12.8 (needed for Blackwell GPUs) and
Parakeet STT via onnx-asr / onnxruntime-gpu (faster-whisper as a fallback). The venv is `.venv`.

### 3. Models

```bash
uv run avatarvoice download
```

AvatarForcing checkpoints (the authors' Google Drive, as in `avatarforcing/download_weights.sh`),
wav2vec2, the STT model and the LLM weights; on macOS also the MLX conversion of the flow
transformer. Skips anything already present; `--check` lists what's missing.

### 4. LLM server environment

The LLM server gets its own venv (it needs newer packages than the avatar). `--no-config` keeps
the project's `pyproject.toml` settings out of it:

```bash
# Linux + NVIDIA
uv venv --no-config --python 3.12 .venv-llm && uv pip install --no-config --python .venv-llm/bin/python vllm
# macOS + Apple Silicon
uv venv --no-config --python 3.12 .venv-llm && uv pip install --no-config --python .venv-llm/bin/python mlx-lm
```

### 5. Run

```bash
uv run avatarvoice run          # starts the LLM + voice servers, opens the browser; Ctrl-C stops both
```

Open `http://<host>:8082` in a browser, pick an avatar, click **Start session**, and talk. On a laptop, use headphones: the avatar's voice from the speakers can otherwise be heard as you talking over it.

### Running the servers yourself

`avatarvoice run` wraps these; run them directly for custom setups.

<details><summary><b>LLM server</b></summary>

Linux + NVIDIA — for a single-GPU setup where vLLM shares the card with the avatar, keep vLLM's KV reservation modest:

```bash
.venv-llm/bin/vllm serve cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.15
```

If vLLM gets its own GPU, you can drop the `--gpu-memory-utilization` flag.

macOS + Apple Silicon:

```bash
.venv-llm/bin/mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
    --port 8000 --prompt-cache-size 1
```

Any OpenAI-compatible server works; point AvatarVoice at it with `AF_LLM_MODEL` / `AF_LLM_BASE_URL`.
</details>

<details><summary><b>AvatarVoice server</b></summary>

**Linux, single-GPU (default — avatar + aux share cuda:0):**
```bash
uv run python voicechat/server.py --port 8082
```

**Linux, dual-GPU (opt-in — avatar on cuda:1, aux on cuda:0):**
```bash
uv run python voicechat/server.py --port 8082 --dual-gpu
```

**macOS** (the backend is picked automatically; `AF_BACKEND=mlx`):
```bash
uv run python voicechat/server.py --port 8082
```
</details>

### Testing without a browser

- `voicechat/tests/test_session_lifecycle.py` — session-lifecycle regression tests (no GPU or models needed).
- `voicechat/tests/test_webrtc_client.py --url http://localhost:8082` — talks to a running server over WebRTC like the browser does (spoken question in, avatar video/audio out → `results/webrtc_client.mp4`); `--sessions N` checks N back-to-back sessions all get a reply. Works with either backend.
- `voicechat/tests/test_closed_loop_mlx.py` — (macOS) the whole chain in-process, no server → `results/closed_loop_mlx.mp4`.

### Useful env vars

| Var | Default | Backend | Notes |
|---|---|---|---|
| `AF_BACKEND` | `mlx` on macOS, else `torch` | both | Inference backend. |
| `AF_PORT` | `8082` | both | Server port. |
| `AF_LLM_MODEL` | `cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit` / `mlx-community/Qwen3-4B-Instruct-2507-4bit` | both | LLM model name sent to the OpenAI-compatible server. |
| `AF_STT` | `parakeet` | torch | STT backend on Linux. `parakeet` = Parakeet-TDT-0.6b-v2 via onnx-asr (onnxruntime-gpu); `whisper` = faster-whisper distil-large-v3. (macOS always uses Parakeet on MLX.) |
| `AF_AOTI` | `1` | torch | **Experimental.** `1` = load prebuilt AOTInductor artifacts from `compiled_models/` when present (see `scripts/compile_aot.py` and `scripts/compile_mae_aot.py`). Startup drops from ~82 s to ~42 s, but current builds produce visibly degraded avatar frames — set `AF_AOTI=0` if you built a `.pt2` and see rendering issues. Default behavior (no `.pt2` in `compiled_models/`) is unaffected. |
| `AF_COMPILE_MODE` | `reduce-overhead` | torch | torch.compile mode for the flow transformer. Set `default` to drop CUDA graphs. |
| `AF_MAE_COMPILE_MODE` | `default` | torch | torch.compile mode for the MAE decoder. Set `reduce-overhead` for +~5% speed at the cost of ~6 GB persistent VRAM. |
| `AF_SINGLE_GPU` | (unset) | torch | `1` forces single-GPU; `0` forces dual-GPU. Overrides CLI flag if set. |
| `AF_LLM_BASE_URL` | `http://localhost:8000/v1` | mlx | OpenAI-compatible LLM endpoint. |
| `AF_MLX_FLOW_DTYPE` | `float32` | mlx | Flow-transformer precision; `bfloat16` is ~25 ms/block faster but drifts further from the fp32 reference. |

### Health / debug endpoints

- `GET /memory` — torch: per-GPU allocator stats (`allocated`, `reserved`, peaks; `?full=1` for the full category breakdown). mlx: MLX active/peak/cache memory and MPS allocator memory.

## Credits

Base model: [Avatar Forcing](https://taekyungki.github.io/AvatarForcing/) (Ki, Jang, Jo, Yoon, Hwang, CVPR 2026).

Voice pipeline shape: [macvoice](https://github.com/johnrobinsn/macvoice) — the Apple-Silicon sibling this fork's voice half is based on.

Apple Silicon: [MLX](https://github.com/ml-explore/mlx) (Apple), [parakeet-mlx](https://github.com/senstella/parakeet-mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm). The MLX port is also published standalone as [avatarforcing-mlx](https://github.com/johnrobinsn/avatarforcing-mlx).

Released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — same as upstream AvatarForcing. Non-commercial use only.
