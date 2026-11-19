# AvatarVoice

Real-time voice chat with a lip-synced diffusion head-avatar. Browser speaks to it; the avatar speaks and lip-syncs back, over WebRTC.

**Works well on a single NVIDIA Blackwell GPU with 32 GB+ VRAM** (hosts the full avatar pipeline + a 4B LLM alongside). Default configuration is single-GPU; a `--dual-gpu` opt-in splits the avatar onto a second card for extra headroom.

Wraps [AvatarForcing](https://github.com/TaekyungKi/AvatarForcing) (Ki et al., CVPR 2026) in a voice-chat loop (Silero VAD + Smart Turn + faster-whisper STT + vLLM + Phonon TTS + wav2vec batching), plus the streaming infrastructure needed to keep the model generating indefinitely on a single GPU.

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

## Numbers

On a single NVIDIA RTX PRO 6000 Blackwell Max-Q (98 GB), single-GPU mode:

| Metric | Value |
|---|---|
| Steady-state EWMA / block | ~210 ms (400 ms realtime budget) |
| Realtime factor | 0.5× (comfortably below 1.0×) |
| Avatar working-set VRAM | ~5 GB resident, ~25 GB warmup peak |
| vLLM (Qwen3-4B-AWQ) VRAM | ~13 GB at `--gpu-memory-utilization 0.15` |
| Combined VRAM (steady state) | ~18 GB |

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

## Layout

```
AvatarVoice/
├── README.md                this file
├── SPEC_MLX_PORT.md         design doc for a MLX / Apple Silicon port (WIP)
├── voicechat/               WebRTC server, VAD, STT, LLM, TTS, worker, rings, browser client
├── webrtc/                  earlier single-track streaming demo (kept for reference)
├── scripts/                 one-off utilities
├── mlx_port/                starter files for the Mac / MLX port (scaffold + weight converter + tests)
├── data/                    avatar portrait images (rumi, kelly, ken)
└── avatarforcing/           submodule → johnrobinsn/AvatarForcing (fork of TaekyungKi's)
```

## Running

### 1. Clone with submodule

```bash
git clone --recurse-submodules https://github.com/johnrobinsn/AvatarVoice.git
cd AvatarVoice
```

### 2. Environment + weights

Follow upstream setup to install torch and download the AvatarForcing weights:

```bash
cd avatarforcing
bash environment.sh              # creates conda env + installs requirements
bash download_weights.sh         # downloads pretrained_dir/
cd ..
```

Install the extra voice-chat dependencies into the same env:

```bash
pip install -r requirements-voicechat.txt
```

### 3. Start a local LLM server

For a single-GPU setup where vLLM shares the card with the avatar, keep vLLM's KV reservation modest:

```bash
vllm serve cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.15
```

If vLLM gets its own GPU, you can drop the `--gpu-memory-utilization` flag.

### 4. Start AvatarVoice

**Single-GPU (default — avatar + aux share cuda:0):**
```bash
python voicechat/server.py --port 8082
```

**Dual-GPU (opt-in — avatar on cuda:1, aux on cuda:0):**
```bash
python voicechat/server.py --port 8082 --dual-gpu
```

Open `http://<host>:8082` in a browser, pick an avatar, click **Start session**, and talk.

### Running on Apple Silicon (MLX)

On macOS the server defaults to `AF_BACKEND=mlx`: the flow transformer runs in MLX, the
motion decoder and wav2vec2 in torch MPS, STT is Parakeet on MLX (as in macvoice), TTS is
Phonon (Metal), and the LLM is `mlx_lm.server`. On an M5 MacBook Pro (32 GB) the avatar
runs at ~200 ms per 400 ms block (~2× realtime) and replies start ~2.3–2.8 s after you stop
talking. Details and numbers: [`mlx_port/MLX_PERF.md`](mlx_port/MLX_PERF.md).

```bash
# environment (Python 3.12) + weights — see mlx_port/README.md for the weight download
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r mlx_port/requirements-mac.txt
uv pip install --python .venv/bin/python --no-deps transformers==4.30.2
.venv/bin/python mlx_port/convert_weights.py \
    --src avatarforcing/pretrained_dir/flow_transformer.pth \
    --dst avatarforcing/pretrained_dir/flow_transformer_mlx.safetensors

# LLM, in its own venv (mlx-lm needs a newer transformers)
uv venv --python 3.12 .venv-llm && uv pip install --python .venv-llm/bin/python mlx-lm
.venv-llm/bin/mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8000 --prompt-cache-size 1

# server
.venv/bin/python voicechat/server.py --port 8082
```

Headless checks (no browser): `voicechat/tests/test_closed_loop_mlx.py` (in-process, full
chain → `results/closed_loop_mlx.mp4`) and `voicechat/tests/test_webrtc_client.py` (talks
to a running server over WebRTC → `results/webrtc_client.mp4`).

### Useful env vars

| Var | Default | Notes |
|---|---|---|
| `AF_COMPILE_MODE` | `reduce-overhead` | torch.compile mode for the flow transformer. Set `default` to drop CUDA graphs. |
| `AF_MAE_COMPILE_MODE` | `default` | torch.compile mode for the MAE decoder. Set `reduce-overhead` for +~5% speed at the cost of ~6 GB persistent VRAM. |
| `AF_SINGLE_GPU` | (unset) | `1` forces single-GPU; `0` forces dual-GPU. Overrides CLI flag if set. |
| `AF_PORT` | `8082` | Server port. |
| `AF_LLM_MODEL` | `cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit` (mlx: `mlx-community/Qwen3-4B-Instruct-2507-4bit`) | LLM model served by vLLM / mlx_lm.server. |
| `AF_BACKEND` | `mlx` on macOS, else `torch` | Inference backend. |
| `AF_LLM_BASE_URL` | `http://localhost:8000/v1` | (mlx) OpenAI-compatible LLM endpoint. |
| `AF_MLX_FLOW_DTYPE` | `float32` | (mlx) flow-transformer precision; `bfloat16` is ~25 ms/block faster but drifts further from the fp32 reference. |

### Health / debug endpoints

- `GET /memory` — per-GPU torch allocator stats (`allocated`, `reserved`, peaks). `?full=1` for the full category breakdown.

## Credits

Base model: [Avatar Forcing](https://taekyungki.github.io/AvatarForcing/) (Ki, Jang, Jo, Yoon, Hwang, CVPR 2026).

Voice pipeline shape: [macvoice](https://github.com/johnrobinsn/macvoice) — the Apple-Silicon sibling this fork's voice half is based on.

Released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — same as upstream AvatarForcing. Non-commercial use only.
