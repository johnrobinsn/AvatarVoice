# AvatarVoice

Real-time voice chat with a lip-synced diffusion head-avatar. Browser speaks to it; the avatar speaks and lip-syncs back, over WebRTC.

Wraps [AvatarForcing](https://github.com/TaekyungKi/AvatarForcing) (Ki et al., CVPR 2026) in a voice-chat loop (Silero VAD + Smart Turn + faster-whisper STT + vLLM + Phonon TTS + wav2vec batching), plus the streaming infrastructure needed to keep the model generating indefinitely on a single GPU.

![Preview](avatarforcing/assets/overview.png)

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

## What's engineered on top of AvatarForcing

Upstream AvatarForcing is a research reference implementation that OOMs within a couple of minutes of streaming. This repo adds the engineering to make it stream indefinitely on one GPU:

- **VRAM leaks fixed**: dynamo `start_pos` specialization (via pre-sliced `freqs_cis`) and autograd retention (via `inference_mode`) — together these are the difference between OOMing at ~90 s and holding flat at 2.4 GB forever.
- **RoPE table extended** to 65 536 so long sessions don't run off the end.
- **Periodic KV reanchor** every ~45 s (deferred until TTS queue is empty and the audio buffer has drained) to combat slow self-conditioning drift, with a 1 s pixel-space crossfade on reconnect.
- **CFG collapsed from 3 → 2 branches**: dropped the user-motion branch (zeroed here anyway), 33 % flow-transformer throughput win.
- **KV update folded into the final denoise step**: saves one forward pass per block.
- **Pipelined decode** on a side CUDA stream — motion autoencoder of block N overlaps with denoise of block N+1.
- **Wall-clock-paced worker** + adaptive playback delay keyed on the worker's EWMA block time.
- **Pink-noise idle features**: zero audio features are out-of-distribution for wav2vec2 and cause phantom silent-mouthing. 5 s of shaped pink noise at -50 dBFS, encoded once at startup, cycled during silence.
- **Soft barge-in**: three-layer mic protection (RMS gate, Silero threshold, sustained-speech hysteresis) plus cancel-future-but-finish-in-flight semantics.

The three model-side edits (RoPE pre-slice, CFG collapse, KV-update fold) live in a fork of AvatarForcing at [johnrobinsn/AvatarForcing](https://github.com/johnrobinsn/AvatarForcing), referenced here as a git submodule.

## Layout

```
AvatarVoice/
├── README.md                this file
├── voicechat/               WebRTC server, VAD, STT, LLM, TTS, worker, rings, browser client
├── webrtc/                  earlier single-track streaming demo (kept for reference)
├── scripts/                 one-off utilities
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

```bash
vllm serve cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit --port 8000
```

### 4. Start AvatarVoice

```bash
CUDA_VISIBLE_DEVICES=0,1 python voicechat/server.py --port 8082
```

Open `http://<host>:8082` in a browser, pick an avatar, click **Start session**, and talk.

Hardware split in the current code:
- `cuda:0` — STT (faster-whisper distil-large-v3, `int8_float16`)
- `cuda:1` — AvatarForcing flow transformer + motion autoencoder + wav2vec audio encoder

vLLM runs in a separate process on any GPU.

## Credits

Base model: [Avatar Forcing](https://taekyungki.github.io/AvatarForcing/) (Ki, Jang, Jo, Yoon, Hwang, CVPR 2026).

Voice pipeline shape: [macvoice](https://github.com/johnrobinsn/macvoice) — the Apple-Silicon sibling this fork's voice half is based on.

Released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — same as upstream AvatarForcing. Non-commercial use only.
