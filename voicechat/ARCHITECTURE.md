# Voice-Chat Avatar Demo — Architecture

Linux-native port of the MacVoice two-way voice pipeline, driving an AvatarForcing avatar. Browser speaks to it; it speaks and shows a face back.

## 1. Goals / non-goals

**Goals (v1 demo):**
- Two-way voice conversation with an animated avatar, browser-initiated over WebRTC.
- Full-duplex: avatar can be interrupted (barge-in) without a visible or audible restart.
- All models on one RTX PRO 6000 Blackwell, running on Linux.
- End-to-end latency comparable to MacVoice: user stops speaking → audio reply in <800 ms, lip motion by <1.2 s.
- Clean enough to serve as the write-up subject for a blog post.

**Non-goals (v1):**
- Multi-user / multi-session. One browser at a time.
- Avatar hot-swap mid-session.
- TURN / WAN access. LAN only.
- Voice cloning. Use Phonon's default voice.
- Echo cancellation in the browser; rely on headphones.
- Reasoning/tool use in the LLM. Plain chat.

## 2. High-level dataflow

```
┌────────────────────────── BROWSER ──────────────────────────┐
│  mic ──┐                                       ┌─── speaker │
│        │ WebRTC audio in     WebRTC audio out  │            │
│  cam ──┤                                       ├─── <video> │
│        │ WebRTC video in     WebRTC video out  │            │
└────────┼───────────────────────────────────────┼────────────┘
         │                                       │
         │             HTTP signaling            │
         │           (aiohttp /offer)            │
         │                                       │
┌────────▼─────────── LINUX SERVER (aiortc) ─────▼────────────┐
│                                                             │
│   AUDIO IN              ORCHESTRATOR             VIDEO OUT  │
│                      ┌────────────────┐                     │
│  mic stream ──┬───▶  │   TurnState    │  ◀─── avatar frames │
│               │      │   (state       │                     │
│               ▼      │    machine)    │                     │
│         [Silero VAD] │                │                     │
│               │      │  + decides     │                     │
│               ▼      │  what feeds    │                     │
│         [Smart Turn] │  avatar_audio  │                     │
│               │      │                │                     │
│          end-of-turn ├────────────────┤                     │
│               ▼      │                │                     │
│    [Parakeet STT] ───┼──text─┐        │                     │
│                      │       ▼        │                     │
│                      │  [vLLM: Qwen3] │                     │
│                      │       │        │                     │
│                      │       ▼        │                     │
│                      │  [Phonon TTS]──┼──(24k→16k)──┐       │
│                      └────────────────┘             │       │
│                                                     ▼       │
│  cam stream ──[face crop]──▶ user_frame ring ──▶ AVATAR     │
│  mic stream ──────────────▶ user_audio ring ─▶  FORCING     │
│                                                 STREAMING   │
│  [switch]                                           │       │
│   TTS pcm ──or── silence ──▶ avatar_audio ring  ────┤       │
│                              (controlled by state)  │       │
│                                                     ▼       │
│                                              avatar frames  │
│                                              avatar_audio ──┼─▶
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 3. Component inventory

| Component | File | Runs on | Lifecycle |
|---|---|---|---|
| WebRTC server + signaling | `voicechat/server.py` | asyncio event loop | per-session |
| Avatar generation worker | `voicechat/avatar_worker.py` | dedicated thread | process-lifetime |
| LLM client (vLLM over SSE) | `voicechat/llm.py` | asyncio task (per-reply) | per-reply |
| TTS worker (Phonon) | `voicechat/tts.py` | dedicated thread | process-lifetime |
| STT worker (Parakeet) | `voicechat/stt.py` | dedicated thread | process-lifetime |
| VAD + Smart Turn (ONNX) | `voicechat/vad.py` | asyncio tasks on mic chunks | per-session |
| Face crop (webcam) | `voicechat/face_crop.py` | asyncio + CPU executor | per-session |
| Ring buffers | `voicechat/rings.py` | shared data | per-session |
| Event wiring | `voicechat/session.py` | asyncio handlers | per-session |

No orchestrator FSM. Just event handlers + a single `reply_task` handle for cancellation.

## 4. Event-driven flow (pipelined per-utterance batch)

Design decision (moved away from rolling-window wav2vec after empirically finding cosine ~0.84/0.92 on the rolling features — wav2vec2's bidirectional attention makes features too context-dependent to approximate from a short window).

Instead we **batch wav2vec on each complete utterance** while keeping everything else streaming. Since Phonon's RTF is 0.22 (produces audio 4.5× faster than realtime) and LLM+TTS pipeline overlap, the added latency vs. rolling is ~100 ms. We get full batch-quality features and much simpler code.

Generation of the avatar still **never stops**. What changes is how `avatar_audio` conditioning gets to the model: instead of raw audio in a ring (needing rolling wav2vec), we have a ring of **pre-computed audio features** that gets written in a batch at each sentence boundary.

```
mic PCM ──┬──▶ user_audio_ring            (continuous, raw audio)
          │
          ├──▶ Silero VAD  ─┐
          │                 ├──▶ Smart Turn ──▶ "turn_complete(audio)" event
          │                 │
          │                 └──▶ "start_of_speech" event  (barge-in signal)
          │
          └──▶ stt_buffer (accumulates during an utterance)

webcam ─[face crop]──▶ user_video_ring   (continuous, face-cropped RGB)

on "turn_complete(audio)":                 # user finished speaking
    text = whisper(audio)                  # STT
    reply_task = asyncio.create_task(
        reply_pipeline(text)               # LLM → Phonon → wav2vec batch → ring writes
    )

on "start_of_speech" (VAD fires again):    # barge-in
    if reply_task and not reply_task.done():
        reply_task.cancel()                # stops LLM, TTS, and ring writes
    # No new feature writes → avatar reads zero-padded features (silence equivalent)
    # → avatar transitions to listening expressions on the next block

async def reply_pipeline(text):
    """Pipeline LLM→Phonon→wav2vec, writing features to the ring at the
    time slot the avatar will read them. Audio is also sent to the browser
    audio track in lockstep so A/V stays synced."""
    audio_buf = []
    phrase_boundary_re = r'[.!?,:;]\s+'       # commit on punctuation for lower latency
    async for text_fragment in stream_llm(text):
        async for pcm_chunk in phonon.stream(text_fragment):
            audio_buf.append(pcm_chunk)
            # On phrase boundary, commit this chunk's audio:
            if is_phrase_end(text_fragment):
                audio_sentence = np.concatenate(audio_buf)
                features = avatar_wav2vec_batch(audio_sentence)   # [1, N_frames, dim_w]
                # Time-align: write features to the ring at the slot where the
                # avatar worker will read them (= when avatar_audio_track will
                # also be emitting this audio to the browser).
                target_ts = next_reply_slot_ts()
                avatar_audio_feat_ring.write(features, ts=target_ts)
                avatar_audio_pcm_ring.write(audio_sentence, ts=target_ts)
                audio_buf.clear()

# Avatar worker (dedicated thread, continuous):
while True:
    end = now()
    start = end - 2.0
    uv = user_video_ring.read_window(start, end)        # webcam face crops
    af = avatar_audio_feat_ring.read_window(start, end) # TTS features (or zero-padded)
    # user_audio features: for v1, feed zeros — rely on user_video for reactivity.
    # (Rolling wav2vec on user mic has the same approximation issue; low priority.)
    uf = torch.zeros_like(af)
    block = model.inference_stream_from_features(af, uf, uv)
    avatar_video_queue.put(block)

# WebRTC audio track for the browser reads from avatar_audio_pcm_ring at the
# same wall-clock cadence the avatar video track emits frames, so A/V is
# aligned at the ring timestamps.
```

Two event handlers + one asyncio task handle + one worker thread, same as before. The only new wrinkle is that ring writes happen in **phrase-sized bursts** rather than chunk-by-chunk, and we write **features** rather than raw audio for the avatar's wav2vec path.

## 5. Threading / task layout

**asyncio event loop (main thread)** — all I/O and cheap CPU work:
- aiohttp HTTP + aiortc RTCPeerConnection
- WebRTC media track `recv()` for mic and webcam
- Mic chunks → VAD → Smart Turn (ONNX; fast enough to run on the loop)
- Webcam frames → face crop (dispatched to a CPU executor; ~10 ms/frame)
- `reply_pipeline` task: awaits LLM SSE stream (vLLM), pushes LLM text to TTS worker's input queue, reads TTS output queue, writes PCM to `avatar_audio_ring`
- Ring writes from all sources

**Dedicated long-lived worker threads** (one of each; dynamo cache lives in the thread):
- **avatar worker** — `AvatarForcing.inference_stream_rolling` loop; reads 3 rings, writes `avatar_video_queue`.
- **STT worker** — Parakeet forward pass on turn-complete audio. Input queue of audio tensors, output queue of text.
- **TTS worker** — Phonon synth. Input queue of text sentences, output queue of PCM chunks.

**vLLM** — separate OS process on `:8000`, standard OpenAI-compatible server. Our code just HTTPs it.

## 6. Interfaces

### Ring buffers (`rings.py`)

Four rings. All time-indexed. All zero-pad / last-good on underflow — that's the one property that makes the whole event-driven design simple.

```python
class AudioRing:
    """16 kHz f32 mono ring with wall-clock timestamping.

    Writes advance an internal write_ts cursor.
    Reads for a window that extends past write_ts return zero-padded tail.
    Reads for a window that starts before the ring's backing buffer return
    zero-padded head.
    """
    def write(self, samples: np.ndarray, start_ts: float) -> None: ...
    def read_window(self, t0: float, t1: float, n_samples: int) -> np.ndarray: ...
    @property
    def write_ts(self) -> float: ...
```

```python
class VideoRing:
    """25 fps face-crop ring. 'Silence' = last-good frame (not zeros)
    because a black frame to the avatar is a worse signal than a
    repeated frame of the user holding still."""
    def write(self, frame: np.ndarray, ts: float) -> None: ...
    def read_window(self, t0: float, t1: float, n_frames: int) -> np.ndarray: ...
```

Four rings instantiated per session:
- `user_audio_ring`      (16 kHz raw audio, 30 s capacity) — for STT buffering
- `user_video_ring`      (25 fps, 30 s, 512², RGB uint8) — for avatar user_video conditioning
- `avatar_audio_feat_ring` (25 fps, 30 s, `dim_w`=512 float) — pre-computed wav2vec features for avatar conditioning
- `avatar_audio_pcm_ring`  (16 kHz raw audio, 30 s) — same audio as features but at native rate, for WebRTC audio-out to browser (A/V synced via timestamps)

Plus a plain `asyncio.Queue` for decoded avatar pixel blocks that the WebRTC video track reads from.

A new ring type is needed:

```python
class FeatureRing:
    """Frame-aligned float32 feature ring (e.g. for wav2vec features).
    Same zero-pad semantics as AudioRing but with a per-frame feature vector."""
    def __init__(self, capacity_seconds: float = 30.0, fps: int = 25, dim: int = 512): ...
    def write(self, feats: np.ndarray, start_ts: float) -> None: ...  # [T, dim]
    def read_window(self, t0: float, t1: float) -> np.ndarray: ...     # [T, dim], zero-pad
```

### What writes what

| Ring | Writer |
|---|---|
| `user_audio_ring` | WebRTC mic track `recv()` handler (every ~20 ms) |
| `user_video_ring` | WebRTC cam track `recv()` handler → face crop → write (every 40 ms at 25 fps) |
| `avatar_audio_feat_ring` | `reply_pipeline`: on each phrase boundary, wav2vec-batch the phrase's PCM audio and write features here at the target time slot |
| `avatar_audio_pcm_ring` | Same phrase boundary, write raw PCM here at same time slot (browser's audio-out track reads this) |
| `avatar_video_queue` | avatar worker thread |

### Barge-in = task cancellation + ring underflow

No swap, no state change. When VAD fires start-of-speech:
1. `reply_task.cancel()` kills the LLM SSE stream.
2. TTS worker sees its input queue drained and goes idle.
3. `avatar_audio_feat_ring` and `avatar_audio_pcm_ring` stop getting new writes; their `write_ts` stops advancing.
4. Avatar worker's next block read is for `[now-2s, now]`. The part past the old `write_ts` zero-pads (features = silence).
5. Over 2 s, the ring's "active" portion rolls off; the avatar transitions smoothly to listening expressions, driven by its user_video conditioning. The browser's audio track also goes silent at the same timestamp.

## 7. Feature-driven avatar generation path

Instead of the existing `inference_stream()` that consumes raw `avatar_audio` and runs wav2vec internally each block, we add `inference_stream_from_features()` that takes **pre-computed wav2vec features** directly.

```python
def inference_stream_from_features(
    self,
    avatar_ref,                       # [1, 3, H, W]
    avatar_feats: torch.Tensor,       # [1, T, dim_w]  — pre-computed wav2vec
    user_feats: torch.Tensor,         # [1, T, dim_w]  — zeros for v1
    user_frames: torch.Tensor,        # [1, T, 3, H, W]
    nfe=10, a_cfg_scale=2.0, u_cfg_scale=2.0, seed=25,
):
    """Same as inference_stream() but skips the audio_encoder calls.
    Features are provided from a FeatureRing the caller owns."""
```

The reply_pipeline writes features to `avatar_audio_feat_ring` at phrase boundaries (one batched wav2vec call per phrase). The avatar worker reads `[now-2s, now]` features from the ring every block and runs the flow transformer + decoder.

Empirical validation: the batch wav2vec path is already proven — that's what the phase C webrtc demo used. No additional validation needed for this split; we're just moving where the features get computed and passing them in as a tensor instead of recomputing per block.

Rolling wav2vec was explored (voicechat/tests/test_rolling_wav2vec.py) and set aside due to cosine 0.84/0.92 fidelity to batch — unacceptable risk when the batch alternative has ~100 ms latency cost and much simpler semantics.

## 8. Latency budget

Target: user stops speaking → audio reply within 800 ms, lip motion within 1.2 s.

Measured individually on this hardware (where known):
- VAD: 0.07 ms/chunk (negligible)
- Smart Turn: ~11 ms (from MacVoice CPU numbers; may be faster on this Linux box)
- End-of-turn silence padding (needed for turn detector): ~250-300 ms of "did user stop talking"
- Parakeet STT: ~60 ms for a 3-5 s utterance (batched, after turn complete)
- Qwen3-4B first-token: ~150-250 ms (vLLM, fresh context)
- Phonon TTFA: 35-45 ms (measured on this box CPU)
- 24 kHz → 16 kHz resample: ~2 ms per 100 ms audio
- Avatar first-reply latency: ~80 ms (one block, no primer since gen runs continuously)

Sum: **~550-700 ms** for audio reply to start. Video lip motion follows by one block (~85 ms) → ~700-800 ms. In budget.

Open risk: if Qwen3-4B generates slower than expected, LLM dominates. Mitigation: smaller model (Qwen3-1.7B) or quantization of the KV cache.

## 9. Failure modes

| Failure | Detection | Response |
|---|---|---|
| vLLM unreachable | HTTP connect error on first call | Return canned apology via Phonon; retry background reconnect |
| TTS worker hangs | Chunk timeout > 500 ms | Kill thread, respawn; ring just underflows → silence |
| Avatar worker hangs | No video frames for 400 ms | Emit last-good placeholder frames, log; in v1, probably drop the session |
| Browser disconnects | `connectionstatechange` → disconnected | Cancel reply_task, stop workers, close rings |
| Mic / webcam gap | Ring write_ts doesn't advance | Read zero-pads (audio) or returns last-good (video); orchestration keeps running |

Explicit non-goals for v1 error handling: retries, graceful degradation, reconnect without re-primer. Session-restart is the recovery mechanism.

## 10. Test plan

Two phases:

**Phase 1 — components in isolation:**
- `voicechat/tests/test_stt.py`: feed a known WAV → text matches.
- `voicechat/tests/test_tts.py`: text → audio, TTFA < 100 ms warm.
- `voicechat/tests/test_llm.py`: prompt → first-token < 300 ms.
- `voicechat/tests/test_rolling_wav2vec.py`: compare output of rolling vs batch wav2vec on same audio — pixel diff on the first 10 avatar blocks should be <1% RMSE.

**Phase 2 — end-to-end:**
- Open browser, say "Hi, what's your name?" → audio reply within 800 ms, lip motion within 1.2 s, measured in browser via `performance.now()` stamps.
- Barge-in test: avatar mid-sentence → user interrupts → avatar audio cuts within 150 ms, avatar transitions to listening expression within one block.
- Durability: 10-minute continuous session, no crashes, no identity drift to the point of unrecognizability.

## 11. Open risks ordered by impact

1. **Model quality on live audio + webcam.** The paper tested on dyadic datasets with high-quality audio and clean face crops. Live mic audio with room noise + a webcam at random pose may degrade quality visibly. No way to know without testing. Mitigation: tune `a_cfg_scale` and `u_cfg_scale` empirically.
2. **Identity drift over long sessions.** Attention-sinks fix we discussed isn't in yet. Issue #10 suggests degradation at 30-60 s. Full-duplex continuous generation may hit this hard. Mitigation: ship attention-sinks patch alongside; it's a 10-line change.
3. **Rolling wav2vec might produce boundary artifacts.** If the first few frames of each 32-frame window have unstable features, the avatar could get jittery lips every block. Mitigation: validate in Phase 1 test, add longer lookback if needed.
4. **Barge-in latency.** 150 ms budget assumes VAD fires fast and we can mutate the avatar_audio ring instantly. If any part of the chain buffers, barge-in feels laggy. Mitigation: keep buffer depths minimal; measure.
5. **Webcam preprocessing throughput.** Face detection per frame at 25 fps = 40 ms/frame budget. `face_alignment`'s S3FD is ~10 ms/frame on this CPU, so we have margin. If we run webcam at higher fps, this becomes tighter; stay at 25.
6. **Phonon English-only.** Default model is `phonon-31m-en`. If you want multilingual, swap to `phonon-40m-multilingual` (CC BY-NC, gated). Phonon-31m-en it is for the demo.

## 12. Build order

1. ✅ **Design review** — this doc (iterated once to drop FSM; iterated again to drop rolling-wav2vec in favor of pipelined per-utterance batching).
2. ✅ **vLLM + Qwen3-4B AWQ** — running, TTFT 26 ms warm, 13 GB VRAM.
3. ✅ **faster-whisper distil-large-v3** — 105 ms warm on 16 s clip.
4. ✅ **Silero VAD + Smart Turn v3.2** — 0.5 ms / 16.5 ms per inference.
5. ✅ **Ring buffers** — AudioRing + VideoRing in `voicechat/rings.py`, smoke tested.
6. **FeatureRing + `inference_stream_from_features()`** — feature-ring class + the new avatar method that accepts pre-computed features. Validation: run a clip batch-compute wav2vec → feed through features API → compare pixel output to the existing batch path (must be bit-identical modulo RNG).
7. **Bidirectional WebRTC**: adapt `webrtc/server.py` to also *receive* mic + webcam from the browser.
8. **Face crop pipeline**: webcam frames → `preprocess_face` per frame → user_video ring.
9. **Event wiring**: session.py builds ring instances + wires VAD/STT/LLM/TTS/wav2vec/avatar, with a single `reply_task` handle and the barge-in cancel path.
10. **End-to-end latency measurement + tuning** on Phonon TTFA / phrase-boundary timing / CFG scales.
11. **Capture a demo recording**; write up the post.

Estimated remaining effort: 2-3 focused days. The event wiring (step 9) is now the hardest piece since we removed the state machine and (new) rolling wav2vec risk.
