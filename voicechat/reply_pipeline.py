"""
reply_pipeline: run LLM → Phonon TTS → wav2vec batch → write to avatar rings.

Runs as an asyncio task. On barge-in the surrounding code cancels this task;
the cancellation propagates through the generators cleanly.
"""
from __future__ import annotations

import asyncio
import logging
import os as _os
import sys as _sys
import time
from typing import Callable

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import torch

from voicechat.llm import VLLMClient
from voicechat.tts import PhononTTS, resample_24k_to_16k
from voicechat.rings import AudioRing, FeatureRing

log = logging.getLogger("af-reply")


import re as _re
_SPEAKABLE = _re.compile(r"[A-Za-z0-9]")
# Strip emoji + pictographic symbols before TTS. Phonon ignores them, but
# other TTS engines sometimes vocalize the emoji name ("sparkle emoji"),
# which is never what you want in a voice chat. Coverage: Misc Symbols
# (U+2600–26FF), Dingbats (U+2700–27BF), and the Supplementary Planes
# where most emoji live (U+1F000–1FFFF). Also strip variation selectors.
_EMOJI = _re.compile(
    r"[\U0001F000-\U0001FFFF☀-➿️‍]+"
)


def _strip_emoji(text: str) -> str:
    return _EMOJI.sub("", text)

async def speak_phrase(
    phrase: str,
    tts: PhononTTS,
    af_model,
    feat_ring: FeatureRing,
    pcm_ring: AudioRing,
    now_fn: Callable[[], float],
    device: str = "cuda",
    worker=None,   # AvatarWorker: if set, delegate wav2vec to worker thread
) -> None:
    """Synthesize `phrase` → resample 24k→16k → submit to avatar worker for wav2vec + ring writes."""
    if not _SPEAKABLE.search(phrase):
        log.info(f"skipping non-speakable phrase: {phrase!r}")
        return
    # Strip emojis before TTS. Phonon ignores them silently; other backends
    # (e.g. some NeMo / piper configurations) vocalize them as the emoji
    # name. Belt-and-suspenders defensive hygiene.
    phrase_tts = _strip_emoji(phrase).strip() or phrase
    chunks = []
    async for c in tts.stream(phrase_tts):
        chunks.append(c)
    if not chunks:
        return
    audio_24k = np.concatenate(chunks)
    audio_16k = resample_24k_to_16k(audio_24k)
    if worker is not None:
        worker.submit_tts(phrase, audio_16k)
        log.info(f"phrase={phrase[:60]!r} audio={audio_16k.size/16000:.2f}s  (submitted to worker)")
        return

    # --- Legacy fallback (wav2vec on this thread) — not used in voice-chat demo ---
    n_video_frames = audio_16k.size * 25 // 16000
    if n_video_frames <= 0:
        return
    audio_16k = audio_16k[:n_video_frames * 640]
    t0 = time.perf_counter()
    audio_tensor = torch.from_numpy(audio_16k).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        feats = af_model.audio_encoder.inference(audio_tensor, seq_len=n_video_frames)
    feats_np = feats[0].float().cpu().numpy()
    t_wav2vec = (time.perf_counter() - t0) * 1000
    target_ts = max(feat_ring.write_ts, pcm_ring.write_ts, now_fn())
    feat_ring.write(feats_np, start_ts=target_ts)
    pcm_ring.write(audio_16k, start_ts=target_ts)
    log.info(f"phrase={phrase[:60]!r} audio={audio_16k.size/16000:.2f}s  "
             f"wav2vec={t_wav2vec:.0f}ms  scheduled@{target_ts:.2f}")


async def reply_pipeline(
    user_text: str,
    llm: VLLMClient,
    tts: PhononTTS,
    af_model,
    feat_ring: FeatureRing,
    pcm_ring: AudioRing,
    now_fn: Callable[[], float],
    device: str = "cuda",
    worker=None,
    on_phrase: Callable[[str], None] | None = None,
) -> None:
    """LLM → TTS → submit audio to avatar worker (wav2vec runs there).

    `on_phrase` fires after each phrase is successfully queued for audio,
    so callers (e.g. the chat UI) stay in sync with what the avatar will
    actually say — even if a barge-in cancels mid-reply.
    """
    log.info(f"reply_pipeline: user={user_text!r}")
    batcher = VLLMClient.phrase_batcher()
    t0 = time.perf_counter()
    first_phrase_t = None
    n_phrases = 0

    async def _handle(phrase: str) -> None:
        nonlocal first_phrase_t, n_phrases
        if first_phrase_t is None:
            first_phrase_t = time.perf_counter() - t0
        n_phrases += 1
        await speak_phrase(phrase, tts, af_model, feat_ring, pcm_ring, now_fn, device, worker=worker)
        if on_phrase is not None and _SPEAKABLE.search(phrase):
            try: on_phrase(phrase)
            except Exception:
                log.exception("on_phrase callback failed")

    try:
        async for frag in llm.stream(user_text):
            for phrase in batcher(frag):
                await _handle(phrase)
        for phrase in batcher.flush():
            await _handle(phrase)
        dt = time.perf_counter() - t0
        log.info(f"reply_pipeline done: {n_phrases} phrases in {dt*1000:.0f}ms  "
                 f"first-phrase-ready {first_phrase_t*1000 if first_phrase_t else -1:.0f}ms")
    except asyncio.CancelledError:
        log.info(f"reply_pipeline cancelled (barge-in). had delivered {n_phrases} phrases.")
        raise


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "avatarforcing"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    async def main():
        # Need the AvatarForcing model's audio_encoder
        from inference import InferenceAgent
        from omegaconf import OmegaConf
        class Opt: pass
        opt = Opt()
        cfg = OmegaConf.load("configs/inference.yaml")
        for k, v in cfg.items(): setattr(opt, k, v)
        opt.infer_config="configs/inference.yaml"; opt.mae_ckpt_path="pretrained_dir/motion_autoencoder.pth"
        opt.ckpt_path="pretrained_dir/flow_transformer.pth"; opt.rank=0; opt.ngpus=1
        opt.result_dir="results/reply_test"; opt.compile=False; opt.warmup=0
        os.makedirs(opt.result_dir, exist_ok=True)
        agent = InferenceAgent(opt)

        llm = VLLMClient(model="cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit")
        tts = PhononTTS()
        feat_ring = FeatureRing(capacity_seconds=30.0, fps=25, dim=512)
        pcm_ring  = AudioRing(capacity_seconds=30.0, sample_rate=16000)
        now = lambda: time.monotonic()
        t_start = now()
        now_rel = lambda: now() - t_start

        for turn in ["Hi!", "Tell me a one-sentence fun fact about octopuses."]:
            await reply_pipeline(turn, llm, tts, agent.G, feat_ring, pcm_ring, now_rel)
            print(f"\n[feat_ring write_ts={feat_ring.write_ts:.2f}]")
            print(f"[pcm_ring write_ts={pcm_ring.write_ts:.2f}]")
        await llm.close()
    asyncio.run(main())
