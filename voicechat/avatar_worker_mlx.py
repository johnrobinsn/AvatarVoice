"""
Avatar worker for Apple Silicon (AF_BACKEND=mlx).

Same session interface and streaming behaviour as `AvatarWorker` (start_session /
submit_tts / barge_in / end_session, ring reads with pink-noise idle
features, wall-clock pacing, EWMA for the adaptive playback delay, periodic KV
reanchor with crossfade) — the CUDA worker's session management, feature reads
and pixel emission are inherited unchanged. What differs is the engine:

  - flow transformer + diffusion-forcing loop: MLX (mlx_port/sampler_mlx.py)
  - motion encoder/decoder, wav2vec2: torch MPS (mlx_port/pipeline_mlx.py)
  - wav2vec runs in this thread between blocks (~10-25 ms per phrase on MPS)
    instead of on a CUDA side stream.
  - no CUDA streams / torch.cuda.* calls.

MLX streams are thread-local, so the pipeline is built inside run() on the
worker thread and used only there.
"""
from __future__ import annotations

import logging
import os
import queue
import sys
import time
from pathlib import Path

import numpy as np
import torch

from voicechat.avatar_worker import AvatarWorker, _shaped_pink_noise

_MLX_PORT = Path(__file__).resolve().parent.parent / "mlx_port"
if str(_MLX_PORT) not in sys.path:
    sys.path.insert(0, str(_MLX_PORT))

log = logging.getLogger("af-avatar")


class MLXAvatarWorker(AvatarWorker):
    def __init__(self, avatar_ref_tensor: torch.Tensor, a_cfg_scale: float = 2.0,
                 nfe: int = 6, seed: int = 25, flow_dtype: str | None = None):
        super().__init__(af_model=None, avatar_ref_tensor=avatar_ref_tensor, device="cpu",
                         a_cfg_scale=a_cfg_scale, nfe=nfe, seed=seed)
        self.name = "af-avatar-mlx"
        self.flow_dtype = flow_dtype or os.environ.get("AF_MLX_FLOW_DTYPE", "float32")
        self.pipe = None
        self.block_time_ewma = 0.3

    # ----- lifecycle -----

    def run(self):
        log.info("avatar worker (mlx): starting (persistent mode)")
        try:
            self._warmup()
            self._warmup_done.set()
            log.info("avatar worker (mlx): idle, waiting for sessions")
            while not self._shutdown.is_set():
                self._session_ready.wait()
                if self._shutdown.is_set():
                    break
                with self._state_lock:
                    self._session_ready.clear()
                log.info("avatar worker (mlx): new session attached, running loop")
                try:
                    self._run_loop()
                except Exception:
                    log.exception("avatar worker (mlx): session loop crashed")
                finally:
                    if self.out_q is not None:
                        try: self.out_q.put_nowait(None)
                        except queue.Full: pass
                    torch.mps.empty_cache()
                log.info("avatar worker (mlx): session ended, back to idle")
        except Exception:
            log.exception("avatar worker (mlx): fatal crash")
            self._warmup_done.set()   # don't leave init_singletons blocked forever

    def _warmup(self):
        import mlx.core as mx
        import pipeline_mlx as P
        t0 = time.perf_counter()
        log.info(f"avatar worker (mlx): building pipeline (flow {self.flow_dtype}, nfe={self.nfe})")
        self.pipe = P.HybridAvatarPipeline(flow_dtype=getattr(mx, self.flow_dtype),
                                           nfe=self.nfe, a_cfg_scale=self.a_cfg)
        self.pipe.set_reference(self.avatar_ref)      # also compiles the decoder
        log.info(f"avatar worker (mlx): pipeline ready ({time.perf_counter()-t0:.1f}s)")

        # Pink-noise "room tone" features (see AvatarWorker._warmup).
        n_sil_sec = 5
        pcm_pink = _shaped_pink_noise(n_sil_sec * 16000, sample_rate=16000, level_dbfs=-50.0)
        self.silence_feats_np = np.array(
            self.pipe.audio_features(pcm_pink, n_sil_sec * 25, normalize=False)[0])
        log.info(f"avatar worker (mlx): cached {self.silence_feats_np.shape} pink-noise silence features")

        # Warm MLX kernels for both shapes (primer + steady block).
        s = self.pipe.sampler
        rng = np.random.default_rng(0)
        feats = mx.array(np.tile(self.silence_feats_np, (2, 1))[None])
        s.reset(1)
        x_t = s.prime(feats[:, :50], self.pipe.r_s, mx.array(rng.standard_normal((1, 50, 512), dtype=np.float32)))
        for b in range(3):
            t = 50 + 10 * b
            x_t = s.step(feats[:, t - 2:t + 10], self.pipe.r_s, x_t,
                         mx.array(rng.standard_normal((1, 10, 512), dtype=np.float32)), t - 2)
            self.pipe.decode(x_t[:, -10:])
        mx.eval(x_t)
        torch.mps.synchronize()
        log.info(f"avatar worker (mlx): warmup DONE in {time.perf_counter()-t0:.1f}s")

    # ----- helpers -----

    def _feats(self, t0: float, n_frames: int):
        """Ring features (with pink-noise silence fill) as an MLX [1, n, 512] array."""
        import mlx.core as mx
        return mx.array(self._features_from_feat_ring(t0, n_frames).numpy())

    def _noise(self, n_frames: int):
        import mlx.core as mx
        return mx.array(self._rng.standard_normal((1, n_frames, 512), dtype=np.float32))

    def _drain_tts_queue(self):
        """wav2vec each queued TTS phrase and schedule features + PCM into the
        rings at the generation horizon (same scheduling as the CUDA worker)."""
        while True:
            epoch = self._tts_epoch
            try:
                phrase, pcm_16k = self.tts_in_q.get_nowait()
            except queue.Empty:
                return
            try:
                n_frames = max(1, (pcm_16k.size + 639) // 640)
                n_samples_needed = n_frames * 640
                if pcm_16k.size < n_samples_needed:
                    pcm_16k = np.concatenate([pcm_16k, np.zeros(n_samples_needed - pcm_16k.size, np.float32)])
                else:
                    pcm_16k = pcm_16k[:n_samples_needed]
                tt = time.perf_counter()
                feats_np = np.array(self.pipe.audio_features(pcm_16k, n_frames, normalize=False)[0])
                ms = (time.perf_counter() - tt) * 1000
                self._schedule_phrase(epoch, phrase, feats_np, pcm_16k, ms, "wav2vec")
            except Exception:
                log.exception("[wav2vec] phrase processing failed")

    def _emit(self, pixels: torch.Tensor, block_start_ts: float):
        self._emit_pixel_block(pixels.unsqueeze(0).float(), block_start_ts)

    # ----- main loop -----

    def _run_loop(self):
        import mlx.core as mx
        pipe, s = self.pipe, self.pipe.sampler
        clip, block = pipe.clip, pipe.block      # 50, 10
        self._rng = np.random.default_rng(self.seed)

        pipe.set_reference(self.avatar_ref)      # avatar may have changed (/set-avatar)

        # Primer: establishes KV state only, not emitted (see AvatarWorker._run_loop).
        t_now = self.now()
        prime_start = max(0.0, t_now - clip / 25.0)
        tp = time.perf_counter()
        s.reset(1)
        x_t = s.prime(self._feats(prime_start, clip), pipe.r_s, self._noise(clip))
        mx.eval(x_t, s.kv_cache)
        log.info(f"avatar worker (mlx): primer done in {(time.perf_counter()-tp)*1000:.0f} ms")

        gen_t = clip
        next_block_ts = max(prime_start + clip / 25.0, self.now())
        start_pos = gen_t - 2
        self.gen_horizon = next_block_ts
        REANCHOR_EVERY = 112   # ~45 s
        MAX_DEFER = 162        # ~65 s hard cap
        blocks_since_reanchor = 0

        while not (self._session_end.is_set() or self._shutdown.is_set()):
            t_loop0 = time.perf_counter()
            self._drain_tts_queue()
            block_end_ts = next_block_ts + block / 25.0

            do_reanchor = False
            if blocks_since_reanchor >= REANCHOR_EVERY and self.tts_in_q.empty():
                pb_delay = max(0.15, min(0.6, self.block_time_ewma * 1.2))
                if self.af_ring.write_ts <= self.now() - pb_delay + 0.05:
                    do_reanchor = True
            if blocks_since_reanchor >= MAX_DEFER:
                do_reanchor = True
                log.info(f"avatar worker (mlx): forcing reanchor (deferred {blocks_since_reanchor} blocks)")

            if do_reanchor:
                # Fresh 50-frame primer ending at this block; its frames aren't
                # emitted — the next block crossfades from the last shown frame.
                window_start_ts = next_block_ts + block / 25.0 - clip / 25.0
                s.reset(1)
                x_t = s.prime(self._feats(window_start_ts, clip), pipe.r_s, self._noise(clip))
                mx.eval(x_t, s.kv_cache)
                if self._last_emitted_frame is not None:
                    self._fade_anchor = self._last_emitted_frame.copy()
                self._reanchor_fade_remaining = self.REANCHOR_FADE_FRAMES
                log.info(f"avatar worker (mlx): reanchored @ gen_t={gen_t} "
                         f"(skipped emit, fade armed {self.REANCHOR_FADE_FRAMES} frames)")
                if self.on_reanchor is not None:
                    try: self.on_reanchor()
                    except Exception:
                        log.exception("on_reanchor callback failed")
                blocks_since_reanchor = 0
                gen_t = clip
                start_pos = gen_t - 2
                compute_time = time.perf_counter() - t_loop0
            else:
                wa = self._feats(next_block_ts - 2 / 25, block + 2)
                x_t = s.step(wa, pipe.r_s, x_t, self._noise(block), start_pos)
                latents = x_t[:, -block:]
                mx.eval(latents, s.kv_cache)
                pixels = pipe.decode(latents)
                torch.mps.synchronize()
                # EWMA over compute (flow + decode) only, not the blocking
                # queue put or pacing sleep — same knob as the CUDA worker.
                compute_time = time.perf_counter() - t_loop0
                self._emit(pixels, next_block_ts)
                blocks_since_reanchor += 1
                start_pos = gen_t + block - 2
                gen_t += block

            next_block_ts = max(block_end_ts, self.now())
            self.gen_horizon = next_block_ts
            self._release_played()
            if compute_time > self.block_time_ewma:
                self.block_time_ewma = 0.5 * self.block_time_ewma + 0.5 * compute_time
            else:
                self.block_time_ewma = 0.92 * self.block_time_ewma + 0.08 * compute_time
            lead = next_block_ts - self.now()
            if lead > 0.0:
                time.sleep(min(lead, 0.4))

            if (gen_t // block) % 10 == 0:
                dt = (time.perf_counter() - t_loop0) * 1000
                log.info(f"avatar worker (mlx): block @ gen_t={gen_t} wall={dt:.0f}ms "
                         f"compute={compute_time*1000:.0f}ms q={self.out_q.qsize()} "
                         f"ewma={self.block_time_ewma*1000:.0f}ms")
