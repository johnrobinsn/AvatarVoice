"""
Avatar worker: continuous generation loop that drives AvatarForcing from ring reads.

- Runs in a dedicated thread (dynamo cache is thread-local — see webrtc/server.py
  where we proved this is necessary).
- On first run: encodes avatar_ref, primes 50-frame KV cache using whatever
  features are in the rings at `now - 2s` (likely zeros since no reply yet).
- Then forever: pull 12-frame windows from rings → one steady-state block →
  push 10 pixel frames to the output queue.

Inputs (all ring-indexed to a common wall-clock):
  user_video_ring        -> drives user_motion conditioning
  avatar_audio_feat_ring -> drives avatar lip-sync (pre-batched wav2vec features)

We write *features* (not raw audio) because the wav2vec batch happens once per
phrase in reply_pipeline, not once per block.
"""
from __future__ import annotations

import logging
import math
import queue
import threading
import time
from typing import Callable

import numpy as np
import torch

log = logging.getLogger("af-avatar")


def _shaped_pink_noise(n_samples: int, sample_rate: int = 16000,
                        level_dbfs: float = -50.0, seed: int = 42) -> np.ndarray:
    """Pink (1/f) noise at `level_dbfs` RMS. Mono f32. Used as a "room tone"
    fallback: wav2vec features of this live closer to the training distribution
    (every training clip had some ambient noise) than absolute-zero features,
    so the downstream model produces less phantom mouth motion during idle."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n_samples).astype(np.float32)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    spec *= 1.0 / np.sqrt(np.maximum(freqs, 1.0))
    pink = np.fft.irfft(spec, n=n_samples).astype(np.float32)
    rms = float(np.sqrt(np.mean(pink * pink)))
    target_rms = 10.0 ** (level_dbfs / 20.0)
    pink *= target_rms / (rms + 1e-12)
    return pink


class AvatarWorker(threading.Thread):
    """Persistent avatar generation thread. Runs torch.compile warmup ONCE at
    server startup; thereafter accepts per-session data via `start_session()`
    and runs the main loop until `end_session()` or server shutdown.

    Only one session at a time (v1).
    """
    def __init__(
        self,
        af_model,
        avatar_ref_tensor: torch.Tensor,   # [1, 3, 512, 512], already preprocessed+normalized
        device: str = "cuda",
        a_cfg_scale: float = 2.0,
        u_cfg_scale: float = 0.0,   # 0 = ignore user-motion/user-audio CFG branch; avatar driven only by its own TTS. (Original model expects a dialog partner's face; a listener's idle webcam drove the mouth as if speaking.)
        nfe: int = 6,
        seed: int = 25,
    ):
        super().__init__(daemon=True, name="af-avatar")
        self.af = af_model
        self.avatar_ref = avatar_ref_tensor
        self.device = device
        self.a_cfg = a_cfg_scale
        self.u_cfg = u_cfg_scale
        self.nfe = nfe
        self.seed = seed
        # Per-session fields; set by start_session()
        self.uv_ring = None
        self.af_ring = None
        self.pcm_ring = None   # avatar_audio_pcm_ring; worker also writes to this
        self.out_q = None
        self.now = None
        # TTS -> wav2vec work queue — reply_pipeline posts (text, pcm_f32_16k) items here
        # and worker handles the wav2vec batch itself so no cross-thread GPU contention.
        self.tts_in_q: "queue.Queue" = queue.Queue(maxsize=32)
        self._session_ready = threading.Event()
        self._session_end = threading.Event()
        self._shutdown = threading.Event()
        self._warmup_done = threading.Event()
        # Serializes start_session / end_session / worker-consume so a
        # start_session arriving during cleanup can't be clobbered.
        self._state_lock = threading.Lock()
        # Barge-in: barge_in() bumps the epoch under _sched_lock; a phrase whose
        # wav2vec started in an older epoch is dropped instead of scheduled.
        self._sched_lock = threading.Lock()
        self._tts_epoch = 0
        # Pink-noise "room tone" features — computed once in _warmup, read in
        # _features_from_feat_ring. Keeps the avatar's mouth quiet during idle
        # by feeding in-distribution noise features instead of zeros.
        self.silence_feats_np: np.ndarray | None = None
        self._silence_phase = 0
        # Reanchor crossfade state. On reanchor we capture the last emitted
        # frame, skip the reanchor's own 10-frame emit (its content is 400 ms
        # of logical video for 1.3 s of wall-clock — would put video behind
        # audio), and then alpha-blend it with the next REANCHOR_FADE_FRAMES
        # frames so the reconnect is a smooth 1 s crossfade instead of a cut.
        self._last_emitted_frame: np.ndarray | None = None   # [H,W,3] uint8
        # Separate anchor: snapshot of the pre-reanchor last frame, held
        # constant through the whole fade. Without this, each blended frame
        # feeds back as the next frame's "anchor" and we get recursive
        # ghosting instead of a clean crossfade.
        self._fade_anchor: np.ndarray | None = None
        self._reanchor_fade_remaining: int = 0
        # Dedicated wav2vec thread. Runs on its own CUDA stream so its kernels
        # can overlap with the main worker's flow-transformer block generation
        # instead of blocking it (previously a 400 ms wav2vec call between
        # blocks would starve the video queue by exactly one block).
        self._wav2vec_thread: threading.Thread | None = None

    @property
    def warmup_done(self) -> bool:
        return self._warmup_done.is_set()

    def start_session(self, uv_ring, af_ring, pcm_ring, out_q, now_fn,
                      playhead_fn=None, on_reanchor=None):
        with self._state_lock:
            self.uv_ring = uv_ring
            self.af_ring = af_ring
            self.pcm_ring = pcm_ring
            self.out_q = out_q
            self.now = now_fn
            # Optional: logical video playhead. If set, TTS is scheduled near
            # this cursor (what's actually playing), not wall-clock — so
            # responses don't drift multi-second behind when generation can't
            # quite hit realtime.
            self.playhead = playhead_fn if playhead_fn is not None else now_fn
            # Optional: called from this thread each time a reanchor fires,
            # so the Session can forward a UI log event to the browser.
            self.on_reanchor = on_reanchor
            # Exponentially-weighted block wall-time. Audio/video tracks read
            # this to compute a self-tuning playback delay: delay ≈ this * 1.2.
            # Fast-rise (~2 blocks) to recover quickly from a spike; slow-decay
            # (~15 blocks) so we don't shrink the buffer before the spike passes.
            self.block_time_ewma = 0.3
            # Drain any stale tts items from a previous session
            try:
                while True: self.tts_in_q.get_nowait()
            except queue.Empty:
                pass
            self._session_end.clear()
            self._session_ready.set()

    def submit_tts(self, phrase: str, pcm_16k: np.ndarray) -> None:
        """Called from reply_pipeline (any thread). Worker thread will run
        wav2vec on this PCM between blocks and write features+pcm to the rings."""
        try:
            self.tts_in_q.put_nowait((phrase, pcm_16k))
        except queue.Full:
            log.warning("tts_in_q full, dropping phrase")

    def barge_in(self, cut_ts: float) -> None:
        """Stop the avatar talking now (called on the user's start-of-speech).

        Drops TTS phrases not yet scheduled (queued, or mid-wav2vec) and cuts the
        PCM and feature rings at `cut_ts`, the audio playback cursor: playback
        goes silent at once, and from there the model is fed the idle (pink-noise)
        features _features_from_feat_ring substitutes past the ring's end.
        Frames already generated ahead of playback (< ~0.6 s) still play out."""
        with self._sched_lock:
            self._tts_epoch += 1
            dropped = 0
            while True:
                try:
                    self.tts_in_q.get_nowait()
                    dropped += 1
                except queue.Empty:
                    break
            cut_s = max(0.0, self.pcm_ring.write_ts - cut_ts) if self.pcm_ring is not None else 0.0
            if self.pcm_ring is not None:
                self.pcm_ring.truncate_at(cut_ts)
            if self.af_ring is not None:
                self.af_ring.truncate_at(cut_ts)
        log.info(f"avatar worker: barge-in at {cut_ts:.2f}: cut {cut_s:.2f}s of queued audio, "
                 f"dropped {dropped} pending phrase(s)")

    # Audio + features kept behind the session clock. Playback reads ≤ ~1 s behind
    # it (playback delay + lip-sync offset), the worker's windows are ahead of it.
    RELEASE_MARGIN_S = 10.0

    def _release_played(self) -> None:
        """Free avatar audio/features that have played (those rings grow; see Session)."""
        t = self.now() - self.RELEASE_MARGIN_S
        if t > 0:
            self.pcm_ring.release_before(t)
            self.af_ring.release_before(t)

    def _schedule_phrase(self, epoch: int, phrase: str, feats_np: np.ndarray,
                         pcm_16k: np.ndarray, w2v_ms: float, tag: str) -> None:
        """Write a wav2vec'd phrase into the rings at the start of the next block the
        worker will generate (anything between playhead and gen_horizon was drawn
        without it). Skipped if a barge-in happened since its wav2vec started."""
        with self._sched_lock:
            if epoch != self._tts_epoch:
                log.info(f"[{tag}] phrase={phrase[:40]!r} dropped (barge-in)")
                return
            horizon = getattr(self, "gen_horizon", self.playhead())
            target_ts = max(self.af_ring.write_ts, self.pcm_ring.write_ts, horizon)
            self.af_ring.write(feats_np, start_ts=target_ts)
            self.pcm_ring.write(pcm_16k, start_ts=target_ts)
        log.info(f"[{tag}] phrase={phrase[:40]!r} audio={pcm_16k.size/16000:.2f}s "
                 f"w2v={w2v_ms:.0f}ms scheduled@{target_ts:.2f}")

    def end_session(self):
        with self._state_lock:
            self._session_end.set()
            self._session_ready.clear()
            # Drain the out video queue so a blocking put() inside
            # _emit_pixel_block unblocks; otherwise the worker loop can't
            # re-check _session_end and the next session never starts.
            if self.out_q is not None:
                try:
                    while True: self.out_q.get_nowait()
                except queue.Empty:
                    pass

    def shutdown(self):
        self._shutdown.set()
        self._session_ready.set()  # unblock the wait loop
        self._session_end.set()

    # Legacy alias used elsewhere
    def stop(self):
        self.end_session()

    def run(self):
        log.info("avatar worker: starting (persistent mode)")
        import torch._dynamo as _dynamo
        # Keep cache small so dynamo evicts aggressively (prevents OOM). Scalar
        # ints are specialized-per-value by default — we accept that for now
        # and rely on cache_size_limit + empty_cache to bound memory.
        _dynamo.config.cache_size_limit = 4
        _dynamo.config.accumulated_cache_size_limit = 4
        # Pin this thread's default CUDA device so operations that touch
        # current-device state (empty_cache, implicit-device factories) don't
        # cross-talk with the main thread's cuda:0 context.
        dev_idx = int(self.device.split(":")[1]) if isinstance(self.device, str) and ":" in self.device else int(self.device)
        torch.cuda.set_device(dev_idx)
        try:
            self._warmup()
            self._warmup_done.set()
            # Start the dedicated wav2vec thread now that audio_encoder has
            # been pre-warmed on the main thread. The wav2vec thread will
            # also pre-warm on its own CUDA context before taking work.
            self._wav2vec_thread = threading.Thread(
                target=self._wav2vec_loop, daemon=True, name="af-wav2vec")
            self._wav2vec_thread.start()
            log.info("avatar worker: idle, waiting for sessions")
            while not self._shutdown.is_set():
                self._session_ready.wait()
                if self._shutdown.is_set():
                    break
                # Consume `ready` atomically with any pending start/end so a
                # start_session that arrives during cleanup isn't clobbered.
                # NO clears in the cleanup path below — start_session and
                # end_session own both events.
                with self._state_lock:
                    self._session_ready.clear()
                log.info("avatar worker: new session attached, running loop")
                try:
                    self._run_loop()
                except torch.cuda.OutOfMemoryError:
                    log.error("avatar worker: session OOM — closing session; "
                              "user can start a fresh one")
                except Exception:
                    log.exception("avatar worker: session loop crashed")
                finally:
                    # Sentinel for the outbound video track
                    if self.out_q is not None:
                        try: self.out_q.put_nowait(None)
                        except queue.Full: pass
                    # Clean up GPU memory between sessions so the next one
                    # starts fresh
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                log.info(f"avatar worker: session ended, back to idle "
                         f"(vram={torch.cuda.memory_allocated()/1e9:.1f}GB)")
        except Exception:
            log.exception("avatar worker: fatal crash")

    def _warmup(self):
        """Pay the torch.compile cost in THIS thread so production runs stay fast.
        Runs on a temporary tensor buffer; no ring dependency so it can run
        BEFORE any session attaches."""
        import time as _t
        log.info("avatar worker: warmup starting (torch.compile pass)")
        t0 = _t.perf_counter()
        af = self.af

        # Load 60 sample frames directly into a tensor; skip the ring entirely
        try:
            from pathlib import Path
            import cv2
            # In the AvatarVoice submodule layout the sample user frames live
            # under the AvatarForcing submodule's data dir.
            sample_dir = Path(__file__).resolve().parent.parent / "avatarforcing/data/user"
            frames_np = []
            for p in sorted(sample_dir.glob("*.jpg"))[:60]:
                img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                frames_np.append(img)
            while len(frames_np) < 60:
                frames_np.append(frames_np[-1])
            frames_np = np.stack(frames_np[:60])  # [60, 512, 512, 3] uint8
        except Exception:
            log.exception("warmup: couldn't load sample frames, aborting")
            return

        # Convert to [1, 60, 3, H, W] normalized as the rest of the model expects
        frames_t = torch.from_numpy(frames_np).to(self.device).float() / 255.0
        frames_t = (frames_t - 0.5) / 0.5
        frames_t = frames_t.permute(0, 3, 1, 2).unsqueeze(0)   # [1, 60, 3, H, W]

        num_frames_for_clip = af.num_frames_for_clip
        num_frames_per_block = af.num_frames_per_block
        block_size = num_frames_per_block

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            s = self.avatar_ref.to(self.device)
            s_r, r_s_lambda, s_r_feats = af.encode_image_into_latent(s)
            r_s = af.motion_autoencoder.dec.direction(r_s_lambda)
            s_r_feats_expanded = [f.repeat_interleave(block_size, dim=0) for f in s_r_feats]
            s_r_unsq = s_r.unsqueeze(1)

        B = 1
        af.denoising_step_list = torch.tensor(
            np.linspace(af.opt.num_train_timestep, 0, self.nfe - 1).tolist())
        af.initialize_kv_cache(batch_size=B, dtype=torch.float32, device=self.device)

        # Run the primer path TWICE during warmup: once to compile, once to
        # confirm cache-hit under production conditions (zero'd kv_cache).
        for primer_pass in (1, 2):
            if primer_pass == 2:
                # Zero kv_cache in place, matching the production reset-path.
                with torch.inference_mode():
                    for (sa, ca) in af.kv_cache:
                        sa["k"].zero_(); sa["v"].zero_()
                        ca["k"].zero_(); ca["v"].zero_()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                avatar_wa_t = torch.zeros(1, num_frames_for_clip, af.opt.dim_w, device=self.device)
                user_wa_t = torch.zeros_like(avatar_wa_t)
                uf = frames_t[:, :num_frames_for_clip]
                uf_list = [uf[0, i].unsqueeze(0) for i in range(num_frames_for_clip)]
                user_r_d_0 = af.encode_user_motion(uf_list)
                x_t = torch.randn(B, num_frames_for_clip, af.opt.dim_w, device=self.device)
                pc_c, pc_wr, pc_adaLN = af.prepare_cfg_condition(
                    avatar_wa_t, user_wa_t, user_r_d_0, r_s, seq_len=num_frames_for_clip, context_len=0)
                _pt0 = _t.perf_counter()
                for idx, ts_ in enumerate(af.denoising_step_list):
                    is_final = (idx == len(af.denoising_step_list) - 1)
                    x_t = af.solve_cfg(
                        B=B, index=idx, current_timestep=ts_, x_t=x_t,
                        precomputed_c=pc_c, precomputed_wr=pc_wr, precomputed_adaLN=pc_adaLN,
                        start_pos=0, context_len=0, use_kv_cache=False,
                        a_cfg_scale=self.a_cfg, u_cfg_scale=self.u_cfg,
                    )
                    if is_final:
                        _kv0 = _t.perf_counter()
                        af.update_kv_cache(final_latents=x_t, precomputed_c=pc_c,
                                           precomputed_wr=pc_wr, precomputed_adaLN=pc_adaLN,
                                           start_pos=0)
                        torch.cuda.synchronize()
                        log.info(f"warmup primer pass {primer_pass}: "
                                 f"denoise+kv={_t.perf_counter()-_pt0:.2f}s  "
                                 f"update_kv_cache={_t.perf_counter()-_kv0:.2f}s")
                for bstart in range(0, num_frames_for_clip, block_size):
                    _ = af._decode_one_block(x_t[:, bstart:bstart+block_size], s_r_unsq, s_r_feats_expanded)

        prev_tail = x_t
        # Warmup 4 steady-state blocks in each pass; 2 passes so dynamo fully settles
        for pass_idx in range(2):
            gen_t = num_frames_for_clip + pass_idx * block_size * 4
            start_pos_w = gen_t - 2
            next_block_ts = num_frames_for_clip / 25.0 + pass_idx * block_size * 4 / 25.0
            for _ in range(4):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    af_feats = torch.zeros(1, block_size + 2, af.opt.dim_w, device=self.device)
                    uf_feats = torch.zeros_like(af_feats)
                    uv_window = frames_t[:, :block_size + 2]
                    uv_list = [uv_window[0, i].unsqueeze(0) for i in range(block_size + 2)]
                    user_r_d_t = af.encode_user_motion(uv_list)
                    offset = prev_tail[:, -2:]
                    noise = torch.randn(B, block_size, af.opt.dim_w, device=self.device)
                    x_t = torch.cat([offset, noise], dim=1)
                    pc_c, pc_wr, pc_adaLN = af.prepare_cfg_condition(
                        af_feats, uf_feats, user_r_d_t, r_s, seq_len=block_size, context_len=2)
                    for idx, ts_ in enumerate(af.denoising_step_list):
                        is_final = (idx == len(af.denoising_step_list) - 1)
                        x_t = af.solve_cfg(
                            B=B, index=idx, current_timestep=ts_, x_t=x_t,
                            precomputed_c=pc_c, precomputed_wr=pc_wr, precomputed_adaLN=pc_adaLN,
                            start_pos=start_pos_w, context_len=2, use_kv_cache=True,
                            a_cfg_scale=self.a_cfg, u_cfg_scale=self.u_cfg,
                        )
                        if is_final:
                            af.update_kv_cache(final_latents=x_t, precomputed_c=pc_c,
                                               precomputed_wr=pc_wr, precomputed_adaLN=pc_adaLN,
                                               start_pos=start_pos_w)
                    _ = af._decode_one_block(x_t[:, -block_size:], s_r_unsq, s_r_feats_expanded)
                prev_tail = x_t
                gen_t += block_size
                start_pos_w = gen_t - 2
                next_block_ts += block_size / 25.0
            log.info(f"avatar worker: warmup pass {pass_idx+1} done ({_t.perf_counter()-t0:.1f}s)")

        # Pre-warm audio_encoder (wav2vec2) on this thread so its first real
        # call during a session doesn't initialize cuDNN/cuBLAS handles in the
        # middle of streaming — that lazy init was racing against the
        # flow_transformer and causing CUDA illegal-memory-access.
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            dummy_pcm = torch.zeros(1, 16000, dtype=torch.float32, device=self.device)   # 1s of silence
            _ = af.audio_encoder.inference(dummy_pcm, seq_len=25)
            torch.cuda.synchronize(self.device)

        # Pre-compute 5 s of wav2vec features over shaped pink noise. We cycle
        # through these in `_features_from_feat_ring` whenever we'd otherwise
        # read zero-padded tail (ring-write_ts is behind read window). Seeded,
        # so the same features every run.
        n_sil_sec = 5
        pcm_pink = _shaped_pink_noise(n_sil_sec * 16000, sample_rate=16000, level_dbfs=-50.0)
        n_sil_frames = n_sil_sec * 25
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pcm_t = torch.from_numpy(pcm_pink).unsqueeze(0).to(self.device)
            sil_feats = af.audio_encoder.inference(pcm_t, seq_len=n_sil_frames)
            torch.cuda.synchronize(self.device)
        self.silence_feats_np = sil_feats[0].float().cpu().numpy()
        log.info(f"avatar worker: cached {self.silence_feats_np.shape} pink-noise silence features")

        # Don't reset the ring — session.user_video_ring points to this ring too.
        # The real primer at session start will read whatever the webcam has written
        # by then; our warmup frames just sit in the past and roll off.
        log.info(f"avatar worker: warmup DONE in {_t.perf_counter()-t0:.1f}s")

    # ----- helpers -----

    def _frames_from_video_ring(self, t0: float, n_frames: int) -> torch.Tensor:
        """Pull n_frames at 25 fps starting at t0; returns torch tensor shape [1, n, 3, H, W]
        normalized to [-1, 1] (same as DataProcessor.transform)."""
        dt = 1.0 / 25
        frames = self.uv_ring.read_window(t0, t0 + n_frames * dt)
        if frames.shape[0] < n_frames:
            # Pad with last frame
            if frames.shape[0] == 0:
                frames = np.zeros((n_frames, self.uv_ring.H, self.uv_ring.W, 3), dtype=np.uint8)
            else:
                last = frames[-1:]
                pad = np.repeat(last, n_frames - frames.shape[0], axis=0)
                frames = np.concatenate([frames, pad], axis=0)
        frames = frames[:n_frames]
        # To [1, n, 3, H, W] f32 normalized to [-1, 1]
        x = torch.from_numpy(frames).to(self.device).float() / 255.0
        x = (x - 0.5) / 0.5
        x = x.permute(0, 3, 1, 2).unsqueeze(0)   # [1, n, 3, H, W]
        return x

    def _features_from_feat_ring(self, t0: float, n_frames: int) -> torch.Tensor:
        """Pull n_frames of 512-dim features from the feat ring; returns [1, n, 512].

        Frames past the ring's write_ts are the "silence" region — the ring
        zero-pads them, but zeros are out-of-distribution for wav2vec (its
        training clips always had a noise floor), which makes the downstream
        model hallucinate faint mouth motion. We substitute pre-computed
        features of shaped pink noise there instead, cycled to avoid any
        obvious periodicity. Keeps the avatar's idle mouth quiet.
        """
        dt = 1.0 / 25
        feats = self.af_ring.read_window(t0, t0 + n_frames * dt)
        if self.silence_feats_np is not None:
            write_ts = self.af_ring.write_ts
            silence_start = max(0, int(np.ceil((write_ts - t0) * 25)))
            silence_start = min(silence_start, n_frames)
            if silence_start < n_frames:
                n_sil = n_frames - silence_start
                L = self.silence_feats_np.shape[0]
                idx = (self._silence_phase + np.arange(n_sil)) % L
                feats[silence_start:] = self.silence_feats_np[idx]
                self._silence_phase = int((self._silence_phase + n_sil) % L)
        return torch.from_numpy(feats).to(self.device).unsqueeze(0)

    def _wav2vec_loop(self):
        """Dedicated thread: pulls items from tts_in_q and runs wav2vec on a
        dedicated CUDA stream so the kernels can overlap with the main worker
        thread's flow-transformer denoise/decode. Previously wav2vec ran on
        the main thread between blocks and could block for 300-500 ms, which
        starved the video queue for exactly that long.

        Side-stream placement lets the GPU scheduler interleave SM allocation
        between wav2vec and the flow transformer. Net effect: wav2vec is
        effectively invisible to the per-block wall budget.
        """
        try:
            dev_idx = int(self.device.split(":")[1]) if isinstance(self.device, str) and ":" in self.device else int(self.device)
            torch.cuda.set_device(dev_idx)
            stream = torch.cuda.Stream(device=self.device)
            # Pre-warm audio_encoder on THIS thread + stream so first real
            # call doesn't race lazy cuDNN/cuBLAS init against the main
            # worker's running kernels.
            with torch.cuda.stream(stream), torch.inference_mode(), \
                 torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                dummy = torch.zeros(1, 16000, dtype=torch.float32, device=self.device)
                _ = self.af.audio_encoder.inference(dummy, seq_len=25)
            torch.cuda.synchronize(self.device)
            log.info("[wav2vec-thread] ready")
        except Exception:
            log.exception("[wav2vec-thread] prewarm failed; thread exiting")
            return

        while not self._shutdown.is_set():
            epoch = self._tts_epoch
            try:
                phrase, pcm_16k = self.tts_in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                # Round UP to next whole video frame and zero-pad (keeps the
                # trailing consonant on short phrases).
                n_frames = max(1, (pcm_16k.size + 639) // 640)
                n_samples_needed = n_frames * 640
                if pcm_16k.size < n_samples_needed:
                    pad = np.zeros(n_samples_needed - pcm_16k.size, dtype=np.float32)
                    pcm_16k = np.concatenate([pcm_16k, pad])
                else:
                    pcm_16k = pcm_16k[:n_samples_needed]
                tt = time.perf_counter()
                with torch.cuda.stream(stream), torch.inference_mode(), \
                     torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    audio_tensor = torch.from_numpy(pcm_16k).unsqueeze(0).to(self.device)
                    feats = self.af.audio_encoder.inference(audio_tensor, seq_len=n_frames)
                # .cpu() blocks the wav2vec thread on its own stream — not the
                # main worker's default stream.
                feats_np = feats[0].float().cpu().numpy()
                ms = (time.perf_counter() - tt) * 1000
                self._schedule_phrase(epoch, phrase, feats_np, pcm_16k, ms, "wav2vec-thread")
            except Exception:
                log.exception("[wav2vec-thread] phrase processing failed")

    def _drain_tts_queue(self):
        """Legacy in-thread wav2vec drain. No longer called — the dedicated
        _wav2vec_loop thread owns this work now (overlaps with block gen on a
        side CUDA stream). Kept temporarily for reference."""
        af = self.af
        while True:
            try:
                phrase, pcm_16k = self.tts_in_q.get_nowait()
            except queue.Empty:
                return
            # Round UP to next whole video frame and zero-pad, instead of
            # truncating. Truncating dropped up to 40 ms of trailing audio
            # (often the final consonant of a short phrase).
            n_frames = max(1, (pcm_16k.size + 639) // 640)
            n_samples_needed = n_frames * 640
            if pcm_16k.size < n_samples_needed:
                pad = np.zeros(n_samples_needed - pcm_16k.size, dtype=np.float32)
                pcm_16k = np.concatenate([pcm_16k, pad])
            else:
                pcm_16k = pcm_16k[:n_samples_needed]
            tt = time.perf_counter()
            # Make sure any pending flow_transformer / motion_autoencoder work
            # is fully committed before we launch wav2vec kernels — we were
            # seeing CUDA illegal-memory-access on wav2vec's conv1d, almost
            # certainly from kernels that hadn't flushed and an allocator
            # reclaiming their memory.
            torch.cuda.synchronize(self.device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                audio_tensor = torch.from_numpy(pcm_16k).unsqueeze(0).to(self.device)
                feats = af.audio_encoder.inference(audio_tensor, seq_len=n_frames)
            feats_np = feats[0].float().cpu().numpy()
            ms = (time.perf_counter() - tt) * 1000
            # Schedule at the START of the next block the worker will
            # generate — NOT the current playhead. Anything queued between
            # playhead and gen_horizon was generated before this TTS was
            # written, so it has no mouth motion; writing there would play
            # audio against silent-mouth frames. gen_horizon guarantees the
            # avatar block that gets drawn for this audio actually sees it.
            horizon = getattr(self, "gen_horizon", self.playhead())
            target_ts = max(self.af_ring.write_ts, self.pcm_ring.write_ts, horizon)
            self.af_ring.write(feats_np, start_ts=target_ts)
            self.pcm_ring.write(pcm_16k, start_ts=target_ts)
            log.info(f"[worker-wav2vec] phrase={phrase[:40]!r} audio={pcm_16k.size/16000:.2f}s "
                     f"w2v={ms:.0f}ms scheduled@{target_ts:.2f}")

    # Reanchor crossfade length in frames (1 s @ 25 fps).
    REANCHOR_FADE_FRAMES = 25

    def _emit_pixel_block(self, pixels: torch.Tensor, block_start_ts: float):
        """pixels: [1, k, 3, H, W] float. Normalize to uint8 and push k frames to
        the queue. Each item is (logical_ts, frame).

        Blocking put throttles the worker to the consumer rate — the queue
        (cap ~1 s) acts as a tiny smoothing buffer, and the worker's logical
        timeline stays close to wall-clock. Drop-oldest would let the worker
        race ahead of real time, making the emitted frame ts (and thus the
        audio playhead) drift multiple seconds behind.

        If a reanchor crossfade is in progress, blend each frame with the
        captured pre-reanchor last frame on a linear alpha ramp (1 → 0)."""
        vid = pixels.squeeze(0).detach().clamp(-1, 1).cpu()
        vid = ((vid - vid.min()) / (vid.max() - vid.min() + 1e-8) * 255.0)
        vid = vid.permute(0, 2, 3, 1).contiguous().to(torch.uint8).numpy()   # [k, H, W, 3]
        dt = 1.0 / 25
        for i, frame in enumerate(vid):
            if (self._reanchor_fade_remaining > 0
                    and self._fade_anchor is not None
                    and self._fade_anchor.shape == frame.shape):
                alpha = self._reanchor_fade_remaining / self.REANCHOR_FADE_FRAMES
                frame = (alpha * self._fade_anchor.astype(np.float32)
                         + (1.0 - alpha) * frame.astype(np.float32)
                         ).clip(0, 255).astype(np.uint8)
                self._reanchor_fade_remaining -= 1
                if self._reanchor_fade_remaining == 0:
                    self._fade_anchor = None   # release
            self._last_emitted_frame = frame
            # Blocking put, but give up once the session ends: end_session()
            # drains the queue only once, and with nobody consuming a plain
            # put() of the rest of the block would hang the worker forever
            # (it never returned to idle for the next session).
            while True:
                try:
                    self.out_q.put((block_start_ts + i * dt, frame), timeout=0.1)
                    break
                except queue.Full:
                    if self._session_end.is_set() or self._shutdown.is_set():
                        return

    # ----- main loop -----

    def _run_loop(self):
        af = self.af
        num_frames_for_clip = af.num_frames_for_clip      # 50
        num_frames_per_block = af.num_frames_per_block    # 10
        block_size = num_frames_per_block

        # Encode the reference portrait once
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            s = self.avatar_ref.to(self.device)
            s_r, r_s_lambda, s_r_feats = af.encode_image_into_latent(s)
            r_s = af.motion_autoencoder.dec.direction(r_s_lambda)
            s_r_feats_expanded = [f.repeat_interleave(block_size, dim=0) for f in s_r_feats]
            s_r_unsq = s_r.unsqueeze(1)

        # KV cache: don't reallocate — warmup already created tensors and
        # torch.compile's dynamo cache is keyed on those tensor identities.
        # Reallocating would force a mid-stream recompile.  Just zero in place.
        B = 1
        af.denoising_step_list = torch.tensor(
            np.linspace(af.opt.num_train_timestep, 0, self.nfe - 1).tolist())
        if getattr(af, "kv_cache", None) is None:
            af.initialize_kv_cache(batch_size=B, dtype=torch.float32, device=self.device)
        else:
            with torch.inference_mode():
                for (sa, ca) in af.kv_cache:
                    sa["k"].zero_(); sa["v"].zero_()
                    ca["k"].zero_(); ca["v"].zero_()

        # No webcam dependency: with u_cfg_scale=0 the user_motion branch's
        # influence is zeroed out in CFG anyway. We feed zero user_motion
        # directly, skipping FaceCropper + webcam pipeline (which were also
        # a prime suspect for the cuda:1 illegal-memory race).
        t_now = self.now()
        prime_start = max(0.0, t_now - num_frames_for_clip / 25.0)
        log.info(f"avatar worker: priming with 50 frames starting at {prime_start:.2f}")

        import sys as _sys, time as _time
        _mt0 = _time.perf_counter()
        def _mark(tag):
            print(f"[primer +{(_time.perf_counter()-_mt0)*1000:7.1f}ms] {tag}", flush=True); _sys.stdout.flush()
        _mark("entering autocast")
        dim_w = af.opt.dim_w
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _mark("reading features from feat ring")
            avatar_wa_t = self._features_from_feat_ring(prime_start, num_frames_for_clip)
            user_wa_t = torch.zeros_like(avatar_wa_t)
            user_r_d_0 = torch.zeros(1, num_frames_for_clip, dim_w, device=self.device)

            x_t = torch.randn(B, num_frames_for_clip, af.opt.dim_w, device=self.device)
            _mark("prepare_cfg_condition")
            pc_c, pc_wr, pc_adaLN = af.prepare_cfg_condition(
                avatar_wa_t, user_wa_t, user_r_d_0, r_s, seq_len=num_frames_for_clip, context_len=0)
            _mark("prepare_cfg_condition done")
            for idx, ts in enumerate(af.denoising_step_list):
                is_final = (idx == len(af.denoising_step_list) - 1)
                x_t = af.solve_cfg(
                    B=B, index=idx, current_timestep=ts, x_t=x_t,
                    precomputed_c=pc_c, precomputed_wr=pc_wr, precomputed_adaLN=pc_adaLN,
                    start_pos=0, context_len=0, use_kv_cache=False,
                    a_cfg_scale=self.a_cfg, u_cfg_scale=self.u_cfg,
                )
                _mark(f"denoise step {idx} done")
                if is_final:
                    af.update_kv_cache(final_latents=x_t, precomputed_c=pc_c,
                                       precomputed_wr=pc_wr, precomputed_adaLN=pc_adaLN,
                                       start_pos=0)
                    _mark("update_kv_cache done")

            # Primer is only used to establish the KV cache state; its pixels
            # aren't shown to the user. Emitting them would push the queue to
            # capacity and leave the worker's logical timeline permanently
            # behind wall-clock (the primer's wall-time is >> its 2 s of
            # logical content). Decoding is skipped entirely — we only need
            # the kv_cache updates from the denoise loop above.

        _mark("primer complete (kv-only, not emitted)")
        prev_tail = x_t  # keep last 2 tokens as overlap into next block

        # Steady-state loop: generate one block per iteration, forever.
        # Start at current wall-clock so the logical timeline is aligned
        # with session time from the first user-visible frame.
        gen_t = num_frames_for_clip   # frame counter (logical, not clamped yet)
        next_block_ts = max(prime_start + num_frames_for_clip / 25.0, self.now())
        start_pos = gen_t - 2
        # Periodic KV reanchor: without it, the model keeps feeding its own
        # output back in and quality drifts after ~60 s. Every N steady
        # blocks we zero the KV cache and run a fresh primer (silently, no
        # emit) to re-establish clean context.
        REANCHOR_EVERY = 112   # ~45 s of video at 25 fps / 10 frames per block
        blocks_since_reanchor = 0
        # Side CUDA stream for decoding, so motion_autoencoder.dec of block
        # N overlaps with denoise of block N+1 instead of serializing.
        # Emission is one iter behind: at the top of iter N we flush the
        # pixels we decoded during iter N-1.
        decode_stream = torch.cuda.Stream(device=self.device)
        pending_pix: torch.Tensor | None = None
        pending_ts: float = 0.0

        while not (self._session_end.is_set() or self._shutdown.is_set()):
            t_loop0 = time.perf_counter()

            # Flush any pipelined decode from last iter before doing new work
            # so emission order is preserved across normal/reanchor paths.
            if pending_pix is not None:
                # Wait for decode_stream to finish producing pending_pix
                # before reading it on the default stream for CPU transfer.
                torch.cuda.current_stream(self.device).wait_stream(decode_stream)
                self._emit_pixel_block(pending_pix, pending_ts)
                pending_pix = None

            block_end_ts = next_block_ts + block_size / 25.0

            # Reanchor periodically, but only once both producer and
            # playback are idle:
            #   - tts_in_q empty: no phrase waiting for wav2vec.
            #   - audio buffer drained: the audio pacer plays
            #     (now - playback_delay); anything in pcm_ring past that
            #     cursor is still-pending playback. Wait for write_ts to
            #     catch up or we'll freeze the mouth mid-playout.
            # The 45 s timer is a MINIMUM; MAX_DEFER forces a reanchor if
            # we've been waiting too long (continuous speech etc.) so model
            # drift can't accumulate unbounded — the crossfade masks the
            # mid-speech seam if it happens.
            MAX_DEFER = 162   # ~65 s wall at RTF=1 (hard cap; fires mid-speech if hit)
            do_reanchor = False
            if blocks_since_reanchor >= REANCHOR_EVERY and self.tts_in_q.empty():
                pb_delay = max(0.15, min(0.6, self.block_time_ewma * 1.2))
                playback_pos = self.now() - pb_delay
                if self.af_ring.write_ts <= playback_pos + 0.05:
                    do_reanchor = True
            if blocks_since_reanchor >= MAX_DEFER:
                do_reanchor = True
                log.info(f"avatar worker: forcing reanchor (deferred {blocks_since_reanchor} blocks)")

            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if do_reanchor:
                    # In-stream KV reanchor: run the model like a primer on a
                    # 50-frame window ending at next_block_ts, emit only the
                    # last 10 frames, and update the KV cache with the full
                    # window. Replaces one steady block, same output rate,
                    # fresh context — no dropout.
                    seq0 = num_frames_for_clip
                    window_start_ts = next_block_ts + (block_size / 25.0) - seq0 / 25.0
                    af_feats_prim = self._features_from_feat_ring(window_start_ts, seq0)
                    uf_feats_prim = torch.zeros_like(af_feats_prim)
                    user_r_d_prim = torch.zeros(1, seq0, dim_w, device=self.device)
                    # Zero KV cache for a clean slate
                    for (sa, ca) in af.kv_cache:
                        sa["k"].zero_(); sa["v"].zero_()
                        ca["k"].zero_(); ca["v"].zero_()
                    x_t = torch.randn(B, seq0, af.opt.dim_w, device=self.device)
                    pc_c, pc_wr, pc_adaLN = af.prepare_cfg_condition(
                        af_feats_prim, uf_feats_prim, user_r_d_prim, r_s,
                        seq_len=seq0, context_len=0)
                    for idx, ts in enumerate(af.denoising_step_list):
                        is_final = (idx == len(af.denoising_step_list) - 1)
                        x_t = af.solve_cfg(
                            B=B, index=idx, current_timestep=ts, x_t=x_t,
                            precomputed_c=pc_c, precomputed_wr=pc_wr, precomputed_adaLN=pc_adaLN,
                            start_pos=0, context_len=0, use_kv_cache=False,
                            a_cfg_scale=self.a_cfg, u_cfg_scale=self.u_cfg,
                        )
                        if is_final:
                            af.update_kv_cache(final_latents=x_t, precomputed_c=pc_c,
                                               precomputed_wr=pc_wr, precomputed_adaLN=pc_adaLN,
                                               start_pos=0)
                    # Skip the reanchor's own emit: the primer's 400 ms of
                    # logical video takes ~1.3 s of wall-clock, so emitting
                    # it puts video ~900 ms behind audio playback. Instead,
                    # leave the video pacer holding the last pre-reanchor
                    # frame and arm a 1 s crossfade onto the next normal
                    # block's output.
                    # Snapshot the last emitted frame as the fade anchor —
                    # held constant through all REANCHOR_FADE_FRAMES frames
                    # of the blend so the crossfade is a clean 1→0 mix
                    # rather than a recursive one that produces ghosting.
                    if self._last_emitted_frame is not None:
                        self._fade_anchor = self._last_emitted_frame.copy()
                    self._reanchor_fade_remaining = self.REANCHOR_FADE_FRAMES
                    log.info(f"avatar worker: reanchored @ gen_t={gen_t} "
                             f"(skipped emit, fade armed {self.REANCHOR_FADE_FRAMES} frames)")
                    if self.on_reanchor is not None:
                        try: self.on_reanchor()
                        except Exception:
                            log.exception("on_reanchor callback failed")
                    blocks_since_reanchor = 0
                else:
                    # Normal steady block: 12 frames audio context + current.
                    af_feats = self._features_from_feat_ring(next_block_ts - 2/25, block_size + 2)
                    uf_feats = torch.zeros_like(af_feats)
                    user_r_d_t = torch.zeros(1, block_size + 2, dim_w, device=self.device)

                    offset = prev_tail[:, -2:]
                    noise = torch.randn(B, block_size, af.opt.dim_w, device=self.device)
                    x_t = torch.cat([offset, noise], dim=1)

                    pc_c, pc_wr, pc_adaLN = af.prepare_cfg_condition(
                        af_feats, uf_feats, user_r_d_t, r_s, seq_len=block_size, context_len=2)

                    for idx, ts in enumerate(af.denoising_step_list):
                        is_final = (idx == len(af.denoising_step_list) - 1)
                        # Fold KV-update into the final denoise step instead
                        # of an extra forward pass — saves ~1 forward per
                        # block. KV stores slightly-less-clean latents but
                        # streaming quality is unaffected in practice.
                        x_t = af.solve_cfg(
                            B=B, index=idx, current_timestep=ts, x_t=x_t,
                            precomputed_c=pc_c, precomputed_wr=pc_wr, precomputed_adaLN=pc_adaLN,
                            start_pos=start_pos, context_len=2, use_kv_cache=True,
                            a_cfg_scale=self.a_cfg, u_cfg_scale=self.u_cfg,
                            update_kv_cache=is_final,
                        )

                    latent_block = x_t[:, -block_size:].contiguous()
                    # Launch decode on side stream; it runs concurrently with
                    # NEXT iter's denoise on the default stream. Emission is
                    # deferred to the top of the next iter.
                    decode_stream.wait_stream(torch.cuda.current_stream(self.device))
                    with torch.cuda.stream(decode_stream):
                        pix = af._decode_one_block(latent_block, s_r_unsq, s_r_feats_expanded)
                    pending_pix = pix
                    pending_ts = next_block_ts
                    blocks_since_reanchor += 1

            prev_tail = x_t
            # After reanchor, restart start_pos from the primer-end position
            # (same sequence of start_pos values as session startup).
            if do_reanchor:
                gen_t = num_frames_for_clip
                start_pos = gen_t - 2
            else:
                start_pos = (gen_t + block_size - 2)
                gen_t += block_size
            # Keep logical timeline aligned with wall-clock. If we fell behind
            # (RTF spike / slow w2v / etc.) jump forward — those lost logical
            # seconds would otherwise permanently delay audio playback.
            next_block_ts = max(block_end_ts, self.now())
            # Publish the logical ts of the NEXT block to be generated so
            # TTS scheduling targets a block whose avatar frames haven't been
            # produced yet.
            self.gen_horizon = next_block_ts
            self._release_played()
            # Measure compute time BEFORE pacing sleep for the EWMA — the
            # delay knob wants to know how long the model actually takes,
            # not how long we spent idle keeping pace.
            compute_time = time.perf_counter() - t_loop0
            if compute_time > self.block_time_ewma:
                self.block_time_ewma = 0.5 * self.block_time_ewma + 0.5 * compute_time
            else:
                self.block_time_ewma = 0.92 * self.block_time_ewma + 0.08 * compute_time
            # If we got ahead of wall-clock, sleep so the queue doesn't buffer
            # "future" frames (would make audio playback trail further behind).
            lead = next_block_ts - self.now()
            if lead > 0.0:
                time.sleep(min(lead, 0.4))
            # Wav2vec now runs on a dedicated thread + side CUDA stream, so
            # no draining is needed here. See _wav2vec_loop.
            # VRAM is stable now (start_pos pre-slice fix + inference_mode).
            # We used to empty_cache() every 20 blocks, but that was racing
            # with async kernels and producing illegal-memory-access during
            # the next wav2vec call — disabling it now. Keep the periodic
            # VRAM log for visibility.
            block_n = gen_t // block_size
            if block_n % 20 == 0:
                log.info(f"avatar worker: @ gen_t={gen_t} "
                         f"vram={torch.cuda.memory_allocated()/1e9:.1f}GB")

            dt = (time.perf_counter() - t_loop0) * 1000
            if (gen_t // block_size) % 10 == 0:
                # 10 frames (one block) at 25 fps = 400 ms of video; wall-time
                # ms / 400 gives the "slowdown vs realtime" factor.
                rtf = dt / 400.0
                log.info(f"avatar worker: block @ gen_t={gen_t} wall={dt:.0f}ms "
                         f"rtf={rtf:.2f}x  q={self.out_q.qsize()} "
                         f"ewma={self.block_time_ewma*1000:.0f}ms")
