"""
Headless WebRTC client for a running voice-chat server — exercises the real
/offer + track path the browser uses, without a browser. Works against any
backend (it only talks WebRTC).

Each session: sends a spoken question as the "mic" track (Phonon TTS, after a
few seconds of silence) plus a blank "webcam" track, opens the `chat` data
channel, records the avatar's outbound video + audio, and checks the avatar
replied.

    # one session, saved to results/webrtc_client.mp4
    python -u voicechat/tests/test_webrtc_client.py --url http://localhost:8082

    # many back-to-back sessions: every one must get a reply (catches leaks that
    # only show up after N sessions, e.g. default-executor exhaustion at ~cpu+4)
    python -u voicechat/tests/test_webrtc_client.py --sessions 40
"""
from __future__ import annotations

import argparse
import asyncio
import fractions
import json
import sys
import time
from pathlib import Path

import av
import httpx
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCConfiguration
from aiortc.mediastreams import MediaStreamError

ROOT = Path(__file__).resolve().parent.parent.parent

QUESTION = "Hi! Can you tell me a quick fun fact about octopuses?"


class PCMTrack(MediaStreamTrack):
    """48 kHz mono 20 ms audio frames from a PCM buffer, then silence."""
    kind = "audio"

    def __init__(self, pcm_48k: np.ndarray):
        super().__init__()
        self.pcm = (np.clip(pcm_48k, -1, 1) * 32767).astype(np.int16)
        self.i = 0
        self.t_start = None

    async def recv(self):
        n = 960
        if self.t_start is None:
            self.t_start = time.perf_counter()
        await asyncio.sleep(max(0.0, self.t_start + self.i / 48000 - time.perf_counter()))
        chunk = self.pcm[self.i:self.i + n]
        if chunk.size < n:
            chunk = np.concatenate([chunk, np.zeros(n - chunk.size, np.int16)])
        f = av.AudioFrame.from_ndarray(chunk[None], format="s16", layout="mono")
        f.sample_rate, f.pts, f.time_base = 48000, self.i, fractions.Fraction(1, 48000)
        self.i += n
        return f


class BlankVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self.n = 0

    async def recv(self):
        await asyncio.sleep(1 / 25)
        f = av.VideoFrame.from_ndarray(np.zeros((240, 320, 3), np.uint8), format="rgb24")
        f.pts, f.time_base = self.n, fractions.Fraction(1, 25)
        self.n += 1
        return f


async def record(track, sink: list, stop: asyncio.Event):
    while not stop.is_set():
        try:
            frame = await asyncio.wait_for(track.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        except MediaStreamError:
            return
        sink.append((time.perf_counter(), frame))


def speak(question: str) -> tuple[np.ndarray, float]:
    """The question as 48 kHz mic audio after 4 s of silence; returns (pcm, question_end_s)."""
    import ptts
    from scipy.signal import resample_poly
    tts = ptts.TTS(lang="en")
    q24 = np.concatenate([np.asarray(c, np.float32).reshape(-1) for c in tts.stream(question, voice="Miles")])
    q48 = resample_poly(q24, 2, 1).astype(np.float32)
    # Peak 0.9 (a browser mic with AGC). At 0.5 this synthetic voice is borderline for the
    # VAD entry rule (15 consecutive 32 ms chunks above the RMS gate) and passes or fails
    # depending on chunk alignment.
    q48 = q48 / (np.abs(q48).max() + 1e-8) * 0.9
    return np.concatenate([np.zeros(48000 * 4, np.float32), q48]), 4.0 + len(q48) / 48000


async def session(url: str, mic: np.ndarray, seconds: float, rms_gate: float | None = None) -> dict:
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    chat = []
    ch = pc.createDataChannel("chat", ordered=True)
    ch.on("message", lambda m: chat.append((time.perf_counter(), m)))

    @ch.on("open")
    def on_open():
        # Like the browser's gear panel, optionally send a VAD energy gate at session start.
        if rms_gate is not None:
            ch.send(json.dumps({"type": "config", "key": "rms", "value": rms_gate}))
    pc.addTrack(PCMTrack(mic))
    pc.addTrack(BlankVideoTrack())
    video, audio, stop, tasks = [], [], asyncio.Event(), []

    @pc.on("track")
    def on_track(track):
        tasks.append(asyncio.create_task(record(track, video if track.kind == "video" else audio, stop)))

    await pc.setLocalDescription(await pc.createOffer())
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{url}/offer", json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    ans = r.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))
    connect_ms = (time.perf_counter() - t0) * 1000
    await asyncio.sleep(seconds)
    stop.set()
    await asyncio.gather(*tasks, return_exceptions=True)
    await pc.close()
    return dict(t0=t0, connect_ms=connect_ms, video=video, audio=audio, chat=chat,
                replies=[m for _, m in chat if '"who": "avatar"' in m])


def mono_f32(f) -> np.ndarray:   # aiortc delivers packed s16, possibly stereo
    return f.to_ndarray().reshape(-1, len(f.layout.channels))[:, 0].astype(np.float32) / 32768.0


def report(s: dict, q_end: float):
    v, a = s["video"], s["audio"]
    print(f"connected in {s['connect_ms']:.0f} ms; received {len(v)} video frames, {len(a)} audio frames")
    if len(v) > 1:
        vt = np.array([t for t, _ in v])
        print(f"video: {len(v) / (vt[-1] - vt[0]):.1f} fps, first frame +{vt[0] - s['t0']:.2f}s, "
              f"max inter-frame gap {np.diff(vt).max() * 1000:.0f} ms")
    if a:
        pcm = np.concatenate([mono_f32(f) for _, f in a])
        loud = np.nonzero(np.abs(pcm) > 0.02)[0]
        if loud.size:
            a0 = a[0][0] - s["t0"] + loud[0] / a[0][1].sample_rate
            print(f"avatar audio starts +{a0:.2f}s; question ended ~+{q_end:.2f}s -> "
                  f"response latency ~{a0 - q_end:.2f}s (incl. playback delay)")
    for t, m in s["chat"]:
        if '"type": "lag"' not in m:
            print(f"  chat +{t - s['t0']:5.2f}s {m[:100]}")


def write_mp4(path: Path, s: dict):
    frames = [f.to_ndarray(format="rgb24") for _, f in s["video"]]
    if not frames:
        return
    pcm = np.concatenate([mono_f32(f) for _, f in s["audio"]]) if s["audio"] else None
    sr = s["audio"][0][1].sample_rate if s["audio"] else 48000
    path.parent.mkdir(exist_ok=True)
    with av.open(str(path), "w") as out:
        vs = out.add_stream("libx264", rate=25)
        vs.width, vs.height, vs.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        if pcm is not None:
            a_s = out.add_stream("aac", rate=sr)
            a_s.layout = "mono"
        for f in frames:
            for pkt in vs.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                out.mux(pkt)
        for pkt in vs.encode():
            out.mux(pkt)
        if pcm is not None:
            for i in range(0, len(pcm), 1024):
                af = av.AudioFrame.from_ndarray(pcm[None, i:i + 1024], format="flt", layout="mono")
                af.sample_rate, af.pts, af.time_base = sr, i, fractions.Fraction(1, sr)
                for pkt in a_s.encode(af):
                    out.mux(pkt)
            for pkt in a_s.encode():
                out.mux(pkt)
    print(f"wrote {path}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8082")
    ap.add_argument("--seconds", type=float, default=25.0, help="recording length per session")
    ap.add_argument("--question", default=QUESTION)
    ap.add_argument("--sessions", type=int, default=1, help="run N back-to-back sessions")
    ap.add_argument("--gap", type=float, default=3.0, help="seconds between sessions")
    ap.add_argument("--rms-gate", type=float, default=-1.0,
                    help="VAD energy gate to send at session start, like the browser's gear panel "
                         "(default: don't send; use the server default)")
    args = ap.parse_args()
    mic, q_end = speak(args.question)
    gate = args.rms_gate if args.rms_gate >= 0 else None

    if args.sessions == 1:
        s = await session(args.url, mic, args.seconds, gate)
        report(s, q_end)
        write_mp4(ROOT / "results" / "webrtc_client.mp4", s)
        sys.exit(0 if s["replies"] else 1)

    seconds = min(args.seconds, q_end + 8.0)       # enough for the reply to start
    failed = []
    for n in range(1, args.sessions + 1):
        s = await session(args.url, mic, seconds, gate)
        ok = bool(s["replies"]) and len(s["video"]) > 0
        if not ok:
            failed.append(n)
        print(f"session {n:3d}/{args.sessions}: {'ok  ' if ok else 'FAIL'} "
              f"{len(s['replies'])} reply phrases, {len(s['video'])} video frames", flush=True)
        await asyncio.sleep(args.gap)
    print(f"\n{args.sessions - len(failed)}/{args.sessions} sessions got a reply"
          + (f"; failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
