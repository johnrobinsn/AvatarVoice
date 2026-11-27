"""Download every model AvatarVoice needs, for this platform, with progress.

    python scripts/download_models.py            # skips anything already present
    python scripts/download_models.py --check    # report what's missing, download nothing

Sources:
  - AvatarForcing motion autoencoder: the authors' Google Drive (as in avatarforcing/download_weights.sh)
  - AvatarForcing flow transformer: an unmodified Hugging Face mirror of the authors' file (Drive
    rate-limits it); attribution and license on the model card
  - everything else: Hugging Face, plus the face-alignment and Phonon voice files their
    libraries fetch on first use.
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRETRAINED = ROOT / "avatarforcing" / "pretrained_dir"
MAC = sys.platform == "darwin" and platform.machine() == "arm64"

GDRIVE = {   # avatarforcing/download_weights.sh
    "motion_autoencoder.pth": "1a0ZqX9GMrkb3JFQgOBmUB0V-Bpwtv4Hp",
}
# Unmodified mirror of the authors' flow_transformer.pth (CC BY-NC 4.0; source and attribution on the model card)
FLOW_MIRROR = ("johnrobinsn/avatarforcing-flow-transformer", "b6e0ca9b03770c6fe12fcbd58569841219bc0ac3")
LLM_MAC = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
LLM_CUDA = "cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit"
PARAKEET = ("johnrobinsn/parakeet-tdt-0.6b-v2-mlx-q8", "aad991e02731e67298700cf1f2b86ddaa45dd18c")
PARAKEET_ONNX = ("nemo-parakeet-tdt-0.6b-v2", "istupakov/parakeet-tdt-0.6b-v2-onnx")   # onnx-asr id, HF repo

BOLD, DIM, GREEN, YELLOW, RED, RESET = (("\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m")
                                        if sys.stdout.isatty() else ("",) * 6)


def size_of(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024


class Item:
    """optional: fetched automatically on first use anyway, so --check doesn't require it."""
    def __init__(self, name, present, fetch, note="", optional=False):
        self.name, self.present, self.fetch, self.note, self.optional = name, present, fetch, note, optional


def hf_present(repo, revision=None, patterns=None) -> bool:
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(repo, revision=revision, allow_patterns=patterns, local_files_only=True)
        return True
    except Exception:
        return False


def hf_fetch(repo, revision=None, patterns=None, local_dir=None):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=revision, allow_patterns=patterns, local_dir=local_dir)


GDRIVE_RETRY_WAITS_S = (10, 30, 90)   # between attempts; each attempt resumes the .part file


def gdrive_fetch(name: str):
    """Download a checkpoint from the authors' Google Drive, retrying with backoff.

    Drive caps how often a popular file can be downloaded per day; once the cap is hit
    it serves an error page instead of the file until it resets (up to ~24 h). Retries
    cover transient failures; for the cap, the error says how to download by hand."""
    import gdown
    dst = PRETRAINED / name
    tmp = dst.with_suffix(dst.suffix + ".part")
    reason = ""
    for attempt in range(len(GDRIVE_RETRY_WAITS_S) + 1):
        if attempt:
            wait = GDRIVE_RETRY_WAITS_S[attempt - 1]
            print(f"    {YELLOW}Google Drive download failed ({reason}); retrying in {wait}s "
                  f"(attempt {attempt + 1} of {len(GDRIVE_RETRY_WAITS_S) + 1}){RESET}", flush=True)
            time.sleep(wait)
        try:
            out = gdown.download(id=GDRIVE[name], output=str(tmp), quiet=False, resume=True)
        except Exception as e:
            out, reason = None, " ".join(str(e).split())[:120] or type(e).__name__
        if out is not None and tmp.exists():
            with open(tmp, "rb") as f:
                if f.read(2) == b"PK":          # torch checkpoints are zip files
                    tmp.rename(dst)
                    return
            reason = "got a web page instead of the file"
            tmp.unlink()                        # an error page, not a partial checkpoint
        elif out is not None:
            reason = "no file was saved"
    limited = any(s in reason.lower() for s in ("too many users", "quota", "web page"))
    raise RuntimeError(
        (f"Google Drive is refusing {name} right now: it caps how often a popular file can be\n"
         f"    downloaded, and the cap resets within about 24 hours.\n" if limited else
         f"Couldn't download {name} from Google Drive ({reason}).\n") +
        f"    Either run `avatarvoice download` again later (it picks up where it left off), or\n"
        f"    download it in your browser (signing in to Google often gets past the limit):\n"
        f"      1. Open  https://drive.google.com/file/d/{GDRIVE[name]}/view\n"
        f"      2. Save it as  {dst}\n"
        f"      3. Run  avatarvoice download  to fetch the rest.")


def items() -> list[Item]:
    PRETRAINED.mkdir(parents=True, exist_ok=True)
    w2v = PRETRAINED / "wav2vec2-base-960h"
    out = [
        Item("AvatarForcing motion autoencoder (181 MB, Google Drive)",
             lambda: (PRETRAINED / "motion_autoencoder.pth").exists(),
             lambda: gdrive_fetch("motion_autoencoder.pth")),
        Item("AvatarForcing flow transformer (614 MB, Hugging Face mirror)",
             lambda: (PRETRAINED / "flow_transformer.pth").exists(),
             lambda: _flow_transformer()),
        Item("wav2vec2-base-960h audio encoder (378 MB)",
             lambda: (w2v / "model.safetensors").exists() and (w2v / "preprocessor_config.json").exists(),
             lambda: hf_fetch("facebook/wav2vec2-base-960h", local_dir=str(w2v),
                              patterns=["config.json", "preprocessor_config.json", "model.safetensors"])),
    ]
    if MAC:
        mlx = PRETRAINED / "flow_transformer_mlx.safetensors"
        out += [
            Item("flow transformer -> MLX conversion (595 MB, local)",
                 lambda: mlx.exists(),
                 lambda: subprocess.run([sys.executable, str(ROOT / "mlx_port" / "convert_weights.py"),
                                         "--src", str(PRETRAINED / "flow_transformer.pth"),
                                         "--dst", str(mlx)], check=True)),
            Item("Parakeet speech-to-text, MLX 8-bit (0.7 GB)",
                 lambda: hf_present(*PARAKEET), lambda: hf_fetch(*PARAKEET)),
            Item(f"LLM {LLM_MAC} (2.3 GB)",
                 lambda: hf_present(LLM_MAC), lambda: hf_fetch(LLM_MAC)),
        ]
    else:
        out += [
            Item("Parakeet speech-to-text, ONNX (onnx-asr)",
                 lambda: hf_present(PARAKEET_ONNX[1]), _parakeet_onnx,
                 note="(AF_STT=whisper downloads faster-whisper's model on first use instead)"),
            Item(f"LLM {LLM_CUDA} (2.7 GB)",
                 lambda: hf_present(LLM_CUDA), lambda: hf_fetch(LLM_CUDA)),
        ]
    out += [
        Item("Phonon text-to-speech voices", lambda: False, _phonon,
             note="(checks the voice cache; quick if already fetched)", optional=True),
        Item("face detector (face-alignment, ~100 MB)", _face_alignment_present, _face_alignment),
    ]
    return out


def _flow_transformer():
    from huggingface_hub import hf_hub_download
    hf_hub_download(FLOW_MIRROR[0], "flow_transformer.pth", revision=FLOW_MIRROR[1], local_dir=str(PRETRAINED))


def _parakeet_onnx():
    """Fetch exactly the files onnx-asr loads, via onnx-asr itself (CPU provider: no CUDA init)."""
    import onnx_asr
    onnx_asr.load_model(PARAKEET_ONNX[0], providers=["CPUExecutionProvider"])


def _phonon():
    import ptts
    tts = ptts.TTS(lang="en")
    for voice in ("Harper", "Miles"):
        for _ in tts.stream("Hello.", voice=voice):
            pass


def _face_alignment_present() -> bool:
    import torch
    ck = Path(torch.hub.get_dir()) / "checkpoints"
    return any(ck.glob("s3fd-*.pth")) and any(ck.glob("2DFAN4-*.pth.tar"))


def _face_alignment():
    import face_alignment
    face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report what's missing")
    args = ap.parse_args()

    todo = items()
    print(f"{BOLD}Models for {'macOS + Apple Silicon' if MAC else 'Linux + NVIDIA'}{RESET}")
    failed = []
    for n, it in enumerate(todo, 1):
        tag = f"[{n}/{len(todo)}]"
        if it.present():
            print(f"  {GREEN}✓{RESET} {tag} {it.name} {DIM}(already present){RESET}")
            continue
        if args.check:
            if it.optional:
                print(f"  {DIM}·{RESET} {tag} {it.name} {DIM}(fetched automatically when needed){RESET}")
            else:
                print(f"  {YELLOW}•{RESET} {tag} {it.name} {DIM}missing{RESET}")
                failed.append(it.name)
            continue
        print(f"  {BOLD}↓{RESET} {tag} {it.name} {DIM}{it.note}{RESET}", flush=True)
        t0 = time.perf_counter()
        try:
            it.fetch()
            print(f"  {GREEN}✓{RESET} {tag} {it.name} {DIM}({time.perf_counter() - t0:.0f}s){RESET}", flush=True)
        except Exception as e:
            failed.append(it.name)
            print(f"  {RED}✗{RESET} {tag} {it.name}\n    {RED}{e}{RESET}", flush=True)
    total = size_of(PRETRAINED)
    print(f"{DIM}avatar weights in {PRETRAINED}: {human(total)}{RESET}")
    if failed:
        print(f"{YELLOW}{len(failed)} item(s) {'missing' if args.check else 'failed'}: "
              f"run `avatarvoice download` to retry.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    main()
