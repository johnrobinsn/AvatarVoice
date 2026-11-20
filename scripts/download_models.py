"""Download every model AvatarVoice needs, for this platform, with progress.

    python scripts/download_models.py            # skips anything already present
    python scripts/download_models.py --check    # report what's missing, download nothing

Sources (nothing is rehosted):
  - AvatarForcing checkpoints: the authors' Google Drive (as in avatarforcing/download_weights.sh)
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
    "flow_transformer.pth": "1lchvHiDka0bBQ29bdY6lHxCyNGbGzvS_",
}
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1rN52J2QXD8A-r2CZ8nDqFdwvZuPRmMsc"
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


def gdrive_fetch(name: str):
    import gdown
    dst = PRETRAINED / name
    tmp = dst.with_suffix(dst.suffix + ".part")
    out = gdown.download(id=GDRIVE[name], output=str(tmp), quiet=False, resume=True)
    if out is None or not tmp.exists() or tmp.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"Google Drive refused the download of {name} (it rate-limits popular files).\n"
            f"    Wait a while and run `avatarvoice download` again, or download {name} by hand from\n"
            f"    {DRIVE_FOLDER}\n    into {PRETRAINED}/")
    tmp.rename(dst)


def items() -> list[Item]:
    PRETRAINED.mkdir(parents=True, exist_ok=True)
    w2v = PRETRAINED / "wav2vec2-base-960h"
    out = [
        Item("AvatarForcing motion autoencoder (181 MB, Google Drive)",
             lambda: (PRETRAINED / "motion_autoencoder.pth").exists(),
             lambda: gdrive_fetch("motion_autoencoder.pth")),
        Item("AvatarForcing flow transformer (614 MB, Google Drive)",
             lambda: (PRETRAINED / "flow_transformer.pth").exists(),
             lambda: gdrive_fetch("flow_transformer.pth")),
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
