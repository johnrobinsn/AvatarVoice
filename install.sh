#!/usr/bin/env bash
# AvatarVoice installer — macOS (Apple Silicon) and Linux (NVIDIA GPU).
#
#   curl -LsSf https://raw.githubusercontent.com/johnrobinsn/AvatarVoice/main/install.sh | bash
#
# Installs into ~/.avatarvoice (override with AVATARVOICE_HOME), adds an `avatarvoice`
# command to ~/.local/bin, and downloads the models. Nothing is installed system-wide.
# Re-running it updates an existing install.
#
# Options (environment): AVATARVOICE_HOME, AVATARVOICE_REF (git branch/tag, default main),
#                        AVATARVOICE_REPO (git URL).  Flags: --update (pull + reinstall).
set -euo pipefail

REPO="${AVATARVOICE_REPO:-https://github.com/johnrobinsn/AvatarVoice.git}"
REF="${AVATARVOICE_REF:-main}"
HOME_DIR="${AVATARVOICE_HOME:-$HOME/.avatarvoice}"
APP="$HOME_DIR/AvatarVoice"
LOG="$HOME_DIR/install.log"
BIN="$HOME/.local/bin"
ORIG_PATH="$PATH"
for a in "$@"; do case "$a" in --update) ;; *) echo "unknown option: $a" >&2; exit 2;; esac; done

# ---------------- output helpers ----------------
if [ -t 1 ]; then
  B=$'\033[1m'; D=$'\033[2m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; C=$'\033[36m'; X=$'\033[0m'
else B=''; D=''; G=''; Y=''; R=''; C=''; X=''; fi
case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in *UTF-8*|*utf8*) FRAMES=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏); TICK="✓"; CROSS="✗";;
  *) FRAMES=('|' '/' '-' '\'); TICK="ok"; CROSS="x";; esac
STEP=0; NSTEPS=7; T0=$SECONDS

step() { STEP=$((STEP + 1)); printf "\n${B}[%d/%d] %s${X}\n" "$STEP" "$NSTEPS" "$1"; }
ok()   { printf "  ${G}%s${X} %s\n" "$TICK" "$1"; }
note() { printf "  ${D}%s${X}\n" "$1"; }
warn() { printf "  ${Y}!${X} %s\n" "$1"; }
die()  {
  printf "\n  ${R}%s %s${X}\n" "$CROSS" "$1"
  [ -n "${2:-}" ] && printf "  ${Y}%s${X}\n" "$2"
  if [ -f "$LOG" ]; then printf "\n  ${D}Last lines of %s:${X}\n" "$LOG"; tail -n 20 "$LOG" | sed 's/^/    /'; fi
  printf "\n  ${D}Full log: %s — re-run the installer to retry (finished steps are skipped).${X}\n" "$LOG"
  exit 1
}

# run_step LABEL WATCH_PATH CMD... — runs CMD with output to the log, showing a spinner,
# elapsed time and (if WATCH_PATH is set) how much has been written there so far.
run_step() {
  local label=$1 watch=$2; shift 2
  local start=$SECONDS i=0 size="" last=-1
  echo "### $label: $*" >>"$LOG"
  ("$@") >>"$LOG" 2>&1 &
  local pid=$!
  if [ -t 1 ]; then
    while kill -0 "$pid" 2>/dev/null; do
      local el=$((SECONDS - start))
      if [ -n "$watch" ] && [ "$el" != "$last" ] && [ $((el % 2)) -eq 0 ]; then
        size=$(du -sh "$watch" 2>/dev/null | cut -f1 | tr -d ' ' || true); last=$el
      fi
      printf "\r  ${C}%s${X} %s ${D}%ds%s${X}\033[K" "${FRAMES[$((i % ${#FRAMES[@]}))]}" "$label" "$el" "${size:+, $size}"
      i=$((i + 1)); sleep 0.15
    done
    printf "\r\033[K"
  fi
  if wait "$pid"; then ok "$label ${D}($((SECONDS - start))s)${X}"; else return 1; fi
}

# ---------------- 1. system ----------------
mkdir -p "$HOME_DIR"; : >"$LOG"
printf "${B}AvatarVoice installer${X} ${D}— real-time voice chat with a lip-synced avatar${X}\n"
printf "${D}Non-commercial use only (CC BY-NC 4.0, like the AvatarForcing model it uses).${X}\n"
step "Checking this machine"
OS=$(uname -s); ARCH=$(uname -m)
if [ "$OS" = Darwin ]; then
  [ "$ARCH" = arm64 ] || die "This Mac has an Intel CPU." "AvatarVoice on macOS needs Apple Silicon (M1 or newer)."
  PLATFORM=mac; PY=3.12; NEED_GB=12
  MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
  MACOS=$(sw_vers -productVersion)
  ok "macOS $MACOS on $(sysctl -n machdep.cpu.brand_string), ${MEM_GB} GB memory"
  [ "${MACOS%%.*}" -ge 14 ] || warn "macOS 14 or newer is recommended (MLX)."
  [ "$MEM_GB" -ge 24 ] || warn "Tested on 32 GB; with ${MEM_GB} GB it may run short of memory (the full stack uses ~12 GB)."
  if ! xcode-select -p >/dev/null 2>&1; then
    die "git is needed (Apple's Command Line Tools)." "Run:  xcode-select --install   then re-run this installer."
  fi
elif [ "$OS" = Linux ]; then
  [ "$ARCH" = x86_64 ] || die "Linux on $ARCH isn't supported." "AvatarVoice on Linux needs x86_64 with an NVIDIA GPU."
  PLATFORM=cuda; PY=3.11; NEED_GB=30
  command -v nvidia-smi >/dev/null 2>&1 || die "No NVIDIA driver found (nvidia-smi)." "Install the NVIDIA driver (CUDA 12.8-capable) first."
  GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) \
    || die "nvidia-smi failed — is the NVIDIA driver working?"
  GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
  MEM_GB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1048576 ))
  ok "Linux, GPU: $GPU, ${MEM_GB} GB RAM"
  [ "${GPU_MB:-0}" -ge 30000 ] || warn "Tested on 32 GB+ GPUs; with $((GPU_MB / 1024)) GB the avatar and LLM may not both fit."
else
  die "Unsupported OS: $OS" "AvatarVoice runs on macOS (Apple Silicon) and Linux (NVIDIA)."
fi
command -v git >/dev/null 2>&1 || die "git is not installed." "Install git and re-run."
command -v curl >/dev/null 2>&1 || die "curl is not installed."
FREE_GB=$(( $(df -Pk "$HOME_DIR" | awk 'NR==2 {print $4}') / 1048576 ))
[ "$FREE_GB" -ge "$NEED_GB" ] || die "Only ${FREE_GB} GB free in $HOME_DIR; about ${NEED_GB} GB is needed." \
  "Free some space, or install elsewhere with AVATARVOICE_HOME=/path/with/space."
ok "${FREE_GB} GB free in $HOME_DIR ${D}(needs ~${NEED_GB} GB)${X}"

# ---------------- 2. uv ----------------
step "Python tooling (uv)"
export PATH="$BIN:$HOME/.cargo/bin:$PATH"
if command -v uv >/dev/null 2>&1; then
  ok "uv $(uv --version | awk '{print $2}') already installed"
else
  run_step "Installing uv (fast Python package manager)" "" sh -c "curl -LsSf https://astral.sh/uv/install.sh | sh" \
    || die "Couldn't install uv." "See https://docs.astral.sh/uv/ to install it by hand, then re-run."
  command -v uv >/dev/null 2>&1 || die "uv installed but isn't on PATH." "Open a new terminal and re-run."
fi

# ---------------- 3. code ----------------
step "Getting AvatarVoice"
if [ -d "$APP/.git" ]; then
  run_step "Updating code ($REF)" "" sh -c "cd '$APP' && git fetch --quiet origin '$REF' && git checkout --quiet '$REF' && git pull --quiet --ff-only origin '$REF' && git submodule update --init --recursive --quiet" \
    || die "Couldn't update $APP." "If you changed files there, commit or discard them, then re-run."
else
  run_step "Cloning $REPO ($REF)" "$APP" git clone --quiet --branch "$REF" --recurse-submodules "$REPO" "$APP" \
    || die "Couldn't clone $REPO." "Check your internet connection and re-run."
fi
note "$(cd "$APP" && git log -1 --format='%h %s' | cut -c1-70)"

# ---------------- 4. Python environment ----------------
step "Python environment (Python $PY, ~2–5 GB, a few minutes the first time)"
cd "$APP"
[ -x .venv/bin/python ] || run_step "Creating environment" "" uv venv --quiet --python "$PY" .venv \
  || die "Couldn't create a Python $PY environment."
if [ "$PLATFORM" = mac ]; then
  run_step "Installing PyTorch, MLX and voice-chat packages" .venv \
    uv pip install --python .venv/bin/python -r mlx_port/requirements-mac.txt \
    || die "Package install failed."
  # AvatarForcing's wav2vec2 wrapper needs transformers 4.30.2, whose tokenizers pin has no
  # Python 3.12 wheels; wav2vec2 doesn't use a tokenizer, so install it without deps.
  run_step "Installing transformers 4.30.2" .venv \
    uv pip install --python .venv/bin/python --no-deps transformers==4.30.2 || die "Package install failed."
else
  run_step "Installing PyTorch (CUDA 12.8)" .venv \
    uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
    || die "PyTorch install failed." "Check the NVIDIA driver supports CUDA 12.8 (nvidia-smi shows the max CUDA version)."
  run_step "Installing voice-chat packages" .venv \
    uv pip install --python .venv/bin/python -r requirements-cuda.txt || die "Package install failed."
fi
run_step "Checking the environment" "" .venv/bin/python -c "import torch, aiortc, voicechat.rings" \
  || die "The environment doesn't import cleanly."

# ---------------- 5. LLM environment ----------------
if [ "$PLATFORM" = mac ]; then
  step "Language model server (mlx-lm)"
  [ -x .venv-llm/bin/python ] || run_step "Creating environment" "" uv venv --quiet --python 3.12 .venv-llm || die "Couldn't create the LLM environment."
  run_step "Installing mlx-lm" .venv-llm uv pip install --python .venv-llm/bin/python mlx-lm || die "mlx-lm install failed."
else
  step "Language model server (vLLM, ~8 GB)"
  [ -x .venv-llm/bin/python ] || run_step "Creating environment" "" uv venv --quiet --python 3.12 .venv-llm || die "Couldn't create the LLM environment."
  run_step "Installing vLLM" .venv-llm uv pip install --python .venv-llm/bin/python vllm || die "vLLM install failed."
fi

# ---------------- 6. models ----------------
step "Models (~4–6 GB; downloads are resumable)"
.venv/bin/python scripts/download_models.py 2>&1 | tee -a "$LOG" | sed 's/^/  /' || true
.venv/bin/python scripts/download_models.py --check >>"$LOG" 2>&1 \
  || die "Some models didn't download." "Run  avatarvoice download  to retry (Google Drive sometimes rate-limits)."

# ---------------- 7. command ----------------
step "Installing the avatarvoice command"
mkdir -p "$BIN"
cat >"$BIN/avatarvoice" <<EOF
#!/usr/bin/env bash
# AvatarVoice launcher (installed by install.sh)
exec "$APP/.venv/bin/python" "$APP/voicechat/launcher.py" "\$@"
EOF
chmod +x "$BIN/avatarvoice"
ok "$BIN/avatarvoice"
printf "\n${G}${B}AvatarVoice is installed${X} ${D}(%dm %02ds)${X}\n\n" $(((SECONDS - T0) / 60)) $(((SECONDS - T0) % 60))
printf "  Start it:   ${B}avatarvoice run${X}      ${D}(opens your browser; Ctrl-C to stop)${X}\n"
printf "  Other:      ${D}avatarvoice status | stop | logs | download | update | uninstall${X}\n"
case ":$ORIG_PATH:" in
  *":$BIN:"*) ;;
  *) printf "\n  ${Y}!${X} Open a new terminal first (or run  ${B}export PATH=\"\$HOME/.local/bin:\$PATH\"${X}),\n"
     printf "    and add that line to your shell profile if 'avatarvoice' still isn't found.\n";;
esac
printf "\n  ${D}Headphones recommended. First start compiles some GPU kernels (~1 min).${X}\n"
