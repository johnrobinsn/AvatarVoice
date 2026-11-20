"""`avatarvoice` command: run the LLM + voice-chat servers with live status.

    avatarvoice run [--port 8082] [--no-browser]   start everything, open the browser; Ctrl-C stops
    avatarvoice download                           fetch / verify models
    avatarvoice status | stop | logs | update | uninstall
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / ".run"                                   # pid files + llm log
MAC = sys.platform == "darwin" and platform.machine() == "arm64"
LLM_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit" if MAC else "cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit"

TTY = sys.stdout.isatty()
BOLD, DIM, GREEN, YELLOW, RED, CYAN, RESET = (("\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m",
                                               "\033[36m", "\033[0m") if TTY else ("",) * 7)


def ok(msg):
    print(f"{GREEN}✓{RESET} {msg}", flush=True)


def warn(msg):
    print(f"{YELLOW}!{RESET} {msg}", flush=True)


def fail(msg, code=1):
    print(f"{RED}✗ {msg}{RESET}", flush=True)
    sys.exit(code)


class Spinner:
    """`with Spinner("Starting LLM"): ...` — animated line with elapsed seconds."""
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label):
        self.label, self.detail, self._stop = label, "", threading.Event()

    def __enter__(self):
        self.t0 = time.perf_counter()
        if TTY:
            self._th = threading.Thread(target=self._spin, daemon=True)
            self._th.start()
        else:
            print(f"… {self.label}", flush=True)
        return self

    def _spin(self):
        i = 0
        while not self._stop.wait(0.1):
            el = time.perf_counter() - self.t0
            sys.stdout.write(f"\r{CYAN}{self.FRAMES[i % 10]}{RESET} {self.label} {DIM}{el:4.0f}s {self.detail}{RESET}\033[K")
            sys.stdout.flush()
            i += 1

    def __exit__(self, *exc):
        self._stop.set()
        if TTY:
            self._th.join()
            sys.stdout.write("\r\033[K")
        return False

    @property
    def elapsed(self):
        return time.perf_counter() - self.t0


def http_ok(url, timeout=2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def tail(path: Path, n=15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except FileNotFoundError:
        return "(no log)"


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def read_pid(name) -> int | None:
    try:
        pid = int((RUN / f"{name}.pid").read_text())
        return pid if pid_alive(pid) else None
    except (FileNotFoundError, ValueError):
        return None


def terminate(proc_or_pid, name, timeout=15):
    pid = proc_or_pid.pid if hasattr(proc_or_pid, "pid") else proc_or_pid
    if not pid or not pid_alive(pid):
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    t0 = time.time()
    while time.time() - t0 < timeout and pid_alive(pid):
        if hasattr(proc_or_pid, "poll") and proc_or_pid.poll() is not None:
            break
        time.sleep(0.2)
    if pid_alive(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    (RUN / f"{name}.pid").unlink(missing_ok=True)


def start(name, cmd, log: Path, env=None):
    RUN.mkdir(exist_ok=True)
    log.parent.mkdir(exist_ok=True)
    f = open(log, "w")
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})},
                         start_new_session=True)
    (RUN / f"{name}.pid").write_text(str(p.pid))
    return p


def wait_ready(proc, url, label, log: Path, timeout, hint=""):
    with Spinner(label) as sp:
        while sp.elapsed < timeout:
            if proc.poll() is not None:
                break
            if http_ok(url):
                el = sp.elapsed
                break
            time.sleep(0.5)
        else:
            el = None
    if proc.poll() is not None or el is None:
        what = "exited" if proc.poll() is not None else f"didn't become ready in {timeout}s"
        print(f"{RED}✗ {label} {what}. Last lines of {log}:{RESET}\n{DIM}{tail(log)}{RESET}")
        if hint:
            print(f"{YELLOW}{hint}{RESET}")
        return False
    ok(f"{label} {DIM}({el:.0f}s){RESET}")
    return True


# ---------------- commands ----------------

def cmd_download(args):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "download_models.py")] + (["--check"] if args.check else []),
                       cwd=ROOT)
    sys.exit(r.returncode)


def cmd_run(args):
    llm_url = f"http://127.0.0.1:{args.llm_port}/v1/models"
    app_url = f"http://localhost:{args.port}"
    print(f"{BOLD}AvatarVoice{RESET} {DIM}({'macOS + Apple Silicon' if MAC else 'Linux + NVIDIA'}){RESET}")

    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "download_models.py"), "--check"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        warn("some models are missing — downloading them first")
        if subprocess.run([sys.executable, str(ROOT / "scripts" / "download_models.py")], cwd=ROOT).returncode:
            fail("model download failed; run `avatarvoice download` to retry")

    if http_ok(f"{app_url}/avatars"):
        ok(f"AvatarVoice is already running at {app_url}")
        _open(app_url, args)
        return

    procs = []

    def shutdown(*_):
        print(f"\n{DIM}stopping…{RESET}", flush=True)
        for name, p in reversed(procs):
            terminate(p, name)
        ok("stopped")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    llm_log = ROOT / "logs" / "llm.log"
    if http_ok(llm_url):
        ok(f"LLM server already running on port {args.llm_port} {DIM}(reusing it){RESET}")
    else:
        if MAC:
            cmd = [str(ROOT / ".venv-llm" / "bin" / "mlx_lm.server"), "--model", LLM_MODEL,
                   "--port", str(args.llm_port), "--prompt-cache-size", "1"]
        else:
            cmd = [str(ROOT / ".venv-llm" / "bin" / "vllm"), "serve", LLM_MODEL, "--port", str(args.llm_port),
                   "--max-model-len", "4096", "--gpu-memory-utilization", str(args.llm_gpu_fraction)]
        procs.append(("llm", start("llm", cmd, llm_log)))
        if not wait_ready(procs[-1][1], llm_url, f"LLM ({LLM_MODEL})", llm_log, timeout=600,
                          hint="Check the log above; on Linux, vLLM needs ~13 GB of free GPU memory."):
            shutdown()

    srv_log = ROOT / "logs" / "server.log"
    procs.append(("server", start("server", [sys.executable, "-u", "voicechat/server.py", "--port", str(args.port)],
                                  ROOT / "logs" / "server-stdout.log",
                                  env={"AF_LLM_MODEL": LLM_MODEL,
                                       "AF_LLM_BASE_URL": f"http://localhost:{args.llm_port}/v1"})))
    first_run = not (ROOT / ".inductor_cache").exists()
    label = "Avatar + speech models (first start compiles kernels, ~1 min)" if first_run else "Avatar + speech models"
    if not wait_ready(procs[-1][1], f"{app_url}/avatars", label, srv_log, timeout=900):
        print(f"{DIM}{tail(ROOT / 'logs' / 'server-stdout.log', 8)}{RESET}")
        shutdown()

    print(f"\n{BOLD}{GREEN}Ready:{RESET} {BOLD}{app_url}{RESET}  {DIM}— pick an avatar, click Start session, and talk."
          f"{RESET}\n{DIM}Headphones recommended. Ctrl-C to stop.{RESET}\n")
    _open(app_url, args)
    while True:
        for name, p in procs:
            if p.poll() is not None:
                warn(f"{name} exited (code {p.returncode}); last lines:\n{DIM}{tail(srv_log if name == 'server' else llm_log)}{RESET}")
                shutdown()
        time.sleep(1)


def _open(url, args):
    if args.no_browser:
        return
    if MAC:
        subprocess.run(["open", url])
    elif os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        subprocess.run(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print(f"{DIM}No display here. Browsers only allow the microphone on localhost or https, so from another\n"
              f"machine use an SSH tunnel:  ssh -L {args.port}:localhost:{args.port} <this-host>  and open {url}{RESET}")


def cmd_status(args):
    for name, url in (("LLM", f"http://127.0.0.1:{args.llm_port}/v1/models"), ("AvatarVoice", f"http://localhost:{args.port}/avatars")):
        print(f"{GREEN + '●' if http_ok(url) else RED + '○'}{RESET} {name:<12s} {url.rsplit('/', 2)[0] if name == 'LLM' else url.rsplit('/', 1)[0]}")


def cmd_stop(args):
    stopped = False
    for name in ("server", "llm"):
        pid = read_pid(name)
        if pid:
            terminate(pid, name)
            ok(f"stopped {name} (pid {pid})")
            stopped = True
    if not stopped:
        print("nothing to stop (only processes started by `avatarvoice run` are tracked)")


def cmd_logs(args):
    for name in ("server.log", "server-stdout.log", "llm.log"):
        print(f"{BOLD}== logs/{name}{RESET}\n{tail(ROOT / 'logs' / name, args.lines)}\n")


def cmd_update(args):
    sys.exit(subprocess.run(["bash", str(ROOT / "install.sh"), "--update"]).returncode)


def cmd_uninstall(args):
    shim = Path.home() / ".local" / "bin" / "avatarvoice"
    print(f"This removes {ROOT} (code, Python environments, avatar weights) and {shim}.\n"
          f"{DIM}Hugging Face downloads (LLM, speech models) stay in your Hugging Face cache.{RESET}")
    if not args.yes and input("Type 'yes' to continue: ").strip() != "yes":
        print("cancelled")
        return
    cmd_stop(args)
    shim.unlink(missing_ok=True)
    shutil.rmtree(ROOT)
    ok("uninstalled")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="avatarvoice", description="Real-time voice chat with a lip-synced avatar")
    ap.add_argument("--port", type=int, default=int(os.environ.get("AF_PORT", 8082)))
    ap.add_argument("--llm-port", type=int, default=8000)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="start the LLM + voice servers and open the browser")
    r.add_argument("--no-browser", action="store_true")
    r.add_argument("--llm-gpu-fraction", type=float, default=0.15, help="(Linux) vLLM --gpu-memory-utilization")
    d = sub.add_parser("download", help="download / verify models")
    d.add_argument("--check", action="store_true", help="only list what's missing")
    sub.add_parser("status", help="show what's running")
    sub.add_parser("stop", help="stop servers started by `avatarvoice run`")
    lg = sub.add_parser("logs", help="show recent log lines")
    lg.add_argument("-n", "--lines", type=int, default=30)
    sub.add_parser("update", help="pull the latest code and update dependencies")
    u = sub.add_parser("uninstall", help="remove AvatarVoice")
    u.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)
    {"run": cmd_run, "download": cmd_download, "status": cmd_status, "stop": cmd_stop, "logs": cmd_logs,
     "update": cmd_update, "uninstall": cmd_uninstall}[args.cmd](args)


if __name__ == "__main__":
    main()
