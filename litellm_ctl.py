#!/usr/bin/env python3
"""
LiteLLM Cross-Platform Controller
=================================

Usage:
    python litellm_ctl.py start   - Start litellm proxy
    python litellm_ctl.py stop    - Stop litellm proxy
    python litellm_ctl.py status  - Check if running
    python litellm_ctl.py restart - Restart litellm proxy
"""

import os
import sys
import signal
import subprocess
import pathlib
import time
import platform

SCRIPT_DIR = pathlib.Path(__file__).parent
SRC_DIR = SCRIPT_DIR / "src"
PID_FILE = SRC_DIR / ".litellm.pid"
LOG_FILE = SRC_DIR / "litellm.log"
ENV_FILE = SRC_DIR / ".env"
CONFIG_FILE = SRC_DIR / "config.yaml"


def load_env():
    """Load environment variables from .env file."""
    if not ENV_FILE.exists():
        return {}
    env = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


def get_pid():
    """Read PID from file, return None if not found or invalid."""
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None


def is_process_running(pid):
    """Check if a process with given PID is running."""
    if pid is None:
        return False

    system = platform.system()

    if system == "Windows":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259

            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                try:
                    exit_code = ctypes.c_ulong()
                    if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                        return exit_code.value == STILL_ACTIVE
                finally:
                    kernel32.CloseHandle(handle)
        except Exception:
            pass
        return False
    else:
        # Unix: use os.kill with signal 0
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def kill_process_tree(pid):
    """Kill a process and all its children."""
    system = platform.system()

    if system == "Windows":
        # Use taskkill for reliable tree termination on Windows
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            timeout=30
        )
    else:
        # Unix: kill process group
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            time.sleep(0.5)
            # Force kill if still running
            if is_process_running(pid):
                os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            # Fallback: kill just the process
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                if is_process_running(pid):
                    os.kill(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass


def start():
    """Start litellm proxy."""
    # Check if already running
    pid = get_pid()
    if pid and is_process_running(pid):
        print(f"LiteLLM already running (PID {pid})")
        return 1

    # Clean up stale PID file
    if PID_FILE.exists():
        PID_FILE.unlink()

    # Check config exists
    if not CONFIG_FILE.exists():
        print(f"Error: Config file not found: {CONFIG_FILE}")
        return 1

    # Load environment
    env = os.environ.copy()
    env.update(load_env())

    # Prepare log file
    log_handle = open(LOG_FILE, "w", encoding="utf-8")

    system = platform.system()

    if system == "Windows":
        # Windows: CREATE_NO_WINDOW prevents any visible console;
        # CREATE_NEW_PROCESS_GROUP lets us kill the tree later.
        CREATE_NO_WINDOW = 0x08000000
        CREATE_NEW_PROCESS_GROUP = 0x00000200

        proc = subprocess.Popen(
            ["litellm", "--config", "config.yaml"],
            cwd=SRC_DIR,
            env=env,
            stdout=log_handle,
            stderr=log_handle,
            stdin=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
        )
    else:
        # Unix: use start_new_session for process group
        proc = subprocess.Popen(
            ["litellm", "--config", "config.yaml"],
            cwd=SRC_DIR,
            env=env,
            stdout=log_handle,
            stderr=log_handle,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    # Write PID file
    PID_FILE.write_text(str(proc.pid))
    log_handle.close()

    # Give it a moment to start
    time.sleep(1)

    if is_process_running(proc.pid):
        print(f"LiteLLM started (PID {proc.pid})")
        print(f"Log: {LOG_FILE}")
        return 0
    else:
        print("Error: LiteLLM failed to start. Check log file.")
        return 1


def stop():
    """Stop litellm proxy."""
    pid = get_pid()

    if not pid:
        print("LiteLLM is not running (no PID file)")
        return 0

    if not is_process_running(pid):
        print("LiteLLM is not running (stale PID file)")
        PID_FILE.unlink()
        return 0

    print(f"Stopping LiteLLM (PID {pid})...")
    kill_process_tree(pid)

    # Clean up PID file
    if PID_FILE.exists():
        PID_FILE.unlink()

    # Wait for process to terminate
    for _ in range(10):
        if not is_process_running(pid):
            print("LiteLLM stopped")
            return 0
        time.sleep(0.5)

    print("Warning: LiteLLM may not have stopped cleanly")
    return 1


def status():
    """Check if litellm proxy is running."""
    pid = get_pid()

    if not pid:
        print("LiteLLM is not running (no PID file)")
        return 1

    if is_process_running(pid):
        print(f"LiteLLM is running (PID {pid})")
        return 0
    else:
        print("LiteLLM is not running (stale PID file)")
        return 1


def restart():
    """Restart litellm proxy."""
    stop()
    time.sleep(1)
    return start()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    command = sys.argv[1].lower()

    commands = {
        "start": start,
        "stop": stop,
        "status": status,
        "restart": restart,
    }

    if command not in commands:
        print(f"Unknown command: {command}")
        print("Valid commands: start, stop, status, restart")
        return 1

    return commands[command]()


if __name__ == "__main__":
    sys.exit(main())