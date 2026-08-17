"""Python client for the persistent Forge sim server (forge_ext SimServer).

Forge pays ~6s of startup (GUI toolkit under xvfb + full card database
load) on EVERY invocation; the evolver and the RL collector launch
hundreds of short batches, so startup dominated wall clock. SimServer
keeps one warm JVM per worker and streams jobs over stdin/stdout.

Usage:
    from sim_server import SimClient
    c = SimClient()                       # starts (or reuses) a worker
    out = c.run("deckA", "deckB", games=10, quiet=True)   # raw sim text
    c.close()

Opt the whole existing pipeline in without code changes by setting
FORGE_SIM_SERVER=1: improve_deck.run_match and verify_combo.run_verbose
route through a shared client automatically (see their fallbacks).
Workers are recycled every RECYCLE_JOBS jobs to bound any slow static
leak inside Forge.
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from improve_deck import FORGE_DIR, java_prefix  # noqa: E402

EXT_CLASSES = Path(__file__).resolve().parent / "forge_ext"
JAR = FORGE_DIR / "forge-gui-desktop-2.0.14-jar-with-dependencies.jar"
RECYCLE_JOBS = 200


class SimClient:
    def __init__(self, env_extra: dict | None = None):
        self.env_extra = env_extra or {}
        self.proc = None
        self.jobs = 0
        self.lock = threading.Lock()
        atexit.register(self.close)

    def _start(self):
        env = dict(os.environ)
        env.update(self.env_extra)
        self.proc = subprocess.Popen(
            java_prefix() + ["-Xmx3g",
             "-Dio.netty.tryReflectionSetAccessible=true",
             "-Dfile.encoding=UTF-8",
             "-cp", f"{EXT_CLASSES}:{JAR}", "forge.view.SimServer"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, cwd=FORGE_DIR, env=env)
        self.jobs = 0
        # wait for ##READY
        for line in self.proc.stdout:
            if line.startswith("##READY"):
                return
        raise RuntimeError("sim server died during startup")

    def run(self, deck_a: str, deck_b: str, games: int,
            quiet: bool = True, timeout_s: int | None = None) -> str:
        """Run one job; returns the raw sim output (parsers unchanged)."""
        with self.lock:
            if self.proc is None or self.proc.poll() is not None \
                    or self.jobs >= RECYCLE_JOBS:
                self.close()
                self._start()
            self.jobs += 1
            a = deck_a[:-4] if deck_a.endswith(".dck") else deck_a
            b = deck_b[:-4] if deck_b.endswith(".dck") else deck_b
            timer = None
            if timeout_s:
                timer = threading.Timer(timeout_s, self._kill)
                timer.start()
            try:
                self.proc.stdin.write(f"SIM {games} {1 if quiet else 0} "
                                      f"{a}\t{b}\n")
                self.proc.stdin.flush()
                out = []
                for line in self.proc.stdout:
                    if line.startswith("##JOB_DONE"):
                        break
                    out.append(line)
                else:
                    self.close()      # EOF mid-job: server died
                return "".join(out)
            except (BrokenPipeError, OSError):
                self.close()
                return ""
            finally:
                if timer:
                    timer.cancel()

    def _kill(self):
        if self.proc is not None:
            self.proc.kill()

    def close(self):
        if self.proc is not None:
            try:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass
            try:
                self.proc.kill()
            except OSError:
                pass
            self.proc = None


_SHARED: dict[str, SimClient] = {}
_SHARED_LOCK = threading.Lock()


def shared_client(key: str = "default", env_extra: dict | None = None):
    """Process-wide clients keyed by env config (thread-safe)."""
    with _SHARED_LOCK:
        if key not in _SHARED:
            _SHARED[key] = SimClient(env_extra)
        return _SHARED[key]


def enabled() -> bool:
    return os.environ.get("FORGE_SIM_SERVER") == "1"
