"""Local companion server for the browser evolver.

Forge is a Java rules engine - it can't run inside a browser tab or on
static hosting, so the evolver page talks to THIS server running on
your own machine, next to your Forge install.

Run:  python3 experiments/evolve_server.py [--port 8765]
Then open http://localhost:8765/  (or the deployed evolver.html, which
finds the local API automatically).

Requirements on this machine: Java 17+, Forge installed (see
experiments/FORGE-NOTES.md), xvfb-run on headless Linux (a desktop
session needs nothing extra).

API (all JSON, CORS-open so the hosted page can call localhost):
  GET  /api/status          current job snapshot + recent events
  POST /api/start           {deck, format, locks[], generations, ...}
  POST /api/stop            stop the running job
"""
from __future__ import annotations

import argparse
import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

JOB = {"evo": None, "thread": None, "events": deque(maxlen=400),
       "status": "idle"}
LOCK = threading.Lock()


def start_job(cfg: dict) -> dict:
    from evolve_core import Evolution
    with LOCK:
        if JOB["thread"] and JOB["thread"].is_alive():
            return {"error": "a job is already running"}
        JOB["events"].clear()

        def on_event(e):
            JOB["events"].append(e)

        try:
            evo = Evolution(
                deck_text=cfg["deck"],
                fmt=cfg.get("format", "legacy"),
                locks=cfg.get("locks", []),
                generations=int(cfg.get("generations", 8)),
                margin=float(cfg.get("margin", 0.03)),
                baseline_games=int(cfg.get("baseline_games", 24)),
                stage2_games=int(cfg.get("stage2_games", 12)),
                name=cfg.get("name", "browser"),
                on_event=on_event,
            )
        except ValueError as exc:
            return {"error": str(exc)}

        def worker():
            JOB["status"] = "running"
            evo.run()
            JOB["status"] = "done"

        JOB["evo"] = evo
        JOB["thread"] = threading.Thread(target=worker, daemon=True)
        JOB["thread"].start()
        JOB["status"] = "running"
        return {"ok": True}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes,
              ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods",
                         "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode())

    def do_OPTIONS(self):                                  # noqa: N802
        self._send(204, b"")

    def do_GET(self):                                      # noqa: N802
        if self.path.startswith("/api/status"):
            evo = JOB["evo"]
            self._json({
                "status": JOB["status"],
                "events": list(JOB["events"]),
                "snapshot": evo.snapshot() if evo else None,
            })
        elif self.path in ("/", "/evolver.html"):
            page = ROOT / "evolver.html"
            self._send(200, page.read_bytes(), "text/html; charset=utf-8")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):                                     # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        try:
            cfg = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad json"}, 400)
            return
        if self.path == "/api/start":
            self._json(start_job(cfg))
        elif self.path == "/api/stop":
            if JOB["evo"]:
                JOB["evo"].stop_flag = True
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, *args):                          # quiet
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"evolver companion listening on http://localhost:{args.port}/")
    srv.serve_forever()


if __name__ == "__main__":
    main()
