"""Rung 2: external-policy pilot bridge - Python side + A/B experiment.

The Java side (forge_ext/) shadows LobbyPlayerAi on the classpath and
routes every "what does the AI want to cast?" decision through a
localhost socket when FORGE_EXT_POLICY=<port> is set. This module is
the other end: a tiny line-JSON policy server plus a driver that
measures the policy's effect.

The demo policy encodes the Tainted Aether deck's actual plan, which
the built-in AI cannot represent:
  - never veto Tainted Aether itself (the lock wants to land ASAP)
  - once Tainted Aether is on OUR battlefield, veto casting our own
    creatures - except the Hunted ones, whose ETB token gift turns
    into forced sacrifices for the opponent under the lock

Usage:
  python3 experiments/pilot_bridge.py --serve 8877          # just the server
  python3 experiments/pilot_bridge.py --experiment          # full A/B
"""
from __future__ import annotations

import argparse
import json
import socket
import socketserver
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from improve_deck import FORGE_DIR, ci95  # noqa: E402
from evolve_core import parse_games, aggregate_stats  # noqa: E402

EXT_CLASSES = Path(__file__).resolve().parent / "forge_ext"
JAR = FORGE_DIR / "forge-gui-desktop-2.0.14-jar-with-dependencies.jar"

STATS = {"decisions": 0, "vetoes": 0}


def tainted_policy(state: dict) -> list[str]:
    """Return card names to veto for this decision."""
    mine = set(state.get("my_battlefield", []))
    if "Tainted Aether" not in mine:
        return []
    vetoes = []
    for prop in state.get("proposed", []):
        if "Creature" not in prop.get("type", ""):
            continue
        if prop["card"] in ("Hunted Horror", "Hunted Phantasm"):
            continue
        vetoes.append(prop["card"])
    return vetoes


class PolicyHandler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            try:
                state = json.loads(line.decode("utf-8"))
                vetoes = tainted_policy(state)
                STATS["decisions"] += 1
                if vetoes:
                    STATS["vetoes"] += 1
                    reply = "veto\t" + "\t".join(vetoes)
                else:
                    reply = "ok"
            except (json.JSONDecodeError, KeyError):
                reply = "ok"
            self.wfile.write((reply + "\n").encode("utf-8"))


def start_server(port: int) -> socketserver.ThreadingTCPServer:
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", port), PolicyHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def run_bridged(deck_a: str, deck_b: str, games: int, timeout_s: int,
                port: int | None, player_filter: str | None = None) -> str:
    """Verbose sim; when port is set, deck_a's AI consults the policy."""
    import os
    cmd = ["xvfb-run", "-a", "java", "-Xmx3g",
           "-Dio.netty.tryReflectionSetAccessible=true",
           "-Dfile.encoding=UTF-8",
           "-cp", f"{EXT_CLASSES}:{JAR}", "forge.view.Main",
           "sim", "-d", f"{deck_a}.dck", f"{deck_b}.dck", "-n", str(games)]
    env = dict(os.environ)
    if port is not None:
        env["FORGE_EXT_POLICY"] = str(port)
        env["FORGE_EXT_PLAYER"] = player_filter or deck_a
    else:
        env.pop("FORGE_EXT_POLICY", None)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, cwd=FORGE_DIR, env=env)
        return proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        return out.decode("utf-8", "replace") if isinstance(out, bytes) else out


def experiment(games: int = 24) -> None:
    """A/B: unblinded built-in AI vs unblinded + veto policy."""
    tracked = ["Tainted Aether", "Acorn Catapult"]
    port = 8877
    srv = start_server(port)
    out_dir = Path(__file__).resolve().parent / "output"

    for label, use_port in (("builtin", None), ("policy", port)):
        all_games = []
        STATS["decisions"] = STATS["vetoes"] = 0
        per = max(6, games // 3)
        for gi in range(3):
            log = run_bridged("evo_tainted2_base", f"evo_tainted2_g{gi}",
                              per, 120 + 30 * per, use_port)
            (out_dir / f"pilot_{label}_g{gi}.log").write_text(log)
            games_p = parse_games(log, "evo_tainted2_base", tracked)
            all_games += games_p
            print(f"[{label}] g{gi}: "
                  f"{sum(g['won'] for g in games_p)}/{len(games_p)}",
                  flush=True)
        stats = aggregate_stats(all_games, tracked)
        p, half = ci95(stats["wins"], stats["games"])
        print(f"[{label}] TOTAL {stats['wins']}/{stats['games']}"
              f" = {p:.0%} ±{half:.0%}"
              + (f"  ({STATS['decisions']} decisions, "
                 f"{STATS['vetoes']} vetoed)" if use_port else ""),
              flush=True)
        for name, t in stats["tracked"].items():
            print(f"  {name}: played {t['played_in']}, "
                  f"first turn {t['median_first_turn']}", flush=True)
    srv.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--serve", type=int, metavar="PORT")
    g.add_argument("--experiment", action="store_true")
    ap.add_argument("--games", type=int, default=24)
    args = ap.parse_args()
    if args.serve:
        start_server(args.serve)
        print(f"policy server on 127.0.0.1:{args.serve}; ctrl-c to stop")
        threading.Event().wait()
    else:
        experiment(args.games)


if __name__ == "__main__":
    main()
