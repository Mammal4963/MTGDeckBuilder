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

STATS = {"decisions": 0, "vetoes": 0, "forces": 0}
COLLECT_FILE = {"fh": None}          # set by --collect / experiment


def bf_names(state: dict, key: str) -> list[str]:
    """Battlefield entries are objects since protocol v3; accept both."""
    return [c["n"] if isinstance(c, dict) else c
            for c in state.get(key, [])]


def tainted_policy(state: dict) -> str:
    """Lock-plan policy (protocol v3 aware).

    1. With Tainted Aether on our battlefield, FORCE an Acorn Catapult
       activation at the opponent (once per turn, enforced Java-side):
       the squirrel gift becomes a forced sacrifice under the lock.
    2. Under the lock, veto casting our own creatures - except the
       Hunted ones, whose ETB token gift feeds the same engine.
    Combat messages ("attackers"/"blockers") are observe-only.
    """
    if state.get("kind") in ("attackers", "blockers"):
        return "ok"
    mine = bf_names(state, "my_battlefield")
    proposed = state.get("proposed", [])
    lock_up = "Tainted Aether" in mine

    if lock_up:
        for cand in state.get("candidates", []):
            if (cand["card"] == "Acorn Catapult"
                    and cand["zone"] == "battlefield"
                    and cand.get("targeted")):
                return f"force\t{cand['i']}\topponent"

    if lock_up and proposed:
        # proposed is a list of card names in v2
        vetoes = []
        by_name = {c["card"]: c for c in state.get("candidates", [])}
        for name in proposed:
            cand = by_name.get(name, {})
            if "Creature" not in cand.get("type", ""):
                continue
            if name in ("Hunted Horror", "Hunted Phantasm"):
                continue
            vetoes.append(name)
        if vetoes:
            return "veto\t" + "\t".join(vetoes)
    return "ok"


class ModelPolicy:
    """The trained pilot (train_pilot2) making live cast decisions.

    kind=cast: encode the board, score candidates + pass, act on argmax:
      - argmax == the built-in AI's proposal -> ok
      - argmax is pass -> veto the proposal (pass priority)
      - argmax is another candidate -> force it
    Combat stays observe-only this rung (serving blocks needs the
    Combat-object write path in Java).
    """

    def __init__(self, ckpt=None):
        import torch
        import train_pilot2 as tp
        self.torch = torch
        self.ckpt = ckpt          # None -> the shared BC checkpoint
        self.feat = tp.Featurizer()
        # rebuild the architecture exactly as trained
        import torch.nn as nn
        feat = self.feat
        D = tp.D
        sdim = feat.scalars({}).shape[0]
        cdim = feat.dim + 3

        class Pilot(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(feat.tok_dim, D)
                self.state_tok = nn.Parameter(torch.randn(1, D) * 0.02)
                layer = nn.TransformerEncoderLayer(
                    d_model=D, nhead=4, dim_feedforward=256,
                    batch_first=True, dropout=0.1)
                self.enc = nn.TransformerEncoder(layer, num_layers=2)
                self.state_mlp = nn.Sequential(nn.Linear(D + sdim, D), nn.ReLU())
                self.cand_proj = nn.Sequential(nn.Linear(cdim, D), nn.ReLU())
                self.cast_head = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(),
                                               nn.Linear(D, 1))
                self.pass_head = nn.Sequential(nn.Linear(D, D), nn.ReLU(),
                                               nn.Linear(D, 1))
                self.atk_head = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(),
                                              nn.Linear(D, 1))
                self.blk_head = nn.Sequential(nn.Linear(3 * D, D), nn.ReLU(),
                                              nn.Linear(D, 1))
                self.noblk_head = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(),
                                                nn.Linear(D, 1))

        self.model = Pilot()
        path = self.ckpt or (Path(__file__).resolve().parent
                             / "output" / "pilot2.pt")
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def encode(self, state: dict):
        torch = self.torch
        tt = self.tt
        toks, ids = self.feat.tokens_and_ids(state)
        x = self.model.proj(tt(toks)).unsqueeze(0)
        x = torch.cat([self.model.state_tok.unsqueeze(0), x], dim=1)
        h = self.model.enc(x)[0]
        hs = self.model.state_mlp(torch.cat(
            [h[0], tt(self.feat.scalars(state))]))
        return hs, h[1:], ids

    @property
    def tt(self):
        torch = self.torch
        import numpy as np
        return lambda x: torch.from_numpy(np.ascontiguousarray(x))

    def _pick(self, logits):
        """argmax, or sample at self.temperature when set (self-play)."""
        torch = self.torch
        t = getattr(self, "temperature", 0.0)
        if t and t > 0:
            return int(torch.multinomial(
                torch.softmax(logits / t, dim=0), 1))
        return int(logits.argmax())

    def __call__(self, state: dict) -> str:
        torch = self.torch
        kind = state.get("kind")
        with torch.no_grad():
            if kind == "cast":
                cands = state.get("candidates", [])
                if not cands:
                    return "ok"
                hs, _h, _ids = self.encode(state)
                scores = [self.model.cast_head(torch.cat(
                    [hs, self.model.cand_proj(
                        self.tt(self.feat.cand_vec(c)))])) for c in cands]
                scores.append(self.model.pass_head(hs))
                pick = self._pick(torch.cat(scores))
                proposed = state.get("proposed", [])
                if pick == len(cands):
                    return ("veto\t" + "\t".join(proposed)) if proposed \
                        else "ok"
                if proposed and cands[pick]["card"] == proposed[0]:
                    return "ok"
                return f"force\t{cands[pick]['i']}"

            if kind == "attackers":
                hs, h, ids = self.encode(state)
                want = []
                for k, x in enumerate(ids):
                    if x is None:
                        continue
                    side, cid, c = x
                    if side != "my" or not c.get("cr") or c.get("tapped"):
                        continue
                    lg = self.model.atk_head(torch.cat([hs, h[k]]))
                    t = getattr(self, "temperature", 0.0)
                    p = float(torch.sigmoid(lg / t if t else lg))
                    go = (torch.rand(1).item() < p) if t else (p > 0.5)
                    if go:
                        want.append(str(cid))
                return "attack\t" + ",".join(want)

            if kind == "blockers":
                attackers = state.get("attackers", [])
                if not attackers:
                    return "ok"
                hs, h, ids = self.encode(state)
                a_tok = {}
                for k, x in enumerate(ids):
                    if x is not None and x[1] in {a["id"] for a in attackers}:
                        a_tok[x[1]] = k
                if len(a_tok) != len(attackers):
                    return "ok"
                a_ids = [a["id"] for a in attackers]
                pairs = []
                for k, x in enumerate(ids):
                    if x is None:
                        continue
                    side, cid, c = x
                    if side != "my" or not c.get("cr") or c.get("tapped"):
                        continue
                    scores = [self.model.blk_head(torch.cat(
                        [hs, h[k], h[a_tok[a]]])) for a in a_ids]
                    scores.append(self.model.noblk_head(
                        torch.cat([hs, h[k]])))
                    pick = self._pick(torch.cat(scores))
                    if pick < len(a_ids):
                        pairs.append(f"{cid}:{a_ids[pick]}")
                return "block\t" + ",".join(pairs)
        return "ok"


class PolicyHandler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            try:
                state = json.loads(line.decode("utf-8"))
                reply = tainted_policy(state)
                STATS["decisions"] += 1
                if reply.startswith("veto"):
                    STATS["vetoes"] += 1
                elif reply.startswith("force"):
                    STATS["forces"] += 1
                if COLLECT_FILE["fh"] is not None:
                    state["_reply"] = reply
                    COLLECT_FILE["fh"].write(
                        json.dumps(state, separators=(",", ":")) + "\n")
                    COLLECT_FILE["fh"].flush()
            except (json.JSONDecodeError, KeyError, ValueError):
                reply = "ok"
            self.wfile.write((reply + "\n").encode("utf-8"))


def start_server(port: int) -> socketserver.ThreadingTCPServer:
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", port), PolicyHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def run_bridged(deck_a: str, deck_b: str, games: int, timeout_s: int,
                port: int | None, player_filter: str | None = None,
                quiet: bool = False, worker: int = 0) -> str:
    """Sim with the policy bridge on matching players (quiet or verbose)."""
    import os
    if os.environ.get("FORGE_SIM_SERVER") == "1":
        import sim_server
        extra = {}
        if port is not None:
            extra = {"FORGE_EXT_POLICY": str(port),
                     "FORGE_EXT_PLAYER": (deck_a if player_filter is None else player_filter)}
        key = f"bridge-{port}-{player_filter or deck_a}-w{worker}"
        return sim_server.shared_client(key, extra).run(
            deck_a, deck_b, games, quiet=quiet, timeout_s=timeout_s)
    from improve_deck import java_prefix
    cmd = java_prefix() + ["-Xmx3g",
           "-Dio.netty.tryReflectionSetAccessible=true",
           "-Dfile.encoding=UTF-8",
           "-cp", f"{EXT_CLASSES}:{JAR}", "forge.view.Main",
           "sim", "-d", f"{deck_a}.dck", f"{deck_b}.dck", "-n", str(games)]
    if quiet:
        cmd.append("-q")
    env = dict(os.environ)
    if port is not None:
        env["FORGE_EXT_POLICY"] = str(port)
        env["FORGE_EXT_PLAYER"] = (deck_a if player_filter is None else player_filter)
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
                 f"{STATS['vetoes']} vetoed, {STATS['forces']} forced)"
                 if use_port else ""),
              flush=True)
        for name, t in stats["tracked"].items():
            print(f"  {name}: played {t['played_in']}, "
                  f"first turn {t['median_first_turn']}", flush=True)
    srv.shutdown()


def collect(games_per_pair: int = 6) -> None:
    """Behavior-cloning dataset: observe the BUILT-IN AI's choices.

    The bridge runs in observer mode (policy always answers "ok"), so
    every logged decision line carries the state, the full candidate
    list, and what the built-in AI chose (`proposed`). Varied matchups
    give the pretraining "empty baseline" the user asked for.
    """
    global tainted_policy
    orig_policy = tainted_policy
    tainted_policy = lambda state: "ok"          # pure observer
    out_path = Path(__file__).resolve().parent / "output" / "pilot_dataset.jsonl"
    COLLECT_FILE["fh"] = open(out_path, "a")
    srv = start_server(0)              # ephemeral port - never collides
    port = srv.server_address[1]
    # mix in the creature-heavy legacy decks (elves, stompy) so combat
    # decisions - attacks and blocks - are well represented
    import os
    if os.environ.get("COLLECT_COMBAT") == "1":
        # combat-dense rotation: creature decks only, where the AI
        # declares attacks and blocks every few turns
        decks = ["gauntlet_0", "gauntlet_1", "gauntlet_2", "burn",
                 "gauntlet_1", "gauntlet_0", "burn", "gauntlet_2"]
    else:
        decks = ["evo_tainted2_base", "gauntlet_1", "evo_tainted2_g0",
                 "gauntlet_2", "evo_tainted2_g1", "burn",
                 "evo_tainted2_g2", "gauntlet_0"]
    try:
        for i, a in enumerate(decks):
            b = decks[(i + 1) % len(decks)]
            n0 = STATS["decisions"]
            # player_filter "evo_" bridges BOTH sides: two observed
            # controllers per game = twice the data per sim
            run_bridged(a, b, games_per_pair,
                        120 + 40 * games_per_pair, port, player_filter="")
            print(f"{a} vs {b}: +{STATS['decisions'] - n0} decisions",
                  flush=True)
    finally:
        COLLECT_FILE["fh"].close()
        COLLECT_FILE["fh"] = None
        tainted_policy = orig_policy
        srv.shutdown()
        srv.server_close()
    print(f"dataset -> {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--serve", type=int, metavar="PORT")
    g.add_argument("--experiment", action="store_true")
    g.add_argument("--collect", action="store_true")
    ap.add_argument("--games", type=int, default=24)
    args = ap.parse_args()
    if args.serve:
        start_server(args.serve)
        print(f"policy server on 127.0.0.1:{args.serve}; ctrl-c to stop")
        threading.Event().wait()
    elif args.collect:
        collect(args.games // 4 or 6)
    else:
        experiment(args.games)


if __name__ == "__main__":
    main()
