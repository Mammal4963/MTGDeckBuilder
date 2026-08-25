"""Distill the champion pilot into a bigger trunk (option 2).

Sets PILOT_D/PILOT_LAYERS, builds a fresh big Pilot, and trains ALL
heads supervised on archived champion-era games: policy loss =
-log p(archived action) (via decision_logp), value loss = MSE toward
the game outcome. eps-forced decisions are excluded (exploration
noise, not the champion's choice).

Usage:
  python experiments/distill_big.py --epochs 2 --max-per-game 80 \
      --out output/pilot_big_bc.pt
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PILOT_D", "256")
os.environ.setdefault("PILOT_LAYERS", "4")

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path(__file__).resolve().parent / "output"


def game_rows(rounds):
    rows = [json.loads(ln) for ln in
            (OUT / "games_index.jsonl").read_text(encoding="utf-8")
            .splitlines()]
    pat = re.compile(rounds)
    return [r for r in rows if pat.match(str(r.get("iter", "")))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--rounds", default=r"r(8|9|10|11)-|cf-rl(8|9)",
                    help="regex over games_index iter field")
    ap.add_argument("--max-per-game", type=int, default=80)
    ap.add_argument("--max-games", type=int, default=1400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-coef", type=float, default=0.5)
    ap.add_argument("--out", default=str(OUT / "pilot_big_bc.pt"))
    args = ap.parse_args()

    import torch
    from pilot_bridge import ModelPolicy
    from self_play import decision_logp
    from self_play_round2 import split_seats

    policy = ModelPolicy(ckpt=OUT / "pilot2.pt")   # tolerant load: shapes
    model = policy.model                           # mismatch -> fresh init
    print(f"big pilot: D={os.environ['PILOT_D']} "
          f"layers={os.environ['PILOT_LAYERS']} params="
          f"{sum(p.numel() for p in model.parameters()):,}", flush=True)

    rows = game_rows(args.rounds)
    rng = np.random.default_rng(17)
    rng.shuffle(rows)
    rows = rows[:args.max_games]
    hold_n = max(1, len(rows) // 12)
    hold, train = rows[:hold_n], rows[hold_n:]
    print(f"{len(train)} train games / {len(hold)} holdout", flush=True)

    def game_decisions(r):
        try:
            with gzip.open(OUT / "games" / r["file"], "rt",
                           encoding="utf-8") as f:
                g = json.load(f)
        except OSError:
            return []
        R = 1.0 if g.get("won") else -1.0
        dec = [(s, rep) for s, rep in g["decisions"]
               if not s.get("_eps_forced")
               and s.get("kind") != "game_end"]
        if len(dec) > args.max_per_game:
            keep = rng.choice(len(dec), args.max_per_game, replace=False)
            dec = [dec[i] for i in sorted(keep)]
        return [(s, rep, R) for s, rep in dec]

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def cast_agreement(rows_):
        model.eval()
        hit = tot = 0
        with torch.no_grad():
            for r in rows_[:40]:
                for s, rep, _R in game_decisions(r):
                    if s.get("kind") != "cast" or not s.get("candidates"):
                        continue
                    live = policy(dict(s))
                    hit += int(live == rep)
                    tot += 1
        return hit / max(1, tot)

    for ep in range(args.epochs):
        model.train()
        rng.shuffle(train)
        tot = seen = 0.0
        nb = 0
        opt.zero_grad()
        for gi, r in enumerate(train):
            for s, rep, R in game_decisions(r):
                try:
                    out = decision_logp(policy, torch, s, rep)
                    if out is None:
                        continue
                    logp, hs, _c = out
                    v = model.val_head(hs)[0]
                    loss = -logp + args.val_coef * (v - R) ** 2
                    loss.backward()
                    tot += float(loss.detach())
                    seen += 1
                    nb += 1
                    if nb % 32 == 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0)
                        opt.step()
                        opt.zero_grad()
                except Exception:
                    continue
            if gi % 100 == 0:
                print(f"  ep{ep} game {gi}/{len(train)} "
                      f"loss {tot/max(1,seen):.3f}", flush=True)
        opt.step()
        agree = cast_agreement(hold)
        print(f"epoch {ep}: loss {tot/max(1,seen):.3f} "
              f"holdout cast-agreement {agree:.1%}", flush=True)
        torch.save(model.state_dict(), args.out)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
