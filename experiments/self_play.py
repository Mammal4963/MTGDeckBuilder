"""Rung 5: self-play fine-tuning - REINFORCE over the BC initialization.

The loop the whole ladder was built for:
  1. the pilot (pilot2 BC weights) plays games through the bridge with
     SAMPLING (temperature > 0) so it explores
  2. every decision it made is journaled; each game's decisions get the
     game's reward:  win/loss  +  lock-bonus (fraction of --lock cards
     that hit our battlefield - the "play YOUR plan" shaping signal)
  3. REINFORCE update: increase log-prob of decisions from winning
     games, decrease from losses, against an EMA baseline
  4. repeat; checkpoints + metrics journaled every iteration

One game per sim-server job gives exact trajectory<->reward assignment.
This container demos the loop at small scale; real convergence is an
overnight run on a beefier box (see FORGE-NOTES rung 5).

Usage:
  FORGE_SIM_SERVER=1 python3 experiments/self_play.py \
      --deck evo_tainted2_base --iters 3 --games 12 \
      --lock "Tainted Aether" --lock "Acorn Catapult"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pilot_bridge  # noqa: E402
from pilot_bridge import (ModelPolicy, start_server,  # noqa: E402
                          run_bridged)

OUT = Path(__file__).resolve().parent / "output"
RNG = np.random.default_rng(3)


class RecordingPolicy:
    """Wraps ModelPolicy: journals (state, reply) per game.

    Each state is stamped with timing for the benchmark lane:
    _dt_ms   wall time since the previous decision this game — mostly
             Forge engine + built-in-AI (opponent) work between our
             decision points
    _model_ms  our own forward-pass cost for this decision
    """

    def __init__(self, inner):
        self.inner = inner
        self.buffer = []
        self._last_t = 0.0

    def _stamp(self, state):
        import time
        now = time.monotonic()
        if self.buffer:
            state["_dt_ms"] = int(1000 * (now - self._last_t))
        self._last_t = now

    def __call__(self, state, _stamped=False):
        import time
        if not _stamped:
            self._stamp(state)
        t0 = time.monotonic()
        reply = self.inner(state)
        state["_model_ms"] = int(1000 * (time.monotonic() - t0))
        self.buffer.append((state, reply))
        return reply


def game_lock_frac(decisions, locks):
    """Fraction of locked cards the pilot actually deployed this game.

    Permanents count via battlefield presence; spells (a sorcery lock
    like Random Encounter never touches the battlefield) count via the
    cast decision the pilot took - forced candidate or ok'd proposal.
    """
    seen = set()
    for state, reply in decisions:
        for c in state.get("my_battlefield", []):
            n = c["n"] if isinstance(c, dict) else c
            if n in locks:
                seen.add(n)
        if state.get("kind") == "cast":
            chosen = None
            if reply.startswith("force\t"):
                idx = int(reply.split("\t")[1])
                chosen = next((c["card"] for c in state.get("candidates", [])
                               if c["i"] == idx), None)
            elif reply == "ok":
                proposed = state.get("proposed", [])
                chosen = proposed[0] if proposed else None
            if chosen in locks:
                seen.add(chosen)
    return len(seen) / max(1, len(locks))


def reinforce_update(model_policy, torch, batch, baseline, lr_opt):
    """batch: list of (decisions, reward). Recompute log-probs and
    ascend reward-weighted likelihood."""
    model = model_policy.model
    model.train()
    total_loss = 0.0
    n_terms = 0
    lr_opt.zero_grad()
    for decisions, reward in batch:
        adv = reward - baseline
        if abs(adv) < 1e-6:
            continue
        for state, reply in decisions:
            kind = state.get("kind")
            try:
                if kind == "cast":
                    cands = state.get("candidates", [])
                    if not cands:
                        continue
                    hs, _h, _ids = model_policy.encode(state)
                    logits = torch.cat(
                        [model.cast_head(torch.cat([hs, model.cand_proj(
                            model_policy.tt(model_policy.feat.cand_vec(c)))]))
                         for c in cands]
                        + [model.pass_head(hs)])
                    # which action did we take?
                    if reply.startswith("force\t"):
                        idx = int(reply.split("\t")[1])
                        action = next((k for k, c in enumerate(cands)
                                       if c["i"] == idx), None)
                    elif reply.startswith("veto"):
                        action = len(cands)
                    else:                       # ok = the AI's proposal
                        proposed = state.get("proposed", [])
                        action = next((k for k, c in enumerate(cands)
                                       if proposed
                                       and c["card"] == proposed[0]),
                                      len(cands))
                    if action is None:
                        continue
                    logp = torch.log_softmax(logits, dim=0)[action]
                elif kind == "mulligan":
                    if not hasattr(model, "mull_head") \
                            or reply not in ("keep", "mull"):
                        continue
                    hs, _h, _ids = model_policy.encode(state)
                    ctr = torch.tensor(
                        [state.get("cards_to_return", 0) / 7.0])
                    lg = model.mull_head(torch.cat([hs, ctr]))[0]
                    logp = -torch.nn.functional \
                        .binary_cross_entropy_with_logits(
                            lg, torch.tensor(
                                1.0 if reply == "keep" else 0.0))
                elif kind == "target":
                    cands = state.get("candidates", [])
                    if not cands or not hasattr(model, "tgt_head"):
                        continue
                    hs, _h, _ids = model_policy.encode(state)
                    hostv = model.cand_proj(model_policy.tt(
                        model_policy.feat.cand_vec(
                            {"card": state.get("host", "")})))
                    logits = torch.cat([model.tgt_head(torch.cat(
                        [hs, model.tgt_proj(model_policy.tt(
                            model_policy.feat.tgt_vec(c))), hostv]))
                        for c in cands])
                    if reply.startswith("target\t"):
                        idx = int(reply.split("\t")[1])
                        action = next((k for k, c in enumerate(cands)
                                       if c["i"] == idx), None)
                    else:
                        prop = state.get("proposed", [])
                        action = next((k for k, c in enumerate(cands)
                                       if prop and c["i"] == prop[0]), None)
                    if action is None:
                        continue
                    logp = torch.log_softmax(logits, dim=0)[action]
                elif kind == "attackers" and reply.startswith("attack\t"):
                    hs, h, ids = model_policy.encode(state)
                    want = {int(x) for x in reply[7:].split(",") if x}
                    logps = []
                    for k, x in enumerate(ids):
                        if x is None:
                            continue
                        side, cid, c = x
                        if side != "my" or not c.get("cr") \
                                or c.get("tapped"):
                            continue
                        lg = model.atk_head(torch.cat([hs, h[k]]))[0]
                        y = 1.0 if cid in want else 0.0
                        logps.append(
                            -torch.nn.functional
                            .binary_cross_entropy_with_logits(
                                lg, torch.tensor(y)))
                    if not logps:
                        continue
                    logp = torch.stack(logps).sum()
                elif kind == "blockers" and reply.startswith("block\t"):
                    attackers = state.get("attackers", [])
                    if not attackers:
                        continue
                    hs, h, ids = model_policy.encode(state)
                    a_ids = [a["id"] for a in attackers]
                    a_tok = {x[1]: k for k, x in enumerate(ids)
                             if x is not None and x[1] in set(a_ids)}
                    if len(a_tok) != len(a_ids):
                        continue
                    chosen = {}
                    for pair in reply[6:].split(","):
                        if ":" in pair:
                            b, a = pair.split(":")
                            chosen[int(b)] = int(a)
                    logps = []
                    for k, x in enumerate(ids):
                        if x is None:
                            continue
                        side, cid, c = x
                        if side != "my" or not c.get("cr") \
                                or c.get("tapped"):
                            continue
                        scores = torch.cat(
                            [model.blk_head(torch.cat(
                                [hs, h[k], h[a_tok[a]]])) for a in a_ids]
                            + [model.noblk_head(torch.cat([hs, h[k]]))])
                        action = (a_ids.index(chosen[cid])
                                  if cid in chosen else len(a_ids))
                        logps.append(
                            torch.log_softmax(scores, dim=0)[action])
                    if not logps:
                        continue
                    logp = torch.stack(logps).sum()
                else:
                    continue
                loss = -adv * logp
                loss.backward()
                total_loss += float(loss.detach())
                n_terms += 1
            except Exception:
                continue
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr_opt.step()
    model.eval()
    return total_loss / max(1, n_terms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="evo_tainted2_base")
    ap.add_argument("--opponents", nargs="+",
                    default=["evo_tainted2_g0", "evo_tainted2_g1",
                             "evo_tainted2_g2"])
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--games", type=int, default=12,
                    help="games per iteration")
    ap.add_argument("--lock", action="append", default=[])
    ap.add_argument("--lock-bonus", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--port", type=int, default=8890)
    ap.add_argument("--name", default="rl")
    args = ap.parse_args()

    import torch
    policy = RecordingPolicy(ModelPolicy())
    policy.inner.temperature = args.temperature
    pilot_bridge.tainted_policy = policy
    opt = torch.optim.Adam(policy.inner.model.parameters(), lr=args.lr)

    srv = start_server(args.port)
    baseline = 0.0
    journal = []
    try:
        for it in range(args.iters):
            batch, wins, lock_sum = [], 0, 0.0
            for g in range(args.games):
                policy.buffer = []
                opp = args.opponents[int(RNG.integers(len(args.opponents)))]
                log = run_bridged(args.deck, opp, 1, 240, args.port,
                                  player_filter=args.deck)
                won = bool(re.search(
                    rf"Game Result.*Ai\(1\)-{re.escape(args.deck)} has won",
                    log))
                lf = game_lock_frac(policy.buffer, set(args.lock))
                reward = (1.0 if won else -1.0) + args.lock_bonus * lf
                batch.append((list(policy.buffer), reward))
                wins += won
                lock_sum += lf
            loss = reinforce_update(policy.inner, torch, batch,
                                    baseline, opt)
            rewards = [r for _d, r in batch]
            baseline = 0.7 * baseline + 0.3 * float(np.mean(rewards))
            entry = {"iter": it, "winrate": wins / args.games,
                     "lock_frac": lock_sum / args.games,
                     "baseline": round(baseline, 3),
                     "loss": round(loss, 4)}
            journal.append(entry)
            print(json.dumps(entry), flush=True)
            torch.save(policy.inner.model.state_dict(),
                       OUT / f"pilot2_{args.name}.pt")
            (OUT / f"selfplay-{args.name}.json").write_text(
                json.dumps(journal, indent=1))
    finally:
        srv.shutdown()
        srv.server_close()
    print(f"checkpoint: {OUT / f'pilot2_{args.name}.pt'}")


if __name__ == "__main__":
    main()
