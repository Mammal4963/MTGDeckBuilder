"""Graft the non-cast heads onto the foundation trunk by behavior
cloning archived decisions.

foundation_D192L6.pt carries trunk + cast/pass/val heads only, and the
rl20/21 heads are D=256 so they cannot transfer. This BC-trains
atk/blk/noblk, tgt_proj/tgt_head, and mull_head (+deck_proj) on the
r18+ corpus (decisions made by trained nets under fixed machinery),
with the trunk FROZEN so the pretrained value function is untouched.

Output: pilot2_fbase.pt - the full-pilot foundation base for RL.

Usage:
  PILOT_D=192 PILOT_LAYERS=6 python experiments/train_foundation_heads.py
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path(__file__).resolve().parent / "output"
GOOD_PREFIXES = ("r18", "r180", "r181", "r19", "r20", "r21")
KINDS = {"mulligan", "attackers", "blockers", "target"}


def gather(max_per_kind=20000):
    rows = list({json.loads(ln)["file"]: json.loads(ln) for ln in
                 (OUT / "games_index.jsonl").read_text(encoding="utf-8")
                 .splitlines()}.values())
    rows = [r for r in rows
            if str(r.get("file", "")).startswith(GOOD_PREFIXES)]
    rng = np.random.default_rng(71)
    rng.shuffle(rows)
    by_kind = {k: [] for k in KINDS}
    for r in rows:
        if all(len(v) >= max_per_kind for v in by_kind.values()):
            break
        try:
            g = json.load(gzip.open(OUT / "games" / r["file"], "rt",
                                    encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for s, rep in g["decisions"]:
            k = s.get("kind")
            if k in by_kind and len(by_kind[k]) < max_per_kind:
                by_kind[k].append((s, rep))
    return by_kind


def target_logp(pol, torch, state, reply):
    """Mirror the serving path: score target candidates, logp of the
    archived pick (decision_logp has no target branch)."""
    model = pol.model
    cands = state.get("candidates", [])
    if not cands:
        return None
    hs, _h, _ids = pol.encode(state)
    hostv = model.cand_proj(pol.tt(
        pol.feat.cand_vec({"card": state.get("host", "")})))
    logits = torch.cat([model.tgt_head(torch.cat(
        [hs, model.tgt_proj(pol.tt(pol.feat.tgt_vec(c))), hostv]))
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
        return None
    return torch.log_softmax(logits, dim=0)[action]


def main():
    import torch
    from pilot_bridge import ModelPolicy
    from self_play import decision_logp

    pol = ModelPolicy(ckpt=OUT / "foundation_D192L6.pt")
    model = pol.model
    head_names = ("atk_head", "blk_head", "noblk_head", "tgt_proj",
                  "tgt_head", "mull_head", "deck_proj", "tuck_head")
    head_params = [p for n, p in model.named_parameters()
                   if n.split(".")[0] in head_names]
    for n, p in model.named_parameters():
        p.requires_grad = n.split(".")[0] in head_names
    opt = torch.optim.Adam(head_params, lr=1e-3)
    print(f"trainable head params: "
          f"{sum(p.numel() for p in head_params):,}", flush=True)

    data = gather()
    for k, v in data.items():
        print(f"{k}: {len(v)} decisions", flush=True)
    flat = [(k, s, rep) for k, v in data.items() for s, rep in v]
    rng = np.random.default_rng(73)

    for ep in range(2):
        rng.shuffle(flat)
        model.train()
        tot = {k: [0.0, 0] for k in KINDS}
        opt.zero_grad()
        nacc = 0
        t0 = time.time()
        for i, (k, s, rep) in enumerate(flat):
            try:
                if k == "target":
                    lp = target_logp(pol, torch, s, rep)
                else:
                    o = decision_logp(pol, torch, s, rep)
                    lp = o[0] if o else None
                if lp is None:
                    continue
                (-lp / 32).backward()
                tot[k][0] += float(lp.detach())
                tot[k][1] += 1
                nacc += 1
                if nacc % 32 == 0:
                    torch.nn.utils.clip_grad_norm_(head_params, 1.0)
                    opt.step()
                    opt.zero_grad()
            except Exception:
                continue
        opt.step()
        stats = {k: round(v[0] / max(1, v[1]), 3)
                 for k, v in tot.items()}
        print(f"epoch {ep}: mean logp {json.dumps(stats)} "
              f"({time.time()-t0:.0f}s)", flush=True)

    model.eval()
    torch.save(model.state_dict(), OUT / "pilot2_fbase.pt")
    print(f"saved -> {OUT / 'pilot2_fbase.pt'}", flush=True)


if __name__ == "__main__":
    main()
