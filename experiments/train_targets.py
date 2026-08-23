"""BC for the protocol-v4 target head: learn the builtin AI's target
choices from output/target_events.jsonl, warm-starting everything else
from an existing checkpoint. Only tgt_proj/tgt_head train (trunk and
other heads frozen: this must not disturb parity behavior).

Usage:
  python experiments/train_targets.py --base output/pilot2_rl.pt \
      --out output/pilot2_v4.pt --epochs 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path(__file__).resolve().parent / "output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(OUT / "pilot2_rl.pt"))
    ap.add_argument("--out", default=str(OUT / "pilot2_v4.pt"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", type=float, default=0.15)
    args = ap.parse_args()

    import torch
    from pilot_bridge import ModelPolicy

    policy = ModelPolicy(ckpt=Path(args.base))
    model, feat, tt = policy.model, policy.feat, policy.tt

    events = [json.loads(ln) for ln in
              open(OUT / "target_events.jsonl", encoding="utf-8")]
    events = [e for e in events if e.get("candidates")
              and e.get("proposed")
              and any(c["i"] == e["proposed"][0] for c in e["candidates"])]
    rng = np.random.default_rng(7)
    rng.shuffle(events)
    n_hold = max(1, int(len(events) * args.holdout))
    hold, train = events[:n_hold], events[n_hold:]
    print(f"{len(train)} train / {len(hold)} holdout target events",
          flush=True)

    for p in model.parameters():
        p.requires_grad = False
    for m in (model.tgt_proj, model.tgt_head):
        for p in m.parameters():
            p.requires_grad = True
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def logits_for(e):
        with torch.no_grad():
            hs, _h, _ids = policy.encode(e)
            hostv = model.cand_proj(tt(feat.cand_vec(
                {"card": e.get("host", "")})))
        return torch.cat([model.tgt_head(torch.cat(
            [hs, model.tgt_proj(tt(feat.tgt_vec(c))), hostv]))
            for c in e["candidates"]])

    def label(e):
        return next(k for k, c in enumerate(e["candidates"])
                    if c["i"] == e["proposed"][0])

    def acc(evs):
        model.eval()
        hits = 0
        with torch.no_grad():
            for e in evs:
                hits += int(int(logits_for(e).argmax()) == label(e))
        return hits / max(1, len(evs))

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(train))
        tot = 0.0
        for bi in range(0, len(order), 16):
            opt.zero_grad()
            loss = 0
            for k in order[bi:bi + 16]:
                e = train[k]
                loss = loss + torch.nn.functional.cross_entropy(
                    logits_for(e).unsqueeze(0),
                    torch.tensor([label(e)]))
            loss.backward()
            opt.step()
            tot += float(loss)
        a = acc(hold)
        print(f"epoch {ep}: loss {tot/max(1,len(train)):.4f} "
              f"holdout acc {a:.1%}", flush=True)
        if a >= best:
            best = a
            torch.save(model.state_dict(), args.out)
    # majority-class baseline for honesty
    from collections import Counter
    base = Counter(label(e) == 0 for e in hold)
    print(f"saved best (holdout {best:.1%}) -> {args.out}; "
          f"always-first-candidate baseline "
          f"{base[True]/max(1,len(hold)):.1%}", flush=True)


if __name__ == "__main__":
    main()
