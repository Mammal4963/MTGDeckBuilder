"""BC for the mulligan head: learn the builtin's keep/mull choices from
output/round3_events.jsonl. Trains mull_head + deck_proj only (trunk
and other heads frozen). Class-weighted BCE: keeps vastly outnumber
mulls and an unweighted fit collapses to "always keep".

Usage:
  python experiments/train_mulligan.py --base output/pilot2_rl2.pt \
      --out output/pilot2_v5.pt --epochs 10
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
    ap.add_argument("--base", default=str(OUT / "pilot2_rl2.pt"))
    ap.add_argument("--out", default=str(OUT / "pilot2_v5.pt"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", type=float, default=0.15)
    args = ap.parse_args()

    import torch
    from pilot_bridge import ModelPolicy

    policy = ModelPolicy(ckpt=Path(args.base))
    model, feat, tt = policy.model, policy.feat, policy.tt

    events = [json.loads(ln) for ln in
              open(OUT / "round3_events.jsonl", encoding="utf-8")
              if '"kind":"mulligan"' in ln]
    events = [e for e in events if e.get("proposed") in ("keep", "mull")]
    rng = np.random.default_rng(13)
    rng.shuffle(events)
    n_hold = max(1, int(len(events) * args.holdout))
    hold, train = events[:n_hold], events[n_hold:]
    n_keep = sum(1 for e in train if e["proposed"] == "keep")
    n_mull = len(train) - n_keep
    pos_weight = torch.tensor(max(1.0, n_mull / max(1, n_keep)))
    mull_weight = max(1.0, n_keep / max(1, n_mull))
    print(f"{len(train)} train ({n_keep} keep / {n_mull} mull), "
          f"{len(hold)} holdout; mull class weight {mull_weight:.1f}",
          flush=True)

    for p in model.parameters():
        p.requires_grad = False
    for m in (model.mull_head, model.deck_proj):
        for p in m.parameters():
            p.requires_grad = True
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def logit_for(e):
        with torch.no_grad():
            hs, _h, _ids = policy.encode(e)
        dv = feat.deck_emb(e.get("player", ""))
        if dv is None:
            dv = np.zeros(feat.dim, np.float32)
        dk = model.deck_proj(tt(dv))
        ex = torch.cat([
            torch.tensor([e.get("cards_to_return", 0) / 7.0]),
            tt(feat.hand_feats(e))])
        return model.mull_head(torch.cat([hs, dk, ex]))[0]

    def metrics(evs):
        model.eval()
        tp = tn = fp = fn = 0
        with torch.no_grad():
            for e in evs:
                keep = float(logit_for(e)) > 0
                actual = e["proposed"] == "keep"
                if keep and actual:
                    tp += 1
                elif not keep and not actual:
                    tn += 1
                elif keep:
                    fp += 1
                else:
                    fn += 1
        acc = (tp + tn) / max(1, len(evs))
        mull_recall = tn / max(1, tn + fp)
        return acc, mull_recall

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(train))
        tot = 0.0
        for bi in range(0, len(order), 16):
            opt.zero_grad()
            loss = 0
            for k in order[bi:bi + 16]:
                e = train[k]
                y = torch.tensor(1.0 if e["proposed"] == "keep" else 0.0)
                w = 1.0 if e["proposed"] == "keep" else mull_weight
                loss = loss + w * torch.nn.functional \
                    .binary_cross_entropy_with_logits(logit_for(e), y)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        acc, mr = metrics(hold)
        score = acc + mr          # value catching mulls, not just accuracy
        print(f"epoch {ep}: loss {tot/max(1,len(train)):.4f} "
              f"holdout acc {acc:.1%} mull-recall {mr:.1%}", flush=True)
        if score >= best:
            best = score
            torch.save(model.state_dict(), args.out)
    # ---- tuck head: which cards to bottom on a London mulligan ------
    tuck = [json.loads(ln) for ln in
            open(OUT / "round3_events.jsonl", encoding="utf-8")
            if '"kind":"mulligan_tuck"' in ln]
    tuck = [e for e in tuck if e.get("options") and e.get("proposed")]
    if tuck:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.tuck_head.parameters():
            p.requires_grad = True
        topt = torch.optim.Adam(model.tuck_head.parameters(), lr=1e-3)

        def opt_name(o):
            return o["n"] if isinstance(o, dict) else o

        def card_scores(e):
            dv = feat.deck_emb(e.get("player", ""))
            if dv is None:
                dv = np.zeros(feat.dim, np.float32)
            with torch.no_grad():
                dk = model.deck_proj(tt(dv))
            return [model.tuck_head(torch.cat(
                [dk, tt(feat.card_emb(opt_name(o)))]))[0]
                for o in e["options"]]

        def labels(e):
            # keep=1, bottomed=0; duplicates matched by name count
            bot = list(e["proposed"])
            ys = []
            for o in e["options"]:
                n = opt_name(o)
                if n in bot:
                    bot.remove(n)
                    ys.append(0.0)
                else:
                    ys.append(1.0)
            return ys

        n_hold2 = max(1, int(len(tuck) * 0.15))
        thold, ttrain = tuck[:n_hold2], tuck[n_hold2:]
        print(f"tuck: {len(ttrain)} train / {len(thold)} holdout events",
              flush=True)
        for ep in range(8):
            model.train()
            tot = 0.0
            for e in ttrain:
                topt.zero_grad()
                loss = sum(torch.nn.functional
                           .binary_cross_entropy_with_logits(
                               s, torch.tensor(y))
                           for s, y in zip(card_scores(e), labels(e)))
                loss.backward()
                topt.step()
                tot += float(loss.detach())
            model.eval()
            hits = tries = 0
            with torch.no_grad():
                for e in thold:
                    ss = [float(s) for s in card_scores(e)]
                    ys = labels(e)
                    amount = int(sum(1 for y in ys if y == 0.0))
                    order = sorted(range(len(ss)), key=lambda k: ss[k])
                    pred_bottom = set(order[:amount])
                    hits += sum(1 for k in pred_bottom if ys[k] == 0.0)
                    tries += amount
            print(f"tuck epoch {ep}: loss {tot/max(1,len(ttrain)):.3f} "
                  f"holdout bottomed-card match {hits}/{tries}", flush=True)
        torch.save(model.state_dict(), args.out)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
