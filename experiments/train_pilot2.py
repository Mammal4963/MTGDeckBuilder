"""Pilot v2: set-transformer over per-card board tokens, multi-task BC.

Upgrades over train_pilot.py (v1 mean-pooling):
- No compression bottleneck: every card on the board is a TOKEN
  [MiniLM text embedding (frozen) + zone one-hot + live state
  (power, toughness, tapped, damage, is-creature)] and a small
  transformer attends over the variable-length set. Ten tokens or
  forty - same model, no averaging mush.
- Three cloned decision types from the same encoder:
    cast      which candidate play (or pass) - CE over candidates+pass
    attackers per eligible creature: attack or not - BCE
    blockers  per eligible blocker: which attacker to block, or none - CE
  Labels are what Forge's built-in AI chose (protocol-v3 observation).

Prints held-out metrics per head; saves output/pilot2.pt.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(__file__).resolve().parent / "output"
RNG = np.random.default_rng(13)

PHASES = ["MAIN1", "MAIN2", "COMBAT_DECLARE_ATTACKERS",
          "COMBAT_DECLARE_BLOCKERS", "UPKEEP", "END_OF_TURN"]
MAX_TOKENS = 60
D = 128


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


class Featurizer:
    def __init__(self):
        self.emb = np.load(OUT / "embeddings.npy").astype(np.float32)
        meta = json.loads((OUT / "cards_meta.json").read_text())
        self.rows = {}
        for i, m in enumerate(meta):
            self.rows.setdefault(norm(m["name"]), i)
        self.dim = self.emb.shape[1]
        # zones: hand, my bf, opp bf, my graveyard, opp graveyard
        self.tok_dim = self.dim + 5 + 5      # zone one-hot + state feats

    def card_emb(self, name: str) -> np.ndarray:
        r = self.rows.get(norm(name))
        return self.emb[r] if r is not None else np.zeros(self.dim, np.float32)

    def token(self, name: str, zone: int, c: dict | None) -> np.ndarray:
        z = np.zeros(5, np.float32)
        z[zone] = 1.0
        if c is None:
            st = np.zeros(5, np.float32)
        else:
            st = np.array([c.get("p", 0) / 10.0, c.get("t", 0) / 10.0,
                           1.0 if c.get("tapped") else 0.0,
                           c.get("dmg", 0) / 5.0,
                           1.0 if c.get("cr") else 0.0], np.float32)
        return np.concatenate([self.card_emb(name), z, st])

    def tokens_and_ids(self, s: dict):
        """-> (tokens [n, tok_dim], ids aligned list (None for hand))."""
        toks, ids = [], []
        for name in s.get("my_hand", []):
            toks.append(self.token(name, 0, None))
            ids.append(None)
        for c in s.get("my_battlefield", []):
            if isinstance(c, dict):
                toks.append(self.token(c["n"], 1, c))
                ids.append(("my", c["id"], c))
            else:
                toks.append(self.token(c, 1, None))
                ids.append(None)
        for c in s.get("opp_battlefield", []):
            if isinstance(c, dict):
                toks.append(self.token(c["n"], 2, c))
                ids.append(("opp", c["id"], c))
            else:
                toks.append(self.token(c, 2, None))
                ids.append(None)
        # graveyards: flashback/escape lines and reanimation targets
        # live here (names only - no per-card state in the bin)
        for n in s.get("my_graveyard", [])[-10:]:
            toks.append(self.token(n, 3, None))
            ids.append(None)
        for n in s.get("opp_graveyard", [])[-10:]:
            toks.append(self.token(n, 4, None))
            ids.append(None)
        toks = toks[:MAX_TOKENS]
        ids = ids[:MAX_TOKENS]
        if not toks:
            toks = [np.zeros(self.tok_dim, np.float32)]
            ids = [None]
        return np.stack(toks), ids

    def scalars(self, s: dict) -> np.ndarray:
        phase = str(s.get("phase", ""))
        ph = [1.0 if p == phase else 0.0 for p in PHASES]
        return np.array([
            min(s.get("turn", 0), 20) / 20.0,
            (s.get("my_life", 20) - s.get("opp_life", 20)) / 20.0,
            len(s.get("my_hand", [])) / 7.0,
            *ph,
        ], np.float32)

    def cand_vec(self, c: dict) -> np.ndarray:
        flags = np.array([
            1.0 if c.get("zone") == "battlefield" else 0.0,
            1.0 if c.get("zone") in ("graveyard", "exile") else 0.0,
            1.0 if c.get("targeted") else 0.0,
        ], np.float32)
        return np.concatenate([self.card_emb(c.get("card", "")), flags])


def load_dataset(feat: Featurizer):
    """-> dict of samples per head."""
    cast, atk, blk = [], [], []
    with open(OUT / "pilot_dataset.jsonl") as fh:
        for line in fh:
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = s.get("kind", "cast")
            toks, ids = feat.tokens_and_ids(s)
            sc = feat.scalars(s)
            if kind == "cast":
                cands = s.get("candidates", [])
                if not cands:
                    continue
                proposed = s.get("proposed", [])
                label = len(cands)                      # pass
                if proposed:
                    label = next((i for i, c in enumerate(cands)
                                  if c["card"] == proposed[0]), None)
                    if label is None:
                        continue
                cast.append((toks, sc,
                             [feat.cand_vec(c) for c in cands], label))
            elif kind == "attackers":
                chosen = set(s.get("chosen", []))
                elig = [(k, (side, cid, c)) for k, x in enumerate(ids)
                        if x is not None
                        for side, cid, c in [x]
                        if side == "my" and c.get("cr")
                        and not c.get("tapped")]
                if not elig:
                    continue
                labels = [(k, 1.0 if cid in chosen else 0.0)
                          for k, (_s, cid, _c) in elig]
                atk.append((toks, sc, labels))
            elif kind == "blockers":
                attackers = s.get("attackers", [])
                if not attackers:
                    continue
                a_ids = [a["id"] for a in attackers]
                assign = {b: a for b, a in s.get("assignments", [])}
                # attacker token index in the sequence, by id
                a_tok = {cid: k for k, x in enumerate(ids) if x is not None
                         for _side, cid, _c in [x]}
                if not all(a in a_tok for a in a_ids):
                    continue
                elig = [(k, cid) for k, x in enumerate(ids) if x is not None
                        for side, cid, c in [x]
                        if side == "my" and c.get("cr")
                        and not c.get("tapped")]
                if not elig:
                    continue
                per_blocker = []
                for k, cid in elig:
                    if cid in assign:
                        label = a_ids.index(assign[cid])
                    else:
                        label = len(a_ids)              # no block
                    per_blocker.append((k, [a_tok[a] for a in a_ids], label))
                blk.append((toks, sc, per_blocker))
    return cast, atk, blk


def main():
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    torch.set_num_threads(4)

    feat = Featurizer()
    cast, atk, blk = load_dataset(feat)
    print(f"samples: {len(cast)} cast, {len(atk)} attack, {len(blk)} block")
    n_pass = sum(1 for _t, _s, c, l in cast if l == len(c))
    print(f"cast pass rate: {n_pass / max(1, len(cast)):.0%}")

    # Downsample pass decisions to 4x the act count: balances the cast
    # head and keeps CPU training tractable. Eval reweights not needed -
    # we report act-only accuracy separately.
    acts = [x for x in cast if x[3] != len(x[2])]
    passes = [x for x in cast if x[3] == len(x[2])]
    RNG.shuffle(acts)
    RNG.shuffle(passes)
    acts = acts[:8000]                     # CPU-tractable cap
    cast = acts + passes[:4 * len(acts)]
    print(f"after caps: {len(cast)} cast samples ({len(acts)} act)")

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

        def encode(self, toks, sc):
            x = self.proj(toks).unsqueeze(0)           # [1, n, D]
            x = torch.cat([self.state_tok.unsqueeze(0), x], dim=1)
            h = self.enc(x)[0]                          # [n+1, D]
            hs = self.state_mlp(torch.cat([h[0], sc]))
            return hs, h[1:]                            # state, per-token

        def cast_logits(self, hs, cvs):
            scores = [self.cast_head(torch.cat([hs, self.cand_proj(cv)]))
                      for cv in cvs]
            scores.append(self.pass_head(hs))
            return torch.cat(scores)

    model = Pilot()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tt = lambda x: torch.from_numpy(np.ascontiguousarray(x))

    def split(xs):
        xs = list(xs)
        RNG.shuffle(xs)
        k = max(1, len(xs) // 10)
        return xs[k:], xs[:k]

    cast_tr, cast_te = split(cast)
    atk_tr, atk_te = split(atk)
    blk_tr, blk_te = split(blk)

    def eval_all():
        model.eval()
        res = {}
        with torch.no_grad():
            hits = 0
            act_hits = act_n = 0
            for toks, sc, cvs, label in cast_te:
                hs, _h = model.encode(tt(toks), tt(sc))
                pred = int(model.cast_logits(
                    hs, [tt(c) for c in cvs]).argmax())
                hits += pred == label
                if label != len(cvs):
                    act_n += 1
                    act_hits += pred == label
            res["cast"] = hits / max(1, len(cast_te))
            res["cast_act"] = act_hits / max(1, act_n)
            tp = fp = fn = tn = 0
            for toks, sc, labels in atk_te:
                hs, h = model.encode(tt(toks), tt(sc))
                for k, y in labels:
                    p = float(torch.sigmoid(
                        model.atk_head(torch.cat([hs, h[k]])))) > 0.5
                    tp += p and y
                    fp += p and not y
                    fn += (not p) and y
                    tn += (not p) and (not y)
            res["atk_acc"] = (tp + tn) / max(1, tp + fp + fn + tn)
            res["atk_f1"] = 2 * tp / max(1, 2 * tp + fp + fn)
            bh = bn = 0
            for toks, sc, per_blocker in blk_te:
                hs, h = model.encode(tt(toks), tt(sc))
                for k, a_toks, label in per_blocker:
                    scores = [model.blk_head(torch.cat([hs, h[k], h[ka]]))
                              for ka in a_toks]
                    scores.append(model.noblk_head(torch.cat([hs, h[k]])))
                    bn += 1
                    bh += int(torch.cat(scores).argmax()) == label
            res["blk"] = bh / max(1, bn)
        model.train()
        return res

    for epoch in range(3):
        jobs = ([("c", x) for x in cast_tr] + [("a", x) for x in atk_tr]
                + [("b", x) for x in blk_tr])
        RNG.shuffle(jobs)
        total = 0.0
        for kind, x in jobs:
            if kind == "c":
                toks, sc, cvs, label = x
                hs, _h = model.encode(tt(toks), tt(sc))
                lg = model.cast_logits(hs, [tt(c) for c in cvs])
                loss = nn.functional.cross_entropy(
                    lg.unsqueeze(0), torch.tensor([label]))
            elif kind == "a":
                toks, sc, labels = x
                hs, h = model.encode(tt(toks), tt(sc))
                lg = torch.cat([model.atk_head(torch.cat([hs, h[k]]))
                                for k, _y in labels])
                y = torch.tensor([y for _k, y in labels])
                loss = nn.functional.binary_cross_entropy_with_logits(lg, y)
            else:
                toks, sc, per_blocker = x
                hs, h = model.encode(tt(toks), tt(sc))
                losses = []
                for k, a_toks, label in per_blocker:
                    scores = [model.blk_head(torch.cat([hs, h[k], h[ka]]))
                              for ka in a_toks]
                    scores.append(model.noblk_head(torch.cat([hs, h[k]])))
                    losses.append(nn.functional.cross_entropy(
                        torch.cat(scores).unsqueeze(0),
                        torch.tensor([label])))
                loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        res = eval_all()
        print(f"epoch {epoch}: loss {total / max(1, len(jobs)):.4f} | "
              + " ".join(f"{k} {v:.0%}" for k, v in res.items()), flush=True)

    torch.save(model.state_dict(), OUT / "pilot2.pt")
    res = eval_all()
    print("final:", " ".join(f"{k} {v:.1%}" for k, v in res.items()))
    print(f"wrote {OUT / 'pilot2.pt'}")


if __name__ == "__main__":
    main()
