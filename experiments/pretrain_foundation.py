"""Foundation pretraining: value-first offline training on the full
game corpus, with a built-in hyperparameter search.

Data: every archived game (all rounds, all decks, both seats). Per
cast decision: (state tokens, outcome R, taken action). Objectives:
  value: predict R from the board (primary - the user's north star)
  policy: behavior-clone the taken action (auxiliary, stabilizes reps)

HPO mode (--search): random-search shapes on a fixed data sample,
short budget each, rank by held-out value AUC + cast agreement;
successive halving extends the top quartile. No Forge involved -
minutes per config on the GPU.

Train mode (--train D,LAYERS): full pretraining run of one shape,
saves output/foundation_D{D}L{L}.pt.

Usage:
  python experiments/pretrain_foundation.py --search --configs 20
  python experiments/pretrain_foundation.py --train 384,5 --epochs 3
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path(__file__).resolve().parent / "output"


def iter_games(limit=None, skip_eps_forced=True):
    rows = [json.loads(ln) for ln in
            (OUT / "games_index.jsonl").read_text(encoding="utf-8")
            .splitlines()]
    if limit:
        rng = np.random.default_rng(41)
        rng.shuffle(rows)
        rows = rows[:limit]
    for r in rows:
        try:
            with gzip.open(OUT / "games" / r["file"], "rt",
                           encoding="utf-8") as f:
                g = json.load(f)
        except OSError:
            continue
        yield r, g


def build_dataset(max_games, max_per_game=40, seed=43,
                  roaming_cap=0.3):
    """-> list of (state, reply, R); split by GAME for holdout.

    roaming_cap bounds the fac_roaming share: the pre-round-15 corpus
    is overwhelmingly our-deck perspectives, and a foundation base
    should not inherit that bias."""
    import torch  # noqa: F401  (ensures torch import before featurizer)
    rng = np.random.default_rng(seed)
    train, hold = [], []
    n = 0
    n_roaming = 0
    for r, g in iter_games(limit=max_games * 3):
        if n >= max_games:
            break
        deck = g.get("deck", "fac_roaming")
        if deck == "fac_roaming":
            if n_roaming > roaming_cap * max(20, n):
                continue
            n_roaming += 1
        R = 1.0 if g.get("won") else -1.0
        dec = [(s, rep) for s, rep in g["decisions"]
               if s.get("kind") == "cast" and s.get("candidates")
               and not s.get("_eps_forced")]
        if not dec:
            continue
        if len(dec) > max_per_game:
            keep = rng.choice(len(dec), max_per_game, replace=False)
            dec = [dec[i] for i in sorted(keep)]
        bucket = hold if (n % 10 == 0) else train
        bucket.extend((s, rep, R) for s, rep in dec)
        n += 1
    return train, hold, n


def featurize(samples, feat):
    """Pad into tensors once (model-independent)."""
    N = len(samples)
    max_tok = max_cand = 1
    toks_l, cands_l = [], []
    for s, rep, R in samples:
        t, _ = feat.tokens_and_ids(s)
        c = np.stack([feat.cand_vec(x) for x in s["candidates"]])
        toks_l.append(t)
        cands_l.append(c)
        max_tok = max(max_tok, t.shape[0])
        max_cand = max(max_cand, c.shape[0])
    tdim, cdim = toks_l[0].shape[1], cands_l[0].shape[1]
    toks = np.zeros((N, max_tok, tdim), np.float32)
    tmask = np.ones((N, max_tok), bool)
    cands = np.zeros((N, max_cand, cdim), np.float32)
    cmask = np.zeros((N, max_cand + 1), bool)
    acts = np.zeros(N, np.int64)
    scals = np.zeros((N, feat.scalars({}).shape[0]), np.float32)
    Rs = np.zeros(N, np.float32)
    for k, (s, rep, R) in enumerate(samples):
        t = toks_l[k]
        toks[k, :t.shape[0]] = t
        tmask[k, :t.shape[0]] = False
        c = cands_l[k]
        cands[k, :c.shape[0]] = c
        cmask[k, :c.shape[0]] = True
        cmask[k, max_cand] = True
        scals[k] = feat.scalars(s)
        Rs[k] = R
        cands_meta = s["candidates"]
        if rep.startswith("force\t"):
            i = int(rep.split("\t")[1])
            a = next((j for j, x in enumerate(cands_meta)
                      if x["i"] == i), max_cand)
        elif rep.startswith("veto"):
            a = max_cand
        else:
            prop = s.get("proposed", [])
            a = next((j for j, x in enumerate(cands_meta)
                      if prop and x["card"] == prop[0]), max_cand)
        acts[k] = a if a < c.shape[0] else max_cand
    return dict(toks=toks, tmask=tmask, cands=cands, cmask=cmask,
                acts=acts, scals=scals, Rs=Rs)


def make_model(torch, feat, D, layers):
    import torch.nn as nn
    nh = 8 if D >= 256 else 4
    sdim = feat.scalars({}).shape[0]
    cdim = feat.dim + 3

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(feat.tok_dim, D)
            self.state_tok = nn.Parameter(torch.randn(1, D) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=D, nhead=nh, dim_feedforward=2 * D,
                batch_first=True, dropout=0.1)
            self.enc = nn.TransformerEncoder(layer, num_layers=layers)
            self.state_mlp = nn.Sequential(nn.Linear(D + sdim, D),
                                           nn.ReLU())
            self.cand_proj = nn.Sequential(nn.Linear(cdim, D), nn.ReLU())
            self.cast_head = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(),
                                           nn.Linear(D, 1))
            self.pass_head = nn.Sequential(nn.Linear(D, D), nn.ReLU(),
                                           nn.Linear(D, 1))
            self.val_head = nn.Sequential(nn.Linear(D, D), nn.ReLU(),
                                          nn.Linear(D, 1))
    return Net()


def run_config(torch, feat, data_tr, data_ho, D, layers, lr,
               epochs, dev, val_coef=1.0, log=print):
    model = make_model(torch, feat, D, layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    T = {k: torch.from_numpy(v).to(dev) for k, v in data_tr.items()}
    H = {k: torch.from_numpy(v).to(dev) for k, v in data_ho.items()}
    N = T["Rs"].shape[0]
    BS = 1024

    def forward(S, sl, m):
        x = m.proj(S["toks"][sl])
        st = m.state_tok.expand(x.shape[0], 1, -1)
        x = torch.cat([st, x], dim=1)
        pad = torch.cat([torch.zeros(x.shape[0], 1, dtype=torch.bool,
                                     device=dev), S["tmask"][sl]], dim=1)
        h = m.enc(x, src_key_padding_mask=pad)
        hs = m.state_mlp(torch.cat([h[:, 0], S["scals"][sl]], dim=1))
        cp = m.cand_proj(S["cands"][sl])
        hse = hs.unsqueeze(1).expand(-1, cp.shape[1], -1)
        lc = m.cast_head(torch.cat([hse, cp], dim=2)).squeeze(-1)
        logits = torch.cat([lc, m.pass_head(hs)], dim=1)
        logits = logits.masked_fill(~S["cmask"][sl], -1e9)
        return logits, m.val_head(hs).squeeze(-1)

    for ep in range(epochs):
        model.train()
        order = torch.randperm(N, device=dev)
        for o in range(0, N, BS):
            sl = order[o:o + BS]
            logits, v = forward(T, sl, model)
            pl = torch.nn.functional.cross_entropy(logits, T["acts"][sl])
            vl = ((v - T["Rs"][sl]) ** 2).mean()
            (pl + val_coef * vl).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
    # holdout metrics
    model.eval()
    hits = tot = 0
    vs, rs = [], []
    with torch.no_grad():
        M = H["Rs"].shape[0]
        for o in range(0, M, BS):
            sl = slice(o, o + BS)
            logits, v = forward(H, sl, model)
            hits += int((logits.argmax(1) == H["acts"][sl]).sum())
            tot += logits.shape[0]
            vs.append(v)
            rs.append(H["Rs"][sl])
    v = torch.cat(vs)
    r = torch.cat(rs)
    vloss = float(((v - r) ** 2).mean())
    # value AUC: does V rank winning states above losing ones?
    wins = v[r > 0]
    losses = v[r < 0]
    if len(wins) and len(losses):
        auc = float((wins.unsqueeze(1) > losses.unsqueeze(0))
                    .float().mean())
    else:
        auc = 0.5
    return model, {"acc": hits / max(1, tot), "vloss": vloss,
                   "auc": auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--configs", type=int, default=16)
    ap.add_argument("--search-games", type=int, default=3000)
    ap.add_argument("--train", default=None, help="D,LAYERS")
    ap.add_argument("--train-games", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    import torch
    from train_pilot2 import Featurizer
    feat = Featurizer()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.search:
        t0 = time.time()
        tr, ho, ng = build_dataset(args.search_games)
        print(f"search data: {ng} games, {len(tr)} train / "
              f"{len(ho)} holdout decisions "
              f"({time.time()-t0:.0f}s)", flush=True)
        t0 = time.time()
        Ftr = featurize(tr, feat)
        Fho = featurize(ho, feat)
        print(f"featurized in {time.time()-t0:.0f}s", flush=True)
        rng = np.random.default_rng(47)
        Ds = [128, 192, 256, 384, 512]
        Ls = [2, 3, 4, 5, 6]
        lrs = [1e-4, 3e-4, 1e-3]
        cfgs = []
        seen = set()
        while len(cfgs) < args.configs:
            c = (int(rng.choice(Ds)), int(rng.choice(Ls)),
                 float(rng.choice(lrs)))
            if c not in seen:
                seen.add(c)
                cfgs.append(c)
        results = []
        for D, L, lr in cfgs:
            t0 = time.time()
            _m, met = run_config(torch, feat, Ftr, Fho, D, L, lr,
                                 epochs=1, dev=dev)
            met.update(D=D, L=L, lr=lr, secs=round(time.time() - t0))
            results.append(met)
            print(json.dumps(met), flush=True)
        # halving: top quartile gets 2 more epochs
        results.sort(key=lambda m: -(m["auc"] + m["acc"]))
        finalists = results[:max(2, len(results) // 4)]
        print("== finalists, extended ==", flush=True)
        for m in finalists:
            t0 = time.time()
            _mm, met = run_config(torch, feat, Ftr, Fho, m["D"], m["L"],
                                  m["lr"], epochs=3, dev=dev)
            met.update(D=m["D"], L=m["L"], lr=m["lr"],
                       secs=round(time.time() - t0))
            print("FINAL " + json.dumps(met), flush=True)
        return

    if args.train:
        D, L = (int(x) for x in args.train.split(","))
        tr, ho, ng = build_dataset(args.train_games, max_per_game=60)
        print(f"train data: {ng} games, {len(tr)}/{len(ho)} decisions",
              flush=True)
        Ftr = featurize(tr, feat)
        Fho = featurize(ho, feat)
        model, met = run_config(torch, feat, Ftr, Fho, D, L, 3e-4,
                                epochs=args.epochs, dev=dev)
        print("metrics:", json.dumps(met), flush=True)
        out = OUT / f"foundation_D{D}L{L}.pt"
        torch.save(model.to("cpu").state_dict(), out)
        print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
