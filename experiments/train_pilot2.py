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
    def __init__(self, deck_ctx: bool = False):
        self.emb = np.load(OUT / "embeddings.npy").astype(np.float32)
        meta = json.loads((OUT / "cards_meta.json").read_text())
        self.rows = {}
        for i, m in enumerate(meta):
            self.rows.setdefault(norm(m["name"]), i)
        self.dim = self.emb.shape[1]
        # zones: hand, my bf, opp bf, my graveyard, opp graveyard
        self.tok_dim = self.dim + 5 + 5      # zone one-hot + state feats
        # deck context (round 3+): one extra token per state = the
        # count-weighted mean embedding of the seat's decklist, so the
        # net can judge a hand RELATIVE to the deck's plan (mulligans)
        self.deck_ctx = deck_ctx
        self._deck_cache: dict = {}

    def deck_emb(self, player: str) -> np.ndarray | None:
        """player 'Ai(1)-fac_roaming' -> mean embedding of fac_roaming.dck"""
        deck = re.sub(r"^Ai\(\d\)-", "", player or "")
        if not deck:
            return None
        if deck not in self._deck_cache:
            path = Path(__file__).resolve().parent / "decks" / f"{deck}.dck"
            vec = None
            if path.exists():
                vecs, weights = [], []
                for ln in path.read_text().splitlines():
                    m = re.match(r"(\d+)\s+(.+)", ln.strip())
                    if m:
                        vecs.append(self.card_emb(m.group(2)))
                        weights.append(int(m.group(1)))
                if vecs:
                    vec = np.average(np.stack(vecs), axis=0,
                                     weights=weights).astype(np.float32)
            self._deck_cache[deck] = vec
        return self._deck_cache[deck]

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
        if self.deck_ctx:
            dv = self.deck_emb(s.get("player", ""))
            if dv is not None:
                toks.append(np.concatenate(
                    [dv, np.zeros(10, np.float32)]))
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

    def tgt_vec(self, c: dict) -> np.ndarray:
        """Protocol v4 target candidate: card embedding (zeros for a
        player) + [is_player, mine, is_creature, p, t, life]."""
        flags = np.array([
            1.0 if c.get("kind") == "player" else 0.0,
            1.0 if c.get("mine") else 0.0,
            1.0 if c.get("cr") else 0.0,
            c.get("p", 0) / 10.0,
            c.get("t", 0) / 10.0,
            c.get("life", 0) / 20.0,
        ], np.float32)
        emb = (self.card_emb(c.get("n", ""))
               if c.get("kind") == "card"
               else np.zeros(self.dim, np.float32))
        return np.concatenate([emb, flags])


def load_dataset(feat: Featurizer):
    """-> samples per head. Streams with bounded memory: ALL combat
    lines, ALL cast lines in the file's tail (recent DAgger rounds -
    the on-policy data that matters most), and a reservoir sample of
    older cast lines. Featurizing every line of a 300k+ dataset at
    once flirts with OOM on small boxes."""
    import os
    path = OUT / "pilot_dataset.jsonl"
    total = sum(1 for _ in open(path, "rb"))
    tail_n = int(os.environ.get("TP2_TAIL", 40000))
    old_cap = int(os.environ.get("TP2_OLD_CAST", 70000))
    tail_start = max(0, total - tail_n)
    rest, old_cast, old_seen = [], [], 0
    with open(path) as fh:
        for li, line in enumerate(fh):
            is_combat = '"kind":"attackers"' in line \
                or '"kind":"blockers"' in line
            if is_combat or li >= tail_start:
                rest.append(line)
            else:
                old_seen += 1
                if len(old_cast) < old_cap:
                    old_cast.append(line)
                elif RNG.random() < old_cap / old_seen:
                    old_cast[int(RNG.integers(0, old_cap))] = line
    kept = old_cast + rest
    print(f"loaded {len(kept)}/{total} lines "
          f"(tail {tail_n} + old-cast cap {old_cap} + all combat)")
    cast, atk, blk = [], [], []
    if True:
        for line in kept:
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
    import os
    import torch
    import torch.nn as nn
    sidecar = OUT / "pilot2_train_state.json"
    target_epochs = int(os.environ.get("TP2_EPOCHS", 3))
    done = 0
    if os.environ.get("TP2_RESUME") == "1" and sidecar.exists():
        done = json.loads(sidecar.read_text()).get("epochs_done", 0)
        if done >= target_epochs and (OUT / "pilot2.pt").exists():
            print(f"already trained ({done}/{target_epochs} epochs) - skip")
            return
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
    import os
    acts = [x for x in cast if x[3] != len(x[2])]
    passes = [x for x in cast if x[3] == len(x[2])]
    RNG.shuffle(acts)
    RNG.shuffle(passes)
    acts = acts[:int(os.environ.get("TP2_ACT_CAP", 8000))]
    ratio = int(os.environ.get("TP2_PASS_RATIO", 4))
    cast = acts + passes[:ratio * len(acts)]
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
    if done > 0 and (OUT / "pilot2.pt").exists():
        model.load_state_dict(torch.load(OUT / "pilot2.pt"))
        print(f"resumed weights at epoch {done}")
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

    steps_done = 0
    if os.environ.get("TP2_RESUME") == "1" and sidecar.exists():
        steps_done = json.loads(sidecar.read_text()).get("steps_done", 0)
    for epoch in range(done, target_epochs):
        jobs = ([("c", x) for x in cast_tr] + [("a", x) for x in atk_tr]
                + [("b", x) for x in blk_tr])
        # per-epoch deterministic shuffle so a mid-epoch resume sees the
        # same order and can skip already-trained steps
        np.random.default_rng(1000 + epoch).shuffle(jobs)
        total = 0.0
        for ji, (kind, x) in enumerate(jobs):
            if ji < steps_done:
                continue
            if ji % 2000 == 1999:
                # intra-epoch checkpoint: restarts cost minutes, not epochs
                torch.save(model.state_dict(), OUT / "pilot2.pt")
                sidecar.write_text(json.dumps(
                    {"epochs_done": epoch, "steps_done": ji + 1}))
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
        # restart-proof: checkpoint every epoch + progress sidecar
        torch.save(model.state_dict(), OUT / "pilot2.pt")
        sidecar.write_text(json.dumps(
            {"epochs_done": epoch + 1, "steps_done": 0}))
        steps_done = 0

    torch.save(model.state_dict(), OUT / "pilot2.pt")
    res = eval_all()
    print("final:", " ".join(f"{k} {v:.1%}" for k, v in res.items()))
    print(f"wrote {OUT / 'pilot2.pt'}")


if __name__ == "__main__":
    main()
