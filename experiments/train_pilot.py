"""Behavior-clone the built-in AI: the first LEARNED pilot component.

Data: pilot_dataset.jsonl from `pilot_bridge.py --collect` - every
priority decision with the full legal-candidate list and what the
built-in AI chose (possibly "pass").

Model: candidate scoring over frozen MiniLM card embeddings.
  state vec  = [mean(hand), mean(my bf), mean(opp bf)] embeddings
               + scalars (turn, phase one-hot, life diff, counts)
  cand vec   = card embedding + zone/targeted flags
  score(s,c) = MLP(proj_s(s) ++ proj_c(c));  pass gets its own head.
Softmax over {candidates..., pass}, cross-entropy on the AI's choice.

Because features are TEXT embeddings, a card never seen in training
still gets a sensible score - the property that lets the same policy
keep piloting while the evolver mutates the deck.

Prints held-out accuracy vs the always-pass baseline; saves
output/pilot_bc.pt.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(__file__).resolve().parent / "output"
RNG = np.random.default_rng(7)

PHASES = ["MAIN1", "MAIN2", "COMBAT_DECLARE_ATTACKERS", "UPKEEP", "END_OF_TURN"]


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

    def card_vec(self, name: str) -> np.ndarray:
        r = self.rows.get(norm(name))
        return self.emb[r] if r is not None else np.zeros(self.dim, np.float32)

    def pool(self, names) -> np.ndarray:
        if not names:
            return np.zeros(self.dim, np.float32)
        return np.mean([self.card_vec(n) for n in names], axis=0)

    def state_vec(self, s: dict) -> np.ndarray:
        phase = str(s.get("phase", ""))
        ph = [1.0 if p in phase else 0.0 for p in PHASES]
        scalars = np.array([
            min(s.get("turn", 0), 20) / 20.0,
            (s.get("my_life", 20) - s.get("opp_life", 20)) / 20.0,
            len(s.get("my_hand", [])) / 7.0,
            len(s.get("my_battlefield", [])) / 10.0,
            len(s.get("opp_battlefield", [])) / 10.0,
            *ph,
        ], dtype=np.float32)
        return np.concatenate([
            self.pool(s.get("my_hand", [])),
            self.pool(s.get("my_battlefield", [])),
            self.pool(s.get("opp_battlefield", [])),
            scalars,
        ])

    def cand_vec(self, c: dict) -> np.ndarray:
        flags = np.array([
            1.0 if c.get("zone") == "battlefield" else 0.0,
            1.0 if c.get("targeted") else 0.0,
        ], dtype=np.float32)
        return np.concatenate([self.card_vec(c.get("card", "")), flags])


def load_samples(feat: Featurizer):
    """-> list of (state_vec, [cand_vecs], label) where label==len(cands)
    means PASS."""
    samples = []
    with open(OUT / "pilot_dataset.jsonl") as fh:
        for line in fh:
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            cands = s.get("candidates", [])
            if not cands:
                continue
            proposed = s.get("proposed", [])
            if proposed:
                chosen = None
                for i, c in enumerate(cands):
                    if c["card"] == proposed[0]:
                        chosen = i
                        break
                if chosen is None:
                    continue          # proposal outside candidate list
                label = chosen
            else:
                label = len(cands)    # pass
            samples.append((feat.state_vec(s),
                            [feat.cand_vec(c) for c in cands], label))
    return samples


def main():
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    torch.set_num_threads(4)

    feat = Featurizer()
    samples = load_samples(feat)
    print(f"{len(samples)} decisions loaded")
    n_pass = sum(1 for _s, c, l in samples if l == len(c))
    print(f"pass rate: {n_pass / len(samples):.0%} "
          f"(= always-pass baseline accuracy)")

    RNG.shuffle(samples)
    n_test = max(1, len(samples) // 10)
    test, train = samples[:n_test], samples[n_test:]

    sdim = samples[0][0].shape[0]
    cdim = samples[0][1][0].shape[0]
    H = 128

    class Scorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.ps = nn.Sequential(nn.Linear(sdim, H), nn.ReLU())
            self.pc = nn.Sequential(nn.Linear(cdim, H), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(2 * H, H), nn.ReLU(),
                                      nn.Linear(H, 1))
            self.pass_head = nn.Sequential(nn.Linear(H, H), nn.ReLU(),
                                           nn.Linear(H, 1))

        def logits(self, sv, cvs):
            hs = self.ps(sv)
            scores = [self.head(torch.cat([hs, self.pc(cv)])) for cv in cvs]
            scores.append(self.pass_head(hs))
            return torch.cat(scores)

    model = Scorer()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def to_t(x):
        return torch.from_numpy(np.ascontiguousarray(x))

    def accuracy(split):
        hits = 0
        with torch.no_grad():
            for sv, cvs, label in split:
                lg = model.logits(to_t(sv), [to_t(c) for c in cvs])
                hits += int(int(lg.argmax()) == label)
        return hits / len(split)

    for epoch in range(3):
        RNG.shuffle(train)
        total = 0.0
        for sv, cvs, label in train:
            lg = model.logits(to_t(sv), [to_t(c) for c in cvs])
            loss = nn.functional.cross_entropy(
                lg.unsqueeze(0), torch.tensor([label]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        print(f"epoch {epoch}: loss {total / len(train):.4f} "
              f"| held-out acc {accuracy(test):.0%}", flush=True)

    torch.save(model.state_dict(), OUT / "pilot_bc.pt")
    print(f"held-out accuracy: {accuracy(test):.1%} "
          f"vs always-pass {n_pass / len(samples):.1%}")
    print(f"wrote {OUT / 'pilot_bc.pt'}")


if __name__ == "__main__":
    main()
