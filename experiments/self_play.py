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


def decision_logp(model_policy, torch, state, reply):
    """log-prob of the taken action under the CURRENT policy, plus the
    state encoding hs (for the value head) and the chosen cast card
    (for lock credit). Returns (logp, hs, chosen) or None."""
    import numpy as _np
    model = model_policy.model
    kind = state.get("kind")
    if kind == "cast":
        cands = state.get("candidates", [])
        if not cands:
            return None
        hs, _h, _ids = model_policy.encode(state)
        logits = torch.cat(
            [model.cast_head(torch.cat([hs, model.cand_proj(
                model_policy.tt(model_policy.feat.cand_vec(c)))]))
             for c in cands] + [model.pass_head(hs)])
        if reply.startswith("force\t"):
            idx = int(reply.split("\t")[1])
            action = next((k for k, c in enumerate(cands)
                           if c["i"] == idx), None)
        elif reply.startswith("veto"):
            action = len(cands)
        else:
            proposed = state.get("proposed", [])
            action = next((k for k, c in enumerate(cands)
                           if proposed and c["card"] == proposed[0]),
                          len(cands))
        if action is None:
            return None
        chosen = cands[action]["card"] if action < len(cands) else None
        return torch.log_softmax(logits, dim=0)[action], hs, chosen
    if kind == "target":
        cands = state.get("candidates", [])
        if not cands or not hasattr(model, "tgt_head"):
            return None
        hs, _h, _ids = model_policy.encode(state)
        hostv = model.cand_proj(model_policy.tt(
            model_policy.feat.cand_vec({"card": state.get("host", "")})))
        logits = torch.cat([model.tgt_head(torch.cat(
            [hs, model.tgt_proj(model_policy.tt(
                model_policy.feat.tgt_vec(c))), hostv])) for c in cands])
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
        return torch.log_softmax(logits, dim=0)[action], hs, None
    if kind == "mulligan":
        if not hasattr(model, "mull_head") or reply not in ("keep", "mull"):
            return None
        hs, _h, _ids = model_policy.encode(state)
        dv = model_policy.feat.deck_emb(state.get("player", ""))
        if dv is None:
            dv = _np.zeros(model_policy.feat.dim, _np.float32)
        dk = model.deck_proj(model_policy.tt(dv))
        ex = torch.cat([
            torch.tensor([state.get("cards_to_return", 0) / 7.0],
                         device=hs.device),
            model_policy.tt(model_policy.feat.hand_feats(state))])
        lg = model.mull_head(torch.cat([hs, dk, ex]))[0]
        logp = -torch.nn.functional.binary_cross_entropy_with_logits(
            lg, torch.tensor(1.0 if reply == "keep" else 0.0,
                             device=lg.device))
        return logp, hs, None
    if kind == "attackers" and reply.startswith("attack\t"):
        hs, h, ids = model_policy.encode(state)
        want = {int(x) for x in reply[7:].split(",") if x}
        logps = []
        for k, x in enumerate(ids):
            if x is None:
                continue
            side, cid, c = x
            if side != "my" or not c.get("cr") or c.get("tapped"):
                continue
            lg = model.atk_head(torch.cat([hs, h[k]]))[0]
            logps.append(-torch.nn.functional
                         .binary_cross_entropy_with_logits(
                             lg, torch.tensor(
                                 1.0 if cid in want else 0.0,
                                 device=lg.device)))
        if not logps:
            return None
        return torch.stack(logps).sum(), hs, None
    if kind == "blockers" and reply.startswith("block\t"):
        attackers = state.get("attackers", [])
        if not attackers:
            return None
        hs, h, ids = model_policy.encode(state)
        a_ids = [a["id"] for a in attackers]
        a_tok = {x[1]: k for k, x in enumerate(ids)
                 if x is not None and x[1] in set(a_ids)}
        if len(a_tok) != len(a_ids):
            return None
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
            if side != "my" or not c.get("cr") or c.get("tapped"):
                continue
            scores = torch.cat(
                [model.blk_head(torch.cat([hs, h[k], h[a_tok[a]]]))
                 for a in a_ids]
                + [model.noblk_head(torch.cat([hs, h[k]]))])
            action = (a_ids.index(chosen[cid])
                      if cid in chosen else len(a_ids))
            logps.append(torch.log_softmax(scores, dim=0)[action])
        if not logps:
            return None
        return torch.stack(logps).sum(), hs, None
    return None


def ppo_update_gpu(model_policy, torch, batch, opt, epochs=3, clip=0.2,
                   lock_credit=None, val_coef=0.5, max_decisions=None,
                   device="cuda"):
    """Batched PPO on the GPU for CAST decisions (~90% of the batch):
    featurize once, pad into big tensors, few large forward passes per
    epoch. Non-cast kinds (combat/target/mulligan) fall back to the
    per-decision CPU path afterward with a single extra epoch.
    Returns (policy_loss, value_loss, mean_V)."""
    import numpy as np
    model = model_policy.model
    feat = model_policy.feat
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # ---- gather + featurize once (model-independent) ----------------
    casts, others = [], []
    for dec, R in batch:
        for s, r in dec:
            kind = s.get("kind")
            if kind == "cast":
                # candidate-less cast states are unscoreable no-ops
                # (decision_logp returns None) - routing them to the
                # CPU fallback wasted 2 forward passes each
                if s.get("candidates"):
                    casts.append((s, r, R))
            elif kind != "game_end":
                others.append((s, r, R))
    if max_decisions and len(casts) > max_decisions:
        idx = np.random.default_rng(0).choice(
            len(casts), max_decisions, replace=False)
        casts = [casts[i] for i in sorted(idx)]

    N = len(casts)
    if N == 0:
        return ppo_update(model_policy, torch, batch, opt, epochs=epochs,
                          clip=clip, lock_credit=lock_credit,
                          val_coef=val_coef, max_decisions=max_decisions)
    tok_list, scal_list, cand_list, act_list, extra_adv, rewards = \
        [], [], [], [], [], []
    max_tok = 1
    max_cand = 1
    for s, r, R in casts:
        toks, _ids = feat.tokens_and_ids(s)
        tok_list.append(toks)
        scal_list.append(feat.scalars(s))
        cands = s["candidates"]
        cvecs = np.stack([feat.cand_vec(c) for c in cands])
        cand_list.append(cvecs)
        max_tok = max(max_tok, toks.shape[0])
        max_cand = max(max_cand, len(cands))
        if r.startswith("force\t"):
            i = int(r.split("\t")[1])
            a = next((k for k, c in enumerate(cands) if c["i"] == i),
                     len(cands))
        elif r.startswith("veto"):
            a = len(cands)
        else:
            prop = s.get("proposed", [])
            a = next((k for k, c in enumerate(cands)
                      if prop and c["card"] == prop[0]), len(cands))
        act_list.append(a)
        ex = 0.0
        if lock_credit and a < len(cands) \
                and cands[a]["card"] in lock_credit["locks"]:
            t = s.get("turn", 16)
            ex = lock_credit["w"] * max(0.0, min(1.0, (16 - t) / 12.0))
        extra_adv.append(ex)
        rewards.append(R)

    tdim = tok_list[0].shape[1]
    cdim = cand_list[0].shape[1]
    toks = np.zeros((N, max_tok, tdim), np.float32)
    tmask = np.ones((N, max_tok), bool)          # True = PAD
    cands_t = np.zeros((N, max_cand, cdim), np.float32)
    cmask = np.zeros((N, max_cand + 1), bool)    # True = real option
    for k in range(N):
        nt = tok_list[k].shape[0]
        toks[k, :nt] = tok_list[k]
        tmask[k, :nt] = False
        nc = cand_list[k].shape[0]
        cands_t[k, :nc] = cand_list[k]
        cmask[k, :nc] = True
        cmask[k, max_cand] = True                # pass option, always last
    acts = np.array([a if a < cand_list[k].shape[0] else max_cand
                     for k, a in enumerate(act_list)], np.int64)

    def _opt_to(o, d):
        for st in o.state.values():
            for k2, v2 in st.items():
                if torch.is_tensor(v2):
                    st[k2] = v2.to(d)

    model.to(dev)
    _opt_to(opt, dev)
    # big tensors stay on the CPU; each minibatch slice streams to the
    # GPU inside forward_slice (a single whole-batch upload can be
    # multiple GB and sporadically OOMs under Windows/WDDM even with
    # free VRAM)
    toks = torch.from_numpy(toks)
    tmask = torch.from_numpy(tmask)
    scals = torch.from_numpy(np.stack(scal_list))
    cands_t = torch.from_numpy(cands_t)
    cmask = torch.from_numpy(cmask)
    acts_t = torch.from_numpy(acts).to(dev)
    R_t = torch.tensor(rewards, dtype=torch.float32, device=dev)
    ex_t = torch.tensor(extra_adv, dtype=torch.float32, device=dev)
    D = model.state_tok.shape[1]

    BS = 1024           # attention activations spike ~4 GB at 2048 on
                        # pool-sized boards; WDDM refuses large asks

    def forward_slice(sl):
        """-> (logp_taken, V) for one minibatch slice (grad respected
        by caller's context)."""
        x = model.proj(toks[sl].to(dev))
        st = model.state_tok.expand(x.shape[0], 1, D)
        x = torch.cat([st, x], dim=1)
        pad = torch.cat([torch.zeros(x.shape[0], 1, dtype=torch.bool,
                                     device=dev),
                         tmask[sl].to(dev)], dim=1)
        h = model.enc(x, src_key_padding_mask=pad)
        hs = model.state_mlp(torch.cat([h[:, 0], scals[sl].to(dev)],
                                       dim=1))
        cp = model.cand_proj(cands_t[sl].to(dev))
        hse = hs.unsqueeze(1).expand(-1, cp.shape[1], -1)
        logits_c = model.cast_head(
            torch.cat([hse, cp], dim=2)).squeeze(-1)
        logit_p = model.pass_head(hs)
        logits = torch.cat([logits_c, logit_p], dim=1)
        logits = logits.masked_fill(~cmask[sl].to(dev), -1e9)
        lp = torch.log_softmax(logits, dim=1)
        return (lp.gather(1, acts_t[sl].unsqueeze(1)).squeeze(1),
                model.val_head(hs).squeeze(-1))

    model.eval()
    olds = []
    with torch.no_grad():
        for o in range(0, N, BS):
            lp, _v = forward_slice(slice(o, o + BS))
            olds.append(lp)
    old_lp = torch.cat(olds).detach()
    ptot = vtot = vsum = 0.0
    gnorms = []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        ep_p = ep_v = ep_m = 0.0
        # backward PER MINIBATCH: one slice's graph in memory at a time
        for o in range(0, N, BS):
            sl = slice(o, o + BS)
            lp, v = forward_slice(sl)
            adv = (R_t[sl] - v.detach()) + ex_t[sl]
            ratio = torch.exp(lp - old_lp[sl])
            pl = -torch.min(ratio * adv,
                            torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
            vl = (v - R_t[sl]) ** 2
            w = lp.shape[0] / N
            ((pl.mean() + val_coef * vl.mean()) * w).backward()
            ep_p += float(pl.mean().detach()) * w
            ep_v += float(vl.mean().detach()) * w
            ep_m += float(v.mean().detach()) * w
        gnorms.append(float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0)))
        opt.step()
        ptot += ep_p
        vtot += ep_v
        vsum += ep_m
    # restore the model (and optimizer state) to wherever serving runs
    # (PILOT_DEVICE=cuda keeps everything on the GPU)
    home = getattr(model_policy, "dev", torch.device("cpu"))
    model.to(home)
    _opt_to(opt, home)
    if dev.type == "cuda" and home.type != "cuda":
        torch.cuda.empty_cache()
    # non-cast kinds: one pass of the per-decision CPU path
    if others:
        rebatch = {}
        for s, r, R in others:
            rebatch.setdefault(R, []).append((s, r))
        small = [(dec, R) for R, dec in rebatch.items()]
        _p, _v, _m, ognorms = ppo_update(
            model_policy, torch, small, opt, epochs=1, clip=clip,
            lock_credit=None, val_coef=val_coef)
        gnorms.extend(ognorms)
    return ptot / epochs, vtot / epochs, vsum / epochs, gnorms


def ppo_update(model_policy, torch, batch, opt, epochs=3, clip=0.2,
               lock_credit=None, val_coef=0.5, max_decisions=None,
               potential=0.0):
    """PPO-style: value-head baseline (advantage = R - V(s)) + clipped
    multi-epoch replay of the same batch. batch: [(decisions, R)].
    Returns (policy_loss, value_loss, mean_V)."""
    import numpy as np
    model = model_policy.model
    # flatten to (state, reply, R, traj, pos); potential-based shaping
    # needs trajectory adjacency, so subsample whole trajectories
    if max_decisions:
        total = sum(len(dec) for dec, _R in batch)
        if total > max_decisions:
            rng = np.random.default_rng(0)
            order = rng.permutation(len(batch))
            kept, cnt = [], 0
            for ti in order:
                kept.append(batch[ti])
                cnt += len(batch[ti][0])
                if cnt >= max_decisions:
                    break
            batch = kept
    flat = [(s, r, R, ti, pi) for ti, (dec, R) in enumerate(batch)
            for pi, (s, r) in enumerate(dec)]
    traj_len = {ti: len(dec) for ti, (dec, _R) in enumerate(batch)}
    # epoch 0 pass: old log-probs + frozen state-values (the potentials)
    model.eval()
    old, vold = [], []
    with torch.no_grad():
        for s, r, R, ti, pi in flat:
            out = decision_logp(model_policy, torch, s, r)
            if out is None:
                old.append(None)
                vold.append(None)
            else:
                old.append(float(out[0]))
                vold.append(float(model.val_head(out[1])[0]))
    # shaping[k] = potential * (V(next state) - V(state)), frozen V;
    # terminal transition uses the actual outcome R as the final value
    shaping = [0.0] * len(flat)
    if potential:
        for k, (s, r, R, ti, pi) in enumerate(flat):
            if vold[k] is None:
                continue
            if pi + 1 < traj_len[ti] and k + 1 < len(flat) \
                    and vold[k + 1] is not None and flat[k + 1][3] == ti:
                shaping[k] = potential * (vold[k + 1] - vold[k])
            else:
                shaping[k] = potential * (R - vold[k])
    ptot = vtot = vsum = vn = 0.0
    gnorms = []
    n_est = max(1, sum(1 for x in old if x is not None))
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        nterms = 0
        for k, (s, r, R, ti, pi) in enumerate(flat):
            if old[k] is None:
                continue
            try:
                out = decision_logp(model_policy, torch, s, r)
                if out is None:
                    continue
                logp, hs, chosen = out
                v = model.val_head(hs)[0]
                adv = R - float(v.detach()) + shaping[k]
                if lock_credit and chosen is not None \
                        and chosen in lock_credit["locks"]:
                    t = s.get("turn", 16)
                    adv += lock_credit["w"] * max(
                        0.0, min(1.0, (16 - t) / 12.0))
                ratio = torch.exp(logp - old[k])
                pl = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
                vl = (v - R) ** 2
                # one averaged step per epoch (like the GPU path):
                # stepping every 64 summed decisions gave this path
                # ~8-64x the update mass of the GPU path and let
                # win-biased combat states whipsaw the value head
                ((pl + val_coef * vl) / n_est).backward()
                ptot += float(pl.detach())
                vtot += float(vl.detach())
                vsum += float(v.detach())
                vn += 1
                nterms += 1
            except Exception:
                continue
        gnorms.append(float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0)))
        opt.step()
    model.eval()
    n = max(1.0, vn)
    return ptot / n, vtot / n, vsum / n, gnorms


def reinforce_update(model_policy, torch, batch, baseline, lr_opt,
                     lock_credit=None):
    """batch: list of (decisions, reward). Recompute log-probs and
    ascend reward-weighted likelihood.

    lock_credit: optional {"locks": set, "w": float} - adds w*(16-t)/12
    directly to the advantage of THE decision that cast a locked card
    (per-decision credit; the trajectory-level bonus alone spreads over
    ~230 decisions and moves a suppressed prior glacially)."""
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
                    if lock_credit and action < len(cands) \
                            and cands[action]["card"] in lock_credit["locks"]:
                        t = state.get("turn", 16)
                        extra = lock_credit["w"] * max(
                            0.0, min(1.0, (16 - t) / 12.0))
                        loss = -extra * logp
                        loss.backward(retain_graph=True)
                        total_loss += float(loss.detach())
                elif kind == "mulligan":
                    if not hasattr(model, "mull_head") \
                            or reply not in ("keep", "mull"):
                        continue
                    hs, _h, _ids = model_policy.encode(state)
                    import numpy as _np
                    dv = model_policy.feat.deck_emb(state.get("player", ""))
                    if dv is None:
                        dv = _np.zeros(model_policy.feat.dim, _np.float32)
                    dk = model.deck_proj(model_policy.tt(dv))
                    ex = torch.cat([
                        torch.tensor(
                            [state.get("cards_to_return", 0) / 7.0],
                            device=hs.device),
                        model_policy.tt(
                            model_policy.feat.hand_feats(state))])
                    lg = model.mull_head(torch.cat([hs, dk, ex]))[0]
                    logp = -torch.nn.functional \
                        .binary_cross_entropy_with_logits(
                            lg, torch.tensor(
                                1.0 if reply == "keep" else 0.0,
                                device=lg.device))
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
                                lg, torch.tensor(y, device=lg.device)))
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
