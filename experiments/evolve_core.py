"""Evolution engine v2 - the methodology the Tainted Aether run taught us.

Lessons encoded here (see FORGE-NOTES.md):
- Self-play mirrors reward whoever adds creatures and attacks; a lock
  deck "improves" by deleting its own win condition. So variants are
  scored against a FIXED EXTERNAL GAUNTLET, never against each other.
- MTGO-meta gauntlets crush casual decks 100-0 (no gradient). The
  gauntlet is picked by PROBING nearest corpus neighbors and keeping
  the ones this deck beats 10-90% of the time - opponents that are
  actually informative.
- Baseline first: many games vs the gauntlet -> winrate + log-derived
  stats. Accept a swap only when its gauntlet winrate beats the
  incumbent's baseline by a margin; re-baseline after each accept.
- Hard locks: cards the evolver may never cut (win conditions the AI
  undervalues - the user knows the deck's plan better than the sim).
- Stats from verbose logs: ramp curve, game length, mulligans, and
  per-locked-card play counts / first-cast turns, so a human can see
  HOW games are going, not just the winrate (and judge AI-pilot bias).

The engine is synchronous and event-driven (on_event callback) so it
can run under a CLI, a server thread, or a notebook equally well.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from improve_deck import (Improver, norm, write_dck, run_match,  # noqa: E402
                          ci95, FORGE_DIR)
from verify_combo import run_verbose  # noqa: E402
from mtg_deckbuilder.collection import parse_collection  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


# ---------------- verbose-log statistics ----------------

def parse_games(log: str, me: str, tracked: list[str]) -> list[dict]:
    """Per-game stats for player `me` from a verbose sim log."""
    games, prev = [], 0
    for m in re.finditer(r"Game Result: Game \d+ ended[^\n]*", log):
        seg, line = log[prev:m.start()], m.group(0)
        prev = m.end()
        me_re = re.escape(me)
        turns = len(re.findall(rf"Turn: Turn \d+ \(Ai\(\d\)-{me_re}\)", seg))
        lands_turn = []          # cumulative lands after my turn t
        lands = 0
        cur_turn = 0
        for lm in re.finditer(
                rf"Turn: Turn (\d+) \(Ai\(\d\)-{me_re}\)"
                rf"|Land: Ai\(\d\)-{me_re} played", seg):
            if lm.group(1):
                while len(lands_turn) < cur_turn:
                    lands_turn.append(lands)
                cur_turn = int(lm.group(1))
            else:
                lands += 1
        while len(lands_turn) < cur_turn:
            lands_turn.append(lands)
        casts = {}
        first = {}
        for name in tracked:
            hits = []
            for cm in re.finditer(
                    rf"Add To Stack: Ai\(\d\)-{me_re} cast {re.escape(name)}"
                    rf"|Land: Ai\(\d\)-{me_re} played {re.escape(name)}", seg):
                t_before = len(re.findall(
                    rf"Turn: Turn \d+ \(Ai\(\d\)-{me_re}\)", seg[:cm.start()]))
                hits.append(max(1, t_before))
            casts[name] = len(hits)
            if hits:
                first[name] = hits[0]
        km = re.search(rf"Ai\(\d\)-{me_re} has kept a hand of (\d+)", seg)
        wm = re.search(r"Ai\(\d\)-(\S+) has won", line)
        games.append({
            "won": bool(wm and wm.group(1) == me),
            "draw": wm is None,
            "turns": turns,
            "lands_by_turn": lands_turn[:8],
            "casts": casts,
            "first_cast": first,
            "kept": int(km.group(1)) if km else 7,
        })
    return games


def aggregate_stats(games: list[dict], tracked: list[str]) -> dict:
    if not games:
        return {}
    real = [g for g in games if not g["draw"]]
    def lands_at(g, t):
        lt = g["lands_by_turn"]
        return lt[min(t, len(lt)) - 1] if lt else 0
    out = {
        "games": len(games),
        "wins": sum(g["won"] for g in games),
        "avg_turns": round(float(np.mean([g["turns"] for g in real])), 1)
        if real else 0,
        "avg_lands_t3": round(float(np.mean([lands_at(g, 3) for g in games])), 2),
        "avg_lands_t5": round(float(np.mean([lands_at(g, 5) for g in games])), 2),
        "mulligan_rate": round(
            sum(g["kept"] < 7 for g in games) / len(games), 2),
        "tracked": {},
    }
    for name in tracked:
        played = [g for g in games if g["casts"].get(name, 0) > 0]
        out["tracked"][name] = {
            "played_in": f"{len(played)}/{len(games)}",
            "avg_casts": round(float(np.mean(
                [g["casts"].get(name, 0) for g in games])), 2),
            "median_first_turn": int(np.median(
                [g["first_cast"][name] for g in played])) if played else None,
        }
    return out


# ---------------- the engine ----------------

class Evolution:
    """One evolution job. Drive with run(); observe via on_event."""

    def __init__(self, deck_text: str, fmt: str, locks: list[str],
                 generations: int = 8, margin: float = 0.03,
                 probe_games: int = 6, baseline_games: int = 24,
                 stage1_games: int = 4, stage2_games: int = 12,
                 candidates: int = 6, gauntlet_size: int = 3,
                 name: str = "job", on_event=None):
        self.on_event = on_event or (lambda e: None)
        self.imp = Improver(fmt)
        self.fmt = fmt
        self.locks = {norm(n) for n in locks}
        self.lock_names = locks
        self.generations = generations
        self.margin = margin
        self.probe_games = probe_games
        self.baseline_games = baseline_games
        self.stage1_games = stage1_games
        self.stage2_games = stage2_games
        self.candidates = candidates
        self.gauntlet_size = gauntlet_size
        self.name = name
        self.stop_flag = False

        self.deck = {}
        coll = parse_collection(deck_text)
        for n, q in coll.items():
            row = self.imp.name_to_row.get(norm(n))
            if row is None:
                raise ValueError(f"unknown card: {n}")
            if not self.imp.playable(row):
                raise ValueError(f"not Forge-playable: {n}")
            self.deck[row] = self.deck.get(row, 0) + q
        for n in locks:
            if norm(n) not in {norm(x) for x, _q in self.pairs()}:
                raise ValueError(f"locked card not in deck: {n}")

        self.history = []
        self.baseline = None
        self.tried = set()
        self.gauntlet = []           # list of (label, deck name pairs)

        # Cards Forge's AI is hard-forbidden from playing
        # (AI:RemoveDeck:All in the card scripts). The sim treats them
        # as blanks; winrates measure the deck WITHOUT them.
        blind_path = (Path(__file__).resolve().parent.parent / "data"
                      / "forge-ai-unplayable.json")
        blind = ({norm(n) for n in json.loads(blind_path.read_text())}
                 if blind_path.exists() else set())
        self.ai_blind = sorted(
            n for n, _q in self.pairs() if norm(n) in blind)

    # -- helpers ---------------------------------------------------------
    def pairs(self, deck=None):
        return sorted((self.imp.meta[r]["name"], q)
                      for r, q in (deck or self.deck).items())

    def emit(self, type_, msg, **data):
        self.on_event({"t": time.time(), "type": type_, "msg": msg, **data})

    def check_stop(self):
        if self.stop_flag:
            raise InterruptedError("stopped")

    # -- phases ----------------------------------------------------------
    def pick_gauntlet(self):
        """Probe nearest corpus neighbors; keep informative opponents."""
        nl = [(r, q) for r, q in self.deck.items() if not self.imp.is_land[r]]
        v = sum(self.imp.vecs[r] * q for r, q in nl)
        v /= max(np.linalg.norm(v), 1e-9)
        sims = self.imp.centroids @ v
        order = np.argsort(-sims)
        write_dck(f"evo_{self.name}_base", self.pairs())
        chosen, probed = [], 0
        for idx in order:
            if len(chosen) >= self.gauntlet_size or probed >= 12:
                break
            self.check_stop()
            d = self.imp.corpus[int(idx)]
            if not all(norm(n) in self.imp.supported for n, _q in d["main"]):
                continue
            probed += 1
            gname = f"evo_{self.name}_g{len(chosen)}"
            write_dck(gname, d["main"])
            w, n = run_match(f"evo_{self.name}_base", gname,
                             self.probe_games, 90 + 30 * self.probe_games)
            # corpus decks carry no archetype label; name them by their
            # highest-count signature nonland cards
            sig = [nm for nm, q in sorted(d["main"], key=lambda p: -p[1])
                   if not (self.imp.is_land[self.imp.name_to_row[norm(nm)]]
                           if norm(nm) in self.imp.name_to_row else True)][:2]
            label = d.get("archetype") or d.get("name") or " + ".join(sig) \
                or f"corpus #{idx}"
            self.emit("probe", f"probe vs {label}: {w}/{n}",
                      label=str(label), wins=w, games=n)
            if n and 0.1 <= w / n <= 0.9:
                chosen.append((str(label), d["main"]))
            elif probed >= 12 and not chosen:
                chosen.append((str(label), d["main"]))
        # fall back to whatever probed closest to 50% if too few kept
        self.gauntlet = chosen
        for gi, (_label, main) in enumerate(self.gauntlet):
            write_dck(f"evo_{self.name}_g{gi}", main)
        self.emit("gauntlet", "gauntlet fixed: "
                  + ", ".join(l for l, _m in self.gauntlet))

    def run_baseline(self, deck=None, tag="baseline"):
        """Verbose games vs every gauntlet deck -> winrate + stats."""
        me = f"evo_{self.name}_base"
        write_dck(me, self.pairs(deck))
        all_games = []
        per = max(6, self.baseline_games // max(1, len(self.gauntlet)))
        for gi, (label, _main) in enumerate(self.gauntlet):
            self.check_stop()
            log = run_verbose(me, f"evo_{self.name}_g{gi}", per,
                              120 + 30 * per)
            games = parse_games(log, me, self.lock_names)
            all_games += games
            self.emit("baseline_part",
                      f"{tag} vs {label}: "
                      f"{sum(g['won'] for g in games)}/{len(games)}")
        stats = aggregate_stats(all_games, self.lock_names)
        p, half = ci95(stats["wins"], stats["games"])
        stats["winrate"] = round(p, 3)
        stats["ci"] = round(half, 3)
        self.emit(tag, f"{tag}: {stats['wins']}/{stats['games']}"
                  f" = {p:.0%} ±{half:.0%}", stats=stats)
        return stats

    def generation(self, gen: int):
        gauntlet_names = [f"evo_{self.name}_g{gi}"
                          for gi in range(len(self.gauntlet))]
        swaps, _pres = self.imp.propose(self.deck, self.candidates * 2,
                                        self.locks, version="auto")
        swaps = [(c, a, q) for c, a, q in swaps
                 if (self.imp.meta[c]["name"], self.imp.meta[a]["name"])
                 not in self.tried and self.imp.playable(a)][:self.candidates]
        if not swaps:
            self.emit("done", "proposer exhausted")
            return False

        variants = {}
        for vi, (cut, add, qty) in enumerate(swaps):
            vname = f"evo_{self.name}_c{vi}"
            newdeck = dict(self.deck)
            take = min(qty, newdeck[cut])
            newdeck[cut] -= take
            if newdeck[cut] <= 0:
                del newdeck[cut]
            newdeck[add] = min(4, newdeck.get(add, 0) + take)
            variants[vname] = (cut, add, take, newdeck)
            write_dck(vname, self.pairs(newdeck))
            self.emit("candidate",
                      f"-{take} {self.imp.meta[cut]['name']}"
                      f" +{take} {self.imp.meta[add]['name']}", gen=gen)

        # stage 1: quick screen vs gauntlet
        s1 = {}
        for vname in variants:
            self.check_stop()
            w = n = 0
            for g in gauntlet_names:
                wi, ni = run_match(vname, g, self.stage1_games,
                                   90 + 30 * self.stage1_games)
                w, n = w + wi, n + ni
            s1[vname] = (w, n)
            self.emit("stage1", f"{vname}: {w}/{n}")
        finalists = sorted(s1, key=lambda v: -(s1[v][0] / max(1, s1[v][1])))[:2]

        # stage 2: finalists get real budgets
        s2 = {}
        for vname in finalists:
            self.check_stop()
            w = n = 0
            for g in gauntlet_names:
                wi, ni = run_match(vname, g, self.stage2_games,
                                   90 + 30 * self.stage2_games)
                w, n = w + wi, n + ni
            s2[vname] = (w, n)
            p, half = ci95(w, n)
            self.emit("stage2", f"{vname}: {w}/{n} = {p:.0%}±{half:.0%}")

        best = max(s2, key=lambda v: s2[v][0] / max(1, s2[v][1]))
        cut, add, take, newdeck = variants[best]
        for c2, a2, _q in swaps:
            self.tried.add((self.imp.meta[c2]["name"], self.imp.meta[a2]["name"]))
        bw, bn = s2[best]
        bp = bw / max(1, bn)
        base_p = self.baseline["winrate"]
        desc = (f"-{take} {self.imp.meta[cut]['name']}"
                f" +{take} {self.imp.meta[add]['name']}")
        event = {"gen": gen, "swap": desc, "stage2": [bw, bn],
                 "vs_baseline": round(bp - base_p, 3)}
        if bp >= base_p + self.margin:
            self.deck = newdeck
            event["accepted"] = True
            self.emit("accept", f"gen {gen}: ACCEPTED {desc} "
                      f"({bp:.0%} vs baseline {base_p:.0%})", **event)
            self.baseline = self.run_baseline(tag="rebaseline")
        else:
            event["accepted"] = False
            self.emit("reject", f"gen {gen}: rejected {desc} "
                      f"({bp:.0%} vs baseline {base_p:.0%})", **event)
        self.history.append(event)
        return True

    def snapshot(self):
        return {
            "name": self.name, "format": self.fmt,
            "locks": self.lock_names,
            "ai_blind": self.ai_blind,
            "incumbent": self.pairs(),
            "gauntlet": [l for l, _m in self.gauntlet],
            "baseline": self.baseline,
            "history": self.history,
        }

    def save(self):
        (OUT / f"evolve2-{self.name}.json").write_text(
            json.dumps(self.snapshot(), indent=1))

    def run(self):
        try:
            if self.ai_blind:
                self.emit("warning",
                          "Forge's AI is forbidden from playing: "
                          + ", ".join(self.ai_blind)
                          + " - the sim scores this deck WITHOUT them. "
                          "Lock them and read winrates accordingly.",
                          ai_blind=self.ai_blind)
            self.emit("phase", "picking gauntlet (probing neighbors)")
            self.pick_gauntlet()
            if not self.gauntlet:
                self.emit("error", "no informative gauntlet found")
                return
            self.emit("phase", "baseline")
            self.baseline = self.run_baseline()
            self.save()
            stale = 0
            for gen in range(1, self.generations + 1):
                self.emit("phase", f"generation {gen}")
                if not self.generation(gen):
                    break
                stale = 0 if self.history[-1]["accepted"] else stale + 1
                self.save()
                if stale >= 3:
                    self.emit("done", "local maximum: 3 stale generations")
                    break
            self.emit("finished", "evolution complete")
        except InterruptedError:
            self.emit("stopped", "stopped by user")
        except Exception as exc:                 # surfaced to the UI
            self.emit("error", f"{type(exc).__name__}: {exc}")
        finally:
            self.save()
