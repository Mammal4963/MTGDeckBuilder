"""Generate a self-contained static dashboard.html for the public site.

Bakes in: current round's training journal, deck stats (ramp + lock
series via watch_train.compute_stats), new-deck confirmation arms, and
all-time game totals. No live endpoints - a snapshot, regenerated and
redeployed at milestones.

Usage: python experiments/make_static_dashboard.py [--out experiments/site/dashboard.html]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from watch_train import compute_stats  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


def newest_journal():
    cands = [p for p in OUT.glob("selfplay_round*.json")
             if re.fullmatch(r"selfplay_round\d*\.json", p.name)]
    cands.sort(key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def alltime_games():
    total = 0
    for jf in OUT.glob("selfplay_round*.json"):
        if not re.fullmatch(r"selfplay_round\d*\.json", jf.name):
            continue
        try:
            j = json.loads(jf.read_text())
            total += sum(e.get("games", 96) for e in j.get("train", []))
            for v in j.get("validation", {}).values():
                if isinstance(v, dict):
                    total += v.get("games", 0)
        except (OSError, json.JSONDecodeError):
            pass
    for cf in ("confirm_round.json", "confirm_newdeck.json"):
        try:
            c = json.loads((OUT / cf).read_text())
            total += sum(a.get("games", 0) for a in c.values()
                         if isinstance(a, dict))
        except (OSError, json.JSONDecodeError):
            pass
    for sc in ("target_events_state.json", "round3_events_state.json"):
        try:
            total += json.loads((OUT / sc).read_text()).get("done", 0) * 4
        except (OSError, json.JSONDecodeError):
            pass
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(
        Path(__file__).resolve().parent / "site" / "dashboard.html"))
    args = ap.parse_args()

    jf = newest_journal()
    journal = json.loads(jf.read_text()) if jf else {"train": [],
                                                     "validation": {}}
    try:
        confirm = json.loads((OUT / "confirm_newdeck.json").read_text())
    except (OSError, json.JSONDecodeError):
        confirm = {}
    stats = compute_stats() or {}
    stats.pop("_", None)

    S = {
        "round": jf.name if jf else "?",
        "train": journal.get("train", []),
        "validation": journal.get("validation", {}),
        "confirm": confirm,
        "stats": stats,
        "alltime": alltime_games(),
        "stamp": time.strftime("%b %d, %I:%M %p"),
    }

    page = TEMPLATE.replace("__STATE__", json.dumps(S))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({len(S['train'])} iters, {S['alltime']} games)")


TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pilot Training — Roaming Encounters</title>
<style>
  body { background:#14161a; color:#d7dae0; font:14px/1.5 system-ui,sans-serif;
         margin:0; padding:24px; }
  .wrap { max-width:860px; margin:0 auto; }
  h1 { font-size:19px; margin:0 0 2px; }
  .sub { color:#8b93a1; font-size:12px; margin-bottom:16px; }
  .sub a { color:#5aa9e6; }
  .cards { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px; }
  .card { background:#1d2026; border:1px solid #2a2e36; border-radius:8px;
          padding:9px 14px; min-width:104px; }
  .card .k { color:#8b93a1; font-size:11px; text-transform:uppercase;
             letter-spacing:.05em; }
  .card .v { font-size:21px; font-weight:600;
             font-variant-numeric:tabular-nums; }
  .card .d { font-size:11px; color:#8b93a1; }
  .panel { background:#1d2026; border:1px solid #2a2e36; border-radius:8px;
           padding:12px 16px; margin-bottom:14px; }
  .panel h2 { font-size:13px; color:#aab2bf; margin:0 0 6px; font-weight:600; }
  canvas { width:100%; height:170px; display:block; }
  .legend { font-size:11px; color:#8b93a1; margin:2px 0 8px; }
  .legend span { margin-right:14px; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:4px;
         margin-right:4px; vertical-align:1px; }
  .win { color:#7ce38b; } .loss { color:#e6785a; }
</style></head><body><div class="wrap">
<h1>Neural Pilot Training — Roaming Encounters</h1>
<div class="sub" id="sub"></div>
<div class="cards" id="cards"></div>
<div class="panel"><h2>Winrate &amp; exploration per iteration (current round)</h2>
<div class="legend"><span><i class="dot" style="background:#5aa9e6"></i>winrate
(vs pilot-flown opponent)</span>
<span><i class="dot" style="background:#7ce38b"></i>lock_frac (Random Encounter
deployed)</span><span><i class="dot" style="background:#8b93a1"></i>eps</span></div>
<canvas id="c1"></canvas></div>
<div class="panel"><h2>Confirmation arms — corrected deck, 288 games each,
greedy vs builtin gauntlet</h2><div id="confirm"></div></div>
<div class="panel"><h2>Random Encounter — first cast turn &amp; cast rate
(recent iterations) <span id="re-stat" style="color:#e0b050;font-weight:400">
</span></h2>
<div style="display:flex;gap:20px;flex-wrap:wrap">
<div style="flex:1;min-width:280px"><div class="legend"><span>
<i class="dot" style="background:#e0b050"></i>avg first-cast turn</span></div>
<canvas id="c-lock" style="height:140px"></canvas></div>
<div style="flex:1;min-width:280px"><div class="legend"><span>
<i class="dot" style="background:#c792ea"></i>cast rate</span></div>
<canvas id="c-lockrate" style="height:140px"></canvas></div></div></div>
<div class="panel"><h2>Mana development (avg lands in play by turn)</h2>
<div class="legend"><span><i class="dot" style="background:#5aa9e6"></i>lands
in play</span><span><i class="dot" style="background:#7ce38b"></i>untapped at
decision points</span></div>
<canvas id="c-ramp" style="height:150px"></canvas></div>
<div class="sub">Static snapshot — regenerated at training milestones.
<a href="index.html">&larr; card map</a></div>
</div>
<script>
const S = __STATE__;
function draw(cv, series, ymin, ymax, xopts) {
  const x0 = (xopts && xopts.x0) || 0;
  const ctx = cv.getContext("2d");
  const W = cv.width = cv.clientWidth * devicePixelRatio;
  const H = cv.height = cv.clientHeight * devicePixelRatio;
  ctx.clearRect(0, 0, W, H);
  const padL = 42 * devicePixelRatio, padB = 30 * devicePixelRatio,
        padT = 8 * devicePixelRatio, padR = 8 * devicePixelRatio;
  const n = Math.max(...series.map(s => s.data.length));
  if (!n) return;
  const x = i => padL + (W - padL - padR) * (n === 1 ? 0.5 : i / (n - 1));
  const y = v => padT + (H - padT - padB) * (1 - (v - ymin) / (ymax - ymin));
  ctx.strokeStyle = "#2a2e36"; ctx.fillStyle = "#8b93a1";
  ctx.font = `${11 * devicePixelRatio}px system-ui`;
  for (let g = 0; g <= 4; g++) {
    const v = ymin + (ymax - ymin) * g / 4;
    ctx.beginPath(); ctx.moveTo(padL, y(v)); ctx.lineTo(W - padR, y(v));
    ctx.stroke();
    ctx.fillText(v.toFixed(2), 4 * devicePixelRatio,
                 y(v) + 4 * devicePixelRatio);
  }
  ctx.strokeStyle = "#3a4150";
  ctx.beginPath(); ctx.moveTo(padL, H - padB); ctx.lineTo(W - padR, H - padB);
  ctx.stroke();
  ctx.textAlign = "center";
  const step = Math.max(1, Math.ceil(n / 10));
  for (let i = 0; i < n; i += step) {
    ctx.beginPath(); ctx.moveTo(x(i), H - padB);
    ctx.lineTo(x(i), H - padB + 4 * devicePixelRatio); ctx.stroke();
    ctx.fillText(String(x0 + i), x(i), H - padB + 16 * devicePixelRatio);
  }
  if (xopts && xopts.label)
    ctx.fillText(xopts.label, padL + (W - padL - padR) / 2,
                 H - 3 * devicePixelRatio);
  ctx.textAlign = "left";
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2 * devicePixelRatio;
    ctx.beginPath();
    s.data.forEach((v, i) => { if (v == null) return;
      const px = x(i), py = y(v);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); });
    ctx.stroke();
  }
}
function card(k, v, d) {
  return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div>` +
         (d ? `<div class="d">${d}</div>` : "") + `</div>`;
}
const t = S.train, last = t[t.length - 1] || {};
const roundNo = (S.round.match(/\\d+/) || ["?"])[0];
document.getElementById("sub").textContent =
  `Round ${roundNo} · ${t.length} iterations · as of ${S.stamp}`;
let cards = "";
cards += card("round", roundNo, `${t.length} iterations`);
cards += card("total games", S.alltime.toLocaleString(), "all runs");
cards += card("last winrate", last.winrate != null ?
              (100 * last.winrate).toFixed(0) + "%" : "–", "self-play");
cards += card("lock_frac", last.lock_frac != null ?
              last.lock_frac.toFixed(2) : "–");
if (S.stats.lock_cast_rate != null)
  cards += card("RE cast rate", (100 * S.stats.lock_cast_rate).toFixed(0) + "%",
                S.stats.lock_first_turn != null ?
                `avg first cast turn ${S.stats.lock_first_turn}` : "");
document.getElementById("cards").innerHTML = cards;
draw(document.getElementById("c1"),
  [{color: "#5aa9e6", data: t.map(e => e.winrate)},
   {color: "#7ce38b", data: t.map(e => e.lock_frac)},
   {color: "#8b93a1", data: t.map(e => e.eps)}], 0, 1,
  {label: "iteration"});
let ch = "";
const armColor = {rl7: "#7ce38b", rl6: "#5aa9e6", rl5: "#5aa9e6",
                  rl2: "#8b93a1", builtin: "#e6785a"};
for (const arm of Object.keys(S.confirm)) {
  const a = S.confirm[arm];
  if (!a || !a.games) continue;
  const p = a.wins / a.games;
  const ci = 1.96 * Math.sqrt(p * (1 - p) / a.games);
  const done = Math.min(100, 100 * a.games / 288);
  ch += `<div style="margin:6px 0">
    <span style="display:inline-block;width:64px">${arm}</span>
    <span style="font-variant-numeric:tabular-nums">${a.wins}/${a.games} =
      ${(100*p).toFixed(0)}% &plusmn;${(100*ci).toFixed(0)}%</span>
    <div style="background:#14161a;border-radius:4px;height:8px;margin-top:3px">
      <div style="background:${armColor[arm] || "#8b93a1"};height:8px;
        border-radius:4px;width:${done}%"></div></div></div>`;
}
document.getElementById("confirm").innerHTML =
  ch || "<span style='color:#8b93a1'>no arms yet</span>";
if (S.stats.lock_first_turn != null)
  document.getElementById("re-stat").textContent =
    ` — avg first cast: turn ${S.stats.lock_first_turn}, cast in ` +
    `${(100 * S.stats.lock_cast_rate).toFixed(0)}% of games`;
const ls = S.stats.lock_series || [];
if (ls.length > 1) {
  draw(document.getElementById("c-lock"),
    [{color: "#e0b050", data: ls.map(e => e.avg_turn)}],
    0, Math.max(10, ...ls.map(e => e.avg_turn || 0)),
    {x0: 0, label: "iterations in window"});
  draw(document.getElementById("c-lockrate"),
    [{color: "#c792ea", data: ls.map(e => e.rate)}], 0, 1,
    {x0: 0, label: "iterations in window"});
}
const ramp = S.stats.ramp || {};
const turns = Object.keys(ramp).map(Number).sort((a, b) => a - b);
if (turns.length)
  draw(document.getElementById("c-ramp"),
    [{color: "#5aa9e6", data: turns.map(t2 => ramp[t2].lands)},
     {color: "#7ce38b", data: turns.map(t2 => ramp[t2].untapped)}],
    0, Math.max(6, ...turns.map(t2 => ramp[t2].lands)),
    {x0: turns[0], label: "turn"});
</script></body></html>
"""


if __name__ == "__main__":
    main()
