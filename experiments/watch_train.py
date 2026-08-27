"""Live dashboard for self-play training: charts the journal as it grows,
counts games, and lets you open any archived training game and step
through every decision the pilot made in it.

Endpoints:
  /        the page
  /data    journal + games-index summary
  /games   latest index rows (games browser)
  /game?f=it012_w3_04.json.gz   one archived game, decompressed

Stdlib only.

Usage:
    python experiments/watch_train.py [--port 8123] [--host 0.0.0.0]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import threading
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

OUT = Path(__file__).resolve().parent / "output"
_STATS_CACHE = {}

EVOLVE_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>deck evolution</title>
<style>
  :root { color-scheme: dark; }
  body { background:#14161a; color:#d7dae0;
         font:14px/1.5 system-ui,sans-serif; margin:0; padding:24px; }
  h1 { font-size:18px; margin:0 0 4px; }
  .sub { color:#8b93a1; font-size:12px; margin-bottom:18px; }
  .cards { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:18px; }
  .card { background:#1d2026; border:1px solid #2a2e36;
          border-radius:8px; padding:10px 16px; min-width:110px; }
  .card .k { color:#8b93a1; font-size:11px; text-transform:uppercase; }
  .card .v { font-size:20px; font-weight:600; }
  .panel { background:#191c21; border:1px solid #2a2e36;
           border-radius:10px; padding:14px 18px; margin-bottom:16px; }
  h2 { font-size:14px; margin:0 0 10px; color:#aeb6c2; }
  canvas { width:100%; height:180px; }
  select { background:#1d2026; color:#d7dae0; border:1px solid #2a2e36;
           border-radius:6px; padding:4px 8px; }
  pre { background:#1d2026; border:1px solid #2a2e36; border-radius:8px;
        padding:12px; font-size:12px; max-height:420px; overflow:auto;
        column-width:220px; }
  a { color:#5aa9e6; }
  .legend { font-size:11px; color:#8b93a1; margin-bottom:6px; }
  .dot { display:inline-block; width:8px; height:8px;
         border-radius:4px; margin-right:4px; }
</style></head><body>
<h1>Deck evolution</h1>
<div class="sub"><a href="/">&larr; training dashboard</a> ·
pure-noise genetic algorithm over 33,585 cards · phase 0 = fitness
within the population, phase 1 = fitness vs the frozen meta
<span id="status"></span></div>
<div class="cards" id="cards"></div>
<div class="panel"><h2>Fitness by generation</h2>
<div class="legend"><span><i class="dot" style="background:#7ce38b"></i>
best</span> <span><i class="dot" style="background:#5aa9e6"></i>mean</span>
<span style="color:#c792ea">| purple line = graduation to meta
fitness</span></div>
<canvas id="c-fit"></canvas></div>
<div class="panel"><h2>Self-adaptive mutation rate (population mean)</h2>
<canvas id="c-mut"></canvas></div>
<div class="panel"><h2>Champion decklist
<select id="gen-pick"></select></h2>
<pre id="deck">select a generation</pre></div>
<script>
function draw(cv, series, ymin, ymax, marks) {
  const ctx = cv.getContext("2d");
  const W = cv.width = cv.clientWidth * devicePixelRatio;
  const H = cv.height = cv.clientHeight * devicePixelRatio;
  ctx.clearRect(0, 0, W, H);
  const pl = 40 * devicePixelRatio, pb = 22 * devicePixelRatio,
        pt = 6 * devicePixelRatio, pr = 8 * devicePixelRatio;
  const n = Math.max(...series.map(s => s.data.length));
  if (!n) return;
  const x = i => pl + (W - pl - pr) * (n === 1 ? 0.5 : i / (n - 1));
  const y = v => pt + (H - pt - pb) * (1 - (v - ymin) / (ymax - ymin));
  ctx.strokeStyle = "#2a2e36"; ctx.fillStyle = "#8b93a1";
  ctx.font = `${10 * devicePixelRatio}px system-ui`;
  for (let g = 0; g <= 4; g++) {
    const v = ymin + (ymax - ymin) * g / 4;
    ctx.beginPath(); ctx.moveTo(pl, y(v)); ctx.lineTo(W - pr, y(v));
    ctx.stroke();
    ctx.fillText(v.toFixed(2), 4, y(v) + 3 * devicePixelRatio);
  }
  for (const m of (marks || [])) {
    ctx.strokeStyle = "#c792ea";
    ctx.beginPath(); ctx.moveTo(x(m), pt); ctx.lineTo(x(m), H - pb);
    ctx.stroke();
  }
  const step = Math.max(1, Math.ceil(n / 12));
  ctx.textAlign = "center";
  for (let i = 0; i < n; i += step)
    ctx.fillText(String(i), x(i), H - 6 * devicePixelRatio);
  ctx.textAlign = "left";
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2 * devicePixelRatio;
    ctx.beginPath();
    let started = false;
    s.data.forEach((v, i) => { if (v == null) return;
      started ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v));
      started = true; });
    ctx.stroke();
  }
}
let CUR = null;
async function tick() {
  const r = await fetch("/evolve-data" + (CUR ? "?c=" + CUR : ""));
  const d = await r.json();
  if (!d.name) {
    document.getElementById("status").textContent = " · no campaigns yet";
    return;
  }
  CUR = d.name;
  const h = d.history;
  const last = h[h.length - 1] || {};
  document.getElementById("cards").innerHTML =
    [["campaign", d.name], ["generation", last.gen],
     ["phase", last.phase],
     ["best fitness", (100 * (last.best || 0)).toFixed(0) + "%"],
     ["mean", (100 * (last.mean || 0)).toFixed(0) + "%"],
     ["mut rate", last.avg_mut],
     ["sec/gen", last.dur_s]].map(([k, v]) =>
      `<div class="card"><div class="k">${k}</div>` +
      `<div class="v">${v}</div></div>`).join("");
  const grad = h.findIndex((e, i) =>
    i > 0 && e.phase === 1 && h[i - 1].phase === 0);
  draw(document.getElementById("c-fit"),
    [{color: "#7ce38b", data: h.map(e => e.best)},
     {color: "#5aa9e6", data: h.map(e => e.mean)}], 0, 1,
    grad >= 0 ? [grad] : []);
  const muts = h.map(e => e.avg_mut);
  draw(document.getElementById("c-mut"),
    [{color: "#e0b050", data: muts}], 0,
    Math.max(0.1, ...muts.filter(x => x != null)));
  const sel = document.getElementById("gen-pick");
  if (sel.options.length !== d.champions.length + 1) {
    const cur = sel.value;
    sel.innerHTML = '<option value="">latest</option>' +
      d.champions.map(g => `<option>${g}</option>`).join("");
    sel.value = cur;
  }
}
async function showDeck() {
  const g = document.getElementById("gen-pick").value ||
    String(Math.max(0, (CUR ? 1 : 1) &&
      (await (await fetch("/evolve-data?c=" + CUR)).json())
        .champions.slice(-1)[0]));
  const r = await fetch(`/evolve-data?c=${CUR}&deck=${g}`);
  const d = await r.json();
  document.getElementById("deck").textContent =
    d.deck ? d.deck.split("[Main]")[1].trim() : "not found";
}
document.getElementById("gen-pick")
  .addEventListener("change", showDeck);
tick(); setInterval(tick, 15000);
setTimeout(showDeck, 1500); setInterval(showDeck, 60000);
</script></body></html>"""
GAMES = OUT / "games"

LOCKS = ("Random Encounter",)
_LANDS = None
_STATS_CACHE = {"key": None, "data": None}


def land_names():
    global _LANDS
    if _LANDS is None:
        meta = json.loads((OUT / "cards_meta.json").read_text())
        _LANDS = {m["name"] for m in meta if "Land" in m.get("type_line", "")}
    return _LANDS


def chosen_card(state, reply):
    if state.get("kind") != "cast":
        return None
    if reply == "ok":
        p = state.get("proposed", [])
        return p[0] if p else None
    if reply.startswith("force\t"):
        idx = int(reply.split("\t")[1])
        return next((c["card"] for c in state.get("candidates", [])
                     if c["i"] == idx), None)
    return None


def iter_num(v):
    """games_index iter field -> orderable (round, iter) key, or None
    for non-training archives (cf-* confirmation games)."""
    if isinstance(v, int):
        return (0, v)
    m = re.fullmatch(r"r(\d+)-(\d+)", str(v))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _game_record(r, g):
    """Extract the tiny per-game stat record from one archive."""
    lands = land_names()
    per_turn = {}   # turn -> (lands, untapped) at last seen decision
    cast_turn = None
    for state, reply in g["decisions"]:
        if state.get("kind") == "game_end":
            continue
        if g.get("deck") and g["deck"] not in state.get("player",
                                                        g["deck"]):
            continue          # our seat only
        t = (state.get("turn", 0) + 1) // 2   # player turns, not
        bf = state.get("my_battlefield", [])   # engine half-turns
        nl = nu = 0
        for c in bf:
            name = c["n"] if isinstance(c, dict) else c
            if name in lands:
                nl += 1
                if not (isinstance(c, dict) and c.get("tapped")):
                    nu += 1
        per_turn[t] = (nl, nu)
        if cast_turn is None and chosen_card(state, reply) in LOCKS:
            cast_turn = t
    return {"iter": iter_num(r.get("iter")), "cast": cast_turn,
            "per_turn": per_turn}


# incremental stats: consume only NEW bytes of the games index per
# refresh, open only the new fac_roaming archives, and keep a rolling
# window of small per-game records. Nothing here ever re-reads the
# whole corpus after the initial build.
from collections import deque
_STATS_STATE = {"offset": 0, "recent": deque(maxlen=600),
                "lock": threading.Lock(), "init": False}


def incremental_stats():
    idx = OUT / "games_index.jsonl"
    if not idx.exists():
        return None
    st = _STATS_STATE
    with st["lock"]:
        size = idx.stat().st_size
        if size < st["offset"]:            # index rewritten/purged
            st["offset"], st["init"] = 0, False
            st["recent"].clear()
        new = []
        with open(idx, "rb") as f:
            f.seek(st["offset"])
            chunk = f.read()
        st["offset"] = size
        for ln in chunk.decode("utf-8", errors="replace").splitlines():
            try:
                new.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
        if not st["init"]:
            # first build: only the newest 600 rows, like before
            new = new[-600:]
            st["init"] = True
        for r in new:
            # generalist era: EVERY seat of every deck contributes to
            # the mana-development picture (RE-specific stats shelved)
            try:
                with gzip.open(OUT / "games" / r["file"], "rt",
                               encoding="utf-8") as f:
                    g = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            st["recent"].append(_game_record(r, g))
        recent = list(st["recent"])
    ramp = {}
    for rec in recent:
        for t, (nl, nu) in rec["per_turn"].items():
            t = int(t)
            if 1 <= t <= 8:
                a = ramp.setdefault(t, [0, 0, 0])
                a[0] += nl
                a[1] += nu
                a[2] += 1
    return {
        "games": len(recent),
        "ramp": {t: {"lands": round(a[0] / a[2], 2),
                     "untapped": round(a[1] / a[2], 2), "n": a[2]}
                 for t, a in sorted(ramp.items())},
    }


PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>self-play training</title>
<style>
  :root { color-scheme: dark; }
  body { background:#14161a; color:#d7dae0; font:14px/1.5 system-ui,sans-serif;
         margin:0; padding:24px; }
  h1 { font-size:18px; margin:0 0 4px; }
  .sub { color:#8b93a1; font-size:12px; margin-bottom:18px; }
  .cards { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:18px; }
  .card { background:#1d2026; border:1px solid #2a2e36; border-radius:8px;
          padding:10px 16px; min-width:110px; }
  .card .k { color:#8b93a1; font-size:11px; text-transform:uppercase;
             letter-spacing:.05em; }
  .card .v { font-size:22px; font-weight:600; margin-top:2px;
             font-variant-numeric:tabular-nums; }
  .card .d { font-size:11px; color:#8b93a1; }
  .chartbox { background:#1d2026; border:1px solid #2a2e36; border-radius:8px;
              padding:12px 16px 4px; margin-bottom:14px; }
  .chartbox h2, .panel h2 { font-size:13px; color:#aab2bf; margin:0 0 6px;
              font-weight:600; }
  canvas { width:100%; height:180px; display:block; }
  .legend { font-size:11px; color:#8b93a1; margin:4px 0 8px; }
  .legend span { margin-right:14px; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:4px;
         margin-right:4px; vertical-align:1px; }
  .stale { color:#e0b050; }
  .panel { background:#1d2026; border:1px solid #2a2e36; border-radius:8px;
           padding:12px 16px; margin-bottom:14px; }
  table { width:100%; border-collapse:collapse; font-size:13px;
          font-variant-numeric:tabular-nums; }
  th { text-align:left; color:#8b93a1; font-size:11px; text-transform:uppercase;
       letter-spacing:.05em; padding:4px 8px; border-bottom:1px solid #2a2e36;
       position:sticky; top:0; background:#1d2026; }
  td { padding:4px 8px; border-bottom:1px solid #23262d; }
  tr.g:hover td { background:#23262d; cursor:pointer; }
  .win { color:#7ce38b; font-weight:600; }
  .loss { color:#e6785a; }
  .scroll { max-height:340px; overflow-y:auto; }
  .filters { margin-bottom:8px; font-size:12px; color:#8b93a1; }
  .filters select, .filters input { background:#14161a; color:#d7dae0;
        border:1px solid #2a2e36; border-radius:5px; padding:2px 6px;
        font-size:12px; margin-right:8px; }
  #inspector { display:none; }
  #insp-head { display:flex; gap:16px; flex-wrap:wrap; align-items:baseline;
               margin-bottom:8px; font-size:13px; }
  #insp-head b { font-size:15px; }
  .dec { border-left:3px solid #2a2e36; padding:2px 10px; margin:2px 0;
         cursor:pointer; }
  .dec:hover { background:#23262d; }
  .dec.acted { border-left-color:#5aa9e6; }
  .dec.forced { border-left-color:#c792ea; }
  .dec.combat { border-left-color:#e0b050; }
  .dec .meta { color:#8b93a1; font-size:11px; margin-right:8px; }
  .dec .eps { color:#c792ea; font-size:11px; margin-left:6px; }
  .dec pre { display:none; background:#14161a; border-radius:6px; padding:8px;
             font-size:11px; overflow-x:auto; margin:6px 0 2px; }
  .dec.open pre { display:block; }
  button { background:#2a2e36; color:#d7dae0; border:0; border-radius:6px;
           padding:4px 12px; font-size:12px; cursor:pointer; }
  button:hover { background:#343945; }
  #board { background:#14161a; border:1px solid #2a2e36; border-radius:8px;
           padding:10px 14px; margin:10px 0; }
  .brow { display:flex; align-items:center; gap:6px; flex-wrap:wrap;
          padding:6px 0; min-height:34px; }
  .brow .side { color:#8b93a1; font-size:11px; text-transform:uppercase;
                letter-spacing:.05em; width:74px; flex-shrink:0; }
  .life { font-size:16px; font-weight:700; width:34px; flex-shrink:0;
          font-variant-numeric:tabular-nums; }
  .chip { background:#1d2026; border:1px solid #2a2e36; border-radius:6px;
          padding:2px 8px; font-size:12px; white-space:nowrap; }
  .chip.cr { border-color:#4a5160; }
  .chip.tapped { opacity:.45; transform:rotate(6deg); }
  .chip.land { color:#8b93a1; }
  .chip.lock { border-color:#e0b050; color:#e0b050; }
  .chip .pt { color:#8b93a1; margin-left:4px; font-size:11px;
              font-variant-numeric:tabular-nums; }
  .chip .dmg { color:#e6785a; margin-left:3px; font-size:11px; }
  .chip.gy { opacity:.6; font-size:11px; padding:1px 6px; }
  #b-events { border-left:2px solid #2a2e36; margin:2px 0 2px 74px;
              padding:2px 10px; font-size:12px; color:#aab2bf; min-height:16px; }
  #b-events .ev-opp { color:#e6785a; }
  #b-events .ev-my { color:#7ce38b; }
  #b-nav { display:flex; align-items:center; gap:10px; margin-top:6px; }
  #b-nav input[type=range] { flex:1; accent-color:#5aa9e6; }
  #b-pos { color:#8b93a1; font-size:12px; min-width:90px; text-align:right;
           font-variant-numeric:tabular-nums; }
  .dec.sel { background:#23262d; outline:1px solid #3a4150; }
  #insp-body { max-height:300px; overflow-y:auto; }
</style></head><body>
<h1>Neural pilot training</h1>
<div class="sub"><a href="/evolve" style="color:#5aa9e6">&rarr; deck
evolution dashboard</a></div>
<div class="sub" id="status">loading&hellip;</div>
<div class="cards" id="cards"></div>
<div class="chartbox"><h2>Policy loss (PPO)</h2>
<canvas id="c2"></canvas></div>
<div class="chartbox"><h2>Value head &mdash; prediction error &amp; mean value</h2>
<div style="display:flex;gap:20px;flex-wrap:wrap">
<div style="flex:1;min-width:260px"><div class="legend"><span><i class="dot"
style="background:#e0b050"></i>vloss (MSE toward outcome; lower = sharper)<span
id="vloss-slope" style="color:#8b93a1"></span></span></div>
<canvas id="c-vloss" style="height:150px"></canvas></div>
<div style="flex:1;min-width:260px"><div class="legend"><span><i class="dot"
style="background:#5aa9e6"></i>mean V (expected outcome of sampled play)</span></div>
<canvas id="c-meanv" style="height:150px"></canvas></div>
<div style="flex:1;min-width:260px"><div class="legend"><span><i class="dot"
style="background:#e06060"></i>clip fraction (share of steps hitting the
gradient clip; sustained ~1.0 = oversized updates)</span></div>
<canvas id="c-clipf" style="height:150px"></canvas></div></div></div>

<div class="panel">
<h2>Benchmarks <select id="bench-label"
  style="background:#1d2026;color:#d7dae0;border:1px solid #2a2e36;
  border-radius:6px;padding:3px 8px;font-size:12px;margin-left:8px">
  <option value="">all labels</option></select></h2>
<div class="scroll"><table id="bench-table"></table></div>
<div class="muted" style="color:#8b93a1;font-size:11px;margin-top:4px">
pilot-skill: seat-swapped vs the frozen benchmark (50% = equal skill) ·
deck-eval: same pilot both seats (measures the deck) ·
ft-progress: fixed deck vs meta_v1 (compare generations by score) ·
league: product vs product</div>
</div>

<div class="panel">
<h2>Mana per turn — all decks <span id="stats-n"
  style="color:#8b93a1;font-weight:400"></span></h2>
<div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start">
  <div style="flex:1;min-width:260px">
    <div class="legend"><span><i class="dot" style="background:#5aa9e6"></i>lands in play</span>
    <span><i class="dot" style="background:#7ce38b"></i>untapped (mana available)</span></div>
    <canvas id="c-ramp" style="height:140px"></canvas>
  </div>
  <div id="lock-stats" style="min-width:200px;font-size:13px"></div>
</div>
</div>

<!-- legacy builtin-gauntlet confirmation panel retired for the
     benchmark era; history lives in confirm_newdeck.json -->>

<div class="panel">
<h2>Training games <span id="gcount" style="color:#8b93a1;font-weight:400"></span></h2>
<div class="filters">
  <select id="f-res"><option value="">all results</option>
    <option value="1">wins</option><option value="0">losses</option></select>
  <select id="f-opp"><option value="">all opponents</option></select>
  <label><input type="checkbox" id="f-lock"> cast Random Encounter</label>
</div>
<div class="scroll"><table id="games">
<tr><th>game</th><th>iter</th><th>opp</th><th>result</th><th>lock</th>
<th>decisions</th><th>when</th></tr>
</table></div>
</div>

<div class="panel" id="inspector">
<div id="insp-head"></div>
<button id="insp-close">close</button>
<div id="board">
  <div class="brow"><span class="side">opponent</span>
    <span class="life" id="b-opp-life"></span><span id="b-opp-bf"
    style="display:contents"></span></div>
  <div id="b-events"></div>
  <div class="brow"><span class="side">pilot</span>
    <span class="life" id="b-my-life"></span><span id="b-my-bf"
    style="display:contents"></span></div>
  <div class="brow"><span class="side">hand</span><span id="b-hand"
    style="display:contents"></span></div>
  <div class="brow"><span class="side">graveyard</span><span id="b-gy"
    style="display:contents"></span></div>
  <div id="b-nav">
    <button id="b-prev">&larr;</button>
    <button id="b-play">&#9654;</button>
    <button id="b-next">&rarr;</button>
    <input type="range" id="b-slider" min="0" max="0" value="0">
    <span id="b-pos"></span>
  </div>
</div>
<div style="margin-top:8px" id="insp-body"></div>
</div>

<script>
let IDX = [];
function draw(cv, series, ymin, ymax, xopts) {
  // xopts: {x0: first x value, label: axis name} - x ticks are drawn
  // for every chart (default: index starting at 0, no axis name)
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
    ctx.beginPath(); ctx.moveTo(padL, y(v)); ctx.lineTo(W - padR, y(v)); ctx.stroke();
    ctx.fillText(v.toFixed(2), 4 * devicePixelRatio, y(v) + 4 * devicePixelRatio);
  }
  // x axis: baseline, ticks, numeric labels, optional axis name
  ctx.strokeStyle = "#3a4150";
  ctx.beginPath(); ctx.moveTo(padL, H - padB); ctx.lineTo(W - padR, H - padB);
  ctx.stroke();
  ctx.textAlign = "center";
  const step = Math.max(1, Math.ceil(n / 10));
  for (let i = 0; i < n; i += step) {
    ctx.strokeStyle = "#3a4150";
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
    ctx.setLineDash(s.dash ? [6 * devicePixelRatio, 5 * devicePixelRatio]
                           : []);
    ctx.beginPath();
    let started = false;
    s.data.forEach((v, i) => { if (v == null) return;
      const px = x(i), py = y(v);
      started ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      started = true; });
    ctx.stroke();
  }
  ctx.setLineDash([]);
}
// least-squares fit over the non-null points -> {b: slope/x, line: []}
function linreg(data) {
  const pts = [];
  data.forEach((v, i) => { if (v != null) pts.push([i, v]); });
  if (pts.length < 3) return null;
  const n = pts.length;
  const sx = pts.reduce((a, p) => a + p[0], 0),
        sy = pts.reduce((a, p) => a + p[1], 0),
        sxx = pts.reduce((a, p) => a + p[0] * p[0], 0),
        sxy = pts.reduce((a, p) => a + p[0] * p[1], 0);
  const den = n * sxx - sx * sx;
  if (!den) return null;
  const b = (n * sxy - sx * sy) / den, a = (sy - b * sx) / n;
  return {b, line: data.map((_v, i) => a + b * i)};
}
function card(k, v, d, cls) {
  return `<div class="card"><div class="k">${k}</div>` +
         `<div class="v ${cls || ""}">${v}</div>` +
         (d ? `<div class="d">${d}</div>` : "") + `</div>`;
}
function fmtAgo(t) {
  const s = Math.round(Date.now() / 1000 - t);
  return s < 90 ? s + "s ago" : s < 5400 ? Math.round(s / 60) + "m ago"
       : Math.round(s / 3600) + "h ago";
}
async function tick() {
  try {
    const r = await fetch("/data"); const d = await r.json();
    const t = d.train || [];
    const last = t[t.length - 1] || {};
    const v = (d.validation && d.validation.rl) || null;
    const p = (d.validation && d.validation._rl_partial) || null;
    const age = Math.round(d.now - d.mtime);
    // per-iter games: trajs/2 (both seats trained) when present -
    // the "games" field only counts our-deck perspectives
    const iterGames = e => e.trajs ? Math.round(e.trajs / 2)
                                   : (e.games || 96);
    const trainGames = t.reduce((a, e) => a + iterGames(e), 0);
    const valGames = v ? v.games : (p ? p.games : 0);
    // games/hour from journal timestamps (last 5 stamped iters)
    const st = t.filter(e => e.t).slice(-6);
    let rate = "";
    if (st.length >= 2) {
      const dt = st[st.length-1].t - st[0].t;
      const g = st.slice(1).reduce((a, e) => a + iterGames(e), 0);
      if (dt > 0) rate = Math.round(3600 * g / dt) + "/h · " +
        (60 * g / dt).toFixed(1) + "/min";
    }
    let cards = "";
    cards += card("iterations", t.length);
    cards += card("games flown", trainGames + valGames,
                  (valGames ? valGames + " validation · " : "") + rate);
    cards += card("eps", last.eps != null ? last.eps : "&ndash;");
    if (v) cards += card("validation", `${v.wins}/${v.games} = ` +
        `${(100 * v.winrate).toFixed(0)}% &plusmn;${(100 * v.ci).toFixed(0)}%`);
    else if (p) cards += card("validation (partial)", `${p.wins}/${p.games}`);
    cards += card("journal age", age + "s", "", age > 3600 ? "stale" : "");
    if (d.alltime) cards += card("total games", d.alltime.toLocaleString(),
                                 "all runs, this box");
    if (d.evolve)
      cards += card("evolver", `gen ${d.evolve.gen}/${d.evolve.gens}`,
        `${d.evolve.name} · phase ${d.evolve.phase} · best ` +
        `${(100 * d.evolve.best).toFixed(0)}% · mean ` +
        `${(100 * d.evolve.mean).toFixed(0)}% · mut ${d.evolve.avg_mut}`);
    if (d.live)
      cards += card("in flight",
        `iter ${d.live.iter}`,
        `${d.live.wins}/${d.live.ours} our-deck won · ` +
        `${d.live.lock} cast RE · ${d.live.games}/${(d.iterprog && d.iterprog.total) || 128} games`);
    else if (d.stage && d.stage.target_events != null)
      cards += card("round 2 stage",
        d.journal.includes("round2") ? "training" :
          d.stage.v4_ckpt ? "v4 ready" : "collecting",
        `${d.stage.target_events} target events` +
        (d.stage.collect_chunks != null ?
          ` · ${d.stage.collect_chunks}/100 chunks` : ""));
    if (d.live && !d.journal.includes("round2"))
      document.getElementById("status").textContent +=
        " · CHARTS STILL SHOW ROUND 1 — round 2's first iteration banks" +
        " its journal when the REINFORCE update finishes";
    document.getElementById("cards").innerHTML = cards;
    let stg = "";
    if (d.stage) {
      const s = d.stage;
      stg = ` · round-2 pipeline: ${s.target_events || 0} target events` +
        ` collected${s.collect_chunks != null ?
          ` (${s.collect_chunks} chunks)` : ""}` +
        `${s.v4_ckpt ? " · v4 checkpoint ready" : ""}`;
    }
    document.getElementById("status").textContent =
      `${d.journal} — updated ` +
      `${new Date(d.mtime * 1000).toLocaleTimeString()}${stg}`;
    const losses = t.map(e => e.loss).filter(x => x != null);
    const lo = Math.min(0, ...losses), hi = Math.max(0.1, ...losses);
    draw(document.getElementById("c2"),
      [{color: "#e6785a", data: t.map(e => e.loss)}], lo, hi,
      {label: "iteration"});
    const vls = t.map(e => e.vloss).filter(x => x != null);
    if (vls.length > 1) {
      const vdata = t.map(e => e.vloss);
      const fit = linreg(vdata);
      const vseries = [{color: "#e0b050", data: vdata}];
      if (fit) {
        vseries.push({color: "#8b93a1", dash: true, data: fit.line});
        const sl = document.getElementById("vloss-slope");
        if (sl) sl.textContent =
          ` · slope ${fit.b >= 0 ? "+" : ""}${fit.b.toFixed(4)}/iter ` +
          `(${fit.b < -1e-4 ? "improving" :
              fit.b > 1e-4 ? "worsening" : "flat"})`;
      }
      draw(document.getElementById("c-vloss"), vseries,
        0, Math.max(1.2, ...vls), {label: "iteration"});
      const mvs = t.map(e => e.meanV).filter(x => x != null);
      draw(document.getElementById("c-meanv"),
        [{color: "#5aa9e6", data: t.map(e => e.meanV)}],
        Math.min(-0.2, ...mvs), Math.max(0.5, ...mvs),
        {label: "iteration"});
      const cfs = t.map(e => e.clipf).filter(x => x != null);
      if (cfs.length > 1)
        draw(document.getElementById("c-clipf"),
          [{color: "#e06060", data: t.map(e => e.clipf)}],
          0, 1, {label: "iteration"});
    }
    renderBenchProgress(d.benchprog);
    renderProgress(d, t);
  } catch (e) {
    document.getElementById("status").textContent = "journal not readable: " + e;
  }
  loadGames();
  loadStats();
  loadBench();
}
function renderBenchProgress(bp) {
  let el = document.getElementById("bench-progress");
  if (!bp) { if (el) el.remove(); return; }
  if (!el) {
    el = document.createElement("div");
    el.id = "bench-progress";
    el.style.cssText = "margin:-8px 0 16px";
    document.getElementById("cards").after(el);
  }
  const pct = Math.min(100, 100 * bp.games / bp.total);
  const wr = bp.games ? (100 * bp.wins / bp.games).toFixed(0) : "–";
  el.innerHTML = `<div style="font-size:12px;color:#8b93a1;margin-bottom:3px">
      benchmark in flight — ${bp.label}: ${bp.a_ckpt} ·
      ${bp.wins}/${bp.games} won (${wr}%) · ${bp.games}/${bp.total} games
      (${pct.toFixed(0)}%)</div>
    <div style="background:#1d2026;border:1px solid #2a2e36;border-radius:5px;height:8px">
      <div style="background:#c792ea;height:8px;border-radius:5px;width:${pct}%"></div>
    </div>`;
}
let BENCH = [];
async function loadBench() {
  try {
    const r = await fetch("/benchmarks");
    BENCH = await r.json();
    const sel = document.getElementById("bench-label");
    const cur = sel.value;
    const labels = [...new Set(BENCH.map(b => b.label))].sort();
    sel.innerHTML = '<option value="">all labels</option>' +
      labels.map(l => `<option${l === cur ? " selected" : ""}>${l}</option>`)
            .join("");
    renderBench();
  } catch (e) {}
}
function renderBench() {
  const fl = document.getElementById("bench-label").value;
  let rows = BENCH.filter(b => !fl || b.label === fl);
  rows = rows.slice(-40).reverse();
  document.getElementById("bench-table").innerHTML =
    "<tr><th>when</th><th>label</th><th>candidate</th><th>deck</th>" +
    "<th>vs</th><th>score</th><th>games</th><th>note</th></tr>" +
    (rows.length ? rows.map(b =>
      `<tr><td>${new Date(b.t * 1000).toLocaleString()}</td>` +
      `<td>${b.label}</td>` +
      `<td>${b.a_ckpt}${b.swap ? " (swap)" : ""}</td>` +
      `<td>${b.a_deck || "&ndash;"}</td><td>${b.b_ckpt}</td>` +
      `<td><b>${(100 * b.wr).toFixed(0)}%</b> &plusmn;${(100 * b.ci).toFixed(0)}%</td>` +
      `<td>${b.games}</td><td>${b.note || ""}</td></tr>`).join("")
     : '<tr><td colspan="8" style="color:#8b93a1">no benchmark runs yet' +
       ' &mdash; gauntlet.py appends here</td></tr>');
}
document.getElementById("bench-label")
  .addEventListener("change", renderBench);
async function loadStats() {
  try {
    const r = await fetch("/stats"); const s = await r.json();
    if (!s.ramp) return;
    document.getElementById("stats-n").textContent =
      `— last ${s.games} archived seats, all decks`;
    const turns = Object.keys(s.ramp).map(Number).sort((a, b) => a - b);
    draw(document.getElementById("c-ramp"),
      [{color: "#5aa9e6", data: turns.map(t => s.ramp[t].lands)},
       {color: "#7ce38b", data: turns.map(t => s.ramp[t].untapped)}],
      0, Math.max(6, ...turns.map(t => s.ramp[t].lands)),
      {x0: turns[0], label: "turn"});
    const ls = document.getElementById("lock-stats");
    if (ls) ls.innerHTML =
      `<div style="color:#8b93a1;font-size:11px">mana development ` +
      `averaged over every seat of every deck in the pool · ` +
      `RE-specific stats shelved for the generalist era</div>`;
  } catch (e) {}
}
const RUN_ITERS = 60;   // matches the launched --iters
function renderProgress(d, t) {
  let el = document.getElementById("run-progress");
  if (!el) {
    el = document.createElement("div");
    el.id = "run-progress";
    el.style.cssText = "margin:-8px 0 16px";
    document.getElementById("cards").after(el);
  }
  // round length: journal "iters" (new), else inferred from the eps
  // schedule eps = eps0*(1 - it/iters), else the old default
  let iters = d.iters;
  if (!iters) {
    const e = [...t].reverse().find(e => e.eps > 0.12 && e.iter > 5);
    iters = e ? Math.round(e.iter / (1 - e.eps / 0.4) / 10) * 10
              : RUN_ITERS;
  }
  const curFrac = (d.iterprog && d.iterprog.total)
    ? Math.min(1, d.iterprog.done / d.iterprog.total) : 0;
  const done = t.length + curFrac;
  const pct = Math.min(100, 100 * done / iters);
  let iterBar = "";
  const ip = d.iterprog;
  if (ip) {
    const ipct = Math.min(100, 100 * ip.done / ip.total);
    iterBar = `<div style="font-size:12px;color:#8b93a1;margin:6px 0 3px">
        iteration ${ip.iter}: ${ip.done} / ${ip.total} games, all matchups
        (${ipct.toFixed(0)}%)</div>
      <div style="background:#1d2026;border:1px solid #2a2e36;border-radius:5px;height:7px">
        <div style="background:#7ce38b;height:7px;border-radius:5px;width:${ipct}%"></div>
      </div>`;
  } else if (d.live) {
    const ipct = Math.min(100, 100 * d.live.games / perIter);
    iterBar = `<div style="font-size:12px;color:#8b93a1;margin:6px 0 3px">
        iteration ${d.live.iter}: ${d.live.games} / ${perIter} games
        (${ipct.toFixed(0)}%)</div>
      <div style="background:#1d2026;border:1px solid #2a2e36;border-radius:5px;height:7px">
        <div style="background:#7ce38b;height:7px;border-radius:5px;width:${ipct}%"></div>
      </div>`;
  }
  el.innerHTML = `<div style="font-size:12px;color:#8b93a1;margin-bottom:3px">
      round progress: iteration ${t.length}${curFrac ?
        "." + Math.round(10 * curFrac) : ""} / ${iters}
      (${pct.toFixed(0)}%)</div>
    <div style="background:#1d2026;border:1px solid #2a2e36;border-radius:5px;height:10px">
      <div style="background:#5aa9e6;height:10px;border-radius:5px;width:${pct}%"></div>
    </div>` + iterBar;
}

function confirmRowsV2(c, mountId) {
  const SC = 0.60;
  const bi = c.builtin;
  const pB = bi && bi.games ? bi.wins / bi.games : null;
  const ciB = pB != null ? 1.96 * Math.sqrt(pB * (1 - pB) / bi.games) : 0;
  const arms = Object.keys(c)
    .filter(k => k !== "_mtime" && k !== "builtin" && c[k] && c[k].games)
    .map(k => {
      const a = c[k], p = a.wins / a.games;
      const ci = 1.96 * Math.sqrt(p * (1 - p) / a.games);
      const m = k.match(/^rl(\d+)(b?)$/);
      return {k, a, p, ci, n: m ? +m[1] : -1, alt: m ? m[2] : ""};
    })
    .sort((x, y) => (y.n - x.n) || x.alt.localeCompare(y.alt));
  const best = arms.reduce((b, x) => (!b || x.p > b.p) ? x : b, null);
  function row(label, p, ci, wins, games, color, tag) {
    const fill = 100 * Math.min(1, p / SC);
    const lo = 100 * Math.max(0, (p - ci) / SC);
    const hi = 100 * Math.min(1, (p + ci) / SC);
    const mark = pB != null ? `<div style="position:absolute;top:-2px;bottom:-2px;left:${100 * pB / SC}%;border-left:2px dashed #e6785a;opacity:.8"></div>` : "";
    return `<div style="display:flex;align-items:center;gap:10px;margin:5px 0">
      <span style="width:52px;font-size:12px">${label}</span>
      <div style="flex:1;position:relative;background:#14161a;border:1px solid #2a2e36;border-radius:4px;height:12px">
        <div style="position:absolute;top:2px;bottom:2px;left:${lo}%;width:${hi - lo}%;background:${color};opacity:.25;border-radius:3px"></div>
        <div style="position:absolute;top:2px;bottom:2px;left:0;width:${fill}%;background:${color};border-radius:3px"></div>
        ${mark}</div>
      <span style="width:160px;font-size:12px;font-variant-numeric:tabular-nums">${wins}/${games} = ${(100 * p).toFixed(0)}% &plusmn;${(100 * ci).toFixed(0)}% ${tag || ""}</span></div>`;
  }
  let html = `<div style="font-size:11px;color:#8b93a1;margin-bottom:4px">bar = winrate, 0&ndash;60% scale &middot; band = 95% CI &middot; dashed line = builtin</div>`;
  if (pB != null)
    html += row("builtin", pB, ciB, bi.wins, bi.games, "#e6785a", "");
  for (const x of arms) {
    const isBest = best && x.k === best.k;
    const isLatest = x === arms[0];
    const color = isBest ? "#7ce38b" : isLatest ? "#5aa9e6" : "#4a5160";
    html += row(x.k, x.p, x.ci, x.a.wins, x.a.games, color,
                isBest ? "&#9733; best" : "");
  }
  if (best && pB != null) {
    const sep = (best.p - best.ci) > (pB + ciB);
    html += `<div style="margin-top:8px;font-size:12px;font-weight:600;color:${sep ? "#7ce38b" : "#e0b050"}">${best.k} ${sep ? "is CI-SEPARATED above builtin" : "leads builtin (CIs overlap)"} &mdash; <span style="font-weight:400;color:#8b93a1">${(100 * best.p).toFixed(0)}% vs ${(100 * pB).toFixed(0)}%</span></div>`;
  }
  document.getElementById(mountId).innerHTML = html;
}

function renderConfirm(c, journalName) {
  const panel = document.getElementById("confirm-panel");
  if (!c) { panel.style.display = "none"; return; }
  panel.style.display = "block";
  panel.querySelector("h2").textContent =
    "Confirmation arms — greedy pilot vs builtin gauntlet";
  confirmRowsV2(c, "confirm-body");
}
async function loadGames() {
  try {
    const r = await fetch("/games"); const d = await r.json();
    IDX = d.rows;
    document.getElementById("gcount").textContent =
      `— ${d.total} archived, newest first`;
    const opps = [...new Set(IDX.map(g => g.opp))].sort();
    const sel = document.getElementById("f-opp");
    const cur = sel.value;
    sel.innerHTML = '<option value="">all opponents</option>' +
      opps.map(o => `<option${o === cur ? " selected" : ""}>${o}</option>`)
          .join("");
    renderGames();
  } catch (e) {}
}
function renderGames() {
  const fr = document.getElementById("f-res").value;
  const fo = document.getElementById("f-opp").value;
  const fl = document.getElementById("f-lock").checked;
  let rows = IDX;
  if (fr !== "") rows = rows.filter(g => (g.won ? "1" : "0") === fr);
  if (fo) rows = rows.filter(g => g.opp === fo);
  if (fl) rows = rows.filter(g => g.lock_frac > 0);
  document.getElementById("games").innerHTML =
    "<tr><th>game</th><th>iter</th><th>opp</th><th>result</th><th>lock</th>" +
    "<th>decisions</th><th>dur</th><th>when</th></tr>" +
    rows.map(g => `<tr class="g" data-f="${g.file}">` +
      `<td>${g.file.replace(".json.gz", "")}</td><td>${g.iter}</td>` +
      `<td>${g.opp}</td>` +
      `<td class="${g.won ? "win" : "loss"}">${g.won ? "WIN" : "loss"}</td>` +
      `<td>${g.lock_frac ? g.lock_frac.toFixed(1) : "–"}</td>` +
      `<td>${g.n_dec}</td>` +
      `<td>${g.dur_s != null ? g.dur_s + "s" : "–"}</td>` +
      `<td>${g.t ? fmtAgo(g.t) : ""}</td></tr>`).join("");
  document.querySelectorAll("tr.g").forEach(tr =>
    tr.addEventListener("click", () => openGame(tr.dataset.f)));
}
["f-res", "f-opp", "f-lock"].forEach(id =>
  document.getElementById(id).addEventListener("change", renderGames));

function names(cards) {
  return (cards || []).map(c => typeof c === "object" ? c.n : c);
}
function summarize(state, reply) {
  const k = state.kind;
  if (k === "cast") {
    const prop = state.proposed || [];
    if (reply === "ok")
      return prop.length ? {a: true, txt: `cast <b>${prop[0]}</b>`}
                         : {a: false, txt: "pass"};
    if (reply.startsWith("veto"))
      return {a: true, txt: `veto <b>${prop.join(", ") || "?"}</b> → pass`};
    if (reply.startsWith("force")) {
      const idx = +reply.split("\t")[1];
      const c = (state.candidates || []).find(c => c.i === idx) || {};
      return {a: true, txt: `force <b>${c.card || "?"}</b>` +
              (c.zone === "graveyard" ? " (from graveyard!)" : "")};
    }
  }
  if (k === "attackers") {
    if (!reply.startsWith("attack")) return {a: false, txt: "observe combat"};
    const ids = reply.slice(7).split(",").filter(Boolean);
    const by = {}; (state.my_battlefield || []).forEach(c =>
        { if (typeof c === "object") by[c.id] = c.n; });
    return {a: true, combat: true, txt: ids.length ?
      `attack with <b>${ids.map(i => by[i] || i).join(", ")}</b>` : "no attacks"};
  }
  if (k === "game_end")
    return {a: false, txt: `game over — final life ${state.my_life} vs ` +
            `${state.opp_life}`};
  if (k === "target") {
    const cand = state.candidates || [];
    const name = j => { const c = cand.find(c => c.i === j);
      return c ? (c.kind === "player" ? (c.mine ? "OUR FACE" : "opponent face")
                  : c.n) : "?"; };
    const prop = (state.proposed || []).map(name).join(", ");
    if (reply.startsWith("target"))
      return {a: true, txt: `${state.host}: retarget &rarr; ` +
        `<b>${reply.slice(7).split(",").map(x => name(+x)).join(", ")}</b>`};
    return {a: false, txt: `${state.host} targets <b>${prop || "?"}</b>` +
      ` <span style="color:#8b93a1">(builtin's pick)</span>`};
  }
  if (k === "blockers") {
    if (!reply.startsWith("block")) return {a: false, txt: "observe blocks"};
    const by = {}; (state.my_battlefield || []).forEach(c =>
        { if (typeof c === "object") by[c.id] = c.n; });
    const ab = {}; (state.attackers || []).forEach(a => ab[a.id] = a.n || a.id);
    const pairs = reply.slice(6).split(",").filter(s => s.includes(":"))
      .map(s => { const [b, a] = s.split(":");
                  return `${by[b] || b} → ${ab[a] || a}`; });
    return {a: true, combat: true,
            txt: pairs.length ? `block: <b>${pairs.join("; ")}</b>` : "no blocks"};
  }
  return {a: false, txt: reply};
}
const LOCKS = ["Random Encounter"];
let G = null, POS = 0, TIMER = null;

function chipHTML(c, zone) {
  const obj = typeof c === "object";
  const name = obj ? c.n : c;
  const isLand = /^(Forest|Mountain|Island|Plains|Swamp)$/.test(name) ||
                 /District|Sanctum|Pool|Town|Verge|Lands/.test(name);
  let cls = "chip";
  if (obj && c.cr) cls += " cr";
  if (obj && c.tapped) cls += " tapped";
  if (!obj || !c.cr) { if (isLand) cls += " land"; }
  if (LOCKS.includes(name)) cls += " lock";
  if (zone === "gy") cls += " gy";
  let extra = "";
  if (obj && c.cr) extra += `<span class="pt">${c.p}/${c.t}</span>`;
  if (obj && c.dmg > 0) extra += `<span class="dmg">-${c.dmg}</span>`;
  return `<span class="${cls}">${name}${extra}</span>`;
}
function bfKey(c) { return typeof c === "object" ? c.id : c; }
function bfNames(list) {
  const m = new Map();
  (list || []).forEach(c => m.set(bfKey(c), typeof c === "object" ? c.n : c));
  return m;
}
function renderBoard(i) {
  POS = i;
  // one virtual position past the last decision = the game result frame
  const end = i >= G.decisions.length;
  let idx = end ? G.decisions.length - 1 : i;
  let [s] = G.decisions[idx];
  let oppLife = s.opp_life, myLife = s.my_life;
  // newer archives carry a real game_end record with the post-fatal-blow
  // life totals (possibly negative); older ones synthesize 0 for the loser
  if (s.kind === "game_end") {
    oppLife = s.opp_life; myLife = s.my_life;
    if (idx > 0) [s] = G.decisions[idx - 1];   // board from last real state
  } else if (end) {
    if (G.won) oppLife = 0; else myLife = 0;
  }
  document.getElementById("b-opp-life").textContent = oppLife;
  document.getElementById("b-my-life").textContent = myLife;
  document.getElementById("b-opp-bf").innerHTML =
    (s.opp_battlefield || []).map(c => chipHTML(c, "bf")).join("") || "&mdash;";
  document.getElementById("b-my-bf").innerHTML =
    (s.my_battlefield || []).map(c => chipHTML(c, "bf")).join("") || "&mdash;";
  document.getElementById("b-hand").innerHTML =
    (s.my_hand || []).map(c => chipHTML(c, "hand")).join("") || "&mdash;";
  document.getElementById("b-gy").innerHTML =
    ((s.my_graveyard || []).map(c => chipHTML(c, "gy")).join("") || "&mdash;") +
    `<span style="color:#8b93a1;font-size:11px;margin-left:8px">` +
    `opp gy: ${(s.opp_graveyard || []).length}</span>`;
  // events since previous decision: inferred opponent/engine activity
  let ev = [];
  if (i > 0) {
    const [p] = G.decisions[i - 1];
    for (const side of ["opp", "my"]) {
      const prev = bfNames(p[side + "_battlefield"]),
            cur = bfNames(s[side + "_battlefield"]);
      const who = side === "opp" ? "opponent" : "pilot";
      const cls = side === "opp" ? "ev-opp" : "ev-my";
      for (const [k, n] of cur) if (!prev.has(k))
        ev.push(`<span class="${cls}">${who}: ${n} entered</span>`);
      for (const [k, n] of prev) if (!cur.has(k))
        ev.push(`<span class="${cls}">${who}: ${n} left</span>`);
    }
    const dl = s.my_life - p.my_life, dol = s.opp_life - p.opp_life;
    if (dl) ev.push(`<span class="ev-my">pilot life ${dl > 0 ? "+" : ""}${dl}` +
                    ` &rarr; ${s.my_life}</span>`);
    if (dol) ev.push(`<span class="ev-opp">opponent life ` +
                     `${dol > 0 ? "+" : ""}${dol} &rarr; ${s.opp_life}</span>`);
  }
  if (end) {
    ev = [G.won
      ? '<span class="ev-my"><b>GAME OVER — pilot WINS</b> (opponent defeated)</span>'
      : '<span class="ev-opp"><b>GAME OVER — pilot LOSES</b></span>'];
  }
  document.getElementById("b-events").innerHTML = ev.length ?
    (end ? "" : "since last decision: ") + ev.join(" &middot; ") : "";
  const sl = document.getElementById("b-slider");
  sl.value = i;
  document.getElementById("b-pos").textContent = end
    ? `turn ${Math.ceil(s.turn/2)} · end`
    : `turn ${Math.ceil(s.turn/2)} · ${i + 1}/${G.decisions.length}`;
  document.querySelectorAll(".dec.sel").forEach(el =>
    el.classList.remove("sel"));
  const row = document.querySelector(`.dec[data-i="${i}"]`);
  if (row) {
    row.classList.add("sel");
    // scroll ONLY the decision list, never the page - following the
    // selection must not steal the viewport from the board view
    const box = document.getElementById("insp-body");
    box.scrollTop = row.offsetTop - box.offsetTop - box.clientHeight / 2;
  }
}
function step(d) {
  renderBoard(Math.max(0, Math.min(G.decisions.length, POS + d)));
}
document.getElementById("b-prev").addEventListener("click", () => step(-1));
document.getElementById("b-next").addEventListener("click", () => step(1));
document.getElementById("b-play").addEventListener("click", function () {
  if (TIMER) { clearInterval(TIMER); TIMER = null;
               this.innerHTML = "&#9654;"; return; }
  this.innerHTML = "&#9646;&#9646;";
  TIMER = setInterval(() => {
    if (POS >= G.decisions.length) { clearInterval(TIMER); TIMER = null;
      document.getElementById("b-play").innerHTML = "&#9654;"; return; }
    step(1);
  }, 700);
});
document.getElementById("b-slider").addEventListener("input",
  function () { renderBoard(+this.value); });
document.addEventListener("keydown", e => {
  if (!G || document.getElementById("inspector").style.display === "none")
    return;
  if (e.key === "ArrowLeft") { step(-1); e.preventDefault(); }
  if (e.key === "ArrowRight") { step(1); e.preventDefault(); }
});

async function openGame(f) {
  const r = await fetch("/game?f=" + encodeURIComponent(f));
  const g = await r.json();
  G = g;
  const insp = document.getElementById("inspector");
  insp.style.display = "block";
  document.getElementById("insp-head").innerHTML =
    `<b>${f.replace(".json.gz", "")}</b>` +
    `<span>${g.deck} vs ${g.opp}</span>` +
    `<span class="${g.won ? "win" : "loss"}">${g.won ? "WIN" : "LOSS"}</span>` +
    `<span>lock_frac ${g.lock_frac}</span>` +
    `<span>${g.decisions.length} decisions</span>` +
    `<span style="color:#8b93a1">&larr;/&rarr; or drag the slider to step ` +
    `the board · opponent plays are inferred between snapshots · ` +
    `blue = model acted · purple = eps-forced · gold = combat</span>`;
  const sl = document.getElementById("b-slider");
  sl.max = g.decisions.length;   // last position = game-result frame
  let html = "", lastTurn = null;
  g.decisions.forEach(([s, reply], i) => {
    const sm = summarize(s, reply);
    if (s.turn !== lastTurn) {
      lastTurn = s.turn;
      html += `<div style="color:#8b93a1;font-size:11px;margin-top:8px;
        text-transform:uppercase;letter-spacing:.05em">turn ${Math.ceil(s.turn/2)}${s.turn % 2 ? "a" : "b"}
        &mdash; life ${s.my_life} vs ${s.opp_life}</div>`;
    }
    const cls = s._eps_forced ? "forced" : sm.combat ? "combat"
              : sm.a ? "acted" : "";
    html += `<div class="dec ${cls}" data-i="${i}">` +
      `<span class="meta">${s.phase || s.kind}</span>${sm.txt}` +
      (s._eps_forced ? `<span class="eps">eps-forced</span>` : "") +
      (s._dt_ms >= 500 ? `<span style="color:#e0b050;font-size:11px;` +
        `margin-left:6px">+${(s._dt_ms / 1000).toFixed(1)}s wait</span>` : "") +
      `<pre></pre></div>`;
  });
  document.getElementById("insp-body").innerHTML = html;
  document.querySelectorAll(".dec").forEach(el =>
    el.addEventListener("click", () => {
      const i = +el.dataset.i;
      if (POS === i) {
        el.classList.toggle("open");
        const pre = el.querySelector("pre");
        if (!pre.textContent)
          pre.textContent = JSON.stringify(g.decisions[i][0], null, 1);
      } else renderBoard(i);
    }));
  renderBoard(0);
  insp.scrollIntoView({behavior: "smooth"});
}
document.getElementById("insp-close").addEventListener("click",
  () => { document.getElementById("inspector").style.display = "none";
          if (TIMER) { clearInterval(TIMER); TIMER = null; } });
tick(); setInterval(tick, 5000);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to allow LAN (phone) access")
    ap.add_argument("--journal", default=str(OUT / "selfplay_round.json"))
    args = ap.parse_args()
    journal = Path(args.journal)

    class H(BaseHTTPRequestHandler):
        def _send(self, data: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/evolve":
                self._send(EVOLVE_PAGE.encode(), "text/html")
            elif url.path == "/evolve-data":
                q = parse_qs(url.query)
                base = OUT / "evolve"
                camps = sorted((p.name for p in base.glob("*")
                                if (p / "history.jsonl").exists()),
                               key=lambda n: (base / n / "history.jsonl")
                               .stat().st_mtime) if base.exists() else []
                body = {"campaigns": camps}
                name = (q.get("c", [None])[0]
                        or (camps[-1] if camps else None))
                if name and re.fullmatch(r"[\w-]+", name) \
                        and (base / name).exists():
                    body["name"] = name
                    hist = []
                    for ln in (base / name / "history.jsonl") \
                            .read_text(encoding="utf-8").splitlines():
                        try:
                            hist.append(json.loads(ln))
                        except json.JSONDecodeError:
                            pass
                    body["history"] = hist
                    body["champions"] = sorted(
                        int(p.stem[len("champion_gen"):])
                        for p in (base / name).glob("champion_gen*.dck"))
                    g = q.get("deck", [None])[0]
                    if g and g.isdigit():
                        p = base / name / f"champion_gen{int(g):03d}.dck"
                        if p.exists():
                            body["deck"] = p.read_text(encoding="utf-8")
                self._send(json.dumps(body).encode(),
                           "application/json")
            elif url.path == "/data":
                # the newest round's journal takes over the charts
                cands = [p for p in OUT.glob("selfplay_round*.json")
                         if re.fullmatch(r"selfplay_round\d*\.json", p.name)]
                cands.sort(key=lambda p: p.stat().st_mtime)
                active = cands[-1] if cands else journal
                try:
                    body = json.loads(active.read_text())
                    mtime = active.stat().st_mtime
                except (OSError, json.JSONDecodeError):
                    body, mtime = {"train": [], "validation": {}}, 0
                body["mtime"] = mtime
                body["now"] = time.time()
                body["journal"] = active.name
                # round-2 pipeline stage indicators
                stage = {}
                sc = OUT / "target_events_state.json"
                te = OUT / "target_events.jsonl"
                if sc.exists():
                    try:
                        stage["collect_chunks"] = json.loads(
                            sc.read_text()).get("done", 0)
                    except (OSError, json.JSONDecodeError):
                        pass
                if te.exists():
                    try:
                        with open(te, "rb") as f:
                            stage["target_events"] = sum(1 for _ in f)
                    except OSError:
                        pass
                stage["v4_ckpt"] = (OUT / "pilot2_v4.pt").exists()
                body["stage"] = stage
                # live evolver progress
                epf = OUT / "evolve_progress.json"
                if epf.exists() and \
                        time.time() - epf.stat().st_mtime < 3600:
                    try:
                        body["evolve"] = json.loads(epf.read_text())
                    except (OSError, json.JSONDecodeError):
                        pass
                # live benchmark-run progress from gauntlet.py
                bpf = OUT / "bench_progress.json"
                if bpf.exists() and \
                        time.time() - bpf.stat().st_mtime < 600:
                    try:
                        body["benchprog"] = json.loads(bpf.read_text())
                    except (OSError, json.JSONDecodeError):
                        pass
                # exact all-games iteration progress from the driver
                ipf = OUT / "iter_progress.json"
                if ipf.exists() and time.time() - ipf.stat().st_mtime < 600:
                    try:
                        body["iterprog"] = json.loads(ipf.read_text())
                    except (OSError, json.JSONDecodeError):
                        pass
                # all-time games simulated on this box, from every journal
                total = 0
                for jf in OUT.glob("selfplay_round*.json"):
                    if not re.fullmatch(r"selfplay_round\d*\.json", jf.name):
                        continue
                    try:
                        j = json.loads(jf.read_text())
                        total += sum(round(e["trajs"] / 2)
                                     if e.get("trajs")
                                     else e.get("games", 96)
                                     for e in j.get("train", []))
                        for v in j.get("validation", {}).values():
                            if isinstance(v, dict):
                                total += v.get("games", 0)
                    except (OSError, json.JSONDecodeError):
                        pass
                try:
                    c = json.loads((OUT / "confirm_round.json").read_text())
                    total += sum(a.get("games", 0) for a in c.values()
                                 if isinstance(a, dict))
                except (OSError, json.JSONDecodeError):
                    pass
                for sc, per in (("target_events_state.json", 4),
                                ("round3_events_state.json", 4)):
                    try:
                        total += json.loads(
                            (OUT / sc).read_text()).get("done", 0) * per
                    except (OSError, json.JSONDecodeError):
                        pass
                body["alltime"] = total
                # live in-flight iteration from the games index (games
                # archive per game; the journal only banks per iteration)
                gidx = OUT / "games_index.jsonl"
                if gidx.exists():
                    try:
                        # tail-read: the index is tens of MB - parsing
                        # it whole on every poll made /data take seconds
                        with open(gidx, "rb") as fh:
                            fh.seek(max(0, gidx.stat().st_size - 200_000))
                            tail = fh.read().decode("utf-8",
                                                    errors="replace")
                        rows = [json.loads(ln) for ln in
                                tail.splitlines()[1:][-300:]]
                        r2 = [r for r in rows if iter_num(r.get("iter")) is not None]
                        if r2:
                            cur = max(iter_num(r["iter"]) for r in r2)
                            mine = [r for r in r2
                                    if iter_num(r["iter"]) == cur]
                            prim = [r for r in mine
                                    if not r["file"].endswith("b.json.gz")]
                            ours = [r for r in mine
                                    if r.get("deck", "fac_roaming")
                                    == "fac_roaming"]
                            body["live"] = {
                                "iter": cur[1], "games": len(prim),
                                "wins": sum(1 for r in ours if r["won"]),
                                "ours": len(ours),
                                "lock": sum(1 for r in ours
                                            if r.get("lock_frac", 0) > 0)}
                    except (OSError, json.JSONDecodeError, ValueError):
                        pass
                cpath = (OUT / "confirm_newdeck.json"
                         if (OUT / "confirm_newdeck.json").exists()
                         else OUT / "confirm_round.json")
                if cpath.exists():
                    try:
                        body["confirm"] = json.loads(cpath.read_text())
                        body["confirm"]["_mtime"] = cpath.stat().st_mtime
                    except (OSError, json.JSONDecodeError):
                        pass
                # Python json emits Infinity/NaN, which browsers refuse
                # to parse - scrub non-finite numbers before serving
                def _finite(o):
                    if isinstance(o, float):
                        return o if math.isfinite(o) else None
                    if isinstance(o, dict):
                        return {k: _finite(x) for k, x in o.items()}
                    if isinstance(o, list):
                        return [_finite(x) for x in o]
                    return o
                self._send(json.dumps(_finite(body)).encode(),
                           "application/json")
            elif url.path == "/benchmarks":
                rows = []
                bf = OUT / "benchmarks.jsonl"
                if bf.exists():
                    for ln in bf.read_text(
                            encoding="utf-8").splitlines():
                        try:
                            rows.append(json.loads(ln))
                        except json.JSONDecodeError:
                            pass
                self._send(json.dumps(rows[-200:]).encode(),
                           "application/json")
            elif url.path == "/stats":
                # opening hundreds of gzipped archives takes ~30s;
                # always serve the cached copy instantly and refresh
                # it in the background at most once a minute
                def _recompute():
                    try:
                        _STATS_CACHE["v"] = (time.time(),
                                             incremental_stats() or {})
                    except Exception:
                        pass
                    finally:
                        _STATS_CACHE.pop("busy", None)
                now = time.time()
                hit = _STATS_CACHE.get("v")
                if (not hit or now - hit[0] >= 15) and \
                        not _STATS_CACHE.get("busy"):
                    _STATS_CACHE["busy"] = True
                    # NEVER block a request: even the first build runs
                    # in the background and fills in on the next poll
                    threading.Thread(target=_recompute,
                                     daemon=True).start()
                data = hit[1] if hit else {}
                self._send(json.dumps(data).encode(), "application/json")
            elif url.path == "/games":
                rows, total = [], 0
                idx = OUT / "games_index.jsonl"
                if idx.exists():
                    lines = idx.read_text(encoding="utf-8").splitlines()
                    total = len(lines)
                    for ln in lines[-400:]:
                        try:
                            rows.append(json.loads(ln))
                        except json.JSONDecodeError:
                            pass
                    rows.reverse()
                self._send(json.dumps({"rows": rows, "total": total}).encode(),
                           "application/json")
            elif url.path == "/game":
                f = parse_qs(url.query).get("f", [""])[0]
                if not re.fullmatch(
                        r"(r\d+)?(it\d+_w\d+|cf\w+_j\d+)_\d+b?\.json\.gz", f) \
                        or not (GAMES / f).exists():
                    self.send_error(404)
                    return
                with gzip.open(GAMES / f, "rt", encoding="utf-8") as fh:
                    self._send(fh.read().encode(), "application/json")
            else:
                self._send(PAGE.encode(), "text/html; charset=utf-8")

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer((args.host, args.port), H)
    print(f"dashboard: http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
