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
from watch_train import incremental_stats  # noqa: E402

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
            total += sum(round(e["trajs"] / 2) if e.get("trajs")
                         else e.get("games", 96)
                         for e in j.get("train", []))
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
    stats = incremental_stats() or {}
    stats.pop("_", None)

    # bundle the newest archived games for the static inspector
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    games_rows = []
    gidx = OUT / "games_index.jsonl"
    if gidx.exists():
        import shutil
        rows = [json.loads(ln) for ln in
                gidx.read_text(encoding="utf-8").splitlines()[-60:]]
        gdir = out.parent / "games"
        gdir.mkdir(exist_ok=True)
        keep = set()
        for r in rows:
            src = OUT / "games" / r["file"]
            if src.exists():
                shutil.copy2(src, gdir / r["file"])
                keep.add(r["file"])
                games_rows.append(r)
        for old in gdir.glob("*.json.gz"):
            if old.name not in keep:
                old.unlink()
        games_rows.reverse()          # newest first

    S = {
        "round": jf.name if jf else "?",
        "train": journal.get("train", []),
        "validation": journal.get("validation", {}),
        "confirm": confirm,
        "stats": stats,
        "games": games_rows,
        "alltime": alltime_games(),
        "stamp": time.strftime("%b %d, %I:%M %p"),
    }

    page = TEMPLATE.replace("__STATE__", json.dumps(S))
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({len(S['train'])} iters, {S['alltime']} games, "
          f"{len(games_rows)} games bundled)")


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
  table { width:100%; border-collapse:collapse; font-size:13px;
          font-variant-numeric:tabular-nums; }
  th { text-align:left; color:#8b93a1; font-size:11px;
       text-transform:uppercase; letter-spacing:.05em; padding:4px 8px;
       border-bottom:1px solid #2a2e36; position:sticky; top:0;
       background:#1d2026; }
  td { padding:4px 8px; border-bottom:1px solid #23262d; }
  tr.g:hover td { background:#23262d; cursor:pointer; }
  .scroll { max-height:280px; overflow-y:auto; }
  #board { background:#14161a; border:1px solid #2a2e36; border-radius:8px;
           padding:10px 14px; margin:10px 0; }
  .brow { display:flex; align-items:center; gap:6px; flex-wrap:wrap;
          padding:5px 0; min-height:30px; }
  .brow .side { color:#8b93a1; font-size:11px; text-transform:uppercase;
                letter-spacing:.05em; width:74px; flex-shrink:0; }
  .life { font-size:16px; font-weight:700; width:34px; flex-shrink:0;
          font-variant-numeric:tabular-nums; }
  .chip { background:#1d2026; border:1px solid #2a2e36; border-radius:6px;
          padding:2px 8px; font-size:12px; white-space:nowrap; }
  .chip.cr { border-color:#4a5160; }
  .chip.tapped { opacity:.45; transform:rotate(6deg); }
  .chip.lock { border-color:#e0b050; color:#e0b050; }
  .chip .pt { color:#8b93a1; margin-left:4px; font-size:11px; }
  .chip .dmg { color:#e6785a; margin-left:3px; font-size:11px; }
  .chip.gy { opacity:.6; font-size:11px; padding:1px 6px; }
  #b-events { border-left:2px solid #2a2e36; margin:2px 0 2px 74px;
              padding:2px 10px; font-size:12px; color:#aab2bf;
              min-height:16px; }
  #b-events .ev-opp { color:#e6785a; } #b-events .ev-my { color:#7ce38b; }
  #b-nav { display:flex; align-items:center; gap:10px; margin-top:6px; }
  #b-nav input[type=range] { flex:1; accent-color:#5aa9e6; }
  #b-pos { color:#8b93a1; font-size:12px; min-width:90px;
           text-align:right; font-variant-numeric:tabular-nums; }
  .dec { border-left:3px solid #2a2e36; padding:2px 10px; margin:2px 0; }
  .dec.acted { border-left-color:#5aa9e6; }
  .dec.forced { border-left-color:#c792ea; }
  .dec.combat { border-left-color:#e0b050; }
  .dec.sel { background:#23262d; outline:1px solid #3a4150; }
  .dec .meta { color:#8b93a1; font-size:11px; margin-right:8px; }
  .dec .eps { color:#c792ea; font-size:11px; margin-left:6px; }
  #insp-body { max-height:280px; overflow-y:auto; }
  button { background:#2a2e36; color:#d7dae0; border:0; border-radius:6px;
           padding:4px 12px; font-size:12px; cursor:pointer; }
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
<div class="panel"><h2>Value head &mdash; prediction error &amp; mean value</h2>
<div style="display:flex;gap:20px;flex-wrap:wrap">
<div style="flex:1;min-width:260px"><div class="legend"><span><i class="dot"
style="background:#e0b050"></i>vloss (lower = sharper predictions)</span></div>
<canvas id="c-vloss" style="height:140px"></canvas></div>
<div style="flex:1;min-width:260px"><div class="legend"><span><i class="dot"
style="background:#5aa9e6"></i>mean V</span></div>
<canvas id="c-meanv" style="height:140px"></canvas></div></div></div>
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
<div class="panel"><h2>Recent games <span id="gcount"
style="color:#8b93a1;font-weight:400"></span></h2>
<div class="scroll"><table id="games"></table></div></div>
<div class="panel" id="inspector" style="display:none">
<div id="insp-head" style="display:flex;gap:14px;flex-wrap:wrap;
align-items:baseline;margin-bottom:8px;font-size:13px"></div>
<button id="insp-close">close</button>
<div id="board">
  <div class="brow"><span class="side">opponent</span>
    <span class="life" id="b-opp-life"></span>
    <span id="b-opp-bf" style="display:contents"></span></div>
  <div id="b-events"></div>
  <div class="brow"><span class="side">pilot</span>
    <span class="life" id="b-my-life"></span>
    <span id="b-my-bf" style="display:contents"></span></div>
  <div class="brow"><span class="side">hand</span>
    <span id="b-hand" style="display:contents"></span></div>
  <div class="brow"><span class="side">graveyard</span>
    <span id="b-gy" style="display:contents"></span></div>
  <div id="b-nav">
    <button id="b-prev">&larr;</button>
    <button id="b-play">&#9654;</button>
    <button id="b-next">&rarr;</button>
    <input type="range" id="b-slider" min="0" max="0" value="0">
    <span id="b-pos"></span>
  </div>
</div>
<div id="insp-body"></div>
</div>
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
                `avg first-cast turn (cast games only) ${S.stats.lock_first_turn}` : "");
document.getElementById("cards").innerHTML = cards;
draw(document.getElementById("c1"),
  [{color: "#5aa9e6", data: t.map(e => e.winrate)},
   {color: "#7ce38b", data: t.map(e => e.lock_frac)},
   {color: "#8b93a1", data: t.map(e => e.eps)}], 0, 1,
  {label: "iteration"});
const vls = t.map(e => e.vloss).filter(x => x != null);
if (vls.length > 1) {
  draw(document.getElementById("c-vloss"),
    [{color: "#e0b050", data: t.map(e => e.vloss)}],
    0, Math.max(1.2, ...vls), {label: "iteration"});
  const mvs = t.map(e => e.meanV).filter(x => x != null);
  draw(document.getElementById("c-meanv"),
    [{color: "#5aa9e6", data: t.map(e => e.meanV)}],
    Math.min(-0.2, ...mvs), Math.max(0.5, ...mvs), {label: "iteration"});
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

if (Object.keys(S.confirm || {}).length) confirmRowsV2(S.confirm, "confirm");
else document.getElementById("confirm").innerHTML = "<span style='color:#8b93a1'>no arms yet</span>";

if (S.stats.lock_first_turn != null)
  document.getElementById("re-stat").textContent =
    ` — avg first cast (cast games only): turn ${S.stats.lock_first_turn}, cast in ` +
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

// ---------- game feed + inspector ----------
const LOCKS = ["Random Encounter"];
let G = null, POS = 0, TIMER = null;
document.getElementById("gcount").textContent =
  `— last ${S.games.length} archived training/validation games`;
document.getElementById("games").innerHTML =
  "<tr><th>game</th><th>iter</th><th>opp</th><th>result</th>" +
  "<th>lock</th><th>dur</th></tr>" +
  S.games.map(g => `<tr class="g" data-f="${g.file}">` +
    `<td>${g.file.replace(".json.gz", "")}</td><td>${g.iter}</td>` +
    `<td>${g.opp}</td>` +
    `<td class="${g.won ? "win" : "loss"}">${g.won ? "WIN" : "loss"}</td>` +
    `<td>${g.lock_frac ? g.lock_frac.toFixed(1) : "–"}</td>` +
    `<td>${g.dur_s != null ? g.dur_s + "s" : "–"}</td></tr>`).join("");
document.querySelectorAll("tr.g").forEach(el =>
  el.addEventListener("click", () => openGame(el.dataset.f)));

function chipHTML(c, zone) {
  const obj = typeof c === "object";
  const name = obj ? c.n : c;
  let cls = "chip";
  if (obj && c.cr) cls += " cr";
  if (obj && c.tapped) cls += " tapped";
  if (LOCKS.includes(name)) cls += " lock";
  if (zone === "gy") cls += " gy";
  let extra = "";
  if (obj && c.cr) extra += `<span class="pt">${c.p}/${c.t}</span>`;
  if (obj && c.dmg > 0) extra += `<span class="dmg">-${c.dmg}</span>`;
  return `<span class="${cls}">${name}${extra}</span>`;
}
function bfNames(list) {
  const m = new Map();
  (list || []).forEach(c =>
    m.set(typeof c === "object" ? c.id : c, typeof c === "object" ? c.n : c));
  return m;
}
function summarize(state, reply) {
  const k = state.kind;
  if (k === "game_end")
    return {a: false, txt: `game over — final life ${state.my_life} vs ` +
            `${state.opp_life}`};
  if (k === "mulligan")
    return {a: true, txt: `mulligan decision: <b>${reply}</b> ` +
      `(hand of ${(state.my_hand || []).length})`};
  if (k === "target") {
    const cand = state.candidates || [];
    const nm = j => { const c = cand.find(c2 => c2.i === j);
      return c ? (c.kind === "player" ? (c.mine ? "OUR FACE" :
        "opponent face") : c.n) : "?"; };
    if (reply.startsWith("target"))
      return {a: true, txt: `${state.host}: retarget &rarr; ` +
        `<b>${reply.slice(7).split(",").map(x => nm(+x)).join(", ")}</b>`};
    return {a: false, txt: `${state.host} targets ` +
      `<b>${(state.proposed || []).map(nm).join(", ") || "?"}</b>`};
  }
  if (k === "cast") {
    const prop = state.proposed || [];
    if (reply === "ok")
      return prop.length ? {a: true, txt: `cast <b>${prop[0]}</b>`}
                         : {a: false, txt: "pass"};
    if (reply.startsWith("veto"))
      return {a: true, txt: `veto <b>${prop.join(", ") || "?"}</b> → pass`};
    if (reply.startsWith("force")) {
      const idx = +reply.split("\\t")[1];
      const c = (state.candidates || []).find(c2 => c2.i === idx) || {};
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
      `attack with <b>${ids.map(i => by[i] || i).join(", ")}</b>` :
      "no attacks"};
  }
  if (k === "blockers") {
    if (!reply.startsWith("block")) return {a: false, txt: "observe blocks"};
    const by = {}; (state.my_battlefield || []).forEach(c =>
        { if (typeof c === "object") by[c.id] = c.n; });
    const ab = {}; (state.attackers || []).forEach(a => ab[a.id] = a.n || a.id);
    const pairs = reply.slice(6).split(",").filter(s => s.includes(":"))
      .map(s => { const [b, a] = s.split(":");
                  return `${by[b] || b} → ${ab[a] || a}`; });
    return {a: true, combat: true, txt: pairs.length ?
      `block: <b>${pairs.join("; ")}</b>` : "no blocks"};
  }
  return {a: false, txt: reply};
}
function renderBoard(i) {
  POS = i;
  const end = i >= G.decisions.length;
  let idx = end ? G.decisions.length - 1 : i;
  let [s] = G.decisions[idx];
  let oppLife = s.opp_life, myLife = s.my_life;
  if (s.kind === "game_end") {
    oppLife = s.opp_life; myLife = s.my_life;
    if (idx > 0) [s] = G.decisions[idx - 1];
  } else if (end) {
    if (G.won) oppLife = 0; else myLife = 0;
  }
  document.getElementById("b-opp-life").textContent = oppLife;
  document.getElementById("b-my-life").textContent = myLife;
  document.getElementById("b-opp-bf").innerHTML =
    (s.opp_battlefield || []).map(c => chipHTML(c, "bf")).join("") ||
    "&mdash;";
  document.getElementById("b-my-bf").innerHTML =
    (s.my_battlefield || []).map(c => chipHTML(c, "bf")).join("") ||
    "&mdash;";
  document.getElementById("b-hand").innerHTML =
    (s.my_hand || []).map(c => chipHTML(c, "hand")).join("") || "&mdash;";
  document.getElementById("b-gy").innerHTML =
    ((s.my_graveyard || []).map(c => chipHTML(c, "gy")).join("") ||
     "&mdash;") +
    `<span style="color:#8b93a1;font-size:11px;margin-left:8px">opp gy: ` +
    `${(s.opp_graveyard || []).length}</span>`;
  let ev = [];
  if (!end && i > 0) {
    const [p] = G.decisions[i - 1];
    for (const side of ["opp", "my"]) {
      const prev = bfNames(p[side + "_battlefield"]),
            cur = bfNames(s[side + "_battlefield"]);
      const who = side === "opp" ? "opponent" : "pilot";
      const cls = side === "opp" ? "ev-opp" : "ev-my";
      for (const [k2, n] of cur) if (!prev.has(k2))
        ev.push(`<span class="${cls}">${who}: ${n} entered</span>`);
      for (const [k2, n] of prev) if (!cur.has(k2))
        ev.push(`<span class="${cls}">${who}: ${n} left</span>`);
    }
  }
  if (end) ev = [G.won
    ? '<span class="ev-my"><b>GAME OVER — pilot WINS</b></span>'
    : '<span class="ev-opp"><b>GAME OVER — pilot LOSES</b></span>'];
  document.getElementById("b-events").innerHTML = ev.length ?
    (end ? "" : "since last decision: ") + ev.join(" &middot; ") : "";
  document.getElementById("b-slider").value = i;
  document.getElementById("b-pos").textContent = end
    ? `turn ${s.turn} · end`
    : `turn ${s.turn} · ${i + 1}/${G.decisions.length}`;
  document.querySelectorAll(".dec.sel").forEach(el =>
    el.classList.remove("sel"));
  const row = document.querySelector(`.dec[data-i="${i}"]`);
  if (row) {
    row.classList.add("sel");
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
  if (!G) return;
  if (e.key === "ArrowLeft") { step(-1); e.preventDefault(); }
  if (e.key === "ArrowRight") { step(1); e.preventDefault(); }
});
async function openGame(f) {
  const resp = await fetch("games/" + f);
  const ds = new DecompressionStream("gzip");
  const g = await new Response(resp.body.pipeThrough(ds)).json();
  G = g;
  const insp = document.getElementById("inspector");
  insp.style.display = "block";
  document.getElementById("insp-head").innerHTML =
    `<b>${f.replace(".json.gz", "")}</b>` +
    `<span>${g.deck} vs ${g.opp}</span>` +
    `<span class="${g.won ? "win" : "loss"}">${g.won ? "WIN" : "LOSS"}</span>` +
    `<span>lock_frac ${g.lock_frac}</span>` +
    `<span style="color:#8b93a1">&larr;/&rarr; or slider to step ·` +
    ` blue = model acted · purple = eps-forced · gold = combat</span>`;
  document.getElementById("b-slider").max = g.decisions.length;
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
      `</div>`;
  });
  document.getElementById("insp-body").innerHTML = html;
  document.querySelectorAll(".dec").forEach(el =>
    el.addEventListener("click", () => renderBoard(+el.dataset.i)));
  renderBoard(0);
  insp.scrollIntoView({behavior: "smooth"});
}
document.getElementById("insp-close").addEventListener("click",
  () => { document.getElementById("inspector").style.display = "none";
          if (TIMER) { clearInterval(TIMER); TIMER = null; } });
</script></body></html>
"""


if __name__ == "__main__":
    main()
