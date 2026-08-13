"""Fetch the official MTGO published decklists into a committed corpus.

Wizards/Daybreak publish league 5-0s and challenge results on
https://www.mtgo.com/decklists - this pulls the constructed 60-card
formats into data/decks/mtgo-decks.jsonl.gz, one deck per line:

  {"event": "...", "format": "pauper", "date": "2026-08-13",
   "player": "...", "wins": 5,
   "main": [["Lightning Bolt", 4], ...], "side": [["Pyroblast", 3], ...]}

Politeness: >=1.2s between requests, resumable (already-fetched events
are skipped), and the result is committed so this only ever runs when
extending the corpus.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
import time
from pathlib import Path

import requests

MONTHS = ["2026/06", "2026/07", "2026/08"]
FORMATS = {"standard", "pioneer", "modern", "legacy", "vintage", "pauper"}
EVENT_RE = re.compile(r'href="(/decklist/([a-z-]+?)-(league|challenge|preliminary|showcase-challenge|showcase-qualifier|super-qualifier)[^"]*)"')
DATA_RE = re.compile(r"window\.MTGO\.decklists\.data\s*=\s*(\{.*?\});", re.S)

OUT = Path(__file__).resolve().parent.parent / "data" / "decks" / "mtgo-decks.jsonl.gz"
UA = {"User-Agent": "mtg-deckbuilder-research/0.1 (one-time corpus fetch; contact via github Mammal4963/MTGDeckBuilder)"}
DELAY = 1.2


def fetch(url: str) -> str:
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=UA, timeout=60)
            if resp.status_code == 200:
                return resp.text
            print(f"  http {resp.status_code} {url}")
        except requests.RequestException as exc:
            print(f"  retry {url}: {exc}")
        time.sleep(3 * (attempt + 1))
    return ""


def cards(entries) -> list:
    out = {}
    for entry in entries or []:
        name = (entry.get("card_attributes") or {}).get("card_name")
        qty = int(entry.get("qty") or 0)
        if name and qty:
            out[name] = out.get(name, 0) + qty
    return sorted(out.items())


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen_events = set()
    existing = []
    if OUT.exists():
        with gzip.open(OUT, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                existing.append(rec)
                seen_events.add(rec["event"])
        print(f"resuming: {len(existing)} decks from {len(seen_events)} events already fetched")

    event_urls = []
    for month in MONTHS:
        html = fetch(f"https://www.mtgo.com/decklists/{month}")
        time.sleep(DELAY)
        for match in EVENT_RE.finditer(html):
            path, fmt = match.group(1), match.group(2)
            if fmt in FORMATS and path not in seen_events:
                event_urls.append((path, fmt))
    event_urls = list(dict.fromkeys(event_urls))
    print(f"{len(event_urls)} new events to fetch across {MONTHS}")

    records = list(existing)
    fetched = 0
    with gzip.open(OUT, "at", encoding="utf-8") as out:
        for i, (path, fmt) in enumerate(event_urls):
            html = fetch("https://www.mtgo.com" + path)
            time.sleep(DELAY)
            match = DATA_RE.search(html)
            if not match:
                print(f"  no data blob in {path}")
                continue
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError as exc:
                print(f"  bad json in {path}: {exc}")
                continue
            date = (data.get("publish_date") or "")[:10]
            for deck in data.get("decklists", []):
                rec = {
                    "event": path,
                    "format": fmt,
                    "date": date,
                    "player": deck.get("player", ""),
                    "wins": int((deck.get("wins") or {}).get("wins", 0) or 0)
                    if isinstance(deck.get("wins"), dict) else 0,
                    "main": cards(deck.get("main_deck")),
                    "side": cards(deck.get("sideboard_deck")),
                }
                out.write(json.dumps(rec, separators=(",", ":")) + "\n")
                records.append(rec)
            fetched += 1
            if fetched % 25 == 0:
                print(f"  {fetched}/{len(event_urls)} events, {len(records)} decks")

    by_fmt = {}
    for rec in records:
        by_fmt[rec["format"]] = by_fmt.get(rec["format"], 0) + 1
    print(f"done: {len(records)} decks total -> {OUT}")
    for fmt, count in sorted(by_fmt.items(), key=lambda kv: -kv[1]):
        print(f"  {fmt:<10} {count}")


if __name__ == "__main__":
    main()
