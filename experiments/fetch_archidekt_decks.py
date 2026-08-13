"""Fetch popular casual decklists from Archidekt's public API.

Purpose: the tournament corpus has no casual archetypes (aristocrats,
tribal, lifegain...), so synergy vectors for casual staples had nothing
to learn from. Archidekt's community decks - mostly Commander - are
exactly that signal.

Respectful use: public JSON API (robots.txt does not disallow /api/),
>=1s between requests, a descriptive User-Agent, fetched once and
committed to data/decks/archidekt-decks.jsonl.gz.
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent.parent / "data" / "decks" / "archidekt-decks.jsonl.gz"
UA = {"User-Agent": "mtg-deckbuilder-research/0.1 (one-time corpus fetch; github Mammal4963/MTGDeckBuilder)"}
API = "https://archidekt.com/api"
DELAY = 1.0

# (label, search query, decks to take, size range)
# deckFormat: 1 standard, 2 modern, 3 commander, 4 legacy, 6 pauper,
# 16 historic. 60-card formats are size-filtered to exclude the cubes,
# binders and wantlists that share those categories.
QUERIES = [
    ("commander-popular", "deckFormat=3&orderBy=-viewCount", 900, (60, 120)),
    ("commander-recent", "deckFormat=3&orderBy=-createdAt", 400, (60, 120)),
    ("standard-popular", "deckFormat=1&orderBy=-viewCount", 400, (55, 90)),
    ("modern-popular", "deckFormat=2&orderBy=-viewCount", 400, (55, 90)),
    ("pauper-popular", "deckFormat=6&orderBy=-viewCount", 400, (55, 90)),
    ("historic-popular", "deckFormat=16&orderBy=-viewCount", 300, (55, 90)),
    ("legacy-popular", "deckFormat=4&orderBy=-viewCount", 200, (55, 90)),
]
FORMAT_NAMES = {3: "commander", 1: "casual-standard", 2: "casual-modern",
                4: "casual-legacy", 6: "casual-pauper", 16: "casual-historic"}


def fetch_json(url: str):
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=UA, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                print("  429 - backing off 30s")
                time.sleep(30)
                continue
            print(f"  http {resp.status_code} {url}")
            return None
        except requests.RequestException as exc:
            print(f"  retry {url}: {exc}")
            time.sleep(5 * (attempt + 1))
    return None


def deck_cards(detail: dict):
    included_cats = {
        c["name"] for c in detail.get("categories", [])
        if c.get("includedInDeck", True)
    }
    out = {}
    for entry in detail.get("cards", []):
        name = ((entry.get("card") or {}).get("oracleCard") or {}).get("name")
        qty = int(entry.get("quantity") or 0)
        cats = entry.get("categories") or []
        if cats and not any(c in included_cats for c in cats):
            continue
        if name and qty:
            out[name] = out.get(name, 0) + qty
    return sorted(out.items())


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    n_existing = 0
    if OUT.exists():
        with gzip.open(OUT, "rt", encoding="utf-8") as fh:
            for line in fh:
                seen.add(json.loads(line)["id"])
                n_existing += 1
        print(f"resuming: {n_existing} decks already fetched")

    todo = []
    for label, query, want, (lo, hi) in QUERIES:
        got, page = 0, 1
        while got < want and page <= (want // 50) + 4:
            data = fetch_json(f"{API}/decks/v3/?{query}&pageSize=50&page={page}")
            time.sleep(DELAY)
            if not data or not data.get("results"):
                break
            for r in data["results"]:
                if (r["id"] not in seen and lo <= (r.get("size") or 0) <= hi
                        and not r.get("private")):
                    todo.append((r["id"], r.get("deckFormat"), r.get("name", "")))
                    seen.add(r["id"])
                    got += 1
            page += 1
        print(f"{label}: queued {got}")
    print(f"{len(todo)} decks to fetch")

    written = 0
    with gzip.open(OUT, "at", encoding="utf-8") as out:
        for deck_id, fmt, name in todo:
            detail = fetch_json(f"{API}/decks/{deck_id}/")
            time.sleep(DELAY)
            if not detail:
                continue
            cards = deck_cards(detail)
            if len(cards) < 20:
                continue
            out.write(json.dumps({
                "id": deck_id,
                "source": "archidekt",
                "format": FORMAT_NAMES.get(fmt, f"fmt{fmt}"),
                "name": name,
                "cards": cards,
            }, separators=(",", ":")) + "\n")
            written += 1
            if written % 100 == 0:
                print(f"  {written}/{len(todo)} decks written")
    print(f"done: {written} new decks (total {n_existing + written}) -> {OUT}")


if __name__ == "__main__":
    main()
