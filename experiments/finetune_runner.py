"""Fine-tune the foundation pilot on YOUR deck, locally.

Give it a decklist (plain text, "4 Lightning Bolt" per line, or a
Forge .dck) and optional locked cards; it registers the deck with
Forge, smoke-verifies it, then runs fine-tune rounds of self-play:
your deck (pilot seat A) vs a random deck from the 850-deck pool
(pilot seat B), warm-started from the foundation-era champion. After
each round's 96-game gate you choose whether to run another.

Watch progress at http://localhost:8123 (started automatically).

Usage:
  python experiments/finetune_runner.py --decklist mydeck.txt \
      --name my_deck --locks "Random Encounter,Chimil" --rounds ask
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from improve_deck import FORGE_DECKS  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
PY = sys.executable


def parse_decklist(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if "[Main]" in text:                       # already a .dck
        m = re.search(r"\[Main\](.*?)(\[|$)", text, re.S | re.I)
        text = m.group(1)
    cards = []
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.lower().startswith("sideboard"):
            break
        if not ln or ln.startswith(("//", "#")):
            continue
        mm = re.match(r"(\d+)\s*x?\s+(.+)", ln)
        if mm:
            cards.append((int(mm.group(1)), mm.group(2).strip()))
    return cards


def write_dck(name: str, cards: list[tuple[int, str]]) -> Path:
    main = "\n".join(f"{n} {c}" for n, c in cards)
    p = FORGE_DECKS / f"{name}.dck"
    p.write_text(f"[metadata]\nName={name}\n[Main]\n{main}\n",
                 encoding="utf-8")
    return p


def verify(name: str) -> bool:
    import sim_server
    client = sim_server.shared_client("ftverify")
    out = client.run(name, "burn", 1, quiet=True, timeout_s=180)
    client.close()
    return len(re.findall(r"Game Result", out)) == 1


def ensure_dashboard():
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:8123/data", timeout=3)
        return
    except OSError:
        pass
    subprocess.Popen([PY, str(HERE / "watch_train.py"), "--port",
                      "8123", "--host", "0.0.0.0"], cwd=HERE.parent,
                     creationflags=0x08000000)   # DETACHED_PROCESS
    time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decklist", required=True)
    ap.add_argument("--name", required=True,
                    help="short deck name, letters/digits/underscore")
    ap.add_argument("--locks", default="",
                    help="comma-separated card names to lock (the "
                    "pilot gets extra credit for casting them early)")
    ap.add_argument("--rounds", default="ask",
                    help="'ask' (prompt after each round) or a number")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--val-games", type=int, default=96)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--ckpt", default=str(OUT / "pilot2_rl22.pt"))
    ap.add_argument("--arch", default="192,6",
                    help="PILOT_D,PILOT_LAYERS of the checkpoint")
    args = ap.parse_args()
    name = re.sub(r"\W", "_", args.name)[:24]
    locks = [c.strip() for c in args.locks.split(",") if c.strip()]

    cards = parse_decklist(Path(args.decklist))
    total = sum(n for n, _ in cards)
    print(f"[deck] {name}: {len(cards)} distinct cards, {total} total")
    if not 40 <= total <= 250:
        sys.exit(f"deck size {total} looks wrong - aborting")
    for c in locks:
        if not any(c.lower() == cn.lower() for _n, cn in cards):
            sys.exit(f"locked card {c!r} is not in the deck")
    write_dck(name, cards)
    print("[verify] playing one test game against a stock deck...")
    if not verify(name):
        sys.exit("Forge could not play this deck (unknown card "
                 "names?) - check spelling against Forge's card list")
    print("[verify] OK")
    ensure_dashboard()

    d, layers = args.arch.split(",")
    import os
    env = dict(os.environ,
               FORGE_SIM_SERVER="1", PILOT_D=d, PILOT_LAYERS=layers)
    try:
        import torch
        if torch.cuda.is_available():
            env["PILOT_DEVICE"] = "cuda"
    except ImportError:
        pass

    rnd = 0
    ckpt = args.ckpt
    while True:
        rnd += 1
        tag = f"ft_{name}_{rnd}"
        cmd = [PY, str(HERE / "self_play_round2.py"),
               "--tag", tag, "--deck", name, "--mull", "--gpu",
               "--ckpt", ckpt, "--iters", str(args.iters),
               "--games", str(args.games),
               "--val-games", str(args.val_games),
               "--parallel", str(args.parallel), "--ppo-epochs", "3",
               "--deck-pool", str(HERE / "decks" / "pool_all"),
               "--pool-frac", "0", "--ramp-bonus", "0",
               "--lock-bonus", "0", "--max-decisions", "16000",
               "--lock-credit", "1.0" if locks else "0"]
        for c in (locks or ["none"]):
            cmd += ["--lock", c]
        print(f"\n[round {rnd}] training {args.iters}x{args.games} "
              f"games - watch http://localhost:8123")
        r = subprocess.run(cmd, cwd=HERE.parent, env=env)
        if r.returncode != 0:
            sys.exit(f"round {rnd} failed (exit {r.returncode}) - "
                     "rerun to resume from the last iteration")
        ckpt = str(OUT / f"pilot2_rl{tag}.pt")
        print(f"[round {rnd}] done -> {ckpt}")
        if args.rounds != "ask":
            if rnd >= int(args.rounds):
                break
        elif input("run another round? [y/N] ").strip().lower() != "y":
            break
    print(f"\nfine-tuned pilot: {ckpt}")
    print(f"play against it:  python experiments/play_vs_pilot.py "
          f"custom {ckpt} {args.arch}")


if __name__ == "__main__":
    main()
