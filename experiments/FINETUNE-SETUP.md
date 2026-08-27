# Fine-tune the pilot on your own deck

The configurator at
https://mtgdeckbuilder.abe141516.workers.dev/finetune.html builds your
command line; the training itself runs on YOUR machine.

## Requirements

- Windows 10/11 (the scripts use Windows paths; Linux works with
  small edits)
- NVIDIA GPU strongly recommended (`PILOT_DEVICE=cuda` is auto-set);
  CPU-only works at roughly 3x the wall-clock
- ~16 GB RAM, ~8 GB free disk
- Java 17 (Temurin), on PATH
- Python 3.12 with: `pip install torch numpy sentence-transformers`
  (install the CUDA build of torch for GPU)
- Forge 2.0.14 installed at `D:\Forge`
  (or edit `FORGE_DIR` in `experiments/improve_deck.py`)
- This repo cloned, plus the released checkpoint files in
  `experiments/output/` (`pilot2_rl22.pt` at minimum) and the deck
  pool in `experiments/decks/pool_all/`

## Run

```
python experiments\finetune_runner.py --decklist decklist.txt ^
    --name my_deck --locks "My Key Card" --rounds ask
```

What happens:

1. Your decklist is registered with Forge and smoke-verified with one
   test game (misspelled card names fail here).
2. Each round: 30 iterations x 128 self-play games. Seat A is always
   YOUR deck, seat B a random deck from the 850-deck pool; one network
   pilots both seats and learns from wins AND losses.
3. Locked cards get per-decision credit for early casts (the same
   shaping that taught the pilot to slam Random Encounter on curve).
4. After each round a 96-game gate measures your deck's win rate vs
   the stock Forge AI gauntlet; the runner then asks whether to run
   another round. 2-4 rounds is typical.
5. Live dashboard: http://localhost:8123 (charts, game inspector).

Output: `experiments/output/pilot2_rlft_<name>_<round>.pt`.

Play against your fine-tuned pilot in the Forge GUI:

```
python experiments\play_vs_pilot.py custom experiments\output\pilot2_rlft_my_deck_2.pt 192,6
```

## Notes

- Interrupted rounds resume: rerun the same command, it continues
  from the last banked iteration.
- Every game is archived under `experiments/output/games/` and
  inspectable in the dashboard.
- The gate compares against the builtin-AI gauntlet with YOUR deck,
  so the absolute number depends on your deck's power level; watch
  the round-over-round trend.
