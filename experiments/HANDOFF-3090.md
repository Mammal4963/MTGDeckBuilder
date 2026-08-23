# Handoff: pilot training on the 3090 box

For the Claude Code instance running on the owner's PC. This picks up
a research line developed in a cloud container; everything below is
committed on branch `claude/embeddings-research`. Full history and
every methodological lesson lives in `experiments/FORGE-NOTES.md` —
read its last four sections first ("DAgger closes the cloning gap",
"DAgger round 2", "Self-play shakedown", and the combo-verification
section for the measurement culture).

## The mission

Train a neural pilot that flies Magic decks in Forge (the open-source
rules engine) BETTER than Forge's built-in AI, so the deck evolver's
fitness function stops being polluted by bad piloting. The flagship
test deck is the owner's "Roaming Encounters" (Naya, 60 cards,
`experiments/decks/roaming-encounters.txt`); its locked centerpiece,
Random Encounter, is a card the built-in AI literally cannot cast
(AI-blind Shuffle effect — it never cast it once in 24 logged games).
The pilot can, including the flashback line from the graveyard.

## Where the science stands (all at 96 games/arm, same gauntlet)

| arm | result | meaning |
|---|---|---|
| built-in AI | 21/96 = 22% ±8% | the teacher |
| BC clone pre-DAgger | 6% | compounding-error collapse |
| clone after 2 DAgger rounds | 25-26% ±9% | PARITY, confirmed twice — imitation is saturated |
| self-play RL (1 round, container scale) | 30/96 = 31% ±9% | best point estimate, but CIs overlap: parity-or-slightly-above, NOT a confirmed gain |

Key negative finding from the RL shakedown: the Random Encounter
habit does not self-sustain at small scale. lock_frac tracked the
forced-exploration rate down to 0.0 as eps annealed to 0.04. The
+0.3 lock bonus with 16-game batches is too weak for REINFORCE.

## Your job, in order

1. **Reproduce the environment** (below), run a 2-game smoke test.
2. **Scaled self-play** from the parity clone (`pilot2.pt`) — this is
   the main event. Use `experiments/self_play_round.py` but apply the
   three exploration fixes first (they are small arg/code changes):
   - floor eps at ~0.1 instead of annealing to 0 (or anneal 5× slower)
   - raise `--lock-bonus` to ~0.5
   - `--games` 64-128 per iteration, `--iters` 50+ (the container ran
     14×16; you have ~20-50× its throughput with parallel JVM workers)
3. **Validate like we do**: 96 games minimum per arm, and treat any
   accept as provisional until a CONFIRMATION run at ≥192 games
   agrees. Point estimates at ±9% have burned us three times
   (two evolver accepts and a factory promotion all refuted on
   confirmation — see FORGE-NOTES).
4. If RL beats builtin at confirmation scale: re-run the Roaming
   Encounters evolver with the RL pilot in the loop
   (`experiments/evolve_core.py`, `policy` param) — that was the
   point of all of this.

## Environment setup

The repo does NOT contain Forge. Install:

1. Java 17+ (container used OpenJDK 21).
2. Forge 2.0.14 desktop release — download
   `forge-gui-desktop-2.0.14.tar.bz2` from the Card-Forge GitHub
   releases, extract anywhere, set `FORGE_HOME=<that dir>`. The code
   needs `forge-gui-desktop-2.0.14-jar-with-dependencies.jar` inside.
   First run downloads card images/db; sim mode needs no GUI but on a
   display-less Linux box needs `xvfb-run` (auto-handled by
   `java_prefix()` in `improve_deck.py`; on a desktop with a display,
   nothing needed).
3. Python: torch (CUDA build), sentence-transformers, numpy. The
   model is tiny (D=128 set-transformer) — the GPU matters less than
   you'd think; **game simulation is the bottleneck**, so parallel
   warm JVMs are the real speedup (`FORGE_SIM_SERVER=1` + the
   `worker=` param / `--val-parallel N` patterns, N ≈ cores/2).
4. Decks: copy `experiments/decks/*.dck` into
   `~/.forge/decks/constructed/` (or set `FORGE_DECKS_DIR`). These
   are the exact gauntlet/opponent decks all published numbers used —
   reuse them so results stay comparable.
5. The Java bridge is pre-compiled and committed:
   `experiments/forge_ext/**/*.class` (source .java alongside). It
   shadows Forge classes via classpath ordering — no Forge rebuild.
   If you change protocol code, recompile against the Forge jar:
   `javac -cp <jar> -d experiments/forge_ext <the .java files>`.
6. Smoke test:
   `FORGE_SIM_SERVER=1 python3 -c "from experiments.pilot_bridge import run_bridged; ..."`
   — or simplest, run `self_play_round.py --iters 1 --games 2
   --val-games 0` and confirm `[train]` lines appear. First run also
   downloads the MiniLM embedding model (~90MB).

## File map (all under experiments/)

- `pilot_bridge.py` — Python side of the bridge: policy server,
  `ModelPolicy` (loads `output/pilot2.pt` by default, `ckpt=` to
  override), `run_bridged()` game runner, data collection.
- `forge_ext/forge/ai/PlayerControllerExt.java` — Java side:
  serializes state (hand/battlefield/graveyard/exile, protocol v3),
  candidate enumeration incl. flashback, verbs ok/veto/force/attack/
  block, fail-open on any error, 3s socket timeout.
- `train_pilot2.py` — BC/DAgger trainer (streaming loader, resume via
  `output/pilot2_train_state.json` + `TP2_RESUME=1`).
- `dagger.py`, `dagger_round.py` — DAgger drivers (round: stage-banked,
  resume-safe; you likely won't need more DAgger).
- `self_play.py` — REINFORCE core (`reinforce_update`, `game_lock_frac`).
- `self_play_round.py` — **your main entry point**: resume-safe RL
  round (train stage banked per iteration in `output/pilot2_rl_state.pt`,
  validation banked per ~8-game chunk, promotion idempotent).
- `sim_server.py` + `forge_ext/forge/view/SimServer.java` — persistent
  warm-JVM job server (`FORGE_SIM_SERVER=1` enables).
- `pilot_factory.py` — deck→pilot pipeline (gauntlet probing lives here).
- `evolve_core.py` / `evolve_server.py` / `improve_deck.py` — the deck
  evolver (v3: locks, log stats, pilot-in-the-loop via `policy=`).
- `output/pilot2.pt` — the parity clone. **Warm start; do not overwrite**
  (trainers write it — keep a backup before any retrain).
- `output/pilot2_rl.pt` — container RL checkpoint (31% point estimate);
  fine to continue from or discard for a fresh RL run from pilot2.pt.
- `output/dagger.json`, `output/selfplay_round.json` — verdict journals.
- `output/pilot_dataset.jsonl` — 383k-line teacher-labeled dataset
  (gitignored, NOT in the repo; only needed if you rerun BC/DAgger —
  regenerate by flying games with collection on, see dagger_round.py).

## Hard-won gotchas (short list; FORGE-NOTES has the full journal)

- **Measurement discipline**: 36-game screens lie; 96-game arms have
  ±9% CIs; only confirmation-scale reruns count. Never promote on a
  point estimate.
- The built-in AI cannot see ~2,525 cards (`AI:RemoveDeck` hints,
  list in `data/forge-ai-unplayable.json`). `ai_unblind.py` can strip
  the hints in cardsfolder.zip (always `--restore` after).
- Double-faced cards: Forge wants front-face names in .dck files
  (`write_dck` handles it).
- A dead policy-handler thread silently degrades every decision to a
  3s-timeout builtin fallback — games crawl and you measure the wrong
  arm. Symptom: JVMs near-idle. The handler now has a catch-all +
  error log, but check `policy_errors.log` if throughput tanks.
- Use ephemeral ports (`start_server(0)`) — fixed ports collide with
  undying listeners.
- The lock-partner/dup-up evolver sources and probe-matched gauntlets
  (10-90% winrate band) are in evolve_core.py — reuse, don't reinvent.

## Reporting back

The owner values honest verdicts over good news: report CIs with
every number, call overlapping intervals "no detectable change", and
log negative findings (they redirected this project three times).
Append findings to FORGE-NOTES.md and keep the journals in
experiments/output/ as the record.
