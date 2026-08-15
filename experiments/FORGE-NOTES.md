# Forge simulation spike - findings (2026-08-15)

Goal: can we use [Forge](https://github.com/Card-Forge/forge) as the
fitness oracle for simulation-driven deck improvement?

**Verdict: yes.** All numbers from this container (4 CPUs, Java 21).

## Setup that works

1. Download `forge-installer-<ver>.jar` from the GitHub release, install
   headlessly (IzPack console mode):
   `printf '\n1\n<target-dir>\n1\nO\n1\n1\n1\n' | java -jar forge-installer-X.jar -console`
2. **Xvfb is required** - Forge's desktop jar initializes the GUI toolkit
   even in sim mode and dies *silently* (exceptions go to sentry, exit 1,
   nothing on stderr) without a display. `xvfb-run -a java ...` fixes it.
3. Decks: plain `.dck` files (`[metadata]/Name=`, `[Main]`, `qty name`
   lines) placed in `~/.forge/decks/constructed/`. Our corpus decks
   convert trivially. (The `-D <dir>` flag did not pick up decks in
   2.0.14; the constructed dir does.)
4. Run:
   ```
   xvfb-run -a java -Xmx4096m -Dio.netty.tryReflectionSetAccessible=true \
     -Dfile.encoding=UTF-8 -jar forge-gui-desktop-<ver>-jar-with-dependencies.jar \
     sim -d deckA.dck deckB.dck -n 20 -q
   ```
   `-q` = one result line per game plus running match score.
   Full syntax (recovered from SimulateMatch constant pool): also
   `-t` tournaments (Bracket/RoundRobin/Swiss), `-f` format,
   `-a` AI profile per player, `-S` RNG seed.

## Numbers

- **Throughput**: 20 games in 58 s wall incl. ~6 s startup -> ~2.6 s/game
  per process; a process uses ~1.6 cores, so 2 parallel processes on
  4 CPUs -> roughly **2,500-3,500 games/hour** on this box.
- Startup is ~6 s per invocation - batch many games per invocation
  (`-n` large) rather than one game per process.
- **Seeding caveat**: `-S` exists but does NOT fully determinize a match:
  with the same seed, game 1 reproduced but later games diverged
  (thread-timing-dependent RNG use). Common-random-numbers variance
  reduction is therefore unreliable; budget for racing / larger N
  instead.
- Corpus decks (MTGO pauper burn, stompy) loaded with no unsupported
  cards; occasional AI warnings in logs (e.g. an activator hiccup on
  Sneaky Snacker) but games complete.
- Sanity: stompy beat burn 16-4 over 20 - direction plausible (Forge AI
  is known to pilot creature aggro better than precise burn math).

## Implications for the improve-loop design

- A candidate-swap evaluation at ~200 games costs ~4-6 min of box time;
  a full improvement run (20 candidates, racing down to finalists,
  ~5-10k games) is a 1.5-3 h offline job. Overnight product shape
  confirmed.
- Gauntlet decks come straight from `data/decks/mtgo-decks.jsonl.gz`.
- Proposal step must filter to Forge-playable cards; detection = dry-run
  deck load and grep for load warnings.

## Repair benchmark results (2026-08-15, fixed parser)

Degraded a corpus pauper burn deck (-4 Lightning Bolt -3 Fireblast,
+4 Goblin Piker +3 Canyon Minotaur), ran improve_deck.py at a small
budget (6 candidates, 3-deck gauntlet, 60 games/variant total).

- **Proposer: perfect.** Top two proposed swaps were exactly the
  removed cards. Geometry alone repaired the list.
- **Simulation referee at this budget: inconclusive, leaning wrong.**
  Baseline 56.7% ±12.5% beat all candidates (Fireblast repair 48.3%
  ±12.6%). All CIs overlap heavily - 60 games cannot resolve
  single-swap deltas (as predicted: ~2,500 games to separate 2%).
  Stage-1 vs stage-2 rankings also swapped order between runs.
- **AI-pilot bias is real and visible**: Forge's AI plays vanilla
  attackers (Goblin Piker) competently and burn reach (Fireblast
  timing) poorly, so the sim's gradient can genuinely prefer the
  *worse-for-humans* deck. Confirms the spike observation (burn 4-16
  vs stompy).

Design implications for the product:
1. Geometry ranking is the primary, instant suggestion signal (it has
   validated recall; see phase 3/4 evals).
2. Simulation is a *screen*, not a fine-grained ranker: use it at
   overnight budgets (>=300-600 games/variant), only claim a
   difference when CIs separate, and always label the AI-pilot caveat.
3. Prefer sim verdicts for large/archetype-level changes over
   single-swap deltas; aggregate over multiple gauntlets.
