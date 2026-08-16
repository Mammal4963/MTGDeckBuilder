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

## Proposer refinement experiments (2026-08-15)

Hypothesis tested: "nearest studied blueprint" (v1: neighbor presence +
centroid similarity) is mean-reverting - it suggests format staples to
every deck. Built v2 (lift over format popularity + synergy to the
deck's core cards + gap residual - redundancy) and benchmarked both on
180 degraded corpus decks (rank the removed real playset; near-duplicate
lists excluded from retrieval).

- Pure v2 (lift-led): much worse at repair (legacy MRR 0.394 -> 0.084).
  The repair task's ground truth IS a popular staple; lift buries
  exactly that signal.
- Reweighted v2b (presence backbone + gentle correctives): ties v1 on
  repair but its suggestions converge back to v1's staples list.
  Interpolating the weights just interpolates the failure modes.

Conclusion: archetype completion and brew enhancement are DIFFERENT
TASKS. v1 stays default (benchmark-validated). The brew path that
showed qualitative promise: query the COMMANDER co-occurrence space for
partners of the deck's core (casual synergy knowledge lives there, not
in tournament lists) - on the Tainted Aether deck it surfaces punisher
enchantments instead of Brainstorm/FoW. Currently data-starved (core
cards in <=8 of 1,887 cmd decks, so vectors are text projections);
the fix is scaling the casual corpus by ~10-50x, then auto-selecting
mode by neighborhood similarity (meta-adjacent deck -> archetype mode;
low max-similarity brew -> core-synergy mode, cmd-space).

## Combo head: fine-tune on Commander Spellbook (2026-08-15)

The pair-synergy model (AUC 0.784 on "same deck?") cannot tell a
mechanical combo from ordinary synergy - miner v0 therefore surfaced
"unplayed archetype fits". Fix: Commander Spellbook's bulk export
(105,692 verified combos, one 27 MB download they publish precisely so
nobody crawls the API) distilled to `data/combos/spellbook-combos.jsonl.gz`
(2.5 MB), then a second bilinear head (`combo_model.py`, same frozen
MiniLM embeddings, rank 128) trained: positives = pairs inside a
verified combo (size <= 4); negatives = 50% deck-co-played non-combo
pairs (the hard case) + 50% random. Card-holdout split (10% of combo
cards never seen in training).

- **Combo head AUC 0.846 vs hard negatives** (combo vs deck-synergy,
  held-out cards); 0.877 vs random.
- Pair-synergy baseline on the same test: **0.334** vs hard negatives -
  below chance, because it actively prefers synergy pairs. The two
  heads measure genuinely different things.
- Showcase (excluded from training): Splinter Twin + Deceiver Exarch
  98.0%, Heliod + Ballista 99.0%, Sanguine Bond + Exquisite Blood
  99.1%, Basalt Monolith + Rings 99.5%; Guttersnipe + Bolt (synergy,
  no combo) 37.0%, nonsense pair 2.9%. Known miss: Thassa's Oracle +
  Demonic Consultation 31.6% - hidden-information/library semantics
  compress poorly in text embeddings.

### Post-crawl retrain (2026-08-16, corpus 3.2k -> 12.1k decks)

The scaled Archidekt crawl (6 chunks, commander 1.9k -> 7.3k decks)
fixed the data starvation diagnosed above:

- Pair-synergy AUC 0.784 -> 0.822; Tainted Aether showcase pairs all
  rose (Aether Vial + Tainted Aether 91% -> 95.5%).
- Combo head held 0.834 hard-negative AUC against a 4x larger co-play
  pool; the Thassa's Oracle blind spot improved 31.6% -> 51.7%.
- **Brew mode transformed.** Pre-crawl the Tainted Aether deck got
  generic blue staples (Counterspell, Negate, Cyclonic Rift...);
  post-crawl it gets on-theme punisher/death-trigger cards: Grave
  Pact, No Mercy, Hissing Miasma, The Meathook Massacre, Dark
  Prophecy, Carnival of Souls, Harvester of Souls. Auto-mode
  similarity 0.62 -> 0.67, still (correctly) routing to brew.
- Blood Artist [cmd] now in 370 decks; neighbors are pure aristocrats
  (Yahenni, Bloodghast, Falkenrath Noble, Viscera Seer, Butcher of
  Malakir).
- Seeker rebuilt with 6,988 recipe decks and redeployed.

## AI:RemoveDeck - Forge's AI is hard-forbidden from 2,525 cards (2026-08-16)

The Tainted Aether evolver run produced a decisive stat: the AI cast
Tainted Aether and Acorn Catapult in **0 of 18** baseline games. Root
cause found in Forge's card scripts: both carry `AI:RemoveDeck:All` -
Forge's developers explicitly forbid the AI from playing cards it
cannot evaluate (symmetric punishers, political cards, etc.).
2,525 cards are flagged; the list is extracted to
`data/forge-ai-unplayable.json` and the evolver warns when a deck
contains any (the sim scores such decks WITHOUT those cards).

Consequences:
- Sim winrates for decks built around AI-blind cards measure only the
  supporting shell. Lock the blind cards; read results accordingly.
- The cheap unlock for a custom pilot experiment: drop modified card
  scripts (hints stripped) into Forge's custom cards folder, or
  implement a PlayerController that overrides the evaluation.

## Custom AI pilot, rung 1: un-blinding experiment (2026-08-16)

`ai_unblind.py` strips AI:RemoveDeck hints from chosen cards inside
the install's cardsfolder.zip (backup/restore built in). Applied to
Tainted Aether + Acorn Catapult, reran the evolved deck's baseline
(24 games, same gauntlet):

- Cast rates: Tainted Aether 0/18 -> 9/24 games (median turn 5);
  Acorn Catapult 0/18 -> 12/24. The ban was the only blocker; generic
  heuristics will cast them.
- But winrate moved 56% -> 38% (overlapping CIs, direction consistent):
  conditional on casting Tainted Aether the AI won 3/9 vs 6/15 when it
  stayed in hand; Catapult 4/12 vs 5/12. The AI casts the lock piece
  then keeps feeding its own creatures into it, and shoots Catapult
  while gifting squirrels. **Attempting != piloting** - Forge's devs
  banned these cards for a reason. This is the empirical case for a
  learned policy (rung 3+), and why rung 2's bridge overrides
  decisions rather than just lifting bans.

Rung 2 architecture (from decompiling forge-gui-desktop 2.0.14):
`forge.game.player.PlayerController` is a ~100-method abstract class -
full reimplementation is out. Instead: subclass
`forge.ai.PlayerControllerAi`, override only the macro decisions
(getAbilityToPlay / spell-choice at priority, chooseTargetsFor,
later combat), delegate everything else to the built-in AI. Inject via
a patched LobbyPlayerAi (Forge is GPL; build from source) with the
policy served over a localhost socket by Python.

## Evolver methodology v2 (evolve_core.py) - lessons from run 1

Run 1 (self-play reference) accepted "cut 4 Acorn Catapult" in gen 1 -
it deleted the deck's win condition, because in mirrors whoever adds
creatures and attacks wins first. User called it immediately. v2:

- FIXED external gauntlet, never self-play. Chosen by PROBING nearest
  corpus neighbors and keeping opponents the deck beats 10-90% of the
  time (informative gradient; the MTGO meta gauntlet gives 0%).
- Baseline first (verbose logs -> winrate + stats: ramp curve, game
  length, mulligans, per-locked-card play rate / first-cast turn),
  accept a swap only if stage-2 gauntlet winrate clears baseline by a
  margin, re-baseline after each accept.
- Hard locks so the evolver can never cut the human-designated plan.
- Served to the browser by experiments/evolve_server.py (local
  companion; Forge can't run in a browser) driving evolver.html.

## Combo verification in Forge (2026-08-16, verify_combo.py)

Method evolved across two iterations, both worth remembering:

1. **Mirror A/B alone is blind to combos.** Combo shell vs curve-matched
   null twin, 40-game h2h: the POSITIVE CONTROL (Sanguine Bond +
   Exquisite Blood, catalogued instant win) came back 50%. Verbose logs
   showed why: with 4+4 pieces in 60 cards, games end before assembly,
   so the mirror measures "8 dead cards vs 8 vanilla bodies".
2. **Assembly-conditional analysis works.** Add symmetric tutors, run
   verbose, bucket each game by whether both pieces resolved and
   whether the interaction visibly FIRED (trigger-pattern threshold):
   - Control: fired games **8/8 wins** (drain chains up to 66 triggers
     deep in the logs - Forge executes the loop); assembled-but-late
     67%; unassembled 22%. Methodology validated.
   - Novel miner candidate (Zodiark, Umbral God + Magnanimous
     Magistrate): fired games **3/3 wins**, but it fired in only 3/20
     assembled games - the AI rarely lines up nontoken deaths with
     Magistrate. Honest verdict: mechanically functional, real when it
     happens, but low-frequency; most of the arm's 70% assembled
     winrate is Zodiark being independently strong.

Takeaway for the evolver/product: verify combos by conditional fired
winrate from verbose logs, never by net A/B winrate; and expect the
AI-pilot ceiling to under-fire engines that need setup sequencing.

Miner v1 (`mine_combos.py`) now scores with the combo head; novelty =
not catalogued in Spellbook (deck co-play is reported, not excluded).
Output character changed exactly as hoped: instead of archetype fits it
proposes activation/untap engines and death-trigger loops - e.g.
Dynaheir, Invoker Adept + tap-ability creatures (ability-copy engines),
Stone-Seeder Hierophant + Copy Land (land-untap mana), Magnanimous
Magistrate + Shimatsu the Bloodcloaked (sac/reanimate loop). These are
mechanically coherent hypotheses, unverified as wins - Forge sim or
manual review is the verification lane.
