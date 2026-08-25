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

## Custom AI pilot, rung 2: the bridge WORKS (2026-08-16)

External-policy bridge built and proven end to end, no Forge rebuild
required:

- `experiments/forge_ext/`: two Java files compiled against the stock
  2.0.14 jar. LobbyPlayerAi is CLASSPATH-SHADOWED (prepend the classes
  dir to -cp and our copy wins over the jar's); it instantiates
  PlayerControllerExt - a PlayerControllerAi subclass overriding only
  chooseSpellAbilityToPlay(). Every "AI wants to cast X" decision
  ships a compact state JSON (turn/phase/life/hand/battlefields/
  proposals) over a localhost socket to Python; the reply can veto
  proposals (AI passes priority instead). Fail-open on any error.
- `experiments/pilot_bridge.py`: policy server + A/B driver.
- Proof of life: 373 decisions round-tripped over 24 games, 9 vetoes
  executed, zero crashes, sims at full speed.

A/B (unblinded Tainted Aether deck vs same gauntlet, 24 games/arm):
builtin 58%+-20%, veto-policy 50%+-20%. The hold-your-creatures rule
fired only 9 times in 24 games - no measurable winrate effect at this
sample. Honest read: the INFRASTRUCTURE is validated, the simple veto
policy is too weak an intervention. The lever the deck actually needs
is FORCING actions (tap Forbidden Orchard, ping with Catapult under
the lock), which means constructing SpellAbilities with targets, not
just filtering - that plus a learned policy is rungs 3-4.

## Custom AI pilot, rung 3: force verb + first learned policy (2026-08-16)

Bridge protocol v2 (forge_ext/PlayerControllerExt): every decision now
carries the FULL legal-candidate list (ComputerUtilAbility enumeration
filtered by canPlay + canPayCost), and the reply may `force\t<idx>
[\topponent]` - playing an ability the built-in AI would never choose,
with optional opponent targeting. Java-side loop guard: a (turn, card)
pair is forced at most once, then falls back to default.

Scripted lock-plan policy (pilot_bridge.py): with Tainted Aether on
our battlefield, force Acorn Catapult activations (the squirrel gift
becomes a forced sacrifice) and veto feeding our own non-Hunted
creatures to the lock. Verified in logs: "activated Acorn Catapult
targeting [Spirit Token]" - it even snipes the opponent's
Orchard-gifted spirits, double value. A/B at 24 games/arm: policy 75%
+-17% vs builtin 71% +-18% with 37 forces executed - the mechanism
demonstrably fires; the winrate delta is within noise at this budget
(the builtin arm itself swung 58% -> 71% between runs; treat n=24 as
mechanism-check, not ranking).

Behavior cloning (train_pilot.py): `pilot_bridge.py --collect` logs
(state, candidates, AI choice) per decision in observer mode - 4,456
decisions from 24 games. Candidate-scoring MLP over frozen MiniLM card
embeddings (state = pooled hand/battlefield embeddings + scalars;
candidate = its embedding + flags; pass head). Held-out: 89.4% overall
(always-pass baseline 84.9%), and on decisions where the AI ACTED it
picks the exact same action 73% (30-sample test; chance ~10-25%).
Text-embedding features mean unseen cards get sensible scores - the
property that lets one policy keep piloting through evolver mutations.

## Custom AI pilot, rung 4: the learned model flies (2026-08-16)

Bridge protocol v3 (PlayerControllerExt): battlefield cards serialize
as {id, name, power, toughness, tapped, damage, is-creature} objects,
and declareAttackers/declareBlockers ship the built-in AI's combat
choices as observation events - the "show it a board at
declare-attackers and record what it does" feed.

Collection at scale (sim-server-backed, both players observed,
creature-heavy legacy decks mixed in for combat coverage): 96 games ->
26,630 cast + 202 attacker + 87 blocker decisions in ~10 min.

Pilot v2 (train_pilot2.py): set-transformer over per-card tokens
[frozen MiniLM text embedding + zone one-hot + live state], 2 layers,
d=128, one encoder with three heads (cast CE, per-creature attack BCE,
per-blocker assignment CE). No pooling bottleneck - 3 permanents or 30,
same model. Held-out: cast 92.3% (act-only 73.1%), ATTACKERS 82.1% F1
from just 202 events, blockers 38.5% (87 events - data-starved, noisy
across epochs; more collection is the fix, not architecture).

Serving (pilot_bridge.ModelPolicy): the trained net makes live cast
decisions through the bridge - agrees with the AI -> ok, prefers pass
-> veto, prefers another candidate -> force. Smoke: 6 games, 1,049
net-made decisions, 15 overrides, 4/6 wins, zero crashes. A learned
model is piloting Forge games end to end.

Dataset regeneration: FORGE_SIM_SERVER=1 python3
experiments/pilot_bridge.py --collect --games 96 (the 27 MB jsonl is
gitignored; ~10 min to rebuild).

## Custom AI pilot, rung 5: the ladder is complete (2026-08-16)

- **Combat write path** (PlayerControllerExt): replies `attack\tids`
  and `block\tblocker:attacker,...` REPLACE the built-in AI's combat
  declarations, each card legality-checked via CombatUtil
  (canAttack/canBlock), illegal requests skipped, fail-open. The
  policy now owns casts AND combat.
- **Combat data scaled**: combat-dense collection (creature decks
  only, both players observed, warm sim server) - 400 games ->
  dataset 303,853 decisions incl. 7,502 attacker (37x) and 1,835
  blocker (21x) events. Fixed en route: player-filter mismatch that
  silently observed nobody; policy-port collisions (ephemeral ports).
- **Retrain on 21x blocker data: blocker head 38.5% -> 53.2%.**
  Attacker F1 reads 72.4% vs the old 82.1% but the numbers are not
  comparable: the old test was ~20 easy events from lock-deck games,
  the new one is 750 events from elves/stompy mirrors - a harder,
  honest benchmark. Cast act-only 70.2% under an 8k act-sample CPU
  cap.
- **Self-play loop runs end to end** (self_play.py): sampled decisions
  (temperature), one game per warm-server job for exact
  trajectory-reward assignment, reward = win/loss + lock-bonus
  (locked cards reaching battlefield), REINFORCE vs EMA baseline,
  grad clipping, journaled checkpoints. Smoke: 2 iters x 8 games,
  checkpoint saved.

State of the ladder: geometry proposes -> evolver verifies in sim ->
learned pilot plays casts and combat -> self-play improves the pilot
beyond its teacher. Remaining work is COMPUTE, not construction:
overnight self-play (10-100k games) and larger BC epochs belong on a
GPU box (see the 3090 notes) - on this container everything runs, just
small. Next experiments: pilot-vs-builtin A/B at real budgets, then
evolver+pilot co-training (re-tune the pilot each accepted mutation,
warm-started).

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

## Validation catches the evolver's false positives (2026-08-17)

Roaming Encounters v3 run (pilot-in-the-loop) accepted two swaps at
+6% and +12% on 36-game stage-2 races. Validation at 96 games/arm vs
the same gauntlet, both lists piloted identically:

    ORIGINAL: 52/96 = 54% +-10%
    EVOLVED:  41/96 = 43% +-10%     (h2h: evolved 44% +-14%)

Both accepts were NOISE - the user's original list is the better
build, by ~11 points. Predicted by our own variance data (builtin arm
swung 58% -> 71% between identical n=24 runs): a 36-game race has
+-16% CI and a +3% margin gate cannot resolve single-swap deltas.
The user's domain read also beat the stats: Nurturing Bristleback's
Forestcycling is hand-smoothing the cast-count stats never see (and
the AI likely never cycles it).

Fix for the evolver: confirmation stage before accept - a candidate
that clears the race must then beat the incumbent at ~100 games
(or SPRT sequential testing: cheap for big effects, exhaustive for
small ones). Race budgets screen; they must never be the verdict.
This validation IS the system working - false positives caught
before the user sleeved cards.

## DAgger closes the cloning gap (2026-08-19)

Pre-DAgger the clone collapsed from compounding error: 6% winrate vs
its teacher's 26% (identical deck, identical gauntlet). One DAgger
round - 196 clone-flown games with free teacher labels (`proposed` +
combat choices are in every bridge message), retrain on the
aggregate - and the 96-game/arm validation reads:

    builtin: 21/96 = 22% +-8%
    clone:   24/96 = 25% +-9%

The DAgger'd clone plays AT PARITY with the built-in AI (difference
within noise). Training metrics on the harder mixed distribution:
cast 87% / act-only 57%, attackers 72.7% F1, blockers 54%. Parity is
the victory condition for cloning; going BEYOND the teacher is
self-play's job, now standing on a base that actually works.

Engineering that made it converge on ~hourly-recycled containers:
per-epoch AND per-2000-step training checkpoints, per-8-game-chunk
validation banking, append-only dataset, a self-healing scheduled
check-in chain, and a policy handler that can never die (a dead
handler had silently degraded games to 3s-timeout builtin fallbacks -
diagnosed via near-idle JVM CPU, fixed with catch-all + error log).

## DAgger round 2: parity holds, still no separation (2026-08-21)

Round 2 at the same scale (dagger_round.py: 200 clone-flown games ->
dataset 382,984 lines -> 3-epoch retrain -> 96-game clone arm;
builtin arm carried forward, it is model-independent):

    builtin:        21/96 = 22% +-8%
    clone round 1:  24/96 = 25% +-9%
    clone round 2:  25/96 = 26% +-9%

Honest read: round 2 is statistically indistinguishable from both the
builtin AI and round 1 - parity is CONFIRMED at a second independent
sample, not improved on. That is the expected shape: DAgger fixes
distribution shift, and once the clone is on-distribution, more
imitation of the same teacher converges to the teacher, not past it.
Training metrics did move (act-only cast agreement 57.3% -> 61.2%,
cast 87.8%, atk F1 70.1%, blk 53.1%), confirming the extra labels
tightened imitation without changing game outcomes - the teacher's
ceiling. Beyond-teacher requires self-play (RL), which is the 3090
plan, warm-starting from this checkpoint.

dagger_round.py is the reusable one-round driver: stage-banked
(per-5-game fly chunks - a slow control mirror outlived the ~hourly
recycle window and re-flew forever until chunked; sidecar-resumed
retrain with a one-shot sidecar clear so the previous round's "done"
state can't fast-skip it; per-chunk validation banking), safe to
re-run after any interruption. Both rounds' final promotion step was
killed by a recycle and hand-promoted from fully-banked chunks -
if extending the driver, fold promotion into the banked state too.

## Self-play shakedown from the parity warm start (2026-08-23)

self_play_round.py: one REINFORCE round (14 iters x 16 games,
fac_roaming vs its gauntlet, reward = win/loss + 0.3 x lock-frac for
Random Encounter, eps-guided lock exploration 0.5 annealed to 0) from
pilot2.pt into pilot2_rl.pt, then the 96-game gate:

    builtin:  21/96 = 22% +-8%
    clone:    25/96 = 26% +-9%   (imitation ceiling, 2x confirmed)
    rl:       30/96 = 31% +-9%

Honest read: the RL arm's point estimate is the highest of the three
but all CIs overlap - consistent with parity-or-slightly-above, NOT a
confirmed gain. ~220 training games of sparse win/loss is a small
signal; the run's real product is that the whole loop now works.

Key negative finding - the lock line does not self-sustain at this
scale: lock_frac tracked eps down (0.25-0.63 mid-run -> 0.13 at
eps 0.07 -> 0.0 at eps 0.04). The +0.3 bonus with 16-game batches is
too weak for REINFORCE to keep casting Random Encounter once forcing
stops. For the 3090 run: floor eps at ~0.1 (or anneal much slower),
raise the lock bonus, and use larger batches so credit assignment has
support. The winrate spike mid-run (50%, 56% at iters 4-5) faded the
same way - suggestive, not durable.

Pipeline validated end to end on a ~hourly-recycled container:
per-iteration model+optimizer+baseline bundle (pilot2_rl_state.pt),
banked validation chunks, and promotion folded into the resume path
(the dagger_round lesson) - the final promotion survived being killed
and completed on relaunch with zero hand-editing. ~12 recycles total,
no lost stages. Scale knobs for the 3090: --iters/--games up 20x,
plus the exploration fixes above.

## Scaled self-play on the 3090: parity confirmed, gain refuted (2026-08-23)

The handoff's main event, run at 26x container scale on the owner's PC
(60 iters x 96 games, 8 parallel warm JVMs, all three exploration
fixes: eps floored at 0.1, lock-bonus 0.5, large batches). ~5,760
training games in ~2h wall. The 96-game gate then the confirmation:

    96-game gate:      rl 30% +-9%  vs clone 26% vs builtin 22%
    CONFIRMATION       rl 78/288 = 27% +-5%
    (288 games/arm):   builtin 76/288 = 26% +-5%   -> NO DETECTABLE CHANGE

Honest read: the fourth point-estimate advantage refuted at
confirmation scale (after two evolver accepts and a factory
promotion). The container's 31% and this run's 30% gate reads were
both noise around parity; note the builtin arm itself moved 22% -> 26%
between samples, exactly the +-CI swing the measurement culture
warns about. Sparse win/loss REINFORCE from the parity clone, even at
26x scale with exploration fixes, converges to the teacher - not past
it.

Second negative finding, sharper than the container's: the lock habit
did not internalize AT ALL. Training lock_frac (~0.20-0.25 at the eps
floor) was pure forcing: the greedy pilot deployed Random Encounter in
0/48 sampled validation-style games. The +0.5 bonus rewards the
forced cast but the policy gradient never transfers it to the unforced
distribution. Implication: the Random Encounter line needs either
denser/shaped reward (credit the *effects* of the cast, not the cast)
or the pilot must own more of the decision (protocol v4 targeting) so
the line actually wins games it otherwise loses.

Infrastructure shipped this session (Windows port of the whole
pipeline): os.pathsep classpaths, Java-17 recompile of forge_ext,
daemonized policy servers (a blocking server_close hung every run at
exit), --parallel collection/validation in self_play_round.py,
per-game trajectory archives + games_index.jsonl, a live dashboard
(experiments/watch_train.py: charts, games browser, board-state
visualizer with per-decision timeline), decision-timing benchmark
(bench_decisions.py: Forge combat on 10+ creature boards averages
209ms/decision vs 35ms on small boards; single worst gap 41.9s on a
41-creature COMBAT_DECLARE_ATTACKERS; our net is 1-9ms), a game_end
bridge event carrying real final life totals, and PROTOCOL V4:
chooseTargetsFor + playSpellAbilityNoStack routed through the bridge
(env-gated FORGE_EXT_TARGETS=1) - trigger targeting observed AND
overridable, proven by flipping Boilerbilges triggers to face in live
games (builtin picks creatures ~70% of the time). Confirmation driver:
confirm_round.py (288 games/arm, banked, resume-safe).

Next levers (in order of expected value): protocol v4 targeting head
(BC from builtin's choices, then RL - the pilot finally owns a
decision the teacher is weak at), both-sides trajectory collection
(pilot flies our deck AND the gauntlet deck: 2x data/game, adaptive
opponent, same matchup distribution as validation - NOT mirrors, per
the run-1 lesson), and reward shaping for the lock line.

## Round 2: both-seats self-play + targeting head (2026-08-23)

Protocol v4 targeting head BC'd from 569 builtin choices (holdout 100%
vs 46% first-candidate baseline - the builtin's targeting rule is
simple), then 60 iters x 64 games of ONE-network-both-seats self-play
(no mirror decks: our deck vs the gauntlet decks, 2 trajectories/game,
128/iter). 96-game gate rl2 30% +-9%; confirmation at 288 games:

    rl2:     84/288 = 29% +-5%
    rl:      78/288 = 27% +-5%   (round 1)
    builtin: 76/288 = 26% +-5%   -> NO DETECTABLE CHANGE (again)

Honest read: three pilot arms now sit 1-3 points above builtin at
288-game scale and none separates. If a real ~+3% effect exists,
resolving it needs ~2,000 games/arm (+-2% CI). Random Encounter still
mostly eps-driven in training (lock_frac ~0.2 at the floor); in the
few unforced casts the avg first-cast turn is ~13 - it's a
nothing-else-to-do play, not a plan.

New instrumentation: mulligan bridge hooks (FORGE_EXT_MULL, keep/mull
write path + London-tuck observe), deck-context token (count-weighted
mean decklist embedding appended to every state; opt-in deck_ctx),
mull head + REINFORCE branches, model-primary mode (FORGE_EXT_PRIMARY:
skip the builtin's cast/combat evaluation entirely - we pay its
thinking cost on every decision otherwise), dashboard deck stats
(ramp curve, per-iteration first-cast-turn + cast-rate series).

## Rounds 3-5: shaping, mulligans, corrected deck (2026-08-23/24)

Rapid-iteration block; every hypothesis measured, most refuted:

- Model-primary mode (skip builtin's cast/combat eval): NO speedup
  (2.10 vs 2.13 s/game) - engine + our own candidate enumeration
  dominate; builtin thinking only matters on rare swarm boards.
  Shelved.
- Deck-context token appended to the state: trained trunk pays it
  ~zero attention (L2 delta 1e-7) - inert. Deck embedding now feeds
  the mulligan/tuck heads DIRECTLY instead.
- Per-decision lock credit (gradient on the RE-cast decision itself,
  earliness-scaled): works - P(cast RE | castable) 5.4% -> 10% over
  ~40 iters, vs glacial movement under trajectory-level bonus alone.
  Still far from greedy dominance (argmax 0/255).
- WHY RE casts average turn ~12: it costs {4}{R}{R} (6 MV), real land
  curve reaches 6 lands ~turn 8-9, and RE is undrawn by turn 6 in 38%
  of games. Turn-6 casts need the perfect ramp curve; realistic
  average floor is ~8-9, not 6.
- Corrected decklist (-4 Threefold Thunderhulk, +1 Coliseum Behemoth,
  +1 Ghalta, +2 Pelakka Wurm): builtin baseline jumped 26% -> 36% +-6%
  at 288 games - the deck fix is worth ~10 points by itself. ALL
  old-deck arms are obsolete as baselines.
- Mulligan ownership rung 1: keep/mull + London tuck heads BC'd from
  builtin (old-deck data). First trainer collapsed to bias (frozen
  trunk is near-blind on turn-0 hand-only states); fixed with direct
  hand features (lands, curve, early plays). But live A/B on the new
  deck: pilot mulligans 27% vs builtin mulligans 34% on the same
  checkpoint - the head UNDER-mulls (kept 96/98 hands) and costs
  points. Disabled until retrained on new-deck collection.
- Round-5 confirmation (targeting on, mulls builtin, 288 games):
  rl5 31% +-5% vs builtin 36% +-6% - first time a pilot arm points
  BELOW builtin (CIs graze). Suspects: shaping tax on general play,
  or insufficient adaptation to the new list. rl2-on-new-deck arm
  running to separate the two.

Round-6 coda (120 iters, retrained mull/tuck heads on new-deck data,
full decision ownership): gate 33% +-9%, confirmation 31% +-5% vs
builtin 36% +-6% - identical to rl5's 31%. The extra 60 iterations
bought nothing. New-deck ladder: rl2 28%, rl5 31%, rl6 31%, builtin
36%. The corrected list is combat-centric and the builtin's
simulation-based combat exploits it better than the pilot's learned
combat heads (trained mostly on old-deck distributions). P(cast RE |
castable) reached 19% (from 5%) without ever tipping greedy behavior.
Recommended next: re-anchor by DAgger-cloning the builtin ON THE NEW
DECK (the ladder's proven fix for distribution shift), then RL from
that parity point - not more RL from the drifted lineage.

Meta-lesson standing after ~18k games: every capability rung (combat,
targeting, mulligans) lands mechanically and clones cleanly, but no
training recipe has yet produced a CONFIRMED winrate gain over the
builtin teacher at 288-game scale.

## CONFIRMED GAIN: PPO pilot beats builtin (2026-08-24)

Rounds 7-8 (PPO: value-head baseline, advantage = R - V(s), 3-epoch
clipped replay; same shaping/heads as round 6) broke the plateau:

    rl8:     282/600 = 47% +-4%
    builtin: 219/600 = 36% +-4%    -> CI-SEPARATED, p < 0.001

Trajectory at 288-game scale: rl6 31 -> rl7 39 -> rl8 45; extended to
600/arm for the verdict. The greedy pilot now argmax-casts Random
Encounter in ~half of castable states (0% for the first six rounds),
mulligans with its own heads (91% acc / 100% mull-recall BC on
new-deck data), and owns targets/combat/casts. What made the
difference, in order: (1) value head turning per-game +-1 into
per-decision advantages - the credit-assignment fix; (2) 3x replay
per batch; (3) per-decision lock credit; (4) the corrected decklist.
Also for the record: PPO source was briefly lost to a botched
stash/rebase during a concurrent-push untangle and recovered from the
dangling stash commit via git fsck (e6b68cec) - stash pops during
active runs writing to tracked files are a trap.

Handoff step 4 is UNLOCKED: rerun the Roaming Encounters evolver with
pilot2_rl8.pt in the loop (evolve_core policy=), so deck mutations are
judged by a pilot that actually plays the deck's plan.

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

Round 9 (same PPO recipe from rl8): confirmation 129/288 = 45% +-6% -
statistically identical to rl8. The recipe has CONVERGED at ~45-47%.
pilot2_rl8.pt remains the champion checkpoint (47% +-4% at 600 games,
+11 pts over builtin, CI-separated). Next lever is not more of the
same training - it is the evolver rerun with this pilot, or a new
algorithmic idea (value-derived shaping, curiosity exploration).


Round 10 (big-batch/low-eps/low-lr refinement from rl8): confirmation
112/288 = 39% +-6% - REGRESSED below rl8. Third data point confirms
rl8 is the peak of this architecture+recipe (rl8 47, rl9 45, rl10 39).
Training chapter closed: pilot2_rl8.pt is the champion. Going higher
needs a different class of change (bigger trunk, value-derived
shaping, curiosity) - or spend the pilot where it already wins: the
evolver rerun.


Round 11 (pure +-1 + potential-based shaping beta=0.5 from the frozen
value head, all manual bonuses OFF): confirmation 114/288 = 40% +-6% -
below rl8. Value-derived shaping does not beat the hand-tuned recipe
for winrate. Real finding though: lock_frac held ~0.45 the whole round
with ZERO lock bonus - the Random Encounter habit is now intrinsic.
rl8 remains champion. Next per roadmap: bigger trunk (option 2).


Option 2 (big trunk, D=256 x4 layers, 3.8M params): distilled from
champion archives at 78.7% cast-agreement, then PPO. rl12 35% +-6
(post-distill round), rl13 42% +-6 (continuation) - climbing at the
same slope the small net showed, one round behind rl8's 47%.
Continuing.


Big-trunk arc: rl12 35, rl13 42, rl14 41 (+-6 each) - stalled below
the small net's 47. Hypothesis: 4 decklists under-feed 3.8M params.
Round 15 = generalist curriculum (50-deck verified pool, mixed
matchups, both-seats; shaping/validation unchanged) from rl14, GPU
updates. Pool grows in future rounds if the gate approves.


Round 15 (generalist, 50-deck pool, 50/50 mix): confirmation 110/288 =
38% +-6% - roughly held rl14's level (41 +-6) while gaining general
skill: sharpest value head yet (vloss 0.52) and record unforced
lock_frac (0.55-0.68 at eps floor). Mild specialist dilution as
predicted. Round 16: rebalanced 75/25 curriculum. Also shipped:
GPU-batched PPO (7.9x update speedup; OOM fixed via per-minibatch
backward), full both-seats archiving of every game (training corpus
for future models), all-games iteration progress, player-turn
reporting fix (Forge counts half-turns).

