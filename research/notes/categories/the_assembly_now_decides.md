# The assembly now decides — gate → record → recall, measured

## The question and the answer

"How do we make the assembly decide?" The answer, from the 2021 parser paper's
division of labor, assembled entirely from parts the repo already had:

- **GATE**: learned voice detection (`_determine_role_order`, MARKER conf 0.96)
  selects the filler sequence — agent-first for active, patient-first for
  passive. Voice is control choosing a program.
- **RECORD**: each content word's core assembly is bind-traversed (T=2) into
  the open slot. The parse's state is the role areas' winners.
- **RECALL**: the **reconstruction readout** — for each role area's current
  winners, which candidate word's projection image reproduces them?
  `occupant(R) = argmax_w overlap(image(w→R), winners(R))`. No `role_lexicons`
  anywhere, so a word never trained in a role can still be read out of it.

`research/experiments/sentence_conditioned_readout.py`, 3 seeds, 4 events x 2
voices, criteria registered before running:

| criterion | phon_weight=1 (default) | phon_weight=6, beta=0.05 | margin route |
|---|---|---|---|
| A actives | 9/12 | **12/12** | 12/12 |
| B passives | 9/12 | **12/12** | 6/12 |
| C1 voice invariance (same event) | 1.0000 | **1.0000** | — |
| C2 event separation (reversed) | 0.9528 | **0.0417** | — |
| D min occupant gap | 0.0000 (ties) | **0.8667** | — |
| D gap varies with seed | yes | yes | bit-identical |

## The three findings, in the order they surfaced

**1. The readout's errors became substrate errors.** First run: ties (gap
0.0000) exactly on `girl/horse` and `child/mouse` frames. Direct measurement:
their images in the role areas overlap **0.93–0.97**, while `dog/cat` — the
pair that always reads correctly — overlaps **0.03**. Dog and cat have
individually trained role fibers; untrained pairs ride the shared-core channel
into the area's attractor, k-WTA amplifying core overlap exactly as the
measured law says. For the first time, parsing accuracy is a measurement OF
the substrate's separation rather than of a symbolic weight.

**2. The registered lever moved the parse.** Phase B measured `phon_weight=6 +
beta=0.05` cutting core duplicates 0.32→0.06 and was deliberately never made
default. Registered prediction: it should separate the collapsed images and
lift the readout. Result: gap 0.50→0.95, ties eliminated, both voices 12/12.
**The parse now responds to a substrate parameter through the readout — the
operational definition of "the assembly decides."** The margin route, on the
same items, is indifferent to it and stays at 6/12 on passives.

**3. Voice-invariant event representation — found by my own mis-registration.**
Criterion C originally demanded "substrate state differs between frames" for
every pair. The run refuted the criterion, not the substrate: for pairs where
the passive RESTATES the active ("dog chases cat" / "cat is chased by the
dog"), the parse state is IDENTICAL — 30/30 winners shared, C1 = 1.0000. That
is not a failure; it is the strongest property available: **the substrate
carries a canonical event representation, with surface order absorbed by the
gating** — `{AGENT: dog-assembly, PATIENT: cat-assembly}` whichever way the
sentence was said. And for pairs stating the REVERSED event ("child enters
mouse" / "child is entered by the mouse"), the states share 0.0417 — nearly
orthogonal. Same words, same lexicon, opposite who-did-what, different brain
state.

The default arm's C2 = **0.9528** is the same fact inverted, and it is the
sharpest statement yet of why the phon_weight default matters: at the current
default, the substrate CANNOT DISTINGUISH "child enters mouse" from "mouse
enters child" — the event representation is degenerate — and the lever takes
that from 0.9528 to 0.0417.

## What this rests on, honestly

- The occupant's image reproducing the winners is partly DETERMINISM (frozen
  parse, same protocol both ways). The substrate's real contribution is the
  GAP to the runner-up — which collapsed at the default and is 0.87+ under the
  lever — and C1/C2, which are properties of the weights, not of the replay.
- The gate still decides WHERE projections go; the substrate records and
  answers. That is the papers' division of labor, not a substrate-only parser.
  What changed is that the ANSWER is now read off the brain and degrades with
  the brain.
- 4 events, 3 seeds, one vocabulary preset, hand-picked probe sentences. The
  scale-up (the passive_payoff 40-probe set, more seeds) is the next
  measurement, not assumed.
- Two probe-harness defects were found and fixed mid-run, both coverage: an
  unregistered surface form ('enters') first stalled the slot sequence, then
  turned out to be unclassifiable too; the sequencer now runs on filler slots
  only, with ACTION taken independently by the first verb-classified token.
- `read_only()` isolation means nothing persists between probes; a production
  wiring needs to decide what happens to the parse state afterwards.

## Consequences

1. **The phon_weight=6 / beta=0.05 default flip now has three independent
   lines of evidence**: duplicates 0.32→0.06 with role retrieval 0.97 (Phase
   B), and now event-representation separation 0.95→0.04 with both voices
   12/12. The re-baselining cost was the reason to defer; the case is now
   strong enough to spend it.
2. **#33's work is specified**: wire the reconstruction readout as the
   parser's role readout (NemoParser gains voice-aware filler sequencing; the
   margin demotes to a thematic-fit prior), then re-run `passive_payoff`
   through it at scale.
3. The C1/C2 pair is a new, cheap invariance probe worth keeping: voice
   invariance ~1 with event separation ~0 is the signature of a working
   event representation, and either number moving is a substrate regression.
