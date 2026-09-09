# The N400 is not broken -- CONTEXT->PREDICTION has no expectation signal (#28)

**Registered:** `research/experiments/n400_landing_2x2.py` +
`prediction_landing_surprise` (erp/adapters.py), both at d3282ce
**Artifacts:** `n400_landing_2x2_results.json` / `.log` (10 seeds, 4 arms,
2 channels, 112 items/arm/channel)
**Figures:** `research/figures/fig_28_{aucs,saturation}.{png,pdf}`

## Question

#28 has stood for weeks as "N400 is saturated / bit-identical across
parse arms, i.e. not reading parse state at all." Is that a readout
defect -- the shipped energy channel measuring the wrong thing, the way
#121 found for role binding -- or a real absence of signal upstream?

## Verdict: NOT A READOUT DEFECT. CONTEXT->PREDICTION carries no
## next-word expectation at ANY grain, on EITHER channel. #28 closes
## as MEASURED.

10 seeds, 3 contrasts (semantic/syntactic/novelty), both channels paired
per parser, AUC mean ± sd:

| contrast   | energy (shipped)   | landing (WHERE)     |
|------------|--------------------|----------------------|
| SEMANTIC   | 0.440 ± 0.159      | 0.473 ± 0.096        |
| SYNTACTIC  | 0.451 ± 0.128      | 0.448 ± 0.138        |
| NOVELTY    | 0.145 ± 0.128      | 0.211 ± 0.144        |

- **P-SAT: PASS.** Energy's semantic AUC (0.440) sits comfortably under
  the 0.65 insensitivity bar -- the #28 premise holds precisely, on the
  post-#151/#121 engine, not merely as a stale carryover.
- **P-LIVE: PASS.** Neither channel is a dead constant: every arm shows
  real per-item range (energy 0.035-0.081, landing 0.067-0.20, n=112
  items/arm). The saturation is compression toward ceiling with graded
  structure underneath (fig_28_saturation), not a broken probe returning
  one number.
- **P-SEM: resolves to a reading NOT in the three pre-stated options.**
  Landing's semantic AUC (0.473 ± 0.096, CI crosses 0.5) is not the
  clean ~0.5-with-P-SYN-separating case, because...
- **P-SYN: FAILS the 0.75 bar.** 0.448 ± 0.138 -- also chance. Even the
  coarsest contrast available (a verb landing in a noun slot) generates
  no landing-channel surprise. Unlike #121, where the binding channel
  clearly beat drive on pathway learning (0.916 vs 0.800), landing does
  NOT beat energy here on either contrast (0.473 vs 0.440 semantic, 0.448
  vs 0.451 syntactic) -- both channels are equally blind. That rules out
  "wrong channel" as the explanation the #121 precedent would suggest:
  the deficit is upstream of the readout, in the CONTEXT->PREDICTION
  settling itself.
- **This is #87 confirmed from the ERP side.** #87 registers "next-token
  architecture cannot represent a transition" as an open architectural
  limit. Measured directly here: the pathway that would carry expectation
  delivers no discriminable signal for either "which word" (semantic) or
  "which category" (syntactic), on two independently-constructed channels
  that disagree sharply elsewhere (#121). The N400 finding and #87's
  hypothesis are now the same measurement.
- **DECISION RULE, applied literally:** "`prediction_landing_surprise`
  becomes sanctioned ONLY if P-LIVE holds and P-SEM is not inverted." Both
  hold (P-LIVE passes; P-SEM's CI straddles 0.5, not a clean inversion) --
  so landing becomes the sanctioned N400 quantity going forward, on
  methodological grounds (shares one settling implementation with the
  energy channel, no MAXIMUM-surprise escape hazard) rather than because
  it succeeds where energy failed. It does not.

## An unresolved, flagged finding: the novelty arm inverts

Both channels read novel (holdout) words as LESS surprising than the
corpus-attested continuation -- energy 0.145 ± 0.128, landing 0.211 ±
0.144, both far below chance and consistent in direction across all 10
seeds (fig_28_aucs, right panel). This is the pre-stated hazard
("the pre-norm_init post-k-WTA sign hazard... resurfaced") but the
mechanism is NOT that one: this is the pre-k-WTA energy channel AND the
neuron-ID landing channel, both post-fix, both inverting together, which
rules out a k-WTA churn explanation. Candidate mechanism, not yet
diagnosed: `_ensure_prediction_lexicon` builds each word's PREDICTION
entry ON DEMAND, the first time it is probed; novel/holdout words are
being entered for the first time within this experiment while
attested/unattested/verb words may carry entries from earlier exposure
in the same parser's lifetime, so the two groups' stored entries may be
snapshotted against different PREDICTION area states. Flagged as an open
diagnosis target for a follow-on unit, not resolved here, per the
registration's own pre-stated scope.

## Infrastructure

One settling implementation (`_settle_context_into_prediction`, extracted
verbatim from `measure_lexical_surprise`) now backs both N400 channels,
closing the same "two ways, one silently wrong" class the #52/#121 units
also hit. Tests cover both directions (undefined escapes on absent
prefix; liveness through the full curriculum). Full ERP-adjacent test
selection green except the pre-existing, separately-diagnosed
`test_shipped_frames_are_trained_on_the_parser` (see
`the_calibration_frame_set_was_never_wrong.md`).
