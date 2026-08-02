# Pre-registration III: is CONTEXT's collapse caused by TRAINING-time recurrence?

**Written 2026-08-02, BEFORE implementing or running.** Follows study II, where
a recurrent CONTEXT area HALVED performance (0.1046 vs 0.2074 no-context) and
collapsed to 0.7566 mean overlap across different prefixes.

The repo's standing claim ([[recurrence-is-the-collapse-channel]]) is that
SELF-RECURRENCE DURING TRAINING is the collapse channel, and that feed-forward
builds show no comparable ceiling. Study II is consistent with that but did not
test it. This does.

## Arms (all else identical to study II; n=10000, k=200, p=0.05, beta=0.10,
## 3 rounds, 200 sentences, seeds 42..51, paired)

| arm | CONTEXT->CONTEXT during TRAINING | during READOUT |
| --- | --- | --- |
| `recurrent` (study II) | open | open |
| **`train_off`** | **CLOSED** | open |
| `ff` (control) | closed | closed |

`train_off` is the arm of interest: the fiber is closed while weights are
changing and opened only to read. If the collapse is a training-time
phenomenon, this keeps the history pathway while avoiding the merge.

## Hypotheses

**H1 — the collapse is training-time.** `train_off` CONTEXT overlap across
different prefixes < 0.5 (vs 0.7566 recurrent).
*Prediction: YES.*

**H2 — and performance recovers.** `train_off` paired difference vs
`recurrent` > 0, and `train_off` reaches at least the no-context 0.2074
(lower bound > 0.2074 − 0.0126 = 0.1948).
*Prediction: YES, recovers to roughly no-context level.*

**H3 — but it still will not beat the bigram optimum (0.2338).**
*Prediction: FAILS to exceed it.* Closing the fiber during training prevents
the merge but also prevents the history pathway from LEARNING, so CONTEXT
should end up carrying little beyond the current word. If H2 passes and H3
also passes, the conclusion is that avoiding collapse is necessary but not
sufficient, and the missing piece is a mechanism that lets history be learned
WITHOUT merging -- i.e. gated recurrence, not absent recurrence.

**H4 — `ff` control ≈ no-context.** Confirms CONTEXT contributes nothing once
recurrence is gone in both phases, so any `train_off` gain over `ff` is
attributable to the readout-time recurrence specifically.

## Committed in advance
Report all arms. No parameter changes. H1/H2 passing with H3 failing is the
expected and most informative outcome, and is reported as such rather than
spun as a success.

## Results
*(empty — appended after the run)*

**Run 2026-08-02, 10 seeds (42..51), paired.**

| arm | MRR | CONTEXT overlap across prefixes |
| --- | ---: | ---: |
| `recurrent` (study II) | 0.1046 ± 0.0128 | 0.7566 ± 0.0958 |
| `train_off` | 0.1107 ± 0.0138 | **0.1755 ± 0.0207** |
| `ff` control | 0.1107 ± 0.0138 | 0.1755 ± 0.0207 |
| *(no-context reference)* | *0.2074 ± 0.0126* | — |
| *(bigram optimum)* | *0.2338* | — |

**H1 CONFIRMED, and it is the clean result.** Closing CONTEXT->CONTEXT during
TRAINING drops cross-prefix overlap from **0.7566 to 0.1755**. The collapse is
a training-time phenomenon, exactly as [[recurrence-is-the-collapse-channel]]
predicts. Pre-registered threshold was < 0.5; observed 0.1755.

**H2 WAS NOT TESTED — the arm was a dead fiber.** `train_off` and `ff` are
identical to four decimals on both metrics across ten seeds, which is not a
result but two arms running the same computation. Cause: with CONTEXT->CONTEXT
closed throughout training the fiber is never materialised, so opening it at
readout projects through a connectome that was never grown -- zero drive, and
k winners returned regardless. This is [[silent-no-op-dead-fibers]]: the
symptom of a dead pathway is "mechanism X has little effect", not an error.

The `train_off` design is therefore invalid as written. Testing H2 needs the
fiber to EXIST while not being trained into a merge -- e.g. materialised and
plastic for a warm-up, then gated closed -- which is a different experiment
and needs its own pre-registration.

**Independent of that, CONTEXT hurts even when it does NOT collapse.** At
overlap 0.1755 the area is not merged, yet MRR is 0.1107 against 0.2074
without CONTEXT. So the damage is not only collapse: an extra afferent that
carries no learned information still injects competing drive into PRED's
k-WTA and displaces the informative LEX->PRED signal. Adding an area is not
free.

## Standing conclusion after three studies

* A transition CAN be learned (study I): 0.2074, 81.6% of the bigram span,
  with the grounding pathway shown necessary by ablation.
* A recurrent helper area CANNOT hold history under Hebbian k-WTA (study II):
  it merges to 0.7566 overlap and halves performance.
* That merge is caused by recurrence DURING TRAINING (study III, H1).
* Whether gated recurrence can hold history without merging is still OPEN --
  the arm meant to test it was dead.
