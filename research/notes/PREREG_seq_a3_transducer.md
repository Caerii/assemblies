# PREREG A3: a transducer state instead of a recurrent accumulator

Registered before implementing or running. Successor to
`PREREG_context_beyond_bigram.md` (#14), whose parameters and reference values
are inherited unchanged so the arms are comparable.

## What #14 established

| arm | MRR |
| --- | ---: |
| chance | 0.0900 |
| unigram | 0.1178 |
| no-context model (bigram by construction) | 0.2074 ± 0.0126 |
| **bigram optimum** | **0.2338** |
| CONTEXT arm (recurrent accumulator) | **0.1046 ± 0.0128** |

CONTEXT did not merely fail to help, it HALVED performance, and the mechanism
was measured: cross-prefix CONTEXT overlap **0.7566 ± 0.0958**, i.e. collapse
to one attractor, which then injects the same drive into PRED on every
prediction and swamps the informative signal.

#14's own conclusion was that the result is not "AC cannot hold context" but
"a recurrent helper area cannot hold context under Hebbian k-WTA". Its proposed
next step was to close CONTEXT's self-recurrence.

## What A3 does differently

The plan of record says context is TRANSITIONS, not accumulation
([[SEQ-TIME-IN-WEIGHTS]]). So the recurrent accumulator is replaced by the
organ A1 and A2 validated -- a feed-forward state with NO self-fiber, updated
only through a refracted conjunctive arc:

    stim[w]           -> LEX                current word
    gstim[w]          -> OUT                grounding signature (as in #14)
    LEX + SEQ_STATE   -> SEQ_ARC            the conjunction (history, word)
    SEQ_ARC           -> SEQ_STATE          state update, feed-forward
    SEQ_ARC           -> OUT                prediction from the conjunction

OUT is fired together with the state update during training, which is what
makes this a transducer rather than an acceptor ([[SEQ-TRANSDUCER]]).

Three differences from #14's CONTEXT, each one a documented failure channel:
no self-fiber ([[recurrence-is-the-collapse-channel]]); the update passes
through a conjunction with refraction ([[ARC-CONJUNCT-EXPOSURE]]); and the
organ runs at its own regime ([[SEQ-ORGAN-EMBEDS]]).

## Parameters

Inherited from #14 and FIXED: n=10000, k=200, p=0.05 ambient, beta=0.10,
3 train rounds per pair, 200 training sentences, seeds 42..51, same generator.

Organ-local, and NOT inherited because #14 had no organ:

* `organ_p` on the arc's and state's fibers, set so kp clears the floor. At
  k=200 the floor for n=10000 is 3 ln 10000 = 27.6, so organ_p = 0.2 gives
  kp = 40. [[SEQ-REGIME]]
* `n_arc` is SWEPT, and this is a design axis rather than a tuning knob.
  [[REFRACTION-NEEDS-LOAD]] says the arc has an operating window in load
  M*k/n, and for an emergent state M -- the number of distinct (state, word)
  conjunctions actually visited -- is NOT KNOWN IN ADVANCE. It cannot be, since
  it is what the experiment is asking about. Sweeping n_arc and reporting the
  whole curve is the honest form; picking the best cell and reporting it alone
  would be exactly the retrospective-tuning failure #14's prereg was written to
  avoid. The load achieved is reported per cell.

## Hypotheses, with honest predictions

**H1 — beats #14's CONTEXT arm.** Lower bound of the paired difference against
0.1046 > 0. *Prediction: PASSES.* The self-fiber was the measured collapse
channel and it is gone.

**H2 — beats the no-context model.** Lower bound > 0.2074.
*Prediction: uncertain, roughly even odds.* This requires the state to carry
information the current word does not, and to deliver it through OUT.

**H3 (the real question) — beats the bigram optimum.** Lower bound > 0.2338.
*Prediction: FAILS.* [[SEQ-STATE-CODE-EMERGENT]] is unproven and this is
exactly where it bites: in A1 and A2 the state alphabet was GIVEN and the
transitions teacher-forced, whereas here nothing tells the state what to
encode. An untaught state is whatever the arc's dynamics produce, and there is
no reason that should align with predictive prefix structure.

**H4 (mechanism, reported regardless) — the state does not collapse.** Mean
pairwise overlap of state assemblies across DIFFERENT prefixes at the same
position < 0.5, against #14's 0.7566. *Prediction: PASSES.*

**H5 (null, runs FIRST) — beta = 0 at chance.** Upper bound < 0.1178. If it
beats chance the study stops and becomes a bug hunt.

## Interpretation, stated now

The informative outcome is H4 passing while H3 fails. That would say the organ
FIXES state maintenance -- the thing #14 diagnosed -- and that the remaining
gap is state INDUCTION, localising the problem to
[[SEQ-STATE-CODE-EMERGENT]] rather than leaving it diffuse. It would also mean
every sequence result so far concerns running a machine over a given alphabet,
and that the alphabet is the whole remaining question for language.

H3 passing would be a much larger claim than anything measured here so far, and
would need an independent replication before it goes anywhere.

H4 failing while H3 fails means the organ did not fix maintenance either, and
the transducer inherits #14's problem rather than solving it.

## Committed in advance

1. H5 first. No parameter changes after seeing results.
2. The full n_arc curve is reported, not its best cell.
3. H4 is reported whatever H3 does -- it is the mechanism, not a consolation.
4. A degenerate-arm audit accompanies any result at or above the bigram
   optimum, since a readout that can reach 0.2338 without using the state is
   the failure mode this architecture is most exposed to.

---

## Amendment (2026-09-04, before any bar is read on the new substrate): the hashed organ, 20 seeds

The registered run (seeds 42..51) was executed on THREE seeds (42, 43, 44)
because a cell cost ~1 minute of numpy per seed; its verdicts stand on a
bound three seeds cannot support. The study is re-run, bars UNCHANGED, on:

* SUBSTRATE: `HashedTransducer` (DESIGN_sequence_port.md) -- the organ's
  areas, fibers and clock on the hashed substrate, gated to reproduce the
  numpy organ's net drive and ARC refraction bias within 5e-6 per
  projection (GATE-1) and to be identical across launch width (GATE-4).
  Connectomes are the engine's pair seeds; stimulus connectomes are hashed
  rather than RNG-drawn, so per-seed values are not the numpy run's, the
  distribution is.
* SEEDS: 42..61 (20). Each seed keeps its own corpus (`corpora(seed)`);
  brains run in launches sized to memory (one brain per launch at
  n_arc = 50,000, where a fiber is 1 GB).
* The CONTEXT arm (#14's accumulator) is numpy and is re-run on the same 20
  seeds in a process pool, so H1 stays paired.
* NOT PORTED, reported as not measured: the achieved arc LOAD (greedy count
  of distinct arc assemblies). H4 (cross-prefix state overlap) is measured
  on the hashed organ's state winners at the same positions.
* Ties in the readout's k-WTA break by the substrate's deterministic
  jitter (1e-6), not the engine's; the `rank` tie rule (random, seeded) is
  the registered one.

Reading order as registered: H5 first; the whole n_arc curve; H1..H4.

## Result on the hashed organ (2026-09-04, 20 seeds, bars unchanged)

    H5 null (beta = 0)      0.0885 +/- 0.0052   upper 0.0937 < unigram 0.1178      PASS
    a3(n_arc = 2000)        0.1801 +/- 0.0089
    a3(n_arc = 10000)       0.2054 +/- 0.0092
    a3(n_arc = 50000)       0.2075 +/- 0.0106   best cell by mean
    CONTEXT (#14, numpy)    0.1099 +/- 0.0085   the same 20 seeds
    H1  a3 - CONTEXT        +0.0976 +/- 0.0158  lower bound 0.029 > 0            PASS
    H2  beats no-context    lower 0.197 vs 0.2074                                FAIL
    H3  beats bigram        0.2338                                               FAIL
    H4  state overlap       0.0728 +/- 0.0060   upper 0.079 < 0.5                PASS
    state-blind audit       0.2144 +/- 0.0107;  a3 - blind = -0.0068 +/- 0.0079

**Reading.** With power, H1 flips: the induced-state organ beats #14's
recurrent accumulator by a paired 0.10 MRR on every seed's own corpus
(three numpy seeds had left the bound below zero). The state does NOT
collapse (H4: 0.07, against #14's 0.76 and the numpy organ's 0.17). And
yet the state carries no information the current word does not: with the
state held EMPTY the readout scores the same or better (-0.007 +/- 0.008),
and the best cell sits exactly on the no-context model's 0.2074 -- the
organ is a bigram model by construction, as the registration predicted for
H3 ([[SEQ-STATE-CODE-EMERGENT]] unproven), and its distinctness is not
information ([[distinctness-is-not-information]], again). The n_arc curve
is flat above 10,000: the arc's load is not what binds.

**What the run found on the way (substrate):** the first width run read
0.12 with the state collapsed at 0.82 and an arc bias of 66. The cause was
`topk_select` ranking NEGATIVE net drives above positives -- refraction is
the only producer of negative drives, and the selector had only ever been
gated on replayed winners. Fixed in 1b475fc; GATE-1 could not have caught
it (it replays winners). A stimulus-model difference found first
(Binomial counts vs the engine's zero-or-size draw) was NOT the cause but
is now the organ's default, as the engine's model. The Binomial-stimuli
run's file is kept (`..._hashed_binomial_stimuli.json`): with the selector
defect it also read a state-blind delta of exactly zero.

**Instrument.** 20 seeds x (null + 3 cells + H4 + blind) = 120 organ
trainings of ~1,060 steps plus the numpy CONTEXT pool: ~45 minutes wall,
the 50,000-neuron arc one brain per launch. The numpy run of three seeds
had cost about the same.

## Amendment 2 (2026-09-09, before running): the arc at half beta

The hashed organ's arc runs at strength 0.1 = beta. PREREG_s5_cliff_anatomy.md
Addendum 5's diagnostics derived that at s = beta refraction cancels
potentiation exactly, so an arc member's net drive is pinned at its base
value with no margin over the best outsider, and relocates when a weight
clips. A state written from an arc on that edge may be unable to
accumulate information -- a candidate for the A3 null (state distinct but
uninformative; a3 - state-blind = -0.007 +/- 0.008).

Cell n_arc = 10,000 (the curve is flat above it), the same 20 seeds and
corpora, strength 0.05, paired against the recorded 0.1 cell
(0.2054 +/- 0.0092, per-seed values in the results JSON). The arc's
MEMBER MARGIN (min winner net minus max outsider net, over the outsider's
drive) is measured at both strengths during scoring.

    A2-1  MRR ABOVE THE BETA CELL, paired: lower bound of a3(0.05) - a3(0.1)
          > 0.  PREDICTION: PASSES, modestly (the arc gains margin; the
          readout is still through the same OUT fiber).
    A2-2  STATE INFORMATIVE: a3(0.05) - state-blind(0.05) lower bound > 0.
          PREDICTION: uncertain. Passing would locate the A3 null in the
          arc's pinned margin; failing (with A2-1 passing) says the state's
          uninformativeness is structural ([[SEQ-STATE-CODE-EMERGENT]]),
          not a strength artifact.
    A2-3  STATE DOES NOT COLLAPSE (H4 restated): overlap upper bound < 0.5.
    Reported: the arc margin at 0.05 vs 0.1 (prediction: ~0 at 0.1, > 0.2
    at 0.05 -- the diagnostic's mechanism, measured in the transducer).

### Amendment 2 -- Result (2026-09-09, n_arc = 10,000, 20 seeds)

    a3(s = 0.05)          0.1883 +/- 0.0085
    a3(s = 0.1)           0.2054 +/- 0.0092   (recorded)
    paired difference    -0.0171 +/- 0.0078   every-seed range -0.044..+0.023
    state overlap (H4)    0.1532 +/- 0.0129   (0.0728 at s = 0.1)
    a3 - state-blind     -0.0178 +/- 0.0071
    arc margin (s=0.05)   0.17 +/- 0.05;  (s=0.1) 88.7 +/- 8.7 -- the ratio
                          is not interpretable as registered: at s = beta the
                          best OUTSIDER's net is near zero (every neuron that
                          ever won carries bias), so the normaliser vanishes;
                          reported, not read.

    A2-1  MRR above the beta cell        FAIL (significantly BELOW: -0.017)
    A2-2  state informative              FAIL (-0.018 +/- 0.007: the blind
                                          readout is BETTER)
    A2-3  state does not collapse        PASS (0.15; doubled from 0.07)

**Reading.** Half beta makes the transducer WORSE on every count, as it
makes the assigned-state organ worse (PREREG_s5_cliff_anatomy.md Addendum
6: at s < beta the arc collapses onto the state conjunct). The A3 null --
a state that is distinct and carries nothing the current word does not --
is NOT the arc's pinned margin. It is structural, the branch the amendment
named: [[SEQ-STATE-CODE-EMERGENT]] stays unproven, and the induced state
is a bigram model's state by construction.
