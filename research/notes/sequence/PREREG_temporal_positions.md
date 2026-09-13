# PREREG: position-specific temporal representation

Registered 2026-09-10 before implementing the study runner or observing any
position-specific neural result.

## Question and correction

`SEQ-TEMPORAL-CARRY` reports that predicted-win improves next-token MRR on the
agreement chain. Its TM-9 mechanism measurement is void: the historical helper
pooled every noninitial input, including VERB and PRON tokens that directly name
subject number. `AUDIT_temporal_position_pooling.md` documents the counterexample
and the replacement observation contract. This study asks the narrower question:
does the ARC at a *distractor noun* represent the sentence's subject number, and
does predicted-win amplify that representation?

The old pooled values (0.11 at g=0 and 0.22 at g=1) motivated this correction but
cannot answer it and are not a baseline. No old frame artifact contains the needed
position identities. The seeds below have not appeared in the temporal registration.

## Fixed protocol

- Engine: `hashed_arc_fsm`; fused CUDA implementation.
- Brain seeds: integers 82 through 101, exactly, paired across all arms.
- Each seed owns its corpus: 200 training sentences generated with that seed and
  25 test sentences generated with seed + 500.
- Corpus: the fixed 50-word agreement chain at gap 2:
  `AUX N N VERB N N PRON N N TAG`. The last TAG is the prediction target and is
  not an observed input frame.
- Model: n = n_arc = n_state = 10,000; k = 200; ambient p = 0.05; organ p = 0.20;
  beta = 0.10; w_max = 20; normalized initialization; refraction strength = 0.10;
  copy state; no horizon or feature register; grounding 5 rounds; training and
  frozen evaluation 3 rounds; 64 maximum potentiations.
- Training: ground the 50 words, then present every adjacent input/target pair in
  the 200 training sentences with the established batched teacher-forced schedule.
- Evaluation: reset at every test-sentence boundary; for each processed input,
  tick with plasticity frozen, snapshot ARC, then call frozen emit so copy state
  advances. No teacher target is written. The capture must prove endpoint equality
  of organ counts, stimulus potentiations and refraction biases or the run is void.
- Arms, executed in this fixed alternating order by seed chunk: copy-state g=0;
  copy-state predicted-win g=1; g=1 with STATE inhibited immediately before every
  tick (`state-blind`). Arm configuration and order are recorded in the artifact.

The runner must use `research.runner`, require an explicit tag, refuse overwrite,
record source/commit/environment/parameters/seeds, and retain each test corpus and
every validated raw frame. A smoke run may reduce n, k, train/test counts and seeds;
it is VOID and cannot alter the registered protocol.

## Estimands

At each processed position, within each brain, compare every unordered pair of test
sentences. The position contrast is mean ARC overlap/k for pairs with the same
subject number minus the corresponding mean for pairs with different subject
numbers. Both pair groups must exist. Pair means are descriptive observations
inside one brain; they are never treated as independent replicates.

The distractor positions are {1, 2, 4, 5, 7, 8}. For each brain and arm:

- `D`: arithmetic mean of its six distractor-position contrasts.
- `D1`: mean at the first noun after each agreeing token, positions {1, 4, 7}.
- `D2`: mean at the second noun, positions {2, 5, 8}.

Twenty brain values produce Student-t 95% intervals through
`ensemble_from_values`. Arm differences use `paired_delta` with the exact seed
keys. All nine position curves are reported with per-seed values and intervals;
the direct agreement positions {0, 3, 6} are labelled positive controls and never
enter `D`, `D1`, or `D2`.

## Bars fixed before data

The 0.05 effect bar is one quarter of an assembly overlap and less than half the
invalid pooled g=1 minus g=0 difference; it asks for a material effect while not
pretending the pooled quantity predicts the corrected one.

    TP-1  Predicted-win amplification: lower 95% bound of paired
          D(g=1) - D(g=0) > 0.05. PREDICTION: uncertain.

    TP-2  State dependence: lower 95% bound of paired
          D(g=1) - D(state-blind g=1) > 0.05. PREDICTION: passes if the
          temporal state, rather than an accidental corpus imbalance, supplies
          the representation.

    TP-3  Plain conjunction carry: lower 95% bound of D(g=0) > 0.02.
          PREDICTION: uncertain. The invalid pooled value cannot settle this.

    TP-4  Constructed negative: upper 95% bound of D(state-blind g=1) < 0.02.
          PREDICTION: passes. Failure voids TP-1 through TP-3 as evidence until
          the corpus balance or instrument is explained.

`D1`, `D2`, and their paired difference are reported to locate decay but have no
confirmatory bar. No position is selected after inspection. TP-1 is the primary
mechanism result. TP-2 is required for its temporal interpretation. TP-3 determines
whether the plain conjunction already carries subject number. TP-4 is an instrument
validity gate. Failed bars and complete curves are retained.

## Adoption boundary

If TP-4 passes, report TP-1 through TP-3 exactly as measured. TP-1 and TP-2 passing
support the statement that predicted-win amplifies a state-dependent subject-number
representation at distractor arcs. TP-3 passing additionally supports inherited
carry under the plain conjunction. None of these outcomes alone proves that ARC to
OUT reads the representation, general sequence learning, robustness to arbitrary
noise, or a decay law. `SEQ-TEMPORAL-CARRY` remains scientifically suspended until
this result is incorporated through an explicit register amendment.

## Smoke validation (2026-09-10, after registration)

The first end-to-end smoke used seeds 1, 2 and 3 with the declared reduced smoke
configuration. It completed all three arms and retained 108 validated frames per
seed per arm. Its source archive validates, its run is tied to commit `45b44ab`,
and its scientific verdict is `VOID` as required. Reusing the tag was refused
before model construction. The smoke's bar outcomes have no scientific meaning.

Evidence: [immutable smoke result](../../results/runs/sequence.temporal-positions/temporal-positions-smoke-20260910/results.json).

## Result (2026-09-10, fixed seeds 82 through 101)

The registered run completed all 20 paired brains on the hashed CUDA substrate.
The source archive validates. All 13,500 raw frames (20 brains x 3 arms x 25
sentences x 9 positions) independently re-run through the analyzer reproduce the
stored position reports and summaries. Test corpora are identical across arms for
each seed. The g=1 and state-blind arms have identical post-training learned-state
digests for every seed, confirming that the negative differs during frozen
evaluation rather than training.

| Brain-level estimand | mean | 95% interval |
|---|---:|---:|
| D, g=0 | 0.0258 | [0.0226, 0.0290] |
| D, g=1 | 0.1849 | [0.1660, 0.2038] |
| D, state-blind g=1 | 0.0022 | [-0.0003, 0.0046] |
| paired D(g=1) - D(g=0) | 0.1591 | [0.1400, 0.1781] |
| paired D(g=1) - D(blind g=1) | 0.1827 | [0.1636, 0.2019] |

    TP-1  amplification lower bound 0.1400 > 0.05                 PASS
    TP-2  state-dependence lower bound 0.1636 > 0.05             PASS
    TP-3  g=0 D lower bound 0.0226 > 0.02                        PASS
    TP-4  blind D upper bound 0.0046 < 0.02                      PASS

The unbarred distance summaries locate the effect. At g=0, D1 is 0.0479
[0.0427, 0.0531], while D2 is 0.0037 [-0.0009, 0.0084]; paired D1-D2 is
0.0441 [0.0366, 0.0516]. Thus the plain conjunction carries subject-number
structure into the first distractor but it is absent at the second. At g=1, D1 is
0.2032 [0.1859, 0.2204] and D2 is 0.1666 [0.1455, 0.1877], with D1-D2 0.0365
[0.0291, 0.0440]. Predicted-win both amplifies the first step and preserves a large
representation through the second distractor. The blind arm has no distance effect.

The direct agreement-token controls are strong in every arm (position means about
0.35 to 0.50), while all six blind distractor intervals include zero. This is the
constructed separation the invalid pooled measurement lacked.

**Reading.** The temporal representation is state-dependent, is amplified by
predicted-win, and remains present after two distractors at g=1. The g=0 native
readout's failure at gap 2 is not evidence of a hidden representation immediately
before the agreement site: its D2 curve is at zero. The earlier pooled 0.11/0.22
values remain void and are not rehabilitated. This result does not measure an
alternative readout, gaps beyond two, natural language, or resistance to arbitrary
noise.

Evidence: [immutable registered result](../../results/runs/sequence.temporal-positions/temporal-positions-study-20260910/results.json).

## Schema 5 storage validation (2026-09-10, after the result)

The same reduced smoke protocol was rerun only to validate the new compressed raw
attachment boundary. Reconstructing its corpus and arm records from the attachment
is exactly equal to the earlier schema 4 smoke; it remains scientifically `VOID`.
Evidence: [schema 5 CUDA smoke](../../results/runs/sequence.temporal-positions/temporal-positions-schema5-smoke-20260910/results.json).

## Amendment 1 (2026-09-12, registered before running): the decay law across gaps 3 to 6

The gap-2 result located the carry (present at the first distractor, absent
at the second under the plain conjunction; present at both with
predicted-win). The backlog asks for the decay law: the same instrument at
gaps 3, 4, 5 and 6, twenty hashed seeds each, with a preregistered decay
model. Nothing in the gap-2 record changes; it stays protocol version 1.

**Instrument (protocol version 2).** `python -m research.runner
temporal-positions --gap G` for G in {2, 3, 4, 5, 6}, everything else as
registered above (seeds 82 to 101; the same n, k, p, organ p, beta, w_max,
rounds, sentence counts and arms). The chain at gap G is `AUX N^G VERB N^G
PRON N^G TAG`; the processed positions are 3G + 3, the agreeing tokens sit
at 0, G+1 and 2G+2, and offset j (1..G) is the j-th distractor noun after an
agreeing token. Per brain and arm the run retains `D` (mean contrast over
all distractors), `D_j` for every offset, the paired `D_1 - D_G`, and the
whole position curve. The gap-2 run under version 2 must reproduce the
version-1 estimands to the comparator's tolerance; that replay is part of
this amendment.

**Decay model, fixed now.** On the g = 1 arm, the brain-level mean carry at
offset j is modelled as geometric, `D_j = A r^(j-1)`, fitted by least
squares on the offset means pooled over the four new gaps (j from 1 to 6,
one point per gap and offset), with r reported with a bootstrap interval
over brains. A pooled r in (0, 1) with the fit's residual below the
between-gap spread of `D_1` supports one decay constant for the mechanism;
a plain conjunction (g = 0) is modelled the same way separately. No bar is
placed on r; the bars are on the quantities that decide whether the carry
exists at all at each gap.

**Bars per gap (each gap judged on its own twenty brains).**

- **DL-1, the carry starts.** Lower 95% bound of `D_1(g=1) > 0.05`.
  PREDICTION: passes at every gap (the first distractor follows an agreeing
  token whatever the gap).
- **DL-2, the carry decays.** Lower 95% bound of paired `D_1(g=1) -
  D_G(g=1) > 0`. PREDICTION: passes at every gap (at gap 2 the paired
  difference was 0.0365 [0.0291, 0.0440]).
- **DL-3, the carry persists to the last distractor.** Lower 95% bound of
  `D_G(g=1) > 0.02`. PREDICTION: uncertain, and the point of the study: the
  register entry's claim (carry across two and three distractors) predicts
  passes at gaps 3; beyond that nothing has been measured.
- **DL-4, instrument.** Upper 95% bound of `D(state-blind g=1) < 0.02`, as
  TP-4. Failure at a gap voids that gap.
- **DL-5, the plain conjunction stops at one.** Upper 95% bound of
  `D_2(g=0) < 0.02` at every gap, as the gap-2 result found (0.0037
  [-0.0009, 0.0084]).

TP-1 to TP-4 are also evaluated at every gap and reported. Failed bars are
retained with their numbers; the gap at which DL-3 first fails, if any, is
the measured horizon of the predicted-win carry and is reported as such,
not as a fitted extrapolation.

## Amendment 1, first results (2026-09-13)

**Gap 2 under protocol version 2 reproduces version 1.** Artifact
`research/results/runs/sequence.temporal-positions/temporal-positions-gap2-20260912/results.json`
(seeds 82 to 101, run from the pinned runs worktree at 24f7767e). Every
per-seed `D`, `D_1` and `D_2` value equals the 2026-09-10 record on every
arm; TP-1 to TP-4 PASS as before and the five decay bars pass at gap 2
(DL-1 0.2032 lower bound 0.1859; DL-2 paired D_1 - D_2 0.0365 [0.0291,
0.0440]; DL-3 D_2(g=1) 0.1666 lower bound 0.1455; DL-4 blind D upper
bound 0.0046; DL-5 D_2(g=0) upper bound 0.0084). The instrument change
adds nothing and removes nothing at gap 2.

**Gap 3 hit the substrate's count range.** The first gap-3 attempt failed
in the organ fiber with `OverflowError: a count passed 127`
(`research/results/runs/sequence.temporal-positions/temporal-positions-gap3-20260912-overflow/failure.json`, retained with its `run.json`): the
longer chain repeats the same noun-to-noun transitions often enough over
200 sentences and 3 rounds that an int8 potentiation count passes 127.
Every such synapse is already at w_max (the chain table clips from count
32 under beta 0.10 and w_max 20, and the drive kernel prices every count
at or beyond the table's last index with that entry), so a count held at
127 gives the same drive as the true count. The organ fiber now treats the
flag as informational exactly when its chain table is saturated within the
count range (`VERIFICATION.md#contract-organ-count-saturation`; an
unclipped table still raises). Gaps 3 to 6 run under that contract; the
protocol is unchanged.

## Amendment 1, results (2026-09-13): one decay constant, a falling amplitude, and a horizon at five distractors

Artifacts, all twenty brains on seeds 82 to 101, run from the pinned worktree
(gaps 3 to 6 at 6efd3654, gap 2 at 24f7767e):

- `research/results/runs/sequence.temporal-positions/temporal-positions-gap3-20260913/results.json`
- `research/results/runs/sequence.temporal-positions/temporal-positions-gap4-20260913/results.json`
- `research/results/runs/sequence.temporal-positions/temporal-positions-gap5-20260913/results.json`
- `research/results/runs/sequence.temporal-positions/temporal-positions-gap6-20260913/results.json`

### Bars

    gap   DL-1  DL-2  DL-3  DL-4  DL-5   TP-1  TP-2  TP-3  TP-4   verdict
    2     PASS  PASS  PASS  PASS  PASS   PASS  PASS  PASS  PASS   PASS
    3     PASS  PASS  PASS  PASS  PASS   PASS  PASS  FAIL  PASS   FAIL
    4     PASS  PASS  PASS  PASS  PASS   PASS  PASS  FAIL  PASS   FAIL
    5     PASS  PASS  PASS  PASS  PASS   PASS  PASS  FAIL  PASS   FAIL
    6     PASS  PASS  FAIL  PASS  PASS   PASS  PASS  FAIL  PASS   FAIL

DL-4 (the state-blind instrument bar) passes at every gap, so no gap is
voided: blind `D` upper bounds are 0.0042, 0.0013, 0.0026 and 0.0015 at
gaps 3 to 6.

### The carry by position

Mean subject-number contrast `D_j` on the g = 1 arm, twenty brains:

    gap   j=1     j=2     j=3     j=4     j=5     j=6
    2     0.2032  0.1666
    3     0.2073  0.1602  0.1272
    4     0.2148  0.1514  0.1173  0.0866
    5     0.1679  0.0984  0.0758  0.0580  0.0418
    6     0.1282  0.0828  0.0654  0.0521  0.0368  0.0243

### The decay model, as registered

Least squares of `D_j = A r^(j-1)` on the offset means pooled over the four
new gaps (18 points), with r bootstrapped over the twenty brains (2000
resamples, brains resampled jointly across gaps because the seeds are
paired):

    pooled r = 0.7093, 95% [0.6889, 0.7286];  A = 0.1792;  rms residual 0.0269

The registered criterion was "a pooled r in (0, 1) with the fit's residual
below the between-gap spread of `D_1`". Both hold: r is inside (0, 1) with
an interval far from either end, and the rms residual 0.0269 is below the
`D_1` standard deviation across gaps, 0.0346. **The decay model is
supported.**

Per-gap fits say where the pooled residual comes from:

    gap   r                       A       rms
    3     0.7815 [0.7329, 0.8208] 0.2067  0.0010
    4     0.7370 [0.7007, 0.7659] 0.2126  0.0031
    5     0.6896 [0.6652, 0.7150] 0.1614  0.0073
    6     0.7275 [0.6778, 0.7687] 0.1244  0.0040

Every per-gap `r` interval contains the pooled 0.71, and every per-gap rms
residual is between three and thirty times smaller than the pooled one: the
geometric SHAPE is one constant across gaps, and the pooled residual is
almost entirely the AMPLITUDE, which falls as the chain lengthens (A 0.207,
0.213, 0.161, 0.124 at gaps 3, 4, 5, 6). The pooled residuals are
systematically positive at gaps 3 and 4 (+0.023 to +0.037) and systematically
negative at gaps 5 and 6 (-0.004 to -0.051), which is that amplitude
difference and not scatter.

This is a refinement of the registered model, not a bar: the amendment fixed
one pooled `A` and one `r`, and the data support the one `r` while rejecting
the one `A`. The chain at gap G is `AUX N^G VERB N^G PRON N^G TAG`, so a
larger gap both lengthens the sentence and repeats the same noun-to-noun
transitions more often (the repetition that saturated the organ's int8
counts from gap 3 on). Nothing here separates sentence length from arc load;
that separation is not measured and is not claimed.

On the g = 0 arm the same fit gives r = 0.0080, 95% [0.0080, 0.0232], with
A = 0.0334: the plain conjunction's carry is gone after the first offset,
which is DL-5 stated as a decay constant.

### The horizon (DL-3)

Lower 95% bound of the carry at the LAST distractor, against the 0.02 bar:

    gap   D_G     95% interval        DL-3
    2     0.1666  [0.1455, 0.1877]    PASS
    3     0.1272  [0.0976, 0.1567]    PASS
    4     0.0866  [0.0662, 0.1070]    PASS
    5     0.0418  [0.0344, 0.0491]    PASS
    6     0.0243  [0.0163, 0.0324]    FAIL

**The measured horizon of the predicted-win carry is five distractors**: the
carry clears the registered bar through gap 5 and does not at gap 6. As the
amendment requires, this is reported as the gap at which DL-3 first fails,
not as a fitted extrapolation.

The failure is a bar, not an absence. At gap 6 the carry at the sixth
distractor is positive on 20 of 20 brains (0.0038 to 0.0586, interval
[0.0163, 0.0324] excluding zero) and above 0.02 on 10 of them. What fails is
the registered threshold, which was set at gap 2 where the contrast was an
order of magnitude larger.

### TP-3 fails from gap 3 on, and the reason is its estimand

TP-3 asks for the lower bound of `D(g = 0) > 0.02`, where `D` is the mean
contrast over ALL distractor positions. The plain conjunction carries at the
first position only, so pooling over more positions divides that one number
by more zeros:

    gap   g = 0 offsets                                        mean D   low      D_1 alone
    2     0.0479  0.0037                                       0.0258   0.0226   0.0479 [0.0427, 0.0531]
    3     0.0683  0.0019  -0.0009                              0.0231   0.0193   0.0683 [0.0590, 0.0775]
    4     0.0323 -0.0028  -0.0011 -0.0019                      0.0066   0.0037   0.0323 [0.0253, 0.0392]
    5     0.0183 -0.0029   0.0001  0.0050 -0.0011              0.0039   0.0017   0.0183 [0.0121, 0.0245]
    6     0.0150 -0.0035  -0.0006  0.0012 -0.0010  0.0032      0.0024   0.0011   0.0150 [0.0065, 0.0235]

At gap 3 TP-3 misses by 0.0007 with its first-position contrast at its
LARGEST value of any gap (0.0683). The bar is kept and recorded as failed;
it is not amended. The position-specific statement that survives is DL-5,
which passes at every gap: the plain conjunction's contrast is present after
the first distractor and absent after the second. Any future pooled-`D`
comparison across gaps is measuring the number of positions pooled as much
as the mechanism.

### What is retained for the register, and what is not

The paired decay contrast is a per-brain fact at every gap: `D_1 - D_G` is
positive on 20 of 20 brains at gap 6, minimum 0.0687 (interval on the mean
0.1039 [0.0929, 0.1148]). That contrast is the register's retained check for
the decay.

The tail carry is NOT a per-brain fact. At gaps 5 and 6 the last offset's
contrast does not exceed the state-blind control on every brain (minimum
paired difference -0.0022 and -0.0037): at the tail the between-brain noise
is the size of the effect, and the carry there is an ensemble statement with
an interval, not a per-brain one. No per-brain claim is made about the last
offset at gaps 5 and 6.
