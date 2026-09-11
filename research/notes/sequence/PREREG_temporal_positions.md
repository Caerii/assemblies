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
