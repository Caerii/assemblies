# Soundness program

*Opened 2026-07-30. The durable plan this work is executed against. Update the
status column in place; do not fork this file.*

## Why this exists

On 2026-07-30 a single afternoon found that the k-WTA pricing law was
implemented twice and the two copies had silently diverged for months, sealing
areas at exactly `k` on the GPU engine while the parity suite that exists to
compare them read stability **1.000** — because a sealed area is trivially
stable. The suite had zero power against the failure class it was written for.

That is not an isolated bug. It is the fourth instance of one pattern:

| Instance | The measurement that read fine |
|---|---|
| `norm_init` skipped on explicit sources | cap-vs-assembly readouts at exactly 1.0000 |
| candidate divisor ignoring `n_pre` | sealed area, stability 1.000 |
| torch mirror of both | parity suite green throughout |
| `associate()` no-op | "marginal" effect, actually exactly chance |

**The through-line: a threshold that a degenerate state also satisfies is not a
test.** The program below is ordered so that the ability to *detect* problems is
restored before we spend effort on new results.

A second through-line, learned the same day: **every headline number in this
program must survive its own confounds before it is reported.** The golden
coverage figure moved 74% → 66% → 57% → 59% → 43% → 19% as confounds came out,
every step downward. Report the number after the corrections, not before.

## Phase table

| Phase | Goal | Exit criterion | Blocks |
|---|---|---|---|
| **0** | Green baseline | full `not slow` suite has 0 unexplained failures | everything |
| **1** | Determinism | same seed ⇒ same result across processes, both engines | precision of 2–5 |
| **2** | Materialization semantics | one definition of "when does a synapse exist" | 3 |
| **3** | Audit standing claims | every headline claim has a falsifier that has been RUN | 5 |
| **4** | Research frontier | open mechanistic questions closed or bounded | 5 |
| **5** | The thesis | compositional generalization vs deep learning, on real input | — |

Phases 0 and 1 are one push. Phase 3 is read-only and interleaves with 2.

---

## Phase 0 — Green baseline

Five tests are red as of `3833139` (full suite: 964 passed, 5 failed, 25m17s).
All five predate the pricing unification — each verified individually against
`d847630`. Until they are green or explicitly labelled, no future suite run can
be read at a glance, and every change has to be hand-diffed against a baseline.

**0.1 — The reciprocal cluster (3 of the 5).** `#53`.
- `test_literature_golden::TestPnas2020ReciprocalGolden::test_reciprocal_restore`
- `test_parity_infrastructure::test_inprocess_verify[pnas2020_reciprocal]`
- `test_torch_parity::TestReciprocalParity::test_reciprocal_recovers[numpy_sparse]`

One root cause. Note the task's own description is **stale**: it records "~0.77x
the reference", but the measured value is 0.1875 against a 0.6 floor. It is on a
CLEAN path per the exposure sweep, so it is not the pricing bugs. Torch passes
the same assertion numpy fails — that asymmetry is the lead.

**0.2 — The ERP calibration pair.**
- `test_erp_calibration::test_calibration_separates_category_violation_from_grammatical`
- `test_erp_calibration::test_fast_calibration_preserves_separation`

Triage: fold into `#32` (P600 root cause: saturated metric + `frozen()` probe
contamination) or file separately. Do not assume; check.

**0.3 — `test_cuda_kernels.py`.** `#40`, 9 of 15 fail against a refactored
engine. Port it or delete it. Leaving it red is what forced every suite run to
exclude it, which is how the fact that the suite *completes* stayed hidden.

---

## Phase 1 — Determinism

Measurement precision gates every number in phases 2–5.

**1.1 — torch_sparse is not reproducible across processes.** Same cell at fixed
`seed=1` read `w=484, 514, 506` on three runs (`#62`). Until closed, no
cross-engine difference below ~10% is readable at all.

**1.2 — `#60`** `test_reference_separate_near_chance` flips xpassed/xfailed.

This repo has been bitten three times by this exact class — `PYTHONHASHSEED`
seeds derived from `hash()`, the global RNG leak between constructions, and now
this. Fix the mechanism, not the instance: there should be one seeding path with
a test that asserts cross-process identity, not per-site patches.

---

## Phase 2 — Materialization semantics

Four open items are one family, exactly as the two pricing bugs were one family.
The pricing fix worked because the law moved to one place
(`neural_assemblies/core/_pricing.py`) and both engines call it. Lazy
materialization needs the same treatment: today there is no single answer to
"when does a synapse exist, and who is allowed to bring it into being".

- **`#62`** engines agree on pricing but not recruitment (torch `w=506` vs numpy
  `w=154` at 1000→10000). The candidate sampler is RULED OUT — forcing torch
  onto the numpy CPU sampler leaves it at 464. Evidence points at allocation:
  numpy over-allocates rows in blocks (632x200 backing `src.w=329, tgt.w=150`),
  torch allocates exactly (387x231 backing `387/231`).
- **`#47`** materialized connectome blocks are not frozen — 56.6% change
  retroactively.
- **`#41`** `read_only()` does not roll back connectome materialization.
- **`#50`** multi-source areas and the deferred-init bug.

Treat as one piece of work with one owning concept, not four tickets.

---

## Phase 3 — Audit standing claims

Mostly unfiled before this program, and where the real risk sits. All read-only;
parallelizes well.

**3.1 — The perfect-score sweep.** Every standing claim sitting at exactly 1.000
needs one invariance test: sweep a parameter that MUST matter and show the
number moves. Zero variance across a wide range is the sealed-area signature,
and we have now seen a sealed area score *better* than a correct one.

| Claim | Recorded | Source |
|---|---|---|
| Q20 reactivation fidelity | **1.000 ± 0.000** across 2–8 stimuli | `open_questions.md`, a *Critical Discovery* |
| Q12 retrieval accuracy | 1.000, no early-vs-late degradation | `open_questions.md` |
| gating role binding | 1.000 gated / 0.000 ungated | memory `gating-is-load-bearing-for-binding` |
| composition depth 3 | flat 1.0000 on shared areas | memory `composition-amplifies-overlap` |
| lexicon 256 words | acc 1.0000 in n=1000 | memory `recurrence-is-the-collapse-channel` |
| role retrieval | 1.000 | memory `role-retrieval-works-key-retracted` |

**3.2 — Seed counts on "Completed" entries.** `open_questions.md` marks nine
questions Completed. Q03 is Completed at `R²=0.601, p=0.070` — a null labelled
as a finding. Given the retraction history (`n^0.29` retracted, `n^1.49`
withdrawn, the OVS anomaly that was n=5 noise, the lesion dissociation I got
wrong twice from n=1), the mechanical check is: for each Completed entry, count
seeds in the linked result artifact and re-label anything single-run.

**3.3 — Self-recorded goldens.** `#54` `reference_pnas_golden.json` is recorded
from our own ops; `#55` the PNAS goldens record `associate` in the merge regime
(rounds=20). A golden recorded from the thing it validates is not a golden.

**3.4 — Dormant mechanisms.** Mutual inhibition: 1373 `project()` calls, ZERO
co-target a group, so the paper's inhibition has never run. Any claim phrased
"the paper's model does X" is a claim about a variant. `fiber_census` and
`ASSEMBLIES_STRICT_DRIVE` exist; the gap is that they have been used
per-investigation rather than as a repo-wide coverage sweep.

**3.5 — Residue.** `#35` back-catalogue for the two silent-failure classes;
`#63` the 5 remaining unchecked golden values + the `tacl2021 roles_found`
promotion (label the missing VERB when re-recording — pinning `[OBJ, SUBJ]`
silently would enshrine a partially-wrong parse as the target).

---

## Phase 4 — Research frontier

- **`#56`** sequence bridge 2/3 → 3/3. Next measurement is already defined: hold
  the within/bridge ratio fixed while scaling both. Merging is RULED OUT
  (overlap 0.000–0.004, below the 0.016 chance level, at every reps to 40) —
  do not re-test it.
- **`#58`** COLT22 Theorem 6 on our engine against the recorded golden.
- **`#57`** 7 papers remain. TACL21 is highest value and now has a *measured*
  reason to be looked at: its golden records the parser finding ADVERB where
  VERB was expected, and passing.
- **`#51`** locate α* at depth 1; **`#52`** role binding is in the crowding
  regime (0.15–0.22, not 1.000); **`#46`** depth is purchased with β.

---

## Phase 5 — The thesis

Everything above is infrastructure for this.

- **`#30` CHILDES.** The binding constraint on every induction result. Synthetic
  corpora have been the ceiling for a while.
- **`#29`** function-word bootstrapping does not clear chance (reopened).
- **`#33`** retire the symbolic role route, or state why it stays.
- **`#28` / `#32`** the ERP 2×2 — N400 saturated, P600 root-caused, both open.

---

## Standing rules for this program

1. **Assert the law, not a threshold.** If a degenerate state passes the
   assertion, the assertion is not measuring what it claims.
2. **Construct the true negative.** A test whose failing case has never been
   built has unmeasured power. Verify each new test FAILS on the broken state.
3. **Invariance is the proof.** Sweep something that must matter.
4. **A perfect score falsifies the measurement** until shown otherwise.
5. **Report distributions.** Determinism is not sample size.
6. **Correct the headline before publishing it.** Every confound found moves the
   number; find them first.
7. **Name the engine.** Any quantitative result on `torch_sparse` with unequal
   area sizes is engine-dependent until `#62` closes.
