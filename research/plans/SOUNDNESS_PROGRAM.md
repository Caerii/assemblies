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

| Phase | Goal | Exit criterion | Tasks | Blocks |
|---|---|---|---|---|
| **0** | Green baseline | full `not slow` suite has 0 unexplained failures | #53, ~~#64~~ ✅, #40 | everything |
| **1** | Determinism | same seed ⇒ same result across processes, both engines | ~~#65~~ ✅, #60 | precision of 2–5 |
| **2** | Materialization semantics | one definition of "when does a synapse exist" | #69 (= #47/#41/#50 — **#62 removed**) | 3 |
| **3** | Audit standing claims | every headline claim has a falsifier that has been RUN | ~~#66~~ ✅, ~~#68~~ ✅, #67, #70, #71, #54, #55, #35, #63 | 5 |
| **4** | Research frontier | open mechanistic questions closed or bounded | #56, #58, #57, #51, #52, #46 | 5 |
| **5** | The thesis | compositional generalization vs deep learning, on real input | #30, #29, #33, #28, #32 | — |

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

**INVESTIGATED 2026-07-30. The threshold is wrong, not only the code — and my
"torch passes where numpy fails" framing was backwards.**

Torch does not pass; **it measures nothing.** Its `B→A` block is never
materialized (`_nrows=0, _ncols=0, val.sum()=0.0`) after the whole protocol, so
the back-projection hits "zero signal → preserve current assembly" and A's
winners never move — reading exactly 1.0000 on 3/3 seeds. numpy is the honest
engine and its 0.1875 is real.

**The 0.6 floor is not achievable by the REFERENCE either** at these parameters:
`.reference/dmitropolsky-assemblies` over 8 seeds gives **0.5875 ± 0.1047**
(max 0.790). The floor traces to a single-seed reading of 0.740 in an `ops.py`
docstring, and the golden's own recorded 1.0 is the dead-fiber artifact.

Our shortfall against that 0.5875 has two measured levers, both parameter
choices rather than defects: `ops.project` defaults `recurrent=False`, so the
`A→A` fiber the recovery phase relies on is never trained (0.2062 → 0.3350 when
enabled); and β=0.1 over T=10 gives `(1.1)^10 = 2.59`, below the norm_init
retention threshold. At **β=0.2** the golden protocol reaches **0.7453 ± 0.0380**
with a live scramble control at 0.1938.

So the fix is a re-derivation, not a patch: set the threshold from a
multi-seed reference distribution, and state the β/recurrence regime the claim
holds in. Do not simply lower the floor to whatever we currently produce.

**0.2 — The ERP calibration pair. ✅ DONE (`#64`, `6222266`).** Neither of the
three hypotheses I offered was the cause. `anchored_p600_live` built a source
set whose *size* was conditioned on the grammatical/violation distinction, so
the violation arm fired one extra trained source. Cohen's d **−2.4 → +1.9**,
3/3 green, ERP surface 13/13. Same family as `#23` (both arms must be matched)
but a distinct instance: there it was the *area*, here the *number of summed
sources*, which area-matching cannot catch.

**0.3 — `test_cuda_kernels.py`.** `#40`, 9 of 15 fail against a refactored
engine. Port it or delete it.

Leaving it red is not cosmetic. It is what forced every full-suite run to pass
`--ignore` or `-x`, and that is precisely how the fact that the suite
*completes* stayed hidden — on 2026-07-30 the suite was reported as finishing
in 6m49s, which was wrong; `-x` had stopped it at this file after about a third
of the files. The real figure is 25m17s. **A permanent known-red does not merely
fail to inform, it distorts how every other run is read.**

If deleting: verify unreachability first and make it its own reviewable commit
stating what was verified. Do not silently `rm` tracked code.

---

## Phase 1 — Determinism

Measurement precision gates every number in phases 2–5.

**1.1 — ✅ DONE (`#65`, `6ca3c7c`).** Both torch candidate samplers took an
`rng: np.random.Generator` and drew from torch's **process-global** stream
anyway — `_sample_truncated_normal_gpu` declared the parameter and never
referenced it; `_sample_dense_candidates` used it only on the `_deterministic`
branch. Before: `w` = 484 / 514 / 506 at a fixed seed. After: bit-identical
across three processes.

My prime suspect was `scatter_add_` float nondeterminism — real, on both hot
paths, and **ruled out by measurement.** The same defect exists in the external
reference for the same reason (`brain.py:414` calls `truncnorm.rvs` with no
`random_state=`).

`test_cross_process_determinism.py` spawns subprocesses, varies
`PYTHONHASHSEED` deliberately, and asserts winner *identities* rather than
counts. True negative verified: it fails at `w=512` vs `484` on the pre-fix
engine.

This unblocked `#62`, which now reads 154 vs 478, stable across three processes.

**1.2 — `#60`** `test_reference_separate_near_chance` flips xpassed/xfailed.
Shares this mechanism per the investigation; re-check and close if stable.

This class had bitten three times before this fix — `PYTHONHASHSEED` seeds
derived from `hash()`, the global RNG leak between constructions, CUDA's
non-Bernoulli init. **No in-process test can catch any of them**, which is the
durable lesson: the test must fork.

---

## Phase 2 — Materialization semantics

Four open items are one family, exactly as the two pricing bugs were one family.
The pricing fix worked because the law moved to one place
(`neural_assemblies/core/_pricing.py`) and both engines call it. Lazy
materialization needs the same treatment: today there is no single answer to
"when does a synapse exist, and who is allowed to bring it into being".

**`#62` HAS BEEN REMOVED FROM THIS FAMILY — my framing was falsified.** The
recruitment gap reproduces with *no area-to-area connectome at all*: one
stimulus into one area gives numpy `w=311` vs torch `w=431` (5 seeds, n=1000).
A lazy-materialization schedule cannot explain a gap that predates any lazy
block. It is two float-precision defects in the torch pricing path, with
opposite signs: the candidate→allocation **round trip** (`(10/50)*50 =
9.9999990463256836`, and `int()` returns 9, because CUDA float32 division runs
up to 1 ULP low), and **`WEIGHT_DTYPE = torch.bfloat16`** quantizing the Hebbian
product enough to reorder the 28–49 entry tie block sitting exactly at the k-WTA
cut. Each was proven by an invariance that moves the number to *exact* equality
with numpy. A third, smaller area-to-area divergence survives both and is not
yet explained. Fix separately from this task.

Remaining, and these genuinely are one family:

- **`#47`** materialized connectome blocks are not frozen — 56.6% change
  retroactively.
- **`#41`** `read_only()` does not roll back connectome materialization.
  Recruitment, not plasticity, is the channel by which measuring changes the
  measured.
- **`#50`** multi-source areas and the deferred-init bug.
- **NEW, found by the `#53` investigation and squarely in this family:** the
  torch engine has no equivalent of numpy's `_init_deferred_area_srcs`
  (`numpy_engine/_sparse.py:868`, called from all three exits of
  `project_into`). A source area whose block into the target has never been
  sized contributes nothing and is never given a Bernoulli(p) block — so it can
  **never** contribute on any later round either. Measured: after the entire
  reciprocal protocol, torch's `B→A` CSRConn is `_nrows=0, _ncols=0,
  val.sum()=0.0`. That is precisely "when does a synapse exist, and who brings
  it into being".

Tracked as one task, **`#69`**.

**The precedent to follow.** `5fa91ed` fixed the pricing divergence by moving
the *law* into `core/_pricing.py` and having both engines call it, while leaving
genuinely storage-specific extraction in each. It worked because it separated
the rule from the backend. Materialization needs the same three-way split:

| | |
|---|---|
| **when** a row/column comes into existence | the rule — shared |
| **how** it is allocated and stored | per-engine |
| **who** may trigger it | a projection may; a read-only probe may not |

Do this **after** Phase 1. While torch is non-reproducible across processes, any
cross-engine difference below ~10% is unreadable and `#62` cannot be verified.

---

## Phase 3 — Audit standing claims

Mostly unfiled before this program, and where the real risk sits. All read-only;
parallelizes well.

**3.1 — The perfect-score sweep.** Every standing claim sitting at exactly 1.000
needs one invariance test: sweep a parameter that MUST matter and show the
number moves. Zero variance across a wide range is the sealed-area signature,
and we have now seen a sealed area score *better* than a correct one.

**DONE 2026-07-30 (`#66`). Four of six do not survive.**

| Claim | Verdict | The measurement that decided it |
|---|---|---|
| Q20 reactivation | **DEAD PROBE** | β=0.0/1 round gives bit-identical assemblies to β=0.10/30 rounds |
| Q12 retrieval | **DEAD PROBE** | same protocol, same cause |
| lexicon 256 words | **VOID** as a memory result | still 1.0000 at β=0.00, i.e. with nothing stored |
| composition depth 3 | **DEGENERATE** | permuted partners score 1.0000, identical to matched |
| gating role binding | SOUND, **over-titled** | `multi` passes at β=0; sentence read == word-alone read |
| role retrieval | SOUND readout, saturated metric | β=0 null also 1.0000 to α=13.9 |

**The shared cause of the first three: one private stimulus fiber per item into
an `explicit=True` area.** Items never compete, so no item can write into
another's synapses; the winner set is fixed from round 1 and independent of β.
A positive control with a real coupling channel, using the *same metric code*,
degrades to 0.9237 by M=64 — the harness can move, the protocol has no
interference channel. Swapping M private fibers for ONE shared fiber breaks the
lexicon result (0.8672 at M=128).

**The lexicon/capacity contradiction dissolves rather than resolving.** α*≈1.15
is a depth-5 *shared-fiber* quantity, not an area constant. Crowding requires a
shared fiber **and** plasticity; neither claim was wrong about its own regime.

**What survives and is stronger than recorded:** the gating 1/S law holds
exactly across S=2,3,5,8, and β buys capacity under load (M=256: 0.5994 at β=0
vs 1.0000 at β=0.10). The depth line (`#46`) also survives — `depth_beta_rescue`
at M=32/depth-8 shows a genuine interior optimum, 0.6250 → **0.9688** → 0.7292
across β = 0 / 0.10 / 0.20. The M=16/depth-3 cell is simply too easy to show it,
which is exactly why it read flat.

**A rule this cost us.** *Shared areas make the degenerate solution unscoreable*
was argued from architecture and never measured. It is false: the cascade is
deterministic given the leaf, so the leaf key alone selects correctly at every
level. **Cue-independence must be measured, not argued.**

**3.2 — Seed counts on "Completed" entries.** `open_questions.md` marks nine
questions Completed. Q03 is Completed at `R²=0.601, p=0.070` — a null labelled
as a finding. Given the retraction history (`n^0.29` retracted, `n^1.49`
withdrawn, the OVS anomaly that was n=5 noise, the lesion dissociation I got
wrong twice from n=1), the mechanical check is: for each Completed entry, count
seeds in the linked result artifact and re-label anything single-run.

**3.3 — Self-recorded goldens.** `#54` `reference_pnas_golden.json` is recorded
from our own ops; `#55` the PNAS goldens record `associate` in the merge regime
(rounds=20). A golden recorded from the thing it validates is not a golden.

**3.4 — Dormant mechanisms. DONE 2026-07-30 (`#68`), and it found the largest
result of the program so far.**

**The neural coin is not neural** (`#70`). `RandomChoiceArea._flip_k_split`
settles with `project({}, {area: [area]})` — and that self-fiber's weight block
is `(0,0)`, so the loop delivers zero drive and returns the incumbent winners.
Verified on four falsifiers: `rounds` = 0 / 1 / 10 give **200/200 per-flip
agreement**, the block is empty, winners move in **0 of 10** rounds, and a cross
fiber control moves them 2/5. The outcome is decided entirely by the numpy RNG
that builds the k-split mix. This underpins `test_pfa`, `test_nemo_fsm`,
`test_computation_value`, four `TestCoin2024*Golden` classes and the
`coin2024_*` protocols — **a paper reproduction that does not use the model.**

Root cause is general: `project_into` registers deferred init only when
`src_name != target`, so **self-fibers are excluded by construction** and one
first driven after its area stops growing is permanently dead. 27 fibers dead on
100% of uses, 23 of them self-fibers. `ensure_area_conn` — the repair for
exactly this — is called **zero** times in 631,925 projections.

**RESOLVED 2026-07-30** (`510285a`, `384fe30`). Waking the fiber was necessary
but not sufficient: lazy materialization sizes the block to `w`, not `n`, so a
uniform `k`-subset of `n` still addressed mostly nothing (82% of the seed at
`n=2000`). `NumpySparseEngine.materialize_area` closes that, and
`RandomChoiceArea(construction="attractor")` ships the validated build.

Measured through the shipped API at `n=2000, k=200`: overlap with the winning
attractor **0.985** vs `legacy`'s **0.159**, against a `k/n = 0.100` floor. The
standalone ladder (`research/notes/neural_coin_fairness.md`) takes it to 1.000
by `n=8000` and finds basin asymmetry scaling as **`k^-1.01`, log-log
r = -0.997** over a 32× range in `k`.

A **third** defect surfaced during the port: `_flip_k_split` — the package
*default* mode — fed `Assembly.winners` (neuron IDs) into `set_winners`
(compact indices), so only 7 of 50 stored ids addressed a real slot. The two
defects were **masking each other**: with the fiber dead, the out-of-range seed
never indexed any weights and so could not crash. Generalizable: in a system
this quiet about failure, a single-defect fix should be assumed insufficient
until measured.

Still open on `#70`: re-record the `coin2024_*` goldens against `attractor`
(they currently pin `legacy`, and their flip counts are decorative by a
decision made when the counts could not move).

**The diagnostic itself was broken** (`#71`). `fiber_census` reads
`conn.weights`; `CSRConn` has no such attribute, so every torch fiber reads
`(0,0)` and a healthy `nnz=1314` fiber is flagged dead. And raw
`silently_ignored` over-reports by **~700x** — 13,905 dead-into-live of which
13,885 were never *driven*. Splitting on "was it ever driven" leaves 20.

**Mutual inhibition confirmed, with a refinement**, because *dormant* and
*unreachable* are different claims. 1 co-target in 142,844 calls on a brain with
a declared group — and that 1 is the test written to force it. But
`NemoParser(competitive=True)` **does** co-target (3 times in one 3-word parse),
because its project map is *derived* from `InhibitionState` rather than named by
the caller. Both tests reaching it are `slow`-marked. Reachable by exactly one
route; nothing in the default suite takes it.

Never firing: `_use_compiled_projection` (611,106 calls, 0 fires),
`ensure_area_conn`, `_sample_area_weights`, `Brain.normalize_weights`,
`Brain.remove_mutual_inhibition`.

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
