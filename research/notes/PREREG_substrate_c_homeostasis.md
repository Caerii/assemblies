# PREREG: substrate C — per-round homeostasis, the theorems' actual hypothesis

Registered before implementing. Follows e73e493 (norm_init intervention:
hub hypothesis dead, init-time 1/d inverts the bias 100x) and the papers
check it triggered: PNAS'20, COLT'22 Thm 6, and the sequences paper's
Thms 1-2 all ASSUME ongoing homeostasis — "after each round ... each
neuron's incoming weights sum to 1." No substrate this repo has measured
satisfies that precondition.

## The algebraic fact that makes this cheap

Plasticity here is value-independent and multiplicative (`w *= 1+beta`),
and column renormalization is a per-column SCALAR. Scalars compose: any
schedule of renormalizations (per round, per presentation, continuous)
yields the same matrix, `w_norm(i,j) = w_raw(i,j) / colsum_j(w_raw)` —
within-column ratios are untouched and the accumulated scalar is the
current column mass. So substrate C is exactly implementable as a
READ-TIME divide by CURRENT column mass: the same mechanism as
`_norm_scale`, with the divisor SUMMING weights instead of COUNTING
synapses. The count is potentiation-invariant (that is substrate B's
defect); the mass tracks learning (that is the theorems' hypothesis).

Two stated approximations, registered now:

* **Per-fiber, not joint.** The theorems normalize a neuron's TOTAL
  incoming mass; the reference code normalizes per-fiber (each matrix by
  its own column sums). We follow the reference's structure because drive
  scales are applied per-fiber in this engine. Single-input projections
  are identical either way; multi-fiber steps differ. If substrate C
  fails its bars, the joint version is the first suspect.
* **w_max.** The stored-raw/divide-at-read equivalence breaks where the
  clip binds (stored weights hit 20x init; explicitly renormalized
  weights would not). Not binding in the organ's regime
  ((1.1)^15 = 4.2 << 20); flagged for long-training regimes.
* **1-D stimulus fibers** have no within-column row structure, so
  per-fiber homeostasis degenerates there. We keep init-mode's implicit-
  population geometry with the CURRENT value in place of the snapshot:
  divisor_j = w_j(now) + p*(n - s). Graded stimulus advantage compresses
  as it potentiates (bounded, the homeostatic contract); the zero/nonzero
  code survives. Risk accepted and stated: if symbol selection degrades,
  this is where.

## Implementation contract

* `Brain(norm_init="homeostatic")` — the existing parameter grows one
  value; `True` keeps meaning init-count (substrate B), `False` off
  (substrate A). One mechanism, three substrates, no parallel flag.
* Mass bookkeeping mirrors `_deg_counts` (same rows/cols/dirty protocol,
  which already has hooks at every structural write site). The one NEW
  invalidator is multiplicative plasticity, which lives in a single
  function; it marks its winner columns mass-dirty.
* `NEURAL_ASSEMBLIES_VERIFY_MASS=1` asserts the maintained mass against a
  brute-force recount on every read — the write-site question is a
  measurement, not an argument, from day one (the count machinery's
  history demands it).
* VirtualWeights is NOT supported under homeostatic mode (its row_sum has
  no mass hook); the gate refuses to create virtual fibers there.
* Compiled-training fast paths are gated on `norm_init` truthiness and
  therefore stay off, as they must ([[compiled-training-incompatible-norm-init]]).

## Bars, stated now

Unit (before any science): U1 — maintained divisor equals brute-force
column mass + p-priced unknown rows after arbitrary train/grow/recruit
sequences (verify mode over the engine surface). U2 — substrates A and B
are byte-identical to their goldens with the feature merged (the flag off
by default changes nothing).

Science (the census, same organs/seeds/words as the registered S5 study,
`norm_init="homeostatic"`):

* **HC1 (machine intact):** zero hard defects on all 40 organs; labels
  perfect, as in both prior arms. *Prediction: PASSES (~85%).*
* **HC2 (the claim):** total soft pairs under C is BELOW substrate A's 30.
  The theorems' precondition should beat no-normalization, not just beat
  the inverted estimator. *Prediction: PASSES (~55%) — genuinely open.*
* **HC3 (consequence, conditional on HC2):** per-group exact@500 >= the
  registered table's. Reported with the tie-fragility caveat
  ([[exact-tables-are-tie-fragile]]); the census is primary, exact@L
  descriptive.
* **HC4 (mechanism readout):** soft-pair overlap distribution and intruder
  identities recorded as in e73e493, so WHATEVER happens is auditable.

## Interpretation, stated now

* HC1+HC2 pass: the theorems' substrate is the right default; register a
  follow-up to re-measure the M-ceiling ([[self-recurrence-stability-window]])
  under C on numpy_exact — the multi-assembly question reopens.
* HC2 fails with soft ~ A's rate: homeostasis neither helps nor hurts at
  this depth; the soft mechanism is elsewhere (connectivity tail);
  substrate A remains the working default.
* HC2 fails with soft >> A (B-like): per-fiber homeostasis shares B's
  defect; the joint version or the stimulus geometry is implicated —
  investigate before concluding against the theorems.
* HC1 fails: the read-time mass divide is buggy or the stimulus geometry
  broke symbol selection; fix before interpreting anything.

## Committed in advance

1. Bars before data; smoke run checks API only, numbers void.
2. The default substrate does not change in this unit regardless of
   outcome — that requires the M-ceiling follow-up and a migration note.
3. Failed bars are committed as evidence with the census artifacts.

---

## Amendment 1 (pre-data): the mechanism already exists — use it

Reading the engine before implementing found `_normalize_area_columns`
(`synaptic_scaling=`, E1 6a8e584 / E9 #138): write-time per-fiber
homeostatic scaling on area->area fibers, applied to the columns
plasticity just touched, setpoint = initial expected column sum
(rows * p, the candidate-commensurable scale). For value-independent
multiplicative Hebbian this is the SAME substrate as the registered
read-time formulation (scalar composition; a touched column is always
renormalized after its write), modulo two knowns:

* fresh recruitment columns sit above setpoint until next touched
  (pinned in test_scoped_synaptic_scaling.py) — read-time divide would
  not have this transient;
* stimulus fibers are excluded (1-D pre-summed; normalizing them erases
  the representation). The registered stimulus-geometry choice is
  therefore NOT exercised: symbol->arc drive stays raw-potentiated while
  state->arc is normalized. This shifts the conjunction's drive ratio
  ([[mood-collapse-is-a-drive-ratio]]) and is the SECOND suspect, after
  per-fiber-vs-joint, if bars fail.

Its documented failure (assembly stability 0.01, completion 0.000) is an
ATTRACTOR property — per-fiber setpoint cancels the net gain a
self-sustaining loop needs. The S5 organ has no self-sustaining loop:
blocks are assigned, state is cued, both fibers are feed-forward. The
failure mode is registered as out of scope here, not refuted.

Substrate C arm is therefore `Brain(norm_init=False, synaptic_scaling=True)`
(per-update, not deferred — the theorems say after each round), with NO new
engine code. U1/U2 unit bars are void (nothing implemented); HC1-HC4 stand
unchanged. Control-arm reproduction is cited from e73e493's fresh N1 PASS
(same engine commit, nothing touched since) rather than re-run a third time.
