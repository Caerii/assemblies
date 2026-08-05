# One canonical way: the refactor plan

**Thesis (the user's, and the evidence agrees):** the bugs come from there being
more than one way to do each thing. Where two ways exist, one is wrong, and the
substrate is TOTALIZING so the wrong one returns a plausible number instead of
raising.

This document is the plan of record. Each phase states what changes, how it is
VERIFIED, and what would count as failure. Phases are ordered so that each is
independently committable and the suite is green between them.

---

## Phase 0 — Settle the ERP cache question (BLOCKER for Phase 5)

The 4 ERP failures behind commit 724a217 **did not reproduce** (62 passed on
re-run). The committed conclusion "the suite leaks state across tests" is NOT
established and must be corrected either way.

Leading hypothesis: the failing run took **352s** where every other run took
75-128s, i.e. it rebuilt the backbone cache and therefore ran against a
DIFFERENTLY-TRAINED parser. That is the `backbone-fingerprint-gap` failure mode:
invisible cache state changing a result with no code difference.

### FINAL SYNTHESIS (third revision — the two below it are superseded)

    test_erp_metric_range.py ALONE, warm      3 passed
    test_erp_metric_range.py ALONE, COLD      3 passed
    full `-k erp` selection, warm            62 passed
    full `-k erp` selection, COLD             4 FAILED

There IS a cross-test channel, and **the warm cache MASKS it** — forks come from
a disk-loaded pristine snapshot rather than a parser trained in this session.
Cold exposes it. My first conclusion (leakage, 724a217) was closer to right than
the cache story that replaced it; the cache is the MODULATOR.

**The dispatch is probably fine.** The 10-seed harness study gives IDENTICAL
numbers warm and fully cold (0.9056 → 0.7167, every seed above chance,
`backbone_cache=OFF trained_fresh=10`). "It inverts on freshly-trained parsers"
is NOT supported. Adoption stays blocked because the SUITE cannot adjudicate —
not because the dispatch is bad. That is a different, weaker claim than the one
committed earlier, and the commit message overstated it.

**Ruled out, each by measurement:** global RNG (byte-identical **under cold**,
where training actually re-runs — the earlier warm test was VACUOUS, since
training never ran and the burn could not have mattered); calibration
contaminating forks (`pristine` precedes `_calibrate`, which defaults off); the
`fork()` pristine fallback; raw-vs-excess (a false premise, now pinned by a
test); harness arm-order.

**Open:** which shared state carries it when cold. Candidates: the in-memory
`_entries` dict, module-level ERP state, pinned-backend globals.

**Method lesson, and the reason this took three tries:** *check what a negative
result was ALLOWED to see.* The RNG test cleared a hypothesis it could not have
detected, and "precursor files pass together" was measured warm and said nothing
about cold. Bisect under every condition, not the convenient one.

### Superseded: "not leakage; not a general cache defect either."

    warm cache, default path            62 passed     75-128s
    warm cache, ERP_EXPECTED_SLOT=1     62 passed
    COLD cache, default path            62 passed     430s
    COLD cache, ERP_EXPECTED_SLOT=1      4 FAILED     339s

The default path is CONSISTENT warm and cold. **Only the expected-slot dispatch
diverges**, so the cache is not broadly poisoning results — but a cached parser
and a freshly-trained one differ in some structure that this dispatch reads and
the shipped one does not.

**The methodological finding, which is the transferable one:** the 10-seed A/B
that appeared to support adopting expected-slot dispatch ran entirely on
`get_parser_cache().fork()` — cached parsers. It does not survive fresh
training. **An A/B built on cached parsers is evidence about cached parsers
only.** This is a direct requirement on Phase 4: the harness must record, and
preferably vary, the substrate provenance.

Likely mechanism (unconfirmed, same family as the VP self-fiber finding):
pre-grown pathways wire the neurons materialised at bootstrap, later training
recruits DIFFERENT neurons, so what ROLE_PATIENT can reach is training-path
dependent.

Consequences: the rollback in 724a217 was correct, but its stated reason
("cross-test state leakage") is WRONG and is corrected in the note, the task,
and memory. Three hypotheses were ruled out by measurement first — cross-test
leakage, global RNG, and the `fork()` pristine fallback — and are recorded so
they are not re-derived.

---

## Phase 1 — Two index spaces become two types

`Area.winners` is COMPACT ENGINE INDICES; `Assembly.winners` is STABLE NEURON
IDS. Both are `uint32` ndarrays, so mixing them is accepted, returns a number,
and reads as chance. Cost so far: the merge line (voided), a retracted
"role retrieval is at chance" theory that survived a 25x beta sweep, and a
precondition gate that read 0.020 — the same bug inside the check written to
catch that class of bug.

`test_index_space_ratchet.py` already contains the honest admission: *"The defect
is MIXING the two spaces, which needs dataflow analysis to detect properly."*
**Types are that dataflow analysis.** The ratchet contains legacy sites; the
types stop new ones. This EXTENDS the existing mechanism, it does not replace it.

- `core/index_spaces.py`: `CompactIdx`, `NeuronIds` NewTypes; `SameSpace`
  value-restricted TypeVar; `to_neuron_ids`; `same_space` runtime smoke check.
  **Done.**
- Annotate the producers: `Area.winners -> CompactIdx`,
  `Assembly.winners: NeuronIds`, `Assembly.neuron_ids -> NeuronIds`,
  `diagnostics.read_assembly -> NeuronIds`.
- `overlap` gets overloads so two-of-the-same-space is accepted and one-of-each
  is a checker error. A union parameter would wrongly accept the mixed call; a
  bare `ndarray` accepts everything, which is the status quo.
### STATUS: done for the library; criterion moved, with the reason recorded.

**The guard has verified power.** `test_index_space_types.py` shells out to
pyright (a NewType is erased at runtime, so a test that does not run the checker
would assert nothing) and asserts BOTH halves: the three mixed calls ARE flagged,
the three same-space calls are NOT. Asserting only the first half would pass for
a checker that rejects everything, which is a wall rather than a guard.

**The criterion "ZERO net new pyright errors" was NOT met, and here is the
honest accounting.** Baseline HEAD 2046 errors / 511 files, measured via
`git stash` so the comparison is against real HEAD rather than a half-edited
tree. After the change: **2083, i.e. +37** — and all 37 are in `tests/` and
`programs/`, ZERO in the library:

    6  tests/test_assembly_calculus.py        3  programs/colt_mnist_tier_a.py
    6  tests/test_metrics_kernels.py          2  programs/colt_mnist_forward_completion.py
    4  tests/test_engine_e2_overlap.py        ... 12 more program/test files, 1-2 each

They are all the same shape: a raw `np.ndarray` literal passed to `overlap`,
which now demands a declared space. This is the predicted cascade, and the plan
allowed narrowing with a reason. The reason: the defect requires code that
touches BOTH an area and a stored assembly, which is library behaviour. A test
that builds two literal arrays and overlaps them cannot commit it. Forcing
`NeuronIds(...)` wrappers into fixtures buys no safety and adds ceremony, which
is how annotations get deleted.

Library sites fixed rather than suppressed, each stating its space at the point
of conversion: `ops._snap` (the one-way door), `diagnostics.assembly_overlap`
(the sanctioned pairing), `metrics.measure_n400`, `constituent_order`.

**Follow-up, tracked not forgotten:** the 37 test/program sites should be
migrated when those files are next touched. Left as-is deliberately — a
mechanical sweep of 21 files now would bury the ERP findings in the same commit.

---

## Phase 2 — Every measurement carries a definedness bit

The substrate has no ⊥, so functions invent one:

- `phrase_stability` returns `1.0` for BOTH "perfectly stable" and "there were
  no phrase areas".
- `_self_recurrent_energy` returns `0.0` for BOTH "no energy" and "the fiber
  does not exist" — which is exactly how the P600 violation arm read a constant
  for months.
- an out-of-vocabulary word returns `p600 0.0000, stability 1.0000`, the
  degenerate no-parse, indistinguishable from data.

`parse_errors.Stability` already has `.trustworthy`, and `diagnostics` already
warns "check `.trustworthy` BEFORE reading numbers". **Generalize that; do not
invent a second convention.**

### STATUS: type built and applied to the readout that caused the worst defect.

`core/measurement.py` — `Measured(value, defined, why, detail)`. The design
decision that matters: **`float()`, comparison, and arithmetic all RAISE on an
undefined value.** `.trustworthy` is opt-in and the caller has to remember; the
P600 and phrase-stability defects are precisely the cases where nobody
remembered. `energy < threshold` on a dead fiber now raises instead of quietly
answering — that comparison is the bug verbatim. `.or_else(fallback)` is the
sanctioned escape, and it makes the default VISIBLE at the call site.

The undefined value still carries NaN so a bypass degrades to NaN rather than to
0.0 or 1.0 — the two numbers that read as findings because they sit at the ends
of the range.

`_self_recurrent_energy` migrated. It previously returned bare 0.0 for FIVE
distinct conditions (area absent / no winners / projection failed /
genuinely-zero drive / **the fiber does not exist**), and the fifth is the one
that made the P600 violation arm a constant. It now checks `engine.fiber_extent`
STRUCTURALLY, before projecting — asking afterwards whether drive was zero
cannot distinguish "no fiber" from "weak assembly", because the totalizing
substrate returns k winners either way.

`measure_live_integration` now aggregates with `defined_values` and records
`stabilities_dropped`. Previously a dead fiber contributed a hard 0.0 into the
mean and pulled it toward "unstable" — a finding rather than a gap.

- **Verified:** `test_measurement.py`, 10 tests, asserting the REFUSALS (float,
  comparison, arithmetic, both operand orders) rather than the storage. A
  version that stored `defined` but let `float()` through would pass a
  storage-only test and be worthless.

**THE AGGREGATION IS DELIBERATELY NOT FIXED, and this is the lesson of the
phase.** My first version switched `measure_live_integration` to
`defined_values`, which drops undefined readings instead of averaging them as
0.0. That is the CORRECT aggregation — and it failed 5 ERP tests, because it
raises `mean_stability` and therefore moves every P600 magnitude, including
tripping the strict xfail that pins the metric's range.

Correct is not the same as free. Changing every published number is a measured
change needing its own A/B, not a side effect of a typing refactor — and today
already produced two adoptions rolled back for exactly that kind of unmeasured
coupling. So the site uses `.or_else(0.0)`, which reproduces the old arithmetic
EXACTLY while making the fallback visible at the call site, and records
`stabilities_dropped` so the honest version's cost is observable. The corrected
aggregation is one line away, behind a measurement.

This is also the first real test of `.or_else` as designed: the escape hatch
exists so that keeping a legacy default is a *stated* decision rather than an
invisible one.

**SECOND MIGRATION: `measure_lexical_surprise` (the N400 readout).** It had FOUR
bare-float escapes, and one returned **1.0 — MAXIMUM SURPRISE — when the word is
absent from the prediction lexicon.** That is exactly what a real anomaly looks
like, and a novel or held-out word is precisely the case that would be absent.
The VP dead-probe defect (#108) in the other half of the 2x2, and a candidate
explanation for #28 ("N400 is saturated / not reading parse state").

**MEASURED, AND THE HYPOTHESIS IS REFUTED: 0 of 37 probes undefined** across all
three conditions (`research/experiments/n400_undefined_census.py`). The reason is
visible once typed: `_ensure_prediction_lexicon([word])` runs BEFORE the lookup,
creating the entry on demand, so the 1.0 branch is effectively dead. It is a
LATENT hazard — worth having typed, since anything that stops pre-creating
entries would silently turn it live — but it does not explain #28. One
explanation cleared, honestly, by a census rather than by reading the code.

Behaviour preserved via `detail["legacy"]`: each undefined branch carries the
float it used to return, so the caller reproduces the old arithmetic byte-for-
byte while the fallback is visible. ERP suite 87 passed after the migration.

- **Remaining:** `anchored_p600_live`, `_predicted_energy`, `afferent_energy`
  (9 more ⊥-inventions in adapters.py), `diagnostics` probes, the `nemo/`
  phrase-stability twins; the corrected aggregation still awaits its A/B.

**NOTE ON CACHE INVALIDATION:** Phase 1 edits `core/` and `assembly_calculus/`,
both of which ARE in `_TRAINING_SOURCE_DIRS`, so the backbone fingerprint
changes and every ERP run after Phase 1 retrains (~340-400s vs 75-128s). Expect
it; it is not a regression, and per Phase 0 the cold path is the trustworthy one.

---

## Phase 3 — Name the three quantities all called "p600"

`s.p600` (raw deficit), `s.p600_excess` (clipped over baseline),
`separation["p600_auc"]` (AUC computed on the excess). Three quantities, one word
in conversation — I compared two of them today and briefly believed they
contradicted each other.

### STATUS: done, and my own characterisation of the problem was wrong.

**Correction first.** I described the third quantity as "AUC on the excess". It
is not — `calibration.py` builds `separation["p600_auc"]` from
`catv_p600_raw, gram_p600_raw`, so it ranks the **RAW deficit**. Believing
otherwise produced a whole false explanation for a real discrepancy and cost
hours. The naming problem is real (the key `p600_auc` says nothing about which
quantity it ranks) but it is a different problem than I stated.

**Not a rename.** Goldens, tests and saved reports depend on `sample.p600`,
`sample.p600_excess` and `separation["p600_auc"]`. Renaming the storage would
fork every one of those into old-name/new-name pairs — literally adding a second
way, which is the disease. So the fix is ONE UNAMBIGUOUS READER:

`ErpQuantities` + `ErpCalibrationReport.p600_quantities()` returns all three side
by side under names that say what they are — `deficit_raw`,
`excess_over_baseline`, `auc_of_raw` — plus `span_of_raw` and `n`, because a
perfect ordering across 0.7% of the scale is both a perfect ordering and a
saturated metric, and either fact alone misleads. `n` exposes the AUC
granularity (3x3 frames → steps of 1/9), so a change under ~0.11 cannot be read
as an effect.

- **Verified:** `test_erp_quantities.py`. The load-bearing one is
  `test_auc_is_computed_on_raw_not_on_excess`, constructed so raw and excess
  rank the arms OPPOSITE ways — it reads 1.0 now and would read 0.0 if the
  computation ever switched. Plus a live check that `p600_quantities().auc_of_raw`
  equals `separation["p600_auc"]`, which fails the moment the name and the
  computation diverge. The belief that cost the time is now a test, not a
  docstring.
- A missing arm yields **NaN**, not 0.0 or 0.5 — the totalizing-substrate
  failure would otherwise reappear in the reporting layer.

---

## Phase 4 — One experiment harness

```
141 experiment scripts
 88 re-implement the sys.path preamble
 60 hand-roll a mean over seeds
 17 use diagnostics.ensemble
```

Every script re-derives its own protocol, so each can independently get pairing,
seeding, or ordering wrong — and mine did today: both arms in one process, `obs`
first, so `exp` was only ever measured WARM while the suite measured it COLD.

- `research/harness.py`: `study(arms, seeds, measure, criteria=...)` that
  - isolates or counterbalances arm order (kills the warm/cold confound),
  - always returns paired ensembles with CIs via the EXISTING
    `diagnostics.ensemble` / `paired_delta`,
  - stamps the backbone fingerprint and cache-hit status into the result,
  - evaluates PRE-REGISTERED criteria and prints PASS/FAIL.
- Pre-registration worked today (it is why a 0.75 was not quietly accepted);
  make it a field, not a matter of discipline.
### STATUS: `research/harness.py` built; 12 tests, all encoding real failures.

`study(arms, seeds, criteria, order)` + `Criteria` + `Provenance`. It CALLS
`diagnostics.ensemble` / `paired_delta` rather than reimplementing them —
statistics live there, protocol lives here, and a second statistics
implementation would be the disease.

What it enforces, each traceable to a specific mistake:

- **Counterbalanced arm order.** The old script ran control-then-candidate every
  seed, so the candidate was only ever measured WARM while the suite measured it
  COLD. `order="as_given"` keeps the old behaviour but must now be chosen.
- **Substrate provenance in the artefact.** `Provenance` records disk hits vs
  fresh trainings. Phase 0's whole finding was that an A/B on cached parsers is
  evidence about cached parsers only; that limitation is now printed, not
  reconstructed later from runtimes.
- **Pre-registration as a FIELD.** `Criteria(above, on_every_seed, must_vary,
  delta_excludes_zero, allow_decrease)` is stored in the result, so the verdict
  is reproducible from the artefact alone rather than from discipline.
- **`allow_decrease=True` by default**, deliberately: removing a confound should
  shrink an inflated effect, and a bar that treats every decrease as failure
  selects for confounded metrics.

**Verified by the failures it catches**, not its happy path:
`test_a_deliberately_inverted_arm_fails` is the true-negative — a harness that
cannot reject is decoration. `test_a_single_inverted_seed_fails_even_when_the_mean_passes`
first PROVES the mean-only check is fooled by the seed-42 shape, then shows
`on_every_seed` catches it. Plus a constant-candidate rejection (the
`afferent_energy` zero-variance signature) and an honest-smaller-effect
acceptance.

`research/experiments/erp_expected_slot_study.py` ports the real A/B onto it and
reads the three p600 quantities through Phase 3's sanctioned reader.

---

## Phase 5 — Retire env flags as the A/B mechanism

28 distinct `os.environ` reads, consulted deep inside call stacks.
`ERP_EXPECTED_SLOT` is read INSIDE `measure_live_integration`, so the same call
means different things depending on process-global state, and experiments mutate
`os.environ` with save/restore around it.

### STATUS: the VALUE exists and owns the semantics; threading is deferred, stated.

`erp/protocol.py` — `ErpProtocol(expected_slot, afferent_energy, debug)`, frozen,
with `from_environment()` as the ONE adapter and `.with_(...)` to derive an arm.
`adapters.py` now delegates both flag reads to it, so the spelling rules
("0"/"false"/"off"/"" are all off; "1"/"true"/"on"/"yes" are on) live in exactly
one place instead of being re-spelled at each site — an earlier version of
`_expected_slot_enabled` defaulted ON and treated any unrecognised value as
enabled, so the two spellings of one intent behaved differently.

Why frozen and why `.with_()`: an arm becomes an EXPRESSION
(`base.with_(expected_slot=True)`) rather than a moment in time. The thing being
removed is `os.environ` mutation as an argument-passing mechanism — it cannot
nest, it leaks on exception, and it makes "which arm produced this number" a
property of when you looked.

**THREADING THE VALUE THROUGH `measure_live_integration` → `phrase_stability` IS
DELIBERATELY NOT DONE.** That touches the live ERP metric, which is under active
investigation (#108) and whose magnitudes must not move as a side effect of a
plumbing change. Two adoptions were rolled back today for exactly that coupling,
and Phase 2 hit it again. The value, the adapter and the tests are in place; the
call-chain change goes behind its own measurement.

- **Verified:** `test_erp_protocol.py`, 19 tests — frozen-ness, derivation
  without mutation, the full truthy/falsey spelling set, and an INJECTABLE
  environment so no test has to mutate the real one (a config reader that can
  only read `os.environ` forces tests to reintroduce the problem being removed).
  Flag semantics confirmed unchanged end-to-end: default off, `=1` on, `=0` off.

---

## Standing rules for this work

1. **Green between phases.** Each phase commits separately with the suite green.
2. **Construct the true negative.** A guard that has never failed has unmeasured
   power. For each new check, show it firing on the defect it targets.
3. **Baseline before changing.** Record pyright counts / test counts BEFORE, so
   "no new errors" is a measurement rather than an impression.
4. **Do not add a second way.** If a mechanism exists (ratchet, `.trustworthy`,
   `diagnostics.ensemble`), extend it. Adding a parallel one is the disease.
