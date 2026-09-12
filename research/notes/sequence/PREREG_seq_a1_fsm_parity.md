# PREREG A1: the transition organ, end-to-end, against the reference golden

Registered before any Brain-side numbers are read. Design rests on
`the_arc_collapses_onto_whichever_conjunct_fires_more.md` (commit d5e64f8),
which characterizes the vendored reference; the plan of record is
`state_tracking_reading_synthesis.md` (065fdca).

## Question

Can our `Brain`, running Theorem 4's three-area FSM, decide the mod-3 language
**end to end** -- with the answer read out of the state assembly rather than
looked up in a table? The vendored reference does (3/3 seeds). Nothing in this
repo has ever demonstrated it, because the FSM programs we ship return a
dictionary lookup and score perfectly untrained.

## Changes required before measurement

Each lands as its own commit and claims no numbers. Behaviour changes are
separated from guards, per the literate-architecture directive.

* **C1 (guard, behaviour-neutral).** `numpy_exact` must REFUSE
  `refracted=True` instead of silently ignoring it. `Engine.set_refracted`
  documents "default is a no-op" and `_UNSUPPORTED_AREA` does not list
  `refracted`, so the mechanism can be configured on that engine and never run.
* **C2 (engine behaviour).** Four items:
  1. refraction accumulates `bias += raw_drive * strength` (drive-proportional,
     the reference rule) instead of a constant. The constant rule stays
     reachable behind an env flag for A/B, following the existing
     `_fixed_target_plasticity_enabled` precedent.
  2. bias accumulation is gated on `plasticity_enabled`. It currently is not,
     so a no-learn test pass still accumulates bias and corrupts the trajectory
     step by step; the reference stops both together under `update=False`.
  3. `refracted` implemented on `numpy_exact`.
  4. per-fiber `p` implemented on `numpy_sparse` (today `add_connectivity` is
     `numpy_exact` only, and the others raise).

  Items 3 and 4 exist because no single engine can currently express an organ
  that needs both a local kp regime and refraction.
* **C3 (program behaviour).** `NemoArcFSM.step_symbol` reads the state assembly
  and returns the nearest stored state; `train_transition` performs the
  reference's single teacher-forced write; the tautological tests are replaced
  with ones a degenerate arm fails. `NemoArcFSM` is fixed in place rather than
  replaced -- `assembly_calculus/fsm.py` is a second, dict-based FSM and merging
  the two is noted as follow-up, not done here.

## Architecture

Faithful port of `reference/nemo_numpy/fsm_network.py`:

* symbols: 11 stimuli of size k. (The reference's symbol area is one
  `symbol -> arc` matrix whose 11 assemblies are disjoint row blocks; 11
  independent stimulus connectomes are structurally the same object.)
* `SEQ_STATE`: area n=500, k=70. The 5 state assemblies are assigned as
  **disjoint neuron-ID blocks**, as the reference does. Emergent assemblies in
  n=500 would overlap ~k/n_state = 0.14 by chance, which is a confound on the
  readout rather than the thing under test.
* `SEQ_ARC`: area n=5000, k=70, refracted, strength 0.1.
* `Brain(p=0.2, beta=0.1)`, 15 presentations of all 33 transitions, k=70.
* Regime check: kp = 14 per fiber, 28 across both conjuncts, against the
  floor 3 ln 5000 = 25.6. The reference's own demo sits just above its floor,
  and only when both fibers are counted.

Training step for transition (q, sigma, q'):

1. inhibit `SEQ_ARC`, `SEQ_STATE`
2. cue `SEQ_STATE` <- assembly(q), fix
3. project {sym_sigma} + `SEQ_STATE` -> `SEQ_ARC`, learning
4. cue `SEQ_STATE` <- assembly(q'), fix
5. project `SEQ_ARC` -> `SEQ_STATE`, learning onto the pinned winners
6. unfix

Step 5 is the reference's `set_input(arc.read()); fire(new_state)`. It relies on
fixed-target plasticity, which is ON by default and documented as the
reference's reciprocal idiom.

Test step: `disable_plasticity`; cue state 0; per symbol project
{sym} + `SEQ_STATE` -> `SEQ_ARC` then `SEQ_ARC` -> `SEQ_STATE` free; label the
final state by nearest overlap among the 5 stored assemblies.

## Amendments, before any data

**Amendment 1 (SUPERSEDED).** Raise `n_state` from the reference's 500 to 5000,
keeping the state assemblies EMERGENT (one stimulus per state, projected once
then replayed). Reasoning: the reference gets disjointness free by assigning
index blocks, and two emergent assemblies in n=500, k=70 would overlap about
k/n = 0.14, failing P-PRE by construction.

**What killed it.** A smoke run (API check; task numbers void at 2
presentations) put P-PRE at **0.224** with n_state=5000 -- sixteen times the
0.014 chance the amendment predicted. Raising `n` does not help because the
cause is not sparsity: it is [[sampler-merges-at-low-load]], the sparse
sampler flattening distinct inputs into overlapping winners while the area is
nearly empty. The amendment addressed the wrong mechanism.

**Amendment 2 (in force).** State assemblies are assigned as DISJOINT NEURON-ID
BLOCKS, exactly as the reference does with
`arange(n_states * cap).reshape(n_states, cap)`, after materializing the state
area so every ID has a compact slot. `n_state` returns to the reference's
**500**. P-PRE then holds by construction rather than by luck, and the
experiment stays on one engine.

This removes a confound rather than measuring around it, and it is the more
faithful port: an emergent state code is a genuinely interesting question, but
it is #91's question, not this one. What A1 tests is whether the arc + state
organ learns transitions given a clean state code -- which is what Theorem 4
assumes.

**Amendment 3 (in force).** `norm_init=False`. The reference's `FSMNetwork`
defaults to raw Bernoulli(p) weights, our `Brain` defaults to normalized ones,
and that changes both the drive scale and the balance between the arc's two
conjuncts -- which is the quantity under test. [[norm-init-substrate-vs-reference]]
already says parity reproductions pin False. It is also the safe choice here:
norm_init exists to stop SELF-recurrence collapsing, and neither area in this
organ has a self fiber.

## Bars

Ten brain seeds, reported as a distribution and never as a bare mean.

* **P-GOLD (primary).** >= 8/10 seeds decide BOTH sequences correctly
  (positive -> accept, negative -> reject). Reference: 3/3.
* **P-CONJ.** Mean across-STATE and across-SYMBOL arc overlap BOTH < 0.15.
  Reference 0.000/0.000; #92 read 0.90-0.99; chance is k/n = 0.014. Both
  directions are required -- one alone cannot distinguish a conjunction from
  collapse onto the other conjunct.
* **P-NULL (negative control, pre-validated on the golden).** With refraction
  off: task <= 2/10 AND across-symbol overlap > 0.5. The reference gives 0/3
  and 0.989. **If the null also passes P-GOLD the entire result is void.**
* **P-DEGEN.** Zero-presentation and beta=0 arms both <= 2/10. These are only
  meaningful once C3 lands; today all three degenerate arms score perfectly.
* **P-PRE (precondition, not a bar).** Report pairwise overlap of the five
  state assemblies. If > 0.05 the readout is confounded and P-GOLD is void.

## Interpretation rules

Stated before the data:

* P-GOLD passes and P-NULL/P-DEGEN fail as required -> the organ works; A2
  proceeds to #92's word-order FSM.
* P-GOLD fails while the reference passes -> the divergence is in OUR
  substrate, not in the architecture. Prime suspect is the sparse sampler
  ([[sampler-is-the-whole-discrepancy]]); the next step is the same experiment
  on `numpy_exact`, which C2 item 3 exists to make possible.
* P-CONJ fails while P-GOLD passes -> either the task is decidable without a
  conjunction, or the readout is size-confounded, which is exactly how #92's
  arc readout went wrong. P-GOLD is then reported but NOT taken as evidence
  that the organ works.
* The refraction rule A/B (proportional vs constant, both on our Brain) is
  reported whatever happens. The reference predicts the constant needs
  retuning whenever the drive scale moves.

## Pre-stated risks

* **#62 recruitment divergence at unequal area sizes.** Our areas are 500 vs
  5000, which is exactly the condition. Run `numpy_sparse` and torch both.
* **`p` is global on `Brain`.** A1 sets p=0.2 brain-wide, which is fine for a
  standalone organ but is not how A2 embeds it in a p=0.05 parser. C2 item 4 is
  what unblocks A2; if it slips, A2 is blocked, not merely slower.
* **`w_max` clamp.** The reference has no weight ceiling; our `Brain` clamps at
  w_max=20. Under refraction a given (state neuron -> arc neuron) synapse is
  potentiated about once per presentation, so 1.1^15 = 4.18 stays clear of the
  clamp -- but the ablation arm potentiates the same winners on all 11
  transitions of a state, reaching 1.1^165, and there our clamp does something
  the reference does not. Report max weight per arm; if the null arm is clamped,
  our null is not the reference's null and must be labelled as such.

Evidence artifact: `research/results/logs/seq_a1_fsm_parity.log`.
