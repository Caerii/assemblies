# The arc collapses onto whichever conjunct fires more

Characterization of the vendored reference FSM (`neural_assemblies/reference/nemo_numpy`)
ahead of building our own transition organ. Script:
`research/experiments/seq_arc_refraction_reference.py`; log alongside it.
No `Brain` is involved except in the audit section.

## The golden exists

`run_mod3_fsm_numpy` decides correctly on every seed tried (42, 7, 13): the
positive digit string is accepted, the negative rejected, read out of the state
assembly by nearest-overlap. Theorem 4's three-area FSM is a real target we can
port against, not a claim on paper.

## #92 measured one of the two collapse directions

An arc area `A_{q,sigma}` has two ways to degenerate, and they are symmetric:

| degeneracy | meaning | detector |
|---|---|---|
| `A_sigma` | arc ignores the state | across-STATE overlap at fixed symbol is high |
| `A_q` | arc ignores the symbol | across-SYMBOL overlap at fixed state is high |

Task #92 measured only the first, read 0.90-0.99, and concluded the conjunctive
arc "has no operating point". A low reading on that one detector is *not*
evidence of conjunction -- it is equally consistent with collapse onto the other
conjunct. Both must be reported together, and this note does.

## Refraction is load-bearing, and it is an anti-swamping force

Ablating refraction from the reference (bias never accumulates; everything else
untouched, manipulation-checked) breaks the task outright:

| refraction | across-state | across-symbol | task |
|---|---|---|---|
| drive-proportional (reference) | 0.000 | 0.000 | 3/3 |
| off | 0.088 | **0.989** | **0/3** |

Chance overlap is k/n = 0.014. Without refraction the reference arc collapses to
`A_q` -- **symbol-blind**, the mirror image of #92's state-blind `A_sigma`.

The direction is set by exposure, not by architecture. In the mod-3 table each
state assembly is trained on 11 transitions per presentation while each symbol is
trained on 3, so the state fiber dominates and the arc collapses onto the state.
In #92, MOOD fired about 3x more often than the state, so the symbol fiber
dominated and the arc collapsed onto the symbol. This is [[role-gain-crowds-not-margins]]'s
law -- gain follows exposure concentration -- appearing in a second organ, and it
means #92's failure is not an argument against conjunctive arcs. It is the
predicted behaviour of a conjunctive arc with no force opposing the dominant
fiber.

Refraction is that force. It penalizes exactly the neurons that keep winning,
and the neurons that keep winning are the ones the dominant fiber selects. It is
first-order, not a nudge: after 15 presentations the accumulated bias averages
113 against a k-WTA boundary drive of 144, and 2501 of 5000 arc neurons carry
bias -- close to the 33 x 70 = 2310 that fully disjoint tiling of the 33
transitions would require.

## Our engines implement a different rule, and it is unnormalized

| | accumulation | scales with drive? |
|---|---|---|
| reference `RefractedArea` | `bias[w] += raw_drive[w] * plasticity` | yes |
| every engine in this repo | `bias[c] += refracted_strength` | no |

Both subtract the bias before k-WTA, so application matches; only accumulation
differs. Substituting the constant rule into the reference and sweeping it:

* an operating point exists but is a knife-edge -- strength 3 fails, 10 gives
  3/3 with 0.000/0.000, 30 fails;
* and it **moves with the drive scale**. At 15 presentations the winner is 10;
  at 30 presentations 10 fails and 30 wins. Drive-proportional passes at both
  without retuning.

Hebbian learning multiplies drive by (1+beta) per presentation, so a constant
increment falls behind by construction. This is [[hebbian-mass-follows-frequency]]
again: normalize the learning rule, not the corpus -- here, normalize the
refraction rule rather than tuning a per-task constant. Note that every existing
call site already passes `refracted_strength=0.1`, and 0.1 is exactly the
reference's `plasticity`, so correcting the rule makes those call sites right
without changing their arguments.

## The FSM programs we ship cannot fail

`NemoArcFSM.step_symbol` computes `to_state = self._table[(from_state, symbol)]`
-- a dictionary lookup -- runs the neural dynamics, discards the result, and
returns the dictionary's answer. Measured consequences:

* the shipped unit test's assertion passes on an **untrained** network;
* `run_mod3_fsm_demo` accepts the positive string and rejects the negative with
  **zero presentations**;
* and again with **beta = 0**.

Three degenerate arms, three perfect scores. See [[fake-perfect-probe-signatures]]:
any score a degenerate arm also achieves is not evidence. Every FSM claim this
repo currently makes rests on this readout.

`train_transition` has a second, independent defect. The reference performs one
teacher-forced write -- `state_area.set_input(arc.read()); state_area.fire(new_state)`
-- strengthening arc->state onto the *target* assembly. Ours calls
`brain.project({}, {arc: [state]})` first, which lets the arc drive the state
area to whatever it prefers and potentiates arc->state onto **those** winners,
then separately drives the target from its stimulus. The transition is never
taught along the pathway that has to carry it at test time.

## Engine gaps found while checking whether the organ is expressible

* per-fiber `p` (`Brain.add_connectivity`) is implemented **only** on
  `numpy_exact`; the others raise.
* `refracted` is implemented on `numpy_sparse`, torch and cuda but **not** on
  `numpy_exact`, where `Engine.set_refracted`'s documented "default is a no-op"
  applies and `_UNSUPPORTED_AREA` does not list it -- so `refracted=True` on that
  engine is silently ignored. That is [[silent-no-op-dead-fibers]] with a live
  fuse.

No single engine can currently express a SEQ organ that needs both a local kp
regime and refraction. Closing both gaps is in scope.
