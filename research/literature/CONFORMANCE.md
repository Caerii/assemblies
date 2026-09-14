# Reference conformance: which of the two clones' constructions exist here

Checked 2026-09-13 against `.reference/mdabagia-nemo` and
`.reference/dmitropolsky-assemblies`, the clones the literature manifest names.
This exists because a provenance claim I made was false, and the same class of
error had already been audited once today in the theory register.

## The eight constructions in `mdabagia-nemo/brain.py`

    reference class      ported here as                              status
    FFArea               core/numpy_engine, core/_pricing            PORTED
    RecurrentArea        assembly_calculus/scaffold.py               PORTED
    RefractedArea        numpy_engine/_sparse, torch_engine/_hashed   PORTED, measured
    RandomChoiceArea     assembly_calculus/coin_config, pfa           PORTED
    ScaffoldNetwork      assembly_calculus/scaffold.py                PORTED
    FSMNetwork           assembly_calculus/fsm.py, HashedArcFSM       PORTED, measured
    PFANetwork           assembly_calculus/pfa.py, transitions.py     PORTED
    AttentionArea        -- not referenced anywhere --                NOT PORTED

## What the gap actually is

`neural_assemblies/assembly_calculus/attention.py` exists, and its own docstring
says what it is: "a readout instrument: it does not mutate a Brain or claim to
learn query-key fibers". The reference's `AttentionArea(RecurrentArea)` is a
LEARNING area with recurrent weights. They are different objects, and the
backlog already says so -- "Finish the learned assembly-attention operator.
Keep the pure snapshot."

The reference's attention carries two things this repository has nowhere:

    def update(self, new_activations):
        self.recurrent_change[ix] = self.plasticity * self.recurrent_weights[ix]
        self.recurrent_weights[ix] = (1 + self.plasticity) * (self.recurrent_weights[ix] > 0)

    def decay_weights(self):
        self.recurrent_weights -= self.recurrent_change
        self.recurrent_change = np.zeros_like(self.recurrent_weights)

1. **A SET rule, not a multiply.** Potentiation assigns `1 + plasticity` to any
   positive weight rather than scaling it, so repeated potentiation SATURATES
   immediately instead of compounding toward `w_max`. Our engine multiplies
   (`w *= 1 + beta`) everywhere. This is a semantic difference that a port must
   decide rather than paper over.
2. **REVERSIBLE plasticity.** `decay_weights` undoes exactly the last update.
   Nothing here can potentiate and then retract.

`FFArea.update` -- the path `FSMNetwork` actually uses, and therefore the path
every sequence result here rests on -- is `w *= 1 + plasticity`, the multiply
rule our engine implements. **There is no parity problem in the sequence line**;
the set rule belongs to `AttentionArea` alone.

## A provenance claim of mine that was FALSE

While porting long range inhibition I wrote, in code and in conversation, that
it is "the mechanism both source papers run on". It is not.

    mdabagia-nemo/brain.py     `inhibit()` is a RESET: clear_input(); activations = []
    dmitropolsky-assemblies    "inhibit" is AREA GATING for parser control flow
    both clones                grep empty for refractory / steps_ago / recently-fired

So LRI-with-a-refractory-period is THIS REPOSITORY'S OWN construction, the same
status as `ordered_recall` and `sequence_memorize` -- which
`PREREG_ordered_recall_reproduction.md` Amendment 1 established this morning for
exactly the same reason. The memory note attributing regimes to
`dabagia2025sequences (lri_hard_test)` cites tests that are not in either clone.

The port is kept, and its justification is restated honestly: RECOVERY is the
one ingredient the substrate lacks. Without a decay term anywhere, idle spacing
is a no-op by construction and the refraction bias is not a relative refractory
period in the biological sense. That is a reason to build it. "The papers use
it" was not.

## What this implies for the order of work

The learned attention operator is the only unported reference construction, it
is an open backlog item, it has no registration and no register entry, and it
carries reversible plasticity that exists nowhere here. That makes it the next
piece of base-proven work -- with a reference to check parity against, which the
LRI port did not have.
