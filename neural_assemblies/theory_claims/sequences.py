"""The sequence organ's foundations: results it rests on, proved or measured elsewhere.

Moved from neural_assemblies/theory.py unchanged; theory.py lists these in its order."""
from __future__ import annotations

from typing import List

from .types import EvidenceRef, Result, Status

CLAIMS: List[Result] = [
    # ---------------------------------------------------------------- sequences
    Result(
        id="SEQ-TIME-IN-WEIGHTS",
        evidence=("demonstrated locally by the autonomous chain "
                  "(PREREG_autonomous_chain.md, autonomous-chain-20260913): a "
                  "chain driven by ONE CONSTANT symbol is recalled exactly to "
                  "128 assemblies on 20 of 20 brains. The symbol carries no "
                  "information, so every advance is carried by the directed "
                  "weights alone; there is no accumulator and no decaying "
                  "trace anywhere in the construction",),
        evidence_refs=(
            EvidenceRef("research/results/runs/sequence.autonomous-chain/autonomous-chain-20260913/results.json", "artifact"),
            EvidenceRef("research/notes/sequence/PREREG_autonomous_chain.md", "registration"),
        ),
        status=Status.PROVED,
        claim="Sequence/temporal structure is carried by DIRECTED inter-assembly "
              "weights, not by an accumulator or a decaying trace.",
        source="Dabagia, Papadimitriou & Vempala, 'Computation with Sequences "
               "in a Model of the Brain' (arXiv:2306.03812), Thm 1.",
        preconditions=("plasticity on the directed fiber",
                       "assemblies stable enough to be re-presented"),
        implemented_by=("neural_assemblies/programs/nemo_fsm.py",),
        caveat="Read the contrapositive too: a recurrent buffer accumulating "
               "context is NOT how this model represents history, which is the "
               "architectural account of the CONTEXT-area collapse in #14.",
    ),
    Result(
        id="SEQ-REGIME",
        status=Status.PROVED,
        claim="The sequence theorems ASSUME that a target neuron receives "
              "kp >= 3 ln n synapses FROM THE DRIVING ASSEMBLY: with it, the "
              "expected drive separates the intended winners from the rest by a "
              "margin the concentration bounds can use. It is a SUFFICIENT "
              "condition inside the proofs, one hypothesis among several: the "
              "theorems also bound the sequence length and the overlap between "
              "stored assemblies, take beta inside a window ([[SEQ-BETA-WINDOW]]) "
              "and assume a normalization schedule on the weights. That crossing "
              "the floor FAILS in practice is this repository's measurement "
              "([[SEQ-REGIME-CLIFF]]), not a theorem: necessity is measured, "
              "sufficiency is proved.",
        source="Dabagia et al. (arXiv:2306.03812); assumed by every theorem in "
               "the paper, and satisfied by its own FSM demo at n=5000, k=70, "
               "p=0.4 (kp=28 per conjunct pair vs floor 25.6). Wording corrected "
               "2026-09-09 after an external review noted the earlier 'reliable "
               "only when' promoted a sufficient condition to a necessary one.",
        preconditions=("counted PER AREA, over the sources that co-fire",
                       "k is the SOURCE assembly's size, not the target's"),
        evidence=("research/experiments/seq_a1_exactness_sweep.py",),
        implemented_by=("neural_assemblies.diagnostics.regime_audit",),
        caveat="An organ is in-regime only when EVERY area in it is. Several "
               "of this repo's null results were recorded an order of magnitude "
               "below floor, where failure is predicted regardless of the "
               "mechanism under test -- those nulls are not evidence.",
    ),
    Result(
        id="SEQ-BETA-WINDOW",
        provenance_gap="Nothing in this repository measures the window. The "
                       "entry is the paper's analysis; the non-monotone recall "
                       "it explains is an observation in a note rather than a "
                       "registered measurement, and no bar has been set on "
                       "where either edge of the window lies.",
        status=Status.PROVED,
        claim="Sequence learning needs beta in a WINDOW: large enough to write a "
              "transition in finite presentations, small enough that the "
              "assemblies formed on presentation 1 do not move.",
        source="Dabagia et al. (arXiv:2306.03812), sequence-memorization "
               "analysis.",
        preconditions=("per-fiber beta, so the window can differ across fibers",),
        caveat="Explains the non-monotone recall in #56 (better at 3 repetitions "
               "than at 8) as a violation from below rather than as noise.",
    ),
    Result(
        id="SEQ-FSM",
        status=Status.PROVED,
        claim="A finite-state machine is simulable by three areas: input, state, "
              "and a CONJUNCTION arc that fires for (state, symbol) and projects "
              "to the next state.",
        source="Dabagia et al. (arXiv:2306.03812), Thm 4; demo at n=5000, k=70, "
               "p=0.4, beta=0.1, 15 presentations.",
        preconditions=("[[SEQ-REGIME]] in every area",
                       "each transition presented comparably often",
                       "teacher-forced write onto the TARGET state assembly"),
        evidence=("research/experiments/seq_a1_fsm_parity.py (10/10 seeds)",
                  "neural_assemblies/reference/nemo_numpy/fsm_network.py"),
        implemented_by=("neural_assemblies.programs.nemo_fsm.NemoArcFSM",),
    ),
    Result(
        id="SEQ-TRANSDUCER",
        status=Status.PROVED,
        claim="Prediction/output is an FSM with one more area, fired together "
              "with the state update during training -- a transducer.",
        source="Dabagia et al. (arXiv:2306.03812), Remark 5.",
        preconditions=("[[SEQ-FSM]]",),
        evidence=("neural_assemblies/programs/sequence_transducer.py: "
                  "LEX + SEQ_STATE -> SEQ_ARC -> SEQ_STATE with SEQ_ARC -> OUT, "
                  "the FSM plus one output area fired with the state update, "
                  "state INDUCED rather than assigned; tested by "
                  "test_sequence_transducer.py and ported to the hashed "
                  "substrate with its own parity test",
                  "measured through [[SEQ-TEMPORAL-CARRY]], whose adopted "
                  "numbers come from this construction"),
        implemented_by=("neural_assemblies/programs/sequence_transducer.py",
                        "neural_assemblies/core/torch_engine/_hashed_transducer.py"),
        caveat="Caveat corrected 2026-09-13 by a provenance pass: it read "
               "'NOT YET BUILT HERE', which was stale. The transducer IS built, "
               "tested and ported, and [[SEQ-TEMPORAL-CARRY]] is measured on "
               "it. What remains true is that our STATELESS next-token model, "
               "which scores exactly the bigram optimum, is the degenerate "
               "one-state case.",
    ),
    Result(
        id="SEQ-TM",
        status=Status.PROVED,
        claim="A Turing machine is simulable by an FSM plus three-area tape "
              "cycles, about ten areas in total.",
        source="Dabagia et al. (arXiv:2306.03812), Thm 7.",
        preconditions=("[[SEQ-FSM]]", "unbounded tape areas"),
        evidence=("neural_assemblies/programs/tm_demo.py: a MINIMAL unary "
                  "increment machine over FSMNetwork plus FiberCircuit tape "
                  "areas, exercised by test_literature_parity.py -- the shape "
                  "of the construction at its smallest nontrivial size, not "
                  "the general theorem",),
        implemented_by=("neural_assemblies/programs/tm_demo.py",),
        caveat="Caveat corrected 2026-09-13 by a provenance pass: it read "
               "'NOT BUILT HERE'. A minimal unary-increment demo IS built and "
               "tested; the general roughly-ten-area construction is not. "
               "[[SEQ-EXACT-RECOVERY]] gives unbounded TIME with "
               "fixed memory, which is the control half only -- it does not by "
               "itself confer more than finite-automaton power.",
    ),
]
