"""The indexed register of results this codebase stands on.

WHY THIS EXISTS
---------------
Most of what this system does is an instance of something proved in the
literature, a measurement we made ourselves, or an EXTRAPOLATION beyond both.
Those three are not interchangeable, and until they are written down they are
indistinguishable in a docstring: "the theory requires kp >= 3 ln n" reads the
same whether it is a theorem, our own sweep, or a guess.

So every claim a docstring leans on gets an ID here, with its PRECONDITIONS and
its STATUS. Code cites it by ID in double brackets. Three consequences:

* an extension is obvious at the point of use, because its status says so;
* preconditions travel with the claim, so "we are inside the theorem" is
  checkable rather than assumed -- this is what `diagnostics.regime_audit`
  automates for the one precondition we kept violating;
* citations cannot rot: `unresolved_citations` scans the package and
  `neural_assemblies/tests/test_theory_citations.py` fails on a dangling one.

CITATION SYNTAX. Result IDs are UPPERCASE-DASHED, written in double
brackets at the point of use. The
lowercase-kebab ``[[silent-no-op-dead-fibers]]`` links already used throughout
the codebase point at operator memory, not at results, and the checker
deliberately ignores them.

STATUS IS NOT QUALITY. A `MEASURED` result can be better evidence for our
purposes than a `PROVED` one whose preconditions we cannot meet. The point of
the label is to say what would have to be true for the claim to transfer, not
to rank the claims.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

#: Matches an UPPERCASE-DASHED result citation in double brackets. Lowercase
#: kebab links are operator memory, not results, and never match.
CITATION = re.compile(r"\[\[([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*)\]\]")
EVIDENCE_ROLES = frozenset({
    "artifact", "registration", "producer", "analysis", "comparison", "log",
})


class Status:
    """How much weight a claim can carry, and what would invalidate it."""

    #: Stated and proved in the cited source. Transfers only inside its
    #: preconditions -- check them before relying on it.
    PROVED = "PROVED"

    #: Established empirically in this repository, with the experiment named.
    #: Transfers to the regime it was measured in; re-measure outside it.
    MEASURED = "MEASURED"

    #: We rely on this BEYOND where it is proved or measured. Every use is a
    #: standing risk and should say what would falsify it.
    EXTENSION = "EXTENSION"


@dataclass(frozen=True)
class EvidenceRef:
    """One resolvable repository file and its role in supporting a result."""

    path: str
    role: str
    limitation: str = ""


@dataclass(frozen=True)
class SensitivityCheck:
    """A retained treatment/control comparison that must still move.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-result-sensitivity

    Paths use RFC 6901 tokens without the optional leading slash and ``*`` to
    expand a JSON list. The resulting vectors are compared pairwise, so a
    missing seed, reordered arm, or dead probe is a register-validation failure
    rather than prose debt.
    """

    artifact: str
    sample_path: str
    treatment_path: str
    control_path: str
    relation: str
    minimum_effect: float
    mechanism: str


@dataclass(frozen=True)
class Result:
    """One citable claim, with the conditions under which it holds."""

    id: str
    status: str
    claim: str
    source: str
    preconditions: Sequence[str] = field(default_factory=tuple)
    evidence: Sequence[str] = field(default_factory=tuple)
    evidence_refs: Sequence[EvidenceRef] = field(default_factory=tuple)
    provenance_gap: str = ""
    sensitivity_checks: Sequence[SensitivityCheck] = field(default_factory=tuple)
    sensitivity_gap: str = ""
    implemented_by: Sequence[str] = field(default_factory=tuple)
    caveat: str = ""
    engine: str = ""  # measurement substrate; empty for non-empirical entries

    def __str__(self) -> str:
        head = f"[{self.status}] {self.id}: {self.claim}"
        bits = [f"    source: {self.source}"]
        if self.engine:
            bits.append(f"    engine: {self.engine}")
        if self.preconditions:
            bits.append("    requires: " + "; ".join(self.preconditions))
        if self.evidence:
            bits.append("    evidence: " + "; ".join(self.evidence))
        if self.evidence_refs:
            bits.append("    evidence files: " + "; ".join(ref.path for ref in self.evidence_refs))
        if self.provenance_gap:
            bits.append("    PROVENANCE GAP: " + self.provenance_gap)
        if self.sensitivity_checks:
            bits.append("    sensitivity: " + "; ".join(
                f"{check.mechanism} ({check.artifact})"
                for check in self.sensitivity_checks))
        if self.sensitivity_gap:
            bits.append("    SENSITIVITY GAP: " + self.sensitivity_gap)
        if self.implemented_by:
            bits.append("    used by: " + "; ".join(self.implemented_by))
        if self.caveat:
            bits.append(f"    CAVEAT: {self.caveat}")
        return head + "\n" + "\n".join(bits)


_RESULTS: List[Result] = [
    # ---------------------------------------------------------------- sequences
    Result(
        id="SEQ-TIME-IN-WEIGHTS",
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
        caveat="NOT YET BUILT HERE. Our stateless next-token model scoring "
               "exactly the bigram optimum is the degenerate one-state case.",
    ),
    Result(
        id="SEQ-TM",
        status=Status.PROVED,
        claim="A Turing machine is simulable by an FSM plus three-area tape "
              "cycles, about ten areas in total.",
        source="Dabagia et al. (arXiv:2306.03812), Thm 7.",
        preconditions=("[[SEQ-FSM]]", "unbounded tape areas"),
        caveat="NOT BUILT HERE. [[SEQ-EXACT-RECOVERY]] gives unbounded TIME with "
               "fixed memory, which is the control half only -- it does not by "
               "itself confer more than finite-automaton power.",
    ),

    # ------------------------------------------------- measured in this repo
    Result(
        id="SEQ-REGIME-CLIFF",
        engine="numpy_sparse; sampled arc in the original sweep; materialized reruns require their own artifact provenance",
        status=Status.MEASURED,
        sensitivity_gap="No retained immutable materialized/hashed sweep pairs "
                        "the regime crossing with a mechanism-disabled null.",
        claim="Crossing the kp >= 3 ln n floor is a CLIFF, not a slope: below it "
              "recovery is almost never exact and the machine fails; above it "
              "every seed runs correctly.",
        source="This repository.",
        evidence=("research/experiments/seq_a1_exactness_sweep.py: state kp "
                  "14 -> 4/100 exact steps and 4/10 trajectories; kp 21 -> "
                  "80/100 and 10/10; kp 28 -> 100/100 and 10/10, with the "
                  "transition at the predicted p = 18.6/70 = 0.266",),
        evidence_refs=(
            EvidenceRef("research/results/sequence/seq_a1_exactness_sweep_results.json", "artifact",
                        "sampled numpy arc; sequence verdict void under sampler audit"),
            EvidenceRef("research/results/sequence/seq_a1_exactness_sweep_results_materialized.json", "artifact"),
            EvidenceRef("research/experiments/seq_a1_exactness_sweep.py", "producer"),
        ),
        provenance_gap="legacy result files have no immutable runner/source record",
        implemented_by=("neural_assemblies.diagnostics.regime_audit",),
        caveat="The original sweep used the sampled numpy arc; its sequence-dynamics "
               "numbers are void under PREREG_sampler_audit.md until reproduced "
               "on a materialized or hashed substrate. Mean overlap read 0.812 at the failing point while exactness "
               "was 4/100 -- the mean hides this mechanism entirely.",
    ),
    Result(
        id="SEQ-EXACT-RECOVERY",
        engine="mixed: vendored nemo_numpy reference, numpy_sparse sampled/materialized, hashed ArcFSM and soft-census organs; see per-evidence caveats",
        status=Status.MEASURED,
        sensitivity_gap="This composite entry spans several legacy protocols; "
                        "no one retained runner artifact encodes its null.",
        claim="The state area is a DISCRETE attractor: k-WTA maps a whole "
              "neighbourhood onto exactly one stored assembly in one step. "
              "Recovery must be EXACT -- 69 of 70 neurons is a failure, not a "
              "near-miss -- because the arc amplifies any residual ~8x per step.",
        source="This repository.",
        evidence=("research/experiments/seq_a1_arc_transfer.py: "
                  "d(output loss)/d(input loss) = 8.17 ours, 10.33 reference",
                  "research/experiments/seq_a1_horizon.py: 2000 steps, 5/5 "
                  "seeds, zero errors, exact recovery every step (p = 0.4); at "
                  "p = 0.3 the one short horizon (seed 1, step 759) was the "
                  "SAMPLER's: materialized, that seed runs 2000 steps exact",
                  "research/experiments/seq_a1_horizon_hashed.py (GATE-3 of "
                  "DESIGN_sequence_port.md): the hashed organ, 20 brains, 2000 "
                  "digits, p = 0.3 and 0.4: 40/40 brains never err; exact "
                  "recovery 0.92-1.00 at p = 0.3, 1.000 at 0.4; the numpy "
                  "machine materialized agrees 5/5",
                  "research/experiments/seq_a1_limit_cycle.py: constant input "
                  "gives an orbit closing bit-identically, 30/30",
                  "research/experiments/seq_s5_soft_census.py: at 60-120 "
                  "states, ~0.25-0.6% of transitions are SOFT (correct label, "
                  "one intruder neuron); first lattice exit equals first "
                  "soft-pair visit, 40/40 seeds, zero parameters (SAMPLED arc)",
                  "seq_s5_soft_census_hashed.py (explicit substrate, 40 "
                  "organs): soft pairs persist at 0.067% (4/6000, one intruder "
                  "each, non-abelian/product groups only), the zero-parameter "
                  "law 4/4 -- and none of the four deviations derailed in 500 "
                  "steps, where the sampled arc derailed 12/40 words"),
        evidence_refs=(
            EvidenceRef("research/results/sequence/seq_a1_horizon_results_hashed_int8_timing.json", "artifact"),
            EvidenceRef("research/results/sequence/seq_a1_horizon_materialized_check.json", "artifact"),
            EvidenceRef("research/results/sequence/seq_s5_soft_census_results.json", "artifact",
                        "sampled numpy arc"),
            EvidenceRef("research/results/sequence/seq_s5_soft_census_results_hashed.json", "artifact"),
            EvidenceRef("research/notes/sequence/PREREG_s5_cliff_anatomy.md", "registration"),
        ),
        provenance_gap="some legacy subclaims have no source archive or raw runner artifact",
        caveat="(1) Expansion and quantization are a PAIR: amplification alone "
               "is chaos, benign only because a quantizing area follows it; "
               "composition steps without a re-quantizing stage drift. "
               "(2) THE MAP HAS SOFT SPOTS: a rare transition emits 69/70 of "
               "the right block (one intruder), and exactness holds until the "
               "word first visits one -- the measured 'horizon' is a "
               "first-hitting time, not a decay constant (zero-parameter law, "
               "40/40 numpy seeds and 36/36 hashed organs). "
               "(3) THE DERAILMENTS WERE THE SAMPLER'S: on the sampled arc "
               "post-visit trajectories derailed into an absorbing off-lattice "
               "regime (12/18) and 12/40 words went wrong within 500 steps; on "
               "the explicit substrate the soft rate is 7.5x lower and no "
               "deviation derailed (Addendum 3 of PREREG_s5_cliff_anatomy.md). "
               "Apparent group differences at fixed L were the pair count "
               "|G| x |gens| at a flat per-pair rate, not solvability. "
               "(4) WHAT A SOFT SPOT IS (Addenda 4-5, 800 organs): the "
               "collision of the target block's weakest member with the area's "
               "best-connected outsider. The outsider is a Binomial(k, p) upper "
               "tail (43-50 of 70 rows at p = 0.4), unrelated to the Cayley "
               "graph (intruder 'other' 31/33), identical in rate across three "
               "groups of order 60 and rising with n_state; the block side is "
               "how much of the test-time arc was potentiated onto the block. "
               "(5) PRESENTATIONS HAVE A WINDOW: 8 unformed (95% soft), 15 in "
               "the window (0.06%, the two tails just touching), 30 RELOCATED "
               "(soft rate doubled, 12% of words derail). Relocation is the "
               "CLIP: at strength = beta refraction cancels potentiation "
               "exactly, a member's net drive is pinned at its base value, and "
               "the cancellation ends when its weight clips -- the "
               "best-connected members fall first (99% of lost arc neurons "
               "clipped vs 12% of kept); c* = ln(w_max max(1, k p) / base) / "
               "ln(1 + beta) ~ 28-30 here. "
               "(6) STRENGTH IS PINNED AT BETA (Addendum 6, 300 organs): below "
               "it the cross-context bias no longer cancels the state-shared "
               "neurons' potentiation and the arc collapses onto the state "
               "conjunct (0.05: across-symbol overlap 0.63; 0.08: soft rate "
               "18x); above it a member's net decays and relocates. The "
               "transducer at half beta is worse on every count (A3 Amendment "
               "2), so its null is structural. "
               "(7) THE FIX IS THE GAIN (Addendum 7): trained just below the "
               "clip edge (20 and 24 presentations, ~0.7-0.85 c*) the organ is "
               "EXACT -- 0 soft pairs in 24,000 and 0 derailments on 100 S5 "
               "organs at each. In-degree normalisation cut the 15-presentation "
               "rate 3.7x by reweighting the conjuncts (STATE -> ARC in-degree "
               "3,360 vs the stimulus's 16,000), not through the tail.",
    ),
    Result(
        id="SEQ-TEMPORAL-CARRY",
        engine="hashed transducer / temporal organ (20 brains per cell)",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/sequence.temporal-positions/temporal-positions-study-20260910/results.json",
            sample_path="run/seeds",
            treatment_path="observations/summaries/g1/D/values",
            control_path="observations/summaries/blind_g1/D/values",
            relation="all-greater", minimum_effect=0.05,
            mechanism="state-dependent distractor carry versus state-blind g=1",
        ),),
        claim="A transducer whose STATE is its previous arc (state_mode='copy') "
              "and whose PREDICTED arc neurons win (the lateral ARC -> ARC "
              "fiber's top-k above half its maximum get (1 + g) x drive, g = 1) "
              "improves next-token prediction on the synthetic agreement chain. On "
              "the agreement chain corpus it beats a bigram by 0.149 MRR with "
              "two distractors between the agreeing words (72% of the oracle "
              "gap; twenty fresh seeds replicate +0.148 to the third digit) and "
              "by 0.106 with three (60%); the copy-state native readout gains nothing "
              "(-0.002), and reading with the state area empty loses the whole "
              "gain (+0.146 full minus blind). POSITION-SPECIFIC MECHANISM: on "
              "twenty further fresh seeds, mean subject-number contrast over the "
              "six distractor positions is 0.0258 at g = 0, 0.1849 at g = 1, "
              "and 0.0022 with state blinded. The paired g = 1 minus g = 0 "
              "amplification is 0.1591 [0.1400, 0.1781]. At g = 0 the contrast "
              "is present after the first distractor (0.0479) but absent after "
              "the second (0.0037); at g = 1 it remains 0.2032 then 0.1666. "
              "Thus predicted-win amplifies and preserves the state-dependent "
              "representation through the second distractor. ORDER: two twelve-word "
              "sequences sharing ten words are continued correctly on 20 of 20 "
              "brains by the induced state and by the copy state at 20 "
              "presentations (19 of 20 with predicted-win at g = 4), and "
              "forgotten past the clip edge c* = ln(20) / ln(1.1) = 31.4 "
              "presentations, where the sequence stimuli relocate the arcs.",
        source="This repository; PREREG_temporal_memory.md (cells A and B, "
               "Amendments 1-2, bars TM-1 to TM-9); "
               "PREREG_temporal_positions.md (bars TP-1 to TP-4); the design is the "
               "literature's temporal memory (predictive cells win) rebuilt on "
               "the refracted-arc transducer.",
        preconditions=("hashed substrate, twenty brains per cell, n = n_arc = "
                       "10,000, k = 200, organ_p = 0.2, beta = 0.1, arc "
                       "strength at beta",
                       "presentations inside the clip window for the order "
                       "result (20 of c* = 31.4)",
                       "a corpus whose oracle gap is large enough to measure: "
                       "0.21 at gap 2, 0.18 at gap 3 (PREREG_agreement_corpus.md)"),
        evidence=("seq_a3_transducer.py --temporal, gap 2, seeds 42-61: g = 1 "
                  "+0.148 +/- 0.009, g = 4 +0.140, g = 0 -0.001 (results_temporal_chain_gap2.json)",
                  "fresh seeds 62-81 (_amend2_fresh): +0.149 +/- 0.009, lower "
                  "bound 0.141 against the 0.085 bar; same-number minus "
                  "different-number arc overlap 0.220 +/- 0.020 at g = 1 (bar "
                  "0.10), 0.111 +/- 0.003 at g = 0; pooled-position values VOID for TM-9 "
                  "(see research/notes/sequence/AUDIT_temporal_position_pooling.md)",
                  "sequence.temporal-positions/temporal-positions-study-20260910, "
                  "seeds 82-101: distractor contrast g = 0 0.0258 [0.0226, "
                  "0.0290], g = 1 0.1849 [0.1660, 0.2038], blind 0.0022 "
                  "[-0.0003, 0.0046]; paired amplification 0.1591 [0.1400, "
                  "0.1781]; all TP-1 to TP-4 bars pass",
                  "gap 3, seeds 42-61 (_amend2_gap3): +0.106 +/- 0.011, lower "
                  "bound 0.095 against the 0.070 bar",
                  "seq_tm_high_order.py --presentations 20: sets I-III exact on "
                  "the induced and copy arms; at 40 presentations set III falls "
                  "to 1 of 20 on every arm (results in research/results/sequence/)"),
        evidence_refs=(
            EvidenceRef("research/results/runs/sequence.temporal-positions/temporal-positions-study-20260910/results.json", "artifact"),
            EvidenceRef("research/results/sequence/seq_a3_transducer_results_temporal_chain_gap2.json", "artifact"),
            EvidenceRef("research/results/sequence/seq_a3_transducer_results_temporal_chain_gap2_amend2_fresh.json", "artifact"),
            EvidenceRef("research/results/sequence/seq_a3_transducer_results_temporal_chain_gap3_amend2_gap3.json", "artifact"),
            EvidenceRef("research/results/sequence/seq_tm_high_order_results_p20.json", "artifact"),
            EvidenceRef("research/notes/sequence/PREREG_temporal_positions.md", "registration"),
        ),
        provenance_gap="pre-schema-4 prediction/order artifacts lack complete source and environment capture",
        implemented_by=("neural_assemblies/core/torch_engine/_hashed_transducer.py",),
        caveat="A prediction result on a synthetic corpus, not a language model: "
               "no baseline beyond bigram and oracle, no scaling curve in n, "
               "and the carry's decay with the gap has two points (72%, 60%). "
               "The historical TM-9 pooled-position values remain void; the "
               "separately preregistered position-specific run replaces their "
               "mechanism interpretation. Its g=0 curve reaches only the first "
               "distractor, while g=1 remains represented after the second; no "
               "alternative readout was tested. The "
               "predicted-win rule is a gain on a lateral fiber applied at "
               "selection, not a plasticity rule. The successor-state "
               "construction (state teacher-forced toward the next h words) "
               "carries nothing on the same corpus (PREREG_successor_state.md).",
    ),
    Result(
        id="ARC-CONJUNCT-EXPOSURE",
        engine="vendored reference/nemo_numpy (explicit NumPy matrices)",
        status=Status.MEASURED,
        sensitivity_gap="The ablation is described in a legacy log but has no "
                        "structured immutable null artifact.",
        claim="A conjunction area collapses onto whichever conjunct is exposed "
              "more often, unless an opposing force (refraction) is present.",
        source="This repository; the same law as the role-binding gain result.",
        evidence=("research/experiments/seq_arc_refraction_reference.py: "
                  "ablating refraction takes across-symbol overlap 0.000 -> "
                  "0.989 and the task 3/3 -> 0/3",),
        evidence_refs=(
            EvidenceRef("research/results/logs/seq_arc_refraction_reference.log", "log"),
            EvidenceRef("research/experiments/seq_arc_refraction_reference.py", "producer"),
        ),
        provenance_gap="legacy log has no structured run, seeds, or source archive",
        preconditions=("BOTH overlap directions measured -- one alone cannot "
                       "distinguish a conjunction from collapse onto the other "
                       "conjunct",),
        caveat="Task #92 measured one direction, read 0.90-0.99, and concluded "
               "a conjunctive arc has no operating point. It has one.",
    ),
    Result(
        id="REFRACTION-PROPORTIONAL",
        engine="vendored reference/nemo_numpy (explicit NumPy matrices)",
        status=Status.MEASURED,
        sensitivity_gap="Constant and proportional charging were retained only "
                        "in a legacy log, not a machine-checkable null record.",
        claim="Refraction must charge in proportion to the winner's raw drive. "
              "A constant increment is not an equivalent parameterization: "
              "Hebbian growth multiplies drive while a constant grows linearly, "
              "so its operating point MOVES with training duration.",
        source="This repository.",
        evidence=("research/experiments/seq_arc_refraction_reference.py: the "
                  "constant winning at 15 presentations fails at 30, while the "
                  "proportional rule passes both untouched",),
        evidence_refs=(
            EvidenceRef("research/results/logs/seq_arc_refraction_reference.log", "log"),
            EvidenceRef("research/experiments/seq_arc_refraction_reference.py", "producer"),
        ),
        provenance_gap="legacy log has no structured run, seeds, or source archive",
        implemented_by=("neural_assemblies/core/_homeostasis.py",),
    ),
    Result(
        id="REFRACTION-NEEDS-LOAD",
        engine="numpy_sparse, sampled versus explicitly materialized arc; sampled load floor is retracted",
        status=Status.MEASURED,
        sensitivity_gap="Legacy load sweeps lack an immutable paired "
                        "mechanism-disabled artifact.",
        claim="A refracted conjunction area has a CEILING in load M*k/n: "
              "above ~1.3 its conjunctions do not fit (10/10 correct at load "
              "1.26, 0/10 at 1.80). RE-SCOPED 2026-09-09 (PREREG_sampler_audit.md): "
              "the lower edge this entry was named for -- 'below ~0.2 its "
              "assemblies never converge' -- was the numpy sampler's; with the "
              "arc materialized a 3-conjunction arc is 10/10 correct at every "
              "load from 0.04 to 0.60. An arc must be sized so its conjunctions "
              "fit; it need not be filled.",
        source="This repository.",
        evidence=("research/experiments/seq_a2_refraction_load.py, SAMPLED arc: "
                  "a 3-conjunction arc goes 1/10 at load 0.04 to 10/10 at 0.21; "
                  "a 9-conjunction arc holds 10/10 from 0.32 to 1.26 and "
                  "collapses to 1/10 at 1.80",
                  "the same script with the arc MATERIALIZED (NEMO_MATERIALIZE=1, "
                  "10 seeds): 3-conjunction arc 10/10 at every load 0.04-0.60; "
                  "9-conjunction arc 10/10 to 1.26, 0/10 at 1.80",
                  "arc assembly stability across training: 0.286 from "
                  "presentation 5 to 15 under-loaded, 0.957 from 10 to 15 loaded"),
        evidence_refs=(
            EvidenceRef("research/results/sequence/seq_a2_refraction_load_results.json", "artifact",
                        "sampled arc; lower-edge inference retracted"),
            EvidenceRef("research/results/sequence/seq_a2_refraction_load_results_materialized.json", "artifact"),
            EvidenceRef("research/experiments/seq_a2_refraction_load.py", "producer"),
        ),
        provenance_gap="legacy artifacts lack runner records and source archives",
        preconditions=("refraction active -- this is a statement about what "
                       "refraction needs, not about k-WTA generally",),
        caveat="The 'silent failure under load' this entry once described -- "
               "assemblies that never stop moving while every diagnostic reads "
               "healthy -- was measured on the sampled engine and does not occur "
               "materialized; treat it as a property of lazily drawn areas, not "
               "of refraction. The ceiling is [[AC-CAP]]'s and is not independent "
               "of it. One task, 10 seeds.",
    ),
    Result(
        id="REFRACTION-ANTI-MERGING",
        engine="hashed AssemblyMemory; materialized numpy_sparse mirror with summed stimulus parts (not an identical stimulus protocol)",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.capacity-scaling/refraction-paired-sensitivity-20260911/results.json",
            sample_path="run/seeds",
            treatment_path="observations/conditions/refracted/cells/B~14000~160/checkpoints/128/rank1",
            control_path="observations/conditions/control/cells/B~14000~160/checkpoints/128/rank1",
            relation="all-greater", minimum_effect=0.5,
            mechanism="refracted memory versus paired Hebbian control at M=128",
        ),),
        sensitivity_gap="The paired A7 check covers the central R1 capacity "
                        "contrast; the masked-vs-net veto, convergence gate, "
                        "strength plateau, and cross-engine mirror still lack "
                        "retained paired sensitivity checks.",
        claim="A recurrent k-WTA area refracted at HALF beta and read with the "
              "refraction bias MASKED holds ~25x the Hebbian ceiling: at n/k = 67 "
              "M* ~ 1600-2200 stored assemblies against 64-89 for the control, "
              "x34-38 at n/k = 33, >= x13-16 at n/k = 133 (censored). Both "
              "ceilings are functions of n/k ALONE (n/k-matched cells agree "
              "within 25%), so refraction multiplies the assembly capacity law "
              "[[AC-CAP]] rather than changing its form. The mechanism is "
              "ANTI-MERGING, not orthogonalization: the stored assemblies are "
              "orthogonal (0.00x chance) only while the area has unvisited "
              "neurons; past fill 1.0 they overlap at chance like random subsets, "
              "yet remain DISTINCT (1.000) and recoverable from a half cue, where "
              "the Hebbian control collapses into hubs (distinct 0.63, overlap "
              "15x chance) long before the area is full. The intrinsic bias "
              "vetoes recall (the net readout reads chance): a refracted memory "
              "is read through the veto or not at all. Fewer rounds per item "
              "raise the ceiling (T = 8 > T = 16) while the items still "
              "converge; refraction's benefit requires GRADED stimulus drive. "
              "SHAPE OF THE LAW (Amendment 4): in regime (k p >= 3 ln n) the "
              "refracted ceiling is ~0.40 (n/k)^2 at n/k = 67 and 133 (6995 at "
              "(8000, 60)), doubling exponents 2.1 then 1.8 -- Willshaw-like, "
              "not adopted as a power law; the control is ~0.017 (n/k)^2 from "
              "n/k = 67 on, so the multiplier is ~23-25x there. Out of regime "
              "(k p = 15 < 3 ln n) the cells fall 20-32% below their n/k pairs "
              "and do not converge at low load. GATED ROUNDS (Amendment 5): "
              "ending an item's rounds at its first repeated winner set under "
              "T_max = 8 raises the ceiling by a CONSTANT fraction of n/k: +35% at "
              "n/k = 67 (2645 vs 1961) and +24% at 133 (8666 vs 6995), both "
              "resolved; at both cells the ceiling sits where items stop "
              "converging inside T_max (the fraction converging is a U in load, "
              "0.98 at mid load, 0 at the ceiling); the same gate STARVES the "
              "Hebbian control, whose winners settle in ~4 rounds before its "
              "memory is written; T_max = 16 under the gate costs the "
              "out-of-regime cell 13x. STRENGTH IS A SWITCH, NOT A DIAL "
              "(Amendment 6): 0.3, 0.4, 0.5 and 0.6 beta give the same ceiling "
              "(1919-1993, one bracket) at n/k = 67 -- the ceiling is the "
              "synaptic memory's, refraction only has to prevent merging while "
              "it is written; the churn transition is in (0.6, 0.7] beta at "
              "T = 8.",
        source="This repository; PREREG_refraction_memory.md (bars R1-R7, N1-N3, "
               "Q1-Q4, G1-G6, S8-S9).",
        evidence=("seq_capacity_scaling.py, arm B on the organ fiber, 20 brains, "
                  "M grid to 4096, s = 0.5 beta masked vs control: n/k = 33: "
                  "431 / 383 vs 11.3; n/k = 67: 1961 / 1589 / 2230 vs 83 / 64 / "
                  "89; n/k = 133: >= 4096 vs 307 / 263",
                  "distinct 1.000 at every refracted ceiling; control 0.63 at "
                  "M = 128 (n = 4000), pairwise 15x chance",
                  "net readout at n = 4000: M* = 8 (chance at every M)",
                  "T = 16: M* = 203 against T = 8's >= 1024 (n = 4000); T = 4 "
                  "does not converge at low load and does at high load",
                  "numpy_sparse gate (refraction_memory_numpy.py, materialized, "
                  "5 brains, summed Binomial stimuli): refracted >= 512 "
                  "(censored, rank-1 1.000 throughout, distinct 1.000) vs "
                  "control 34 -- the claim holds on the engine whose k-WTA has "
                  "no selector defect",
                  "grid to 16384, 20 brains: (8000, 60) REF 6995 [6144, 8192) = "
                  "0.395 (n/k)^2, CTL 307; (4000, 30) REF 4749 [4096, 6144), "
                  "CTL 263 (out of regime)",
                  "(4000, 60) gated T_max 8: REF 2645 [2560, 2816) vs 1961; CTL "
                  "gated rank-1 0.32 at M = 8 (no memory formed); (4000, 30) "
                  "gated T_max 16: 362 vs 4749",
                  "(8000, 60) gated T_max 8: 8666 [8192, 10240) vs 6995 (x1.24)",
                  "strength 0.3 / 0.4 / 0.5 / 0.6 beta at (4000, 60): 1986 / "
                  "1919 / 1961 / 1993, all [1536, 2048)"),
        evidence_refs=(
            EvidenceRef("research/notes/memory/PREREG_refraction_memory.md", "registration"),
            EvidenceRef("research/results/runs/memory.capacity-scaling/capacity-record-consumed-20260910/results.json", "artifact",
                        "registered protocol-consumption replay, not the whole historical grid"),
            EvidenceRef("research/results/memory/refraction_memory_numpy_results.json", "artifact"),
            EvidenceRef("research/results/memory/capacity_scaling_results_figure_ref.json", "artifact"),
            EvidenceRef("research/results/memory/capacity_scaling_results_figure_ctl.json", "artifact"),
            EvidenceRef("research/results/runs/memory.capacity-scaling/refraction-paired-sensitivity-20260911/results.json", "artifact",
                        "paired version-3 reproduction and sensitivity run"),
            EvidenceRef("research/results/comparisons/refraction-paired-sensitivity-20260911.json", "comparison",
                        "2090-scalar migration comparison receipt"),
        ),
        provenance_gap="most capacity-grid artifacts predate immutable source/environment records",
        preconditions=("recurrent k-WTA area, weight clip, norm_init, no column "
                       "scaling (arm B); refraction strength 0.5 beta; T = 8 "
                       "rounds per item from an inhibited area; readout = "
                       "half-cue recall with the bias masked",
                       "GRADED stimulus drive: a zero-or-size stimulus (the "
                       "engine's into a materialized area, at one Bernoulli "
                       "draw) makes the refracted item rotate through its tied "
                       "connected set and the benefit vanishes"),
        caveat="Found by re-measuring PREREG_refraction_capacity.md after the "
               "hashed selector's sign defect (1b475fc) -- its Amendment 1 "
               "('spends the substrate') was that defect. The n/k law was "
               "counted as holding at n/k = 133 while both cells were censored; "
               "resolved, they disagree by 0.68 and the k = 30 one is out of "
               "regime -- k p >= 3 ln n is a precondition, not a footnote. Three "
               "ratios do not fix an exponent that is falling (2.1 -> 1.8). The "
               "multiplier is at 0.5 beta; 0.7 beta converges only given T = 16 "
               "and holds ~185 at n/k = 67, but strength is otherwise a plateau "
               "whose lower edge (below 0.3 beta) is unmeasured. The gate is "
               "two in-regime cells (n/k = 67, 133); n/k = 33 and out of regime "
               "are unmeasured, and the gated ceiling's doubling exponent (1.71) "
               "may keep falling.",
    ),
    Result(
        id="REFRACTION-CANCELS-CONVERGENCE",
        engine="hashed substrate (HashedArea with AreaFiber/StimulusFiber)",
        status=Status.MEASURED,
        sensitivity_gap="The claimed mechanism has no identified immutable "
                        "result artifact, so its null cannot yet be checked.",
        claim="[RE-MEASURED 2026-09-04 with the selector fixed (1b475fc): the "
              "churn above ~0.75 beta stands; the intermediate-strength rows "
              "were a selector artefact -- at 0.5 beta the recurrent assembly "
              "converges, relocates once when the clip binds (~round 40-60, "
              "the registered P2 prediction) and holds; at 0.7 beta most "
              "brains no longer converge. The transition lies in 0.5-0.7 "
              "beta. And BELOW it, with the bias-MASKED readout, a refracted "
              "recurrent area at 0.5 beta stored every assembly of the grid "
              "(rank-1 1.000 to M = 32, above the bar to 256, pairwise 0.00 x "
              "chance) against a Hebbian ceiling of 23.5 -- capacity becomes "
              "FILL-limited far above the interference limit; the earlier "
              "'spends the substrate' reading was the defect. Post hoc; a "
              "registration is owed. PREREG_refraction_capacity.md.] "
              "Refraction at strength s is the anti-Hebbian counterweight on a "
              "neuron's own repeated input: raw*(1+beta)^t minus the charged "
              "bias leaves net drive growing by (beta - s)*raw per win, so at "
              "s = beta it is CONSTANT. A feedforward area needs no convergence "
              "force (its input ranking is fixed) and holds; a RECURRENT "
              "assembly converges only through rich-get-richer, and above "
              "s ~ 0.75 beta it never converges and churns through the whole "
              "area -- refraction there is a firing-rate equalizer, and "
              "firing-rate homeostasis is incompatible with attractor memory "
              "in a recurrent k-WTA area. Below the transition it is the "
              "anti-merging force of [[REFRACTION-ANTI-MERGING]]: ~25x the "
              "Hebbian ceiling, read with the bias masked.",
        source="This repository; PREREG_refraction_capacity.md.",
        evidence=("seq_refraction_wander.py at n=4000 k=100 p=0.5 beta=0.1 "
                  "w_max=20, 16 brains, 240 rounds: s/beta = 0.5, 0.7 converge "
                  "(rounds 48, 45 vs control 4; late stability 1.000); 0.8, "
                  "0.9, 0.95, 1.0 never converge (late stability <= 0.22, fill "
                  "1.000); feedforward at s = beta holds (late 0.993)",
                  "the w_max saturation arithmetic ln(w_max)/ln(1+beta) + "
                  "(1-1/w_max)/beta ~ 41 appears as a transient re-ranking at "
                  "rounds 44-48 below the transition, which the assembly "
                  "survives",
                  "capacity protocol at s = 0.5 beta, re-measured with the "
                  "selector fixed: see [[REFRACTION-ANTI-MERGING]] (the "
                  "earlier 'M* 19.6 vs 23.5' reading was the defect)",
                  "bias-on partial-cue recall 0.250 vs bias-masked 0.984 at "
                  "M=8, same training: the intrinsic bias vetoes recall from "
                  "a partial cue, as the identity predicts"),
        evidence_refs=(
            EvidenceRef("research/experiments/seq_refraction_wander.py", "producer"),
            EvidenceRef("research/notes/memory/PREREG_refraction_capacity.md", "registration"),
            EvidenceRef("research/notes/memory/AUDIT_refraction_scaling.md", "analysis"),
        ),
        provenance_gap="wander and bias-readout numbers have no identified immutable result artifact",
        preconditions=("recurrent area; strength quoted relative to beta; "
                       "T=8 rounds per item in the capacity protocol",
                       "the reference uses RefractedArea only as a FEEDFORWARD "
                       "conjunction area driven by its full input at recall, "
                       "where none of this applies"),
        implemented_by=("neural_assemblies/core/_homeostasis.py",
                        "neural_assemblies/core/torch_engine/_hashed.py"),
        caveat="The critical ratio is bracketed in (0.7, 0.8) at one operating "
               "point; a transient-handicap estimate gives ~2/3. Whether "
               "REFRACTION-NEEDS-LOAD's under-loaded non-convergence is this "
               "mechanism (the arc's state input is itself changing) is "
               "suggested, not established. TWO FURTHER LIMITS (AUDIT_"
               "refraction_scaling.md): refraction + synaptic scaling on the "
               "same area is INCOMPATIBLE -- a feedforward arc that holds "
               "under either alone loses its assemblies within ~10 "
               "presentations under both (late stability 0.12 vs 1.00), "
               "because scaling moves the raw drive the bias is charged "
               "against; and with no clip the identity dies of float32 "
               "cancellation at ~100-150 wins. Never scale a refracted area.",
    ),
    Result(
        id="AC-CAP",
        engine="incompletely recorded: graded-similarity evidence compares explicit, materialized and sampled numpy_sparse; capacity-note run provenance remains unresolved",
        status=Status.MEASURED,
        sensitivity_gap="The underlying capacity run is unidentified and has no "
                        "retained mechanism-null comparison.",
        claim="Assembly capacity is EXTENSIVE: about M_max ~ 1.15 n/k distinct "
              "assemblies per area.",
        source="This repository (critical-load measurement).",
        evidence=("research/notes/categories/capacity_is_not_the_constraint_separation_is.md",
                  "research/notes/substrate/graded_similarity_and_sampler_load.md"),
        evidence_refs=(
            EvidenceRef("research/notes/categories/capacity_is_not_the_constraint_separation_is.md", "analysis"),
            EvidenceRef("research/notes/substrate/graded_similarity_and_sampler_load.md", "analysis"),
        ),
        provenance_gap="capacity-note run has no identifiable immutable artifact",
        caveat="The capacity-note run's engine provenance remains unresolved; "
               "this entry does not certify the numerical capacity claim. "
               "This is why k and p are not interchangeable routes to a regime: "
               "raising k to reach kp spends capacity and forces n up with it.",
    ),

    Result(
        id="RATE-HETEROGENEITY",
        engine="numpy_explicit (materialized copied-fiber protocol)",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/mechanism.per-fiber-plasticity/per-fiber-plasticity-20260910/results.json",
            sample_path="observations/rows/*/seed",
            treatment_path="observations/rows/*/cells/forward/ratio",
            control_path="observations/rows/*/cells/equal/ratio",
            relation="all-greater", minimum_effect=9.0,
            mechanism="per-fiber rate contrast versus equal-rate null",
        ),),
        claim="Learning rate is settable PER FIBER: copied fibers with identical "
              "pre/post activity at beta 0.06 and 0.005 diverge 14.35-fold after "
              "50 updates without reaching the weight clip.",
        source="This repository.",
        evidence=("20 seeds: fast geometric mean 18.4201012, slow 1.2832253, "
                  "ratio 14.3545340; swapped labels reproduce the ratio, equal-beta "
                  "null is 1.0, and maximum weight 18.4201012 is below w_max=20",),
        evidence_refs=(
            EvidenceRef("research/notes/memory/PREREG_per_fiber_plasticity.md", "registration"),
            EvidenceRef("research/experiments/per_fiber_plasticity.py", "producer"),
            EvidenceRef("research/results/runs/mechanism.per-fiber-plasticity/per-fiber-plasticity-20260910/results.json", "artifact"),
        ),
        implemented_by=("neural_assemblies.core.brain.Brain.update_plasticity",
                        "research.experiments.per_fiber_plasticity"),
        caveat="This isolates the multiplicative rate on copied materialized NumPy "
               "fibers; it does not measure assembly quality or biological "
               "heterogeneity, and it does not establish sampled or hashed backend "
               "equivalence. Near-zero intervals reflect the controlled arithmetic, "
               "not population certainty. The old saturated beta=0.5 number remains "
               "unreproduced and is not evidence for this entry.",
    ),

    # ------------------------------------------------------------- extensions
    Result(
        id="DUAL-RATE",
        status=Status.EXTENSION,
        claim="Running fast and slow pathways at once is FUNCTIONALLY useful: "
              "a high-beta fiber binds in one shot (episodic) while a low-beta "
              "fiber accumulates statistics (semantic), and a system with both "
              "does something neither does alone.",
        source="Proposed here; not proved and not measured.",
        preconditions=("[[RATE-HETEROGENEITY]] -- the mechanism exists",
                       "[[SEQ-BETA-WINDOW]] -- each rate must sit inside its "
                       "own window, and the windows may not overlap"),
        caveat="UNTESTED. That the knob exists is measured; that turning it "
               "buys anything is not. Two known tensions to design against: "
               "beta trades capacity against depth (low favours capacity, high "
               "favours depth), and the afferent count flips the SIGN of beta's "
               "effect, so a rate that helps one fiber can hurt another at a "
               "different kp. Falsified if a dual-rate organ matches the better "
               "of its two single-rate controls.",
    ),
    Result(
        id="SEQ-ORGAN-EMBEDS",
        engine="numpy_sparse (original organ-density experiment; sampled-arc provenance limitation)",
        status=Status.MEASURED,
        sensitivity_gap="The materialized density rerun predates immutable run "
                        "records and has no encoded disabled-organ null.",
        claim="A sequence organ runs at its own regime INSIDE a brain whose "
              "ambient density is far lower, given per-fiber p.",
        source="This repository.",
        preconditions=("per-fiber p (`Brain.add_connectivity`, numpy_exact and "
                       "numpy_sparse)",
                       "EVERY per-fiber quantity scaled by the fiber's p, not "
                       "the global one -- see the caveat"),
        evidence=("research/experiments/seq_a1_local_regime.py: ambient p=0.05 "
                  "with organ fibers at p=0.4 gives 10/10 correct trajectories, "
                  "matching the uniform p=0.4 result, while the same organ left "
                  "at the ambient density gives 0/10",),
        evidence_refs=(
            EvidenceRef("research/results/sequence/seq_a1_local_regime_results.json", "artifact",
                        "sampled arc; sequence verdict void"),
            EvidenceRef("research/results/sequence/seq_a1_local_regime_results_materialized.json", "artifact"),
            EvidenceRef("research/experiments/seq_a1_local_regime.py", "producer"),
        ),
        provenance_gap="legacy artifacts lack runner records and source archives",
        caveat="The original organ-density experiment used the sampled numpy arc; "
               "its sequence-dynamics numbers are void under PREREG_sampler_audit.md "
               "until reproduced materialized or hashed. "
               "The heterogeneous path found a defect that the regime audit "
               "could NOT see: the stimulus weight clamp was scaled by the "
               "global p while the weights were drawn at the fiber's p, so a "
               "dense fiber in a sparse brain saturated at a sparse ceiling. "
               "The organ read 1/10 while every area reported comfortably "
               "in-regime. Any NEW per-fiber quantity is a candidate for the "
               "same class. The pooled candidate draw also remains "
               "moment-matched -- exact in the first two moments, an "
               "approximation beyond them.",
    ),
    Result(
        id="SEQ-STATE-CODE-EMERGENT",
        status=Status.EXTENSION,
        claim="The state alphabet can be INDUCED from data rather than assigned.",
        source="Not proved and not measured, here or in the source paper.",
        caveat="THE LOAD-BEARING GAP. Both our FSM and the reference assign "
               "state assemblies as disjoint blocks by hand. Every result under "
               "[[SEQ-FSM]] and [[SEQ-EXACT-RECOVERY]] is therefore about "
               "running a machine over a GIVEN alphabet, not about discovering "
               "one. Language needs the latter.",
    ),

    # ------------------------------------------------- representation algebra
    Result(
        id="KWTA-TIE-FRAGILE",
        engine="torch/CUDA selector prototype; not a Brain-engine conformance claim",
        status=Status.MEASURED,
        sensitivity_gap="The selector prototype retained no raw artifact for a "
                        "tie-free negative comparison.",
        claim="The k-WTA bar is routinely TIED, so anything that perturbs the "
              "drive in its last bits -- a change of summation order, of "
              "arithmetic, or of tie-break policy -- can change WHICH neurons "
              "fire, not merely their order.",
        source="The base drive is a Bernoulli COUNT, hence integer-valued, so "
               "exact ties are the common case rather than an edge case. "
               "Measured 5-18 columns tied at the bar per brain at n=4000-50000 "
               "(research/experiments/gpu_radix_select_prototype.py). An ulp-level "
               "change to a summation once moved sixteen cells of an exact "
               "table with no direction to it.",
        preconditions=("integer or near-integer drive, i.e. before heavy "
                       "potentiation spreads the values",
                       "k-WTA selecting at a bar that several columns reach"),
        evidence=("research/experiments/gpu_radix_select_prototype.py",),
        evidence_refs=(EvidenceRef("research/experiments/gpu_radix_select_prototype.py", "producer"),),
        provenance_gap="prototype measurement has no retained raw result artifact",
        implemented_by=("neural_assemblies.core.numpy_engine._kwta_prune",),
        caveat="`_kwta_prune` records the operational rule this implies: making "
               "the selector's tie-break CANONICAL is a science-affecting "
               "change needing its own registration, not an optimisation to "
               "smuggle in. The fused selector folds the index into the sort "
               "key so that ties break to the smallest index BY CONSTRUCTION, "
               "which is exactly why it is a separate entry point.",
    ),

    Result(
        id="HEBB-OUTER-PRODUCT",
        status=Status.PROVED,
        claim="The Hebbian co-firing count is a SUM OF RANK-1 OUTER PRODUCTS: "
              "count = SUM_t x_{t-1} x_t^T, with x_t the 0/1 winner indicator "
              "at round t. Restricted to the rows and columns a window of T "
              "rounds touches it is a matrix product R^T C of two thin 0/1 "
              "matrices; equivalently count[i,j] = popcount(rm[i] & cm[j]) "
              "with rm, cm the per-neuron round bitmasks.",
        source="Immediate from the update rule: `w[i,j] *= (1+beta)` fires "
               "exactly when i is a source winner and j a target winner, so "
               "the exponent counts co-firing rounds and nothing else.",
        preconditions=("plasticity purely MULTIPLICATIVE, so the exponent is "
                       "the whole state",
                       "the pairing must match the engine: source winners x "
                       "target winners, i.e. prev x new for a recurrent fiber"),
        evidence=("research/experiments/gpu_writeback_gemm_prototype.py",),
        implemented_by=("neural_assemblies.core.torch_engine._batched",
                        "neural_assemblies.core.torch_engine._fused_cuda"),
        caveat="An event carries 2k numbers, so materialising the k x k cross "
               "product writes the same information k/2 times over -- a "
               "redundancy created BEFORE any sort is reached, which is why "
               "sorting strategies cannot recover it. The two forms answer "
               "different questions: the GEMM materialises a block, the "
               "popcount evaluates one cell. `batched_project_independent` "
               "uses new x new rather than prev x new and is therefore NOT "
               "interchangeable with the engine.",
    ),
    Result(
        id="DRIVE-SPLIT",
        status=Status.PROVED,
        claim="With a Bernoulli 0/1 base B and G[i,j] = chain(1, count[i,j]), "
              "the drive splits as "
              "1_S^T (B (.) G) = 1_S^T B + SUM_{i in S, j} B[i,j] D[i,j] "
              "where D = G - 1 is nonzero only on potentiated cells. The "
              "correction is a SPARSE MATVEC over D restricted to |S| = k "
              "rows, so its intrinsic cost is the number of stored deviations "
              "in those rows and nothing else.",
        source="Algebraic identity, given that the base is 0/1: an ABSENT cell "
               "stays absent however often it is potentiated, and a present "
               "one starts at exactly 1.0, so chain(base, c) = base * tab[c].",
        preconditions=("base strictly 0/1 -- recruitment OVERRIDES cells, so "
                       "the effective base is 1 wherever a cell was written",
                       "tab replayed with the engine's own per-step "
                       "multiply-and-clip, NOT min((1+beta)^c, w_max)"),
        evidence=("research/experiments/gpu_hashed_deviations_prototype.py",),
        implemented_by=("neural_assemblies.core.torch_engine._fused_cuda",),
        caveat="A representation that cannot say WHICH cells are nonzero must "
               "visit every (row, column) pair: the bitmask form costs "
               "O(k n W) with W = ceil(rounds/64) against the intrinsic "
               "O(sum_{i in S} nnz_i), a ratio n^2 T / (64 k^2) that is "
               "INDEPENDENT of M -- 8894x at n=16000, k=60, T=8. Changing the "
               "representation changes the SUMMATION ORDER, and float32 "
               "addition is not associative, so ties at the k-WTA bar can "
               "flip ([[KWTA-TIE-FRAGILE]]).",
    ),
    Result(
        id="CAP-RATIO",
        engine="hashed AssemblyMemory / exact count-then-apply path",
        status=Status.MEASURED,
        sensitivity_gap="The immutable replay covers one registered capacity "
                        "cell and does not retain a mechanism-disabled null.",
        claim="The assembly-capacity ceiling M* is a function of n/k ALONE, "
              "not of n and k separately.",
        source="Held-out test registered in "
               "research/notes/memory/PREREG_capacity_nk_law.md (bar CS1) before the "
               "data existed. Holding n/k = 66.67 while n varies four-fold "
               "gives M* = 68.8 / 73.1 / 67.6 at n = 4000 / 8000 / 16000 on "
               "the exact count-then-apply path (engine-parity verified) -- "
               "constant to +/-4%, where any law M* = f(n) predicts ~4x. All "
               "three cells uncensored (fill 0.69-0.76), and all inside the "
               "band registered before the data existed.",
        preconditions=("in regime, kp >= 3 ln n [[SEQ-REGIME]]",
                       "ceiling read from the CURVE, gated on distinctness",
                       "fill at the ceiling below 0.95, else the tiling limit "
                       "is what is being measured",
                       "AT FIXED p AND beta: M* moved 1.6x across p in "
                       "[0.3, 0.7] and ~6x across beta in [0.05, 0.20] at "
                       "fixed n/k = 40 (PREREG_crosstalk_mechanism.md), so "
                       "the ratio law holds within an operating point, not "
                       "across them"),
        evidence=("research/experiments/seq_capacity_scaling.py",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.capacity-scaling/capacity-record-consumed-20260910/results.json", "artifact"),
            EvidenceRef("research/notes/memory/PREREG_capacity_nk_law.md", "registration"),
            EvidenceRef("research/experiments/seq_capacity_scaling.py", "producer"),
        ),
        caveat="This result survived a WRONG RETRACTION: an intermediate "
               "CSR deviation store applied the potentiation table per count "
               "FRAGMENT, and (tab[c0]-1)+(tab[c1]-1) != tab[c0+c1]-1, "
               "inflating ceilings ~25% (88.3/92.1/83.2); those numbers were "
               "briefly recorded here as the verified ones. Multi-episode "
               "ENGINE parity exposed it (rel 2e-3) and the exact path lands "
               "back on the first run's values. The EXPONENT remains "
               "unestablished: b = 2.19 +/- 0.05, above 2, NO mechanism -- "
               "reported, never quoted. A synapse bound "
               "M ~ n^2 p / (k ln(n/k)) is REFUTED by CS1: it is not a "
               "function of n/k alone -- and so is plain second-order "
               "CROSSTALK, refuted by registered test: the crosstalk ratio "
               "cancels p and beta, but M* ~ p^-0.6 and ~ beta^-1.3 at fixed "
               "n/k. No mechanism is adopted. Operational rule from the wrong "
               "retraction: when two implementations disagree, a test they "
               "BOTH pass verifies neither -- arbitrate with engine parity on "
               "the DRIVE.",
    ),
    Result(
        id="CAP-ANCHOR-RATIO",
        engine="hashed AssemblyMemory / capacity-scaling protocol",
        status=Status.MEASURED,
        sensitivity_gap="Registered legacy outcomes are not yet packaged with "
                        "an immutable anchor-disabled comparison.",
        claim="The capacity ceiling is set at FORMATION by the ratio of the "
              "stimulus anchor to the trained recurrent pull. Density p and "
              "gain beta enter through that ratio, so an excursion in either "
              "is undone by a computed change in anchor size.",
        source="This repository; PREREG_formation_interference.md (F2) and "
               "PREREG_anchor_ratio.md (A1-A3).",
        evidence=("F2: anchor 100 -> 200 lifts M* 23.5 -> 67.1 (2.86x), while "
                  "retrieval never sees the stimulus",
                  "A1-A3: beta 0.20 / p 0.7 / p 0.3 excursions with "
                  "uncompensated M* 8.0 / 17.6 / 28.8 land at 29.4 / 25.0 / "
                  "25.3 once the anchor is set from the measured exponents -- "
                  "a 3.6x spread collapses to 1.18x, all within 25% of 23.5",
                  "F3: past the cliff, erosion is retroactive and diffuse "
                  "(early items fail worst), not one-shot capture"),
        evidence_refs=(
            EvidenceRef("research/notes/memory/PREREG_anchor_ratio.md", "registration"),
            EvidenceRef("research/notes/memory/PREREG_formation_interference.md", "registration"),
            EvidenceRef("research/experiments/seq_capacity_scaling.py", "producer"),
        ),
        provenance_gap="registered legacy outcomes are not yet packaged as immutable runner artifacts",
        preconditions=("n=4000, k=100, T=8, w_max=20, arm B, 16 brains; "
                       "one operating-point neighbourhood",
                       "exponents s^1.52 p^-0.6 beta^-1.3 are two- and "
                       "three-point fits used ONLY to place cells; none is "
                       "adopted"),
        caveat="A1 sits on the numeric band's edge (29.4 vs < 29.4) and "
               "passes on the registered prose criterion; all three "
               "compensated cells overshoot upward, so the anchor exponent is "
               "probably slightly high. The (n/k)^2 dependence of CAP-RATIO is "
               "the pull's chance-overlap term and is NOT derived here.",
    ),
    Result(
        id="CAP-CLIFF",
        engine="hashed AssemblyMemory / exact count-then-apply path",
        status=Status.MEASURED,
        sensitivity_checks=(SensitivityCheck(
            artifact="research/results/runs/memory.capacity-scaling/"
                     "capacity-cliff-sensitivity-20260911/results.json",
            sample_path="observations/cells/*/seeds",
            treatment_path="observations/cells/*/checkpoints/192/rank1",
            control_path="observations/cells/*/checkpoints/512/rank1",
            relation="all-greater", minimum_effect=0.5,
            mechanism="safe-load versus overloaded capacity readout",
        ),),
        claim="Capacity failure is a CLIFF, not a slope: past the ceiling the "
              "assemblies shatter rather than degrading gracefully.",
        source="research/experiments/seq_capacity_scaling.py, on the exact "
               "count-then-apply path (engine-parity verified). At n=8000, "
               "k=60, 16 brains: M=256 gives rank-1 0.938 at pairwise overlap "
               "1.37x chance with every assembly distinct; M=320 gives 0.486; "
               "M=384 gives 0.014 at overlap 4.75x. One doubling (256 -> 512) "
               "takes rank-1 from 0.938 to 0.000. Ceiling M* = 284 at fill "
               "0.944, uncensored.",
        preconditions=("half-cue rank-1 readout against ALL M stored items",
                       "distinctness gate applied, so a collapsed set scores 0"),
        evidence=("research/experiments/seq_capacity_scaling.py",),
        evidence_refs=(
            EvidenceRef("research/results/runs/memory.capacity-scaling/capacity-record-consumed-20260910/results.json", "artifact",
                        "maintained-run replay at (4000,60), not the registered cliff cell"),
            EvidenceRef("research/results/runs/memory.capacity-scaling/capacity-cliff-sensitivity-20260911/results.json", "artifact",
                        "fresh exact-path reproduction and retained sensitivity vectors"),
            EvidenceRef("research/experiments/seq_capacity_scaling.py", "producer"),
        ),
        caveat="There is no soft capacity margin to trade against: a design "
               "must know where the ceiling is and stay under it. Sharing "
               "itself is healthy -- at M=256 the load Mk/n is 1.9, nearly "
               "two assemblies per neuron, with overlap still 1.37x chance -- "
               "so the cliff is not caused by sharing. The transition occupies "
               "roughly one 1.5x step in M (256 -> 384). A 20-brain maintained-"
               "path reproduction gave 0.909 / 0.433 / 0.0188 at those three "
               "checkpoints; its interpolated M* = 310.1 remains fill-censored.",
    ),
]

RESULTS: Dict[str, Result] = {r.id: r for r in _RESULTS}


def cite(result_id: str) -> Result:
    """Look up a result, raising if the ID is unknown."""
    try:
        return RESULTS[result_id]
    except KeyError:
        raise KeyError(
            f"unknown result {result_id!r}. Add it to neural_assemblies/theory.py "
            f"rather than citing an ID that does not resolve. Known: "
            f"{', '.join(sorted(RESULTS))}"
        ) from None


def extensions() -> List[Result]:
    """Everything the codebase relies on beyond what is proved or measured."""
    return [r for r in _RESULTS if r.status == Status.EXTENSION]


def _source_files(root: str) -> List[str]:
    out = []
    for base, _dirs, names in os.walk(root):
        if any(part in base for part in (".git", "__pycache__", "reference")):
            continue
        out += [os.path.join(base, n) for n in names
                if n.endswith((".py", ".md"))]
    return out


def unresolved_citations(root: str) -> Dict[str, List[str]]:
    """Map each dangling result citation to the files citing it.

    Lowercase-kebab links are operator memory and are ignored by `CITATION`.
    """
    missing: Dict[str, List[str]] = {}
    for path in _source_files(root):
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:                                      # noqa: PERF203
            continue
        for rid in set(CITATION.findall(text)):
            if rid not in RESULTS:
                missing.setdefault(rid, []).append(path)
    return missing


def _json_values(document, path: str) -> list:
    """Resolve RFC 6901 tokens with ``*`` list expansion into scalar values."""
    nodes = [document]
    for raw_token in path.split('/'):
        if not raw_token:
            raise ValueError("empty sensitivity path component")
        if re.search(r"~(?![01])", raw_token):
            raise ValueError("invalid RFC 6901 escape in sensitivity path")
        token = raw_token.replace("~1", "/").replace("~0", "~")
        expanded = []
        for node in nodes:
            if token == '*':
                if not isinstance(node, list):
                    raise ValueError("sensitivity wildcard requires a list")
                expanded.extend(node)
            elif isinstance(node, dict) and token in node:
                expanded.append(node[token])
            else:
                raise ValueError(f"missing sensitivity path component {token!r}")
        nodes = expanded
    if nodes and all(isinstance(node, list) for node in nodes):
        return [value for node in nodes for value in node]
    if any(isinstance(node, (dict, list)) for node in nodes):
        raise ValueError("sensitivity path must resolve only to scalar values")
    return nodes


def _sensitivity_errors(result: Result, check: SensitivityCheck, root: str) -> list[str]:
    prefix = f"{result.id}: sensitivity {check.mechanism or '<unnamed>'}"
    if not check.mechanism.strip():
        return [f"{prefix} must name the mechanism under test"]
    if not any(ref.role == "artifact" and ref.path == check.artifact
               for ref in result.evidence_refs):
        return [f"{prefix} artifact is not a typed artifact evidence edge"]
    allowed = {"all-greater", "all-less", "all-different"}
    if check.relation not in allowed:
        return [f"{prefix} has unsupported relation {check.relation!r}"]
    if (type(check.minimum_effect) not in (int, float)
            or not math.isfinite(check.minimum_effect)
            or check.minimum_effect <= 0):
        return [f"{prefix} has invalid minimum effect"]
    normalized = os.path.normpath(check.artifact).replace("\\", "/")
    target = os.path.abspath(os.path.join(root, check.artifact))
    if (not check.artifact or normalized != check.artifact
            or os.path.commonpath((root, target)) != root):
        return [f"{prefix} has unsafe artifact path {check.artifact!r}"]
    try:
        with open(target, encoding="utf-8") as handle:
            document = json.load(handle)
        samples = _json_values(document, check.sample_path)
        treatment = _json_values(document, check.treatment_path)
        control = _json_values(document, check.control_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return [f"{prefix} cannot resolve retained values: {exc}"]
    if (not samples or len(samples) != len(set(map(repr, samples)))):
        return [f"{prefix} sample identities are empty or duplicated"]
    if len(samples) != len(treatment) or len(treatment) != len(control):
        return [f"{prefix} sample/treatment/control vectors have unequal lengths"]
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for value in [*treatment, *control]):
        return [f"{prefix} values must be finite numbers"]
    if check.relation == "all-greater":
        effects = [left - right for left, right in zip(treatment, control, strict=True)]
    elif check.relation == "all-less":
        effects = [right - left for left, right in zip(treatment, control, strict=True)]
    else:
        effects = [abs(left - right) for left, right in zip(treatment, control, strict=True)]
    if any(effect < check.minimum_effect for effect in effects):
        return [
            f"{prefix} does not move by {check.minimum_effect:g} under "
            f"{check.relation}; minimum retained effect is {min(effects):g}",
        ]
    return []


def evidence_reference_errors(root: str) -> List[str]:
    """Validate local evidence edges and retained mechanism sensitivity.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-result-sensitivity

    This checks whether the evidence still exists and whether each declared
    treatment/control contrast still clears its frozen minimum effect. It does
    not decide whether the scientific claim follows from that contrast.
    """
    root = os.path.abspath(root)
    errors = []
    for result in _RESULTS:
        if result.status == Status.MEASURED and not (result.evidence_refs or result.provenance_gap):
            errors.append(f"{result.id}: measured result has no typed evidence or provenance gap")
        if (result.status == Status.MEASURED
                and not any(ref.role == "artifact" for ref in result.evidence_refs)
                and not result.provenance_gap):
            errors.append(f"{result.id}: no result artifact and no provenance gap")
        if (result.status == Status.MEASURED
                and not result.sensitivity_checks and not result.sensitivity_gap):
            errors.append(f"{result.id}: no retained sensitivity check or explicit gap")
        for ref in result.evidence_refs:
            normalized = os.path.normpath(ref.path).replace("\\", "/")
            target = os.path.abspath(os.path.join(root, ref.path))
            if (not ref.path or normalized != ref.path or os.path.commonpath((root, target)) != root):
                errors.append(f"{result.id}: unsafe evidence path {ref.path!r}")
            elif not os.path.isfile(target):
                errors.append(f"{result.id}: dangling evidence path {ref.path}")
            if ref.role not in EVIDENCE_ROLES:
                errors.append(f"{result.id}: unsupported evidence role {ref.role!r}")
        for check in result.sensitivity_checks:
            errors.extend(_sensitivity_errors(result, check, root))
    return errors


def format_index(results: Sequence[Result] = ()) -> str:
    """Render the register, extensions last so they are what you read last."""
    order = {Status.PROVED: 0, Status.MEASURED: 1, Status.EXTENSION: 2}
    items = list(results) or _RESULTS
    return "\n\n".join(str(r) for r in
                       sorted(items, key=lambda r: (order[r.status], r.id)))


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _split_findings(text: str) -> List[str]:
    """A caveat written as '(1) ... (2) ...' becomes one item per number;
    any other caveat is one item."""
    parts = re.split(r"\s*\(\d+\)\s+", text.strip())
    parts = [x.strip() for x in parts if x.strip()]
    return parts if len(parts) > 1 else [text.strip()]


def render_markdown() -> str:
    """The register as Markdown: one section per entry, in file order."""
    out = ["# Register of results", "",
           "Rendered from `neural_assemblies/theory.py` by "
           "`python -m neural_assemblies.theory --render`; do not edit by hand. "
           "Each entry is cited elsewhere by its ID in double brackets. Statuses: PROVED (in the "
           "cited source, inside its preconditions), MEASURED (in this repository, "
           "in the regime named), EXTENSION (relied on beyond either).", ""]
    by_status: Dict[str, List[Result]] = {}
    for r in _RESULTS:
        by_status.setdefault(r.status, []).append(r)
    out.append("| ID | Status | Engine / substrate | Claim |")
    out.append("|----|--------|--------------------|-------|")
    for r in _RESULTS:
        first = r.claim.split(". ")[0].rstrip(".") + "."
        out.append(f"| [`{r.id}`](#{r.id.lower()}) | {r.status} | {r.engine or 'Not an empirical entry'} | {first} |")
    out.append("")
    for r in _RESULTS:
        out.append(f"## {r.id}")
        out.append("")
        out.append(f"**Status.** {r.status}. **Source.** {r.source}")
        out.append("")
        if r.engine:
            out.append(f"**Engine / substrate.** {r.engine}")
            out.append("")
        out.append(f"**Claim.** {r.claim}")
        out.append("")
        if r.preconditions:
            out.append("**Requires.**")
            out.extend(f"- {x}" for x in r.preconditions)
            out.append("")
        if r.evidence:
            out.append("**Evidence.**")
            out.extend(f"- {x}" for x in r.evidence)
            out.append("")
        if r.evidence_refs:
            out.append("**Evidence files.**")
            for ref in r.evidence_refs:
                suffix = f" â€” {ref.limitation}" if ref.limitation else ""
                out.append(f"- [{ref.path}](../{ref.path}) ({ref.role}){suffix}")
            out.append("")
        if r.provenance_gap:
            out.append(f"**Provenance gap.** {r.provenance_gap}")
            out.append("")
        if r.sensitivity_checks:
            out.append("**Mechanism sensitivity.**")
            for check in r.sensitivity_checks:
                out.append(
                    f"- {check.mechanism}: `{check.treatment_path}` "
                    f"{check.relation} `{check.control_path}` by at least "
                    f"{check.minimum_effect:g}, retained in "
                    f"[{check.artifact}](../{check.artifact}) and paired by "
                    f"`{check.sample_path}`."
                )
            out.append("")
        if r.sensitivity_gap:
            out.append(f"**Sensitivity gap.** {r.sensitivity_gap}")
            out.append("")
        if r.implemented_by:
            out.append("**Used by.** " + "; ".join(f"`{x}`" for x in r.implemented_by))
            out.append("")
        if r.caveat:
            items = _split_findings(r.caveat)
            if len(items) > 1:
                out.append("**Findings and caveats.**")
                out.extend(f"{i}. {x}" for i, x in enumerate(items, 1))
            else:
                out.append(f"**Caveat.** {items[0]}")
            out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


if __name__ == "__main__":
    import sys as _sys
    if "--render" in _sys.argv:
        _sys.stdout.write(render_markdown())
    else:
        print(format_index())
