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

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

#: Matches an UPPERCASE-DASHED result citation in double brackets. Lowercase
#: kebab links are operator memory, not results, and never match.
CITATION = re.compile(r"\[\[([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*)\]\]")


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
class Result:
    """One citable claim, with the conditions under which it holds."""

    id: str
    status: str
    claim: str
    source: str
    preconditions: Sequence[str] = field(default_factory=tuple)
    evidence: Sequence[str] = field(default_factory=tuple)
    implemented_by: Sequence[str] = field(default_factory=tuple)
    caveat: str = ""

    def __str__(self) -> str:
        head = f"[{self.status}] {self.id}: {self.claim}"
        bits = [f"    source: {self.source}"]
        if self.preconditions:
            bits.append("    requires: " + "; ".join(self.preconditions))
        if self.evidence:
            bits.append("    evidence: " + "; ".join(self.evidence))
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
        claim="Winner selection is reliable only when a target neuron receives "
              "kp >= 3 ln n synapses FROM THE DRIVING ASSEMBLY.",
        source="Dabagia et al. (arXiv:2306.03812); assumed by every theorem in "
               "the paper, and satisfied by its own FSM demo at n=5000, k=70, "
               "p=0.4 (kp=28 per conjunct pair vs floor 25.6).",
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
        status=Status.MEASURED,
        claim="Crossing the kp >= 3 ln n floor is a CLIFF, not a slope: below it "
              "recovery is almost never exact and the machine fails; above it "
              "every seed runs correctly.",
        source="This repository.",
        evidence=("research/experiments/seq_a1_exactness_sweep.py: state kp "
                  "14 -> 4/100 exact steps and 4/10 trajectories; kp 21 -> "
                  "80/100 and 10/10; kp 28 -> 100/100 and 10/10, with the "
                  "transition at the predicted p = 18.6/70 = 0.266",),
        implemented_by=("neural_assemblies.diagnostics.regime_audit",),
        caveat="Mean overlap read 0.812 at the failing point while exactness "
               "was 4/100 -- the mean hides this mechanism entirely.",
    ),
    Result(
        id="SEQ-EXACT-RECOVERY",
        status=Status.MEASURED,
        claim="The state area is a DISCRETE attractor: k-WTA maps a whole "
              "neighbourhood onto exactly one stored assembly in one step. "
              "Recovery must be EXACT -- 69 of 70 neurons is a failure, not a "
              "near-miss -- because the arc amplifies any residual ~8x per step.",
        source="This repository.",
        evidence=("research/experiments/seq_a1_arc_transfer.py: "
                  "d(output loss)/d(input loss) = 8.17 ours, 10.33 reference",
                  "research/experiments/seq_a1_horizon.py: 2000 steps, 5/5 "
                  "seeds, zero errors, exact recovery every step",
                  "research/experiments/seq_a1_limit_cycle.py: constant input "
                  "gives an orbit closing bit-identically, 30/30",
                  "research/experiments/seq_s5_soft_census.py: at 60-120 "
                  "states, ~0.25-0.6% of transitions are SOFT (correct label, "
                  "one intruder neuron); first lattice exit equals first "
                  "soft-pair visit, 40/40 seeds, zero parameters"),
        caveat="Expansion and quantization are a PAIR. Amplification alone is "
               "chaos; it is only benign because a quantizing area follows it. "
               "Composition steps without a re-quantizing stage should be "
               "expected to drift. AND THE MAP HAS SOFT SPOTS at scale: a "
               "~0.5%-per-transition subset emits 69/70 of the right block "
               "(one intruder), so exactness holds until the word first "
               "visits a soft pair -- the measured 'horizon' is a "
               "first-hitting time, not a decay constant. Post-visit, "
               "trajectories recover (6/18), wander metastably for up to "
               "~330 steps, or derail into an absorbing off-lattice regime "
               "(12/18). Apparent group differences at fixed L were the "
               "pair-count |G|x|gens| at a FLAT per-pair rate, not "
               "solvability. Intruder mechanism open; norm_init=False (the "
               "parity substrate) leaves state-area hubs unchecked and is "
               "the registered suspect.",
    ),
    Result(
        id="ARC-CONJUNCT-EXPOSURE",
        status=Status.MEASURED,
        claim="A conjunction area collapses onto whichever conjunct is exposed "
              "more often, unless an opposing force (refraction) is present.",
        source="This repository; the same law as the role-binding gain result.",
        evidence=("research/experiments/seq_arc_refraction_reference.py: "
                  "ablating refraction takes across-symbol overlap 0.000 -> "
                  "0.989 and the task 3/3 -> 0/3",),
        preconditions=("BOTH overlap directions measured -- one alone cannot "
                       "distinguish a conjunction from collapse onto the other "
                       "conjunct",),
        caveat="Task #92 measured one direction, read 0.90-0.99, and concluded "
               "a conjunctive arc has no operating point. It has one.",
    ),
    Result(
        id="REFRACTION-PROPORTIONAL",
        status=Status.MEASURED,
        claim="Refraction must charge in proportion to the winner's raw drive. "
              "A constant increment is not an equivalent parameterization: "
              "Hebbian growth multiplies drive while a constant grows linearly, "
              "so its operating point MOVES with training duration.",
        source="This repository.",
        evidence=("research/experiments/seq_arc_refraction_reference.py: the "
                  "constant winning at 15 presentations fails at 30, while the "
                  "proportional rule passes both untouched",),
        implemented_by=("neural_assemblies/core/_refraction.py",),
    ),
    Result(
        id="REFRACTION-NEEDS-LOAD",
        status=Status.MEASURED,
        claim="A refracted conjunction area has an operating WINDOW in load "
              "M*k/n: below ~0.2 its assemblies never converge, above ~1.15 "
              "they do not fit. An arc must be SIZED to the number of "
              "conjunctions it holds.",
        source="This repository.",
        evidence=("research/experiments/seq_a2_refraction_load.py: sweeping arc "
                  "size at fixed content, a 3-conjunction arc goes 1/10 at load "
                  "0.04 to 10/10 at 0.21; a 9-conjunction arc holds 10/10 from "
                  "0.32 to 1.26 and collapses to 1/10 at 1.80",
                  "arc assembly stability across training: 0.286 from "
                  "presentation 5 to 15 under-loaded, 0.957 from 10 to 15 loaded"),
        preconditions=("refraction active -- this is a statement about what "
                       "refraction needs, not about k-WTA generally",),
        caveat="A NEW SILENT-FAILURE MODE. Under-loaded, every local diagnostic "
               "reads healthy -- the conjunction is clean, every area is "
               "in-regime, refraction is charging -- while the organ fails, "
               "because the assemblies never stopped moving. `regime_audit` "
               "cannot see it; assembly stability across training can. The "
               "upper bound is [[AC-CAP]] and is not independent of it. "
               "Bounds are approximate and from one task.",
    ),
    Result(
        id="REFRACTION-CANCELS-CONVERGENCE",
        status=Status.MEASURED,
        claim="Refraction at strength s is the anti-Hebbian counterweight on a "
              "neuron's own repeated input: raw*(1+beta)^t minus the charged "
              "bias leaves net drive growing by (beta - s)*raw per win, so at "
              "s = beta it is CONSTANT. A feedforward area needs no convergence "
              "force (its input ranking is fixed) and holds; a RECURRENT "
              "assembly converges only through rich-get-richer, and above "
              "s ~ 0.75 beta it never converges and churns through the whole "
              "area -- refraction there is a firing-rate equalizer, and "
              "firing-rate homeostasis is incompatible with attractor memory "
              "in a recurrent k-WTA area. Below the transition it converges "
              "~10x slower and orthogonalizes stored assemblies to below "
              "chance overlap, but spends fill and LOWERS capacity.",
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
                  "capacity protocol at s = 0.5 beta: pairwise overlap "
                  "0.00-0.05x chance (control ~1.5x) yet M* 19.6 vs 23.5, "
                  "fill 0.977 at M=24 -- the ceiling becomes fill-limited",
                  "bias-on partial-cue recall 0.250 vs bias-masked 0.984 at "
                  "M=8, same training: the intrinsic bias vetoes recall from "
                  "a partial cue, as the identity predicts"),
        preconditions=("recurrent area; strength quoted relative to beta; "
                       "T=8 rounds per item in the capacity protocol",
                       "the reference uses RefractedArea only as a FEEDFORWARD "
                       "conjunction area driven by its full input at recall, "
                       "where none of this applies"),
        implemented_by=("neural_assemblies/core/_refraction.py",
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
        status=Status.MEASURED,
        claim="Assembly capacity is EXTENSIVE: about M_max ~ 1.15 n/k distinct "
              "assemblies per area.",
        source="This repository (critical-load measurement).",
        evidence=("research/notes/capacity_is_not_the_constraint_separation_is.md",
                  "research/notes/graded_similarity_and_sampler_load.md"),
        caveat="This is why k and p are not interchangeable routes to a regime: "
               "raising k to reach kp spends capacity and forces n up with it.",
    ),

    Result(
        id="RATE-HETEROGENEITY",
        status=Status.MEASURED,
        claim="Learning rate is settable PER FIBER and genuinely bites: two "
              "fibers into the same area, driven by the same projections, "
              "diverge by more than an order of magnitude in weight.",
        source="This repository.",
        evidence=("beta=0.5 vs beta=0.01 into one target over 10 rounds: max "
                  "weight 20.000 (at the w_max clamp) vs 1.094",),
        implemented_by=("neural_assemblies.core.brain.Brain.update_plasticity",),
        caveat="Density is NOT yet settable per fiber on the production engine "
               "-- see [[SEQ-ORGAN-EMBEDS]]. And a fast fiber saturates against "
               "w_max, so 'fast' has a ceiling that 'slow' does not: an A/B "
               "across rates is confounded unless w_max is checked.",
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
        status=Status.MEASURED,
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
        caveat="The heterogeneous path found a defect that the regime audit "
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
        status=Status.MEASURED,
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
        status=Status.MEASURED,
        claim="The assembly-capacity ceiling M* is a function of n/k ALONE, "
              "not of n and k separately.",
        source="Held-out test registered in "
               "research/notes/PREREG_capacity_nk_law.md (bar CS1) before the "
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
        status=Status.MEASURED,
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
        status=Status.MEASURED,
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
        caveat="There is no soft capacity margin to trade against: a design "
               "must know where the ceiling is and stay under it. Sharing "
               "itself is healthy -- at M=256 the load Mk/n is 1.9, nearly "
               "two assemblies per neuron, with overlap still 1.37x chance -- "
               "so the cliff is not caused by sharing. The transition occupies "
               "roughly one 1.5x step in M (256 -> 384).",
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


def format_index(results: Sequence[Result] = ()) -> str:
    """Render the register, extensions last so they are what you read last."""
    order = {Status.PROVED: 0, Status.MEASURED: 1, Status.EXTENSION: 2}
    items = list(results) or _RESULTS
    return "\n\n".join(str(r) for r in
                       sorted(items, key=lambda r: (order[r.status], r.id)))


if __name__ == "__main__":
    print(format_index())
