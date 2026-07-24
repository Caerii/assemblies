"""
Assembly-calculus theory of MNIST visual classification.
=========================================================

This module records **hypotheses**, **ventral-stream information structures**, and a
**research roadmap** for advancing assembly-based vision beyond the current ~80%
regime toward robust digit recognition (>95%).  It is the canonical reference
for why simple hierarchical models plateau near 64% while recurrent models reach
~81%, and which assembly-calculus primitives are plausible next steps.

Empirical anchors (real MNIST CSV, seed=42, n_examples=50 unless noted):

+---------------------------+----------+-------------------------------------------+
| Model                     | Accuracy | Circuit signature                         |
+---------------------------+----------+-------------------------------------------+
| Simple hierarchical       | ~64%     | LOW→MID→HIGH feedforward, 5 rounds/class  |
| Advanced ventral stream   | ~79%     | + repeated exposure + CLASS consolidation |
| Advanced recurrent cortex | ~80%     | LOW→HIGH + HIGH→HIGH attractor loop       |
| Notebook single-layer     | ~81%     | Same recurrence, numpy/brain explicit     |
| Paper scale (n=5000)      | ~73%     | Single-layer, more exposure, not deeper   |
| Tier A multi-prototype    | ~79%     | K-exemplar fuzzy + connectome readout     |
| Tier B pattern complete   | ~82%     | Ventral readout + recovery metric         |
| Tier B consolidation      | ~79%     | wire_class_from_prototypes replay         |
| Tier C LRI cascade        | ~82%     | Readout baseline; confused-pair cascade   |
| Cross-domain vis/lang     | ~79%     | SEMANTIC hub (binding gated, H6)          |
| Representational panel    | diag.    | Phase I geometry + Phase II encoding    |
| Tier A merge-halves       | ~38%     | H9 falsified: merge overlap ~0.94 vs ventral ~0.23 |
+---------------------------+----------+-------------------------------------------+

The ~17 percentage-point gap between **simple hierarchical** and **recurrent**
models is the central phenomenon this document explains.


Part I — Hypotheses: why feedforward hierarchy plateaus
-------------------------------------------------------

**H1 — Attractor deficiency.**

*Statement.* A single feedforward pass ``LOW → MID → HIGH`` produces a transient
pattern vector, not a stabilized assembly attractor.  Recurrent cortex runs
``k_cap(h @ W + x @ A + bias)`` for multiple implicit steps via ``HIGH→HIGH``
self-projection; each step is a ``project`` round that lets Hebbian weights
pull the representation toward a class-conditional fixed point.

*Prediction.* Adding MID→MID or HIGH→HIGH recurrence to the two-layer stack
without matching the notebook's joint input+recurrence update order will not
fully close the gap (confirmed empirically: naive mid/high recurrence hurts).

*Assembly-calculus primitive.* ``project`` with ``target→target`` recurrence;
``learn_assembly`` until persistence ≥ 0.9 between consecutive rounds.

**H2 — Insufficient within-class exposure.**

*Statement.* The simple hierarchical trainer applies Hebbian updates on only
``n_rounds`` (default 5) examples per digit before advancing bias.  The ventral
stream learns from **many views** of the same object; each view is one
``project`` step, not an arbitrary extra epoch.

*Prediction.* Training on all ``n_examples`` per class (repeated exposure)
raises hierarchical accuracy toward ~79% without changing architecture.

*Primitive.* Repeated ``project`` (experience), not larger ``class_passes`` alone.

**H3 — Cross-class overlap at chance for confused pairs.**

*Statement.* Per-class accuracy in the simple model collapses for digits 2 and 8
(~8% and ~32% in parity golden).  Prototype overlap for these pairs remains
near the chance level ``k/n`` because feedforward features do not separate
curved-stroke confusions without stabilization or part structure.

*Prediction.* ``separate(stim_a, stim_b, area)`` overlap for 2-vs-8 stimuli
will remain elevated until part-based or completion-based representations are
introduced.

*Primitive.* ``separate`` as a **training diagnostic**, not merely a test.

**H4 — Readout–representation mismatch.**

*Statement.* Slot-WTA overlap on CLASS after a single ``HIGH→CLASS`` ``project``
discards graded synaptic evidence.  Sum readout over consolidated connectome
columns (PNAS Fig. 2) or ``fuzzy_readout`` on a digit lexicon aligns readout
with the stored assembly geometry.

*Prediction.* Prototype consolidation + connectome sum makes ``readout="class"``
match prototype accuracy (~64% simple, ~79% ventral).

*Primitives.* ``reciprocal_project``, ``wire_class_from_prototypes`` (consolidation
replay), ``fuzzy_readout``.


Part II — What the human ventral stream encodes (and we do not yet)
-----------------------------------------------------------------

The ventral pathway (Felleman & Van Essen; DiCarlo & Cox) builds a **nested
code** that assembly calculus can express only if we add areas and primitives
beyond bare feedforward ``project``:

1. **Parts and conjunctions.**
   Stroke fragments, loops, junctions — merged into wholes.
   *AC:* ``merge(LOOP, STROKE, PART)``; multiple part areas projecting jointly
   into HIGH.

2. **Invariance manifolds.**
   Same digit under translation, mild scale, stroke thickness — one IT assembly
   family, not one brittle prototype.
   *AC:* Multiple assemblies per class in a lexicon; ``fuzzy_readout`` with
   threshold; or ``associate`` views to a shared CLASS hub.

3. **Pattern completion.**
   Partial / occluded digits still evoke the full assembly.
   *AC:* ``pattern_complete(brain, partial_cue, HIGH, rounds)`` after training.

4. **Competitive selection among hypotheses.**
   When two categories overlap (2/8), cortex suppresses the leading hypothesis
   and re-evaluates.
   *AC:* **LRI-gated iterative readout** — after first ``fuzzy_readout`` winner,
   ``set_lri(HIGH)``, suppress winners, ``project`` again, second-pass readout
   (adaptation of ``ordered_recall`` mechanics to discrimination, not serial
   recall).

5. **Top-down gating.**
   Attention restricts which LOW pixels drive MID (figure-ground).
   *AC:* ``FiberCircuit`` masking LOW→MID projections; future work.

6. **Structured uncertainty.**
   Abstain when max overlap < threshold (improper parse).
   *AC:* ``fuzzy_readout(..., threshold=τ)`` returning ``None``.


Part III — Roadmap toward >95% (assembly-calculus strategies)
-------------------------------------------------------------

The COLT MNIST notebook itself reaches ~74% at n=5000 with a **single recurrent
layer** — evidence that 95% requires **representational enrichment**, not only
scale.  Ordered by expected impact:

**Tier A — Representation (target 88–92%).**

- ``learn_assembly`` with convergence criterion instead of fixed ``n_rounds``.
- **Multi-prototype lexicon**: K assemblies per digit (cluster HIGH snapshots);
  readout = max over lexicon entries (ensemble fuzzy readout).
- ``merge`` part streams: LOW_edge + LOW_blob → MID_parts → HIGH.
- Per-confusion-pair ``associate`` strengthening (curriculum on 2/8, 3/5, 4/9).

**Tier B — Completion and recurrence (target 92–95%).**

- ``pattern_complete`` on occluded / jittered MNIST at eval time — forces
  completion-capable assemblies during training via random pixel dropout.
- **Bidirectional ventral loop**: HIGH→MID top-down ``project`` during
  stabilization (predictive coding analogue); 2–3 rounds only at eval.
- ``consolidate`` / ``PathwayReplay`` overnight replay of HIGH→CLASS without
  resetting area→area weights (systems consolidation).

**Tier C — LRI and sequential primitives (target 95%+).**

Long-Range Inhibition (Dabagia et al. 2025) enables ``ordered_recall``: after
memorizing a sequence, LRI suppresses the current assembly so the next
self-projection activates the **successor** assembly along Hebbian bridges.

*Visual reinterpretation for discrimination (hypothesis H5):*

    Cue digit → HIGH assembly A
    → enable LRI on HIGH
    → self-project → successor assembly A' (within-class variation or sub-feature)
    → readout on ensemble {A, A'} vs lexicon
    → if margin < ε, suppress top lexicon neurons via LRI and recurse (race model)

This is **not** digit-sequence recall (reading "179" as three digits) but
**competitive cascade**: each LRI step suppresses the leading interpretation's
neurons and forces the network to expose the next-best assembly, improving
margin on ambiguous pairs.

Implementation: ``colt_mnist_lri_readout`` (Tier C / H5).

1. Train ventral stream as today.
2. Build digit lexicon with ``build_lexicon`` / ``prototypes_to_lexicon``.
3. At inference: ``fuzzy_readout`` → if top−second margin < ε, ``set_lri(HIGH)``,
   ``clear_refractory``, re-project from LOW, second readout excluding prior
   winner's neurons.

**Tier D — Architecture scale (diminishing returns alone).**

- n=5000+ exposure (paper golden ~73% single-layer — necessary but not sufficient).
- Larger k, tuned sparsity p — only after representational tiers A–C.

**Honest ceiling note.** 95% on full MNIST with sparse Hebbian assemblies and
no convolutions requires either (a) rich part-merge-completion structure above,
or (b) explicit invariance areas (e.g. position-normalized MID).  Pure deeper
feedforward ``project`` stacks have empirically saturated at ~64–79%.


Part IV — Primitive checklist for the next advanced module
----------------------------------------------------------

+-------------------+----------------------------------------+----------------+
| Primitive         | Visual role                            | Priority       |
+-------------------+----------------------------------------+----------------+
| learn_assembly    | Stable IT attractor per view           | A              |
| merge             | Part → whole (strokes → digit)         | A              |
| associate         | View ↔ category hub; confusion pairs   | A              |
| fuzzy_readout     | Lexicon + abstention                   | done           |
| pattern_complete  | Occlusion robustness                   | B              |
| consolidate       | Sleep replay HIGH→CLASS                | B              |
| set_lri + recall  | Competitive cascade readout            | C (implemented) |
| ScaffoldNetwork   | Faster sequence of feature probes      | C              |
| FiberCircuit      | Attention gating LOW→MID               | D              |
| separate          | Capacity audit during training         | diagnostic     |
+-------------------+----------------------------------------+----------------+


Part V — Cross-modal fusion (CLIP / LLaVA in Assembly Calculus)
----------------------------------------------------------------

Modern VLMs use **contrastive alignment** (CLIP/SigLIP) and **two-stage
curriculum** (LLaVA: language-first, then vision alignment).  We map these
faithfully onto AC primitives in ``cross_domain_assemblies``:

+----------------------+---------------------------+---------------------------+
| VLM concept          | AC implementation         | Area / route              |
+----------------------+---------------------------+---------------------------+
| Text encoder         | LANG orthographic patterns| LANG (784-d sparse codes) |
| Image encoder        | Ventral HIGH assemblies   | HIGH (IT)                 |
| Shared embedding     | SEMANTIC hub assemblies   | SEMANTIC (ATL-like)       |
| Stage-1 (LLaVA ph.1) | LANG→SEMANTIC associate   | freeze vision             |
| Stage-2 alignment    | HIGH+LANG→SEMANTIC bind   | multi-view associate      |
| Contrastive loss     | diagonal associate +      | hard negatives on         |
|                      | activate_assembly reset   | CONFUSED_PAIRS            |
| Multi-caption        | WORD_FORMS per digit      | ("two","2","TWO",…)       |
| Retrieval@1          | anchor overlap readout    | i2t / t2i metrics         |
| Classifier head      | fused readout vote        | semantic + connectome +   |
|                      |                           | multi-prototype           |
+----------------------+---------------------------+---------------------------+

Empirical target: fused visual accuracy should match or exceed ventral stream
(~79%) while language route stays near-perfect; agreement and retrieval@1 track
cross-modal binding quality.

Empirical finding (causal profile): fusion tracks connectome readout because
vision→SEMANTIC binding preserves stroke confusability.  Cross-modal work is
**gated** until representational metrics pass (see Part VI).


Part VI — Representational learning thesis (encoding before binding)
--------------------------------------------------------------------

**Master thesis.**  Digit accuracy is bounded by HIGH assembly geometry on
stroke-confusable pairs.  All visual readout routes (CLASS connectome, fuzzy
lexicon, SEMANTIC hub, fusion) estimate the same latent code; hub fusion cannot
exceed ventral information content.

**Corollaries.**

1. Fix **encoding** (parts, recurrence, completion, pair curriculum) before
   investing in cross-modal binding.
2. **Pair errors** (2/8, 3/5, …) are the scientific object — global mean accuracy
   hides the mechanism.
3. **Readout tricks** (LRI, fusion) help only at the margin when the leader
   hypothesis is low-confidence, not when HIGH is structurally wrong.
4. **No benchmark-maxxing** — oracle readout fallbacks and label-conditional
   routing are excluded; report honest inference paths only.

**Accuracy budget (scientific, not leaderboard).**

+------------------+--------+-------------------------------+
| Regime           | ~Acc   | Mechanism                     |
+------------------+--------+-------------------------------+
| Feedforward      | ~64%   | Transient codes (H1)          |
| Ventral exposure | ~79%   | Experience + connectome (H2,H4) |
| Recurrent IT     | ~81%   | HIGH→HIGH attractor (H1)      |
| Honest completion| target | Pair-gated pattern_complete   |
| Part-merge       | open   | merge stream (Tier A)         |
| Cross-modal hub  | gated  | H6: binding after separability|
+------------------+--------+-------------------------------+

**Research program (closed discovery loop).**

*Phase I — Characterize geometry* (``colt_mnist_geometry_panel``):

- 10×10 prototype overlap matrix on HIGH
- Per-pair connectome margin and ``separate()`` overlap diagnostics
- Confused vs non-confused digit accuracy split
- Occlusion recovery curves

*Phase II — Intervene on encoding* (``colt_mnist_representational``):

1. Recurrent ventral bundle as universal base (H1, H8)
2. Honest pair-gated pattern completion at inference — no oracle (H7)
3. Confused-pair HIGH→CLASS curriculum (H3, H9)
4. Train-time LOW occlusion for completion-capable assemblies (Tier B)
5. Re-open cross-modal binding only when vision→SEMANTIC > 70% or contrastive
   margin > 0.05 (H6 gate)

*Phase III — Readout after encoding moves pair metrics* (Tier C LRI).

*Phase IV — Cross-modal binding* (``cross_domain_assemblies``) when H6 gate opens.

*Phase V — Generative completeness* (``colt_mnist_regeneration_panel``,
``colt_mnist_attractor``, ``colt_mnist_forward_completion``):

- H11: HIGH attractor recovery via ``pattern_complete`` (~99% on explicit recurrent;
  prior ~0.5% report was E2 double-/k metric bug)
- H12: Digit-3 structured absence (Ember test; ``colt_mnist_absence``)
- H13: HIGH→LOW reciprocal regeneration (top-down decode)

*Phase VI — Spatial / CNN-inspired encoding* (``colt_mnist_spatial_ventral``):

- Local RF LOW→MID connectomes (``colt_mnist_spatial_connectome``)
- Digit-3 center-band curriculum during training
- Forward completion as additional honest inference primitive for partial LOW cues


Part VII — Generative completeness and engine gaps
--------------------------------------------------

**Empirical finding (updated after E2).**  The ~0.5% ``pattern_complete`` recovery
report was a **measurement artifact**: program code divided normalized overlap
by ``k`` again.  After E2 fixes, explicit recurrent MNIST recovers **~99%**
(H11).  Forward encoding from partial structured LOW remains the preferred
generative inference path for H7/H12 (occlusion-native, no HIGH self-mask).

**Interim program strategy:**

1. Forward completion for H7/H12 honest inference on confused pairs
2. Spatial RF connectomes at program layer
3. Digit-3 center_band curriculum
4. Regeneration panel (H11–H13) gates cross-modal binding with H6

**Engine upgrade roadmap** — see ``colt_mnist_engine_roadmap`` (E1–E8):

+----------------+----------+-----------------------------------------------+
| ID             | Priority | Status                                        |
+----------------+----------+-----------------------------------------------+
| E2 winner IDs  | P0       | **done** — ``Assembly.from_area``, ``_snap``  |
| E1 attractor   | P0       | partial — explicit MNIST OK; sparse TBD       |
| E6 plasticity  | P0       | partial — ``plasticity_mask`` on reinforce    |
| E8 learn_asm   | P0       | **done** — ``learn_assembly_from_pattern``    |
| E8 learn_asm   | P0       | area-level convergence loops                  |
| E4 top-down    | P1       | H13 LOW regeneration                          |
| E3 spatial RF  | P1       | engine-native connectome init                 |
| E5 partial drv | P1       | occlusion-native fill-in                      |
| E7 batch mixed | P2       | explicit+sparse parity                        |
+----------------+----------+-----------------------------------------------+

Work order: E2 → E1 → E6 → E8 → E4 → E3 → E5 → E7.


References
----------

- Papadimitriou et al., PNAS 2020 — projection, association, merge.
- Dabagia et al., Neural Computation 2025 — sequences, LRI, ordered recall.
- Mitropolsky & Papadimitriou, 2023 — fuzzy lexicon readout.
- Dabagia et al., COLT 2022 — MNIST.ipynb Hebbian classifier.


See Also
--------

``colt_mnist_visual_advanced`` — current Tier-0/1 implementations (ventral,
recurrent, fuzzy_lexicon).  ``colt_mnist_tier_a`` / ``colt_mnist_tier_b`` /
``colt_mnist_lri_readout`` — Tier A–C experiments.  ``cross_domain_assemblies``
— vision-language SEMANTIC hub.  ``cross_domain_profile`` — causal bottleneck
profiling.  ``colt_mnist_geometry_panel`` — Phase I pair/overlap diagnostics.
``colt_mnist_representational`` — Phase II encoding interventions (honest
completion, pair curriculum, occlusion training).  ``colt_mnist_evidence`` —
unified empirical ladder.  ``colt_mnist_regeneration_panel`` — H11–H13
generative completeness gates.  ``colt_mnist_forward_completion`` — partial
LOW→HIGH inference primitive.  ``colt_mnist_spatial_ventral`` — local RF
LOW→MID→HIGH spatial ventral stream.  ``colt_mnist_engine_roadmap`` — E1–E8
engine upgrade plan with acceptance tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

# ---------------------------------------------------------------------------
# Structured hypothesis registry (for programmatic access / future dashboards)
# ---------------------------------------------------------------------------


class HypothesisId(str, Enum):
    """Registered ventral-stream hypotheses."""

    ATTRACTOR_DEFICIENCY = "H1_attractor_deficiency"
    INSUFFICIENT_EXPOSURE = "H2_insufficient_exposure"
    CROSS_CLASS_OVERLAP = "H3_cross_class_overlap"
    READOUT_MISMATCH = "H4_readout_mismatch"
    LRI_CASCADE_DISCRIMINATION = "H5_lri_cascade_discrimination"
    SEPARABILITY_BEFORE_BINDING = "H6_separability_before_binding"
    HONEST_COMPLETION_GAIN = "H7_honest_completion_gain"
    RECURRENT_BASE_SUPERIORITY = "H8_recurrent_base_superiority"
    PART_MERGE_REDUCES_OVERLAP = "H9_part_merge_reduces_overlap"
    LRI_ON_CONFUSED_TOP2 = "H10_lri_on_confused_top2"
    ATTRACTOR_RECOVERY = "H11_attractor_recovery"
    DIGIT3_STRUCTURED_ABSENCE = "H12_digit3_structured_absence"
    LOW_REGENERATION = "H13_low_regeneration"


@dataclass(frozen=True)
class VentralHypothesis:
    """One falsifiable claim linking biology to assembly-calculus mechanism."""

    id: HypothesisId
    title: str
    statement: str
    ac_primitive: str
    empirical_status: Literal["confirmed", "partial", "open", "falsified"]
    simple_vs_recurrent: str


HYPOTHESES: tuple[VentralHypothesis, ...] = (
    VentralHypothesis(
        id=HypothesisId.ATTRACTOR_DEFICIENCY,
        title="Feedforward hierarchy lacks recurrent attractor stabilization",
        statement=(
            "Single-pass LOW→MID→HIGH produces transient codes; HIGH→HIGH "
            "recurrence in the notebook implements multi-step project convergence "
            "toward class-specific fixed points. Naive recurrence on two-layer "
            "stacks without joint input+recurrence updates does not recover "
            "full accuracy."
        ),
        ac_primitive="project + target→target recurrence; learn_assembly",
        empirical_status="confirmed",
        simple_vs_recurrent=(
            "Explains ~17pt gap: simple ~64% vs recurrent ~81% at n=50."
        ),
    ),
    VentralHypothesis(
        id=HypothesisId.INSUFFICIENT_EXPOSURE,
        title="Five rounds per class under-samples the ventral learning regime",
        statement=(
            "Hebbian plasticity during class blocks sees only n_rounds examples "
            "in the simple trainer; biological object learning integrates many "
            "views. Repeated exposure (all n_examples per class) raises "
            "hierarchical accuracy to ~79% without adding recurrence."
        ),
        ac_primitive="repeated project (experience)",
        empirical_status="confirmed",
        simple_vs_recurrent=(
            "Partially closes gap on hierarchical path; recurrence still "
            "helps single-pass inference."
        ),
    ),
    VentralHypothesis(
        id=HypothesisId.CROSS_CLASS_OVERLAP,
        title="Stroke-confusable digits share prototype support",
        statement=(
            "Digits 2 and 8 (and 3/8, 4/9) show collapsed per-class accuracy "
            "under feedforward codes because prototypes overlap near chance k/n. "
            "Part-structured or completion-based codes should reduce pairwise "
            "overlap measured by separate()."
        ),
        ac_primitive="separate; merge (parts)",
        empirical_status="partial",
        simple_vs_recurrent=(
            "Recurrent model improves global accuracy but pair errors persist "
            "without part structure."
        ),
    ),
    VentralHypothesis(
        id=HypothesisId.READOUT_MISMATCH,
        title="Slot-WTA readout discards graded connectome evidence",
        statement=(
            "CLASS slot overlap after one project loses synaptic weight "
            "structure; connectome sum readout and fuzzy_readout on a digit "
            "lexicon align decision with PNAS Fig. 2 population-sum readout."
        ),
        ac_primitive="reciprocal_project; fuzzy_readout; consolidation",
        empirical_status="confirmed",
        simple_vs_recurrent="Applies equally; closes class vs prototype gap.",
    ),
    VentralHypothesis(
        id=HypothesisId.LRI_CASCADE_DISCRIMINATION,
        title="LRI enables competitive cascade readout on ambiguous digits",
        statement=(
            "ordered_recall uses LRI to suppress the current assembly and "
            "activate successors along Hebbian bridges. Adapted to vision: "
            "after a low-margin fuzzy_readout, LRI suppresses the leading "
            "hypothesis neurons and re-project exposes the runner-up — a "
            "race model for 2/8-style confusions without backprop."
        ),
        ac_primitive="set_lri; ordered_recall (adapted); fuzzy_readout",
        empirical_status="partial",
        simple_vs_recurrent=(
            "Tier C ``colt_mnist_lri_readout`` implements competitive cascade; "
            "marginal gains on global accuracy, targeted for confused pairs."
        ),
    ),
    VentralHypothesis(
        id=HypothesisId.SEPARABILITY_BEFORE_BINDING,
        title="Representational separability precedes cross-modal binding",
        statement=(
            "Contrastive HIGH→SEMANTIC binding fails when pairwise digit overlap "
            "in HIGH remains near chance; language anchors are separable but "
            "vision codes are not.  Improving pair geometry without changing the "
            "hub protocol should raise contrastive margin and vision→SEMANTIC "
            "readout in lockstep."
        ),
        ac_primitive="associate; separate (diagnostic); geometry panel",
        empirical_status="open",
        simple_vs_recurrent=(
            "Cross-domain profile: fusion ≈ connectome; SEMANTIC margin ≈ 0."
        ),
    ),
    VentralHypothesis(
        id=HypothesisId.HONEST_COMPLETION_GAIN,
        title="Honest pair-gated pattern completion adds inference-time information",
        statement=(
            "When connectome top-two digits form a confused pair, recurrent "
            "pattern_complete on HIGH and connectome re-readout should improve "
            "pair accuracy without label-conditional oracle fallback."
        ),
        ac_primitive="pattern_complete; project (HIGH→HIGH)",
        empirical_status="open",
        simple_vs_recurrent="Tier B oracle fallback excluded in representational tier.",
    ),
    VentralHypothesis(
        id=HypothesisId.RECURRENT_BASE_SUPERIORITY,
        title="Recurrent ventral bundle is the correct encoding base",
        statement=(
            "Representational interventions should train and evaluate on "
            "enable_high_recurrence=True bundles (~81%) rather than feedforward "
            "ventral alone (~79%)."
        ),
        ac_primitive="project; HIGH→HIGH recurrence",
        empirical_status="partial",
        simple_vs_recurrent="Recurrent cortex ~81% vs ventral ~79% confirmed.",
    ),
    VentralHypothesis(
        id=HypothesisId.PART_MERGE_REDUCES_OVERLAP,
        title="Part-merge stream reduces confused-pair prototype overlap",
        statement=(
            "merge(TOP,BOT)→MID before HIGH should lower separate(2,8) overlap "
            "before global accuracy rises; merge-halves routing must be trained "
            "before HIGH readout."
        ),
        ac_primitive="merge; separate",
        empirical_status="falsified",
        simple_vs_recurrent=(
            "Merge-halves/grid merge raise confused-pair overlap to ~0.94 vs "
            "~0.23 recurrent; H9 diagnostic falsified (k-cap binding collapses geometry)."
        ),
    ),
    VentralHypothesis(
        id=HypothesisId.LRI_ON_CONFUSED_TOP2,
        title="LRI cascade when top-2 is a confused pair",
        statement=(
            "Connectome LRI should trigger when the leading two CLASS hypotheses "
            "are a stroke-confusable pair, improving confused-digit accuracy "
            "without easy-digit regression."
        ),
        ac_primitive="set_lri; connectome readout cascade",
        empirical_status="open",
        simple_vs_recurrent="Tier C LRI partial; pair-gated trigger under test.",
    ),
    VentralHypothesis(
        id=HypothesisId.ATTRACTOR_RECOVERY,
        title="HIGH assemblies are attractors recoverable by pattern_complete",
        statement=(
            "After convergent learn_assembly-style LOW+HIGH training and HIGH-only "
            "consolidation, pattern_complete at 50% masking should recover >=50% "
            "overlap for confused digits (especially 3). Default recurrent training "
            "without consolidation yields ~1% recovery."
        ),
        ac_primitive="learn_assembly; pattern_complete; project (HIGH→HIGH)",
        empirical_status="open",
        simple_vs_recurrent="Attractor bundle targets recovery; baseline recurrent ~1%.",
    ),
    VentralHypothesis(
        id=HypothesisId.DIGIT3_STRUCTURED_ABSENCE,
        title="Digit 3 survives structured absence (Ember test)",
        statement=(
            "When digit-3-distinctive structure is removed (top half, center band), "
            "the circuit must still classify or complete to digit 3. Random pixel "
            "dropout is insufficient; structured absence is the generative test."
        ),
        ac_primitive="pattern_complete; merge; structured absence curriculum",
        empirical_status="open",
        simple_vs_recurrent="Regeneration panel gates cross-modal binding with H6.",
    ),
    VentralHypothesis(
        id=HypothesisId.LOW_REGENERATION,
        title="Top-down HIGH→LOW reciprocal projection regenerates pixel assemblies",
        statement=(
            "A trained HIGH→LOW connectome should overlap the original LOW assembly "
            "after projecting from a consolidated HIGH attractor — the AC-native "
            "decoder analogue to GAN/diffusion reverse paths."
        ),
        ac_primitive="reciprocal_project; project (HIGH→LOW)",
        empirical_status="open",
        simple_vs_recurrent="Attractor training includes top-down consolidation.",
    ),
)


class ResearchTier(str, Enum):
    """Roadmap tiers toward >95% MNIST accuracy."""

    A_REPRESENTATION = "A_representation"      # target 88–92%
    B_COMPLETION = "B_completion"              # target 92–95%
    C_LRI_CASCADE = "C_lri_cascade"            # target 95%+
    D_SCALE_ATTENTION = "D_scale_attention"    # necessary, not sufficient


@dataclass(frozen=True)
class ResearchDirection:
    """One assembly-calculus strategy on the roadmap."""

    tier: ResearchTier
    name: str
    ac_primitives: tuple[str, ...]
    target_accuracy: str
    visual_analogue: str
    module: str | None = None


ROADMAP_TO_95: tuple[ResearchDirection, ...] = (
    ResearchDirection(
        tier=ResearchTier.A_REPRESENTATION,
        name="Convergent learn_assembly per digit",
        ac_primitives=("learn_assembly", "project"),
        target_accuracy="88–92%",
        visual_analogue="IT attractor stabilization across views",
        module="colt_mnist_attractor",
    ),
    ResearchDirection(
        tier=ResearchTier.A_REPRESENTATION,
        name="Multi-prototype lexicon (K assemblies per digit)",
        ac_primitives=("build_lexicon", "fuzzy_readout", "merge"),
        target_accuracy="88–92%",
        visual_analogue="View/manifold family in IT",
        module="colt_mnist_tier_a",
    ),
    ResearchDirection(
        tier=ResearchTier.A_REPRESENTATION,
        name="Part-merge stream (strokes → digit)",
        ac_primitives=("merge", "project", "associate"),
        target_accuracy="88–92%",
        visual_analogue="V4 part decomposition",
        module="colt_mnist_tier_a",
    ),
    ResearchDirection(
        tier=ResearchTier.A_REPRESENTATION,
        name="Geometry panel (pair overlap diagnostics)",
        ac_primitives=("separate", "overlap", "pattern_complete"),
        target_accuracy="diagnostic",
        visual_analogue="Representational adequacy audit before binding",
        module="colt_mnist_geometry_panel",
    ),
    ResearchDirection(
        tier=ResearchTier.B_COMPLETION,
        name="Generative completeness panel (H11–H13)",
        ac_primitives=("pattern_complete", "separate", "reciprocal_project"),
        target_accuracy="diagnostic",
        visual_analogue="Attractor recovery + Ember digit-3 absence test",
        module="colt_mnist_regeneration_panel",
    ),
    ResearchDirection(
        tier=ResearchTier.A_REPRESENTATION,
        name="Spatial RF ventral stream (local LOW→MID fibers)",
        ac_primitives=("project", "local_rf", "structured_absence"),
        target_accuracy="≥ ventral - 2% confused digits",
        visual_analogue="V1 local receptive fields / CNN first layer",
        module="colt_mnist_spatial_ventral",
    ),
    ResearchDirection(
        tier=ResearchTier.B_COMPLETION,
        name="Engine upgrade roadmap (E1–E8)",
        ac_primitives=("pattern_complete", "learn_assembly", "reciprocal_project"),
        target_accuracy="E1: pattern_complete recovery ≥50%",
        visual_analogue="Attractor persistence + top-down decode in core engine",
        module="colt_mnist_engine_roadmap",
    ),
    ResearchDirection(
        tier=ResearchTier.A_REPRESENTATION,
        name="Honest pair-gated completion + pair curriculum",
        ac_primitives=("pattern_complete", "reinforce_connectome", "project"),
        target_accuracy="82%+ honest",
        visual_analogue="Attractor completion without oracle readout",
        module="colt_mnist_representational",
    ),
    ResearchDirection(
        tier=ResearchTier.B_COMPLETION,
        name="Pattern completion on occluded digits",
        ac_primitives=("pattern_complete", "project"),
        target_accuracy="92–95%",
        visual_analogue="Perceptual filling-in",
        module="colt_mnist_tier_b",
    ),
    ResearchDirection(
        tier=ResearchTier.B_COMPLETION,
        name="Systems consolidation replay",
        ac_primitives=("consolidate", "PathwayReplay", "reciprocal_project"),
        target_accuracy="92–95%",
        visual_analogue="Sleep replay of category pathways",
        module="colt_mnist_tier_b",
    ),
    ResearchDirection(
        tier=ResearchTier.C_LRI_CASCADE,
        name="LRI competitive cascade readout",
        ac_primitives=("set_lri", "fuzzy_readout", "project", "ordered_recall"),
        target_accuracy="95%+",
        visual_analogue="Hypothesis suppression / race models in decision cortex",
        module="colt_mnist_lri_readout",
    ),
    ResearchDirection(
        tier=ResearchTier.C_LRI_CASCADE,
        name="Cross-domain vision-language semantic hub",
        ac_primitives=(
            "associate", "reciprocal_project", "activate_assembly",
            "fuzzy_readout", "project",
        ),
        target_accuracy="82%+ fused (semantic + ventral readout)",
        visual_analogue="CLIP contrastive + LLaVA two-stage + ATL hub",
        module="cross_domain_assemblies",
    ),
    ResearchDirection(
        tier=ResearchTier.C_LRI_CASCADE,
        name="Scaffold areas for feature-probe sequences",
        ac_primitives=("ScaffoldNetwork", "sequence_memorize", "ordered_recall"),
        target_accuracy="95%+",
        visual_analogue="Sequential saccades / feature sampling",
    ),
    ResearchDirection(
        tier=ResearchTier.D_SCALE_ATTENTION,
        name="Paper-scale exposure + FiberCircuit gating",
        ac_primitives=("project", "FiberCircuit"),
        target_accuracy="73% alone; +tiers A–C",
        visual_analogue="More experience + spatial attention",
    ),
)


@dataclass(frozen=True)
class VentralInformationStructure:
    """Information the ventral stream encodes; assembly area mapping."""

    name: str
    biological_locus: str
    proposed_area: str
    ac_primitives: tuple[str, ...]
    encoded_in_current_models: bool


VENTRAL_CODE_STRUCTURES: tuple[VentralInformationStructure, ...] = (
    VentralInformationStructure(
        name="Sparse edge / pixel field",
        biological_locus="V1",
        proposed_area="LOW",
        ac_primitives=("project",),
        encoded_in_current_models=True,
    ),
    VentralInformationStructure(
        name="Local features / contours",
        biological_locus="V2/V4",
        proposed_area="MID",
        ac_primitives=("project", "merge"),
        encoded_in_current_models=True,
    ),
    VentralInformationStructure(
        name="Object-level assembly",
        biological_locus="IT",
        proposed_area="HIGH",
        ac_primitives=("project", "learn_assembly", "pattern_complete"),
        encoded_in_current_models=True,
    ),
    VentralInformationStructure(
        name="Category decision index",
        biological_locus="PFC / category areas",
        proposed_area="CLASS",
        ac_primitives=("reciprocal_project", "fuzzy_readout"),
        encoded_in_current_models=True,
    ),
    VentralInformationStructure(
        name="Part/stroke conjunctions",
        biological_locus="V4 parts",
        proposed_area="LOW_TOP + LOW_BOT -> MID",
        ac_primitives=("merge", "associate"),
        encoded_in_current_models=True,
    ),
    VentralInformationStructure(
        name="View / exemplar manifolds",
        biological_locus="IT subpopulations",
        proposed_area="HIGH lexicon K>1",
        ac_primitives=("build_lexicon", "associate", "fuzzy_readout"),
        encoded_in_current_models=True,
    ),
    VentralInformationStructure(
        name="Competitive hypothesis queue",
        biological_locus="Decision / LIP-like",
        proposed_area="HIGH + LRI",
        ac_primitives=("set_lri", "ordered_recall", "fuzzy_readout"),
        encoded_in_current_models=True,
    ),
    VentralInformationStructure(
        name="Cross-modal semantic hub",
        biological_locus="ATL / semantic association cortex",
        proposed_area="SEMANTIC (HIGH + LANG associate)",
        ac_primitives=(
            "associate", "reciprocal_project", "activate_assembly", "fuzzy_readout",
        ),
        encoded_in_current_models=True,
    ),
    VentralInformationStructure(
        name="Top-down attention mask",
        biological_locus="Parietal → V1 feedback",
        proposed_area="FiberCircuit on LOW→MID",
        ac_primitives=("FiberCircuit", "project"),
        encoded_in_current_models=False,
    ),
)


def hypothesis_by_id(hypothesis_id: HypothesisId) -> VentralHypothesis:
    """Return a registered hypothesis by id."""
    for h in HYPOTHESES:
        if h.id == hypothesis_id:
            return h
    raise KeyError(hypothesis_id)


def roadmap_for_tier(tier: ResearchTier) -> tuple[ResearchDirection, ...]:
    """Filter roadmap directions by research tier."""
    return tuple(r for r in ROADMAP_TO_95 if r.tier == tier)


def structures_not_yet_implemented() -> tuple[VentralInformationStructure, ...]:
    """Ventral codes required for >95% that current models lack."""
    return tuple(s for s in VENTRAL_CODE_STRUCTURES if not s.encoded_in_current_models)


def experiment_modules() -> dict[str, str]:
    """Map experiment ids to implementing modules (for evidence dashboards)."""
    return {
        "baseline_hierarchical": "colt_mnist_hierarchical",
        "advanced_ventral": "colt_mnist_visual_advanced",
        "advanced_recurrent": "colt_mnist_visual_advanced",
        "advanced_fuzzy_lexicon": "colt_mnist_visual_advanced",
        "tier_a_multi_prototype": "colt_mnist_tier_a",
        "tier_a_merge_halves": "colt_mnist_tier_a",
        "tier_b_pattern_complete": "colt_mnist_tier_b",
        "tier_b_consolidation": "colt_mnist_tier_b",
        "tier_c_lri_cascade": "colt_mnist_lri_readout",
        "cross_domain_vision_language": "cross_domain_assemblies",
        "cross_domain_profile": "cross_domain_profile",
        "geometry_panel": "colt_mnist_geometry_panel",
        "representational_honest_completion": "colt_mnist_representational",
        "representational_pair_curriculum": "colt_mnist_representational",
        "representational_recurrent_base": "colt_mnist_representational",
        "representational_occlusion_train": "colt_mnist_representational",
        "representational_stack": "colt_mnist_representational",
        "attractor_recurrent": "colt_mnist_attractor",
        "regeneration_panel": "colt_mnist_regeneration_panel",
        "forward_completion": "colt_mnist_forward_completion",
        "spatial_ventral": "colt_mnist_spatial_ventral",
        "engine_roadmap": "colt_mnist_engine_roadmap",
        "evidence_suite": "colt_mnist_evidence",
    }


def simple_vs_recurrent_summary() -> str:
    """One-paragraph summary of the empirical gap and its mechanistic explanation."""
    return (
        "Simple hierarchical MNIST (~64%) applies assembly-calculus project "
        "once per layer (LOW->MID->HIGH) with only five Hebbian updates per "
        "digit class before cross-class bias. Recurrent cortex (~81%) and the "
        "COLT notebook instead run a HIGH->HIGH attractor loop — mathematically "
        "multiple project rounds with shared input — which stabilizes IT-like "
        "assemblies. Advanced ventral stream (~79%) closes much of the gap by "
        "repeated exposure (many project steps per class) and principled readout "
        "without recurrence, confirming H1 and H2. Neither path yet encodes parts, "
        "completion, or LRI cascade discrimination (H5); those are the plausible "
        "assembly-calculus routes toward >95%."
    )
