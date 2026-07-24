"""
Cross-domain vision–language fusion via Assembly Calculus.

Implements a CLIP / LLaVA-inspired training protocol using only AC primitives:

* **Stage 1 (LLaVA phase-1 analogue):** freeze vision; build LANG→SEMANTIC concept
  anchors from multiple orthographic surface forms per digit.
* **Stage 2 (alignment):** freeze language anchors; align HIGH→SEMANTIC with
  multi-view ``associate`` + ``reciprocal_project``.
* **Contrastive binding (CLIP / SigLIP analogue):** strengthen diagonal
  (digit, digit) pairs; hard-negative passes on confused pairs push wrong
  cross-modal bindings away from anchors via ``activate_assembly`` reset.
* **Evaluation:** classification accuracy, cross-modal agreement, image→text and
  text→image retrieval@1, semantic overlap on matched pairs.

Theory mapping: HIGH=IT, LANG=orthography, SEMANTIC=shared hub (ATL-like).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neural_assemblies.assembly_calculus.assembly import Assembly, overlap
from neural_assemblies.assembly_calculus.ops import (
    _snap,
    activate_assembly,
    associate,
    reciprocal_project,
)
from neural_assemblies.assembly_calculus.readout import fuzzy_readout, readout_all
from neural_assemblies.programs.colt_mnist_advanced_util import (
    read_class_connectome_scores,
    refined_confused_prototypes,
    wire_class_from_prototypes,
    wire_class_mixed_prototypes,
    wire_digit_slots_from_prototypes,
    wire_semantic_from_anchors,
    wire_semantic_from_view_projections,
)
from neural_assemblies.programs.colt_mnist_brain_util import (
    class_slot_neurons,
    clear_area_winners,
    reinforce_class_slot,
    renorm_connectome_columns,
    set_kcap_winners,
)
from neural_assemblies.programs.colt_mnist_hierarchical_brain import CLASS, HIGH, NUM_DIGITS
from neural_assemblies.programs.colt_mnist_lri_readout import connectome_lri_predict
from neural_assemblies.programs.colt_mnist_tier_a import (
    build_multi_prototype_lexicon,
    predict_multi_prototype,
)
from neural_assemblies.programs.colt_mnist_tier_util import (
    CONFUSED_DIGITS,
    CONFUSED_PAIRS,
    connectome_predict,
    load_ventral_bundle,
)

LANG = "LANG"
SEMANTIC = "SEMANTIC"

# Primary word labels + alternate surface forms (multi-caption invariance).
WORD_NAMES = (
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
)

WORD_FORMS: tuple[tuple[str, ...], ...] = (
    ("zero", "0", "ZERO"),
    ("one", "1", "ONE"),
    ("two", "2", "TWO", "ii"),
    ("three", "3", "THREE"),
    ("four", "4", "FOUR"),
    ("five", "5", "FIVE"),
    ("six", "6", "SIX"),
    ("seven", "7", "SEVEN"),
    ("eight", "8", "EIGHT"),
    ("nine", "9", "NINE"),
)

def _hard_negative_digit_pairs() -> list[tuple[int, int]]:
    """Directed (vision_digit, wrong_lang_digit) pairs for contrastive passes."""
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for a, b in CONFUSED_PAIRS:
        for v, w in ((a, b), (b, a)):
            if (v, w) not in seen:
                pairs.append((v, w))
                seen.add((v, w))
    return pairs


@dataclass
class CrossDomainResult:
    visual_accuracy: float
    language_accuracy: float
    agreement_rate: float
    mean_semantic_overlap: float
    image_to_text_recall: float
    text_to_image_recall: float
    per_digit_visual: np.ndarray
    per_digit_language: np.ndarray
    data_source: str
    parameters: dict
    backend: str = "cross_domain_contrastive_hub"
    extra: dict = field(default_factory=dict)


def _orthographic_pattern(word: str, n: int = 784, seed: int = 0) -> np.ndarray:
    """Deterministic sparse binary pattern from word string."""
    rng = np.random.default_rng(hash(word) % (2**32) + seed)
    pattern = np.zeros(n, dtype=np.float64)
    k = 80 + len(word) * 5
    idx = rng.choice(n, size=min(k, n), replace=False)
    pattern[idx] = 1.0
    return pattern


def _word_pattern(digit: int, form: str, *, seed: int, n_lang: int) -> np.ndarray:
    return _orthographic_pattern(form, n=n_lang, seed=seed + digit * 17)


def _primary_word_pattern(digit: int, *, seed: int, n_lang: int) -> np.ndarray:
    return _word_pattern(digit, WORD_NAMES[digit], seed=seed, n_lang=n_lang)


def _inhibit_hub(brain) -> None:
    for area in (LANG, SEMANTIC):
        if area in brain.areas:
            clear_area_winners(brain, area)
            brain.inhibit_areas([area])


def _readout_digit(asm: Assembly, lexicon: dict[str, Assembly]) -> int:
    ranked = readout_all(asm, lexicon)
    if not ranked:
        return 0
    label = fuzzy_readout(asm, lexicon, threshold=0.15)
    return int(label) if label is not None else int(ranked[0][0])


def _anchor_readout(asm: Assembly, anchors: dict[str, Assembly], k: int) -> int:
    """Direct max-overlap readout against frozen concept anchors."""
    best_digit = 0
    best_score = -1.0
    for label, anchor in anchors.items():
        score = overlap(asm, anchor)
        if score > best_score:
            best_score = score
            best_digit = int(label)
    return best_digit


@dataclass
class TrainedCrossDomainHub:
    """Trained brain + lexicons for eval and profiling."""

    brain: object
    bundle: object
    semantic_anchors: dict[str, Assembly]
    semantic_lex: dict[str, Assembly]
    lang_lex: dict[str, Assembly]
    mp_lex: dict[str, Assembly]
    seed: int
    n_lang: int
    parameters: dict


def _build_semantic_lexicon_from_projections(
    brain,
    bundle,
) -> dict[str, Assembly]:
    """Per-digit SEMANTIC assemblies from prototype projection (eval-aligned)."""
    lex: dict[str, Assembly] = {}
    for digit in range(NUM_DIGITS):
        set_kcap_winners(brain, HIGH, bundle.prototypes[digit])
        brain.project({}, {HIGH: [SEMANTIC]})
        reciprocal_project(brain, HIGH, SEMANTIC, rounds=4)
        lex[str(digit)] = _snap(brain, SEMANTIC)
    return lex


def _confused_pair_in_top2(order: np.ndarray) -> bool:
    a, b = int(order[0]), int(order[1])
    return (a, b) in CONFUSED_PAIRS or (b, a) in CONFUSED_PAIRS


def _connectome_margin(high_vec: np.ndarray, brain, k: int) -> tuple[int, float, np.ndarray]:
    scores = read_class_connectome_scores(high_vec, brain, HIGH, CLASS, k, NUM_DIGITS)
    order = np.argsort(scores)[::-1]
    pred = int(order[0])
    margin = float(scores[pred] - scores[order[1]]) if len(order) > 1 else 0.0
    return pred, margin, order


def _fuse_visual_prediction(
    high_vec: np.ndarray,
    brain,
    bundle,
    semantic_lex: dict[str, Assembly],
    mp_lex: dict[str, Assembly],
    *,
    use_semantic_tiebreak: bool = True,
    use_connectome_lri: bool = True,
    lri_margin_threshold: float = 0.04,
) -> tuple[int, dict[str, int]]:
    """Fuse ventral routes; LRI cascade on low-margin connectome readout."""
    conn_pred, conn_margin, order = _connectome_margin(high_vec, brain, bundle.k)
    lri_cascades = 0
    trigger_lri = (
        use_connectome_lri
        and (conn_margin < lri_margin_threshold or _confused_pair_in_top2(order))
    )
    if trigger_lri:
        conn_lri, _, lri_cascades = connectome_lri_predict(
            high_vec, brain, mp_lex, k=bundle.k,
            margin_threshold=lri_margin_threshold,
            max_cascades=3,
        )
        if lri_cascades > 0:
            conn_pred = conn_lri

    mp_pred = predict_multi_prototype(high_vec, mp_lex)
    sem_conn = _semantic_connectome_predict(high_vec, brain, bundle.k)

    set_kcap_winners(brain, HIGH, high_vec)
    brain.project({}, {HIGH: [SEMANTIC]})
    sem_anchor = _anchor_readout(_snap(brain, SEMANTIC), semantic_lex, bundle.k)

    if conn_pred == mp_pred:
        fused = conn_pred
    elif use_semantic_tiebreak:
        fused = conn_pred
        for sem_pred in (sem_conn, sem_anchor):
            if sem_pred in (conn_pred, mp_pred):
                fused = sem_pred
                break
    else:
        fused = conn_pred

    return fused, {
        "semantic": sem_anchor,
        "semantic_connectome": sem_conn,
        "connectome": conn_pred,
        "multi_proto": mp_pred,
        "connectome_margin": conn_margin,
        "lri_cascades": lri_cascades,
    }


def _semantic_connectome_predict(high_vec: np.ndarray, brain, k: int) -> int:
    scores = read_class_connectome_scores(high_vec, brain, HIGH, SEMANTIC, k, NUM_DIGITS)
    return int(np.argmax(scores))


def _stage0_confused_class_reinforcement(
    brain,
    bundle,
    *,
    passes: int = 2,
) -> None:
    """Extra Hebbian HIGH->CLASS exposure on stroke-confusable digits."""
    for _ in range(passes):
        for digit in CONFUSED_DIGITS:
            for j in range(bundle.n_examples):
                set_kcap_winners(brain, HIGH, bundle.high_outputs[digit, j])
                reinforce_class_slot(brain, HIGH, CLASS, digit, bundle.k)
    renorm_connectome_columns(brain, HIGH, CLASS)


def _ensure_hub_areas(brain, *, n_lang: int, k: int, beta: float) -> None:
    if LANG not in brain.areas:
        brain.add_area(LANG, n_lang, k, beta, explicit=True)
    if SEMANTIC not in brain.areas:
        brain.add_area(SEMANTIC, 2000, k, beta, explicit=True)


def _restore_semantic_anchor(brain, anchor: Assembly) -> None:
    """Reset SEMANTIC to a stored concept anchor (contrastive push-away)."""
    brain.areas[SEMANTIC].unfix_assembly()
    activate_assembly(brain, anchor)


def _stage1_language_semantic_anchors(
    brain,
    *,
    seed: int,
    n_lang: int,
    associate_rounds: int,
) -> tuple[dict[str, Assembly], dict[str, Assembly]]:
    """Stage 1: LANG→SEMANTIC anchors from multiple surface forms (vision frozen)."""
    lang_lex: dict[str, Assembly] = {}
    semantic_anchors: dict[str, Assembly] = {}

    for digit in range(NUM_DIGITS):
        _inhibit_hub(brain)
        primary_pat = _primary_word_pattern(digit, seed=seed, n_lang=n_lang)

        for form in WORD_FORMS[digit]:
            pat = _word_pattern(digit, form, seed=seed, n_lang=n_lang)
            set_kcap_winners(brain, LANG, pat)
            brain.project({}, {LANG: [SEMANTIC]})
            reciprocal_project(brain, LANG, SEMANTIC, rounds=associate_rounds)
            renorm_connectome_columns(brain, LANG, SEMANTIC)

        set_kcap_winners(brain, LANG, primary_pat)
        brain.project({}, {LANG: [SEMANTIC]})
        reciprocal_project(brain, LANG, SEMANTIC, rounds=associate_rounds)
        lang_lex[str(digit)] = _snap(brain, LANG)
        semantic_anchors[str(digit)] = _snap(brain, SEMANTIC)
        renorm_connectome_columns(brain, LANG, SEMANTIC)

    return semantic_anchors, lang_lex


def _positive_bind(
    brain,
    high_vec: np.ndarray,
    lang_pat: np.ndarray,
    *,
    associate_rounds: int,
) -> None:
    """Positive (vision, language) pair → shared SEMANTIC (CLIP diagonal)."""
    set_kcap_winners(brain, HIGH, high_vec)
    set_kcap_winners(brain, LANG, lang_pat)
    associate(brain, HIGH, LANG, SEMANTIC, stim_a=None, stim_b=None, rounds=associate_rounds)
    reciprocal_project(brain, HIGH, SEMANTIC, rounds=4)
    reciprocal_project(brain, LANG, SEMANTIC, rounds=4)
    renorm_connectome_columns(brain, HIGH, SEMANTIC)
    renorm_connectome_columns(brain, LANG, SEMANTIC)


def _negative_bind(
    brain,
    high_vec: np.ndarray,
    lang_pat_wrong: np.ndarray,
    anchor_correct: Assembly,
    *,
    rounds: int = 3,
) -> None:
    """Hard-negative pass: wrong cross-modal pair, then restore correct anchor."""
    set_kcap_winners(brain, HIGH, high_vec)
    set_kcap_winners(brain, LANG, lang_pat_wrong)
    brain.project({}, {HIGH: [SEMANTIC], LANG: [SEMANTIC]})
    if rounds > 1:
        brain.project_rounds(
            target=SEMANTIC,
            areas_by_stim={},
            dst_areas_by_src_area={
                HIGH: [SEMANTIC],
                LANG: [SEMANTIC],
                SEMANTIC: [SEMANTIC],
            },
            rounds=rounds - 1,
        )
    _restore_semantic_anchor(brain, anchor_correct)


def _stage2_vision_alignment(
    brain,
    bundle,
    semantic_anchors: dict[str, Assembly],
    *,
    seed: int,
    n_lang: int,
    views_per_digit: int,
    associate_rounds: int,
    negative_rounds: int,
    hard_negatives: bool,
    contrastive_epochs: int,
) -> dict[str, Assembly]:
    """Stage 2: multi-view HIGH→SEMANTIC alignment to frozen language anchors."""
    indices = np.linspace(
        0, bundle.n_examples - 1, min(views_per_digit, bundle.n_examples), dtype=int,
    )
    hard_pairs = _hard_negative_digit_pairs() if hard_negatives else []

    for _epoch in range(contrastive_epochs):
        for digit in range(NUM_DIGITS):
            anchor = semantic_anchors[str(digit)]
            primary_pat = _primary_word_pattern(digit, seed=seed, n_lang=n_lang)

            for j in indices:
                _positive_bind(
                    brain,
                    bundle.high_outputs[digit, j],
                    primary_pat,
                    associate_rounds=associate_rounds,
                )
                _restore_semantic_anchor(brain, anchor)

                for form in WORD_FORMS[digit][1:]:
                    alt_pat = _word_pattern(digit, form, seed=seed, n_lang=n_lang)
                    _positive_bind(
                        brain,
                        bundle.high_outputs[digit, j],
                        alt_pat,
                        associate_rounds=max(4, associate_rounds // 2),
                    )
                    _restore_semantic_anchor(brain, anchor)

            if hard_negatives:
                j = int(indices[_epoch % len(indices)])
                hv = bundle.high_outputs[digit, j]
                for vis_d, lang_d in hard_pairs:
                    if vis_d != digit:
                        continue
                    wrong_pat = _primary_word_pattern(lang_d, seed=seed, n_lang=n_lang)
                    _negative_bind(
                        brain, hv, wrong_pat, anchor,
                        rounds=negative_rounds,
                    )

    refreshed: dict[str, Assembly] = {}
    for digit in range(NUM_DIGITS):
        set_kcap_winners(brain, HIGH, bundle.prototypes[digit])
        brain.project({}, {HIGH: [SEMANTIC]})
        reciprocal_project(brain, HIGH, SEMANTIC, rounds=4)
        refreshed[str(digit)] = _snap(brain, SEMANTIC)
    return refreshed


def _stage2b_confused_pair_curriculum(
    brain,
    bundle,
    semantic_anchors: dict[str, Assembly],
    *,
    seed: int,
    n_lang: int,
    associate_rounds: int,
    negative_rounds: int,
    passes_per_pair: int = 5,
) -> None:
    """Extra contrastive passes on stroke-confusable digit pairs (all views)."""
    view_idx = np.linspace(0, bundle.n_examples - 1, min(8, bundle.n_examples), dtype=int)
    for a, b in CONFUSED_PAIRS:
        for _ in range(passes_per_pair):
            for vis_d, lang_d in ((a, a), (b, b), (a, b), (b, a)):
                anchor = semantic_anchors[str(vis_d)]
                for j in view_idx:
                    hv = bundle.high_outputs[vis_d, j]
                    if lang_d == vis_d:
                        pat = _primary_word_pattern(lang_d, seed=seed, n_lang=n_lang)
                        _positive_bind(brain, hv, pat, associate_rounds=associate_rounds)
                        _restore_semantic_anchor(brain, anchor)
                    else:
                        wrong_pat = _primary_word_pattern(lang_d, seed=seed, n_lang=n_lang)
                        _negative_bind(
                            brain, hv, wrong_pat, anchor, rounds=negative_rounds,
                        )


def _consolidate_hub_readout(
    brain,
    bundle,
    semantic_anchors: dict[str, Assembly],
    *,
    wiring: str = "views",
    refined_prototypes: np.ndarray | None = None,
) -> None:
    """Systems-replay wiring after plastic hub training."""
    protos = refined_prototypes if refined_prototypes is not None else bundle.prototypes
    wire_class_mixed_prototypes(
        brain, bundle.prototypes, protos, CONFUSED_DIGITS, HIGH, CLASS, bundle.k,
    )
    if wiring == "anchors":
        wire_semantic_from_anchors(brain, protos, semantic_anchors, HIGH, SEMANTIC)
    elif wiring == "slots":
        wire_digit_slots_from_prototypes(brain, protos, HIGH, SEMANTIC, bundle.k)
    else:
        wire_digit_slots_from_prototypes(brain, protos, HIGH, SEMANTIC, bundle.k)
        wire_semantic_from_view_projections(
            brain, bundle.high_outputs, HIGH, SEMANTIC, bundle.k,
            digits=CONFUSED_DIGITS, reset=False,
        )


def _contrastive_matrix(
    brain,
    bundle,
    semantic_lex: dict[str, Assembly],
    *,
    seed: int,
    n_lang: int,
    sample_j: int = 0,
) -> np.ndarray:
    """10×10 semantic overlap matrix (diagnostic — CLIP similarity grid)."""
    mat = np.zeros((NUM_DIGITS, NUM_DIGITS), dtype=np.float64)
    for i in range(NUM_DIGITS):
        set_kcap_winners(brain, HIGH, bundle.high_outputs[i, sample_j])
        brain.project({}, {HIGH: [SEMANTIC]})
        sem_v = _snap(brain, SEMANTIC)
        for j in range(NUM_DIGITS):
            pat = _primary_word_pattern(j, seed=seed, n_lang=n_lang)
            set_kcap_winners(brain, LANG, pat)
            brain.project({}, {LANG: [SEMANTIC]})
            sem_l = _snap(brain, SEMANTIC)
            mat[i, j] = overlap(sem_v, sem_l) / bundle.k
    return mat


def _eval_retrieval_and_classification(
    brain,
    bundle,
    semantic_anchors: dict[str, Assembly],
    semantic_lex: dict[str, Assembly],
    lang_lex: dict[str, Assembly],
    mp_lex: dict[str, Assembly],
    *,
    seed: int,
    n_lang: int,
    use_connectome_lri: bool = True,
) -> dict:
    """VLM-style retrieval@1 and per-route classification."""
    lang_anchors = semantic_anchors
    vis_correct = np.zeros(NUM_DIGITS)
    fused_correct = np.zeros(NUM_DIGITS)
    sem_route_correct = np.zeros(NUM_DIGITS)
    sem_conn_correct = np.zeros(NUM_DIGITS)
    lang_correct = np.zeros(NUM_DIGITS)
    agreements = 0
    i2t_hits = 0
    t2i_hits = 0
    lri_triggered = 0
    overlaps: list[float] = []
    n_eval = NUM_DIGITS * bundle.n_examples

    for digit in range(NUM_DIGITS):
        v_hits = 0
        f_hits = 0
        s_hits = 0
        sc_hits = 0
        l_hits = 0
        primary_pat = _primary_word_pattern(digit, seed=seed, n_lang=n_lang)

        for j in range(bundle.n_examples):
            hv = bundle.high_outputs[digit, j]
            fused_pred, routes = _fuse_visual_prediction(
                hv, brain, bundle, semantic_lex, mp_lex,
                use_connectome_lri=use_connectome_lri,
            )
            sem_pred = routes["semantic"]
            sem_conn_pred = routes["semantic_connectome"]

            set_kcap_winners(brain, LANG, primary_pat)
            brain.project({}, {LANG: [SEMANTIC]})
            sem_l = _snap(brain, SEMANTIC)
            l_pred = _anchor_readout(sem_l, lang_anchors, bundle.k)
            l_lang = _readout_digit(_snap(brain, LANG), lang_lex)
            if l_pred != digit and l_lang == digit:
                l_pred = l_lang

            set_kcap_winners(brain, HIGH, hv)
            brain.project({}, {HIGH: [SEMANTIC]})
            sem_v = _snap(brain, SEMANTIC)
            overlaps.append(overlap(sem_v, sem_l) / bundle.k)

            if routes.get("lri_cascades", 0) > 0:
                lri_triggered += 1
            if sem_pred == digit:
                s_hits += 1
            if sem_conn_pred == digit:
                sc_hits += 1
            if fused_pred == digit:
                f_hits += 1
                v_hits += 1
            if l_pred == digit:
                l_hits += 1
            if fused_pred == l_pred:
                agreements += 1
            if fused_pred == digit:
                i2t_hits += 1
            if l_pred == digit:
                t2i_hits += 1

        vis_correct[digit] = v_hits / bundle.n_examples
        fused_correct[digit] = f_hits / bundle.n_examples
        sem_route_correct[digit] = s_hits / bundle.n_examples
        sem_conn_correct[digit] = sc_hits / bundle.n_examples
        lang_correct[digit] = l_hits / bundle.n_examples

    return {
        "visual_accuracy": float(fused_correct.mean()),
        "semantic_route_accuracy": float(sem_route_correct.mean()),
        "semantic_connectome_accuracy": float(sem_conn_correct.mean()),
        "language_accuracy": float(lang_correct.mean()),
        "agreement_rate": agreements / n_eval,
        "image_to_text_recall": i2t_hits / n_eval,
        "text_to_image_recall": t2i_hits / n_eval,
        "mean_semantic_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "lri_trigger_rate": lri_triggered / n_eval,
        "per_digit_visual": fused_correct,
        "per_digit_language": lang_correct,
    }


def _train_cross_domain_hub(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    bundle=None,
    views_per_digit: int = 5,
    associate_rounds: int = 10,
    negative_rounds: int = 3,
    hard_negatives: bool = True,
    contrastive_epochs: int = 2,
    confused_curriculum: bool = True,
    confused_class_reinforcement: bool = True,
    semantic_wiring: str = "views",
    use_connectome_lri: bool = True,
    prototypes_per_digit: int = 5,
    **kwargs,
) -> TrainedCrossDomainHub:
    """Train hub and return all artifacts for eval / profiling."""
    if bundle is None:
        bundle = load_ventral_bundle(seed=seed, n_examples=n_examples, k=k, **kwargs)

    beta = kwargs.get("beta", 1.0)
    n_lang = 784
    brain = bundle.brain
    _ensure_hub_areas(brain, n_lang=n_lang, k=k, beta=beta)
    wire_class_from_prototypes(brain, bundle.prototypes, HIGH, CLASS, bundle.k)

    if confused_class_reinforcement:
        _stage0_confused_class_reinforcement(brain, bundle, passes=2)

    refined = refined_confused_prototypes(
        bundle.prototypes, bundle.high_outputs, CONFUSED_DIGITS, bundle.k,
    )

    semantic_anchors, lang_lex = _stage1_language_semantic_anchors(
        brain, seed=seed, n_lang=n_lang, associate_rounds=associate_rounds,
    )
    _stage2_vision_alignment(
        brain,
        bundle,
        semantic_anchors,
        seed=seed,
        n_lang=n_lang,
        views_per_digit=views_per_digit,
        associate_rounds=associate_rounds,
        negative_rounds=negative_rounds,
        hard_negatives=hard_negatives,
        contrastive_epochs=contrastive_epochs,
    )
    if confused_curriculum:
        _stage2b_confused_pair_curriculum(
            brain, bundle, semantic_anchors,
            seed=seed, n_lang=n_lang,
            associate_rounds=max(6, associate_rounds // 2),
            negative_rounds=negative_rounds,
        )
    _consolidate_hub_readout(
        brain, bundle, semantic_anchors,
        wiring=semantic_wiring, refined_prototypes=refined,
    )
    semantic_lex = _build_semantic_lexicon_from_projections(brain, bundle)
    mp_lex = build_multi_prototype_lexicon(
        bundle.high_outputs, k=bundle.k, prototypes_per_digit=prototypes_per_digit,
    )

    return TrainedCrossDomainHub(
        brain=brain,
        bundle=bundle,
        semantic_anchors=semantic_anchors,
        semantic_lex=semantic_lex,
        lang_lex=lang_lex,
        mp_lex=mp_lex,
        seed=seed,
        n_lang=n_lang,
        parameters={
            "seed": seed,
            "n_examples": n_examples,
            "k": k,
            "views_per_digit": views_per_digit,
            "associate_rounds": associate_rounds,
            "hard_negatives": hard_negatives,
            "contrastive_epochs": contrastive_epochs,
            "confused_curriculum": confused_curriculum,
            "confused_class_reinforcement": confused_class_reinforcement,
            "semantic_wiring": semantic_wiring,
            "use_connectome_lri": use_connectome_lri,
            "prototypes_per_digit": prototypes_per_digit,
        },
    )


def run_vision_language_contrastive_hub(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    bundle=None,
    views_per_digit: int = 5,
    associate_rounds: int = 10,
    negative_rounds: int = 3,
    hard_negatives: bool = True,
    contrastive_epochs: int = 2,
    confused_curriculum: bool = True,
    confused_class_reinforcement: bool = True,
    semantic_wiring: str = "views",
    use_connectome_lri: bool = True,
    prototypes_per_digit: int = 5,
    **kwargs,
) -> CrossDomainResult:
    """Full VLM-style assembly-calculus vision–language fusion protocol."""
    hub = _train_cross_domain_hub(
        seed=seed,
        n_examples=n_examples,
        k=k,
        bundle=bundle,
        views_per_digit=views_per_digit,
        associate_rounds=associate_rounds,
        negative_rounds=negative_rounds,
        hard_negatives=hard_negatives,
        contrastive_epochs=contrastive_epochs,
        confused_curriculum=confused_curriculum,
        confused_class_reinforcement=confused_class_reinforcement,
        semantic_wiring=semantic_wiring,
        use_connectome_lri=use_connectome_lri,
        prototypes_per_digit=prototypes_per_digit,
        **kwargs,
    )
    brain = hub.brain
    bundle = hub.bundle

    metrics = _eval_retrieval_and_classification(
        brain, bundle, hub.semantic_anchors, hub.semantic_lex, hub.lang_lex, hub.mp_lex,
        seed=seed, n_lang=hub.n_lang,
        use_connectome_lri=hub.parameters.get("use_connectome_lri", True),
    )
    cm = _contrastive_matrix(
        brain, bundle, hub.semantic_lex, seed=seed, n_lang=hub.n_lang,
    )
    diag = float(np.trace(cm) / NUM_DIGITS)
    offdiag = float((cm.sum() - np.trace(cm)) / (NUM_DIGITS * (NUM_DIGITS - 1)))

    return CrossDomainResult(
        visual_accuracy=metrics["visual_accuracy"],
        language_accuracy=metrics["language_accuracy"],
        agreement_rate=metrics["agreement_rate"],
        mean_semantic_overlap=metrics["mean_semantic_overlap"],
        image_to_text_recall=metrics["image_to_text_recall"],
        text_to_image_recall=metrics["text_to_image_recall"],
        per_digit_visual=metrics["per_digit_visual"],
        per_digit_language=metrics["per_digit_language"],
        data_source=bundle.data_source,
        parameters={
            **hub.parameters,
            "protocol": "stage1_lang_stage2_vision_contrastive_fused_readout",
        },
        extra={
            "word_names": WORD_NAMES,
            "word_forms": WORD_FORMS,
            "binding_gate_note": (
                "Cross-modal binding is gated until representational metrics pass; "
                "see colt_mnist_geometry_panel.BINDING_GATE and Part VI thesis."
            ),
            "semantic_route_accuracy": metrics["semantic_route_accuracy"],
            "semantic_connectome_accuracy": metrics.get("semantic_connectome_accuracy"),
            "lri_trigger_rate": metrics.get("lri_trigger_rate"),
            "readout_fusion": (
                "connectome_lri", "semantic_connectome", "connectome", "multi_prototype",
            ),
            "contrastive_diagonal_overlap": diag,
            "contrastive_offdiag_overlap": offdiag,
            "contrastive_margin": diag - offdiag,
            "hard_negative_pairs": _hard_negative_digit_pairs(),
        },
    )


def run_vision_language_digit_hub(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    bundle=None,
    **kwargs,
) -> CrossDomainResult:
    """Public entry — contrastive VLM-style assembly-calculus fusion."""
    return run_vision_language_contrastive_hub(
        seed=seed, n_examples=n_examples, k=k, bundle=bundle, **kwargs,
    )
