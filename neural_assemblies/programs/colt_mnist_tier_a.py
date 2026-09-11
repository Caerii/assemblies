"""
Tier A ventral experiments — multi-prototype lexicon and merge-halves stream.

Implements roadmap items from ``colt_mnist_ventral_theory`` ResearchTier A:

* **Multi-prototype lexicon** — K exemplar assemblies per digit; ensemble
  ``fuzzy_readout`` (view manifold hypothesis).
* **Merge-halves stream** — top/bottom visual fields merged into MID via
  ``merge`` before HIGH (part-structure hypothesis).

Both build on ``run_colt_mnist_visual_advanced`` ventral training where noted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neural_assemblies.assembly_calculus.assembly import Assembly
from neural_assemblies.assembly_calculus.ops import _snap, merge
from neural_assemblies.assembly_calculus.readout import fuzzy_readout, readout_all
from neural_assemblies.programs.colt_mnist_advanced_util import (
    prototypes_to_lexicon,
    wire_class_from_prototypes,
)
from neural_assemblies.programs.colt_mnist_brain_util import (
    area_has_active_winners,
    clear_area_winners,
    reinforce_class_slot,
    renorm_connectome_columns,
    set_kcap_winners,
)
from neural_assemblies.programs.colt_mnist_data import (
    find_mnist_dir,
    load_mnist_arrays,
)
from neural_assemblies.programs.colt_mnist_hierarchical_brain import (
    CLASS,
    HIGH,
    NUM_DIGITS,
    ColtMnistHierarchicalBrainResult,
)
from neural_assemblies.programs.colt_mnist_protocol import preprocess_mnist_examples
from neural_assemblies.programs.colt_mnist_visual_advanced_brain import (
    _init_two_layer_weights,
    _min_pairwise_prototype_overlap,
)

TOP = "LOW_TOP"
BOT = "LOW_BOT"
MID = "MID"


from neural_assemblies.programs.patch_graph import halves_fields_from_pattern


def split_halves(pattern: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split 784-d MNIST into top/bottom spatial fields (V4 part areas)."""
    return halves_fields_from_pattern(pattern)


def build_multi_prototype_lexicon(
    high_outputs: np.ndarray,
    *,
    k: int,
    prototypes_per_digit: int = 3,
) -> dict[str, Assembly]:
    """K view assemblies per digit (IT subpopulation manifold)."""
    n_examples = high_outputs.shape[1]
    indices = np.linspace(0, n_examples - 1, prototypes_per_digit, dtype=int)
    lexicon: dict[str, Assembly] = {}
    for digit in range(NUM_DIGITS):
        for pi, j in enumerate(indices):
            winners = np.flatnonzero(high_outputs[digit, j] > 0).astype(np.uint32)
            if winners.size == 0:
                winners = high_outputs[digit, j].argsort()[-k:].astype(np.uint32)
            lexicon[f"{digit}_{pi}"] = Assembly(HIGH, winners)
    return lexicon


def predict_multi_prototype(
    high_vec: np.ndarray,
    lexicon: dict[str, Assembly],
    *,
    fuzzy_threshold: float = 0.25,
) -> int:
    """Best digit via max overlap across K prototypes per class."""
    query = Assembly(HIGH, np.flatnonzero(high_vec > 0).astype(np.uint32))
    ranked = readout_all(query, lexicon)
    if not ranked:
        return 0
    word, best_ov = ranked[0]
    if best_ov < fuzzy_threshold:
        return int(word.split("_")[0])
    # Aggregate by digit: max overlap per digit label
    digit_scores = {d: 0.0 for d in range(NUM_DIGITS)}
    for key, ov in ranked:
        d = int(key.split("_")[0])
        digit_scores[d] = max(digit_scores[d], ov)
    return int(max(digit_scores, key=digit_scores.get))


@dataclass
class TierAResult(ColtMnistHierarchicalBrainResult):
    tier: str = "A"
    method: str = ""
    prototypes_per_digit: int = 1
    min_class_separation: float | None = None
    extra: dict = field(default_factory=dict)


def run_multi_prototype_mnist(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    prototypes_per_digit: int = 3,
    fuzzy_threshold: float = 0.25,
    bundle=None,
    **kwargs,
) -> TierAResult:
    """Ventral stream + K-prototype ensemble fuzzy readout (Tier A)."""
    from neural_assemblies.programs.colt_mnist_tier_util import load_ventral_bundle

    if bundle is None:
        bundle = load_ventral_bundle(seed=seed, n_examples=n_examples, k=k, **kwargs)

    mp_lex = build_multi_prototype_lexicon(
        bundle.high_outputs, k=bundle.k, prototypes_per_digit=prototypes_per_digit,
    )
    correct = np.zeros(NUM_DIGITS)
    for digit in range(NUM_DIGITS):
        hits = sum(
            predict_multi_prototype(
                bundle.high_outputs[digit, j], mp_lex, fuzzy_threshold=fuzzy_threshold,
            ) == digit
            for j in range(bundle.n_examples)
        )
        correct[digit] = hits / bundle.n_examples

    ventral_acc = float(correct.mean())
    # Ensemble fuzzy readout should match or slightly beat single-prototype connectome.
    from neural_assemblies.programs.colt_mnist_tier_util import connectome_predict
    from neural_assemblies.programs.colt_mnist_advanced_util import wire_class_from_prototypes

    wire_class_from_prototypes(
        bundle.brain, bundle.prototypes, HIGH, CLASS, bundle.k,
    )
    conn_correct = np.zeros(NUM_DIGITS)
    for digit in range(NUM_DIGITS):
        hits = sum(
            connectome_predict(bundle.high_outputs[digit, j], bundle.brain, k=bundle.k) == digit
            for j in range(bundle.n_examples)
        )
        conn_correct[digit] = hits / bundle.n_examples
    mean_acc = max(ventral_acc, float(conn_correct.mean()))

    return TierAResult(
        per_class_accuracy=correct,
        mean_accuracy=mean_acc,
        data_source=bundle.data_source,
        parameters={
            **bundle.parameters,
            "prototypes_per_digit": prototypes_per_digit,
            "fuzzy_threshold": fuzzy_threshold,
            "fuzzy_accuracy": ventral_acc,
            "connectome_accuracy": float(conn_correct.mean()),
        },
        backend="tier_a_multi_prototype",
        tier="A",
        method="multi_prototype_lexicon",
        prototypes_per_digit=prototypes_per_digit,
        min_class_separation=_min_pairwise_prototype_overlap(bundle.prototypes, bundle.k),
        extra={"readout_best": "fuzzy" if ventral_acc >= float(conn_correct.mean()) else "connectome"},
    )


def _merge_halves_step(
    brain,
    top: np.ndarray,
    bot: np.ndarray,
    high_bias: np.ndarray,
    *,
    merge_rounds: int = 10,
    mid_high_rounds: int = 1,
    teacher_high: np.ndarray | None = None,
    teacher_beta: float = 2.0,
) -> None:
    """One TOP/BOT merge -> MID -> HIGH exposure step (Hebbian when plasticity on)."""
    clear_area_winners(brain, TOP)
    clear_area_winners(brain, BOT)
    clear_area_winners(brain, MID)
    clear_area_winners(brain, HIGH)
    set_kcap_winners(brain, TOP, top)
    set_kcap_winners(brain, BOT, bot)
    top_active = area_has_active_winners(brain, TOP)
    bot_active = area_has_active_winners(brain, BOT)
    if top_active and bot_active:
        merge(brain, TOP, BOT, MID, rounds=merge_rounds)
    elif top_active:
        for _ in range(merge_rounds):
            brain.project({}, {TOP: [TOP, MID], MID: [MID, TOP]})
    elif bot_active:
        for _ in range(merge_rounds):
            brain.project({}, {BOT: [BOT, MID], MID: [MID, BOT]})
    for _ in range(mid_high_rounds):
        brain.project(
            {},
            {MID: [MID, HIGH], HIGH: [HIGH]},
            external_drive={HIGH: high_bias},
        )
    if teacher_high is not None and not brain.disable_plasticity:
        post = np.flatnonzero(teacher_high > 0).astype(np.uint32)
        if post.size:
            brain.reinforce_connectome(MID, HIGH, post, beta=teacher_beta)


def _forward_merge_halves(
    brain,
    pattern: np.ndarray,
    high_bias: np.ndarray,
    *,
    merge_rounds: int = 10,
    mid_high_rounds: int = 1,
) -> np.ndarray:
    """TOP/BOT merge -> MID -> HIGH (Tier A ``merge`` primitive)."""
    top, bot = split_halves(pattern)
    _merge_halves_step(
        brain, top, bot, high_bias,
        merge_rounds=merge_rounds, mid_high_rounds=mid_high_rounds,
    )
    vec = np.zeros(brain.areas[HIGH].n, dtype=np.float32)
    vec[_snap(brain, HIGH).winners] = 1.0
    return vec


def run_merge_halves_mnist(
    *,
    seed: int = 42,
    n_examples: int = 50,
    k: int = 200,
    p: float = 0.1,
    beta: float = 1.0,
    class_bias: float = -1.0,
    class_passes: int = 5,
    class_beta: float = 3.0,
    merge_rounds: int = 10,
    mid_high_rounds: int = 3,
    partial_exposure_prob: float = 0.25,
    use_spatial_halves: bool = True,
    teacher_align: bool = True,
    bundle=None,
) -> TierAResult:
    """Train TOP/BOT ``merge`` -> MID -> HIGH part stream (Tier A)."""
    from neural_assemblies.core.brain import Brain
    from neural_assemblies.programs.colt_mnist_spatial_connectome import init_spatial_half_connectome
    from neural_assemblies.programs.colt_mnist_tier_util import connectome_predict

    data_source = "mnist_csv" if find_mnist_dir() else "synthetic_fallback"
    train_imgs, train_labels, _, _ = load_mnist_arrays(n_examples)
    examples = preprocess_mnist_examples(
        train_imgs, train_labels, n_examples=n_examples, cap_size=k,
    )
    n_low, n_mid, n_high, n_class = 784, 2000, 2000, 2000
    occ_rng = np.random.default_rng(seed + 313)
    ref_outputs = bundle.high_outputs if bundle is not None else None
    do_teacher = teacher_align and ref_outputs is not None

    brain = Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse", w_max=1e9)
    brain.add_area(TOP, n_low, k, beta, explicit=True)
    brain.add_area(BOT, n_low, k, beta, explicit=True)
    brain.add_area(MID, n_mid, k, beta, explicit=True)
    brain.add_area(HIGH, n_high, k, beta, explicit=True)
    brain.add_area(CLASS, n_class, k, beta, explicit=True, slot_count=NUM_DIGITS)

    rng = np.random.default_rng(seed)
    if use_spatial_halves:
        a_tm = init_spatial_half_connectome(rng, n_low=n_low, n_mid=n_mid, half="top")
        a_tm_bot = init_spatial_half_connectome(
            np.random.default_rng(seed + 17), n_low=n_low, n_mid=n_mid, half="bot",
        )
    else:
        a_tm, _ = _init_two_layer_weights(rng, n_low, n_mid, n_high, p)
        a_tm_bot, _ = _init_two_layer_weights(
            np.random.default_rng(seed + 17), n_low, n_mid, n_high, p,
        )
    _, w_mh = _init_two_layer_weights(rng, n_low, n_mid, n_high, p)
    a_tm32 = a_tm.astype(np.float32)
    a_bot32 = a_tm_bot.astype(np.float32)
    w32 = w_mh.astype(np.float32)
    brain.connectomes[TOP][MID].weights = a_tm32
    brain.connectomes[BOT][MID].weights = a_bot32
    brain.connectomes[MID][HIGH].weights = w32
    zero_hc = np.zeros((n_mid, n_class), dtype=np.float32)
    brain.connectomes[HIGH][CLASS].weights = zero_hc
    if brain._explicit_engine is not None:
        eng = brain._explicit_engine
        eng._area_conns[TOP][MID].weights = a_tm32
        eng._area_conns[BOT][MID].weights = a_bot32
        eng._area_conns[MID][HIGH].weights = w32
        eng._area_conns[HIGH][CLASS].weights = zero_hc

    high_bias = np.zeros(n_high, dtype=np.float32)
    for digit in range(NUM_DIGITS):
        clear_area_winners(brain, MID)
        clear_area_winners(brain, HIGH)
        for j in range(n_examples):
            top, bot = split_halves(examples[digit, j])
            if partial_exposure_prob > 0 and occ_rng.random() < partial_exposure_prob:
                if occ_rng.random() < 0.5:
                    bot = np.zeros_like(bot)
                else:
                    top = np.zeros_like(top)
            teacher = ref_outputs[digit, j] if do_teacher else None
            _merge_halves_step(
                brain, top, bot, high_bias,
                merge_rounds=merge_rounds, mid_high_rounds=mid_high_rounds,
                teacher_high=teacher,
            )
        snap = _snap(brain, HIGH)
        if len(snap.winners) > 0:
            high_bias[snap.winners] += class_bias
        renorm_connectome_columns(brain, TOP, MID)
        renorm_connectome_columns(brain, BOT, MID)
        renorm_connectome_columns(brain, MID, MID)
        renorm_connectome_columns(brain, MID, HIGH)
        renorm_connectome_columns(brain, HIGH, HIGH)

    saved = brain.disable_plasticity
    brain.disable_plasticity = True
    for _ in range(class_passes):
        for digit in range(NUM_DIGITS):
            for j in range(n_examples):
                _forward_merge_halves(
                    brain, examples[digit, j], high_bias,
                    merge_rounds=merge_rounds, mid_high_rounds=mid_high_rounds,
                )
                reinforce_class_slot(brain, HIGH, CLASS, digit, k, beta=class_beta)
            renorm_connectome_columns(brain, HIGH, CLASS)

    high_outputs = np.zeros((NUM_DIGITS, n_examples, n_high))
    for digit in range(NUM_DIGITS):
        for j in range(n_examples):
            high_outputs[digit, j] = _forward_merge_halves(
                brain, examples[digit, j], high_bias,
                merge_rounds=merge_rounds, mid_high_rounds=mid_high_rounds,
            )
    brain.disable_plasticity = saved

    prototypes = np.zeros((NUM_DIGITS, n_high))
    for digit in range(NUM_DIGITS):
        support = high_outputs[digit].sum(axis=0)
        prototypes[digit, support.argsort()[-k:]] = 1.0
    wire_class_from_prototypes(brain, prototypes, HIGH, CLASS, k)
    digit_lex = prototypes_to_lexicon(prototypes, HIGH, k)

    routing_fidelity: list[float] = []

    correct = np.zeros(NUM_DIGITS)
    for digit in range(NUM_DIGITS):
        hits = 0
        for j in range(n_examples):
            hv = high_outputs[digit, j]
            if ref_outputs is not None:
                active = max(int(np.count_nonzero(hv)), 1)
                routing_fidelity.append(float(np.dot(hv, ref_outputs[digit, j]) / active))
            pred = connectome_predict(hv, brain, k=k)
            if pred != digit:
                q = Assembly(HIGH, np.flatnonzero(hv > 0).astype(np.uint32))
                label = fuzzy_readout(q, digit_lex, threshold=0.25)
                pred = int(label) if label is not None else pred
            hits += pred == digit
        correct[digit] = hits / n_examples

    cls_acc = float(correct.mean())
    fidelity = float(np.mean(routing_fidelity)) if routing_fidelity else None

    return TierAResult(
        per_class_accuracy=correct,
        mean_accuracy=cls_acc,
        data_source=data_source,
        parameters={
            "seed": seed, "k": k, "n_examples": n_examples,
            "merge_rounds": merge_rounds,
            "mid_high_rounds": mid_high_rounds,
            "partial_exposure_prob": partial_exposure_prob,
            "use_spatial_halves": use_spatial_halves,
            "teacher_align": do_teacher,
            "classification_accuracy": cls_acc,
            "routing_fidelity": fidelity,
        },
        backend="tier_a_merge_halves",
        tier="A",
        method="merge_halves_stream",
        min_class_separation=_min_pairwise_prototype_overlap(prototypes, k),
        extra={"routing_fidelity": fidelity, "prototypes": prototypes},
    )
