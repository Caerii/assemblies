"""
Advanced MNIST Brain models — ventral stream and recurrent cortex.

Implements three research-tier-0/1 models on top of the simple illustration
trainers in ``colt_mnist_hierarchical_brain`` and ``colt_mnist_brain_explicit``:

* **ventral** — LOW→MID→HIGH→CLASS with repeated exposure + connectome readout
* **recurrent** — LOW→HIGH with HIGH→HIGH recurrence (notebook dynamics)
* **fuzzy_lexicon** — ventral training + ``fuzzy_readout`` digit lexicon

Theory, hypotheses (H1–H5), ventral information structures, and the roadmap
toward >95% (including LRI competitive cascade readout) are documented in
``colt_mnist_ventral_theory`` — read that module for the full scientific narrative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from neural_assemblies.assembly_calculus.assembly import Assembly, overlap
from neural_assemblies.assembly_calculus.ops import _snap, reciprocal_project
from neural_assemblies.assembly_calculus.readout import fuzzy_readout
from neural_assemblies.programs.colt_mnist_advanced_util import (
    prototypes_to_lexicon,
    read_class_connectome_scores,
    wire_class_from_prototypes,
)
from neural_assemblies.programs.colt_mnist_brain_util import (
    clear_area_winners,
    reinforce_class_slot,
    renorm_connectome_columns,
)
from neural_assemblies.programs.colt_mnist_data import (
    find_mnist_dir as _find_mnist_dir,
    load_mnist_arrays as _load_mnist_arrays,
)
from neural_assemblies.programs.colt_mnist_hierarchical_brain import (
    CLASS,
    HIGH,
    NUM_DIGITS,
    ColtMnistHierarchicalBrainResult,
    _forward_high,
    _init_two_layer_weights,
    _sync_two_layer,
)
from neural_assemblies.programs.colt_mnist_brain_util import set_kcap_winners
from neural_assemblies.programs.colt_mnist_protocol import preprocess_mnist_examples

ReadoutMode = Literal["prototype", "class", "fuzzy", "connectome"]
ModelKind = Literal["ventral", "recurrent", "fuzzy_lexicon"]


@dataclass
class ColtMnistVisualAdvancedResult(ColtMnistHierarchicalBrainResult):
    min_class_separation: float | None = None
    readout_mode: str = "prototype"


def _min_pairwise_prototype_overlap(prototypes: np.ndarray, k: int) -> float:
    assemblies = [
        Assembly(HIGH, np.flatnonzero(prototypes[d] > 0).astype(np.uint32))
        for d in range(NUM_DIGITS)
    ]
    min_ov = 1.0
    for i in range(NUM_DIGITS):
        for j in range(i + 1, NUM_DIGITS):
            min_ov = min(min_ov, overlap(assemblies[i], assemblies[j]))
    return float(min_ov)


def _extend_hierarchical_training(
    brain,
    examples: np.ndarray,
    *,
    n_examples: int,
    train_iters: int,
    class_bias: float,
    class_passes: int,
    class_beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Continue from simple hierarchical HIGH weights with repeated exposure."""
    from neural_assemblies.programs.colt_mnist_hierarchical_brain import LOW, MID

    n_high = brain.areas[HIGH].n
    k = brain.areas[HIGH].k
    high_bias = np.zeros(n_high, dtype=np.float32)

    for digit in range(NUM_DIGITS):
        clear_area_winners(brain, MID)
        clear_area_winners(brain, HIGH)
        for j in range(train_iters):
            set_kcap_winners(brain, LOW, examples[digit, j])
            brain.project({}, {LOW: [MID]})
            brain.project({}, {MID: [HIGH]}, external_drive={HIGH: high_bias})
        snap = _snap(brain, HIGH)
        if len(snap.winners) > 0:
            high_bias[snap.winners] += class_bias
        renorm_connectome_columns(brain, LOW, MID)
        renorm_connectome_columns(brain, MID, HIGH)

    saved = brain.disable_plasticity
    brain.disable_plasticity = True
    for _ in range(class_passes):
        for digit in range(NUM_DIGITS):
            for j in range(n_examples):
                _forward_high(brain, examples[digit, j], high_bias)
                reinforce_class_slot(
                    brain, HIGH, CLASS, digit, k, beta=class_beta,
                )
            renorm_connectome_columns(brain, HIGH, CLASS)

    high_outputs = np.zeros((NUM_DIGITS, n_examples, n_high))
    for digit in range(NUM_DIGITS):
        for j in range(n_examples):
            high_outputs[digit, j] = _forward_high(
                brain, examples[digit, j], high_bias,
            )
    brain.disable_plasticity = saved
    return high_bias, high_outputs


def _attach_class_to_recurrent(
    *,
    seed: int,
    n_low: int,
    n_high: int,
    n_class: int,
    k: int,
    beta: float,
    n_rounds: int,
    n_examples: int,
    p: float,
    class_bias: float,
    class_passes: int,
    class_beta: float,
    absence_exposure_prob: float = 0.0,
) -> tuple[object, np.ndarray, np.ndarray]:
    """Train recurrent HIGH (notebook dynamics), then associate CLASS slots.

    When ``absence_exposure_prob > 0``, structured partial views are mixed into
    the per-digit exposure loop (emergent occlusion training — no separate stage).
    """
    from neural_assemblies.core.brain import Brain
    from neural_assemblies.programs.colt_mnist_protocol import init_protocol_weights
    from neural_assemblies.programs.colt_mnist_brain_util import sync_protocol_weights
    from neural_assemblies.programs.colt_mnist_absence import (
        STRUCTURED_PROTOCOLS,
        apply_absence_mask,
    )

    train_imgs, train_labels, _, _ = _load_mnist_arrays(n_examples)
    examples = preprocess_mnist_examples(
        train_imgs, train_labels, n_examples=n_examples, cap_size=k,
    )
    occ_rng = np.random.default_rng(seed + 991)

    brain = Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse", w_max=1e9)
    brain.add_area("LOW", n_low, k, beta, explicit=True)
    brain.add_area(HIGH, n_high, k, beta, explicit=True)
    brain.add_area(CLASS, n_class, k, beta, explicit=True, slot_count=NUM_DIGITS)

    rng = np.random.default_rng(seed)
    w_hh, a_lh = init_protocol_weights(rng, n_low, n_high, p)
    sync_protocol_weights(brain, w_hh, a_lh, "LOW", HIGH)
    zero_hc = np.zeros((n_high, n_class), dtype=np.float32)
    brain.connectomes[HIGH][CLASS].weights = zero_hc
    if brain._explicit_engine is not None:
        brain._explicit_engine._area_conns[HIGH][CLASS].weights = zero_hc

    high_bias = np.zeros(n_high, dtype=np.float32)
    for digit in range(NUM_DIGITS):
        clear_area_winners(brain, HIGH)
        for j in range(n_rounds):
            pat = examples[digit, j % n_examples]
            if absence_exposure_prob > 0 and occ_rng.random() < absence_exposure_prob:
                proto = STRUCTURED_PROTOCOLS[int(occ_rng.integers(0, len(STRUCTURED_PROTOCOLS)))]
                pat = apply_absence_mask(pat, proto, rng=occ_rng)
            winners = set_kcap_winners(brain, "LOW", pat)
            brain.project(
                external_inputs={"LOW": winners},
                projections={"LOW": [HIGH], HIGH: [HIGH]},
                external_drive={HIGH: high_bias},
            )
        snap = _snap(brain, HIGH)
        if len(snap.winners) > 0:
            high_bias[snap.winners] += class_bias
        renorm_connectome_columns(brain, "LOW", HIGH)
        renorm_connectome_columns(brain, HIGH, HIGH)

    saved = brain.disable_plasticity
    brain.disable_plasticity = True
    for _ in range(class_passes):
        for digit in range(NUM_DIGITS):
            for j in range(n_examples):
                clear_area_winners(brain, HIGH)
                winners = set_kcap_winners(brain, "LOW", examples[digit, j])
                brain.project(
                    external_inputs={"LOW": winners},
                    projections={"LOW": [HIGH], HIGH: [HIGH]},
                    external_drive={HIGH: high_bias},
                )
                reinforce_class_slot(
                    brain, HIGH, CLASS, digit, k, beta=class_beta,
                )
            renorm_connectome_columns(brain, HIGH, CLASS)

    high_outputs = np.zeros((NUM_DIGITS, n_examples, n_high))
    for digit in range(NUM_DIGITS):
        for j in range(n_examples):
            clear_area_winners(brain, HIGH)
            winners = set_kcap_winners(brain, "LOW", examples[digit, j])
            brain.project(
                external_inputs={"LOW": winners},
                projections={"LOW": [HIGH], HIGH: [HIGH]},
                external_drive={HIGH: high_bias},
            )
            snap = _snap(brain, HIGH)
            high_outputs[digit, j, snap.winners] = 1.0
    brain.disable_plasticity = saved
    return brain, high_outputs, high_bias


def _build_prototypes(high_outputs: np.ndarray, k: int) -> np.ndarray:
    prototypes = np.zeros((NUM_DIGITS, high_outputs.shape[2]))
    for digit in range(NUM_DIGITS):
        support = high_outputs[digit].sum(axis=0)
        prototypes[digit, support.argsort()[-k:]] = 1.0
    return prototypes


def _predict(
    brain,
    high_vec: np.ndarray,
    prototypes: np.ndarray,
    digit_lexicon: dict[str, Assembly],
    k: int,
    *,
    readout: ReadoutMode,
    fuzzy_threshold: float,
) -> int:
    if readout == "prototype":
        return int(np.argmax(prototypes @ high_vec))

    if readout == "fuzzy":
        query = Assembly(HIGH, np.flatnonzero(high_vec > 0).astype(np.uint32))
        label = fuzzy_readout(query, digit_lexicon, threshold=fuzzy_threshold)
        return int(label) if label is not None else int(np.argmax(prototypes @ high_vec))

    if readout == "connectome":
        return int(np.argmax(
            read_class_connectome_scores(high_vec, brain, HIGH, CLASS, k, NUM_DIGITS),
        ))

    winners = np.flatnonzero(high_vec > 0).astype(np.uint32)
    brain.areas[HIGH].winners = winners
    brain.areas[HIGH].w = len(winners)
    if brain._explicit_engine is not None:
        brain._explicit_engine.set_winners(HIGH, winners)
    clear_area_winners(brain, CLASS)
    reciprocal_project(brain, HIGH, CLASS, rounds=1)
    return int(np.argmax(
        read_class_connectome_scores(high_vec, brain, HIGH, CLASS, k, NUM_DIGITS),
    ))


def run_colt_mnist_visual_advanced(
    *,
    model: ModelKind = "ventral",
    seed: int = 42,
    n_low: int = 784,
    n_mid: int = 2000,
    n_high: int = 2000,
    n_class: int = 2000,
    k: int = 200,
    beta: float = 1.0,
    n_rounds: int = 5,
    n_examples: int = 50,
    p: float = 0.1,
    class_bias: float = -1.0,
    class_passes: int = 5,
    class_beta: float = 3.0,
    consolidate_class: bool = True,
    readout: ReadoutMode | None = None,
    fuzzy_threshold: float = 0.3,
    repeated_exposure: bool | None = None,
) -> ColtMnistVisualAdvancedResult:
    """Run an advanced ventral-stream MNIST model.

    Models
    ------
    ventral
        Ventral hierarchy (LOW→MID→HIGH→CLASS).  Uses repeated exposure —
        every example in the class block drives one Hebbian ``project`` step —
        and connectome-sum readout after prototype consolidation.  ~79% at n=50.
        Tests **H2** (exposure) and **H4** (readout) in ``colt_mnist_ventral_theory``.

    recurrent
        Recurrent cortex (LOW→HIGH with HIGH→HIGH).  Matches the COLT notebook
        attractor loop; adds CLASS association on top.  ~80% at n=50.
        Tests **H1** (attractor stabilization via recurrence).

    fuzzy_lexicon
        Same training as ``ventral``; classifies with ``fuzzy_readout`` over a
        digit lexicon (Mitropolsky 2023 language-organ readout applied to IT
        assemblies).  ~79% at n=50.

    Parameters beyond the simple illustration trainers are documented in
    ``colt_mnist_ventral_theory`` (Tier A–C roadmap for >95%).
    """
    if model == "recurrent":
        if repeated_exposure is None:
            repeated_exposure = False
        if readout is None:
            readout = "connectome"
        backend = "visual_recurrent_cortex"
        brain, high_outputs, high_bias = _attach_class_to_recurrent(
            seed=seed,
            n_low=n_low,
            n_high=n_high,
            n_class=n_class,
            k=k,
            beta=beta,
            n_rounds=n_rounds,
            n_examples=n_examples,
            p=p,
            class_bias=class_bias,
            class_passes=class_passes,
            class_beta=class_beta,
        )
    else:
        if repeated_exposure is None:
            repeated_exposure = True
        if readout is None:
            readout = "fuzzy" if model == "fuzzy_lexicon" else "connectome"
        backend = (
            "visual_fuzzy_lexicon" if model == "fuzzy_lexicon"
            else "visual_ventral_stream"
        )
        train_iters = n_examples if repeated_exposure else n_rounds

        from neural_assemblies.core.brain import Brain

        train_imgs, train_labels, _, _ = _load_mnist_arrays(n_examples)
        examples = preprocess_mnist_examples(
            train_imgs, train_labels, n_examples=n_examples, cap_size=k,
        )

        brain = Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse", w_max=1e9)
        brain.add_area("LOW", n_low, k, beta, explicit=True)
        brain.add_area("MID", n_mid, k, beta, explicit=True)
        brain.add_area(HIGH, n_high, k, beta, explicit=True)
        brain.add_area(CLASS, n_class, k, beta, explicit=True, slot_count=NUM_DIGITS)
        rng = np.random.default_rng(seed)
        a_lm, w_mh = _init_two_layer_weights(rng, n_low, n_mid, n_high, p)
        _sync_two_layer(brain, a_lm, w_mh)

        _, high_outputs = _extend_hierarchical_training(
            brain,
            examples,
            n_examples=n_examples,
            train_iters=train_iters,
            class_bias=class_bias,
            class_passes=class_passes,
            class_beta=class_beta,
        )

    prototypes = _build_prototypes(high_outputs, k)
    min_sep = _min_pairwise_prototype_overlap(prototypes, k)

    if consolidate_class:
        wire_class_from_prototypes(brain, prototypes, HIGH, CLASS, k)

    digit_lexicon = prototypes_to_lexicon(prototypes, HIGH, k)
    data_source = "mnist_csv" if _find_mnist_dir() else "synthetic_fallback"

    correct = np.zeros(NUM_DIGITS)
    for digit in range(NUM_DIGITS):
        hits = 0
        for j in range(n_examples):
            pred = _predict(
                brain,
                high_outputs[digit, j],
                prototypes,
                digit_lexicon,
                k,
                readout=readout,
                fuzzy_threshold=fuzzy_threshold,
            )
            hits += pred == digit
        correct[digit] = hits / n_examples

    params = {
        "model": model,
        "seed": seed,
        "k": k,
        "n_rounds": n_rounds,
        "n_examples": n_examples,
        "class_passes": class_passes,
        "class_beta": class_beta,
        "consolidate_class": consolidate_class,
        "readout": readout,
        "repeated_exposure": repeated_exposure,
    }
    return ColtMnistVisualAdvancedResult(
        per_class_accuracy=correct,
        mean_accuracy=float(correct.mean()),
        data_source=data_source,
        parameters=params,
        backend=backend,
        min_class_separation=min_sep,
        readout_mode=readout,
    )


def run_ventral_stream_mnist(**kwargs) -> ColtMnistVisualAdvancedResult:
    """V1→V2/V4→IT ventral hierarchy with CLASS consolidation (~79% n=50)."""
    return run_colt_mnist_visual_advanced(model="ventral", **kwargs)


def run_recurrent_cortex_mnist(**kwargs) -> ColtMnistVisualAdvancedResult:
    """V1→IT recurrent attractor + CLASS (~80% n=50); tests H1."""
    return run_colt_mnist_visual_advanced(model="recurrent", **kwargs)


def run_fuzzy_lexicon_mnist(**kwargs) -> ColtMnistVisualAdvancedResult:
    """Ventral training + ``fuzzy_readout`` lexicon (~79% n=50)."""
    return run_colt_mnist_visual_advanced(model="fuzzy_lexicon", **kwargs)
