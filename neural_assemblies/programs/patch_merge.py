"""
Grid patch merge stream — K local part areas bound via ``merge`` chain → MID → HIGH.

Generalizes merge-halves (K=2) to arbitrary ``PatchGraph`` lattices.  Each patch
gets a dedicated PART area with spatial connectome restricted to patch support.

Run::

    python -m neural_assemblies.programs.patch_merge --n-examples 50
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neural_assemblies.assembly_calculus.assembly import Assembly
from neural_assemblies.assembly_calculus.ops import _snap, merge
from neural_assemblies.assembly_calculus.readout import fuzzy_readout
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
from neural_assemblies.programs.colt_mnist_hierarchical_brain import (
    CLASS,
    HIGH,
    MID,
    NUM_DIGITS,
    ColtMnistHierarchicalBrainResult,
    _init_two_layer_weights,
)
from neural_assemblies.programs.colt_mnist_spatial_connectome import (
    init_spatial_patch_connectome,
)
from neural_assemblies.programs.colt_mnist_visual_advanced_brain import (
    _min_pairwise_prototype_overlap,
)
from neural_assemblies.programs.patch_graph import PatchGraph, build_grid_patch_graph


def part_area_name(patch_id: int) -> str:
    return f"PART_{patch_id}"


def bind_area_name(level: int) -> str:
    return f"BIND_{level}"


@dataclass
class PatchMergeResult(ColtMnistHierarchicalBrainResult):
    tier: str = "A"
    method: str = "grid_patch_merge_chain"
    extra: dict = field(default_factory=dict)


def _sync_part_connectome(brain, src: str, dst: str, weights: np.ndarray) -> None:
    w32 = weights.astype(np.float32)
    brain.connectomes[src][dst].weights = w32
    if brain._explicit_engine is not None:
        brain._explicit_engine._area_conns[src][dst].weights = w32


def _merge_two_active(
    brain,
    area_a: str,
    area_b: str,
    target: str,
    *,
    merge_rounds: int,
) -> None:
    """``merge`` when both areas active; single-source project otherwise."""
    a_active = area_has_active_winners(brain, area_a)
    b_active = area_has_active_winners(brain, area_b)
    if a_active and b_active:
        merge(brain, area_a, area_b, target, rounds=merge_rounds)
    elif a_active:
        for _ in range(merge_rounds):
            brain.project({}, {area_a: [area_a, target], target: [target, area_a]})
    elif b_active:
        for _ in range(merge_rounds):
            brain.project({}, {area_b: [area_b, target], target: [target, area_b]})


def _teacher_align_high_to_mid(
    brain,
    teacher_high: np.ndarray,
    mid_area: str,
    *,
    teacher_beta: float = 2.5,
) -> None:
    """Supervised MID→HIGH reinforcement toward ventral teacher assembly."""
    if brain.disable_plasticity or teacher_high is None:
        return
    post = np.flatnonzero(teacher_high > 0).astype(np.uint32)
    if post.size and mid_area in brain.areas:
        brain.reinforce_connectome(mid_area, HIGH, post, beta=teacher_beta)


def _patch_merge_chain_step(
    brain,
    graph: PatchGraph,
    fields: list[np.ndarray],
    high_bias: np.ndarray,
    *,
    merge_rounds: int = 8,
    mid_high_rounds: int = 3,
    teacher_high: np.ndarray | None = None,
    teacher_beta: float = 2.5,
    merge_mode: str = "chain",
) -> None:
    """Merge patch fields (balanced tree, linear chain, or simultaneous), then MID → HIGH."""
    n = len(graph.patches)
    part_names = [part_area_name(p.patch_id) for p in graph.patches]
    scale0_ids = [p.patch_id for p in graph.patches if p.scale == 0]
    if not scale0_ids:
        scale0_ids = list(range(n))

    clear_area_winners(brain, MID)
    clear_area_winners(brain, HIGH)
    for name in part_names:
        clear_area_winners(brain, name)
    n_bind = max(len(scale0_ids) - 2, 0)
    for i in range(n_bind):
        clear_area_winners(brain, bind_area_name(i))

    for pid, feature_field in zip(range(n), fields):
        set_kcap_winners(brain, part_names[pid], feature_field)

    if merge_mode == "simultaneous" and len(scale0_ids) > 1:
        projections: dict[str, list[str]] = {MID: [MID]}
        for pid in scale0_ids:
            name = part_names[pid]
            projections[name] = [name, MID]
            projections[MID].append(name)
        for _ in range(merge_rounds):
            brain.project({}, projections)
    elif len(scale0_ids) == 1:
        brain.project({}, {part_names[scale0_ids[0]]: [part_names[scale0_ids[0]], MID]})
    elif len(scale0_ids) == 2:
        _merge_two_active(
            brain, part_names[scale0_ids[0]], part_names[scale0_ids[1]], MID,
            merge_rounds=merge_rounds,
        )
    elif merge_mode == "balanced" and len(scale0_ids) == 4:
        _merge_two_active(
            brain, part_names[scale0_ids[0]], part_names[scale0_ids[1]], bind_area_name(0),
            merge_rounds=merge_rounds,
        )
        _teacher_align_high_to_mid(
            brain, teacher_high, bind_area_name(0), teacher_beta=teacher_beta * 0.5,
        )
        _merge_two_active(
            brain, part_names[scale0_ids[2]], part_names[scale0_ids[3]], bind_area_name(1),
            merge_rounds=merge_rounds,
        )
        _teacher_align_high_to_mid(
            brain, teacher_high, bind_area_name(1), teacher_beta=teacher_beta * 0.5,
        )
        _merge_two_active(
            brain, bind_area_name(0), bind_area_name(1), MID,
            merge_rounds=merge_rounds,
        )
    else:
        ids = scale0_ids
        _merge_two_active(
            brain,
            part_names[ids[0]],
            part_names[ids[1]],
            bind_area_name(0),
            merge_rounds=merge_rounds,
        )
        _teacher_align_high_to_mid(
            brain, teacher_high, bind_area_name(0), teacher_beta=teacher_beta * 0.5,
        )
        current = bind_area_name(0)
        for i, pid in enumerate(ids[2:-1]):
            _merge_two_active(
                brain, current, part_names[pid], bind_area_name(i + 1),
                merge_rounds=merge_rounds,
            )
            _teacher_align_high_to_mid(
                brain, teacher_high, bind_area_name(i + 1), teacher_beta=teacher_beta * 0.5,
            )
            current = bind_area_name(i + 1)
        _merge_two_active(
            brain, current, part_names[ids[-1]], MID,
            merge_rounds=merge_rounds,
        )

    for _ in range(mid_high_rounds):
        brain.project(
            {},
            {MID: [MID, HIGH], HIGH: [HIGH]},
            external_drive={HIGH: high_bias},
        )
    _teacher_align_high_to_mid(brain, teacher_high, MID, teacher_beta=teacher_beta)


def _forward_patch_merge(
    brain,
    graph: PatchGraph,
    pattern: np.ndarray,
    high_bias: np.ndarray,
    **kwargs,
) -> np.ndarray:
    fields = graph.split_patch_fields(pattern, tuple(range(len(graph.patches))))
    _patch_merge_chain_step(brain, graph, fields, high_bias, **kwargs)
    vec = np.zeros(brain.areas[HIGH].n, dtype=np.float32)
    snap = _snap(brain, HIGH)
    if snap.winners.size:
        vec[np.asarray(snap.winners, dtype=int)] = 1.0
    return vec


def run_grid_patch_merge_mnist(
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
    mid_high_rounds: int = 4,
    grid: int = 2,
    rf_radius: int = 3,
    partial_exposure_prob: float = 0.35,
    teacher_beta: float = 2.5,
    teacher_align: bool = True,
    merge_mode: str = "chain",
    bundle=None,
    patch_graph: PatchGraph | None = None,
) -> PatchMergeResult:
    """Train K-patch merge chain (default 2×2 grid → 4 patches)."""
    from neural_assemblies.core.brain import Brain
    from neural_assemblies.programs.colt_mnist_data import find_mnist_dir, load_mnist_arrays
    from neural_assemblies.programs.colt_mnist_protocol import preprocess_mnist_examples
    from neural_assemblies.programs.colt_mnist_tier_util import connectome_predict

    graph = patch_graph or build_grid_patch_graph(grid=grid, radii=(rf_radius,), seed=seed)
    n_patches = len(graph.patches)
    data_source = "mnist_csv" if find_mnist_dir() else "synthetic_fallback"
    train_imgs, train_labels, _, _ = load_mnist_arrays(n_examples)
    examples = preprocess_mnist_examples(
        train_imgs, train_labels, n_examples=n_examples, cap_size=k,
    )
    n_low, n_mid, n_high, n_class = 784, 2000, 2000, 2000
    occ_rng = np.random.default_rng(seed + 419)
    ref_outputs = bundle.high_outputs if bundle is not None else None
    do_teacher = teacher_align and ref_outputs is not None

    brain = Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse", w_max=1e9)
    part_names = [part_area_name(i) for i in range(n_patches)]
    n_bind = max(n_patches - 2, 0)
    for name in part_names:
        brain.add_area(name, n_low, k, beta, explicit=True)
    for i in range(n_bind):
        brain.add_area(bind_area_name(i), n_mid, k, beta, explicit=True)
    brain.add_area(MID, n_mid, k, beta, explicit=True)
    brain.add_area(HIGH, n_high, k, beta, explicit=True)
    brain.add_area(CLASS, n_class, k, beta, explicit=True, slot_count=NUM_DIGITS)

    rng = np.random.default_rng(seed)
    for i, spec in enumerate(graph.patches):
        idx = graph.patch_indices(spec)
        w_pm = init_spatial_patch_connectome(
            np.random.default_rng(seed + i * 31),
            idx, n_low=n_low, n_mid=n_mid, rf_radius=spec.radius,
        )
        _sync_part_connectome(brain, part_names[i], MID, w_pm)

    _, w_mh = _init_two_layer_weights(rng, n_low, n_mid, n_high, p)
    brain.connectomes[MID][HIGH].weights = w_mh.astype(np.float32)
    zero_hc = np.zeros((n_high, n_class), dtype=np.float32)
    brain.connectomes[HIGH][CLASS].weights = zero_hc
    if brain._explicit_engine is not None:
        eng = brain._explicit_engine
        eng._area_conns[MID][HIGH].weights = w_mh.astype(np.float32)
        eng._area_conns[HIGH][CLASS].weights = zero_hc

    high_bias = np.zeros(n_high, dtype=np.float32)
    step_kw = dict(
        merge_rounds=merge_rounds,
        mid_high_rounds=mid_high_rounds,
        teacher_beta=teacher_beta,
        merge_mode=merge_mode,
    )
    for digit in range(NUM_DIGITS):
        for j in range(n_examples):
            pat = examples[digit, j]
            fields = graph.split_patch_fields(pat, tuple(range(n_patches)))
            if partial_exposure_prob > 0 and occ_rng.random() < partial_exposure_prob:
                n_drop = int(occ_rng.integers(1, max(n_patches, 2)))
                for drop in occ_rng.choice(n_patches, size=min(n_drop, n_patches), replace=False):
                    fields[int(drop)] = np.zeros_like(fields[int(drop)])
            teacher = ref_outputs[digit, j] if do_teacher else None
            _patch_merge_chain_step(
                brain, graph, fields, high_bias, teacher_high=teacher, **step_kw,
            )
        snap = _snap(brain, HIGH)
        if snap.winners.size:
            high_bias[snap.winners] += class_bias
        for name in part_names:
            renorm_connectome_columns(brain, name, MID)
        for i in range(n_bind):
            renorm_connectome_columns(brain, bind_area_name(i), MID)
        renorm_connectome_columns(brain, MID, MID)
        renorm_connectome_columns(brain, MID, HIGH)
        renorm_connectome_columns(brain, HIGH, HIGH)

    saved = brain.disable_plasticity
    brain.disable_plasticity = True
    for _ in range(class_passes):
        for digit in range(NUM_DIGITS):
            for j in range(n_examples):
                _forward_patch_merge(
                    brain, graph, examples[digit, j], high_bias, **step_kw,
                )
                reinforce_class_slot(brain, HIGH, CLASS, digit, k, beta=class_beta)
            renorm_connectome_columns(brain, HIGH, CLASS)

    high_outputs = np.zeros((NUM_DIGITS, n_examples, n_high))
    for digit in range(NUM_DIGITS):
        for j in range(n_examples):
            high_outputs[digit, j] = _forward_patch_merge(
                brain, graph, examples[digit, j], high_bias, **step_kw,
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

    fidelity = float(np.mean(routing_fidelity)) if routing_fidelity else None
    return PatchMergeResult(
        per_class_accuracy=correct,
        mean_accuracy=float(correct.mean()),
        data_source=data_source,
        parameters={
            "seed": seed, "k": k, "n_examples": n_examples,
            "grid": grid, "n_patches": n_patches,
            "merge_rounds": merge_rounds,
            "routing_fidelity": fidelity,
            "patch_graph": f"grid_{grid}",
        },
        backend="grid_patch_merge",
        extra={
            "routing_fidelity": fidelity,
            "n_patches": n_patches,
            "min_class_separation": _min_pairwise_prototype_overlap(prototypes, k),
            "prototypes": prototypes,
        },
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Grid patch merge MNIST")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-examples", type=int, default=50)
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument(
        "--merge-mode",
        choices=("simultaneous", "balanced", "chain"),
        default="chain",
    )
    args = parser.parse_args()

    from neural_assemblies.programs.colt_mnist_tier_util import clear_ventral_bundle_cache, load_ventral_bundle

    clear_ventral_bundle_cache()
    ref = load_ventral_bundle(seed=args.seed, n_examples=args.n_examples, use_cache=False)
    r = run_grid_patch_merge_mnist(
        seed=args.seed, n_examples=args.n_examples, grid=args.grid, bundle=ref,
        merge_mode=args.merge_mode,
    )
    print(f"accuracy={r.mean_accuracy:.1%}  routing_fidelity={r.extra.get('routing_fidelity'):.1%}")


if __name__ == "__main__":
    main()
