"""In-process protocol executors (metrics without pytest)."""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Dict

from neural_assemblies.reference.nemo_numpy import compare_scaffold_vs_simple
from neural_assemblies.exceptions import RetractedProtocol

from .paths import golden_dir, parity_root



def _retracted_coin(protocol_id: str) -> dict:
    raise RetractedProtocol(
        f"{protocol_id} is a retracted legacy record: its reported output did "
        "not validate the neural mechanism. Use the explicit SeedMixtureChoice "
        "or ContextAttractorChoice contracts and register a new protocol ID; "
        "do not overwrite or rebaseline this historical golden."
    )

def _load_golden(name: str) -> dict:
    path = golden_dir() / name
    if not path.is_file():
        path = parity_root() / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _metrics_close(actual: Any, expected: Any, tol: float) -> bool:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(actual - expected) <= tol
    return actual == expected


def execute_coin2024_demo() -> dict:
    return _retracted_coin("coin2024_demo")


def execute_coin2024_compete() -> dict:
    return _retracted_coin("coin2024_compete")


def execute_nemo2025_scaffold() -> dict:
    g = _load_golden("nemo2025_scaffold.json")
    p = g["parameters"]
    result = compare_scaffold_vs_simple(
        seq_len=p["seq_len"],
        n=p["n"],
        k=p["k"],
        density=p["p"],
        n_presentations=p["n_presentations"],
        seed=p["seed"],
    )
    return {
        "simple_last_recall": round(result.simple_last, 4),
        "scaffold_last_recall": round(result.scaffold_last, 4),
        "scaffold_beats_simple": result.scaffold_beats_simple,
    }


def execute_colt2022_mnist() -> dict:
    from neural_assemblies.assembly_calculus.ops import project
    from neural_assemblies.core.brain import Brain
    from neural_assemblies.programs.learn import classify, learn_separable_classes

    g = _load_golden("colt2022_mnist.json")
    p = g["parameters"]
    brain = Brain(p=p["p"], save_winners=True, seed=p["seed"], engine="numpy_sparse")
    brain.add_area("CLASS", p["n"], p["k"], p["beta"])
    class_stimuli = {str(i): f"s{i}" for i in range(p["n_classes"])}
    for stim in class_stimuli.values():
        brain.add_stimulus(stim, p["k"])
    lexicon, min_ov = learn_separable_classes(
        brain, class_stimuli, "CLASS", rounds=p["rounds"],
    )
    correct = sum(
        1
        for label, stim in class_stimuli.items()
        if classify(lexicon, project(brain, stim, "CLASS", rounds=3), threshold=0.3) == label
    )
    return {
        "train_classify_accuracy": round(correct / p["n_classes"], 4),
        "min_pairwise_overlap": round(min_ov, 6),
    }


def execute_colt2022_mnist_notebook() -> dict:
    from neural_assemblies.programs.colt_mnist_numpy import run_colt_mnist_numpy

    g = _load_golden("colt2022_mnist_notebook.json")
    result = run_colt_mnist_numpy(**g["parameters"])
    return {
        "mean_accuracy": round(result.mean_accuracy, 4),
        "data_source": result.data_source,
    }


def execute_direct2026_pearl() -> dict:
    raise RetractedProtocol(
        "direct2026_pearl is retracted: its overlap readout is insensitive to "
        "the learned CAUSE->BIND fiber. Define a synaptic-asymmetry readout "
        "with a wipe negative control before recording a replacement protocol."
    )


def execute_hoff2026_size_dist() -> dict:
    from neural_assemblies.assembly_calculus.ops import project
    from neural_assemblies.compute import EPercentPolicy
    from neural_assemblies.core.brain import Brain

    g = _load_golden("hoff2026_size_dist.json")
    targets = g["repo_smoke_targets"]
    n, k = 1000, 80
    brain = Brain(p=0.05, save_winners=True, seed=42, engine="numpy_sparse")
    policy = EPercentPolicy(
        fraction_of_max=targets["fraction_of_max"],
        e_fraction=targets["e_fraction"],
        min_winners=targets["min_winners"],
    )
    brain.add_area("M", n, k, 0.1, winner_policy=policy)
    brain.add_stimulus("s", k)
    asm = project(brain, "s", "M", rounds=10)
    return {"assembly_size": len(asm)}


def execute_nemo2025_fsm_mod3() -> dict:
    """RETRACTED protocol -- the golden certified a dictionary lookup.

    It was recorded while `NemoArcFSM.step_symbol` returned a table lookup, so
    its metrics came from the transition table rather than from the network:
    the same values are returned by an untrained brain and at beta=0. Its
    parameters also put the arc at kp = 2 against a floor of 22.8. Re-running
    it would compare live dynamics against numbers no network produced.

    Raising rather than returning numbers, because a parity executor that
    quietly reports SOMETHING is how a retracted golden comes back. See the
    RETRACTED block in the golden and
    `research/notes/sequence/the_arc_is_a_conjunction_and_the_state_drifts.md`; the
    condition for re-recording is P-GOLD passing in seq_a1_fsm_parity.py.
    """
    raise RetractedProtocol(
        "nemo2025_fsm_mod3 was retracted on 2026-08-10 (commit fc2a036): the "
        "golden was recorded against a table-lookup readout, so it is not "
        "evidence about the network. Do not re-record until the FSM decides "
        "reliably from the assembly.")


def execute_coin2024_softmax() -> dict:
    return _retracted_coin("coin2024_softmax")


def execute_colt2022_mnist_brain() -> dict:
    from neural_assemblies.programs.colt_mnist_brain import run_colt_mnist_brain

    g = _load_golden("colt2022_mnist_brain.json")
    result = run_colt_mnist_brain(**g["parameters"])
    return {
        "mean_accuracy": round(result.mean_accuracy, 4),
        "data_source": result.data_source,
    }


def execute_coin2024_markov_arc() -> dict:
    return _retracted_coin("coin2024_markov_arc")


def execute_nemo2025_fsm_mod3_numpy() -> dict:
    from neural_assemblies.reference.nemo_numpy.fsm_network import run_mod3_fsm_numpy

    g = _load_golden("nemo2025_fsm_mod3_numpy.json")
    result = run_mod3_fsm_numpy(**g["parameters"])
    return {
        "positive_accepted": result["positive_accepted"],
        "negative_rejected": result["negative_rejected"],
        "positive_final_state": result["positive_final_state"],
    }


def execute_colt2022_mnist_hierarchical() -> dict:
    from neural_assemblies.programs.colt_mnist_hierarchical import run_colt_mnist_hierarchical

    g = _load_golden("colt2022_mnist_hierarchical.json")
    result = run_colt_mnist_hierarchical(**g["parameters"])
    return {
        "mean_accuracy": round(result.mean_accuracy, 4),
        "data_source": result.data_source,
    }


def execute_colt2022_mnist_notebook_full() -> dict:
    from neural_assemblies.programs.colt_mnist_protocol import run_colt_mnist_protocol

    g = _load_golden("colt2022_mnist_notebook_full.json")
    p = g["parameters"]
    result = run_colt_mnist_protocol(
        seed=p["seed"],
        n_in=p["n_in"],
        n_neurons=p["n_neurons"],
        cap_size=p["cap_size"],
        sparsity=p["sparsity"],
        beta=p["beta"],
        n_rounds=p["n_rounds"],
        n_examples=p["n_examples"],
    )
    return {
        "mean_accuracy": round(result.mean_accuracy, 4),
        "data_source": result.data_source,
    }


def execute_colt2022_mnist_notebook_paper() -> dict:
    from neural_assemblies.programs.colt_mnist_protocol import run_colt_mnist_protocol

    g = _load_golden("colt2022_mnist_notebook_paper.json")
    p = g["parameters"]
    result = run_colt_mnist_protocol(
        seed=p["seed"],
        n_in=p["n_in"],
        n_neurons=p["n_neurons"],
        cap_size=p["cap_size"],
        sparsity=p["sparsity"],
        beta=p["beta"],
        n_rounds=p["n_rounds"],
        n_examples=p["n_examples"],
    )
    return {
        "mean_accuracy": round(result.mean_accuracy, 4),
        "data_source": result.data_source,
    }


def execute_pnas2020_reciprocal() -> dict:
    from neural_assemblies.programs.pnas_extended import run_pnas_reciprocal

    g = _load_golden("pnas2020_reciprocal.json")
    result = run_pnas_reciprocal(**g["parameters"])
    return {"reciprocal_restore_overlap": round(result.reciprocal_restore_overlap, 4)}


def execute_pnas2020_pattern_complete() -> dict:
    from neural_assemblies.programs.pnas_extended import run_pnas_pattern_complete

    g = _load_golden("pnas2020_pattern_complete.json")
    result = run_pnas_pattern_complete(**g["parameters"])
    return {
        "pattern_complete_50pct_overlap": round(
            result.pattern_complete_50pct_overlap, 4,
        ),
    }


def execute_seq25_multi_recall() -> dict:
    from neural_assemblies.programs.seq_multi_recall import run_seq_multi_recall

    g = _load_golden("seq25_multi_recall.json")
    result = run_seq_multi_recall(**g["parameters"])
    return {
        "first_item_overlap": round(result.first_item_overlap, 4),
        "second_item_overlap": round(result.second_item_overlap, 4),
        "recalled_length": result.recalled_length,
    }


def execute_tacl2021_parser_f1() -> dict:
    from neural_assemblies.programs.tacl_parser_f1 import run_tacl_parser_f1_smoke

    g = _load_golden("tacl2021_parser_f1.json")
    result = run_tacl_parser_f1_smoke(**g["parameters"])
    return {
        "role_recall": round(result.role_recall, 4),
        "roles_found": result.roles_found,
    }


def execute_pnas2020_scaling() -> dict:
    from neural_assemblies.assembly_calculus.ops import overlap, project, separate
    from neural_assemblies.core.brain import Brain

    def _record(n, k, p, beta, seed):
        # Parity reproduction of Papadimitriou et al. (PNAS 2020). The goldens
        # (project_persistence=1.0, separate near chance) are the un-normalized
        # model's numbers, and the cross-repo reference brain (ref_brain) has no
        # norm_init. norm_init is our PRODUCTION enhancement, not part of the
        # reference model, so parity must reproduce the reference substrate:
        # pinned off here (measured: persistence 1.0 off vs 0.86-0.92 on).
        brain = Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse",
                      norm_init=False)
        brain.add_stimulus("s1", k)
        brain.add_stimulus("s2", k)
        brain.add_area("A", n, k, beta)
        a1 = project(brain, "s1", "A", rounds=10)
        a1b = project(brain, "s1", "A", rounds=10)
        _, _, sep_ov = separate(brain, "s1", "s2", "A", rounds=10)
        return {
            "chance_overlap": round(k / n, 6),
            "project_persistence": round(overlap(a1, a1b), 4),
            "separate_overlap": round(sep_ov, 4),
        }

    return {
        "ci_parity": _record(5000, 80, 0.05, 0.1, 42),
        "paper_canonical": _record(10000, 100, 0.01, 0.05, 42),
        "test_assembly_calculus": _record(10000, 100, 0.05, 0.1, 42),
    }


def execute_cross_lang_pnas_scaling() -> dict:
    from research.literature.cross_lang.runner import run_python_pnas_scaling

    return run_python_pnas_scaling()


EXECUTORS: Dict[str, Callable[[], dict]] = {
    "coin2024_demo": execute_coin2024_demo,
    "coin2024_compete": execute_coin2024_compete,
    "nemo2025_scaffold": execute_nemo2025_scaffold,
    "colt2022_mnist": execute_colt2022_mnist,
    "colt2022_mnist_notebook": execute_colt2022_mnist_notebook,
    "direct2026_pearl": execute_direct2026_pearl,
    "hoff2026_size_dist": execute_hoff2026_size_dist,
    "nemo2025_fsm_mod3": execute_nemo2025_fsm_mod3,
    "coin2024_softmax": execute_coin2024_softmax,
    "colt2022_mnist_brain": execute_colt2022_mnist_brain,
    "coin2024_markov_arc": execute_coin2024_markov_arc,
    "nemo2025_fsm_mod3_numpy": execute_nemo2025_fsm_mod3_numpy,
    "colt2022_mnist_hierarchical": execute_colt2022_mnist_hierarchical,
    "colt2022_mnist_notebook_full": execute_colt2022_mnist_notebook_full,
    "colt2022_mnist_notebook_paper": execute_colt2022_mnist_notebook_paper,
    "pnas2020_reciprocal": execute_pnas2020_reciprocal,
    "pnas2020_pattern_complete": execute_pnas2020_pattern_complete,
    "seq25_multi_recall": execute_seq25_multi_recall,
    "tacl2021_parser_f1": execute_tacl2021_parser_f1,
    "pnas2020_scaling": execute_pnas2020_scaling,
    "cross_lang.pnas_scaling": execute_cross_lang_pnas_scaling,
}


def verify_against_golden(
    protocol_id: str,
    metrics: dict,
    golden: dict,
) -> tuple[bool, dict]:
    """Compare metrics to golden thresholds / expected values."""
    diffs: dict = {}
    tol = golden.get("tolerance", 0.02)
    if isinstance(tol, dict):
        default_tol = 0.02
    else:
        default_tol = float(tol)

    gm = golden.get("metrics", {})
    th = golden.get("thresholds", {})
    exp = golden.get("expected", {})
    passed = True
    checked = 0

    for key, expected in exp.items():
        # Older goldens repeat a bound under "expected". Interpret the
        # suffix as a bound, rather than looking for a fictitious metric.
        if key.endswith(("_min", "_max")):
            if key in th and th[key] != expected:
                diffs[key] = {"error": "conflicting expected and threshold"}
            th = {**th, key: expected}
            continue
        checked += 1
        if key not in metrics:
            diffs[key] = {"error": "missing required metric", "expected": expected}
            continue
        actual = metrics[key]
        if isinstance(expected, bool):
            ok = actual is expected
        elif isinstance(expected, (int, float)):
            t = tol.get(key, default_tol) if isinstance(tol, dict) else default_tol
            ok = _metrics_close(actual, expected, t)
        else:
            ok = actual == expected
        if not ok:
            passed = False
            diffs[key] = {"actual": actual, "expected": expected}

    for key, bound in th.items():
        if key == "metrics_match_tolerance":
            continue
        checked += 1
        if key.endswith(("_min", "_max")):
            mkey = key[:-4]
            actual = metrics.get(mkey)
            numeric = (
                isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual)
                and isinstance(bound, (int, float)) and not isinstance(bound, bool)
                and math.isfinite(bound)
            )
            ok = numeric and (actual >= bound if key.endswith("_min") else actual <= bound)
            if not ok:
                diffs[key] = {"actual": actual, "bound": bound,
                              "error": "missing, nonfinite, or outside bound"}
        elif isinstance(bound, bool):
            if metrics.get(key) is not bound:
                diffs[key] = {"actual": metrics.get(key), "expected": bound}
        else:
            diffs[key] = {"error": "unsupported threshold; needs an explicit comparator"}

    for key, gval in gm.items():
        if key in metrics and isinstance(gval, (int, float)):
            if key not in exp and key not in th and "metrics_match_tolerance" not in th:
                continue
            checked += 1
            t = th.get("metrics_match_tolerance",
                       tol.get(key, default_tol) if isinstance(tol, dict) else default_tol)
            if not _metrics_close(metrics[key], gval, t):
                if key not in diffs:
                    diffs[f"golden.{key}"] = {"actual": metrics[key], "golden": gval}
        elif isinstance(gval, (int, float)) and "metrics_match_tolerance" in th:
            checked += 1
            diffs[f"golden.{key}"] = {"error": "missing required metric"}

    if checked == 0:
        diffs["criteria"] = {"error": f"{protocol_id}: no evaluable acceptance criteria"}
    return passed and len(diffs) == 0, diffs
