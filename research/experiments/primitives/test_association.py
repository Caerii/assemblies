"""Historical association: driven regeneration and learning-on identity readout.

Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#historical-association-trials

Establish A and B separately with stimulus+self, snapshot them, then associate
through co-stimulation and directed cross-area fibers. Evaluation continues
learning. A-driven regeneration of B does not read the replaced B winners and
therefore does not measure partial-cue completion. Identity evaluation re-trains
each area with its stimulus and recurrent fiber, comparing against the original
pre-association snapshot.

The historical outer harness reports basic overlap, association-duration and
network-size grids, directionality comparisons and identity overlap. Its chance
reference k/n and paired test are descriptive; nonsignificance does not establish
equivalence, and none of these summaries alone certifies an association mechanism.
Explicit seeds, raw observations and tagged runner provenance accompany all grids.
"""

import sys
from pathlib import Path
from numbers import Integral

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from dataclasses import dataclass, replace
from typing import Dict, Any
from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    measure_overlap,
    chance_overlap,
    summarize,
    reported_null_test, summarize_paired,
)

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.registration import validate_area_registration, validate_round_count
from research.experiment_config import resolve_seed_ids, resolve_real_grid

N_SEEDS = 10


def association_round_count(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("association rounds must be a nonnegative integer")
    return int(value)


@dataclass
class AssocConfig:
    """Configuration for association trials."""
    n: int
    k: int
    p: float
    beta: float
    w_max: float
    establish_rounds: int = 30
    assoc_rounds: int = 30
    test_rounds: int = 20


    def __post_init__(self):
        self.n, self.k = validate_area_registration("A", self.n, self.k)
        self.establish_rounds = validate_round_count(self.establish_rounds)
        self.assoc_rounds = association_round_count(self.assoc_rounds)
        self.test_rounds = validate_round_count(self.test_rounds)
        self.p = resolve_real_grid([self.p], name="connection probability", maximum=1.)[0]
        self.beta = resolve_real_grid([self.beta], name="plasticity")[0]
        self.w_max = resolve_real_grid([self.w_max], name="weight clip")[0]


# ── Core trial runner ──────────────────────────────────────────────


# Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#historical-association-trials
def _establish_and_associate(cfg, seed, bidirectional):
    if type(bidirectional) is not bool:
        raise ValueError("bidirectional must be boolean")
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max, engine="numpy_sparse")

    b.add_area("A", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_area("B", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_stimulus("sa", cfg.k)
    b.add_stimulus("sb", cfg.k)

    # Establish assembly A (stim+self)
    for _ in range(cfg.establish_rounds):
        b.project({"sa": ["A"]}, {"A": ["A"]})
    trained_a = np.array(b.areas["A"].winners, dtype=np.uint32)

    # Establish assembly B (stim+self)
    for _ in range(cfg.establish_rounds):
        b.project({"sb": ["B"]}, {"B": ["B"]})
    trained_b = np.array(b.areas["B"].winners, dtype=np.uint32)

    fibers = {"A": ["B"], **({"B": ["A"]} if bidirectional else {})}
    for _ in range(cfg.assoc_rounds):
        b.project({"sa": ["A"], "sb": ["B"]}, fibers)
    return b, trained_a, trained_b


def run_association_trial(
    cfg: AssocConfig, seed: int, bidirectional: bool = True,
    rng: np.random.Generator = None,
) -> Dict[str, Any]:
    """
    Train association between A and B, then test recovery of B from A.

    Learning-on driven regeneration; the replaced B state is not read.
    Returns overlap with pre-association B after A→B projection).
    """
    b, trained_a, trained_b = _establish_and_associate(cfg, seed, bidirectional)

    # Corrupt B: replace all winners with random neurons
    if rng is None:
        rng = np.random.default_rng(seed + 99999)
    random_winners = rng.choice(cfg.n, cfg.k, replace=False).tolist()
    b.areas["B"].winners = random_winners

    # Recover B via A→B projection
    for _ in range(cfg.test_rounds):
        b.project({"sa": ["A"]}, {"A": ["B"]})

    recovery = measure_overlap(trained_b, np.array(b.areas["B"].winners, dtype=np.uint32))

    return {"recovery": recovery, "trained_a": trained_a, "trained_b": trained_b}


def run_identity_trial(
    cfg: AssocConfig, seed: int,
) -> Dict[str, float]:
    """Learning-on stimulus+self overlap with pre-association A and B."""
    b, trained_a, trained_b = _establish_and_associate(cfg, seed, True)

    # Re-activate A via stim+self
    for _ in range(cfg.test_rounds):
        b.project({"sa": ["A"]}, {"A": ["A"]})
    recovery_a = measure_overlap(trained_a, np.array(b.areas["A"].winners, dtype=np.uint32))

    # Re-activate B via stim+self
    for _ in range(cfg.test_rounds):
        b.project({"sb": ["B"]}, {"B": ["B"]})
    recovery_b = measure_overlap(trained_b, np.array(b.areas["B"].winners, dtype=np.uint32))

    return {"recovery_a": recovery_a, "recovery_b": recovery_b}


# ── Main experiment ──────────────────────────────────────────────────


class AssociationExperiment(ExperimentBase):
    """Test cross-area association via co-stimulation."""

    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="association",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "primitives",
            verbose=verbose,
        )

    # Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#historical-association-harness
    def run(self, n=1000, k=100, p=.05, beta=.1, w_max=20., n_seeds=None, *,
            seed_ids=None, establish_rounds=30, assoc_rounds=30, test_rounds=20,
            round_values=None, h1e_sizes=None):
        seeds = resolve_seed_ids(n_seeds, seed_ids, base_seed=self.seed, default_count=N_SEEDS)
        cfg = AssocConfig(n, k, p, beta, w_max, establish_rounds, assoc_rounds, test_rounds)
        rounds = [association_round_count(value) for value in
                  ([1, 5, 10, 20, 30, 50] if round_values is None else round_values)]
        sizes = [validate_area_registration("A", value, 1)[0] for value in
                 ([200, 500, 1000, 2000] if h1e_sizes is None else h1e_sizes)]
        if not rounds or len(set(rounds)) != len(rounds) or not sizes or len(set(sizes)) != len(sizes):
            raise ValueError("association grids must be nonempty and unique")
        training_configs = [replace(cfg, assoc_rounds=value) for value in rounds]
        size_configs = [replace(cfg, n=value, k=int(np.sqrt(value))) for value in sizes]
        self._start_timer()
        rng = np.random.default_rng(self.seed)
        raw = {}

        def association(label, config, bidirectional=True):
            values = [run_association_trial(config, seed, bidirectional, rng=rng)["recovery"]
                      for seed in seeds]
            raw[label] = values
            return {"recovery": summarize(values),
                    "test_vs_null": reported_null_test(values, chance_overlap(config.k, config.n))}

        metrics = {"basic_association": association("basic", cfg)}
        metrics["recovery_vs_training"] = [
            {"assoc_rounds": item.assoc_rounds, **association(f"training/{item.assoc_rounds}", item)}
            for item in training_configs]
        bidirectional, unidirectional = [], []
        for seed in seeds:
            bidirectional.append(run_association_trial(cfg, seed, True, rng=rng)["recovery"])
            unidirectional.append(run_association_trial(cfg, seed, False, rng=rng)["recovery"])
        comparison = summarize_paired(bidirectional, unidirectional, seed_ids=seeds)
        differences = comparison["values"]
        raw.update(bidirectional=bidirectional, unidirectional=unidirectional, paired_difference=differences)
        null = chance_overlap(cfg.k, cfg.n)
        metrics["directionality"] = {
            "bidirectional": {"recovery": summarize(bidirectional), "test_vs_null": reported_null_test(bidirectional, null)},
            "unidirectional": {"recovery": summarize(unidirectional), "test_vs_null": reported_null_test(unidirectional, null)},
            "paired_difference": comparison["summary"],
            "paired_test": comparison["test"],
        }
        identities = [run_identity_trial(cfg, seed) for seed in seeds]
        metrics["identity_preservation"] = {}
        for area in ("A", "B"):
            values = [row[f"recovery_{area.lower()}"] for row in identities]
            raw[f"identity/{area}"] = values
            metrics["identity_preservation"][f"stimulus_recovery_{area}"] = {
                "stats": summarize(values), "test_vs_null": reported_null_test(values, null)}
        metrics["recovery_vs_size"] = [
            {"n": item.n, "k": item.k, **association(f"size/{item.n}/{item.k}", item)} for item in size_configs]
        return ExperimentResult(
            experiment_name=self.name,
            parameters={"n_seeds": len(seeds), "seed_ids": seeds,
                        "base_n": cfg.n, "base_k": cfg.k, "base_p": cfg.p,
                        "base_beta": cfg.beta, "base_wmax": cfg.w_max,
                        "establish_rounds": cfg.establish_rounds, "assoc_rounds": cfg.assoc_rounds,
                        "test_rounds": cfg.test_rounds, "round_values": rounds, "h1e_sizes": sizes,
                        "engine": "numpy_sparse", "area_engine": "numpy_explicit",
                        "readout": "learning-on A-driven regeneration; pre-association references",
                        "size_assembly_rule": "floor(sqrt(n))", "statistics_version": "paired-difference-v1"},
            metrics=metrics, raw_data={"seed_ids": seeds, "values": raw},
            duration_seconds=self._stop_timer())


def main(argv=None):
    from research.experiments.historical_association import main as run
    return run(argv)


if __name__ == "__main__":
    main()
