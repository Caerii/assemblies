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
Seed/provenance/raw-data and runner migration remain outstanding in this harness.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    measure_overlap,
    chance_overlap,
    summarize,
    ttest_vs_null,
    paired_ttest,
)

from neural_assemblies.core.brain import Brain

N_SEEDS = 10


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

    # Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#legacy-experiment-configuration
    def run(
        self,
        n: int = 1000,
        k: int = 100,
        p: float = 0.05,
        beta: float = 0.10,
        w_max: float = 20.0,
        n_seeds: int = N_SEEDS,
    ) -> ExperimentResult:
        self._start_timer()
        seeds = list(range(n_seeds))
        rng = np.random.default_rng(self.seed)

        cfg = AssocConfig(n=n, k=k, p=p, beta=beta, w_max=w_max)
        null = chance_overlap(k, n)

        self.log("=" * 60)
        self.log("Association Experiment (Co-Stimulation Protocol)")
        self.log(f"  n={n}, k={k}, p={p}, beta={beta}, w_max={w_max}")
        self.log(f"  establish_rounds={cfg.establish_rounds}")
        self.log(f"  assoc_rounds={cfg.assoc_rounds}")
        self.log(f"  test_rounds={cfg.test_rounds}")
        self.log(f"  null overlap (k/n) = {null:.3f}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 60)

        metrics: Dict[str, Any] = {}

        # ================================================================
        # H1: Basic association (bidirectional, 30 rounds)
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H1: Basic Association (bidirectional co-stimulation)")
        self.log("=" * 60)

        h1_recoveries = []
        for s in seeds:
            trial = run_association_trial(cfg, seed=self.seed + s, bidirectional=True, rng=rng)
            h1_recoveries.append(trial["recovery"])

        metrics["basic_association"] = {
            "recovery": summarize(h1_recoveries),
            "test_vs_null": ttest_vs_null(h1_recoveries, null),
        }

        self.log(
            f"  Recovery: {metrics['basic_association']['recovery']['mean']:.3f}"
            f"+/-{metrics['basic_association']['recovery']['sem']:.3f}  "
            f"d={metrics['basic_association']['test_vs_null']['d']:.1f}"
        )

        # ================================================================
        # H2: Recovery vs training rounds
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H2: Recovery vs Association Training Rounds")
        self.log("=" * 60)

        round_values = [1, 5, 10, 20, 30, 50]
        h2_results = []

        for n_rounds in round_values:
            cfg_h2 = AssocConfig(n=n, k=k, p=p, beta=beta, w_max=w_max,
                                 assoc_rounds=n_rounds)
            recoveries = []
            for s in seeds:
                trial = run_association_trial(cfg_h2, seed=self.seed + s,
                                             bidirectional=True, rng=rng)
                recoveries.append(trial["recovery"])

            row = {
                "assoc_rounds": n_rounds,
                "recovery": summarize(recoveries),
                "test_vs_null": ttest_vs_null(recoveries, null),
            }
            h2_results.append(row)

            self.log(
                f"  rounds={n_rounds:2d}: "
                f"{row['recovery']['mean']:.3f}+/-{row['recovery']['sem']:.3f}  "
                f"d={row['test_vs_null']['d']:.1f}"
            )

        metrics["recovery_vs_training"] = h2_results

        # ================================================================
        # H3: Bidirectional vs unidirectional
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H3: Bidirectional vs Unidirectional Association")
        self.log("=" * 60)

        bidir_recoveries = []
        unidir_recoveries = []

        for s in seeds:
            trial_bi = run_association_trial(cfg, seed=self.seed + s,
                                            bidirectional=True, rng=rng)
            bidir_recoveries.append(trial_bi["recovery"])

            trial_uni = run_association_trial(cfg, seed=self.seed + s,
                                             bidirectional=False, rng=rng)
            unidir_recoveries.append(trial_uni["recovery"])

        metrics["directionality"] = {
            "bidirectional": {
                "recovery": summarize(bidir_recoveries),
                "test_vs_null": ttest_vs_null(bidir_recoveries, null),
            },
            "unidirectional": {
                "recovery": summarize(unidir_recoveries),
                "test_vs_null": ttest_vs_null(unidir_recoveries, null),
            },
            "paired_test": paired_ttest(bidir_recoveries, unidir_recoveries),
        }

        self.log(
            f"  Bidirectional:  {summarize(bidir_recoveries)['mean']:.3f}"
            f"+/-{summarize(bidir_recoveries)['sem']:.3f}"
        )
        self.log(
            f"  Unidirectional: {summarize(unidir_recoveries)['mean']:.3f}"
            f"+/-{summarize(unidir_recoveries)['sem']:.3f}"
        )
        paired = metrics["directionality"]["paired_test"]
        self.log(f"  Paired t-test: t={paired['t']:.2f}, p={paired['p']:.3f}, d={paired['d']:.1f}")

        # ================================================================
        # H4: Identity preservation
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H4: Identity Preservation After Association")
        self.log("=" * 60)

        identity_a_vals = []
        identity_b_vals = []

        for s in seeds:
            trial = run_identity_trial(cfg, seed=self.seed + s)
            identity_a_vals.append(trial["recovery_a"])
            identity_b_vals.append(trial["recovery_b"])

        metrics["identity_preservation"] = {
            "stimulus_recovery_A": {
                "stats": summarize(identity_a_vals),
                "test_vs_null": ttest_vs_null(identity_a_vals, null),
            },
            "stimulus_recovery_B": {
                "stats": summarize(identity_b_vals),
                "test_vs_null": ttest_vs_null(identity_b_vals, null),
            },
        }

        self.log(f"  Stim→A recovery: {summarize(identity_a_vals)['mean']:.3f}")
        self.log(f"  Stim→B recovery: {summarize(identity_b_vals)['mean']:.3f}")

        # ================================================================
        # H1 Extended: Recovery vs network size (k=sqrt(n))
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H1 Extended: Recovery vs Network Size (k=sqrt(n))")
        self.log("=" * 60)

        h1e_sizes = [200, 500, 1000, 2000]
        h1e_results = []

        for n_val in h1e_sizes:
            k_val = int(np.sqrt(n_val))
            cfg_h1e = AssocConfig(n=n_val, k=k_val, p=p, beta=beta, w_max=w_max)
            null_h1e = chance_overlap(k_val, n_val)

            recoveries = []
            for s in seeds:
                trial = run_association_trial(cfg_h1e, seed=self.seed + s,
                                             bidirectional=True, rng=rng)
                recoveries.append(trial["recovery"])

            row = {
                "n": n_val,
                "k": k_val,
                "recovery": summarize(recoveries),
                "test_vs_null": ttest_vs_null(recoveries, null_h1e),
            }
            h1e_results.append(row)

            self.log(
                f"  n={n_val:4d}, k={k_val:2d}: "
                f"{row['recovery']['mean']:.3f}+/-{row['recovery']['sem']:.3f}  "
                f"d={row['test_vs_null']['d']:.1f}"
            )

        metrics["recovery_vs_size"] = h1e_results

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n_seeds": n_seeds,
                "base_n": n, "base_k": k, "base_p": p,
                "base_beta": beta, "base_wmax": w_max,
                "establish_rounds": cfg.establish_rounds,
                "test_rounds": cfg.test_rounds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Association Experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer seeds)")

    args = parser.parse_args()

    exp = AssociationExperiment(verbose=True)

    if args.quick:
        result = exp.run(n_seeds=5)
        exp.save_result(result, "_quick")
    else:
        result = exp.run()
        exp.save_result(result)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("ASSOCIATION EXPERIMENT SUMMARY")
    print("=" * 70)

    m = result.metrics
    print(f"\nH1: Basic association recovery: {m['basic_association']['recovery']['mean']:.3f}")
    print("\nH2: Recovery vs training rounds:")
    for r in m["recovery_vs_training"]:
        print(f"  {r['assoc_rounds']:2d} rounds: {r['recovery']['mean']:.3f}")
    print("\nH3: Bidirectional vs unidirectional:")
    print(f"  Bidir:  {m['directionality']['bidirectional']['recovery']['mean']:.3f}")
    print(f"  Unidir: {m['directionality']['unidirectional']['recovery']['mean']:.3f}")
    print("\nH4: Identity preservation:")
    print(f"  Stim→A: {m['identity_preservation']['stimulus_recovery_A']['stats']['mean']:.3f}")
    print(f"  Stim→B: {m['identity_preservation']['stimulus_recovery_B']['stats']['mean']:.3f}")
    print("\nH1 Extended: Recovery vs size:")
    for r in m["recovery_vs_size"]:
        print(f"  n={r['n']:4d}: {r['recovery']['mean']:.3f}")

    print(f"\nTotal time: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
