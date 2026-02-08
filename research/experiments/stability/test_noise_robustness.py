"""
Noise Robustness: Assembly Recovery Under Perturbation

Tests whether Hebbian-trained assemblies can recover from partial or total
corruption of their winner neurons, under three recovery mechanisms:

1. Stimulus-driven recovery (H1): After noise injection, re-apply the
   original stimulus with stim+self projection. Tests whether the learned
   stimulus pathway can restore the assembly.
2. Autonomous recovery (H2): After noise injection, run self-projection
   only (no stimulus). Tests whether the self-connectome alone can restore
   the assembly from its attractor basin.
3. Association-based recovery (H3): Establish assemblies in A and B,
   associate via co-stimulation, corrupt B, recover B by projecting from
   clean A. Tests cross-area pattern completion as error correction.
4. Sparsity scaling (H4): Repeat autonomous recovery at k=sqrt(n) for
   n=200,500,1000,2000 to find where the attractor basin narrows.

Noise injection: Replace a fraction of winner neurons with uniformly random
non-winner neurons. At noise_frac=1.0, every winner is replaced.

Establishment: All assemblies trained via stim+self:
project({"s": ["A"]}, {"A": ["A"]}) x 30 rounds.
Association: project({"sa": ["A"], "sb": ["B"]}, {"A": ["B"]}) x 30 rounds.

Statistical methodology:
- N_SEEDS=10 independent seeds per condition.
- One-sample t-test against null k/n.
- Cohen's d effect sizes. Mean +/- SEM.

References:
- Papadimitriou et al., PNAS 117(25):14464-14472, 2020
- Dabagia et al., "Coin-Flipping in the Brain", 2024 (weight saturation)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any
from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    measure_overlap,
    chance_overlap,
    summarize,
    ttest_vs_null,
)

from src.core.brain import Brain

N_SEEDS = 10


@dataclass
class NoiseConfig:
    """Configuration for noise robustness trials."""
    n: int
    k: int
    p: float
    beta: float
    w_max: float
    establish_rounds: int = 30
    recovery_rounds: int = 20


def inject_noise(
    winners: np.ndarray, n_neurons: int, noise_frac: float, rng,
) -> np.ndarray:
    """Replace a fraction of winners with random non-winner neurons."""
    k = len(winners)
    n_replace = int(noise_frac * k)
    if n_replace == 0:
        return winners.copy()

    noisy = winners.copy()
    replace_idx = rng.choice(k, n_replace, replace=False)
    non_winners = np.array([i for i in range(n_neurons) if i not in set(winners.tolist())])
    new_neurons = rng.choice(non_winners, n_replace, replace=False)
    noisy[replace_idx] = new_neurons.astype(np.uint32)
    return noisy


# -- Core trial runners --------------------------------------------------------


def run_stimulus_recovery_trial(
    cfg: NoiseConfig, noise_frac: float, seed: int,
) -> float:
    """
    Establish via stim+self, inject noise, recover via stim+self.
    Returns final overlap with trained assembly.
    """
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    b.add_area("A", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_stimulus("s", cfg.k)

    # Establish
    b.project({"s": ["A"]}, {})
    for _ in range(cfg.establish_rounds):
        b.project({"s": ["A"]}, {"A": ["A"]})
    trained = np.array(b.areas["A"].winners, dtype=np.uint32)

    # Inject noise
    rng = np.random.default_rng(seed + 77777)
    noisy = inject_noise(trained, cfg.n, noise_frac, rng)
    b.areas["A"].winners = noisy

    # Recovery via stim+self
    for _ in range(cfg.recovery_rounds):
        b.project({"s": ["A"]}, {"A": ["A"]})

    return measure_overlap(trained, np.array(b.areas["A"].winners, dtype=np.uint32))


def run_autonomous_recovery_trial(
    cfg: NoiseConfig, noise_frac: float, seed: int,
) -> float:
    """
    Establish via stim+self, inject noise, recover via self-only.
    Returns final overlap with trained assembly.
    """
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    b.add_area("A", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_stimulus("s", cfg.k)

    # Establish
    b.project({"s": ["A"]}, {})
    for _ in range(cfg.establish_rounds):
        b.project({"s": ["A"]}, {"A": ["A"]})
    trained = np.array(b.areas["A"].winners, dtype=np.uint32)

    # Inject noise
    rng = np.random.default_rng(seed + 77777)
    noisy = inject_noise(trained, cfg.n, noise_frac, rng)
    b.areas["A"].winners = noisy

    # Recovery via self-only (autonomous)
    for _ in range(cfg.recovery_rounds):
        b.project({}, {"A": ["A"]})

    return measure_overlap(trained, np.array(b.areas["A"].winners, dtype=np.uint32))


def run_association_recovery_trial(
    cfg: NoiseConfig, noise_frac: float, seed: int,
) -> Dict[str, float]:
    """
    Establish A and B via stim+self, associate via co-stimulation,
    corrupt B, recover via A->B projection.
    Returns B recovery overlap and A integrity.
    """
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    b.add_area("A", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_area("B", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_stimulus("sa", cfg.k)
    b.add_stimulus("sb", cfg.k)

    # Establish A via stim+self
    b.project({"sa": ["A"]}, {})
    for _ in range(cfg.establish_rounds):
        b.project({"sa": ["A"]}, {"A": ["A"]})
    trained_a = np.array(b.areas["A"].winners, dtype=np.uint32)

    # Establish B via stim+self
    b.project({"sb": ["B"]}, {})
    for _ in range(cfg.establish_rounds):
        b.project({"sb": ["B"]}, {"B": ["B"]})
    trained_b = np.array(b.areas["B"].winners, dtype=np.uint32)

    # Associate via co-stimulation: A->B
    for _ in range(cfg.establish_rounds):
        b.project({"sa": ["A"], "sb": ["B"]}, {"A": ["B"]})

    # Corrupt B
    rng = np.random.default_rng(seed + 77777)
    noisy_b = inject_noise(trained_b, cfg.n, noise_frac, rng)
    b.areas["B"].winners = noisy_b

    # Recovery: project A->B (stimulus keeps A intact)
    for _ in range(cfg.recovery_rounds):
        b.project({"sa": ["A"]}, {"A": ["B"]})

    b_recovery = measure_overlap(
        trained_b, np.array(b.areas["B"].winners, dtype=np.uint32)
    )
    a_intact = measure_overlap(
        trained_a, np.array(b.areas["A"].winners, dtype=np.uint32)
    )

    return {"b_recovery": b_recovery, "a_intact": a_intact}


# -- Main experiment -----------------------------------------------------------


class NoiseRobustnessExperiment(ExperimentBase):
    """Test noise robustness: recovery under perturbation."""

    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="noise_robustness",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "stability",
            verbose=verbose,
        )

    def run(
        self,
        n: int = 1000,
        k: int = 100,
        p: float = 0.05,
        beta: float = 0.10,
        w_max: float = 20.0,
        n_seeds: int = N_SEEDS,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()
        seeds = list(range(n_seeds))

        noise_fracs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        cfg = NoiseConfig(n=n, k=k, p=p, beta=beta, w_max=w_max)
        null = chance_overlap(k, n)

        self.log("=" * 60)
        self.log("Noise Robustness Experiment")
        self.log(f"  n={n}, k={k}, p={p}, beta={beta}, w_max={w_max}")
        self.log(f"  establish_rounds={cfg.establish_rounds}")
        self.log(f"  recovery_rounds={cfg.recovery_rounds}")
        self.log(f"  null overlap (k/n) = {null:.3f}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 60)

        metrics: Dict[str, Any] = {}

        # ================================================================
        # H1: Stimulus-Driven Recovery
        # ================================================================
        self.log("\nH1: Stimulus-Driven Recovery (stim+self)")

        h1_results = []
        for nf in noise_fracs:
            vals = []
            for s in seeds:
                vals.append(run_stimulus_recovery_trial(cfg, nf, seed=self.seed + s))

            row = {
                "noise_frac": nf,
                "final_overlap": summarize(vals),
                "test_vs_chance": ttest_vs_null(vals, null),
            }
            h1_results.append(row)

            self.log(
                f"  noise={nf:.1f}: "
                f"{row['final_overlap']['mean']:.3f}+/-{row['final_overlap']['sem']:.3f}  "
                f"d={row['test_vs_chance']['d']:.1f}"
            )

        metrics["h1_stimulus_recovery"] = h1_results

        # ================================================================
        # H2: Autonomous Recovery (self-only)
        # ================================================================
        self.log("\nH2: Autonomous Recovery (self-only)")

        h2_results = []
        for nf in noise_fracs:
            vals = []
            for s in seeds:
                vals.append(run_autonomous_recovery_trial(cfg, nf, seed=self.seed + s))

            row = {
                "noise_frac": nf,
                "final_overlap": summarize(vals),
                "test_vs_chance": ttest_vs_null(vals, null),
            }
            h2_results.append(row)

            self.log(
                f"  noise={nf:.1f}: "
                f"{row['final_overlap']['mean']:.3f}+/-{row['final_overlap']['sem']:.3f}  "
                f"d={row['test_vs_chance']['d']:.1f}"
            )

        metrics["h2_autonomous_recovery"] = h2_results

        # ================================================================
        # H3: Association-Based Recovery
        # ================================================================
        self.log("\nH3: Association-Based Recovery")

        h3_results = []
        for nf in noise_fracs:
            b_vals = []
            a_vals = []
            for s in seeds:
                trial = run_association_recovery_trial(cfg, nf, seed=self.seed + s)
                b_vals.append(trial["b_recovery"])
                a_vals.append(trial["a_intact"])

            row = {
                "noise_frac": nf,
                "final_overlap": summarize(b_vals),
                "a_intact": summarize(a_vals),
                "test_vs_chance": ttest_vs_null(b_vals, null),
            }
            h3_results.append(row)

            self.log(
                f"  noise={nf:.1f}: "
                f"B={row['final_overlap']['mean']:.3f}  "
                f"A={row['a_intact']['mean']:.3f}  "
                f"d={row['test_vs_chance']['d']:.1f}"
            )

        metrics["h3_association_recovery"] = h3_results

        # ================================================================
        # H4: Autonomous Recovery vs Network Size (k=sqrt(n))
        # ================================================================
        self.log("\nH4: Autonomous Recovery vs Network Size (k=sqrt(n))")

        h4_sizes = [200, 500, 1000, 2000]
        h4_noise_fracs = [0.3, 0.5, 0.7, 1.0]
        h4_results = []

        for n_val in h4_sizes:
            k_val = int(np.sqrt(n_val))
            null_h4 = chance_overlap(k_val, n_val)
            cfg_h4 = NoiseConfig(n=n_val, k=k_val, p=p, beta=beta, w_max=w_max)

            noise_entries = []
            for nf in h4_noise_fracs:
                vals = []
                for s in seeds:
                    vals.append(
                        run_autonomous_recovery_trial(cfg_h4, nf, seed=self.seed + s)
                    )

                entry = {
                    "noise_frac": nf,
                    "final_overlap": summarize(vals),
                    "test_vs_chance": ttest_vs_null(vals, null_h4),
                }
                noise_entries.append(entry)

            h4_results.append({
                "n": n_val,
                "k": k_val,
                "noise_levels": noise_entries,
            })

            summary_str = "  ".join(
                f"{nf:.1f}:{e['final_overlap']['mean']:.3f}"
                for nf, e in zip(h4_noise_fracs, noise_entries)
            )
            self.log(f"  n={n_val:4d}, k={k_val:2d}: {summary_str}")

        metrics["h4_sparsity_scaling"] = h4_results

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": n, "k": k, "p": p, "beta": beta, "w_max": w_max,
                "establish_rounds": cfg.establish_rounds,
                "recovery_rounds": cfg.recovery_rounds,
                "noise_fracs": noise_fracs,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Noise Robustness Experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer seeds)")

    args = parser.parse_args()

    exp = NoiseRobustnessExperiment(verbose=True)

    if args.quick:
        result = exp.run(n_seeds=5)
        exp.save_result(result, "_quick")
    else:
        result = exp.run()
        exp.save_result(result)

    print(f"\nTotal time: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
