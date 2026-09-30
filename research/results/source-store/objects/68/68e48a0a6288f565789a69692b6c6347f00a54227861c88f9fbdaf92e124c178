"""
Parameter Robustness: Core Effects Across Parameter Space

Four 1D sweeps, each varying one parameter while holding others at reference:
  - n (network size): [2000, 5000, 10000, 20000]
  - beta (plasticity): [0.05, 0.10, 0.15, 0.20]
  - p (connectivity): [0.02, 0.05, 0.10]
  - k/n ratio (sparsity): [0.005, 0.01, 0.02, 0.05]

At each parameter point, a compact mini-trial trains on RecursiveCFG and
measures three core phenomena:
  1. P600 double dissociation (CatViol > Gram at object position)
  2. SRC vs ORC dual-binding P600 difference
  3. Garden-path N400 difference (GP > unambiguous)

Hypotheses:
  H1: All three effects d > 0.5 at reference params
  H2: Effects robust across >= 50% of parameter combinations
  H3: There exists a minimum n below which effects collapse

Usage:
    uv run python research/experiments/primitives/test_parameter_robustness.py
    uv run python research/experiments/primitives/test_parameter_robustness.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    summarize,
    paired_ttest,
)
from research.experiments.lib.vocabulary import RECURSIVE_VOCAB
from research.experiments.lib.grammar import RecursiveCFG
from research.experiments.lib.brain_setup import (
    BrainConfig,
    create_language_brain,
    build_lexicon,
    activate_word,
)
from research.experiments.lib.training import train_sentence
from research.experiments.lib.measurement import measure_n400, measure_p600


@dataclass
class RobustnessConfig:
    # Reference parameters
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.10
    w_max: float = 20.0
    lexicon_rounds: int = 20
    train_rounds_per_pair: int = 5
    binding_rounds: int = 10
    n_settling_rounds: int = 10
    lexicon_readout_rounds: int = 5
    # Training
    n_train_sentences: int = 80
    training_reps: int = 2
    # Test
    n_test_items: int = 5


def run_mini_trial(
    n: int, k: int, p: float, beta: float,
    cfg: RobustnessConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run a compact trial measuring three core phenomena."""
    rng = np.random.default_rng(seed)
    vocab = RECURSIVE_VOCAB

    bcfg = BrainConfig(
        n=n, k=k, p=p, beta=beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain = create_language_brain(bcfg, vocab, seed)

    grammar = RecursiveCFG(
        pp_prob=0.3, rel_prob=0.4, orc_prob=0.3,
        max_pp_depth=1, vocab=vocab, rng=rng)

    n_train = cfg.n_train_sentences * cfg.training_reps
    train_sents = grammar.generate_batch(n_train)

    for sent in train_sents:
        train_sentence(brain, sent, vocab,
                       cfg.train_rounds_per_pair, cfg.binding_rounds)

    brain.disable_plasticity = True
    lexicon = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)

    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    ni = cfg.n_test_items

    # 1. P600 double dissociation at object position
    p600_gram, p600_cv = [], []
    for i in range(ni):
        gram_obj = nouns[(i + 1) % len(nouns)]
        cv_obj = verbs[(i + 1) % len(verbs)]
        p600_gram.append(measure_p600(
            brain, gram_obj, "NOUN_CORE", "ROLE_PATIENT", cfg.n_settling_rounds))
        p600_cv.append(measure_p600(
            brain, cv_obj, "VERB_CORE", "ROLE_PATIENT", cfg.n_settling_rounds))

    # 2. SRC vs ORC dual-binding P600
    src_dual, orc_dual = [], []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        src_dual.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_REL_AGENT", cfg.n_settling_rounds))
        orc_dual.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_REL_PATIENT", cfg.n_settling_rounds))

    # 3. Garden-path N400
    gp_n400, unamb_n400 = [], []
    for i in range(ni):
        agent = nouns[i % len(nouns)]
        rel_verb = verbs[i % len(verbs)]
        rel_patient = nouns[(i + 2) % len(nouns)]
        main_verb = verbs[(i + 1) % len(verbs)]

        # Unambiguous
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, "that", "COMP_CORE", 3)
        activate_word(brain, rel_verb, "VERB_CORE", 3)
        activate_word(brain, rel_patient, "NOUN_CORE", 3)
        activate_word(brain, agent, "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_u = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)
        unamb_n400.append(measure_n400(pred_u, lexicon[main_verb]))

        # Garden-path (no "that")
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, rel_verb, "VERB_CORE", 3)
        activate_word(brain, rel_patient, "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_g = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)
        gp_n400.append(measure_n400(pred_g, lexicon[main_verb]))

    brain.disable_plasticity = False

    return {
        "p600_gram": float(np.mean(p600_gram)),
        "p600_cv": float(np.mean(p600_cv)),
        "src_dual_p600": float(np.mean(src_dual)),
        "orc_dual_p600": float(np.mean(orc_dual)),
        "gp_n400": float(np.mean(gp_n400)),
        "unamb_n400": float(np.mean(unamb_n400)),
    }


def run_sweep(
    param_name: str,
    param_values: List,
    cfg: RobustnessConfig,
    n_seeds: int,
    base_seed: int,
    log_fn,
) -> Dict[str, Any]:
    """Run one parameter sweep."""
    results = {}

    for pval in param_values:
        # Set parameters
        if param_name == "n":
            n, k = pval, max(10, int(0.01 * pval))
            p, beta = cfg.p, cfg.beta
        elif param_name == "beta":
            n, k, p, beta = cfg.n, cfg.k, cfg.p, pval
        elif param_name == "p":
            n, k, p, beta = cfg.n, cfg.k, pval, cfg.beta
        elif param_name == "k_ratio":
            n = cfg.n
            k = max(10, int(pval * n))
            p, beta = cfg.p, cfg.beta
        else:
            raise ValueError(f"Unknown param: {param_name}")

        keys = ["p600_gram", "p600_cv", "src_dual_p600", "orc_dual_p600",
                "gp_n400", "unamb_n400"]
        vals = {k_: [] for k_ in keys}

        log_fn(f"    {param_name}={pval} (n={n}, k={k}, p={p}, beta={beta})")

        for s in range(n_seeds):
            trial = run_mini_trial(n, k, p, beta, cfg, base_seed + s)
            for k_ in keys:
                vals[k_].append(trial[k_])

        t_dissoc = paired_ttest(vals["p600_cv"], vals["p600_gram"])
        t_orc = paired_ttest(vals["orc_dual_p600"], vals["src_dual_p600"])
        t_gp = paired_ttest(vals["gp_n400"], vals["unamb_n400"])

        results[str(pval)] = {
            "n": n, "k": k, "p": p, "beta": beta,
            "dissociation_d": t_dissoc["d"],
            "orc_src_d": t_orc["d"],
            "garden_path_d": t_gp["d"],
            "tests": {
                "dissociation": t_dissoc,
                "orc_src": t_orc,
                "garden_path": t_gp,
            },
        }

        log_fn(f"      Dissoc d={t_dissoc['d']:.2f}  "
               f"ORC/SRC d={t_orc['d']:.2f}  "
               f"GP d={t_gp['d']:.2f}")

    return results


class ParameterRobustnessExperiment(ExperimentBase):
    """Parameter robustness across four dimensions."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="parameter_robustness",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 5,
        config: Optional[RobustnessConfig] = None,
        quick: bool = False,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or RobustnessConfig(
            **{k: v for k, v in kwargs.items()
               if k in RobustnessConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Parameter Robustness: Four 1D Sweeps")
        self.log(f"  Reference: n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        if quick:
            sweeps = {
                "n": [2000, 5000, 10000],
                "beta": [0.05, 0.10, 0.20],
                "p": [0.02, 0.05, 0.10],
                "k_ratio": [0.005, 0.01, 0.02],
            }
        else:
            sweeps = {
                "n": [2000, 5000, 10000, 20000],
                "beta": [0.05, 0.10, 0.15, 0.20],
                "p": [0.02, 0.05, 0.10],
                "k_ratio": [0.005, 0.01, 0.02],
            }

        all_results = {}
        for param_name, param_values in sweeps.items():
            self.log(f"\n  === Sweep: {param_name} ===")
            all_results[param_name] = run_sweep(
                param_name, param_values, cfg, n_seeds,
                self.seed, self.log)

        # Count robust points (all three effects d > 0.5)
        n_total = 0
        n_robust = 0
        for sweep_name, sweep_results in all_results.items():
            for pval, r in sweep_results.items():
                n_total += 1
                if (r["dissociation_d"] > 0.5 and
                        r["orc_src_d"] > 0.5 and
                        r["garden_path_d"] > 0.5):
                    n_robust += 1

        # Check reference point
        ref_results = None
        for pval, r in all_results.get("n", {}).items():
            if r["n"] == cfg.n and r["k"] == cfg.k:
                ref_results = r
                break

        h1 = (ref_results is not None and
               ref_results["dissociation_d"] > 0.5 and
               ref_results["orc_src_d"] > 0.5 and
               ref_results["garden_path_d"] > 0.5)
        h2 = n_robust >= n_total * 0.5
        h3 = False
        # Check if smallest n shows collapse
        n_sweep = all_results.get("n", {})
        if n_sweep:
            sorted_ns = sorted(n_sweep.items(), key=lambda x: x[1]["n"])
            if sorted_ns:
                smallest = sorted_ns[0][1]
                if (smallest["dissociation_d"] < 0.5 or
                        smallest["orc_src_d"] < 0.5 or
                        smallest["garden_path_d"] < 0.5):
                    h3 = True

        self.log(f"\n  === Summary ===")
        self.log(f"    Robust points (all d > 0.5): {n_robust}/{n_total}")
        self.log(f"    H1 (Reference all d > 0.5): {'PASS' if h1 else 'FAIL'}")
        self.log(f"    H2 (>= 50% robust):         {'PASS' if h2 else 'FAIL'}")
        self.log(f"    H3 (Min-n collapse):         {'YES' if h3 else 'NO'}")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "sweeps": all_results,
            "n_robust": n_robust,
            "n_total": n_total,
            "hypotheses": {
                "H1_reference_robust": h1,
                "H2_majority_robust": h2,
                "H3_min_n_collapse": h3,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n_ref": cfg.n, "k_ref": cfg.k,
                "p_ref": cfg.p, "beta_ref": cfg.beta,
                "n_seeds": n_seeds,
                "quick": quick,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Parameter Robustness Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = ParameterRobustnessExperiment(verbose=True)

    if args.quick:
        cfg = RobustnessConfig(
            n=5000, k=50,
            n_train_sentences=50, training_reps=2)
        n_seeds = args.seeds or 3
    else:
        cfg = RobustnessConfig()
        n_seeds = args.seeds or 5

    result = exp.run(n_seeds=n_seeds, config=cfg, quick=args.quick)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    m = result.metrics
    print("\n" + "=" * 70)
    print("PARAMETER ROBUSTNESS SUMMARY")
    print("=" * 70)

    for sweep_name, sweep_results in m["sweeps"].items():
        print(f"\n{sweep_name}:")
        for pval, r in sweep_results.items():
            print(f"  {pval}: dissoc={r['dissociation_d']:.2f}"
                  f"  orc/src={r['orc_src_d']:.2f}"
                  f"  gp={r['garden_path_d']:.2f}")

    print(f"\nRobust points: {m['n_robust']}/{m['n_total']}")
    h = m["hypotheses"]
    print(f"H1 Reference robust: {'PASS' if h['H1_reference_robust'] else 'FAIL'}")
    print(f"H2 Majority robust:  {'PASS' if h['H2_majority_robust'] else 'FAIL'}")
    print(f"H3 Min-n collapse:   {'YES' if h['H3_min_n_collapse'] else 'NO'}")

    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
