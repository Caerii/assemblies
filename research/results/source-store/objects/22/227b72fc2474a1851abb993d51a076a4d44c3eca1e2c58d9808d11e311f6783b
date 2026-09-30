"""
Role Area Reuse via LRI Wraparound

Tests whether fewer STRUCT areas than noun positions works via LRI
refractory period decay. With 3 STRUCT areas and refractory_period=2,
the first area's LRI should decay by the time the 4th noun arrives.

Setup:
  - Create brain with N STRUCT areas (N < number of noun positions)
  - Train on long sentences (recursive PP: "dog chases cat in park near river")
  - Measure routing success, position disambiguation, and binding quality

Hypotheses:
  H1: Reuse works (>90% routing success with 3 areas, 4 nouns)
  H2: LRI period controls reuse timing
  H3: Binding quality maintained at reused positions

Usage:
    uv run python research/experiments/primitives/test_role_reuse.py
    uv run python research/experiments/primitives/test_role_reuse.py --quick
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
    measure_overlap,
)
from research.experiments.lib.vocabulary import DEFAULT_VOCAB, RECURSIVE_VOCAB
from research.experiments.lib.grammar import RecursiveCFG
from research.experiments.lib.brain_setup import activate_word
from research.experiments.lib.unsupervised import (
    UnsupervisedConfig,
    create_unsupervised_brain,
    discover_role_area,
    train_sentence_unsupervised,
)


@dataclass
class RoleReuseConfig:
    # Brain
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.15
    w_max: float = 20.0
    lexicon_rounds: int = 20
    # Training
    n_train_sentences: int = 100
    train_rounds_per_pair: int = 5
    binding_rounds: int = 10
    stabilize_rounds: int = 3
    inhibition_strength: float = 1.0
    # Test
    n_test_sentences: int = 30


def count_noun_positions(sentence: Dict[str, Any]) -> int:
    """Count the number of noun/location positions in a sentence."""
    return sum(1 for c in sentence["categories"] if c in ("NOUN", "LOCATION"))


def test_routing_success(
    brain,
    struct_areas: List[str],
    sentences: List[Dict[str, Any]],
    vocab,
    stabilize_rounds: int,
) -> Dict[str, Any]:
    """Test how often routing succeeds for each noun position."""
    brain.disable_plasticity = True

    position_successes = {}  # noun_pos -> [True/False, ...]
    position_areas = {}  # noun_pos -> [area_name, ...]

    for sent in sentences:
        words = sent["words"]
        categories = sent["categories"]

        for name in struct_areas:
            brain.clear_refractory(name)
        brain.inhibit_areas(struct_areas)

        noun_idx = 0
        for w, c in zip(words, categories):
            if c in ("NOUN", "LOCATION"):
                core = vocab.core_area_for(w)
                activate_word(brain, w, core, 3)
                winner = discover_role_area(
                    brain, "NOUN_MARKER", struct_areas, stabilize_rounds)

                position_successes.setdefault(noun_idx, []).append(
                    winner is not None)
                if winner is not None:
                    position_areas.setdefault(noun_idx, []).append(winner)
                noun_idx += 1

    brain.disable_plasticity = False

    # Compute success rates per position
    success_rates = {}
    for pos, results in sorted(position_successes.items()):
        success_rates[pos] = float(np.mean(results))

    return {
        "success_rates": success_rates,
        "position_areas": {
            pos: list(set(areas))
            for pos, areas in position_areas.items()
        },
        "overall_success": float(np.mean([
            v for results in position_successes.values() for v in results
        ])),
    }


def test_lri_scaling(
    cfg: RoleReuseConfig,
    vocab,
    seed: int,
    n_struct_values: List[int],
    refractory_values: List[int],
    n_test: int,
) -> List[Dict[str, Any]]:
    """Test different n_struct x refractory_period combinations."""
    rng = np.random.default_rng(seed)
    results = []

    for n_struct in n_struct_values:
        for ref_period in refractory_values:
            ucfg = UnsupervisedConfig(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds,
                n_struct_areas=n_struct,
                refractory_period=ref_period,
                inhibition_strength=cfg.inhibition_strength,
                stabilize_rounds=cfg.stabilize_rounds,
                train_rounds_per_pair=cfg.train_rounds_per_pair,
                binding_rounds=cfg.binding_rounds,
            )
            brain, struct_areas = create_unsupervised_brain(
                ucfg, vocab, seed)

            # Generate long sentences with recursive PP
            grammar = RecursiveCFG(
                pp_prob=0.9, recursive_pp_prob=0.8,
                rel_prob=0.0, max_pp_depth=3,
                vocab=vocab,
                rng=np.random.default_rng(seed + 1000),
            )
            # Filter for sentences with enough noun positions
            min_nouns = n_struct + 1  # need more nouns than areas
            test_sents = []
            for _ in range(n_test * 10):
                s = grammar.generate()
                if count_noun_positions(s) >= min_nouns:
                    test_sents.append(s)
                if len(test_sents) >= n_test:
                    break

            if not test_sents:
                results.append({
                    "n_struct": n_struct,
                    "refractory_period": ref_period,
                    "success_rates": {},
                    "overall_success": 0.0,
                    "n_test_sents": 0,
                    "max_nouns_tested": 0,
                })
                continue

            # Train on some sentences first
            train_grammar = RecursiveCFG(
                pp_prob=0.7, recursive_pp_prob=0.5,
                rel_prob=0.0, max_pp_depth=3,
                vocab=vocab, rng=rng,
            )
            train_sents = train_grammar.generate_batch(cfg.n_train_sentences)
            for sent in train_sents:
                train_sentence_unsupervised(
                    brain, sent, vocab, struct_areas, ucfg)

            # Test routing
            routing = test_routing_success(
                brain, struct_areas, test_sents, vocab,
                cfg.stabilize_rounds)

            max_nouns = max(count_noun_positions(s) for s in test_sents)

            results.append({
                "n_struct": n_struct,
                "refractory_period": ref_period,
                "success_rates": routing["success_rates"],
                "overall_success": routing["overall_success"],
                "n_test_sents": len(test_sents),
                "max_nouns_tested": max_nouns,
            })

    return results


class RoleReuseExperiment(ExperimentBase):
    """Role area reuse experiment."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="role_reuse",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 5,
        config: Optional[RoleReuseConfig] = None,
        n_struct_values: List[int] = None,
        refractory_values: List[int] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or RoleReuseConfig(
            **{k: v for k, v in kwargs.items()
               if k in RoleReuseConfig.__dataclass_fields__})

        if n_struct_values is None:
            n_struct_values = [2, 3, 4, 6]
        if refractory_values is None:
            refractory_values = [1, 2, 3, 5]

        vocab = RECURSIVE_VOCAB

        self.log("=" * 70)
        self.log("Role Area Reuse via LRI Wraparound")
        self.log(f"  n={cfg.n}, k={cfg.k}")
        self.log(f"  n_struct values: {n_struct_values}")
        self.log(f"  refractory values: {refractory_values}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        # Accumulate across seeds
        all_results = []

        for s in range(n_seeds):
            self.log(f"  Seed {s + 1}/{n_seeds} ...")
            seed_results = test_lri_scaling(
                cfg, vocab, self.seed + s,
                n_struct_values, refractory_values,
                cfg.n_test_sentences,
            )
            all_results.append(seed_results)

        # Aggregate: for each (n_struct, ref_period), average success rates
        config_results = {}
        for seed_results in all_results:
            for r in seed_results:
                key = (r["n_struct"], r["refractory_period"])
                config_results.setdefault(key, []).append(r)

        aggregated = {}
        for (ns, rp), runs in sorted(config_results.items()):
            overall = [r["overall_success"] for r in runs]
            # Per-position success rates (positions that exist across runs)
            all_positions = set()
            for r in runs:
                all_positions.update(r["success_rates"].keys())

            pos_rates = {}
            for pos in sorted(all_positions):
                rates = [r["success_rates"].get(pos, 0.0) for r in runs
                         if pos in r["success_rates"]]
                if rates:
                    pos_rates[pos] = float(np.mean(rates))

            agg = {
                "n_struct": ns,
                "refractory_period": rp,
                "overall_success": summarize(overall) if len(overall) > 1 else {"mean": overall[0] if overall else 0.0},
                "position_success_rates": pos_rates,
            }
            aggregated[f"ns{ns}_rp{rp}"] = agg

        # Report
        self.log(f"\n  {'n_struct':>8} | {'ref_per':>7} | {'overall':>8} | per-position success")
        self.log(f"  {'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*30}")
        for key, agg in sorted(aggregated.items()):
            overall_mean = (agg["overall_success"]["mean"]
                           if isinstance(agg["overall_success"], dict)
                           else agg["overall_success"])
            pos_str = ", ".join(
                f"p{p}={r:.0%}"
                for p, r in sorted(agg["position_success_rates"].items())
            )
            self.log(f"  {agg['n_struct']:>8} | {agg['refractory_period']:>7} | "
                     f"{overall_mean:>7.1%} | {pos_str}")

        # Hypotheses (using n_struct=3, refractory=2 as primary test)
        primary_key = "ns3_rp2"
        if primary_key in aggregated:
            primary = aggregated[primary_key]
            overall_mean = (primary["overall_success"]["mean"]
                           if isinstance(primary["overall_success"], dict)
                           else primary["overall_success"])
            pos_rates = primary["position_success_rates"]

            h1 = overall_mean > 0.9
            # H2: later positions (3+) should have non-zero success
            h2 = any(pos_rates.get(p, 0.0) > 0.5 for p in range(3, 10))
            # H3: reused positions have comparable success to early positions
            early_rate = np.mean([pos_rates.get(p, 0.0) for p in [0, 1] if p in pos_rates]) if pos_rates else 0
            late_rate = np.mean([pos_rates.get(p, 0.0) for p in range(3, 10) if p in pos_rates]) if any(p in pos_rates for p in range(3, 10)) else 0
            h3 = late_rate > early_rate * 0.5 if early_rate > 0 else False
        else:
            h1 = h2 = h3 = False
            overall_mean = 0.0

        self.log(f"\n  === Hypotheses (n_struct=3, ref_period=2) ===")
        self.log(f"    H1 (>90% routing success):     "
                 f"{'PASS' if h1 else 'FAIL'} ({overall_mean:.1%})")
        self.log(f"    H2 (Reuse at position 3+):     "
                 f"{'PASS' if h2 else 'FAIL'}")
        self.log(f"    H3 (Reused quality comparable): "
                 f"{'PASS' if h3 else 'FAIL'}")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "configurations": aggregated,
            "hypotheses": {
                "H1_routing_success": h1,
                "H2_lri_reuse": h2,
                "H3_quality_maintained": h3,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "n_struct_values": n_struct_values,
                "refractory_values": refractory_values,
                "n_train_sentences": cfg.n_train_sentences,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Role Area Reuse Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = RoleReuseExperiment(verbose=True)

    if args.quick:
        cfg = RoleReuseConfig(
            n=5000, k=50,
            n_train_sentences=50,
            n_test_sentences=15)
        n_struct_values = [2, 3, 4]
        refractory_values = [1, 2, 3]
        n_seeds = args.seeds or 3
    else:
        cfg = RoleReuseConfig()
        n_struct_values = [2, 3, 4, 6]
        refractory_values = [1, 2, 3, 5]
        n_seeds = args.seeds or 5

    result = exp.run(
        n_seeds=n_seeds, config=cfg,
        n_struct_values=n_struct_values,
        refractory_values=refractory_values,
    )
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    h = result.metrics["hypotheses"]
    print("\n" + "=" * 70)
    print("ROLE REUSE SUMMARY")
    print("=" * 70)
    print(f"\nH1 Routing success >90%:    {'PASS' if h['H1_routing_success'] else 'FAIL'}")
    print(f"H2 LRI reuse at pos 3+:     {'PASS' if h['H2_lri_reuse'] else 'FAIL'}")
    print(f"H3 Quality maintained:      {'PASS' if h['H3_quality_maintained'] else 'FAIL'}")
    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
