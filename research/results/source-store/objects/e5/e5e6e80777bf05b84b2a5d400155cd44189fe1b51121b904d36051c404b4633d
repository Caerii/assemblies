"""
Incremental Variable-Length Learning with Context-Free Grammar

Generates training sentences from a simple context-free grammar (CFG) that
produces variable-length utterances with optional PPs. Sentences are processed
one at a time with plasticity ON; ERP signals are measured at checkpoints
to track learning curves.

This tests whether the AC prediction+binding mechanism can implicitly learn
to process CFG-generated structure from distributional evidence alone — no
grammar rules are encoded in the architecture.

Grammar:
  S  -> NP VP
  VP -> V NP (PP)?    [PP with probability pp_prob]
  PP -> P NP

Architecture (vocabulary-driven, 7 areas + PREDICTION):
  NOUN_CORE, VERB_CORE, PREP_CORE  -- lexical assemblies
  PREDICTION                        -- forward projection target
  ROLE_AGENT, ROLE_PATIENT, ROLE_PP_OBJ -- structural role slots

Hypotheses:
  H1: N400 at all positions decreases with exposure (word-frequency effect)
  H2: P600 for category violations stays high regardless of exposure
  H3: P600 for novel nouns stays low (correct-category bindings transfer)
  H4: PP-position ERPs develop alongside SVO ERPs (no special handling)

Usage:
    uv run python research/experiments/primitives/test_variable_incremental.py
    uv run python research/experiments/primitives/test_variable_incremental.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    summarize,
    paired_ttest,
)
from research.experiments.lib.vocabulary import DEFAULT_VOCAB
from research.experiments.lib.grammar import SimpleCFG
from research.experiments.lib.brain_setup import (
    BrainConfig,
    create_language_brain,
    build_lexicon,
    activate_word,
)
from research.experiments.lib.training import train_sentence
from research.experiments.lib.measurement import (
    measure_erps_at_position,
    generate_test_triples,
    generate_pp_test_triples,
)


@dataclass
class VariableIncrementalConfig:
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
    pp_prob: float = 0.4
    novel_obj_prob: float = 0.2
    measurement_points: tuple = (0, 5, 15, 30, 60)
    n_test_triples: int = 5


def _measure_all_erps(
    brain,
    cfg: VariableIncrementalConfig,
    lexicon: Dict[str, np.ndarray],
    obj_triples,
    pp_triples,
) -> Dict[str, float]:
    """Measure N400/P600 at object and PP-object positions."""
    obj_results = {
        "obj_n400_gram": [], "obj_n400_catviol": [], "obj_n400_novel": [],
        "obj_p600_gram": [], "obj_p600_catviol": [], "obj_p600_novel": [],
    }
    pp_results = {
        "pp_n400_gram": [], "pp_n400_catviol": [], "pp_n400_novel": [],
        "pp_p600_gram": [], "pp_p600_catviol": [], "pp_p600_novel": [],
    }

    # Object position
    for agent, verb_word, gram_obj, cv_obj, novel_obj in obj_triples:
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, verb_word, "VERB_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"VERB_CORE": ["PREDICTION"]})
        predicted = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

        erps = measure_erps_at_position(
            brain, predicted, lexicon,
            gram_word=gram_obj, gram_core="NOUN_CORE",
            catviol_word=cv_obj, catviol_core="VERB_CORE",
            novel_word=novel_obj, novel_core="NOUN_CORE",
            role_area="ROLE_PATIENT", n_settling_rounds=cfg.n_settling_rounds)

        for k in ["n400_gram", "n400_catviol", "n400_novel",
                   "p600_gram", "p600_catviol", "p600_novel"]:
            obj_results[f"obj_{k}"].append(erps[k])

    # PP-object position
    for (agent, verb_word, patient, prep,
         gram_pp, cv_pp, novel_pp) in pp_triples:
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, verb_word, "VERB_CORE", 3)
        activate_word(brain, patient, "NOUN_CORE", 3)
        activate_word(brain, prep, "PREP_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"PREP_CORE": ["PREDICTION"]})
        predicted_pp = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

        erps = measure_erps_at_position(
            brain, predicted_pp, lexicon,
            gram_word=gram_pp, gram_core="NOUN_CORE",
            catviol_word=cv_pp, catviol_core="VERB_CORE",
            novel_word=novel_pp, novel_core="NOUN_CORE",
            role_area="ROLE_PP_OBJ", n_settling_rounds=cfg.n_settling_rounds)

        for k in ["n400_gram", "n400_catviol", "n400_novel",
                   "p600_gram", "p600_catviol", "p600_novel"]:
            pp_results[f"pp_{k}"].append(erps[k])

    combined = {}
    for k, v in {**obj_results, **pp_results}.items():
        combined[k] = float(np.mean(v))
    return combined


def run_trial(
    cfg: VariableIncrementalConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run one incremental variable-length trial."""
    rng = np.random.default_rng(seed)
    vocab = DEFAULT_VOCAB

    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain = create_language_brain(bcfg, vocab, seed)

    # Generate CFG sentence pool
    grammar = SimpleCFG(
        pp_prob=cfg.pp_prob,
        novel_obj_prob=cfg.novel_obj_prob,
        vocab=vocab, rng=rng)
    max_sentences = max(cfg.measurement_points)
    sentence_pool = grammar.generate_batch(max_sentences)

    # Fixed test items across checkpoints
    obj_triples = generate_test_triples(rng, cfg.n_test_triples, vocab)
    pp_triples = generate_pp_test_triples(rng, cfg.n_test_triples, vocab)

    curves = {}
    sentences_processed = 0
    length_counts = {"svo": 0, "svopp": 0}

    for checkpoint in sorted(cfg.measurement_points):
        while sentences_processed < checkpoint:
            sent = sentence_pool[sentences_processed]
            train_sentence(brain, sent, vocab,
                           cfg.train_rounds_per_pair, cfg.binding_rounds)
            if sent["has_pp"]:
                length_counts["svopp"] += 1
            else:
                length_counts["svo"] += 1
            sentences_processed += 1

        brain.disable_plasticity = True
        lexicon = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)
        erps = _measure_all_erps(brain, cfg, lexicon, obj_triples, pp_triples)
        brain.disable_plasticity = False

        curves[checkpoint] = erps

    return {"curves": curves, "length_counts": length_counts}


class VariableIncrementalExperiment(ExperimentBase):
    """Incremental learning from CFG-generated variable-length sentences."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="variable_incremental",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[VariableIncrementalConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or VariableIncrementalConfig(
            **{k: v for k, v in kwargs.items()
               if k in VariableIncrementalConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Incremental Variable-Length Learning (CFG-generated)")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  pp_prob={cfg.pp_prob}, novel_obj_prob={cfg.novel_obj_prob}")
        self.log(f"  measurement_points={cfg.measurement_points}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        checkpoints = sorted(cfg.measurement_points)
        erp_keys = [
            "obj_n400_gram", "obj_n400_catviol", "obj_n400_novel",
            "obj_p600_gram", "obj_p600_catviol", "obj_p600_novel",
            "pp_n400_gram", "pp_n400_catviol", "pp_n400_novel",
            "pp_p600_gram", "pp_p600_catviol", "pp_p600_novel",
        ]
        all_curves = {cp: {k: [] for k in erp_keys} for cp in checkpoints}

        for s in range(n_seeds):
            self.log(f"  Seed {s+1}/{n_seeds} ...")
            result = run_trial(cfg, self.seed + s)

            for cp in checkpoints:
                for key in erp_keys:
                    all_curves[cp][key].append(result["curves"][cp][key])

            if s == 0:
                lc = result["length_counts"]
                self.log(f"    Sentence mix: {lc['svo']} SVO, {lc['svopp']} SVO+PP")

        # Report learning curves
        for pos, prefix, label in [
            ("obj", "obj_", "Object position (word 3)"),
            ("pp", "pp_", "PP-object position (word 5)"),
        ]:
            self.log(f"\n  {label}:")
            self.log(f"  {'Sent':>5}  "
                     f"{'N400_g':>7}{'N400_c':>7}{'N400_n':>7} | "
                     f"{'P600_g':>7}{'P600_c':>7}{'P600_n':>7}")
            self.log("  " + "-" * 55)
            for cp in checkpoints:
                cv = all_curves[cp]
                self.log(f"  {cp:>5}  "
                         f"{np.mean(cv[f'{prefix}n400_gram']):>7.3f}"
                         f"{np.mean(cv[f'{prefix}n400_catviol']):>7.3f}"
                         f"{np.mean(cv[f'{prefix}n400_novel']):>7.3f} | "
                         f"{np.mean(cv[f'{prefix}p600_gram']):>7.3f}"
                         f"{np.mean(cv[f'{prefix}p600_catviol']):>7.3f}"
                         f"{np.mean(cv[f'{prefix}p600_novel']):>7.3f}")

        # Final checkpoint tests
        final = all_curves[checkpoints[-1]]
        tests = {}
        for prefix, label in [("obj_", "obj"), ("pp_", "pp")]:
            tests[f"{label}_n400_cv"] = paired_ttest(
                final[f"{prefix}n400_catviol"], final[f"{prefix}n400_gram"])
            tests[f"{label}_n400_novel"] = paired_ttest(
                final[f"{prefix}n400_novel"], final[f"{prefix}n400_gram"])
            tests[f"{label}_p600_cv"] = paired_ttest(
                final[f"{prefix}p600_catviol"], final[f"{prefix}p600_gram"])
            tests[f"{label}_p600_novel"] = paired_ttest(
                final[f"{prefix}p600_novel"], final[f"{prefix}p600_gram"])

        self.log(f"\n  Final checkpoint ({checkpoints[-1]} sentences):")
        for prefix, label in [("obj", "Object"), ("pp", "PP-obj")]:
            self.log(f"    {label}:")
            self.log(f"      N400 CatViol>Gram: d={tests[f'{prefix}_n400_cv']['d']:.2f}")
            self.log(f"      P600 CatViol>Gram: d={tests[f'{prefix}_p600_cv']['d']:.2f},"
                     f" Novel~Gram: d={tests[f'{prefix}_p600_novel']['d']:.2f}")

        # Learning effect
        first_cp, last_cp = checkpoints[0], checkpoints[-1]
        novel_first = all_curves[first_cp]["obj_n400_novel"]
        novel_last = all_curves[last_cp]["obj_n400_novel"]
        learning = paired_ttest(novel_first, novel_last)
        self.log(f"\n  Learning effect (novel N400 at object):")
        self.log(f"    First ({first_cp}): {np.mean(novel_first):.4f}")
        self.log(f"    Last ({last_cp}):  {np.mean(novel_last):.4f}")
        self.log(f"    d={learning['d']:.2f}, p={learning['p']:.4f}")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        curve_metrics = {}
        for cp in checkpoints:
            cv = all_curves[cp]
            curve_metrics[f"checkpoint_{cp}"] = {
                k: summarize(cv[k]) for k in erp_keys
            }

        metrics = {
            "curves": curve_metrics,
            "final_tests": tests,
            "learning_effect": {
                "novel_n400_first": float(np.mean(novel_first)),
                "novel_n400_last": float(np.mean(novel_last)),
                "test": learning,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "w_max": cfg.w_max, "pp_prob": cfg.pp_prob,
                "novel_obj_prob": cfg.novel_obj_prob,
                "measurement_points": list(cfg.measurement_points),
                "n_test_triples": cfg.n_test_triples,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Incremental Variable-Length ERP Experiment (CFG)")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = VariableIncrementalExperiment(verbose=True)

    if args.quick:
        cfg = VariableIncrementalConfig(
            n=5000, k=50,
            measurement_points=(0, 3, 10, 20),
            n_test_triples=3,
        )
        n_seeds = args.seeds or 3
    else:
        cfg = VariableIncrementalConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    m = result.metrics
    print("\n" + "=" * 70)
    print("INCREMENTAL VARIABLE-LENGTH SUMMARY (CFG)")
    print("=" * 70)

    for pos, prefix, label in [
        ("obj", "obj_", "Object position"),
        ("pp", "pp_", "PP-object position"),
    ]:
        print(f"\n{label}:")
        print(f"  {'Sent':>5}  "
              f"{'N400_g':>7}{'N400_c':>7}{'N400_n':>7} | "
              f"{'P600_g':>7}{'P600_c':>7}{'P600_n':>7}")
        print("  " + "-" * 55)
        for cp_key in sorted(m["curves"].keys(),
                             key=lambda x: int(x.split("_")[1])):
            cp_num = int(cp_key.split("_")[1])
            cv = m["curves"][cp_key]
            print(f"  {cp_num:>5}  "
                  f"{cv[f'{prefix}n400_gram']['mean']:>7.3f}"
                  f"{cv[f'{prefix}n400_catviol']['mean']:>7.3f}"
                  f"{cv[f'{prefix}n400_novel']['mean']:>7.3f} | "
                  f"{cv[f'{prefix}p600_gram']['mean']:>7.3f}"
                  f"{cv[f'{prefix}p600_catviol']['mean']:>7.3f}"
                  f"{cv[f'{prefix}p600_novel']['mean']:>7.3f}")

    ft = m["final_tests"]
    print(f"\nFinal: Obj P600 cv>g d={ft['obj_p600_cv']['d']:.2f},"
          f" PP P600 cv>g d={ft['pp_p600_cv']['d']:.2f}")

    le = m["learning_effect"]
    print(f"\nNovel N400 learning: {le['novel_n400_first']:.3f}"
          f" -> {le['novel_n400_last']:.3f} (d={le['test']['d']:.2f})")

    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
