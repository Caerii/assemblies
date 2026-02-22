"""
Prediction via Forward Projection

Tests whether Hebbian bridges formed during co-projection training enable
category-level next-word prediction. After training on SVO sentences where
noun contexts are co-projected with verb targets into a PREDICTION area,
projecting only the noun context should evoke assemblies overlapping with
verb reference assemblies more than noun reference assemblies.

This is the AC-native mechanism for anticipation: co-projection training
builds Hebbian bridges that create forward projections resembling expected
future input without any external supervision.

Protocol:
  1. Areas: NOUN_CORE, VERB_CORE, PREDICTION
  2. Establish word assemblies in core areas via stimulus projection
  3. Train: present SVO sentences. At each transition, co-project:
     - NOUN_CORE assembly + verb stimulus -> PREDICTION (noun predicts verb)
     - VERB_CORE assembly + noun stimulus -> PREDICTION (verb predicts noun)
     Hebbian plasticity strengthens bridges from context -> target neurons.
     No PREDICTION self-recurrence during training to prevent attractor
     collapse (all assemblies converging to the same dominant neurons).
  4. Build prediction lexicon: with plasticity OFF, project each word's
     stimulus into PREDICTION and snapshot the reference assembly.
  5. Test: project only context area -> PREDICTION (no target stimulus).
     Compare evoked assembly against lexicon entries. Category-correct
     entries should have higher overlap than category-incorrect entries.

Hypotheses:

H1: Category prediction at verb position -- After subject noun context,
    forward-projecting NOUN_CORE -> PREDICTION yields assemblies whose
    mean overlap with verb reference assemblies exceeds mean overlap with
    noun reference assemblies. Category advantage > 0.
    Null: advantage = 0 (no category-selective bridges).

H2: Category prediction at object position -- After verb context,
    forward-projecting VERB_CORE -> PREDICTION yields assemblies whose
    mean overlap with noun reference assemblies exceeds mean overlap with
    verb reference assemblies. Category advantage > 0.
    Null: advantage = 0.

Usage:
    uv run python research/experiments/primitives/test_forward_prediction.py
    uv run python research/experiments/primitives/test_forward_prediction.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    measure_overlap,
    chance_overlap,
    summarize,
    ttest_vs_null,
)
from src.core.brain import Brain


NOUNS = ["dog", "cat", "bird", "boy", "girl"]
VERBS = ["chases", "sees", "eats", "finds", "hits"]


@dataclass
class PredictionConfig:
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.10
    w_max: float = 20.0
    lexicon_rounds: int = 20
    train_rounds_per_pair: int = 5
    training_reps: int = 3
    n_train_sentences: int = 20
    lexicon_readout_rounds: int = 5


def generate_svo_sentences(
    n_sentences: int,
    rng: np.random.Generator,
) -> List[Tuple[str, str, str]]:
    """Generate random SVO triples (agent, verb, patient)."""
    sentences = []
    for _ in range(n_sentences):
        agent = rng.choice(NOUNS)
        patient = rng.choice([n for n in NOUNS if n != agent])
        verb = rng.choice(VERBS)
        sentences.append((agent, verb, patient))
    return sentences


def _activate_word(brain: Brain, stim_name: str, area: str, rounds: int):
    """Activate a word's assembly in its core area via stimulus projection."""
    brain.inhibit_areas([area])
    for _ in range(rounds):
        brain.project({stim_name: [area]}, {area: [area]})


def _lexicon_diversity(lexicon: Dict[str, np.ndarray],
                       nouns: List[str], verbs: List[str]) -> Dict[str, float]:
    """Compute within-category and across-category overlap of lexicon entries."""
    noun_pairs = []
    for i in range(len(nouns)):
        for j in range(i + 1, len(nouns)):
            noun_pairs.append(
                measure_overlap(lexicon[nouns[i]], lexicon[nouns[j]]))

    verb_pairs = []
    for i in range(len(verbs)):
        for j in range(i + 1, len(verbs)):
            verb_pairs.append(
                measure_overlap(lexicon[verbs[i]], lexicon[verbs[j]]))

    cross_pairs = []
    for n in nouns:
        for v in verbs:
            cross_pairs.append(measure_overlap(lexicon[n], lexicon[v]))

    return {
        "within_noun": float(np.mean(noun_pairs)) if noun_pairs else 0.0,
        "within_verb": float(np.mean(verb_pairs)) if verb_pairs else 0.0,
        "cross_category": float(np.mean(cross_pairs)) if cross_pairs else 0.0,
    }


def run_trial(
    cfg: PredictionConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run one forward prediction trial.

    Returns overlap metrics for verb-position and object-position prediction.
    """
    brain = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    rng = np.random.default_rng(seed)

    # Create areas
    brain.add_area("NOUN_CORE", cfg.n, cfg.k, cfg.beta)
    brain.add_area("VERB_CORE", cfg.n, cfg.k, cfg.beta)
    brain.add_area("PREDICTION", cfg.n, cfg.k, cfg.beta)

    # Register stimuli
    for noun in NOUNS:
        brain.add_stimulus(f"PHON_{noun}", cfg.k)
    for verb in VERBS:
        brain.add_stimulus(f"PHON_{verb}", cfg.k)

    # -- Build word assemblies in core areas --------------------
    for noun in NOUNS:
        brain._engine.reset_area_connections("NOUN_CORE")
        _activate_word(brain, f"PHON_{noun}", "NOUN_CORE", cfg.lexicon_rounds)

    for verb in VERBS:
        brain._engine.reset_area_connections("VERB_CORE")
        _activate_word(brain, f"PHON_{verb}", "VERB_CORE", cfg.lexicon_rounds)

    # -- Training -----------------------------------------------
    # Co-project context area + next word stimulus -> PREDICTION.
    # NO self-recurrence in PREDICTION to prevent attractor collapse.
    n_train = cfg.n_train_sentences * cfg.training_reps
    train_sents = generate_svo_sentences(n_train, rng)

    for agent, verb_word, patient in train_sents:
        # Activate agent in NOUN_CORE
        _activate_word(brain, f"PHON_{agent}", "NOUN_CORE", 3)

        # Co-project: NOUN_CORE + verb stimulus -> PREDICTION
        brain.inhibit_areas(["PREDICTION"])
        for _ in range(cfg.train_rounds_per_pair):
            brain.project(
                {f"PHON_{verb_word}": ["PREDICTION"]},
                {"NOUN_CORE": ["PREDICTION"]},
            )

        # Activate verb in VERB_CORE
        _activate_word(brain, f"PHON_{verb_word}", "VERB_CORE", 3)

        # Co-project: VERB_CORE + object stimulus -> PREDICTION
        brain.inhibit_areas(["PREDICTION"])
        for _ in range(cfg.train_rounds_per_pair):
            brain.project(
                {f"PHON_{patient}": ["PREDICTION"]},
                {"VERB_CORE": ["PREDICTION"]},
            )

    # -- Build prediction lexicon (post-training, plasticity OFF) --
    # Stimulus-only projection captures each word's PREDICTION-area
    # fingerprint as shaped by training, without modifying bridges.
    brain.disable_plasticity = True
    lexicon = {}
    for word in NOUNS + VERBS:
        brain.inhibit_areas(["PREDICTION"])
        for _ in range(cfg.lexicon_readout_rounds):
            brain.project({f"PHON_{word}": ["PREDICTION"]}, {})
        lexicon[word] = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)
    brain.disable_plasticity = False

    lex_div = _lexicon_diversity(lexicon, NOUNS, VERBS)

    # -- Testing ------------------------------------------------
    # Forward-project context area -> PREDICTION (no stimulus, no recurrence).
    # Single projection step: assembly = top-k by context->PREDICTION weights.
    brain.disable_plasticity = True

    # H1: Verb prediction (after noun context)
    verb_pos_correct = []
    verb_pos_incorrect = []
    verb_top1_correct = 0

    for noun in NOUNS:
        _activate_word(brain, f"PHON_{noun}", "NOUN_CORE", 3)

        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})

        pred = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

        v_overlaps = [measure_overlap(pred, lexicon[v]) for v in VERBS]
        n_overlaps = [measure_overlap(pred, lexicon[n]) for n in NOUNS]

        mean_v = float(np.mean(v_overlaps))
        mean_n = float(np.mean(n_overlaps))
        verb_pos_correct.append(mean_v)
        verb_pos_incorrect.append(mean_n)

        all_overlaps = {w: measure_overlap(pred, lexicon[w])
                        for w in NOUNS + VERBS}
        best_word = max(all_overlaps, key=all_overlaps.get)
        if best_word in VERBS:
            verb_top1_correct += 1

    # H2: Object prediction (after verb context)
    obj_pos_correct = []
    obj_pos_incorrect = []
    obj_top1_correct = 0

    for verb_word in VERBS:
        _activate_word(brain, f"PHON_{verb_word}", "VERB_CORE", 3)

        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"VERB_CORE": ["PREDICTION"]})

        pred = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

        n_overlaps = [measure_overlap(pred, lexicon[n]) for n in NOUNS]
        v_overlaps = [measure_overlap(pred, lexicon[v]) for v in VERBS]

        mean_n = float(np.mean(n_overlaps))
        mean_v = float(np.mean(v_overlaps))
        obj_pos_correct.append(mean_n)
        obj_pos_incorrect.append(mean_v)

        all_overlaps = {w: measure_overlap(pred, lexicon[w])
                        for w in NOUNS + VERBS}
        best_word = max(all_overlaps, key=all_overlaps.get)
        if best_word in NOUNS:
            obj_top1_correct += 1

    brain.disable_plasticity = False

    # -- Metrics ------------------------------------------------
    verb_advantage = float(
        np.mean(verb_pos_correct) - np.mean(verb_pos_incorrect))
    obj_advantage = float(
        np.mean(obj_pos_correct) - np.mean(obj_pos_incorrect))

    return {
        "verb_pos_correct_mean": float(np.mean(verb_pos_correct)),
        "verb_pos_incorrect_mean": float(np.mean(verb_pos_incorrect)),
        "verb_pos_advantage": verb_advantage,
        "verb_pos_top1_acc": verb_top1_correct / len(NOUNS),
        "obj_pos_correct_mean": float(np.mean(obj_pos_correct)),
        "obj_pos_incorrect_mean": float(np.mean(obj_pos_incorrect)),
        "obj_pos_advantage": obj_advantage,
        "obj_pos_top1_acc": obj_top1_correct / len(VERBS),
        "lexicon_diversity": lex_div,
    }


class ForwardPredictionExperiment(ExperimentBase):
    """Test category-level prediction via forward projection."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="forward_prediction",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[PredictionConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or PredictionConfig(
            **{k: v for k, v in kwargs.items()
               if k in PredictionConfig.__dataclass_fields__})

        null = chance_overlap(cfg.k, cfg.n)

        self.log("=" * 70)
        self.log("Forward Prediction via Hebbian Bridges")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  train_rounds_per_pair={cfg.train_rounds_per_pair}")
        self.log(f"  training_reps={cfg.training_reps}, "
                 f"n_train_sentences={cfg.n_train_sentences}")
        self.log(f"  chance overlap (k/n)={null:.4f}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        verb_adv_vals = []
        obj_adv_vals = []
        verb_correct_vals = []
        verb_incorrect_vals = []
        obj_correct_vals = []
        obj_incorrect_vals = []
        verb_top1_vals = []
        obj_top1_vals = []
        lex_div_vals = []

        for s in range(n_seeds):
            self.log(f"  Seed {s+1}/{n_seeds} ...")
            result = run_trial(cfg, self.seed + s)

            verb_adv_vals.append(result["verb_pos_advantage"])
            obj_adv_vals.append(result["obj_pos_advantage"])
            verb_correct_vals.append(result["verb_pos_correct_mean"])
            verb_incorrect_vals.append(result["verb_pos_incorrect_mean"])
            obj_correct_vals.append(result["obj_pos_correct_mean"])
            obj_incorrect_vals.append(result["obj_pos_incorrect_mean"])
            verb_top1_vals.append(result["verb_pos_top1_acc"])
            obj_top1_vals.append(result["obj_pos_top1_acc"])
            lex_div_vals.append(result["lexicon_diversity"])

            if s == 0:
                ld = result["lexicon_diversity"]
                self.log(f"    Lexicon diversity: "
                         f"noun-noun={ld['within_noun']:.3f}  "
                         f"verb-verb={ld['within_verb']:.3f}  "
                         f"cross={ld['cross_category']:.3f}")
                self.log(f"    Verb pred: correct={result['verb_pos_correct_mean']:.4f}  "
                         f"incorrect={result['verb_pos_incorrect_mean']:.4f}  "
                         f"adv={result['verb_pos_advantage']:.4f}  "
                         f"top1={result['verb_pos_top1_acc']:.2f}")
                self.log(f"    Obj  pred: correct={result['obj_pos_correct_mean']:.4f}  "
                         f"incorrect={result['obj_pos_incorrect_mean']:.4f}  "
                         f"adv={result['obj_pos_advantage']:.4f}  "
                         f"top1={result['obj_pos_top1_acc']:.2f}")

        verb_adv_test = ttest_vs_null(verb_adv_vals, 0.0)
        obj_adv_test = ttest_vs_null(obj_adv_vals, 0.0)

        self.log(f"\n  H1 -- Verb prediction (noun context -> verb):")
        self.log(f"    Correct overlap:   {np.mean(verb_correct_vals):.4f} "
                 f"+/- {np.std(verb_correct_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    Incorrect overlap:  {np.mean(verb_incorrect_vals):.4f} "
                 f"+/- {np.std(verb_incorrect_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    Advantage:          {np.mean(verb_adv_vals):.4f} "
                 f"(d={verb_adv_test['d']:.2f}, p={verb_adv_test['p']:.4f})")
        self.log(f"    Top-1 accuracy:     {np.mean(verb_top1_vals):.3f}")

        self.log(f"\n  H2 -- Object prediction (verb context -> noun):")
        self.log(f"    Correct overlap:   {np.mean(obj_correct_vals):.4f} "
                 f"+/- {np.std(obj_correct_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    Incorrect overlap:  {np.mean(obj_incorrect_vals):.4f} "
                 f"+/- {np.std(obj_incorrect_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    Advantage:          {np.mean(obj_adv_vals):.4f} "
                 f"(d={obj_adv_test['d']:.2f}, p={obj_adv_test['p']:.4f})")
        self.log(f"    Top-1 accuracy:     {np.mean(obj_top1_vals):.3f}")

        # Lexicon diversity summary
        mean_nn = float(np.mean([d["within_noun"] for d in lex_div_vals]))
        mean_vv = float(np.mean([d["within_verb"] for d in lex_div_vals]))
        mean_xc = float(np.mean([d["cross_category"] for d in lex_div_vals]))
        self.log(f"\n  Lexicon diversity (mean across seeds):")
        self.log(f"    noun-noun: {mean_nn:.3f}  verb-verb: {mean_vv:.3f}  "
                 f"cross: {mean_xc:.3f}")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "verb_prediction": {
                "correct_overlap": summarize(verb_correct_vals),
                "incorrect_overlap": summarize(verb_incorrect_vals),
                "advantage": summarize(verb_adv_vals),
                "advantage_test": verb_adv_test,
                "top1_accuracy": summarize(verb_top1_vals),
            },
            "object_prediction": {
                "correct_overlap": summarize(obj_correct_vals),
                "incorrect_overlap": summarize(obj_incorrect_vals),
                "advantage": summarize(obj_adv_vals),
                "advantage_test": obj_adv_test,
                "top1_accuracy": summarize(obj_top1_vals),
            },
            "lexicon_diversity": {
                "within_noun": mean_nn,
                "within_verb": mean_vv,
                "cross_category": mean_xc,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "w_max": cfg.w_max,
                "lexicon_rounds": cfg.lexicon_rounds,
                "train_rounds_per_pair": cfg.train_rounds_per_pair,
                "training_reps": cfg.training_reps,
                "n_train_sentences": cfg.n_train_sentences,
                "lexicon_readout_rounds": cfg.lexicon_readout_rounds,
                "n_seeds": n_seeds,
                "n_nouns": len(NOUNS),
                "n_verbs": len(VERBS),
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Forward Prediction Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = ForwardPredictionExperiment(verbose=True)

    if args.quick:
        cfg = PredictionConfig(
            n=5000, k=50, training_reps=3, n_train_sentences=15)
        n_seeds = args.seeds or 5
    else:
        cfg = PredictionConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)

    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    print("\n" + "=" * 70)
    print("FORWARD PREDICTION SUMMARY")
    print("=" * 70)

    m = result.metrics
    vp = m["verb_prediction"]
    op = m["object_prediction"]

    print(f"\nH1 -- Verb prediction (noun -> verb):")
    print(f"  Correct overlap:  {vp['correct_overlap']['mean']:.4f} "
          f"+/- {vp['correct_overlap']['sem']:.4f}")
    print(f"  Incorrect overlap: {vp['incorrect_overlap']['mean']:.4f} "
          f"+/- {vp['incorrect_overlap']['sem']:.4f}")
    print(f"  Advantage:         {vp['advantage']['mean']:.4f} "
          f"(d={vp['advantage_test']['d']:.2f}, "
          f"p={vp['advantage_test']['p']:.4f})")
    print(f"  Top-1 accuracy:    {vp['top1_accuracy']['mean']:.3f}")

    print(f"\nH2 -- Object prediction (verb -> noun):")
    print(f"  Correct overlap:  {op['correct_overlap']['mean']:.4f} "
          f"+/- {op['correct_overlap']['sem']:.4f}")
    print(f"  Incorrect overlap: {op['incorrect_overlap']['mean']:.4f} "
          f"+/- {op['incorrect_overlap']['sem']:.4f}")
    print(f"  Advantage:         {op['advantage']['mean']:.4f} "
          f"(d={op['advantage_test']['d']:.2f}, "
          f"p={op['advantage_test']['p']:.4f})")
    print(f"  Top-1 accuracy:    {op['top1_accuracy']['mean']:.3f}")

    ld = m["lexicon_diversity"]
    print(f"\nLexicon diversity: noun-noun={ld['within_noun']:.3f}  "
          f"verb-verb={ld['within_verb']:.3f}  "
          f"cross={ld['cross_category']:.3f}")

    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
