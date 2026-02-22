"""
Incremental Language Learning with Continuous ERP Measurement

Models the developmental trajectory of language learning where words
accumulate prediction and binding experience through sentence exposure,
and N400/P600 signals evolve as a natural consequence.

Instead of batch train -> test, sentences are processed one at a time with
plasticity ON, and ERP signals are measured at checkpoints.

This demonstrates that the AC prediction+binding mechanism naturally produces:
  - Word-frequency effects: graded N400 via accumulated co-projection experience
  - Category-level generalization: novel nouns bind correctly from first exposure
  - Persistent selectional restriction violations: verbs never fit noun slots

Architecture:
  PHON_<word>    -- stimulus for each word
  NOUN_CORE      -- noun assemblies
  VERB_CORE      -- verb assemblies
  PREDICTION     -- forward projection target
  ROLE_AGENT     -- agent slot
  ROLE_PATIENT   -- patient slot

Protocol:
  1. Build word assemblies in NOUN_CORE and VERB_CORE (lexicon formation)
  2. Process sentences incrementally with plasticity ON:
     - Each sentence trains prediction bridges AND role bindings
     - At measurement checkpoints, pause plasticity and measure ERPs
  3. Output: learning curves showing N400 and P600 at each checkpoint

Hypotheses:
  H1: Novel noun N400 decreases monotonically with exposure
  H2: Category violation N400 stays high regardless of exposure
  H3: P600 for novel nouns stays low (correct category bindings transfer)
  H4: P600 for category violations stays high
  H5: Grammatical N400 and P600 both stay low throughout

Usage:
    uv run python research/experiments/primitives/test_incremental_erp.py
    uv run python research/experiments/primitives/test_incremental_erp.py --quick
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
    summarize,
    paired_ttest,
)
from research.experiments.metrics.instability import compute_anchored_instability
from src.core.brain import Brain


NOUNS = ["dog", "cat", "bird", "boy", "girl"]
VERBS = ["chases", "sees", "eats", "finds", "hits"]
NOVEL_NOUNS = ["table", "chair"]


@dataclass
class IncrementalConfig:
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
    # Measurement checkpoints: measure ERPs after this many sentences
    measurement_points: tuple = (0, 5, 15, 30, 60)
    # Fraction of sentences that include novel nouns
    novel_fraction: float = 0.3
    # Number of test triples per measurement point
    n_test_triples: int = 5


def _activate_word(brain: Brain, stim_name: str, area: str, rounds: int):
    """Activate a word's assembly in its core area via stimulus projection."""
    brain.inhibit_areas([area])
    for _ in range(rounds):
        brain.project({stim_name: [area]}, {area: [area]})


def generate_sentence_pool(
    n_sentences: int,
    novel_fraction: float,
    rng: np.random.Generator,
) -> List[Tuple[str, str, str]]:
    """Generate a pool of SVO sentences, mixing familiar and novel nouns.

    novel_fraction of sentences will use novel nouns in object position.
    """
    sentences = []
    n_novel = int(n_sentences * novel_fraction)
    n_familiar = n_sentences - n_novel

    # Familiar sentences (all trained nouns)
    for _ in range(n_familiar):
        agent = rng.choice(NOUNS)
        patient = rng.choice([n for n in NOUNS if n != agent])
        verb = rng.choice(VERBS)
        sentences.append((agent, verb, patient))

    # Novel noun sentences (novel nouns in object position)
    for _ in range(n_novel):
        agent = rng.choice(NOUNS)
        verb = rng.choice(VERBS)
        patient = rng.choice(NOVEL_NOUNS)
        sentences.append((agent, verb, patient))

    # Shuffle to interleave
    rng.shuffle(sentences)
    return sentences


def generate_test_triples(
    rng: np.random.Generator,
    n_triples: int = 5,
) -> List[Tuple[str, str, str, str, str]]:
    """Generate matched test triples: (agent, verb, gram_obj, catviol_obj, novel_obj)."""
    triples = []
    nouns = list(NOUNS)
    verbs = list(VERBS)
    novels = list(NOVEL_NOUNS)

    for i in range(n_triples):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        remaining = [n for n in nouns if n != agent]
        gram_obj = remaining[i % len(remaining)]
        catviol_obj = verbs[(i + 1) % len(verbs)]
        novel_obj = novels[i % len(novels)]
        triples.append((agent, verb, gram_obj, catviol_obj, novel_obj))

    return triples


def _train_sentence(brain: Brain, cfg: IncrementalConfig,
                    agent: str, verb_word: str, patient: str):
    """Process one sentence with plasticity ON: prediction bridges + role bindings."""
    # Determine patient's core area
    patient_area = ("NOUN_CORE" if patient in NOUNS or patient in NOVEL_NOUNS
                    else "VERB_CORE")

    # --- Prediction training ---
    # Activate agent, co-project with verb stimulus -> PREDICTION
    _activate_word(brain, f"PHON_{agent}", "NOUN_CORE", 3)
    brain.inhibit_areas(["PREDICTION"])
    for _ in range(cfg.train_rounds_per_pair):
        brain.project(
            {f"PHON_{verb_word}": ["PREDICTION"]},
            {"NOUN_CORE": ["PREDICTION"]},
        )

    # Activate verb, co-project with patient stimulus -> PREDICTION
    _activate_word(brain, f"PHON_{verb_word}", "VERB_CORE", 3)
    brain.inhibit_areas(["PREDICTION"])
    for _ in range(cfg.train_rounds_per_pair):
        brain.project(
            {f"PHON_{patient}": ["PREDICTION"]},
            {"VERB_CORE": ["PREDICTION"]},
        )

    # --- Binding training ---
    # Bind agent -> ROLE_AGENT
    _activate_word(brain, f"PHON_{agent}", "NOUN_CORE", 3)
    brain.inhibit_areas(["ROLE_AGENT"])
    for _ in range(cfg.binding_rounds):
        brain.project(
            {f"PHON_{agent}": ["NOUN_CORE", "ROLE_AGENT"]},
            {"NOUN_CORE": ["ROLE_AGENT"],
             "ROLE_AGENT": ["NOUN_CORE"]},
        )

    # Bind patient -> ROLE_PATIENT
    _activate_word(brain, f"PHON_{patient}", patient_area, 3)
    brain.inhibit_areas(["ROLE_PATIENT"])
    for _ in range(cfg.binding_rounds):
        brain.project(
            {f"PHON_{patient}": [patient_area, "ROLE_PATIENT"]},
            {patient_area: ["ROLE_PATIENT"],
             "ROLE_PATIENT": [patient_area]},
        )


def _build_lexicon(brain: Brain, cfg: IncrementalConfig) -> Dict[str, np.ndarray]:
    """Build prediction lexicon with plasticity OFF."""
    lexicon = {}
    for word in NOUNS + VERBS + NOVEL_NOUNS:
        brain.inhibit_areas(["PREDICTION"])
        for _ in range(cfg.lexicon_readout_rounds):
            brain.project({f"PHON_{word}": ["PREDICTION"]}, {})
        lexicon[word] = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)
    return lexicon


def _measure_erps(
    brain: Brain,
    cfg: IncrementalConfig,
    lexicon: Dict[str, np.ndarray],
    test_triples: List[Tuple[str, str, str, str, str]],
) -> Dict[str, float]:
    """Measure N400 and P600 at the critical object position.

    Plasticity must be OFF before calling.
    Returns mean N400 and P600 for gram, catviol, and novel conditions.
    """
    noun_refs = [lexicon[n] for n in NOUNS]

    n400_gram, n400_catviol, n400_novel = [], [], []
    p600_gram, p600_catviol, p600_novel = [], [], []

    for agent, verb_word, gram_obj, catviol_obj, novel_obj in test_triples:
        # Position 1: subject
        _activate_word(brain, f"PHON_{agent}", "NOUN_CORE", 3)

        # Position 2: verb -> forward predict object
        _activate_word(brain, f"PHON_{verb_word}", "VERB_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"VERB_CORE": ["PREDICTION"]})
        predicted = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

        # Position 3: critical word -- measure N400 and P600 per condition

        # Grammatical
        n400_gram.append(1.0 - measure_overlap(predicted, lexicon[gram_obj]))
        bd_g = compute_anchored_instability(
            brain, gram_obj, "NOUN_CORE", "ROLE_PATIENT",
            cfg.n_settling_rounds)
        p600_gram.append(bd_g["instability"])

        # Category violation
        n400_catviol.append(
            1.0 - measure_overlap(predicted, lexicon[catviol_obj]))
        bd_c = compute_anchored_instability(
            brain, catviol_obj, "VERB_CORE", "ROLE_PATIENT",
            cfg.n_settling_rounds)
        p600_catviol.append(bd_c["instability"])

        # Novel
        n400_novel.append(
            1.0 - measure_overlap(predicted, lexicon[novel_obj]))
        bd_n = compute_anchored_instability(
            brain, novel_obj, "NOUN_CORE", "ROLE_PATIENT",
            cfg.n_settling_rounds)
        p600_novel.append(bd_n["instability"])

    return {
        "n400_gram": float(np.mean(n400_gram)),
        "n400_catviol": float(np.mean(n400_catviol)),
        "n400_novel": float(np.mean(n400_novel)),
        "p600_gram": float(np.mean(p600_gram)),
        "p600_catviol": float(np.mean(p600_catviol)),
        "p600_novel": float(np.mean(p600_novel)),
    }


def run_trial(
    cfg: IncrementalConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run one incremental learning trial.

    Processes sentences one at a time, measuring ERPs at checkpoints.
    Returns learning curves: ERP values at each measurement point.
    """
    brain = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    rng = np.random.default_rng(seed)

    # Create areas
    brain.add_area("NOUN_CORE", cfg.n, cfg.k, cfg.beta)
    brain.add_area("VERB_CORE", cfg.n, cfg.k, cfg.beta)
    brain.add_area("PREDICTION", cfg.n, cfg.k, cfg.beta)
    brain.add_area("ROLE_AGENT", cfg.n, cfg.k, cfg.beta)
    brain.add_area("ROLE_PATIENT", cfg.n, cfg.k, cfg.beta)

    # Register stimuli
    for noun in NOUNS + NOVEL_NOUNS:
        brain.add_stimulus(f"PHON_{noun}", cfg.k)
    for verb in VERBS:
        brain.add_stimulus(f"PHON_{verb}", cfg.k)

    # Build word assemblies (lexicon formation -- separate from sentence learning)
    for noun in NOUNS:
        brain._engine.reset_area_connections("NOUN_CORE")
        _activate_word(brain, f"PHON_{noun}", "NOUN_CORE", cfg.lexicon_rounds)

    for verb in VERBS:
        brain._engine.reset_area_connections("VERB_CORE")
        _activate_word(brain, f"PHON_{verb}", "VERB_CORE", cfg.lexicon_rounds)

    for noun in NOVEL_NOUNS:
        brain._engine.reset_area_connections("NOUN_CORE")
        _activate_word(brain, f"PHON_{noun}", "NOUN_CORE", cfg.lexicon_rounds)

    # Generate sentence pool (enough for the max measurement point)
    max_sentences = max(cfg.measurement_points)
    sentence_pool = generate_sentence_pool(max_sentences, cfg.novel_fraction, rng)

    # Generate test triples (fixed across measurement points for comparability)
    test_triples = generate_test_triples(rng, cfg.n_test_triples)

    # Learning curve: measure ERPs at each checkpoint
    curves = {}
    sentences_processed = 0

    for checkpoint in sorted(cfg.measurement_points):
        # Train up to this checkpoint
        while sentences_processed < checkpoint:
            agent, verb_word, patient = sentence_pool[sentences_processed]
            _train_sentence(brain, cfg, agent, verb_word, patient)
            sentences_processed += 1

        # Measure ERPs (plasticity OFF)
        brain.disable_plasticity = True
        lexicon = _build_lexicon(brain, cfg)
        erps = _measure_erps(brain, cfg, lexicon, test_triples)
        brain.disable_plasticity = False

        curves[checkpoint] = erps

    return {"curves": curves}


class IncrementalERPExperiment(ExperimentBase):
    """Incremental language learning with ERP measurement at checkpoints."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="incremental_erp",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[IncrementalConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or IncrementalConfig(
            **{k: v for k, v in kwargs.items()
               if k in IncrementalConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Incremental Language Learning with ERP Measurement")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  measurement_points={cfg.measurement_points}")
        self.log(f"  novel_fraction={cfg.novel_fraction}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        # Collect per-checkpoint, per-seed values
        checkpoints = sorted(cfg.measurement_points)
        all_curves = {cp: {
            "n400_gram": [], "n400_catviol": [], "n400_novel": [],
            "p600_gram": [], "p600_catviol": [], "p600_novel": [],
        } for cp in checkpoints}

        for s in range(n_seeds):
            self.log(f"  Seed {s+1}/{n_seeds} ...")
            result = run_trial(cfg, self.seed + s)

            for cp in checkpoints:
                erps = result["curves"][cp]
                for key in all_curves[cp]:
                    all_curves[cp][key].append(erps[key])

        # Report learning curves
        self.log(f"\n  {'Sent':>5}  "
                 f"{'N400_g':>7}{'N400_c':>7}{'N400_n':>7} | "
                 f"{'P600_g':>7}{'P600_c':>7}{'P600_n':>7}")
        self.log("  " + "-" * 55)

        for cp in checkpoints:
            cv = all_curves[cp]
            self.log(f"  {cp:>5}  "
                     f"{np.mean(cv['n400_gram']):>7.3f}"
                     f"{np.mean(cv['n400_catviol']):>7.3f}"
                     f"{np.mean(cv['n400_novel']):>7.3f} | "
                     f"{np.mean(cv['p600_gram']):>7.3f}"
                     f"{np.mean(cv['p600_catviol']):>7.3f}"
                     f"{np.mean(cv['p600_novel']):>7.3f}")

        # Statistical tests at final checkpoint
        final = all_curves[checkpoints[-1]]

        h1_test = paired_ttest(final["n400_novel"], final["n400_gram"])
        h2_test = paired_ttest(final["n400_catviol"], final["n400_gram"])
        h3_test = paired_ttest(final["p600_novel"], final["p600_gram"])
        h4_test = paired_ttest(final["p600_catviol"], final["p600_gram"])

        self.log(f"\n  Statistical tests at final checkpoint "
                 f"({checkpoints[-1]} sentences):")
        self.log(f"    H1 Novel N400 > Gram N400:     "
                 f"d={h1_test['d']:.2f}, p={h1_test['p']:.4f}")
        self.log(f"    H2 CatViol N400 > Gram N400:   "
                 f"d={h2_test['d']:.2f}, p={h2_test['p']:.4f}")
        self.log(f"    H3 Novel P600 ~ Gram P600:     "
                 f"d={h3_test['d']:.2f}, p={h3_test['p']:.4f}")
        self.log(f"    H4 CatViol P600 > Gram P600:   "
                 f"d={h4_test['d']:.2f}, p={h4_test['p']:.4f}")

        # Check if novel N400 decreased from first to last checkpoint
        first_cp = checkpoints[0]
        last_cp = checkpoints[-1]
        novel_n400_first = all_curves[first_cp]["n400_novel"]
        novel_n400_last = all_curves[last_cp]["n400_novel"]
        learning_test = paired_ttest(novel_n400_first, novel_n400_last)
        self.log(f"\n  Learning effect (novel N400 decrease):")
        self.log(f"    First ({first_cp}): {np.mean(novel_n400_first):.4f}")
        self.log(f"    Last ({last_cp}):  {np.mean(novel_n400_last):.4f}")
        self.log(f"    d={learning_test['d']:.2f}, p={learning_test['p']:.4f}")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        # Build metrics
        curve_metrics = {}
        for cp in checkpoints:
            cv = all_curves[cp]
            curve_metrics[f"checkpoint_{cp}"] = {
                "n400_gram": summarize(cv["n400_gram"]),
                "n400_catviol": summarize(cv["n400_catviol"]),
                "n400_novel": summarize(cv["n400_novel"]),
                "p600_gram": summarize(cv["p600_gram"]),
                "p600_catviol": summarize(cv["p600_catviol"]),
                "p600_novel": summarize(cv["p600_novel"]),
            }

        metrics = {
            "curves": curve_metrics,
            "final_tests": {
                "h1_novel_n400_vs_gram": h1_test,
                "h2_catviol_n400_vs_gram": h2_test,
                "h3_novel_p600_vs_gram": h3_test,
                "h4_catviol_p600_vs_gram": h4_test,
            },
            "learning_effect": {
                "novel_n400_first": float(np.mean(novel_n400_first)),
                "novel_n400_last": float(np.mean(novel_n400_last)),
                "test": learning_test,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "w_max": cfg.w_max,
                "lexicon_rounds": cfg.lexicon_rounds,
                "train_rounds_per_pair": cfg.train_rounds_per_pair,
                "binding_rounds": cfg.binding_rounds,
                "n_settling_rounds": cfg.n_settling_rounds,
                "measurement_points": list(cfg.measurement_points),
                "novel_fraction": cfg.novel_fraction,
                "n_test_triples": cfg.n_test_triples,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Incremental Language Learning ERP Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = IncrementalERPExperiment(verbose=True)

    if args.quick:
        cfg = IncrementalConfig(
            n=5000, k=50,
            measurement_points=(0, 3, 10, 20),
            n_test_triples=3,
        )
        n_seeds = args.seeds or 3
    else:
        cfg = IncrementalConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)

    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    print("\n" + "=" * 70)
    print("INCREMENTAL LEARNING ERP SUMMARY")
    print("=" * 70)

    m = result.metrics

    print(f"\nLearning curves (mean across seeds):")
    print(f"  {'Sent':>5}  "
          f"{'N400_g':>7}{'N400_c':>7}{'N400_n':>7} | "
          f"{'P600_g':>7}{'P600_c':>7}{'P600_n':>7}")
    print("  " + "-" * 55)

    for cp_key in sorted(m["curves"].keys(),
                         key=lambda x: int(x.split("_")[1])):
        cp_num = int(cp_key.split("_")[1])
        cv = m["curves"][cp_key]
        print(f"  {cp_num:>5}  "
              f"{cv['n400_gram']['mean']:>7.3f}"
              f"{cv['n400_catviol']['mean']:>7.3f}"
              f"{cv['n400_novel']['mean']:>7.3f} | "
              f"{cv['p600_gram']['mean']:>7.3f}"
              f"{cv['p600_catviol']['mean']:>7.3f}"
              f"{cv['p600_novel']['mean']:>7.3f}")

    ft = m["final_tests"]
    print(f"\nFinal checkpoint tests:")
    print(f"  H1 Novel N400 > Gram:     d={ft['h1_novel_n400_vs_gram']['d']:.2f}, "
          f"p={ft['h1_novel_n400_vs_gram']['p']:.4f}")
    print(f"  H2 CatViol N400 > Gram:   d={ft['h2_catviol_n400_vs_gram']['d']:.2f}, "
          f"p={ft['h2_catviol_n400_vs_gram']['p']:.4f}")
    print(f"  H3 Novel P600 ~ Gram:     d={ft['h3_novel_p600_vs_gram']['d']:.2f}, "
          f"p={ft['h3_novel_p600_vs_gram']['p']:.4f}")
    print(f"  H4 CatViol P600 > Gram:   d={ft['h4_catviol_p600_vs_gram']['d']:.2f}, "
          f"p={ft['h4_catviol_p600_vs_gram']['p']:.4f}")

    le = m["learning_effect"]
    print(f"\nLearning effect (novel N400):")
    print(f"  First: {le['novel_n400_first']:.4f}")
    print(f"  Last:  {le['novel_n400_last']:.4f}")
    print(f"  d={le['test']['d']:.2f}, p={le['test']['p']:.4f}")

    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
