"""
P600 from Binding Difficulty

Tests whether binding difficulty -- measured via anchored instability during
role-slot settling -- produces graded P600-like signals across sentence
conditions.

Anchored instability: prime the role area with one stimulus-driven co-projection
round, then settle with area-to-area connections only. Trained pathways sustain
the initial pattern (low instability = low P600). Untrained pathways cannot
sustain it (high instability = high P600).

Architecture:
  PHON_<word>    — stimulus for each word
  NOUN_CORE      — noun assemblies
  VERB_CORE      — verb assemblies
  ROLE_AGENT     — agent slot
  ROLE_PATIENT   — patient slot

Protocol:
  1. Build word assemblies in NOUN_CORE and VERB_CORE
  2. Train role bindings via SVO sentences (agent->ROLE_AGENT,
     patient->ROLE_PATIENT co-projection)
  3. Test binding at critical (patient) position:
       Grammatical:  "dog chases cat"   — noun binds to ROLE_PATIENT (trained)
       CatViol:      "dog chases likes" — verb binds to ROLE_PATIENT (untrained)

Hypotheses:
  H1: anchored_instability(catviol) > anchored_instability(gram)
  H2: convergence_rounds(catviol) > convergence_rounds(gram)

Usage:
    uv run python research/experiments/primitives/test_binding_p600.py
    uv run python research/experiments/primitives/test_binding_p600.py --quick
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
    summarize,
    paired_ttest,
)
from research.experiments.metrics.instability import compute_anchored_instability
from src.core.brain import Brain


NOUNS = ["dog", "cat", "bird", "boy", "girl"]
VERBS = ["chases", "sees", "eats", "finds", "hits"]


@dataclass
class P600Config:
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.10
    w_max: float = 20.0
    lexicon_rounds: int = 20
    binding_rounds: int = 10
    training_reps: int = 3
    n_train_sentences: int = 20
    n_settling_rounds: int = 10


def generate_svo_sentences(
    n_sentences: int,
    rng: np.random.Generator,
) -> List[Tuple[str, str, str]]:
    """Generate random SVO triples from trained vocab."""
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


def _measure_binding_difficulty(
    brain: Brain,
    word: str,
    core_area: str,
    role_area: str,
    n_settling_rounds: int,
) -> Dict[str, float]:
    """Measure binding difficulty via anchored instability.

    Uses anchored instability (Phase A: stimulus co-projection, Phase B:
    area-to-area settling) as the primary P600 metric. Also computes
    convergence speed from the settling dynamics.

    Plasticity must be OFF before calling.
    """
    result = compute_anchored_instability(
        brain, word, core_area, role_area, n_settling_rounds)

    round_winners = result["round_winners"]

    # Convergence speed: first round where Jaccard >= 0.95 with previous
    convergence_round = len(round_winners)
    for i in range(1, len(round_winners)):
        prev = round_winners[i - 1]
        curr = round_winners[i]
        union = prev | curr
        if len(union) > 0:
            jaccard = len(prev & curr) / len(union)
        else:
            jaccard = 1.0
        if jaccard >= 0.95:
            convergence_round = i
            break

    return {
        "anchored_instability": result["instability"],
        "convergence_round": convergence_round,
    }


def run_trial(
    cfg: P600Config,
    seed: int,
) -> Dict[str, Any]:
    """Run one P600 binding difficulty trial.

    Returns binding instability and convergence metrics at the critical
    (patient) position for grammatical vs category violation conditions.
    """
    brain = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    rng = np.random.default_rng(seed)

    # Create areas
    brain.add_area("NOUN_CORE", cfg.n, cfg.k, cfg.beta)
    brain.add_area("VERB_CORE", cfg.n, cfg.k, cfg.beta)
    brain.add_area("ROLE_AGENT", cfg.n, cfg.k, cfg.beta)
    brain.add_area("ROLE_PATIENT", cfg.n, cfg.k, cfg.beta)

    # Register stimuli
    for noun in NOUNS:
        brain.add_stimulus(f"PHON_{noun}", cfg.k)
    for verb in VERBS:
        brain.add_stimulus(f"PHON_{verb}", cfg.k)

    # -- Build word assemblies in core areas -----------------------
    for noun in NOUNS:
        brain._engine.reset_area_connections("NOUN_CORE")
        _activate_word(brain, f"PHON_{noun}", "NOUN_CORE", cfg.lexicon_rounds)

    for verb in VERBS:
        brain._engine.reset_area_connections("VERB_CORE")
        _activate_word(brain, f"PHON_{verb}", "VERB_CORE", cfg.lexicon_rounds)

    # -- Binding training (SVO sentences) --------------------------
    n_train = cfg.n_train_sentences * cfg.training_reps
    train_sents = generate_svo_sentences(n_train, rng)

    for agent, verb_word, patient in train_sents:
        # Bind agent noun -> ROLE_AGENT
        _activate_word(brain, f"PHON_{agent}", "NOUN_CORE", 3)
        brain.inhibit_areas(["ROLE_AGENT"])
        for _ in range(cfg.binding_rounds):
            brain.project(
                {f"PHON_{agent}": ["NOUN_CORE", "ROLE_AGENT"]},
                {"NOUN_CORE": ["ROLE_AGENT"],
                 "ROLE_AGENT": ["NOUN_CORE"]},
            )

        # Bind patient noun -> ROLE_PATIENT
        _activate_word(brain, f"PHON_{patient}", "NOUN_CORE", 3)
        brain.inhibit_areas(["ROLE_PATIENT"])
        for _ in range(cfg.binding_rounds):
            brain.project(
                {f"PHON_{patient}": ["NOUN_CORE", "ROLE_PATIENT"]},
                {"NOUN_CORE": ["ROLE_PATIENT"],
                 "ROLE_PATIENT": ["NOUN_CORE"]},
            )

    # -- Test binding at critical position -------------------------
    brain.disable_plasticity = True

    gram_instabilities = []
    gram_convergences = []
    catviol_instabilities = []
    catviol_convergences = []

    # Test each noun as grammatical object, each verb as violation
    for noun in NOUNS:
        result = _measure_binding_difficulty(
            brain, noun, "NOUN_CORE", "ROLE_PATIENT", cfg.n_settling_rounds)
        gram_instabilities.append(result["anchored_instability"])
        gram_convergences.append(result["convergence_round"])

    for verb in VERBS:
        result = _measure_binding_difficulty(
            brain, verb, "VERB_CORE", "ROLE_PATIENT", cfg.n_settling_rounds)
        catviol_instabilities.append(result["anchored_instability"])
        catviol_convergences.append(result["convergence_round"])

    brain.disable_plasticity = False

    return {
        "gram_instability_mean": float(np.mean(gram_instabilities)),
        "gram_convergence_mean": float(np.mean(gram_convergences)),
        "catviol_instability_mean": float(np.mean(catviol_instabilities)),
        "catviol_convergence_mean": float(np.mean(catviol_convergences)),
        "gram_instabilities": gram_instabilities,
        "catviol_instabilities": catviol_instabilities,
        "gram_convergences": gram_convergences,
        "catviol_convergences": catviol_convergences,
    }


class BindingP600Experiment(ExperimentBase):
    """Test P600-like binding difficulty signals at critical sentence positions."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="binding_p600",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[P600Config] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or P600Config(
            **{k: v for k, v in kwargs.items()
               if k in P600Config.__dataclass_fields__})

        self.log("=" * 70)
        self.log("P600 from Binding Difficulty")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  binding_rounds={cfg.binding_rounds}, "
                 f"n_settling_rounds={cfg.n_settling_rounds}")
        self.log(f"  training_reps={cfg.training_reps}, "
                 f"n_train_sentences={cfg.n_train_sentences}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        gram_inst_vals = []
        catviol_inst_vals = []
        gram_conv_vals = []
        catviol_conv_vals = []

        for s in range(n_seeds):
            self.log(f"  Seed {s+1}/{n_seeds} ...")
            result = run_trial(cfg, self.seed + s)

            gram_inst_vals.append(result["gram_instability_mean"])
            catviol_inst_vals.append(result["catviol_instability_mean"])
            gram_conv_vals.append(result["gram_convergence_mean"])
            catviol_conv_vals.append(result["catviol_convergence_mean"])

            if s == 0:
                self.log(f"    Gram anchored instability:    "
                         f"{result['gram_instability_mean']:.4f}")
                self.log(f"    CatViol anchored instability: "
                         f"{result['catviol_instability_mean']:.4f}")
                self.log(f"    Gram convergence:    "
                         f"{result['gram_convergence_mean']:.1f}")
                self.log(f"    CatViol convergence: "
                         f"{result['catviol_convergence_mean']:.1f}")

        h1_test = paired_ttest(catviol_inst_vals, gram_inst_vals)
        h2_test = paired_ttest(catviol_conv_vals, gram_conv_vals)

        self.log(f"\n  H1 -- Anchored Instability: CatViol > Gram:")
        self.log(f"    Gram:    {np.mean(gram_inst_vals):.4f} "
                 f"+/- {np.std(gram_inst_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    CatViol: {np.mean(catviol_inst_vals):.4f} "
                 f"+/- {np.std(catviol_inst_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    d={h1_test['d']:.2f}, p={h1_test['p']:.4f}")

        self.log(f"\n  H2 -- Convergence: CatViol > Gram:")
        self.log(f"    Gram:    {np.mean(gram_conv_vals):.1f} "
                 f"+/- {np.std(gram_conv_vals)/np.sqrt(n_seeds):.1f}")
        self.log(f"    CatViol: {np.mean(catviol_conv_vals):.1f} "
                 f"+/- {np.std(catviol_conv_vals)/np.sqrt(n_seeds):.1f}")
        self.log(f"    d={h2_test['d']:.2f}, p={h2_test['p']:.4f}")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "p600_instability_gram": summarize(gram_inst_vals),
            "p600_instability_catviol": summarize(catviol_inst_vals),
            "convergence_gram": summarize(gram_conv_vals),
            "convergence_catviol": summarize(catviol_conv_vals),
            "h1_instability_test": h1_test,
            "h2_convergence_test": h2_test,
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "w_max": cfg.w_max,
                "lexicon_rounds": cfg.lexicon_rounds,
                "binding_rounds": cfg.binding_rounds,
                "training_reps": cfg.training_reps,
                "n_train_sentences": cfg.n_train_sentences,
                "n_settling_rounds": cfg.n_settling_rounds,
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
        description="P600 Binding Difficulty Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = BindingP600Experiment(verbose=True)

    if args.quick:
        cfg = P600Config(
            n=5000, k=50, training_reps=3, n_train_sentences=15)
        n_seeds = args.seeds or 5
    else:
        cfg = P600Config()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)

    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    print("\n" + "=" * 70)
    print("P600 BINDING DIFFICULTY SUMMARY")
    print("=" * 70)

    m = result.metrics
    print(f"\nAnchored instability at patient position (P600):")
    print(f"  Grammatical: {m['p600_instability_gram']['mean']:.4f} "
          f"+/- {m['p600_instability_gram']['sem']:.4f}")
    print(f"  CatViol:     {m['p600_instability_catviol']['mean']:.4f} "
          f"+/- {m['p600_instability_catviol']['sem']:.4f}")

    print(f"\nH1 -- Anchored Instability CatViol > Gram: "
          f"d={m['h1_instability_test']['d']:.2f}, "
          f"p={m['h1_instability_test']['p']:.4f}")

    print(f"\nConvergence rounds:")
    print(f"  Grammatical: {m['convergence_gram']['mean']:.1f} "
          f"+/- {m['convergence_gram']['sem']:.1f}")
    print(f"  CatViol:     {m['convergence_catviol']['mean']:.1f} "
          f"+/- {m['convergence_catviol']['sem']:.1f}")
    print(f"  H2 -- d={m['h2_convergence_test']['d']:.2f}, "
          f"p={m['h2_convergence_test']['p']:.4f}")

    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
