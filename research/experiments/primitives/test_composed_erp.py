"""
Composed ERP Signals: Unified Prediction + Binding Pipeline

Chains forward prediction and role binding into a single sentence processor
where N400 and P600 emerge as different aspects of the same prediction+binding
mechanism.

The key insight: both ERP components are prediction error at different levels:
  N400 = lexical prediction error (overlap failure in PREDICTION area)
  P600 = structural integration difficulty (anchored instability in role area)

The double dissociation emerges naturally:
  Grammatical:       low N400, low P600  (prediction matches, binding trained)
  Category violation: high N400, high P600 (wrong category, untrained binding)
  Novel object:      moderate N400, low P600 (right category but novel item,
                     binding still works because it's a noun)

Architecture:
  PHON_<word>    — stimulus for each word
  NOUN_CORE      — noun assemblies
  VERB_CORE      — verb assemblies
  PREDICTION     — forward projection target
  ROLE_AGENT     — agent slot
  ROLE_PATIENT   — patient slot

Protocol:
  1. Build word assemblies in NOUN_CORE and VERB_CORE
  2. Train prediction bridges (SVO co-projection) AND role bindings
  3. Build prediction lexicon (plasticity OFF)
  4. Test word-by-word, measuring N400 and P600 at each position

Hypotheses:
  H1 (N400 ordering): N400_catviol > N400_novel > N400_gram
  H2 (P600 ordering): P600_catviol > P600_gram; P600_novel ~ P600_gram
  H3 (Double dissociation): Novel shows N400 but not P600; CatViol shows both

Usage:
    uv run python research/experiments/primitives/test_composed_erp.py
    uv run python research/experiments/primitives/test_composed_erp.py --quick
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
class ComposedConfig:
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.10
    w_max: float = 20.0
    lexicon_rounds: int = 20
    train_rounds_per_pair: int = 5
    binding_rounds: int = 10
    training_reps: int = 3
    n_train_sentences: int = 20
    lexicon_readout_rounds: int = 5
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
    """Measure binding difficulty when settling a word into a role slot.

    Plasticity must be OFF before calling.

    Uses anchored instability as the primary P600 metric:
      Phase A: one round with stimulus co-projection to create initial pattern
      Phase B: area-to-area settling without stimulus
    Trained pathways sustain the pattern (low instability = low P600).
    Untrained pathways cannot sustain it (high instability = high P600).
    """
    result = compute_anchored_instability(
        brain, word, core_area, role_area, n_settling_rounds)
    return {
        "anchored_instability": result["instability"],
    }


def run_trial(
    cfg: ComposedConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run one composed ERP trial.

    Processes test sentences word by word, measuring both N400 (prediction
    error) and P600 (binding instability) at each position.
    """
    brain = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    rng = np.random.default_rng(seed)

    # Create all areas
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

    # -- Build word assemblies in core areas -----------------------
    for noun in NOUNS:
        brain._engine.reset_area_connections("NOUN_CORE")
        _activate_word(brain, f"PHON_{noun}", "NOUN_CORE", cfg.lexicon_rounds)

    for verb in VERBS:
        brain._engine.reset_area_connections("VERB_CORE")
        _activate_word(brain, f"PHON_{verb}", "VERB_CORE", cfg.lexicon_rounds)

    # Novel nouns (assemblies exist but never trained in prediction/binding)
    for noun in NOVEL_NOUNS:
        brain._engine.reset_area_connections("NOUN_CORE")
        _activate_word(brain, f"PHON_{noun}", "NOUN_CORE", cfg.lexicon_rounds)

    # -- Training phase: prediction bridges + role bindings --------
    n_train = cfg.n_train_sentences * cfg.training_reps
    train_sents = generate_svo_sentences(n_train, rng)

    for agent, verb_word, patient in train_sents:
        # --- Prediction training ---
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

        # --- Binding training ---
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

    # -- Build prediction lexicon (plasticity OFF) -----------------
    brain.disable_plasticity = True
    lexicon = {}
    for word in NOUNS + VERBS + NOVEL_NOUNS:
        brain.inhibit_areas(["PREDICTION"])
        for _ in range(cfg.lexicon_readout_rounds):
            brain.project({f"PHON_{word}": ["PREDICTION"]}, {})
        lexicon[word] = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)
    brain.disable_plasticity = False

    # -- Test sentence processing (plasticity OFF) -----------------
    brain.disable_plasticity = True

    test_triples = generate_test_triples(rng, n_triples=5)

    # Metrics at critical (object) position
    n400_gram = []
    n400_catviol = []
    n400_novel = []
    n400_cat_gram = []
    n400_cat_catviol = []
    n400_cat_novel = []
    p600_gram = []
    p600_catviol = []
    p600_novel = []

    # Also collect verb-position metrics (control)
    n400_verb = []
    p600_subj = []

    # Category reference assemblies for category-match N400
    noun_refs = [lexicon[n] for n in NOUNS]
    verb_refs = [lexicon[v] for v in VERBS]

    for agent, verb_word, gram_obj, catviol_obj, novel_obj in test_triples:
        # === Position 1: subject noun ===
        _activate_word(brain, f"PHON_{agent}", "NOUN_CORE", 3)

        # Bind to ROLE_AGENT, measure difficulty
        bd_subj = _measure_binding_difficulty(
            brain, agent, "NOUN_CORE", "ROLE_AGENT", cfg.n_settling_rounds)
        p600_subj.append(bd_subj["anchored_instability"])

        # Re-activate agent (binding measurement may have changed NOUN_CORE)
        _activate_word(brain, f"PHON_{agent}", "NOUN_CORE", 3)

        # Forward project: NOUN_CORE -> PREDICTION (verb expectation)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})

        # === Position 2: verb ===
        # N400 at verb: compare prediction with verb's lexicon entry
        predicted_at_verb = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)
        n400_v = 1.0 - measure_overlap(predicted_at_verb, lexicon[verb_word])
        n400_verb.append(n400_v)

        # Activate verb in VERB_CORE
        _activate_word(brain, f"PHON_{verb_word}", "VERB_CORE", 3)

        # Forward project: VERB_CORE -> PREDICTION (object expectation)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"VERB_CORE": ["PREDICTION"]})
        predicted_at_obj = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

        # === Position 3: critical word (object slot) ===
        # --- Grammatical condition ---
        n400_g = 1.0 - measure_overlap(predicted_at_obj, lexicon[gram_obj])
        n400_gram.append(n400_g)
        n400_cat_gram.append(
            1.0 - max(measure_overlap(predicted_at_obj, r) for r in noun_refs))

        _activate_word(brain, f"PHON_{gram_obj}", "NOUN_CORE", 3)
        bd_g = _measure_binding_difficulty(
            brain, gram_obj, "NOUN_CORE", "ROLE_PATIENT",
            cfg.n_settling_rounds)
        p600_gram.append(bd_g["anchored_instability"])

        # --- Category violation condition ---
        n400_c = 1.0 - measure_overlap(predicted_at_obj, lexicon[catviol_obj])
        n400_catviol.append(n400_c)
        n400_cat_catviol.append(
            1.0 - max(measure_overlap(predicted_at_obj, r) for r in verb_refs))

        _activate_word(brain, f"PHON_{catviol_obj}", "VERB_CORE", 3)
        bd_c = _measure_binding_difficulty(
            brain, catviol_obj, "VERB_CORE", "ROLE_PATIENT",
            cfg.n_settling_rounds)
        p600_catviol.append(bd_c["anchored_instability"])

        # --- Novel object condition ---
        n400_n = 1.0 - measure_overlap(predicted_at_obj, lexicon[novel_obj])
        n400_novel.append(n400_n)
        n400_cat_novel.append(
            1.0 - max(measure_overlap(predicted_at_obj, r) for r in noun_refs))

        _activate_word(brain, f"PHON_{novel_obj}", "NOUN_CORE", 3)
        bd_n = _measure_binding_difficulty(
            brain, novel_obj, "NOUN_CORE", "ROLE_PATIENT",
            cfg.n_settling_rounds)
        p600_novel.append(bd_n["anchored_instability"])

    brain.disable_plasticity = False

    return {
        "n400_gram": n400_gram,
        "n400_catviol": n400_catviol,
        "n400_novel": n400_novel,
        "n400_verb": n400_verb,
        "n400_cat_gram": n400_cat_gram,
        "n400_cat_catviol": n400_cat_catviol,
        "n400_cat_novel": n400_cat_novel,
        "p600_gram": p600_gram,
        "p600_catviol": p600_catviol,
        "p600_novel": p600_novel,
        "p600_subj": p600_subj,
        "n400_gram_mean": float(np.mean(n400_gram)),
        "n400_catviol_mean": float(np.mean(n400_catviol)),
        "n400_novel_mean": float(np.mean(n400_novel)),
        "n400_cat_gram_mean": float(np.mean(n400_cat_gram)),
        "n400_cat_catviol_mean": float(np.mean(n400_cat_catviol)),
        "n400_cat_novel_mean": float(np.mean(n400_cat_novel)),
        "p600_gram_mean": float(np.mean(p600_gram)),
        "p600_catviol_mean": float(np.mean(p600_catviol)),
        "p600_novel_mean": float(np.mean(p600_novel)),
    }


class ComposedERPExperiment(ExperimentBase):
    """Composed prediction + binding pipeline for N400/P600 double dissociation."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="composed_erp",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[ComposedConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or ComposedConfig(
            **{k: v for k, v in kwargs.items()
               if k in ComposedConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Composed ERP: Prediction (N400) + Binding (P600)")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  train_rounds_per_pair={cfg.train_rounds_per_pair}, "
                 f"binding_rounds={cfg.binding_rounds}")
        self.log(f"  training_reps={cfg.training_reps}, "
                 f"n_train_sentences={cfg.n_train_sentences}")
        self.log(f"  n_settling_rounds={cfg.n_settling_rounds}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        # Collect per-seed means
        n400_gram_vals = []
        n400_catviol_vals = []
        n400_novel_vals = []
        n400_cat_gram_vals = []
        n400_cat_catviol_vals = []
        n400_cat_novel_vals = []
        p600_gram_vals = []
        p600_catviol_vals = []
        p600_novel_vals = []

        for s in range(n_seeds):
            self.log(f"  Seed {s+1}/{n_seeds} ...")
            result = run_trial(cfg, self.seed + s)

            n400_gram_vals.append(result["n400_gram_mean"])
            n400_catviol_vals.append(result["n400_catviol_mean"])
            n400_novel_vals.append(result["n400_novel_mean"])
            n400_cat_gram_vals.append(result["n400_cat_gram_mean"])
            n400_cat_catviol_vals.append(result["n400_cat_catviol_mean"])
            n400_cat_novel_vals.append(result["n400_cat_novel_mean"])
            p600_gram_vals.append(result["p600_gram_mean"])
            p600_catviol_vals.append(result["p600_catviol_mean"])
            p600_novel_vals.append(result["p600_novel_mean"])

            if s == 0:
                self.log(f"    N400 (word) — gram={result['n400_gram_mean']:.4f}  "
                         f"catviol={result['n400_catviol_mean']:.4f}  "
                         f"novel={result['n400_novel_mean']:.4f}")
                self.log(f"    N400 (cat)  — gram={result['n400_cat_gram_mean']:.4f}  "
                         f"catviol={result['n400_cat_catviol_mean']:.4f}  "
                         f"novel={result['n400_cat_novel_mean']:.4f}")
                self.log(f"    P600 — gram={result['p600_gram_mean']:.4f}  "
                         f"catviol={result['p600_catviol_mean']:.4f}  "
                         f"novel={result['p600_novel_mean']:.4f}")

        # --- Statistical tests ---
        # H1: N400 ordering (word-specific)
        h1_catviol_gram = paired_ttest(n400_catviol_vals, n400_gram_vals)
        h1_novel_gram = paired_ttest(n400_novel_vals, n400_gram_vals)
        h1_catviol_novel = paired_ttest(n400_catviol_vals, n400_novel_vals)

        # H1: N400 ordering (category-match)
        h1_cat_catviol_gram = paired_ttest(n400_cat_catviol_vals, n400_cat_gram_vals)
        h1_cat_novel_gram = paired_ttest(n400_cat_novel_vals, n400_cat_gram_vals)

        # H2: P600 ordering
        h2_catviol_gram = paired_ttest(p600_catviol_vals, p600_gram_vals)
        h2_novel_gram = paired_ttest(p600_novel_vals, p600_gram_vals)

        self.log(f"\n  === N400 Word-specific (Prediction Error) ===")
        self.log(f"    Gram:    {np.mean(n400_gram_vals):.4f} "
                 f"+/- {np.std(n400_gram_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    Novel:   {np.mean(n400_novel_vals):.4f} "
                 f"+/- {np.std(n400_novel_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    CatViol: {np.mean(n400_catviol_vals):.4f} "
                 f"+/- {np.std(n400_catviol_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    H1 CatViol>Gram:  d={h1_catviol_gram['d']:.2f}, "
                 f"p={h1_catviol_gram['p']:.4f}")
        self.log(f"    H1 Novel>Gram:    d={h1_novel_gram['d']:.2f}, "
                 f"p={h1_novel_gram['p']:.4f}")
        self.log(f"    H1 CatViol>Novel: d={h1_catviol_novel['d']:.2f}, "
                 f"p={h1_catviol_novel['p']:.4f}")

        self.log(f"\n  === N400 Category-match ===")
        self.log(f"    Gram:    {np.mean(n400_cat_gram_vals):.4f} "
                 f"+/- {np.std(n400_cat_gram_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    Novel:   {np.mean(n400_cat_novel_vals):.4f} "
                 f"+/- {np.std(n400_cat_novel_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    CatViol: {np.mean(n400_cat_catviol_vals):.4f} "
                 f"+/- {np.std(n400_cat_catviol_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    CatViol>Gram: d={h1_cat_catviol_gram['d']:.2f}, "
                 f"p={h1_cat_catviol_gram['p']:.4f}")
        self.log(f"    Novel>Gram:   d={h1_cat_novel_gram['d']:.2f}, "
                 f"p={h1_cat_novel_gram['p']:.4f}")

        self.log(f"\n  === P600 (Anchored Instability) ===")
        self.log(f"    Gram:    {np.mean(p600_gram_vals):.4f} "
                 f"+/- {np.std(p600_gram_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    Novel:   {np.mean(p600_novel_vals):.4f} "
                 f"+/- {np.std(p600_novel_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    CatViol: {np.mean(p600_catviol_vals):.4f} "
                 f"+/- {np.std(p600_catviol_vals)/np.sqrt(n_seeds):.4f}")
        self.log(f"    H2 CatViol>Gram:  d={h2_catviol_gram['d']:.2f}, "
                 f"p={h2_catviol_gram['p']:.4f}")
        self.log(f"    H2 Novel~Gram:    d={h2_novel_gram['d']:.2f}, "
                 f"p={h2_novel_gram['p']:.4f}")

        # H3: Double dissociation summary
        n400_effect_novel = float(np.mean(n400_novel_vals) - np.mean(n400_gram_vals))
        p600_effect_novel = float(np.mean(p600_novel_vals) - np.mean(p600_gram_vals))
        n400_effect_catviol = float(np.mean(n400_catviol_vals) - np.mean(n400_gram_vals))
        p600_effect_catviol = float(np.mean(p600_catviol_vals) - np.mean(p600_gram_vals))

        self.log(f"\n  === Double Dissociation (H3) ===")
        self.log(f"    Novel:   N400 effect={n400_effect_novel:+.4f}  "
                 f"P600 effect={p600_effect_novel:+.4f}")
        self.log(f"    CatViol: N400 effect={n400_effect_catviol:+.4f}  "
                 f"P600 effect={p600_effect_catviol:+.4f}")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "n400": {
                "gram": summarize(n400_gram_vals),
                "catviol": summarize(n400_catviol_vals),
                "novel": summarize(n400_novel_vals),
            },
            "n400_cat": {
                "gram": summarize(n400_cat_gram_vals),
                "catviol": summarize(n400_cat_catviol_vals),
                "novel": summarize(n400_cat_novel_vals),
            },
            "p600": {
                "gram": summarize(p600_gram_vals),
                "catviol": summarize(p600_catviol_vals),
                "novel": summarize(p600_novel_vals),
            },
            "n400_tests": {
                "catviol_vs_gram": h1_catviol_gram,
                "novel_vs_gram": h1_novel_gram,
                "catviol_vs_novel": h1_catviol_novel,
            },
            "n400_cat_tests": {
                "catviol_vs_gram": h1_cat_catviol_gram,
                "novel_vs_gram": h1_cat_novel_gram,
            },
            "p600_tests": {
                "catviol_vs_gram": h2_catviol_gram,
                "novel_vs_gram": h2_novel_gram,
            },
            "double_dissociation": {
                "novel_n400_effect": n400_effect_novel,
                "novel_p600_effect": p600_effect_novel,
                "catviol_n400_effect": n400_effect_catviol,
                "catviol_p600_effect": p600_effect_catviol,
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
                "training_reps": cfg.training_reps,
                "n_train_sentences": cfg.n_train_sentences,
                "lexicon_readout_rounds": cfg.lexicon_readout_rounds,
                "n_settling_rounds": cfg.n_settling_rounds,
                "n_seeds": n_seeds,
                "n_nouns": len(NOUNS),
                "n_verbs": len(VERBS),
                "n_novel": len(NOVEL_NOUNS),
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Composed ERP Experiment (N400 + P600)")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = ComposedERPExperiment(verbose=True)

    if args.quick:
        cfg = ComposedConfig(
            n=5000, k=50, training_reps=3, n_train_sentences=15)
        n_seeds = args.seeds or 5
    else:
        cfg = ComposedConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)

    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    print("\n" + "=" * 70)
    print("COMPOSED ERP SUMMARY (Double Dissociation)")
    print("=" * 70)

    m = result.metrics

    print(f"\nN400 (Prediction Error) at object position:")
    print(f"  Grammatical: {m['n400']['gram']['mean']:.4f} "
          f"+/- {m['n400']['gram']['sem']:.4f}")
    print(f"  Novel:       {m['n400']['novel']['mean']:.4f} "
          f"+/- {m['n400']['novel']['sem']:.4f}")
    print(f"  CatViol:     {m['n400']['catviol']['mean']:.4f} "
          f"+/- {m['n400']['catviol']['sem']:.4f}")

    nt = m["n400_tests"]
    print(f"\n  CatViol > Gram:  d={nt['catviol_vs_gram']['d']:.2f}, "
          f"p={nt['catviol_vs_gram']['p']:.4f}")
    print(f"  Novel > Gram:    d={nt['novel_vs_gram']['d']:.2f}, "
          f"p={nt['novel_vs_gram']['p']:.4f}")
    print(f"  CatViol > Novel: d={nt['catviol_vs_novel']['d']:.2f}, "
          f"p={nt['catviol_vs_novel']['p']:.4f}")

    print(f"\nP600 (Anchored Instability) at object position:")
    print(f"  Grammatical: {m['p600']['gram']['mean']:.4f} "
          f"+/- {m['p600']['gram']['sem']:.4f}")
    print(f"  Novel:       {m['p600']['novel']['mean']:.4f} "
          f"+/- {m['p600']['novel']['sem']:.4f}")
    print(f"  CatViol:     {m['p600']['catviol']['mean']:.4f} "
          f"+/- {m['p600']['catviol']['sem']:.4f}")

    pt = m["p600_tests"]
    print(f"\n  CatViol > Gram:  d={pt['catviol_vs_gram']['d']:.2f}, "
          f"p={pt['catviol_vs_gram']['p']:.4f}")
    print(f"  Novel ~ Gram:    d={pt['novel_vs_gram']['d']:.2f}, "
          f"p={pt['novel_vs_gram']['p']:.4f}")

    dd = m["double_dissociation"]
    print(f"\nDouble Dissociation:")
    print(f"  Novel:   N400 effect={dd['novel_n400_effect']:+.4f}  "
          f"P600 effect={dd['novel_p600_effect']:+.4f}")
    print(f"  CatViol: N400 effect={dd['catviol_n400_effect']:+.4f}  "
          f"P600 effect={dd['catviol_p600_effect']:+.4f}")

    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
