"""
P600 Syntactic Violations — Global Energy in Role-Binding Areas

The P600 is a positive ERP component peaking ~600ms post-stimulus,
associated with syntactic violations and reanalysis. While the N400
reflects semantic processing difficulty (measured in core areas), the
P600 reflects syntactic processing difficulty (measured in role/syntactic
areas).

Design:
- Train parser with role-binding on standard SVO sentences
- Process test sentences word-by-word, measuring global pre-k-WTA
  energy at the critical word (object position) in:
  (a) the word's core area — N400 analogue (semantic access difficulty)
  (b) role-binding areas — P600 analogue (syntactic integration difficulty)

Conditions:
  C1 — Grammatical: "the dog chases the cat"
  C2 — Semantic violation: "the dog chases the table"
    (noun in correct position, but semantically incongruent)
  C3 — Category violation: "the dog chases the likes"
    (verb where noun expected — different verb from sentence verb)

Measurement protocol:
  N400: energy in core area DURING the first recurrent projection step
  of the critical word (self-recurrence from subject noun's residual
  assembly captures context-dependent facilitation).
  P600: energy in ROLE_PATIENT when attempting to bind the critical
  word to the patient role.

Predictions:
  N400 (core): semviol > gram (table harder than cat in NOUN_CORE)
               catviol: measured in VERB_CORE, not directly comparable
  P600 (role): catviol > semviol > gram (role binding difficulty)

This maps N400 to lexical access difficulty and P600 to syntactic
integration difficulty, consistent with Kuperberg (2007).

References:
- Osterhout & Holcomb 1992: P600 discovery
- Hagoort et al. 1993: P600 for syntactic violations
- Kuperberg 2007: N400/P600 dissociation framework
- research/claims/N400_GLOBAL_ENERGY.md: N400 claim
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any

from research.experiments.base import (
    ExperimentBase, ExperimentResult, summarize, paired_ttest,
)
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence
from src.assembly_calculus.emergent.areas import (
    NOUN_CORE, VERB_CORE, DET_CORE, CORE_AREAS,
    ROLE_AGENT, ROLE_PATIENT,
    SUBJ, OBJ, VP,
)
from src.assembly_calculus.ops import project


@dataclass
class P600Config:
    n: int = 50000
    k: int = 100
    p: float = 0.05
    beta: float = 0.05
    rounds: int = 10
    n_seeds: int = 5


def _build_vocab():
    """Standard vocabulary for syntactic tests."""
    return {
        # Animals (trained as subjects and objects)
        "dog":    GroundingContext(visual=["DOG", "ANIMAL"]),
        "cat":    GroundingContext(visual=["CAT", "ANIMAL"]),
        "bird":   GroundingContext(visual=["BIRD", "ANIMAL"]),
        "horse":  GroundingContext(visual=["HORSE", "ANIMAL"]),
        "fish":   GroundingContext(visual=["FISH", "ANIMAL"]),
        "mouse":  GroundingContext(visual=["MOUSE", "ANIMAL"]),
        # Objects (never trained in sentences — semantic violations)
        "table":  GroundingContext(visual=["TABLE", "FURNITURE"]),
        "chair":  GroundingContext(visual=["CHAIR", "FURNITURE"]),
        "book":   GroundingContext(visual=["BOOK", "OBJECT"]),
        "ball":   GroundingContext(visual=["BALL", "TOY"]),
        # Verbs (trained as actions)
        "chases": GroundingContext(motor=["CHASING", "PURSUIT"]),
        "sees":   GroundingContext(motor=["SEEING", "PERCEPTION"]),
        "finds":  GroundingContext(motor=["FINDING", "PERCEPTION"]),
        "likes":  GroundingContext(motor=["LIKING", "EMOTION"]),
        # Function
        "the":    GroundingContext(),
    }


def _build_training(vocab):
    """Training sentences with role annotations."""
    def ctx(w):
        return vocab[w]

    sentences = []
    for _ in range(3):
        for subj, verb, obj in [
            ("dog", "chases", "cat"),
            ("cat", "chases", "bird"),
            ("bird", "sees", "fish"),
            ("horse", "chases", "dog"),
            ("dog", "sees", "bird"),
            ("cat", "sees", "horse"),
        ]:
            sentences.append(GroundedSentence(
                words=["the", subj, verb, "the", obj],
                contexts=[ctx("the"), ctx(subj), ctx(verb),
                          ctx("the"), ctx(obj)],
                roles=[None, "agent", "action", None, "patient"],
            ))

    for subj, verb, obj in [
        ("fish", "sees", "mouse"),
        ("horse", "finds", "cat"),
        ("dog", "finds", "mouse"),
        ("bird", "chases", "horse"),
        ("mouse", "sees", "dog"),
        ("cat", "finds", "fish"),
    ]:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb),
                      ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    return sentences


# -- Energy measurement during sentence processing ----------------------------

def _measure_critical_word(
    parser: EmergentParser,
    context_words: List[str],
    critical_word: str,
    p600_areas: List[str],
    rounds: int,
) -> Dict[str, Any]:
    """Process context words, then measure energy at the critical word.

    N400 measurement: On the first recurrent projection step of the critical
    word into its core area, capture global pre-k-WTA energy. Self-recurrence
    from the subject noun's residual assembly (still active in NOUN_CORE)
    provides context-dependent facilitation.

    P600 measurement: After projecting the critical word into its core area,
    measure energy when projecting core -> role-binding area (ROLE_PATIENT).

    Returns dict with:
      n400_energy: float  — global energy in core area during first recurrent step
      p600_energies: {area: float}  — energy in each role/syntactic area
      core_area: str  — which core area the word projected into
    """
    engine = parser.brain._engine
    brain = parser.brain

    # Clear all areas to start fresh for each sentence
    for area_name in list(brain.areas.keys()):
        brain.inhibit_areas([area_name])

    # Process context words normally (builds sentence context)
    for word in context_words:
        core = parser._word_core_area(word)
        phon = parser.stim_map.get(word)
        if phon is not None:
            project(brain, phon, core, rounds=rounds)

    # Now measure the critical word
    crit_core = parser._word_core_area(critical_word)
    crit_phon = parser.stim_map.get(critical_word)
    if crit_phon is None:
        return {
            "n400_energy": 0.0,
            "p600_energies": {a: 0.0 for a in p600_areas},
            "core_area": crit_core,
        }

    # --- N400: measure during first projection WITH self-recurrence ---
    # Round 1: stimulus only (no recurrence), just like normal project()
    brain.project({crit_phon: [crit_core]}, {})

    # Round 2: stimulus + self-recurrence WITH record_activation
    # This is where context (residual assembly from subject noun) provides
    # facilitation via Hebbian-trained weights within the core area
    n400_result = engine.project_into(
        crit_core,
        from_stimuli=[crit_phon],
        from_areas=[crit_core],
        plasticity_enabled=True,
        record_activation=True,
    )
    n400_energy = n400_result.pre_kwta_total

    # Sync winners back to brain area
    brain.areas[crit_core].winners = n400_result.winners
    brain.areas[crit_core].w = n400_result.num_ever_fired

    # Complete remaining rounds (normal processing)
    if rounds > 2:
        brain.project_rounds(
            target=crit_core,
            areas_by_stim={crit_phon: [crit_core]},
            dst_areas_by_src_area={crit_core: [crit_core]},
            rounds=rounds - 2,
        )

    # --- P600: measure role-binding energy ---
    # After word is settled in core area, attempt role binding
    # and measure energy in role/syntactic areas
    p600_energies = {}
    for area in p600_areas:
        if area not in brain.areas:
            p600_energies[area] = 0.0
            continue
        try:
            p600_result = engine.project_into(
                area,
                from_stimuli=[],
                from_areas=[crit_core, area],
                plasticity_enabled=False,
                record_activation=True,
            )
            p600_energies[area] = p600_result.pre_kwta_total
        except (IndexError, KeyError):
            p600_energies[area] = 0.0

    return {
        "n400_energy": n400_energy,
        "p600_energies": p600_energies,
        "core_area": crit_core,
    }


# -- Test sentences -----------------------------------------------------------

def _make_test_sentences(vocab):
    """Generate matched sentence triples for N400/P600 dissociation.

    Each triple has the same context ("the SUBJ VERB the ___") with:
    - Grammatical: trained animal as object
    - Semantic violation: untrained object-category noun
    - Category violation: verb in object position (DIFFERENT from sentence verb)
    """
    def ctx(w):
        return vocab[w]

    tests = []

    # Triple 1: "the dog chases the ___"
    tests.append({
        "frame": "the dog chases the ___",
        "context_words": ["the", "dog", "chases", "the"],
        "grammatical": "cat",
        "semantic_violation": "table",
        "category_violation": "likes",  # different verb, avoids repetition
    })

    # Triple 2: "the cat sees the ___"
    tests.append({
        "frame": "the cat sees the ___",
        "context_words": ["the", "cat", "sees", "the"],
        "grammatical": "bird",
        "semantic_violation": "chair",
        "category_violation": "finds",
    })

    # Triple 3: "the bird chases the ___"
    tests.append({
        "frame": "the bird chases the ___",
        "context_words": ["the", "bird", "chases", "the"],
        "grammatical": "fish",
        "semantic_violation": "book",
        "category_violation": "sees",
    })

    # Triple 4: "the horse finds the ___"
    tests.append({
        "frame": "the horse finds the ___",
        "context_words": ["the", "horse", "finds", "the"],
        "grammatical": "mouse",
        "semantic_violation": "ball",
        "category_violation": "chases",
    })

    return tests


class P600SyntacticExperiment(ExperimentBase):
    """Test P600 via global energy in role/syntactic areas."""

    def __init__(self, results_dir=None, seed=42, verbose=True):
        super().__init__(
            name="p600_syntactic",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick=False, **kwargs):
        self._start_timer()
        cfg = P600Config()
        if quick:
            cfg.n_seeds = 3

        vocab = _build_vocab()
        training = _build_training(vocab)
        test_sentences = _make_test_sentences(vocab)
        seeds = list(range(cfg.n_seeds))

        p600_areas = [ROLE_AGENT, ROLE_PATIENT, SUBJ, OBJ, VP]

        # Per-seed accumulators
        # N400: core area energy (only comparable for gram vs semviol, both NOUN_CORE)
        n400_gram_seeds = []
        n400_semviol_seeds = []
        n400_catviol_seeds = []
        # P600: mean role-binding area energy (comparable across all conditions)
        p600_gram_seeds = []
        p600_semviol_seeds = []
        p600_catviol_seeds = []
        # Per-area P600 for detailed analysis
        p600_per_area_gram = {a: [] for a in p600_areas}
        p600_per_area_sem = {a: [] for a in p600_areas}
        p600_per_area_cat = {a: [] for a in p600_areas}

        for seed_idx, seed in enumerate(seeds):
            self.log(f"\n=== Seed {seed_idx + 1}/{len(seeds)} ===")

            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=seed, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            n400_gram, n400_sem, n400_cat = [], [], []
            p600_gram, p600_sem, p600_cat = [], [], []
            area_gram = {a: [] for a in p600_areas}
            area_sem = {a: [] for a in p600_areas}
            area_cat = {a: [] for a in p600_areas}

            for test in test_sentences:
                for cond_label, cond_key in [
                    ("gram", "grammatical"),
                    ("semviol", "semantic_violation"),
                    ("catviol", "category_violation"),
                ]:
                    critical_word = test[cond_key]
                    result = _measure_critical_word(
                        parser, test["context_words"], critical_word,
                        p600_areas, cfg.rounds,
                    )

                    n400_val = result["n400_energy"]
                    p600_mean = float(np.mean([
                        result["p600_energies"].get(a, 0.0)
                        for a in p600_areas
                    ]))

                    if cond_label == "gram":
                        n400_gram.append(n400_val)
                        p600_gram.append(p600_mean)
                        for a in p600_areas:
                            area_gram[a].append(result["p600_energies"].get(a, 0.0))
                    elif cond_label == "semviol":
                        n400_sem.append(n400_val)
                        p600_sem.append(p600_mean)
                        for a in p600_areas:
                            area_sem[a].append(result["p600_energies"].get(a, 0.0))
                    elif cond_label == "catviol":
                        n400_cat.append(n400_val)
                        p600_cat.append(p600_mean)
                        for a in p600_areas:
                            area_cat[a].append(result["p600_energies"].get(a, 0.0))

            if n400_gram:
                n400_gram_seeds.append(float(np.mean(n400_gram)))
                n400_semviol_seeds.append(float(np.mean(n400_sem)))
                n400_catviol_seeds.append(float(np.mean(n400_cat)))
                p600_gram_seeds.append(float(np.mean(p600_gram)))
                p600_semviol_seeds.append(float(np.mean(p600_sem)))
                p600_catviol_seeds.append(float(np.mean(p600_cat)))
                for a in p600_areas:
                    p600_per_area_gram[a].append(float(np.mean(area_gram[a])))
                    p600_per_area_sem[a].append(float(np.mean(area_sem[a])))
                    p600_per_area_cat[a].append(float(np.mean(area_cat[a])))

                self.log(
                    f"  N400 (core): gram={np.mean(n400_gram):.1f}  "
                    f"sem={np.mean(n400_sem):.1f}  "
                    f"cat={np.mean(n400_cat):.1f}")
                self.log(
                    f"  P600 (role): gram={np.mean(p600_gram):.1f}  "
                    f"sem={np.mean(p600_sem):.1f}  "
                    f"cat={np.mean(p600_cat):.1f}")

        # ===== Analysis =====
        self.log(f"\n{'='*70}")
        self.log("P600 / N400 DISSOCIATION RESULTS")
        self.log(f"{'='*70}")

        metrics = {}

        if len(n400_gram_seeds) < 2:
            self.log("Insufficient seeds for analysis")
            duration = self._stop_timer()
            result = ExperimentResult(
                experiment_name=self.name,
                parameters={"n": cfg.n, "k": cfg.k, "p": cfg.p,
                             "beta": cfg.beta, "rounds": cfg.rounds,
                             "n_seeds": cfg.n_seeds},
                metrics=metrics, duration_seconds=duration)
            self.save_result(result)
            return result

        # --- N400 analysis (core area energy) ---
        self.log("\nN400 (core area energy at critical word)")
        self.log("  Note: gram and semviol both project to NOUN_CORE (comparable)")
        self.log("  Note: catviol projects to VERB_CORE (different area)")
        self.log("  Prediction: semviol > gram (N400 for semantic anomaly)")

        for label, a, b in [
            ("sem_vs_gram", n400_semviol_seeds, n400_gram_seeds),
            ("cat_vs_gram", n400_catviol_seeds, n400_gram_seeds),
        ]:
            stats = paired_ttest(a, b)
            a_s = summarize(a)
            b_s = summarize(b)
            direction = "N400_EFFECT" if a_s["mean"] > b_s["mean"] else "NO_N400"
            self.log(f"  {label}: viol={a_s['mean']:.1f}  "
                     f"gram={b_s['mean']:.1f}  "
                     f"d={stats['d']:.3f}  p={stats['p']:.4f}  {direction}")
            metrics[f"n400_{label}"] = {
                "violation": a_s, "grammatical": b_s,
                "test": stats, "direction": direction,
            }

        # --- P600 analysis (role-binding area energy) ---
        self.log("\nP600 (role/syntactic area energy — mean across areas)")
        self.log("  Prediction: catviol > semviol > gram")

        for label, a, b in [
            ("cat_vs_gram", p600_catviol_seeds, p600_gram_seeds),
            ("sem_vs_gram", p600_semviol_seeds, p600_gram_seeds),
            ("cat_vs_sem", p600_catviol_seeds, p600_semviol_seeds),
        ]:
            stats = paired_ttest(a, b)
            a_s = summarize(a)
            b_s = summarize(b)
            direction = "P600_EFFECT" if a_s["mean"] > b_s["mean"] else "NO_P600"
            self.log(f"  {label}: a={a_s['mean']:.1f}  "
                     f"b={b_s['mean']:.1f}  "
                     f"d={stats['d']:.3f}  p={stats['p']:.4f}  {direction}")
            metrics[f"p600_{label}"] = {
                "a": a_s, "b": b_s,
                "test": stats, "direction": direction,
            }

        # --- Per-area P600 breakdown ---
        self.log(f"\nP600 per-area breakdown (category violation vs grammatical)")
        for area in p600_areas:
            if len(p600_per_area_gram[area]) >= 2:
                stats = paired_ttest(
                    p600_per_area_cat[area], p600_per_area_gram[area])
                cat_s = summarize(p600_per_area_cat[area])
                gram_s = summarize(p600_per_area_gram[area])
                self.log(f"  {area:<15}: cat={cat_s['mean']:.1f}  "
                         f"gram={gram_s['mean']:.1f}  "
                         f"d={stats['d']:.3f}  p={stats['p']:.4f}")
                metrics[f"p600_per_area_{area}"] = {
                    "category_violation": cat_s, "grammatical": gram_s,
                    "test": stats,
                }

        # --- Dissociation summary ---
        self.log(f"\n{'-'*70}")
        self.log("DISSOCIATION SUMMARY")
        self.log(f"{'-'*70}")

        n400_sem = metrics.get("n400_sem_vs_gram", {})
        p600_cat = metrics.get("p600_cat_vs_gram", {})
        p600_sem = metrics.get("p600_sem_vs_gram", {})
        p600_cat_vs_sem = metrics.get("p600_cat_vs_sem", {})

        sem_n400_d = n400_sem.get("test", {}).get("d", 0)
        sem_n400_p = n400_sem.get("test", {}).get("p", 1)
        cat_p600_d = p600_cat.get("test", {}).get("d", 0)
        cat_p600_p = p600_cat.get("test", {}).get("p", 1)
        sem_p600_d = p600_sem.get("test", {}).get("d", 0)
        sem_p600_p = p600_sem.get("test", {}).get("p", 1)

        self.log(f"  N400 for semantic violation: d={sem_n400_d:.3f} p={sem_n400_p:.4f}")
        self.log(f"  P600 for category violation: d={cat_p600_d:.3f} p={cat_p600_p:.4f}")
        self.log(f"  P600 for semantic violation: d={sem_p600_d:.3f} p={sem_p600_p:.4f}")

        # N400 selective = semantic violation produces N400
        n400_works = (n400_sem.get("direction") == "N400_EFFECT" and
                      sem_n400_p < 0.05)
        # P600 selective = category violation produces larger P600 than semantic
        p600_works = (p600_cat.get("direction") == "P600_EFFECT" and
                      cat_p600_p < 0.05)
        p600_graded = (p600_cat_vs_sem.get("direction") == "P600_EFFECT" and
                       p600_cat_vs_sem.get("test", {}).get("p", 1) < 0.05)

        self.log(f"\n  N400 for semantic anomaly: {'YES' if n400_works else 'NO'}")
        self.log(f"  P600 for syntactic anomaly: {'YES' if p600_works else 'NO'}")
        self.log(f"  P600 graded (cat > sem): {'YES' if p600_graded else 'NO'}")

        metrics["dissociation"] = {
            "n400_semantic": n400_works,
            "p600_syntactic": p600_works,
            "p600_graded": p600_graded,
        }

        duration = self._stop_timer()
        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p,
                "beta": cfg.beta, "rounds": cfg.rounds,
                "n_seeds": cfg.n_seeds,
                "p600_areas": p600_areas,
            },
            metrics=metrics,
            duration_seconds=duration,
        )
        self.save_result(result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P600 Syntactic Violation Experiment")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with fewer seeds (3 instead of 5)")
    args = parser.parse_args()

    exp = P600SyntacticExperiment()
    exp.run(quick=args.quick)
