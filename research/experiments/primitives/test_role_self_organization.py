"""
Unsupervised Role Discovery via Shared Category Marker + LRI

Tests whether a pool of undifferentiated structural areas can learn to
specialize for agent vs patient roles from SVO word order alone, with
NO role labels and NO training.

Mechanism:
  1. A shared NOUN_MARKER stimulus fires for every noun (category-level signal)
  2. NOUN_MARKER projects to the structural pool; mutual inhibition picks an area
  3. Losers' LRI is cleared; only the winner carries refractory suppression
  4. The marker re-projects to the winner several times (stabilization),
     building up cumulative LRI that makes the winner's neurons uncompetitive
  5. Verb stimulus fires (temporal separator; doesn't touch struct areas)
  6. NOUN_MARKER fires again for Noun2; the first area's neurons are suppressed
     by LRI, so mutual inhibition picks a DIFFERENT area
  7. No Hebbian training is needed for position routing — the effect is
     architectural (MI + LRI + shared marker)

Key insights:
  - Different nouns activate different assemblies in NOUN_CORE, so LRI on
    specific neurons does NOT prevent the same area from winning again.
    The shared marker solves this: it always activates the same stimulus
    neurons, so the same struct neurons fire regardless of which noun.
  - All areas accumulate LRI during the competition projection (before MI).
    Clearing losers' LRI after MI ensures only the winner is suppressed.

Hypotheses:

H1: Position consistency — each structural area is dominated by one
    positional role (pre-verbal or post-verbal). Purity > chance.

H2: Role separation — Noun1 and Noun2 within a sentence consistently
    go to DIFFERENT areas. Separation rate > 0.90.

Usage:
    uv run python research/experiments/primitives/test_role_self_organization.py
    uv run python research/experiments/primitives/test_role_self_organization.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    summarize,
    ttest_vs_null,
)
from src.core.brain import Brain


NOUNS = ["dog", "cat", "bird", "boy", "girl",
         "ball", "book", "food", "table", "car"]
VERBS = ["chases", "sees", "eats", "finds", "plays"]


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


@dataclass
class RoleDiscoveryConfig:
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.15
    w_max: float = 20.0
    n_struct_areas: int = 4
    refractory_period: int = 5
    inhibition_strength: float = 1.0
    lexicon_rounds: int = 20
    stabilize_rounds: int = 3
    n_test_sentences: int = 50


def _find_winner(brain: Brain, areas: List[str]) -> Optional[str]:
    for name in areas:
        if len(brain.areas[name].winners) > 0:
            return name
    return None


def _activate_word(brain: Brain, stim_name: str, core_area: str, rounds: int):
    """Activate a word's assembly in its core area via stimulus projection."""
    brain.inhibit_areas([core_area])
    for _ in range(rounds):
        brain.project({stim_name: [core_area]}, {core_area: [core_area]})


def _project_marker_to_struct(
    brain: Brain,
    marker_name: str,
    struct_areas: List[str],
    stabilize_rounds: int,
) -> Optional[str]:
    """Project shared category marker to structural pool.

    Phase 1 (competition): plasticity OFF, marker → all struct areas.
       Mutual inhibition picks the area with highest total activation.
       Then losers' LRI is cleared so only the winner is suppressed.

    Phase 2 (stabilization): plasticity OFF, marker → winner only.
       Repeated projection builds cumulative LRI in the winner area,
       ensuring its neurons are strongly suppressed for the next position.

    Returns name of winning structural area.
    """
    # Phase 1: Competition (no plasticity)
    brain.disable_plasticity = True
    brain.project({marker_name: list(struct_areas)}, {})

    winner = _find_winner(brain, struct_areas)
    if winner is None:
        brain.disable_plasticity = False
        return None

    # Clear losers' LRI — only the winner carries refractory suppression.
    # (All areas accumulate LRI during projection, before MI fires.)
    for name in struct_areas:
        if name != winner:
            brain.clear_refractory(name)

    # Phase 2: Stabilization (marker → winner, builds LRI history)
    for _ in range(stabilize_rounds):
        brain.project({marker_name: [winner]}, {})

    brain.disable_plasticity = False
    return winner


def run_trial(
    cfg: RoleDiscoveryConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run one role discovery trial."""
    brain = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    rng = np.random.default_rng(seed)

    # Core areas
    brain.add_area("NOUN_CORE", cfg.n, cfg.k, cfg.beta)
    brain.add_area("VERB_CORE", cfg.n, cfg.k, cfg.beta)

    # Structural pool with LRI + mutual inhibition
    struct_areas = [f"STRUCT_{i}" for i in range(cfg.n_struct_areas)]
    for name in struct_areas:
        brain.add_area(
            name, cfg.n, cfg.k, cfg.beta,
            refractory_period=cfg.refractory_period,
            inhibition_strength=cfg.inhibition_strength,
        )
    brain.add_mutual_inhibition(struct_areas)

    # Register stimuli
    for noun in NOUNS:
        brain.add_stimulus(f"PHON_{noun}", cfg.k)
    for verb in VERBS:
        brain.add_stimulus(f"PHON_{verb}", cfg.k)

    # Shared category marker — same stimulus for all nouns
    brain.add_stimulus("NOUN_MARKER", cfg.k)

    # ── Build lexicon ────────────────────────────────────────────
    for noun in NOUNS:
        brain._engine.reset_area_connections("NOUN_CORE")
        _activate_word(brain, f"PHON_{noun}", "NOUN_CORE", cfg.lexicon_rounds)

    for verb in VERBS:
        brain._engine.reset_area_connections("VERB_CORE")
        _activate_word(brain, f"PHON_{verb}", "VERB_CORE", cfg.lexicon_rounds)

    # ── Test ─────────────────────────────────────────────────────
    # No training phase needed — role separation is architectural,
    # arising from MI + LRI + shared marker without Hebbian learning.
    test_sents = generate_svo_sentences(cfg.n_test_sentences, rng)

    pos1_areas = []
    pos2_areas = []
    separation = []

    for agent_word, verb_word, patient_word in test_sents:
        for name in struct_areas:
            brain.clear_refractory(name)
        brain.inhibit_areas(struct_areas)

        # Noun1 (agent)
        _activate_word(brain, f"PHON_{agent_word}", "NOUN_CORE", 3)
        w1 = _project_marker_to_struct(
            brain, "NOUN_MARKER", struct_areas, cfg.stabilize_rounds)

        # Verb (temporal separator)
        _activate_word(brain, f"PHON_{verb_word}", "VERB_CORE", 2)

        # Noun2 (patient)
        _activate_word(brain, f"PHON_{patient_word}", "NOUN_CORE", 3)
        w2 = _project_marker_to_struct(
            brain, "NOUN_MARKER", struct_areas, cfg.stabilize_rounds)

        if w1:
            pos1_areas.append(w1)
        if w2:
            pos2_areas.append(w2)
        if w1 and w2:
            separation.append(1.0 if w1 != w2 else 0.0)

    # ── Metrics ──────────────────────────────────────────────────
    # Position purity
    all_assignments = []
    all_labels = []
    for area in pos1_areas:
        all_assignments.append(area)
        all_labels.append("pos1")
    for area in pos2_areas:
        all_assignments.append(area)
        all_labels.append("pos2")

    area_label_counts = defaultdict(Counter)
    for area, label in zip(all_assignments, all_labels):
        area_label_counts[area][label] += 1

    correct = sum(
        counts.most_common(1)[0][1]
        for counts in area_label_counts.values()
    )
    total = len(all_assignments)
    position_purity = correct / total if total > 0 else 0.0

    sep_rate = float(np.mean(separation)) if separation else 0.0

    # Position consistency
    pos1_consistency = 0.0
    pos2_consistency = 0.0
    if pos1_areas:
        pos1_consistency = Counter(pos1_areas).most_common(1)[0][1] / len(pos1_areas)
    if pos2_areas:
        pos2_consistency = Counter(pos2_areas).most_common(1)[0][1] / len(pos2_areas)

    return {
        "position_purity": position_purity,
        "separation_rate": sep_rate,
        "pos1_consistency": pos1_consistency,
        "pos2_consistency": pos2_consistency,
        "pos1_counter": dict(Counter(pos1_areas)),
        "pos2_counter": dict(Counter(pos2_areas)),
        "n_active_struct": len(set(pos1_areas) | set(pos2_areas)),
    }


class RoleDiscoveryExperiment(ExperimentBase):

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="role_self_organization",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[RoleDiscoveryConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or RoleDiscoveryConfig(
            **{k: v for k, v in kwargs.items()
               if k in RoleDiscoveryConfig.__dataclass_fields__})

        chance_purity = 1.0 / cfg.n_struct_areas + 0.5 * (
            1.0 - 1.0 / cfg.n_struct_areas)

        self.log("=" * 70)
        self.log("Unsupervised Role Discovery: Shared Marker + LRI")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  n_struct_areas={cfg.n_struct_areas}, "
                 f"refractory={cfg.refractory_period}, "
                 f"LRI_strength={cfg.inhibition_strength}")
        self.log(f"  stabilize_rounds={cfg.stabilize_rounds}")
        self.log(f"  test: {cfg.n_test_sentences} sentences")
        self.log(f"  n_seeds={n_seeds}")
        self.log(f"  chance purity={chance_purity:.3f}")
        self.log("=" * 70)

        purity_vals = []
        sep_vals = []
        p1_consist_vals = []
        p2_consist_vals = []
        active_vals = []

        for s in range(n_seeds):
            self.log(f"  Seed {s+1}/{n_seeds} ...")
            result = run_trial(cfg, self.seed + s)

            purity_vals.append(result["position_purity"])
            sep_vals.append(result["separation_rate"])
            p1_consist_vals.append(result["pos1_consistency"])
            p2_consist_vals.append(result["pos2_consistency"])
            active_vals.append(result["n_active_struct"])

            if s == 0:
                self.log(f"    Pos1 dist: {result['pos1_counter']}")
                self.log(f"    Pos2 dist: {result['pos2_counter']}")

        purity_test = ttest_vs_null(purity_vals, chance_purity)
        sep_test = ttest_vs_null(sep_vals, 0.5)

        self.log(f"\n  Position purity: {np.mean(purity_vals):.3f} "
                 f"+/- {np.std(purity_vals)/np.sqrt(n_seeds):.3f}  "
                 f"(d={purity_test['d']:.2f}, p={purity_test['p']:.4f})")
        self.log(f"  Separation rate: {np.mean(sep_vals):.3f} "
                 f"+/- {np.std(sep_vals)/np.sqrt(n_seeds):.3f}  "
                 f"(d={sep_test['d']:.2f}, p={sep_test['p']:.4f})")
        self.log(f"  Pos1 consistency: {np.mean(p1_consist_vals):.3f}")
        self.log(f"  Pos2 consistency: {np.mean(p2_consist_vals):.3f}")
        self.log(f"  Active struct areas: {np.mean(active_vals):.1f}"
                 f"/{cfg.n_struct_areas}")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "position_purity": summarize(purity_vals),
            "purity_vs_chance": purity_test,
            "separation_rate": summarize(sep_vals),
            "separation_vs_chance": sep_test,
            "pos1_consistency": summarize(p1_consist_vals),
            "pos2_consistency": summarize(p2_consist_vals),
            "n_active_struct": summarize(active_vals),
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "w_max": cfg.w_max,
                "n_struct_areas": cfg.n_struct_areas,
                "refractory_period": cfg.refractory_period,
                "inhibition_strength": cfg.inhibition_strength,
                "stabilize_rounds": cfg.stabilize_rounds,
                "n_test_sentences": cfg.n_test_sentences,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Unsupervised Role Discovery Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = RoleDiscoveryExperiment(verbose=True)

    if args.quick:
        cfg = RoleDiscoveryConfig(
            n=5000, k=50, n_test_sentences=30)
        n_seeds = args.seeds or 5
    else:
        cfg = RoleDiscoveryConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)

    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    print("\n" + "=" * 70)
    print("ROLE DISCOVERY SUMMARY")
    print("=" * 70)

    m = result.metrics
    pur = m["position_purity"]
    sep = m["separation_rate"]
    p1 = m["pos1_consistency"]
    p2 = m["pos2_consistency"]
    pt = m["purity_vs_chance"]
    st = m["separation_vs_chance"]

    print(f"\nPosition purity:  {pur['mean']:.3f} +/- {pur['sem']:.3f}  "
          f"(d={pt['d']:.2f}, p={pt['p']:.4f})")
    print(f"Separation rate:  {sep['mean']:.3f} +/- {sep['sem']:.3f}  "
          f"(d={st['d']:.2f}, p={st['p']:.4f})")
    print(f"Pos1 consistency: {p1['mean']:.3f} +/- {p1['sem']:.3f}")
    print(f"Pos2 consistency: {p2['mean']:.3f} +/- {p2['sem']:.3f}")
    print(f"Active areas:     {m['n_active_struct']['mean']:.1f}")
    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
