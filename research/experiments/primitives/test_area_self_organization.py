"""
Self-Organization of Core Areas via Mutual Inhibition

Tests whether undifferentiated brain areas can specialize for word categories
through mutual inhibition + Hebbian plasticity, replacing the hand-specified
GROUNDING_TO_CORE mapping.

The Assembly Calculus provides three mechanisms that should enable this:
1. k-WTA (within-area): sparse competition selects top-k neurons
2. Mutual inhibition (across-area): only the area with highest drive fires
3. Hebbian plasticity: strengthens stim->winner pathways, creating positive
   feedback loops where shared features attract similar words to the same area

Key design insight: the competition phase must NOT have plasticity enabled,
otherwise ALL areas (including losers) accumulate Hebbian strengthening during
each round, diluting the winner's advantage and preventing specialization.
The correct protocol is:
  Phase 1 (competition): plasticity OFF, project stim → all areas, MI picks
  Phase 2 (stabilization): plasticity ON, project stim → winner only + recurrence

Conditions tested:
  A: Specific features only (DOG, ANIMAL, RUNNING, MOTION, etc.)
  B: Specific + modality features (A + shared VISUAL, MOTOR, etc.)

Condition A tests whether sub-group features provide enough clustering signal.
Condition B adds a modality-level feature shared by ALL words of the same POS
category, providing the grouping signal that the real parser gets from routing
through modality-specific input areas.

Usage:
    uv run python research/experiments/primitives/test_area_self_organization.py
    uv run python research/experiments/primitives/test_area_self_organization.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    measure_overlap,
    chance_overlap,
    summarize,
    ttest_vs_null,
    paired_ttest,
)
from src.core.brain import Brain
from src.assembly_calculus.emergent.grounding import VOCABULARY, GroundingContext


# ── Ground truth categories by dominant modality ──────────────────────

def get_ground_truth() -> Dict[str, str]:
    """Return word -> modality mapping for ground truth categories."""
    return {word: gc.dominant_modality for word, gc in VOCABULARY.items()}


def get_word_features(word: str, gc: GroundingContext) -> List[str]:
    """Extract grounding feature names for a word."""
    features = []
    for modality in ("visual", "motor", "properties", "spatial",
                     "social", "temporal", "emotional"):
        for feat in getattr(gc, modality):
            features.append(feat)
    return features


# ── Clustering metrics ────────────────────────────────────────────────

def compute_purity(word_to_area: Dict[str, str],
                   ground_truth: Dict[str, str]) -> float:
    """Weighted purity: fraction of words in each area matching majority class."""
    area_words = defaultdict(list)
    for word, area in word_to_area.items():
        area_words[area].append(ground_truth[word])

    total = 0
    correct = 0
    for area, labels in area_words.items():
        counts = Counter(labels)
        total += len(labels)
        correct += counts.most_common(1)[0][1]

    return correct / total if total > 0 else 0.0


def compute_completeness(word_to_area: Dict[str, str],
                         ground_truth: Dict[str, str]) -> float:
    """Weighted completeness: fraction of each category in its dominant area."""
    cat_areas = defaultdict(list)
    for word, area in word_to_area.items():
        cat_areas[ground_truth[word]].append(area)

    total = 0
    correct = 0
    for cat, areas in cat_areas.items():
        counts = Counter(areas)
        total += len(areas)
        correct += counts.most_common(1)[0][1]

    return correct / total if total > 0 else 0.0


def compute_chance_purity(ground_truth: Dict[str, str],
                          n_areas: int, n_samples: int = 1000) -> float:
    """Monte Carlo estimate of purity for random area assignments."""
    rng = np.random.default_rng(42)
    words = list(ground_truth.keys())
    labels = [ground_truth[w] for w in words]

    purities = []
    for _ in range(n_samples):
        random_areas = rng.integers(0, n_areas, size=len(words))
        area_labels = defaultdict(list)
        for i, a in enumerate(random_areas):
            area_labels[a].append(labels[i])

        correct = 0
        for _, labs in area_labels.items():
            if labs:
                correct += Counter(labs).most_common(1)[0][1]
        purities.append(correct / len(words))

    return float(np.mean(purities))


def build_contingency_table(word_to_area: Dict[str, str],
                            ground_truth: Dict[str, str],
                            core_areas: List[str]) -> Dict[str, Any]:
    """Build area × category contingency table."""
    categories = sorted(set(ground_truth.values()))
    table = {area: Counter() for area in core_areas}

    for word, area in word_to_area.items():
        table[area][ground_truth[word]] += 1

    rows = []
    for area in core_areas:
        row = {"area": area}
        for cat in categories:
            row[cat] = table[area].get(cat, 0)
        row["total"] = sum(table[area].values())
        rows.append(row)

    return {"categories": categories, "rows": rows}


# ── Core trial runner ─────────────────────────────────────────────────

@dataclass
class SelfOrgConfig:
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.10
    w_max: float = 20.0
    n_core_areas: int = 8
    rounds_per_word: int = 10
    training_reps: int = 10
    add_modality_features: bool = False


def _find_winner(brain: Brain, core_areas: List[str]) -> Optional[str]:
    """Return the name of the area with active winners, or None."""
    for area_name in core_areas:
        if len(brain.areas[area_name].winners) > 0:
            return area_name
    return None


def run_trial(
    cfg: SelfOrgConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run one self-organization trial.

    Training protocol (per word):
      1. Inhibit all core areas
      2. Competition phase (plasticity OFF): project stim → all areas,
         mutual inhibition selects the area with highest total drive
      3. Stabilization phase (plasticity ON): project stim → winner only
         with self-recurrence, strengthening stim→winner connections
    """
    brain = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    rng = np.random.default_rng(seed)

    # Create undifferentiated core areas
    core_areas = [f"CORE_{i}" for i in range(cfg.n_core_areas)]
    for area_name in core_areas:
        brain.add_area(area_name, cfg.n, cfg.k, cfg.beta)

    # Mutual inhibition: only one core area fires per projection
    brain.add_mutual_inhibition(core_areas)

    # Register all stimuli
    registered_stims = set()
    word_stim_names: Dict[str, List[str]] = {}

    for word, gc in VOCABULARY.items():
        phon = f"PHON_{word}"
        brain.add_stimulus(phon, cfg.k)
        registered_stims.add(phon)
        stims = [phon]

        for feat in get_word_features(word, gc):
            feat_name = f"FEAT_{feat}"
            if feat_name not in registered_stims:
                brain.add_stimulus(feat_name, cfg.k)
                registered_stims.add(feat_name)
            stims.append(feat_name)

        if cfg.add_modality_features:
            modality = gc.dominant_modality
            mod_name = f"MOD_{modality.upper()}"
            if mod_name not in registered_stims:
                brain.add_stimulus(mod_name, cfg.k)
                registered_stims.add(mod_name)
            stims.append(mod_name)

        word_stim_names[word] = stims

    # ── Training ──────────────────────────────────────────────────
    word_list = list(VOCABULARY.keys())

    for rep in range(cfg.training_reps):
        rng.shuffle(word_list)

        for word in word_list:
            # Clear all core areas
            brain.inhibit_areas(core_areas)

            # Phase 1: COMPETITION (plasticity OFF)
            # Project stim → all areas; mutual inhibition picks winner
            all_areas_stim = {s: list(core_areas) for s in word_stim_names[word]}
            brain.disable_plasticity = True
            brain.project(all_areas_stim, {})
            brain.disable_plasticity = False

            # Find which area won
            winner = _find_winner(brain, core_areas)
            if winner is None:
                continue

            # Phase 2: STABILIZATION (plasticity ON, winner only)
            # Project stim → winner with self-recurrence
            winner_stim = {s: [winner] for s in word_stim_names[word]}
            winner_recur = {winner: [winner]}
            for _ in range(cfg.rounds_per_word - 1):
                brain.project(winner_stim, winner_recur)

    # ── Testing ───────────────────────────────────────────────────
    word_to_area: Dict[str, str] = {}
    word_assemblies: Dict[str, np.ndarray] = {}

    for word in VOCABULARY:
        brain.inhibit_areas(core_areas)

        # Competition phase (plasticity OFF for clean test)
        all_areas_stim = {s: list(core_areas) for s in word_stim_names[word]}
        brain.disable_plasticity = True
        brain.project(all_areas_stim, {})
        brain.disable_plasticity = False

        winner = _find_winner(brain, core_areas)
        if winner is None:
            word_to_area[word] = "NONE"
            word_assemblies[word] = np.array([], dtype=np.uint32)
            continue

        # Stabilize in winner for clean assembly snapshot
        winner_stim = {s: [winner] for s in word_stim_names[word]}
        winner_recur = {winner: [winner]}
        brain.disable_plasticity = True
        for _ in range(cfg.rounds_per_word - 1):
            brain.project(winner_stim, winner_recur)
        brain.disable_plasticity = False

        word_to_area[word] = winner
        word_assemblies[word] = np.array(
            brain.areas[winner].winners, dtype=np.uint32)

    # ── Compute metrics ───────────────────────────────────────────
    ground_truth = get_ground_truth()
    purity = compute_purity(word_to_area, ground_truth)
    completeness = compute_completeness(word_to_area, ground_truth)
    n_active = len(set(word_to_area.values()) - {"NONE"})

    # Assembly overlaps
    within_overlaps = []
    across_overlaps = []

    words_by_area = defaultdict(list)
    for word, area in word_to_area.items():
        words_by_area[area].append(word)

    for area, words in words_by_area.items():
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                w1, w2 = words[i], words[j]
                a1 = word_assemblies.get(w1, np.array([]))
                a2 = word_assemblies.get(w2, np.array([]))
                if len(a1) > 0 and len(a2) > 0:
                    ov = measure_overlap(a1, a2)
                    if ground_truth[w1] == ground_truth[w2]:
                        within_overlaps.append(ov)
                    else:
                        across_overlaps.append(ov)

    # Cross-area overlaps (always across-category by definition)
    all_area_names = list(words_by_area.keys())
    for ai in range(len(all_area_names)):
        for aj in range(ai + 1, len(all_area_names)):
            for w1 in words_by_area[all_area_names[ai]]:
                for w2 in words_by_area[all_area_names[aj]]:
                    a1 = word_assemblies.get(w1, np.array([]))
                    a2 = word_assemblies.get(w2, np.array([]))
                    if len(a1) > 0 and len(a2) > 0:
                        across_overlaps.append(measure_overlap(a1, a2))

    return {
        "word_to_area": word_to_area,
        "purity": purity,
        "completeness": completeness,
        "n_active_areas": n_active,
        "within_category_overlap": within_overlaps,
        "across_category_overlap": across_overlaps,
        "contingency": build_contingency_table(
            word_to_area, ground_truth, core_areas),
    }


# ── Main experiment ───────────────────────────────────────────────────

class SelfOrganizationExperiment(ExperimentBase):
    """Test self-organization of core areas via mutual inhibition."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="area_self_organization",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[SelfOrgConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or SelfOrgConfig(**{k: v for k, v in kwargs.items()
                                         if k in SelfOrgConfig.__dataclass_fields__})
        ground_truth = get_ground_truth()
        chance_pur = compute_chance_purity(ground_truth, cfg.n_core_areas)
        null_overlap = chance_overlap(cfg.k, cfg.n)

        self.log("=" * 70)
        self.log("Self-Organization of Core Areas via Mutual Inhibition")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  n_core_areas={cfg.n_core_areas}")
        self.log(f"  rounds_per_word={cfg.rounds_per_word}, "
                 f"training_reps={cfg.training_reps}")
        self.log(f"  chance purity = {chance_pur:.3f}")
        self.log(f"  chance overlap (k/n) = {null_overlap:.4f}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        # ── Condition A: specific features only ───────────────────
        self.log("\n--- Condition A: Specific features only ---")
        cfg_a = SelfOrgConfig(
            n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta, w_max=cfg.w_max,
            n_core_areas=cfg.n_core_areas,
            rounds_per_word=cfg.rounds_per_word,
            training_reps=cfg.training_reps,
            add_modality_features=False,
        )
        results_a = self._run_condition(cfg_a, n_seeds, "A")

        # ── Condition B: specific + modality features ─────────────
        self.log("\n--- Condition B: Specific + modality features ---")
        cfg_b = SelfOrgConfig(
            n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta, w_max=cfg.w_max,
            n_core_areas=cfg.n_core_areas,
            rounds_per_word=cfg.rounds_per_word,
            training_reps=cfg.training_reps,
            add_modality_features=True,
        )
        results_b = self._run_condition(cfg_b, n_seeds, "B")

        # ── Compare A vs B ────────────────────────────────────────
        paired_purity = paired_ttest(
            results_b["purity_values"], results_a["purity_values"])
        paired_complete = paired_ttest(
            results_b["completeness_values"], results_a["completeness_values"])

        self.log("\n--- Condition A vs B (paired) ---")
        self.log(f"  Purity:       A={results_a['purity_mean']:.3f}  "
                 f"B={results_b['purity_mean']:.3f}  "
                 f"p={paired_purity['p']:.4f}")
        self.log(f"  Completeness: A={results_a['completeness_mean']:.3f}  "
                 f"B={results_b['completeness_mean']:.3f}  "
                 f"p={paired_complete['p']:.4f}")

        # ── Assemble metrics ──────────────────────────────────────
        metrics = {
            "condition_A_features_only": results_a["metrics"],
            "condition_B_with_modality": results_b["metrics"],
            "A_vs_B_purity": paired_purity,
            "A_vs_B_completeness": paired_complete,
            "chance_purity": chance_pur,
            "chance_overlap": null_overlap,
        }

        duration = self._stop_timer()
        self.log(f"\nTotal duration: {duration:.1f}s")

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "w_max": cfg.w_max, "n_core_areas": cfg.n_core_areas,
                "rounds_per_word": cfg.rounds_per_word,
                "training_reps": cfg.training_reps,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )

    def _run_condition(
        self,
        cfg: SelfOrgConfig,
        n_seeds: int,
        label: str,
    ) -> Dict[str, Any]:
        """Run one experimental condition across seeds."""
        ground_truth = get_ground_truth()
        chance_pur = compute_chance_purity(ground_truth, cfg.n_core_areas)
        null_overlap = chance_overlap(cfg.k, cfg.n)

        purity_values = []
        completeness_values = []
        n_active_values = []
        within_ov_values = []
        across_ov_values = []
        contingencies = []

        for s in range(n_seeds):
            seed = self.seed + s
            self.log(f"  Seed {s+1}/{n_seeds} ...")
            result = run_trial(cfg, seed)

            purity_values.append(result["purity"])
            completeness_values.append(result["completeness"])
            n_active_values.append(result["n_active_areas"])

            if result["within_category_overlap"]:
                within_ov_values.append(
                    np.mean(result["within_category_overlap"]))
            if result["across_category_overlap"]:
                across_ov_values.append(
                    np.mean(result["across_category_overlap"]))

            contingencies.append(result["contingency"])

        pur_mean = float(np.mean(purity_values))
        comp_mean = float(np.mean(completeness_values))
        active_mean = float(np.mean(n_active_values))
        within_mean = (float(np.mean(within_ov_values))
                       if within_ov_values else 0.0)
        across_mean = (float(np.mean(across_ov_values))
                       if across_ov_values else 0.0)

        pur_test = ttest_vs_null(purity_values, chance_pur)

        self.log(f"  Purity:       {pur_mean:.3f} (chance={chance_pur:.3f}, "
                 f"d={pur_test['d']:.2f}, p={pur_test['p']:.4f})")
        self.log(f"  Completeness: {comp_mean:.3f}")
        self.log(f"  Active areas: {active_mean:.1f}/{cfg.n_core_areas}")
        self.log(f"  Within-cat overlap: {within_mean:.4f}  "
                 f"Across-cat overlap: {across_mean:.4f}  "
                 f"(chance={null_overlap:.4f})")

        if contingencies:
            self._print_contingency(contingencies[0], label)

        metrics = {
            "purity": summarize(purity_values),
            "purity_vs_chance": pur_test,
            "completeness": summarize(completeness_values),
            "n_active_areas": summarize(n_active_values),
            "within_category_overlap": (
                summarize(within_ov_values) if within_ov_values else None),
            "across_category_overlap": (
                summarize(across_ov_values) if across_ov_values else None),
            "example_contingency": contingencies[0] if contingencies else None,
        }

        return {
            "metrics": metrics,
            "purity_values": purity_values,
            "completeness_values": completeness_values,
            "purity_mean": pur_mean,
            "completeness_mean": comp_mean,
        }

    def _print_contingency(self, contingency: Dict, label: str) -> None:
        """Pretty-print a contingency table."""
        cats = contingency["categories"]
        rows = contingency["rows"]

        hdr = f"  {'Area':<10}"
        for cat in cats:
            hdr += f"{cat:>10}"
        hdr += f"{'total':>8}"
        self.log(f"\n  Contingency table (seed 0, condition {label}):")
        self.log(hdr)
        self.log("  " + "-" * (10 + 10 * len(cats) + 8))

        for row in rows:
            if row["total"] == 0:
                continue
            line = f"  {row['area']:<10}"
            for cat in cats:
                line += f"{row.get(cat, 0):>10}"
            line += f"{row['total']:>8}"
            self.log(line)


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Self-Organization of Core Areas Experiment")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (fewer seeds, smaller network)")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--reps", type=int, default=None,
                        help="Training repetitions")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = SelfOrganizationExperiment(verbose=True)

    if args.quick:
        cfg = SelfOrgConfig(n=5000, k=50, training_reps=5)
        n_seeds = args.seeds or 5
    else:
        cfg = SelfOrgConfig()
        n_seeds = args.seeds or 10

    if args.n is not None:
        cfg.n = args.n
    if args.k is not None:
        cfg.k = args.k
    if args.beta is not None:
        cfg.beta = args.beta
    if args.reps is not None:
        cfg.training_reps = args.reps

    result = exp.run(n_seeds=n_seeds, config=cfg)

    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    print("\n" + "=" * 70)
    print("SELF-ORGANIZATION SUMMARY")
    print("=" * 70)

    m = result.metrics
    print(f"\nChance purity: {m['chance_purity']:.3f}")

    for cond_key, cond_label in [
        ("condition_A_features_only", "A (features only)"),
        ("condition_B_with_modality", "B (+ modality)"),
    ]:
        c = m[cond_key]
        pur = c["purity"]
        comp = c["completeness"]
        test = c["purity_vs_chance"]
        within = c.get("within_category_overlap")
        across = c.get("across_category_overlap")

        print(f"\nCondition {cond_label}:")
        print(f"  Purity:       {pur['mean']:.3f} +/- {pur['sem']:.3f}  "
              f"(d={test['d']:.2f}, p={test['p']:.4f})")
        print(f"  Completeness: {comp['mean']:.3f} +/- {comp['sem']:.3f}")
        print(f"  Active areas: {c['n_active_areas']['mean']:.1f}")
        if within:
            print(f"  Within-cat:   {within['mean']:.4f}")
        if across:
            print(f"  Across-cat:   {across['mean']:.4f}")

    ab = m["A_vs_B_purity"]
    print(f"\nA vs B purity: t={ab['t']:.2f}, p={ab['p']:.4f}, d={ab['d']:.2f}")
    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
