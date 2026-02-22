"""
Variable Binding and Selective Retrieval

Tests whether sequential word-role bindings via bidirectional projection
support selective retrieval -- the fundamental variable binding mechanism
for compositional representation.

When multiple words are sequentially bound to a structural role area,
each creates a distinct binding assembly in the ROLE area (different
neurons for different words). The WORD<->ROLE fiber weights encode which
WORD neurons correspond to which ROLE neurons. Reinstating a specific
binding assembly in ROLE and projecting back to WORD should recover the
original word's assembly.

This tests the binding CAPACITY of a single structural area: how many
word-role bindings can coexist before interference degrades retrieval?

Protocol:
  1. Areas: WORD (word assemblies), ROLE (structural bindings)
  2. Build M word assemblies in WORD via stimulus projection
  3. Sequential binding: for each word, activate in WORD, project
     bidirectionally to ROLE with Hebbian learning
  4. Retrieval: for each word, re-evoke its WORD assembly (plasticity OFF),
     project to ROLE to reinstate binding, then project ROLE -> WORD
     to retrieve. Compare with original word assemblies.

Hypotheses:

H1: Binding retrieval -- After binding M words to ROLE, reinstating each
    binding and projecting back to WORD recovers the correct word's
    assembly with overlap >> chance (k/n).
    Null: overlap = k/n.

H2: Selective retrieval -- Overlap with the correct (bound) word exceeds
    overlap with all other words. Selectivity > 0.
    Null: overlap with correct = overlap with incorrect.

H3: Capacity scaling -- Retrieval quality degrades as M increases.
    There exists a critical capacity beyond which retrieval drops to chance.
    Tested with M = 2, 3, 5, 8.

Usage:
    uv run python research/experiments/primitives/test_binding_retrieval.py
    uv run python research/experiments/primitives/test_binding_retrieval.py --quick
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
    measure_overlap,
    chance_overlap,
    summarize,
    ttest_vs_null,
)
from src.core.brain import Brain


WORDS = ["dog", "cat", "bird", "boy", "girl", "ball", "book", "food"]


@dataclass
class BindingConfig:
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.10
    w_max: float = 20.0
    lexicon_rounds: int = 20
    binding_rounds: int = 10


def _activate_word(brain: Brain, stim_name: str, area: str, rounds: int):
    """Activate a word's assembly in its core area via stimulus projection."""
    brain.inhibit_areas([area])
    for _ in range(rounds):
        brain.project({stim_name: [area]}, {area: [area]})


def _degenerate_result(n_words):
    """Return a degenerate result when word assemblies can't be re-evoked."""
    return {
        "n_words": n_words,
        "correct_overlap_mean": 0.0,
        "incorrect_overlap_mean": 0.0,
        "selectivity": 0.0,
        "top1_accuracy": 0.0,
        "correct_overlaps": [0.0] * n_words,
        "incorrect_overlaps": [0.0] * n_words,
        "degenerate": True,
    }


def run_capacity_trial(
    cfg: BindingConfig,
    n_words: int,
    seed: int,
) -> Dict[str, Any]:
    """Run one binding capacity trial with n_words bound to a single ROLE area.

    Returns per-word retrieval overlap and accuracy metrics.
    """
    brain = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    words = WORDS[:n_words]

    # Create areas
    brain.add_area("WORD", cfg.n, cfg.k, cfg.beta)
    brain.add_area("ROLE", cfg.n, cfg.k, cfg.beta)

    # Register stimuli
    for word in words:
        brain.add_stimulus(f"PHON_{word}", cfg.k)

    # -- Build word assemblies -----------------------------------
    # Reset area-to-area connections between words to prevent assembly
    # collapse.  Stimulus->WORD connections are PRESERVED by
    # reset_area_connections, so each word remains re-evocable.
    word_assemblies = {}
    for word in words:
        brain._engine.reset_area_connections("WORD")
        _activate_word(brain, f"PHON_{word}", "WORD", cfg.lexicon_rounds)
        word_assemblies[word] = np.array(
            brain.areas["WORD"].winners, dtype=np.uint32)

    # Verify re-evocability: can we re-activate each word?
    brain.disable_plasticity = True
    for word in words:
        _activate_word(brain, f"PHON_{word}", "WORD", 3)
        reevoked = np.array(brain.areas["WORD"].winners, dtype=np.uint32)
        recall = measure_overlap(reevoked, word_assemblies[word])
        if recall < 0.5:
            # Assembly not recoverable -- degenerate case
            brain.disable_plasticity = False
            return _degenerate_result(n_words)
    brain.disable_plasticity = False

    # -- Sequential binding: each word -> ROLE -------------------
    # Co-project word stimulus to BOTH areas: stimulus anchors WORD
    # and bootstraps ROLE.  Bidirectional area projections build the
    # WORD<->ROLE bridges via Hebbian learning.
    for word in words:
        _activate_word(brain, f"PHON_{word}", "WORD", 3)
        brain.inhibit_areas(["ROLE"])
        for _ in range(cfg.binding_rounds):
            brain.project(
                {f"PHON_{word}": ["WORD", "ROLE"]},
                {"WORD": ["ROLE"],
                 "ROLE": ["WORD"]},
            )

    # -- Retrieval: re-evoke each binding, project back ----------
    brain.disable_plasticity = True

    correct_overlaps = []
    incorrect_overlaps = []
    top1_correct = 0

    for word in words:
        # Re-evoke word's assembly in WORD
        _activate_word(brain, f"PHON_{word}", "WORD", 3)

        # Forward project to ROLE (reinstate binding assembly)
        brain.inhibit_areas(["ROLE"])
        for _ in range(3):
            brain.project({}, {"WORD": ["ROLE"]})

        # Reverse project from ROLE -> WORD (retrieve)
        brain.inhibit_areas(["WORD"])
        for _ in range(3):
            brain.project({}, {"ROLE": ["WORD"]})

        retrieved = np.array(
            brain.areas["WORD"].winners, dtype=np.uint32)

        # Measure overlap with all word assemblies
        overlaps = {}
        for w in words:
            overlaps[w] = measure_overlap(retrieved, word_assemblies[w])

        correct_overlaps.append(overlaps[word])
        incorrect = [overlaps[w] for w in words if w != word]
        incorrect_overlaps.append(float(np.mean(incorrect)) if incorrect else 0.0)

        # Top-1 accuracy
        best = max(overlaps, key=overlaps.get)
        if best == word:
            top1_correct += 1

    brain.disable_plasticity = False

    selectivity = float(
        np.mean(correct_overlaps) - np.mean(incorrect_overlaps))

    return {
        "n_words": n_words,
        "correct_overlap_mean": float(np.mean(correct_overlaps)),
        "incorrect_overlap_mean": float(np.mean(incorrect_overlaps)),
        "selectivity": selectivity,
        "top1_accuracy": top1_correct / n_words,
        "correct_overlaps": correct_overlaps,
        "incorrect_overlaps": incorrect_overlaps,
    }


def run_two_role_trial(
    cfg: BindingConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run a two-role binding trial (agent + patient) with retrieval.

    Tests cross-role selectivity: binding dog=AGENT, cat=PATIENT,
    then retrieving from each role should recover the correct word.
    """
    brain = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    rng = np.random.default_rng(seed)

    nouns = WORDS[:5]

    brain.add_area("WORD", cfg.n, cfg.k, cfg.beta)
    brain.add_area("ROLE_AGENT", cfg.n, cfg.k, cfg.beta)
    brain.add_area("ROLE_PATIENT", cfg.n, cfg.k, cfg.beta)

    for word in nouns:
        brain.add_stimulus(f"PHON_{word}", cfg.k)

    # Build word lexicon with reset between words (preserves stimulus
    # connections but clears self-recurrence to prevent assembly collapse)
    word_assemblies = {}
    for word in nouns:
        brain._engine.reset_area_connections("WORD")
        _activate_word(brain, f"PHON_{word}", "WORD", cfg.lexicon_rounds)
        word_assemblies[word] = np.array(
            brain.areas["WORD"].winners, dtype=np.uint32)

    # Generate test sentences (random agent-patient pairs)
    n_test = 10
    agent_correct = 0
    patient_correct = 0
    agent_selectivity_vals = []
    patient_selectivity_vals = []

    for _ in range(n_test):
        agent = rng.choice(nouns)
        patient = rng.choice([n for n in nouns if n != agent])

        # Bind agent -> ROLE_AGENT (co-projection)
        _activate_word(brain, f"PHON_{agent}", "WORD", 3)
        brain.inhibit_areas(["ROLE_AGENT"])
        for _ in range(cfg.binding_rounds):
            brain.project(
                {f"PHON_{agent}": ["WORD", "ROLE_AGENT"]},
                {"WORD": ["ROLE_AGENT"],
                 "ROLE_AGENT": ["WORD"]},
            )

        # Bind patient -> ROLE_PATIENT (co-projection)
        _activate_word(brain, f"PHON_{patient}", "WORD", 3)
        brain.inhibit_areas(["ROLE_PATIENT"])
        for _ in range(cfg.binding_rounds):
            brain.project(
                {f"PHON_{patient}": ["WORD", "ROLE_PATIENT"]},
                {"WORD": ["ROLE_PATIENT"],
                 "ROLE_PATIENT": ["WORD"]},
            )

        # Retrieval (plasticity OFF)
        brain.disable_plasticity = True

        # Retrieve from ROLE_AGENT
        _activate_word(brain, f"PHON_{agent}", "WORD", 3)
        brain.inhibit_areas(["ROLE_AGENT"])
        for _ in range(3):
            brain.project({}, {"WORD": ["ROLE_AGENT"]})
        brain.inhibit_areas(["WORD"])
        for _ in range(3):
            brain.project({}, {"ROLE_AGENT": ["WORD"]})
        agent_retrieved = np.array(
            brain.areas["WORD"].winners, dtype=np.uint32)

        agent_overlaps = {w: measure_overlap(agent_retrieved, word_assemblies[w])
                          for w in nouns}
        best_agent = max(agent_overlaps, key=agent_overlaps.get)
        if best_agent == agent:
            agent_correct += 1
        agent_sel = agent_overlaps[agent] - agent_overlaps[patient]
        agent_selectivity_vals.append(agent_sel)

        # Retrieve from ROLE_PATIENT
        _activate_word(brain, f"PHON_{patient}", "WORD", 3)
        brain.inhibit_areas(["ROLE_PATIENT"])
        for _ in range(3):
            brain.project({}, {"WORD": ["ROLE_PATIENT"]})
        brain.inhibit_areas(["WORD"])
        for _ in range(3):
            brain.project({}, {"ROLE_PATIENT": ["WORD"]})
        patient_retrieved = np.array(
            brain.areas["WORD"].winners, dtype=np.uint32)

        patient_overlaps = {w: measure_overlap(patient_retrieved, word_assemblies[w])
                            for w in nouns}
        best_patient = max(patient_overlaps, key=patient_overlaps.get)
        if best_patient == patient:
            patient_correct += 1
        patient_sel = patient_overlaps[patient] - patient_overlaps[agent]
        patient_selectivity_vals.append(patient_sel)

        brain.disable_plasticity = False

    return {
        "agent_accuracy": agent_correct / n_test,
        "patient_accuracy": patient_correct / n_test,
        "agent_selectivity": float(np.mean(agent_selectivity_vals)),
        "patient_selectivity": float(np.mean(patient_selectivity_vals)),
    }


class BindingRetrievalExperiment(ExperimentBase):
    """Test variable binding capacity and selective retrieval."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="binding_retrieval",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[BindingConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or BindingConfig(
            **{k: v for k, v in kwargs.items()
               if k in BindingConfig.__dataclass_fields__})

        null = chance_overlap(cfg.k, cfg.n)

        self.log("=" * 70)
        self.log("Variable Binding and Selective Retrieval")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  binding_rounds={cfg.binding_rounds}")
        self.log(f"  chance overlap (k/n)={null:.4f}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        metrics = {}

        # -- H1 + H2 + H3: Capacity sweep -----------------------
        self.log("\nH1-H3: Binding capacity sweep")
        self.log("-" * 50)

        capacity_levels = [2, 3, 5, 8]
        h1_results = []

        for m in capacity_levels:
            correct_vals = []
            incorrect_vals = []
            selectivity_vals = []
            top1_vals = []

            for s in range(n_seeds):
                result = run_capacity_trial(cfg, m, self.seed + s)
                correct_vals.append(result["correct_overlap_mean"])
                incorrect_vals.append(result["incorrect_overlap_mean"])
                selectivity_vals.append(result["selectivity"])
                top1_vals.append(result["top1_accuracy"])

            correct_test = ttest_vs_null(correct_vals, null)
            select_test = ttest_vs_null(selectivity_vals, 0.0)

            row = {
                "n_words": m,
                "correct_overlap": summarize(correct_vals),
                "incorrect_overlap": summarize(incorrect_vals),
                "selectivity": summarize(selectivity_vals),
                "selectivity_test": select_test,
                "correct_vs_chance": correct_test,
                "top1_accuracy": summarize(top1_vals),
            }
            h1_results.append(row)

            self.log(
                f"  M={m}: correct={np.mean(correct_vals):.3f}  "
                f"incorrect={np.mean(incorrect_vals):.3f}  "
                f"selectivity={np.mean(selectivity_vals):.3f}  "
                f"top1={np.mean(top1_vals):.3f}  "
                f"d={select_test['d']:.2f}  p={select_test['p']:.4f}")

        metrics["capacity_sweep"] = h1_results

        # -- Two-role binding ------------------------------------
        self.log("\nTwo-role binding (agent + patient)")
        self.log("-" * 50)

        agent_acc_vals = []
        patient_acc_vals = []
        agent_sel_vals = []
        patient_sel_vals = []

        for s in range(n_seeds):
            result = run_two_role_trial(cfg, self.seed + s)
            agent_acc_vals.append(result["agent_accuracy"])
            patient_acc_vals.append(result["patient_accuracy"])
            agent_sel_vals.append(result["agent_selectivity"])
            patient_sel_vals.append(result["patient_selectivity"])

        agent_test = ttest_vs_null(agent_sel_vals, 0.0)
        patient_test = ttest_vs_null(patient_sel_vals, 0.0)

        self.log(
            f"  Agent acc:  {np.mean(agent_acc_vals):.3f}  "
            f"selectivity={np.mean(agent_sel_vals):.3f}  "
            f"d={agent_test['d']:.2f}  p={agent_test['p']:.4f}")
        self.log(
            f"  Patient acc: {np.mean(patient_acc_vals):.3f}  "
            f"selectivity={np.mean(patient_sel_vals):.3f}  "
            f"d={patient_test['d']:.2f}  p={patient_test['p']:.4f}")

        metrics["two_role"] = {
            "agent_accuracy": summarize(agent_acc_vals),
            "patient_accuracy": summarize(patient_acc_vals),
            "agent_selectivity": summarize(agent_sel_vals),
            "agent_selectivity_test": agent_test,
            "patient_selectivity": summarize(patient_sel_vals),
            "patient_selectivity_test": patient_test,
        }

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "w_max": cfg.w_max,
                "binding_rounds": cfg.binding_rounds,
                "lexicon_rounds": cfg.lexicon_rounds,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Binding Retrieval Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = BindingRetrievalExperiment(verbose=True)

    if args.quick:
        cfg = BindingConfig(n=5000, k=50)
        n_seeds = args.seeds or 5
    else:
        cfg = BindingConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)

    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    print("\n" + "=" * 70)
    print("BINDING RETRIEVAL SUMMARY")
    print("=" * 70)

    m = result.metrics

    print("\nCapacity sweep:")
    for r in m["capacity_sweep"]:
        M = r["n_words"]
        corr = r["correct_overlap"]["mean"]
        sel = r["selectivity"]["mean"]
        t1 = r["top1_accuracy"]["mean"]
        d = r["selectivity_test"]["d"]
        p = r["selectivity_test"]["p"]
        print(f"  M={M}: correct={corr:.3f}  selectivity={sel:.3f}  "
              f"top1={t1:.3f}  d={d:.2f}  p={p:.4f}")

    tr = m["two_role"]
    print(f"\nTwo-role binding:")
    print(f"  Agent:   acc={tr['agent_accuracy']['mean']:.3f}  "
          f"selectivity={tr['agent_selectivity']['mean']:.3f}  "
          f"d={tr['agent_selectivity_test']['d']:.2f}")
    print(f"  Patient: acc={tr['patient_accuracy']['mean']:.3f}  "
          f"selectivity={tr['patient_selectivity']['mean']:.3f}  "
          f"d={tr['patient_selectivity_test']['d']:.2f}")

    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
