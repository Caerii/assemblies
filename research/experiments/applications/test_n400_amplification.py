"""
N400 Amplification Experiment

The N400 diagnostic experiment (test_n400_diagnostic.py) found:
- H3: Convergence speed priming at round 0 (d=0.77 rel vs unrel)
- H5: Semantic layer with k=300 saturated at 1.000

This experiment systematically tests interventions to amplify the
convergence-speed priming signal, aiming for larger effect sizes
and a robust N400-like pattern.

Key insight from the diagnostic: the round-0 priming effect comes from
ACTIVATION persistence — after projecting the prime, some of its neurons
remain active and overlap with the target's assembly when both share
features (e.g., ANIMAL).  This is an analogue of the brain's subthreshold
pre-activation that produces the N400 reduction for expected words.

Interventions tested:

A (baseline): Phon-only prime, no activation reset.  Same as H3 in the
    diagnostic, included as control.

B (weight isolation): Phon-only prime, then inhibit_areas() to clear
    activations but keep Hebbian weight traces.  Tests whether weights
    alone (without residual activation) produce priming.

C (grounded prime): Prime projects phon + grounding features.  Shared
    ANIMAL features should create stronger activation overlap with
    related targets.

D (grounded + reset): Grounded prime, then inhibit_areas().  Tests
    whether grounding-strengthened weight traces alone produce priming.

E (full grounded): Phon + grounding for BOTH prime and target.  The
    shared grounding features (ANIMAL) directly create assembly overlap
    during target processing.

F (semantic hub): Add a SEMANTIC area with moderate k, route words
    through it.  Tests whether a dedicated semantic layer produces
    priming that the core area cannot.

For each intervention, we measure:
- Round-0 convergence overlap (the earliest N400 signal)
- Cohen's d for related vs unrelated and related vs no-prime
- Convergence curves for the first 6 rounds

Statistical methodology:
- Within-subject design: same parser, same target, different primes.
- Paired t-tests for related vs unrelated conditions.
- Cohen's d effect sizes across seeds x test pairs.

References:
- Kutas & Hillyard (1980). Reading senseless sentences.
- Kutas & Federmeier (2011). Thirty years and counting: N400.
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
    ExperimentBase, ExperimentResult, summarize, paired_ttest,
)
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence
from src.assembly_calculus.emergent.areas import NOUN_CORE, VERB_CORE
from src.assembly_calculus.ops import project, _snap
from src.assembly_calculus.assembly import overlap as asm_overlap


@dataclass
class AmpConfig:
    """Configuration for N400 amplification experiment."""
    n: int = 50000
    k: int = 100
    n_seeds: int = 5
    p: float = 0.05
    beta: float = 0.05
    rounds: int = 10
    semantic_k: int = 150   # moderate k for semantic area (not 300!)
    max_conv_rounds: int = 8


# -- Vocabulary (same as N400 diagnostic) ------------------------------------

def _build_vocab() -> Dict[str, GroundingContext]:
    return {
        "dog":    GroundingContext(visual=["DOG", "ANIMAL"]),
        "cat":    GroundingContext(visual=["CAT", "ANIMAL"]),
        "bird":   GroundingContext(visual=["BIRD", "ANIMAL"]),
        "horse":  GroundingContext(visual=["HORSE", "ANIMAL"]),
        "fish":   GroundingContext(visual=["FISH", "ANIMAL"]),
        "table":  GroundingContext(visual=["TABLE", "FURNITURE"]),
        "chair":  GroundingContext(visual=["CHAIR", "FURNITURE"]),
        "book":   GroundingContext(visual=["BOOK", "OBJECT"]),
        "ball":   GroundingContext(visual=["BALL", "OBJECT"]),
        "car":    GroundingContext(visual=["CAR", "VEHICLE"]),
        "chases": GroundingContext(motor=["CHASING", "PURSUIT"]),
        "sees":   GroundingContext(motor=["SEEING", "PERCEPTION"]),
        "finds":  GroundingContext(motor=["FINDING", "PERCEPTION"]),
        "the":    GroundingContext(),
    }


def _build_training(vocab):
    def ctx(w):
        return vocab[w]
    sentences = []
    for subj, verb, obj in [
        ("dog", "chases", "cat"), ("cat", "sees", "bird"),
        ("bird", "chases", "fish"), ("horse", "chases", "dog"),
        ("fish", "sees", "horse"), ("dog", "sees", "bird"),
        ("cat", "chases", "horse"), ("horse", "sees", "cat"),
        ("bird", "finds", "horse"), ("fish", "finds", "cat"),
    ]:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb),
                      ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))
    for subj, verb, obj in [
        ("dog", "finds", "ball"), ("cat", "sees", "book"),
        ("bird", "finds", "car"),
    ]:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb),
                      ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))
    return sentences


# Priming test triplets: (target, related_prime, unrelated_prime)
PRIMING_TESTS = [
    ("cat",   "dog",   "table"),
    ("bird",  "cat",   "chair"),
    ("horse", "fish",  "book"),
    ("fish",  "bird",  "car"),
    ("dog",   "horse", "ball"),
    ("table", "chair", "dog"),
    ("chair", "table", "cat"),
    ("book",  "ball",  "bird"),
]


# -- Convergence measurement functions ---------------------------------------

def _convergence_curve_A(
    parser: EmergentParser, prime: Optional[str], target: str,
    max_rounds: int,
) -> List[float]:
    """Intervention A: phon-only prime, no activation reset."""
    core = parser._word_core_area(target)
    lex = parser.core_lexicons.get(core, {}).get(target)
    if lex is None:
        return [0.0] * max_rounds

    parser.brain._engine.reset_area_connections(core)

    if prime:
        project(parser.brain, parser.stim_map[prime], core,
                rounds=parser.rounds)

    phon = parser.stim_map[target]
    overlaps = []
    parser.brain.project({phon: [core]}, {})
    overlaps.append(float(asm_overlap(_snap(parser.brain, core), lex)))
    for _ in range(max_rounds - 1):
        parser.brain.project({phon: [core]}, {core: [core]})
        overlaps.append(float(asm_overlap(_snap(parser.brain, core), lex)))
    return overlaps


def _convergence_curve_B(
    parser: EmergentParser, prime: Optional[str], target: str,
    max_rounds: int,
) -> List[float]:
    """Intervention B: phon-only prime, then clear activation (keep weights)."""
    core = parser._word_core_area(target)
    lex = parser.core_lexicons.get(core, {}).get(target)
    if lex is None:
        return [0.0] * max_rounds

    parser.brain._engine.reset_area_connections(core)

    if prime:
        project(parser.brain, parser.stim_map[prime], core,
                rounds=parser.rounds)
        # Clear activation but KEEP Hebbian weight traces
        parser.brain.inhibit_areas([core])

    phon = parser.stim_map[target]
    overlaps = []
    parser.brain.project({phon: [core]}, {})
    overlaps.append(float(asm_overlap(_snap(parser.brain, core), lex)))
    for _ in range(max_rounds - 1):
        parser.brain.project({phon: [core]}, {core: [core]})
        overlaps.append(float(asm_overlap(_snap(parser.brain, core), lex)))
    return overlaps


def _convergence_curve_C(
    parser: EmergentParser, prime: Optional[str], target: str,
    max_rounds: int,
) -> List[float]:
    """Intervention C: grounded prime (phon + grounding), phon-only target."""
    core = parser._word_core_area(target)
    lex = parser.core_lexicons.get(core, {}).get(target)
    if lex is None:
        return [0.0] * max_rounds

    parser.brain._engine.reset_area_connections(core)

    if prime:
        # Project prime with GROUNDING features (activates ANIMAL etc.)
        prime_phon = parser.stim_map[prime]
        prime_ctx = parser.word_grounding.get(prime, GroundingContext())
        grounding_names = parser._grounding_stim_names(prime_ctx)
        stim_dict = {prime_phon: [core]}
        for gs in grounding_names:
            stim_dict[gs] = [core]
        parser.brain.project(stim_dict, {})
        for _ in range(parser.rounds - 1):
            parser.brain.project(stim_dict, {core: [core]})

    # Target: phon only
    phon = parser.stim_map[target]
    overlaps = []
    parser.brain.project({phon: [core]}, {})
    overlaps.append(float(asm_overlap(_snap(parser.brain, core), lex)))
    for _ in range(max_rounds - 1):
        parser.brain.project({phon: [core]}, {core: [core]})
        overlaps.append(float(asm_overlap(_snap(parser.brain, core), lex)))
    return overlaps


def _convergence_curve_D(
    parser: EmergentParser, prime: Optional[str], target: str,
    max_rounds: int,
) -> List[float]:
    """Intervention D: grounded prime + activation reset (weight isolation)."""
    core = parser._word_core_area(target)
    lex = parser.core_lexicons.get(core, {}).get(target)
    if lex is None:
        return [0.0] * max_rounds

    parser.brain._engine.reset_area_connections(core)

    if prime:
        prime_phon = parser.stim_map[prime]
        prime_ctx = parser.word_grounding.get(prime, GroundingContext())
        grounding_names = parser._grounding_stim_names(prime_ctx)
        stim_dict = {prime_phon: [core]}
        for gs in grounding_names:
            stim_dict[gs] = [core]
        parser.brain.project(stim_dict, {})
        for _ in range(parser.rounds - 1):
            parser.brain.project(stim_dict, {core: [core]})
        # Clear activation, keep weight traces
        parser.brain.inhibit_areas([core])

    phon = parser.stim_map[target]
    overlaps = []
    parser.brain.project({phon: [core]}, {})
    overlaps.append(float(asm_overlap(_snap(parser.brain, core), lex)))
    for _ in range(max_rounds - 1):
        parser.brain.project({phon: [core]}, {core: [core]})
        overlaps.append(float(asm_overlap(_snap(parser.brain, core), lex)))
    return overlaps


def _convergence_curve_E(
    parser: EmergentParser, prime: Optional[str], target: str,
    max_rounds: int,
) -> List[float]:
    """Intervention E: grounded prime + grounded target (full grounded)."""
    core = parser._word_core_area(target)
    lex = parser.core_lexicons.get(core, {}).get(target)
    if lex is None:
        return [0.0] * max_rounds

    parser.brain._engine.reset_area_connections(core)

    if prime:
        prime_phon = parser.stim_map[prime]
        prime_ctx = parser.word_grounding.get(prime, GroundingContext())
        grounding_names = parser._grounding_stim_names(prime_ctx)
        stim_dict = {prime_phon: [core]}
        for gs in grounding_names:
            stim_dict[gs] = [core]
        parser.brain.project(stim_dict, {})
        for _ in range(parser.rounds - 1):
            parser.brain.project(stim_dict, {core: [core]})

    # Target: phon + grounding
    target_phon = parser.stim_map[target]
    target_ctx = parser.word_grounding.get(target, GroundingContext())
    target_grounding = parser._grounding_stim_names(target_ctx)
    target_stim = {target_phon: [core]}
    for gs in target_grounding:
        target_stim[gs] = [core]

    overlaps = []
    parser.brain.project(target_stim, {})
    overlaps.append(float(asm_overlap(_snap(parser.brain, core), lex)))
    for _ in range(max_rounds - 1):
        parser.brain.project(target_stim, {core: [core]})
        overlaps.append(float(asm_overlap(_snap(parser.brain, core), lex)))
    return overlaps


def _convergence_curve_F(
    parser: EmergentParser, prime: Optional[str], target: str,
    max_rounds: int, semantic_area: str, semantic_lexicon: dict,
) -> List[float]:
    """Intervention F: semantic hub with moderate k.

    Project prime phon+grounding -> core -> SEMANTIC, then target
    phon+grounding -> core -> SEMANTIC.  Measure target overlap in
    the SEMANTIC area (which has larger k for co-activation).
    """
    sem_lex = semantic_lexicon.get(target)
    if sem_lex is None:
        return [0.0] * max_rounds

    core = parser._word_core_area(target)

    # Reset SEMANTIC recurrence (not core, since we need core weights)
    parser.brain._engine.reset_area_connections(semantic_area)
    parser.brain.inhibit_areas([semantic_area])

    if prime:
        parser.brain._engine.reset_area_connections(core)
        prime_phon = parser.stim_map[prime]
        prime_ctx = parser.word_grounding.get(prime, GroundingContext())
        grounding_names = parser._grounding_stim_names(prime_ctx)
        stim_dict = {prime_phon: [core]}
        for gs in grounding_names:
            stim_dict[gs] = [core]
        # Prime: phon+grounding -> core -> SEMANTIC
        parser.brain.project(stim_dict, {})
        parser.brain.project(
            {}, {core: [core, semantic_area],
                 semantic_area: [semantic_area]})
        for _ in range(parser.rounds - 1):
            parser.brain.project(
                {}, {core: [semantic_area],
                     semantic_area: [semantic_area]})

    # Target: measure convergence in SEMANTIC area
    parser.brain._engine.reset_area_connections(core)
    target_phon = parser.stim_map[target]
    target_ctx = parser.word_grounding.get(target, GroundingContext())
    target_grounding = parser._grounding_stim_names(target_ctx)
    target_stim = {target_phon: [core]}
    for gs in target_grounding:
        target_stim[gs] = [core]

    overlaps = []
    # Round 0: project target through core to semantic
    parser.brain.project(target_stim, {})
    parser.brain.project(
        {}, {core: [semantic_area], semantic_area: [semantic_area]})
    asm = _snap(parser.brain, semantic_area)
    overlaps.append(float(asm_overlap(asm, sem_lex)))

    # Subsequent rounds
    for _ in range(max_rounds - 1):
        parser.brain.project(
            {}, {core: [semantic_area], semantic_area: [semantic_area]})
        asm = _snap(parser.brain, semantic_area)
        overlaps.append(float(asm_overlap(asm, sem_lex)))

    return overlaps


# -- Experiment ----------------------------------------------------------------

class N400AmplificationExperiment(ExperimentBase):
    """Test interventions to amplify the N400 convergence-speed signal."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="n400_amplification",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def _run_intervention(
        self, label: str, curve_fn, cfg: AmpConfig,
        vocab, training, seeds, extra_kwargs=None,
    ) -> Dict[str, Any]:
        """Run one intervention across all seeds and test pairs."""
        self.log(f"\n{'=' * 60}")
        self.log(f"Intervention {label}")
        self.log("=" * 60)

        conv_related = {r: [] for r in range(cfg.max_conv_rounds)}
        conv_unrelated = {r: [] for r in range(cfg.max_conv_rounds)}
        conv_noprime = {r: [] for r in range(cfg.max_conv_rounds)}

        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            # For intervention F, build semantic layer
            kwargs = {}
            if extra_kwargs and "semantic_setup" in extra_kwargs:
                kwargs = extra_kwargs["semantic_setup"](parser, cfg)

            for target, related, unrelated in PRIMING_TESTS:
                c_r = curve_fn(parser, related, target,
                               cfg.max_conv_rounds, **kwargs)
                c_u = curve_fn(parser, unrelated, target,
                               cfg.max_conv_rounds, **kwargs)
                c_n = curve_fn(parser, None, target,
                               cfg.max_conv_rounds, **kwargs)
                for r in range(cfg.max_conv_rounds):
                    conv_related[r].append(c_r[r])
                    conv_unrelated[r].append(c_u[r])
                    conv_noprime[r].append(c_n[r])

        # Log convergence table
        self.log(f"  {'Round':>5} {'NoPrime':>8} {'Related':>8} "
                 f"{'Unrelated':>10} {'R-U':>6} {'R-N':>6}")
        by_round = []
        for r in range(min(cfg.max_conv_rounds, 6)):
            mn = float(np.mean(conv_noprime[r]))
            mr = float(np.mean(conv_related[r]))
            mu = float(np.mean(conv_unrelated[r]))
            self.log(f"  {r:>5} {mn:>8.3f} {mr:>8.3f} {mu:>10.3f} "
                     f"{mr - mu:>+6.3f} {mr - mn:>+6.3f}")
            by_round.append({
                "round": r,
                "no_prime": mn, "related": mr, "unrelated": mu,
                "r_minus_u": mr - mu, "r_minus_n": mr - mn,
            })

        # Test at round 0
        t_ru = paired_ttest(conv_related[0], conv_unrelated[0])
        t_rn = paired_ttest(conv_related[0], conv_noprime[0])

        self.log(f"\n  Round 0 rel vs unrel: t={t_ru['t']:.2f} "
                 f"p={t_ru['p']:.4f} d={t_ru['d']:.2f} "
                 f"{'*' if t_ru['significant'] else ''}")
        self.log(f"  Round 0 rel vs none:  t={t_rn['t']:.2f} "
                 f"p={t_rn['p']:.4f} d={t_rn['d']:.2f} "
                 f"{'*' if t_rn['significant'] else ''}")

        # Also test at round 1 for latent effects
        t_ru_r1 = paired_ttest(conv_related[1], conv_unrelated[1])
        t_rn_r1 = paired_ttest(conv_related[1], conv_noprime[1])
        self.log(f"  Round 1 rel vs unrel: t={t_ru_r1['t']:.2f} "
                 f"p={t_ru_r1['p']:.4f} d={t_ru_r1['d']:.2f} "
                 f"{'*' if t_ru_r1['significant'] else ''}")

        return {
            "by_round": by_round,
            "round0_rel_vs_unrel": t_ru,
            "round0_rel_vs_none": t_rn,
            "round1_rel_vs_unrel": t_ru_r1,
            "round1_rel_vs_none": t_rn_r1,
            "raw_round0": {
                "related": conv_related[0],
                "unrelated": conv_unrelated[0],
                "noprime": conv_noprime[0],
            },
        }

    def run(self, quick: bool = False, **kwargs) -> ExperimentResult:
        self._start_timer()

        cfg = AmpConfig()
        if quick:
            cfg.n_seeds = 3

        vocab = _build_vocab()
        training = _build_training(vocab)
        seeds = list(range(cfg.n_seeds))

        self.log(f"Training: {len(training)} sentences")
        self.log(f"Seeds: {cfg.n_seeds}")
        self.log(f"Params: n={cfg.n}, k={cfg.k}, p={cfg.p}, "
                 f"beta={cfg.beta}, rounds={cfg.rounds}")

        # Run all interventions
        results_by_intervention = {}

        # A: Baseline (phon-only, no reset)
        results_by_intervention["A_baseline"] = self._run_intervention(
            "A: Baseline (phon-only, no reset)",
            _convergence_curve_A, cfg, vocab, training, seeds,
        )

        # B: Weight isolation (phon-only + activation reset)
        results_by_intervention["B_weight_only"] = self._run_intervention(
            "B: Weight isolation (phon + activation reset)",
            _convergence_curve_B, cfg, vocab, training, seeds,
        )

        # C: Grounded prime (phon+grounding, no reset)
        results_by_intervention["C_grounded_prime"] = self._run_intervention(
            "C: Grounded prime (phon+grounding -> phon target)",
            _convergence_curve_C, cfg, vocab, training, seeds,
        )

        # D: Grounded + weight isolation
        results_by_intervention["D_grounded_weight"] = self._run_intervention(
            "D: Grounded prime + activation reset",
            _convergence_curve_D, cfg, vocab, training, seeds,
        )

        # E: Full grounded (both prime and target)
        results_by_intervention["E_full_grounded"] = self._run_intervention(
            "E: Full grounded (phon+grounding for both)",
            _convergence_curve_E, cfg, vocab, training, seeds,
        )

        # F: Semantic hub with moderate k
        SEMANTIC = "SEMANTIC_HUB"
        animals = ["dog", "cat", "bird", "horse", "fish"]
        objects = ["table", "chair", "book", "ball", "car"]

        def setup_semantic(parser, cfg):
            sem_k = min(cfg.semantic_k, cfg.n // 10)
            parser.brain.add_area(SEMANTIC, cfg.n, sem_k, cfg.beta)

            # Build semantic lexicon: project each word through core
            semantic_lexicon = {}
            for word in animals + objects:
                core = parser._word_core_area(word)
                phon = parser.stim_map[word]
                word_ctx = parser.word_grounding.get(
                    word, GroundingContext())
                grounding_names = parser._grounding_stim_names(word_ctx)

                parser.brain._engine.reset_area_connections(core)
                parser.brain._engine.reset_area_connections(SEMANTIC)
                parser.brain.inhibit_areas([SEMANTIC])

                stim_dict = {phon: [core]}
                for gs in grounding_names:
                    stim_dict[gs] = [core]
                parser.brain.project(stim_dict, {})
                parser.brain.project(
                    {}, {core: [core, SEMANTIC],
                         SEMANTIC: [SEMANTIC]})
                for _ in range(parser.rounds - 1):
                    parser.brain.project(
                        {}, {core: [SEMANTIC],
                             SEMANTIC: [SEMANTIC]})
                semantic_lexicon[word] = _snap(parser.brain, SEMANTIC)

            return {
                "semantic_area": SEMANTIC,
                "semantic_lexicon": semantic_lexicon,
            }

        results_by_intervention["F_semantic_hub"] = self._run_intervention(
            f"F: Semantic hub (k={cfg.semantic_k})",
            _convergence_curve_F, cfg, vocab, training, seeds,
            extra_kwargs={"semantic_setup": setup_semantic},
        )

        # ==============================================================
        # Summary comparison
        # ==============================================================
        self.log(f"\n{'=' * 60}")
        self.log("INTERVENTION COMPARISON (Round 0)")
        self.log("=" * 60)
        self.log(f"  {'Intervention':<35} {'d(R-U)':>8} {'p(R-U)':>8} "
                 f"{'d(R-N)':>8} {'R-U':>8}")

        best_d = 0
        best_label = ""
        for label, res in results_by_intervention.items():
            d_ru = res["round0_rel_vs_unrel"]["d"]
            p_ru = res["round0_rel_vs_unrel"]["p"]
            d_rn = res["round0_rel_vs_none"]["d"]
            r_u = res["by_round"][0]["r_minus_u"]
            sig = "*" if res["round0_rel_vs_unrel"]["significant"] else ""
            self.log(f"  {label:<35} {d_ru:>7.2f}{sig} {p_ru:>8.4f} "
                     f"{d_rn:>8.2f} {r_u:>+8.3f}")
            if d_ru > best_d:
                best_d = d_ru
                best_label = label

        self.log(f"\n  Best intervention: {best_label} (d={best_d:.2f})")

        # Also show round 1 comparison
        self.log(f"\n  {'Intervention':<35} {'d(R-U) r1':>10} "
                 f"{'p(R-U) r1':>10}")
        for label, res in results_by_intervention.items():
            d_r1 = res["round1_rel_vs_unrel"]["d"]
            p_r1 = res["round1_rel_vs_unrel"]["p"]
            sig = "*" if res["round1_rel_vs_unrel"]["significant"] else ""
            self.log(f"  {label:<35} {d_r1:>9.2f}{sig} {p_r1:>10.4f}")

        duration = self._stop_timer()
        self.log(f"\n  Duration: {duration:.1f}s")

        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "rounds": cfg.rounds, "n_seeds": cfg.n_seeds,
                "semantic_k": cfg.semantic_k,
                "max_conv_rounds": cfg.max_conv_rounds,
                "n_training": len(training),
            },
            metrics=results_by_intervention,
            duration_seconds=duration,
        )
        self.save_result(result)
        return result


def main():
    parser = argparse.ArgumentParser(
        description="N400 amplification experiment")
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer seeds for faster iteration")
    args = parser.parse_args()

    exp = N400AmplificationExperiment(verbose=True)
    result = exp.run(quick=args.quick)

    print(f"\nCompleted in {result.duration_seconds:.1f}s")
    for label, res in result.metrics.items():
        d = res["round0_rel_vs_unrel"]["d"]
        p = res["round0_rel_vs_unrel"]["p"]
        sig = "*" if res["round0_rel_vs_unrel"]["significant"] else ""
        print(f"  {label}: d={d:.2f} p={p:.4f} {sig}")


if __name__ == "__main__":
    main()
