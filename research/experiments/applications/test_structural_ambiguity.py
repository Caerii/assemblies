"""
Structural Ambiguity Experiment (Tier 2)

Tests how the Assembly Calculus parser handles structurally ambiguous
sentences, specifically prepositional phrase (PP) attachment ambiguity.

Scientific Questions:
1. Does the parser resolve PP attachment ambiguity differently depending
   on training bias (instrument vs modifier)?
2. Does training on instrument-biased sentences cause PPs to attach to VP?
3. Does training on modifier-biased sentences cause PPs to attach to NP?
4. Does the parser deterministically settle on one parse, with no
   lingering ambiguity in the assembly state?

Hypotheses:
H1: PP attachment in "saw the man with the telescope" produces different
    role-area assemblies depending on whether training is biased toward
    instrument readings ("cut bread with knife") or modifier readings
    ("saw man with hat").
H2: Instrument-biased training causes the PP test assembly to overlap
    more with VP training exemplars than with NP training exemplars.
H3: Modifier-biased training causes the PP test assembly to overlap
    more with NP training exemplars than with VP training exemplars.
H4: The parser settles deterministically: self-projection of the PP
    assembly converges to a stable attractor (no split representation).

Protocol:
1. Build vocabulary with nouns, verbs, prepositions, and determiners,
   including items for ambiguous PP-attachment constructions.
2. Condition A (instrument bias): Train with sentences where PP is
   clearly an instrument ("cut the bread with the knife").
3. Condition B (modifier bias): Train with sentences where PP is
   clearly a modifier ("saw the man with the hat").
4. Test: Parse the ambiguous sentence "saw the man with the telescope"
   under each condition.
5. Measure VP-area and NP-area overlap between the test PP assembly
   and each condition's training exemplars.

Statistical methodology:
- Each condition replicated across n_seeds independent random seeds.
- VP vs NP attachment scores compared via paired t-test.
- Determinism assessed by self-projection convergence rate.
- Effect sizes reported as Cohen's d.

References:
- Papadimitriou et al., PNAS 117(25):14464-14472, 2020
- Mitropolsky & Papadimitriou (2025), "Simulated Language Acquisition."
- Frazier & Fodor (1978), "The Sausage Machine" (PP attachment).
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any

from research.experiments.base import ExperimentBase, ExperimentResult, summarize, ttest_vs_null
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence


@dataclass
class AmbiguityConfig:
    """Configuration for the structural ambiguity experiment."""
    n: int = 10000           # neurons per area
    k: int = 100             # assembly size
    n_seeds: int = 5         # independent replications
    p: float = 0.05          # connection probability
    beta: float = 0.1        # Hebbian plasticity rate
    rounds: int = 10         # projection rounds
    convergence_rounds: int = 15  # self-projection rounds for H4


# -- Vocabulary ----------------------------------------------------------------

def _build_ambiguity_vocab() -> Dict[str, GroundingContext]:
    """Build vocabulary for PP-attachment ambiguity experiments.

    Includes nouns that can serve as instruments vs modifiers,
    verbs that take instruments vs those that take modifiers,
    and prepositions for PP construction.
    """
    vocab = {
        # Nouns — agents
        "man":       GroundingContext(visual=["MAN", "PERSON"]),
        "woman":     GroundingContext(visual=["WOMAN", "PERSON"]),
        "boy":       GroundingContext(visual=["BOY", "PERSON"]),
        "girl":      GroundingContext(visual=["GIRL", "PERSON"]),

        # Nouns — patients / objects
        "bread":     GroundingContext(visual=["BREAD", "FOOD"]),
        "cake":      GroundingContext(visual=["CAKE", "FOOD"]),
        "paper":     GroundingContext(visual=["PAPER", "OBJECT"]),
        "rope":      GroundingContext(visual=["ROPE", "OBJECT"]),

        # Nouns — instruments (clearly tool-like)
        "knife":     GroundingContext(visual=["KNIFE", "TOOL"]),
        "fork":      GroundingContext(visual=["FORK", "TOOL"]),
        "scissors":  GroundingContext(visual=["SCISSORS", "TOOL"]),
        "spoon":     GroundingContext(visual=["SPOON", "TOOL"]),

        # Nouns — modifiers (clearly attribute-like)
        "hat":       GroundingContext(visual=["HAT", "CLOTHING"]),
        "glasses":   GroundingContext(visual=["GLASSES", "CLOTHING"]),
        "scarf":     GroundingContext(visual=["SCARF", "CLOTHING"]),
        "badge":     GroundingContext(visual=["BADGE", "CLOTHING"]),

        # Ambiguous noun (can be instrument or modifier)
        "telescope": GroundingContext(visual=["TELESCOPE", "OBJECT"]),

        # Verbs — instrument-biased (actions that use tools)
        "cut":       GroundingContext(motor=["CUTTING", "ACTION"]),
        "sliced":    GroundingContext(motor=["SLICING", "ACTION"]),
        "tied":      GroundingContext(motor=["TYING", "ACTION"]),
        "stirred":   GroundingContext(motor=["STIRRING", "ACTION"]),

        # Verbs — perception / neutral
        "saw":       GroundingContext(motor=["SEEING", "PERCEPTION"]),
        "watched":   GroundingContext(motor=["WATCHING", "PERCEPTION"]),
        "noticed":   GroundingContext(motor=["NOTICING", "PERCEPTION"]),
        "found":     GroundingContext(motor=["FINDING", "PERCEPTION"]),

        # Preposition
        "with":      GroundingContext(spatial=["WITH", "ACCOMPANY"]),

        # Determiners
        "the":       GroundingContext(),
        "a":         GroundingContext(),
    }
    return vocab


def _ctx(word: str, vocab: Dict[str, GroundingContext]) -> GroundingContext:
    """Look up grounding context for a word."""
    return vocab[word]


# -- Training sentence builders ------------------------------------------------

def _build_instrument_sentences(
    vocab: Dict[str, GroundingContext],
) -> List[GroundedSentence]:
    """Build training sentences where PP attaches to VP (instrument reading).

    Pattern: SUBJ VERB OBJ with INSTRUMENT
    The PP "with X" modifies the verb (how the action was performed).
    """
    sentences = []
    instrument_triples = [
        ("man", "cut", "bread", "knife"),
        ("woman", "cut", "cake", "knife"),
        ("boy", "sliced", "bread", "knife"),
        ("girl", "sliced", "cake", "fork"),
        ("man", "tied", "rope", "scissors"),
        ("woman", "stirred", "cake", "spoon"),
        ("boy", "cut", "paper", "scissors"),
        ("girl", "tied", "rope", "scissors"),
        ("man", "stirred", "cake", "fork"),
        ("woman", "sliced", "paper", "knife"),
    ]
    for subj, verb, obj, instr in instrument_triples:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj, "with", "the", instr],
            contexts=[
                _ctx("the", vocab), _ctx(subj, vocab), _ctx(verb, vocab),
                _ctx("the", vocab), _ctx(obj, vocab), _ctx("with", vocab),
                _ctx("the", vocab), _ctx(instr, vocab),
            ],
            roles=[None, "agent", "action", None, "patient",
                   None, None, None],
        ))
    return sentences


def _build_modifier_sentences(
    vocab: Dict[str, GroundingContext],
) -> List[GroundedSentence]:
    """Build training sentences where PP attaches to NP (modifier reading).

    Pattern: SUBJ VERB OBJ with MODIFIER
    The PP "with X" modifies the object noun (which man/woman).
    """
    sentences = []
    modifier_triples = [
        ("boy", "saw", "man", "hat"),
        ("girl", "saw", "woman", "glasses"),
        ("man", "watched", "boy", "scarf"),
        ("woman", "watched", "girl", "badge"),
        ("boy", "noticed", "man", "glasses"),
        ("girl", "noticed", "woman", "hat"),
        ("man", "found", "boy", "badge"),
        ("woman", "found", "girl", "scarf"),
        ("boy", "saw", "woman", "scarf"),
        ("girl", "watched", "man", "badge"),
    ]
    for subj, verb, obj, modifier in modifier_triples:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj, "with", "the", modifier],
            contexts=[
                _ctx("the", vocab), _ctx(subj, vocab), _ctx(verb, vocab),
                _ctx("the", vocab), _ctx(obj, vocab), _ctx("with", vocab),
                _ctx("the", vocab), _ctx(modifier, vocab),
            ],
            roles=[None, "agent", "action", None, "patient",
                   None, None, None],
        ))
    return sentences


# -- Core measurement functions ------------------------------------------------

def measure_pp_attachment(
    parser: EmergentParser,
    sentence_words: List[str],
) -> Dict[str, float]:
    """Parse a PP-attachment sentence and measure VP and NP overlap.

    Projects the full sentence through the parser's merge pipeline:
    1. Merge subject + verb -> VP assembly
    2. Merge object noun + PP noun -> NP-modifier assembly
    3. Project PP noun toward both VP and NP areas
    4. Compare overlaps to determine attachment preference

    Args:
        parser: Trained EmergentParser instance.
        sentence_words: Token list, expected format:
            ["the", SUBJ, VERB, "the", OBJ, "with", "the", PP_NOUN]

    Returns:
        Dict with vp_overlap, np_overlap, and preference label.
    """
    from src.assembly_calculus.ops import project, merge, reciprocal_project, _snap
    from src.assembly_calculus.assembly import overlap as asm_overlap
    from src.assembly_calculus.emergent.areas import VERB_CORE, NOUN_CORE, VP, NP, PP

    subj = sentence_words[1]
    verb = sentence_words[2]
    obj_noun = sentence_words[4]
    pp_noun = sentence_words[7]

    subj_core = parser._word_core_area(subj)
    obj_core = parser._word_core_area(obj_noun)
    pp_core = parser._word_core_area(pp_noun)

    subj_phon = parser.stim_map.get(subj)
    verb_phon = parser.stim_map.get(verb)
    obj_phon = parser.stim_map.get(obj_noun)
    pp_phon = parser.stim_map.get(pp_noun)

    if not all([subj_phon, verb_phon, obj_phon, pp_phon]):
        return {"vp_overlap": 0.0, "np_overlap": 0.0, "preference": "UNKNOWN"}

    # Step 1: Build VP assembly (subject + verb)
    project(parser.brain, subj_phon, subj_core, rounds=parser.rounds)
    project(parser.brain, verb_phon, VERB_CORE, rounds=parser.rounds)
    vp_asm = merge(parser.brain, subj_core, VERB_CORE, VP, rounds=parser.rounds)

    # Step 2: Build NP assembly (object noun in NP area)
    project(parser.brain, obj_phon, obj_core, rounds=parser.rounds)
    np_asm = reciprocal_project(parser.brain, obj_core, NP, rounds=parser.rounds)

    # Step 3: Project PP noun into PP area
    project(parser.brain, pp_phon, pp_core, rounds=parser.rounds)
    pp_asm = reciprocal_project(parser.brain, pp_core, PP, rounds=parser.rounds)

    # Step 4: Now project PP towards VP and NP to see where it attaches
    # VP attachment: merge PP into VP area
    parser.brain.areas[PP].fix_assembly()
    parser.brain.areas[VP].fix_assembly()
    for _ in range(parser.rounds):
        parser.brain.project({}, {PP: [VP], VP: [VP]})
    parser.brain.areas[VP].unfix_assembly()
    parser.brain.areas[PP].unfix_assembly()
    vp_pp_asm = _snap(parser.brain, VP)
    vp_overlap = asm_overlap(vp_pp_asm, vp_asm)

    # NP attachment: merge PP into NP area
    parser.brain._engine.reset_area_connections(VP)
    parser.brain.areas[PP].fix_assembly()
    parser.brain.areas[NP].fix_assembly()
    for _ in range(parser.rounds):
        parser.brain.project({}, {PP: [NP], NP: [NP]})
    parser.brain.areas[NP].unfix_assembly()
    parser.brain.areas[PP].unfix_assembly()
    np_pp_asm = _snap(parser.brain, NP)
    np_overlap = asm_overlap(np_pp_asm, np_asm)

    # Clean up
    parser.brain._engine.reset_area_connections(VP)
    parser.brain._engine.reset_area_connections(NP)
    parser.brain._engine.reset_area_connections(PP)

    preference = "VP" if vp_overlap > np_overlap else "NP"
    return {
        "vp_overlap": float(vp_overlap),
        "np_overlap": float(np_overlap),
        "preference": preference,
    }


def measure_determinism(
    parser: EmergentParser,
    sentence_words: List[str],
    n_rounds: int = 15,
) -> Dict[str, float]:
    """Measure whether the PP assembly converges to a stable attractor.

    After parsing, self-projects the PP area assembly and tracks
    step-to-step overlap. High consecutive overlap (>0.95) indicates
    deterministic settlement with no split representation.

    Args:
        parser: Trained EmergentParser instance.
        sentence_words: Token list for the ambiguous sentence.
        n_rounds: Number of self-projection rounds.

    Returns:
        Dict with convergence metrics.
    """
    from src.assembly_calculus.ops import project, reciprocal_project, _snap
    from src.assembly_calculus.assembly import overlap as asm_overlap
    from src.assembly_calculus.emergent.areas import PP

    pp_noun = sentence_words[7]
    pp_core = parser._word_core_area(pp_noun)
    pp_phon = parser.stim_map.get(pp_noun)

    if pp_phon is None:
        return {"converged": False, "final_stability": 0.0, "steps_to_converge": n_rounds}

    project(parser.brain, pp_phon, pp_core, rounds=parser.rounds)
    reciprocal_project(parser.brain, pp_core, PP, rounds=parser.rounds)

    step_overlaps = []
    for _ in range(n_rounds):
        prev = _snap(parser.brain, PP)
        parser.brain.project({}, {PP: [PP]})
        curr = _snap(parser.brain, PP)
        step_overlaps.append(asm_overlap(prev, curr))

    parser.brain._engine.reset_area_connections(PP)

    # Find convergence point (3 consecutive rounds > 0.95)
    converged = False
    convergence_step = n_rounds
    for i in range(len(step_overlaps) - 2):
        if all(o >= 0.95 for o in step_overlaps[i:i + 3]):
            converged = True
            convergence_step = i + 1
            break

    return {
        "converged": converged,
        "final_stability": step_overlaps[-1] if step_overlaps else 0.0,
        "steps_to_converge": convergence_step,
        "overlap_history": step_overlaps,
    }


# -- Experiment class ----------------------------------------------------------

class StructuralAmbiguityExperiment(ExperimentBase):
    """Test how assembly calculus resolves PP-attachment ambiguity."""

    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="structural_ambiguity",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "applications",
            verbose=verbose,
        )

    def run(self, quick: bool = False, **kwargs) -> ExperimentResult:
        """Run the full structural ambiguity experiment.

        Args:
            quick: If True, use reduced seeds (2) for fast validation.
        """
        self._start_timer()

        cfg = AmbiguityConfig()
        if quick:
            cfg.n_seeds = 2

        seeds = list(range(cfg.n_seeds))
        vocab = _build_ambiguity_vocab()
        instrument_sents = _build_instrument_sentences(vocab)
        modifier_sents = _build_modifier_sentences(vocab)

        # The ambiguous test sentence
        ambig_sentence = ["the", "boy", "saw", "the", "man", "with", "the", "telescope"]

        null_overlap = cfg.k / cfg.n
        self.log(f"Null (chance) overlap: {null_overlap:.4f}")
        self.log(f"Instrument training sentences: {len(instrument_sents)}")
        self.log(f"Modifier training sentences: {len(modifier_sents)}")
        self.log(f"Ambiguous test: {' '.join(ambig_sentence)}")

        # ================================================================
        # H1 + H2: Instrument-biased training -> VP attachment
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H1/H2: Instrument-biased training")
        self.log("=" * 60)

        instr_vp_overlaps = []
        instr_np_overlaps = []
        instr_preferences = []

        for s in seeds:
            self.log(f"  Seed {s}/{cfg.n_seeds - 1}")
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=instrument_sents)

            result = measure_pp_attachment(parser, ambig_sentence)
            instr_vp_overlaps.append(result["vp_overlap"])
            instr_np_overlaps.append(result["np_overlap"])
            instr_preferences.append(result["preference"])
            self.log(f"    VP={result['vp_overlap']:.3f}  "
                     f"NP={result['np_overlap']:.3f}  "
                     f"pref={result['preference']}")

        instr_vp_stats = summarize(instr_vp_overlaps)
        instr_np_stats = summarize(instr_np_overlaps)
        # Test: VP overlap > NP overlap under instrument bias
        instr_deltas = [v - n for v, n in zip(instr_vp_overlaps, instr_np_overlaps)]
        h2_test = ttest_vs_null(instr_deltas, 0.0)
        instr_vp_pct = instr_preferences.count("VP") / len(instr_preferences)

        self.log(f"  Instrument bias results:")
        self.log(f"    VP overlap: {instr_vp_stats['mean']:.3f} +/- {instr_vp_stats['sem']:.3f}")
        self.log(f"    NP overlap: {instr_np_stats['mean']:.3f} +/- {instr_np_stats['sem']:.3f}")
        self.log(f"    VP preference: {instr_vp_pct:.0%}")
        self.log(f"    H2 test (VP > NP): t={h2_test['t']:.2f} p={h2_test['p']:.4f} "
                 f"d={h2_test['d']:.2f} {'*' if h2_test['significant'] else ''}")

        # ================================================================
        # H1 + H3: Modifier-biased training -> NP attachment
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H1/H3: Modifier-biased training")
        self.log("=" * 60)

        mod_vp_overlaps = []
        mod_np_overlaps = []
        mod_preferences = []

        for s in seeds:
            self.log(f"  Seed {s}/{cfg.n_seeds - 1}")
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=modifier_sents)

            result = measure_pp_attachment(parser, ambig_sentence)
            mod_vp_overlaps.append(result["vp_overlap"])
            mod_np_overlaps.append(result["np_overlap"])
            mod_preferences.append(result["preference"])
            self.log(f"    VP={result['vp_overlap']:.3f}  "
                     f"NP={result['np_overlap']:.3f}  "
                     f"pref={result['preference']}")

        mod_vp_stats = summarize(mod_vp_overlaps)
        mod_np_stats = summarize(mod_np_overlaps)
        # Test: NP overlap > VP overlap under modifier bias
        mod_deltas = [n - v for v, n in zip(mod_vp_overlaps, mod_np_overlaps)]
        h3_test = ttest_vs_null(mod_deltas, 0.0)
        mod_np_pct = mod_preferences.count("NP") / len(mod_preferences)

        self.log(f"  Modifier bias results:")
        self.log(f"    VP overlap: {mod_vp_stats['mean']:.3f} +/- {mod_vp_stats['sem']:.3f}")
        self.log(f"    NP overlap: {mod_np_stats['mean']:.3f} +/- {mod_np_stats['sem']:.3f}")
        self.log(f"    NP preference: {mod_np_pct:.0%}")
        self.log(f"    H3 test (NP > VP): t={h3_test['t']:.2f} p={h3_test['p']:.4f} "
                 f"d={h3_test['d']:.2f} {'*' if h3_test['significant'] else ''}")

        # ================================================================
        # H1: Cross-condition difference (instrument vs modifier)
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H1: Cross-condition preference difference")
        self.log("=" * 60)

        # Compare VP-NP delta across conditions
        cross_deltas = [i - m for i, m in zip(instr_deltas, mod_deltas)]
        h1_test = ttest_vs_null(cross_deltas, 0.0)

        self.log(f"  Instrument VP-NP delta: {np.mean(instr_deltas):.3f}")
        self.log(f"  Modifier NP-VP delta:   {np.mean(mod_deltas):.3f}")
        self.log(f"  Cross-condition test: t={h1_test['t']:.2f} p={h1_test['p']:.4f} "
                 f"d={h1_test['d']:.2f} {'*' if h1_test['significant'] else ''}")

        # ================================================================
        # H4: Deterministic settlement
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H4: Deterministic settlement (PP attractor convergence)")
        self.log("=" * 60)

        convergence_results = []
        for s in seeds:
            # Test under instrument bias
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=instrument_sents)
            det_result = measure_determinism(
                parser, ambig_sentence, n_rounds=cfg.convergence_rounds)
            convergence_results.append(det_result)
            self.log(f"  Seed {s}: converged={det_result['converged']}  "
                     f"stability={det_result['final_stability']:.3f}  "
                     f"steps={det_result['steps_to_converge']}")

        convergence_rate = sum(
            1 for r in convergence_results if r["converged"]
        ) / len(convergence_results)
        final_stabilities = [r["final_stability"] for r in convergence_results]
        stability_stats = summarize(final_stabilities)
        h4_test = ttest_vs_null(final_stabilities, 0.5)

        self.log(f"  Convergence rate: {convergence_rate:.0%}")
        self.log(f"  Final stability: {stability_stats['mean']:.3f} +/- {stability_stats['sem']:.3f}")
        self.log(f"  H4 test (stability > 0.5): t={h4_test['t']:.2f} p={h4_test['p']:.4f} "
                 f"d={h4_test['d']:.2f} {'*' if h4_test['significant'] else ''}")

        # ================================================================
        # Summary
        # ================================================================
        duration = self._stop_timer()

        self.log(f"\n{'=' * 60}")
        self.log("STRUCTURAL AMBIGUITY SUMMARY")
        self.log(f"  H1 (cross-condition diff):  "
                 f"{'SUPPORTED' if h1_test['significant'] else 'NOT SUPPORTED'} "
                 f"(d={h1_test['d']:.2f})")
        self.log(f"  H2 (instrument -> VP):      "
                 f"{'SUPPORTED' if h2_test['significant'] else 'NOT SUPPORTED'} "
                 f"(VP pref={instr_vp_pct:.0%})")
        self.log(f"  H3 (modifier -> NP):        "
                 f"{'SUPPORTED' if h3_test['significant'] else 'NOT SUPPORTED'} "
                 f"(NP pref={mod_np_pct:.0%})")
        self.log(f"  H4 (deterministic):         "
                 f"{'SUPPORTED' if h4_test['significant'] else 'NOT SUPPORTED'} "
                 f"(conv={convergence_rate:.0%})")
        self.log(f"  Duration: {duration:.1f}s ({cfg.n_seeds} seeds)")

        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "n_seeds": cfg.n_seeds,
                "p": cfg.p, "beta": cfg.beta, "rounds": cfg.rounds,
                "null_overlap": null_overlap,
                "ambiguous_sentence": " ".join(ambig_sentence),
                "n_instrument_sents": len(instrument_sents),
                "n_modifier_sents": len(modifier_sents),
            },
            metrics={
                "h1_cross_condition": {
                    "test": h1_test,
                    "instr_mean_delta": float(np.mean(instr_deltas)),
                    "mod_mean_delta": float(np.mean(mod_deltas)),
                },
                "h2_instrument_bias": {
                    "vp_overlap": instr_vp_stats,
                    "np_overlap": instr_np_stats,
                    "vp_preference_rate": instr_vp_pct,
                    "test": h2_test,
                },
                "h3_modifier_bias": {
                    "vp_overlap": mod_vp_stats,
                    "np_overlap": mod_np_stats,
                    "np_preference_rate": mod_np_pct,
                    "test": h3_test,
                },
                "h4_determinism": {
                    "convergence_rate": convergence_rate,
                    "stability": stability_stats,
                    "test": h4_test,
                },
            },
            duration_seconds=duration,
        )

        self.save_result(result)
        return result


def main():
    """Run structural ambiguity experiment."""
    parser = argparse.ArgumentParser(
        description="Structural ambiguity (PP-attachment) experiment")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation run (2 seeds)")
    args = parser.parse_args()

    exp = StructuralAmbiguityExperiment(verbose=True)
    result = exp.run(quick=args.quick)

    print(f"\nExperiment completed in {result.duration_seconds:.1f}s")
    print(f"H1 (cross-condition): {result.metrics['h1_cross_condition']['test']['significant']}")
    print(f"H2 (instrument->VP): {result.metrics['h2_instrument_bias']['test']['significant']}")
    print(f"H3 (modifier->NP):   {result.metrics['h3_modifier_bias']['test']['significant']}")
    print(f"H4 (deterministic):  {result.metrics['h4_determinism']['test']['significant']}")


if __name__ == "__main__":
    main()
