"""
Agreement Attraction: Binding Interference from Intervening Nouns

Tests whether intervening PP nouns interfere with the subject's AGENT binding,
producing a graded distance effect analogous to agreement attraction in
human sentence processing.

Phenomenon:
  "The key to the cabinets is..." — humans sometimes accept "are" because
  "cabinets" (plural, recent) attracts agreement away from "key" (singular,
  distant). This is a binding interference effect.

In AC terms:
  After processing "dog in garden", the location noun "garden" partially
  activates in NOUN_CORE (the same area as "dog"). When the verb arrives
  and the system needs to retrieve the agent, the PP noun's residual
  activation may interfere with the agent's binding to ROLE_AGENT.

  The interference should increase with:
    - More intervening nouns (longer PP chains)
    - Category similarity (same core area)

Architecture: 11-area RECURSIVE_VOCAB.
Training: SVO and SVO+PP sentences via RecursiveCFG (VP-attached PPs).
Test: Manually construct subject-PP contexts and measure at verb position.

Hypotheses:
  H1 (Distance effect): AGENT P600 is higher after a short PP than without.
  H2 (Graded distance): AGENT P600 increases further with recursive PP
      (more intervening nouns = more interference).
  H3 (Prediction weakening): N400 for the verb increases with PP length
      (prediction from agent weakens with more intervening material).
  H4 (Leakage): The PP noun's assembly has measurable overlap with the
      agent's footprint in ROLE_AGENT (direct interference metric).

Usage:
    uv run python research/experiments/primitives/test_agreement_attraction.py
    uv run python research/experiments/primitives/test_agreement_attraction.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    measure_overlap,
    summarize,
    paired_ttest,
)
from research.experiments.lib.vocabulary import RECURSIVE_VOCAB
from research.experiments.lib.grammar import RecursiveCFG
from research.experiments.lib.brain_setup import (
    BrainConfig,
    create_language_brain,
    build_lexicon,
    activate_word,
)
from research.experiments.lib.training import train_sentence
from research.experiments.lib.measurement import (
    measure_n400,
    measure_p600,
    measure_role_leakage,
)


@dataclass
class AgreementAttractionConfig:
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
    # Grammar
    pp_prob: float = 0.5
    recursive_pp_prob: float = 0.4
    rel_prob: float = 0.3
    max_pp_depth: int = 2
    # Training
    n_train_sentences: int = 80
    training_reps: int = 3
    # Test
    n_test_items: int = 5


def run_trial(
    cfg: AgreementAttractionConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run one agreement attraction trial.

    Trains on SVO/SVO+PP sentences, then measures:
      - AGENT P600 with no PP (baseline)
      - AGENT P600 after short PP (one intervening noun)
      - AGENT P600 after long PP (two intervening nouns)
      - N400 for verb prediction at each distance
      - Leakage of PP noun into ROLE_AGENT
    """
    rng = np.random.default_rng(seed)
    vocab = RECURSIVE_VOCAB

    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain = create_language_brain(bcfg, vocab, seed)

    grammar = RecursiveCFG(
        pp_prob=cfg.pp_prob,
        recursive_pp_prob=cfg.recursive_pp_prob,
        rel_prob=cfg.rel_prob,
        max_pp_depth=cfg.max_pp_depth,
        vocab=vocab, rng=rng)

    n_train = cfg.n_train_sentences * cfg.training_reps
    train_sents = grammar.generate_batch(n_train)

    n_pp = sum(1 for s in train_sents if s.get("has_pp"))
    n_deep = sum(1 for s in train_sents if s.get("pp_depth", 0) >= 2)

    for sent in train_sents:
        train_sentence(brain, sent, vocab,
                       cfg.train_rounds_per_pair, cfg.binding_rounds)

    brain.disable_plasticity = True
    lexicon = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)

    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")
    preps = vocab.words_for_category("PREP")
    locs = vocab.words_for_category("LOCATION")

    # ── Measurements across three conditions ───────────────────────
    p600_no_pp, p600_short_pp, p600_long_pp = [], [], []
    n400_no_pp, n400_short_pp, n400_long_pp = [], [], []
    leakage_no_pp, leakage_short_pp, leakage_long_pp = [], [], []

    for i in range(cfg.n_test_items):
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        prep1 = preps[i % len(preps)]
        prep2 = preps[(i + 1) % len(preps)]
        pp_obj1 = locs[i % len(locs)]
        pp_obj2 = locs[(i + 1) % len(locs)]

        # --- Condition A: No PP ---
        # "agent VERB patient"
        activate_word(brain, agent, "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_no = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)
        n400_no_pp.append(measure_n400(pred_no, lexicon[verb]))

        p600_no_pp.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_AGENT", cfg.n_settling_rounds))

        # Leakage: how much does a location noun activate in ROLE_AGENT?
        leak_no = measure_role_leakage(
            brain, pp_obj1, "NOUN_CORE", "ROLE_AGENT")
        agent_footprint = measure_role_leakage(
            brain, agent, "NOUN_CORE", "ROLE_AGENT")
        leakage_no_pp.append(measure_overlap(leak_no, agent_footprint))

        # --- Condition B: Short PP (one intervening noun) ---
        # "agent prep1 pp_obj1 VERB patient"
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, prep1, "PREP_CORE", 3)
        activate_word(brain, pp_obj1, "NOUN_CORE", 3)
        # Re-activate agent for verb prediction
        activate_word(brain, agent, "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_short = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)
        n400_short_pp.append(measure_n400(pred_short, lexicon[verb]))

        p600_short_pp.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_AGENT", cfg.n_settling_rounds))

        # Leakage after PP processing
        leak_short = measure_role_leakage(
            brain, pp_obj1, "NOUN_CORE", "ROLE_AGENT")
        agent_fp_short = measure_role_leakage(
            brain, agent, "NOUN_CORE", "ROLE_AGENT")
        leakage_short_pp.append(measure_overlap(leak_short, agent_fp_short))

        # --- Condition C: Long PP (two intervening nouns) ---
        # "agent prep1 pp_obj1 prep2 pp_obj2 VERB patient"
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, prep1, "PREP_CORE", 3)
        activate_word(brain, pp_obj1, "NOUN_CORE", 3)
        activate_word(brain, prep2, "PREP_CORE", 3)
        activate_word(brain, pp_obj2, "NOUN_CORE", 3)
        # Re-activate agent for verb prediction
        activate_word(brain, agent, "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_long = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)
        n400_long_pp.append(measure_n400(pred_long, lexicon[verb]))

        p600_long_pp.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_AGENT", cfg.n_settling_rounds))

        # Leakage after long PP
        leak_long = measure_role_leakage(
            brain, pp_obj2, "NOUN_CORE", "ROLE_AGENT")
        agent_fp_long = measure_role_leakage(
            brain, agent, "NOUN_CORE", "ROLE_AGENT")
        leakage_long_pp.append(measure_overlap(leak_long, agent_fp_long))

    brain.disable_plasticity = False

    return {
        "n_train": n_train, "n_pp": n_pp, "n_deep": n_deep,
        # P600 (agent binding)
        "p600_no_pp": float(np.mean(p600_no_pp)),
        "p600_short_pp": float(np.mean(p600_short_pp)),
        "p600_long_pp": float(np.mean(p600_long_pp)),
        # N400 (verb prediction)
        "n400_no_pp": float(np.mean(n400_no_pp)),
        "n400_short_pp": float(np.mean(n400_short_pp)),
        "n400_long_pp": float(np.mean(n400_long_pp)),
        # Leakage
        "leakage_no_pp": float(np.mean(leakage_no_pp)),
        "leakage_short_pp": float(np.mean(leakage_short_pp)),
        "leakage_long_pp": float(np.mean(leakage_long_pp)),
    }


class AgreementAttractionExperiment(ExperimentBase):
    """Agreement attraction / binding interference experiment."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="agreement_attraction",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[AgreementAttractionConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or AgreementAttractionConfig(
            **{k: v for k, v in kwargs.items()
               if k in AgreementAttractionConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Agreement Attraction: Binding Interference from PP Nouns")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  pp_prob={cfg.pp_prob}, recursive_pp={cfg.recursive_pp_prob}")
        self.log(f"  n_train={cfg.n_train_sentences}, reps={cfg.training_reps}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        keys = [
            "p600_no_pp", "p600_short_pp", "p600_long_pp",
            "n400_no_pp", "n400_short_pp", "n400_long_pp",
            "leakage_no_pp", "leakage_short_pp", "leakage_long_pp",
        ]
        vals = {k: [] for k in keys}

        for s in range(n_seeds):
            self.log(f"  Seed {s+1}/{n_seeds} ...")
            result = run_trial(cfg, self.seed + s)
            for k in keys:
                vals[k].append(result[k])

            if s == 0:
                self.log(f"    Training: {result['n_train']} sentences"
                         f" ({result['n_pp']} PP, {result['n_deep']} deep PP)")

        # ── Report: P600 distance effect ──
        t_short = paired_ttest(vals["p600_short_pp"], vals["p600_no_pp"])
        t_long_short = paired_ttest(vals["p600_long_pp"], vals["p600_short_pp"])
        t_long_no = paired_ttest(vals["p600_long_pp"], vals["p600_no_pp"])

        self.log(f"\n  === AGENT Binding P600 (distance effect) ===")
        self.log(f"    No PP:    {np.mean(vals['p600_no_pp']):.3f}")
        self.log(f"    Short PP: {np.mean(vals['p600_short_pp']):.3f}"
                 f"  (vs no PP: d={t_short['d']:.2f})")
        self.log(f"    Long PP:  {np.mean(vals['p600_long_pp']):.3f}"
                 f"  (vs short: d={t_long_short['d']:.2f},"
                 f" vs no PP: d={t_long_no['d']:.2f})")

        # ── Report: N400 prediction weakening ──
        t_n400_short = paired_ttest(vals["n400_short_pp"], vals["n400_no_pp"])
        t_n400_long = paired_ttest(vals["n400_long_pp"], vals["n400_no_pp"])

        self.log(f"\n  === Verb N400 (prediction weakening) ===")
        self.log(f"    No PP:    {np.mean(vals['n400_no_pp']):.3f}")
        self.log(f"    Short PP: {np.mean(vals['n400_short_pp']):.3f}"
                 f"  (vs no PP: d={t_n400_short['d']:.2f})")
        self.log(f"    Long PP:  {np.mean(vals['n400_long_pp']):.3f}"
                 f"  (vs no PP: d={t_n400_long['d']:.2f})")

        # ── Report: Leakage ──
        self.log(f"\n  === Role Leakage (PP noun in ROLE_AGENT) ===")
        self.log(f"    No PP:    {np.mean(vals['leakage_no_pp']):.3f}")
        self.log(f"    Short PP: {np.mean(vals['leakage_short_pp']):.3f}")
        self.log(f"    Long PP:  {np.mean(vals['leakage_long_pp']):.3f}")

        # Hypotheses
        h1 = np.mean(vals["p600_short_pp"]) > np.mean(vals["p600_no_pp"])
        h2 = np.mean(vals["p600_long_pp"]) > np.mean(vals["p600_short_pp"])
        h3 = np.mean(vals["n400_long_pp"]) > np.mean(vals["n400_no_pp"])
        h4 = np.mean(vals["leakage_short_pp"]) > np.mean(vals["leakage_no_pp"])

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    H1 (Short PP > No PP, P600): "
                 f"{'PASS' if h1 else 'FAIL'}"
                 f" (d={t_short['d']:.2f})")
        self.log(f"    H2 (Long PP > Short PP, P600): "
                 f"{'PASS' if h2 else 'FAIL'}"
                 f" (d={t_long_short['d']:.2f})")
        self.log(f"    H3 (Long PP > No PP, N400): "
                 f"{'PASS' if h3 else 'FAIL'}"
                 f" (d={t_n400_long['d']:.2f})")
        self.log(f"    H4 (PP leakage > baseline): "
                 f"{'PASS' if h4 else 'FAIL'}")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "p600_distance": {
                "no_pp": summarize(vals["p600_no_pp"]),
                "short_pp": summarize(vals["p600_short_pp"]),
                "long_pp": summarize(vals["p600_long_pp"]),
                "short_vs_none": t_short,
                "long_vs_short": t_long_short,
                "long_vs_none": t_long_no,
            },
            "n400_prediction": {
                "no_pp": summarize(vals["n400_no_pp"]),
                "short_pp": summarize(vals["n400_short_pp"]),
                "long_pp": summarize(vals["n400_long_pp"]),
                "short_vs_none": t_n400_short,
                "long_vs_none": t_n400_long,
            },
            "leakage": {
                "no_pp": summarize(vals["leakage_no_pp"]),
                "short_pp": summarize(vals["leakage_short_pp"]),
                "long_pp": summarize(vals["leakage_long_pp"]),
            },
            "hypotheses": {
                "H1_short_pp_higher_p600": h1,
                "H2_long_pp_higher_p600": h2,
                "H3_long_pp_higher_n400": h3,
                "H4_pp_leakage": h4,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "pp_prob": cfg.pp_prob,
                "recursive_pp_prob": cfg.recursive_pp_prob,
                "n_train_sentences": cfg.n_train_sentences,
                "training_reps": cfg.training_reps,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Agreement Attraction Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = AgreementAttractionExperiment(verbose=True)

    if args.quick:
        cfg = AgreementAttractionConfig(
            n=5000, k=50,
            n_train_sentences=50, training_reps=2)
        n_seeds = args.seeds or 5
    else:
        cfg = AgreementAttractionConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    m = result.metrics
    print("\n" + "=" * 70)
    print("AGREEMENT ATTRACTION SUMMARY")
    print("=" * 70)

    print("\nAGENT P600 (distance effect):")
    pd = m["p600_distance"]
    print(f"  No PP:    {pd['no_pp']['mean']:.3f}")
    print(f"  Short PP: {pd['short_pp']['mean']:.3f}"
          f"  (d={pd['short_vs_none']['d']:.2f})")
    print(f"  Long PP:  {pd['long_pp']['mean']:.3f}"
          f"  (d={pd['long_vs_none']['d']:.2f})")

    print("\nVerb N400 (prediction weakening):")
    np_ = m["n400_prediction"]
    print(f"  No PP:    {np_['no_pp']['mean']:.3f}")
    print(f"  Short PP: {np_['short_pp']['mean']:.3f}"
          f"  (d={np_['short_vs_none']['d']:.2f})")
    print(f"  Long PP:  {np_['long_pp']['mean']:.3f}"
          f"  (d={np_['long_vs_none']['d']:.2f})")

    print("\nRole leakage (PP noun in ROLE_AGENT):")
    lk = m["leakage"]
    print(f"  No PP:    {lk['no_pp']['mean']:.3f}")
    print(f"  Short PP: {lk['short_pp']['mean']:.3f}")
    print(f"  Long PP:  {lk['long_pp']['mean']:.3f}")

    h = m["hypotheses"]
    print(f"\nH1 Short>None P600:  {'PASS' if h['H1_short_pp_higher_p600'] else 'FAIL'}")
    print(f"H2 Long>Short P600:  {'PASS' if h['H2_long_pp_higher_p600'] else 'FAIL'}")
    print(f"H3 Long>None N400:   {'PASS' if h['H3_long_pp_higher_n400'] else 'FAIL'}")
    print(f"H4 PP leakage:       {'PASS' if h['H4_pp_leakage'] else 'FAIL'}")

    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
