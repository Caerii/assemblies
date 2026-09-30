"""
Relative Clause Asymmetry: Subject-Relative vs Object-Relative

Tests whether the bottom-up prediction+binding mechanism reproduces the
universal SRC > ORC processing asymmetry.

Subject-relative clause (SRC):
  "dog that chases cat sees bird"
  Head noun "dog" is AGENT of both embedded and main verb.
  Dual binding: AGENT + REL_AGENT (same direction — both agent roles).

Object-relative clause (ORC):
  "dog that cat chases sees bird"
  Head noun "dog" is AGENT of main verb but PATIENT of embedded verb.
  Dual binding: AGENT + REL_PATIENT (conflicting directions).

The SRC/ORC asymmetry is one of the most robust findings in psycholinguistics.
ORCs are harder across all tested languages, independent of word order.

In AC terms, the asymmetry should emerge from:
  1. Prediction interference: In ORC, an intervening noun ("cat") disrupts
     the prediction chain from head noun to main verb.
  2. Binding conflict: In ORC, the head noun binds to ROLE_AGENT and
     ROLE_REL_PATIENT — conflicting structural roles that may interfere.
  3. Retrieval difficulty: At the main verb, the head noun must be
     retrieved from more intervening material in ORC.

Architecture: 11-area RECURSIVE_VOCAB (same as center-embedding).
Grammar: RecursiveCFG with orc_prob=0.5 to train on both SRC and ORC.

Hypotheses:
  H1 (N400 asymmetry): N400 at the main verb position is higher for ORC
      than SRC — weaker prediction after conflicting role processing.
  H2 (P600 AGENT binding): P600 for the head noun's ROLE_AGENT binding
      is higher for ORC — conflicting REL_PATIENT binding interferes.
  H3 (Dual-binding direction): P600 for the conflicting dual-binding
      (ORC: AGENT + REL_PATIENT) is higher than the same-direction
      dual-binding (SRC: AGENT + REL_AGENT).
  H4 (Embedded verb N400): Embedded verb N400 may differ between SRC
      and ORC due to different prediction contexts.

Usage:
    uv run python research/experiments/primitives/test_rc_asymmetry.py
    uv run python research/experiments/primitives/test_rc_asymmetry.py --quick
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
from research.experiments.lib.measurement import measure_n400, measure_p600


@dataclass
class RCAsymmetryConfig:
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
    pp_prob: float = 0.3
    rel_prob: float = 0.6      # high to get lots of RC training
    orc_prob: float = 0.5      # 50/50 SRC vs ORC
    max_pp_depth: int = 1
    # Training
    n_train_sentences: int = 100
    training_reps: int = 3
    # Test
    n_test_items: int = 5


def run_trial(
    cfg: RCAsymmetryConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run one SRC vs ORC trial.

    Trains on mixed SRC/ORC sentences, then measures:
      - N400 at main verb for SRC vs ORC
      - P600 for head noun's AGENT binding in SRC vs ORC context
      - P600 for dual-binding (same-direction vs conflicting)
      - N400 at embedded verb for SRC vs ORC
    """
    rng = np.random.default_rng(seed)
    vocab = RECURSIVE_VOCAB

    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain = create_language_brain(bcfg, vocab, seed)

    grammar = RecursiveCFG(
        pp_prob=cfg.pp_prob,
        rel_prob=cfg.rel_prob,
        orc_prob=cfg.orc_prob,
        max_pp_depth=cfg.max_pp_depth,
        vocab=vocab, rng=rng)

    n_train = cfg.n_train_sentences * cfg.training_reps
    train_sents = grammar.generate_batch(n_train)

    n_src = sum(1 for s in train_sents if s.get("rel_type") == "SRC")
    n_orc = sum(1 for s in train_sents if s.get("rel_type") == "ORC")
    n_plain = sum(1 for s in train_sents if not s.get("has_rel"))

    for sent in train_sents:
        train_sentence(brain, sent, vocab,
                       cfg.train_rounds_per_pair, cfg.binding_rounds)

    brain.disable_plasticity = True
    lexicon = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)

    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")

    # ── Test 1: Main verb prediction — SRC vs ORC ─────────────────
    src_n400_main, orc_n400_main = [], []

    for i in range(cfg.n_test_items):
        agent = nouns[i % len(nouns)]
        rel_verb = verbs[i % len(verbs)]
        rel_patient = nouns[(i + 2) % len(nouns)]
        orc_agent = nouns[(i + 1) % len(nouns)]
        main_verb = verbs[(i + 1) % len(verbs)]

        # SRC: "agent that rel_verb rel_patient MAIN_VERB"
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, "that", "COMP_CORE", 3)
        activate_word(brain, rel_verb, "VERB_CORE", 3)
        activate_word(brain, rel_patient, "NOUN_CORE", 3)
        activate_word(brain, agent, "NOUN_CORE", 3)  # re-activate for main
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_src = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)
        src_n400_main.append(measure_n400(pred_src, lexicon[main_verb]))

        # ORC: "agent that orc_agent rel_verb MAIN_VERB"
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, "that", "COMP_CORE", 3)
        activate_word(brain, orc_agent, "NOUN_CORE", 3)
        activate_word(brain, rel_verb, "VERB_CORE", 3)
        activate_word(brain, agent, "NOUN_CORE", 3)  # re-activate for main
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_orc = np.array(brain.areas["PREDICTION"].winners, dtype=np.uint32)
        orc_n400_main.append(measure_n400(pred_orc, lexicon[main_verb]))

    # ── Test 2: Embedded verb prediction — SRC vs ORC ─────────────
    src_n400_emb, orc_n400_emb = [], []

    for i in range(cfg.n_test_items):
        agent = nouns[i % len(nouns)]
        orc_agent = nouns[(i + 1) % len(nouns)]
        rel_verb = verbs[i % len(verbs)]

        # SRC: "agent that REL_VERB ..." — predict from "that"
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, "that", "COMP_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"COMP_CORE": ["PREDICTION"]})
        pred_src_emb = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)
        src_n400_emb.append(measure_n400(pred_src_emb, lexicon[rel_verb]))

        # ORC: "agent that orc_agent REL_VERB ..." — predict from orc_agent
        activate_word(brain, agent, "NOUN_CORE", 3)
        activate_word(brain, "that", "COMP_CORE", 3)
        activate_word(brain, orc_agent, "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        pred_orc_emb = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)
        orc_n400_emb.append(measure_n400(pred_orc_emb, lexicon[rel_verb]))

    # ── Test 3: P600 for AGENT binding — SRC vs ORC context ───────
    src_p600_agent, orc_p600_agent = [], []
    src_p600_rel_role, orc_p600_rel_role = [], []

    for i in range(cfg.n_test_items):
        agent = nouns[i % len(nouns)]

        # P600 for agent in ROLE_AGENT (trained in both SRC and ORC)
        src_p600_agent.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_AGENT", cfg.n_settling_rounds))
        orc_p600_agent.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_AGENT", cfg.n_settling_rounds))

        # P600 for same-direction dual binding (SRC: ROLE_REL_AGENT)
        src_p600_rel_role.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_REL_AGENT", cfg.n_settling_rounds))

        # P600 for conflicting dual binding (ORC: ROLE_REL_PATIENT)
        orc_p600_rel_role.append(measure_p600(
            brain, agent, "NOUN_CORE", "ROLE_REL_PATIENT", cfg.n_settling_rounds))

    # ── Test 4: P600 for main patient — SRC vs ORC ────────────────
    src_p600_patient, orc_p600_patient = [], []

    for i in range(cfg.n_test_items):
        main_patient = nouns[(i + 3) % len(nouns)]
        src_p600_patient.append(measure_p600(
            brain, main_patient, "NOUN_CORE", "ROLE_PATIENT",
            cfg.n_settling_rounds))
        orc_p600_patient.append(measure_p600(
            brain, main_patient, "NOUN_CORE", "ROLE_PATIENT",
            cfg.n_settling_rounds))

    brain.disable_plasticity = False

    return {
        "n_train": n_train, "n_src": n_src, "n_orc": n_orc, "n_plain": n_plain,
        # Main verb N400
        "src_n400_main": float(np.mean(src_n400_main)),
        "orc_n400_main": float(np.mean(orc_n400_main)),
        # Embedded verb N400
        "src_n400_emb": float(np.mean(src_n400_emb)),
        "orc_n400_emb": float(np.mean(orc_n400_emb)),
        # AGENT binding P600
        "src_p600_agent": float(np.mean(src_p600_agent)),
        "orc_p600_agent": float(np.mean(orc_p600_agent)),
        # Relative role dual-binding P600
        "src_p600_rel_role": float(np.mean(src_p600_rel_role)),
        "orc_p600_rel_role": float(np.mean(orc_p600_rel_role)),
        # Main patient P600
        "src_p600_patient": float(np.mean(src_p600_patient)),
        "orc_p600_patient": float(np.mean(orc_p600_patient)),
    }


class RCAsymmetryExperiment(ExperimentBase):
    """Subject-relative vs object-relative clause asymmetry experiment."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="rc_asymmetry",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[RCAsymmetryConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or RCAsymmetryConfig(
            **{k: v for k, v in kwargs.items()
               if k in RCAsymmetryConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("RC Asymmetry: Subject-Relative vs Object-Relative")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  rel_prob={cfg.rel_prob}, orc_prob={cfg.orc_prob}")
        self.log(f"  n_train={cfg.n_train_sentences}, reps={cfg.training_reps}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        keys = [
            "src_n400_main", "orc_n400_main",
            "src_n400_emb", "orc_n400_emb",
            "src_p600_agent", "orc_p600_agent",
            "src_p600_rel_role", "orc_p600_rel_role",
            "src_p600_patient", "orc_p600_patient",
        ]
        vals = {k: [] for k in keys}

        for s in range(n_seeds):
            self.log(f"  Seed {s+1}/{n_seeds} ...")
            result = run_trial(cfg, self.seed + s)
            for k in keys:
                vals[k].append(result[k])

            if s == 0:
                self.log(f"    Training: {result['n_train']} sentences"
                         f" ({result['n_src']} SRC, {result['n_orc']} ORC,"
                         f" {result['n_plain']} plain)")

        # ── Report: Main verb N400 ──
        t_main_n400 = paired_ttest(vals["orc_n400_main"], vals["src_n400_main"])
        self.log(f"\n  === Main Verb N400 ===")
        self.log(f"    SRC: {np.mean(vals['src_n400_main']):.3f}"
                 f"  ORC: {np.mean(vals['orc_n400_main']):.3f}"
                 f"  d={t_main_n400['d']:.2f}")

        # ── Report: Embedded verb N400 ──
        t_emb_n400 = paired_ttest(vals["orc_n400_emb"], vals["src_n400_emb"])
        self.log(f"\n  === Embedded Verb N400 ===")
        self.log(f"    SRC: {np.mean(vals['src_n400_emb']):.3f}"
                 f"  ORC: {np.mean(vals['orc_n400_emb']):.3f}"
                 f"  d={t_emb_n400['d']:.2f}")

        # ── Report: AGENT binding P600 ──
        t_agent_p600 = paired_ttest(
            vals["orc_p600_agent"], vals["src_p600_agent"])
        self.log(f"\n  === AGENT Binding P600 ===")
        self.log(f"    SRC: {np.mean(vals['src_p600_agent']):.3f}"
                 f"  ORC: {np.mean(vals['orc_p600_agent']):.3f}"
                 f"  d={t_agent_p600['d']:.2f}")

        # ── Report: Dual-binding P600 (REL_AGENT vs REL_PATIENT) ──
        t_dual = paired_ttest(
            vals["orc_p600_rel_role"], vals["src_p600_rel_role"])
        self.log(f"\n  === Dual-Binding P600 ===")
        self.log(f"    SRC (ROLE_REL_AGENT):   {np.mean(vals['src_p600_rel_role']):.3f}")
        self.log(f"    ORC (ROLE_REL_PATIENT):  {np.mean(vals['orc_p600_rel_role']):.3f}")
        self.log(f"    ORC > SRC: d={t_dual['d']:.2f}")

        # ── Report: Main patient P600 ──
        t_patient = paired_ttest(
            vals["orc_p600_patient"], vals["src_p600_patient"])
        self.log(f"\n  === Main Patient P600 ===")
        self.log(f"    SRC: {np.mean(vals['src_p600_patient']):.3f}"
                 f"  ORC: {np.mean(vals['orc_p600_patient']):.3f}"
                 f"  d={t_patient['d']:.2f}")

        # Hypotheses
        orc_higher_n400 = np.mean(vals["orc_n400_main"]) > np.mean(vals["src_n400_main"])
        orc_higher_agent_p600 = (np.mean(vals["orc_p600_agent"])
                                 > np.mean(vals["src_p600_agent"]))
        orc_higher_dual = (np.mean(vals["orc_p600_rel_role"])
                           > np.mean(vals["src_p600_rel_role"]))

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    H1 (ORC > SRC main verb N400): "
                 f"{'PASS' if orc_higher_n400 else 'FAIL'}"
                 f" (d={t_main_n400['d']:.2f})")
        self.log(f"    H2 (ORC > SRC AGENT P600): "
                 f"{'PASS' if orc_higher_agent_p600 else 'FAIL'}"
                 f" (d={t_agent_p600['d']:.2f})")
        self.log(f"    H3 (Conflicting > same dual-binding): "
                 f"{'PASS' if orc_higher_dual else 'FAIL'}"
                 f" (d={t_dual['d']:.2f})")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "main_verb_n400": {
                "src": summarize(vals["src_n400_main"]),
                "orc": summarize(vals["orc_n400_main"]),
                "test": t_main_n400,
            },
            "embedded_verb_n400": {
                "src": summarize(vals["src_n400_emb"]),
                "orc": summarize(vals["orc_n400_emb"]),
                "test": t_emb_n400,
            },
            "agent_binding_p600": {
                "src": summarize(vals["src_p600_agent"]),
                "orc": summarize(vals["orc_p600_agent"]),
                "test": t_agent_p600,
            },
            "dual_binding_p600": {
                "src_rel_agent": summarize(vals["src_p600_rel_role"]),
                "orc_rel_patient": summarize(vals["orc_p600_rel_role"]),
                "test": t_dual,
            },
            "main_patient_p600": {
                "src": summarize(vals["src_p600_patient"]),
                "orc": summarize(vals["orc_p600_patient"]),
                "test": t_patient,
            },
            "hypotheses": {
                "H1_orc_higher_n400": orc_higher_n400,
                "H2_orc_higher_agent_p600": orc_higher_agent_p600,
                "H3_conflicting_dual_binding": orc_higher_dual,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "rel_prob": cfg.rel_prob, "orc_prob": cfg.orc_prob,
                "pp_prob": cfg.pp_prob,
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
        description="RC Asymmetry Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = RCAsymmetryExperiment(verbose=True)

    if args.quick:
        cfg = RCAsymmetryConfig(
            n=5000, k=50,
            n_train_sentences=60, training_reps=2)
        n_seeds = args.seeds or 5
    else:
        cfg = RCAsymmetryConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    m = result.metrics
    print("\n" + "=" * 70)
    print("RC ASYMMETRY SUMMARY")
    print("=" * 70)

    print("\nMain verb N400:")
    mv = m["main_verb_n400"]
    print(f"  SRC: {mv['src']['mean']:.3f}  ORC: {mv['orc']['mean']:.3f}"
          f"  d={mv['test']['d']:.2f}")

    print("\nEmbedded verb N400:")
    ev = m["embedded_verb_n400"]
    print(f"  SRC: {ev['src']['mean']:.3f}  ORC: {ev['orc']['mean']:.3f}"
          f"  d={ev['test']['d']:.2f}")

    print("\nAGENT binding P600:")
    ab = m["agent_binding_p600"]
    print(f"  SRC: {ab['src']['mean']:.3f}  ORC: {ab['orc']['mean']:.3f}"
          f"  d={ab['test']['d']:.2f}")

    print("\nDual-binding P600:")
    db = m["dual_binding_p600"]
    print(f"  SRC (REL_AGENT):  {db['src_rel_agent']['mean']:.3f}")
    print(f"  ORC (REL_PATIENT): {db['orc_rel_patient']['mean']:.3f}"
          f"  d={db['test']['d']:.2f}")

    h = m["hypotheses"]
    print(f"\nH1 ORC>SRC N400:        {'PASS' if h['H1_orc_higher_n400'] else 'FAIL'}")
    print(f"H2 ORC>SRC AGENT P600:  {'PASS' if h['H2_orc_higher_agent_p600'] else 'FAIL'}")
    print(f"H3 Conflicting binding: {'PASS' if h['H3_conflicting_dual_binding'] else 'FAIL'}")

    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
