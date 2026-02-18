"""
P600 Syntactic Violations — Assembly Instability in Structural Areas

The P600 is a positive ERP component peaking ~600ms post-stimulus,
associated with syntactic violations and structural reanalysis. We model
it as assembly INSTABILITY during structural integration: when projecting
a word's core assembly into syntactic/role areas (SUBJ, OBJ, ROLE_AGENT,
ROLE_PATIENT, VP), trained pathways produce stable assemblies that
converge quickly, while untrained or weakly-trained pathways produce
oscillating assemblies.

Mechanism:
  After a word settles in its core area, the parser attempts structural
  integration by projecting core -> role/syntactic areas. For grammatical
  continuations (e.g., "cat" after "the dog chases the"), NOUN_CORE ->
  SUBJ/OBJ has Hebbian-strengthened weights from training -> assembly
  converges quickly (low instability = small P600). For category violations
  (e.g., "likes" [verb] in object position), VERB_CORE -> SUBJ/OBJ has
  only random baseline weights (never trained) -> assembly oscillates
  (high instability = large P600).

P600 metrics (both measured, instability is primary after consolidation):
  INSTABILITY: sum(1 - Jaccard(winners[r], winners[r-1])) across rounds
  CUMULATIVE ENERGY: sum(pre_kwta_total) across settling rounds

Consolidation pass:
  After parser.train(), the parser calls reset_area_connections() on role
  and VP areas, wiping all Hebbian-trained weights. We replay the same
  role/phrase training WITHOUT reset, creating persistent Hebbian-strengthened
  connections for trained pathways (NOUN_CORE→ROLE_AGENT, NOUN_CORE→VP, etc).
  Untrained pathways (VERB_CORE→ROLE_*) retain only random baseline weights.
  This asymmetry is what makes instability differentiate conditions.

Design:
  C1 -- Grammatical: "the dog chases the cat" (trained noun as object)
  C2 -- Semantic violation: "the dog chases the table" (untrained noun)
  C3 -- Category violation: "the dog chases the likes" (verb as noun)

N400 measurement:
  Global pre-k-WTA energy in the core area DURING the first recurrent
  projection step of the critical word. Self-recurrence from the subject
  noun's residual assembly captures context-dependent facilitation.

Predictions:
  N400 (core energy): sem > gram (semantic access difficulty)
  P600 (instability): cat > sem > gram (structural integration difficulty)
  Double dissociation: N400 selective for semantic, P600 graded for syntactic

Bootstrap connectivity:
  After training, some core->structural weight matrices remain empty (0x0
  in sparse engine). We bootstrap by projecting through each pathway once
  with plasticity OFF, materializing random binomial(p) baseline weights.
  This models anatomical fibers that exist before learning.

Literature:
  - Osterhout & Holcomb 1992: P600 discovery
  - Hagoort et al. 1993, Hagoort 2005: P600 / MUC model
  - Friederici 2002: P600 as Phase 3 reanalysis
  - Vosse & Kempen 2000: P600 = settling time
  - Brouwer & Crocker 2017: P600 = integration update cost
  - Kuperberg 2007: N400/P600 biphasic model
  - Lewis & Vasishth 2005: Cue-based retrieval interference
  - van Herten et al. 2005: Semantic P600

See also:
  - research/plans/P600_REANALYSIS.md: design rationale and theory
  - research/claims/N400_GLOBAL_ENERGY.md: N400 claim and evidence
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
from research.experiments.vocab import (
    build_svo_vocab, build_svo_sentences, make_p600_test_sentences,
)
from research.experiments.metrics import measure_p600_settling, compute_jaccard_instability
from research.experiments.metrics.measurement import measure_critical_word
from research.experiments.infrastructure import (
    bootstrap_structural_connectivity,
    consolidate_role_connections,
    consolidate_vp_connections,
)
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.areas import (
    ROLE_AGENT, ROLE_PATIENT, SUBJ, OBJ, VP,
)
from src.assembly_calculus.ops import project

# Backward-compat aliases for external importers
_measure_critical_word = measure_critical_word
_make_test_sentences = make_p600_test_sentences


@dataclass
class P600Config:
    n: int = 50000
    k: int = 100
    p: float = 0.05
    beta: float = 0.05
    rounds: int = 10
    n_seeds: int = 5
    p600_settling_rounds: int = 5


class P600SyntacticExperiment(ExperimentBase):
    """Test P600 via assembly instability in structural areas.

    Combines N400 measurement (global pre-k-WTA energy in core area) with
    P600 measurement (assembly instability during structural integration)
    to test for double dissociation between semantic and syntactic processing.
    """

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

        vocab = build_svo_vocab()
        training = build_svo_sentences(vocab)
        test_sentences = make_p600_test_sentences(vocab)
        seeds = list(range(cfg.n_seeds))

        p600_areas = [ROLE_AGENT, ROLE_PATIENT, SUBJ, OBJ, VP]

        # Per-seed accumulators
        n400_gram_seeds = []
        n400_semviol_seeds = []
        n400_catviol_seeds = []
        # Core-area instability (Jaccard during word settling in core area)
        core_inst_gram_seeds = []
        core_inst_semviol_seeds = []
        core_inst_catviol_seeds = []
        # P600 structural instability (with consolidation)
        p600_inst_gram_seeds = []
        p600_inst_semviol_seeds = []
        p600_inst_catviol_seeds = []
        # P600 cumulative energy
        p600_cum_gram_seeds = []
        p600_cum_semviol_seeds = []
        p600_cum_catviol_seeds = []
        # Per-area P600 instability for detailed analysis
        p600_per_area_inst_gram = {a: [] for a in p600_areas}
        p600_per_area_inst_sem = {a: [] for a in p600_areas}
        p600_per_area_inst_cat = {a: [] for a in p600_areas}

        for seed_idx, seed in enumerate(seeds):
            self.log(f"\n=== Seed {seed_idx + 1}/{len(seeds)} ===")

            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=seed, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            # Bootstrap FIRST: materialize random baseline weights for all
            # empty core->structural pairs. Must come before consolidation
            # because empty weights trigger the zero-signal early return,
            # preventing consolidation projections from running.
            bootstrap_structural_connectivity(
                parser, p600_areas, log_fn=self.log)

            # Consolidation SECOND: replay role/phrase training WITHOUT reset.
            # Now that random baseline weights exist, Hebbian learning can
            # strengthen the trained pathways above baseline.
            consolidate_role_connections(
                parser, training, log_fn=self.log)
            consolidate_vp_connections(
                parser, training, log_fn=self.log)

            n400_gram, n400_sem, n400_cat = [], [], []
            cinst_gram, cinst_sem, cinst_cat = [], [], []  # core instability
            inst_gram, inst_sem, inst_cat = [], [], []
            cum_gram, cum_sem, cum_cat = [], [], []
            area_inst_gram = {a: [] for a in p600_areas}
            area_inst_sem = {a: [] for a in p600_areas}
            area_inst_cat = {a: [] for a in p600_areas}

            for test in test_sentences:
                for cond_label, cond_key in [
                    ("gram", "grammatical"),
                    ("semviol", "semantic_violation"),
                    ("catviol", "category_violation"),
                ]:
                    critical_word = test[cond_key]
                    result = measure_critical_word(
                        parser, test["context_words"], critical_word,
                        p600_areas, cfg.rounds, cfg.p600_settling_rounds,
                    )

                    n400_val = result["n400_energy"]
                    core_inst_val = result["core_instability"]
                    mean_inst = result["p600_mean_instability"]
                    mean_cum = float(np.mean([
                        result["p600_cumulative_energy"].get(a, 0.0)
                        for a in p600_areas
                    ]))

                    if cond_label == "gram":
                        n400_gram.append(n400_val)
                        cinst_gram.append(core_inst_val)
                        inst_gram.append(mean_inst)
                        cum_gram.append(mean_cum)
                        for a in p600_areas:
                            area_inst_gram[a].append(
                                result["p600_instability"].get(a, 0.0))
                    elif cond_label == "semviol":
                        n400_sem.append(n400_val)
                        cinst_sem.append(core_inst_val)
                        inst_sem.append(mean_inst)
                        cum_sem.append(mean_cum)
                        for a in p600_areas:
                            area_inst_sem[a].append(
                                result["p600_instability"].get(a, 0.0))
                    elif cond_label == "catviol":
                        n400_cat.append(n400_val)
                        cinst_cat.append(core_inst_val)
                        inst_cat.append(mean_inst)
                        cum_cat.append(mean_cum)
                        for a in p600_areas:
                            area_inst_cat[a].append(
                                result["p600_instability"].get(a, 0.0))

            if n400_gram:
                n400_gram_seeds.append(float(np.mean(n400_gram)))
                n400_semviol_seeds.append(float(np.mean(n400_sem)))
                n400_catviol_seeds.append(float(np.mean(n400_cat)))
                core_inst_gram_seeds.append(float(np.mean(cinst_gram)))
                core_inst_semviol_seeds.append(float(np.mean(cinst_sem)))
                core_inst_catviol_seeds.append(float(np.mean(cinst_cat)))
                p600_inst_gram_seeds.append(float(np.mean(inst_gram)))
                p600_inst_semviol_seeds.append(float(np.mean(inst_sem)))
                p600_inst_catviol_seeds.append(float(np.mean(inst_cat)))
                p600_cum_gram_seeds.append(float(np.mean(cum_gram)))
                p600_cum_semviol_seeds.append(float(np.mean(cum_sem)))
                p600_cum_catviol_seeds.append(float(np.mean(cum_cat)))
                for a in p600_areas:
                    p600_per_area_inst_gram[a].append(
                        float(np.mean(area_inst_gram[a])))
                    p600_per_area_inst_sem[a].append(
                        float(np.mean(area_inst_sem[a])))
                    p600_per_area_inst_cat[a].append(
                        float(np.mean(area_inst_cat[a])))

                self.log(
                    f"  N400 (core): gram={np.mean(n400_gram):.1f}  "
                    f"sem={np.mean(n400_sem):.1f}  "
                    f"cat={np.mean(n400_cat):.1f}")
                self.log(
                    f"  Core instability: gram={np.mean(cinst_gram):.3f}  "
                    f"sem={np.mean(cinst_sem):.3f}  "
                    f"cat={np.mean(cinst_cat):.3f}")
                self.log(
                    f"  P600 (struct instability): gram={np.mean(inst_gram):.3f}  "
                    f"sem={np.mean(inst_sem):.3f}  "
                    f"cat={np.mean(inst_cat):.3f}")

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
                             "n_seeds": cfg.n_seeds,
                             "p600_settling_rounds": cfg.p600_settling_rounds},
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

        # --- Core-area instability (Jaccard in core during word settling) ---
        self.log("\nCORE-AREA INSTABILITY (Jaccard during word settling)")
        self.log("  Prediction: sem > gram (untrained settles slower in NOUN_CORE)")
        self.log("  Note: catviol is in VERB_CORE (different area, not comparable)")

        for label, a, b in [
            ("sem_vs_gram", core_inst_semviol_seeds, core_inst_gram_seeds),
            ("cat_vs_gram", core_inst_catviol_seeds, core_inst_gram_seeds),
        ]:
            stats = paired_ttest(a, b)
            a_s = summarize(a)
            b_s = summarize(b)
            direction = ("HIGHER_INSTAB" if a_s["mean"] > b_s["mean"]
                         else "LOWER_INSTAB")
            self.log(f"  {label}: viol={a_s['mean']:.4f}  "
                     f"gram={b_s['mean']:.4f}  "
                     f"d={stats['d']:.3f}  p={stats['p']:.4f}  {direction}")
            metrics[f"core_inst_{label}"] = {
                "violation": a_s, "grammatical": b_s,
                "test": stats, "direction": direction,
            }

        # --- P600 analysis: STRUCTURAL INSTABILITY (primary P600 metric) ---
        self.log("\nP600 STRUCTURAL INSTABILITY (primary — with consolidated connections)")
        self.log("  Prediction: catviol > semviol > gram")
        self.log("  Higher instability = larger P600 = harder integration")

        for label, a, b in [
            ("cat_vs_gram", p600_inst_catviol_seeds, p600_inst_gram_seeds),
            ("sem_vs_gram", p600_inst_semviol_seeds, p600_inst_gram_seeds),
            ("cat_vs_sem", p600_inst_catviol_seeds, p600_inst_semviol_seeds),
        ]:
            stats = paired_ttest(a, b)
            a_s = summarize(a)
            b_s = summarize(b)
            direction = "P600_EFFECT" if a_s["mean"] > b_s["mean"] else "NO_P600"
            self.log(f"  {label}: a={a_s['mean']:.4f}  "
                     f"b={b_s['mean']:.4f}  "
                     f"d={stats['d']:.3f}  p={stats['p']:.4f}  {direction}")
            metrics[f"p600_inst_{label}"] = {
                "a": a_s, "b": b_s,
                "test": stats, "direction": direction,
            }

        # --- P600 analysis: CUMULATIVE ENERGY (secondary — reversed after consolidation) ---
        self.log("\nP600 CUMULATIVE ENERGY (note: reversed after consolidation)")

        for label, a, b in [
            ("cat_vs_gram", p600_cum_catviol_seeds, p600_cum_gram_seeds),
            ("sem_vs_gram", p600_cum_semviol_seeds, p600_cum_gram_seeds),
            ("cat_vs_sem", p600_cum_catviol_seeds, p600_cum_semviol_seeds),
        ]:
            stats = paired_ttest(a, b)
            a_s = summarize(a)
            b_s = summarize(b)
            direction = "P600_EFFECT" if a_s["mean"] > b_s["mean"] else "NO_P600"
            self.log(f"  {label}: a={a_s['mean']:.1f}  "
                     f"b={b_s['mean']:.1f}  "
                     f"d={stats['d']:.3f}  p={stats['p']:.4f}  {direction}")
            metrics[f"p600_cum_{label}"] = {
                "a": a_s, "b": b_s,
                "test": stats, "direction": direction,
            }

        # --- Per-area P600 instability breakdown ---
        self.log(f"\nP600 instability per area (category violation vs grammatical)")
        for area in p600_areas:
            if len(p600_per_area_inst_gram[area]) >= 2:
                stats = paired_ttest(
                    p600_per_area_inst_cat[area],
                    p600_per_area_inst_gram[area])
                cat_s = summarize(p600_per_area_inst_cat[area])
                gram_s = summarize(p600_per_area_inst_gram[area])
                self.log(f"  {area:<15}: cat={cat_s['mean']:.4f}  "
                         f"gram={gram_s['mean']:.4f}  "
                         f"d={stats['d']:.3f}  p={stats['p']:.4f}")
                metrics[f"p600_per_area_{area}"] = {
                    "category_violation": cat_s, "grammatical": gram_s,
                    "test": stats,
                }

        self.log(f"\nP600 instability per area (semantic violation vs grammatical)")
        for area in p600_areas:
            if len(p600_per_area_inst_gram[area]) >= 2:
                stats = paired_ttest(
                    p600_per_area_inst_sem[area],
                    p600_per_area_inst_gram[area])
                sem_s = summarize(p600_per_area_inst_sem[area])
                gram_s = summarize(p600_per_area_inst_gram[area])
                self.log(f"  {area:<15}: sem={sem_s['mean']:.4f}  "
                         f"gram={gram_s['mean']:.4f}  "
                         f"d={stats['d']:.3f}  p={stats['p']:.4f}")
                metrics[f"p600_per_area_sem_{area}"] = {
                    "semantic_violation": sem_s, "grammatical": gram_s,
                    "test": stats,
                }

        # --- Dissociation summary ---
        self.log(f"\n{'-'*70}")
        self.log("DISSOCIATION SUMMARY")
        self.log(f"{'-'*70}")

        n400_sem = metrics.get("n400_sem_vs_gram", {})
        # P600 cumulative energy (total structural processing cost)
        p600_cat = metrics.get("p600_cum_cat_vs_gram", {})
        p600_sem = metrics.get("p600_cum_sem_vs_gram", {})
        p600_cat_vs_sem = metrics.get("p600_cum_cat_vs_sem", {})
        # P600 structural instability (with consolidation)
        inst_cat = metrics.get("p600_inst_cat_vs_gram", {})
        inst_sem_m = metrics.get("p600_inst_sem_vs_gram", {})
        inst_cat_sem = metrics.get("p600_inst_cat_vs_sem", {})
        # Core-area instability
        core_sem = metrics.get("core_inst_sem_vs_gram", {})

        sem_n400_d = n400_sem.get("test", {}).get("d", 0)
        sem_n400_p = n400_sem.get("test", {}).get("p", 1)
        cat_p600_d = p600_cat.get("test", {}).get("d", 0)
        cat_p600_p = p600_cat.get("test", {}).get("p", 1)
        sem_p600_d = p600_sem.get("test", {}).get("d", 0)
        sem_p600_p = p600_sem.get("test", {}).get("p", 1)
        cat_sem_d = p600_cat_vs_sem.get("test", {}).get("d", 0)
        cat_sem_p = p600_cat_vs_sem.get("test", {}).get("p", 1)

        self.log(f"  N400 energy (sem vs gram):      d={sem_n400_d:.3f} p={sem_n400_p:.4f}")

        core_sem_d = core_sem.get("test", {}).get("d", 0)
        core_sem_p = core_sem.get("test", {}).get("p", 1)
        self.log(f"  Core instability (sem vs gram):  d={core_sem_d:.3f} p={core_sem_p:.4f}")

        self.log(f"  P600 cum energy (cat vs gram):  d={cat_p600_d:.3f} p={cat_p600_p:.4f}")
        self.log(f"  P600 cum energy (sem vs gram):  d={sem_p600_d:.3f} p={sem_p600_p:.4f}")
        self.log(f"  P600 cum energy (cat vs sem):   d={cat_sem_d:.3f} p={cat_sem_p:.4f}")

        inst_cat_d = inst_cat.get("test", {}).get("d", 0)
        inst_cat_p = inst_cat.get("test", {}).get("p", 1)
        inst_sem_d = inst_sem_m.get("test", {}).get("d", 0)
        inst_sem_p = inst_sem_m.get("test", {}).get("p", 1)
        inst_cs_d = inst_cat_sem.get("test", {}).get("d", 0)
        inst_cs_p = inst_cat_sem.get("test", {}).get("p", 1)
        self.log(f"  Struct instab (cat vs gram):    d={inst_cat_d:.3f} p={inst_cat_p:.4f}")
        self.log(f"  Struct instab (sem vs gram):    d={inst_sem_d:.3f} p={inst_sem_p:.4f}")
        self.log(f"  Struct instab (cat vs sem):     d={inst_cs_d:.3f} p={inst_cs_p:.4f}")

        n400_works = (n400_sem.get("direction") == "N400_EFFECT" and
                      sem_n400_p < 0.05)
        p600_cum_works = (p600_cat.get("direction") == "P600_EFFECT" and
                          cat_p600_p < 0.05)
        p600_inst_works = (inst_cat.get("direction") == "P600_EFFECT" and
                           inst_cat_p < 0.05)
        p600_graded_cum = (p600_cat_vs_sem.get("direction") == "P600_EFFECT"
                           and cat_sem_p < 0.10)
        p600_graded_inst = (inst_cat_sem.get("direction") == "P600_EFFECT"
                            and inst_cs_p < 0.10)
        core_inst_works = (core_sem.get("direction") == "HIGHER_INSTAB" and
                           core_sem_p < 0.05)
        double_dissoc = n400_works and (p600_cum_works or p600_inst_works)

        self.log(f"\n  N400 for semantic anomaly:      {'YES' if n400_works else 'NO'}")
        self.log(f"  Core instability (sem > gram):   {'YES' if core_inst_works else 'NO'}")
        self.log(f"  P600 cum energy (cat > gram):    {'YES' if p600_cum_works else 'NO'}")
        self.log(f"  P600 instability (cat > gram):   {'YES' if p600_inst_works else 'NO'}")
        self.log(f"  P600 graded cum (cat > sem):     {'YES' if p600_graded_cum else 'NO'}")
        self.log(f"  P600 graded instab (cat > sem):  {'YES' if p600_graded_inst else 'NO'}")
        self.log(f"  Double dissociation:             {'YES' if double_dissoc else 'NO'}")

        metrics["dissociation"] = {
            "n400_semantic": n400_works,
            "core_instability_semantic": core_inst_works,
            "p600_cum_syntactic": p600_cum_works,
            "p600_inst_syntactic": p600_inst_works,
            "p600_graded_cum": p600_graded_cum,
            "p600_graded_inst": p600_graded_inst,
            "double_dissociation": double_dissoc,
        }

        duration = self._stop_timer()
        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p,
                "beta": cfg.beta, "rounds": cfg.rounds,
                "n_seeds": cfg.n_seeds,
                "p600_settling_rounds": cfg.p600_settling_rounds,
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
