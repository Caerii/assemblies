"""
Diagnostic Investigation: N400/P600 Dynamics

Instruments the internals of the composed prediction+binding pipeline to
understand the dynamics that produce N400 and P600 signals. Five probes:

  1. geometry     -- PREDICTION area representational geometry
  2. signal_flow  -- activation strength through trained vs untrained pathways
  3. p600_metrics -- compare four candidate P600 metrics head-to-head
  4. exposure     -- novel noun exposure gradient for graded N400
  5. settling     -- per-round dynamics trace during binding settling

Usage:
    uv run python research/experiments/primitives/diagnose_erp_dynamics.py --quick
    uv run python research/experiments/primitives/diagnose_erp_dynamics.py --quick --probe geometry
    uv run python research/experiments/primitives/diagnose_erp_dynamics.py --quick --probe signal_flow
    uv run python research/experiments/primitives/diagnose_erp_dynamics.py --quick --probe p600_metrics
    uv run python research/experiments/primitives/diagnose_erp_dynamics.py --quick --probe exposure
    uv run python research/experiments/primitives/diagnose_erp_dynamics.py --quick --probe settling
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
from research.experiments.metrics.instability import compute_jaccard_instability
from src.core.brain import Brain


NOUNS = ["dog", "cat", "bird", "boy", "girl"]
VERBS = ["chases", "sees", "eats", "finds", "hits"]
NOVEL_NOUNS = ["table", "chair"]


@dataclass
class DiagnosticConfig:
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


def _activate_word(brain: Brain, stim_name: str, area: str, rounds: int):
    """Activate a word's assembly in its core area via stimulus projection."""
    brain.inhibit_areas([area])
    for _ in range(rounds):
        brain.project({stim_name: [area]}, {area: [area]})


def _generate_svo_sentences(n_sentences, rng):
    sentences = []
    for _ in range(n_sentences):
        agent = rng.choice(NOUNS)
        patient = rng.choice([n for n in NOUNS if n != agent])
        verb = rng.choice(VERBS)
        sentences.append((agent, verb, patient))
    return sentences


# ---------------------------------------------------------------------------
# Shared setup: build a trained brain
# ---------------------------------------------------------------------------

def build_trained_brain(
    cfg: DiagnosticConfig,
    seed: int,
    novel_training_sentences: int = 0,
) -> Tuple[Brain, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Build and train a brain with prediction bridges + role bindings.

    Returns (brain, lexicon, core_assemblies).
    brain has plasticity re-enabled after setup.
    """
    brain = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    rng = np.random.default_rng(seed)

    brain.add_area("NOUN_CORE", cfg.n, cfg.k, cfg.beta)
    brain.add_area("VERB_CORE", cfg.n, cfg.k, cfg.beta)
    brain.add_area("PREDICTION", cfg.n, cfg.k, cfg.beta)
    brain.add_area("ROLE_AGENT", cfg.n, cfg.k, cfg.beta)
    brain.add_area("ROLE_PATIENT", cfg.n, cfg.k, cfg.beta)

    for noun in NOUNS + NOVEL_NOUNS:
        brain.add_stimulus(f"PHON_{noun}", cfg.k)
    for verb in VERBS:
        brain.add_stimulus(f"PHON_{verb}", cfg.k)

    # Build word assemblies
    for noun in NOUNS:
        brain._engine.reset_area_connections("NOUN_CORE")
        _activate_word(brain, f"PHON_{noun}", "NOUN_CORE", cfg.lexicon_rounds)
    for verb in VERBS:
        brain._engine.reset_area_connections("VERB_CORE")
        _activate_word(brain, f"PHON_{verb}", "VERB_CORE", cfg.lexicon_rounds)
    for noun in NOVEL_NOUNS:
        brain._engine.reset_area_connections("NOUN_CORE")
        _activate_word(brain, f"PHON_{noun}", "NOUN_CORE", cfg.lexicon_rounds)

    # Training: prediction bridges + role bindings
    n_train = cfg.n_train_sentences * cfg.training_reps
    train_sents = _generate_svo_sentences(n_train, rng)

    # Add novel noun sentences if requested
    novel_sents = []
    for _ in range(novel_training_sentences):
        agent = rng.choice(NOUNS)
        verb = rng.choice(VERBS)
        patient = rng.choice(NOVEL_NOUNS)
        novel_sents.append((agent, verb, patient))

    for agent, verb_word, patient in train_sents + novel_sents:
        # Prediction training
        _activate_word(brain, f"PHON_{agent}", "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        for _ in range(cfg.train_rounds_per_pair):
            brain.project(
                {f"PHON_{verb_word}": ["PREDICTION"]},
                {"NOUN_CORE": ["PREDICTION"]},
            )

        _activate_word(brain, f"PHON_{verb_word}", "VERB_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        for _ in range(cfg.train_rounds_per_pair):
            brain.project(
                {f"PHON_{patient}": ["PREDICTION"]},
                {"VERB_CORE": ["PREDICTION"]},
            )

        # Binding training
        _activate_word(brain, f"PHON_{agent}", "NOUN_CORE", 3)
        brain.inhibit_areas(["ROLE_AGENT"])
        for _ in range(cfg.binding_rounds):
            brain.project(
                {f"PHON_{agent}": ["NOUN_CORE", "ROLE_AGENT"]},
                {"NOUN_CORE": ["ROLE_AGENT"],
                 "ROLE_AGENT": ["NOUN_CORE"]},
            )

        _activate_word(brain, f"PHON_{patient}", "NOUN_CORE", 3)
        brain.inhibit_areas(["ROLE_PATIENT"])
        for _ in range(cfg.binding_rounds):
            brain.project(
                {f"PHON_{patient}": ["NOUN_CORE", "ROLE_PATIENT"]},
                {"NOUN_CORE": ["ROLE_PATIENT"],
                 "ROLE_PATIENT": ["NOUN_CORE"]},
            )

    # Build prediction lexicon
    brain.disable_plasticity = True
    lexicon = {}
    for word in NOUNS + VERBS + NOVEL_NOUNS:
        brain.inhibit_areas(["PREDICTION"])
        for _ in range(cfg.lexicon_readout_rounds):
            brain.project({f"PHON_{word}": ["PREDICTION"]}, {})
        lexicon[word] = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

    # Capture core assemblies
    core_assemblies = {}
    for word in NOUNS + NOVEL_NOUNS:
        _activate_word(brain, f"PHON_{word}", "NOUN_CORE", 3)
        core_assemblies[word] = np.array(
            brain.areas["NOUN_CORE"].winners, dtype=np.uint32)
    for word in VERBS:
        _activate_word(brain, f"PHON_{word}", "VERB_CORE", 3)
        core_assemblies[word] = np.array(
            brain.areas["VERB_CORE"].winners, dtype=np.uint32)

    brain.disable_plasticity = False
    return brain, lexicon, core_assemblies


# ---------------------------------------------------------------------------
# Probe 1: PREDICTION Representational Geometry
# ---------------------------------------------------------------------------

def probe_prediction_geometry(
    brain: Brain,
    cfg: DiagnosticConfig,
    lexicon: Dict[str, np.ndarray],
    log=print,
) -> Dict[str, Any]:
    """Map representational geometry of the PREDICTION area."""
    brain.disable_plasticity = True

    # Stimulus-driven assemblies (already in lexicon)
    stim_nouns = {n: lexicon[n] for n in NOUNS}
    stim_verbs = {v: lexicon[v] for v in VERBS}
    stim_novels = {n: lexicon[n] for n in NOVEL_NOUNS}

    # Context-driven assemblies: VERB_CORE -> PREDICTION
    context_driven = {}
    for verb in VERBS:
        _activate_word(brain, f"PHON_{verb}", "VERB_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"VERB_CORE": ["PREDICTION"]})
        context_driven[verb] = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

    # Also: NOUN_CORE -> PREDICTION (noun context predicts verbs)
    context_from_noun = {}
    for noun in NOUNS:
        _activate_word(brain, f"PHON_{noun}", "NOUN_CORE", 3)
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {"NOUN_CORE": ["PREDICTION"]})
        context_from_noun[noun] = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

    brain.disable_plasticity = False

    # --- Stimulus-mode overlaps ---
    noun_noun = []
    for i, n1 in enumerate(NOUNS):
        for n2 in NOUNS[i+1:]:
            noun_noun.append(measure_overlap(stim_nouns[n1], stim_nouns[n2]))
    verb_verb = []
    for i, v1 in enumerate(VERBS):
        for v2 in VERBS[i+1:]:
            verb_verb.append(measure_overlap(stim_verbs[v1], stim_verbs[v2]))
    cross_cat = []
    for n in NOUNS:
        for v in VERBS:
            cross_cat.append(measure_overlap(stim_nouns[n], stim_verbs[v]))
    novel_trained = []
    for nv in NOVEL_NOUNS:
        for n in NOUNS:
            novel_trained.append(measure_overlap(stim_novels[nv], stim_nouns[n]))

    # --- Context-mode overlaps (verb predictions) ---
    ctx_ctx = []
    for i, v1 in enumerate(VERBS):
        for v2 in VERBS[i+1:]:
            ctx_ctx.append(measure_overlap(context_driven[v1], context_driven[v2]))

    # --- Cross-mode: context-driven vs stimulus-driven ---
    ctx_stim_noun = []
    for v in VERBS:
        for n in NOUNS:
            ctx_stim_noun.append(
                measure_overlap(context_driven[v], stim_nouns[n]))
    ctx_stim_novel = []
    for v in VERBS:
        for nv in NOVEL_NOUNS:
            ctx_stim_novel.append(
                measure_overlap(context_driven[v], stim_novels[nv]))
    ctx_stim_verb = []
    for v in VERBS:
        for v2 in VERBS:
            ctx_stim_verb.append(
                measure_overlap(context_driven[v], stim_verbs[v2]))

    results = {
        "stimulus_mode": {
            "noun_noun": float(np.mean(noun_noun)),
            "verb_verb": float(np.mean(verb_verb)),
            "cross_category": float(np.mean(cross_cat)),
            "novel_to_trained_noun": float(np.mean(novel_trained)),
        },
        "context_mode": {
            "verb_verb": float(np.mean(ctx_ctx)),
        },
        "cross_mode": {
            "context_to_trained_noun": float(np.mean(ctx_stim_noun)),
            "context_to_novel_noun": float(np.mean(ctx_stim_novel)),
            "context_to_verb_stim": float(np.mean(ctx_stim_verb)),
        },
    }

    log("=== Probe 1: PREDICTION Representational Geometry ===")
    sm = results["stimulus_mode"]
    log(f"  Stimulus mode:")
    log(f"    noun-noun={sm['noun_noun']:.4f}  "
        f"verb-verb={sm['verb_verb']:.4f}  "
        f"cross={sm['cross_category']:.4f}  "
        f"novel-noun={sm['novel_to_trained_noun']:.4f}")
    cm = results["context_mode"]
    log(f"  Context mode (verb predictions):")
    log(f"    verb-verb={cm['verb_verb']:.4f}")
    xm = results["cross_mode"]
    log(f"  Cross-mode (context vs stimulus):")
    log(f"    ctx->trained_noun={xm['context_to_trained_noun']:.4f}  "
        f"ctx->novel_noun={xm['context_to_novel_noun']:.4f}  "
        f"ctx->verb_stim={xm['context_to_verb_stim']:.4f}")
    gap = xm["context_to_trained_noun"] - xm["context_to_novel_noun"]
    log(f"  Representational gap (trained - novel): {gap:+.4f}")

    return results


# ---------------------------------------------------------------------------
# Probe 2: Signal Flow Analysis
# ---------------------------------------------------------------------------

def probe_signal_flow(
    brain: Brain,
    cfg: DiagnosticConfig,
    log=print,
) -> Dict[str, Any]:
    """Measure pre_kwta_total signal strength for each pathway."""
    brain.disable_plasticity = True
    engine = brain._engine
    results = {}

    pathways = [
        ("NOUN->PREDICTION", NOUNS, "NOUN_CORE", "PREDICTION"),
        ("VERB->PREDICTION", VERBS, "VERB_CORE", "PREDICTION"),
        ("NOUN->ROLE_PAT", NOUNS, "NOUN_CORE", "ROLE_PATIENT"),
        ("VERB->ROLE_PAT", VERBS, "VERB_CORE", "ROLE_PATIENT"),
        ("NOUN->ROLE_AGT", NOUNS, "NOUN_CORE", "ROLE_AGENT"),
        ("VERB->ROLE_AGT", VERBS, "VERB_CORE", "ROLE_AGENT"),
    ]

    log("=== Probe 2: Signal Flow Analysis ===")
    log(f"  {'Pathway':<22s} {'PreKWTA':>8} {'WeightSum':>10} "
        f"{'WeightMax':>10} {'WeightMean':>11}")
    log("  " + "-" * 65)

    for label, words, src_area, tgt_area in pathways:
        energies = []
        for word in words:
            _activate_word(brain, f"PHON_{word}", src_area, 3)
            brain.inhibit_areas([tgt_area])

            # Sync to engine and project with activation recording
            src_winners = np.asarray(
                brain.areas[src_area].winners, dtype=np.uint32)
            engine.set_winners(src_area, src_winners)

            result = engine.project_into(
                tgt_area,
                from_stimuli=[],
                from_areas=[src_area],
                plasticity_enabled=False,
                record_activation=True,
            )
            engine.set_winners(tgt_area, result.winners)
            brain.areas[tgt_area].winners = result.winners
            brain.areas[tgt_area].w = result.num_ever_fired

            energies.append(result.pre_kwta_total)

        # Weight statistics
        conn = engine._area_conns[src_area][tgt_area]
        w = conn.weights
        if w.ndim == 2 and w.size > 0:
            w_sum = float(np.sum(w))
            w_max = float(np.max(w))
            w_mean = float(np.mean(w))
        else:
            w_sum = w_max = w_mean = 0.0

        mean_energy = float(np.mean(energies))
        results[label] = {
            "pre_kwta_total": mean_energy,
            "weight_sum": w_sum,
            "weight_max": w_max,
            "weight_mean": w_mean,
        }

        log(f"  {label:<22s} {mean_energy:>8.1f} {w_sum:>10.1f} "
            f"{w_max:>10.4f} {w_mean:>11.6f}")

    brain.disable_plasticity = False
    return results


# ---------------------------------------------------------------------------
# Probe 3: P600 Metric Comparison
# ---------------------------------------------------------------------------

def probe_p600_metrics(
    brain: Brain,
    cfg: DiagnosticConfig,
    core_assemblies: Dict[str, np.ndarray],
    log=print,
) -> Dict[str, Any]:
    """Compare four P600 metrics at the critical position."""
    brain.disable_plasticity = True
    engine = brain._engine

    conditions = [
        ("gram", NOUNS, "NOUN_CORE"),
        ("catviol", VERBS, "VERB_CORE"),
        ("novel", NOVEL_NOUNS, "NOUN_CORE"),
    ]
    role_area = "ROLE_PATIENT"

    all_results = {}

    for cond_name, words, core_area in conditions:
        instabilities = []
        weaknesses = []
        cum_energies = []
        anchored_instabilities = []

        for word in words:
            # (i) Jaccard instability: standard settling
            _activate_word(brain, f"PHON_{word}", core_area, 3)
            brain.inhibit_areas([role_area])
            round_winners = []
            for _ in range(cfg.n_settling_rounds):
                brain.project(
                    {},
                    {core_area: [role_area],
                     role_area: [role_area, core_area]},
                )
                round_winners.append(
                    set(brain.areas[role_area].winners.tolist()))
            instabilities.append(compute_jaccard_instability(round_winners))

            # (ii) Binding weakness: forward+reverse retrieval
            _activate_word(brain, f"PHON_{word}", core_area, 3)
            brain.inhibit_areas([role_area])
            for _ in range(3):
                brain.project({}, {core_area: [role_area]})
            core_snap = np.array(brain.areas[core_area].winners, dtype=np.uint32)
            brain.inhibit_areas([core_area])
            for _ in range(3):
                brain.project({}, {role_area: [core_area]})
            retrieved = np.array(brain.areas[core_area].winners, dtype=np.uint32)
            weaknesses.append(1.0 - measure_overlap(retrieved, core_snap))

            # (iii) Cumulative activation: sum pre_kwta_total over rounds
            _activate_word(brain, f"PHON_{word}", core_area, 3)
            src_winners = np.asarray(
                brain.areas[core_area].winners, dtype=np.uint32)
            engine.set_winners(core_area, src_winners)
            engine.fix_assembly(core_area)

            brain.inhibit_areas([role_area])
            cum_energy = 0.0
            for _ in range(cfg.n_settling_rounds):
                result = engine.project_into(
                    role_area,
                    from_stimuli=[],
                    from_areas=[core_area, role_area],
                    plasticity_enabled=False,
                    record_activation=True,
                )
                engine.set_winners(role_area, result.winners)
                brain.areas[role_area].winners = result.winners
                brain.areas[role_area].w = result.num_ever_fired
                cum_energy += result.pre_kwta_total
            cum_energies.append(cum_energy)
            engine.unfix_assembly(core_area)

            # (iv) Anchored instability: prime with stimulus, then settle
            _activate_word(brain, f"PHON_{word}", core_area, 3)
            brain.inhibit_areas([role_area])
            # Phase A: one round with stimulus to create initial pattern
            brain.project(
                {f"PHON_{word}": [core_area, role_area]},
                {core_area: [role_area]},
            )
            anchored_rw = [set(brain.areas[role_area].winners.tolist())]
            # Phase B: settle without stimulus
            for _ in range(cfg.n_settling_rounds - 1):
                brain.project(
                    {},
                    {core_area: [role_area],
                     role_area: [role_area, core_area]},
                )
                anchored_rw.append(
                    set(brain.areas[role_area].winners.tolist()))
            anchored_instabilities.append(
                compute_jaccard_instability(anchored_rw))

        all_results[cond_name] = {
            "jaccard_instability": float(np.mean(instabilities)),
            "binding_weakness": float(np.mean(weaknesses)),
            "cumulative_energy": float(np.mean(cum_energies)),
            "anchored_instability": float(np.mean(anchored_instabilities)),
        }

    brain.disable_plasticity = False

    # Compute activation deficit relative to gram (max trained energy)
    max_energy = all_results["gram"]["cumulative_energy"]
    for cond in all_results:
        all_results[cond]["activation_deficit"] = (
            max_energy - all_results[cond]["cumulative_energy"])

    log("=== Probe 3: P600 Metric Comparison ===")
    log(f"  {'Metric':<25s} {'Gram':>8} {'CatViol':>8} {'Novel':>8}")
    log("  " + "-" * 52)
    for metric in ["jaccard_instability", "binding_weakness",
                    "cumulative_energy", "activation_deficit",
                    "anchored_instability"]:
        g = all_results["gram"][metric]
        c = all_results["catviol"][metric]
        n = all_results["novel"][metric]
        log(f"  {metric:<25s} {g:>8.4f} {c:>8.4f} {n:>8.4f}")

    return all_results


# ---------------------------------------------------------------------------
# Probe 4: Novel Noun Exposure Gradient
# ---------------------------------------------------------------------------

def probe_exposure_gradient(
    cfg: DiagnosticConfig,
    seed: int,
    log=print,
) -> Dict[str, Any]:
    """Train with varying exposure to novel nouns, measure N400 gradient."""
    levels = [0, 1, 3]
    results = {}

    log("=== Probe 4: Novel Noun Exposure Gradient ===")
    log(f"  {'Exposure':<10s} {'Trained_N400':>13} {'Novel_N400':>11} "
        f"{'CatViol_N400':>13}")
    log("  " + "-" * 50)

    for level in levels:
        brain, lexicon, _ = build_trained_brain(
            cfg, seed, novel_training_sentences=level)
        brain.disable_plasticity = True

        # Measure N400 at object position after verb context
        trained_errors = []
        novel_errors = []
        catviol_errors = []

        for verb in VERBS:
            _activate_word(brain, f"PHON_{verb}", "VERB_CORE", 3)
            brain.inhibit_areas(["PREDICTION"])
            brain.project({}, {"VERB_CORE": ["PREDICTION"]})
            predicted = np.array(
                brain.areas["PREDICTION"].winners, dtype=np.uint32)

            for noun in NOUNS:
                trained_errors.append(
                    1.0 - measure_overlap(predicted, lexicon[noun]))
            for noun in NOVEL_NOUNS:
                novel_errors.append(
                    1.0 - measure_overlap(predicted, lexicon[noun]))
            for verb2 in VERBS:
                if verb2 != verb:
                    catviol_errors.append(
                        1.0 - measure_overlap(predicted, lexicon[verb2]))

        brain.disable_plasticity = False

        t_mean = float(np.mean(trained_errors))
        n_mean = float(np.mean(novel_errors))
        c_mean = float(np.mean(catviol_errors))

        results[f"exposure_{level}"] = {
            "trained_n400": t_mean,
            "novel_n400": n_mean,
            "catviol_n400": c_mean,
        }

        log(f"  {level:<10d} {t_mean:>13.4f} {n_mean:>11.4f} {c_mean:>13.4f}")

    return results


# ---------------------------------------------------------------------------
# Probe 5: Per-Round Settling Dynamics
# ---------------------------------------------------------------------------

def probe_settling_dynamics(
    brain: Brain,
    cfg: DiagnosticConfig,
    core_assemblies: Dict[str, np.ndarray],
    log=print,
) -> Dict[str, Any]:
    """Record per-round dynamics during binding settling."""
    brain.disable_plasticity = True
    engine = brain._engine
    role_area = "ROLE_PATIENT"

    test_words = [
        ("gram", NOUNS[0], "NOUN_CORE"),
        ("catviol", VERBS[0], "VERB_CORE"),
        ("novel", NOVEL_NOUNS[0], "NOUN_CORE"),
    ]

    results = {}

    for cond_name, word, core_area in test_words:
        _activate_word(brain, f"PHON_{word}", core_area, 3)
        src_winners = np.asarray(
            brain.areas[core_area].winners, dtype=np.uint32)
        engine.set_winners(core_area, src_winners)
        engine.fix_assembly(core_area)

        brain.inhibit_areas([role_area])

        rounds_data = []
        prev_winners = set()

        for r in range(cfg.n_settling_rounds):
            result = engine.project_into(
                role_area,
                from_stimuli=[],
                from_areas=[core_area, role_area],
                plasticity_enabled=False,
                record_activation=True,
            )
            engine.set_winners(role_area, result.winners)
            brain.areas[role_area].winners = result.winners
            brain.areas[role_area].w = result.num_ever_fired

            curr_winners = set(int(w) for w in result.winners)

            if prev_winners and (prev_winners | curr_winners):
                jac = len(prev_winners & curr_winners) / len(
                    prev_winners | curr_winners)
            else:
                jac = None

            core_ovlp = measure_overlap(
                result.winners, core_assemblies[word])

            rounds_data.append({
                "round": r,
                "energy": result.pre_kwta_total,
                "jaccard_prev": jac,
                "core_overlap": core_ovlp,
            })
            prev_winners = curr_winners

        engine.unfix_assembly(core_area)
        results[cond_name] = {
            "word": word,
            "core_area": core_area,
            "rounds": rounds_data,
        }

        log(f"\n  \"{word}\" -> {role_area} ({cond_name}, "
            f"{'trained' if cond_name != 'catviol' else 'untrained'})")
        log(f"  {'Rnd':>4s} {'Energy':>8s} {'Jac(prev)':>10s} "
            f"{'CoreOvlp':>9s}")
        for rd in rounds_data:
            jac_str = f"{rd['jaccard_prev']:.4f}" if rd[
                "jaccard_prev"] is not None else "--"
            log(f"  {rd['round']:>4d} {rd['energy']:>8.1f} "
                f"{jac_str:>10s} {rd['core_overlap']:>9.4f}")

    brain.disable_plasticity = False
    return results


# ---------------------------------------------------------------------------
# Experiment orchestrator
# ---------------------------------------------------------------------------

class DiagnoseERPExperiment(ExperimentBase):
    """Diagnostic investigation of N400/P600 dynamics."""

    def __init__(self, results_dir=None, seed=42, verbose=True):
        super().__init__(
            name="diagnose_erp_dynamics",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 3,
        config: Optional[DiagnosticConfig] = None,
        probes: Optional[List[str]] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or DiagnosticConfig(
            **{k: v for k, v in kwargs.items()
               if k in DiagnosticConfig.__dataclass_fields__})

        all_probes = ["geometry", "signal_flow", "p600_metrics",
                      "exposure", "settling"]
        run_probes = probes or all_probes

        self.log("=" * 70)
        self.log("Diagnostic Investigation: N400/P600 Dynamics")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  Probes: {', '.join(run_probes)}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        all_metrics = {}

        for s in range(n_seeds):
            self.log(f"\n{'='*70}")
            self.log(f"Seed {s+1}/{n_seeds} (seed={self.seed + s})")
            self.log(f"{'='*70}")

            brain, lexicon, core_assemblies = build_trained_brain(
                cfg, self.seed + s)

            if "geometry" in run_probes:
                r = probe_prediction_geometry(
                    brain, cfg, lexicon, log=self.log)
                all_metrics.setdefault("geometry", []).append(r)

            if "signal_flow" in run_probes:
                r = probe_signal_flow(brain, cfg, log=self.log)
                all_metrics.setdefault("signal_flow", []).append(r)

            if "p600_metrics" in run_probes:
                r = probe_p600_metrics(
                    brain, cfg, core_assemblies, log=self.log)
                all_metrics.setdefault("p600_metrics", []).append(r)

            if "exposure" in run_probes:
                r = probe_exposure_gradient(
                    cfg, self.seed + s, log=self.log)
                all_metrics.setdefault("exposure", []).append(r)

            if "settling" in run_probes:
                r = probe_settling_dynamics(
                    brain, cfg, core_assemblies, log=self.log)
                all_metrics.setdefault("settling", []).append(r)

        # --- Aggregate across seeds ---
        self.log(f"\n{'='*70}")
        self.log("AGGREGATE SUMMARY")
        self.log(f"{'='*70}")

        agg = {}

        if "geometry" in all_metrics:
            geo = all_metrics["geometry"]
            agg["geometry"] = {
                "context_to_trained_noun": summarize(
                    [g["cross_mode"]["context_to_trained_noun"] for g in geo]),
                "context_to_novel_noun": summarize(
                    [g["cross_mode"]["context_to_novel_noun"] for g in geo]),
                "context_to_verb_stim": summarize(
                    [g["cross_mode"]["context_to_verb_stim"] for g in geo]),
                "novel_to_trained_stim": summarize(
                    [g["stimulus_mode"]["novel_to_trained_noun"] for g in geo]),
            }
            ag = agg["geometry"]
            self.log(f"\n  Geometry (cross-mode, mean across seeds):")
            self.log(f"    ctx->trained_noun: "
                     f"{ag['context_to_trained_noun']['mean']:.4f}")
            self.log(f"    ctx->novel_noun:   "
                     f"{ag['context_to_novel_noun']['mean']:.4f}")
            self.log(f"    ctx->verb_stim:    "
                     f"{ag['context_to_verb_stim']['mean']:.4f}")

        if "p600_metrics" in all_metrics:
            p6 = all_metrics["p600_metrics"]
            self.log(f"\n  P600 Metrics (mean across seeds):")
            for metric in ["jaccard_instability", "binding_weakness",
                           "activation_deficit", "anchored_instability"]:
                g_vals = [p[("gram")][metric] for p in p6]
                c_vals = [p["catviol"][metric] for p in p6]
                n_vals = [p["novel"][metric] for p in p6]
                if n_seeds >= 2:
                    test = paired_ttest(c_vals, g_vals)
                    d_str = f"d={test['d']:.2f}"
                else:
                    d_str = "n/a"
                self.log(f"    {metric:<25s} g={np.mean(g_vals):.4f}  "
                         f"c={np.mean(c_vals):.4f}  "
                         f"n={np.mean(n_vals):.4f}  {d_str}")

        if "exposure" in all_metrics:
            exp = all_metrics["exposure"]
            self.log(f"\n  Exposure gradient (mean across seeds):")
            for key in ["exposure_0", "exposure_1", "exposure_3"]:
                t_vals = [e[key]["trained_n400"] for e in exp]
                n_vals = [e[key]["novel_n400"] for e in exp]
                c_vals = [e[key]["catviol_n400"] for e in exp]
                self.log(f"    {key}: trained={np.mean(t_vals):.4f}  "
                         f"novel={np.mean(n_vals):.4f}  "
                         f"catviol={np.mean(c_vals):.4f}")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "n_seeds": n_seeds,
                "probes": run_probes,
            },
            metrics=agg,
            raw_data=all_metrics,
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic investigation of ERP dynamics")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--probe", type=str, default=None,
                        choices=["geometry", "signal_flow", "p600_metrics",
                                 "exposure", "settling"])
    args = parser.parse_args()

    exp = DiagnoseERPExperiment(verbose=True)

    if args.quick:
        cfg = DiagnosticConfig(
            n=5000, k=50, training_reps=3, n_train_sentences=15)
        n_seeds = args.seeds or 3
    else:
        cfg = DiagnosticConfig()
        n_seeds = args.seeds or 5

    probes = [args.probe] if args.probe else None
    result = exp.run(n_seeds=n_seeds, config=cfg, probes=probes)

    suffix = "_quick" if args.quick else ""
    probe_suffix = f"_{args.probe}" if args.probe else ""
    exp.save_result(result, f"{probe_suffix}{suffix}")


if __name__ == "__main__":
    main()
