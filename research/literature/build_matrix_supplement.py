#!/usr/bin/env python3
"""Build reproduction_matrix_supplement.json from audit findings."""

from __future__ import annotations

import json
from pathlib import Path

LIT = Path(__file__).resolve().parent

# (claim_id, paper_id, category, claim, status, test, protocol_id, config_ref, gap, priority)
SUPPLEMENT = [
    # --- ITCS / CCNeuro ---
    ("ITCS19-E02", "papadimitriou2019random", "empirical", "O(log n) convergence rounds vs n sweep", "research", "research/experiments/stability/test_scaling_laws.py", None, "itcs2019.yaml", "Promote to parity golden", "P3"),
    ("ITCS19-E03", "papadimitriou2019random", "empirical", "Intra-assembly density vs beta sweep", "missing", None, "itcs2019.density", "itcs2019.yaml", "Pin density_simulator golden", "P3"),
    ("ITCS19-M05", "papadimitriou2019random", "mechanism", "Attractor invariance under extra recurrent rounds", "partial", "neural_assemblies/tests/test_assembly_calculus.py -k stable_assembly", None, "pnas2020.yaml", "Add literature parity at paper n", "P2"),
    ("ITCS19-M06", "papadimitriou2019random", "mechanism", "FiberCircuit inhibit/disinhibit gating", "partial", "neural_assemblies/tests/test_assembly_calculus.py -k FiberCircuit", None, "pnas2020.yaml", "Matrix row for fiber primitive", "P1"),
    ("ITCS19-C01", "papadimitriou2019random", "config", "CI n=5000 vs paper n=10000 param regime documented", "partial", None, "global.param_regimes", "pnas2020.yaml", "Reconcile all test defaults", "P0"),
    ("CCN19-D01", "papadimitriou2019ccneuro", "demo", "NV/VN language production via LearnBrain_SimpleSyntax", "legacy", "legacy/root_modules/learner.py", "ccneuro.nv_vn", "pnas2020.yaml", "Port to package or pin legacy", "P3"),
    ("CCN19-M02", "papadimitriou2019ccneuro", "mechanism", "CORE areas custom_inner_p=0.9 syntax gating", "legacy", None, None, "pnas2020.yaml", "Not in neural_assemblies", "P4"),
    # --- PNAS extended ---
    ("PNAS20-M07", "papadimitriou2020brain", "mechanism", "Reciprocal restore overlap > 0.6", "partial", "neural_assemblies/tests/test_assembly_calculus.py -k reciprocal", None, "pnas2020.yaml", "Golden vs ref", "P1"),
    ("PNAS20-M08", "papadimitriou2020brain", "mechanism", "Pattern completion 50% cue recovery > 0.6", "partial", "neural_assemblies/tests/test_assembly_calculus.py -k pattern_completion", None, "pnas2020.yaml", "Reconcile 0.6 test vs 0.8 doc", "P1"),
    ("PNAS20-M09", "papadimitriou2020brain", "mechanism", "Pattern completion monotonic in cue fraction", "partial", "neural_assemblies/tests/test_assembly_calculus.py -k degrades_with_less", None, "pnas2020.yaml", "Pin alpha sweep protocol", "P2"),
    ("PNAS20-M10", "papadimitriou2020brain", "mechanism", "Merge responds to either source > 2× chance", "partial", "neural_assemblies/tests/test_assembly_calculus.py -k merge_responds", None, "pnas2020.yaml", "Distinct from parity merge test", "P1"),
    ("PNAS20-M11", "papadimitriou2020brain", "mechanism", "FiberCircuit for language/TM circuits", "partial", "neural_assemblies/tests/test_assembly_calculus.py -k FiberCircuit", None, "pnas2020.yaml", None, "P1"),
    ("PNAS20-E03", "papadimitriou2020brain", "empirical", "Cross-repo associate overlap golden ±0.02", "pinned", "neural_assemblies/tests/test_cross_repo_parity.py::TestCrossRepoExtendedPNAS", "pnas2020.associate", "pnas2020.yaml", None, "P1"),
    ("PNAS20-E04", "papadimitriou2020brain", "empirical", "Cross-repo merge cue overlap golden", "pinned", "neural_assemblies/tests/test_cross_repo_parity.py::TestCrossRepoExtendedPNAS", "pnas2020.merge", "pnas2020.yaml", None, "P1"),
    ("PNAS20-E05", "papadimitriou2020brain", "empirical", "Cross-repo reciprocal restore golden", "pinned", "neural_assemblies/tests/test_cross_repo_parity.py::TestCrossRepoExtendedPNAS", "pnas2020.reciprocal", "pnas2020.yaml", None, "P2"),
    ("PNAS20-E06", "papadimitriou2020brain", "empirical", "Pattern completion curves vs alpha/comp_iter", "pinned", "neural_assemblies/tests/test_cross_repo_parity.py::TestCrossRepoExtendedPNAS", "pnas2020.pattern_com", "pnas2020.yaml", "Single alpha=0.5 point", "P2"),
    ("PNAS20-E07", "papadimitriou2020brain", "empirical", "Associate overlap vs interleave iterations", "missing", None, "pnas2020.associate_iter", "pnas2020.yaml", "association_grand_sim", "P2"),
    ("PNAS20-E08", "papadimitriou2020brain", "empirical", "Assembly capacity per area vs n,k", "research", "research/experiments/distinctiveness/test_capacity_limits.py", None, "pnas2020.yaml", "Separate from ITCS19-E01", "P3"),
    ("PNAS20-C01", "papadimitriou2020brain", "config", "All PNAS param regimes reconciled in one registry", "partial", None, "pnas2020.regimes", "pnas2020.yaml", "See parity/configs/pnas2020.yaml", "P0"),
    ("PNAS20-C02", "papadimitriou2020brain", "config", "Associate 3-phase rounds + overlap_iter=10", "partial", None, "pnas2020.associate_config", "pnas2020.yaml", "Legacy vs ops.associate", "P1"),
    ("PNAS20-D02", "papadimitriou2020brain", "demo", "Interrogative mood reverses NV/VN order", "legacy", None, "pnas2020.interrogative", "pnas2020.yaml", "legacy learner.py", "P3"),
    # --- COLT ---
    ("COLT22-M02", "dabagia2022classify", "mechanism", "fuzzy_readout classify threshold=0.3", "partial", "neural_assemblies/programs/learn.py", "colt2022.classify", "colt2022.yaml", "Add classify parity test", "P1"),
    ("COLT22-C01", "dabagia2022classify", "config", "learn_assembly convergence 0.90 API vs 0.85 protocol", "partial", None, "colt2022.learn_assembly", "colt2022.yaml", "Document in PROTOCOLS", "P0"),
    ("COLT22-C02", "dabagia2022classify", "config", "learn_separable_classes vs parity hyperparam mismatch", "partial", None, "colt2022.separable", "colt2022.yaml", "Unify entrypoints", "P1"),
    ("COLT22-E04", "dabagia2022classify", "empirical", "Hierarchical MNIST/CIFAR assembly-per-class", "legacy", "legacy/root_modules/image_learner.py", "colt2022_mnist", "colt2022.yaml", "Official: .reference/mdabagia-learning-with-assemblies/MNIST.ipynb", "P1"),
    ("COLT22-E05", "dabagia2022classify", "empirical", "Multi-class online interference in shared CLASS area", "missing", None, "colt2022.multiclass", "colt2022.yaml", None, "P2"),
    ("COLT22-M03", "dabagia2022classify", "mechanism", "learn_class_assembly wrapper", "partial", "neural_assemblies/programs/learn.py", None, "colt2022.yaml", "Add wrapper test", "P2"),
    # --- Sequences ---
    ("SEQ25-M05", "dabagia2025sequences", "mechanism", "FiberCircuit gates TM tape symbol presentation", "partial", "neural_assemblies/programs/tm_demo.py", "sequences2025.tm_fiber", "sequences2025.yaml", None, "P1"),
    ("SEQ25-M06", "dabagia2025sequences", "mechanism", "ordered_recall requires refractory_period>0", "partial", "neural_assemblies/tests/test_sequences.py -k requires_lri", None, "sequences2025.yaml", "LRI ablation protocol", "P1"),
    ("SEQ25-M07", "dabagia2025sequences", "mechanism", "Hard vs soft LRI inhibition strength effects", "partial", "neural_assemblies/tests/test_lri.py", None, "sequences2025.yaml", "Golden LRI curves", "P1"),
    ("SEQ25-C01", "dabagia2025sequences", "config", "TM/FSM/parity param regime reconciliation", "partial", None, "sequences2025.regimes", "sequences2025.yaml", "See sequences2025.yaml", "P0"),
    ("SEQ25-C02", "dabagia2025sequences", "config", "sequence_memorize phase_b_ratio/beta_boost", "research", "neural_assemblies/tests/test_sequence_recall_sweep.py", None, "sequences2025.yaml", "Pin sweep winners", "P2"),
    ("SEQ25-C03", "dabagia2025sequences", "config", "ordered_recall termination thresholds", "partial", None, "sequences2025.recall_terminates", "sequences2025.yaml", "Add to PROTOCOLS", "P1"),
    ("SEQ25-C04", "dabagia2025sequences", "config", "TM FSM transition table pinned", "pinned", "neural_assemblies/programs/tm_demo.py", "sequences2025.tm_table", "sequences2025.yaml", None, "P0"),
    ("SEQ25-E06", "dabagia2025sequences", "empirical", "Recall recovers 2nd+ memorized items", "partial", "neural_assemblies/tests/test_sequences.py -k recovers_memorized", None, "sequences2025.yaml", "Parity only checks item 0", "P1"),
    ("SEQ25-E07", "dabagia2025sequences", "empirical", "Recall length 5/5 at paper scale", "research", "neural_assemblies/tests/test_sequence_recall_sweep.py", "sequences2025_recall_len", "sequences2025.yaml", "Golden from sweep", "P1"),
    ("SEQ25-E08", "dabagia2025sequences", "empirical", "Memorize repetitions strengthen links", "partial", "neural_assemblies/tests/test_sequences.py -k repetitions_strengthen", None, "sequences2025.yaml", None, "P2"),
    ("SEQ25-E09", "dabagia2025sequences", "empirical", "Memorized items pairwise overlap < 0.5", "partial", "neural_assemblies/tests/test_sequences.py -k distinct_assemblies", None, "sequences2025.yaml", None, "P1"),
    ("SEQ25-M08", "dabagia2025sequences", "mechanism", "FSM learner rejects contradictory traces", "partial", "neural_assemblies/programs/fsm_learn.py", None, "sequences2025.yaml", "Add conflict test", "P2"),
    ("SEQ25-E10", "dabagia2025sequences", "empirical", "Multi-state FSM inference beyond 2-state toy", "missing", None, "sequences2025.fsm_multistate", "sequences2025.yaml", "Ref repo traces", "P2"),
    ("SEQ25-E11", "dabagia2025sequences", "empirical", "turing_simulations larger_k / turing_erase", "research", "neural_assemblies/simulation/turing_simulations.py", None, "sequences2025.yaml", None, "P3"),
    ("SEQ25-E12", "dabagia2025sequences", "empirical", "lri_recall_sweep diagnostics n=4000", "research", "neural_assemblies/assembly_calculus/tracing/sweeps.py", None, "sequences2025.yaml", None, "P3"),
    ("SEQ25-E13", "dabagia2025sequences", "empirical", "GPU full-chain recall parity (not item-0 only)", "partial", "neural_assemblies/tests/test_literature_parity.py::TestLiteratureParityGPU", None, "sequences2025.yaml", "Extend GPU assertion", "P1"),
    # --- TACL ---
    ("TACL21-C01", "mitropolsky2021parser", "config", "English parser hyperparameters pinned", "partial", None, "tacl2021.english", "tacl2021.yaml", "See tacl2021.yaml", "P0"),
    ("TACL21-C02", "mitropolsky2021parser", "config", "Russian parser hyperparameters pinned", "partial", None, "tacl2021.russian", "tacl2021.yaml", None, "P0"),
    ("TACL21-C03", "mitropolsky2021parser", "config", "English grammar area graph + fiber names", "partial", None, "tacl2021.areas_en", "tacl2021.yaml", None, "P1"),
    ("TACL21-C04", "mitropolsky2021parser", "config", "Russian case paradigm → NOM/ACC/DAT", "partial", None, "tacl2021.russian_cases", "tacl2021.yaml", None, "P1"),
    ("TACL21-M03", "mitropolsky2021parser", "mechanism", "LEX war of fibers ≤2 targets per step", "partial", "neural_assemblies/language/parser.py", None, "tacl2021.yaml", "Assert in test", "P2"),
    ("TACL21-M04", "mitropolsky2021parser", "mechanism", "PRE_RULES → project_rounds → POST_RULES loop", "partial", "neural_assemblies/language/__init__.py", None, "tacl2021.yaml", None, "P1"),
    ("TACL21-M05", "mitropolsky2021parser", "mechanism", "FIXED_MAP_READOUT alternative readout", "missing", None, "tacl2021.fixed_readout", "tacl2021.yaml", "Only FIBER tested", "P2"),
    ("TACL21-E06", "mitropolsky2021parser", "empirical", "DAT-initial ditransitive roles", "pinned", "neural_assemblies/tests/test_tacl_parser_suite.py::test_russian_dative_initial_order_roles", "tacl2021.ru_dat_initial", "tacl2021.yaml", None, "P0"),
    ("TACL21-E07", "mitropolsky2021parser", "empirical", "Extended lexicon words parse", "missing", None, "tacl2021.extended_lex", "tacl2021.yaml", "Add sentences for untested lemmas", "P2"),
    ("TACL21-E08", "mitropolsky2021parser", "empirical", "Cross-engine parser on mixed LEX+sparse", "pinned", "neural_assemblies/tests/test_cross_engine_projection.py", "cross_engine.parser", "tacl2021.yaml", None, "P0"),
    # --- NALOMA ---
    ("NALO22-M03", "mitropolsky2022center", "mechanism", "that-triggered DEP_CLAUSE entry + outer freeze", "partial", "neural_assemblies/language/__init__.py", "naloma2022.that_entry", "naloma2022.yaml", None, "P1"),
    ("NALO22-M04", "mitropolsky2022center", "mechanism", "Comma WM replay plasticity-off token range", "partial", "neural_assemblies/language/__init__.py", "naloma2022.wm_replay", "naloma2022.yaml", "Pin saved_outer_start→inner", "P1"),
    ("NALO22-M05", "mitropolsky2022center", "mechanism", "Inner/outer lexeme cache scopes", "partial", None, "naloma2022.caches", "naloma2022.yaml", "Golden readout", "P1"),
    ("NALO22-E03", "mitropolsky2022center", "empirical", "Depth-2 center embedding parses", "missing", None, "naloma2022.depth2", "naloma2022.yaml", "Parser support + lexicon", "P2"),
    ("NALO22-E04", "mitropolsky2022center", "empirical", "CE processing difficulty / N400 gradient", "research", "research/experiments/primitives/test_center_embedded_baseline.py", None, "naloma2022.yaml", "Psycholinguistic tier", "P3"),
    ("NALO22-E05", "mitropolsky2022center", "empirical", "Inner-clause SUBJ via DEP_CLAUSE path", "missing", None, "naloma2022.inner_subj", "naloma2022.yaml", None, "P2"),
    # --- AAAI ---
    ("AAAI22-C01", "damore2022planning", "config", "BlocksWorldAC neural scaffold defaults", "partial", None, "aaai2022.ac_defaults", "aaai2022.yaml", None, "P0"),
    ("AAAI22-C02", "damore2022planning", "config", "FSM pick/put control graph", "partial", None, "aaai2022.fsm", "aaai2022.yaml", None, "P1"),
    ("AAAI22-C03", "damore2022planning", "config", "STRIPS BlocksAction encoding", "pinned", "neural_assemblies/programs/planning.py", "aaai2022.strips", "aaai2022.yaml", None, "P0"),
    ("AAAI22-E04", "damore2022planning", "empirical", "Neural execution tracks discrete STRIPS state", "partial", "neural_assemblies/programs/planning.py", "aaai2022.discrete_track", "aaai2022.yaml", "Document shortcut", "P1"),
    ("AAAI22-E05", "damore2022planning", "empirical", "Holding-area rounds-1 recurrent per pick/put", "partial", None, "aaai2022.holding_rounds", "aaai2022.yaml", None, "P2"),
    # --- Organ ---
    ("ORG23-C01", "mitropolsky2023architecture", "config", "45-area emergent organ graph", "partial", None, "organ2023.areas", "organ2023.yaml", None, "P1"),
    ("ORG23-C02", "mitropolsky2023architecture", "config", "GROUNDING_TO_CORE modality wiring", "partial", None, "organ2023.grounding", "organ2023.yaml", None, "P1"),
    ("ORG23-M04", "mitropolsky2023architecture", "mechanism", "Toy NemoParser pipeline areas", "partial", "neural_assemblies/assembly_calculus/parser.py", None, "organ2023.yaml", None, "P2"),
    ("ORG23-M05", "mitropolsky2023architecture", "mechanism", "Organ-level LRI on SEQ", "partial", "neural_assemblies/tests/test_lri.py", None, "organ2023.yaml", "Integration test missing", "P2"),
    ("ORG23-E03", "mitropolsky2023architecture", "empirical", "Mini organ noun/verb ≥80% (6 words)", "partial", "neural_assemblies/tests/test_nemo_patterns_core.py", None, "organ2023.yaml", None, "P2"),
    ("ORG23-E04", "mitropolsky2023architecture", "empirical", "Within-category lexicon distinctness <0.5", "partial", "neural_assemblies/tests/test_nemo_patterns_core.py -k distinct", None, "organ2023.yaml", None, "P2"),
    ("ORG23-E05", "mitropolsky2023architecture", "empirical", "SVO via sequence_memorize at calculus layer", "partial", "neural_assemblies/tests/test_nemo_patterns_core.py -k svo", None, "organ2023.yaml", None, "P2"),
    # --- Coin ---
    ("COIN24-C01", "dabagia2024coinflipping", "config", "CoinFlipModel code defaults", "partial", None, "coin2024.defaults", "coin2024.yaml", None, "P0"),
    ("COIN24-C02", "dabagia2024coinflipping", "config", "Ambient noise std + E%-WTA wiring", "retracted", "neural_assemblies/tests/test_literature_parity.py::TestLiteratureParity::test_coin_ambient_noise_epwta_wiring", "coin2024.ambient", "coin2024.yaml", "Legacy coin construction", "P0"),
    ("COIN24-C03", "dabagia2024coinflipping", "config", "dabagia.org demo URL params scraped", "retracted", "neural_assemblies/tests/test_literature_golden.py::TestRetractedCoinGoldens", "coin2024_demo", "coin2024.yaml", "Package simulation used an invalid instrument; demo has no numeric API", "P1"),
    ("COIN24-C04", "dabagia2024coinflipping", "config", "Markov order / multi-state chains", "partial", None, "coin2024.markov_order", "coin2024.yaml", "Higher-order PFA", "P2"),
    ("COIN24-M04", "dabagia2024coinflipping", "mechanism", "Flip via proportional k-split (not softmax)", "partial", "neural_assemblies/assembly_calculus/pfa.py", None, "coin2024.yaml", "Reconcile E04 claim", "P1"),
    ("COIN24-E05", "dabagia2024coinflipping", "empirical", "PFA 30-step stochastic routing", "retracted", "neural_assemblies/tests/test_literature_parity.py::TestLiteratureParity::test_pfa_stochastic_both_targets", "coin2024.pfa_route", "coin2024.yaml", "Implicit selector was not a calibrated probability", "P0"),
    # --- NEMO 2025 extended ---
    ("NEMO25-C01", "mitropolsky2025simulated", "config", "Vocabulary size + grounding modalities", "partial", None, "nemo2025.vocab", "nemo2025.yaml", "45 words pinned", "P0"),
    ("NEMO25-C02", "mitropolsky2025simulated", "config", "train() phase order vs curriculum", "partial", None, "nemo2025.train_phases", "nemo2025.yaml", "Document divergence", "P1"),
    ("NEMO25-C03", "mitropolsky2025simulated", "config", "Developmental _STAGE_CONFIG betas/phases", "partial", None, "nemo2025.stages", "nemo2025.yaml", "See nemo2025.yaml", "P0"),
    ("NEMO25-C04", "mitropolsky2025simulated", "config", "STAGE_TRAINING_ROUNDS + DISTRIBUTIONAL_REPS", "partial", None, "nemo2025.stage_reps", "nemo2025.yaml", None, "P0"),
    ("NEMO25-C05", "mitropolsky2025simulated", "config", "Parity n=5000 vs test n=10000 brain scale", "partial", None, "nemo2025.brain_regimes", "nemo2025.yaml", None, "P0"),
    ("NEMO25-E07", "mitropolsky2025simulated", "empirical", "All 45 areas registered", "partial", "neural_assemblies/tests/test_emergent_parser.py -k areas", None, "nemo2025.yaml", None, "P1"),
    ("NEMO25-E08", "mitropolsky2025simulated", "empirical", "Full VOCABULARY POS ≥80%", "partial", "neural_assemblies/tests/test_emergent_parser.py::test_overall_accuracy_above_80_percent", None, "nemo2025.yaml", None, "P1"),
    ("NEMO25-E09", "mitropolsky2025simulated", "empirical", "Parsing category+role ≥80% (5 sentences)", "partial", "neural_assemblies/tests/test_emergent_parser.py::test_overall_parsing_accuracy", None, "nemo2025.yaml", None, "P1"),
    ("NEMO25-E10", "mitropolsky2025simulated", "empirical", "Neural role accuracy ≥80% multi-sentence", "partial", "neural_assemblies/tests/test_emergent_parser.py::test_neural_role_accuracy_above_80_percent", None, "nemo2025.yaml", None, "P1"),
    ("NEMO25-E11", "mitropolsky2025simulated", "empirical", "Holdout POS generalization ≥66%", "partial", "neural_assemblies/tests/test_emergent_parser.py::test_generalization_accuracy", None, "nemo2025.yaml", None, "P2"),
    ("NEMO25-E12", "mitropolsky2025simulated", "empirical", "Scaled 200-word preset POS ≥80%", "partial", "neural_assemblies/tests/test_emergent_parser.py::test_scaled_pos_accuracy_above_80", None, "nemo2025.yaml", "Paper-scale E03", "P1"),
    ("NEMO25-E13", "mitropolsky2025simulated", "empirical", "Incremental parse matches batch ≥90%", "partial", "neural_assemblies/tests/test_emergent_parser.py::test_incremental_matches_batch_categories", None, "nemo2025.yaml", None, "P2"),
    ("NEMO25-E14", "mitropolsky2025simulated", "empirical", "Unsupervised roles ≥60% (package test)", "partial", "neural_assemblies/tests/test_emergent_parser.py::test_unsupervised_accuracy_above_60", None, "nemo2025.yaml", None, "P2"),
    ("NEMO25-E15", "mitropolsky2025simulated", "empirical", "NP/PP phrase identification", "partial", "neural_assemblies/tests/test_emergent_parser.py -k np_identification", None, "nemo2025.yaml", None, "P2"),
    ("NEMO25-E16", "mitropolsky2025simulated", "empirical", "Generation roundtrip produce→parse", "partial", "neural_assemblies/tests/test_emergent_parser.py::test_generate_roundtrip", None, "nemo2025.yaml", None, "P2"),
    ("NEMO25-D01", "mitropolsky2025simulated", "demo", "Agent blocks/JSON/multi-tool extension", "partial", "neural_assemblies/tests/test_emergent_agent.py", "emergent.agent", "nemo2025.yaml", "Out of paper scope", "P3"),
    # --- Hoff extended ---
    ("EPWTA26-C01", "hoff2026epwta", "config", "Paper vs repo E%-WTA param mapping", "partial", None, "hoff2026.regimes", "hoff2026.yaml", "See hoff2026.yaml", "P1"),
    ("EPWTA26-E04", "hoff2026epwta", "empirical", "Fig 2 failure rate vs beta/omega_inh", "missing", None, "hoff2026.fig2", "hoff2026.yaml", "Feedforward inh missing", "P2"),
    ("EPWTA26-E05", "hoff2026epwta", "empirical", "Fig 3b recovery median 1.0 vs k-WTA", "missing", None, "hoff2026.recovery", "hoff2026.yaml", None, "P2"),
    ("EPWTA26-E06", "hoff2026epwta", "empirical", "Multi-assembly overlap matrices Fig 3c-d", "missing", None, "hoff2026.overlap_matrix", "hoff2026.yaml", None, "P3"),
    # --- DIRECT extended ---
    ("DIR26-C01", "kopadi2026causal", "config", "Paper adaptive_soft schedule params", "missing", None, "direct2026.schedule", "direct2026.yaml", "Port to direct.py", "P2"),
    ("DIR26-C02", "kopadi2026causal", "config", "Synaptic Δ readout vs overlap readout", "missing", None, "direct2026.readout", "direct2026.yaml", None, "P2"),
    ("DIR26-E04", "kopadi2026causal", "empirical", "Alzheimer 12-edge SCM Precision@K=1.0", "missing", None, "direct2026.alzheimer", "direct2026.yaml", None, "P2"),
    ("DIR26-E05", "kopadi2026causal", "empirical", "Perturbation suites R1/R2/R3", "missing", None, "direct2026.perturbations", "direct2026.yaml", None, "P3"),
    # --- Ting (all from paper params) ---
    ("TING26-C01", "ting2026speech", "config", "TIMIT phone classification params", "missing", None, "ting2026.phones", "ting2026.yaml", "Encoder module", "P4"),
    ("TING26-C02", "ting2026speech", "config", "Mel/MFCC binarisation params", "missing", None, "ting2026.mfcc", "ting2026.yaml", None, "P4"),
    ("TING26-C03", "ting2026speech", "config", "Boundary detection two-level architecture", "missing", None, "ting2026.boundaries", "ting2026.yaml", None, "P4"),
    # --- Onasch ---
    ("ONA25-C01", "onasch2025dendritic", "config", "Compartment gating proposed keys", "missing", None, "onasch2025.gating", "onasch2025.yaml", "Extract from bioRxiv", "P4"),
    ("ONA25-C02", "onasch2025dendritic", "config", "Forgetting protocol task sequence", "missing", None, "onasch2025_forgetting", "onasch2025.yaml", None, "P4"),
    # --- Cross-infra ---
    ("XINF-C01", None, "config", "Global PROTOCOLS.md vs per-paper config YAML index", "partial", None, "global.registry", "parity/configs/README.md", None, "P0"),
    ("XINF-C02", None, "config", "Legacy MATLAB assembly_sim.m params", "legacy", "legacy/matlab/assembly_sim.m", None, "pnas2020.yaml", "Cross-lang golden", "P4"),
    ("XINF-E03", None, "empirical", "Julia AssemblyCalculus.jl capacity JSON", "missing", None, "cross_lang.julia", "pnas2020.yaml", "XINF-E02 extended", "P3"),
]


def row(t):
    cid, pid, cat, claim, status, test, proto, cfg, gap, pri = t
    cmd = None
    if test and test.endswith(".py") or (test and "pytest" in str(test)):
        cmd = f"uv run pytest {test} -v" if test and not test.startswith("uv") else test
    elif test and "::" in test:
        cmd = f"uv run pytest {test} -v"
    elif test and test.startswith("neural_assemblies"):
        cmd = f"uv run pytest {test} -v"
    return {
        "claim_id": cid,
        "paper_id": pid,
        "category": cat,
        "claim": claim,
        "status": status,
        "module": None,
        "test": test if test and not test.startswith("research/") else test,
        "protocol_id": proto,
        "config_ref": f"parity/configs/{cfg}" if cfg and cfg.endswith(".yaml") else cfg,
        "golden": None,
        "gap": gap,
        "priority": pri,
        "repro_command": cmd,
    }


def main() -> None:
    claims = [row(t) for t in SUPPLEMENT]
    out = {
        "version": 1,
        "updated": "2026-06-23",
        "description": "Supplemental claim rows from exhaustive paper audit (configs, legacy, extended empirical).",
        "claims": claims,
    }
    path = LIT / "reproduction_matrix_supplement.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(claims)} supplement claims to {path}")


if __name__ == "__main__":
    main()
