# Orphan result files: disposition inventory (2026-09-13)

Source: `uv run python -m research.evidence audit` in the worktree
`assemblies-astra-audit-20260909`; `candidate_orphan_results` = 374 files
(research/evidence.py:330: a results file with no literal incoming reference from a
tracked .py or .md). Backlog item: TODO.md:249-252. Nothing here proposes deletion.

Conventions. "runner" = the `EXPERIMENTS` dict, research/runner.py:535-560.
Line names follow research/notes/README.md:47-58 (memory, sequence, aligner,
substrate), research/notes/language/README.md:3 (language), and the
research/notes/categories and research/notes/coin folders. Timestamped files
`<name>_<YYYYMMDD_HHMMSS>[_quick|_full].json` are written by
`ExperimentBase.save_result` (research/experiments/base.py:167-171); `_quick` is the
script's `--quick` flag (e.g. primitives/test_freeform_learner.py:441).
"Sampled?" answers whether the producer runs `numpy_sparse` without
materializing; that voids SEQUENCE numbers only (PREREG_sampler_audit.md:1-9;
research/notes/README.md:257-260).

## Clusters

| # | Cluster (dir / stem) | n | Example | Producer (exists? runner?) | Line | Note cites producer? | Sampled? | Last commit | Disposition |
|---|---|---|---|---|---|---|---|---|---|
| 1 | results/primitives/freeform_{learner,det,adj,opt_adj}_* | 47 | primitives/freeform_learner_20260222_131302_quick.json | primitives/test_freeform_learner.py (+ _det, _adj, _opt_adj); exist; not in runner. Writes results/primitives at test_freeform_learner.py:272-273 | language | none found (grep `freeform_` in research/notes) | no engine arg (base.py sets none); not a sequence claim | 2026-02-22 | HISTORICAL |
| 2 | results/primitives calculus primitives: association_binding, merge_composition, projection_convergence, association, association_chain, projection, inhibition | 28 | primitives/association_binding_20251128_230322_quick.json; primitives/merge_composition_20260206_145449_full.json | earlier producer names (docs/reviews/whole-codebase/STATIC_DEBT_DISPOSITIONS.md:48); current test_association.py (docstring "Historical association", 2026-09-11), test_merge.py, test_projection.py exist; not in runner | substrate (core questions Q01, research/core_questions/index.json:5-8) | RESULTS_association.md / RESULTS_merge.md / RESULTS_projection.md cite ONE chosen timestamp each (e.g. association_20260206_144409.json); these are the other runs | explicit (test_association.py mentions `explicit` 5x); not a sequence claim | 2026-02-05 | HISTORICAL |
| 3 | results/primitives ERP, binding, prediction suite: composed_erp*, binding_*, forward_prediction*, prediction_n400, incremental_erp, diagnose_erp_dynamics, garden_path_erp, p600_informativeness, cloze_probability, agreement_attraction, rc_asymmetry, recursive_structure, variable_* | 42 | primitives/composed_erp_20260222_003923_quick.json; primitives/binding_retrieval_20260221_232045_quick.json | primitives/test_composed_erp.py etc.; exist; not in runner | language | none found; diagnose_erp_dynamics appears only as a code-audit row (research/notes/substrate/w_alias_back_catalogue.md:53). RESULTS_composed_erp.md is cited (gates.py, docs/claim_audit.md) but names none of these files | no engine arg; not sequence | 2026-02-22 | HISTORICAL |
| 4 | results/primitives self-organization and curriculum suite: area_self_organization, role_*, self_org_exploration, developmental_curriculum, curriculum_ablation, integrated_learner, unsupervised_binding, grounded_language, online_word_acquisition, generalization, depth_generalization, semantic_similarity, sentence_generation, subgrammar_decomposition, vocabulary_scaling, grammaticality_judgment, parameter_robustness | 32 | primitives/integrated_learner_20260222_043703_quick.json | primitives/test_integrated_learner.py etc.; exist; not in runner | language | none found | no engine arg; not sequence | 2026-02-22 | HISTORICAL |
| 5 | results/applications/agreement_{number,violations,exploration}_* | 29 | applications/agreement_number_20260217_190403.json | applications/test_agreement_number.py:100-102 (writes results/applications); exists (2026-07-24); not in runner | language | none found | no engine arg; not sequence | 2026-02-17 | HISTORICAL |
| 6 | results/applications/n400_* (amplification, cloze, context, controls, diagnostic, graded, parameter_sweep, pre_kwta, vocab_scaling) | 22 | applications/n400_pre_kwta_20260211_124000.json | applications/test_n400_*.py; exist; not in runner. test_n400_pre_kwta.py:134 writes results/applications | language (N400 closed: research/notes/language/n400_saturated_because_prediction_has_no_signal.md) | research/claims/N400_GLOBAL_ENERGY.md cites the sibling n400_pre_kwta_20260211_131944.json (not an orphan); no note names the 22 | test_n400_pre_kwta.py:143,396 defaults to `numpy_sparse`; language claim, so the sequence void rule does not apply | 2026-02-11 | HISTORICAL (18); KEEP-AS-CITED (4 n400_pre_kwta_*: link beside the 131944 citation) |
| 7 | results/applications/p600_syntactic_* | 10 | applications/p600_syntactic_20260211_140623.json | applications/test_p600_syntactic.py; exists; not in runner | language | none found | no engine arg; not sequence | 2026-02-11 | HISTORICAL |
| 8 | results/applications parser phenomena: compositional_*, garden_path, relative_clauses, semantic_priming, cross_linguistic, developmental_*, function_word_discovery, l2_processing, language_syntax, learned_gating, morphological_agreement, multiclause_coordination, nemo_benchmark, parser_scaling_laws, structural_ambiguity | 29 | applications/garden_path_20260210_232020.json; applications/language_syntax_20251129_155811_quick.json | applications/test_<stem>.py for every stem; exist; not in runner | language | none found; docs/claim_audit.md cites the sibling crosslinguistic_typology_20260209_171109_quick.json only | no engine arg; not sequence | 2026-02-10 | HISTORICAL |
| 9 | results/stability/{assembly_distinctiveness,loop_switching,noise_robustness(_v2),phase_diagram,scaling_laws}_* | 18 | stability/phase_diagram_20251128_230642_quick.json | stability/test_phase_diagram.py:148 (writes results/stability) etc.; exist (edited 2026-09-11); not in runner | substrate (core questions Q01/Q03/Q20, research/core_questions/index.json:6 status `curated_question`) | RESULTS_phase_diagram.md etc. cite one chosen timestamp each (phase_diagram_20260206_160402.json ...); Q01/Q03/Q20 experiments.md cite the RESULTS files | explicit (`explicit` 4x in test_phase_diagram.py); not sequence | 2026-02-05 | HISTORICAL |
| 10 | results/biological_validation/* + results/information_theory/* | 5 | biological_validation/biological_parameters_20251128_231133_quick.json | biological_validation/test_biological_parameters.py:105, information_theory/test_coding_capacity.py:59; exist (2026-09-10); not in runner | substrate | none found | explicit; not sequence | 2026-02-05 | HISTORICAL |
| 11 | results/ root: freeform_rc_*, center_embedded_baseline_* | 6 | research/results/freeform_rc_20260222_221420_quick.json | primitives/test_freeform_rc.py, primitives/test_center_embedded_baseline.py; exist; not in runner; results_dir line not found by grep (files sit at the results root) | language | none found | no engine arg; not sequence | 2026-02-22 | HISTORICAL |
| 12 | results/dev_runs/* | 16 | dev_runs/compare_battery_v2.json; dev_runs/ablate_seed42.json | compare_training_paths.py:163-168 (compare_battery*.json/.csv), ablate_developmental.py:94-99 (ablate_*), examples/train_developmental.py:8 (--output-dir dev_runs); exist; not in runner. compare_paths_seed42.json, compare_dev_only_v2.json, dev_sentences_42.json: producer unknown (grep none) | language (developmental parser training) | none found | no engine arg; not sequence | 2026-07-24 | HISTORICAL (13); REVIEW (3 unknown-producer files) |
| 13 | results/sweeps/*.csv | 9 | sweeps/dynamics_n300.csv; sweeps/gpu.csv | sweep_dynamics.py:471-473 (default sweeps/dynamics.csv; `--output` names the file); the gpu/explore_perf*/_vis_smoke*/explore_checkpoint headers match its columns (seed,depth,holdout_pattern,wobbly_bootstrap,...,n400_cohens_d,p600_cohens_d); exists; not in runner | language (ERP dynamics sweep) | none found | no engine arg; not sequence | 2026-07-24 | HISTORICAL; mark `_vis_smoke*.csv`, `explore_perf*.csv`, `explore_checkpoint.csv` as smoke/perf, not evidence |
| 14 | experiments/recruitment/results_recruitment_shard{0..7}of8.json | 8 | recruitment/results_recruitment_shard0of8.json | recruitment/recruitment_mechanisms.py:312-337 (`--shard`); merged by recruitment/analyze.py:521-524; exists; not in runner | substrate (capacity follow-up; docstring :1-6) | REPORT.md:6 and PROSE.md:6 cite `results_recruitment_shard*.json` as a glob, so the literal audit misses them | `numpy_sparse` at recruitment_mechanisms.py:126; a lexicon-capacity claim, not sequence | 2026-07-24 | KEEP-AS-CITED (spell out the 8 names at REPORT.md:6) |
| 15 | experiments/mass_readout_gate_results_raw.json | 1 | (same) | mass_readout_gate.py:40-46 writes mass_readout_gate_results.json (tracked, cited); no `_raw` writer in the script (grep none) | categories (#151 readout campaign, research/notes/categories/calibration_is_not_information.md:3) | none names the raw file | no engine arg | 2026-08-09 | REVIEW (raw dump of a cited result; writer unknown) |
| 16 | experiments/seq_a1_fsm_parity_results.json, _p040_results.json | 2 | experiments/seq_a1_fsm_parity_results.json | seq_a1_fsm_parity.py; exists; runner `a1-fsm-parity` (runner.py:543) | sequence | theory.py:207 cites the SCRIPT for SEQ-FSM; PREREG_seq_a1_fsm_parity.md:167 cites the log only; research/notes/README.md:159-162 lists it among "the numpy studies these correct" | YES: `engine="numpy_sparse"` at :89 and :245, no materialization (grep `materializ` = 0) | 2026-08-10 | VOID (keep, mark: sampled-engine sequence numbers superseded by the hashed horizon run) |
| 17 | results/aligner/word_capacity_results{,_hashed,_scheduled}.json | 3 | aligner/word_capacity_results_hashed.json | word_capacity.py (legacy entry, PREREG_word_capacity.md:300); exists; superseded by runner `word-capacity` (runner.py:555) | aligner | PREREG_word_capacity.md:103 (Amendment 2, hashed) and :125 (Result 2026-09-03) describe these runs but link only word_capacity_results_scheduled_feat4000x100.json (:306); log research/results/logs/word_capacity_hashed.log | hashed (word_capacity.py `hashed` 12x); not sampled | 2026-09-09 | KEEP-AS-CITED |
| 18 | results/memory/capacity_scaling_results_{amend4_ref133, amend5_ctl_gated, amend5_g5, amend6_g6_gated133, amend6_s0.3/0.4/0.6, ctl_T16, ksweep_ctl}.json | 9 | memory/capacity_scaling_results_amend6_s0.3.json | seq_capacity_scaling.py; exists; runner `capacity-scaling` (runner.py:553) | memory | PREREG_refraction_memory.md reports each: Amendment 2 k sweep :185/:231 (ksweep_ctl), Amendment 4 result :378 (ref133), Amendment 5 result :433-440 (CTL gated, G5), Amendment 6 strength :539 (s0.3/0.4/0.6), gate :566 (g6_gated133), post hoc control T=16 :595 (ctl_T16); results/README.md:5-6 "one per registered run, tagged" | hashed batched brains (`torch` 19x); not sampled | 2026-09-09 | KEEP-AS-CITED |
| 19 | results/sequence/seq_s5_soft_census_results_hashed_amend{4_100,5_p8/p15/p30,6_s0.0/0.05/0.08/0.1,7_norm/p20/p24,8_p20_groups/p28}.json | 13 | sequence/seq_s5_soft_census_results_hashed_amend5_p15.json | seq_s5_soft_census_hashed.py; exists; runner `s5-soft-census` (runner.py:551) | sequence | PREREG_s5_cliff_anatomy.md:20-22 gives the commands; Addenda 4-8 results at :344, :413, :516, :581, :623, :632 report the numbers; figures_notes.py:119-121,129,214-217 reads them by f-string tag (invisible to the literal audit) | hashed explicit substrate (docstring :1-3); not sampled | 2026-09-09 | KEEP-AS-CITED |
| 20 | results/sequence/seq_a3_transducer_results_hashed_{s0.05,binomial_stimuli}.json | 2 | sequence/seq_a3_transducer_results_hashed_s0.05.json | seq_a3_transducer.py `--engine hashed`; exists; runner `a3-transducer` (runner.py:559) | sequence (transducer, closed null) | PREREG_seq_a3_transducer.md:232 (Amendment 2 result, strength 0.05) and :196 ("The Binomial-stimuli run's file is kept") name them in prose, no link | hashed; not sampled | 2026-09-09 | KEEP-AS-CITED (s0.05); VOID-kept (binomial_stimuli: selector defect 1b475fc, already labelled at :196) |
| 21 | results/runs/memory.* and mechanism.per-fiber-plasticity: run.json (19) + comparison-v2/v3.json (2) | 21 | runs/memory.capacity-scaling/anchor-pair-20260912/run.json; runs/memory.capacity-scaling/capacity-record-consumed-20260910/comparison-v3.json | historical_*.py, context_noise.py, seq_capacity_scaling.py, per_fiber_plasticity.py; all in runner (runner.py:535-541,552-553). run.json is the provenance record (runner.py:487; fields engine, git_commit, registration_sha256, script_sha256, seeds, source_archive) | memory | every sibling results.json is cited by its PREREG (audit resolved_edges); PREREG_refraction_memory.md:647,:656 link comparison.json / comparison-v2.json but not the later comparison-v3.json and migration-capacity comparison-v2.json | historical-* smoke runs are VOID API checks by design (research/experiments/README.md:16-21); context-noise numpy arm is numpy_sparse (context_noise.py:94) but a memory claim | 2026-09-10 to 09-12 | KEEP-AS-CITED (add run.json and the newest comparison receipt beside each results.json link) |
| 22 | results/runs/sequence.*: run.json (15) + comparison-v2/v3.json (2) | 17 | runs/sequence.a3-oracle-ceiling/oracle-study4-20260912/run.json; runs/sequence.a1-horizon/horizon-record-consumed-20260910/comparison-v3.json | seq_a1_horizon_hashed.py, seq_a1_learning_null.py, seq_a1_exactness_sweep.py, seq_a1_local_regime.py, seq_a2_refraction_load.py, seq_a3_oracle_ceiling.py, seq_a3_transducer.py, seq_arc_refraction_reference.py, seq_temporal_positions.py; all in runner (runner.py:542-559) | sequence | every sibling results.json cited (DESIGN_sequence_port.md, PREREG_sampler_audit.md, PREREG_agreement_corpus.md, PREREG_temporal_memory.md, PREREG_temporal_positions.md, PREREG_arc_refraction_reference.md, PREREG_a1_learning_null.md); DESIGN_sequence_port.md:237 links comparison-v2 of horizon-record-consumed, not comparison-v3 | no: sens-*-v3 runs are `--materialized` (README.md:26-28); horizon/temporal are hashed; oracle is `computed_baseline`; arc-ref is `reference_nemo_numpy` | 2026-09-10 to 09-12 | KEEP-AS-CITED |
| 23 | results/runs/substrate.kwta-tie-fragility/{kwta-smoke,kwta-study}-20260912/run.json | 2 | runs/substrate.kwta-tie-fragility/kwta-study-20260912/run.json | kwta_tie_fragility.py; exists; runner (runner.py:558) | substrate | PREREG_kwta_tie_fragility.md cites both results.json | explicit engine (README.md:33) | 2026-09-12 | KEEP-AS-CITED |
| 24 | results/runs/aligner.word-capacity{,-ladder}/*/run.json | 3 | runs/aligner.word-capacity/word-capacity-cell-a-schema8-replay-20260911/run.json | word_capacity_run.py, word_capacity_ladder_run.py; exist; runner (runner.py:555-556) | aligner | PREREG_word_capacity.md:303,:318,:331 cite the three results.json | hashed/scheduled aligner; not sampled | 2026-09-11 | KEEP-AS-CITED |

Count check: 47+28+42+32+29+22+10+29+18+5+6+16+9+8+1+2+3+9+13+2+21+17+2+3 = 374.

## Summary

| Disposition | Files | Clusters |
|---|---|---|
| HISTORICAL | 286 | 1, 2, 3, 4, 5, 6 (18 of 22), 7, 8, 9, 10, 11, 12 (13 of 16), 13 |
| KEEP-AS-CITED | 81 | 6 (4 n400_pre_kwta), 14, 17, 18, 19, 20 (s0.05), 21, 22, 23, 24 |
| VOID (kept, marked) | 3 | 16 (2), 20 (binomial_stimuli) |
| REVIEW | 4 | 12 (3 unknown-producer dev_runs files), 15 (mass_readout raw) |

Largest clusters: 1 freeform primitives (47), 3 ERP/binding primitives (42),
4 self-organization primitives (32), 5 agreement applications (29), 8 parser
phenomena applications (29). All five are the February 2026 language line
(no registration, no register edge: theory.py has zero refs under
results/primitives, results/applications) and should be marked historical in
research/results/README.md, which today says only that older folders "belong
to their own studies" (README.md:11-12).

Void reasoning. Only cluster 16 is a sequence claim produced on the sampled
numpy engine without materialization (seq_a1_fsm_parity.py:89,245; grep
`materializ` = 0), so PREREG_sampler_audit.md applies; the register keeps
SEQ-FSM as PROVED from the paper (theory.py:196-210), so the script stays as
cited evidence and the two JSONs are marked void, not removed. Every other
sequence artifact is hashed, materialized, computed, or the vendored
reference (clusters 19, 20, 22). The numpy_sparse uses in clusters 6, 14 and
21 are language, capacity and memory claims, outside the void rule.

## Cheapest fixes: files that should already be linked from an existing note

1. PREREG_refraction_memory.md, beside the Amendment results at :185/:231,
   :378, :433-440, :539, :566, :595: link the 9 `capacity_scaling_results_*`
   files of cluster 18 (each tag is named in the results README convention).
2. PREREG_s5_cliff_anatomy.md Addenda 4-8 results (:344, :413, :516, :581,
   :623, :632): link the 13 `seq_s5_soft_census_results_hashed_amend*` files;
   note figures_notes.py:119-121 as their consumer.
3. PREREG_word_capacity.md :103 and :125: link `word_capacity_results.json`,
   `_hashed.json`, `_scheduled.json` (the F1/F2 sweep of 2026-09-03) and the
   log research/results/logs/word_capacity_hashed.log.
4. PREREG_seq_a3_transducer.md :232: link `..._hashed_s0.05.json`; :196: turn
   the prose mention of `..._hashed_binomial_stimuli.json` into a link with
   the existing "selector defect, kept" label.
5. research/experiments/recruitment/REPORT.md:6 (and PROSE.md:6): replace the
   `shard*.json` glob with the eight literal names.
6. Registration link lists for every runs/ folder (e.g. PREREG_refraction_memory.md
   :646-656, DESIGN_sequence_port.md:237, PREREG_temporal_positions.md,
   PREREG_historical_*_migration.md): add `run.json` beside each `results.json`,
   and the newest comparison receipt (capacity-record-consumed comparison-v3,
   migration-capacity comparison-v2, horizon-record-consumed comparison-v3,
   migration-a1 comparison-v2). 43 files, one mechanical edit per note.
7. PREREG_seq_a1_fsm_parity.md:167: link `seq_a1_fsm_parity_results.json` and
   `_p040_results.json` beside the log, labelled sampled-engine / superseded.
8. research/claims/N400_GLOBAL_ENERGY.md: list the four `n400_pre_kwta_*`
   reruns beside the cited 131944 run.
9. RESULTS_*.md in results/primitives and results/stability: one line each
   naming the sibling timestamps as earlier or repeat runs (clusters 2, 9).

## Open questions for REVIEW rows

- dev_runs/compare_paths_seed42.json, compare_dev_only_v2.json, dev_sentences_42.json:
  no tracked script names them (grep over examples/, research/, neural_assemblies/).
- experiments/mass_readout_gate_results_raw.json: mass_readout_gate.py writes only
  the non-raw file (:40-46); the raw dump's writer is not in the tree.
- primitives/run_all.py, the launcher the static-debt review names at
  STATIC_DEBT_DISPOSITIONS.md:44-48, is absent from `git ls-files research/experiments`;
  the review already records both grids failing at the producer.

## Applied (2026-09-13)

`candidate_orphan_results` fell from **374 to 297**: 77 files gained a literal
incoming reference from the note, claim or report that already described the
run in prose. Nothing was deleted, moved or rewritten; every added link points
at a tracked file and was checked to resolve.

Applied: fixes 1 (nine `capacity_scaling_results_*` tags at the Amendment
results of `PREREG_refraction_memory.md`), 2 (thirteen
`seq_s5_soft_census_results_hashed_amend*` tags at Addenda 4-8 of
`PREREG_s5_cliff_anatomy.md`, with `figures_notes.py` named as their
by-tag consumer), 3 (the three aligner `word_capacity_results*` files and the
hashed log at `PREREG_word_capacity.md`), 4 (the s = 0.05 and Binomial-stimuli
transducer files at `PREREG_seq_a3_transducer.md`, the second carrying its
existing selector-defect label), 5 (the eight recruitment shards spelled out
in `REPORT.md` and `PROSE.md` -- `PROSE.md` is the hand-written source and
`REPORT.md` is generated from it by `analyze.py`, so both carry the same text
and a regeneration is a no-op), 7 (the two `seq_a1_fsm_parity` result files at
their registration, labelled sampled-engine and superseded by the hashed
horizon run), 8 (the four `n400_pre_kwta` reruns beside the cited run in
`research/claims/N400_GLOBAL_ENERGY.md`) and 9 (one line per `RESULTS_*.md`
under `results/primitives` and `results/stability` naming the sibling
timestamps as earlier or repeat runs).

`research/results/README.md` gains a "Historical folders" section marking the
February 2026 language and substrate lines as kept history that no
registration cites, and naming the three smoke/performance CSV families under
`sweeps/` as probes rather than evidence.

**Deferred: fix 6** (adding `run.json` and the newest comparison receipt beside
each `results.json` link in the runner registrations, 43 files). It is
mechanical but touches registrations that were being edited concurrently; the
remaining 297 orphans are dominated by those `run.json` provenance records and
by the historical clusters, both of which are inventory signals rather than
unlinked evidence.
