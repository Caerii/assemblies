# Static-debt clusters: dispositions (2026-09-12)

> Drafted by a read-only pass at commit 6754199d over the eight files the backlog names under 'Repair or explicitly retire the remaining static-debt clusters', with Pyright run per file. Each entry records retained role, consumers (grep hits), contract or evidence link, owner line, Pyright profile and a decision (MIGRATE / RETAIN-AS-DIAGNOSTIC / ARCHIVE) with its justification. Decisions are carried out one reviewable commit at a time; the first is the two hashed parity scripts, whose import has been broken since 637593ee and whose replacement test already runs.

# Static-debt dispositions: the eight named experiment files

Worktree read: C:\Users\locke\AppData\Local\Temp\assemblies-astra-audit-20260909 (audit/astra-codebase-20260909), 2026-09-12. Backlog item: TODO.md:184-191.
Gate: scripts/verify_maintained.py:20-38 (MAINTAINED_SCOPES + MAINTAINED_FILES); no research/experiments file is in it. Runner registry: research/runner.py:535-558 (EXPERIMENTS); migrated studies are tabled in research/experiments/README.md:13-40.
Owner lines are the rows of research/notes/README.md:45-58 (memory, sequence, transducer, aligner, substrate). The notes tree also holds categories/, coin/ and language/ directories the index does not list; files whose only note lives there are marked "unassigned" with the directory named.
Line numbers are from the working copy, which carries pre-existing uncommitted edits (not made by this audit): research/experiments/README.md has nine lines inserted after L56, so README citations at or beyond L57 sit nine lines lower at HEAD, and research/runner.py has one extra EXPERIMENTS entry appended at L558-559.
Pyright: `uv run pyright <file>` in the worktree today. Seven counts match TODO.md:186-188; gpu_hashed_deviations_prototype.py (23) is not in the TODO list. The "X is not a known attribute of module torch" class is file-local, not a stub gap: neural_assemblies/core/torch_engine/_engine.py does a top-level `import torch` (L25) and reports 0 errors; the cause inside the scripts was not diagnosed here.

## 1. research/experiments/gpu_writeback_gemm_prototype.py (235 lines)
- Retained role: Times plasticity write-back computed as a batched 0/1 GEMM per (brain, item) block against the committed append+compact key store, on identical winner streams in one process (docstring L1-35; `block_from_window` L93, `run_baseline`/`run_gemm` L165-190). `check_exact` (L111) aborts main() unless both paths agree cell for cell; the script prints a speed table and writes no file.
- Consumers: none in code. Named by neural_assemblies/theory.py:987 and docs/register.md:508 (evidence), research/notes/substrate/DESIGN_gpu_hashed_drive.md:250 (Amendment 3), TODO.md:186.
- Contract or evidence link: register entry HEBB-OUTER-PRODUCT (theory.py:971-997, PROVED) lists this file as its only evidence. No [[ID]] inside the file; no PREREG; no results file.
- Owner: substrate.
- Pyright: 40 errors, all reportAttributeAccessIssue on torch attributes (`arange`/`int64` L52, `cat` L60, `zeros`, `randint`, ...).
- Decision: RETAIN-AS-DIAGNOSTIC. It is the register's evidence for a PROVED identity and the DESIGN note's Amendment 3, yet it is a one-shot timing prototype with an internal exactness gate rather than a seeded study, so the runner has nothing to record; document it as "prototype, prints only" and keep it out of the gate. Single commit fbe84828 (2026-08-25).

## 2. research/experiments/gpu_hashed_deviations_prototype.py (267 lines)
- Retained role: Builds a private inline CUDA extension `na_drive_full2` (`build` L119-120 via `load_inline`) that adds the full deviation-store correction (col, count, eff_bit, raw_bit) with a host-replayed chain table to the hashed base drive (docstring L1-22; `chain_table` L125, `store_from_vw` L137). main() checks the GPU drive against CPU `VirtualWeights.row_sum` on brains trained through research/experiments/_substrate_arms.py (sys.path hack + import at L200-202, ASSEMBLIES_VIRTUAL_WEIGHTS=1), then prints a cost table; writes no file.
- Consumers: none in code. Named by theory.py:1016, docs/register.md:525, DESIGN_gpu_hashed_drive.md:124 (Amendment 1). Not in the TODO list.
- Contract or evidence link: register entry DRIVE-SPLIT (theory.py:998-1030, PROVED) lists it as sole evidence. No [[ID]]; no PREREG; no results file.
- Owner: substrate.
- Pyright: 23 errors, all reportAttributeAccessIssue: torch attributes (L220 ...), `numpy.random.seed` (L207), and "Cannot access attribute drive_full for class str" x3 (L219: `build()`'s `load_inline` return is typed str).
- Decision: RETAIN-AS-DIAGNOSTIC, same grounds as file 1. Record two cautions in its docstring: its kernel is not the shipped `_fused_cuda`, so it is not an engine parity test, and it depends on _substrate_arms.py (9 Pyright errors of its own). Single commit 286cd27b (2026-08-25).

## 3. research/experiments/gpu_hashed_stim_parity.py (140 lines)
- Retained role: Captures a numpy_sparse winner trajectory on a materialized area with one stimulus (`engine_trace` L38), injects the engine's own stimulus base, replays the trajectory through the `_fused_cuda` kernels (`replay` L64) and prints the per-round relative drive error for five arms (`arm` L112; main L131-137). The question is whether stimulus PRICING (the `tgt.n` divisor and the `w_max*stim_size*p` cap) is right under norm_init.
- Consumers: none found (only TODO.md:187 and docs/reviews/whole-codebase/inventory.json).
- Contract or evidence link: none; no [[ID]]; no PREREG or DESIGN names it; no results file. Superseded: neural_assemblies/tests/test_hashed_substrate_parity.py::test_stimulus_pricing_reproduces_numpy_sparse (L215-228) is the same method, the same injected base, the same cases (1024,30,0.1) and (2048,50,0.5), under pytest, and feeds research/results/substrate/parity_errors.json through neural_assemblies/tests/_parity_dump.py (L1-9).
- Owner: substrate.
- Pyright: 26 errors: 22 reportAttributeAccessIssue (torch attributes; and L28 `_chain_table` "unknown import symbol"), 3 reportOptionalMemberAccess, 1 reportArgumentType. L28 is real: `_chain_table` left torch_engine/_batched.py in 637593ee (2026-08-25) and lives at torch_engine/_hashed.py:47, so the script raises ImportError before main().
- Decision: ARCHIVE to legacy/experiments/ with a README line naming the maintained test above; unrunnable since 2026-08-25, replaced, and uncited. Last commit 4e708ff6 (2026-09-09, path fix only).

## 4. research/experiments/gpu_hashed_substrate_parity.py (130 lines)
- Retained role: Same replay method for the recurrent fiber without a stimulus: engine trajectory captured (`engine_trace` L38), replayed through `hashed_drive`/`dev_correct`/`hashed_indegree`/`column_mass` (`replay` L62), max and relative drive error printed for arms NONE, B (norm_init), C (scaling), G (both) at two densities (`arm` L99; main L118-127).
- Consumers: none found (TODO.md:187, inventory.json only).
- Contract or evidence link: none; no [[ID]]; no PREREG/DESIGN; no results file. Superseded by neural_assemblies/tests/test_hashed_substrate_parity.py (docstring L1-22; replay at L90-98; same four arms and cases), added in a8a7e875 (2026-08-25).
- Owner: substrate.
- Pyright: 26 errors: 22 reportAttributeAccessIssue (torch attributes; L27 `_chain_table` unknown import symbol, the same ImportError as file 3), 4 reportOptionalMemberAccess.
- Decision: ARCHIVE alongside file 3 in one change, same README line; identical grounds. Last commit 4e708ff6 (2026-09-09).

## 5. research/experiments/primitives/run_all.py (184 lines)
- Retained role: Aggregate launcher for the projection, association and merge producers with a quick grid (L24-110) and a full grid (L113-170), saving through `save_result(..., "_quick"|"_full")` (base.py:145-172, filename `<name>_<timestamp><suffix>.json` under research/results/primitives). All six `run()` calls pass `n_neurons_range`/`k_active_range`/`p_connect_range`/`beta_range`/`n_trials` (L41-45, 55, 69, 127, 141, 155), which none of the producers accept (test_projection.py:252-257 `run(self, n, k, p, beta, ...)`; test_association.py:170; test_merge.py:185).
- Consumers: research/registry.json:55 still lists `run_all.py --quick` as a primitives entry point; docs/reviews/whole-codebase/VALIDATION.md:2264, CLAUDE_HANDOFF.md:1313 and research/experiments/README.md:86-89 record that both paths now fail at the producer boundary. No .py imports it.
- Contract or evidence link: none; no [[ID]]; no PREREG. research/results/primitives/association_binding_*_quick.json (2025-11-28) and *_full.json (2026-02-07) come from an earlier producer name; theory.py cites nothing under primitives/.
- Owner: unassigned (registry.json "primitives" line; no notes-index row).
- Pyright: 30 errors, all reportCallIssue "No parameter named n_neurons_range/k_active_range/p_connect_range/..." (L41-43 onward): the static form of the runtime refusal.
- Decision: ARCHIVE; a pre-runner aggregate whose every call is rejected, while projection/association/merge already run under the runner as historical-projection/-association/-merge (runner.py:535-539; README:16-20). Remove the registry.json:55 entry point in the same change so nothing points into legacy/. Last substantive commit efd21d94 (2026-02-10).

## 6. research/experiments/primitives/test_unified_phenomena.py (485 lines)
- Retained role: Trains one language brain on a RecursiveCFG corpus (SVO, PP, SRC, ORC) and runs nine ERP/binding suites, reporting Cohen's d per phenomenon plus H1 (most d>0.5) and H2 (object-position d>1.0) (docstring L1-33; `UnifiedConfig` L65, `UnifiedPhenomenaExperiment` L315, `run` L328). Saves under name "unified_phenomena" (L321, L465) into research/results/primitives.
- Consumers: none in code. neural_assemblies/tests/methodology_baseline.json:100 pins its 22 hand-rolled statistic sites (test_methodology_ratchet.py:52-55: lower when fixed; raising needs a comment). TODO.md:187.
- Contract or evidence link: none; no [[ID]]; no PREREG/DESIGN. One result, research/results/primitives/unified_phenomena_20260222_030824_quick.json (H1 True, H2 True, n_surviving 5 of 6; a --quick run at n=5000, 5 seeds), uncited by theory.py.
- Owner: unassigned (ERP/parser work; research/notes/language/ exists but the index has no language row).
- Pyright: 30 errors, all reportArgumentType: 19x float stored into a Dict[str, int] (L145, L146, L158, ...), 5x uint32 NDArray against a "predictions" parameter, 5x ndarray against a "lexicon" parameter. Annotation mismatches, cheap to fix.
- Decision: RETAIN-AS-DIAGNOSTIC; a real multi-suite probe with a saved quick result but no registration, no register citation and no replacement, so migration would need a PREREG first and archival would discard the only coexistence measurement. Document as unregistered/post hoc; keep out of the gate. Single commit c529bc92 (2026-02-22).

## 7. research/experiments/run_all_experiments.py (186 lines)
- Retained role: Self-described "Legacy aggregate launcher (not a registered validation protocol)" (L1-2). It keeps the eight historical quick calls as data (`QUICK_EXPERIMENTS` L49), `validate_suite` (L95) refuses the six producer-signature mismatches before any constructor runs, `--full` errors out (L175-181), and `generate_summary` (L135) stamps every quick run scientific_status VOID before `write_new_document` writes master_summary_<ts>.json (L129-131).
- Consumers: neural_assemblies/tests/test_legacy_aggregate_summary.py:5,46,60,71,88 imports generate_summary, print_summary, QUICK_EXPERIMENTS and validate_suite (a live pytest consumer); docs/reviews/whole-codebase/SEMANTIC_CARDS.md:1323-1364; research/experiments/README.md:79-84; TODO.md:188.
- Contract or evidence link: SEMANTIC_CARDS.md#legacy-aggregate-summary-2026-09-10 (cited in its own docstring, L136) and #legacy-experiment-configuration. No [[ID]]; no PREREG. Result: research/results/master_summary_20251128_231237.json (pre-contract run).
- Owner: unassigned (a legacy aggregate over memory-line producers).
- Pyright: 24 errors, all reportArgumentType at L125 `experiment.run(**parameters)`: list[int] passed to int/float parameters, i.e. the six preserved mismatches the module refuses at runtime by design.
- Decision: RETAIN-AS-DIAGNOSTIC; a tested, contract-bearing refusal harness rather than a study, whose remaining errors are the inventory it deliberately preserves. Keep out of the gate; its count falls only when those six calls are migrated (README:79-84). Last commits 177dbbc0 and 9a3537c0 (2026-09-10).

## 8. research/experiments/surprise_gain_recall.py (211 lines)
- Retained role: E2 of the frequency-imbalance program (task #131): paired arms OFF/GAIN/SCALED/BOTH x seeds 42-46 at n=3000 test whether label-free novelty gain, gain = min(GAIN_MAX, sqrt(mean_count/count)), composes with scoped synaptic scaling on tense/number recall, with predictions R1-R5 and a decision rule pre-registered in the docstring (L1-65). Trains through the shared curriculum and role guards (imports L133, L141), caches parsers via `_parallel.cached_parser` (L146), and writes surprise_gain_recall_results.json beside itself (L94-95, L190).
- Consumers: none in code (VALIDATION.md:5107 records the role_guards refactor). research/notes/categories/gain_and_scaling_compose.md:3 names it as the experiment behind that note.
- Contract or evidence link: the in-file preregistration (L48-65) and its result record gain_and_scaling_compose.md (R1 refuted by 0.010, R2 refuted as registered, R3-R5 confirmed; decision rule: sweep GAIN_MAX and add seeds). No [[ID]]; theory.py does not cite it. Results: research/experiments/surprise_gain_recall_results.json (made immutable by 2da6fe4c, 2026-09-11) and research/results/logs/surprise_gain_recall.log.
- Owner: unassigned (its note lives in research/notes/categories/, not an index line; language-adjacent).
- Pyright: 24 errors: 21 reportArgumentType because `kwargs = dict(n=..., vocabulary=...)` is inferred as dict[str, int | Dict[str, GroundingContext]], so the L121/L123 assignments and `EmergentParser(**kwargs)` at L124 mismatch; 3 reportAttributeAccessIssue. One annotation (`kwargs: dict[str, Any]`) clears most.
- Decision: MIGRATE to research/runner.py under a new command (e.g. `surprise-gain-recall` -> research.experiments.surprise_gain_recall), running the E2 preregistration held in its docstring with gain_and_scaling_compose.md as the registered result. It is live (the registered decision rule still owes the GAIN_MAX sweep and more seeds), and the runner supplies the seed identities and tags it now improvises; it does not yet call run_experiment (grep: none).

## Summary
- MIGRATE 1 (surprise_gain_recall.py); RETAIN-AS-DIAGNOSTIC 4 (gpu_writeback_gemm_prototype.py, gpu_hashed_deviations_prototype.py, primitives/test_unified_phenomena.py, run_all_experiments.py); ARCHIVE 3 (gpu_hashed_stim_parity.py, gpu_hashed_substrate_parity.py, primitives/run_all.py).
- Start with gpu_hashed_substrate_parity.py, taking its stimulus twin in the same commit: both have raised ImportError since 637593ee (2026-08-25), their exact replacement already runs in neural_assemblies/tests/test_hashed_substrate_parity.py, and nothing cites them, so archiving removes 52 errors with no scientific decision to make.
- None of the above adds research/experiments/ to the maintained gate; the two register-cited prototypes and run_all_experiments.py keep their counts until the named causes are addressed.
