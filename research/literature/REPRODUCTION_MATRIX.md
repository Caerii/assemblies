# Literature Reproduction Matrix

**Goal:** frictionless reproducibility (Donoho) for every Assembly Calculus paper — each
claim maps to a **protocol**, a **test or script**, and (when empirical) a **golden
artifact**. This matrix is the master checklist; close a row by pinning the protocol,
not by hand-waving.

**Companion files**

| File | Role |
|------|------|
| [reproduction_matrix.json](reproduction_matrix.json) | Base claim rows (CI-parity focused) |
| [reproduction_matrix_supplement.json](reproduction_matrix_supplement.json) | Pass-2 audit: configs, legacy, extended empirical |
| [build_matrix_supplement.py](build_matrix_supplement.py) | Regenerate supplement from `SUPPLEMENT` table |
| [parity/configs/](parity/configs/) | **Per-paper YAML param registries** (all hyperparams) |
| [parity/PROTOCOLS.md](parity/PROTOCOLS.md) | Pinned parameters for passing protocols |
| [index.json](index.json) | Paper inventory + `implementation_status` |
| [../../docs/scientific_status.md](../../docs/scientific_status.md) | What the package may legally claim |

---

## Donoho contract (what “done” means)

For each **empirical** or **mechanistic** claim row:

| Requirement | Artifact |
|-------------|----------|
| **Protocol** | Row in `parity/PROTOCOLS.md` or paper-specific `parity/<paper_id>.yaml` |
| **Pinned inputs** | `seed`, `n`, `k`, `p`, `beta`, `rounds`, corpus slice, hardware note |
| **Golden or bound** | JSON under `parity/golden/` **or** statistical tolerance vs reference |
| **One command** | `uv run pytest …` or `uv run python research/literature/reproduce.py --claim …` |
| **Manifest** | `claim_id`, git hash, engine, duration (future: `reproduce.py` emits) |

**Status legend (per claim row)**

| Status | Meaning |
|--------|---------|
| `pinned` | Protocol + test + golden/bound; CI green |
| `partial` | Mechanism exists; protocol incomplete or toy-scale only |
| `missing` | No maintained protocol |
| `theorem` | Formal result — cite paper; optional simulation sanity only |
| `research` | Exploratory code under `research/`; not a package guarantee |

**Progress (2026-06-23, pass 2):** 16 papers · **204 claim rows** (95 base + 109 supplement) · **40 pinned** · **87 partial** · **55 missing** · **5 theorem** · **12 research** · **5 legacy** · **37 config rows**

Pass 2 adds **every named hyperparameter, config regime, legacy path, and extended empirical claim** found in code, tests, and paper params — including items not yet pinned in CI.

---

## Config registry (pass 2)

Every paper’s **full parameter surface** lives in YAML under [parity/configs/](parity/configs/):

| Config file | What it pins |
|-------------|--------------|
| `pnas2020.yaml` | Paper `n=10⁴,k=100,p=0.01,β=0.05` vs CI `n=5000,k=80`; associate/merge/pattern_complete thresholds; legacy `n=10⁵,k=317` |
| `itcs2019.yaml` | ITCS/CCNeuro; append gap; density simulator sweeps |
| `colt2022.yaml` | `learn_assembly` convergence 0.90 vs protocol 0.85; MNIST hierarchical legacy arch |
| `sequences2025.yaml` | LRI defaults, TM FSM table, ordered_recall thresholds, TM vs parity param split |
| `tacl2021.yaml` | English/Russian `p, lex_k, betas, areas`; all pinned sentences + untested lexicon lemmas |
| `naloma2022.yaml` | Depth-1/2 patterns, WM replay steps, SUBJ overwrite bug |
| `aaai2022.yaml` | BlocksWorldAC `n,k,β,rounds`; FSM pick/put; STRIPS problems 2–4 block |
| `organ2023.yaml` | 45-area graph; toy NemoParser vs emergent organ |
| `coin2024.yaml` | CoinFlipModel defaults; k-split vs softmax note; demo URL gap |
| `nemo2025.yaml` | Full `_STAGE_CONFIG`, stage reps, accuracy thresholds, brain regimes |
| `hoff2026.yaml` | Paper `ε, p_i, ω_inh, τ_m` vs repo `EPercentPolicy` mapping |
| `direct2026.yaml` | `adaptive_soft` schedule; Alzheimer 12-edge SCM; Pearl battery |
| `ting2026.yaml` | TIMIT 39-phone 47.5%; mel/MFCC bins; boundary F1 targets |
| `onasch2025.yaml` | Proposed dendritic gating keys (paper supplement TBD) |

Config claims use category **`config`** in the supplement matrix (`*-C01` rows).

---

## Known param mismatches (must reconcile for 100% parity)

| Area | Paper / legacy | CI parity today | Config row |
|------|----------------|-----------------|------------|
| Brain size | PNAS `n=10000,k=100,p=0.01,β=0.05` | `n=5000,k=80,p=0.05,β=0.1` | `PNAS20-C01`, `ITCS19-C01` |
| COLT convergence | API default 0.90 | Protocol test 0.85 | `COLT22-C01` |
| TM demo | `k=50,β=0.08,rounds=8` | Global `k=80,β=0.1,rounds=10` | `SEQ25-C01` |
| TACL parser | `non_LEX_n=1000` (code default) | PROTOCOLS implied 5000 | `TACL21-C01` |
| NEMO brain | Full tests `n=10000,k=100` | Parity `n=5000,k=80` | `NEMO25-C05` |
| Coin flip | Paper demo (unknown) | k-split not softmax | `COIN24-M04`, `COIN24-C03` |
| Hoff E%-WTA | Continuous E/I, `ε=0.1` | Discrete projection + `fraction_of_max≈0.35` | `EPWTA26-C01` |
| DIRECT binding | `adaptive_soft` β schedule | Fixed round counts | `DIR26-C01` |

---

## Summary by paper (merged matrix)

| Paper | ID | Pinned | Partial | Missing / theorem | Target |
|-------|-----|--------|---------|-------------------|--------|
| ITCS 2019 | `papadimitriou2019random` | 0 | 3 | 4 | `partial` → mechanism pins |
| CCNeuro 2019 | `papadimitriou2019ccneuro` | 0 | 2 | 1 | cite ITCS |
| PNAS 2020 | `papadimitriou2020brain` | 8 | 2 | 1 | **`implemented`** |
| COLT 2022 | `dabagia2022classify` | 2 | 2 | 2 | MNIST protocol |
| Sequences 24/25 | `dabagia2025sequences` | 4 | 5 | 3 | full TM + long recall |
| TACL 2021 | `mitropolsky2021parser` | 3 | 6 | 2 | full benchmark suite |
| NALOMA 2022 | `mitropolsky2022center` | 1 | 3 | 2 | depth-2 + SUBJ fix |
| AAAI 2022 | `damore2022planning` | 4 | 2 | 2 | full AC program |
| Language organ 2023 | `mitropolsky2023architecture` | 0 | 5 | 3 | unify `nemo/` |
| Coin-flipping 2024 | `dabagia2024coinflipping` | 4 | 2 | 2 | demo URL golden |
| Lang acquisition 2025 | `mitropolsky2025simulated` | 4 | 6 | 4 | paper-scale curriculum |
| Dendritic 2025 | `onasch2025dendritic` | 0 | 0 | 5 | `not_started` |
| E%-WTA 2026 | `hoff2026epwta` | 2 | 3 | 2 | E/I dynamics golden |
| Speech 2026 | `ting2026speech` | 0 | 0 | 5 | encoder + TIMIT slice |
| DIRECT 2026 | `kopadi2026causal` | 2 | 2 | 3 | full do-calculus battery |

---

## Tier 1 — Foundations

### Papadimitriou & Vempala (2019) — ITCS · `papadimitriou2019random`

| ID | Category | Claim / result | Module | Test / protocol | Status | Gap → protocol |
|----|----------|----------------|--------|-----------------|--------|----------------|
| ITCS19-T01 | theorem | Random projection + Hebbian plasticity yields assembly convergence (main ITCS theorem) | `compute/`, `ops.project` | — | theorem | Optional: convergence-rate sweep vs `n` (research) |
| ITCS19-M01 | mechanism | `project` stabilizes stimulus assembly in target area | `ops.py` | `test_assembly_calculus`, `test_literature_parity::test_project_stability_seed42` | pinned | Extend to paper `n=10⁴` golden |
| ITCS19-M02 | mechanism | `associate` binds across areas | `ops.py` | `test_assembly_calculus::test_associate_creates_shared_response` | partial | Pin associate overlap golden (ref repo) |
| ITCS19-M03 | mechanism | `merge` combines two source assemblies | `ops.py` | `test_literature_parity::test_merge_responds_above_chance` | pinned | — |
| ITCS19-M04 | mechanism | `append` chains projections (ITCS primitive) | — | — | missing | Add `append()` API + parity vs ref |
| ITCS19-E01 | empirical | Capacity / overlap scaling vs `n`, `k` | `simulation/` | — | missing | `parity/itcs2019_capacity.yaml` + Julia/ref compare |

### Papadimitriou (2019) — CCNeuro · `papadimitriou2019ccneuro`

| ID | Category | Claim / result | Module | Test / protocol | Status | Gap |
|----|----------|----------------|--------|-----------------|--------|-----|
| CCN19-T01 | theorem | RP&C as brain computation primitive | — | — | theorem | Cite ITCS 2019 |
| CCN19-M01 | mechanism | project / associate / merge as calculus | `assembly_calculus/` | PNAS parity tests (inherited) | partial | Explicit CCNeuro doc cross-ref only |

### Papadimitriou et al. (2020) — PNAS · `papadimitriou2020brain`

| ID | Category | Claim / result | Module | Test / protocol | Status | Gap |
|----|----------|----------------|--------|-----------------|--------|-----|
| PNAS20-M01 | mechanism | Project persistence across rounds | `ops.project` | `test_literature_parity::test_project_stability_seed42` | pinned | — |
| PNAS20-M02 | mechanism | Separate: distinct stimuli → low overlap | `ops.separate` | `test_literature_parity::test_separate_near_chance_seed42` | pinned | — |
| PNAS20-M03 | mechanism | Merge above chance | `ops.merge` | `test_literature_parity::test_merge_responds_above_chance` | pinned | — |
| PNAS20-M04 | mechanism | Reciprocal project | `ops.reciprocal_project` | `test_assembly_calculus` | partial | Golden vs ref |
| PNAS20-M05 | mechanism | Pattern completion from partial cue | `ops.pattern_complete` | `test_assembly_calculus` | partial | Golden vs ref |
| PNAS20-M06 | mechanism | Associate cross-area | `ops.associate` | `test_assembly_calculus` | partial | Golden vs ref |
| PNAS20-E01 | empirical | Cross-repo overlap metrics (persistence, separate) | `ops.py` | `test_cross_repo_parity` + `reference_pnas_golden.json` ±0.02 | pinned | — |
| PNAS20-E02 | empirical | Paper-scale `n`, `k`, `p` sweeps | `simulation/` | — | missing | `parity/pnas2020_scaling.yaml` |
| PNAS20-D01 | demo | Language production sketch (Fig. language circuit) | `language/`, fibers | — | missing | Port ref `word_order` protocol |

### Dabagia et al. (2022) — COLT · `dabagia2022classify`

| ID | Category | Claim / result | Module | Test / protocol | Status | Gap |
|----|----------|----------------|--------|-----------------|--------|-----|
| COLT22-T01 | theorem | Online learnability of well-separated classes | — | — | theorem | — |
| COLT22-M01 | mechanism | `learn_assembly` converges within epochs | `ops.learn_assembly` | `test_literature_parity::test_learn_assembly_converges` | pinned | — |
| COLT22-E01 | empirical | 2-class separable toy classification | `ops`, `programs/learn.py` | `test_literature_parity::test_colt_separable_classes` | pinned | — |
| COLT22-E02 | empirical | MNIST digits via assembly-per-class | `programs/learn.py` | — | missing | `parity/colt2022_mnist.yaml` + golden accuracy |
| COLT22-E03 | empirical | Sample complexity / margin scaling | — | — | missing | Research experiment + bound check |

### Dabagia et al. (2024/25) — Sequences · `dabagia2025sequences`

| ID | Category | Claim / result | Module | Test / protocol | Status | Gap |
|----|----------|----------------|--------|-----------------|--------|-----|
| SEQ25-T01 | theorem | Turing completeness via sequence constructions | — | — | theorem | Do not claim in package docs |
| SEQ25-M01 | mechanism | `sequence_memorize` + `ordered_recall` | `ops`, `test_sequences.py` | `test_literature_parity::test_sequence_memorize_ordered_recall` | pinned | — |
| SEQ25-M02 | mechanism | LRI enables ordered recall | `test_lri.py` | `test_lri.py` (package) | partial | Pin LRI ablation protocol |
| SEQ25-M03 | mechanism | FSM inferred from transition traces | `fsm.py`, `fsm_learn.py` | `test_literature_parity::test_fsm_learn_from_traces` | pinned | — |
| SEQ25-M04 | mechanism | PFA / probabilistic transitions | `pfa.py` | `test_pfa.py` | partial | Link to coin-flip paper protocol |
| SEQ25-E01 | empirical | Unary TM increment halts (`11_` → `111_`) | `tm_demo.py` | `test_literature_parity::test_tm_demo_unary_increment` | pinned | — |
| SEQ25-E02 | empirical | Longer unary tape (`111_`) | `tm_demo.py` | `test_literature_parity::test_tm_demo_longer_unary_tape` | pinned | — |
| SEQ25-E03 | empirical | Multi-step recall at paper length | `ops` | — | missing | `parity/sequences2025_recall_len.yaml` |
| SEQ25-E04 | empirical | Full paper TM construction runnable | `simulation/turing_simulations.py` | — | research | Port `dmitropolsky/assemblies` `turing_sim` |
| SEQ25-E05 | empirical | GPU path recall parity | `torch_engine` | `test_literature_parity_gpu::test_sequence_memorize_ordered_recall_torch` | partial | Golden on CI with CUDA |

---

## Tier 2 — NEMO / Language / Cognition

### Mitropolsky et al. (2021) — TACL · `mitropolsky2021parser`

| ID | Category | Claim / result | Module | Test / protocol | Status | Gap |
|----|----------|----------------|--------|-----------------|--------|-----|
| TACL21-M01 | mechanism | Fiber-gated incremental parse | `language/parser.py` | `test_cross_engine_projection` | partial | End-to-end fiber step golden |
| TACL21-M02 | mechanism | Explicit LEX ↔ sparse grammar projection | `core/brain.py` | `test_cross_engine_projection` (6 tests) | pinned | — |
| TACL21-E01 | empirical | English closed-lexicon sentences parse | `RuleParser` | `test_tacl_parser_suite` (5 EN) | partial | Expand to paper full list |
| TACL21-E02 | empirical | Russian free word order | `grammar_rules` | `test_russian_parser_sentences`, OSV roles | partial | Paper-scale RU benchmark |
| TACL21-E03 | empirical | Ditransitive NOM/ACC/DAT roles | `RuleParser` | `test_russian_ditransitive_roles` | pinned | — |
| TACL21-E04 | empirical | Lexical readout via `area_lexeme_cache` | `parser.py` | `test_english_cats_chase_mice_roles` | pinned | — |
| TACL21-E05 | empirical | Paper parser accuracy / dependency F1 | — | — | missing | `parity/tacl2021_benchmark.json` from ref |
| TACL21-D01 | demo | `cats chase mice` smoke | `RuleParser` | `test_literature_parity::test_rule_parser_cats_chase_mice` | pinned | — |

### Mitropolsky et al. (2022) — NALOMA · `mitropolsky2022center`

| ID | Category | Claim / result | Module | Test / protocol | Status | Gap |
|----|----------|----------------|--------|-----------------|--------|-----|
| NALO22-T01 | theorem | AC parser + WM ⟹ CFG characterization | — | — | theorem | — |
| NALO22-M01 | mechanism | Center-embedded relative clause parse | `parser.py` | `test_tacl_parser_suite` (2 CE sentences) | partial | Paper depth-2 lexicon |
| NALO22-M02 | mechanism | `DEP-VERB` fiber readout | `parser.py` | `test_literature_parity::test_center_embedding_dep_verb_readout` | pinned | — |
| NALO22-E01 | empirical | Outer-clause SUBJ stable under embedding | `parser.py` | — | missing | Fix SUBJ overwrite; pin roles |
| NALO22-E02 | empirical | Working-memory comma replay | `parser.py` | CE sentences (structural) | partial | Explicit WM step protocol |
| NALO22-R01 | research | Depth-2 at scale | `research/experiments/primitives/test_center_embedded_baseline.py` | research only | research | Promote winner to package |

### d'Amore et al. (2022) — AAAI · `damore2022planning`

| ID | Category | Claim / result | Module | Test / protocol | Status | Gap |
|----|----------|----------------|--------|-----------------|--------|-----|
| AAAI22-M01 | mechanism | STRIPS blocks-world BFS planner | `planning.py` | `TestBlocksPlanning` 2–4 block | pinned | — |
| AAAI22-M02 | mechanism | Neural plan execution scaffold | `planning.solve_and_run` | `test_three_block_neural_execution`, `test_four_block_plan_and_neural_run` | pinned | — |
| AAAI22-E01 | empirical | Plan reaches goal state | `verify_plan` | blocks tests | pinned | — |
| AAAI22-E02 | empirical | Full paper AC operator program (no discrete STRIPS shortcut) | — | — | missing | Port AAAI operator sequence |
| AAAI22-E03 | empirical | Paper-scale tower depth / search stats | — | — | missing | `parity/aaai2022_planning.yaml` |

### Mitropolsky & Papadimitriou (2023) — Language organ · `mitropolsky2023architecture`

| ID | Category | Claim / result | Module | Test / protocol | Status | Gap |
|----|----------|----------------|--------|-----------------|--------|-----|
| ORG23-M01 | mechanism | NEMO areas: LEX, roles, fibers | `nemo/`, `parser.py` | `test_nemo_patterns_core.py` | partial | Single organ entrypoint |
| ORG23-M02 | mechanism | LRI replaces imperative control | `ops`, LRI | `test_lri.py` | partial | Organ-level LRI integration test |
| ORG23-M03 | mechanism | Differential lexicon | `emergent/` | NEMO2025 parity (inherited) | partial | — |
| ORG23-E01 | empirical | End-to-end organ on paper toy corpus | `research/nemo/` | — | research | Promote stable path to package |
| ORG23-E02 | empirical | Russian + English organ parity | — | — | missing | Unified benchmark |
| ORG23-D01 | demo | Toy `NemoParser` | `assembly_calculus/parser.py` | — | partial | Document limits in protocol |

### Dabagia et al. (2024) — Coin-flipping · `dabagia2024coinflipping`

| ID | Category | Claim / result | Module | Test / protocol | Status | Gap |
|----|----------|----------------|--------|-----------------|--------|-----|
| COIN24-M01 | mechanism | Trace counts become a symbolic branch schedule | `pfa.py`, `markov_coin.py` | `test_markov_transition_frequencies_from_traces` | partial | No calibrated neural probability follows |
| COIN24-M02 | mechanism | Explicit seed-mixture branch selection | `pfa.py` | `test_pfa_choice_contract.py` | partial | Register a mechanism-sensitive attractor protocol |
| COIN24-M03 | mechanism | Ambient noise + E%-WTA on coin area | `markov_coin.py` | `test_coin_ambient_noise_epwta_wiring` | retracted | Legacy golden used the dead construction |
| COIN24-E01 | empirical | Fair / biased coin empirical outcomes | `RandomChoiceArea` | `coin2024_demo` | retracted | New preregistered attractor study required |
| COIN24-E02 | empirical | Markov chain flip protocol | `arc_markov.py` | `coin2024_markov_arc` | retracted | Old wrapper omitted the explicit arc protocol |
| COIN24-E03 | empirical | Match [dabagia.org demo](http://dabagia.org/nemo/coinflipping/) statistics | ? | `coin2024_demo` | retracted | Demo lacks an exported numerical reference |
| COIN24-E04 | empirical | Softmax coin-flip trajectories | ? | `coin2024_softmax` | retracted | SoftmaxContextCoin was an invalid instrument |

### Mitropolsky & Papadimitriou (2025) — Language acquisition · `mitropolsky2025simulated`

| ID | Category | Claim / result | Module | Test / protocol | Status | Gap |
|----|----------|----------------|--------|-----------------|--------|-----|
| NEMO25-M01 | mechanism | Grounded differential lex → POS | `emergent/` | `test_pos_classification_at_least_80_percent` | pinned | Paper vocab size |
| NEMO25-M02 | mechanism | Role binding (agent/patient) | `emergent/` | `test_pinned_role_binding` (3 sentences) | pinned | Expand role grid |
| NEMO25-E01 | empirical | Novel composition generalization | `emergent/` | `test_novel_generalization_bird_chases_boy` | pinned | — |
| NEMO25-E02 | empirical | Word order emergence (SVO) | `emergent/` | `test_word_order_svo` | pinned | — |
| NEMO25-E03 | empirical | Curriculum stages at paper scale | `emergent/training/` | — | missing | `parity/nemo2025_curriculum.yaml` |
| NEMO25-E04 | empirical | Continuous grounding encoders | `research/experiments/lib/grounding.py` | research | research | Package-stable encoder |
| NEMO25-E05 | empirical | Multi-turn / chat tuning | `emergent/agent/` | `test_emergent_agent.py` | partial | Not paper protocol; extension |
| NEMO25-E06 | empirical | Unsupervised role discovery | `research/experiments/primitives/test_unsupervised_binding.py` | research | research | Promote if stable |
| NEMO25-D01 | demo | Blocks / JSON / multi-tool (repo extension) | `emergent/blocks_bridge.py` | `test_emergent_agent.py` | partial | Out of paper scope; keep separate tier |

---

## Tier 3 — Extensions

### Onasch et al. (2025) — Dendritic gating · `onasch2025dendritic`

| ID | Category | Claim / result | Module | Test / protocol | Status | Gap |
|----|----------|----------------|--------|-----------------|--------|-----|
| ONA25-M01 | mechanism | Context-gated plasticity per compartment | — | — | missing | `compute/dendritic.py` design |
| ONA25-E01 | empirical | Reduced catastrophic forgetting vs baseline | — | — | missing | `parity/onasch2025_forgetting.yaml` |
| ONA25-E02 | empirical | Paper figure reproduction | — | — | missing | Golden from bioRxiv supplement |

### Hoff et al. (2026) — E%-WTA · `hoff2026epwta`

| ID | Category | Claim / result | Module | Test / protocol | Status | Gap |
|----|----------|----------------|--------|-----------------|--------|-----|
| EPWTA26-M01 | mechanism | Variable winner count capped by `e_fraction` | `winner_policies.py` | `test_epwta_variable_size_cap` | pinned | — |
| EPWTA26-M02 | mechanism | `EPercentPolicy` on brain areas | `brain.py` | `test_literature_parity::test_epercent_policy_on_area` | pinned | — |
| EPWTA26-E01 | empirical | E/I ratio dynamics vs fixed k-WTA | `torch_engine` | `test_epwta_gpu.py` | partial | Golden recovery curves |
| EPWTA26-E02 | empirical | Cross-engine E%-WTA parity | `torch_engine` | `test_epwta_gpu.py` | partial | CI CUDA job |
| EPWTA26-E03 | empirical | Paper assembly size distributions | — | — | missing | `parity/hoff2026_size_dist.json` |

### Ting et al. (2026) — Speech · `ting2026speech`

| ID | Category | Claim / result | Module | Test / protocol | Status | Gap |
|----|----------|----------------|--------|-----------------|--------|-----|
| TING26-M01 | mechanism | Continuous speech → assembly encoding | — | — | missing | `encoders/speech.py` |
| TING26-M02 | mechanism | Mel binarisation / population MFCC | — | — | missing | — |
| TING26-E01 | empirical | Phone classification accuracy | — | — | missing | TIMIT slice + golden |
| TING26-E02 | empirical | Boundary detection F1 | — | — | missing | — |
| TING26-E03 | empirical | vs deep-learning baseline | — | — | missing | Research tier |

### Kopadi & Kalles (2026) — DIRECT · `kopadi2026causal`

| ID | Category | Claim / result | Module | Test / protocol | Status | Gap |
|----|----------|----------------|--------|-----------------|--------|-----|
| DIR26-M01 | mechanism | Directional binding asymmetry | `programs/direct.py` | `test_direct_binding_rejects_the_retired_instrument` | retracted | overlap survives wiping learned fiber |
| DIR26-M02 | mechanism | `do(effect)` intervention | `direct.py` | same refusal contract | retracted | requires synaptic-asymmetry readout and wipe control |
| DIR26-E01 | empirical | Full Pearl do-calculus battery | — | — | missing | `parity/direct2026_pearl.yaml` |
| DIR26-E02 | empirical | Synaptic asymmetry training fidelity | — | — | missing | — |
| DIR26-E03 | empirical | Paper causal graph examples | — | — | missing | Golden per graph |

---

## Cross-cutting infrastructure (enables all papers)

See supplement rows `XINF-C01` (config registry), `XINF-C02` (MATLAB legacy), `XINF-E03` (Julia golden).

| ID | Claim | Status | Gap |
|----|-------|--------|-----|
| XINF-M01 | Explicit ↔ sparse connectome protocol | pinned | `test_cross_engine_projection` |
| XINF-M02 | Parser `engine=auto` → `numpy_sparse` for mixed LEX | pinned | `resolve_mixed_engine` |
| XINF-M03 | `CSRConn.sparse` flag for protocol checks | pinned | `connectome.is_sparse_connectome` |
| XINF-E01 | `dmitropolsky/assemblies` live clone optional parity | partial | CI submodule or pinned wheel |
| XINF-E02 | `AssemblyCalculus.jl` cross-language golden | missing | Export Julia metrics JSON |
| XINF-D01 | `reproduce.py --claim <id>` one-command runner | missing | Wire matrix JSON → pytest -k |

---

## Implementation phases (recommended)

| Phase | Focus | Exit criterion |
|-------|-------|----------------|
| **P0** | PNAS golden + cross-engine + current 65 tests | All `pinned` rows green in CI |
| **P1** | COLT MNIST, Sequences long recall, TACL full list, coin demo golden | +15 rows pinned |
| **P2** | AAAI full AC program, NALOMA SUBJ fix + depth-2, NEMO2025 curriculum scale | Language tier `implemented` |
| **P3** | TM full construction, Hoff E/I curves, DIRECT Pearl battery | Foundations + extensions parity |
| **P4** | Onasch dendritic, Ting speech encoder | New modules + goldens |

---

## How this helps “for free”

- **Debugging:** failing row → exact protocol + module; no guessing which paper broke.
- **Refactors:** connectome / engine changes run against full claim grid.
- **Science:** emergent / developmental work inherits substrate guarantees (P0).
- **Papers:** each `pinned` row is a citable “Reproduced via `claim_id`” footnote.
- **Onboarding:** new contributor picks a `missing` row, adds golden, flips status.

---

## Maintenance

1. Add a row to [reproduction_matrix.json](reproduction_matrix.json) before merging new paper code.
2. Update `status` when protocol lands; sync [index.json](index.json) `gaps`.
3. Run `uv run python research/literature/validate_matrix.py` (validates base + supplement).
4. Regenerate supplement after audit edits: `uv run python research/literature/build_matrix_supplement.py`.
4. Full parity CI:

```bash
uv run pytest neural_assemblies/tests/test_literature_parity.py \
  neural_assemblies/tests/test_tacl_parser_suite.py \
  neural_assemblies/tests/test_cross_repo_parity.py \
  neural_assemblies/tests/test_cross_engine_projection.py -v
```
