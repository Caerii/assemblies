"""

Literature parity protocols — pinned parameters for reproducible checks.



All tests in ``neural_assemblies/tests/test_literature_parity.py`` use these defaults

unless a paper-specific protocol says otherwise.



## Global defaults



| Parameter | Value | Rationale |

|-----------|-------|-----------|

| seed | 42 | Reproducibility across CI |

| n | 5000 | Sparse regime, faster than paper n=10⁴ |

| k | 80 | k/n = 0.016 |

| p | 0.05 | Standard connection probability |

| beta | 0.1 | Plasticity for PNAS/COLT primitives |

| rounds | 10 | Stabilization rounds |



## PNAS 2020 — project / separate / merge



- **Project**: ``project(brain, stim, area, rounds=10)``; persistence overlap > 0.85.

- **Separate**: two stimuli → same area; overlap < 3× chance (k/n).

- **Merge**: two source areas → target; cue overlap > 2× chance.



## COLT 2022 — learn_assembly



- ``learn_assembly(brain, stim, area, max_epochs=12, project_rounds=6, convergence=0.85)``

- Success: convergence within max_epochs OR high overlap with fresh project.



## TACL 2021 — rule parser



- ``RuleParser(p=0.1, lex_k=20).parse("cats chase mice")`` — non-empty fiber readout.

- **Russian**: NOM/ACC/DAT roles on ditransitive ``kot dayet kotu sobaku``; OSV ``sobaku vidit kot``.
- **Lexical readout**: fiber dependencies include content words via ``area_lexeme_cache`` (e.g. ``cats``, ``chase``, ``mice``).



## Sequences 2023/25 — FSM / TM / LRI



- **FSM learn**: infer transitions from ``(state, symbol, next)`` traces; step matches table.

- **TM demo**: unary tapes ``11_`` and ``111_`` halt in ``q_halt``.

- **Memorize + recall**: ``sequence_memorize`` then ``ordered_recall`` with LRI; first recall overlap > 0.3 (numpy + torch_sparse when CUDA)..



## Coin-flipping 2024



- Train PFA from transition frequency traces; ``RandomChoiceArea.flip()`` returns 0 or 1.

- **Markov frequencies**: 5× ``(q0, flip, q0)`` + 5× ``(q0, flip, q1)`` → estimated P=0.5 each.

- **PFA stochasticity**: 50/50 transition table → both target states over 30 seeded steps.

- **Fair coin**: bias=0.5 over 24 flips → both outcomes appear.

- **Biased coin**: bias=0.85 over 20 flips → more 0s than 1s.

- **Ambient noise**: ``CoinFlipModel(..., input_noise_std=0.02)`` wires E%-WTA on coin area.



## Center-embedding 2022 (NALOMA)



- Sentence pattern: ``outer that inner , outer_continued`` using lexicon words + markers ``that``, ``,``.

- Example: ``cats chase mice that dogs chase cats , love mice``

- Success: fiber readout length ≥ 5 and at least one ``DEP-VERB`` role; inner clause words (e.g. ``dogs``, ``chase``) via ``_inner_lexeme_cache``.



## AAAI 2022 — blocks-world planning



- **2-block swap** (``make_toy_problem``): A-on-B → B-on-A; BFS plan non-empty; ``verify_plan`` reaches goal.

- **3-block tower** (``make_three_block_problem``): A-on-B-on-C → C-on-B-on-A; same checks.

- **Neural scaffold**: ``solve_and_run`` → ``(plan, trajectory, final_state)``; ``final_state.on == goal.on``.
- **4-block tower**: ``make_four_block_problem`` BFS + neural run.



## E%-WTA 2026



- ``Brain.add_area(..., winner_policy=EPercentPolicy(...))`` and ``set_competition_policy``.

- Variable winner count: ``len(assembly) <= max(min_winners, round(e_fraction * n))`` after stabilization.

- Hoff et al.: fraction_of_max gates relative threshold; e_fraction caps population fraction.



## DIRECT 2026 — causal binding



- ``direct_bind`` then ``measure_directional_asymmetry``; forward overlap > 0.1.

- **do-calculus**: ``validate_direct_do_calculus`` — ``do(effect)`` preserves cause→bind (≥ 45% of forward).



## Cross-engine projection (explicit ↔ sparse)



- **Connectome protocol**: use ``connectome.is_dense_connectome`` / ``dense_connectome_or_new``;
  do not read ``.sparse`` on engine-native CSR objects directly.
- **Parser brains**: ``resolve_mixed_engine("auto")`` → ``numpy_sparse`` (explicit LEX + sparse grammar).
- **Explicit → sparse**: dense connectomes on sparse engine; `_bootstrap_from_explicit_dense`

- **Sparse → explicit**: `_sparse_sources_drive_to_explicit` + `external_drive` in explicit engine

- **GPU**: `torch_sparse` mirrors numpy path via `_dense_area_conns` / `set_dense_area_conn`

- Tests: `tests/test_cross_engine_projection.py` (numpy + torch when CUDA available)



## NEMO 2025 — simulated language acquisition (mitropolsky2025simulated)



- **Path**: ``EmergentParser`` in ``assembly_calculus/emergent/``; train on grounded curriculum.

- **Parameters**: same global defaults (``n=5000``, ``k=80``, ``seed=42``, ``rounds=10``).

- **POS**: classify all ``VOCABULARY`` words via differential readout; accuracy ≥ 80%.

- **Roles** (pinned): ``the dog runs`` (dog=AGENT); ``the cat chases the bird`` (cat=AGENT, bird=PATIENT); ``she sees the bird`` (she=AGENT, bird=PATIENT).

- **Novel**: ``the bird chases the boy`` — bird=AGENT, boy=PATIENT, chases=ACTION (not in training).

- **Word order**: ``EvaluationSuite.evaluate_word_order(target="SVO")`` → ``correct=True``.

- **Tests**: ``test_literature_parity.py::TestLiteratureParityNEMO2025``; full suite in ``test_emergent_parser.py``.



## Emergent agent layer (chat → instruction → tools)



- **Train**: ``parser.train_for_agent()`` — grammar + imperatives + ``train_next_token``.

- **Session**: ``EmergentSession.interact(text)`` — Q-A and imperative acks on ``numpy_sparse``.

- **Frames**: ``parse_instruction(words)`` → ``InstructionFrame`` (JSON via ``.to_json()``).

- **Tools**: ``ToolRegistry.dispatch(frame)`` → ``ToolCall`` (``{"tool", "arguments"}``).

- **Roadmap**: ``assembly_calculus/emergent/ROADMAP.md`` (phases 2–5: chat tuning, JSON, multi-tool).

- **Tests**: ``tests/test_emergent_agent.py``.



## Blocks-world instructions — Phase 3



- **Bridge**: ``blocks_bridge.BlocksLanguageExecutor`` — language → ``BlocksAction`` → STRIPS state.

- **Patterns**: ``move a to b``, ``put a on table`` (normalized to ``blk_*`` tokens).

- **Train**: ``train_for_agent(include_blocks=True)`` adds ``blocks_curriculum`` sentences.

- **Tools**: ``move_block`` tool with ``block`` + ``destination`` args; ``execute_tools=True`` on session.

- **Eval**: ``evaluate_tool_compliance()``, ``evaluate_blocks_execution()``.



## Structured JSON — Phase 4



- **Records**: ``StructuredRecord`` — schema-tagged JSON; ``validate_record()`` against ``JsonSchema``.

- **Emit**: ``parser.words_to_json(words)`` — language → ``{"tool", "arguments"}``.

- **Roundtrip**: ``structured_roundtrip(words)`` — language → JSON → words → tool call match.

- **Session**: ``interact_json(text)``, ``execute_json(json_text)``.

- **Eval**: ``evaluate_json_roundtrip()``, ``evaluate_schema_compliance()``.



## Multi-tool plans — Phase 5



- **Plan**: ``ToolPlan`` — ordered ``move_block`` steps as JSON ``{source, goal, steps}``.

- **Explicit**: ``move a to table then move a to b`` — split on ``then`` / ``;``.

- **Goal-directed**: ``stack a on b`` — BFS via ``BlocksWorldPlanner`` → tool step list.

- **Session**: ``interact_plan(text)``, ``execute_plan_json(plan_json)``.

- **Eval**: ``evaluate_multi_tool_plan()``.



## Scaled conversation curriculum — Phase 6



- **Vocabulary**: ``build_vocabulary_preset("core" | "medium" | "large" | "discussion")``.

- **Curriculum**: ``CurriculumTrainer.train_conversation_path(max_stage="DIALOGUE" | "CONVERSATION")``.

- **Training**: ``parser.train_for_conversation()``; ``EmergentSession.bootstrap_conversation(preset=...)``.

- **Chat CLI**: ``python examples/chat_emergent.py --preset medium --stage DIALOGUE``.


- **State**: ``DialogueState`` — entity slots, pronoun resolution (``it`` → last patient).

- **Curriculum**: ``dialogue_curriculum.get_dialogue_curriculum()`` — Q-A pairs on closed vocab.

- **Train**: ``train_for_agent()`` includes dialogue sentences + ``train_dialogue()`` CONTEXT→PREDICTION bridges.

- **Online**: ``present_turn(words, learn=True)`` per chat turn; ``parse_instruction_with_context()``.

- **Eval**: ``EvaluationSuite.evaluate_dialogue(qa_pairs)``.



## Cross-repo PNAS parity (dmitropolsky/assemblies)



- **Golden file**: ``research/literature/parity/reference_pnas_golden.json`` (seed=42 metrics from reference ``brain.py``).

- **Metrics**: ``project_persistence`` (re-project same stimulus); ``separate_overlap`` (two stimuli → one area).

- **Tolerance**: overlap ± 0.02 when n, k, p, beta, rounds match.

- **Recorded golden** (n=5000, k=80): persistence **1.0**, separate **0.025**, chance **0.016**.

- **Tests**: ``tests/test_cross_repo_parity.py`` — always compares package vs golden; live reference clone optional (``.reference/dmitropolsky-assemblies`` or ``REF_ASSEMBLIES_PATH``).

