# Claim Audit: Dependence on Confirmed Defects

> **STATUS UPDATE (2026-07-22, later same day).** Several defects this audit
> depends on have since been FIXED. This header records what changed; the body
> below is preserved as the dated record of the pre-fix state and its verdicts
> should be read against this update, not as current.
>
> **Lifted:**
> - **BUG 7-E** (`EmergentParser` role read-out dominated by an AGENT-first
>   positional prior). The read-out is now `_role_binding_margin` (residual
>   above the area-generic attractor) and is training-dependent. VERIFIED with
>   the audit's own Condition B discriminator: role-exclusive OVS training moves
>   the object-initial score from **0.000 (untrained null) to 1.000 (trained)**,
>   3 seeds, sd 0.00. Training now changes role assignment on an object-initial
>   order, which is exactly what the invalidated claims required. Claims 9, 11,
>   12, 13, 14 (BUG 7-E rows) are candidates for re-validation — RE-RUN the
>   cited experiments before citing them; this header is not itself the
>   re-validation of each result file.
> - **BUG 12 / BUG 13** (hand-authored English SVO prior applied regardless of
>   typology; non-SVO/SOV/VSO generation falling through to SVO). Word order is
>   now a six-label space in `emergent/core/word_order.py`, inferred from role
>   annotations; position profiles are sourced `learned` on the default corpus;
>   generation walks the inferred order. A hard identifiability limit was found
>   and surfaced (`word_order_identifiable`): POS transitions alone cannot
>   distinguish subject-initial from object-initial, so a transitions-only "SVO"
>   is not evidence against OVS.
>
> **NOT fixed — these verdicts stand:**
> - **BUG 7 / BUG 6** (`NemoParser.parse`, `parser.py`). This is a DIFFERENT
>   parser from `EmergentParser`; its hardcoded SVO role template was not
>   touched. Claims 1-3, 8 remain INVALIDATED. `assign_role` remains an
>   unreachable latent landmine.
> - **BUG 4** (`score_corpus` trained on the eval corpus) — PARTIALLY FIXED.
>   `predict_next_token` now disables *plasticity* by default, which stops
>   weight-training on the eval set. But `brain.frozen()` disables plasticity,
>   NOT materialization: on the sparse engine a "frozen" projection still samples
>   candidates and recruits new neurons, so the connectome's STRUCTURE still
>   grows during scoring (measured: `w` 1270→1274 in a small run, ~50 neurons in
>   a larger one). So `score_corpus` remained not-strictly-read-only even after
>   the plasticity fix — the substrate still moves across a corpus, and
>   predictions can be order-dependent. Magnitude is config-dependent (often
>   small, but it moved ~18% of predictions in one batched-inference audit).
>   The true fix is the engine `readonly` mode (2026-07-24, `torch_engine`):
>   suppress candidate sampling so a frozen projection selects only among
>   materialized neurons. Any score_corpus number wanting a *fixed* substrate
>   should set it. Direction of the affected claims (above chance) is unchanged;
>   the held-out/deterministic framing of the numbers is what this weakens.
> - **BUG 3** (PFA `_flip_k_split` plasticity) — a plasticity guard was added,
>   but deeper analysis showed the coin performs NO attractor dynamics at all
>   (its recurrent fiber is never materialised), so PFA/coin results measure
>   seed overlap, not assembly competition. That is a stronger caveat than this
>   audit recorded and applies to both flip modes.
>
> A primitive-level conformance suite (`tests/test_ac_conformance.py`) was added
> after this audit and found deeper issues than any claim here: independence and
> pattern completion failed until `norm_init` (now default) fixed a degree-hub
> collapse. Every result in this audit predates that substrate fix and was
> measured on collapsed assemblies.

**Date:** 2026-07-22
**Scope:** Read-only audit. No library code was modified. The audit traces every
scientific claim in this repository that could rest on eight confirmed defects
found during the code-exposition pass.

**Verdict vocabulary**

| Verdict | Meaning |
|---|---|
| INVALIDATED | The claim cannot be supported by the cited evidence at all. The measurement is produced by a code path that does not measure what the claim says. |
| SUSPECT | The claim may well be true, but the cited evidence is contaminated and cannot be used as-is. Re-run under a corrected protocol before citing. |
| UNAFFECTED | The claim runs through a different code path. The specific path is named. |
| UNDETERMINED | Dependence could not be settled from reading alone; the settling experiment is named. |

---

## Bug inventory under audit

| ID | Location | Defect |
|---|---|---|
| BUG 7 | `neural_assemblies/assembly_calculus/parser.py:339` | `NemoParser.parse` assigns roles by hardcoded SVO template for sentences of >= 3 words. Consults neither role areas nor SEQ. |
| BUG 6 | `neural_assemblies/assembly_calculus/parser.py:289` | `NemoParser.assign_role` is dict-membership bookkeeping, not neural readout. Ambiguous words resolve by insertion order. |
| BUG 4 | `neural_assemblies/assembly_calculus/next_token.py:162` | `score_corpus` trains on the evaluation corpus (plasticity default-on inside `predict_next_token`). |
| BUG 8 | `neural_assemblies/assembly_calculus/metrics/instability.py:69` | `compute_anchored_instability` requires plasticity OFF but never checks. |
| BUG 12 | `neural_assemblies/assembly_calculus/emergent/parser_mixins/distributional.py:59` | `_CATEGORY_POSITION_PROFILES` is a hand-authored English SVO prior applied regardless of inferred typology. |
| BUG 13 | `neural_assemblies/assembly_calculus/emergent/parser_mixins/generation.py:183` | Non-SVO/SOV/VSO word orders fall through to SVO in `generate`. |
| BUG 3 | `neural_assemblies/assembly_calculus/pfa.py:158` | `_flip_k_split` (package default) leaves plasticity ON; successive flips are not independent. |

### New finding surfaced by this audit (BUG 7-E)

`EmergentParser` is a **different parser** from `NemoParser` and its
`parse` (`emergent/parser_mixins/core.py:1102`) does route through a genuine
neural readout, `_assign_roles_neural`. However that readout carries a
**hardcoded AGENT-first structural prior** that dominates it:

* `_determine_role_order` (`core.py:924-969`) returns `[ROLE_AGENT, ROLE_PATIENT]`
  unconditionally unless a *learned passive gating* rule fires. It never
  consults `word_order_type`, `infer_word_order`, or any typological state.
* `_assign_roles_neural` (`core.py:1073`) adds `prior = 1.2 * 0.5**rank`, so the
  first uninhibited role gets +1.2 and the second +0.6. Lexical overlap
  (`assembly_overlap`, range [0,1]) can never make up the 0.6 gap.
* Consequence: **the first noun is always AGENT and the second is always
  PATIENT**, for any word order, trained or untrained.

This is the *emergent* analogue of BUG 7 and it is the load-bearing defect for
the cross-linguistic typology result.

**Empirical proof (run for this audit, n=2000, k=40, seed=42, 20 training
sentences, 10 held-out transitive test sentences, `EmergentParser`):**

| Condition | `infer_word_order` | role accuracy |
|---|---|---|
| A. train SVO -> test SVO | SVO | **1.000** |
| B. **no training at all** -> test SVO | (untrained) | **1.000** |
| C. train VSO -> test VSO | VSO | **1.000** |
| D. **no training at all** -> test VSO | (untrained) | **1.000** |
| E. train OSV -> test OSV | SOV (wrong) | **0.000** |
| F. **no training at all** -> test OSV | (untrained) | **0.000** |
| G. train OVS -> test OVS | SVO (wrong) | **0.000** |

Role accuracy is bit-for-bit identical between the trained and untrained
parser. Training contributes nothing to the metric. Object-initial orders score
0.000 whether trained or not — training cannot repair them.

Word ORDER inference (`infer_word_order`) is, by contrast, genuinely learned:
it correctly separated SVO from VSO from category transition statistics, and
it fails on OSV/OVS only because its label space is literally the three-way
`{SVO, SOV, VSO}` (`distributional.py:794-835`).

---

## Claim table

### BUG 7 / BUG 6 — `NemoParser` role binding

| # | Claim | Location | Bug | Verdict | Evidence |
|---|---|---|---|---|---|
| 1 | "Novel SVO sentence should get correct role assignments" (`test_novel_sentence_roles`) | `neural_assemblies/tests/test_parser_composition.py:231-240` | 7 | **INVALIDATED** as role-binding evidence | `_build_parser` (line 56) constructs `NemoParser`. Test asserts `roles["cat"]=="AGENT"` on a 3-word sentence, so `parse` takes the `len(words) >= 3` template branch (`parser.py:339`). The assertion holds identically if `train_roles` (line 70) is deleted. |
| 2 | "Train on 4 sentences, parse 2 novel ones... Target >= 80% accuracy on both [category and role]" (`test_full_pipeline_accuracy`) | `neural_assemblies/tests/test_parser_composition.py:242-280` | 7 | **INVALIDATED** for the role half; category half UNAFFECTED | Same `NemoParser` path. All test sentences are 3-word SVO. Role accuracy is 100% by construction of the template. The category assertions do go through `classify_word` (real differential readout) and stand. |
| 3 | "Parse a sentence that appeared in training data" role assertions (`test_trained_sentence_parse`) | `neural_assemblies/tests/test_parser_composition.py:206-217` | 7 | **INVALIDATED** for role assertions | Same path. |
| 4 | Reproduction matrix `ORG23-D01` "Toy `NemoParser`", status `partial` | `research/literature/REPRODUCTION_MATRIX.md:218` | 7 | **SUSPECT** (already flagged `Document limits in protocol`) | Entry has no pinned test and already carries a caveat. It must be amended to state explicitly that `NemoParser` role output is a Python template. |
| 5 | Notebook: staged training (`train_lexicon`/`train_roles`/`train_word_order`) followed by a role table from `parser.parse` | `examples/notebooks/volume-03-language/01_nemo_parser_toy_sentence.ipynb` | 7 | **UNAFFECTED (already correctly stated)** | The notebook's own markdown says "Because the parser has a simple SVO rule at this level, the first noun becomes AGENT and the second noun becomes PATIENT." It does not claim emergence. Recommend one sentence added noting that the `train_roles` stage above does not feed this table. |
| 6 | Notebook: "Role assignments by probe case" failure-surface chart | `examples/notebooks/volume-03-language/02_parser_failure_cases.ipynb` | 7 | **UNAFFECTED** | Purpose is to show limits, and the concluding markdown already frames the output as "category/role/order plumbing for a tiny controlled setup." |
| 7 | `docs/scientific_status.md:19` — "NEMO and emergent-parser tests cover narrow behaviors such as word-category separation, **role binding**, and word-order structure" | `docs/scientific_status.md:19` | 7, 12, 7-E | **INVALIDATED as written** | The two things this sentence points at are `test_parser_composition.py` (BUG 7, template) and `test_emergent_parser.py` / `test_literature_parity.py` (BUG 7-E, structural prior). Neither test demonstrates role binding. Word-category separation and word-order structure do stand. |
| 8 | Any word trained in two roles resolves correctly | (no claim found) | 6 | **UNAFFECTED — no claim depends on it** | `assign_role` (`parser.py:289`) is only reachable from `parse` for sentences of **fewer than 3 words**. No test, notebook, doc, or result in this repo parses a 1- or 2-word sentence through `NemoParser.parse`. BUG 6 is a live landmine, not a current invalidator. |

### BUG 7-E / BUG 12 — `EmergentParser` role binding and typology

| # | Claim | Location | Bug | Verdict | Evidence |
|---|---|---|---|---|---|
| 9 | **Cross-linguistic typology: role accuracy 1.000 (SEM 0.000) for SVO, SOV and VSO; H5 "role assignment adapts to word order"** | `research/results/applications/crosslinguistic_typology_20260209_171109_quick.json`; experiment `research/experiments/applications/test_crosslinguistic_typology.py:411-440` | 7-E, 12 | **INVALIDATED** | Conditions B and D above reproduce 1.000 with a completely untrained parser. Additionally, all three tested typologies place subject before object, so a first-noun-is-AGENT rule scores 100% on every one of them by construction. The metric has zero discriminative power over the hypothesis it was built to test. H5 as stated ("agent before verb in SVO, agent after verb in VSO") is not what the code measures — `parse` never consults verb position. |
| 10 | Cross-linguistic typology: **order** accuracy 1.000 and order confidence for SVO/SOV/VSO | same result file, `metrics.*_order_accuracy` | 12 | **SUSPECT (mild) — likely sound** | Order inference runs through `infer_word_order` (`distributional.py:794`), which reads `dist_stats.category_transitions` — genuinely learned. Reproduced independently in conditions A and C above. The contamination risk is indirect: the categories fed into those transitions come from `classify_word_cached`, which can fall through to `classify_distributional` and its English position prior. In *this* experiment every one of the 20 vocabulary items is explicitly grounded, so the grounding route wins and the prior is not consulted — but that is a property of the vocabulary, not of the code. Safe to keep with that caveat recorded. |
| 11 | Cross-linguistic typology: H4 "classification accuracy is comparable across typologies" (all paired t = 0.0, p = 1.0) | same result file | 7-E | **INVALIDATED as informative** | The paired tests compare three vectors that are all exactly `[1,1,1]`. A zero-variance comparison of a saturated metric is not evidence of equivalence. |
| 12 | `NEMO25-M02` "Role binding (agent/patient)" — status **pinned** to `test_pinned_role_binding` (3 sentences) | `research/literature/REPRODUCTION_MATRIX.md:237` | 7-E | **INVALIDATED as pinned evidence** | Fixture `nemo_parser` (`test_literature_parity.py:368-376`) is an `EmergentParser`. All three pinned sentences (`the dog runs`, `the cat chases the bird`, `she sees the bird`) are subject-first. Condition B shows an untrained `EmergentParser` passes this shape. Downgrade from `pinned` to `missing` until an object-initial or role-ambiguous probe exists. |
| 13 | `test_novel_generalization_bird_chases_boy` — novel-sentence role generalization | `neural_assemblies/tests/test_literature_parity.py:415-419` | 7-E | **INVALIDATED as generalization evidence** | Same `EmergentParser` fixture, subject-first SVO. Passes untrained. The `chases == "ACTION"` assertion is separately trivial: `_assign_roles_neural` (`core.py:1032`) assigns ACTION to anything classified VERB with no readout at all. |
| 14 | "At least 60% role accuracy on trained sentences" from **unsupervised** role training (`test_unsupervised_accuracy_above_60`) | `neural_assemblies/tests/test_emergent_parser.py:1060-1080` | 7-E | **INVALIDATED** | All three probe sentences are subject-first SVO. Condition B shows the structural prior alone yields 100%, so the 60% floor is met without `train_unsupervised` running. The companion tests `test_unsupervised_agent_emerges` / `test_unsupervised_patient_emerges` (lines 1010-1021), which assert that role *lexicons* become non-empty, are **UNAFFECTED** — they inspect `role_lexicons` directly and never call `parse`. |
| 15 | `research/SCIENCE_GAPS.md:5` — "A parser that ... **binds** roles compositionally on novel [input]"; `novel_strain` named as "primary path discriminator" | `research/SCIENCE_GAPS.md:5,24` | 7-E | **UNDETERMINED** | The prose is a goal statement, not a result. Whether the `novel_strain` probe suite discriminates depends on whether its 6 probes include any non-subject-first or role-ambiguous item. Settle by: enumerate the `novel_strain` probe sentences and check whether any has the patient in first position. If none do, the discriminator is measuring the prior. |
| 16 | `research/open_questions.md:166` — "EmergentParser implements a 44-area parser with grounded vocabulary, **role binding**, ..." | `research/open_questions.md:166` | 7-E | **SUSPECT** | "Implements" is a capability statement, and the machinery (role areas, `role_lexicons`, mutual inhibition, cross-area projection) genuinely exists. But as currently weighted the machinery is overridden by the prior on every sentence tested in this repo. Reword to "role areas and binding machinery, currently dominated by an AGENT-first positional prior". |
| 17 | Distributional POS classification for **ungrounded / function** words from raw text | `neural_assemblies/tests/test_emergent_parser.py:1240-1300, 1416-1480`; `classify_distributional` (`distributional.py:277`) | 12 | **SUSPECT, low severity, for English corpora only** | The English SVO position prior contributes at most `1.0 * 0.5 = 0.5` per category (`distributional.py:360-364`), against up to 3.0 from verb-relative position and 2.0 from transitions. On the English test corpora it agrees with ground truth, so it inflates rather than flips. Note that lines 379-386 contain a **second**, undocumented English word-order prior (DET-before-NOUN, NOUN-VERB-NOUN) with weights 1.0-2.0 — larger than the documented one. Any claim of language-independent distributional POS induction is **INVALIDATED**; no such claim is currently made. |

### BUG 13 — non-SVO/SOV/VSO generation

| # | Claim | Location | Bug | Verdict | Evidence |
|---|---|---|---|---|---|
| 18 | **"Six-word-order generation (SVO/SOV/VSO/OSV/OVS/VOS) at high transitive accuracy"** | **No such result exists in this repository** | 13 | **NOT FOUND — nothing to invalidate** | Exhaustive search of `*.md`, `*.ipynb`, `*.json`, `*.py` (excluding `.venv`) found the strings `OSV`/`OVS`/`VOS` in exactly one tracked file: `neural_assemblies/nemo/archive/hierarchical_full_v1.py:100-102, 351-363`, an **archived v1 module** with its own six-branch order table. `git grep` across all 200 most recent revisions of every branch returns only that archived file. The live experiment (`test_crosslinguistic_typology.py:80`) tests `["SVO","SOV","VSO"]` only, and the live result file records only those three. If a six-order result was reported elsewhere (a paper draft, a slide, a prior session), it was **not** produced by `emergent/parser_mixins/generation.py` and its provenance must be established separately — see the retraction list. |
| 19 | `test_generate_uses_learned_sov` — SOV-trained parser generates verb-final | `neural_assemblies/tests/test_emergent_parser.py:1553-1570` | 13 | **UNAFFECTED** | Sets `word_order_type = "SOV"` explicitly, hitting the SOV branch at `generation.py:174`, not the fallthrough at 183. |
| 20 | `test_generate_uses_learned_svo` | `neural_assemblies/tests/test_emergent_parser.py:1536-1551` | 13 | **UNAFFECTED** | SVO branch, `generation.py:170`. |
| 21 | Any object-initial generation capability | (none claimed) | 13 | **UNAFFECTED — unreachable by construction** | `word_order_type` is set only by `train_word_order_typological` -> `infer_word_order` (`distributional.py:830-835`), whose label space is `{SVO, SOV, VSO}`. The fallthrough at `generation.py:183` is unreachable unless a caller assigns `word_order_type` by hand. It is a correctness trap for future work, not a current invalidator. |

### BUG 4 — `score_corpus` trains on the evaluation corpus

| # | Claim | Location | Bug | Verdict | Evidence |
|---|---|---|---|---|---|
| 22 | "Overall prediction accuracy should be above chance (1/\|V\|)" — MRR > 0.125 | `neural_assemblies/tests/test_next_token.py:153-175` | 4 | **SUSPECT** | Calls `score_corpus` (line 161) with no `disable_plasticity` anywhere in the file. `score_corpus` -> `predict_next_token` -> `brain.project` with plasticity on, once per position. The 2-sentence test corpus is a subset of the training corpus shape, so the measured MRR is partly adaptation to the scored items. The direction of the claim (above chance) is very likely still true, but the number is not a held-out number. |
| 23 | "MRR should be above chance (1/50)" at 50-word vocabulary scale | `neural_assemblies/tests/test_next_token_scaling.py:236-258` | 4 | **SUSPECT** | Same pattern (`score_corpus` at line 241, no plasticity guard in the file). Here the test corpus IS generated separately (`_generate_test_sentences(10)`), so it is nominally held out — but it stops being held out the moment `score_corpus` runs, and the result is order-dependent across the 10 sentences. |
| 24 | "Prediction accuracy stratified by next-word class: DET easier than NOUN" | `neural_assemblies/tests/test_next_token_scaling.py:260+` (`_score_by_word_class`) | 4 | **SUSPECT** | Uses `predict_next_token` in a loop, same plasticity-on drive. A *comparative* claim is more robust to uniform contamination than an absolute one, but the contamination is not uniform: earlier-scored positions get less adaptation than later ones, so the comparison is confounded with corpus order. |
| 25 | Notebook cell displaying `score_corpus` output as a results table | `examples/notebooks/volume-08-prediction-and-erp-signals/01_toy_next_token_and_signal_map.ipynb` cell 27 | 4 | **SUSPECT** | Scores the same 2-sentence corpus that `train_on_corpus` just trained on, with plasticity on. The notebook presents the number without a caveat. |
| 26 | `next_token.py` module docstring "HOW TO READ THE NUMBERS" | `neural_assemblies/assembly_calculus/next_token.py:15-43` | 4 | **UNAFFECTED — this is the correct disclosure** | Item 1 of that docstring states the defect precisely and gives the remedy (`brain.disable_plasticity = True` around the call). No caller in the repo follows it. |

### BUG 8 — `compute_anchored_instability` plasticity precondition

Every located caller **does** set `brain.disable_plasticity = True` around the
measurement, and every call site falls inside the guarded window. Verified line
by line:

| # | Claim | Location | Bug | Verdict | Evidence |
|---|---|---|---|---|---|
| 27 | "Binding P600: does anchored instability produce graded P600 signals? Validated" | `research/experiments/README.md:75`; `research/experiments/primitives/test_binding_p600.py` | 8 | **UNAFFECTED** | Guard at line 196 (`= True`) / 216 (`= False`); the `compute_anchored_instability` calls are at lines 207 and 213, inside the window. |
| 28 | "Composed ERP: N400/P600 double dissociation" | `research/results/primitives/RESULTS_composed_erp.md`; `research/experiments/primitives/test_composed_erp.py` | 8 | **UNAFFECTED** | Guard at 252 / 344; measurement calls at 282, 318, 330, 342 — all inside. |
| 29 | "Incremental ERP: graded developmental ERP curves" | `research/experiments/primitives/test_incremental_erp.py` | 8 | **UNAFFECTED** | Guard at 330 / 333 wraps `_measure_erps`, which is the function containing the three `compute_anchored_instability` calls at 242/250/258. Training deliberately runs with plasticity ON *outside* the guard, which is the intended design. |
| 30 | ERP dynamics diagnostics, `anchored_instability` fields in results | `research/experiments/primitives/diagnose_erp_dynamics.py:494`; `research/results/primitives/diagnose_erp_dynamics_*_p600_metrics_quick.json` | 8 | **UNAFFECTED** | Guard at 400 / 497 encloses line 494. |
| 31 | Free-form relative-clause P600 feedback results | `research/experiments/lib/freeform.py:469`; `research/results/freeform_rc_*.json` | 8 | **UNAFFECTED** | Explicit `disable_plasticity = True` at 468, `False` at 473, tightly wrapping `measure_p600`. |
| 32 | `p600_syntactic` results in the N400 claim index | `research/experiments/applications/test_p600_syntactic.py`; `research/claims/index.json:17` | 8 | **UNAFFECTED — different function** | Uses `measure_p600_settling` (import at line 87), not `compute_anchored_instability`. Whether *that* function has the same unchecked precondition was not audited and is listed as follow-up work below. |
| 33 | All `n400_*` claims in `research/claims/index.json` | `research/claims/N400_GLOBAL_ENERGY.md` and the 6 `test_n400_*` experiments | 4, 8 | **UNAFFECTED** | The N400 claim is built on pre-k-WTA global energy, not on `assembly_calculus/next_token.py` and not on `compute_anchored_instability`. Different module entirely. |

BUG 8 is therefore a **latent hazard with no currently invalidated claim**. The
missing assertion should still be added, because the failure is silent and
compresses the effect toward zero — a future caller that forgets the guard will
see a *weakened* dissociation and may read it as a null result.

### BUG 3 — `_flip_k_split` leaves plasticity on

| # | Claim | Location | Bug | Verdict | Evidence |
|---|---|---|---|---|---|
| 34 | "A fair coin (bias=0.5) should produce roughly 50/50 over many flips" — 40 sequential flips on one coin | `neural_assemblies/tests/test_pfa.py:40-56` | 3 | **SUSPECT** | `coin.flip(...)` at line 49 uses the default `mode="k_split"`. All 40 flips reuse one `RandomChoiceArea`, so each flip potentiates the basin it landed in. The assertion is weak (both outcomes must appear at least once) and survives substantial drift, so the test still passes — but it cannot be cited as evidence that the flip distribution is fair or that flips are i.i.d. |
| 35 | "A strongly biased coin (bias=0.9) should mostly produce 0" — 30 sequential flips | `neural_assemblies/tests/test_pfa.py:58-70` | 3 | **SUSPECT** | Same default mode, same shared coin. Drift toward the 0 basin *reinforces* the expected direction, so this test is biased toward passing for the wrong reason. |
| 36 | `COIN24-M02` "Stochastic PFA branch selection", status **pinned** to `test_pfa_stochastic_both_targets` | `research/literature/REPRODUCTION_MATRIX.md:225` | 3 | **SUSPECT** | Runs through `PFANetwork`, which uses `flip` at its default mode (`pfa.py:156`). "Both targets are reachable" is a weak enough property to survive drift, but a *distributional* parity claim against the Dabagia et al. coin-flipping paper would not be. |
| 37 | `COIN24-M01` "PFA from transition frequency traces", status pinned | `research/literature/REPRODUCTION_MATRIX.md:224` | 3 | **UNDETERMINED** | Depends on whether `test_markov_transition_frequencies_from_traces` measures *frequencies* (drift-sensitive) or only reachability (drift-robust), and on whether it uses a fresh coin per trace. Settle by reading that test and re-running it with `mode="compete"`; if the measured frequencies move, the pin is SUSPECT. |
| 38 | `test_literature_golden.py:84` golden fixture with `flip_mode="compete"` | `neural_assemblies/tests/test_literature_golden.py:84` | 3 | **UNAFFECTED** | Explicitly selects `compete`, which disables plasticity for the duration of the flip (`pfa.py:145-152`). This is the correct instrument and the only place in the repo that uses it. |
| 39 | Notebook: "PFA sample outcomes after five days" bar chart over 8 seeds | `examples/notebooks/volume-02-memory-and-computation/02_fsm_and_pfa.ipynb` | 3 | **SUSPECT** | Calls `pfa.reset()` between seeds — but `reset()` restores the *state pointer*, not the connectome. Whether the trained weights are also reset was not verified; if not, the 8 trajectories are not independent samples and the bar chart misrepresents the transition probabilities it appears to illustrate. The README for that volume claims "independent samples from a small `PFANetwork`" (`volume-02.../README.md:19`), which is the specific wording at risk. |

---

## Prioritized action list

### Tier 1 — Retract or re-run before any external use

1. **Retract the role-accuracy half of the cross-linguistic typology result.**
   `research/results/applications/crosslinguistic_typology_20260209_171109_quick.json`:
   every `*_role_accuracy` and `overall_role_accuracy` field, and hypothesis H5.
   The order-accuracy half survives and should be reported separately.
   The re-run must include **at least one object-initial typology (OSV, OVS, or
   VOS)** and an **untrained-parser control arm**. Without the control arm the
   re-run proves nothing, because conditions B and D above show the trained and
   untrained numbers are identical.

2. **Downgrade `NEMO25-M02` from `pinned` to `missing`** in
   `research/literature/REPRODUCTION_MATRIX.md:237`. The three pinned sentences
   pass on an untrained parser.

3. **Establish the provenance of any six-word-order generation result.**
   No such result exists in this repository's tracked history — the only
   six-order code is `neural_assemblies/nemo/archive/hierarchical_full_v1.py`,
   an archived v1 module. If such a figure or table was ever reported, it either
   (a) came from that archived module, in which case it must be re-derived
   against the live code before being cited, or (b) came from
   `emergent/.../generation.py`, in which case OSV/OVS/VOS were silently
   rendered as SVO (BUG 13) and the transitive-accuracy number is an artifact.
   Determine which before the result is used anywhere.

4. **Fix `docs/scientific_status.md:19`.** Remove "role binding" from the list of
   package-defensible behaviors, or qualify it as "role-area machinery exists;
   role *assignment* is currently determined by a positional prior."

### Tier 2 — Re-run under corrected protocol

5. **All `score_corpus` numbers** (claims 22-25). Wrap the call in
   `brain.disable_plasticity = True` / `False` and re-run. If the accuracy drops
   materially, the previously reported numbers were measuring adaptation.
   The class-stratified comparison (claim 24) additionally needs shuffled-order
   replicates to separate the effect from corpus-order confounding.

6. **PFA flip-distribution tests** (claims 34-36). Re-run `test_pfa.py` with
   `mode="compete"` and compare. If the fair-coin balance or the biased-coin
   ratio shifts, the k_split numbers were drift, and the tests should be pinned
   to `compete` (the k_split default can remain for reproducing historical
   numbers, as the docstring at `pfa.py:145-152` already argues).

7. **Verify `PFANetwork.reset()` semantics** for claim 39. If it does not reset
   the connectome, correct the volume-02 README wording ("independent samples")
   and either reconstruct the network per seed in the notebook or relabel the
   chart as a single drifting trajectory.

### Tier 3 — Reword, do not re-run

8. `research/open_questions.md:166` — qualify "role binding" (claim 16).
9. `examples/notebooks/volume-03-language/01_...ipynb` — add one sentence noting
   the `train_roles` stage does not feed the parse table (claim 5).
10. `REPRODUCTION_MATRIX.md:218` (`ORG23-D01`) — make the existing
    "Document limits in protocol" note concrete: state that `NemoParser` role
    output is a Python SVO template.
11. Notebook `volume-08` cell 27 — add the `next_token.py` "HOW TO READ THE
    NUMBERS" caveat next to the score table (claim 25).

### Tier 4 — Latent hazards with no current invalid claim

12. Add the missing `assert brain.disable_plasticity` to
    `compute_anchored_instability` (BUG 8). No current result is wrong, but the
    failure mode is silent and *understates* the effect, so a future omission
    reads as a null result rather than an error.
13. BUG 6 (`assign_role`) is unreachable from any current claim (only fires for
    sentences under 3 words, which nothing in the repo does). Leave or fix, but
    do not treat as urgent.
14. BUG 13's fallthrough is unreachable while `infer_word_order` has a
    three-label output space. It becomes live the moment object-initial
    typologies are added — which is exactly what Tier 1 item 1 requires. Fix it
    **before** running that re-run.

### Follow-up not covered by this audit

15. `measure_p600_settling` (`research/experiments/metrics`) carries the same
    plasticity precondition risk as `compute_anchored_instability` and was not
    audited. It backs the `p600_syntactic` result in `research/claims/index.json`.
16. The second, undocumented English word-order prior at
    `distributional.py:379-386` (DET-before-NOUN, NOUN-VERB-NOUN frames, weights
    1.0-2.0) is **larger** than the documented `_CATEGORY_POSITION_PROFILES`
    prior that BUG 12 names. It should be documented with the same caveat.
17. The `novel_strain` probe suite (`research/SCIENCE_GAPS.md:24`) needs its six
    probes enumerated to determine whether any is non-subject-first (claim 15).

---

## Method notes

* Verdicts on BUG 7 vs BUG 7-E required distinguishing two parsers with the same
  method name. `NemoParser.parse` is `assembly_calculus/parser.py:301`.
  `EmergentParser.parse` is `assembly_calculus/emergent/parser_mixins/core.py:1102`.
  Every claim was traced to one or the other by reading the constructing fixture,
  not by name.
* The trained-vs-untrained comparison (conditions A-G) was run at reduced scale
  (n=2000, k=40 rather than the experiment's n=10000, k=100) for speed. The
  effect is structural — a `+1.2` vs `+0.6` prior against a bounded-[0,1] lexical
  score — so it does not depend on scale. A full-scale replication with the
  experiment's own parameters would make the retraction airtight and is cheap
  (roughly two minutes per condition).
* No library code, and nothing under `research/experiments/`, was modified.
