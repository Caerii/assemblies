# Assembly Calculus Literature

This page is the field map for Assembly Calculus (AC): every major paper,
what it contributes, where it lives in this repo, and what is still missing.

For BibTeX entries use
[research/papers/_shared_assets/bibliography/references.bib](../research/papers/_shared_assets/bibliography/references.bib).

For machine-readable parity tracking use
[research/literature/index.json](../research/literature/index.json).

For **claim-level reproduction** (every paper result → protocol → test → golden),
use [research/literature/REPRODUCTION_MATRIX.md](../research/literature/REPRODUCTION_MATRIX.md)
and [reproduction_matrix.json](../research/literature/reproduction_matrix.json).

The package implements and experiments with these ideas. It is not the proof
of the underlying theorems. Cite the papers directly for theoretical claims.

---

## Reference Implementations

| Implementation | URL | Role |
|----------------|-----|------|
| **dmitropolsky/assemblies** | <https://github.com/dmitropolsky/assemblies> | Canonical Python reference (parser, learner, Turing sim) |
| **AssemblyCalculus.jl** | <https://github.com/djpasseyjr/AssemblyCalculus.jl> | Fast Julia reimplementation |
| **This repo** | <https://github.com/Caerii/assemblies> | Extended platform: `neural_assemblies`, GPU, EmergentParser, research |

Target state: **pinned parity tests** against `dmitropolsky/assemblies` for core
ops, plus **paper-protocol tests** for each tier-1 and tier-2 result.

---

## Tier 1 — Foundational Assembly Calculus

### Papadimitriou & Vempala (2019) — ITCS

**Random Projection in the Brain and Computation with Assemblies of Neurons**

- Link: <https://doi.org/10.4230/LIPIcs.ITCS.2019.57>
- Contributes: convergence proofs for assembly projection; associate, merge,
  append; theoretical foundation for PNAS 2020
- **Repo status:** `partial` — core ops exist; no ITCS-specific parity tests

### Papadimitriou (2019) — CCNeuro

**A Calculus for Brain Computation**

- Link: <https://ccneuro.org/2019/proceedings/0000998.pdf>
- Contributes: conceptual precursor (RP&C primitive, project/associate/merge)
- **Repo status:** `partial` — cite ITCS 2019 for formal results

### Papadimitriou et al. (2020) — PNAS

**Brain Computation by Assemblies of Neurons**

- Link: <https://doi.org/10.1073/pnas.2001893117>
- Contributes: project, reciprocal project, associate, merge, pattern
  completion, separation; language production sketch
- **Repo status:** `implemented`
- Package: `neural_assemblies/assembly_calculus/ops.py`,
  `tests/test_assembly_calculus.py`
- **Parity:** `tests/test_cross_repo_parity.py` + `reference_pnas_golden.json` (±0.02 vs dmitropolsky/assemblies)

### Dabagia, Vempala, Papadimitriou (2022) — COLT

**Assemblies of Neurons Learn to Classify Well-Separated Distributions**

- Links: <https://arxiv.org/abs/2110.03171>,
  <https://proceedings.mlr.press/v178/dabagia22a.html>
- Contributes: online classification; assembly-per-class; MNIST demonstration
- **Repo status:** `partial`
- **Gap:** no maintained COLT/MNIST protocol; no explicit `learn()` API

### Dabagia, Papadimitriou, Vempala (2024/2025) — Sequences

**Computation with Sequences of Assemblies in a Model of the Brain**

- Links: <https://arxiv.org/abs/2306.03812> (full),
  <https://proceedings.mlr.press/v237/dabagia24a.html> (ALT abstract),
  *Neural Computation* 37(1):193–233 (journal)
- Contributes: sequence memorization, ordered recall, LRI, FSM learning,
  Turing completeness argument
- **Repo status:** `partial`
- Package: `sequence_memorize`, `ordered_recall`, `FSMNetwork`, `PFANetwork`,
  `tests/test_sequences.py`, `tests/test_lri.py`
- **Gaps:** FSM-from-sequences not end-to-end; no runnable full TM; Turing
  sims are exploratory (`simulation/turing_simulations.py`)

---

## Tier 2 — NEMO / Language / Cognition

### Mitropolsky, Collins, Papadimitriou (2021) — TACL

**A Biologically Plausible Parser**

- Links: <https://doi.org/10.1162/tacl_a_00432>,
  <https://arxiv.org/abs/2108.02189>
- Contributes: fiber-gated dependency parser; Russian variant; readout
- **Repo status:** `partial` — explicit LEX → sparse grammar; fiber readout with lexical cache

### Mitropolsky et al. (2022) — NALOMA

**Center-Embedding and Constituency in the Brain…**

- Links: <https://arxiv.org/abs/2206.13217>,
  <https://aclanthology.org/2022.naloma-1.4/>
- Contributes: constituency, center-embedding, working-memory replay
- **Repo status:** `partial` — `DEP_CLAUSE` + comma replay in maintained parser;
  `DEP-VERB` fiber readout pinned in `test_literature_parity.py`
- **Gap:** paper-scale depth-2 sentences need expanded lexicon (see `research/experiments/`)

### d'Amore et al. (2022) — AAAI

**Planning with Biological Neurons and Synapses**

- Links: <https://doi.org/10.1609/aaai.v36i1.19875>,
  <https://arxiv.org/abs/2112.08186>
- Contributes: blocks-world planning as large AC program
- **Repo status:** `partial` — `programs/planning.py` with STRIPS `verify_plan` + neural FSM scaffold

### Mitropolsky & Papadimitriou (2023)

**The Architecture of a Biologically Plausible Language Organ**

- Link: <https://arxiv.org/abs/2306.15364>
- Contributes: NEMO organ; LRI replaces imperative control; differential lex,
  roles, fibers
- **Repo status:** `partial` — `NemoParser`, `nemo/`, `research/nemo/`

### Dabagia et al. (2024)

**Coin-Flipping In The Brain: Statistical Learning with Neuronal Assemblies**

- Link: <https://arxiv.org/abs/2406.07715>
- Demo: <http://dabagia.org/nemo/coinflipping/>
- Contributes: conditional probabilistic assemblies; Markov chains from sequences
- **Repo status:** `partial` — `PFANetwork`, `RandomChoiceArea`, `programs/markov_coin.py`

### Mitropolsky & Papadimitriou (2025)

**Simulated Language Acquisition in a Biologically Realistic Model of the Brain**

- Link: <https://arxiv.org/abs/2507.11788>
- Contributes: grounded POS emergence, role binding, word order, curriculum
- **Repo status:** `partial` — emergent parser + agent stack through JSON tool output (Phase 4)
- Package: `assembly_calculus/emergent/`, `tests/test_emergent_agent.py` (24 tests)
- **Gap:** multi-tool plans (Phase 5); not LLM-scale chat

---

## Tier 3 — Field Extensions (2025–2026)

### Onasch et al. (2025) — bioRxiv

**Assembly-based Computations through Contextual Dendritic Gating of Plasticity**

- Link: <https://doi.org/10.1101/2025.07.22.666089>
- Contributes: dendrite-level gated plasticity; catastrophic forgetting mitigation
- **Repo status:** `not_started`

### Hoff et al. (2026) — arXiv

**Formation of Artificial Neural Assemblies by Biologically Plausible Inhibition
Mechanisms** (E%-WTA)

- Link: <https://arxiv.org/abs/2603.12416>
- Contributes: variable assembly size from E/I ratios; improved recovery vs
  fixed k-WTA
- **Repo status:** `partial` — `EPercentPolicy` on numpy + `torch_sparse`; `test_epwta_gpu.py`

### Ting, Sethu, Dang (2026) — arXiv

**Beyond Deep Learning: Speech Segmentation and Phone Classification with
Neural Assemblies**

- Link: <https://arxiv.org/abs/2603.16923>
- Contributes: continuous speech → assembly encoding; boundary detection;
  per-class recurrent areas
- **Repo status:** `not_started`

### Kopadi & Kalles (2026) — arXiv

**Causal Learning with Neural Assemblies** (DIRECT)

- Link: <https://arxiv.org/abs/2604.26919>
- Contributes: directional causal binding; synaptic asymmetry readout;
  Pearl do-calculus validation
- **Repo status:** `retracted` - the former overlap readout was
  insensitive to wiping its learned fiber; `programs/direct.py` refuses it

---

## Implementation Parity Matrix

| Paper | Year | Status | Primary package path |
|-------|------|--------|----------------------|
| ITCS 2019 | 2019 | partial | `assembly_calculus/`, `compute/` |
| PNAS 2020 | 2020 | **implemented** | `assembly_calculus/ops.py` |
| COLT 2022 | 2022 | partial | `ops.learn_assembly`, `programs/learn.py` |
| Sequences | 2024/25 | partial | `programs/fsm_learn.py`, `programs/tm_demo.py` |
| TACL parser | 2021 | partial | `test_tacl_parser_suite.py` (EN+RU) |
| Center-embed | 2022 | partial | `test_tacl_parser_suite.py`, `test_literature_parity.py` |
| AAAI planning | 2022 | partial | `programs/planning.py` (`BlocksWorldPlanner`) |
| Language organ | 2023 | partial | `parser.py`, `nemo/` |
| Coin-flipping | 2024 | partial | `programs/markov_coin.py` |
| Lang acquisition | 2025 | partial | `emergent/` |
| Dendritic gating | 2025 | not_started | — |
| E%-WTA | 2026 | partial | `EPercentPolicy`, `torch_engine`, `test_epwta_gpu.py`, size-cap parity |
| Speech AC | 2026 | not_started | — |
| Causal DIRECT | 2026 | retracted | `programs/direct.py` refusal boundary |

---

## Recommended Implementation Order

This is the order that maximizes **literature parity** before chasing extensions:

1. **Parity test suite** — `tests/test_literature_parity.py` + `research/literature/parity/PROTOCOLS.md`
2. **Cross-engine projection** — explicit↔sparse both directions (readout path done)
3. **TACL parser** — expand sentence suite; Russian parity
4. **Sequences / TM** — scale `MinimalTMDemo` toward paper construction
5. **Coin-flipping** — pin Markov statistics vs paper demo
6. **E%-WTA** — GPU engine parity
7. **DIRECT** — do-calculus validation protocol
8. ~~**NEMO 2025 curriculum**~~ — parity test in `test_literature_parity.py` (notebook optional)
9. **Planning** — blocks-world from AAAI 2022
10. **Speech encoder** — continuous input path (Ting et al. 2026)
11. **Dendritic gating** — compartment plasticity (Onasch et al. 2025)

---

## Background Literature

See [references.md](references.md) for Hebb, Buzsáki, Hopfield, Olshausen &
Field, and other neuroscience context cited in the papers above.
