# Research Organization

Research work lives here.

The installable package lives under `neural_assemblies/`. The material here is
for scientific questions, experiments, results, claims, and paper scaffolding
that should not be confused with released package behavior.

## Principle

Organize research by questions and evidence, not by hoped-for papers.

Papers should emerge from validated claims. They should not be the structure
that everything else bends around.

## Research Layers

### Broad tracker

- `open_questions.md`
  The wide inventory of active, incomplete, or still-uncertain questions.

### Curated question set

- `core_questions/index.json`
  Machine-readable inventory of the smaller set of question directories that
  already map cleanly onto evidence.
- `core_questions/`
  Per-question directories with `hypothesis.md`, `theoretical_basis.md`,
  `experiments.md`, `results.md`, `analysis.md`, and `conclusions.md`.

### Experiment and suite inventory

- `registry.json`
  Canonical map of experiment families, code paths, result directories, and
  recommended entry points.
- `experiments/`
  Experimental code, manifests, infrastructure, and suite-level notes.

### Results and claims

- `results/`
  Result artifacts produced by experiments.
- `claims/index.json`
  Inventory of formalized claims and claim-ready evidence summaries.
- `claims/`
  Claim documents and supporting material.

### Literature inventory

- `literature/index.json`
  Canonical Assembly Calculus bibliography with per-paper implementation status,
  package module mapping, and gap list.
- `literature/README.md`
  How to use the literature index when implementing paper features.
- [../docs/literature.md](../docs/literature.md)
  Human-readable field map and parity matrix.

### Papers

- `papers/`
  LaTeX infrastructure and paper drafting space that should only be used after
  claims are bounded and evidence-backed.

## Typical Workflow

1. Start with `open_questions.md`.
2. Check whether the topic already exists in `core_questions/index.json`.
3. If it is mature enough, create or update a curated question directory.
4. Register or update the relevant experiment suite in `registry.json`.
5. Run experiments and store outputs under `results/`.
6. Promote bounded evidence into `claims/`.
7. Draft papers from validated claims.

## Current State

### Curated questions

The curated question set includes:

- `Q01` assembly stability
- `Q03` scaling laws
- `Q20` competition and distinctiveness
- `Q22` N400 as global pre-k-WTA energy

These live under `core_questions/` and are indexed in
`core_questions/index.json`.

### Active suites

The suite registry tracks:

- `applications`
- `biological_validation`
- `distinctiveness`
- `information_theory`
- `infrastructure`
- `primitives`
- `stability`
- `vocab`

### Claims status

The claims inventory distinguishes between fully formalized claims and
claim-ready evidence summaries. See `claims/index.json` for the exact status.

## Quality Rules

- Questions may be exploratory.
- Experiments must be reproducible enough to rerun.
- Results should state caveats, not just wins.
- Claims should only say what the evidence supports.
- Papers should be downstream of claims, not upstream of them.

## Validators

Use the repo tooling to check the research indexes:

```bash
uv run python research/literature/validate_index.py
uv run python research/experiments/infrastructure/validate_registry.py
uv run python research/claims/validate_index.py
uv run python research/core_questions/validate_index.py
```

## Where To Start

If you are new to this tree:

1. Read `open_questions.md`.
2. Read `core_questions/index.json`.
3. Read `claims/index.json`.
4. Read `GETTING_STARTED.md`.

That order gives you the broad tracker, the curated subset, the claim
inventory, and the practical workflow.


## Source identity

[The shared runner](runner.py) records `git_commit`, `source_sha256`, and
`source_inventory` before measurement and rejects a changed identity afterward.
`source-inputs-v2` hashes Git-discovered tracked and nonignored untracked code,
compiler headers, build scripts/configuration, Lean/Dafny specifications, the
Lean toolchain pin and manifest, and JSON contracts under `neural_assemblies/ir`.
The inventory policy identifier is also included in the digest. Older records
without `source_inventory` used the narrower original suffix inventory; their
hashes must not be interpreted as covering these additional inputs.

Results JSON is excluded so publishing observations does not invalidate its own
run. Registrations and explicitly declared `input_artifacts` are hashed separately.
Other data/configuration JSON must be declared as an input artifact. This is a
start/end repository-content guard, not a hermetic execution certificate: it does
not capture installed binaries, ignored sources, external
data, or changes made and reverted between the two checks. A completed record
remains UNJUDGED (or VOID for smoke), never automatically adopted evidence.

The constructed negative in
[test_research_runner.py](../neural_assemblies/tests/test_research_runner.py)
changes a Lean file during measurement and requires a retained failure record
with no completed results artifact. Separate cases cover added/modified compiler
and verification inputs and show that emitting results leaves identity unchanged.


## Environment identity

[The shared fingerprint](../neural_assemblies/core/environment.py) covers exact current
values of `ASSEMBLIES_*`, `NEURAL_ASSEMBLIES_*`, and `EMERGENT_*`. Run schema 2
requires an `environment` record naming `repository-environment-v1` and mapping
variable names to SHA-256 digests. No raw environment values are persisted. The
runner compares this snapshot after measurement and retains failure if it changed.
The evidence validator checks its structure; schema 1 remains readable as historical
records that did not require environment identity.

The parser training cache consumes the same fingerprint, excluding only its cache
location (`ASSEMBLIES_BACKBONE_CACHE`) and separately keyed calibration mode
(`EMERGENT_ERP_FAST`). Backend switches under `NEURAL_ASSEMBLIES_*` now invalidate
both memory and disk reuse. Experiment records exclude neither of those settings.

This fingerprint establishes equality, not reconstruction or secrecy of low-entropy
settings. Reproduction still requires explicit resolved protocol parameters.
It describes the current process environment, not values previously captured at
module import, external thread/CUDA settings, installed binary versions, or changes
reverted before the final check. Resolving every semantic switch into an immutable
model configuration remains open; this guard does not substitute for that work.
