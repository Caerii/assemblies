# Research Organization

This directory is the research-first side of the repository.

The installable package lives under `neural_assemblies/`. The material here is
for scientific questions, experiments, results, claims, and paper scaffolding
that should not be confused with the default package contract.

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

The current curated question set includes:

- `Q01` assembly stability
- `Q03` scaling laws
- `Q20` competition and distinctiveness
- `Q22` N400 as global pre-k-WTA energy

These live under `core_questions/` and are indexed in
`core_questions/index.json`.

### Active suites

The suite registry currently tracks:

- `applications`
- `biological_validation`
- `distinctiveness`
- `information_theory`
- `infrastructure`
- `primitives`
- `stability`
- `vocab`

### Claims status

The current claims inventory distinguishes between fully formalized claims and
claim-ready evidence summaries. See `claims/index.json` for the exact status.

## Quality Rules

- Questions may be exploratory.
- Experiments must be reproducible enough to rerun.
- Results should state caveats, not just wins.
- Claims should only say what the evidence supports.
- Papers should be downstream of claims, not upstream of them.

## Validators

Use the current repo tooling to check the research indexes:

```bash
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

That order gives you the broad tracker, the curated subset, the current claim
surface, and the practical workflow.
