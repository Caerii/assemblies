# Getting Started with the Research Tree

This guide explains how to work in `research/` without blurring the line
between exploratory work, defended results, and package-level claims.

## Read First

Before adding anything new, read these in order:

1. `open_questions.md`
2. `core_questions/index.json`
3. `claims/index.json`
4. `registry.json`

That tells you:

- what the broad question surface looks like
- which questions are already curated
- what the repo can currently defend as claims or evidence summaries
- which experiment suites already exist

## Research Layout

```text
research/
|-- open_questions.md
|-- core_questions/
|-- experiments/
|-- results/
|-- claims/
|-- papers/
`-- registry.json
```

Use those directories deliberately:

- `open_questions.md` for the broad tracker
- `core_questions/` for mature, evidence-linked question directories
- `experiments/` for protocols, code, and suite-specific infrastructure
- `results/` for outputs and analysis artifacts
- `claims/` for bounded statements backed by evidence
- `papers/` for writing that emerges from claims

## Starting a New Line of Work

### 1. Decide whether the question is new

Check `open_questions.md` and `core_questions/index.json` first.

If the topic is already represented, extend the existing question rather than
creating a near-duplicate directory.

### 2. Write the question before the experiment

If the topic is genuinely new, start by documenting:

- the hypothesis
- why it matters
- what would count as supporting evidence
- what would falsify or weaken it

If the question is mature enough for a curated directory, add a new
`QXX_*` directory under `core_questions/`.

### 3. Register the experimental surface

Before adding ad hoc scripts, check `registry.json` and
`experiments/MANIFEST_TEMPLATE.json`.

If you are creating a new suite or experiment family:

- update `registry.json`
- use the manifest template where appropriate
- keep the suite naming consistent with the rest of the tree

### 4. Run experiments and store artifacts deliberately

- keep experiment code under `experiments/`
- keep generated outputs under `results/`
- avoid scattering one-off outputs across the repo

### 5. Promote evidence carefully

When a result becomes stable and bounded:

- update `claims/index.json`
- add or expand the corresponding claim material under `claims/`
- state limitations explicitly

Do not promote a broad theory claim when you only have a narrow experimental
result.

## Practical Validation

Run the repo validators after updating the research indexes:

```bash
uv run python research/experiments/infrastructure/validate_registry.py
uv run python research/claims/validate_index.py
uv run python research/core_questions/validate_index.py
```

## Paper Workflow

Paper writing is downstream of the research tree.

Only move into `papers/` when:

- the question is already clear
- the evidence is already collected
- the claim surface is already bounded

For paper-specific structure and LaTeX workflow, see `papers/README.md`.

## Common Failure Modes

- Creating a paper draft before the claim exists
- Creating a new question directory for something already tracked elsewhere
- Treating an evidence summary as if it were a fully defended claim
- Leaving experimental outputs outside `results/`
- Making package-facing statements from research-only artifacts

## Good Working Pattern

1. Frame the question.
2. Register the suite.
3. Run the experiment.
4. Save the result artifact.
5. Update the claim inventory.
6. Only then start writing a paper.

That order keeps the repo scientifically honest and easier to maintain.
