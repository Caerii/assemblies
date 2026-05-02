# Core Research Questions

Use this directory for research questions that are mature enough to have their
own evidence-linked narrative.

The broad tracker is `../open_questions.md`. This directory is narrower: a
question belongs here only when the repo can already connect it to experiments,
results, analysis, and caveats.

## Directory Shape

```text
QXX_question_name/
|-- hypothesis.md
|-- theoretical_basis.md
|-- experiments.md
|-- results.md
|-- analysis.md
`-- conclusions.md
```

## Curated Questions

The curated set is indexed in `index.json` and checked by
`validate_index.py`.

Current question directories:

- `Q01_assembly_stability/`
  Recurrent stim+self stability, with explicit limits on scope.
- `Q03_scaling_laws/`
  Empirical scaling evidence in the tested `k = sqrt(n)` regime.
- `Q20_competition_distinctiveness/`
  Same-brain distinctiveness with mechanism caveats.
- `Q22_n400_global_energy/`
  Question-first wrapper around the strongest formalized claim.

## Evidence That Is Not Yet Curated

Some results are useful but not ready for a full question directory.

- Q07 has parameter-alignment evidence, but not a full neural-data comparison.
- Q12 has a promising quick retrieval result, but the broader learning-rules
  claim still needs a cleaner experiment.

Keep those in `open_questions.md` until the evidence can support a bounded
question narrative.

## Adding A Question

1. Check `index.json`.
2. Check `../open_questions.md`.
3. Create `QXX_descriptive_name/` only if the question is mature enough.
4. Fill out `hypothesis.md` before writing conclusions.
5. Link experiments and result artifacts.
6. Add caveats before promoting the question as completed.
7. Update `index.json` and run:

```bash
uv run python research/core_questions/validate_index.py
```

## Standard

Good question docs state what was tested, what happened, what it means, and
what remains unresolved.
