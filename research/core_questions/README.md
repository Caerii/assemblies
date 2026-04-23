# Core Research Questions

This directory contains the **fundamental scientific questions** driving this research. Each question gets its own subdirectory with a complete investigation.

## Philosophy

**Start with questions, not papers.** Papers emerge naturally when questions are answered with validated evidence.

## Question Template

Each research question should have:

```
QXX_question_name/
├── hypothesis.md              # What exactly are we claiming?
├── theoretical_basis.md       # Mathematical/theoretical foundation
├── experiments.md             # How do we test this?
├── results.md                 # What have we found?
├── analysis.md                # Interpretation of results
└── conclusions.md             # What can we claim?
```

## Curated Questions

This directory is no longer only a template surface. A small set of questions
has been curated because the repo already has enough evidence to support a
question-level narrative without overclaiming.

Machine-readable inventory:

- `index.json`
- `validate_index.py`

Current curated questions:

- **Q01: Assembly Stability**
  - `Q01_assembly_stability/`
  - recurrent stim+self stability, with explicit limits on scope
- **Q03: Scaling Laws**
  - `Q03_scaling_laws/`
  - empirical scaling evidence in the tested `k = sqrt(n)` regime
- **Q20: Competition and Distinctiveness**
  - `Q20_competition_distinctiveness/`
  - same-brain distinctiveness with mechanism caveats
- **Q22: N400 as Global Pre-k-WTA Energy**
  - `Q22_n400_global_energy/`
  - question-first wrapper around the strongest formalized claim

## Evidence-Backed But Not Yet Curated

Some questions have nontrivial evidence but are not yet clean enough to promote
into full question directories.

- **Q07: Sparsity Measurements**
  - useful parameter-alignment evidence exists
  - the writeup should stay narrow and not imply a full neural-data comparison
- **Q12: Learning Rules**
  - the quick retrieval result is promising
  - the broader learning-rules claim still needs a cleaner bounded experiment

## Open Tracker vs Curated Questions

- `open_questions.md` remains the broad tracker of all questions and ideas.
- `core_questions/` is intentionally narrower. A question should only be added
  here when the repo can already tell an honest, artifact-backed story about it.

## Creating a New Question

1. Check `index.json` first to make sure the question is not already curated.
2. If it is new, create directory: `QXX_descriptive_name/`
3. Copy the standard question file layout
4. Fill out `hypothesis.md` first
5. Design experiments to test the hypothesis
6. Run experiments and document results
7. Add the question to `index.json` once it is curated enough to defend

## Status Tracking

### Not Started
*Questions identified but not yet investigated*

### In Progress
*Questions under active investigation*

### Completed
*Questions with validated conclusions*

### Abandoned
*Questions that didn't pan out (document why!)*

---

**Remember: Good questions are more valuable than quick answers.**
