# P8 draft: the budget of a sequence area

The manuscript for paper P8 of [../../../plans/PAPERS.md](../../../plans/PAPERS.md):
how much one assembly area holds as sequences, how it fails, and what a program
of areas must budget for.

- `main.tex` -- the manuscript, organised by claims. Every number carries its
  evidence: REGISTERED (an amendment of
  [PREREG_refraction_memory.md](../../../notes/memory/PREREG_refraction_memory.md)),
  EXPLORATORY (a probe under
  [../../../notes/memory/probes/2026-10-08/](../../../notes/memory/probes/2026-10-08/README.md)),
  or PROVED within a stated model.
- `figures/` -- built, never edited by hand:

      python research/experiments/memory_budget_figures.py

  reads the committed run records (Amendments 37-43) and probe logs and writes
  the five figures, `crossings.json` and `crossings_table.tex` (bootstrap
  intervals over brains beside the ladder points that bracket each crossing).
- Build: `tectonic main.tex` from this directory (or `pdflatex` + `bibtex`); the
  bibliography is the shared
  [../../_shared_assets/bibliography/references.bib](../../_shared_assets/bibliography/references.bib).

The full chronological record -- every probe, every revision of every
conjecture -- is the theory notebook,
[../../../theory/assembly_statmech.tex](../../../theory/assembly_statmech.tex).
This draft is what survives into a paper; the notebook is where it came from.

Open before submission: Amendment 43 (the hazard's predictions); the author
list; the link law registered at new area pairs; the universality test (E/I
dynamics in place of k-WTA).
