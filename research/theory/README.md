# Theory

`assembly_statmech.tex` — the substrate as a statistical-mechanical system:
microstates, order parameters, control parameters, and the measured phase
structure.

Every formal statement is marked with its epistemic status:

- **[PROVED]** — proved in the document.
- **[MEASURED]** — observed, with the measurement and its conditions stated.
  Not proved, and the conditions matter (several results here are size- or
  density-dependent).
- **[OPEN]** — conjectured or sketched, not established.

The distinction is load-bearing. Mixing the three is how a measurement at one
operating point turns into a "law" nobody re-checks.

Build: `pdflatex assembly_statmech.tex` (no external packages beyond a standard
TeX distribution).

Supporting measurements live in `../experiments/`:
`critical_point_scan.py` (tidy CSV), `plot_critical_point.py` (figures),
`beta_cliff_location.py`, `depth_snr_scaling.py`, `depth_at_low_load.py`.
