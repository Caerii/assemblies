# Research Plans

This directory holds planning notes, roadmaps, curriculum sketches, and
evidence analyses that are not yet validated results.

Nothing here is a completed experiment. Use these files to plan implementation,
track missing evidence, and decide what should move into `experiments/`,
`results/`, `core_questions/`, or `claims/`.

## Main Documents

| Document | Focus | Status (2026-09-09) |
|----------|-------|---------------------|
| `PAPERS.md` | The papers the measured results support, in order of readiness, with what each still needs; reconciles the plans below with the register. | current |
| `ASSEMBLY_SYSTEMS_PAPER_VISION.md` | Long systems-paper vision: assemblies as learned macrostates, measurable regimes, operation contracts, and cognitive-computational implications. | thesis kept; the single long paper is split into `PAPERS.md` P1 to P4 |
| `THEORETICAL_THROUGHLINES.md` | Linear-algebra, dynamical-systems, and complex-systems views of assembly dynamics. | current; feeds `PAPERS.md` P4 |
| `PRIORITIES_AND_GAPS.md` | Highest-value gaps: autonomous recurrence, noise robustness, theory, claims, biological comparison, and falsifiability. | items 1, 3, 5, 6 resolved or overtaken; see its header |
| `SOUNDNESS_PROGRAM.md` | The phased program for measurement soundness; its standing rules apply to every draft. | current |
| `VISUALS_DYNAMICAL_SYSTEMS.md` | Figures that would make stability, attractors, phase diagrams, recurrence, and learned weights visible. | partly done in `../notes/figures/` |
| `IMPLICATIONS_AND_PREDICTIONS.md` | The N400 and P600 triple dissociation and its predictions. | overtaken by the language notes; see its header |
| `BRIDGE_WEBSCALE_CURRICULUM.md` | How assembly inputs, curricula, and next-token prediction might connect to larger-scale data. | aspirational; title rule in `PAPERS.md` section 1 |
| `ASSEMBLIES_AS_NEURAL_COMPILER.md` | Assemblies as a possible programming substrate or compiler target. |
| `SELF_ASSEMBLING_NEURAL_NANOTECH.md` | Speculative links between assembly dynamics and self-organizing hardware or materials. |
| `ASSEMBLIES_MAPPED_CONNECTOMES.md` | Using mapped connectomes as graph substrates for assembly dynamics. |

## Subdirectories

- `control/`
  Control-rate and motor-control planning.
- `robotics_embodiment/`
  Isaac Lab, embodied control, and language-grounding brainstorms.
- `curriculum/`
  Embodied and social curriculum sketches, including task-order analysis.

## How To Use These Files

Plans can be speculative. Claims cannot.

When a plan becomes concrete:

1. create or update an experiment under `research/experiments/`
2. store outputs under `research/results/`
3. update `research/open_questions.md`
4. promote bounded evidence into `research/claims/` only when the result is
   defensible

Do not cite a plan as evidence. Cite the experiment, result artifact, or claim.
