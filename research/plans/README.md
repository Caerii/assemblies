# Research plans and stubs

This folder holds **planning documents and stubs** — roadmaps, curriculum designs, and evidence analyses that are not yet implemented or validated. They are organized by theme.

**Status:** Plans/stubs only. Nothing here is a completed experiment or a canonical result. Use them to guide implementation and to track what evidence exists (or is missing).

---

## Structure

| Subfolder / doc | Contents |
|------------------|----------|
| **THEORETICAL_THROUGHLINES.md** | ML/linear algebra view (fixed points, stability, capacity) and complex systems/network physics view (phase transitions, mean-field, criticality); whether to add both to the exploratory plan. |
| **PRIORITIES_AND_GAPS.md** | What else would add the most value: unblock Q21/Q10 (autonomous recurrence, noise robustness), one derived theory result, claims pipeline, one biological comparison, Q12 learning stability, falsifiability pass. |
| **VISUALS_DYNAMICAL_SYSTEMS.md** | Dynamical-systems visuals: trajectories (overlap vs t, 2D overlap space), basins, phase diagram/scaling, stability (perturbation decay, Jacobian), multi-area flow, W before/after, recurrence plot; data sources and build order. |
| **BRIDGE_WEBSCALE_CURRICULUM.md** | Bridge assembly calculus ↔ web-scale data and curricula: token → assembly input, next-token loss, curriculum ordering; implications; minimal recipe; can do next-token prediction. |
| **control/** | Control-rate and motor-control plans: CUDA benchmark → achievable control Hz, biological realism of loop rates. |
| **robotics_embodiment/** | Isaac Lab + assembly brain: brainstorm for embodied control, embodied/social cognition, multi-robot, language at realistic rates, what could emerge. |
| **curriculum/** | Embodied + social curriculum (stages 0–8), MHC alignment, and task-by-task evidence analysis (what we have vs what we don’t). |

---

## Index of documents

### Theory (exploratory)
- **THEORETICAL_THROUGHLINES.md** — ML/linear algebra perspective (top-k dynamics, fixed points, stability, capacity, Hopfield/sparse coding); complex systems/network physics (phase transitions, mean-field, percolation, criticality); recommendation to include both in the exploratory plan alongside statistical mechanics.
- **PRIORITIES_AND_GAPS.md** — What else would be most valuable: fix Q21/Q10 (autonomous recurrence, noise robustness); one derived theory result (phase boundary, convergence rate, or capacity); populate claims/ for validated results; one biological comparison; Q12 learning stability; falsifiability pass (Q19).
- **VISUALS_DYNAMICAL_SYSTEMS.md** — Visuals for the dynamical-system view: overlap vs time, 2D overlap trajectories, basins of attraction, phase diagram and scaling plots, perturbation decay and Jacobian eigenvalues, multi-area raster and flow, W before/after and spectrum, recurrence plot; data sources (existing vs new) and suggested build order.
- **BRIDGE_WEBSCALE_CURRICULUM.md** — Bridge between assembly calculus and web-scale data + training curricula: token→assembly mapping, next-token prediction (yes), readout + optional STE, curriculum ordering (length/frequency/domain); implications; minimal recipe for next-token training.

### control/
- **CONTROL_SPEED_FROM_CUDA.md** — Derive maximum control-loop frequency (Hz) from CUDA assembly benchmarks; biological realism of control rate; how to state it for a paper.

### robotics_embodiment/
- **ISAAC_LAB_ASSEMBLY_BRAINSTORM.md** — Isaac Lab + assembly brain: minimal loop, parallel envs, hierarchy, association, sequences, training; embodied cognition, embodied social cognition, realistic speech/processing rates; what could emerge.

### curriculum/
- **EMBODIED_SOCIAL_CURRICULUM.md** — Rich curriculum for embodied + social assembly cognition: stages 0–8 (sensorimotor → affordances → first words → phrases → commands → dialogue → two robots → social dialogue → open-ended); MHC mapping and implications for structure.
- **CURRICULUM_TASK_ANALYSIS_AND_EVIDENCE.md** — Task-by-task analysis of the curriculum; what makes sense; what evidence we have (or lack); verdicts per stage; recommendations.

---

## Relationship to the rest of research/

- **experiments/** — Runnable experiments and scripts. Results live in **results/**.
- **open_questions.md** — Living list of open questions (some validated, some not).
- **core_questions/**, **papers/** — Core questions and paper infrastructure.

Plans in this folder **reference** experiments and results (e.g. association recovery, distinctiveness, phase diagram) but are not substitutes for running experiments or publishing results.
