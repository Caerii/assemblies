# Dynamical-systems visuals for the assembly calculus

**Goal:** List **concrete visualizations** that would be valuable when viewing the assembly calculus as a **dynamical system** (x_{t+1} = topk(W x_t + b), fixed points = attractors, phase transitions). For each: what it shows, why it matters, and where the data comes from (existing experiments vs new logging).

---

## 1. Trajectories in reduced state space

**Idea:** State space is huge (n neurons or k winners). Project trajectories into 2D/3D so we *see* convergence, drift, or switching.

**Concrete plots:**

- **Overlap vs time (single run)**  
  - **X:** round t. **Y:** overlap of current winners with a *target* assembly (e.g. trained assembly, or “correct” attractor).  
  - **Shows:** Convergence curve (overlap → 1), persistence (stays high), or failure (overlap drops to chance).  
  - **Data:** Already in projection/scaling/phase experiments: `step_overlaps`, `persist_overlaps`, `recovery_trajectory`.  
  - **Use:** Paper figure: “assembly converges in 2–3 steps and persists.”

- **Overlap with A vs overlap with B (2D trajectory)**  
  - **X:** overlap(current, assembly_A). **Y:** overlap(current, assembly_B).  
  - **Shows:** Trajectory in “assembly space”: approach to one attractor (corner), or mixed state (middle). For multiple assemblies (e.g. distinctiveness), shows separation.  
  - **Data:** Per round, compute overlap with each of 2–3 reference assemblies; log (ov_A, ov_B) per t.  
  - **Use:** “Basin of attraction” intuition without full n-dimensional state.

- **PCA (or t-SNE) of winner sets over time**  
  - Encode each round as a binary vector (winners = 1) or as overlap vector vs a set of reference assemblies; PCA to 2–3 components; plot trajectory.  
  - **Shows:** Clusters = attractors; trajectories flowing into clusters.  
  - **Data:** Log winner sets (or overlaps with references) for a run; post-process.  
  - **Use:** High-level “landscape” for talks/papers when n is large.

**Why valuable:** Makes “convergence” and “attractor” *visible* instead of just a number (mean persistence). Supports theory (contraction → trajectory approaches fixed point).

---

## 2. Basins of attraction

**Idea:** For different *initial* conditions (e.g. different stimuli or perturbed winners), where does the system end up? Color by final attractor.

**Concrete plots:**

- **Initial overlap vs final overlap (scatter)**  
  - **X:** overlap(initial_winners, target). **Y:** overlap(final_winners, target).  
  - **Shows:** Recovery: points above diagonal = improved; horizontal band at 1 = all converge to target.  
  - **Data:** Noise-robustness / recovery experiments: vary initial (e.g. perturb k/2 winners); record final overlap.  
  - **Use:** “Recovery from perturbation” figure.

- **Which attractor? (discrete)**  
  - For multiple stable assemblies A, B, C: from many initial conditions (e.g. random subset of A∪B, or stimulus-driven), run dynamics; label end state as “A”, “B”, “C”, or “other”.  
  - **Plot:** 2D where (x,y) = some initial feature (e.g. overlap with A vs overlap with B); color = which attractor was reached.  
  - **Shows:** Basins in “initial condition” space.  
  - **Data:** New experiment: grid of initial conditions; run to fixpoint; record winner set and match to A/B/C.  
  - **Use:** “Basins of attraction” figure for theory/ML audience.

**Why valuable:** Directly illustrates “attractor” and “basin”; connects to theory (contraction region, capacity).

---

## 3. Bifurcation / phase diagrams (already in progress)

**Idea:** As parameters (p, k/n, β, n) vary, behavior changes *qualitatively*. Plot order parameter vs parameters.

**Concrete plots:**

- **Phase diagram: persistence vs (k/n, β)**  
  - **X:** k/n (sparsity). **Y:** β. **Color:** mean persistence (or “stable” vs “unstable”).  
  - **Shows:** Phase boundary: region where attractors exist vs not.  
  - **Data:** Already in `test_phase_diagram.py`; just need to plot heatmap + optional critical line.  
  - **Use:** Central figure for “phase transition” / complex-systems story.

- **Scaling: convergence time and persistence vs n**  
  - **X:** n (log scale). **Y:** T_conv or mean persistence.  
  - **Shows:** O(log n) convergence; persistence improving with n.  
  - **Data:** Already in `test_scaling_laws.py`.  
  - **Use:** “Scaling laws” figure; compare to theory (e.g. derived exponent).

- **Order parameter vs p (or vs k/n) at fixed β**  
  - **X:** p (connectivity) or k/n. **Y:** persistence (or overlap order parameter).  
  - **Shows:** Sharp transition at critical p or k/n if present.  
  - **Data:** Slice phase diagram or dedicated sweep.  
  - **Use:** “Critical point” figure; compare to mean-field prediction.

**Why valuable:** You already collect this; turning it into clear heatmaps and 1D slices makes the *phase transition* and *scaling* story immediate.

---

## 4. Stability and Jacobian (theory–experiment link)

**Idea:** Theory says stability = spectral radius of Jacobian &lt; 1 (or contraction). Visualize *where* the system is stable and how perturbations shrink.

**Concrete plots:**

- **Perturbation decay**  
  - Start at fixed point (or trained assembly); perturb by flipping a few winners; plot overlap with unperturbed fixpoint vs round.  
  - **Shows:** Exponential (or geometric) decay → stable; growth or no decay → unstable.  
  - **Data:** New: one run from fixpoint + small perturbation; log overlap per round.  
  - **Use:** “Stability of attractor” figure; compare decay rate to theoretical contraction factor.

- **Jacobian at fixpoint (if computed)**  
  - Compute Jacobian J of the map topk(W·) at a fixed point (e.g. numerically); plot eigenvalues in complex plane.  
  - **Shows:** All inside unit circle ⇒ stable.  
  - **Data:** From W and fixed-point support; need a small routine to form J (or approximate by finite differences).  
  - **Use:** “Linear stability” figure for theory paper.

**Why valuable:** Connects *empirical* stability (persistence, recovery) to *theoretical* stability (spectral radius, contraction). One such figure strengthens the ML/linear-algebra throughline.

---

## 5. Multi-area flow (projection, association)

**Idea:** Activity flows A → B → C. Visualize *who* is active *where* over time.

**Concrete plots:**

- **Raster / timeline by area**  
  - **X:** round t. **Y:** neuron index (or binned). **Color:** activity (e.g. 1 if winner, 0 else), or one row per area with winner indices binned.  
  - **Shows:** Which area “lights up” when; propagation A→B→C.  
  - **Data:** Log winners per area per round (already available in projection/association).  
  - **Use:** “Information flow” figure for multi-area experiments.

- **Overlap matrix between areas over time**  
  - At each t, compute overlap(area_A_winners, area_B_winners), etc.; plot as small matrix or heatmap sequence.  
  - **Shows:** When B “matches” A (projection); when C “matches” A and B (association).  
  - **Data:** From winners per area per round.  
  - **Use:** “Assembly alignment across areas.”

- **Graph: areas as nodes, edges = “activity propagated”**  
  - Node size = activity (e.g. sum of weights of winners); directed edge A→B = “B’s winners overlap with A’s” or “B received input from A.” Animate over t.  
  - **Shows:** Flow of activity on the network.  
  - **Data:** Connectomes + winners per area.  
  - **Use:** Conceptual figure for “assembly calculus as dynamics on a graph.”

**Why valuable:** Makes *multi-area* dynamics (projection, association) visible; supports “assembly calculus as dynamical system on a network.”

---

## 6. Weight matrix and Hebbian structure

**Idea:** W starts random (Bernoulli); Hebbian updates structure it. Visualize W before/after and its effect on dynamics.

**Concrete plots:**

- **W before vs after training (heatmap or block)**  
  - Plot W (or a principal submatrix) at init vs after forming one assembly (or several).  
  - **Shows:** Sparse random → structured (blocks or bands corresponding to co-active neurons).  
  - **Data:** Save connectome (or sample of rows/columns) at init and after training.  
  - **Use:** “Hebbian structure” figure.

- **Weight distribution**  
  - Histogram of W_ij before and after Hebbian updates.  
  - **Shows:** Bimodal or heavy tail after learning (strong weights on assembly pairs).  
  - **Data:** From connectome dump.  
  - **Use:** “Plasticity changes weight distribution.”

- **Eigenvalue spectrum of W (or W restricted to assembly support)**  
  - Histogram or scatter of eigenvalues in complex plane.  
  - **Shows:** Spectral radius; clustering; comparison to random matrix.  
  - **Data:** From W (or W_assembly); standard linear algebra.  
  - **Use:** Theory figure: “Spectrum of learned W.”

**Why valuable:** Links *plasticity* (Hebbian) to *structure* (W) and to *dynamics* (eigenvalues, stability). Supports SM + ML throughlines.

---

## 7. Recurrence and consistency

**Idea:** Recurrence plot (or similar) shows when the system “revisits” similar states — useful for attractors and cycles.

**Concrete plots:**

- **Recurrence plot**  
  - R(t, s) = 1 if overlap(winners_t, winners_s) &gt; θ, else 0. Plot R as 2D image.  
  - **Shows:** Diagonal line = persistence; horizontal/vertical bands = periodic or stuck; scattered = chaotic or drifting.  
  - **Data:** Log winner sets for a long run; post-process.  
  - **Use:** “Attractor = recurrent structure” figure.

- **Step-to-step overlap over time (already used for convergence)**  
  - **X:** round. **Y:** overlap(winners_t, winners_{t-1}).  
  - **Shows:** Jumps to 1 when at fixpoint; &lt;1 when moving.  
  - **Data:** Already in projection: `step_overlaps`.  
  - **Use:** “Convergence in 2–3 steps” figure.

**Why valuable:** Standard dynamical-systems tool; reinforces “attractor” and “convergence” in a recognizable form.

---

## 8. Summary: what to build first

| Priority | Visual | Data source | Purpose |
|----------|--------|-------------|---------|
| **1** | Overlap vs time (single run) | Existing (step_overlaps, persist_overlaps, recovery_trajectory) | Convergence + persistence; paper figure |
| **2** | Phase diagram heatmap (persistence vs k/n × β) | Existing (phase_diagram) | Phase transition; central figure |
| **3** | Scaling: T_conv and persistence vs n | Existing (scaling_laws) | Scaling laws; compare to theory |
| **4** | Overlap with A vs overlap with B (2D trajectory) | Log overlaps with 2 ref assemblies per round | Basins / attractor space |
| **5** | Perturbation decay (overlap with fixpoint vs t) | New: perturb then run; log overlap | Stability; link to theory |
| **6** | Raster / timeline by area (multi-area) | Winners per area per round | Flow A→B→C |
| **7** | W before/after (heatmap or distribution) | Connectome dump before/after training | Hebbian structure |
| **8** | Recurrence plot | Winner sets over long run | Attractor = recurrent |
| **9** | Basin coloring (initial vs final attractor) | New: grid of initial conditions | Basins of attraction |
| **10** | Jacobian eigenvalues at fixpoint | From W + fixpoint support | Linear stability |

**Recommendation:** Start with **1–3** (data already there; just plot). Add **4** and **5** when you want to push the “dynamical system” and “stability” story. **6–7** for multi-area and plasticity. **8–10** for deeper theory/ML audience.

All of these are **dynamical-systems visuals**: they show trajectories, basins, phase structure, stability, or structure of W that governs the dynamics. They support the theoretical throughlines (ML/linear algebra, complex systems, SM) and make the assembly calculus readable as a dynamical system.
