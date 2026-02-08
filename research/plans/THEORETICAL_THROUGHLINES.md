# Theoretical throughlines: ML/linear algebra and complex systems

**Goal:** Spell out the **machine learning / linear algebra** perspective and the **complex systems / network physics** perspective on the assembly calculus; say whether each is a **worthy throughline** to explore; and recommend whether to include them in the **exploratory plan**.

---

## 1. Machine learning theory / linear algebra perspective

### What it is

The assembly calculus can be written in **linear-algebraic** and **dynamical-system** terms:

- **Projection step:** For a target area, input = sum of weighted inputs from source areas (and stimuli). So **input = W @ one_hot(winners)** (or sum of columns of W indexed by winners). That’s a **matrix–vector product** (sparse: only k columns/rows matter).
- **Winner-take-all:** **x_{t+1} = topk(input)** — take the indices of the k largest entries. So one step is **x_{t+1} = topk(W x_t + b)** (with b from stimuli). That’s a **nonlinear dynamical system**: affine map then top-k.
- **Hebbian update:** **W_ij := W_ij * (1 + β)** for (i, j) in (winners × pre-synaptic winners), then clamp at w_max. So **W** is updated with a **rank-k** (or sparse) update driven by the current assembly. Over time, W becomes a **structured** matrix (strong weights on co-activated pairs).
- **Fixed points:** Attractors = states **x*** such that **x* = topk(W x*)**. So we’re studying **fixed points of a nonlinear map** defined by W and k.
- **Convergence:** Stability of a fixed point = whether small perturbations shrink. That’s **Jacobian** or **contraction** analysis: does the map **topk(W ·)** contract in some norm or on some subset? Spectral radius of the linearization (or a suitable matrix norm) below 1 ⇒ stability.

So from **linear algebra / dynamical systems**:

- **W** is the key object: random (Bernoulli) at init, then **Hebbian** structure.
- Dynamics = **iterated top-k of (W x + b)**.
- Questions: **spectral / norm conditions** for uniqueness and stability of fixed points; **capacity** (number of stable attractors) as a function of n, k, p; **convergence rate** (contraction factor per step).

### ML theory angle

- **Hopfield networks:** Binary or continuous units, **symmetric W**, energy function, attractors = local minima. Assembly calculus is **asymmetric** (area A → B ≠ B → A in general), **sparse** (k winners), and **Hebbian** (W learned). So it’s a **sparse, asymmetric, Hebbian** variant of Hopfield-like associative memory.
- **Sparse coding / dictionary learning:** Representations are sparse (k active). Assembly formation = finding a sparse code given input; Hebbian = learning the “dictionary” (W). So there’s a link to **sparse coding** and **recurrent sparse autoencoders**.
- **Attractor networks in ML:** Modern view: RNNs as dynamical systems, attractors for memory. Assembly calculus = a **discrete-time, sparse, Hebbian** attractor net with a **prescribed** nonlinearity (top-k). Theory from **random matrix** or **mean-field** analysis could predict capacity and convergence.
- **Linear algebra:** **Eigenstructure** of W (or of the effective Jacobian at a fixed point) controls stability. **Rank** of Hebbian updates, **spectral radius** of W (or of W restricted to support of assembly), **contractivity** of top-k(W ·) on a suitable set — all are concrete, analyzable.

### Is it a worthy throughline?

**Yes.**

- **Precise:** You get **testable** conditions (e.g. “if spectral radius of J < 1 at x*, then x* is stable”) and **capacity bounds** (e.g. number of fixed points as function of n, k, p).
- **Bridges to ML:** Hopfield, sparse coding, attractor RNNs — same community and venues. A “sparse Hebbian attractor net with top-k” is a clear **ML-theory** object.
- **Complements statistical mechanics:** SM gives **distributional** (phase diagram, fluctuations); linear algebra gives **per-instance** (this W, this x*, stable or not) and **deterministic** bounds. Both are useful.
- **Implementation-friendly:** Your code is already matrix/vector (connectomes, winner indices); the same objects (W, support of x) are what you’d analyze.

**Worth including in the exploratory plan:** Yes. Add a **theory** thread: (1) write dynamics as **x_{t+1} = topk(W x_t + b)**; (2) characterize fixed points and **stability** (Jacobian/contraction); (3) **capacity** (number of attractors) from random-matrix or mean-field arguments; (4) **convergence rate** (e.g. contraction per step). This can feed the “needs theory” part of Q03 (scaling) and the phase diagram (Q02) with **derived** conditions, not just empirical.

---

## 2. Complex systems / network physics perspective

### What it is

- **Network:** Areas and stimuli = nodes; connectomes = weighted, directed edges. So the brain is a **directed, weighted graph** (or multigraph: multiple areas, multiple connectomes).
- **Dynamics on the network:** Activity (assembly = set of k winners) propagates along edges; each area updates by top-k of incoming weighted input. So it’s **discrete-time dynamics on a random graph** with **nonlinear** node update (top-k).
- **Random graph:** W is Bernoulli(p) at init → **Erdős–Rényi** (or similar) random graph. Then Hebbian **rewires** weights (strengthens some edges). So we have **random graph + plasticity** — a classic **network physics** setup.
- **Phase transitions:** As parameters (p, k/n, β) vary, the system can have **qualitative** change: e.g. from “no stable assembly” to “unique attractor” to “multiple attractors.” That’s a **phase transition**; critical exponents and scaling are **complex systems** bread-and-butter.
- **Mean-field / cavity:** In the limit of large n, we can try **mean-field** equations (e.g. distribution of overlaps, or order parameters) and **cavity** methods (one node in a field created by the rest). That’s how spin glasses and random neural nets are analyzed — **network physics**.
- **Percolation / connectivity:** Existence of a giant component, or of a “path” from stimulus to stable assembly, can be seen as **percolation** on the random graph. So **percolation theory** (critical p, scaling) is relevant.
- **Criticality:** “Brain at criticality” = systems tuned near a phase transition. If assembly formation has a critical point (e.g. critical p or k/n), then **critical exponents** and **universality** are natural — **complex systems**.

### Is it a worthy throughline?

**Yes.**

- **You’re already there:** Phase diagram (Q02), scaling laws (Q03), statistical mechanics framing — that *is* complex systems / network physics. Making it **explicit** as a throughline ties the experiments to a **theory** (phase transitions, mean-field, criticality) and to a **community** (physics, complex systems, network science).
- **Predictive:** Mean-field or cavity can **derive** phase boundaries and scaling exponents; then you **test** them in simulation. That’s the right loop.
- **Unifying:** Same language (order parameters, critical point, universality) for **assembly formation**, **capacity**, and **convergence** — one framework.

**Worth including in the exploratory plan:** Yes. You already have phase diagram and scaling experiments; the exploratory plan should **explicitly** list: (1) **mean-field / cavity** derivation of phase diagram and critical exponents; (2) **finite-size scaling** and universality; (3) **percolation** or connectivity view (e.g. critical p for “stimulus → assembly” path). That’s the complex systems / network physics thread.

---

## 3. How the three fit together

| Throughline | Main question | Tools | Output |
|-------------|----------------|-------|--------|
| **Statistical mechanics** | Distribution of overlaps, convergence time, phase diagram | Ensembles, Fokker–Planck, cavity, free energy | Phase diagram, scaling, capacity (thermodynamic limit) |
| **ML / linear algebra** | Fixed points, stability, capacity (per W), convergence rate | Jacobian, spectral radius, contraction, random matrices | Stability conditions, capacity bounds, convergence rate |
| **Complex systems / network physics** | Phase transitions, criticality, dynamics on random graphs | Mean-field, percolation, critical exponents, universality | Critical point, exponents, scaling, “brain at criticality” |

- **SM** and **complex systems** overlap a lot (phase transitions, mean-field, cavity). The difference is emphasis: SM = thermodynamic/ensemble view; complex systems = networks, graphs, criticality.
- **ML / linear algebra** is more **per-instance** and **deterministic** (this W, this x*); SM / complex systems is more **distributional** and **typical** (over random W, over runs). Both are useful: linear algebra for **sufficient conditions** and **algorithms**; SM/complex systems for **phase diagram** and **scaling**.

So: **three throughlines**, not two. All three are worthy. They reinforce each other.

---

## 4. Recommendation for the exploratory plan

**Include both (and keep SM explicit):**

1. **ML theory / linear algebra**
   - Formulate dynamics as **x_{t+1} = topk(W x_t + b)**.
   - Study **fixed points** and **stability** (Jacobian, contraction).
   - Derive **capacity** (number of attractors) and **convergence rate** (e.g. O(log n) from contraction).
   - Link to Hopfield, sparse coding, attractor RNNs.

2. **Complex systems / network physics**
   - Treat the brain as **dynamics on a random graph** (Bernoulli + Hebbian).
   - **Mean-field / cavity** for order parameters and phase diagram.
   - **Critical exponents** and **finite-size scaling** at the transition.
   - **Percolation** view (critical connectivity for assembly formation).
   - Frame as “assembly formation as a critical phenomenon” / “brain at criticality” where appropriate.

3. **Statistical mechanics** (already in your monograph vision)
   - Keep as the **distributional** and **thermodynamic** layer (free energy, entropy, phase diagram from ensemble).
   - SM and complex systems share phase transitions and mean-field; keep both labels so you can cite **physics** (SM) and **network science** (complex systems) as appropriate.

**Concrete add to exploratory plan:**

- **Theory (exploratory):**  
  - **Thread A — Linear algebra / ML:** Fixed-point and stability analysis of top-k(W·); capacity and convergence rate; connection to Hopfield/sparse coding.  
  - **Thread B — Complex systems / network physics:** Mean-field/cavity derivation of phase diagram; critical exponents and finite-size scaling; percolation/connectivity interpretation.  
  - **Thread C — Statistical mechanics:** (Existing) Ensemble, free energy, thermodynamic limit, scaling.

- **Experiments:**  
  - Phase diagram and scaling (already there) **inform** Threads B and C.  
  - Add (if not already): **per-instance** stability checks (e.g. Jacobian at fixed point, or perturbation tests) to **validate** Thread A.

So: **yes**, the ML/linear algebra perspective and the complex systems/network physics perspective are **both** worthy throughlines; **yes**, they’re worth including in the exploratory plan alongside statistical mechanics; and the note above is a compact way to record that and to spell out how the three fit together.
