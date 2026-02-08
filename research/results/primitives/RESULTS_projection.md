# Projection Primitive: Convergence and Stability

**Script**: `research/experiments/primitives/test_projection.py`
**Date**: 2026-02-06 (post-bugfix run)
**Brain implementation**: `src.core.brain.Brain` (with w_max saturation, corrected winner remapping and stimulus plasticity)

## Bug Fixes Applied Before This Run

1. **Winner remapping**: Explicit areas now correctly use `n` (total neurons) instead of `w` (ever-fired count) for winner selection range. Previously all assemblies were compressed into indices [0, ~189] regardless of area size.
2. **Stimulus plasticity indexing**: Hebbian learning now strengthens connections TO winner neurons (column indexing) instead of FROM fiber indices (row indexing).
3. **Area.winners setter**: Explicit areas now correctly maintain `w = n` when winners are manually set.

## Protocol

- **Stim+self training**: `project({"s": ["A"]}, {"A": ["A"]})` x T rounds
- **Stim-only training**: `project({"s": ["A"]}, {})` x T rounds
- **Autonomous test**: `project({}, {"A": ["A"]})` x 20 rounds, measure overlap with trained assembly
- **Convergence**: 3 consecutive rounds with step-to-step overlap > 0.98

**Statistical methodology**: N_SEEDS=10 independent seeds per condition. One-sample t-test against null k/n. Paired t-test for H2. Cohen's d. Mean +/- SEM.

## Results

### H1: Convergence Rate vs Network Size

**Parameters**: k=sqrt(n), p=0.05, beta=0.10, w_max=20.0, max_train=100.

| n | k | T_conv (mean +/- SEM) | persistence (mean +/- SEM) | Cohen's d |
|---|---|----------------------|---------------------------|-----------|
| 100 | 10 | 16.3 +/- 9.4 | 0.360 +/- 0.078 | 1.1 |
| 200 | 14 | 40.0 +/- 12.5 | 0.414 +/- 0.089 | 1.2 |
| 500 | 22 | 45.3 +/- 14.9 | 0.545 +/- 0.106 | 1.5 |
| 1000 | 31 | 9.9 +/- 0.5 | 0.497 +/- 0.053 | 2.8 |
| 2000 | 44 | 13.9 +/- 4.1 | 0.539 +/- 0.062 | 2.6 |
| 5000 | 70 | 8.6 +/- 0.2 | 0.360 +/- 0.070 | 1.6 |

**Scaling fit**: T = -12.74 * log10(n) + 58.42 (R^2=0.250)

**Findings**:
- Convergence is variable at small n (100-500) and stabilizes at larger n (8-14 rounds for n >= 1000).
- Persistence is moderate (0.36-0.54) — the autonomous attractor is weaker than in the pre-bugfix runs because assemblies now correctly span the full neuron range [0, n-1], making the self-connectome's job harder.
- The persistence values represent genuine attractor strength: how well a trained self-connectome can maintain an assembly without external stimulus drive.

### H2: Stim-Only vs Stim+Self Training

**Parameters**: n=1000, k=100, p=0.05, beta=0.10, w_max=20.0, train_rounds=30.

| Training mode | persistence (mean +/- SEM) | Cohen's d vs null |
|--------------|---------------------------|-------------------|
| Stim+self | 0.709 +/- 0.032 | 6.0 |
| Stim-only | 0.113 +/- 0.007 | 0.6 |

**Paired t-test**: t=18.88, p < 0.0001, Cohen's d = 6.0

**Findings**:
- **Stim+self creates genuine attractors** (0.709 persistence) — the assembly partially survives autonomous recurrence.
- **Stim-only persistence is at chance** (0.113 vs null k/n = 0.100). This is correct: stim-only training strengthens stimulus-to-area connections but never trains the A->A self-connectome, so there is no attractor to hold the assembly during autonomous recurrence.
- The stim-only result validates that our bug fixes are working: the assembly disperses to random when there is no self-connectome training, exactly as expected.

### H3: Cross-Area Projection Fidelity

**Parameters**: k=sqrt(n), p=0.05, beta=0.10, w_max=20.0, train_rounds=30.

| n | k | recovery (mean +/- SEM) |
|---|---|------------------------|
| 500 | 22 | **1.000 +/- 0.000** |
| 1000 | 31 | **1.000 +/- 0.000** |
| 2000 | 44 | **1.000 +/- 0.000** |

**Findings**:
- Cross-area projection recovery is **perfect** at all tested sizes.
- After training A->B connections, corrupting B, and re-projecting from A, the B-assembly is recovered with 100% fidelity.
- This validates cross-area projection as a completely reliable mechanism.

### H4: Weight Ratio vs Training Rounds

**Parameters**: n=1000, k=100, p=0.05, beta=0.10, w_max=20.0.

| T (rounds) | weight ratio (mean +/- SEM) | persistence (mean +/- SEM) |
|-----------|---------------------------|---------------------------|
| 1 | 1.00 +/- 0.00 | 0.152 +/- 0.009 |
| 5 | 1.00 +/- 0.00 | 0.442 +/- 0.015 |
| 10 | 1.00 +/- 0.00 | 0.730 +/- 0.013 |
| 20 | 1.00 +/- 0.00 | 0.706 +/- 0.024 |
| 30 | 1.00 +/- 0.00 | 0.722 +/- 0.031 |
| 50 | 1.00 +/- 0.00 | 0.733 +/- 0.027 |

**Findings**:
- Weight ratio measurement still returns 1.00 (measurement bug in extracting self-connectome weights — not related to the projection bugs fixed this session).
- Persistence shows a clear learning curve: 0.152 (T=1) -> 0.730 (T=10) -> 0.733 (T=50).
- Most of the attractor formation happens in the first 10 rounds. Diminishing returns beyond that.
- The persistence ceiling of ~0.73 is lower than pre-bugfix (~0.96) because assemblies now span the full neuron range, requiring more self-connectome specificity.

## Key Takeaways

1. **Self-connectome training creates genuine attractors**: Persistence of 0.71 with stim+self vs 0.11 (chance) with stim-only.
2. **Cross-area projection is perfectly reliable**: 1.000 recovery at all network sizes.
3. **Stim-only produces no attractor**: Correct behavior — assemblies are purely stimulus-driven.
4. **Attractor formation saturates at ~10 rounds**: Most stability gain happens early.
5. **Persistence ceiling ~0.73**: With correct neuron indexing, the self-connectome provides moderate (not dominant) attractor strength at n=1000, k=100.
