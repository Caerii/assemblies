# Phase Diagram of Assembly Attractor Formation

**Script**: `research/experiments/stability/test_phase_diagram.py`
**Results file**: `phase_diagram_20260206_160402.json`
**Date**: 2026-02-06
**Brain implementation**: `src.core.brain.Brain` (with w_max saturation, corrected winner remapping and stimulus plasticity)

## Protocol

Maps the boundary in (k/n, beta) parameter space where assemblies transition from drifting activations to stable fixed-point attractors under autonomous recurrence.

1. **Establish**: `project({"s": ["A"]}, {})` -- initial stimulus activation.
2. **Train**: `project({"s": ["A"]}, {"A": ["A"]})` x 30 rounds -- stim+self training.
3. **Test**: `project({}, {"A": ["A"]})` x 20 rounds -- autonomous recurrence.
4. **Measure**: Final persistence = overlap between current winners and trained assembly after 20 autonomous rounds.

**Stability criterion**: persistence >= 0.95 marked as [S] (stable), otherwise [D] (drifting).

**Parameters**: n=1000, p=0.05, w_max=20.0, train_rounds=30, test_rounds=20.

**Statistical methodology**: N_SEEDS=10 independent seeds per condition. One-sample t-test against null k/n. Cohen's d. Mean +/- SEM.

## Results

### H1/H2: Sparsity (k/n) x Beta Phase Diagram

Persistence values (mean across 10 seeds). **Bold** = stable (>= 0.95).

| k/n | beta=0.01 | beta=0.02 | beta=0.05 | beta=0.10 | beta=0.20 |
|-----|-----------|-----------|-----------|-----------|-----------|
| 0.01 (k=10) | 0.010 | 0.020 | 0.760 | 0.730 | 0.550 |
| 0.02 (k=20) | 0.045 | 0.015 | 0.745 | 0.910 | 0.800 |
| 0.05 (k=50) | 0.138 | 0.224 | 0.922 | **1.000** | **0.976** |
| 0.10 (k=100) | 0.251 | 0.663 | **0.986** | **1.000** | **0.999** |
| 0.15 (k=150) | 0.393 | 0.834 | **0.995** | **1.000** | **1.000** |
| 0.20 (k=200) | 0.614 | 0.879 | **0.997** | **1.000** | **1.000** |
| 0.30 (k=300) | 0.781 | **0.965** | **1.000** | **1.000** | **1.000** |

**Phase boundary** (minimum beta for stability at each sparsity):

| k/n | min beta for stability | persistence at boundary |
|-----|----------------------|------------------------|
| 0.01 | *never stable* | max 0.760 |
| 0.02 | *never stable* | max 0.910 |
| 0.05 | 0.10 | 1.000 |
| 0.10 | 0.05 | 0.986 |
| 0.15 | 0.05 | 0.995 |
| 0.20 | 0.05 | 0.997 |
| 0.30 | 0.02 | 0.965 |

**Findings**:

1. **Sharp phase boundary exists**: The transition from drifting to stable is abrupt -- typically a single step in beta doubles persistence (e.g., k/n=0.10: beta=0.02 gives 0.663, beta=0.05 gives 0.986).

2. **Denser assemblies need less Hebbian learning**: At k/n=0.30, stability is achieved with beta=0.02. At k/n=0.05, beta=0.10 is required. The relationship is approximately k/n * beta_critical ~ 0.005. Denser assemblies have more intra-assembly connections by chance, so less Hebbian strengthening is needed.

3. **Very sparse assemblies never stabilize**: k/n=0.01 (k=10) and k/n=0.02 (k=20) fail to reach 0.95 persistence even at beta=0.20. With only 10-20 neurons in the assembly, the self-connectome has too few intra-assembly connections (expected: k^2 * p = 10^2 * 0.05 = 5 connections at k=10) to form a dominant attractor.

4. **Non-monotonicity at very sparse k/n**: At k/n=0.01, persistence *decreases* from 0.760 (beta=0.05) to 0.730 (beta=0.10) to 0.550 (beta=0.20). At k/n=0.02, it decreases from 0.910 (beta=0.10) to 0.800 (beta=0.20). Excessive Hebbian learning at extreme sparsity appears counterproductive -- the few intra-assembly connections hit w_max quickly, and further training strengthens spurious connections to non-assembly neurons that happened to co-fire, destabilizing the attractor.

5. **The stable region is large**: For k/n >= 0.05 and beta >= 0.05, all conditions achieve >= 0.922 persistence. The default parameters used throughout our experiments (k/n=0.10, beta=0.10) sit comfortably inside the stable regime.

### H3: Connection Probability Effect

**Parameters**: n=1000, k=100, beta=0.10, w_max=20.0, train_rounds=30, test_rounds=20.

| p | persistence (mean +/- SEM) | Cohen's d |
|---|---------------------------|-----------|
| 0.01 | 0.908 +/- 0.014 | 18.5 |
| 0.02 | 0.973 +/- 0.006 | 47.7 |
| 0.05 | **1.000 +/- 0.000** | inf |
| 0.10 | **1.000 +/- 0.000** | inf |
| 0.20 | **1.000 +/- 0.000** | inf |

**Findings**:
- Sharp threshold between p=0.01 and p=0.02. At p=0.01, expected intra-assembly connections = k^2 * p = 100^2 * 0.01 = 100 (out of 10,000 possible), which is marginal for attractor formation. At p=0.02, this doubles to 200, crossing the threshold.
- For p >= 0.05, persistence is perfect. The default p=0.05 provides ample connectivity.
- The transition from 0.908 to 1.000 over a 5x range of p (0.01 to 0.05) confirms that random connectivity provides the scaffold on which Hebbian learning builds -- too few connections and there is nothing to strengthen.

## Key Takeaways

1. **A sharp phase boundary separates stable attractors from drifting assemblies** in (k/n, beta) space. The critical product is approximately k/n * beta >= 0.005.
2. **Denser assemblies are easier to stabilize**: k/n=0.30 needs only beta=0.02; k/n=0.05 needs beta=0.10.
3. **Very sparse assemblies (k/n <= 0.02) cannot form stable attractors** at any tested beta, because the self-connectome has too few connections to train.
4. **Excessive beta destabilizes sparse assemblies**: Non-monotonic persistence at k/n=0.01-0.02 suggests weight saturation at extreme sparsity causes interference.
5. **Connection probability has a sharp threshold**: p=0.01 is marginal (0.908), p >= 0.02 is sufficient (0.973+) for k=100 at n=1000.
6. **Default parameters (k/n=0.10, beta=0.10, p=0.05) are deep inside the stable regime**: persistence = 1.000 with no variance.

## Biological Interpretation

The phase diagram suggests that biological assemblies (estimated k/n ~ 0.01-0.05 in cortex) operate near or below the stability boundary. This has implications:
- Cortical assemblies may rely on external input (stimulus drive, top-down attention) rather than pure autonomous recurrence for maintenance.
- Neuromodulation that increases effective beta (e.g., dopamine-mediated plasticity enhancement) could push assemblies across the phase boundary into stable attractor territory.
- The non-monotonicity at extreme sparsity suggests that homeostatic mechanisms (like synaptic scaling) may be necessary to prevent excessive Hebbian learning from destabilizing sparse representations.
