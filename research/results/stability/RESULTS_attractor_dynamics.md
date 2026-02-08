# Attractor Dynamics Under Autonomous Recurrence

**Script**: `research/experiments/stability/test_attractor_dynamics.py`
**Results file**: `attractor_dynamics_20260206_121741.json`
**Date**: 2026-02-06
**Brain implementation**: `src.core.brain.Brain` (with w_max saturation)

## Protocol

1. Create explicit area A (n neurons, k assembly size) with stimulus s.
2. **Establish**: `project({"s": ["A"]}, {})` -- single stimulus activation.
3. **Train**: `project({"s": ["A"]}, {"A": ["A"]})` x train_rounds -- strengthens both stim->A and A->A weights via Hebbian plasticity, clamped at w_max.
4. Record trained assembly winners.
5. **Test**: `project({}, {"A": ["A"]})` x test_rounds -- pure autonomous recurrence with no external input.
6. Measure overlap between current winners and trained assembly at each test step.

**Statistical methodology**: N_SEEDS=10 independent seeds per condition. One-sample t-test against null mean k/n (chance overlap of two random k-subsets). Cohen's d effect size. Mean +/- SEM with 95% CI.

## Results

### H1: Training Threshold

**Question**: What is the critical Hebbian exposure (beta x train_rounds) for stable attractor formation?

**Base parameters**: n=1000, k=100, p=0.05, w_max=20.0, test_rounds=20.

| beta | rounds | beta*T | persistence (mean+/-SEM) | weight ratio | Cohen's d |
|------|--------|--------|--------------------------|-------------|-----------|
| 0.01 | 1 | 0.01 | 0.886 +/- 0.007 | 1.0 | 36.2 |
| 0.01 | 5 | 0.05 | 0.914 +/- 0.008 | 1.1 | 33.7 |
| 0.01 | 10 | 0.10 | 0.918 +/- 0.012 | 1.2 | 22.1 |
| 0.01 | 20 | 0.20 | 0.918 +/- 0.011 | 1.3 | 24.1 |
| 0.01 | 50 | 0.50 | 0.972 +/- 0.008 | 1.7 | 34.5 |
| 0.05 | 1 | 0.05 | 0.866 +/- 0.009 | 1.0 | 26.0 |
| 0.05 | 5 | 0.25 | 0.923 +/- 0.008 | 1.3 | 31.3 |
| 0.05 | 10 | 0.50 | 0.954 +/- 0.007 | 1.7 | 37.6 |
| 0.05 | 20 | 1.00 | 0.988 +/- 0.003 | 2.7 | 96.6 |
| 0.05 | 50 | 2.50 | 0.997 +/- 0.002 | 11.2 | 185.7 |
| 0.10 | 1 | 0.10 | 0.890 +/- 0.006 | 1.0 | 39.5 |
| 0.10 | 5 | 0.50 | 0.947 +/- 0.006 | 1.6 | 44.8 |
| 0.10 | 10 | 1.00 | 0.988 +/- 0.003 | 2.5 | 86.0 |
| 0.10 | 20 | 2.00 | 0.997 +/- 0.002 | 6.5 | 185.7 |
| 0.10 | 50 | 5.00 | 0.998 +/- 0.001 | 21.1 | 213.0 |
| 0.20 | 1 | 0.20 | 0.878 +/- 0.011 | 1.0 | 23.2 |
| 0.20 | 5 | 1.00 | 0.970 +/- 0.003 | 2.3 | 82.5 |
| 0.20 | 10 | 2.00 | 0.997 +/- 0.002 | 5.4 | 185.7 |
| 0.20 | 20 | 4.00 | 1.000 +/- 0.000 | 21.3 | inf |
| 0.20 | 50 | 10.00 | 1.000 +/- 0.000 | 21.5 | inf |

**Findings**:
- All conditions are significantly above the chance null (k/n = 0.10), p < 0.0001.
- The product beta*T governs stability, not either variable alone.
- Transition from partial (~0.87-0.93) to near-perfect (>0.95) at beta*T ~ 0.5.
- Near-frozen attractors (>0.997) at beta*T >= 1.0.
- Perfect fixed points (1.000) at beta*T >= 4.0.
- Weight ratio >= 2 corresponds to the transition to high stability.

**Caveat**: Even 1 round of training gives ~0.88 persistence, far above chance. The null (k/n = 0.10) is too easy to beat. A stronger null would be stim-only training persistence (no self-projection strengthening).

### H2: Attractor Basin (Noise Tolerance)

**Question**: What fraction of the assembly can be replaced with random neurons while still recovering the original attractor?

**Parameters**: n=1000, k=100, p=0.05, beta=0.1, w_max=20.0, train_rounds=30, test_rounds=20.

| noise fraction | final overlap (mean +/- SEM) | range | Cohen's d |
|---------------|------------------------------|-------|-----------|
| 0.0 | 0.994 +/- 0.002 | [0.98, 1.00] | 127.9 |
| 0.1 | 0.996 +/- 0.002 | [0.99, 1.00] | 173.5 |
| 0.2 | 0.997 +/- 0.002 | [0.99, 1.00] | 185.7 |
| 0.3 | 0.997 +/- 0.002 | [0.98, 1.00] | 132.9 |
| 0.4 | 0.997 +/- 0.002 | [0.99, 1.00] | 185.7 |
| 0.5 | 0.997 +/- 0.002 | [0.98, 1.00] | 132.9 |
| 0.6 | 0.997 +/- 0.002 | [0.99, 1.00] | 185.7 |
| 0.7 | 0.998 +/- 0.001 | [0.99, 1.00] | 213.0 |
| 0.8 | 0.992 +/- 0.003 | [0.98, 1.00] | 97.1 |
| 0.9 | 0.998 +/- 0.001 | [0.99, 1.00] | 213.0 |
| 1.0 | 0.994 +/- 0.002 | [0.98, 1.00] | 127.9 |

**Findings**:
- The attractor recovers from **100% noise** (all winners replaced with random neurons).
- Recovery to >0.99 overlap within 20 self-projection steps at all noise levels.
- The basin of attraction is effectively the entire state space (global attractor).

**Caveat**: This result is for a **single trained assembly** with heavy training (beta*T = 3.0). With multiple competing assemblies, basins would shrink and interference would appear. This is the most important follow-up experiment.

### H3: Network Scaling

**Question**: How does attractor persistence scale with network size n (at k = sqrt(n))?

**Parameters**: p=0.05, beta=0.1, w_max=20.0, train_rounds=30, test_rounds=20.

| n | k = floor(sqrt(n)) | k/n | null overlap | persistence (mean +/- SEM) | Cohen's d |
|---|---------------------|-----|-------------|----------------------------|-----------|
| 100 | 10 | 0.100 | 0.100 | 0.800 +/- 0.039 | 5.6 |
| 200 | 14 | 0.070 | 0.070 | 0.821 +/- 0.044 | 5.4 |
| 500 | 22 | 0.044 | 0.044 | 0.977 +/- 0.010 | 29.0 |
| 1000 | 31 | 0.031 | 0.031 | 0.990 +/- 0.005 | 61.6 |
| 2000 | 44 | 0.022 | 0.022 | 0.995 +/- 0.003 | 101.6 |
| 5000 | 70 | 0.014 | 0.014 | 0.999 +/- 0.001 | 217.9 |

**Findings**:
- All sizes significantly above chance (p < 0.0001).
- Transition from partial stability (~0.80) to near-perfect (>0.97) between n=200 and n=500.
- Monotonic improvement: variance decreases from SEM=0.039 (n=100) to SEM=0.001 (n=5000).
- At n >= 500, assemblies form reliable fixed-point attractors.
- Small networks (n < 300) have too much assembly-background overlap for stable attractors.

### H4: Weight Ceiling (w_max) Transition

**Question**: What is the minimum w_max that supports stable attractor formation?

**Parameters**: n=1000, k=100, p=0.05, beta=0.1, train_rounds=30, test_rounds=20.

| w_max | persistence (mean +/- SEM) | weight ratio (mean +/- SEM) | Cohen's d |
|-------|----------------------------|---------------------------|-----------|
| 1.0 | 0.921 +/- 0.006 | 1.1 +/- 0.0 | 40.5 |
| 1.2 | 0.930 +/- 0.005 | 1.2 +/- 0.0 | 53.1 |
| 1.5 | 0.932 +/- 0.010 | 1.6 +/- 0.0 | 27.3 |
| 2.0 | 0.963 +/- 0.006 | 2.1 +/- 0.0 | 48.8 |
| 3.0 | 0.983 +/- 0.005 | 3.2 +/- 0.1 | 59.1 |
| 5.0 | 0.994 +/- 0.002 | 5.2 +/- 0.1 | 127.9 |
| 10.0 | 0.995 +/- 0.002 | 10.4 +/- 0.2 | 169.8 |
| 20.0 | 0.995 +/- 0.002 | 16.5 +/- 0.4 | 169.8 |
| 50.0 | 0.998 +/- 0.001 | 16.2 +/- 0.3 | 213.0 |

**Findings**:
- Gradual transition from 0.92 (w_max=1.0) to 0.998 (w_max=50.0).
- This is a **smooth sigmoid**, not a sharp phase transition.
- Weight ratio saturates at ~16.5 for w_max >= 20 (training duration, not ceiling, becomes the bottleneck).
- Practical minimum: w_max >= 2.0 for persistence > 0.95.
- Default w_max = 20.0 provides near-optimal performance.

## Key Structural Predictor

Across all hypotheses, the **intra-assembly / inter-assembly weight ratio** is the fundamental structural predictor of attractor stability:
- Ratio ~ 1.0: no attractor (drift at ~0.88)
- Ratio >= 2.0: transition to stability (>0.95)
- Ratio >= 5.0: near-perfect attractor (>0.99)
- Ratio saturates at w_max (clamped by Dabagia et al. saturation rule)

## Known Limitations

1. **Single assembly per area**: H2 tests one attractor in isolation. With multiple competing assemblies, basins would shrink. Follow-up needed: `test_assembly_distinctiveness.py`.
2. **Weak null for H1**: The chance null (k/n) is trivially beaten. A stim-only-training null would better isolate the contribution of Hebbian self-strengthening.
3. **20 test rounds may miss slow drift**: Some conditions might appear stable at 20 rounds but drift over 100+ rounds.
4. **H4 is not a phase transition**: The smooth sigmoid should not be called a "phase transition" without evidence of critical scaling.

## Suggested Follow-up Experiments

1. **Competing attractors** (test_assembly_distinctiveness.py): Multiple assemblies in same area, measure basin interference.
2. **Long-horizon persistence**: Extend test_rounds to 100-500 to detect slow drift.
3. **Stim-only null**: Run H1 with stim-only training as the null distribution.
4. **Multi-area pattern completion**: Can area B recover a corrupted assembly in area A via learned associations?
