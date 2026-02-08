# Noise Robustness: Assembly Recovery Under Perturbation

**Script**: `research/experiments/stability/test_noise_robustness.py`
**Results file**: `noise_robustness_20260206_155522.json`
**Date**: 2026-02-06
**Brain implementation**: `src.core.brain.Brain` (with w_max saturation, corrected winner remapping and stimulus plasticity)

## Protocol

Tests whether Hebbian-trained assemblies can recover from partial or total corruption of their winner neurons, under three recovery mechanisms:

1. **Stimulus-driven recovery** (H1): After noise injection, re-apply the original stimulus with stim+self projection. Tests whether the learned stimulus pathway can restore the assembly.
2. **Autonomous recovery** (H2): After noise injection, run self-projection only (no stimulus). Tests whether the self-connectome alone can restore the assembly from its attractor basin.
3. **Association-based recovery** (H3): Establish two assemblies in areas A and B, associate via co-stimulation, corrupt B, recover B by projecting from clean A. Tests cross-area pattern completion as error correction.
4. **Sparsity scaling** (H4): Repeat autonomous recovery at k=sqrt(n) for n=200,500,1000,2000 to find where the attractor basin narrows.

### Noise injection

Replace a fraction of winner neurons with uniformly random non-winner neurons. At noise_frac=1.0, every winner is replaced -- the assembly is completely destroyed.

### Establishment

All assemblies trained via stim+self: `project({"s": ["A"]}, {"A": ["A"]})` x 30 rounds. Association trained via co-stimulation: `project({"sa": ["A"], "sb": ["B"]}, {"A": ["B"]})` x 30 rounds.

**Statistical methodology**: N_SEEDS=10 independent seeds per condition. One-sample t-test against null k/n. Cohen's d. Mean +/- SEM.

## Results

### H1: Stimulus-Driven Recovery (n=1000, k=100)

**Parameters**: n=1000, k=100, p=0.05, beta=0.10, w_max=20.0, recovery_rounds=20.

| noise fraction | recovery (mean +/- SEM) | Cohen's d |
|---------------|------------------------|-----------|
| 0.0 | **1.000 +/- 0.000** | inf |
| 0.1 | **1.000 +/- 0.000** | inf |
| 0.2 | **1.000 +/- 0.000** | inf |
| 0.3 | **1.000 +/- 0.000** | inf |
| 0.4 | **1.000 +/- 0.000** | inf |
| 0.5 | **1.000 +/- 0.000** | inf |
| 0.6 | **1.000 +/- 0.000** | inf |
| 0.8 | **1.000 +/- 0.000** | inf |
| 1.0 | **1.000 +/- 0.000** | inf |

**Finding**: Perfect recovery at all noise levels, including 100% corruption. The learned stimulus pathway (stim->A connections strengthened over 30 rounds of training) completely determines which neurons fire, regardless of the current winner state. This is expected: the stimulus provides a strong, specific external drive that overrides any internal state.

### H2: Autonomous Recovery -- Self-Only (n=1000, k=100)

**Parameters**: n=1000, k=100, p=0.05, beta=0.10, w_max=20.0, recovery_rounds=20.

| noise fraction | recovery (mean +/- SEM) | Cohen's d |
|---------------|------------------------|-----------|
| 0.0 | **1.000 +/- 0.000** | inf |
| 0.1 | **1.000 +/- 0.000** | inf |
| 0.2 | 0.999 +/- 0.001 | 284.3 |
| 0.3 | 0.999 +/- 0.001 | 284.3 |
| 0.4 | **1.000 +/- 0.000** | inf |
| 0.5 | **1.000 +/- 0.000** | inf |
| 0.6 | **1.000 +/- 0.000** | inf |
| 0.8 | **1.000 +/- 0.000** | inf |
| 1.0 | **1.000 +/- 0.000** | inf |

**Finding**: Near-perfect autonomous recovery across all noise levels. Even at 100% corruption (every winner replaced), the self-connectome alone recovers the full assembly.

**Mechanism**: At k/n=0.10 (10% density), after replacing all 100 winners with random neurons, roughly 10 of the new random winners happen to be trained assembly neurons by chance. These ~10 neurons have w_max=20 intra-assembly connections to other trained neurons, creating a strong input advantage (~14.5 vs ~5.0 expected input). This bootstraps a cascade: each subsequent self-projection round pulls in more trained neurons until the full assembly is restored.

**Implication**: At dense representations (k/n=0.10), the attractor basin covers the entire state space. This is a ceiling effect -- the experiment needs sparser representations to find the transition.

### H3: Association-Based Recovery (n=1000, k=100)

**Parameters**: n=1000, k=100, p=0.05, beta=0.10, w_max=20.0, recovery_rounds=20, association via co-stimulation x 30 rounds.

| noise fraction | B recovery (mean +/- SEM) | A intact (mean +/- SEM) | Cohen's d |
|---------------|--------------------------|------------------------|-----------|
| 0.0 | 0.994 +/- 0.003 | 1.000 +/- 0.000 | 106.0 |
| 0.1 | 0.994 +/- 0.002 | 1.000 +/- 0.000 | 127.9 |
| 0.2 | 0.997 +/- 0.002 | 1.000 +/- 0.000 | 185.7 |
| 0.3 | 0.993 +/- 0.003 | 1.000 +/- 0.000 | 84.3 |
| 0.4 | 0.984 +/- 0.005 | 1.000 +/- 0.000 | 51.6 |
| 0.5 | 0.993 +/- 0.003 | 1.000 +/- 0.000 | 108.5 |
| 0.6 | 0.995 +/- 0.002 | 1.000 +/- 0.000 | 169.8 |
| 0.8 | 0.989 +/- 0.004 | 1.000 +/- 0.000 | 74.3 |
| 1.0 | 0.993 +/- 0.002 | 1.000 +/- 0.000 | 132.3 |

**Findings**:
- Cross-area recovery is near-perfect (~0.984-0.997) across all noise levels. The A->B connections learned during co-stimulation association reliably reconstruct B from A.
- Recovery is flat across noise fractions -- the noise level in B is irrelevant because A drives B through the learned cross-area pathway.
- Source assembly A is perfectly preserved (1.000) in all conditions. Projecting from A to recover B does not corrupt A.
- Slightly below H1/H2 (~0.99 vs 1.00): the cross-area pathway is one hop removed, so recovery depends on the quality of the learned A->B connections rather than direct stimulus drive.

### H4: Autonomous Recovery vs Network Size (k=sqrt(n))

**Parameters**: p=0.05, beta=0.10, w_max=20.0, establish_rounds=30, recovery_rounds=20, autonomous (self-only) recovery.

| n | k | k/n | noise=0.3 | noise=0.5 | noise=0.7 | noise=1.0 |
|---|---|-----|-----------|-----------|-----------|-----------|
| 200 | 14 | 0.070 | 0.800 +/- 0.071 | 0.664 +/- 0.066 | 0.536 +/- 0.080 | 0.529 +/- 0.068 |
| 500 | 22 | 0.044 | 0.923 +/- 0.024 | 0.914 +/- 0.033 | 0.855 +/- 0.038 | 0.877 +/- 0.030 |
| 1000 | 31 | 0.031 | 0.945 +/- 0.019 | 0.971 +/- 0.009 | 0.945 +/- 0.017 | 0.958 +/- 0.014 |
| 2000 | 44 | 0.022 | 0.986 +/- 0.005 | 0.998 +/- 0.002 | 0.993 +/- 0.003 | 0.995 +/- 0.005 |

**Findings**:

Two gradients are visible:

1. **Recovery improves with network size** (reading down each column): At 100% noise: 0.529 (n=200) -> 0.877 (n=500) -> 0.958 (n=1000) -> 0.995 (n=2000). Larger networks have proportionally stronger intra-assembly weight structure relative to random drift.

2. **Noise level matters at small n** (reading across each row): At n=200, recovery degrades from 0.800 (30% noise) to 0.529 (100% noise) -- a clear monotonic decline. At n=2000, recovery is ~0.99 regardless of noise level. The attractor basin narrows with smaller networks, making noise fraction relevant.

**n=200 is the vulnerable regime**: With k=14 neurons in a 200-neuron area (k/n=0.07), the self-connectome cannot reliably self-heal from severe corruption. At 100% noise, only ~1 trained neuron lands in the random winner set by chance (k/n=0.07 x k=14 ≈ 1 neuron), which is insufficient to bootstrap recovery.

**n>=500 is robust**: Recovery exceeds 0.85 at all noise levels. The attractor basin is large enough that even complete corruption is recoverable.

**Critical threshold**: The transition from vulnerable to robust falls between n=200 and n=500 at k=sqrt(n). This corresponds to the regime where the expected number of chance-seeded trained neurons (k^2/n = k/n * k) transitions from ~1 to ~1 as well, but the total intra-assembly weight grows as k^2 * w_max, providing more total drive.

## Key Takeaways

1. **Stimulus-driven recovery is trivially perfect**: The learned stimulus pathway overrides any internal state. Noise injection is irrelevant when the original stimulus is available.
2. **Autonomous recovery at dense k/n=0.10 is also perfect**: The attractor basin covers the entire state space. Even 100% corruption recovers to 1.000.
3. **Cross-area association provides reliable error correction**: ~0.99 recovery via co-stimulation-trained A->B connections, regardless of noise level. Source assembly remains intact.
4. **Sparse representations reveal the actual transition**: At k=sqrt(n), n=200 shows genuine vulnerability (0.53 at 100% noise) while n>=500 is robust (>0.85).
5. **Recovery improves monotonically with network size**: 0.53 -> 0.88 -> 0.96 -> 0.99 at 100% noise for n=200 through n=2000.
6. **Hebbian-trained assemblies are error-correcting codes**: The learned weight structure (w_max=20 intra-assembly vs ~1.0 baseline) provides a strong enough signal to reconstruct assemblies from pure noise, given sufficient network size.

## Comparison with Attractor Dynamics Results

The attractor dynamics experiment (H2) also tested noise recovery at n=1000, k=100 and found identical results: ~0.994-0.998 recovery across all noise fractions. The noise robustness experiment extends this by:
- Adding stimulus-driven and association-based recovery mechanisms
- Testing the sparse regime (k=sqrt(n)) where recovery is non-trivial
- Identifying the critical network size threshold (n~200-500)
