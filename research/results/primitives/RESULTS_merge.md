# Merge Primitive: Compositional Assembly Formation

**Script**: `research/experiments/primitives/test_merge.py`
**Results file**: `merge_composition_20260206_150412.json`
**Date**: 2026-02-06
**Brain implementation**: `src.core.brain.Brain` (with w_max saturation, corrected winner remapping and stimulus plasticity)

## Protocol

Merge tests the third fundamental Assembly Calculus operation: when two assemblies in different areas project simultaneously to a third area, does the resulting assembly encode information about both parents?

Biologically, merge models compositional representation -- a child sees a RED BALL and visual cortex merges "red" and "ball" into a bound representation in a downstream area.

### Setup

- Three explicit areas: A (source 1), B (source 2), C (target)
- Two stimuli: sa (drives A), sb (drives B)
- All areas use the same parameters (n, k, p, beta, w_max)

### Phases (single brain instance)

1. **Establish**: Train assemblies in A and B via stim+self (30 rounds each)
2. **A-only projection**: Project A->C via `project({"sa": ["A"]}, {"A": ["C"]})`, record C_A
3. **Reset C**, **B-only projection**: Project B->C, record C_B
4. **Reset C**, **Merge**: Simultaneous co-stimulation `project({"sa": ["A"], "sb": ["B"]}, {"A": ["C"], "B": ["C"]})`, record C_AB
5. **Measure overlaps**: C_AB vs C_A, C_AB vs C_B, C_A vs C_B

C is reset (random winners) between phases to prevent carryover. All phases use the same brain instance so connectome weights accumulate realistically.

**Statistical methodology**: N_SEEDS=10 independent seeds per condition. One-sample t-test against null k/n. Cohen's d. Mean +/- SEM.

## Results

### H1: Merge Composition

**Parameters**: n=1000, k=100, p=0.05, beta=0.10, w_max=20.0, merge_rounds=30.

| Metric | Value |
|--------|-------|
| C_AB vs C_A overlap (mean +/- SEM) | 0.552 +/- 0.010 |
| C_AB vs C_B overlap (mean +/- SEM) | 0.548 +/- 0.011 |
| C_A vs C_B overlap (mean +/- SEM) | 0.100 +/- 0.008 |
| Merge quality = min(cab_ca, cab_cb) | 0.523 +/- 0.006 |
| Composition score = avg(cab_ca, cab_cb) | 0.550 +/- 0.004 |
| Cohen's d (quality vs chance) | 21.1 |
| p-value | < 0.0001 |

**Findings**:
- The merged assembly C_AB has **symmetric overlap** with both parent projections (~0.55 for both C_A and C_B). This confirms compositional encoding -- C_AB is not biased toward either parent.
- C_A and C_B are **independent** (0.100 overlap = chance k/n). The A-only and B-only projections produce unrelated assemblies in C, as expected since A and B have independent connectomes to C.
- Merge quality (0.523) is far above chance (0.100), with Cohen's d = 21.1. The merged representation genuinely encodes both inputs.
- The ~0.55 overlap means roughly half of C_AB's neurons come from C_A and half from C_B, which is the optimal split for encoding two equal-strength inputs.

### H2: Merge Quality vs Training Rounds

**Parameters**: n=1000, k=100, p=0.05, beta=0.10, w_max=20.0.

| merge rounds | quality (mean +/- SEM) | Cohen's d |
|-------------|----------------------|-----------|
| 1 | 0.498 +/- 0.006 | 19.5 |
| 5 | 0.520 +/- 0.009 | 15.1 |
| 10 | 0.508 +/- 0.008 | 17.0 |
| 20 | 0.514 +/- 0.008 | 16.5 |
| 30 | 0.519 +/- 0.008 | 16.7 |
| 50 | 0.518 +/- 0.014 | 9.6 |

**Findings**:
- Merge quality is **essentially flat** across training durations (0.498-0.520). Even a single round of merge achieves the same composition as 50 rounds.
- This is fundamentally different from association (which shows a clear learning curve). Merge composition is structurally determined by the random graph -- when A and B project simultaneously to C, the winner-take-all competition in C naturally selects neurons that receive strong input from both sources.
- Extended training does not improve merge quality because the A->C and B->C connectomes are already well-established from phases 1-2. The merge step simply combines two existing projections.

### H3: Merge Recovery (Partial Reactivation)

**Parameters**: n=1000, k=100, p=0.05, beta=0.10, w_max=20.0, merge_rounds=30.

| Test | recovery (mean +/- SEM) |
|------|------------------------|
| A-only recovery of C_AB | **1.000 +/- 0.000** |
| B-only recovery of C_AB | **1.000 +/- 0.000** |

**Findings**:
- After merge training, activating just A (via stimulus sa) and projecting A->C **perfectly recovers the full merged assembly C_AB**. Same for B-only.
- This is a strong result: each parent independently contains enough information to reconstruct the merged representation. The A->C connections learned during merge training point to the same C_AB neurons that the B->C connections point to.
- This validates merge as a robust compositional primitive. In biological terms: if you have a merged representation of "red ball", seeing just "red" can reactivate the full composite representation in the downstream area.

### H4: Merge Quality vs Network Size

**Parameters**: k=sqrt(n), p=0.05, beta=0.10, w_max=20.0, merge_rounds=30.

| n | k | quality (mean +/- SEM) | null (k/n) | Cohen's d |
|---|---|----------------------|------------|-----------|
| 200 | 14 | 0.450 +/- 0.024 | 0.070 | 5.0 |
| 500 | 22 | 0.432 +/- 0.023 | 0.044 | 5.4 |
| 1000 | 31 | 0.455 +/- 0.016 | 0.031 | 8.2 |
| 2000 | 44 | 0.461 +/- 0.011 | 0.022 | 12.9 |

**Findings**:
- Merge quality is stable across network sizes (~0.43-0.46) with k=sqrt(n) scaling. This is slightly lower than the 0.52 seen at n=1000, k=100, because the sparser representation (k/n = 0.022-0.070) gives fewer neurons to split between two parents.
- All conditions are highly significant vs chance (d = 5.0-12.9). Effect size increases with n because chance overlap decreases while merge quality stays constant.
- The consistency across scales suggests merge is a robust fundamental operation that does not depend on specific parameter regimes.

## Key Takeaways

1. **Merge creates genuine compositional representations**: ~55% overlap with each parent, far above the 10% chance level (d = 21.1).
2. **Symmetric encoding**: C_AB overlaps equally with C_A and C_B, confirming unbiased composition.
3. **Perfect partial recovery**: Activating either parent alone perfectly recovers the full merged assembly (1.000 for both A and B).
4. **Training rounds don't matter**: Quality is flat from 1 to 50 rounds. Merge composition is structurally determined by the graph topology, unlike association which requires Hebbian reinforcement.
5. **Scales robustly**: Quality ~0.43-0.46 across n=200 to n=2000 with k=sqrt(n).
6. **C_A and C_B are independent**: 0.100 overlap (chance), confirming that the two sources produce genuinely different projections that merge combines.

## Comparison with Other Primitives

| Property | Projection | Association | Merge |
|----------|-----------|-------------|-------|
| Key metric | Persistence 0.71 | Recovery 0.99 | Quality 0.52 |
| Training dependence | Saturates ~10 rounds | Saturates ~30 rounds | Flat (1 round sufficient) |
| Cross-area recovery | 1.000 | 0.99 | 1.000 (partial reactivation) |
| Identity preservation | N/A | 1.000 | N/A |
| Mechanism | Self-connectome attractor | Hebbian cross-area learning | Winner-take-all competition |
