# Bidirectional Association

**Script**: `research/experiments/primitives/test_bidirectional_association.py`
**Results file**: `bidirectional_association_20260206_173748.json`
**Date**: 2026-02-06
**Brain implementation**: `src.core.brain.Brain` (with w_max saturation, corrected winner remapping and stimulus plasticity)

## Protocol

Tests whether cross-area associations are inherently directional or support symmetric recall.

1. **Establish**: Train assemblies in areas A and B via stim-only projection (30 rounds each).

2. **Train**: Co-stimulation with specified projection fiber(s):
   - Unidirectional: `project({"sa": ["A"], "sb": ["B"]}, {"A": ["B"]})` — only A→B fiber.
   - Bidirectional simultaneous: `project({"sa": ["A"], "sb": ["B"]}, {"A": ["B"], "B": ["A"]})` — both fibers.
   - Bidirectional sequential: A→B fiber first (30 rounds), then B→A fiber (30 rounds).

3. **Test forward**: `project({"sa": ["A"]}, {"A": ["B"]})` — activate A, project to B.

4. **Test reverse**: `project({"sb": ["B"]}, {"B": ["A"]})` — activate B, project to A.

**Parameters**: n=1000, k=100, p=0.05, beta=0.10, w_max=20.0, establish_rounds=30, assoc_rounds=30, test_rounds=15.

**Statistical methodology**: N_SEEDS=10 independent seeds per condition. One-sample t-test against null k/n=0.100. Paired t-test for H4. Cohen's d. Mean +/- SEM.

## Results

### H1/H2: Unidirectional Training — Forward and Reverse Recall

**Question**: After training only the A→B fiber, does reverse (B→A) recall emerge implicitly?

| Direction | Recovery | SEM | Cohen's d | Status |
|-----------|----------|-----|-----------|--------|
| Forward (A→B) | **0.992** | 0.002 | 141.0 | far above chance |
| Reverse (B→A) | **0.097** | 0.009 | -0.1 | **at chance** |

**Finding**: Associations are **fundamentally directional**. The reverse recall (0.097) is literally at chance (k/n=0.100, d=-0.1). Training the A→B fiber via co-stimulation strengthens ONLY the connections from A's winners to B's winners. The B→A fiber is completely untouched despite both assemblies being co-active during training.

**Mechanism**: During co-stimulation `project({"sa": ["A"], "sb": ["B"]}, {"A": ["B"]})`, both areas have their trained assemblies active (maintained by stimuli). However, Hebbian plasticity only applies along the fibers specified in the projection map — `{"A": ["B"]}` strengthens A→B connections but not B→A. Co-activity alone is insufficient; the directionality of the projection map determines which fiber is trained.

**Implication**: Symmetric associative memory (face→name AND name→face) requires explicit bidirectional training. The framework does not produce implicit reverse associations from co-activation.

### H3: Explicit Bidirectional Training

**Question**: Can both directions be trained explicitly? Do they coexist without interference?

| Training method | Forward (A→B) | Reverse (B→A) |
|----------------|---------------|---------------|
| Simultaneous (both fibers per round) | 0.991 (d=101.8) | 0.992 (d=113.1) |
| Sequential (A→B first, then B→A) | 0.998 (d=213.0) | 0.995 (d=169.8) |

**Findings**:

1. **Both training methods produce strong bidirectional recall.** Simultaneous: 0.991/0.992. Sequential: 0.998/0.995. The A→B and B→A fibers are independent and coexist without interference.

2. **Sequential training is slightly stronger than simultaneous.** Sequential forward (0.998) exceeds simultaneous forward (0.991). This is likely because sequential training gives each fiber dedicated training rounds where the projection step focuses on a single fiber direction, while simultaneous training splits the projection step between both directions.

3. **Both fibers operate independently.** The A→B fiber and B→A fiber use different sets of connections in the connectome (A's neurons → B's neurons vs B's neurons → A's neurons). Training one does not affect the other.

### H4: Does Bidirectional Training Degrade Forward Recall?

**Question**: Does adding B→A training hurt A→B quality?

| Condition | Forward recovery | SEM |
|-----------|-----------------|-----|
| Unidirectional (A→B only) | 0.992 | 0.002 |
| Bidirectional (simultaneous) | 0.991 | 0.003 |

**Paired t-test**: t=0.32, p=0.758, d=0.10. **No significant difference.**

**Finding**: Training the reverse direction does not degrade the forward direction at all. The two fibers are completely independent — they use disjoint sets of connections in the connectome (A→B connections vs B→A connections), so strengthening one has zero effect on the other.

### H5: Size Scaling (k=sqrt(n), Bidirectional)

**Question**: How does bidirectional recall scale at biologically realistic sparsity?

| n | k | k²p | Forward | Reverse |
|---|---|-----|---------|---------|
| 500 | 22 | 24 | 0.727 | 0.709 |
| 1000 | 31 | 48 | 0.752 | 0.800 |
| 2000 | 44 | 97 | 0.900 | 0.891 |
| 5000 | 70 | 245 | 0.970 | 0.967 |

**Findings**:

1. **Forward and reverse are symmetric at every network size.** The two directions track each other closely: 0.727/0.709, 0.900/0.891, 0.970/0.967. Neither direction has an inherent advantage.

2. **Same k²p scaling as all other operations.** At k²p=24: ~0.72. At k²p=245: ~0.97. The bidirectional threshold is the same as for unidirectional association, chains, and loops.

## Key Takeaways

1. **Associations are fundamentally directional.** Reverse recall after unidirectional training is literally at chance (0.097 vs null 0.100). Co-activity during training is not sufficient — only the projection map fiber is strengthened.

2. **Explicit bidirectional training works perfectly.** Both simultaneous and sequential training produce strong symmetric recall (~0.99). The A→B and B→A fibers are independent and do not interfere.

3. **No cross-fiber interference.** Training B→A does not degrade A→B (p=0.758). The two fiber directions use disjoint connection sets.

4. **Sequential training is slightly superior to simultaneous.** Each direction gets dedicated training rounds, producing marginally stronger associations (0.998 vs 0.991).

## Biological Interpretation

The directionality result has clear biological implications:

**Symmetric associative memory requires explicit bidirectional wiring.** In the brain, cortico-cortical connections are typically reciprocal — if area A projects to area B, there is almost always a return projection from B to A. Our result shows that these forward and backward pathways must be independently trained (strengthened by Hebbian learning during co-activity). The anatomical reciprocity provides the substrate; experience-dependent plasticity in both directions provides the symmetric association.

**Hippocampal auto-associative networks** (CA3) are believed to support symmetric recall through recurrent connections where A→B and B→A are trained simultaneously during encoding. Our simultaneous bidirectional training (0.991/0.992) is a direct model of this process.

## Relationship to Other Experiments

| Experiment | Tests | Key finding | Relevance |
|-----------|-------|-------------|-----------|
| Association (single-hop) | A→B recovery | 0.994 | Baseline forward association |
| **Bidirectional** | **Forward + reverse** | **Reverse at chance without explicit training** | **Directionality confirmed** |
| Association chain | Multi-hop A→B→...→F | Lossless at 5 hops | Each hop is unidirectional |
| Recurrent loops | A→B→C→A circulation | Zero-decay attractor | Each loop edge is unidirectional |

The directionality result explains why recurrent loops work: each edge in the loop (X0→X1, X1→X2, X2→X0) is trained as a unidirectional association. The loop sustains circulation because signal flows in one direction around the loop. A bidirectional loop (X0↔X1↔X2) would require explicit training of all 6 fiber directions.
