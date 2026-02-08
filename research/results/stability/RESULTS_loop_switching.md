# Multi-Pattern Loop Switching (Working Memory Capacity)

**Script**: `research/experiments/stability/test_loop_switching.py`
**Results file**: `loop_switching_20260206_191220.json`
**Date**: 2026-02-06
**Brain implementation**: `src.core.brain.Brain` (with w_max saturation, corrected winner remapping, stimulus plasticity, and optional homeostatic normalization)

## Protocol

Tests whether recurrent loops can store and switch between multiple distinct circulating patterns.

1. **Establish**: For each pattern i (of N total), establish assemblies in all 3 loop areas via independent stimuli (30 rounds each): si0→X0, si1→X1, si2→X2.

2. **Train loop associations**: For each pattern i, train the full loop via co-stimulation (30 rounds per edge): X0(i)→X1(i), X1(i)→X2(i), X2(i)→X0(i). All patterns share the same 3 areas and same cross-area connectomes.

3. **Test switching**: Kick-start pattern A (15 rounds stimulus-driven circulation), run autonomous (20 rounds), measure. Then kick-start pattern B, run autonomous, measure overlap with both A and B.

**Parameters**: n=1000, k=100, p=0.05, beta=0.10, w_max=20.0, establish_rounds=30, assoc_rounds=30, kick_rounds=15, autonomous_rounds=20.

**Statistical methodology**: N_SEEDS=10. One-sample t-test against null k/n=0.100. Paired t-test for H3. Cohen's d. Mean +/- SEM.

## Results

### H1: Single-Pattern Baseline

| Metric | Value | SEM | Cohen's d |
|--------|-------|-----|-----------|
| Final overlap (1 pattern) | **0.996** | 0.001 | 398.1 |

Replicates the recurrent loop experiment: one pattern circulates as a perfect fixed-point attractor.

### H2: Pattern Switching (2 Patterns Trained)

**Question**: Can the loop switch from pattern A to pattern B?

| Phase | Overlap with A | Overlap with B |
|-------|---------------|---------------|
| After kick A, autonomous | **0.874** | — |
| After kick B, autonomous | **0.708** | **0.376** (d=2.0) |

**Findings**:

1. **Pattern A does not reach 0.99 even before switching.** With two patterns trained, pattern A's autonomous circulation is only 0.874 — significantly below the 0.996 single-pattern baseline. The interference from training pattern B has already degraded pattern A's cross-area connection quality.

2. **Switching does not cleanly replace A with B.** After kicking pattern B, overlap with A drops from 0.874 to 0.708 — but NOT to chance (0.100). Pattern A's assembly neurons remain partially active. Meanwhile, pattern B only reaches 0.376 — well above chance but far below useful fidelity.

3. **The loop enters a blended state.** Neither pattern dominates. The cross-area connectomes contain strengthened connections for BOTH patterns, and winner-take-all selection at each area produces a compromise set of winners that partially matches both patterns. During autonomous circulation, this blended state reinforces itself — there is no mechanism to separate the two patterns once they are mixed.

4. **The blending problem is fundamental.** In stimulus-driven association tests, the stimulus provides a clean external anchor that biases winner selection toward the correct pattern. In autonomous circulation, no such anchor exists after the kick-start phase ends. The loop's own blended state becomes its own input, perpetuating the blend.

### H3: Interference from Dual Training

**Question**: Does training a second pattern degrade the first?

| Condition | Pattern A overlap | SEM |
|-----------|------------------|-----|
| Single-pattern (A only trained) | 0.996 | 0.001 |
| Dual-pattern (A and B trained) | 0.874 | 0.005 |

**Paired t-test**: t=12.33, p<0.0001. **Highly significant degradation.**

**Finding**: Training a second pattern in the same loop causes a 0.122 drop in pattern A's circulation quality (0.996 → 0.874). This is much worse than cross-area association interference at 2 pairs (which showed ~0.994 recovery). The difference: association interference is tested with stimulus support, while loop interference compounds through autonomous circulation without any external anchor.

### H4: Pattern Capacity

**Question**: How does switching quality degrade with the number of trained patterns?

| Patterns | Mean switching quality | SEM | Cohen's d |
|----------|----------------------|-----|-----------|
| 1 | **0.994** | 0.002 | 138.8 |
| 2 | **0.606** | 0.012 | 13.0 |
| 3 | **0.458** | 0.012 | 9.1 |
| 5 | **0.332** | 0.012 | 6.1 |

**Findings**:

1. **Steep capacity cliff.** Switching quality drops from 0.994 (1 pattern) to 0.606 (2 patterns) — a 39% relative loss from adding a single additional pattern. By 5 patterns, quality is 0.332 — above chance (d=6.1) but functionally unusable.

2. **Much steeper than association interference.** Cross-area association interference in a shared A→B connectome showed: 1 pair=0.997, 10 pairs=0.770 (23% relative loss over 10 items). Loop switching shows: 1 pattern=0.994, 2 patterns=0.606 (39% relative loss from 1 item). The loop is dramatically more sensitive to multi-item interference.

3. **Compounding mechanism.** The steep degradation arises because loop interference compounds through three mechanisms:
   - **Per-area interference**: Each area (X0, X1, X2) contains multiple trained assemblies that compete during winner selection.
   - **Per-hop interference**: Each cross-area connectome (X0→X1, X1→X2, X2→X0) has multiple pattern-specific connections that create cross-talk.
   - **Temporal compounding**: During autonomous circulation, the degraded output of one hop becomes the input to the next, and the error is amplified each cycle rather than corrected (because there is no clean external signal).

4. **The loop is a single-item buffer.** Practical circulation fidelity (>0.9) requires a single trained pattern. The recurrent loop is not a multi-item working memory store — it is a sustained-attention mechanism for maintaining ONE active representation.

### H5: Two-Pattern Switching at Sparse Coding (k=sqrt(n))

| n | k | k²p | Switch quality | SEM | Cohen's d |
|---|---|-----|---------------|-----|-----------|
| 500 | 22 | 24 | 0.361 | 0.050 | 2.0 |
| 1000 | 31 | 48 | 0.386 | 0.037 | 3.0 |
| 2000 | 44 | 97 | 0.479 | 0.042 | 3.4 |
| 5000 | 70 | 245 | 0.466 | 0.068 | 2.1 |

**Finding**: Scaling the network does not help. Two-pattern switching quality is ~0.4-0.5 regardless of network size. This is strikingly different from every other experiment, where increasing k²p consistently improved performance.

**Mechanism**: Two competing effects cancel out:
- Higher k²p → better per-hop relay quality → should help
- But: the fundamental blending problem (no external anchor during autonomous circulation) is independent of relay quality. Better relay just means the blended state is maintained more faithfully — it doesn't help separate the patterns.

**Comparison with single-pattern loop at same k²p** (from recurrent loop H4):

| k²p | Single-pattern loop | Two-pattern switching | Relative penalty |
|-----|--------------------|-----------------------|-----------------|
| 24 | 0.488 | 0.339 | -31% |
| 48 | 0.587 | 0.461 | -21% |
| 97 | 0.861 | 0.420 | -51% |
| 245 | 0.967 | 0.469 | -52% |

The relative penalty from multi-pattern training is roughly constant (-30 to -50%) across all k²p values. At high k²p, the single-pattern loop is excellent (0.967) but two-pattern switching is still poor (0.469). This confirms that the blending problem is orthogonal to relay quality.

### H6: Homeostatic Synaptic Normalization — Negative Result

Turrigiano-style synaptic scaling (preserving each source neuron's total outgoing weight budget after plasticity) was tested and **does not help**. Normalization reduces training interference (pattern A improves from 0.874 → 0.938 when a second pattern is trained) but makes switching *worse* (pattern B takeover drops from 0.376 → 0.268) because the conserved synaptic budget makes the active pattern stickier. Net effect on capacity: slightly negative (-0.007 to -0.021). The multi-pattern problem requires active inhibition, not passive homeostasis. This extension has been reverted from the codebase.

## Key Takeaways

1. **Recurrent loops are single-item working memory buffers.** One pattern circulates at 0.994 as a perfect fixed-point attractor. Adding a second pattern drops switching quality to 0.606. By 5 patterns, quality is 0.332. The loop supports sustained attention for one representation, not multi-item working memory.

2. **Pattern blending, not pattern erasure.** Switching does not replace pattern A with pattern B. Instead, both patterns remain partially active in a blended state. The loop lacks a mechanism to suppress the previous pattern during switching.

3. **Interference compounds through autonomous circulation.** Unlike stimulus-driven association (where the source stimulus provides a clean anchor), autonomous loops amplify interference at each cycle. This is the fundamental reason loops are more sensitive to multi-item interference than associations.

4. **Network size does not help.** Two-pattern switching at k=sqrt(n) is ~0.4-0.5 regardless of n. The blending problem is structural, not a capacity limitation that can be overcome with larger networks.

5. **Training interference alone causes significant degradation.** Even before any switching attempt, training two patterns drops single-pattern quality from 0.996 to 0.874 (p<0.0001).

6. **Homeostatic normalization does not help (and slightly hurts).** Turrigiano-style synaptic scaling reduces training interference (+0.064 on pattern A fidelity) but makes the active pattern harder to suppress during switching (-0.108 on pattern B takeover). The net effect on capacity is slightly negative. The multi-pattern problem requires active inhibition, not passive homeostasis.

## Biological Interpretation

This result has important implications for the neural basis of working memory:

**Multi-item working memory requires parallel, independent loops.** The well-known "4±1" working memory capacity limit (Cowan, 2001) cannot arise from a single recurrent loop holding multiple items. Instead, it likely reflects the number of independent recurrent circuits (separate area-to-area loops or separate neural subpopulations within areas) that can simultaneously sustain autonomous circulation without cross-talk.

**Pattern switching requires active suppression, not homeostasis.** Our framework uses only Hebbian excitatory learning — there is no inhibitory mechanism to suppress the previous pattern when switching. Homeostatic synaptic normalization (Turrigiano scaling) was tested and found to make the problem *worse* by making active patterns stickier. Biological circuits include GABAergic inhibitory interneurons (particularly PV+ basket cells) that can rapidly suppress the current active representation via perisomatic inhibition, clearing the way for a new pattern. This is the most promising next extension.

**Sustained attention vs flexible working memory.** The Assembly Calculus recurrent loop is a model of sustained attention (maintaining one item indefinitely) rather than flexible working memory (maintaining and switching between multiple items). This distinction maps onto the difference between simple maintenance tasks (e.g., holding a phone number) and complex manipulation tasks (e.g., mental arithmetic with multiple operands).

**Separate storage vs active maintenance.** The contrast between association interference (graceful, 10+ items) and loop interference (catastrophic, ~1 item) mirrors the biological distinction between long-term storage (hippocampal/cortical associations) and active working memory (prefrontal recurrent loops). The framework naturally produces both: associations for durable multi-item storage, loops for single-item active maintenance.

## Relationship to Other Experiments

| Experiment | Substrate | Items | Degradation pattern | Anchor |
|-----------|-----------|-------|--------------------|----|
| Catastrophic forgetting | A→B connectome | 1-10 pairs | Graceful: 0.997→0.770 | Stimulus |
| **Loop switching** | **3-area loop** | **1-5 patterns** | **Steep: 0.994→0.332** | **None (autonomous)** |
| Loop + normalization (reverted) | 3-area loop | 1-5 patterns | Steep: 0.987→0.311 | None (autonomous) |
| Recurrent loops (single) | 3-area loop | 1 pattern | Perfect: 0.991 at t=50 | None |

The critical difference is the presence or absence of an external anchor (stimulus). With stimulus support, interference is manageable (associations store 10+ items with graceful degradation). Without stimulus support (autonomous loops), interference compounds through recurrent circulation, limiting capacity to a single item.
