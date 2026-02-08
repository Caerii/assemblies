# Association Primitive: Cross-Area Binding and Pattern Completion

**Script**: `research/experiments/primitives/test_association.py`
**Results file**: `association_20260206_144409.json`
**Date**: 2026-02-06 (co-stimulation protocol)
**Brain implementation**: `src.core.brain.Brain` (with w_max saturation, corrected winner remapping and stimulus plasticity)

## Protocol Changes from Prior Run

The previous association test used bare projection (`project({}, {"A": ["B"]})`) during association training. This allowed both assemblies to drift freely, producing a flat ~0.86 recovery ceiling regardless of training rounds.

The current test uses **co-stimulation**: both stimuli fire simultaneously during association training, holding each assembly stable through its learned stimulus pathway while cross-area connections learn:

```python
# Bidirectional association via co-stimulation
project({"sa": ["A"], "sb": ["B"]}, {"A": ["B"], "B": ["A"]})

# Unidirectional (A->B only)
project({"sa": ["A"], "sb": ["B"]}, {"A": ["B"]})
```

This is biologically realistic: association occurs when two signals co-occur in the environment (e.g., a child hearing "dog" while seeing a dog -- both cortical representations are simultaneously active).

## Protocol

1. **Establish** assemblies in areas A and B via stim+self training (30 rounds each).
2. **Associate** via co-stimulation: both stimuli fire simultaneously while cross-area projections learn (N rounds).
3. **Test recovery**: Corrupt B (replace winners with random neurons), activate A via its stimulus, project A->B for 20 rounds, measure B overlap with original trained B.
4. **Test identity**: After association, re-activate stimulus A with stim+self, measure overlap with original trained A.

**Statistical methodology**: N_SEEDS=10 independent seeds per condition. One-sample t-test against null k/n. Paired t-test for H3. Cohen's d. Mean +/- SEM.

## Results

### H1: Basic Association Formation

**Parameters**: n=1000, k=100, p=0.05, beta=0.10, w_max=20.0, assoc_rounds=30.

| Metric | Value |
|--------|-------|
| Recovery (mean +/- SEM) | 0.994 +/- 0.002 |
| Chance null (k/n) | 0.100 |
| Cohen's d | 127.9 |
| p-value | < 0.0001 |

**Finding**: Co-stimulation association enables near-perfect recovery of B from A (0.994). This is a dramatic improvement over the old bare-projection protocol (0.856). Because both stimuli anchor their assemblies during training, the cross-area connections learn precisely which B-neurons should fire when A is active.

### H2: Recovery vs Association Training Rounds

**Parameters**: n=1000, k=100, p=0.05, beta=0.10, w_max=20.0.

| assoc rounds | recovery (mean +/- SEM) | Cohen's d |
|-------------|------------------------|-----------|
| 1 | 0.120 +/- 0.006 | 1.0 |
| 5 | 0.437 +/- 0.018 | 6.1 |
| 10 | 0.722 +/- 0.009 | 22.7 |
| 20 | 0.965 +/- 0.007 | 38.1 |
| 30 | 0.989 +/- 0.003 | 101.5 |
| 50 | 0.993 +/- 0.003 | 108.5 |

**Finding**: Recovery now shows a clear monotonic learning curve: 0.120 (1 round, near chance) -> 0.722 (10 rounds) -> 0.993 (50 rounds). This is the expected behavior -- Hebbian reinforcement progressively strengthens cross-area connections. The old protocol's flat curve (0.86 at all durations) was an artifact of unstable assemblies, not a property of the association mechanism.

**Key insight**: Most learning happens in the first 20-30 rounds. Beyond 30 rounds, returns are diminishing (0.989 -> 0.993). The association saturates around 0.99.

### H3: Bidirectional vs Unidirectional

**Parameters**: n=1000, k=100, p=0.05, beta=0.10, w_max=20.0, assoc_rounds=30.

| Direction | recovery (mean +/- SEM) | Cohen's d |
|-----------|------------------------|-----------|
| Bidirectional | 0.994 +/- 0.003 | 92.5 |
| Unidirectional | 0.990 +/- 0.003 | 109.0 |

**Paired t-test**: t=0.94, p=0.373 (not significant), d=0.3

**Finding**: No significant difference between bidirectional and unidirectional association. Both achieve ~0.99 recovery. This is expected: the recovery test projects A->B, so B->A connections (present only in bidirectional) are irrelevant to this particular recovery direction.

### H4: Identity Preservation

**Parameters**: n=1000, k=100, p=0.05, beta=0.10, w_max=20.0, assoc_rounds=30.

| Test | recovery (mean +/- SEM) |
|------|------------------------|
| Stim -> A | **1.000 +/- 0.000** |
| Stim -> B | **1.000 +/- 0.000** |

**Finding**: Perfect identity preservation. After 30 rounds of co-stimulation association, re-activating the original stimulus with stim+self recovers the original assembly with 100% fidelity across all seeds. Association does not corrupt source assemblies.

### H1 Extended: Recovery vs Network Size

**Parameters**: k=sqrt(n), p=0.05, beta=0.10, w_max=20.0, assoc_rounds=30.

| n | k | recovery (mean +/- SEM) | Cohen's d |
|---|---|------------------------|-----------|
| 200 | 14 | 0.471 +/- 0.039 | 3.3 |
| 500 | 22 | 0.700 +/- 0.034 | 6.1 |
| 1000 | 31 | 0.800 +/- 0.019 | 12.7 |
| 2000 | 44 | 0.911 +/- 0.013 | 21.8 |

**Finding**: Recovery improves monotonically with network size: 0.471 (n=200) -> 0.911 (n=2000). The k=sqrt(n) scaling means the assembly representation becomes sparser relative to area size as n grows, which improves the signal-to-noise ratio for cross-area connections. At n=2000 (k=44), recovery reaches 0.91 -- strong but below the 0.99 seen at n=1000, k=100, because the k/n ratio is much smaller (0.022 vs 0.100).

## Key Takeaways

1. **Co-stimulation enables near-perfect association**: 0.994 recovery at n=1000, k=100 -- a substantial improvement over bare projection (0.856).
2. **Training rounds matter**: Clear learning curve from 0.120 (1 round) to 0.993 (50 rounds), with saturation at ~30 rounds. This confirms genuine Hebbian reinforcement of cross-area connections.
3. **Directionality is irrelevant for same-direction recovery**: Bidirectional and unidirectional produce equivalent A->B recovery.
4. **Identity is perfectly preserved**: Association does not damage source assemblies (1.000 recovery).
5. **Recovery scales with network size**: 0.471 (n=200) to 0.911 (n=2000) at k=sqrt(n).
6. **Biological plausibility**: The co-stimulation protocol models natural associative learning -- two sensory signals co-occurring and strengthening cross-area links while individual representations remain anchored by ongoing input.
