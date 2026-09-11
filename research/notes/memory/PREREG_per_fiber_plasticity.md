# Per-fiber plasticity routing and unclipped gain

Registered before implementing the experiment or observing any of its measurements.
The existing `RATE-HETEROGENEITY` register entry reports beta 0.5 against 0.01
for ten rounds, but has no identified producer, seeds, engine or artifact; its
fast arm reaches `w_max`, so the reported ratio does not isolate the two rates.
That number remains unreproduced. This protocol replaces it with a causal,
unclipped test of the public per-fiber route.

## Question and scope

Does `Brain.update_plasticity(source, target, beta)` make the named fiber use its
own multiplicative Hebbian rate under otherwise identical materialized projection
conditions? This is a finite implementation-level mechanism claim. It is not a
claim about learning quality, biological rate heterogeneity, sampled recurrence,
hashed-backend parity, or an emergent assembly-calculus operation.

## Protocol

Use the materialized NumPy engine through `Brain(engine="numpy_explicit")` and
ordinary areas. For brain seeds 201..220,
construct source areas A and B and three target areas, all with n=80, k=10,
p=0.2, default beta=0.005, `norm_init=False`, and `w_max=20`. Give A and B the
same fixed neuron-ID winners 0..9. Before traffic, copy A's complete materialized
connectome into B's corresponding connectome for each target. Assert each pair is
bit-identical, contains at least one present synapse in the active pre/post block,
and is stored independently after copying.

Project A and B together into each target for 50 rounds, reinjecting the identical
source winners each round:

| Cell | A beta | B beta | Purpose |
|---|---:|---:|---|
| forward | 0.06 | 0.005 | primary contrast |
| swapped | 0.005 | 0.06 | source-label control |
| equal | 0.005 | 0.005 | constructed null |

Set every rate only through `Brain.update_plasticity`. Both fibers into a target
therefore see the same target winners on every round. Retain each round's target
winners, the initial present-edge mask for the final co-active block, and the
final selected weights. Do not discard a seed or a zero edge. A missing present
edge, changed source activity, unequal within-target base matrices, nonfinite
weight, or winner-count mismatch invalidates the run.

For each seed and cell, compute the geometric mean over initially present edges in
the cross-product of source winners and the target's final winners, separately for
A and B. Report `fast/slow` for the two directional cells, `A/B` for the equal
cell, and the maximum selected weight. The theoretical unclipped values after 50
co-firings from unit weight are `(1.06)^50` and `(1.005)^50`; the implementation
must record its observed values rather than substitute these formulas.

Run through the shared immutable runner as protocol
`mechanism.per-fiber-plasticity`, version 1, with resolved engine
`numpy_explicit`, a mandatory unique tag, source archive, exact
parameters and all twenty seed identities. A three-seed smoke is VOID and cannot
satisfy the hypothesis.

## Statistics and predeclared bars

Summarize the three per-seed ratios and both rate-specific geometric means with
`ensemble_from_values` and nominal 95% Student-t intervals over the twenty brain
seeds. Preserve zero-width intervals if the controlled construction is
deterministic; they do not establish a population probability.

The primary hypothesis passes only if all conditions hold for every seed:

1. forward fast/slow > 10 and swapped fast/slow > 10;
2. equal A/B differs from 1 by at most 1e-6;
3. the fast and slow observed means differ from `(1 + beta)^50` by at most
   2e-5 relative error;
4. maximum selected weight is strictly below 20, so clipping cannot explain the
   contrast;
5. target winner histories are identical between forward and swapped cells after
   their base target connectomes are made identical.

The constructed null is required evidence: if equal beta produces a directional
split, the instrument is invalid. Conversely, passing only the null does not show
that the public setter reached the engine.

## API rejection controls

Before any experiment measurement, unit tests must show that an unknown source,
unknown target, boolean, negative, NaN or infinite beta is rejected by
`Brain.update_plasticity` before either the public area record or an engine store
changes. The legacy area-level setters must continue to raise rather than silently
write the dead bookkeeping route. These rejection checks are API contracts, not
seeded scientific observations.

## Adoption rule

On PASS, `RATE-HETEROGENEITY` may cite this registration, producer and immutable
artifact and replace its saturated legacy number with the unclipped result. Any
failed bar is retained and the claim remains provenance-gapped. Backend agreement
requires separate registered conformance studies; this result alone cannot extend
to sampled NumPy, hashed CUDA or other production substrates.

## Amendment 1 (before implementation or data)

The initial wording said "through explicit areas" and named the engine
`numpy_explicit-materialized`. The public materialized route is actually ordinary
areas owned by `Brain(engine="numpy_explicit")`; `explicit=True` means an auxiliary
engine inside another primary backend. The protocol now names the public route and
the engine's canonical identifier `numpy_explicit`. No parameter, bar or seed
changed.

## Results (2026-09-10)

The exact twenty-seed study passed all seven predeclared checks. The immutable
[run artifact](../../results/runs/mechanism.per-fiber-plasticity/per-fiber-plasticity-20260910/results.json)
records source commit `70d386a`, the canonical `numpy_explicit` engine, parameters,
seeds 201..220, summaries and complete raw selected blocks in a digest-bound gzip
attachment.

The fast rate produced geometric mean weight 18.4201012 and the slow rate
1.2832253, a ratio of 14.3545340 [14.3545340, 14.3545340]. Swapping which source
received the fast rate produced the same ratio and exactly the same target winner
history in every seed. The equal-rate null was exactly 1.0. The largest selected
weight was 18.4201012, strictly below `w_max=20`, and both rate-specific weights
matched their predeclared multiplicative formulas within tolerance. The raw-block
recomputation reproduced every stored per-seed mean, present-edge count, maximum
and history comparison.

This resolves the old entry's producer, engine, seed and clipping gaps for the
materialized NumPy mechanism. The original beta 0.5 versus 0.01 saturated number
remains unreproduced and is retired as evidence. The controlled result shows that
the public route selects a fiber-specific multiplicative rate; it does not show a
learning-quality advantage or establish equivalent semantics on another backend.
