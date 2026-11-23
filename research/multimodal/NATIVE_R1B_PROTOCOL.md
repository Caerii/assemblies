# R1b — Native Brain identity-retention study

**Status: implementation committed; not yet executed on the native engine.**
This preregisters protocol, interpretation bounds, and execution commands before
any outcome is inspected. Branch: `research/multimodal-continual-assemblies`.

## Question

Does a learned recurrent assembly remain recoverable from its original
stimulus as other independent stimuli are successively learned in the same
area? Does recurrence improve retention over feedforward-only projection,
and is any apparent benefit actually attributable to plasticity?

R1a used an externally specified random target assembly in a single explicit
synaptic fiber, which cannot recruit targets. R1b asks a DIFFERENT question:
the native engine chooses an assembly via k-WTA, and its learned winners become
the item identity to be recalled.

## Arms (same n, k, p, seed, registrations, item order)

- **recurrent**: beta=.1; stimulus->memory with memory recurrence;
- **feedforward**: beta=.1; stimulus->memory without recurrence;
- **no_plasticity**: beta=0; same recurrent schedule.

Use `Brain(engine="numpy_exact", seed=seed, p=.1, norm_init=False)` for
the primary diagnostic to avoid sampled-connectome recruitment artifacts.
Clear memory activity between stimulus writes; each new stimulus is registered
up front. Each arm runs in its own seeded Brain. Refraction is NOT part of
this study: it needs a verified masked-readout protocol and would otherwise
receive an unfair evaluation.

## Readout protocol and possible confounds

The stored reference is the winner snapshot at the end of the item's initial
write. Recall projects its original stimulus under a nested
`brain.read_only(seed=...)` and `brain.frozen()` context, clearing
memory activity in the isolated scope. This is not ordinary training and
is not an unconstrained plastic reactivation. This observation policy
suppresses recruitment; it can therefore shift measured recall. Report the
mode explicitly. The decoder compares stable neuron IDs, never compact IDs.

Metrics: mean fraction of original k winners recovered, exact k-of-k
recovery, earliest-quartile retention, and saved winner-ID digest.

For the first run:

```sh
uv sync --no-group gpu
uv run python -m pytest -q research/multimodal/test_native_retention.py
uv run python research/multimodal/native_retention.py \
  --n 512 --k 20 --p .1 --beta .1 --write-rounds 8 \
  --read-rounds 8 --loads 4 8 16 \
  --seeds 0 1 2 --output /tmp/native_r1b_diagnostic.json
```

Do not conflate this study with the repository's registered
`REFRACTION-ANTI-MERGING` claim. That study uses different hardware,
connectivity parameters, readout, and memory formation constraints.

## Decision rules

- Report raw seed-level results and variation, including failures.
- Treat differences with overlapping seed ranges as exploratory.
- Do not claim native-versus-Rust parity based on passing these tests.
- If zero-plasticity equals or exceeds recurrent Hebbian, inspect whether
  read-only suppresses access to the trained attractor, before drawing a
  mechanism conclusion.
- If recurrent beats feedforward, follow up with equalized total
  potentiation and fixed-connectome controls before causal attribution.
- Independently register the masked-refraction arm (R1c), followed by
  source-target arbitrary-binding and compositional-transfer tests.

## Current evidence

The only verified numbers are the separate R1a simplified-fiber exploratory
results. **No R1b native recovery numbers are claimed here.** Mock-level
protocol tests are not equivalent to executing native Brain, nor to
proving plasticity / observation state is preserved on exceptions.
