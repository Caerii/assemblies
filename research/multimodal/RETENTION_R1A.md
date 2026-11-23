# R1a — Connectivity and retention, exploratory diagnostic

**Status:** exploratory; not preregistered evidence, not native-Brain behavior.
**Date:** 2026-11-23.
**Code:** `research/multimodal/retention_stream.py`, fixed seed stream.
**Environment:** Python 3.13, NumPy 2.3.5, pytest 9.0.2, CPU.

## Protocol

`n=128, k=6, beta=0.5, w_max=20, presentations=3, seeds=[0,1,2]`,
loads `[8,16,32]`, initial Bernoulli connectivity `p in [0.1,0.3,0.6,1.0]`.
Each stored item is an independent random k-element source assembly and target
assembly. We test every stored source after the indicated number of writes.
Exact target recovery compares all k winners; Jaccard is secondary.
The episodic baseline performs exact-key lookup with newest-value-wins, without
memory-budget matching. A target's *input coverage* is the fraction with at
least one existing fiber from the active source (not a strict accuracy bound).

## Measured mean exact recovery, 3 seeds

| p | 8 writes | 16 writes | 32 writes |
|---:|---:|---:|---:|
| 0.1 | 0.000 | 0.000 | 0.000 |
| 0.3 | 0.000 | 0.000 | 0.021 |
| 0.6 | 0.417 | 0.229 | 0.125 |
| 1.0 | 1.000 | 0.958 | 0.802 |

At p=0.1, mean target input coverage at loads 8/16/32 was
0.444/0.444/0.436. At p=0.6, coverage was 0.986/0.993/0.990.
The p=1.0 arm is a *positive control* for high-support binding, not evidence
that fully connected assemblies are a better design.

## Independent results provenance

The local JSON files were generated with:

```sh
for p in 0.1 0.3 0.6 1.0; do
  python research/multimodal/retention_stream.py --n 128 --k 6 \
    --p "$p" --beta .5 --presentations 3 \
    --loads 8 16 32 --seeds 0 1 2 --output "p${p}.json"
done
```

Corresponding SHA-256 of local JSON outputs (files not yet checked into Git):

- p0.1: `4d2bea926917b344269c3ba2e04271c8d062753f64d215c84dfd27fa19b733e9`
- p0.3: `f0ab2db5570db4d5526c37d99d89b5db487d199dc9d9fc7f683ef622cc623bdf`
- p0.6: `210956f8df33b64c980a0d8bcc82560a83f61c5a2f0da05124a110354f41a6cc`
- p1.0: `a1f6b6408b07b96976531a24cf781bf0de8458fa94250e7700138f0898942076`

## Interpretation and next falsification

The synthetic one-fiber rule poorly recovers independent random bindings at
small p/k. Sparse Hebbian updates only potentiate *existing* synapses, so
targets with no initial cue connection cannot be strengthened by this rule.
At high p the same substrate eventually loses earlier bindings to interference.
Tests must next distinguish poor support, winner competition, and saturation,
and compare to native Brain projection (with recruitment) and retrievable
episode/indexed baselines before interpreting this as an assembly-calculus
limitation.

**Important:** This is a diagnostic of a simplified fiber and cannot be used
to claim native Brain fails, refraction fails, or online compositional learning
is impossible. Never substitute the synthetic result for a verified model claim.
