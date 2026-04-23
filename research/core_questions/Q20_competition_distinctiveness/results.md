# Results

## Shared-Area Distinctiveness

From `RESULTS_assembly_distinctiveness.md`:

- With `n=1000, k=100`, pairwise overlap stays near chance:
  `0.104 - 0.121` vs chance `0.100`
- Recovery after reactivation is `1.000 +/- 0.000` across 2, 3, 5, and 8
  trained stimuli
- Overlap tracks `k/n` closely across assembly sizes
- At `k = sqrt(n)`, excess overlap shrinks as `n` increases

This is strong evidence that same-brain shared-area training can still produce
distinct, recoverable assemblies in the tested stim-only regime.

## Mechanism Follow-Up

From `competition_mechanisms_20251128_232917_quick.json`:

- neuron reuse rate is low (about `5 - 7%`)
- simple hypotheses such as "interleaving is always better" or "larger beta
  always helps" were not supported in that quick experiment

So the empirical distinctiveness result is stronger than the current
mechanistic decomposition.
