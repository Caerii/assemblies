# Registration: how often the k-WTA bar is tied, and what a last bit does to it

> **Status (2026-09-12): registered, not yet run.** Bars fixed before the
> twenty-seed run. The register entry `KWTA-TIE-FRAGILE` rests on a GPU
> selector prototype with no retained artifact; this registration retains
> the tie census on the explicit CPU engine with a constructed
> last-bit perturbation and a tie-free null, and says explicitly which
> part of the entry the CPU can and cannot reproduce.

## Why

`KWTA-TIE-FRAGILE` says the k-WTA bar is routinely tied, so anything that
perturbs the drive in its last bits can change WHICH neurons fire. The
evidence is `gpu_radix_select_prototype.py` and, since, the fused-kernel
gate that found float32 reciprocal multiplication and float64 division
selecting different winners on 19 of 40 members (VALIDATION.md, "GPU gate:
explicit storage and normalization arithmetic"). Neither is a retained
per-seed artifact. The mechanism has two halves: ties at the bar are
common (a substrate-independent counting fact), and a perturbation of one
unit in the last place decides among tied candidates (a property of the
tie rule and the arithmetic). The CPU can retain the first half and
demonstrate the second by construction; whether the CPU's own float32
reciprocal path splits rational ties is measured and reported without a
bar, because a three-seed pilot on seeds 0 to 4 (not registered seeds)
saw no split there.

## What runs

`python -m research.runner kwta-tie-fragility --tag UNIQUE`

- Engine `numpy_explicit` (dense float32 matrices, lowest-neuron-id ties),
  `observation_policy` none: the drives are computed from the connectome
  weights exactly as the engine's first projection computes them (float32
  sum over the stimulus rows), so no readout projection runs and nothing
  learns.
- One area of n = 1000, k = 50, p = 0.1; one stimulus of 50 rows. Seeds 42
  to 61, each drawing its own connectome.
- Per seed the run retains:
  - `ties_raw`: candidates whose float32 stimulus drive equals the k-th
    largest (the bar), on the raw 0/1 weights.
  - `ties_normalized`: the same census on the in-degree-normalized drive
    (count divided by the column's full in-degree), counted in exact
    rational arithmetic.
  - `ties_jittered`: the census after a deterministic per-neuron jitter of
    `1e-3 * index` is added, which makes every drive distinct: the null.
  - `flip_one_ulp_raw`: 1 if raising the drive of the highest-index
    candidate tied at the bar but outside the winner set by one float32
    unit in the last place changes the winner set, else 0.
  - `flip_one_ulp_jittered`: the same construction on the jittered drive
    applied to the candidate at rank k + 1 (there is no tied candidate
    to choose), expected 0.
  - `split_f32_div_vs_recip`, `split_f32_vs_f64`: the number of winners
    that differ between the normalized drive computed by float32 division,
    float32 reciprocal multiplication, and float64 division, under the
    engine's lowest-id tie rule.

Smoke (`--smoke --seeds 1 2 3`) runs the same instrument on three seeds and
is VOID.

## Bars

Priors: pilot on seeds 0 to 4 (raw ties 23 to 77 with a synthetic 0/1
matrix, 3 to 34 on the engine after one plastic round; normalized ties 1
to 7; no split under any arithmetic path).

- **KT-1, the bar is routinely tied.** `ties_raw >= 5` on every one of the
  twenty seeds.
- **KT-2, the last bit decides.** `flip_one_ulp_raw == 1` on every seed
  (a one-ulp raise of a tied outsider puts it in the winner set), and
  `flip_one_ulp_jittered == 0` on every seed.
- **KT-3, normalization does not remove the ties.** `ties_normalized >= 2`
  on at least 10 of 20 seeds.
- **Reported, not judged:** the two split counts. If they are zero on every
  seed the register's caveat records that the CPU reciprocal path does not
  reproduce the GPU selector's split at these sizes, and the entry's engine
  field keeps naming the GPU prototype for that half of the claim.

Retained sensitivity for the register: `ties_raw` against `ties_jittered`,
every seed greater by at least 4 (the jittered census is exactly 1).

Not claimed: anything about how often ties are decided differently in a
trained organ, or the rate at which a real run's winners change under a
change of summation order. This retains the counting fact and the
one-bit demonstration, no more.

## Result (2026-09-12, twenty seeds: KT-1 PASS, KT-2 FAIL as registered, KT-3 PASS)

Artifacts: study
`research/results/runs/substrate.kwta-tie-fragility/kwta-study-20260912/results.json`
(seeds 42 to 61); smoke, VOID,
`research/results/runs/substrate.kwta-tie-fragility/kwta-smoke-20260912/results.json`.

| quantity | per seed |
|---|---|
| ties at the bar, raw 0/1 drive | 26 to 73 (median 35) |
| ties at the bar, in-degree normalized (exact rational) | 1 to 12; at least 2 on 17 of 20 |
| ties at the bar, jittered drive | 1 on every seed |
| one-ulp raise of a tied outsider flips the winner set | 19 of 20 seeds |
| one-ulp raise on the jittered drive flips | 0 of 20 |
| winners differing, float32 division vs float32 reciprocal | 0 on every seed |
| winners differing, float32 vs float64 division | 0 on every seed |

- **KT-1 PASS.** The bar is tied on every seed, by 26 to 73 candidates.
- **KT-2 FAIL as registered.** The one-ulp construction changed the winner
  set on 19 of 20 seeds and never on the jittered drive, but the bar said
  "every seed". On seed 49 there are 19 candidates strictly above the bar
  and 31 tied at it, so all 50 are inside k and no outsider is tied; the
  construction has nothing to decide. The mechanism stands on the 19 seeds
  where a tied outsider exists; the bar as written did not anticipate the
  exact-fit case and is not amended after the fact.
- **KT-3 PASS.** Normalization keeps at least two candidates tied on 17 of
  20 seeds.
- **Reported, not judged.** The three arithmetic paths select the same
  winners on every seed at these sizes. The split the register entry cites
  (float32 reciprocal against float64 division, 19 of 40 members) was seen
  in the fused-kernel gate on the GPU organ fiber, not here; the entry's
  engine field now says which half of the claim each substrate carries.

Retained sensitivity for `KWTA-TIE-FRAGILE`: `ties_raw` against
`ties_jittered`, every seed greater by at least 4 (smallest effect 25).
