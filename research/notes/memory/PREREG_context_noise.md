# Context readout sensitivity and finite noise tolerance

Registered before intermediate-noise measurements. Development diagnostics already
observed noise 0 and 1000 on brain seeds 1/2/3; those are not novel predictions.
This study uses new brain identities 101..120, independently for each engine.

## Protocol

Run context-attractor-v1 on numpy_sparse (fully materialized) and torch_sparse
(hashed, CUDA), separately. No between-backend equality or pooled inference is
claimed. p=.05; assigned context n=400,k=100, names left/right; output n=2000,k=200,
beta=3, rounds_train=10, fires=2; coupling_beta=1; presentations ((4,0),(0,4));
read_rounds=3. Fresh identically seeded construction for every cell. Read seeds
700..709, each used for both contexts, provide 20 observations per brain and cell.
They are repeated observations, NOT independent brain replicates. Every observation
retains label, both overlaps and margin. Ties count as incorrect, never dropped.

Trained noise standard deviations: 0,.1,.3,1,3,10,30,100,1000, in that order.
Noise is native additive Gaussian input noise in the backend's drive units, enabled
only after teaching. Two extra cells at noise zero: coupling_beta=0, and context
source disabled during observation. The first preserves recurrent learning; the
second preserves all training. They are not equivalent biological interventions.
Use one GPU job at a time; this is not a performance comparison.

## Statistics and bars

Per brain: accuracy = correct labels / 20; target_overlap = sum of intended-target
overlaps / 20; joint_success = fraction with correct label, target overlap >.8,
and absolute margin >.5. Summarize each per-brain metric with the existing keyed
ensemble_from_values 95% Student-t interval over 20 brain seeds. Retain raw values;
do not pool reads as independent samples, clip intervals, discard failures, or
interpret a zero empirical variance as certainty of population-perfect behavior.
These are nominal marginal intervals with finite-sample/normal-approximation limits.

The primary hypothesis is tolerance at the preselected std=1. Require ALL:
- trained noise 0: accuracy lower bound >.90 and target_overlap lower bound >.80;
- trained noise 1: accuracy lower bound >.90 and target_overlap lower bound >.80;
- zero-coupling noise 0: joint_success upper bound <.90;
- context-disabled noise 0: joint_success upper bound <.90;
- trained noise 1000: target_overlap upper bound <.40.

Evaluate each backend independently and retain each failed bar. A claim across both
backends requires both to pass. This conjunction is not a claim that every secondary
noise level passes. All other levels are descriptive: no selected largest passing
noise or continuous interval is certified, and no simultaneous confidence band is
claimed. A failed primary hypothesis is not repaired by selecting another level.

## Execution and evidence

`python -m research.runner context-noise --engine ENGINE --tag UNIQUE` uses the
shared exclusive writer and captures source, protocol, engine and seeds. A smoke
run uses exactly 3 supplied seeds, read seeds 700/701 and noise 0/1000 plus both
controls; it is VOID and cannot satisfy the primary hypothesis. No data is generated
until this registration, runner and analysis tests are committed. Results will be
linked here after completion without rewriting the pre-run criteria.

This concerns finite readout performance of assigned-context, trained-attractor
fixtures. It does not establish biological noise tolerance, probability calibration,
a softmax law, general assembly-calculus robustness, or a formal backend proof.


## Results (2026-09-10)

Both independent backend runs passed all seven predeclared checks. Registration and
analysis were committed at f83413a before either run; each run captured that source
and registration in its own validated archive. [NumPy raw observations and record](../../results/runs/memory.context-noise/context-noise-numpy-20260910/results.json)
and [CUDA raw observations and record](../../results/runs/memory.context-noise/context-noise-torch-20260910/results.json)
retain all 20 brain identities, 11 cells and 20 repeated reads per brain/cell.

At the preselected noise standard deviation 1:

| Backend | Accuracy, nominal 95% interval | Target overlap, nominal 95% interval | Registered verdict |
|---|---|---|---|
| NumPy materialized | 1.0000 [1.0000, 1.0000] | 0.92635 [0.92312, 0.92958] | PASS |
| Torch hashed CUDA | 1.0000 [1.0000, 1.0000] | 0.92606 [0.92305, 0.92907] | PASS |

Each zero-coupling and context-disabled control had joint_success=0 in all 20
brains on both backends. Extreme-noise target overlap was .10081 [.09985,.10178]
on NumPy and .09840 [.09721,.09959] on CUDA, below the registered .40 upper-bound
bar. Zero-noise target overlap was .99900 [.99830,.99970] and .99925
[.99858,.99992], respectively. All raw failures/incorrect labels remain in the data.

The descriptive grid reveals a distinction hidden by label accuracy: at std=3,
accuracy remained 1 in both runs, while target overlap was .46746 [.46303,.47190]
on NumPy and .46675 [.46298,.47052] on CUDA. Joint success was zero for every brain.
A correct argmax label therefore does not imply recovery of the stored assembly.
[Curve of both measurements](../../../docs/reviews/whole-codebase/context-noise.svg).

This is finite evidence for the registered noise-1 criterion in this fixture.
Zero-width accuracy intervals reflect no observed between-brain variance; they do
not prove perfect population accuracy or exact coverage. The unbounded Student-t
interval at CUDA std=10 extends above 1 and is retained unmodified. No continuous
noise range, selected maximum tolerance, formal backend equivalence, biological
interpretation or calibrated outcome probability is adopted from these results.
