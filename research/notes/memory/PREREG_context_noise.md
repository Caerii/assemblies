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
