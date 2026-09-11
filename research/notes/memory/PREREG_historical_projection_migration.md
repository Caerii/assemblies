# Historical projection runner migration, protocol version 2

Software migration registration, not a scientific adoption hypothesis. Version 2
includes the corrected selected-pairs/all-pairs recurrent weight ratio from 37772eb;
old constant-1 H4 outputs are not reproducible weight evidence. Evaluation learns.
H3 is A-driven regeneration of B, not autonomous completion. H1's convergence_time
still conflates a timeout with convergence in the final round; no convergence-rate
claim is licensed by that scalar alone.

## Resolved protocol

Full: n1000,k100,p.05,beta.1,w_max20; training30,evaluation20,max-training100.
H1 sizes100,200,500,1000,2000,5000; H3 sizes500,1000,2000; k=floor(sqrt(n)) for
those two grids. H4 training counts1,5,10,20,30,50. Default seeds42 through51.
H1 stops after three consecutive overlaps >.98. Primary engine numpy_sparse,
actual explicit area owner numpy_explicit. These semantics are not hashed parity.

Smoke: n60,k6,p.2,beta.1,w_max20; training3,evaluation3,max-training8;
H1 sizes60,80; H3 size60; H4 training counts1,3. Seeds1,2,3 for the migration run.
Every grid, schedule and ordered seed identity must be captured before compute.
Raw values must be retained for all six cells (H2 has paired arms).

## Acceptance before running

Retain the 15 pre-change trial trajectory/weight fixtures, with H4's defective old
ratio explicitly excluded and independently recomputed from actual weights. A
shared-runner smoke must match direct execution's metrics, raw_data, parameters
and execution status exactly, excluding timestamps/duration. Validate its record
and source archive. Require at least three distinct nonnegative seeds, an exclusive
tag, and VOID status for smoke and legacy --quick. Full outputs remain UNADOPTED.
Undefined null tests retain their reason and null statistics; a constant scaling
response has undefined R-squared/p-value, not a fabricated fit significance.

Commit this registration before the tagged smoke. This migration does not
reconstruct missing historical source/seed provenance, justify old statistical
nulls, or repair historical scientific artifacts. It does not migrate the obsolete
aggregate parameter grid. Existing paired-test behavior remains descriptive and
must not be treated as independent evidence of adoption.


## Migration result (2026-09-10)

Registration and adapter were committed at c5ef5e7 before execution. The
[archived smoke](../../results/runs/memory.historical-projection/historical-projection-smoke-20260910/results.json)
contains six cells with seed identities1/2/3. Direct execution using the saved
parameters exactly matched metrics, raw_data, parameters and success under the
canonical JSON comparison; only timestamps/duration were excluded. The run record
and source archive validate. Verdict remains VOID, with no full-grid scientific
adoption or claim of reconstructing older artifacts.


## Version 3 amendment: stopping status (before execution)

Version 2 above remains the record for its archived smoke. Version 3 preserves
its projection/evaluation schedules but adds explicit convergence_window=3 and
convergence_threshold=.98 to the resolved parameters. The window counts comparisons,
not snapshots; the strict comparison is unchanged. Training iterations are recorded
separately from convergence_time, which is null on timeout. Each brain retains a
converged boolean. Reaching the criterion on the final round is still convergence.

The H1 summary reports capped training work and the observed convergence fraction.
A scaling fit is unavailable if any included brain times out: failed brains may
not be dropped or relabeled as convergence at the cap. This conservatively refuses
an uncensored fit; it does not estimate a censored convergence-time distribution.

Acceptance: preserve all 15 source-177dbbc trial trajectories and final weight
hashes; interpret the old H1 scalar as elapsed training, not an observed event.
Construct final-round success and failure with identical elapsed rounds and verify
opposite statuses. Verify a mixed censored cell retains all seeds and blocks fitting.
After committing this amendment, execute a version-3 smoke with the same seeds/grid,
compare it exactly to direct version-3 output, and validate source/archive/VOID.
No change to adopted scientific claims or reconstruction of version-2 stopping
statuses from its scalar-only record is permitted.
