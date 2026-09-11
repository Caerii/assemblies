# Historical phase-grid runner migration

Software migration registration, not scientific adoption. Preserve the corrected
4d06478 trial and reporting semantics. Every phase learns: initial stimulus-only,
then stimulus+self training, then self-only evaluation. This measures finite
persistence, not frozen fixed points or a physical phase transition.

## Resolved configurations

Full: n1000,p.05,w_max20; requested sparsities .01,.02,.05,.10,.15,.20,.30;
betas .01,.02,.05,.10,.20; connection grid .01,.02,.05,.10,.20 with H3 k100,beta.1.
One initialization round, 30 training rounds, 20 evaluation rounds. Threshold.95.
Seed identities42..51. Primary engine numpy_sparse, actual owner numpy_explicit.

Smoke: n60,p.2,w_max20,sparsities.1/.2,betas0/.1,connection grid.1/.2,H3 k6,beta.1;
initialization1,training3,evaluation3,threshold.95,seeds1/2/3. Six cells with every
seed retained. Duplicate k after floor(sparsity*n), invalid grids/schedules and
fewer than three distinct seed identities must fail before trial computation.

## Acceptance before execution

Retain all six source-f0a0de8 trial projection traces, winner states, final weight
hashes and persistence values. Spies must verify every explicit grid/schedule and
seed order with no extra base-seed offset. The old CLI requires --tag; --quick is
VOID smoke. Full outputs are UNADOPTED. Commit this registration before the smoke.

Compare saved metrics/raw_data/parameters/success exactly with direct execution
using its recorded inputs, excluding only timestamps/duration. Validate its record
and source archive. Missing crossings remain explicit; a sample mean crossing the
threshold is not enough for above_threshold interval status. Crossings are selected
descriptive grid observations, not simultaneous confidence or monotonicity claims.
No old provenance gaps or obsolete aggregate grids are repaired by inference.
