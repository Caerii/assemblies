# Historical merge harness migration

Software migration only. Preserve the six source4650438 trial traces and weights.
Composition prepares C through separate A-only/B-only phases before joint training;
recovery uses joint training only, then learns during sequential A/B readout.

Full defaults: n1000,k100,p.05,beta.1,w_max20,establishment30/merge30/readout20;
duration grid1/5/10/20/30/50; size grid200/500/1000/2000 with k=floor(sqrt(n));
seeds42..51. Engine numpy_sparse; actual area owner numpy_explicit.
Smoke: n60,k6,p.2,beta.1,w_max20,schedules3/3/3; duration grid1/3 and size grid60
(k7); seeds1/2/3. Five cells retain every observation and both parent overlaps.

Report parent-overlaps-v1 renames old arithmetic to mean_parent_overlap and
max_parent_overlap. Neither is a success bar. Keep chance tests only for the mean
and recovery observables, with undefined t/p/d explicit. Record every resolved
parameter and raw per-seed vector. Smoke is VOID; full outputs UNADOPTED.

Before execution, commit registration/code, run trial replay, configuration and
misuse controls, then execute a tagged smoke. Compare its metrics/raw_data/parameters/
success exactly with direct execution of recorded inputs, excluding timestamps and
duration. Validate the archive and link the result here. This does not reproduce
source-less old evidence, infer biological composition, or adopt old hypotheses.


## Migration result (2026-09-10)

Implementation and registration committed at b8984cd before the
[smoke run](../../results/runs/memory.historical-merge/historical-merge-smoke-20260910/results.json).
Five cells retain seeds1/2/3 and all overlap vectors. Metrics/raw_data/parameters/
success match direct execution exactly using recorded inputs; archive validation
passes. This is VOID software migration evidence, not adoption of a composition
claim. The six prior trial fixtures also remain exact at their original schedules.
