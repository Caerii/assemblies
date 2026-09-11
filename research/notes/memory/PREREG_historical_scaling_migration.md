# Historical scaling runner migration

This registers software migration acceptance, not scientific adoption. Preserve
the corrected c1ba555 learning-on scaling protocol, including its separate initial
stimulus-only activation, explicit stopping status, censored fit handling, and
absence of coefficient-based asymptotic classifications.

## Resolved configurations

Full: populations100,200,500,1000,2000,5000, with k=floor(sqrt(n)); p.05,beta.1,
weight clip20; initial stimulus-only rounds1, maximum recurrent training100,
evaluation20; strict overlap>.98 over three consecutive comparisons; seeds42..51.
Primary engine numpy_sparse, actual fully explicit area owner numpy_explicit.
Initialization is excluded from the training convergence observation count.
Evaluation remains self-only with learning enabled, not frozen persistence.

Smoke: populations60,80; p.2,beta.1,clip20,initialization1,max training8,evaluation3,
window3,threshold.98, explicit seeds1,2,3. Both cells retain every stopping status,
nullable event time, elapsed round count and persistence observation. Timed-out
brains prevent the ordinary convergence-time regression; they cannot be dropped.

## Acceptance before execution

The six source-0b9909c trial fixtures must retain initialization, projection calls,
winners and final weights. Their historical convergence_time is elapsed work only.
Configuration spies must verify grid order, seed order with no second offset, and
all supplied schedules. Invalid grids, stopping rules and fewer than three seeds
must fail before trials. The old CLI must require --tag; --quick is VOID smoke.

Commit this registration before a tagged smoke. Compare saved metrics/raw_data/
parameters/success exactly with direct execution of the saved configuration,
excluding only timestamps/duration. Validate the record and source archive.
Smoke remains VOID and full output UNADOPTED. No coefficient establishes a
complexity class, no older provenance gaps are filled by inference, and the obsolete
aggregate grid is not silently translated into this different protocol.
