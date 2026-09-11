# Experiments

Shared-runner experiments write immutable tagged records under
`../results/runs/<protocol>/<tag>/`. Other active scripts use
`../results/<line>/` through `_results.results_path`; some older scripts still
write beside themselves. Logs of older runs are under `../results/logs/`.

GPU scripts need the CUDA build environment for the fused kernels (an
MSVC developer shell on Windows, `CUDA_HOME` set); the first run compiles
them.

## Active scripts by line

| Line | Script | What it measures | Typical run |
|------|--------|------------------|-------------|
| memory | `seq_capacity_scaling.py` | capacity M* of a recurrent area, Hebbian or refracted, with gating and readout options | `python -m research.runner capacity-scaling --tag UNIQUE --registration research/notes/memory/PREREG_refraction_memory.md --nk 4000:60 --arms B`; see registration for the full load grid |
| memory | `historical_merge.py` | sequential parent overlaps and learning-on driven recovery; smoke VOID | `python -m research.runner historical-merge --smoke --seeds 1 2 3 --tag UNIQUE` |
| memory | `historical_association.py` | learning-on driven regeneration and identity; explicit paired differences | `python -m research.runner historical-association --smoke --seeds 1 2 3 --tag UNIQUE` |
| memory | `historical_phase.py` | learning-on persistence grid and descriptive sampled crossings | `python -m research.runner historical-phase --smoke --seeds 1 2 3 --tag UNIQUE` |
| memory | `historical_scaling.py` | learning-on persistence and censored convergence times; no asymptotic classification | `python -m research.runner historical-scaling --smoke --seeds 1 2 3 --tag UNIQUE` |
| memory | `historical_projection.py` | corrected historical learning-on projection study; smoke VOID, full UNADOPTED | `python -m research.runner historical-projection --smoke --seeds 1 2 3 --tag UNIQUE` |
| memory | `historical_noise.py` | preserved learning-on-recovery protocol; smoke is VOID, full output UNADOPTED | `python -m research.runner historical-noise --smoke --seeds 1 2 3 --tag UNIQUE` |
| memory | `refraction_memory_numpy.py` | the same protocol on the numpy engine, 5 brains | ~10 min |
| sequence | `seq_a1_horizon_hashed.py` | the mod-3 machine's horizon at width, paired to the numpy seeds | `python -m research.runner a1-horizon --tag UNIQUE` (20 seeds by default) |
| sequence | `seq_a1_learning_null.py` | preregistered paired sensitivity control, beta/strength disabled | `python -m research.runner a1-learning-null --tag UNIQUE` |
| sequence | `seq_s5_soft_census_hashed.py` | soft transitions in the word-problem organs at width | `--seeds 100 --groups S5 --presentations 20`, ~6 min |
| sequence | `seq_s5_arc_drift.py`, `seq_s5_arc_clip.py` | post hoc diagnostics: arc relocation across presentations, and its cause | ~1 min each |
| sequence | `seq_a3_transducer.py --engine hashed` | the induced-state transducer at width (`--strength` for Amendment 2) | ~45 min |
| sequence | `seq_a3_oracle_ceiling.py` | the corpus's oracle-state ceiling, computed | seconds |
| aligner | `word_capacity.py` | word capacity of the cross-situational learner | see its docstring |

The numpy-era scripts these supersede (`seq_a1_horizon.py`,
`seq_s5_soft_census.py`, `seq_s5_word_problem.py`, ...) remain runnable and
their results sit in the same results folders.

## Shared modules

- `_substrate.py`: the measurement standards (ceiling from a curve, distinctness checks, seed handling).
- `_results.py`: `results_path(line, name)`.
- `_parallel.py`: worker pools for the numpy scripts.
- `study4/`: the next-token corpus generator (`ntp.py`) and its context study.

Subfolders (`capacity/`, `stability/`, `vocab/`, ...) hold the older
studies; each keeps its own results.


## Legacy aggregate configuration errors

`run_all_experiments.py --quick` currently refuses its historical configuration
inventory before constructing any experiment: six entries request parameters that
the producers do not implement. Those arguments previously selected default grids
silently. Use an individually registered shared-runner protocol above; changing the
old argument names is not sufficient to reproduce the intended grid. The seven
remaining legacy producers now reject unknown keywords directly as well.

`primitives/run_all.py` also contains obsolete grid arguments; both its quick and
full paths now fail at the producer boundary. Its full protocol migration remains
open. Individual explicit producer arguments remain supported, but execution alone
does not establish provenance or scientific adoption.


The historical projection study's H4 weight-ratio probe was corrected after
177dbbc: earlier constant-1 outputs came from a missing-attribute fallback and are
not weight measurements. Its H3 readout is A-driven regeneration while learning,
not autonomous completion of a corrupted B cue. See the source-linked
[measurement card](../../docs/reviews/whole-codebase/SEMANTIC_CARDS.md#historical-projection-measurement)
before interpreting this study. The corrected version now has the shared runner
entry above; earlier source-less artifacts are not thereby reproduced.


Historical projection protocol v3 distinguishes elapsed training from observed
convergence. A timeout has `converged=false` and `convergence_time=null`; any timed-out
H1 brain makes the ordinary scaling fit unavailable. `training_rounds` includes
capped work and is summarized separately. Version-2 scalar records cannot resolve
that stopping distinction without new trajectory evidence.


The legacy scaling study now shares the convergence stopping phase with projection,
while retaining its extra initial stimulus-only activation. It records timeouts and
raw seeds and refuses a convergence fit when any observation is censored. Its former
coefficient-based complexity labels have been removed: the fitted slope does not
establish an asymptotic class. Its assembly size is floor(sqrt(n)), so k/n varies.
New runs use the configurable shared-runner entry above; old artifacts retain
their original provenance limitations.


The legacy phase-diagram study now reports marginal interval classifications and
descriptive sampled threshold crossings, including absent crossings. Its old
mean-based "stable" labels did not establish a phase boundary. Evaluation continues
learning; all cells now retain raw seed observations. Its H3 k, grids and schedules are now explicit; the default H3 k100 requires
n>=100, while compatible smaller configurations are available through the runner.
