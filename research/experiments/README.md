# Experiments

Scripts only. Results of the active lines are written to
`../results/<line>/` through `_results.results_path`; older scripts still
write beside themselves. Logs of runs are under `../results/logs/`.

GPU scripts need the CUDA build environment for the fused kernels (an
MSVC developer shell on Windows, `CUDA_HOME` set); the first run compiles
them.

## Active scripts by line

| Line | Script | What it measures | Typical run |
|------|--------|------------------|-------------|
| memory | `seq_capacity_scaling.py` | capacity M* of a recurrent area, Hebbian or refracted, with gating and readout options | `--nk 4000:60 --arms B --brains 20 --refracted --refracted-factor 0.5 --readout masked --ms 8,...,4096`, ~35 s |
| memory | `refraction_memory_numpy.py` | the same protocol on the numpy engine, 5 brains | ~10 min |
| sequence | `seq_a1_horizon_hashed.py` | the mod-3 machine's horizon at width, paired to the numpy seeds | `--brains 20`, 9 s |
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
