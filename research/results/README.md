# Results

Evidence files, written by the scripts in `../experiments/` through
`results_path(line, name)`.

- `memory/`: capacity grids (`capacity_scaling_results*.json`, one per
  registered run, tagged) and the numpy mirror.
- `sequence/`: the horizon runs, the soft-transition censuses (tagged by
  addendum), the transducer studies.
- `aligner/`: word capacity.
- `logs/`: console logs of runs, kept as they were written.

Older result folders (`applications/`, `coin_fairness/`, ...) belong to
their own studies. Registrations in `../notes/` cite the file they read.


Shared-runner artifacts live under `runs/<protocol>/<tag>/`: `run.json`,
`results.json` and a verified source archive. The registered context-noise study
is under `runs/memory.context-noise/`; its
[registration and result interpretation](../notes/memory/PREREG_context_noise.md)
distinguish label accuracy from assembly recovery.
