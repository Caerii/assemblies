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
`results.json` and a `source.manifest.json` naming the run's source in the
content-addressed store `source-store/` (older runs may still carry a
`source.zip`; see research/README.md#recoverable-source). The registered context-noise study
is under `runs/memory.context-noise/`; its
[registration and result interpretation](../notes/memory/PREREG_context_noise.md)
distinguish label accuracy from assembly recovery.
The corrected temporal-position protocol is under
`runs/sequence.temporal-positions/`; its registration separates within-brain
sentence-pair summaries from the twenty independent brain-seed replicates.

Runner schema 6 keeps compact conclusions in `results.json` and may store large raw
JSON as digest-bound `.json.gz` siblings. Read those through
`research.evidence.load_json_attachment`, which validates the complete artifact
before decoding. A sidecar is part of its result, not an optional cache.

## Historical folders (2026-09-13)

`primitives/`, `applications/`, `stability/`, `biological_validation/`,
`information_theory/`, `dev_runs/`, `sweeps/` and the timestamped files at this
folder's root are the February 2026 language and substrate lines. No
registration or register entry cites them. They are kept as history and never
deleted; dispositions per cluster are in
`docs/reviews/whole-codebase/ORPHAN_RESULTS_DISPOSITIONS.md`. Under `sweeps/`,
`_vis_smoke*.csv`, `explore_perf*.csv` and `explore_checkpoint.csv` are smoke or
performance probes, not evidence.

## Storage (measured 2026-09-30)

What the repository pack actually spends, from `git cat-file` over every
reachable blob (pack 413 MB):

| What | In the pack | Status |
|------|-------------|--------|
| CIFAR-10 raw batches (`data/cifar-10-batches-py/`) | 162 MB | reachable only from the branch `temp-branch-delete-later`; not on dev |
| `legacy/artifacts/image_learning/` animations | 105 MB | in the current tree (114 MB on disk) |
| `runs/` (records, raw-frame attachments, old zips) | 89 MB | the science |
| `source-store/` | 10 MB | 35.5 MB raw; replaces 313 MB of per-run zips |

Git compresses text in the pack: the 59.6 MB
`temporal-positions-study-20260910/results.json` costs about 7 MB there. The
`.json.gz` attachments are already at gzip-9 density; xz would halve them
(54.9 to 29.3 MB) but records bind the gzip bytes by digest, so that is a
format change for new runs only. The 58 historical `source.zip` blobs cost
26 MB packed (git deltas them against each other), so rewriting history to
purge them would save little.

Git LFS is not available here: this repository is a public fork of
dmitropolsky/assemblies, and GitHub refuses new LFS objects on such a fork
(tested 2026-09-30: "can not upload new objects to public fork").

## Critical paths (2026-09-30)

Every run directory is referenced by something; there are no orphans. Of 77
runs, 33 back an adopted register entry (267 MB on disk), 42 are linked from a
registration only, mostly smoke runs whose former size was almost entirely
their source zip, and 2 are void or failed.

Where each adopted claim's own evidence is weakest, from the register's
`provenance_gap` and `sensitivity_gap` fields, ordered by what depends on it:

1. `REFRACTION-ANTI-MERGING`, the capacity law of the first planned paper
   (research/plans/PAPERS.md, P1). One retained paired check (the central
   contrast); the capacity grid that establishes the (n/k)^2 form predates
   immutable source records, and the masked-vs-net veto, the convergence
   gate and the strength plateau have no retained checks. A registered
   replay of the grid cells under the shared runner is the most valuable
   single run in the repository.
2. `AC-CAP`: the 1.15 n/k constant has no identifiable run; the retained
   bracket (0.48 to 1.9 n/k) does not pin it.
3. `SEQ-EXACT-RECOVERY`: the soft-census subclaims (rate, first-hitting law,
   presentation window) have no retained mechanism-disabled null.
4. `REFRACTION-NEEDS-LOAD` and `SEQ-REGIME-CLIFF`: the refraction ablation
   and a plasticity-disabled arm at fixed p predate the runner.
5. `CAP-ANCHOR-RATIO`: the p and beta excursions are logs without a run record.

The sequence line's recent entries (`SEQ-TEMPORAL-CARRY`,
`SEQ-INTEGER-ARC-LOAD`, `SEQ-STATE-COLLISION-TOLERATED`,
`REFRACTION-CANCELS-CONVERGENCE`) were measured under the runner and carry
their raw data; their open gaps are mechanistic (why the carry's amplitude
falls with chain length; why a roomy random code still costs 3.7 steps of
256), not evidential.
