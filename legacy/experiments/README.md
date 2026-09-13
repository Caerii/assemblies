# Archived experiments

Scripts that predate the shared runner and have a maintained replacement,
kept for history with `git mv` so their log survives. Nothing here is a
supported entry point; `research/experiments/README.md` lists what is.

## archived_2026_09/

Moved on 2026-09-12 per
`docs/reviews/whole-codebase/STATIC_DEBT_DISPOSITIONS.md`:

- `gpu_hashed_substrate_parity.py`, `gpu_hashed_stim_parity.py`: the first
  drive-replay parity scripts for the hashed substrate. Both import
  `_chain_table` from `torch_engine/_batched.py`, which moved to
  `_hashed.py` in 637593ee (2026-08-25), so they have raised ImportError
  since; their replacement with the same arms, parameter cases and
  injected stimulus base runs as
  `neural_assemblies/tests/test_hashed_substrate_parity.py`. Nothing cites
  them.
- `primitives_run_all.py` (was `research/experiments/primitives/run_all.py`):
  the primitives suite runner. Its grid arguments no longer match the
  producers, so both its quick and full paths failed at the producer
  boundary; the individual `primitives/test_*.py` scripts remain and are
  the entry points the registry names.

## Earlier material

The hyperdimensional-computing investigation an earlier version of this
README described (`assembly_hdc_investigation.py`) is not tracked; the
maintained module is `neural_assemblies/compute/hyperdimensional.py` with
`neural_assemblies/tests/test_hyperdimensional_contract.py`.

## Current practice

New runnable experiments live under `research/experiments/` and run through
`python -m research.runner <command>`.
