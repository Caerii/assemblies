# Verification of the isolated migration

Worktree: `a`verify_maintained.py` now runs the non-slow package suite with pytest-xdist by default (`-n 4 --dist loadfile`), using a bounded worker count for the repository's fast path. `--workers N` and `ASSEMBLIES_TEST_WORKERS` bound parallelism, while `--serial` preserves the one-process diagnostic path. The static gate remains unchanged. Validation: xdist smoke (`test_area.py` plus index-space tests) 11 passed in 13.23s; maintained Pyright gate 286 files, 0 diagnostics.

## Bounded parallel gate and full-suite measurement (2026-09-12)

The default worker count is now bounded at four. An unconstrained -n auto run reached 99% without finalizing on this Windows numerical workload; four workers completed reliably. The maintained gate exposes the same override via --workers and ASSEMBLIES_TEST_WORKERS, with --serial for diagnosis.
Validation: 3,829 passed, 141 skipped, 8 xfailed, 10 subtests passed in 597.89s (9:57) using -n 4 --dist loadfile; the prior serial gate took 1,982.82s (33:03), a 69.9% wall-time reduction.
