# Back-catalogue audit: did the `area.w` alias void any standing result?

**Date:** 2026-07-31
**Trigger:** `Area.get_num_ever_fired()` was returning `w`, which the `winners`
setter clobbers to `len(winners)`. Any reader that touched `winners` and then
asked how many neurons had ever fired got `k` (or `0`), not recruitment.
That is the signature of several "sealed area" readings in the memory, so the
back catalogue had to be checked rather than assumed.

**Verdict: largely clean.** One documented workaround, two descriptive
statistics affected, no headline claim overturned. Details below.

---

## The mechanism, measured

`scratchpad/w_clobber_probe.py`, n=1000, k=20, 12 rounds:

| reading point | `engine.get_num_ever_fired` | `area.get_num_ever_fired()` | `area.w` |
| --- | ---: | ---: | ---: |
| after training | 129 | 129 | 129 |
| after `inhibit_areas` | 129 | 129 | **0** |
| after restoring a k-set | 129 | 129 | **20** |

Two things follow, and the second is the useful one:

1. The clobber is real and reproduces on demand.
2. **The engine always had the right number.** `AreaState` (`_state.py`) holds a
   plain `winners` attribute with no property setter, so engine-side `w` is
   never clobbered. `area.w` is a *mirror* that goes stale on assignment.

So this was never a lost quantity -- only a stale copy of one. That bounds the
blast radius: any harness reading through the engine was always correct.

## Where the bug could bite

`.w` is wrong only when a `winners` assignment happened with no intervening
projection. Both `brain.py` sync sites set `area.w = result.num_ever_fired`
immediately after `area.winners = result.winners`, so the normal path is
self-repairing. The assigners that do NOT repair are `inhibit_areas`
(`inhibition.py:221`), `_restore_outer_state` (`incremental.py:721`), and the
hand-built winner sets in the simulation harnesses.

Cross-referencing those against every `.w` reader in `research/`:

| site | status |
| --- | --- |
| `capacity/analyze.py:76` | **already found and routed around** -- `final_wn()` docstring names this exact bug and reads the per-word record instead |
| `capacity/lexicon_capacity.py:218` | safe -- reads `w_before` *before* the inhibit, with a comment saying why |
| `recruitment/recruitment_mechanisms.py:185,194` | safe -- straddles `_drive()`, no assignment between |
| `recurrent_assembly_decay.py:185,197` | safe -- reads immediately after `brain.project(...)` |
| `distinctiveness/test_competition_mechanisms.py:156,165` | safe -- explicit area, reads after `project` |
| `metrics/measurement.py`, `metrics/settling.py`, `metrics/instability.py`, `primitives/diagnose_erp_dynamics.py` | safe -- every one of these assigns `winners` and repairs `.w` on the very next line |
| `capacity/parser_recruitment.py:103` | **AFFECTED** -- feeds `w`, `w_over_n`, `tiling_ratio` |
| `prediction_paths_compare.py:114` | **AFFECTED** -- `area_sizes()` |
| `erp_p600_probe_contamination.py:73` | benign -- `total_w` is a *change* detector; a constant offset cancels |
| `p600_metric_comparison.py:404` | benign -- diagnostic buffer, not an asserted quantity |

The ERP/metrics cluster is worth calling out: it repairs `.w` by hand at seven
separate sites. Someone hit this before and fixed it locally each time instead
of at the source. That is the actual lesson here -- the same defect was
re-discovered and re-patched rather than named.

## What was affected

Both affected sites report *descriptive* statistics, not a claim under test:

* `parser_recruitment.py` -- `tiling_ratio = area_w / (len(words) * k)`. Under
  a clobber this reads `1 / len(words)`, i.e. it would understate tiling by the
  vocabulary size. The recruitment and overlap metrics in the same function are
  computed from the stored assemblies (`asms`), not from `area_w`, so the
  substantive findings there do not route through the stale value.
* `prediction_paths_compare.py` -- `area_sizes()` is printed for comparison
  between the CONTEXT-buffer and bounded-state arms. Both arms would be
  clobbered identically, so the *comparison* survives even where the absolute
  numbers do not.

No standing conclusion in `MEMORY.md` is overturned. In particular the
capacity line (`critical-load-alpha-star`, `beta-opposes-capacity-and-depth`)
runs through `analyze.py`, which already avoided the bug.

## Fix

The accessor now reads `_num_ever_fired`, a shadow that survives the setter and
is synced at the two `brain.py` result sites. Direct `.w` readers are NOT fixed
by that -- they must be migrated to `get_num_ever_fired()`. The two affected
sites above have been migrated; the rest read `.w` at points where it is
correct and were left alone, with this note as the record of why.

See [[same-name-two-meanings]] -- `area.w` is one of four names in this
codebase that mean two different things, and it is the third to produce a
measurement bug.
