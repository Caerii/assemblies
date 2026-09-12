# Making the assumptions unrepresentable

Every defect this repository keeps rediscovering has the same shape: **an
assumption that is true somewhere, false elsewhere, and checked nowhere.** This
is the plan to make each one impossible to violate silently, using one mechanism
per class rather than a seventh ad-hoc fix.

## The constraint that decides the design

    pyright and ruff are available in the maintained development environment
    semantic index brands are runtime-checked at public boundaries
    ratchets and true-negative tests guard migration paths

**Static annotations are paired with runtime checks here.** Branded ndarray
subclasses preserve NumPy behavior while public boundaries reject mixed spaces;
Pyright and true-negative tests verify that the static contract remains useful.

This matches the repo's evidence. Every real defect on record was caught by
MEASUREMENT, not by inspection, and the only guard family that has demonstrably
worked is the ratchet -- a frozen baseline that fails when a pattern grows. One
caught me earlier today adding hand-rolled seed statistics.

## The four mechanisms

### Tier A -- rename, so the wrong read is an AttributeError

**For: two quantities sharing one name.** The dominant defect generator
([[same-name-two-meanings]]). Known instances:

| name | meaning 1 | meaning 2 | damage on record |
|---|---|---|---|
| `w` | neurons materialized (sparse engine) | `len(winners)` (explicit engine) | 247 vs 50 for one area; a ratio read 1.87 / 1.65 / 2.84 through three candidate extents |
| `winners` | COMPACT engine indices | NEURON IDs (`Assembly`) | merge recall read exactly chance; the line was silently void |
| `p600` | unbounded cumulative churn (0.12 vs 5.24) | bounded energy deficit (0.989 vs 0.995) | detector 11.9x from ever firing |
| `stim_names` | registered stimuli | firing stimuli | -- |
| `support` | two meanings | | -- |

Python enforces attribute lookup. If the two meanings have different NAMES, the
wrong read raises `AttributeError` at the first call, with no checker involved
and no runtime cost. This is the strongest mechanism available to us and it
targets the biggest class.

Cost: 309 `.winners` readers and 172 `.w` readers outside tests. Too many to
migrate blind, which is what Tier C is for.

### Tier B -- runtime value types at module boundaries

**For: values that flow as ARRAYS, where a name alone cannot carry the
distinction.** `winners` is an `np.ndarray`; renaming the attribute does not stop
the array being passed into a function expecting the other index space.

A tagged wrapper checked **at boundaries only, never in hot loops**: engine
returns, `Assembly` construction, diagnostics entry points. The cost has to land
where a projection is not happening.

### Tier C -- ratchets, so migration is monotone

**For: everything too large to migrate in one pass.** Freeze the current count
of a pattern per file; fail when it grows. Proven mechanism, already used for
hand-rolled seed statistics and unpinned `Brain()`.

This is what makes Tier A affordable: the 481 sites do not have to move today,
but no new one may appear.

### Tier D -- bind constants to the quantity they describe

**For: a threshold that outlives its metric.** `P600_EXCESS_MARGIN = 0.152` was
correct for a quantity with a 44x separation, the quantity was replaced with one
bounded in [0,1], and nothing noticed. The calibration still reports
`source="empirical"` while using the stale constant.

A metric declares its expected range and units; constants are declared against a
metric; observed values are validated against the declared range. A redefinition
then fails loudly instead of silently producing a dead detector.

## What "unified" means here, concretely

Not "one class hierarchy". It means: **for each assumption class there is
exactly one mechanism, it is enforced at runtime or by a test, and its failure
message says what to do.** Seven bespoke guards is what we have; four mechanisms
with one owner each is the target.

## Order of work

1. **Tier D** first -- smallest, self-contained, and fixes a LIVE defect (#104).
   It is also the template: a declared invariant, validated where the value is
   produced, with a message naming the fix.
2. **Tier C ratchet** for the ambiguous-name readers -- freezes the problem so
   Tier A can proceed incrementally instead of as a 481-site big bang.
3. **Tier A** rename passes, one name at a time, ratchet ticking down.
4. **Tier B** last, and only where Tier A provably cannot reach -- it is the only
   tier with a runtime cost, so it must earn its place.

## The rule for adding a fifth mechanism

Don't. If a new assumption class appears, first check whether it is an instance
of A-D. The failure this program exists to prevent is a codebase with a
different guard for every bug.
