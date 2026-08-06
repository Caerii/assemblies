# The role margin's second quantity, and the corpus that hid it

`_role_binding_margin` returned two different things under one name:

```python
if not others:
    return own                                        # RAW OVERLAP
base = sum(others) / len(others)
return max(0.0, (own - base) / max(1e-6, 1.0 - base))  # RESIDUAL
```

The function's own docstring explains why the raw overlap is useless — "~0.9 for
EVERY candidate role", "nearly independent of what was learned" — which is the
entire reason the residual exists. So the single-filler branch returned exactly
the quantity the function was written to reject, and returned it *large*.

That would matter because `_assign_roles_neural` normalizes these margins
against each other into a distribution over candidate roles
(`total = sum(margins.values()) + eps * len(margins)`). A raw ~0.9 competing
against properly-baselined residuals takes probability mass it did not earn, so
**lexicon size — not binding strength — would move the role decision.**

## Measure before fixing

`research/experiments/role_margin_branch_census.py` counts which branch fires in
a real parse. Across seeds 11 / 12 / 42:

| branch | calls | share |
|---|---|---|
| residual (intended) | 20 | 83.3% |
| no stored binding → 0.0 | 4 | 16.7% |
| **single filler → RAW OVERLAP** | **0** | **0%** |

**The confounded branch never fires.** The fix is free, and the post-fix margins
are identical to four decimal places on every seed (0.8980 / 0.9048 / 0.8523,
before and after).

## The first run of that census was wrong, and it looked like a finding

Version one used hand-written probes — `the dog chases the cat`. It reported
**77.8% of calls taking the no-binding branch, and every single AGENT call
taking it**, which reads as "the lexical route is dead for agents."

It was not. Those words are not in the trained vocabulary at all:

```
ROLE_AGENT    46 entries: ['bag', 'beach', 'bottle', 'bread', 'cake', ...]
ROLE_PATIENT  36 entries: ['baby', 'bed', 'book', 'bread', 'cake', ...]
present=[]   MISSING=['the','dog','chases','cat','sees','boy',...]
```

The census was measuring **my probe corpus**, not the parser. Deriving the
probes from the role lexicons themselves fixes it and makes the artefact
unrepeatable. Worth stating plainly: a census is a measurement, and a
measurement with the wrong input produces a confident wrong number exactly like
any other.

## A real finding the census did surface

Four of seven role lexicons are **empty**:

```
ROLE_AGENT=46  ROLE_ACTION=48  ROLE_PATIENT=36
ROLE_THEME=0  ROLE_GOAL=0  ROLE_SOURCE=0  ROLE_LOCATION=0
```

Any margin computed for THEME, GOAL, SOURCE or LOCATION is structurally
guaranteed to take the no-binding branch — not because the word was never bound
there, but because *nothing was ever stored there at all*. Those roles were not
candidates in these sentences, so it costs nothing here; it is the same shape as
[[silent-no-op-dead-fibers]] and is worth checking wherever those roles do enter
`role_order` (see #107, nouns after a preposition).

## The fix

Both non-residual cases return `Measured.undefined` with the reason, and the two
call sites (`roles.py`, `nemo_parse.py`) use an explicit `.or_else(0.0)` — an
undefined margin contributes nothing and leaves the structural prior in charge.
The raw overlap is still reported in `detail["own"]`: not lost, just no longer
passed off as a residual.

`nemo_parse.py` had been calling `float(margin_fn(...))` inside a bare
`except Exception`, which would have swallowed the `UndefinedMeasurement`
alongside every real error and produced the same 0.0 either way. Stated
explicitly now.

Guards: `neural_assemblies/tests/test_role_binding_margin.py`, 5 tests — the
lone-filler case is undefined AND does not return the raw overlap; the residual
is still computed when a baseline exists; a DEFINED zero stays distinct from an
undefined one.
