# `input_drive` divides by the alias, so its normalization silently does nothing

## The defect

`input_drive` is the P600 measurement (`anchored_p600_live` calls it). Its
docstring states, with a worked reason:

> The pre-k-WTA figure is normalized per candidate because areas do not have the
> same number of recruited neurons: two role areas measured here differed by
> **391 vs 449**, and comparing raw sums across them reverses the ranking purely
> on size. **Any cross-area comparison must divide out the candidate count.**

It implements that as:

```python
scores = {a: v / max(int(brain.areas[a].w), 1) for a, v in scores.items() ...}
```

`brain.areas[a].w` is the **winners-length alias**, not the recruited count.
`Area.winners`'s setter runs `self.w = len(self._winners)`, clobbering the
num-ever-fired value Brain syncs from the engine — the collision recorded in
[[same-name-two-meanings]] and audited under #83 as "largely clean". This call
site was missed.

Measured on a SENTENCES parser, three seeds:

| area | `Area.w` | `engine.w` |
|---|---|---|
| `NOUN_CORE` | 2546 | 2546 |
| `ROLE_PATIENT` | **0** | **675** |
| `ROLE_AGENT` | **0** | **874** |

Both role areas have empty winners at probe time, so the divisor is
`max(0, 1) = 1` for both. **The normalization does nothing**, and the areas
differ 675 vs 874 — a 1.3× gap, larger than the 391-vs-449 case the docstring
says reverses rankings. The guard is disabled exactly where it was written to
apply.

`NOUN_CORE` agrees because its winners were never cleared, which is why this
does not show up everywhere and why a spot check would miss it.

## Blast radius, stated conservatively

- **Any cross-area `input_drive` comparison is suspect** whenever a target
  area's winners are empty. That includes the multi-target form, whose docstring
  advertises commensurability as the reason to use it.
- **The shipped P600 uses a single target**, so no cross-area comparison happens
  *within* a probe. What does move is the SCALE: the divisor is `len(winners)`
  — `k` after a parse, `1` after a clear — so the magnitude of `p600` depends on
  a state that has nothing to do with what is being measured.
- This is adjacent to but **distinct from** #104. #104 says `w` is the wrong
  *quantity* (materialised count, not `n`). This says `w` is not even that — at
  this call site it is the winners alias and frequently reads 0.

## How it was found

By using it. `role_binding_writes_anything.py` compared, per word, the drive its
stored assembly delivers to `ROLE_PATIENT` versus `ROLE_AGENT`, on the strength
of that same docstring promise. The first version of that script is therefore
**invalid and was rewritten to a single target**, where a constant divisor
cancels in a rank statistic.

Worth stating plainly: the docstring was not wrong about the *need*, and the
code was not obviously wrong to read. The defect is that the identifier means
one thing at the definition and another at the read.

## Fix, not yet applied

Read the recruited count (`engine._areas[a].w`, or whatever the sanctioned
accessor is after #83) rather than `Area.w`. It changes every `input_drive`
magnitude, therefore every P600 magnitude, so it wants its own pre-registered
measurement — and per `what_the_p600_can_and_cannot_support.md` the magnitudes
are not quotable today anyway, so nothing published depends on the current
scale. Rank statistics are unaffected wherever the divisor is constant across
the compared items.
