# Role binding works. `input_drive` cannot see it, by construction.

## The measurement

Three seeds, SENTENCES parser.

| area | stored | spread | vs chance (k/n = 0.0100) |
|---|---|---|---|
| `NOUN_CORE` | 71 | 0.1384 / 0.1337 / 0.1375 | ~13.5× |
| `ROLE_PATIENT` | 36 | 0.1582 / 0.1692 / 0.1690 | ~16.5× |
| `ROLE_AGENT` | 46 | 0.1286 / 0.1643 / 0.1261 | ~14× |

**Retrieval** — activate a word's stored core assembly, project `core → role`,
ask which stored role assembly the result best matches:

| area | rank-1 accuracy | chance | above chance | margin |
|---|---|---|---|---|
| `ROLE_PATIENT` | 0.333 / 0.417 / 0.417 | 0.028 | **12–15×** | 1.13–1.29 |
| `ROLE_AGENT` | 0.326 / 0.348 / 0.391 | 0.022 | **15–18×** | 1.24–1.71 |

## Two things this settles

**The role assemblies did not merge.** Spread 0.13–0.17, statistically
indistinguishable from `NOUN_CORE`'s 0.13 on the same parser — and `NOUN_CORE`
is the built-in control proving the protocol can tell distinct from collapsed.
#31's "94 VPs, one assembly" does not reproduce in the role areas.

**Binding writes a word-specific, findable target.** A third to 40% rank-1
retrieval against 1/36 chance is 12–19× above chance. `bind()` works.

## So why did the drive probe find nothing?

Because **`input_drive` measures how much, and role binding encodes where.**

`input_drive` sums pre-kWTA energy over *every* candidate neuron in the target
area. If `cake` is bound to one 30-neuron assembly in `ROLE_PATIENT` and `bread`
to a different 30, both deliver comparable **total** energy to the area — they
concentrate it on different neurons. The sum is blind to which.

That is not a bug in either piece. It is an orthogonality:

| | what it encodes | what `input_drive` reads |
|---|---|---|
| role binding | *which* assembly, within the area | — |
| `input_drive` | — | *how much*, over the whole area |

**The P600 metric is structurally incapable of reading role binding**, however
well the binding works. And it explains the whole chain without any further
defect:

- pathway null, AUC 0.515 — drive cannot see per-word targets;
- the P600 reads area identity, AUC 0.980 — source area is the only thing left
  that changes total energy;
- the minimal substrate showed 1.23× because with bound vs *never-projected*
  assemblies the totals do differ; in the parser both groups had been bound
  somewhere, so the totals equalise and only the target differs.

Every earlier result stands. They were measuring a quantity that does not carry
the signal.

## What this costs the ERP line

`anchored_p600_live` is built on `input_drive`. Its premise — that a violation
"routes a wrongly-typed core through an untrained pathway and delivers LESS" —
survives only in the weak form already measured: it detects the **source area**,
not the pathway. Adding headroom (#104) would not change that; a saturated
measure of the wrong quantity and a well-scaled measure of the wrong quantity
say the same thing.

**If the ERP line wants to index role integration, it needs a WHERE readout, not
a HOW MUCH one.** `bind_strength` / `recall` already exist and are exactly that
— "use `input_drive` rather than `bind_strength` whenever the question is *which
area* rather than *which assembly*", says `input_drive`'s own docstring. The
question here was always which assembly.

## The honest limits of the positive result

Retrieval is **0.33–0.42**, so 58–67% of words are *not* retrieved correctly, and
the rank1/rank2 margin is **1.13–1.29** — thin. Binding is real and weak, which
is what #52's crowding regime (overlap 0.15–0.22) predicts. This is a
demonstration that the mechanism works, not that it works well enough to carry a
parser.

The retrieval probe drives `core → role` directly, which is not what a live
parse does — the parse arrives through the incremental circuit with other areas
active. So 0.33–0.42 is an upper bound on what the parser gets in situ.

## Status

#120 resolved: the inverted drive reading was neither a probe bug nor a dormant
mechanism. It was the wrong instrument. Nothing in the chain needs a fix; the
readout needs replacing.
