# `norm_init` does not invert trained drive — and a beta=0 null is not a null

## The question

The parser reads back **inverted**: nouns bound into `ROLE_PATIENT` deliver
*less* drive there than nouns never bound there (AUC 0.458 / 0.348 / 0.358).
The hypothesis worth taking seriously was `norm_init`: it normalises each
postsynaptic neuron's incoming weights by in-degree, and the neurons a bound
assembly drives are precisely the ones that *received* the binding — so a
well-trained pathway could deliver less per-candidate drive than an untrained
one.

That mattered beyond role binding. The ERP readout was moved from post-kWTA
churn to pre-kWTA energy **because** the post-kWTA family reverses sign under
`norm_init`. If pre-kWTA inverted too, the fix had relocated the sign flip
rather than removed it, and the N400 is built the same way.

## The answer: refuted

Two areas, 9 stimuli, `a0` bound into `DST`, the rest never bound. 5 seeds.

| `norm_init` | gain(`a0`) | gain(unbound) | **plasticity effect** | seeds |
|---|---|---|---|---|
| True | 1.4097 | 1.1479 | **1.2280** | 5/5 |
| False | 1.6204 | 1.1963 | **1.3544** | 5/5 |

**Plasticity raises a bound pathway's drive under both settings**, on every
seed. `norm_init` changes the magnitude — the absolute drives differ by ~100×
between the two settings — but not the sign.

So the parser-level inversion is **not a substrate property**. The next suspect
is the probe or the parser's state: #120 step (a), the index-space check on
`activate_assembly`.

## The methodological finding, which is the reusable part

**A `beta=0` arm does not hold the connectome constant.** It took two attempts
to get a null that meant anything, and both failures pointed the same way:

**Attempt 1 — null read 1.678 / 2.182.** `brain.project({"a0": [SRC]}, {SRC: [DST]})`
does not only apply plasticity, it **materialises** the `SRC→DST` columns for
the neurons it drives. `a0`'s synapse rows existed and the unbound assemblies'
did not, so the "bound" advantage was lazy instantiation. Without the beta=0 arm
this would have read as a clean **5.5× "trained pathways deliver more drive"**,
and about a third of that was the connectome being allocated.

**Attempt 2 — null read 1.230 / 1.260**, after a materialisation pass that
instantiated every assembly's rows at beta=0. Still not 1.0, because
**recruitment continues at beta=0**. Each projection into `DST` can recruit new
neurons, and a newly recruited neuron draws its incoming synapses against
whatever source is *active* — which throughout the bind loop is always `a0`. So
`a0` accumulates materialised targets with plasticity switched off.

> **Plasticity and growth are separate switches**, and `beta=0` holds only one
> of them. In a lazily-materialised substrate, "no learning" is not "nothing
> happened".

## What finally worked

A **paired** statistic, which differences both out instead of arguing them away:

```
gain(word) = drive(word | beta=0.10) / drive(word | beta=0.0)
```

at the same seed and same `norm_init`. Construction is deterministic, so the two
runs share wiring, build, materialisation pass and recruitment history — the
only difference between them is whether the bind loop learned. Then
`gain(a0)` against `mean gain(a1..aM)` reads plasticity alone.

Note the direction of the correction: every unguarded framing **overstated** the
effect (5.5× → 1.5× → 1.23×), and all three were positive. The conclusion was
robust to the protocol; the magnitude was not remotely.

## Why this generalises past this experiment

Three of this session's errors are now the same shape — a knob that does more
than its name says:

- `forced_category` reads like a label, and is a **projection target**;
- `Area.w` reads like a neuron count, and is a **winners-length alias**;
- `beta=0` reads like "nothing happens", and still **materialises and recruits**.

Related: [[silent-no-op-dead-fibers]], [[same-name-two-meanings]],
`input_drive_normalization_is_disabled.md`,
`role_binding_readback_is_inverted_UNRESOLVED.md`.
