# The stored lexicon and the live parse address different neurons (overlap 0.27)

Two checks on the SENTENCES parser's 71 stored `NOUN_CORE` assemblies, three
seeds.

| check | result |
|---|---|
| **round trip** — activate the stored assembly, read the area back | **1.0000** on every word, every seed |
| **freshness** — re-project `phon → NOUN_CORE` and compare to the snapshot | **0.2737 / 0.2751 / 0.2939** mean, **many words at 0.000** |

`assembly_is_current` returns True for 71/71.

## What the round trip settles

**`activate_assembly` is exonerated.** The neuron-ID ↔ compact-index mapping is
exact right now, and nothing is being silently truncated or mis-addressed. That
was #120's step (a) and it is closed: the index space is not the problem.

## What the freshness number means

The stored snapshots **no longer match what the words produce**. Mean overlap
0.27, and `dog`, `bread`, `cake`, `king`, `man` read **0.000** — completely
disjoint from the assemblies their own stimuli now generate.

Measured under `brain.probe()`, so the re-projection could not itself learn or
recruit. This is a read of the current connectome, not a side effect of taking
it.

The codebase half-knows this. `train_roles` replays the stabilised snapshot
rather than re-projecting, and says why:

> re-projecting `phon -> core` carries plasticity, so the core assembly drifts
> between the moment a role binding is stored and the moment it is read back,
> and retrieval then misses.

What that comment does not say is **how far** it drifts, or that the divergence
persists with plasticity off. 0.27 is not drift around a stable attractor; at
0.000 the two have nothing in common.

## The consequence: binding and parsing address different neurons

Two code paths, two different sources:

- `train_roles` binds **from the stored snapshot** — `bind(brain, core_area, role_area, stored_core)`
- `_advance_incremental_word` parses with a **fresh projection** — `project(brain, phon, core_area, rounds)`

So the synapses a role binding strengthened belong to the *stored* assembly,
while the ERP probe drives with the *fresh* one — and those overlap ~0.27.
**Roughly three quarters of the drive during a live parse comes from neurons
that were never part of the binding.** The learned pathway is largely not being
addressed by the thing that reads it.

That is a mechanical candidate for why the P600 shows no pathway learning
(`the_p600_is_area_identity_not_pathway_learning.md`, AUC 0.515). It is a
candidate, not the finding — see below.

## What it does NOT explain, and this is the honest part

`role_binding_writes_anything.py` probed with the **stored** assemblies — the
same ones the binding was written from, with a verified 1.0000 round trip — and
still found no positive pathway effect (AUC 0.458 / 0.348 / 0.358). So
source-side divergence cannot be the whole story.

Meanwhile the minimal substrate shows binding IS readable when **one** assembly
is bound: plasticity effect 1.2280 / 1.3544, 5/5 seeds
(`a_beta_zero_null_does_not_hold_the_connectome_still.md`).

The difference between the two is **how many assemblies share the target**:

| | bindings into the target | pathway readable? |
|---|---|---|
| minimal substrate | 1 | yes, 1.23–1.35× |
| parser `ROLE_PATIENT` | 36 | no, AUC ~0.4 |

## The hypothesis that now sits at the front, and how to kill it

**Target-side capacity.** With many assemblies bound into one role area,
individual pathway strength stops being recoverable. Source-side crowding is
already ruled out — `NOUN_CORE` pairwise overlap is 0.126–0.132, so the source
assemblies are distinct.

**The experiment is a sweep, and it is cheap**: in the minimal substrate, vary
M = the number of assemblies bound into `DST` from 1 upward and find where the
paired plasticity effect decays to 1.0. That produces a capacity law rather than
a yes/no, it uses a harness that is already built and already has a working
null, and it moves the question from ERP plumbing to substrate capacity — which
is the composition line's question, not the ERP line's.

If the effect survives to M = 36, target capacity is refuted and the parser's
state is doing something neither experiment has isolated.

## Open, and separable

Whether the stored/fresh divergence is drift or something sharper (consolidation
re-issuing IDs would look like this too, and `assembly_is_current` checks
presence rather than identity) is **not** settled here and does not need to be
settled to use the number above. Both readings imply the same thing: a snapshot
taken at training time does not address the neurons a later parse activates.
