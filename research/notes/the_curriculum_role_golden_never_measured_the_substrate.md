# RETRACTED TITLE — the role probe DOES measure the substrate. The failure is a one-sided lexical binding.

> **CORRECTION, same day.** The original version of this note concluded that
> `evaluate_roles` is "substrate-invariant" and "never measured the assembly
> calculus", from the fact that β and `phon_weight` left the accuracy
> byte-identical. **That inference was wrong**, and a census of the actual
> mechanism refutes it. The corrected finding is below. Commit `008ba9d`'s
> message carries the wrong claim.

## What is still true

`nemo2025_curriculum` asserts `role_probe_accuracy_min = 1.0` and delivers
**0.6667**. Pre-existing — reproduced in a worktree at `6ecfa15` with
`ASSEMBLIES_BACKBONE_CACHE=0`. It stayed green because warm backbones
deserialize instead of training, so a golden recorded pre-`norm_init`
(2026-06-23) could rot while CI passed.

And β=0.05 / `phon_weight`=6 do leave the accuracy at exactly 0.6667.

## What I got wrong

I read "accuracy does not move" as "the metric cannot see the substrate."
Censusing the mechanism shows the opposite. `_assign_roles_neural` computes

```
score = prior + lexical,   lexical = (margin + eps)/total if any(margins) else 0
```

and I predicted every `margin` would be undefined, making `lexical` identically
zero. Measured, over both probes:

| | margin reads | defined | non-zero | value |
|---|---|---|---|---|
| golden as recorded | 5 | **5/5 (100%)** | 1/5 | `dog → ROLE_PATIENT` **0.9855** |
| β=0.05 + phon_weight=6 | 5 | **5/5 (100%)** | 1/5 | `dog → ROLE_PATIENT` **1.0000** |

The lexical term is **live**, and it **is** substrate-sensitive — 0.9855 → 1.0000
under the drive-share fix. My claim that it was inert was false.

## The actual root cause

```
['the','dog','runs']            -> dog = PATIENT   (expected AGENT)   WRONG
['the','cat','chases','the','bird'] -> cat = AGENT, bird = PATIENT    correct
```

`ROLE_AGENT` holds 46 entries, `ROLE_PATIENT` 36 — and **`dog` is in
`ROLE_PATIENT` only**. It was never bound as an agent during training. So at
probe time the lexical evidence for PATIENT is ~1.0, there is no AGENT evidence
at all, and the lexical term outvotes the structural prior that would otherwise
make the preverbal noun the agent.

**One noun with a one-sided training history, assigned its habitual role in a
sentence where it has a different one.**

## Why the arms are byte-identical, correctly explained

The substrate genuinely improved — the margin rose 0.9855 → 1.0000. But the
*decision* was already saturated and wrong, so a better substrate makes the
wrong answer **more confident** without flipping it. With a denominator of 3, an
unflipped decision is an unmoved metric.

So the invariance was never evidence about the metric's wiring; it was evidence
that the error is **categorical, not marginal**. Improving binding quality
cannot fix a word that has no AGENT binding to retrieve.

## What this says about the architecture

A per-word role lexicon encodes *which role this word usually had*, not *which
role it has in this sentence*. The structural prior exists to supply the latter,
but the current combination lets a single strong lexical match override it. For
an intransitive subject — where the lexicon has nothing useful and position has
everything — that is exactly backwards.

That is a real design question, and it is #33's question (retire the symbolic
route?) restated with a measurement attached: the two routes are combined by
`prior + lexical` with no notion of which one is *entitled* to decide for this
word in this frame.

## What to do

1. **Still do not re-record at 0.667.** The claim is broken, not the threshold.
2. The fix is not #121 (a better readout) as I previously wrote — the readout
   already works. It is either (a) train `dog` as an agent, which fixes the
   probe and nothing else, or (b) make the lexical term unable to override the
   prior when the word has evidence for only ONE role and the frame implies
   another. (b) is the real change and needs its own study.
3. Anything asserting "role accuracy" on 3 items is under-powered; one item is
   0.333.

## Limits

One seed (42), the golden's own. Two sentences, three role slots. I did not
check whether `dog` is absent from `ROLE_AGENT` because the corpus never uses it
as one, or because the agent binding was written and lost — that distinction
matters for fix (a) and is not yet measured.
