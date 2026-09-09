# The assembly DOES decide — it is answering the wrong question

## What I predicted, and why it was wrong

I expected the neural term to be inert. The arithmetic looked decisive:

```
score = _STRUCTURAL_PRIOR * (0.5 ** rank)  +  (margin + eps) / total
        \___ 1.2 at rank 0, 0.6 at rank 1 ___/   \___ in [0, 1] ___/
```

so the assembly-derived term can only overturn the prior when the normalised
margin gap exceeds **0.6**, and #52 recorded role binding at 0.15–0.22 — the
crowding regime. `passive_payoff` then found the parse answer bit-identical
across 10 distinct substrates, which is what an inert term looks like from
outside.

**Both readings were wrong.** Measured from the inside:

| | full | prior_only | neural_only |
|---|---|---|---|
| accuracy | 0.7250 | **1.0000** | 0.675–0.725 |

* margins reach **1.0000**, not 0.15–0.22;
* they are non-zero on **174 of 360** candidate evaluations;
* they **flip 33 of 120 decisions** (11 per seed, every seed).

The neural term is loud, and the structural route alone is *perfect* on these
held-out reversible probes. Adding the assemblies takes 1.0000 down to 0.7250.

The earlier seed-invariance was real but I drew the wrong conclusion from it:
the answer is stable across substrates not because the assemblies are ignored,
but because what they contribute is stable across substrates — a per-word
statistic that every seed learns identically from the same corpus.

## What the flips are tracking

For every flip, does the decision move toward the word's **trained majority
role**?

```
39 toward    9 against    18 unknown/both
```

4.3 : 1, and the examples are unambiguous:

```
'the child is entered by the mouse':  child  PATIENT -> AGENT  (trained bias AGENT)
'the girl  is entered by the horse':  girl   PATIENT -> AGENT  (trained bias AGENT)
'the bear  is entered by the dog'  :  bear   PATIENT -> AGENT  (trained bias AGENT)
```

The passive says `child` is the patient. The gating route gets that right. The
assembly overrides it, because `child` was usually an agent in training.

## The mechanism, visible in a signature

```python
self._role_binding_margin(word, core_area, role_area)
```

**There is no sentence parameter.** The method activates the word's *stored*
core assembly and asks which role area it best matches. Nothing about the
current sentence — not the voice, not the marker, not position, not the other
participant — enters the computation.

So the term is structurally incapable of being anything but a word-level role
prior. It is not weak, mis-weighted, or crowded. It is answering
*"what role does this word usually play?"* when the parser needs
*"what role does this word play HERE?"* On any sentence that departs from a
word's usual role — which is exactly what a passive is — it must be wrong.

This is the same fact recorded from the other side in
`gating-is-load-bearing-for-binding`: *sentence read == word-alone read,
1.0000*. There is no sentence-conditioned state for a readout to read.

## So: how do we make the assembly decide?

Not by re-weighting. Turning the prior down does not help, because the evidence
being weighted is the wrong evidence — `neural_only` sits at 0.675–0.725, below
the structural 1.0000.

The requirement is a **sentence-conditioned binding**: the readout must consult
an assembly formed *during this parse*, in which the word and the sentence's
structure have both participated, rather than the lexicon entry written during
training. Concretely, the margin needs to be a function of

    (word assembly, the role area's state under THIS sentence's gating)

and the repo already has the pieces that would produce such a state — fiber
gating (#45) opens a role area conditioned on the parse — but the readout does
not use them; it queries the stored lexicon instead.

**The test that this has been fixed is not accuracy.** It is that the parse of a
word *in a sentence* stops being identical to the parse of the word *alone*, and
that the margin for a given word CHANGES between the active and passive frames
of the same event. Until that quantity moves, any accuracy gain is coming from
somewhere other than the substrate.

## Limits

* 40 probes, 3 seeds, one preset. The flip counts are identical across seeds
  (11/40 every time), consistent with a per-word statistic.
* `prior_only = 1.0000` is on **reversible** probes drawn from the same
  generator family as training; it is not a claim that word order solves
  parsing generally. It does mean the structural route is not the bottleneck
  here, so the assemblies cannot be credited for what it already does.
* `neural_only` is not a clean arm: with the prior at zero and no lexical
  evidence, the argmax falls to `role_order[0]`, which is still the structural
  answer. Its accuracy is an upper bound, and the flip count is the honest
  statistic.
* The trained-bias reading uses membership in the AGENT/PATIENT role lexicons,
  which is a coarse proxy for a frequency; 18 of the 66 flipped words were in
  both lexicons and are counted as unknown rather than assumed.
