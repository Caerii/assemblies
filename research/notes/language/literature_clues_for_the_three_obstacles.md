# What the literature says about the three obstacles

Read of all ten PDFs in `research/literature/papers/`, aimed at exactly three
questions and nothing else:

1. **Categories are annotation-driven.** Cut the three hand-authored routes and
   the parser answers one class for everything, at majority baseline.
2. **No real corpus.** #30, flagged in-repo as "the binding constraint on every
   induction result."
3. **Multi-mood word order collapses.** The emergent chains form at init
   (SYNTAX overlap 0.04) and merge within ~20 sentences (-> 1.00).

The short version: **two of the three have a published mechanism we have not
implemented, and the third is smaller than it looks.** In each case our
implementation substituted a hand-authored structure for a mechanism the paper
derives, and in each case the substitution is the thing that fails.

---

## Obstacle 1 — categories

### What we do

`emergent/core/grounding.py` maps a word to a core area by **dominant
modality**, resolved by a fixed priority order over hand-authored modality
lists:

    VISUAL -> NOUN_CORE,  MOTOR -> VERB_CORE,  PROPERTY -> ADJ_CORE, ...

The docstring cites Mitropolsky & Papadimitriou (2025). That citation does not
survive contact with the paper: **the paper has no such map.**

### What the paper does (mitropolsky2025_acquisition, §2.1-2.2)

Two lexical areas, LEX1 and LEX2, **both tabula rasa**, no label anywhere. The
only prior is *architectural asymmetry*:

> "Four of these 2m + 6 fibers, namely the ones between PHON and the two lexical
> areas, as well as the one between LEX1 and VISUAL, and the one between LEX2
> and MOTOR, have **increased parameters β and p**."

Every word is presented to *both* lexical areas. Which one forms a real
assembly is decided by NEMO dynamics plus co-occurrence statistics. The readout
is **not a classifier** — it is a stability probe (Property 3):

> "firing PHON[w] into LEX2 results in a 'wobbly' set of neurons: firing this
> set again recurrently ... results in a quite different set."

So: fire `PHON[w]` into both; the area whose k-cap is self-sustaining *and*
fires back onto `PHON[w]` is the word's class.

**The graded prediction is the valuable part.** Instability in the wrong area
grows with the number of distinct complements the word occurred with — "dog"
with 7 verbs reads ~30% stability in LEX2 against 100% in LEX1 (Fig 3d). That
is a curve, not a point, and it is reproducible.

Sample efficiency: ~10 sentences per word, linear in lexicon size (Fig 3a).

### Why this matters to us specifically

We have been treating recurrent instability as a **defect** — it is the subject
of `[[self-recurrence-stability-window]]`, `[[recurrence-is-the-collapse-channel]]`,
and `[[norm-init-stability-threshold]]`. In this paper the same quantity is the
**signal**: it is how the system knows a noun is not a verb. We already have the
measurement (`assembly_calculus/metrics/instability.py`) — pointed at the P600
rather than at part of speech.

### The blocker, named

**`p` is a Brain-level constant. There is no per-fiber `p`.** `update_plasticity`
gives per-fiber β; nothing gives per-fiber density. The paper's four privileged
fibers need both. Either add per-fiber `p`, or test whether β asymmetry alone
carries the noun/verb split — that is itself a clean experiment and the answer
is not obvious.

---

## Obstacle 2 — corpus

No paper in the set uses a natural corpus; the acquisition paper generates
sentences by uniform choice of noun and verb. So the literature does not hand us
CHILDES. What it does hand us is **the sample-efficiency claim to test against**
(~10 sentences/word, linear in ℓ) and three constraints that decide whether a
corpus experiment is even well-posed:

- Function words are **ignored**, not learned. Our #29 ("function-word
  bootstrapping does not clear chance") is attacking a problem the source
  material explicitly declines.
- Abstract words are out of scope — the grounding assumption requires a
  sensorimotor representation per word.
- Word-order learning **requires >50% transitive input**. Below that, the paper
  proves VOS/OSV/OVS are unlearnable by this mechanism *with unbounded input*.
  Any corpus we pick has a transitivity ratio, and it is a gate, not a nuisance
  parameter. CHILDES child-directed speech should be measured for this before
  any run.

`ting2026_speech` is the only paper that touches real signal: phone (F1 0.69)
and word (F1 0.61) boundary detection from continuous speech **with no weight
training**, plus 47.5% phone classification. That is a candidate for the PHON
area the acquisition paper assumes into existence. Out of scope for now, but it
means the input-convention gap is not unbridged in the literature.

---

## Obstacle 3 — multi-mood collapse

### Where we actually are

`reference/word_order_learner.py` already carries the fix — `per_mood_syntax`,
one SUBJ/VERB/OBJ triple **per mood** — and it gets 24/24 against the emergent
version's 17/24 (`b756d0d`). The docstring is honest that this is a deviation:
the distinctness is made *structural* rather than learned.

Which makes obstacle 3 the same disease as obstacle 1. It is not "we can't do
multi-mood"; it is "we can only do multi-mood by architecting the answer in."

### The mechanism we are missing (dabagia2025_sequences, Thm 4)

NEMO can learn an **arbitrary finite state machine**, and multi-mood word order
*is* an FSM: state = last constituent emitted, symbol = mood, transition = next
constituent. The paper's architecture is explicit about what carries the
transition:

> "There is an assembly for each symbol, state, and transition; each pair of
> state and symbol assemblies projects to the associated **arc assembly**, which
> in turn projects back to the assembly corresponding to the state the FSM would
> switch to."

The arc assembly `A_{q,σ}` is a **conjunction** of (state, symbol) with its own
identity. We have no such thing — and "SYNTAX stays mood-blind (0.95)" is
precisely the reading you get when the conjunction is represented nowhere.

Sizing is not the constraint: Theorem 4 wants `n ≥ |Q|²|Σ|²`. For 3-4
constituents and a handful of moods that is a few hundred neurons, which is
consistent with our measured finding that the collapse survives n=1e5.

### The one-line version of the fix

Our HELPER areas are already positioned to *be* `A_{q,σ}`. They receive
`SYNTAX_prev -> helper` (the state) and `MOOD -> helper` (the symbol). But in
`_project_training`:

```python
if first_word:
    pmap[MOOD] = pmap[MOOD] + [helper]
```

**MOOD reaches the helper on the first word only.** Under Theorem 4 the arc is
conjunctive at *every* transition. Every non-initial transition in our model is
therefore mood-blind by construction, which is exactly the failure profile
recorded in the code comment: the failing pairs are "the pairs that need to
diverge after a shared opening constituent (3/6 and 2/6)."

This is cheap to test and it is a *derivation*, not a guess.

### Three further divergences worth checking, in cost order

**(a) Mutual inhibition is dormant.** The acquisition paper says of the three
ROLE areas: "this is **the only** use of interarea inhibition in our model," and
it is load-bearing for generation —

> "the current constituent will continue to receive the most input from its own
> recurrent firing, and because the role areas are in mutual inhibition, will
> continue to fire until the next time all the role areas are inhibited."

`[[mutual-inhibition-prefers-untrained]]` measured 1373 `project()` calls with
**zero** co-targeting a group. The paper's only inhibition mechanism never runs
here. Whatever else is true, the generation dynamics are not the paper's.

**(b) `p` is at the bottom of the paper's own sweep.** The sequence experiments
run at **p = 0.2**; Figure 8 sweeps p from 0.04 to 0.4 and shows max-overlap
between sequence elements *falling* as p rises. Our chain work is at p = 0.05 —
the low end, where the paper reports high overlap. Given
`[[kp-decides-whether-beta-helps]]`, this is a first-class suspect and a
one-parameter experiment.

**(c) β, and homeostasis.** Theorem 2's bound,

    β ≤ ln(n/(2kL)) / (2·max{Δp, 6 ln n}²)

evaluates to ~5e-4 with T ≳ 3e4 presentations at our sizes — far below anything
practical, and the paper's own experiments run at β = 0.1 with 10 presentations.
**So the bound is not directly actionable and should not be quoted as one.**
What *is* actionable is the sentence next to it: "the number of presentations
needed grows inversely with the plasticity, and crucially, **the plasticity
cannot be too high**." Our recorded collapse — chains present at init, gone in
~20 sentences — is the shape of β-too-high, and `[[mood-chain-collapse-mechanism]]`
lists beta as "ruled out" without recording how far down it was swept.

Second: Theorem 2 assumes **homeostasis after every round** ("each neuron's
incoming weights sum to 1"). We do one-time `norm_init` and
`[[recurrence-needs-norm-init]]` explicitly records the choice as
"not homeostasis". The theorems that promise non-merging chains assume the
thing we decided against. That is worth re-opening as a measurement rather than
a preference.

---

## A fourth thing, unasked for, that bears on every result

`hoff2026_epwta` replaces k-WTA with **E%-WTA**: fire

    F_t = { j : h_j(t) ∈ [(1-ε)·h_max(t), h_max(t)] }

so assembly size is set by the drive distribution instead of fixed at k.

Adopting it as production is a large change and not proposed here. But as a
**diagnostic** it is close to free and it targets our single most persistent
defect class. Under k-WTA a degenerate area returns exactly k winners and reads
as a healthy assembly — that is `[[silent-no-op-dead-fibers]]` ("a zero-drive
projection still returns k winners"), the index tie-break in
`[[reset-in-training-loop-collapses-areas]]`, and the 1.000-that-means-nothing
family in `[[fake-perfect-probe-signatures]]`. Under E%-WTA a degenerate drive
produces a *wrong-sized* set — near-empty or near-n — and degeneracy stops being
silent.

`|F|` computed alongside the existing k-cap is a drive-degeneracy meter that
costs one extra comparison per projection.

---

## What is NOT here

- No claim that any of this fixes anything. These are derivations from the
  source papers plus one code reading each; nothing below has been run.
- The β bound above is a worst-case theoretical quantity and is reported here
  *because* it is unusable at our scale, not as evidence.
- `kopadi2026_direct` (causal direction) and `dabagia2024_coinflipping` were
  read and are not relevant to these three obstacles.
