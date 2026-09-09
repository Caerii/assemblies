# One shared lexical area: what grounding buys, and what Hebbian costs

Harness: `research/experiments/shared_lexical_area.py`
Vocabulary: `medium` preset, 211 grounded words, 7 categories.
Substrate: one area, `n=6104 k=30 p=0.05`, feed-forward only, `norm_init=True`.
`n` set from the critical load (alpha = 1.037 < alpha* ~ 1.15) so no result here
is a capacity artifact. 8 seeds, numpy/random re-seeded per trial.

## 0. The grounding table cannot answer the question it looks like it answers

Before running anything, the feature statistics:

    distinct grounding features   163
    CATEGORY-PURE features        163 / 163  (100%)
    feature Jaccard within-cat    0.1273
    feature Jaccard between-cat   0.0000     <-- exactly zero
    cross-category pairs sharing >=1 feature   0 / 19365

Features are named `visual:ANIMATE`, `motor:TRANSITIVE`. A word's category is
its `dominant_modality`, and every feature carries the modality in its name, so
**the feature set is a one-hot encoding of the category**. No noun shares a
single feature with any verb.

Run as-is, "do categories emerge from grounding in a shared area?" scores 0.96
and means nothing. That is the same degeneracy that retracted the depth-3
composition result: a control that must succeed for trivial reasons.

The three arms below exist because of this.

| arm | features | role |
|---|---|---|
| PREFIXED | `visual:ANIMATE` | positive control, must score high trivially |
| STRIPPED | `ANIMATE` | the real test — categories now collide |
| PERMUTED | shuffled across words | null |

STRIPPED is not cosmetic: `QUALITY` occurs as both `properties:QUALITY`
(adjectives) and `motor:QUALITY` (verbs), so dropping the prefix makes a word's
nearest neighbours genuinely category-ambiguous. Between-category overlap goes
from 0.0068 to 0.0935 — a real channel opens.

## 1. Category IS recoverable from a single shared area

    arm        nn-acc            nn-chance   majority
    PREFIXED   0.9637 +/- 0.007    0.2343      0.3365
    STRIPPED   0.8215 +/- 0.022    0.2343      0.3365
    PERMUTED   0.2528 +/- 0.048    0.2343      0.3365

The baseline for a nearest-neighbour score is the collision probability
`sum(p_c^2) = 0.2343`, **not** the majority frequency 0.3365 — a constant
classifier and a 1-NN classifier have different chance levels. PERMUTED lands
on 0.2343, which is what a working null looks like.

So the eight-way pre-partition into core areas is not required to keep word
categories apart. One area holds all 211 words and the category is legible in
the overlap structure at 0.82 against a chance of 0.23.

## 2. But almost none of that is learned

The beta=0 control is the whole result:

    beta    nn-acc             within   between   within/between
    0.00    0.7796 +/- 0.018   0.0134   0.0047      2.88x
    0.05    0.7980 +/- 0.020   0.1811   0.0649      2.79x
    0.10    0.8199 +/- 0.022   0.2354   0.0939      2.51x
    0.30    0.8021 +/- 0.024   0.2760   0.1344      2.05x

With plasticity switched off entirely, category accuracy is **0.7796**.
Learning buys +0.04, which is under two pooled standard deviations.

The structure is geometric, not learned. Words that share input stimuli receive
overlapping drive, so k-WTA elects overlapping winners — the projection is
approximately similarity-preserving, and the similarity was already in the
input. This is a real and useful substrate property (it is why generalisation to
held-out words works at all), but it is not category formation. Nothing is
discovered.

## 3. Hebbian learning is actively anti-categorical here

The separation ratio **decreases monotonically in beta**: 2.88 -> 2.79 -> 2.51
-> 2.05. Between-category overlap is amplified *faster* than within:

    beta 0 -> 0.1     within  0.0134 -> 0.2354   17.6x
                      between 0.0047 -> 0.0939   19.9x

Accuracy barely moves because 1-NN reads rank order, not magnitude; the ratio is
what degrades.

The mechanism is visible in the PREFIXED arm, where between-overlap grows only
1.9x (0.0036 -> 0.0068) because no feature is shared across categories. So the
amplification is carried entirely by the **shared feature stimulus**: Hebbian
potentiation on a stimulus common to two words drags both toward a common
assembly, and it has no way to represent "these two share QUALITY but differ
elsewhere". Conjunctions are lost; the strongest shared channel wins.

## 4. This is the drive-ratio problem again

Same shape as three results already on record:

* mood collapse — MOOD holds ~4% of SYN's drive, separation tracks the ratio at
  r = 0.999
* role binding stuck in the crowding regime (0.15-0.22)
* shared-area collapse under recurrence

In each case a strongly-potentiated shared channel swamps the weaker
distinguishing channels, and plasticity makes it worse because it potentiates
the shared channel hardest. That is now measured at the lexicon too, with a
clean beta=0 null and a monotone dose-response.

## What this rules in and out

* **Ruled out:** the 8-way core-area partition is *not* load-bearing for keeping
  categories apart, and adding plasticity to a shared area is not the fix.
* **Ruled out:** "grounded category learning" cannot be evaluated on the current
  grounding table without stripping the modality prefix.
* **Ruled in:** the missing ingredient is a *sharpener* — something that turns a
  graded similarity gradient into discrete clusters against the pull of shared
  features. Plain Hebbian is the wrong sign.

Candidates already implemented in-repo and unwired:
`PlasticityEngine.anti_hebbian_update` (test-only importers),
`Brain.add_mutual_inhibition` (dormant — no `project()` co-targets a group).

## Reproduce

    python research/experiments/shared_lexical_area.py --seeds 0 1 2 3 4 5 6 7
    python research/experiments/shared_lexical_area.py --regimes stripped \
        --seeds 0 1 2 3 4 5 6 7 --beta 0.0
