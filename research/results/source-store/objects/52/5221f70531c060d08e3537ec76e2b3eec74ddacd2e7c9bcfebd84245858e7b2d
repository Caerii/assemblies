"""A POS-tagged, feature-annotated SVO lexicon and grammar.

WHY THIS IS NEEDED, and it is not "more data for its own sake". The bridge
experiment's role-retrieval metric has NO RESOLUTION on the 6-word toy corpus:
with 3 words per role area, chance is 0.333 and a single extra hit moves the
statistic by 0.111, so the measured 0.444 and chance 0.333 are the same
measurement. No claim about role binding -- positive or negative -- can be made
at that scale.

The map says the scale is affordable. Capacity is extensive, M_max ~ 1.15 n/k,
so at n=10000, k=100 a role area holds about 115 items and 40 types sits well
inside it. At n=1000, k=50 the same 40 types sit ABOVE M_max ~ 23. That
contrast is not incidental -- it is the sharpest quantitative prediction the
phase map can make about the parser, and it is only available once the
vocabulary is big enough to straddle a capacity bound.

WHERE THE PART OF SPEECH COMES FROM. Positional mining out of the repo's
curriculum corpora was tried and it DOES NOT WORK; the record is kept in
`_svo_triples` below because the failure is informative. What is used instead:

  1. `emergent.core.grounding.VOCABULARY` is genuinely labelled --
     dominant_modality "visual" for object words, "motor" for action words.
     That is repo ground truth, but it covers only 10 nouns and 8 verbs.
  2. An AUTHORED lexicon, below, marked as authored. At this scale that is the
     right tool rather than a compromise: the part of speech of sixty concrete
     English words is not in doubt, and hand-authoring buys something a tagged
     corpus would not give for free -- SELECTIONAL RESTRICTIONS.

WHY THE FEATURES ARE LOAD-BEARING. A bare (nouns, verbs) split can only
generate sentences whose roles are recoverable from word order, because any
noun can fill any slot. Annotating nouns with semantic features and verbs with
what they select for makes the REVERSIBLE / IRREVERSIBLE distinction derivable:

  * "dog chases cat"  -- reversible: "cat chases dog" is equally licit, so ONLY
                        word order carries who did what.
  * "boy eats apple"  -- irreversible: "apple eats boy" is illicit, so lexical
                        semantics alone fixes the roles.

That distinction is exactly where the two candidate role mechanisms come apart.
The gated parser was measured as a perfect positional template -- 1.000 on
reversible sentences and 0.000 on irreversible -- i.e. flawless structure with
no lexical sensitivity at all. A corpus that cannot separate the two cases
cannot see that, and cannot test whether a lexical route supplies what gating
lacks. So the features are the experiment, not decoration.

ZIPFIAN BY CONSTRUCTION, because uniform frequency was the least plausible
thing about the earlier campaign and because it changes the physics: plasticity
is cumulative, so an item seen c times reaches effective gain g^c and the gain
becomes a distribution over items rather than a control parameter.

TWO SCHEDULES ARE EXPOSED and the difference is load-bearing:

  * `occurrences` -- every token, Zipfian. What a learner hears.
  * `bindings`    -- each (word, role) pair ONCE. What role training must use.

The derivation in role_binding_design.md gives the reason: both walls must hold
at once, so a single beta exists only if need^(c_max/c_min) <= g_c, and with a
Zipfian c_max that is violated by an order of magnitude. Driving c_max to 1 is
the only change that makes a single beta feasible at all.
"""

from __future__ import annotations

import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# --------------------------------------------------------------------------
# Semantic features. Deliberately minimal -- only the distinctions some verb
# in the lexicon actually selects on. Features nothing selects on would be
# unfalsifiable decoration.
# --------------------------------------------------------------------------
ANIM = "animate"      # can be an agent
HUMAN = "human"        # subset of animate; no verb selects on it yet, kept
EDIBLE = "edible"      # solid food
DRINKABLE = "drinkable"
OBJECT = "object"      # inanimate, manipulable, not ingestible

#: Authored nouns, word -> semantic features. Ordered by rough child-directed
#: familiarity, because that ordering is what the Zipf ranks are read off (see
#: `occurrences`). The ordering is an AUTHORED ASSUMPTION, not a frequency
#: measurement -- it asserts only that "dog" is commoner than "rabbit".
NOUNS = {
    # animate: humans (6)
    "boy": {ANIM, HUMAN}, "girl": {ANIM, HUMAN}, "baby": {ANIM, HUMAN},
    "man": {ANIM, HUMAN}, "woman": {ANIM, HUMAN}, "teacher": {ANIM, HUMAN},
    # animate: animals (10)
    "dog": {ANIM}, "cat": {ANIM}, "bird": {ANIM}, "horse": {ANIM},
    "cow": {ANIM}, "duck": {ANIM}, "fish": {ANIM}, "mouse": {ANIM},
    "bear": {ANIM}, "rabbit": {ANIM},
    # edible (7)
    "apple": {EDIBLE}, "cookie": {EDIBLE}, "bread": {EDIBLE},
    "banana": {EDIBLE}, "egg": {EDIBLE}, "cheese": {EDIBLE},
    "cake": {EDIBLE},
    # drinkable (3)
    "milk": {DRINKABLE}, "water": {DRINKABLE}, "juice": {DRINKABLE},
    # inanimate objects (14)
    "ball": {OBJECT}, "book": {OBJECT}, "cup": {OBJECT}, "spoon": {OBJECT},
    "shoe": {OBJECT}, "hat": {OBJECT}, "box": {OBJECT}, "doll": {OBJECT},
    "block": {OBJECT}, "bottle": {OBJECT}, "sock": {OBJECT},
    "towel": {OBJECT}, "key": {OBJECT}, "clock": {OBJECT},
}

ANY = frozenset({ANIM, HUMAN, EDIBLE, DRINKABLE, OBJECT})

#: Authored transitive verbs, word -> (features the AGENT must have,
#: features the PATIENT may have). Third-person singular forms throughout, so
#: that no verb string collides with a noun string -- checked in `build`.
#:
#: Every verb requires an animate agent, which is a fact about transitive
#: action verbs and not a simplification. Its consequence is measurable and
#: must be reported: ROLE_AGENT only ever sees the 16 animate nouns, so its
#: chance level (1/16) differs from ROLE_PATIENT's. Per-role chance, never a
#: single global number.
VERBS = {
    "chases": ({ANIM}, {ANIM}),
    "sees": ({ANIM}, ANY),
    "eats": ({ANIM}, {EDIBLE}),
    "holds": ({ANIM}, {OBJECT, EDIBLE}),
    "wants": ({ANIM}, ANY),
    "throws": ({ANIM}, {OBJECT}),
    "catches": ({ANIM}, {ANIM, OBJECT}),
    "finds": ({ANIM}, {ANIM, OBJECT, EDIBLE}),
    "drinks": ({ANIM}, {DRINKABLE}),
    "pushes": ({ANIM}, {ANIM, OBJECT}),
    "carries": ({ANIM}, {OBJECT, EDIBLE}),
    "touches": ({ANIM}, ANY),
    "pulls": ({ANIM}, {OBJECT}),
    "brings": ({ANIM}, {OBJECT, EDIBLE, DRINKABLE}),
    "drops": ({ANIM}, {OBJECT, EDIBLE}),
    "washes": ({ANIM}, {ANIM, OBJECT}),
    "opens": ({ANIM}, {OBJECT}),
    "licks": ({ANIM}, {ANIM, EDIBLE}),
    "hides": ({ANIM}, {OBJECT, EDIBLE}),
    "takes": ({ANIM}, {OBJECT, EDIBLE, DRINKABLE}),
}

_FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in",
    "on", "at", "with", "and", "or", "but", "my", "your", "his", "her",
    "its", "this", "that", "these", "those", "i", "you", "he", "she", "it",
    "we", "they", "me", "him", "them", "us", "for", "from", "by", "up",
    "down", "out", "off", "over", "very", "so", "not", "no", "yes",
}


def _svo_triples():
    """Three-token curriculum sentences. KEPT AS A NEGATIVE RESULT.

    POSITIONAL MINING DOES NOT WORK ON THIS CURRICULUM. Two attempts:

      1. Stripping function words BEFORE checking arity SHIFTS THE SLOTS:
         "the dog runs fast" becomes ["dog", "runs", "fast"] and "fast" is
         mined as a noun. Observed damage: 'will', 'can', 'why', 'fast' in the
         nouns and 'baby' in the verbs.
      2. Restricting to genuine three-token function-word-free sentences, as
         below, fixes the shift and still mislabels: entries like "big red
         ball" are adjective-adjective-noun, so the yield was 11 triples with
         'dog', 'cat', 'ball', 'under', 'mommy' mined as VERBS.

    The curriculum corpora are not POS-tagged and not uniformly SVO, so
    position carries no part-of-speech information. This function is retained
    only so the yield is reportable alongside the authored lexicon.
    """
    triples = []
    for mod_name, attr in (
        ("stage3_two_word", "STAGE3_CORPUS"),
        ("stage4_sentences", "STAGE4_CORPUS"),
        ("stage1_first_words", "STAGE1_CORPUS"),
        ("stage2_vocabulary_spurt", "STAGE2_CORPUS"),
    ):
        try:
            mod = __import__(
                f"neural_assemblies.lexicon.curriculum.{mod_name}",
                fromlist=["*"])
            corpus = getattr(mod, attr)
        except Exception:                                    # noqa: BLE001
            continue
        for entry in corpus:
            text = entry if isinstance(entry, str) else entry[0]
            toks = str(text).lower().split()
            if len(toks) != 3:
                continue
            if any(t in _FUNCTION_WORDS for t in toks):
                continue
            triples.append(tuple(toks))
    return triples


def grounded_words():
    """Words the repo actually labels, from VOCABULARY.dominant_modality."""
    n, v = [], []
    try:
        from neural_assemblies.assembly_calculus.emergent.core.grounding import (
            VOCABULARY,
        )
        for w, entry in VOCABULARY.items():
            mod = getattr(entry, "dominant_modality", None)
            if mod == "visual":
                n.append(w)
            elif mod == "motor":
                v.append(w)
    except Exception:                                        # noqa: BLE001
        pass
    return n, v


def licit(subj, verb, obj):
    """Does (subj, verb, obj) satisfy the verb's selectional restrictions?"""
    need_a, allow_p = VERBS[verb]
    if not (NOUNS[subj] & set(need_a)):
        return False
    if not (NOUNS[obj] & set(allow_p)):
        return False
    return subj != obj


def reversible(subj, verb, obj):
    """Is the role assignment recoverable ONLY from word order?

    True when swapping the two arguments yields an equally licit sentence, so
    semantics cannot disambiguate and order must. False when the swap is
    illicit, so the lexicon alone fixes the roles. This is the standard
    psycholinguistic sense of the term and it is the diagnostic the gated
    route fails.
    """
    return licit(subj, verb, obj) and licit(obj, verb, subj)


def build(n_nouns=40, n_verbs=20):
    """Return (nouns, verbs) with the repo's labelled words ordered first.

    Validates the two properties the parser depends on: no string is both a
    noun and a verb, and every noun/verb carries features. A collision would
    silently ask one token to live in two LEX areas.
    """
    lab_n, lab_v = grounded_words()

    def order(labelled, authored):
        out = [w for w in labelled if w in authored]
        out += [w for w in authored if w not in out]
        return out

    nouns = order(lab_n, list(NOUNS))[:n_nouns]
    verbs = order(lab_v, list(VERBS))[:n_verbs]
    both = set(nouns) & set(verbs)
    if both:
        raise ValueError(f"strings tagged as both noun and verb: {sorted(both)}")
    missing = [w for w in nouns if not NOUNS.get(w)]
    if missing:
        raise ValueError(f"nouns with no features: {missing}")
    return nouns, verbs


def occurrences(nouns, verbs, n_sentences=400, zipf_s=1.0, seed=0,
                enforce=True):
    """Zipfian licit SVO token stream: what a learner hears.

    Ranks come from list position, which is the authored familiarity ordering.
    `enforce` applies the selectional restrictions by rejection sampling; with
    it off the same sampler produces the semantically unconstrained corpus,
    which is the control for any claim that the restrictions matter.

    ONE COUPLING TO BE AWARE OF WHEN READING REVERSIBILITY RATES. The authored
    familiarity ordering happens to put all 16 animate nouns first, so rank and
    animacy are correlated and a steeper Zipf samples animates preferentially.
    That is why zipf_s=1.0 yields MORE reversible sentences than zipf_s=0.0
    (269 vs 112 in 400) even though reversibility is the minority class over
    the licit set (2160 of 7376). Child-directed speech genuinely is
    animate-heavy, so the direction is not wrong, but the magnitude here is an
    artifact of the authored ordering and must not be read as a corpus fact.
    """
    rng = random.Random(seed)

    def weights(items):
        return [1.0 / ((i + 1) ** zipf_s) for i in range(len(items))]

    nw, vw = weights(nouns), weights(verbs)
    out = []
    for _ in range(n_sentences):
        for _attempt in range(200):
            s = rng.choices(nouns, weights=nw, k=1)[0]
            o = rng.choices(nouns, weights=nw, k=1)[0]
            v = rng.choices(verbs, weights=vw, k=1)[0]
            if s == o:
                continue
            if not enforce or licit(s, v, o):
                out.append([s, v, o])
                break
    return out


def bindings(sentences):
    """Each (word, role-slot) pair ONCE, in first-occurrence order.

    Slots are positional: 0 AGENT, 1 ACTION, 2 PATIENT. This is the schedule
    role training must use -- c_max=1 is a feasibility requirement, not an
    optimisation. See the module docstring.
    """
    seen, out = set(), []
    for sent in sentences:
        for slot, word in enumerate(sent):
            if (word, slot) not in seen:
                seen.add((word, slot))
                out.append((word, slot))
    return out


def summary(nouns, verbs, sentences):
    counts = Counter(w for s in sentences for w in s)
    per_slot = [Counter(s[i] for s in sentences) for i in range(3)]
    rev = sum(1 for s in sentences if reversible(*s))
    return {
        "n_nouns": len(nouns), "n_verbs": len(verbs),
        "n_sentences": len(sentences),
        "tokens": sum(counts.values()), "types": len(counts),
        "c_max": max(counts.values()), "c_min": min(counts.values()),
        "per_slot_types": [len(c) for c in per_slot],
        "n_bindings": len(bindings(sentences)),
        "reversible": rev,
        "irreversible": len(sentences) - rev,
    }


if __name__ == "__main__":
    nouns, verbs = build()
    lab_n, lab_v = grounded_words()
    print("\n  LEXICON  (authored, POS-tagged, feature-annotated)")
    print(f"    nouns {len(nouns):>3}   verbs {len(verbs):>3}   "
          f"repo-labelled: {len(set(lab_n) & set(nouns))} nouns, "
          f"{len(set(lab_v) & set(verbs))} verbs")
    print(f"    curriculum positional mining yield (unusable, see docstring): "
          f"{len(_svo_triples())} triples")
    for feat in (ANIM, EDIBLE, DRINKABLE, OBJECT):
        ws = [w for w in nouns if feat in NOUNS[w]]
        print(f"    {feat:<10} {len(ws):>3}  {ws[:8]}")

    licit_all = [(s, v, o) for s in nouns for v in verbs for o in nouns
                 if licit(s, v, o)]
    rev_all = [t for t in licit_all if reversible(*t)]
    print(f"\n    licit SVO sentences : {len(licit_all)} of "
          f"{len(nouns) ** 2 * len(verbs)} combinations "
          f"({100 * len(licit_all) / (len(nouns) ** 2 * len(verbs)):.1f}%)")
    print(f"      reversible        : {len(rev_all)}  "
          f"(order is the only cue)")
    print(f"      irreversible      : {len(licit_all) - len(rev_all)}  "
          f"(lexical semantics fixes roles)")

    for s in (0.0, 1.0):
        sents = occurrences(nouns, verbs, n_sentences=400, zipf_s=s)
        info = summary(nouns, verbs, sents)
        print(f"\n    zipf_s={s}: " + "  ".join(
            f"{k}={v}" for k, v in info.items()))
        for slot, name in enumerate(("AGENT", "ACTION", "PATIENT")):
            t = info["per_slot_types"][slot]
            print(f"      {name:<8} {t:>3} types -> chance "
                  f"{1 / max(t, 1):.4f}")
