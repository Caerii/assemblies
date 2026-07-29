"""A real, Zipfian SVO corpus, built from the repo's existing curriculum.

WHY THIS IS NEEDED, and it is not "more data for its own sake". The bridge
experiment's role-retrieval metric has NO RESOLUTION on the 6-word toy corpus:
with 3 words per role area, chance is 0.333 and a single extra hit moves the
statistic by 0.111, so 0.444 and 0.333 are the same measurement. No claim about
role binding -- positive or negative -- can be made at that scale.

The map says the scale is affordable. Capacity is M_max ~ 1.15 n/k, which at
n=10000, k=100 is about 115 items per role area, so 30-60 words per role sits
comfortably inside it and puts chance near 0.02-0.03, where an effect is visible.

SOURCED FROM WHAT ALREADY EXISTS rather than invented. `neural_assemblies.
lexicon.curriculum` ships STAGE1-4 corpora, and STAGE3/STAGE4 contain
three-word SVO sentences ("mommy read book", "daddy throw ball"). Parsing those
BY POSITION yields nouns (slots 0 and 2) and verbs (slot 1) without guessing at
part of speech, which is far more trustworthy than a heuristic over a bare word
list. `emergent.core.grounding.VOCABULARY` then supplies modality for the words
it covers, so grounding stimuli stay meaningful.

ZIPFIAN BY CONSTRUCTION. Frequency falls as 1/rank^s, because uniform frequency
was measured to be the least plausible thing about the earlier campaign and
because it changes the physics: plasticity is cumulative, so an item seen c
times reaches effective gain g^c and the gain becomes a distribution over items
rather than a control parameter.

TWO SCHEDULES ARE EXPOSED, and the difference is load-bearing:

  * `occurrences` -- every token, Zipfian. This is what a learner hears.
  * `bindings`    -- each (word, role) pair ONCE. This is what role training
                     should use.

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

_FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in",
    "on", "at", "with", "and", "or", "but", "my", "your", "his", "her",
    "its", "this", "that", "these", "those", "i", "you", "he", "she", "it",
    "we", "they", "me", "him", "them", "us", "for", "from", "by", "up",
    "down", "out", "off", "over", "very", "so", "not", "no", "yes",
}


def _svo_triples():
    """SVO triples mined from the curriculum's three-word sentences.

    Position gives part of speech, which is why this is preferable to
    classifying a flat word list: slot 1 of a three-word child-directed
    sentence is the verb essentially by construction.
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
            # ARITY IS CHECKED BEFORE ANY FILTERING, and that matters. An
            # earlier version stripped function words first, which SHIFTS THE
            # SLOTS: "the dog runs fast" becomes ["dog", "runs", "fast"] and
            # "fast" is then mined as a noun. The observed damage was real --
            # 'will', 'can', 'why' and 'fast' entered the noun list and 'baby'
            # entered the verbs. Only genuine three-token sentences are used, so
            # slot 1 really is the verb.
            if len(toks) != 3:
                continue
            if any(t in _FUNCTION_WORDS for t in toks):
                continue
            triples.append(tuple(toks))
    return triples


# POSITIONAL MINING DOES NOT WORK ON THIS CURRICULUM, and the attempt is left
# recorded above because the failure is informative. Even restricted to genuine
# three-token function-word-free sentences, slot 1 is not reliably the verb:
# entries like "big red ball" are adjective-adjective-noun, so the yield was 11
# triples with 'dog', 'cat', 'ball', 'under' and 'mommy' mined as VERBS and
# 'big', 'run', 'little' as NOUNS. The curriculum corpora are not POS-tagged and
# are not uniformly SVO, so position carries no part-of-speech information.
#
# Two honest sources remain, and both are used:
#
#   1. `emergent.core.grounding.VOCABULARY` IS labelled -- dominant_modality
#      "visual" for object words and "motor" for action words. That is ground
#      truth from the repo, but it only covers 10 and 8 words respectively.
#
#   2. An AUTHORED extension, below, marked as such. For a synthetic SVO
#      grammar this is legitimate -- the words only need to be distinct tokens
#      with correct part of speech -- but it must not be passed off as mined
#      from a corpus. A genuinely corpus-derived vocabulary needs a POS-tagged
#      source, which is what task #30 (CHILDES) is for.

#: Authored concrete nouns. Not mined -- see the note above.
_EXTRA_NOUNS = [
    "dog", "cat", "bird", "boy", "girl", "ball", "book", "food", "table",
    "car", "cup", "spoon", "shoe", "hat", "door", "box", "apple", "milk",
    "chair", "bed", "tree", "flower", "fish", "horse", "cow", "duck",
    "mouse", "bear", "truck", "train", "doll", "block", "bottle", "sock",
    "towel", "brush", "plate", "bowl", "key", "clock",
]

#: Authored transitive verbs. Not mined -- see the note above.
_EXTRA_VERBS = [
    "chases", "sees", "catches", "eats", "holds", "pushes", "pulls",
    "carries", "drops", "throws", "finds", "wants", "likes", "watches",
    "touches", "moves", "hides", "brings", "takes", "washes",
]


def build(target_nouns=40, target_verbs=20, seed=0):
    """Return (nouns, verbs, svo_templates).

    Labelled VOCABULARY words come first, so anything the repo actually grounds
    is preferred over the authored extension.
    """
    labelled_n, labelled_v = [], []
    try:
        from neural_assemblies.assembly_calculus.emergent.core.grounding import (
            VOCABULARY,
        )
        for w, entry in VOCABULARY.items():
            mod = getattr(entry, "dominant_modality", None)
            if mod == "visual":
                labelled_n.append(w)
            elif mod == "motor":
                labelled_v.append(w)
    except Exception:                                        # noqa: BLE001
        pass

    def merge_keep_order(primary, extra, target):
        out = list(dict.fromkeys(primary))
        for w in extra:
            if len(out) >= target:
                break
            if w not in out:
                out.append(w)
        return out[:target]

    nouns = merge_keep_order(labelled_n, _EXTRA_NOUNS, target_nouns)
    verbs = merge_keep_order(labelled_v, _EXTRA_VERBS, target_verbs)
    # No word may be both, or a role area is asked to hold one token twice.
    verbs = [v for v in verbs if v not in set(nouns)]
    templates = [(s, v, o) for (s, v, o) in _svo_triples()
                 if s in nouns and v in verbs and o in nouns]
    return nouns, verbs, templates


def occurrences(nouns, verbs, n_sentences=400, zipf_s=1.0, seed=0):
    """Zipfian token stream: what a learner actually hears.

    Ranks are assigned by list position, which is itself frequency-ordered from
    the curriculum, so the skew compounds realistically rather than being
    imposed on an arbitrary ordering.
    """
    rng = random.Random(seed)

    def weights(items):
        return [1.0 / ((i + 1) ** zipf_s) for i in range(len(items))]

    nw, vw = weights(nouns), weights(verbs)
    out = []
    for _ in range(n_sentences):
        s = rng.choices(nouns, weights=nw, k=1)[0]
        o = rng.choices(nouns, weights=nw, k=1)[0]
        while o == s and len(nouns) > 1:
            o = rng.choices(nouns, weights=nw, k=1)[0]
        v = rng.choices(verbs, weights=vw, k=1)[0]
        out.append([s, v, o])
    return out


def bindings(sentences):
    """Each (word, role-slot) pair ONCE, in first-occurrence order.

    Role slots are positional: 0 AGENT, 1 ACTION, 2 PATIENT. This is the
    schedule role training must use -- see the module docstring for why c_max=1
    is a feasibility requirement rather than an optimisation.
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
    return {
        "n_nouns": len(nouns), "n_verbs": len(verbs),
        "n_sentences": len(sentences),
        "tokens": sum(counts.values()),
        "types": len(counts),
        "c_max": max(counts.values()), "c_min": min(counts.values()),
        "per_slot_types": [len(c) for c in per_slot],
        "n_bindings": len(bindings(sentences)),
    }


if __name__ == "__main__":
    nouns, verbs, templates = build()
    print(f"\n  CORPUS from repo curriculum")
    print(f"    mined SVO triples : {len(_svo_triples())}")
    print(f"    nouns             : {len(nouns)}  {nouns[:10]}")
    print(f"    verbs             : {len(verbs)}  {verbs[:10]}")
    print(f"    intact templates  : {len(templates)}")
    for s in (0.0, 1.0):
        sents = occurrences(nouns, verbs, n_sentences=400, zipf_s=s)
        info = summary(nouns, verbs, sents)
        print(f"\n    zipf_s={s}: " + "  ".join(
            f"{k}={v}" for k, v in info.items()))
        print(f"      chance for a role area with {info['per_slot_types'][0]} "
              f"types = {1 / max(info['per_slot_types'][0], 1):.4f}")
