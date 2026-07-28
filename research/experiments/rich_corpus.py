"""A corpus big enough for distributional induction to have something to chew.

WHAT THE OLD CORPUS COULD NOT SUPPORT, measured in `category_induction.py`
--------------------------------------------------------------------------
36 types, 765 tokens, balanced by design. Consequences, all measured:

* Intransitive verbs were unlearnable. Each occurred ~5 times and always with a
  DIFFERENT subject, so "the dog runs" / "a bird plays" / "the cat sleeps" give
  frames (dog,#), (bird,#), (cat,#) that share NOTHING, and each verb became a
  2%-coverage singleton. No clustering rule recovers a class from five
  non-overlapping examples.
* Closed-class anchoring was untestable. `the` is rank 9 of 36; the
  198-sentence lesion sub-corpus has no determiners at all. Frequency-ranking
  nominates `bird`, `boy`, `girl` as the "function words".
* S-vs-O without semantic bias was untestable: nearly every noun is strongly
  role-biased by construction.

DESIGN, each choice answering one of those
-------------------------------------------
* ZIPFIAN frequencies, so function words actually dominate the token
  distribution as they do in real language. This is what makes "closed class"
  detectable by frequency at all.
* Every content word appears with MANY different partners, so frames REPEAT.
  That is the specific fix for the intransitive failure: `runs` needs to occur
  with many subjects AND those subjects need to recur, or no frame is shared.
* A mix of role-BIASED nouns (animals act, objects are acted on) and role-
  BALANCED ones (people do both), so the S-vs-O question can be asked of a
  lexicon where agreement has nothing to work with.
* Both transitive and intransitive verbs, in quantity.
* Full grounding: per-word `GroundingContext` plus a `SceneEvent` carrying
  perceived participants in CAUSAL order, so roles can be derived from
  perception rather than annotation (`core/scene.py`).
* Generable in ANY of the six word orders from the same underlying events,
  which is what makes word-order induction testable rather than assumed.

WHAT THIS IS NOT. It is still synthetic, and synthetic corpora can only falsify,
not confirm: if induction fails here it certainly fails on real speech, but
succeeding here does not establish it works on CHILDES. The generator is
deliberately regular, and real language is not. Task #30 remains the real test.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

#: (word, features). Animals act, objects are acted upon, people do both --
#: the third group is what makes the no-lexical-bias case testable.
ANIMATE = [("dog", ["DOG", "ANIMAL"]), ("cat", ["CAT", "ANIMAL"]),
           ("bird", ["BIRD", "ANIMAL"]), ("horse", ["HORSE", "ANIMAL"]),
           ("mouse", ["MOUSE", "ANIMAL"]), ("fox", ["FOX", "ANIMAL"])]
INANIMATE = [("ball", ["BALL", "OBJECT"]), ("book", ["BOOK", "OBJECT"]),
             ("food", ["FOOD", "OBJECT"]), ("table", ["TABLE", "OBJECT"]),
             ("car", ["CAR", "OBJECT"]), ("cup", ["CUP", "OBJECT"]),
             ("stick", ["STICK", "OBJECT"]), ("box", ["BOX", "OBJECT"])]
BALANCED = [("boy", ["BOY", "PERSON"]), ("girl", ["GIRL", "PERSON"]),
            ("man", ["MAN", "PERSON"]), ("woman", ["WOMAN", "PERSON"]),
            ("child", ["CHILD", "PERSON"])]

TRANSITIVE = [("chases", ["CHASING", "PURSUIT"]), ("finds", ["FINDING"]),
              ("sees", ["SEEING"]), ("eats", ["EATING"]),
              ("holds", ["HOLDING"]), ("pushes", ["PUSHING"]),
              ("carries", ["CARRYING"]), ("wants", ["WANTING"])]
INTRANSITIVE = [("runs", ["RUNNING"]), ("sleeps", ["SLEEPING"]),
                ("plays", ["PLAYING"]), ("jumps", ["JUMPING"]),
                ("sits", ["SITTING"]), ("walks", ["WALKING"])]

DETERMINERS = ["the", "a"]
ADJECTIVES = [("big", ["BIG"]), ("small", ["SMALL"]),
              ("red", ["RED"]), ("fast", ["FAST"])]

#: Where each constituent goes, per order. "S"/"V"/"O" as in the name.
_ORDER_SLOTS = {"SVO": "SVO", "SOV": "SOV", "VSO": "VSO",
                "VOS": "VOS", "OSV": "OSV", "OVS": "OVS"}


def _zipf_weights(n: int, exponent: float = 1.0) -> List[float]:
    """Rank-frequency weights. Real lexicons are Zipfian and this one must be
    too, or 'closed class' is not detectable by frequency."""
    return [1.0 / ((i + 1) ** exponent) for i in range(n)]


def _noun_phrase(word: str, rng: random.Random, *,
                 determiner_rate: float, adjective_rate: float
                 ) -> Tuple[List[str], List[Optional[str]]]:
    """Build a noun phrase; returns (tokens, per-token adjective features).

    Determiners are frequent because that is the point -- they are the frames
    that let nouns be recognised as nouns without any semantics.
    """
    tokens: List[str] = []
    extra: List[Optional[str]] = []
    if rng.random() < determiner_rate:
        tokens.append(rng.choice(DETERMINERS))
        extra.append(None)
    if rng.random() < adjective_rate:
        adj, feats = rng.choice(ADJECTIVES)
        tokens.append(adj)
        extra.append(feats[0])
    tokens.append(word)
    extra.append(None)
    return tokens, extra


def generate(n_sentences: int = 4000, *, seed: int = 0,
             word_order: str = "SVO", determiner_rate: float = 0.8,
             adjective_rate: float = 0.25,
             intransitive_rate: float = 0.35) -> List:
    """A grounded corpus in `word_order`, with scenes attached.

    `intransitive_rate` is high on purpose. Intransitives were the class the
    old corpus could not teach, and the fix is not a cleverer learner but
    enough occurrences for frames to repeat.
    """
    from neural_assemblies.assembly_calculus.emergent.core.grounding import (
        GroundingContext,
    )
    from neural_assemblies.assembly_calculus.emergent.core.scene import SceneEvent
    from neural_assemblies.assembly_calculus.emergent.core.sentence import (
        GroundedSentence,
    )

    rng = random.Random(seed)
    slots = _ORDER_SLOTS[word_order]
    agents = ANIMATE + BALANCED
    patients = INANIMATE + BALANCED
    a_w = _zipf_weights(len(agents))
    p_w = _zipf_weights(len(patients))
    t_w = _zipf_weights(len(TRANSITIVE))
    i_w = _zipf_weights(len(INTRANSITIVE))

    corpus = []
    for _ in range(n_sentences):
        if rng.random() < intransitive_rate:
            (subj, subj_f), = rng.choices(agents, weights=a_w, k=1)
            (verb, verb_f), = rng.choices(INTRANSITIVE, weights=i_w, k=1)
            np_tokens, np_extra = _noun_phrase(
                subj, rng, determiner_rate=determiner_rate,
                adjective_rate=adjective_rate)
            # Intransitives have no object, so order reduces to whether the
            # verb precedes or follows its subject.
            verb_first = slots.index("V") < slots.index("S")
            words = ([verb] + np_tokens) if verb_first else (np_tokens + [verb])
            contexts, roles = [], []
            for tok in words:
                if tok == verb:
                    contexts.append(GroundingContext(motor=list(verb_f)))
                    roles.append("action")
                elif tok == subj:
                    contexts.append(GroundingContext(visual=list(subj_f)))
                    roles.append("agent")
                else:
                    idx = np_tokens.index(tok) if tok in np_tokens else -1
                    feat = np_extra[idx] if 0 <= idx < len(np_extra) else None
                    contexts.append(GroundingContext(properties=[feat])
                                    if feat else GroundingContext())
                    roles.append(None)
            sentence = GroundedSentence(words=words, contexts=contexts,
                                        roles=roles)
            sentence.event = SceneEvent(action=list(verb_f),
                                        participants=[list(subj_f)])
            corpus.append(sentence)
            continue

        (subj, subj_f), = rng.choices(agents, weights=a_w, k=1)
        (obj, obj_f), = rng.choices(patients, weights=p_w, k=1)
        if obj == subj:
            continue
        (verb, verb_f), = rng.choices(TRANSITIVE, weights=t_w, k=1)

        s_tokens, s_extra = _noun_phrase(
            subj, rng, determiner_rate=determiner_rate,
            adjective_rate=adjective_rate)
        o_tokens, o_extra = _noun_phrase(
            obj, rng, determiner_rate=determiner_rate,
            adjective_rate=adjective_rate)
        piece = {"S": s_tokens, "V": [verb], "O": o_tokens}
        extra = {"S": s_extra, "V": [None], "O": o_extra}

        words, contexts, roles = [], [], []
        for slot in slots:
            for tok, feat in zip(piece[slot], extra[slot]):
                words.append(tok)
                if slot == "V":
                    contexts.append(GroundingContext(motor=list(verb_f)))
                    roles.append("action")
                elif tok == subj:
                    contexts.append(GroundingContext(visual=list(subj_f)))
                    roles.append("agent")
                elif tok == obj:
                    contexts.append(GroundingContext(visual=list(obj_f)))
                    roles.append("patient")
                else:
                    contexts.append(GroundingContext(properties=[feat])
                                    if feat else GroundingContext())
                    roles.append(None)

        sentence = GroundedSentence(words=words, contexts=contexts, roles=roles)
        # Participants in CAUSAL order -- actor first -- independent of the
        # linguistic order above. That independence is what makes word-order
        # induction from perceived structure non-circular.
        sentence.event = SceneEvent(action=list(verb_f),
                                    participants=[list(subj_f), list(obj_f)])
        corpus.append(sentence)
    return corpus


def summarize(corpus: Sequence) -> Dict[str, object]:
    import collections
    tok = collections.Counter()
    for s in corpus:
        tok.update(s.words)
    total = sum(tok.values())
    return {
        "sentences": len(corpus),
        "types": len(tok),
        "tokens": total,
        "top10": tok.most_common(10),
        "hapax": sum(1 for c in tok.values() if c == 1),
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    corpus = generate(4000, seed=0)
    info = summarize(corpus)
    print(f"\n  sentences {info['sentences']}  types {info['types']}  "
          f"tokens {info['tokens']}  hapax {info['hapax']}")
    print(f"  top 10 by frequency (function words should lead):")
    for w, c in info["top10"]:
        print(f"    {w:<10}{c:>6}")
    print("\n  samples:")
    for s in corpus[:5]:
        print(f"    {' '.join(s.words):<34} roles={s.roles}")
    from neural_assemblies.assembly_calculus.emergent.core.scene import (
        roles_from_scene,
    )
    ok = sum(1 for s in corpus
             if roles_from_scene(s) == list(s.roles))
    print(f"\n  roles derivable from perception alone: {ok}/{len(corpus)}")
