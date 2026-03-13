"""
Grounded training sentences for the emergent NEMO parser.

Each sentence provides:
- words: the token sequence
- contexts: GroundingContext per word (what the learner experiences)
- roles: thematic role annotations ("agent", "action", "patient", None)
- mood: sentence mood (currently always "declarative")

The training set covers all structural patterns:
intransitive, transitive, adjective+noun, prepositional, pronoun subjects,
adverbs, and varied verb/noun combinations.
"""

import random as _random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .grounding import GroundingContext, VOCABULARY


@dataclass
class GroundedSentence:
    """A sentence with grounding context for each word."""
    words: List[str]
    contexts: List[GroundingContext]
    roles: List[Optional[str]] = None
    mood: str = "declarative"

    def __post_init__(self):
        if self.roles is None:
            self.roles = [None] * len(self.words)
        assert len(self.words) == len(self.contexts), (
            f"words ({len(self.words)}) != contexts ({len(self.contexts)})"
        )
        assert len(self.words) == len(self.roles), (
            f"words ({len(self.words)}) != roles ({len(self.roles)})"
        )


def _ctx(word: str) -> GroundingContext:
    """Look up grounding context for a vocabulary word."""
    return VOCABULARY[word]


def create_training_sentences() -> List[GroundedSentence]:
    """Create ~30 grounded training sentences covering all structural patterns."""
    data = []

    # ---- Intransitive: DET NOUN VERB ----
    data.append(GroundedSentence(
        words=["the", "dog", "runs"],
        contexts=[_ctx("the"), _ctx("dog"), _ctx("runs")],
        roles=[None, "agent", "action"],
    ))
    data.append(GroundedSentence(
        words=["the", "cat", "sleeps"],
        contexts=[_ctx("the"), _ctx("cat"), _ctx("sleeps")],
        roles=[None, "agent", "action"],
    ))
    data.append(GroundedSentence(
        words=["a", "bird", "plays"],
        contexts=[_ctx("a"), _ctx("bird"), _ctx("plays")],
        roles=[None, "agent", "action"],
    ))
    data.append(GroundedSentence(
        words=["the", "boy", "reads"],
        contexts=[_ctx("the"), _ctx("boy"), _ctx("reads")],
        roles=[None, "agent", "action"],
    ))
    data.append(GroundedSentence(
        words=["the", "girl", "plays"],
        contexts=[_ctx("the"), _ctx("girl"), _ctx("plays")],
        roles=[None, "agent", "action"],
    ))

    # ---- Transitive: DET NOUN VERB DET NOUN ----
    data.append(GroundedSentence(
        words=["the", "cat", "chases", "the", "bird"],
        contexts=[_ctx("the"), _ctx("cat"), _ctx("chases"),
                  _ctx("the"), _ctx("bird")],
        roles=[None, "agent", "action", None, "patient"],
    ))
    data.append(GroundedSentence(
        words=["the", "dog", "sees", "a", "cat"],
        contexts=[_ctx("the"), _ctx("dog"), _ctx("sees"),
                  _ctx("a"), _ctx("cat")],
        roles=[None, "agent", "action", None, "patient"],
    ))
    data.append(GroundedSentence(
        words=["the", "boy", "finds", "the", "ball"],
        contexts=[_ctx("the"), _ctx("boy"), _ctx("finds"),
                  _ctx("the"), _ctx("ball")],
        roles=[None, "agent", "action", None, "patient"],
    ))
    data.append(GroundedSentence(
        words=["a", "girl", "reads", "a", "book"],
        contexts=[_ctx("a"), _ctx("girl"), _ctx("reads"),
                  _ctx("a"), _ctx("book")],
        roles=[None, "agent", "action", None, "patient"],
    ))
    data.append(GroundedSentence(
        words=["the", "bird", "eats", "the", "food"],
        contexts=[_ctx("the"), _ctx("bird"), _ctx("eats"),
                  _ctx("the"), _ctx("food")],
        roles=[None, "agent", "action", None, "patient"],
    ))
    data.append(GroundedSentence(
        words=["the", "dog", "chases", "the", "cat"],
        contexts=[_ctx("the"), _ctx("dog"), _ctx("chases"),
                  _ctx("the"), _ctx("cat")],
        roles=[None, "agent", "action", None, "patient"],
    ))
    data.append(GroundedSentence(
        words=["the", "cat", "finds", "the", "food"],
        contexts=[_ctx("the"), _ctx("cat"), _ctx("finds"),
                  _ctx("the"), _ctx("food")],
        roles=[None, "agent", "action", None, "patient"],
    ))

    # ---- With adjectives: DET ADJ NOUN VERB ----
    data.append(GroundedSentence(
        words=["a", "big", "cat", "sleeps"],
        contexts=[_ctx("a"), _ctx("big"), _ctx("cat"), _ctx("sleeps")],
        roles=[None, None, "agent", "action"],
    ))
    data.append(GroundedSentence(
        words=["the", "small", "dog", "runs"],
        contexts=[_ctx("the"), _ctx("small"), _ctx("dog"), _ctx("runs")],
        roles=[None, None, "agent", "action"],
    ))
    data.append(GroundedSentence(
        words=["the", "fast", "bird", "plays"],
        contexts=[_ctx("the"), _ctx("fast"), _ctx("bird"), _ctx("plays")],
        roles=[None, None, "agent", "action"],
    ))
    data.append(GroundedSentence(
        words=["a", "red", "car", "runs"],
        contexts=[_ctx("a"), _ctx("red"), _ctx("car"), _ctx("runs")],
        roles=[None, None, "agent", "action"],
    ))

    # ---- With adjectives + transitive: DET ADJ NOUN VERB DET NOUN ----
    data.append(GroundedSentence(
        words=["the", "big", "dog", "chases", "a", "cat"],
        contexts=[_ctx("the"), _ctx("big"), _ctx("dog"), _ctx("chases"),
                  _ctx("a"), _ctx("cat")],
        roles=[None, None, "agent", "action", None, "patient"],
    ))
    data.append(GroundedSentence(
        words=["a", "small", "cat", "sees", "the", "bird"],
        contexts=[_ctx("a"), _ctx("small"), _ctx("cat"), _ctx("sees"),
                  _ctx("the"), _ctx("bird")],
        roles=[None, None, "agent", "action", None, "patient"],
    ))

    # ---- With prepositions: DET NOUN VERB PREP DET NOUN ----
    data.append(GroundedSentence(
        words=["the", "cat", "sleeps", "on", "the", "table"],
        contexts=[_ctx("the"), _ctx("cat"), _ctx("sleeps"),
                  _ctx("on"), _ctx("the"), _ctx("table")],
        roles=[None, "agent", "action", None, None, None],
    ))
    data.append(GroundedSentence(
        words=["the", "dog", "plays", "in", "the", "car"],
        contexts=[_ctx("the"), _ctx("dog"), _ctx("plays"),
                  _ctx("in"), _ctx("the"), _ctx("car")],
        roles=[None, "agent", "action", None, None, None],
    ))
    data.append(GroundedSentence(
        words=["the", "ball", "runs", "near", "the", "table"],
        contexts=[_ctx("the"), _ctx("ball"), _ctx("runs"),
                  _ctx("near"), _ctx("the"), _ctx("table")],
        roles=[None, "agent", "action", None, None, None],
    ))

    # ---- With pronouns: PRON VERB DET NOUN ----
    data.append(GroundedSentence(
        words=["he", "sees", "the", "bird"],
        contexts=[_ctx("he"), _ctx("sees"), _ctx("the"), _ctx("bird")],
        roles=["agent", "action", None, "patient"],
    ))
    data.append(GroundedSentence(
        words=["she", "finds", "the", "book"],
        contexts=[_ctx("she"), _ctx("finds"), _ctx("the"), _ctx("book")],
        roles=["agent", "action", None, "patient"],
    ))
    data.append(GroundedSentence(
        words=["it", "eats", "the", "food"],
        contexts=[_ctx("it"), _ctx("eats"), _ctx("the"), _ctx("food")],
        roles=["agent", "action", None, "patient"],
    ))
    data.append(GroundedSentence(
        words=["they", "chases", "the", "dog"],
        contexts=[_ctx("they"), _ctx("chases"), _ctx("the"), _ctx("dog")],
        roles=["agent", "action", None, "patient"],
    ))

    # ---- With adverbs: DET NOUN VERB ADV ----
    data.append(GroundedSentence(
        words=["the", "dog", "runs", "quickly"],
        contexts=[_ctx("the"), _ctx("dog"), _ctx("runs"), _ctx("quickly")],
        roles=[None, "agent", "action", None],
    ))
    data.append(GroundedSentence(
        words=["the", "cat", "eats", "slowly"],
        contexts=[_ctx("the"), _ctx("cat"), _ctx("eats"), _ctx("slowly")],
        roles=[None, "agent", "action", None],
    ))

    # ---- Mixed patterns ----
    data.append(GroundedSentence(
        words=["a", "happy", "girl", "sees", "the", "dog"],
        contexts=[_ctx("a"), _ctx("happy"), _ctx("girl"), _ctx("sees"),
                  _ctx("the"), _ctx("dog")],
        roles=[None, None, "agent", "action", None, "patient"],
    ))
    data.append(GroundedSentence(
        words=["the", "boy", "eats", "food", "quickly"],
        contexts=[_ctx("the"), _ctx("boy"), _ctx("eats"),
                  _ctx("food"), _ctx("quickly")],
        roles=[None, "agent", "action", "patient", None],
    ))
    data.append(GroundedSentence(
        words=["he", "chases", "the", "fast", "cat"],
        contexts=[_ctx("he"), _ctx("chases"),
                  _ctx("the"), _ctx("fast"), _ctx("cat")],
        roles=["agent", "action", None, None, "patient"],
    ))

    # ---- Passive voice: DET NOUN was VERB by DET NOUN ----
    data.append(GroundedSentence(
        words=["the", "cat", "was", "chases", "by", "the", "dog"],
        contexts=[_ctx("the"), _ctx("cat"), _ctx("was"), _ctx("chases"),
                  _ctx("by"), _ctx("the"), _ctx("dog")],
        roles=[None, "patient", None, "action", None, None, "agent"],
    ))
    data.append(GroundedSentence(
        words=["the", "bird", "was", "sees", "by", "the", "cat"],
        contexts=[_ctx("the"), _ctx("bird"), _ctx("was"), _ctx("sees"),
                  _ctx("by"), _ctx("the"), _ctx("cat")],
        roles=[None, "patient", None, "action", None, None, "agent"],
    ))
    data.append(GroundedSentence(
        words=["the", "ball", "was", "finds", "by", "the", "boy"],
        contexts=[_ctx("the"), _ctx("ball"), _ctx("was"), _ctx("finds"),
                  _ctx("by"), _ctx("the"), _ctx("boy")],
        roles=[None, "patient", None, "action", None, None, "agent"],
    ))
    data.append(GroundedSentence(
        words=["the", "book", "was", "reads", "by", "the", "girl"],
        contexts=[_ctx("the"), _ctx("book"), _ctx("was"), _ctx("reads"),
                  _ctx("by"), _ctx("the"), _ctx("girl")],
        roles=[None, "patient", None, "action", None, None, "agent"],
    ))
    data.append(GroundedSentence(
        words=["the", "food", "was", "eats", "by", "the", "bird"],
        contexts=[_ctx("the"), _ctx("food"), _ctx("was"), _ctx("eats"),
                  _ctx("by"), _ctx("the"), _ctx("bird")],
        roles=[None, "patient", None, "action", None, None, "agent"],
    ))

    return data


def generate_training_sentences(
    vocab: Dict[str, GroundingContext],
    n_sentences: int = 100,
    seed: int = 42,
) -> List[GroundedSentence]:
    """Generate training sentences for a scaled vocabulary.

    Uses template patterns (intransitive, transitive, adj+noun, etc.) to
    create sentences that cover every word at least twice.  Role annotations
    are automatically assigned from the template structure.

    Args:
        vocab: Word → GroundingContext mapping.
        n_sentences: Target number of sentences (may produce more to ensure
            coverage).
        seed: Random seed for reproducibility.

    Returns:
        List of GroundedSentence objects.
    """
    rng = _random.Random(seed)

    # Categorize words by dominant modality → POS
    nouns = [w for w, c in vocab.items() if c.dominant_modality == "visual"]
    verbs = [w for w, c in vocab.items() if c.dominant_modality == "motor"]
    adjs = [w for w, c in vocab.items() if c.dominant_modality == "properties"]
    preps = [w for w, c in vocab.items() if c.dominant_modality == "spatial"]
    prons = [w for w, c in vocab.items() if c.dominant_modality == "social"]
    advs = [w for w, c in vocab.items() if c.dominant_modality == "temporal"]
    dets = [w for w, c in vocab.items() if c.dominant_modality == "none"]

    # Ensure we have at least basic categories
    if not dets:
        dets = ["the"]
    if not verbs:
        return []

    def _pick(lst: list) -> str:
        return rng.choice(lst) if lst else ""

    data: List[GroundedSentence] = []
    word_count: Dict[str, int] = {w: 0 for w in vocab}

    def _add(words, roles):
        contexts = [vocab.get(w, GroundingContext()) for w in words]
        data.append(GroundedSentence(
            words=words, contexts=contexts, roles=roles,
        ))
        for w in words:
            if w in word_count:
                word_count[w] += 1

    # Template patterns
    templates = []

    # 1. DET NOUN VERB (intransitive)
    if nouns and verbs:
        templates.append("intransitive")
    # 2. DET NOUN VERB DET NOUN (transitive)
    if len(nouns) >= 2 and verbs:
        templates.append("transitive")
    # 3. DET ADJ NOUN VERB (adjective)
    if nouns and verbs and adjs:
        templates.append("adj_intransitive")
    # 4. DET NOUN VERB PREP DET NOUN (prepositional)
    if len(nouns) >= 2 and verbs and preps:
        templates.append("prepositional")
    # 5. PRON VERB DET NOUN (pronoun subject)
    if prons and verbs and nouns:
        templates.append("pronoun_transitive")
    # 6. DET NOUN VERB ADV (adverb)
    if nouns and verbs and advs:
        templates.append("adverb")
    # 7. DET NOUN was VERB by DET NOUN (passive)
    if len(nouns) >= 2 and verbs:
        templates.append("passive")

    if not templates:
        return data

    # Generate sentences
    for i in range(n_sentences):
        template = templates[i % len(templates)]

        if template == "intransitive":
            d, n, v = _pick(dets), _pick(nouns), _pick(verbs)
            _add([d, n, v], [None, "agent", "action"])

        elif template == "transitive":
            d1, n1 = _pick(dets), _pick(nouns)
            v = _pick(verbs)
            d2, n2 = _pick(dets), _pick(nouns)
            # Avoid same noun as subject and object
            for _ in range(5):
                if n2 != n1:
                    break
                n2 = _pick(nouns)
            _add([d1, n1, v, d2, n2],
                 [None, "agent", "action", None, "patient"])

        elif template == "adj_intransitive":
            d, a, n, v = _pick(dets), _pick(adjs), _pick(nouns), _pick(verbs)
            _add([d, a, n, v], [None, None, "agent", "action"])

        elif template == "prepositional":
            d1, n1, v = _pick(dets), _pick(nouns), _pick(verbs)
            p = _pick(preps)
            d2, n2 = _pick(dets), _pick(nouns)
            _add([d1, n1, v, p, d2, n2],
                 [None, "agent", "action", None, None, None])

        elif template == "pronoun_transitive":
            pr, v = _pick(prons), _pick(verbs)
            d, n = _pick(dets), _pick(nouns)
            _add([pr, v, d, n],
                 ["agent", "action", None, "patient"])

        elif template == "adverb":
            d, n, v = _pick(dets), _pick(nouns), _pick(verbs)
            av = _pick(advs)
            _add([d, n, v, av],
                 [None, "agent", "action", None])

        elif template == "passive":
            d1, n1 = _pick(dets), _pick(nouns)
            v = _pick(verbs)
            d2, n2 = _pick(dets), _pick(nouns)
            for _ in range(5):
                if n2 != n1:
                    break
                n2 = _pick(nouns)
            _add([d1, n1, "was", v, "by", d2, n2],
                 [None, "patient", None, "action", None, None, "agent"])

    # Ensure every word appears at least twice
    underrepresented = [w for w, c in word_count.items() if c < 2]
    for w in underrepresented:
        ctx = vocab[w]
        mod = ctx.dominant_modality

        if mod == "visual":
            d, v = _pick(dets), _pick(verbs)
            _add([d, w, v], [None, "agent", "action"])
        elif mod == "motor":
            d, n = _pick(dets), _pick(nouns)
            _add([d, n, w], [None, "agent", "action"])
        elif mod == "properties":
            d, n, v = _pick(dets), _pick(nouns), _pick(verbs)
            _add([d, w, n, v], [None, None, "agent", "action"])
        elif mod == "spatial":
            d1, n1, v = _pick(dets), _pick(nouns), _pick(verbs)
            d2, n2 = _pick(dets), _pick(nouns)
            _add([d1, n1, v, w, d2, n2],
                 [None, "agent", "action", None, None, None])
        elif mod == "social":
            v, d, n = _pick(verbs), _pick(dets), _pick(nouns)
            _add([w, v, d, n], ["agent", "action", None, "patient"])
        elif mod == "temporal":
            d, n, v = _pick(dets), _pick(nouns), _pick(verbs)
            _add([d, n, v, w], [None, "agent", "action", None])
        else:
            # Function word (det/conj) — use in noun phrase
            n, v = _pick(nouns), _pick(verbs)
            _add([w, n, v], [None, "agent", "action"])

    return data
