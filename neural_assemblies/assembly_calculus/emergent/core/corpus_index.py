"""Corpus compiler: one-pass indexing for emergent training phases.

Compiles ``GroundedSentence`` corpora into shared indices (categories,
roles, weighted bridge transitions, sizing stats) so every training phase
reads precomputed data instead of re-walking raw sentences.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from .areas import (
    CORE_TO_CATEGORY, GROUNDING_TO_CORE, ROLE_ACTION, ROLE_AGENT, ROLE_PATIENT,
    ROLE_LABEL_TO_AREA,
)
from .grounding import GroundingContext
from .sentence import GroundedSentence

if TYPE_CHECKING:
    from ..parser_mixins.core import CoreParserMixin


@dataclass(frozen=True)
class SentenceIndex:
    """Per-sentence compiled features."""
    words: Tuple[str, ...]
    categories: Tuple[str, ...]
    verb_pos: Optional[int]
    noun_roles: Tuple[Tuple[int, str, str], ...]  # (idx, word, ROLE_*)


@dataclass(frozen=True)
class BridgeTransition:
    """Unique prefix → next-word bridge with corpus frequency."""
    prefix_words: Tuple[str, ...]
    next_word: str
    category_signature: Tuple[str, ...]
    count: int


@dataclass(frozen=True)
class RoleUpdate:
    """Single unsupervised role projection (word, role area)."""
    word: str
    role_area: str
    count: int


@dataclass
class CorpusIndex:
    """Compiled view of a training corpus."""
    grounded: List[GroundedSentence]
    raw: List[List[str]]
    sentences: List[SentenceIndex]
    corpus_vocab: Set[str]
    max_sentence_length: int
    word_freq: Dict[str, int]
    transitions: List[BridgeTransition]
    role_updates: List[RoleUpdate]
    content_words: Set[str]

    @property
    def grounded_for_roles(self) -> List[GroundedSentence]:
        return self.grounded

    @property
    def bridge_vocab(self) -> Set[str]:
        """Words that appear in prefix or next position of any bridge."""
        words: Set[str] = set()
        for trans in self.transitions:
            words.update(trans.prefix_words)
            words.add(trans.next_word)
        return words


class TransitionCache:
    """Cumulative bridge-transition store across curriculum stages."""

    def __init__(self) -> None:
        self._counts: Dict[Tuple[Tuple[str, ...], str], int] = defaultdict(int)
        self._sigs: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
        self._trained: Set[Tuple[Tuple[str, ...], str]] = set()

    def absorb(self, index: CorpusIndex) -> None:
        for trans in index.transitions:
            key = (trans.prefix_words, trans.next_word)
            self._counts[key] += trans.count
            self._sigs[trans.prefix_words] = trans.category_signature

    def overlay(self, base: CorpusIndex) -> CorpusIndex:
        transitions = [
            BridgeTransition(
                prefix_words=prefix,
                next_word=nxt,
                category_signature=self._sigs[prefix],
                count=count,
            )
            for (prefix, nxt), count in self._counts.items()
        ]
        transitions.sort(key=lambda t: t.count, reverse=True)
        base.transitions = transitions
        return base

    def merge_indices(self, *indices: CorpusIndex) -> CorpusIndex:
        for idx in indices:
            self.absorb(idx)
        merged = merge_corpus_indices(*indices)
        return self.overlay(merged)

    def filter_untrained(
        self,
        transitions: List[BridgeTransition],
    ) -> List[BridgeTransition]:
        """Return transitions not yet trained across curriculum stages."""
        return [
            t for t in transitions
            if (t.prefix_words, t.next_word) not in self._trained
        ]

    def mark_trained(self, transitions: List[BridgeTransition]) -> None:
        """Record bridges trained in the current pass."""
        for t in transitions:
            self._trained.add((t.prefix_words, t.next_word))


def category_oracle(
    parser: "CoreParserMixin",
    word: str,
    grounding: Optional[GroundingContext] = None,
) -> str:
    """O(1) category when grounding or distributional stats suffice."""
    ctx = grounding if grounding is not None else parser.word_grounding.get(word)
    if ctx is not None and ctx.is_grounded:
        from ..acquisition.pos_inference import classify_word_bootstrapped, is_word_in_lexicon

        if not is_word_in_lexicon(parser, word):
            cat, _ = classify_word_bootstrapped(parser, word, ctx)
            if cat != "UNKNOWN":
                return cat
        core = GROUNDING_TO_CORE[ctx.dominant_modality]
        return CORE_TO_CATEGORY[core]

    if hasattr(parser, "_dist_categories") and word in parser._dist_categories:
        return parser._dist_categories[word]

    if parser.dist_stats.word_count.get(word, 0) > 0:
        cat, _ = parser.classify_distributional(word)
        if cat != "UNKNOWN":
            return cat

    cat, _ = parser.classify_word_cached(word, grounding=grounding)
    return cat


def _assign_noun_roles(
    words: List[str],
    categories: List[str],
    verb_pos: int,
    order: str,
) -> List[Tuple[int, str, str]]:
    noun_positions = [
        (idx, word) for idx, (word, cat) in enumerate(zip(words, categories))
        if cat in ("NOUN", "PRON")
    ]
    # Map the nouns onto the noun slots of the typology, split at the verb.
    # Nouns before the verb take the noun slots that precede V in the order,
    # nouns after it take those that follow. For "OVS" the pre-verb slot is O,
    # so the first noun is the PATIENT -- which the old SVO/SOV/VSO ladder had
    # no way to express. For SVO this reproduces the old idx-vs-verb_pos rule
    # exactly, including verb-initial fragments like an imperative.
    from .word_order import WORD_ORDERS, noun_slots, order_slots

    label = order if order in WORD_ORDERS else "SVO"
    slots = order_slots(label)
    v_at = slots.index("V")
    all_noun_slots = noun_slots(label)
    pre_slots = tuple(s for i, s in enumerate(slots) if s != "V" and i < v_at)
    post_slots = tuple(s for i, s in enumerate(slots) if s != "V" and i > v_at)
    slot_role = {"S": ROLE_AGENT, "O": ROLE_PATIENT}

    def pick(seq: Tuple[str, ...], rank: int) -> str:
        if not seq:
            seq = all_noun_slots
        return seq[rank] if rank < len(seq) else seq[-1]

    out: List[Tuple[int, str, str]] = []
    pre_rank = post_rank = 0
    for rank, (idx, word) in enumerate(noun_positions):
        if verb_pos is None or verb_pos < 0:
            slot = pick(all_noun_slots, rank)
        elif idx < verb_pos:
            slot = pick(pre_slots, pre_rank)
            pre_rank += 1
        else:
            slot = pick(post_slots, post_rank)
            post_rank += 1
        out.append((idx, word, slot_role[slot]))

    # THE VERB IS A ROLE FILLER TOO, and omitting it here is why ROLE_ACTION
    # held ZERO words while VERB_CORE had learned 51 verbs -- the verb sat
    # outside the role system entirely, so its position could not be learned.
    # `roles.py`'s own comment says exactly that about the supervised path
    # ("Skipping it here left the verb outside the role system"); the
    # unsupervised path had the same omission one layer further up, in this
    # function, which only ever iterated `noun_positions`.
    if verb_pos is not None and 0 <= verb_pos < len(words):
        out.append((verb_pos, words[verb_pos], ROLE_ACTION))
    return out


def compile_corpus(
    parser: "CoreParserMixin",
    sentences: List[GroundedSentence],
    *,
    stim_map: Optional[dict] = None,
) -> CorpusIndex:
    """Compile corpus features in a single pass."""
    smap = stim_map if stim_map is not None else parser.stim_map
    order = getattr(parser, "word_order_type", None) or "SVO"

    grounded: List[GroundedSentence] = []
    raw: List[List[str]] = []
    sent_indices: List[SentenceIndex] = []
    word_freq: Dict[str, int] = defaultdict(int)
    transition_counts: Dict[Tuple[Tuple[str, ...], str], int] = defaultdict(int)
    sig_by_prefix: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
    role_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    corpus_vocab: Set[str] = set()
    content_words: Set[str] = set()
    max_len = 0

    for sent in sentences:
        words = [w for w in sent.words if w in smap]
        if not words:
            continue
        max_len = max(max_len, len(words))
        grounded.append(sent)
        raw.append(list(sent.words))

        categories = [
            category_oracle(parser, w, parser.word_grounding.get(w))
            for w in words
        ]
        parser._category_cache.update(zip(words, categories))

        verb_pos = None
        for idx, cat in enumerate(categories):
            if cat == "VERB":
                verb_pos = idx
                break

        # PERCEPTION OUTRANKS POSITION. `_assign_noun_roles` reads roles off
        # word order, which `unsupervised.py`'s own docstring flags as unsound
        # exactly here: "Sentences that violate the typology (passives,
        # scrambling) will be assigned the wrong role and trained on it,
        # silently." When the sentence carries a perceived event, its roles
        # were derived from that event and are correct in EITHER voice, so they
        # are used instead. Sentences with no event (raw CDS text) still fall
        # through to the positional inducer, which is the right answer when the
        # corpus is 100% canonical order and there is nothing else to go on.
        noun_roles: Tuple[Tuple[int, str, str], ...] = ()
        scene_roles = None
        if getattr(sent, "event", None) is not None:
            kept = [r for w, r in zip(sent.words, sent.roles) if w in smap]
            if len(kept) == len(words):
                scene_roles = [
                    (i, w, ROLE_LABEL_TO_AREA[r])
                    for i, (w, r) in enumerate(zip(words, kept))
                    if r in ROLE_LABEL_TO_AREA
                ]
        if scene_roles is not None:
            roles = scene_roles
        elif verb_pos is not None:
            roles = _assign_noun_roles(words, categories, verb_pos, order)
        else:
            roles = []
        if roles:
            noun_roles = tuple(roles)
            for _, word, role_area in roles:
                role_counts[(word, role_area)] += 1
                content_words.add(word)

        sent_indices.append(SentenceIndex(
            words=tuple(words),
            categories=tuple(categories),
            verb_pos=verb_pos,
            noun_roles=noun_roles,
        ))

        for w in words:
            corpus_vocab.add(w)
            word_freq[w] += 1

        for i in range(len(words) - 1):
            prefix = tuple(words[: i + 1])
            nxt = words[i + 1]
            transition_counts[(prefix, nxt)] += 1
            sig_by_prefix[prefix] = tuple(categories[: i + 1])

    transitions = [
        BridgeTransition(
            prefix_words=prefix,
            next_word=nxt,
            category_signature=sig_by_prefix[prefix],
            count=count,
        )
        for (prefix, nxt), count in transition_counts.items()
    ]
    transitions.sort(key=lambda t: t.count, reverse=True)

    role_updates = [
        RoleUpdate(word=word, role_area=role, count=count)
        for (word, role), count in role_counts.items()
    ]
    role_updates.sort(key=lambda r: r.count, reverse=True)

    return CorpusIndex(
        grounded=grounded,
        raw=raw,
        sentences=sent_indices,
        corpus_vocab=corpus_vocab,
        max_sentence_length=max_len,
        word_freq=dict(word_freq),
        transitions=transitions,
        role_updates=role_updates,
        content_words=content_words,
    )


def merge_corpus_indices(
    *indices: CorpusIndex,
    extra_grounded: Optional[List[GroundedSentence]] = None,
) -> CorpusIndex:
    """Merge compiled indices (e.g. stage sents + instruction sents)."""
    all_grounded: List[GroundedSentence] = []
    for idx in indices:
        all_grounded.extend(idx.grounded)
    if extra_grounded:
        all_grounded.extend(extra_grounded)
    if not all_grounded:
        return CorpusIndex(
            grounded=[], raw=[], sentences=[], corpus_vocab=set(),
            max_sentence_length=0, word_freq={}, transitions=[],
            role_updates=[], content_words=set(),
        )

    word_freq: Dict[str, int] = defaultdict(int)
    transition_counts: Dict[Tuple[Tuple[str, ...], str], int] = defaultdict(int)
    sig_by_prefix: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
    role_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    corpus_vocab: Set[str] = set()
    content_words: Set[str] = set()
    max_len = 0
    sent_indices: List[SentenceIndex] = []
    raw: List[List[str]] = []

    for idx in indices:
        max_len = max(max_len, idx.max_sentence_length)
        corpus_vocab |= idx.corpus_vocab
        content_words |= idx.content_words
        for w, c in idx.word_freq.items():
            word_freq[w] += c
        for t in idx.transitions:
            transition_counts[(t.prefix_words, t.next_word)] += t.count
            sig_by_prefix[t.prefix_words] = t.category_signature
        for r in idx.role_updates:
            role_counts[(r.word, r.role_area)] += r.count
        sent_indices.extend(idx.sentences)
        raw.extend(idx.raw)

    transitions = [
        BridgeTransition(
            prefix_words=prefix,
            next_word=nxt,
            category_signature=sig_by_prefix[prefix],
            count=count,
        )
        for (prefix, nxt), count in transition_counts.items()
    ]
    transitions.sort(key=lambda t: t.count, reverse=True)

    role_updates = [
        RoleUpdate(word=w, role_area=role, count=c)
        for (w, role), c in role_counts.items()
    ]
    role_updates.sort(key=lambda r: r.count, reverse=True)

    return CorpusIndex(
        grounded=all_grounded,
        raw=raw,
        sentences=sent_indices,
        corpus_vocab=corpus_vocab,
        max_sentence_length=max_len,
        word_freq=dict(word_freq),
        transitions=transitions,
        role_updates=role_updates,
        content_words=content_words,
    )


def grounded_fraction(
    parser: "CoreParserMixin",
    corpus_vocab: Set[str],
) -> float:
    """Share of corpus tokens with grounded ``word_grounding`` entries."""
    if not corpus_vocab:
        return 0.0
    grounded = sum(
        1 for w in corpus_vocab
        if (ctx := parser.word_grounding.get(w)) is not None and ctx.is_grounded
    )
    return grounded / len(corpus_vocab)


def ingest_index_stats(
    parser: "CoreParserMixin",
    index: CorpusIndex,
) -> None:
    """Populate ``dist_stats`` from compiled categories (no re-classification)."""
    stats = parser.dist_stats

    for sent in index.sentences:
        words = list(sent.words)
        categories = list(sent.categories)
        n = len(words)
        if n == 0:
            continue
        stats.sentences_seen += 1
        verb_pos = sent.verb_pos

        for idx, word in enumerate(words):
            stats.word_count[word] += 1
            stats.position_counts[word][idx] += 1

            if idx + 1 < n:
                stats.transitions[(word, words[idx + 1])] += 1

            for j in range(max(0, idx - 2), min(n, idx + 3)):
                if j != idx:
                    stats.word_cooccurrence[word][words[j]] += 1

            if verb_pos is not None:
                if idx < verb_pos:
                    stats.word_as_pre_verb[word] += 1
                elif idx > verb_pos:
                    stats.word_as_post_verb[word] += 1
                elif idx == verb_pos:
                    stats.word_as_action[word] += 1

        for idx in range(len(categories) - 1):
            a, b = categories[idx], categories[idx + 1]
            if a and b:
                stats.category_transitions[(a, b)] += 1
