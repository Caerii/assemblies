"""
Neural training compiler: prefix trie, bridge programs, and link-time plans.

Compiles corpus indices into minimal execution schedules so runtime training
does not rebuild shared prefixes or re-discover connectome topology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..parser_mixins.core import CoreParserMixin
    from ..core.corpus_index import BridgeTransition, CorpusIndex
    from ..curriculum.dialogue import DialoguePair
    from ..training.compiled import CompiledTopologySpec


@dataclass(frozen=True)
class BridgeOp:
    """One bridge-training step in compiled order."""

    prefix_words: Tuple[str, ...]
    next_word: str
    bridge_rounds: int
    count: int = 1


@dataclass
class PrefixTrie:
    """Prefix tree over bridge context prefixes."""

    children: Dict[str, "PrefixTrie"] = field(default_factory=dict)
    is_terminal: bool = False

    def insert(self, prefix: Sequence[str]) -> None:
        node = self
        for word in prefix:
            node = node.children.setdefault(word, PrefixTrie())
        node.is_terminal = True

    def iter_prefixes_dfs(self) -> List[Tuple[str, ...]]:
        """Terminal prefixes in trie DFS order (shared prefixes adjacent)."""
        out: List[Tuple[str, ...]] = []

        def dfs(node: PrefixTrie, path: List[str]) -> None:
            if node.is_terminal and path:
                out.append(tuple(path))
            for word in sorted(node.children.keys()):
                dfs(node.children[word], path + [word])

        dfs(self, [])
        return out


@dataclass
class CompiledTrainingPlan:
    """Link-time artifact for one bridge-training episode."""

    topology: "CompiledTopologySpec"
    fidelity: str = "compiled"
    prefix_trie: Optional[PrefixTrie] = None
    bridge_ops: List[BridgeOp] = field(default_factory=list)
    prefix_order: List[Tuple[str, ...]] = field(default_factory=list)
    pregrow_done: bool = False


@dataclass(frozen=True)
class RoleOp:
    """One unsupervised role projection in compiled order."""

    word: str
    role_area: str
    rounds: int


@dataclass
class CompiledRolePlan:
    """Link-time artifact for one role-training episode."""

    topology: "CompiledTopologySpec"
    role_ops: List[RoleOp] = field(default_factory=list)
    repetitions: int = 1


def compile_role_plan(
    parser: "CoreParserMixin",
    corpus_index: "CorpusIndex",
    repetitions: int,
) -> CompiledRolePlan:
    """Compile role updates into a flat execution schedule."""
    from ..training.compiled import role_topology_spec
    from ..training.perf import adaptive_rounds

    ops: List[RoleOp] = []
    word_freq = corpus_index.word_freq
    for _rep in range(repetitions):
        for update in corpus_index.role_updates:
            if update.word not in parser.stim_map:
                continue
            rounds = adaptive_rounds(
                parser.rounds, word_freq.get(update.word, 1),
            )
            ops.append(
                RoleOp(
                    word=update.word,
                    role_area=update.role_area,
                    rounds=rounds,
                ),
            )
    return CompiledRolePlan(
        topology=role_topology_spec(parser),
        role_ops=ops,
        repetitions=repetitions,
    )


@dataclass(frozen=True)
class LexiconOp:
    """One lexicon word projection in compiled order."""

    word: str
    core_area: str
    rounds: int


@dataclass
class CompiledLexiconPlan:
    """Link-time artifact for one lexicon-training episode."""

    topology: "CompiledTopologySpec"
    lexicon_ops: List[LexiconOp] = field(default_factory=list)
    core_areas: Tuple[str, ...] = ()


def compile_lexicon_plan(
    parser: "CoreParserMixin",
    *,
    holdout_words: Optional[set] = None,
    skip_known: bool = True,
    words: Optional[Sequence[str]] = None,
) -> CompiledLexiconPlan:
    """Compile pending lexicon words into a flat execution schedule."""
    from ..core.areas import GROUNDING_TO_CORE
    from ..training.compiled import lexicon_topology_spec
    from ..training.perf import adaptive_rounds

    holdout = holdout_words or set()
    word_set = set(words) if words is not None else None
    ops: List[LexiconOp] = []
    word_count = getattr(parser.dist_stats, "word_count", {})

    for word, ctx in parser.word_grounding.items():
        if word_set is not None and word not in word_set:
            continue
        if word in holdout:
            continue
        if word not in parser.stim_map:
            continue
        core = GROUNDING_TO_CORE[ctx.dominant_modality]
        if skip_known and word in parser.core_lexicons.get(core, {}):
            continue
        freq = word_count.get(word, 1)
        rounds = adaptive_rounds(parser.rounds, freq)
        ops.append(LexiconOp(word=word, core_area=core, rounds=rounds))

    core_areas = tuple(sorted({op.core_area for op in ops}))
    return CompiledLexiconPlan(
        topology=lexicon_topology_spec(parser, core_areas),
        lexicon_ops=ops,
        core_areas=core_areas,
    )


def group_lexicon_ops_by_core(
    plan: CompiledLexiconPlan,
) -> Dict[str, List[LexiconOp]]:
    grouped: Dict[str, List[LexiconOp]] = {}
    for op in plan.lexicon_ops:
        grouped.setdefault(op.core_area, []).append(op)
    return grouped


def build_prefix_trie(prefixes: Sequence[Sequence[str]]) -> PrefixTrie:
    trie = PrefixTrie()
    for prefix in prefixes:
        if prefix:
            trie.insert(prefix)
    return trie


def compile_bridge_ops(
    transitions: Sequence["BridgeTransition"],
    *,
    default_bridge_rounds: int,
    adaptive_rounds_fn,
) -> List[BridgeOp]:
    """Turn corpus transitions into ordered bridge operations."""
    ops: List[BridgeOp] = []
    for trans in transitions:
        rounds = adaptive_rounds_fn(default_bridge_rounds, trans.count)
        ops.append(
            BridgeOp(
                prefix_words=trans.prefix_words,
                next_word=trans.next_word,
                bridge_rounds=rounds,
                count=trans.count,
            ),
        )
    return ops


def compile_training_plan(
    parser: "CoreParserMixin",
    corpus_index: "CorpusIndex",
    *,
    transitions: Optional[Sequence["BridgeTransition"]] = None,
) -> CompiledTrainingPlan:
    """Compile a bridge-training plan from a corpus index."""
    from ..training.compiled import bridge_topology_spec
    from ..training.perf import adaptive_rounds

    trans = list(transitions if transitions is not None else corpus_index.transitions)
    prefixes = sorted({t.prefix_words for t in trans}, key=lambda p: (len(p), p))
    trie = build_prefix_trie(prefixes)
    ops = compile_bridge_ops(
        trans,
        default_bridge_rounds=parser.bridge_rounds,
        adaptive_rounds_fn=adaptive_rounds,
    )
    return CompiledTrainingPlan(
        topology=bridge_topology_spec(parser),
        prefix_trie=trie,
        bridge_ops=ops,
        prefix_order=trie.iter_prefixes_dfs(),
    )


def lcp_length(a: Tuple[str, ...], b: Tuple[str, ...]) -> int:
    """Longest common prefix length between two word tuples."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def group_bridge_ops_by_prefix(
    ops: Sequence[BridgeOp],
) -> Dict[Tuple[str, ...], List[BridgeOp]]:
    grouped: Dict[Tuple[str, ...], List[BridgeOp]] = {}
    for op in ops:
        grouped.setdefault(op.prefix_words, []).append(op)
    return grouped


def compile_dialogue_pairs(
    parser: "CoreParserMixin",
    pairs: Sequence["DialoguePair"],
) -> "CorpusIndex":
    """Compile Q→A pairs into bridge transitions (question context → answer word)."""
    from collections import defaultdict

    from ..core.corpus_index import BridgeTransition, CorpusIndex, category_oracle

    transition_counts: Dict[Tuple[Tuple[str, ...], str], int] = defaultdict(int)
    sig_by_prefix: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
    corpus_vocab: set = set()
    max_len = 0

    for pair in pairs:
        q = [w for w in pair.question.words if w in parser.stim_map]
        a = [w for w in pair.answer.words if w in parser.stim_map]
        if len(q) < 1 or len(a) < 1:
            continue
        max_len = max(max_len, len(q), len(a))
        corpus_vocab.update(q)
        corpus_vocab.update(a[:3])
        categories = [
            category_oracle(parser, w, parser.word_grounding.get(w))
            for w in q
        ]
        prefix = tuple(q)
        sig_by_prefix[prefix] = tuple(categories)
        for ans_word in a[:3]:
            transition_counts[(prefix, ans_word)] += 1
            corpus_vocab.add(ans_word)

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

    return CorpusIndex(
        grounded=[],
        raw=[],
        sentences=[],
        corpus_vocab=corpus_vocab,
        max_sentence_length=max_len,
        word_freq={},
        transitions=transitions,
        role_updates=[],
        content_words=set(corpus_vocab),
    )


def link_preallocate_stim_targets(
    parser: "CoreParserMixin",
    area_names: Sequence[str],
) -> None:
    """Pre-grow stim→area 1-D vectors to current ever-fired depth."""
    for area_name in area_names:
        if area_name not in parser.brain.areas:
            continue
        engine = parser.brain._engine_for(parser.brain.areas[area_name])
        if not getattr(engine, "supports_stim_preallocation", False):
            continue
        w = parser.brain.population_counts(area_name).ever_fired
        if w > 0:
            engine.preallocate_stim_targets(area_name, w)
