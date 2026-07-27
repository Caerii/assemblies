"""Replay wobbly episodes: remedial training and POS commit."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from .hypotheses import trial_category_in_sentence
from .memory import WobblyMemory

if TYPE_CHECKING:
    from ...parser import EmergentParser

from .memory import STRUCTURAL_SIGNATURES


def resolve_wobbly_hypotheses(
    parser: "EmergentParser",
    memory: WobblyMemory,
) -> WobblyMemory:
    """Compete POS hypotheses by re-parsing with forced category (NEMO merge test)."""
    for ep in memory.episodes:
        if ep.resolved_category is not None:
            continue
        candidates = [ep.probe.category]
        candidates.extend(cat for _w, cat in ep.hypotheses)

        best_cat = ep.probe.category
        best_combined = ep.probe.combined
        best_stability = ep.probe.phrase_stability

        seen: Set[str] = set()
        for cat in candidates:
            if cat in seen:
                continue
            seen.add(cat)
            trial = trial_category_in_sentence(
                parser,
                list(ep.sentence),
                ep.probe.position,
                cat,
            )
            if trial is None:
                continue
            combined, _p600, _n400, stability = trial
            if combined < best_combined or (
                abs(combined - best_combined) < 0.02
                and stability > best_stability
            ):
                best_combined = combined
                best_cat = cat
                best_stability = stability

        ep.resolved_category = best_cat
        ep.resolved_combined = best_combined
        ep.resolved_stability = best_stability
    return memory


def episodes_to_grounded_sentences(parser: "EmergentParser", memory: WobblyMemory) -> List:
    """Build grounded replay sentences with resolved POS roles."""
    from ...core.grounding import GroundingContext
    from ...core.sentence import GroundedSentence

    sentences = []
    seen: Set[Tuple[str, ...]] = set()
    resolved_map: Dict[Tuple[str, Tuple[int, str]], str] = {}
    for ep in memory.episodes:
        if ep.resolved_category:
            resolved_map[(ep.sentence, (ep.probe.position, ep.probe.word))] = (
                ep.resolved_category
            )

    for ep in memory.episodes:
        key = ep.sentence
        if key in seen:
            continue
        seen.add(key)
        words = list(key)
        contexts = [
            parser.word_grounding.get(w, GroundingContext()) for w in words
        ]
        roles: List[Optional[str]] = []
        for i, w in enumerate(words):
            cat = resolved_map.get((key, (i, w)), None)
            if cat is None:
                roles.append(None)
            elif cat == "VERB":
                roles.append("action")
            elif cat in ("NOUN", "PRON") and i == 0:
                roles.append("agent")
            elif cat in ("NOUN", "PRON"):
                roles.append("patient")
            else:
                roles.append(None)
        sentences.append(GroundedSentence(words=words, contexts=contexts, roles=roles))
    return sentences


def replay_wobbly_episodes(
    parser: "EmergentParser",
    memory: Optional[WobblyMemory] = None,
    *,
    trainer=None,
) -> Dict[str, object]:
    """Replay wobbly episodes: remedial ingest, consolidation, POS commit."""
    from ..adaptive import MisclassificationTarget, build_remedial_sentences
    from ..pos_inference import (
        infer_holdout_categories,
        ingest_holdout_sentence_stats,
        record_wobbly_resolution,
    )
    from ...training.consolidation import (
        consolidate_role_pathways,
        consolidate_vp_pathways,
    )

    mem = memory or getattr(parser, "_wobbly_memory", None)
    if mem is None or not mem.episodes:
        return {"episodes": 0, "assigned": {}, "remedial_sentences": 0}

    resolve_wobbly_hypotheses(parser, mem)

    targets: List[MisclassificationTarget] = []
    word_set: Set[str] = set()

    for ep in mem.episodes:
        word = ep.probe.word
        resolved = ep.resolved_category or ep.probe.category
        if ep.probe.failure_signature not in STRUCTURAL_SIGNATURES:
            continue
        if resolved == ep.probe.category and not ep.hypotheses:
            continue
        if resolved != ep.probe.category or ep.probe.wobbly:
            targets.append(
                MisclassificationTarget(
                    word=word,
                    expected=resolved,
                    predicted=ep.probe.category,
                    failure_mode=ep.probe.failure_signature,
                ),
            )
            word_set.add(word)

    for ep in mem.episodes:
        word = ep.probe.word
        resolved = ep.resolved_category or ep.probe.category
        if word in word_set:
            record_wobbly_resolution(
                parser,
                word,
                resolved,
                stability=ep.resolved_stability or ep.probe.phrase_stability,
                exposure=parser.dist_stats.word_count.get(word, 1),
            )

    # dict.fromkeys, not a set comprehension: this drives TRAINING order, and
    # set-of-tuple iteration order varies with PYTHONHASHSEED across processes.
    # Deduplicate while keeping first-seen order so replay is reproducible.
    for sent in dict.fromkeys(ep.sentence for ep in mem.episodes):
        parser.ingest_raw_sentence(list(sent))

    ingest_holdout_sentence_stats(
        parser, word_set, allow_canonical_fallback=True,
    )

    remedial = build_remedial_sentences(parser, targets, seed=17)
    replay_sentences = episodes_to_grounded_sentences(parser, mem)
    remedial_count = len(remedial)

    if trainer is not None and remedial:
        trainer.train_remedial(
            remedial,
            phases=["distributional", "lexicon"],
            label="WOBBLY_BOOTSTRAP",
        )
    else:
        from ...core.corpus_index import compile_corpus, ingest_index_stats

        if remedial:
            idx = compile_corpus(parser, remedial)
            ingest_index_stats(parser, idx)

    all_replay = remedial + replay_sentences
    if all_replay:
        consolidate_role_pathways(parser, all_replay, passes=1)
        consolidate_vp_pathways(parser, all_replay, passes=1)
        # Consolidation re-issues neuron IDs in the core/role areas it replays
        # through, orphaning any assembly snapshot stored before it. Drop those
        # stale lexicon entries so later stages re-project instead of injecting
        # a snapshot that no longer maps. See consolidation.drop_stale_assemblies.
        from ....consolidation import drop_stale_assemblies
        drop_stale_assemblies(parser)

    assigned = infer_holdout_categories(parser, word_set) if word_set else {}

    return {
        "episodes": len(mem.episodes),
        "wobbly_words": sorted(word_set),
        "assigned": dict(assigned),
        "remedial_sentences": remedial_count,
        "replay_sentences": len(replay_sentences),
        "consolidated": bool(all_replay),
    }


bootstrap_from_wobbly_memory = replay_wobbly_episodes


def mine_and_bootstrap_from_exposure(
    parser: "EmergentParser",
    *,
    trainer=None,
    max_sentences: int = 40,
    target_words: Optional[Set[str]] = None,
) -> Dict[str, object]:
    """Mine wobbly probes from exposure log and run episode replay."""
    from ..pos_inference import is_word_in_lexicon, sentences_from_exposure_log
    from ...evaluation.erp import assess_erp_readiness, ensure_parser_erp_calibration
    from .mining import mine_wobbly_episodes

    if target_words:
        words = set(target_words)
    else:
        words = {
            w for w in parser.stim_map
            if not is_word_in_lexicon(parser, w)
        }
    if not words:
        return {"episodes": 0, "assigned": {}, "remedial_sentences": 0}

    readiness = assess_erp_readiness(parser)
    if not readiness.p600_ready:
        return {
            "episodes": 0,
            "assigned": {},
            "remedial_sentences": 0,
            "skipped": "p600_not_ready",
            "readiness": {
                "n400_ready": readiness.n400_ready,
                "p600_ready": readiness.p600_ready,
                "sentences_seen": readiness.sentences_seen,
            },
        }

    ensure_parser_erp_calibration(parser)

    sents = sentences_from_exposure_log(parser, words)
    if not sents:
        return {"episodes": 0, "assigned": {}, "remedial_sentences": 0}

    mem = mine_wobbly_episodes(
        parser, sents[-max_sentences:], target_words=words,
    )
    return replay_wobbly_episodes(parser, mem, trainer=trainer)


def format_wobbly_report(memory: WobblyMemory) -> str:
    lines = [
        f"Wobbly memory: {len(memory.episodes)} episodes",
        "",
    ]
    for ep in memory.episodes[:20]:
        p = ep.probe
        lines.append(
            f"  {' '.join(ep.sentence)} @ {p.word!r} "
            f"cat={p.category} n400={p.n400:.2f} p600={p.p600:.2f} "
            f"stab={p.phrase_stability:.2f} err={p.error_active} "
            f"sig={p.failure_signature}",
        )
        if ep.hypotheses:
            lines.append(f"    hypotheses: {list(ep.hypotheses)}")
        if ep.resolved_category:
            lines.append(
                f"    resolved: {ep.resolved_category} "
                f"(stab={ep.resolved_stability:.2f})",
            )
    return "\n".join(lines)
