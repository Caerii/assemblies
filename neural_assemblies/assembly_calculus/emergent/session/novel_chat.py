"""Train on large generated corpora and prepare parsers for novel interactive chat."""

from __future__ import annotations

from typing import Dict, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.grounding import GroundingContext
    from ..parser import EmergentParser


def build_chat_corpus(
    vocabulary: Dict[str, "GroundingContext"],
    *,
    n_sentences: int = 400,
    seed: int = 42,
) -> list:
    """Generate a large grounded sentence corpus covering the vocabulary."""
    from ..curriculum.data import generate_training_sentences

    return generate_training_sentences(vocabulary, n_sentences=n_sentences, seed=seed)


def register_corpus_memory(
    parser: "EmergentParser",
    sentences: list,
) -> None:
    """Store training sentences for novelty scoring at generation time."""
    parser._corpus_sentence_set = {
        tuple(s.words) for s in sentences
    }
    parser._corpus_bigram_set = set()
    for s in sentences:
        ws = s.words
        for i in range(len(ws) - 1):
            parser._corpus_bigram_set.add((ws[i], ws[i + 1]))


def train_for_novel_chat(
    parser: "EmergentParser",
    *,
    n_corpus_sentences: int = 400,
    seed: int = 42,
    max_stage: str = "CONVERSATION",
    skip_early_curriculum: bool = False,
    include_dialogue: bool = True,
) -> Dict[str, object]:
    """Full path: large corpus grammar + bridges + multi-turn conversation.

    1. Generate hundreds of template sentences over the loaded vocabulary.
    2. Train lexicon, roles, phrases, word order, morphology, prediction bridges.
    3. Run conversation curriculum (DIALOGUE / CONVERSATION).
    4. Train QA bridges for interactive chat.

    Returns a summary dict (counts and stages run).
    """
    from ..acquisition import run_developmental_acquisition
    from ..curriculum.conversation import get_conversation_pairs
    from ..core.corpus_index import compile_corpus
    from ..curriculum import CurriculumTrainer
    from ..curriculum.dialogue import get_dialogue_pairs
    from ..train_progress import current_progress
    from ..training.perf import developmental_curriculum_enabled

    vocab = parser.word_grounding
    acquisition_report = None

    if developmental_curriculum_enabled():
        acquisition_report = run_developmental_acquisition(
            parser,
            max_stage="SENTENCES",
            seed=seed,
            babble=True,
            fuzzy_early_words=True,
            adaptive=True,
        )

    corpus = build_chat_corpus(vocab, n_sentences=n_corpus_sentences, seed=seed)
    register_corpus_memory(parser, corpus)

    prog = current_progress()
    prog.info(
        f"novel chat corpus: {len(corpus)} sentences, "
        f"{len(vocab)} words registered",
    )

    with prog.phase("grammar_corpus"):
        parser.train(
            sentences=corpus,
            train_prediction=True,
        )

    trainer = CurriculumTrainer(parser)
    with prog.section(max_stage):
        trainer.train_conversation_path(
            max_stage=max_stage,
            skip_early_if_loaded=not developmental_curriculum_enabled()
            and skip_early_curriculum,
        )

    idx = compile_corpus(parser, corpus)
    with prog.phase("corpus_bridges", f"{len(idx.transitions)} transitions"):
        parser._ensure_prediction_lexicon(list(idx.corpus_vocab))
        parser.train_next_token(
            corpus,
            corpus_index=idx,
            transition_cache=trainer._transition_cache,
        )

    dialogue_pairs = 0
    if include_dialogue:
        pairs = (
            get_dialogue_pairs(seed=seed)
            + get_conversation_pairs(parser.word_grounding)
        )
        dialogue_pairs = len(pairs)
        with prog.phase("dialogue", f"{dialogue_pairs} Q-A pairs"):
            parser.train_dialogue(
                pairs,
                transition_cache=trainer._transition_cache,
            )

    return {
        "corpus_sentences": len(corpus),
        "corpus_vocab": len(idx.corpus_vocab),
        "bridge_transitions": len(idx.transitions),
        "dialogue_pairs": dialogue_pairs,
        "max_stage": max_stage,
        "vocab_size": len(parser.stim_map),
        "developmental": developmental_curriculum_enabled(),
        "acquisition_stages": (
            acquisition_report.stages_run if acquisition_report else []
        ),
    }
