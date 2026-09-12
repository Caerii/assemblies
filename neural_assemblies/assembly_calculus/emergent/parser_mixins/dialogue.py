"""DialogueMixin — multi-turn context, chat training, online turn presentation."""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from ..session.dialogue_state import DialogueState
from ..structured_io import InstructionFrame

if TYPE_CHECKING:
    from ..curriculum.dialogue import DialoguePair
    from ..core.corpus_index import TransitionCache


class DialogueMixin:
    """Chat tuning: dialogue curriculum, CONTEXT carryover, ``present_turn``."""

    def train_dialogue(
        self,
        pairs: Optional[List["DialoguePair"]] = None,
        *,
        transition_cache: Optional["TransitionCache"] = None,
    ) -> None:
        """Train Q→A bridges via compiled corpus transitions."""
        from ..curriculum.dialogue import get_dialogue_pairs
        from ..training.compiler import compile_dialogue_pairs

        if pairs is None:
            pairs = get_dialogue_pairs()

        qa_words: set = set()
        for pair in pairs:
            qa_words.update(pair.question.words)
            qa_words.update(pair.answer.words)

        lex_targets = [w for w in qa_words if w in self.stim_map]
        self._ensure_prediction_lexicon(lex_targets)

        dialogue_index = compile_dialogue_pairs(self, pairs)
        if not dialogue_index.transitions:
            return

        self.train_next_token(
            [],
            corpus_index=dialogue_index,
            transition_cache=transition_cache,
            dedupe_sentences=False,
        )

    def _train_qa_bridge(self, question: List[str], answer: List[str]) -> None:
        """Legacy single-pair bridge (prefer ``train_dialogue``)."""
        q = [w for w in question if w in self.stim_map]
        a = [w for w in answer if w in self.stim_map]
        if len(q) < 1 or len(a) < 1:
            return

        self.build_context_incremental(q, reset=True, direct=True)

        bridge_rounds = self.bridge_rounds
        for ans_word in a[:3]:
            phon = self.stim_map.get(ans_word)
            if phon is None:
                continue
            self._train_next_token_bridge(phon, bridge_rounds=bridge_rounds)

    def present_turn(
        self,
        words: List[str],
        *,
        speaker: str = "user",
        learn: bool = True,
    ) -> InstructionFrame:
        """Ingest one dialogue turn and preserve its speaker in the frame."""
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError("speaker must be a nonempty string")
        known = [w for w in words if w in self.stim_map]
        if known:
            self.parse_incremental(known, light=True)
            if learn:
                for w in known:
                    self.ingest_raw_sentence([w])
        frame = self.parse_instruction(words)
        frame.speaker = speaker
        return frame

    def parse_instruction_with_context(
        self,
        words: List[str],
        state: Optional[DialogueState] = None,
    ) -> InstructionFrame:
        """Parse with pronoun resolution and optional prior CONTEXT."""
        if state is not None:
            words = state.resolve_words(words)
        frame = self.parse_instruction(words)
        if state is not None and len(state.recent_context_words()) > 0:
            prior = state.recent_context_words(n_turns=1)
            if prior:
                self.parse_incremental(prior[-8:], light=True)
        return frame

    def train_for_conversation(
        self,
        *,
        max_stage: str = "DIALOGUE",
        include_agent: bool = True,
    ) -> "DialogueMixin":
        """Deep curriculum: grammar stages → dialogue → optional agent skills."""
        from ..curriculum import CurriculumTrainer
        from ..train_progress import ensure_progress, finish_progress

        ensure_progress(f"train_for_conversation->{max_stage}")
        trainer = CurriculumTrainer(self)
        trainer.train_conversation_path(max_stage=max_stage)

        if include_agent:
            from ..curriculum.blocks import blocks_vocabulary
            from ..train_progress import current_progress

            with current_progress().phase("agent_tools"):
                extra = {
                    k: v for k, v in blocks_vocabulary().items()
                    if k not in self.stim_map
                }
                if extra:
                    self._register_vocabulary(extra)
                    self.word_grounding.update(extra)
                from ..core.corpus_index import compile_corpus
                from ..curriculum.data import create_instruction_sentences

                inst = create_instruction_sentences()
                idx = compile_corpus(self, inst)
                self.train_next_token(
                    inst,
                    corpus_index=idx,
                    transition_cache=trainer._transition_cache,
                )
                if hasattr(self, "train_dialogue"):
                    from ..curriculum.conversation import get_conversation_pairs
                    from ..curriculum.dialogue import get_dialogue_pairs

                    self.train_dialogue(
                        get_dialogue_pairs() + get_conversation_pairs(
                            self.word_grounding,
                        ),
                        transition_cache=trainer._transition_cache,
                    )

        finish_progress()
        return self
