"""BlocksMixin — blocks-world instructions and scaled agent training."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..core.sentence import GroundedSentence
    from ..train_progress import TrainProgress

from ..blocks_bridge import is_blocks_command, normalize_blocks_command, parse_blocks_command
from ..structured_io import InstructionFrame


class BlocksMixin:
    """Blocks-world command parsing and training integration."""

    def parse_instruction(self, words: List[str]) -> InstructionFrame:
        """Parse with blocks-world pattern priority."""
        if is_blocks_command(words):
            blocks_frame = parse_blocks_command(words)
            if blocks_frame is not None:
                return blocks_frame
        normalized = normalize_blocks_command(words)
        frame = super().parse_instruction(normalized)  # type: ignore[misc]
        if is_blocks_command(words):
            bf = parse_blocks_command(words)
            if bf is not None:
                return bf
        return frame

    def train_for_agent(
        self,
        sentences: Optional[List["GroundedSentence"]] = None,
        holdout_words: Optional[set] = None,
        include_dialogue: bool = True,
        include_blocks: bool = True,
        include_word_order: Optional[bool] = None,
        include_morphology: Optional[bool] = None,
        include_extra_dialogue: bool = False,
        progress: Optional["TrainProgress"] = None,
    ) -> "BlocksMixin":
        """Full agent training with optional blocks instruction curriculum."""
        from ..curriculum.blocks import (
            blocks_vocabulary,
            create_blocks_instruction_sentences,
        )
        from ..curriculum.dialogue import get_dialogue_curriculum
        from ..curriculum.data import (
            create_instruction_sentences,
            create_training_sentences,
        )
        from ..train_progress import current_progress

        prog = progress if progress is not None else current_progress()

        if include_word_order is None:
            include_word_order = not self.fast_training
        if include_morphology is None:
            include_morphology = not self.fast_training

        if include_blocks:
            with prog.phase("blocks_vocab"):
                full_vocab = blocks_vocabulary()
                extra = {k: v for k, v in full_vocab.items() if k not in self.stim_map}
                if extra:
                    self._register_vocabulary(extra)
                self.word_grounding.update(extra)

        if sentences is None:
            sentences = create_training_sentences() + create_instruction_sentences()
            if include_blocks:
                sentences = sentences + create_blocks_instruction_sentences()
            if include_dialogue:
                sentences = sentences + get_dialogue_curriculum()

        with prog.phase("core_train", f"{len(sentences)} sentences"):
            self.train(
                sentences=sentences,
                holdout_words=holdout_words,
                train_prediction=True,
                include_word_order=include_word_order,
                include_morphology=include_morphology,
                progress=prog,
            )
        if include_extra_dialogue:
            with prog.phase("dialogue"):
                self.train_dialogue()
        return self
