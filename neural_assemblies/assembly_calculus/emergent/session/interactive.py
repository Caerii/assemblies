"""Interactive session on EmergentParser (numpy_sparse chat loop).

THE THING TO UNDERSTAND ABOUT CHATTING WITH THIS MODEL.  There is no
inference mode.  Parsing a user turn runs the same projections that training
runs, and plasticity is on by default, so the connectome is being modified by
the conversation.  A long session is continued training on whatever the user
happened to say.

That is a feature of the model rather than an oversight -- a biological
learner does not stop learning to answer a question -- but it has two
practical consequences worth stating.  Replies are not reproducible: the same
input twice gives different output, because the first pass changed the
weights the second reads.  And a session can degrade previously learned
grammar; ``acquisition.continual`` exists to measure exactly that, and its
before/after snapshot is the right instrument to reach for after a long
session.

Set ``brain.disable_plasticity = True`` if you want a frozen model that
answers without learning.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from ..session.dialogue_state import DialogueState
from ..structured_json import record_to_tool_call
from ..structured_io import InstructionFrame, ToolCall
from ..tools import ToolRegistry
from ..blocks_bridge import BlocksLanguageExecutor, frame_to_blocks_action


@dataclass
class Turn:
    speaker: str
    text: str
    words: List[str]
    frame: Optional[InstructionFrame] = None
    tool_call: Optional[ToolCall] = None


@dataclass
class EmergentSession:
    """Multi-turn wrapper: parse → tool dispatch → response generation."""

    parser: object
    registry: ToolRegistry = field(default_factory=ToolRegistry)
    dialogue: DialogueState = field(default_factory=DialogueState)
    history: List[Turn] = field(default_factory=list)
    execute_tools: bool = False
    online_learn: bool = True
    blocks_executor: Optional[BlocksLanguageExecutor] = None

    def __post_init__(self) -> None:
        if self.blocks_executor is None and self.execute_tools:
            self.blocks_executor = BlocksLanguageExecutor.from_toy_problem()
        if self.execute_tools:
            self.registry.register_handler(
                "move_block",
                self._execute_move_block,
            )

    def _execute_move_block(self, call: ToolCall) -> str:
        if self.blocks_executor is None:
            self.blocks_executor = BlocksLanguageExecutor.from_toy_problem()
        from ..blocks_bridge import tool_call_to_blocks_action

        action = tool_call_to_blocks_action(call)
        if action is None:
            return "cannot execute"
        try:
            self.blocks_executor.apply_action(action)
        except ValueError:
            return f"illegal move {action.src} to {action.dst}"
        return f"ok {action.src} on {action.dst}"

    @classmethod
    def bootstrap(
        cls,
        n: int = 3000,
        k: int = 30,
        seed: int = 42,
        **parser_kwargs,
    ) -> "EmergentSession":
        from ..parser import EmergentParser

        parser = EmergentParser(n=n, k=k, seed=seed, **parser_kwargs)
        parser.train_for_agent()
        return cls(parser=parser)

    @classmethod
    def bootstrap_novel_chat(
        cls,
        preset: str = "medium",
        *,
        n_corpus_sentences: int = 400,
        max_stage: str = "CONVERSATION",
        n: int = 3000,
        k: int = 30,
        seed: int = 42,
        **parser_kwargs,
    ) -> "EmergentSession":
        """Large-corpus grammar + bridges + conversation; tuned for novel generation."""
        from ..session.novel_chat import train_for_novel_chat
        from ..parser import EmergentParser
        from ..vocabulary_builder import build_vocabulary_preset
        from ..train_progress import finish_progress, start_progress

        vocab = build_vocabulary_preset(preset)
        prog = start_progress(f"novel_chat_{preset}")
        prog.info(
            f"vocab={len(vocab)} corpus_sents={n_corpus_sentences} "
            f"stage={max_stage}",
        )
        parser = EmergentParser(
            n=n, k=k, seed=seed, vocabulary=vocab,
            fast_training=parser_kwargs.pop("fast_training", True),
            **parser_kwargs,
        )
        os.environ["EMERGENT_DEV_CURRICULUM"] = "1"
        train_for_novel_chat(
            parser,
            n_corpus_sentences=n_corpus_sentences,
            seed=seed,
            max_stage=max_stage,
            skip_early_curriculum=False,
        )
        finish_progress(f"{len(parser.stim_map)} words ready for novel chat")
        return cls(parser=parser, online_learn=True)

    @classmethod
    def bootstrap_conversation(
        cls,
        preset: str = "medium",
        max_stage: str = "DIALOGUE",
        n: int = 3000,
        k: int = 30,
        seed: int = 42,
        include_agent: bool = True,
        **parser_kwargs,
    ) -> "EmergentSession":
        """Train scaled vocabulary + conversation curriculum; return chat session."""
        from ..parser import EmergentParser
        from ..vocabulary_builder import build_vocabulary_preset
        from ..train_progress import finish_progress, start_progress

        vocab = build_vocabulary_preset(preset)
        prog = start_progress(f"bootstrap_{preset}->{max_stage}")
        prog.info(f"vocab={len(vocab)} words n={n} k={k}")
        parser = EmergentParser(
            n=n, k=k, seed=seed, vocabulary=vocab, **parser_kwargs,
        )
        parser.train_for_conversation(
            max_stage=max_stage,
            include_agent=include_agent,
        )
        finish_progress(f"{len(parser.stim_map)} words registered")
        return cls(parser=parser, online_learn=True)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.strip().lower()
        text = re.sub(r"([?.!,])", r" \1", text)
        return [w for w in text.split() if w]

    def interact(self, user_text: str) -> str:
        """Process one user turn; return assistant response string."""
        from ..acquisition.continual import capture_stability_snapshot

        words = self.tokenize(user_text)
        if not words:
            return ""

        before = capture_stability_snapshot(self.parser) if self.online_learn else None

        resolved = self.dialogue.resolve_words(words)
        for i, w in enumerate(resolved):
            if hasattr(self.parser, "resolve_surface_word"):
                resolved[i] = self.parser.resolve_surface_word(w)
        for w in resolved:
            if w not in self.parser.stim_map:
                self.parser.register_word(w)
            elif hasattr(self.parser, "resolve_surface_word"):
                canon = self.parser.resolve_surface_word(w)
                if canon != w and canon not in self.parser.stim_map:
                    self.parser.register_word(canon)
        if self.online_learn:
            self.parser.present_turn(resolved, speaker="user", learn=True)

        frame = self.parser.parse_instruction_with_context(resolved, self.dialogue)
        tool_call = self.registry.dispatch(frame)

        turn = Turn(
            speaker="user",
            text=user_text.strip(),
            words=resolved,
            frame=frame,
            tool_call=tool_call,
        )
        self.history.append(turn)

        if frame.mood == "INTERROGATIVE":
            reply = self._answer_question(resolved, frame)
        elif frame.mood == "IMPERATIVE":
            reply = self._handle_instruction(frame, tool_call)
        else:
            reply = self._describe(frame)

        reply_words = self.tokenize(reply)
        if self.online_learn and reply_words:
            self.parser.present_turn(reply_words, speaker="assistant", learn=True)

        self.dialogue.record_turn("user", resolved, frame=frame, reply=reply)
        self.history.append(
            Turn(speaker="assistant", text=reply, words=reply_words)
        )

        if self.online_learn and before is not None:
            from ..acquisition.continual import stability_gate_after_session

            corpus = getattr(self.parser, "_corpus_sentence_set", None)
            replay = [list(s) for s in corpus][:15] if corpus else None
            stability_gate_after_session(
                self.parser, before, replay_sentences=replay,
            )

        return reply

    def interact_json(self, user_text: str) -> Optional[str]:
        """Parse user text and return validated tool-call JSON, or None."""
        words = self.tokenize(user_text)
        if not words:
            return None
        resolved = self.dialogue.resolve_words(words)
        return self.parser.words_to_json(resolved)

    def execute_json(self, json_text: str) -> str:
        """Parse JSON tool call, execute if possible, return status string."""
        record = self.parser.parse_json_command(json_text)
        call = record_to_tool_call(record)
        if call is None:
            return "invalid tool call"
        if call.name == "move_block" and self.execute_tools:
            return self._execute_move_block(call)
        words = self.parser.json_to_words(json_text)
        frame = self.parser.parse_instruction(words)
        tool_call = self.registry.dispatch(frame)
        if tool_call and self.execute_tools:
            result = self.registry.execute(tool_call)
            if isinstance(result, str):
                return result
        return call.to_json()

    def interact_plan(self, user_text: str) -> Optional[str]:
        """Parse multi-step or goal-directed instruction → plan JSON."""
        if self.blocks_executor is None:
            self.blocks_executor = BlocksLanguageExecutor(blocks=("A", "B", "C"))
        plan = self.parser.text_to_tool_plan(user_text, self.blocks_executor)
        if plan is None or plan.length == 0:
            return None
        return plan.to_json()

    def execute_plan_json(self, json_text: str) -> str:
        """Execute a multi-step tool plan; return result JSON."""
        if self.blocks_executor is None:
            self.blocks_executor = BlocksLanguageExecutor(blocks=("A", "B", "C"))
        plan = self.parser.json_to_plan(json_text)
        result = self.parser.execute_plan(plan, self.blocks_executor)
        return result.to_json()

    def _answer_question(self, words: List[str], frame: InstructionFrame) -> str:
        wh = words[0] if words else ""

        if wh in ("who", "whom"):
            agent = frame.roles.get("agent") or self._role_lookup(frame, "AGENT")
            if agent and agent not in ("who", "whom", "what"):
                if hasattr(self.parser, "generate_novel_sentence"):
                    novel = self.parser.generate_novel_sentence(
                        seed_prefix=["the", agent],
                        min_words=3,
                        max_words=7,
                    )
                    if novel and self.parser.sentence_novelty(novel) > 0.3:
                        return " ".join(novel)
                return agent
            preds = self.parser.predict_next(words)
            if preds:
                for w, _ in preds[:5]:
                    if w in self.parser.word_grounding:
                        ctx = self.parser.word_grounding[w]
                        if ctx.dominant_modality == "visual":
                            return w

        if wh == "what":
            patient = frame.roles.get("patient") or self._role_lookup(frame, "PATIENT")
            if patient:
                return patient
            for word, role in frame.roles.items():
                if role == "PATIENT":
                    return word

        if words and words[0] == "does":
            preds = self.parser.predict_next(words)
            if preds and preds[0][0] == "yes":
                return "yes"

        if wh in ("does", "do", "did", "can", "is", "are") and len(words) > 1:
            preds = self.parser.predict_next(words[:-1])
            if preds:
                return preds[0][0]

        if self.dialogue.last_agent and wh in ("who", "whom"):
            return self.dialogue.last_agent

        return "unknown"

    @staticmethod
    def _role_lookup(frame: InstructionFrame, role: str) -> Optional[str]:
        for word, r in frame.roles.items():
            if r == role:
                return word
        return None

    def _handle_instruction(
        self,
        frame: InstructionFrame,
        tool_call: Optional[ToolCall],
    ) -> str:
        if tool_call is None:
            words = self.parser.generate_confirmation(frame)
            return " ".join(words)

        if self.execute_tools:
            result = self.registry.execute(tool_call)
            if isinstance(result, str):
                return result
            action = frame_to_blocks_action(frame)
            if action is not None and self.blocks_executor is not None:
                try:
                    self.blocks_executor.apply_action(action)
                    return f"ok {action.src} on {action.dst}"
                except ValueError:
                    return f"illegal move {action.src} to {action.dst}"

        args = tool_call.arguments
        target = args.get("target") or args.get("object") or args.get("block")
        if target:
            return f"ok {tool_call.name} {target}"
        return f"ok {tool_call.name}"

    def _describe(self, frame: InstructionFrame) -> str:
        if frame.agent and frame.action:
            sem: dict = {"agent": frame.agent, "action": frame.action}
            if frame.patient:
                sem["patient"] = frame.patient
            try:
                generated = self.parser.generate(sem)
                if generated and len(generated) >= 2:
                    return " ".join(generated)
            except Exception:
                pass

        prefix = list(frame.raw_words) if frame.raw_words else []
        if prefix and hasattr(self.parser, "continue_sentence"):
            extended = self.parser.continue_sentence(prefix, max_words=10)
            if len(extended) > len(prefix):
                return " ".join(extended)

        if frame.agent and frame.action:
            parts = [frame.agent, frame.action]
            if frame.patient:
                parts.append(frame.patient)
            return " ".join(parts)

        if hasattr(self.parser, "generate_novel_sentence"):
            novel = self.parser.generate_novel_sentence(min_words=3, max_words=8)
            if novel:
                return " ".join(novel)

        preds = self.parser.predict_next(frame.raw_words)
        if preds:
            return preds[0][0]
        return "ok"

    def last_tool_call(self) -> Optional[ToolCall]:
        for turn in reversed(self.history):
            if turn.tool_call is not None:
                return turn.tool_call
        return None

    def reset_dialogue(self) -> None:
        self.dialogue.clear()
        self.history.clear()
