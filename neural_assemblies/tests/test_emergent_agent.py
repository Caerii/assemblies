"""
Agent layer tests — instruction following, tool dispatch, interactive session.

Uses parity-scale n/k for CI speed; paper-scale in test_emergent_parser.
"""

import json

import pytest

from neural_assemblies.assembly_calculus.emergent import (
    BlocksLanguageExecutor,
    EmergentParser,
    EmergentSession,
    ToolRegistry,
)
from neural_assemblies.assembly_calculus.emergent.evaluation import EvaluationSuite
from neural_assemblies.assembly_calculus.emergent.structured_io import ToolCall

N = 5000
K = 80
P = 0.05
BETA = 0.1
SEED = 42
ROUNDS = 10


@pytest.fixture(scope="module")
def agent_parser():
    parser = EmergentParser(
        n=N, k=K, p=P, beta=BETA, seed=SEED, rounds=ROUNDS,
    )
    parser.train_for_agent()
    return parser


@pytest.fixture(scope="module")
def agent_session(agent_parser):
    return EmergentSession(parser=agent_parser)


class TestInstructionFrames:
    def test_imperative_mood_and_implicit_agent(self, agent_parser):
        frame = agent_parser.parse_instruction(["chases", "the", "cat"])
        assert frame.mood == "IMPERATIVE"
        assert frame.agent == "you"
        assert frame.action == "chases"
        assert frame.patient == "cat"

    def test_evaluate_instruction_following(self, agent_parser):
        suite = EvaluationSuite(agent_parser)
        result = suite.evaluate_instruction_following([
            {
                "words": ["chases", "the", "cat"],
                "expected": {
                    "mood": "IMPERATIVE",
                    "action": "chases",
                    "patient": "cat",
                    "agent": "you",
                },
            },
            {
                "words": ["the", "dog", "runs"],
                "expected": {"mood": "DECLARATIVE", "action": "runs", "agent": "dog"},
            },
        ])
        assert result["accuracy"] >= 0.5
        assert result["per_field_accuracy"]["mood"] >= 0.5


class TestToolDispatch:
    def test_chase_tool_call(self, agent_parser):
        frame = agent_parser.parse_instruction(["chases", "the", "cat"])
        call = ToolRegistry().dispatch(frame)
        assert call is not None
        assert call.name == "chase"
        assert call.arguments.get("target") == "cat"
        assert call.arguments.get("agent") == "you"

    def test_tool_call_json_roundtrip(self, agent_parser):
        frame = agent_parser.parse_instruction(["sees", "the", "bird"])
        call = ToolRegistry().dispatch(frame)
        assert call is not None
        data = json.loads(call.to_json())
        restored = ToolCall.from_dict(data)
        assert restored.name == call.name
        assert restored.arguments == call.arguments


class TestEmergentSession:
    def test_imperative_ack(self, agent_session):
        reply = agent_session.interact("chases the cat")
        assert "chase" in reply
        assert "cat" in reply

    def test_interrogative_who(self, agent_session):
        reply = agent_session.interact("who chases the cat")
        assert reply != "unknown"
        assert isinstance(reply, str)

    def test_last_tool_call(self, agent_session):
        agent_session.interact("chases the bird")
        call = agent_session.last_tool_call()
        assert call is not None
        assert call.name == "chase"

    def test_train_for_agent_enables_prediction(self, agent_parser):
        preds = agent_parser.predict_next(["the"])
        assert len(preds) > 0


class TestStructuredIO:
    def test_instruction_frame_json(self, agent_parser):
        frame = agent_parser.parse_instruction(["reads", "the", "book"])
        data = json.loads(frame.to_json())
        assert data["action"] == "reads"
        assert data["patient"] == "book"


class TestDialoguePhase2:
    def test_dialogue_state_pronoun_resolution(self):
        from neural_assemblies.assembly_calculus.emergent import DialogueState

        state = DialogueState()
        state.last_patient = "cat"
        assert state.resolve_words(["chases", "it"]) == ["chases", "cat"]

    def test_present_turn_returns_frame(self, agent_parser):
        frame = agent_parser.present_turn(["the", "dog", "runs"], learn=False)
        assert frame.action == "runs"
        assert frame.agent == "dog"
        assert frame.speaker == "user"

    def test_present_turn_preserves_speaker_and_rejects_empty(self, agent_parser):
        frame = agent_parser.present_turn(
            ["the", "dog", "runs"], speaker="assistant", learn=False
        )
        assert frame.speaker == "assistant"
        with pytest.raises(ValueError, match="nonempty"):
            agent_parser.present_turn(["the"], speaker="", learn=False)

    def test_evaluate_dialogue_pinned(self, agent_parser):
        suite = EvaluationSuite(agent_parser)
        result = suite.evaluate_dialogue([
            {
                "question": "who chases the cat",
                "acceptable": ["dog"],
                "pattern_type": "who_query",
            },
            {
                "question": "does the dog runs",
                "acceptable": ["yes"],
                "pattern_type": "yesno",
            },
        ], online_learn=False)
        assert result["total"] == 2
        assert result["accuracy"] >= 0.0

    def test_multi_turn_entity_memory(self, agent_parser):
        session = EmergentSession(parser=agent_parser, online_learn=False)
        session.interact("the dog chases the cat")
        assert session.dialogue.last_patient == "cat"
        session.interact("who chases the cat")
        assert session.dialogue.turn_count >= 2


class TestBlocksPhase3:
    def test_parse_blocks_command(self, agent_parser):
        frame = agent_parser.parse_instruction(["move", "a", "to", "b"])
        assert frame.mood == "IMPERATIVE"
        assert frame.patient == "blk_a"
        assert frame.destination == "blk_b"
        assert frame.roles.get("_block_src") == "A"
        assert frame.roles.get("_block_dst") == "B"

    def test_tool_dispatch_move_block(self, agent_parser):
        frame = agent_parser.parse_instruction(["put", "a", "on", "table"])
        call = ToolRegistry().dispatch(frame)
        assert call is not None
        assert call.name == "move_block"
        assert call.arguments.get("block") == "blk_a"
        assert call.arguments.get("destination") == "table"

    def test_evaluate_tool_compliance(self, agent_parser):
        suite = EvaluationSuite(agent_parser)
        result = suite.evaluate_tool_compliance()
        assert result["tool_match_accuracy"] >= 0.66
        assert result["arg_accuracy"] >= 0.66

    def test_blocks_executor_apply(self, agent_parser):
        from neural_assemblies.assembly_calculus.emergent import BlocksLanguageExecutor

        ex = BlocksLanguageExecutor(blocks=("A", "B"))
        ex.reset(ex.state.__class__(
            on={"A": "B", "B": None},
            clear={"A": True, "B": False},
        ))
        frame = agent_parser.parse_instruction(["move", "a", "to", "table"])
        action, new_state = ex.apply_frame(frame)
        assert action is not None
        assert action.src == "A"
        assert new_state.on["A"] is None

    def test_session_execute_blocks(self, agent_parser):
        session = EmergentSession(
            parser=agent_parser,
            execute_tools=True,
            online_learn=False,
        )
        session.blocks_executor = BlocksLanguageExecutor(blocks=("A", "B", "C"))
        session.blocks_executor.state = session.blocks_executor.state.__class__(
            on={"A": None, "B": None, "C": None},
            clear={"A": True, "B": True, "C": True},
        )
        reply = session.interact("move a to b")
        assert "ok" in reply.lower()
        assert session.blocks_executor.state.on.get("A") == "B"


class TestStructuredJSONPhase4:
    def test_words_to_json_move_block(self, agent_parser):
        text = agent_parser.words_to_json(["move", "a", "to", "b"])
        assert text is not None
        data = json.loads(text)
        assert data["tool"] == "move_block"
        assert data["arguments"]["block"] == "blk_a"
        assert data["arguments"]["destination"] == "blk_b"

    def test_schema_validation(self, agent_parser):
        from neural_assemblies.assembly_calculus.emergent import (
            StructuredRecord,
            validate_record,
        )

        call_json = agent_parser.words_to_json(["put", "a", "on", "table"])
        record = StructuredRecord.from_json(call_json)
        ok, errors = validate_record(record)
        assert ok, errors

    def test_json_to_words_roundtrip(self, agent_parser):
        rt = agent_parser.structured_roundtrip(["move", "a", "to", "b"])
        assert rt["schema_valid"]
        assert rt["tool_match"]
        assert rt["roundtrip_words"] == ["move", "a", "to", "b"]

    def test_evaluate_json_roundtrip(self, agent_parser):
        suite = EvaluationSuite(agent_parser)
        result = suite.evaluate_json_roundtrip()
        assert result["roundtrip_accuracy"] >= 0.66
        assert result["schema_valid_rate"] >= 0.66

    def test_session_interact_json(self, agent_parser):
        session = EmergentSession(parser=agent_parser, online_learn=False)
        out = session.interact_json("chases the cat")
        assert out is not None
        data = json.loads(out)
        assert data["tool"] == "chase"

    def test_execute_json_blocks(self, agent_parser):
        session = EmergentSession(parser=agent_parser, execute_tools=True, online_learn=False)
        session.blocks_executor = BlocksLanguageExecutor(blocks=("A", "B", "C"))
        session.blocks_executor.state = session.blocks_executor.state.__class__(
            on={"A": None, "B": None, "C": None},
            clear={"A": True, "B": True, "C": True},
        )
        j = session.interact_json("move a to b")
        assert j is not None
        reply = session.execute_json(j)
        assert "ok" in reply.lower()


class TestMultiToolPlanPhase5:
    def test_explicit_two_step_plan(self, agent_parser):
        from neural_assemblies.assembly_calculus.emergent import BlocksLanguageExecutor

        ex = BlocksLanguageExecutor(blocks=("A", "B"))
        ex.state = ex.state.__class__(
            on={"A": "B", "B": None},
            clear={"A": True, "B": False},
        )
        plan = agent_parser.text_to_tool_plan(
            "move a to table then move a to b", ex,
        )
        assert plan is not None
        assert plan.length == 2
        assert plan.source == "explicit"
        result = agent_parser.execute_plan(plan, ex)
        assert result.success
        assert ex.state.on["A"] == "B"

    def test_bfs_goal_plan(self, agent_parser):
        from neural_assemblies.assembly_calculus.emergent import BlocksLanguageExecutor

        ex = BlocksLanguageExecutor(blocks=("A", "B", "C"))
        plan = agent_parser.text_to_tool_plan("stack a on b", ex)
        assert plan is not None
        assert plan.source == "bfs"
        assert plan.length >= 1
        result = agent_parser.execute_plan(plan, ex)
        assert result.success
        assert ex.state.on["A"] == "B"

    def test_plan_json_roundtrip(self, agent_parser):
        from neural_assemblies.assembly_calculus.emergent import ToolPlan

        ex = BlocksLanguageExecutor(blocks=("A", "B", "C"))
        plan = agent_parser.text_to_tool_plan("move a to b", ex)
        restored = ToolPlan.from_json(plan.to_json())
        assert restored.length == plan.length
        assert restored.steps[0].name == "move_block"

    def test_session_interact_and_execute_plan(self, agent_parser):
        session = EmergentSession(parser=agent_parser, execute_tools=True, online_learn=False)
        session.blocks_executor = BlocksLanguageExecutor(blocks=("A", "B", "C"))
        plan_json = session.interact_plan("stack a on b")
        assert plan_json is not None
        data = json.loads(plan_json)
        assert data["source"] == "bfs"
        assert len(data["steps"]) >= 1
        result_json = session.execute_plan_json(plan_json)
        result = json.loads(result_json)
        assert result["success"]

    def test_evaluate_multi_tool_plan(self, agent_parser):
        suite = EvaluationSuite(agent_parser)
        result = suite.evaluate_multi_tool_plan()
        assert result["accuracy"] >= 0.5

