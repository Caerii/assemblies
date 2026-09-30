"""
Emergent NEMO parser — 48-area architecture on numpy_sparse.

Ports the full emergent NEMO architecture (Mitropolsky & Papadimitriou 2025)
from cupy/CUDA to the proven numpy_sparse engine. Categories emerge from
grounding patterns, not hardcoded labels.

Package layout::

    core/           — areas, grounding, corpus index, GroundedSentence
    parser_mixins/  — runtime capabilities composed into EmergentParser
    curriculum/     — staged training content and CurriculumTrainer
    training/       — compile → link → run pipeline
    acquisition/    — developmental babble → grammar orchestration
    evaluation/     — erp/ (gates, adapters, runner, frames, calibration)
                      generalization gates, parity probes
    session/        — interactive chat and novel-corpus training

Agent path (chat / instruction / tools): see ``ROADMAP.md``.

Usage::

    from neural_assemblies.assembly_calculus.emergent import EmergentParser

    parser = EmergentParser(n=10000, k=100, seed=42)
    parser.train_for_agent()
    result = parser.parse(["the", "big", "dog", "chases", "a", "cat"])
"""

from .parser import (
    EmergentParser,
    CurriculumTrainer,
    StageResult,
    EvaluationSuite,
)
from .core import GroundingContext, GroundedSentence
from .curriculum import (
    create_instruction_sentences,
    generate_training_sentences,
)
from .session import EmergentSession, Turn, DialogueState
from .vocabulary_builder import (
    build_vocabulary,
    build_vocabulary_preset,
    VOCAB_PRESETS,
    verb_surface_form,
    words_by_modality,
)
from .train_progress import (
    TrainProgress,
    current_progress,
    ensure_progress,
    finish_progress,
    progress_enabled,
    start_progress,
)
from .curriculum.conversation import (
    ConversationScript,
    create_conversation_pairs,
    get_conversation_curriculum,
    get_conversation_pairs,
)
from .blocks_bridge import BlocksLanguageExecutor, frame_to_blocks_action
from .curriculum.blocks import (
    blocks_compliance_test_cases,
    create_blocks_instruction_sentences,
    blocks_vocabulary,
)
from .tools import ToolRegistry, ToolSpec, DEFAULT_TOOL_SPECS
from .structured_io import InstructionFrame, ToolCall
from .structured_json import StructuredRecord, validate_record, structured_to_words
from .json_schemas import JsonSchema, TOOL_CALL_SCHEMA, SCHEMA_REGISTRY
from .curriculum.dialogue import DialoguePair, get_dialogue_curriculum, get_dialogue_pairs
from .tool_plan import ToolPlan, ToolPlanResult, tokenize_text
from .core import (
    ALL_AREAS,
    CORE_AREAS,
    CORE_TO_CATEGORY,
    CATEGORY_TO_CORE,
    GROUNDING_TO_CORE,
    PHRASE_AREAS,
    THEMATIC_AREAS,
    CONTEXT,
    PRODUCTION,
    PREDICTION,
    DEP_CLAUSE,
)
from .training import (
    build_role_pathway_protocol,
    build_vp_pathway_protocol,
    consolidate_role_pathways,
    consolidate_vp_pathways,
)

__all__ = [
    "EmergentParser",
    "CurriculumTrainer",
    "StageResult",
    "EvaluationSuite",
    "EmergentSession",
    "Turn",
    "DialogueState",
    "DialoguePair",
    "get_dialogue_curriculum",
    "get_dialogue_pairs",
    "BlocksLanguageExecutor",
    "frame_to_blocks_action",
    "blocks_compliance_test_cases",
    "create_blocks_instruction_sentences",
    "blocks_vocabulary",
    "ToolRegistry",
    "ToolSpec",
    "DEFAULT_TOOL_SPECS",
    "InstructionFrame",
    "ToolCall",
    "StructuredRecord",
    "JsonSchema",
    "TOOL_CALL_SCHEMA",
    "SCHEMA_REGISTRY",
    "validate_record",
    "structured_to_words",
    "ToolPlan",
    "ToolPlanResult",
    "tokenize_text",
    "GroundingContext",
    "GroundedSentence",
    "create_instruction_sentences",
    "build_vocabulary",
    "build_vocabulary_preset",
    "VOCAB_PRESETS",
    "verb_surface_form",
    "words_by_modality",
    "ConversationScript",
    "create_conversation_pairs",
    "get_conversation_curriculum",
    "get_conversation_pairs",
    "TrainProgress",
    "start_progress",
    "finish_progress",
    "progress_enabled",
    "generate_training_sentences",
    "ALL_AREAS",
    "CORE_AREAS",
    "CORE_TO_CATEGORY",
    "CATEGORY_TO_CORE",
    "GROUNDING_TO_CORE",
    "PHRASE_AREAS",
    "THEMATIC_AREAS",
    "CONTEXT",
    "PRODUCTION",
    "PREDICTION",
    "DEP_CLAUSE",
    "build_role_pathway_protocol",
    "build_vp_pathway_protocol",
    "consolidate_role_pathways",
    "consolidate_vp_pathways",
]
