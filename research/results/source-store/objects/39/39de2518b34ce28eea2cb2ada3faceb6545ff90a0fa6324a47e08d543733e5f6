"""One-shot codemod: legacy emergent shim imports -> canonical package paths."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REPLACEMENTS = [
    ("neural_assemblies.assembly_calculus.emergent._parser_blocks", "neural_assemblies.assembly_calculus.emergent.parser_mixins.blocks"),
    ("neural_assemblies.assembly_calculus.emergent._parser_core", "neural_assemblies.assembly_calculus.emergent.parser_mixins.core"),
    ("neural_assemblies.assembly_calculus.emergent._parser_dialogue", "neural_assemblies.assembly_calculus.emergent.parser_mixins.dialogue"),
    ("neural_assemblies.assembly_calculus.emergent._parser_distributional", "neural_assemblies.assembly_calculus.emergent.parser_mixins.distributional"),
    ("neural_assemblies.assembly_calculus.emergent._parser_generation", "neural_assemblies.assembly_calculus.emergent.parser_mixins.generation"),
    ("neural_assemblies.assembly_calculus.emergent._parser_incremental", "neural_assemblies.assembly_calculus.emergent.parser_mixins.incremental"),
    ("neural_assemblies.assembly_calculus.emergent._parser_instructions", "neural_assemblies.assembly_calculus.emergent.parser_mixins.instructions"),
    ("neural_assemblies.assembly_calculus.emergent._parser_morphosyntax", "neural_assemblies.assembly_calculus.emergent.parser_mixins.morphosyntax"),
    ("neural_assemblies.assembly_calculus.emergent._parser_plans", "neural_assemblies.assembly_calculus.emergent.parser_mixins.plans"),
    ("neural_assemblies.assembly_calculus.emergent._parser_prediction", "neural_assemblies.assembly_calculus.emergent.parser_mixins.prediction"),
    ("neural_assemblies.assembly_calculus.emergent._parser_structured", "neural_assemblies.assembly_calculus.emergent.parser_mixins.structured"),
    ("neural_assemblies.assembly_calculus.emergent._parser_unsupervised", "neural_assemblies.assembly_calculus.emergent.parser_mixins.unsupervised"),
    ("neural_assemblies.assembly_calculus.emergent.babble_curriculum", "neural_assemblies.assembly_calculus.emergent.acquisition.babble"),
    ("neural_assemblies.assembly_calculus.emergent.blocks_curriculum", "neural_assemblies.assembly_calculus.emergent.curriculum.blocks"),
    ("neural_assemblies.assembly_calculus.emergent.classification_bootstrap", "neural_assemblies.assembly_calculus.emergent.acquisition.pos_inference"),
    ("neural_assemblies.assembly_calculus.emergent.consolidation", "neural_assemblies.assembly_calculus.emergent.training.consolidation"),
    ("neural_assemblies.assembly_calculus.emergent.continual_learning", "neural_assemblies.assembly_calculus.emergent.acquisition.continual"),
    ("neural_assemblies.assembly_calculus.emergent.conversation_curriculum", "neural_assemblies.assembly_calculus.emergent.curriculum.conversation"),
    ("neural_assemblies.assembly_calculus.emergent.corpus_chat", "neural_assemblies.assembly_calculus.emergent.session.novel_chat"),
    ("neural_assemblies.assembly_calculus.emergent.corpus_index", "neural_assemblies.assembly_calculus.emergent.core.corpus_index"),
    ("neural_assemblies.assembly_calculus.emergent.dialogue_curriculum", "neural_assemblies.assembly_calculus.emergent.curriculum.dialogue"),
    ("neural_assemblies.assembly_calculus.emergent.dialogue_state", "neural_assemblies.assembly_calculus.emergent.session.dialogue_state"),
    ("neural_assemblies.assembly_calculus.emergent.grounding", "neural_assemblies.assembly_calculus.emergent.core.grounding"),
    ("neural_assemblies.assembly_calculus.emergent.interactive", "neural_assemblies.assembly_calculus.emergent.session.interactive"),
    ("neural_assemblies.assembly_calculus.emergent.phonology", "neural_assemblies.assembly_calculus.emergent.acquisition.phonology"),
    ("neural_assemblies.assembly_calculus.emergent.topology_linker", "neural_assemblies.assembly_calculus.emergent.training.linker"),
    ("neural_assemblies.assembly_calculus.emergent.training_batch", "neural_assemblies.assembly_calculus.emergent.training.batch"),
    ("neural_assemblies.assembly_calculus.emergent.training_compiled", "neural_assemblies.assembly_calculus.emergent.training.compiled"),
    ("neural_assemblies.assembly_calculus.emergent.training_compiler", "neural_assemblies.assembly_calculus.emergent.training.compiler"),
    ("neural_assemblies.assembly_calculus.emergent.training_data", "neural_assemblies.assembly_calculus.emergent.curriculum.data"),
    ("neural_assemblies.assembly_calculus.emergent.training_generalization", "neural_assemblies.assembly_calculus.emergent.evaluation.generalization"),
    ("neural_assemblies.assembly_calculus.emergent.training_parity", "neural_assemblies.assembly_calculus.emergent.evaluation.parity"),
    ("neural_assemblies.assembly_calculus.emergent.training_perf", "neural_assemblies.assembly_calculus.emergent.training.perf"),
    ("neural_assemblies.assembly_calculus.emergent.training_schedule", "neural_assemblies.assembly_calculus.emergent.training.schedule"),
    ("neural_assemblies.assembly_calculus.emergent.areas", "neural_assemblies.assembly_calculus.emergent.core.areas"),
    ("neural_assemblies.assembly_calculus.emergent.evaluation.erp_metrics", "neural_assemblies.assembly_calculus.emergent.evaluation.erp.gates"),
    ("neural_assemblies.assembly_calculus.emergent.evaluation.erp_probes", "neural_assemblies.assembly_calculus.emergent.evaluation.erp"),
    ("neural_assemblies.assembly_calculus.emergent.evaluation.erp_calibration", "neural_assemblies.assembly_calculus.emergent.evaluation.erp.calibration"),
]

SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules"}


def is_shim(path: Path, text: str) -> bool:
    if path.name == "bootstrap.py" and "acquisition" in path.parts:
        return True
    return text.lstrip().startswith('"""Backward-compatible shim')


def codemod_file(path: Path) -> bool:
    if path.name == "codemod_emergent_imports.py":
        return False
    text = path.read_text(encoding="utf-8")
    if is_shim(path, text):
        return False
    original = text
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed = []
    for path in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if codemod_file(path):
            changed.append(path.relative_to(ROOT))
    print(f"updated {len(changed)} files")
    for p in sorted(changed):
        print(f"  {p}")


if __name__ == "__main__":
    main()
