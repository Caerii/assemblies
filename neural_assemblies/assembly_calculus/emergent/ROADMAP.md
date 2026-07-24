# EmergentParser → Agent Roadmap

## Phases 1–5 ✓
Agent I/O, chat, blocks-world, JSON structured output, multi-tool plans.

## Phase 6 — Scaled language & conversation (in progress)

### Vocabulary scaling
- `build_vocabulary_preset("core" | "medium" | "large" | "discussion")`
- Merges lexicon-frequency words with closed agent `VOCABULARY`
- `discussion` preset tuned for multi-turn chat

### Deep curriculum path
- `CurriculumTrainer.train_conversation_path(max_stage="DIALOGUE" | "CONVERSATION")`
- New stages: **DIALOGUE** (Q-A bridges), **CONVERSATION** (multi-turn scripts)
- `parser.train_for_conversation(preset via init vocabulary, max_stage=...)`

### Developmental training path (preferred for composition probes)
Ground-up CDS curriculum with babble, fuzzy surfaces, exit gates, and adaptive remediation.
**Not** the fast chat bootstrap (`chat_emergent.py` skips early stages when vocab ≥ 80).

```python
import os
os.environ["EMERGENT_DEV_CURRICULUM"] = "1"

from neural_assemblies.assembly_calculus.emergent import EmergentParser, build_vocabulary_preset
from neural_assemblies.assembly_calculus.emergent.acquisition import run_developmental_acquisition

parser = EmergentParser(n=3000, k=30, vocabulary=build_vocabulary_preset("core"))
report = run_developmental_acquisition(parser, max_stage="SENTENCES")
```

```bash
# CLI: writes report + metrics JSON
python examples/train_developmental.py --max-stage SENTENCES --preset core --fast
```

- Stage exit gates in `acquisition/stage_gates.py` (composition + holdout floors at SENTENCES)
- CDS corpora wired for `FIRST_WORDS` / `VOCABULARY_SPURT` / `TWO_WORD` via `lexicon/curriculum/`
- Inter-stage plasticity-off replay when `EMERGENT_DEV_CURRICULUM=1`
- SENTENCES includes **prediction** phase + holdout bridge corpus (`holdout_bridge_corpus.py`)
- Science battery: `evaluation/composition_battery.py`; compare: `research/experiments/compare_training_paths.py`

### Chat entry points (fast path — not for developmental eval)
```python
from neural_assemblies.assembly_calculus.emergent import (
    EmergentSession, build_vocabulary_preset, EmergentParser,
)

# Full path: scaled vocab + curriculum + agent tools
session = EmergentSession.bootstrap_conversation(
    preset="medium", max_stage="DIALOGUE",
)
reply = session.interact("who chases the cat")

# CLI
# python examples/chat_emergent.py --preset discussion --stage CONVERSATION
```

### Performance (Phase 6b)

- `training_perf.py` — engine auto-select, round budgets, stage skips
- `train_progress.py` — streaming stderr progress (`TRAIN_PROGRESS=1`)
- `classify_word_cached`, `_ensure_prediction_lexicon`, incremental circuit reuse
- `EMERGENT_FAST_TRAINING=1` — fewer rounds / sequence reps
- Preset vocab skips early curriculum stages when `len(vocab) >= 80`

```bash
TRAIN_PROGRESS=1 EMERGENT_FAST_TRAINING=1 python examples/chat_emergent.py --preset medium
python research/experiments/benchmark_emergent_training.py --quick
```
1. Larger presets + `CONVERSATION` stage for narrative follow-ups
2. Online learning during chat (`online_learn=True`, unknown-word `register_word`)
3. Open-domain sentence ingestion (`train_from_text`)
4. Neural blocks loop (`BlocksWorldAC.solve_and_run`)

## Phase 7 — Neural VM (parallel worktree)
Opcode layer: ToolCall → FiberCircuit traces; arithmetic syscalls.

## Agent / tools entry points

```python
session = EmergentSession(parser=parser, execute_tools=True)
plan_json = session.interact_plan("stack a on b")
result_json = session.execute_plan_json(plan_json)
```
