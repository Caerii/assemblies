# `neural_assemblies/nemo/` -- 24,126 lines, superseded on every axis checked

An audit, not a deletion. The subtree is 84 files and ~27% of the package by
line count, and it is reachable from exactly ONE place: a docs smoke test doing
`from neural_assemblies.nemo.language import LanguageLearner, SentenceGenerator`.

The `test_nemo_*.py` files do NOT rescue it -- they import from `programs/`,
`assembly_calculus/` and `core/`. The name overlaps; the package does not.

## What I checked, and what I found

I went looking for capability the live tree lacks, because deleting a parallel
implementation without reading it is how a good idea gets lost. Four candidates
looked promising from their filenames. Three are superseded and one is
off-substrate.

### Grounding -- SUPERSEDED, and it is the design we moved AWAY from

`language/emergent/params.py` defines `GroundingContext` as a **modality to
category map**:

    visual -> NOUN,  motor -> VERB,  properties -> ADJECTIVE,
    spatial -> PREPOSITION,  social -> PRONOUN,  temporal -> ADVERB

That is the annotation-driven design [[categories-are-annotation-driven]]
measured as circular: the category is GIVEN by the modality tag, not learned.
It is also the map that `research/notes/language/literature_clues_for_the_three_obstacles.md`
records as citing Mitropolsky & Papadimitriou 2025 for something the paper does
not contain.

The live tree's `assembly_calculus/emergent/core/scene.py` supersedes it
explicitly -- `SceneEvent` holds participants in CAUSAL order and says "Feature
bundles, never words". [[grounded-role-learning]] is the record of that being
the better idea.

**Do not port.**

### Mood generators -- SUPERSEDED

`curriculum/generators/{declarative,interrogative,self_referential}.py` looked
like the most valuable thing here, because [[word-order-is-degenerate]] says
multi-mood is the prerequisite for the compositional-advantage claim and "the
repo lacks it".

It does not. `assembly_calculus/emergent/curriculum/data.py` already emits
`GroundedSentence(..., mood="interrogative")`, seven live modules handle
interrogatives, `MOOD` is a real control area in `core/areas.py`, and
`reference/word_order_learner.py` carries full `mood_orders`, `num_moods` and
`per_mood_syntax`. The live tree has strictly more.

**Do not port.**

### Hopfield associative memory -- UNIQUE, and off the substrate

`generation/hopfield_memory.py` is the one thing with no counterpart: zero hits
for "hopfield" anywhere in the live tree. Modern Hopfield / attention-equivalent
content-addressable memory, storing `(key_assembly, value_assembly)` pairs and
retrieving by dot-product similarity plus softmax.

The idea is relevant -- #92 (the FSM arc drops the state) and
[[role-retrieval-works-key-retracted]] both want content-addressable retrieval
without string keys.

But the implementation is `assembly_to_dense` -> numpy/cupy softmax ->
`dense_to_assembly`. No `Brain`, no `project`, no areas in the mechanism. It
answers "what can attention do on vectors we happened to obtain from an assembly
substrate", which is the wrong side of this project's line: the question is what
the SUBSTRATE can do. It also `import cupy as cp` at module scope, which
[[cupy-torch-import-order]] records as breaking torch CUDA outright.

**Record the idea; do not port the code.** If content-addressable retrieval is
wanted, it has to be built from projection and k-WTA, and that is a research
task, not a move.

### `archive/` -- self-describing

Eight `*_v1.py` files in a directory named `archive`. No further audit needed.

## `text_generation/` -- 1,910 lines, zero importers

Three modules describing themselves as prototypes, built "based on the
architecture from parser.py, recursive_parser.py, and learner.py" -- which live
in `legacy/root_modules/`. Superseded by
`assembly_calculus/emergent/parser.py`. No unique mechanism found.

## Recommendation

Remove `nemo/` and `text_generation/`, in a reviewable commit of their own,
after the one docs-smoke-test import is retargeted or dropped. That is ~26,000
lines, and the argument for it is not tidiness: **a parallel implementation is
[[same-name-two-meanings]] at maximum scale.** Two `EmergentLearner`s exist, and
nothing prevents a future result from being produced by the wrong one.

Not done here, per [[dead-code-deletion-caution]] -- deleting 26k tracked lines
is a decision to take deliberately, with the audit above as its justification
rather than a line count.

One consequence worth noting: `assembly_calculus/emergent/training_data.py` is a
deprecation shim kept alive ONLY by `nemo/language/emergent/profiler.py`. It
becomes dead the moment nemo goes.
