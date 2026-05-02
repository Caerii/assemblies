# Documentation Style

This repo needs prose that is clear enough for engineers and careful enough for
researchers. The writing should feel authored, not generated.

This guide follows the same basic direction as the Google developer
documentation style guide and the Microsoft Writing Style Guide: use active
voice, simple words, concise sentences, and reader-centered structure. Break
those rules only when the local science or code needs more precision.

## Voice

Write like a maintainer explaining the repo to a serious reader.

- Prefer concrete nouns over abstract labels.
- Prefer short claims with evidence links over broad positioning.
- Avoid filler openings such as "This document describes" when the heading
  already says what the file is.
- Do not repeat "surface", "contract", "current", or "package-backed" when a
  plainer word works.
- Cut sentences that only announce importance, lineage, ambition, or
  seriousness. Replace them with the actual fact.
- Keep personal context where it matters: authorship, provenance,
  collaboration, and motivation.

Example:

```text
Weak: That lineage matters: the repo is not an anonymous port of an idea.
Better: Alif Jakir maintains Assemblies and continued the work in collaboration
with Daniel Mitropolsky.
```

## Claim Discipline

Every strong sentence should have one of four anchors:

- package behavior: code and tests
- measured research: experiment, result artifact, or claim index
- theory: cited paper
- future direction: explicitly labeled as research work

If no anchor exists, weaken the sentence or remove it.

## Structure

Lead with the reader's question.

Good order for most docs:

1. What is this?
2. When should I use it?
3. What is stable?
4. What is experimental or legacy?
5. What commands or files matter?

Avoid long inventories unless the reader needs a reference table.

## Research Writing

Research docs should distinguish:

- hypothesis
- protocol
- result
- interpretation
- limitation
- next experiment

Do not turn a promising observation into a general law. State the parameter
regime, dataset, seed policy, and result artifact when possible.

## API Writing

API docs should show the import first, then the object, then the caveat.

Bad pattern:

```text
This module provides a flexible surface for...
```

Better pattern:

```python
from neural_assemblies.core.brain import Brain
```

Use `Brain` when you need areas, stimuli, routing, and projection cycles.

## Words To Use Sparingly

- surface
- contract
- aspirational
- current
- robust
- scalable
- biologically plausible
- foundational

These words are sometimes correct. They become noise when they substitute for
the actual object, test, or result.

## External References

- Google developer documentation style guide:
  <https://developers.google.com/style/highlights>
- Google voice and tone:
  <https://developers.google.com/style/tone>
- Google active voice:
  <https://developers.google.com/style/voice>
- Microsoft Writing Style Guide, simple words and concise sentences:
  <https://learn.microsoft.com/en-us/style-guide/word-choice/use-simple-words-concise-sentences>
