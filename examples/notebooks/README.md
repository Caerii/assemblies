# Notebook Curriculum

These notebooks are a guided first tour of the package. They are meant to make
assembly calculus feel concrete: a stimulus leaves a sparse trace, traces can
be compared, labels can be read out, sequences can be inspected, and small
structured computations can be followed step by step.

The tone should be curious and hands-on, not grandiose. Treat each notebook as
a small instrument on the bench. Run it, look at the numbers, change one
parameter, and ask what moved.

Read the volumes in order if you are new to the package.

## Volumes

- `volume-01-foundations`: Brain setup, projection, merge, lexicons, and
  readout. You learn how assemblies are formed and inspected.
- `volume-02-memory-and-computation`: sequence memory, LRI diagnostics, FSMs,
  and PFAs. You learn how assemblies can carry state over time.
- `volume-03-language`: controlled NEMO-style parsing examples that run on the
  maintained assembly-calculus stack. You learn how category, role, and order
  can be composed in a toy language setting.
- `volume-04-research-workflow`: how to inspect claims, experiments, and
  scientific status without turning demos into evidence. You learn how to keep
  software demonstrations separate from scientific conclusions.

## Reader Path

By the end, a reader should be able to:

- create a brain with named areas and stimuli
- form an assembly and compare it to another assembly
- build a tiny label-to-assembly lexicon
- inspect a memorized sequence and understand why recall is parameter-sensitive
- run a deterministic and probabilistic automaton helper
- parse a toy sentence through the maintained NEMO-style parser
- find the evidence behind a claim before repeating it

## Standards

- Start each notebook with the question it answers.
- Keep parameters visible near the top.
- Use deterministic seeds unless the notebook is specifically about
  stochasticity.
- Print overlaps, trajectories, or parse dictionaries instead of only saying
  that something "worked."
- State whether the notebook demonstrates a package API, an experimental
  behavior, or a literature result.
- End with a small "try next" prompt so readers can safely explore.

Avoid notebooks that make broad claims from a single seed, hide parameters, or
mix legacy artifacts with maintained package APIs.
