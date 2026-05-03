# Notebook Curriculum

These notebooks are a guided first tour of the package. They are meant to make
assembly calculus feel concrete: a stimulus leaves a sparse trace, traces can
be compared, labels can be read out, sequences can be inspected, and small
structured computations can be followed step by step.

The tone should be curious and hands-on, not grandiose. Treat each notebook as
a small instrument on the bench. Run it, look at the numbers, change one
parameter, and ask what moved.

Read the volumes in order if you are new to the package.
Volumes 01-04 are the core path. Volumes 05-14 are thematic tracks for the
larger research program.

For the longer teaching and visualization roadmap, see
`NOTEBOOK_PEDAGOGY_ROADMAP.md`.

For the full notebook environment:

```bash
uv sync --group notebooks
uv run jupyter lab
```

The package itself does not require the richer notebook stack. Keep reusable
trace helpers in `neural_assemblies.assembly_calculus.tracing`, reusable
visualization helpers in `neural_assemblies.viz`, and optional interactive
tools in notebooks.

## Volumes

- `volume-01-foundations`: Brain setup, projection, merge, lexicons, and
  readout, plus parameter and pattern-completion labs. You learn how
  assemblies are formed, inspected, and stress-tested.
- `volume-02-memory-and-computation`: sequence memory, LRI diagnostics, FSMs,
  PFAs, and LRI parameter sweeps. You learn how assemblies can carry state
  over time.
- `volume-03-language`: controlled NEMO-style parsing examples that run on the
  maintained assembly-calculus stack. You learn how category, role, and order
  can be composed in a toy language setting.
- `volume-04-research-workflow`: how to inspect claims, experiments, and
  scientific status without turning demos into evidence. You learn how to keep
  software demonstrations separate from scientific conclusions and how to trace
  an observation toward a claim.
- `volume-05-dynamics-and-statistical-mechanics`: order parameters,
  stabilization regimes, scaling, and phase-diagram thinking.
- `volume-06-binding-composition-and-interference`: binding diagnostics,
  source-response checks, composition reliability, and interference.
- `volume-07-control-circuits-and-runtime`: fiber gating, operation schedules,
  runtime protocols, and controlled information flow.
- `volume-08-prediction-and-erp-signals`: toy next-token prediction plus the
  research boundary around N400/P600-style signal claims.
- `volume-09-embodiment-and-grounding`: object/action/word grounding maps,
  missing encoders and readouts, and future embodied tasks.
- `volume-10-biological-validation-and-connectomes`: biological parameter
  alignment, neural-data validation levels, and mapped-connectome hypotheses.
- `volume-11-scaling-engines-and-performance`: backend selection, winner
  policies, sparse compute, and hardware-sensitive performance work.
- `volume-12-systems-paper-and-theory`: paper-claim maps, theory synthesis,
  limitation gates, and figure/evidence planning.
- `volume-13-programming-assemblies`: transition objects, operation graphs,
  fiber schedules, and the neural-programming-language frame.
- `volume-14-frontiers-and-open-problems`: web-scale curricula, compiler
  ideas, connectomes, robotics, self-assembling substrates, and open-problem
  triage.

## Reader Path

By the end, a reader should be able to:

- create a brain with named areas and stimuli
- form an assembly and compare it to another assembly
- build a tiny label-to-assembly lexicon
- inspect a memorized sequence and understand why recall is parameter-sensitive
- run a small projection or LRI parameter sweep without hiding the regime
- run a deterministic and probabilistic automaton helper
- parse a toy sentence through the maintained NEMO-style parser
- identify parser failure surfaces before making broader language claims
- find the evidence behind a claim before repeating it
- map a larger research idea to the correct evidence level before turning it
  into a notebook, experiment, or paper claim

## Standards

- Start each notebook with the question it answers.
- Keep parameters visible near the top.
- Use deterministic seeds unless the notebook is specifically about
  stochasticity.
- Print overlaps, trajectories, or parse dictionaries instead of only saying
  that something "worked."
- Prefer trace helpers such as `project_trace` and `merge_trace` when the
  reader needs to see dynamics over rounds rather than only final snapshots.
- Use animations for neurodynamical processes when a static before/after plot
  would hide the interesting transition.
- State whether the notebook demonstrates a package API, an experimental
  behavior, or a literature result.
- Use code comments to explain intent, parameters, and what to inspect. Avoid
  comments that merely repeat Python syntax.
- End with a small "try next" prompt so readers can safely explore.

Avoid notebooks that make broad claims from a single seed, hide parameters, or
mix legacy artifacts with maintained package APIs.
