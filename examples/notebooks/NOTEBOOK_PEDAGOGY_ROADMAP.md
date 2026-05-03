# Notebook Pedagogy Roadmap

Status: planning note for teaching notebooks and reusable helpers.

This note preserves the curriculum and visualization ideas behind the notebook
rewrite. It is not a scientific result. It records how the notebooks should
teach the package and where new helper code should live.

## 1. First Principle

The notebooks should not feel like API demos. They should feel like small
experiments where the reader can see activity forming, stabilizing, binding,
failing, and recovering.

The reader should learn to ask:

- What is active?
- How sparse is it?
- Did it stabilize?
- What changed from the previous round?
- Does it overlap with the intended source, target, or reference assembly?
- Is that overlap above chance?
- Which parameter or perturbation moved the result?

The notebooks should teach intuition before formalism, but every intuition
should point to an inspectable trace, plot, table, or diagnostic.

## 2. Volume 01 As The First Encounter

Volume 01 is the reader's first real contact with the system. It should be
concrete and memorable.

The first notebook should use a concept-binding scene rather than abstract
placeholder names:

| Cue | Area | Assembly Role |
|-----|------|---------------|
| `red` | `COLOR` | color cue assembly |
| `triangle` | `SHAPE` | shape cue assembly |
| `red triangle` | `OBJECT` | bound object assembly |

The teaching arc should be:

1. A stimulus leaves a sparse trace.
2. Projection turns that cue into a stable winner set.
3. A second cue forms a second winner set.
4. Merge binds the two source traces into a target trace.
5. Diagnostics show what changed and what can be recovered.

The strongest first-contact claim is modest:

> In this toy system, a sparse winner set can stabilize under projection, and a
> target winner set can be driven by two source assemblies through merge.

Avoid claims that the notebook demonstrates real color perception, shape
understanding, semantic binding, cortical anatomy, or object recognition.

The visuals are index maps and process diagrams. They are not brain maps.

## 3. What Each Volume Should Teach

The notebook curriculum should grow in carefully controlled layers.

| Volume | Question | Main Lesson |
|--------|----------|-------------|
| 01 foundations | What is an assembly? | Sparse traces, projection, merge, readout, perturbation, and overlap. |
| 02 memory and computation | How does activity move over time? | Sequence traces, LRI, refractory effects, automata utilities, and parameter sensitivity. |
| 03 language | What can composition look like in toy language? | Category, role, order, parser traces, and explicit failure surfaces. |
| 04 research workflow | When does a demo become evidence? | Claim discipline, experiment artifacts, status indexes, and limitations. |

Future volumes should be added only when they introduce a new inspectable
mechanism rather than merely a bigger demo.

Good future notebook directions:

- basin-of-attraction maps from partial cues,
- interference between multiple bound objects,
- capacity and crowding as more assemblies are added,
- finite-size scaling labs for `N`, `K`, `p`, and `beta`,
- phase diagrams for stability and recall regimes,
- recurrent pattern-completion drills,
- sequence drift and LRI failure cases,
- lexicon threshold and damage labs,
- parser ambiguity and error-analysis notebooks,
- research-result reproduction notebooks that load saved artifacts instead of
  rerunning expensive sweeps.

## 4. Animation Philosophy

Animations should show neurodynamical process, not decorate a static result.

Use animation when a before/after plot would hide the lesson:

- winner turnover across projection rounds,
- convergence toward a stable assembly,
- merge target formation under two fixed sources,
- pattern completion from a damaged cue,
- LRI recall stepping through a sequence,
- drift, cycling, or collapse under bad parameters.

Every animation should have a nearby static diagnostic. The animation shows
"what happened"; the table or plot says "how much."

Useful paired views:

| Animation | Paired Static Diagnostic |
|-----------|--------------------------|
| winner grid over rounds | consecutive overlap and new-winner count |
| merge target animation | source-response overlap table |
| damaged cue recovery | recovery overlap curve |
| LRI recall animation | recall-vs-reference overlap matrix |
| parameter sweep animation | final stability heatmap |

The reader should never have to infer success from motion alone.

## 5. Animation Library Choices

Default notebook animations should stay lightweight:

- Matplotlib `FuncAnimation` for winner grids and trace playback.
- Jupyter `HTML(animation.to_jshtml())` when embedding the animation in a
  notebook output is useful.
- `ipywidgets` only for actual controls such as round sliders, parameter
  selectors, or seed selectors.

Optional richer tools are useful later, but should not become required for the
basic package:

- Plotly for interactive phase diagrams and hoverable sweep surfaces.
- NetworkX plus Matplotlib or Plotly for area graphs and operation flows.
- Manim only for polished explanatory videos, not ordinary notebook execution.
- JavaScript canvas only if Matplotlib becomes too slow for large trace
  playback.

Do not add heavy visual dependencies to normal package imports. Keep them in
the notebook dependency group or optional extras.

## 6. Trace Helpers Are First-Class Pedagogy

Tracing is not merely debugging. It is the way the notebooks make dynamics
inspectable.

Reusable trace behavior belongs under:

```text
neural_assemblies/assembly_calculus/tracing/
```

That package should own:

- trace data models,
- traced operation wrappers,
- trace diagnostics,
- small sweep helpers,
- conversion to records suitable for notebook tables.

Compatibility imports such as `neural_assemblies.assembly_calculus.trace` can
remain thin, but new trace behavior should be compositional and kept in the
tracing subpackage.

Reusable visualization behavior belongs under:

```text
neural_assemblies/viz/
neural_assemblies/viz/tracing/
```

Trace-specific plotting and animation should stay under `viz/tracing/`.
General grids, overlap plots, flow diagrams, and sweep plots should remain
separate modules. Avoid turning `viz/__init__.py` or a single plotting file into
a large mixed-purpose module.

## 7. Notebook Writing Standards

Notebook prose should be direct and curious. It should not inflate toy results.

Use:

- concrete labels such as `red`, `triangle`, `COLOR`, `SHAPE`, and `OBJECT`,
- visible parameters near the top,
- deterministic seeds unless stochasticity is the topic,
- tables for cast-of-characters and trace summaries,
- explicit chance baselines where overlap interpretation needs them,
- `plt.show()` and `plt.close(fig)` to avoid duplicate displays,
- short markdown cells that tell the reader what to inspect next.

Avoid:

- abstract placeholders such as `s1`, `A1`, and `B` in first-contact notebooks,
- comments that explain obvious Python syntax,
- comments apologizing for notebook display behavior,
- broad claims from one seed,
- plots without a nearby number or diagnostic,
- animations without static metrics,
- hidden parameters,
- large notebooks that introduce several new concepts at once.

Code comments should explain intent, parameter choice, or interpretation. They
should not narrate the syntax.

## 8. Volume 01 Detailed Expansion Ideas

Volume 01 can still become stronger.

Add or improve:

- A cast-of-characters table before code runs.
- A conceptual flow diagram for `red -> COLOR`, `triangle -> SHAPE`, and
  `COLOR + SHAPE -> OBJECT`.
- A sparse grid map for each assembly.
- Trace metric plots for projection and merge.
- Winner-turnover heatmaps.
- A small "what changed?" diagnostic after merge.
- A source-response probe: reactivate one source and inspect whether the target
  response resembles the bound object assembly.
- A chance-overlap explanation only where the comparison needs a baseline.
- A final "try next" section that changes one variable at a time.

Potential first-notebook sequence:

1. Build the brain with named areas.
2. Project `red` into `COLOR` and animate stabilization.
3. Project `triangle` into `SHAPE` and inspect sparse winners.
4. Merge `COLOR` and `SHAPE` into `OBJECT`.
5. Compare source and object assemblies by overlap and density.
6. Replay a source-response diagnostic.
7. Perturb one parameter and observe whether stability changes.

## 9. Volume 02 Detailed Expansion Ideas

Volume 02 should make time and control visible.

Add or improve:

- sequence memorization traces with a clear timeline,
- LRI recall animation showing accepted steps,
- refractory-period and inhibition-strength parameter labs,
- plots showing when recall terminates early,
- examples where LRI helps and examples where it fails,
- deterministic automata traces side by side with assembly-coded state,
- probabilistic automata sample histograms with seed-visible stochasticity.

The point is not that every sequence works. The point is to expose the regime
where recall is stable, where it drifts, and where the control policy fails.

## 10. Volume 03 Detailed Expansion Ideas

Volume 03 should avoid claiming broad language ability. It should show how toy
language computations can be made inspectable.

Add or improve:

- sentence parse traces with category and role assemblies,
- unknown-word failure cases,
- swapped-word-order failure cases,
- role-confusion diagnostics,
- ambiguity examples where two parses compete,
- readout threshold sensitivity,
- comparisons between maintained parser APIs and legacy material.

Every language notebook should include a "what this does not show" section.

## 11. Volume 04 Detailed Expansion Ideas

Volume 04 should teach research hygiene.

Add or improve:

- a walkthrough from notebook observation to experiment protocol,
- a checklist for claim promotion,
- examples of weak claims and repaired claims,
- links between tests, experiment scripts, result artifacts, and claims,
- a failure-case writeup that becomes more valuable because it is measured.

This volume should make the repo harder to misuse in papers, talks, and README
copy.

## 12. Connection To The Systems Paper

The notebooks and the systems paper should reinforce each other.

The notebooks are the reader-facing instruments:

- traces,
- animations,
- overlap tables,
- parameter labs,
- failure cases.

The paper is the claim-bearing synthesis:

- order parameters,
- phase regimes,
- reliability curves,
- finite-size scaling,
- basin maps,
- composition and interference,
- limits and falsification criteria.

Notebook outputs should not be cited as evidence unless they are tied to a
registered experiment or result artifact. Their main job is to make the
mechanisms legible enough that the experiments and claims are easier to trust.

## 13. Short Version

Make the notebooks vivid, but keep the claims careful.

Show activity moving, stabilizing, binding, and failing. Then give the reader
the numbers that explain what they just saw.
