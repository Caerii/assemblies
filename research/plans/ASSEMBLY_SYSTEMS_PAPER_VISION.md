# Neural Assembly Systems Paper Vision

Status: planning note, not evidence.

This document records the current conceptual frame for a long systems paper
around neural assembly learning dynamics, statistical mechanics, and
compositional cognitive computation. It is intentionally ambitious, but it is
not a claim document. Claims must still be backed by registered experiments,
result artifacts, limitations, and falsification criteria.

## 1. Core Thesis

The strongest version of the paper is not simply:

> We implemented assembly-calculus demos.

The stronger thesis is:

> Sparse Hebbian winner-take-all networks support learned macrostates whose
> stability, recovery, and compositional transitions can be characterized by
> measurable order parameters; those regimes define a substrate for
> assembly-based cognitive computation.

An even shorter framing:

> Cognition can be studied as controlled motion through learned sparse
> neural macrostates.

The paper should argue that neural assembly computation becomes scientifically
useful when three things are made explicit:

1. the microscopic dynamics that create assemblies,
2. the mesoscopic order parameters that make assemblies measurable,
3. the cognitive-computational primitives that compose reliable assemblies into
   memory, binding, readout, sequence, and control.

## 2. Deep Mental Model

### 2.1 Assemblies as Macrostates

An assembly should be treated as a macrostate, not merely as a set of active
neurons.

Microscopic degrees of freedom include:

- neuron identities,
- sparse random connectivity,
- learned weights,
- stochastic initialization,
- k-winner-take-all competition,
- recurrent projection rounds,
- inhibition and refractory history.

The assembly is the coarse-grained object that emerges from this substrate. It
is functionally meaningful only insofar as it is stable, recoverable,
distinguishable, readable, and composable.

This provides the main bridge between statistical mechanics and cognition:

> Cognitive computation does not happen directly at the individual-neuron
> level. It happens at an assembly macrostate level, but the validity of those
> macrostates depends on microscopic dynamics.

### 2.2 Four-Layer Stack

The paper can organize the system into four levels.

| Level | Objects | Main Question |
|-------|---------|---------------|
| Microscopic dynamics | neurons, sparse graph, k-WTA, Hebbian updates, inhibition | What local stochastic dynamics are being run? |
| Mesoscopic order parameters | assemblies, overlap, turnover, basins, drift | Did a stable, recoverable macrostate emerge? |
| Dynamical primitives | projection, recurrence, association, merge, completion, LRI recall | What protocol moves the system through macrostate space? |
| Cognitive-computational layer | binding, memory, readout, automata, toy parsing, control | What useful computation is built from those primitives? |

The top layer depends on the lower layers. The paper should make that
dependency explicit instead of treating demos as standalone capability claims.

### 2.3 Cognitive Operations as Dynamical Protocols

Assembly operations are not abstract Python functions. They are dynamical
protocols over learned macrostates.

| Operation | Dynamical Interpretation | Cognitive Interpretation |
|-----------|--------------------------|--------------------------|
| Projection | external field induces a sparse macrostate | write or induce a representation |
| Recurrence | recurrent dynamics stabilize a macrostate | hold or settle a representation |
| Reciprocal projection | source macrostate routes into another area | copy, route, or translate |
| Association | sequential source drives form shared access through a target | link related representations |
| Merge | simultaneous source drives form a conjunctive target | bind features, roles, or objects |
| Pattern completion | partial cue flows back toward an attractor | recover memory from incomplete evidence |
| LRI recall | inhibition destabilizes the current attractor and permits transition | move to a next state in a sequence |
| Readout | symbolic label is assigned by overlap measurement | name or measure a macrostate |
| FSM/PFA helper | explicit control over assembly-coded states | controlled discrete computation |
| Parser demo | lexical, role, and sequence assemblies are coordinated | structured cognitive workflow |

The deeper architecture is:

> cognition as programmable control over assembly dynamics.

## 3. Statistical-Mechanics Framing

The paper should be careful about the phrase "statistical mechanics." It does
not need to claim a literal thermodynamic energy unless one is formally
defined. A safer framing is:

> nonequilibrium statistical mechanics of learned neural macrostates.

The system is driven, finite, stochastic, plastic, and out of equilibrium. The
natural scientific objects are therefore:

- control parameters,
- order parameters,
- phase-like regimes,
- finite-size effects,
- basins of attraction,
- fluctuation and reliability over seeds,
- failure boundaries.

### 3.1 Control Parameters

Core control parameters:

- `N`: neurons per area,
- `K`: active winners per assembly,
- `K / N`: sparsity or density,
- `p`: connection probability,
- `beta`: Hebbian plasticity strength,
- projection rounds,
- recurrence rounds,
- source feedback,
- refractory period,
- inhibition strength,
- noise or damage level,
- number of trained assemblies,
- number of bindings per area,
- number of random seeds.

### 3.2 Order Parameters

The paper should define these carefully.

| Order Parameter | Meaning |
|-----------------|---------|
| Consecutive overlap | whether the current macrostate is stabilizing |
| Winner turnover | how many active neurons churn between rounds |
| New-winner count | local formation or drift signal |
| Chance overlap | random baseline for interpreting overlaps |
| Recovery overlap | basin-of-attraction quality after damage |
| Source-response overlap | whether a bound target remains responsive to constituents |
| Same-pair replay overlap | whether repeating a binding protocol recovers similar target state |
| Readout margin | separation between best and second-best symbolic readout |
| Recall length | number of accepted LRI recall steps |
| Ordered match count | sequence positions recovered in order |
| Drift score | distance from known memories or stable basin |
| Cycle score | repeated return to previous assembly |
| Entropy/diversity | whether winners are concentrated, diverse, or collapsed |
| Energy-like concentration | optional pre/post-WTA activation concentration measure |

The paper's most important methodological claim may be:

> Every cognitive primitive is accompanied by diagnostic order parameters and
> chance baselines.

### 3.3 Regimes

The system should classify behavior into regimes rather than only reporting
single-run examples.

Candidate regimes:

- stable: high consecutive overlap and low late turnover,
- drifting: persistent turnover and low convergence,
- collapsed: excessive reuse or low diversity,
- recoverable: damaged state returns to reference above threshold,
- nonrecoverable: partial cue fails to return to reference,
- over-bound: merged representation loses constituent specificity,
- source-responsive: bound target can be partially recovered from either source,
- cycling: LRI or recurrence repeatedly revisits previous assemblies,
- escaped: inhibition successfully leaves current attractor,
- noisy transition: sequence recall leaves known-memory neighborhood,
- ambiguous readout: best label margin is too small,
- saturated capacity: additional memories increase overlap and confusion.

The major systems idea:

> A primitive is not implemented until its operating regime is mapped.

## 4. Cognitive Implications

### 4.1 Cognition as Physics of Learned State Spaces

Learning is not only changing weights. Learning reshapes a dynamical landscape
so that useful macrostates exist, can be recovered, and can transition into
other useful macrostates under controlled drives.

The implication chain:

1. sparse Hebbian WTA dynamics create assemblies,
2. assemblies behave like macrostates,
3. operations are protocols for moving between macrostates,
4. cognitive primitives are allowed transitions in a learned dynamical system,
5. cognition is constrained by assembly physics.

The stronger question is not merely:

> What representations and rules do we need?

but:

> What dynamical regimes make those representations and rules physically
> realizable?

### 4.2 Operation Contracts

Each primitive should have a regime contract.

| Primitive | Contract |
|-----------|----------|
| Projection | final consecutive overlap clears stability threshold; late turnover is low |
| Readout | best label is above threshold and separated from alternatives |
| Merge | target is stable, above chance, and source-responsive from either constituent |
| Association | source-alone target responses overlap above chance after association |
| Pattern completion | recovery overlap clears threshold for a specified damage regime |
| LRI recall | current attractor is escaped without immediate drift or cycling |
| Sequence memory | ordered match count remains above threshold over seeds |
| Parser composition | category, role, and order outputs remain inspectable and failure cases are explicit |

The paper can state:

> Cognitive primitives should be built from operations with measured dynamical
> contracts, not from one-off demonstrations.

### 4.3 Semantics as Stability Under Perturbation

The paper should avoid claiming full semantics from toy demos. A more
defensible proto-semantic criterion is:

A representation earns functional semantic usefulness only when it is:

- stable under recurrence,
- recoverable from partial cues,
- distinguishable from nearby alternatives,
- readable by downstream systems,
- composable with other assemblies,
- robust across seeds or parameter regimes.

For example, instead of saying:

> The system understands a red triangle.

say:

> The red-triangle target is a stable, inspectable, source-responsive macrostate
> that can be compared, routed, and partially recovered under the tested
> regime.

### 4.4 Failure as Scientific Signal

Failure modes reveal the geometry of the assembly landscape.

| Failure | Interpretation |
|---------|----------------|
| Drift | no stable basin or insufficient training |
| Collapse | excessive attraction or poor diversity |
| Interference | insufficient separability or capacity |
| Weak recovery | shallow basin of attraction |
| Cycling | uncontrolled recurrence or inhibition timing |
| Over-binding | loss of part recoverability |
| Failed readout | symbolic measurement is unreliable |
| Ambiguous readout | representation is too close to alternatives |

These failures should be reported, not hidden. They are regime-boundary
evidence.

### 4.5 Cognitive Control as Landscape Navigation

Stability is not enough. If every attractor is too sticky, the system cannot
think, parse, act, or transition. If everything is too unstable, the system
drifts.

Cognitive control requires balancing:

- stabilization,
- destabilization,
- routing,
- binding,
- measurement,
- gating,
- inhibition,
- allocation of new areas or memories.

LRI should be framed as more than a sequence trick:

> LRI is a primitive for controlled escape from an attractor.

This connects to attention, working memory, action selection, parsing, and
bounded reasoning.

### 4.6 Symbols as Measurement Layer

Readout, FSMs, PFAs, and parser demos should not be framed as proof that the
system is symbolic. A better frame:

> Symbols are stable measurements of neural macrostates, not the substrate
> itself.

The assembly system provides neural macrostates. The symbolic layer names,
compares, routes, and checks them.

This gives a hybrid view:

> neural macrostates are the substrate; symbolic labels are measurement and
> control interfaces.

### 4.7 Compositionality Has Physical Cost

Compositionality is not free. Every association, merge, role binding, or
sequence bridge consumes capacity and changes future dynamics.

Costs to measure:

- source interference,
- target collapse,
- loss of part recoverability,
- spurious overlap with other bindings,
- reduced future capacity,
- readout confusion,
- failure under repeated binding load.

The deeper claim:

> compositional operations are constrained by attractor geometry and
> interference budgets.

### 4.8 Process-Level Interpretability

Most neural systems expose outputs and add explanations afterward. This system
can expose process:

- winner traces,
- overlap curves,
- turnover maps,
- basin recovery,
- source-response diagnostics,
- chance baselines,
- phase regimes,
- ablation results.

Potential claim:

> Assembly computation is interpretable because its functional states are
> macroscopic and measurable.

This is one of the strongest systems contributions.

### 4.9 Regime-First AI

The systems principle:

1. map stable representation regimes,
2. map recovery regimes,
3. map binding regimes,
4. map transition regimes,
5. compose only inside reliable regions.

The paper can call this a regime-first methodology for assembly-based cognitive
systems.

### 4.10 Self-Diagnosing Assembly Runtime

Because operations are traced, a future runtime could detect and respond to
failure:

- projection did not stabilize,
- merge is below source-response threshold,
- recall is drifting,
- readout is ambiguous,
- memory area is near interference capacity,
- pattern completion failed at expected damage level.

Possible adaptive responses:

- increase rounds,
- adjust inhibition,
- lower or raise beta,
- allocate a new area,
- retrain a binding,
- reject a readout,
- request more evidence,
- route through an alternate pathway.

This suggests a future "assembly operating system" or "assembly runtime" where
diagnostics are part of execution.

## 5. Systems Framing

### 5.1 Assembly Operating System Metaphor

For systems audiences, the analogy is useful:

| Assembly System | Systems Analogy |
|-----------------|-----------------|
| Areas | memory/register spaces |
| Assemblies | values or latent states |
| Fibers/projections | routing instructions |
| Plasticity | write/update |
| Recurrence | stabilization/commit |
| Inhibition | control/transition |
| Readout | measurement/query |
| Traces | execution logs |
| Sweeps | regime maps |
| Operation contracts | type or reliability contracts |

This should be used carefully. The paper should not over-literalize the
computer metaphor, but it is valuable for communicating that assembly
calculus can be treated as a programmable substrate.

### 5.2 Scientific Framing

For neuroscience and cognitive-science audiences, use:

> an instrumented dynamical substrate for compositional cognition.

or:

> a controlled dynamical system over learned sparse neural macrostates.

Avoid overly broad claims such as:

- the system understands language,
- the system explains cortex,
- the system is a complete cognitive architecture,
- the system is spike-accurate or biologically complete.

Use stronger but bounded claims:

- assembly operations can be made process-inspectable,
- cognitive primitives can be treated as dynamical protocols,
- reliability depends on measurable regimes,
- symbolic interfaces can be built as readout/control layers over assemblies,
- failures reveal regime boundaries.

## 6. Paper Shape

The ambitious paper can combine statistical mechanics and compositional systems
if the bridge is clear:

> The statistical mechanics tells us when the primitives are reliable.

Possible title candidates:

- Neural Assembly Systems: Statistical Mechanics, Compositional Dynamics, and
  Cognitive Computation
- Statistical Mechanics of Neural Assembly Learning and Compositional
  Computation
- A Traced Neural Assembly System: Learning Dynamics, Phase Regimes, and
  Compositional Computation
- Learning Dynamics and Compositional Computation in Neural Assembly Systems
- Toward a Physics of Cognitive Assembly Systems

### 6.1 Proposed Structure

1. Introduction
   - Cognitive systems need representations that are neural, compositional,
     inspectable, and dynamically learned.
   - Assembly demos often hide the dynamics.
   - Contribution: traced simulator, measured regimes, compositional
     primitives, and research workflow.

2. Model
   - Areas, stimuli, sparse graph, k-WTA, Hebbian plasticity.
   - Recurrence, feedback, inhibition, LRI.
   - Assembly snapshots and overlap metrics.

3. Macrostates and Order Parameters
   - Define assemblies as coarse-grained macrostates.
   - Define overlap, turnover, recovery, source response, recall, margins.
   - Define chance baselines and regime classification.

4. Assembly Formation Dynamics
   - Projection traces.
   - Stabilization regimes over `N`, `K`, `p`, `beta`, rounds, and seeds.
   - Stable, drifting, and collapsed behavior.

5. Attractors and Pattern Completion
   - Partial cue experiments.
   - Basin-size maps.
   - Noise injection and damage sweeps.

6. Binding and Composition
   - Association vs merge.
   - Source-response diagnostics.
   - Same-pair replay.
   - Compositional reliability curves and interference.

7. Sequential Dynamics and Control
   - Sequence memorization.
   - LRI recall.
   - Refractory/inhibition phase diagrams.
   - Drift, cycling, and controlled escape.

8. Symbolic Interfaces and Cognitive Workflows
   - Lexicon/readout.
   - FSM/PFA controlled computation.
   - Toy parser.
   - Failure cases.
   - Frame these as demonstrations of composability, not proof of cognition.

9. System Architecture and Instrumentation
   - Trace models.
   - Visualization.
   - Parameter sweeps.
   - Research registry and claim discipline.

10. Limitations
    - Not spiking.
    - Not full biological cortex.
    - Not a broad language system.
    - Empirical phase diagrams do not equal full analytic statistical
      mechanics.
    - Current cognitive demos are toy workflows.

11. Discussion
    - Why measured regimes matter.
    - How this supports future cognitive architectures.
    - Open theory and systems directions.

### 6.2 Three Claim Levels

Keep these distinct:

1. Regime claim:
   sparse Hebbian WTA systems exhibit measurable assembly formation,
   stability, recovery, drift, and capacity regimes.

2. Primitive claim:
   within reliable regimes, assembly operations such as projection, merge,
   association, completion, and recall become computational primitives.

3. System claim:
   these primitives can be composed into early cognitive workflows such as
   labels, memory, binding, automata, and toy parsing.

The system claim depends on the first two.

## 7. Low-Hanging Fruit That Would Deepen The Paper

The most valuable additions are not necessarily larger architectures. They are
better measurements.

### 7.1 Finite-Size Scaling

Sweep:

- `N`,
- `K`,
- `K / N`,
- `p`,
- `beta`,
- rounds,
- random seeds.

Measure:

- stabilization time,
- final overlap,
- turnover,
- recovery overlap,
- source-response overlap,
- variance over seeds.

Why it matters:

> It turns the work from one-off demonstrations into empirical statistical
> mechanics.

### 7.2 Basin-of-Attraction Maps

For pattern completion, sweep:

- retained fraction,
- random neuron injection,
- deletion noise,
- training rounds,
- beta,
- recurrence rounds.

Measure:

- recovery overlap,
- basin threshold,
- probability of recovery over seeds.

Why it matters:

> Memory is only useful if recovery basins are measurable.

### 7.3 Operation Reliability Curves

For projection, association, merge, completion, and recall:

- run many seeds,
- classify success/failure,
- report success probability with confidence intervals.

Why it matters:

> A primitive is a reliable operation only if it works across seeds and not
> merely in a chosen example.

### 7.4 Regime Classification

Implement classifiers for:

- stable,
- drifting,
- collapsed,
- recoverable,
- nonrecoverable,
- over-bound,
- source-responsive,
- cycling,
- ambiguous readout.

Why it matters:

> This makes failure mechanistic and supports phase-diagram figures.

### 7.5 Energy-Like Metrics

Add carefully named energy-like or concentration metrics, possibly based on:

- pre-k-WTA activation concentration,
- margin between selected and non-selected winners,
- total input into winners,
- variance or entropy of activation mass,
- change in concentration over rounds.

Important caveat:

Do not call this thermodynamic energy unless formalized. Use "energy-like" or
"activation concentration" until the derivation exists.

### 7.6 Entropy and Diversity Metrics

Track:

- winner diversity across rounds,
- entropy over winner frequency,
- area reuse,
- concentration of winners across memories.

Why it matters:

> Stable behavior and collapsed behavior can both look like high overlap; a
> diversity metric distinguishes them.

### 7.7 Compositional Reliability and Interference

Train many bindings:

- many feature pairs,
- many object assemblies,
- many role bindings,
- many associations in a shared target.

Measure:

- source-response overlap,
- spurious recovery,
- readout confusion,
- binding capacity,
- degradation with number of stored compositions.

Why it matters:

> Compositionality has a physical cost and an interference budget.

### 7.8 Ablations

Turn off or alter:

- recurrence,
- plasticity,
- source feedback,
- source fixing,
- LRI,
- inhibition strength,
- target recurrence,
- stimulus drive persistence.

Measure which primitive fails and why.

Why it matters:

> Ablations show which mechanistic ingredients are necessary, not just
> sufficient.

### 7.9 Integrated Runtime Diagram

Add one canonical figure:

- areas,
- stimuli,
- assembly macrostates,
- operation traces,
- diagnostics,
- readout/control layer,
- experiment registry.

Why it matters:

> The paper needs a single figure that says what the system is.

### 7.10 Registered Experiment Artifacts

Every paper figure should come from:

- an experiment script,
- a manifest,
- a JSON result artifact,
- a generated plot,
- a result summary,
- a claim or evidence index entry when mature.

Why it matters:

> It turns a systems paper into reproducible research rather than a notebook
> tour.

## 8. Highest-Priority Experiments

If only five experimental programs are added before drafting, prioritize:

1. finite-size scaling for projection stability,
2. pattern-completion basin maps,
3. merge/association reliability over seeds,
4. LRI recall phase diagrams,
5. ablations for recurrence, plasticity, feedback, and inhibition.

These five support the central thesis most directly:

> Cognitive operations work only inside measurable assembly-dynamical regimes.

## 9. What The Current Codebase Already Supports

The current repo now supports a strong basis for:

- traced projection dynamics,
- traced reciprocal projection,
- traced merge,
- traced association,
- traced pattern completion,
- traced LRI recall,
- source-response diagnostics,
- lexicon/readout demos,
- FSM/PFA demos,
- toy parser demos and failure cases,
- notebook-level parameter sweeps,
- notebook-level visual inspection,
- research registry and claim discipline.

This is enough to motivate the paper and guide experiments, but not yet enough
to claim full theory or broad cognitive competence.

## 10. What Is Not Yet Defensible

Do not yet claim:

- full statistical mechanics in the analytic physics sense,
- general cognitive architecture,
- full biological plausibility,
- real language understanding,
- cortical explanation,
- spiking-level realism,
- large-scale cognitive competence,
- robust compositionality under high memory load.

Current safer language:

- neurobiologically inspired,
- assembly-based cognitive substrate,
- traced neural assembly system,
- measurable learned macrostates,
- compositional primitives under tested regimes,
- toward cognitive assembly systems.

## 11. Future Research Program

The long-term program is:

1. Build measurement primitives for assembly macrostates.
2. Map regime boundaries for formation, recovery, binding, and recall.
3. Define operation contracts.
4. Compose primitives only inside validated regimes.
5. Add adaptive runtime control.
6. Scale to richer cognitive workflows.
7. Promote bounded evidence into formal claims.
8. Draft papers from validated claims.

The most ambitious long-term claim:

> Cognition may be programmable at the level of learned neural macrostates, but
> only if the physics of those macrostates is measured and controlled.

## 12. Relationship To Existing Planning Notes

This note complements:

- `THEORETICAL_THROUGHLINES.md`
  for the linear-algebra, dynamical-systems, and statistical-mechanics
  perspectives.
- `ASSEMBLIES_AS_NEURAL_COMPILER.md`
  for the programming-language and compiler metaphor.
- `VISUALS_DYNAMICAL_SYSTEMS.md`
  for figure ideas around attractors, phase diagrams, recurrence, and learned
  weights.
- `PRIORITIES_AND_GAPS.md`
  for the concrete gaps that must become experiments before strong claims.

This document is the integrated paper-vision layer tying those themes together.

## 13. One-Sentence Center

The sentence to keep returning to:

> We do not merely demonstrate assembly operations; we characterize the
> dynamical regimes under which assembly operations become reliable cognitive
> primitives.
