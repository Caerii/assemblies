# Codebase critique and a proposal for doing assembly calculus reliably

Review date: 2026-09-09. Base commit: `2e17382ef658b9cba2995286c1a5f2dbc13aa7c8`, branch `dev`, with a changing working tree. This is an architectural and scientific-method audit, not certification of every experiment or a fresh reproduction of the GPU results. No runtime, registration, or existing result files were changed for this review.

The review inventoried the Python tree and examined the onboarding, register, engine interfaces and implementations, calculus operations, measurement helpers, parity infrastructure, active memory and sequence protocols, research indexes, language entry points, and release workflow. The working tree contained 586 package Python files (178 under its main tests directory) and 463 research Python files, excluding environments, worktrees, and references. Those are inventory counts, not a claim that every line was reviewed. Existing parity edits and additional research changes appeared during the review; their ownership was preserved. GPU jobs were not started.

## The diagnosis

The strongest part of this repository is its willingness to record why a result failed. The weakest part is that a collaborator must remember that history to choose a scientifically meaningful operation.

There are already useful building blocks: immutable assembly snapshots, the result register and rendering test, undefined measurements that raise, shared pricing code, fixed-connectome CPU execution, drive replay, width parity, and reusable memory and arc units. The next step is to make those define the ordinary workflow.

The repository currently interleaves five different objects:

1. A mathematical model with explicit assumptions.
2. Several numerical implementations, with different approximations and supported rules.
3. Protocols that construct useful computations from those rules.
4. Measurements of those protocols under particular training and readout conditions.
5. Scientific claims inferred from those measurements.

The confusion occurs at the boundaries. A kernel match becomes a model-validity claim; a winning set becomes a memory; an assumption becomes a necessary law; an experimentally useful construction becomes a universal computational statement. Directory cleanup cannot prevent those transfers. Contracts and evidence links can.

## Findings that should change the immediate priorities

### 1. The entry path does not teach the protocol the repository now recommends

The [README quick start](../../README.md:231) uses `numpy_sparse` and calls `project` without recurrence. [The operation itself](../../neural_assemblies/assembly_calculus/ops.py:364) defaults `recurrent=False`, explicitly explains that this does not strengthen the internal recurrent fiber, and directs new work to a private helper under `research/experiments/_substrate.py`.

This is a maintenance compatibility choice, not evidence that every feedforward representation is invalid. The problem is presenting that route as the introductory assembly-formation procedure. A user should not have to read a warning deep in the implementation to discover which formation protocol they ran.

Introduce a supported formation API with explicit recurrence, engine/model configuration, convergence observations, and subsequent cue-recovery checks. Keep the historical behavior behind a named compatibility route. Do not flip the default and regenerate every golden indiscriminately.

Likewise, [the calculus package introduction](../../neural_assemblies/assembly_calculus/__init__.py) calls disabling plasticity a pure read, whereas [Brain.probe](../../neural_assemblies/core/brain.py:622) explains why that can still change topology and subsequent measurements. Teach mutation, frozen dynamics, and observational probes separately from the first example.

### 2. “Exact” is overloaded across genuinely different execution semantics

[ProjectionFidelity.EXACT](../../neural_assemblies/core/projection_fidelity.py:1) explicitly includes truncated-normal candidate sampling. [ENGINE.md](../../neural_assemblies/ENGINE.md) explains why that sampling does not preserve fixed-connectome input correlations. Meanwhile [NumpyExactEngine](../../neural_assemblies/core/numpy_engine/_exact.py) already provides a different, fixed-hash drive path and different normalization semantics. The architecture's engine table omits `numpy_exact` even though the registry includes it.

Use independent configuration fields for graph generation, candidate domain, normalization, weight evolution, inhibition/refraction, tie selection, precision, and backend. Reserve “exact” for a stated relationship: exact drive on this fixed graph, parity with this reference under this schedule, or exact winner equality under this tie rule.

Materializing a graph removes candidate sampling. It does not by itself make stimulus distributions, normalization, tie policy, or higher-level clocks identical across implementations. Hash regeneration is a storage/execution technique; the mathematical assumptions about graph randomness remain a separate contract.

### 3. A PROVED entry overstates what its theorem licenses

[`SEQ-REGIME`](../../neural_assemblies/theory.py:105) says winner selection is reliable **only when** `kp >= 3 ln n`, and its caveat predicts failure below that threshold regardless of the mechanism.

The cited sequence-copy theorem supplies sufficient conditions, including sequence length, input overlap, plasticity, repetitions, and homeostasis after each presentation. Satisfying one inequality does not establish applicability; violating it does not establish failure. [Dabagia et al., Theorem 1](https://arxiv.org/html/2306.03812v2#S2.SS1).

Split this into a faithful theorem record, a theorem-applicability evaluator, and separately scoped empirical operating-region claims. “Outside the sufficient bound” is a legitimate experimental condition. It should remain measurable.

The broader design distinction also matters: the sequence paper explicitly distinguishes externally programmed control from control implemented with inhibitory circuitry. A host-scheduled organ and a self-contained neural controller should have separate claim scopes. [Dabagia et al., introduction and neural model](https://arxiv.org/html/2306.03812v2#S1).

### 4. The memory line does not have one unambiguous capacity definition across its artifacts

[The registration's protocol](../../research/notes/memory/PREREG_refraction_memory.md:53) describes stimulus-cued rank-1 recovery crossing 0.9. The current GPU script measures half-cue recurrent recovery and uses [`HALF_BAR = 0.50`](../../research/experiments/seq_capacity_scaling.py:57). The CPU mirror uses [`THRESHOLD = 0.90`](../../research/experiments/refraction_memory_numpy.py:49). Later prose explains half-cue recall, but a reader still has to reconstruct which definition belongs to which result.

The mirror also deliberately uses a different stimulus distribution: ten size-six inputs approximate graded input, rather than the hashed harness's binomial count over sixty inputs. The script explains this explicitly. That supports a robustness comparison, not an identical-protocol numerical reproduction.

This review does not establish that the reported memory advantage is false. It establishes that “capacity” and “mirror” need versioned definitions. Each artifact must state cue construction, recall dynamics, ranking universe, tie handling, threshold, sampling, distinctness gate, and interpolation/censoring rule. Regenerate headline quantities from saved curves under named definitions before deciding which scientific reruns are required.

### 5. The capacity runner can overwrite both runs and parameter cells

[`seq_capacity_scaling.py`](../../research/experiments/seq_capacity_scaling.py:339) keys cells by `arm/n`, although `--nk` accepts `(n,k)` pairs. A sweep containing the same `n` at two `k` values assigns both to the same key. That is a direct loss-of-evidence risk even within one run.

The script also defaults to sixteen brains, accepts an empty tag, and writes with `open(..., "w")`. Its explicit smoke route is labeled and avoids writing results, which is good. But a non-smoke invocation with reduced brains can still write into the ordinary result namespace. The result JSON's top level consists of cells rather than a full run manifest.

Use a canonical cell ID derived from the complete resolved configuration. Create immutable run directories with collision refusal, record requested and completed cells, and make deviations from the registration change the run's validity status automatically. A tag should be a human label, not the identity or overwrite protection.

### 6. Scientific guardrails are partly conventions, and several currently fail

Focused CPU checks completed with **33 passed, 4 failed, 1 deselected**:

- Theory citations and methodology: 9 passed, 1 failed. The methodology scanner flags `refraction_memory_numpy.py` and `seq_tm_high_order.py`.
- Measurement, read-only probes, index-space ratchet/types: 24 passed, 3 failed, 1 deselected. The three failures concern a baseline still pointing to `tests/test_brain_core.py`, whose scanned counterpart is now `legacy/scripts/simulations/test_brain_core.py`.

These failures require different actions. The CPU mirror's flagged `np.mean(pw)` is a within-brain pair statistic; calling it an invalid seed estimate would be wrong. The high-order script retains per-brain values and judges success counts, but its displayed means lack intervals. Ratchet failures are review signals, not automatic scientific retractions. Resolve each site by its estimand rather than simply raising baselines.

The only checked-in GitHub workflow found is [publish.yml](../../.github/workflows/publish.yml), triggered on release publication or manual dispatch. It tests before publishing, but does not provide a checked-in PR/push gate. This says nothing about external CI configuration that was not inspected.

The [index-type guard](../../neural_assemblies/tests/test_index_space_types.py:76) is marked slow and skips when Pyright is unavailable; Pyright is not declared in the inspected development dependencies. It therefore does not run under the onboarding command. Even when run, its small negative example tests the types' capability, not complete package-wide dataflow coverage.

### 7. The statistical helper does less than the onboarding implies

[`ensemble_from_values`](../../neural_assemblies/diagnostics.py:1421) checks the number of values, not whether they are independent seeds. It accepts duplicate seed keys. Its small-sample t table ends at twelve observations and falls back to 1.96 afterward.

A direct CPU probe with values 0 through 19 produced CI half-width 2.592836; Student's t with 19 degrees of freedom gives 2.768811. The current half-width is about 6.4% smaller. This is an interval-policy defect, not evidence that any particular registered bar changes verdict. Recompute affected summaries before making that claim.

Store seed identities with observations, validate counts and finite values, use a stated interval method, and distinguish independent brains, sampled items within a brain, and corpus variation. Twenty seeds is a project minimum, not a power analysis or a guarantee of precision. A run with zero errors also needs a bound at the declared sampling level, not an inference of universal exactness.

### 8. Index identities still have escape paths into plausible numbers

[`to_neuron_ids`](../../neural_assemblies/core/index_spaces.py) silently drops out-of-range indices, whereas [`_snap`](../../neural_assemblies/assembly_calculus/ops.py:74) passes some invalid indices through unchanged. The private research `pinned` helper drops IDs missing from an inverse mapping; `activate_assembly` instead raises. These are inconsistent meanings of invalid identity.

A direct probe converted compact `[0,99]` through mapping `[42]` to `[42]`, without error. Another gave overlap 1.0 between assemblies named `A` and `B` containing the same integers. That second result is defined by the current API, but shared integer ranges alone do not establish a biological or representational correspondence between distinct areas or brains.

Public snapshots should carry a brain/connectome identity and area identity. Same-population overlap should reject mismatches; cross-population comparisons should require an explicit correspondence. Engine-internal invalid indices should fail. If filtering is intentional, return the number and reason dropped as a separate operation.

### 9. The headline layer is stronger than the evidence layer

The [README](../../README.md:76) says capacity follows a square law; the [register](../../neural_assemblies/theory.py:401) and [registration](../../research/notes/memory/PREREG_refraction_memory.md:489) explicitly decline a power-law claim and report drifting effective exponents. The README also says sequences of any order are remembered, while the highlighted experiment tests order ten in a finite training window.

Use the bounded statements: approximately quadratic capacity over the measured range; order-ten disambiguation on the specified memorized corpus inside the measured window. Architectural arguments about higher orders need a separate resource/collision argument or theorem.

The register is a good nucleus, but its `source`, `preconditions`, and `evidence` are mostly strings. The tests establish citation resolution, nonempty evidence, caveats, and faithful rendering. They do not establish that a run has the right engine, all registered cells, or a valid adoption decision.

### 10. The package boundary admits too many competing definitions of the system

Language behavior lives in `language/`, `nemo/language/`, `assembly_calculus/emergent/`, `lexicon/`, `programs/`, and `text_generation/`. Some are legitimate separate systems, but the maintained-package label alone does not distinguish their roles. For example, `lexicon/true_assembly_learner.py` asserts CUDA availability and prints device information at import; the older emergent learner directly imports CuPy and maintains symbolic frequency/co-occurrence tables. These are not interchangeable backends of `Brain`.

The current `EmergentParser` assembles fourteen mixins with cross-mixin calls resolved through the shared object. Splitting files has reduced local file size without making the runtime dependencies explicit. Before adding more functionality, map state ownership, required collaborators, and the path from input to decision for each active route.

Preserve these systems as reproduction targets where useful. Assign each a current status, entry point, supported configuration, responsible tests, and reason to exist. Migrate only the active route to explicit components; move unsupported research prototypes out of the public guarantee through a deprecation process.

## The architecture to organize around

Use three layers for execution and a separate layer for evidence. The names below are proposed destinations, not APIs that already exist.

| Boundary | Owns | Must not silently decide |
|---|---|---|
| `model/` | Graph law, clock, weight rule, normalization, inhibition/refraction, ties, identity | Hardware selection or experimental conclusions |
| `backends/` | NumPy/CUDA storage and execution of supported model configurations | A different stimulus model or learning schedule |
| `protocols/` and `organs/` | Formation, association, merge, recall; memory, FSM, transducer, aligner | Metrics, adoption, or hidden test-time teachers |
| `measurement/` and research tooling | Probe semantics, readouts, interventions, estimators, run records, claims | Modifying dynamics to improve the measured answer |

Keep `Brain` as a facade where it remains useful. The hashed units need not be forced through its per-area CPU-array interface: that would threaten the batch execution advantage. Share declarative semantics and conformance cases; keep backend-specific execution strategies. An independently understandable tiny reference oracle is valuable even if it duplicates some mathematics; two paths calling the same buggy helper are weak independent evidence.

An operation should return an outcome with the snapshot, rounds used, termination reason, convergence observations, and provenance. Recovery competence is tested by a separate read protocol. The memory registration already demonstrates why stable winners and a successfully written associative memory are different facts.

A model configuration must state the order of each tick: read previous winners, gather inputs, apply selection modifiers, select, potentiate, charge bias, and publish next state. It must also state behavior under zero drive, ties, fixed targets, masked reads, and exception exits. These are model decisions, not backend conveniences.

For every application, label what the host provides: stimulus sequencing, target assemblies during training, reset signals, gating schedule, label dictionaries, and ranking. Distinguish an external decoder that measures information in an assembly from a neural decoder that computes with it. Both are useful, but they answer different questions.

## Evidence should be a graph with immutable run records

The durable chain should be:

`question -> registration revision -> resolved protocol -> run -> analysis -> claim revision -> paper/README statement`

Extend the existing register, parity manifests, and IR rather than creating unrelated competing catalogs. A `RunRecord` should include the registration content hash, commit and dirty patch hash, actual imported module paths, model configuration, engine and kernel identity, dependency/compiler/hardware details, seeds, corpus/split hashes, resolved grid, command, and completed-cell inventory. Sensitive data need not be embedded; dataset identifiers and hashes can supply provenance.

Track validity separately from outcome. A valid failed prediction is valuable; a successful smoke run is not confirmatory evidence. Track claim lifecycle separately from evidence kind: PROVED/MEASURED/EXTENSION does not express superseded, retracted, narrowed, or awaiting replication.

For each measured claim, resolve concrete run and analysis IDs. For each theorem, resolve its exact source version and proposition, complete assumptions, and implementation mapping. Historical result files without recoverable provenance should say “unknown,” never inherit today's engine metadata.

Do not reorganize evidence by moving every old artifact. First register existing paths and hashes. Then put new runs under immutable directories and generate navigation views by research line. This preserves citation history while making new work legible.

## Replace the onboarding reading burden with a first successful investigation

The supplied onboarding teaches admirable scientific conduct. It starts too far inside the research history and postpones the executable model until the end. A newcomer needs a verified small investigation before reading hundreds of lines of amendments.

Recommended sequence:

1. **Model:** one page defining a tick, a graph, a snapshot, an attractor, and an operation; distinguish refraction, neuronal refractoriness, and circuit-level inhibition.
2. **First CPU session:** show the checkout/import paths, resolve a fixed-connectome configuration, form one representation with explicit recurrence, read from a partial cue, and compare to no-learning and wrong-cue controls. Treat small runs as instructional evidence only.
3. **State semantics:** demonstrate that training changes weights; a dynamics-only step advances state; an observational probe restores its declared state. Show an undefined measurement and how it remains undefined through reporting.
4. **One operation contract:** inspect its schedule, expected regime, readout, and true negative. Explain why an operation can execute successfully while its scientific criterion fails.
5. **One complete research chain:** open a registration, its run manifest, per-seed values, analysis, failed bars, and claim revision. Then reproduce a bounded registered run if resources permit.
6. **Choose a lane:** core calculus, memory, sequence, language, or backend implementation; read the lane-specific history and tests.

Keep worktree isolation, explicit-path commits, data restrictions, and one-GPU-job policy. Enforce import provenance, output exclusivity, and device ownership in a supported runner. A manual `sys.meta_path` workaround should be an implementation detail of tooling, not mandatory collaborator knowledge.

Replace the required July [GPU design sketch](../gpu_scale_design.md:3), which still says no batched/dense code has been committed, with the current architecture and built sequence-port note. Retain the sketch as dated history.

Rephrase “a perfect score falsifies the measurement” as “a perfect score requires sensitivity and negative controls before interpretation.” Likewise, identical arm scores can reflect a correct invariance or a coarse metric. Mechanism engagement and test sensitivity decide which; a score alone does not.

## Research order that will produce the most understanding

**First, repair the instrument and claim boundary.** Reconcile capacity definitions, exactness naming, theorem assumptions, run identity, and the focused failing checks. Recompute intervals and run the affected analyses. Scientific work can continue provisionally, but new headline adoption should wait for these controls.

**Second, recover the actual calculus as a small conformance suite.** For projection, reciprocal projection, association, merge, and completion, provide one fixed-connectome protocol with a meaningful negative control and a parameter sweep. The paper plan itself says the earlier merge/association evidence contained dead probes. New organs cannot substitute for a clean account of the primitives named in the package.

**Third, separate the temporal representation question from the readout question.** At `g=0` and `g=1`, measure number information by position while matching token, position, and distractor structure. Fit a diagnostic decoder on training data and evaluate on held-out combinations and fresh brains, using an appropriate within-brain training/test split. Compare it with the native ARC-to-OUT readout. A diagnostic decoder finding information at `g=0` shows representational availability; it does not establish a local neural readout until that readout is built and evaluated.

**Fourth, work on durable sequence learning with a retention/adaptation test.** Treat 200 presentations as an evaluation horizon, not a magic target. Measure old-sequence retention and acquisition of new transitions together, across exposure frequency and order. Inspect weight saturation, bias, winner drift, and readout margins. Freezing all learning preserves old trajectories but does not solve continual learning. Count saturation and clipping times are mechanistic predictions to test, not automatic derivations of the behavioral failure time.

**Fifth, strengthen the memory claim.** Test randomized partial cues, cue corruption, item age/order, low-load competence, and capacity at several explicitly named recovery thresholds. Report storage cost and recall computation along with number of items: ranking against a saved dictionary of all assemblies is an evaluation readout whose cost and role should be visible. Compare against classical associative-memory baselines under a stated resource budget. Fit finite-range scaling with uncertainty before seeking an asymptotic exponent or constant.

**Sixth, evaluate sequence value against serious baselines.** Add trigram, variable-context or prefix-tree memorization, and a small recurrent learner alongside bigram and oracle. Match exposure/data and report parameter memory, compute, and training separately; neural “steps” and network optimizer steps are not inherently comparable. Use gaps 4–6 and train/test structural splits. This distinguishes remembered prefixes, portable features, and useful prediction.

Keep the two training-window mechanisms separate: the memory registration explicitly withdraws a shared explanation for memory hub dominance and sequence arc relocation under clip/refraction. A single homeostasis fix is a hypothesis to test on both, not something already implied by the evidence.

## A staged implementation backlog

| Order | Bounded change | Acceptance evidence |
|---|---|---|
| 1 | Reconcile current docs, theorem scope, and capacity definitions | Every headline resolves to a bounded claim and named metric/protocol version; historical statements remain visible |
| 2 | Repair statistical and ratchet gate issues | Focused suite passes; CI values verified; duplicate identities rejected; legitimate within-brain means documented |
| 3 | Add canonical cell IDs and immutable run manifests | Same-n/different-k cells survive serialization; duplicate output creation fails; incomplete/smoke runs cannot be adopted |
| 4 | Add CPU onboarding protocol and public safe operations | A new collaborator runs one positive and one true negative without private imports or index conversion |
| 5 | Add PR checks and required contract coverage | CPU checks run on PRs; type checks cannot silently skip; GPU conformance is a separately recorded required gate for GPU changes |
| 6 | Extract model configuration and backend capability contracts | Unsupported requests fail before execution; reference and backend traces agree for supported configurations |
| 7 | Consolidate active organs and measurement APIs | Memory and sequence use the same run/evidence machinery while retaining backend-specific performance |
| 8 | Classify language surfaces and reduce implicit state coupling | Every supported entry point has an owner/status, explicit dependencies, a smoke route, and a cited claim boundary |

Do semantic migrations one at a time on isolated worktrees. Preserve old protocols and evidence. A changed primitive, graph rule, readout, or tie policy requires a new protocol/model version and an affected-claim review; it should not be hidden inside a folder move.

## Verification performed

Commands used the existing local environment, without dependency installation:

```powershell
.venv/Scripts/python.exe -m pytest neural_assemblies/tests/test_theory_citations.py neural_assemblies/tests/test_methodology_ratchet.py -q
.venv/Scripts/python.exe -m pytest neural_assemblies/tests/test_measurement.py neural_assemblies/tests/test_read_only_probes.py neural_assemblies/tests/test_index_space_ratchet.py neural_assemblies/tests/test_index_space_types.py -q -m 'not slow'
```

Direct CPU probes confirmed invalid-index filtering, cross-area integer-overlap acceptance, duplicate seed-key acceptance, and the twenty-observation interval discrepancy. Static inspection established the capacity cell-key collision and the conflicting protocol thresholds. The register currently contains 8 PROVED, 15 MEASURED, and 2 EXTENSION entries. Citation/render checks passed.

The full package suite, GPU parity gates, manuscript figures, and scientific replications were not run. No claim is made that the active dirty parity edits pass or fail. Historical incidents in the soundness program were treated as records motivating contracts, not assumed to remain current engine defects.
