# Neural Assemblies: Multimodal Continual Intelligence Research Program

**Status:** experimental, 2026-11-23. **Authoritative codebase:** `Caerii/assemblies` (`master` at inspection). **Companion prototype:** this folder. No external hardware or foundation-model claims have been demonstrated by these scripts.

## North-star question

Can local, continuously updated neural assembly circuits achieve **measurably better compositional generalization and retention** than strong indexed episodic, retrieval, and parameter-efficient adaptation baselines, with a latency/energy envelope that benefits from sparse hardware?

A faster associative table does not answer this question. Every scientific claim is registered before a sweep; all seeds and failures are retained.

## Workstreams, dependencies, and exit gates

| ID | Phase and dependency | Deliverable | Evidence gate | Status |
|---|---|---|---|---|
| P0 | Scientific foundations | frozen definitions, provenance, and baseline inventory | claims and benchmark protocol match current native `Brain` behavior | in progress |
| P1 | Native semantics (P0) | multiplicative Hebbian, clipping, stable IDs, gating, WTA, refraction parity | equal winner IDs and exact weights vs native engine across seeded explicit graphs, including read-only and error paths | partial: independent standalone parity; native comparison outstanding |
| P2 | Systems core (P1) | Rust packed CSR + copy-on-write delta + safe compaction | native compiled/loaded; zero mismatches against reference across >10k mixed operations, checkpoint/restore/fault tests | prototype only; Rust not compiled here |
| P3 | Memory scaling (P1,P2) | lifelong association, capacity and forgetting curves | paired seeds; retention vs inverted episodic, nearest-neighbor, and refraction/no-refraction arms | pending |
| P4 | Typed compositional binding (P1,P3) | role-factored Phrase/Clause/Discourse with reciprocal readout | unseen role swaps and nested trees; full tree reconstruction, not just assembly cardinality | pending |
| P5 | Multimodal grounding (P4) | visual/audio/text encoders to stable assembly identities | held-out entities, paraphrases, transforms, false matches and OOD shift; vision-language transfer better than baseline | pending |
| P6 | Local plasticity policy (P3,P5) | learn/review/correct plus scoped decay, pruning, refraction | verified corrections do not erase unrelated memories; gate calibrated for false positives | pending |
| P7 | Hardware scale (P2,P4,P5) | Rust SIMD/Rayon, CUDA, optional AWS F2 HLS | complete end-to-end latency/energy including I/O and WTA; speed claims from hardware measurements | pending |
| P8 | Foundation-model co-processor (P5-P7) | streaming perception + persistent assembly memory + generation | downstream gains at matched compute/memory budget; real multimodal tasks | pending |

## Immediate implementation order

1. **Do not equate additive prototype updates with native `Brain` updates.** Keep two contracts, test each independently, then write a native-Brain parity adapter with fixed explicit connectome.
2. Add a versioned serialization format for compiled CSR and delta state; test atomic compaction, write rollback, invalid inputs, tie breaks, and checkpoint migration.
3. Build paired memory curves: 16, 64, 256, 1024, 4096, 10000 learning events; include old-item recall, noisy recall, correction survival, overlap distributions and actual process RSS.
4. Introduce role-factored training and zero-shot evaluation: subject/object swaps, nonce nouns, unseen relational triples, scrambled-role control, de novo relations, nested phrase depth.
5. Only after (4) use real ViT/JEPA/text embedding outputs; explicitly separate perceived labels from taught labels and report encoder-only nearest-neighbor baseline.
6. Compile Rust extensions and benchmark same seeds, same connectome, same semantics. Record CPU model, threads, compiler flags, warm-up, p50/p95/p99, memory, nonzero edge rate.
7. Model F2 from *measured* per-tract transactions and bank conflicts; synthetic roofline is not a performance result. Port only after a validated bottleneck warrants it.

## Mandatory benchmark baselines

- **B0:** chance and untrained assembly.
- **B1:** exact nearest-neighbor episodic store, inverted posting lists and comparable compression.
- **B2:** sparse additive Hebbian, with/without explicit normalization (only comparison within this family).
- **B3:** native multiplicative Hebbian, refraction on/off, with all recurrence and tie-break policies pinned.
- **B4:** role-symbolic ground-truth program and a label-permutation control.
- **B5:** encoder similarity or vector search without assembly memory; RAG/parameter-efficient adaptation where meaningful.

## Registrations (must be decided before looking at test outcomes)

### R1: continual retention
- Independent seed is experimental unit; preregister training stream and noise levels.
- Primary: exact target assembly recovery, all k winners; secondary Jaccard, early-quarter retention and retrieval margin.
- Compare same memory budgets and number of presentations; report intervals over independent graphs, not just queries.
- Success only if assembly arm exceeds B1 at comparable storage/latency in a prespecified relevant setting.

### R2: compositional transfer
- Train on atomic entities and relations with held-out combinations; provide explicitly novel role swaps and center embedding.
- Primary: full role-labeled tree reconstruction, not membership in k-sparse assembly.
- Evaluate depths 1..8 with controlled distractors; record collision, readout errors, and invalid binding cases.
- Include symbolic upper bound, naive concatenation, episodic and label-shuffle controls.

### R3: sensory stability
- Fix encoder weights and seeds; test identity under image transformations, paraphrases, audio noise.
- Measure false matches, distribution overlap and calibration. Reject arbitrary >85% overlap assumptions.

### R4: correction plasticity
- No automatic anti-Hebbian erase from novelty or 'N400/P600 proxies'. Only verified or independently trusted teacher corrections.
- Measure accuracy immediately before and after correction, plus collateral loss across all old tasks.
- Separate neural 'surprise' features from human ERP claims unless human EEG evidence exists.

### R5: acceleration
- Strict parity: exact IDs, deterministic ties and state transitions where mathematically possible; where float order differs, prereg tolerance and check decision equivalence.
- Native Rust tests MUST fail if extension absent; separate Python-only smoke suite may skip.
- Track wall clock, compaction cost, bytes moved, resident memory, read/write mix, and tail latency. Compare against vectorized NumPy and an indexed retrieval baseline.

## Hardware reality checks

- The past discussion's **137k multi-area projections/sec, 7.3 μs/step, 100k sentences in 0.7 sec, and 15 μs PCIe round trip** are unvalidated modeling assumptions, not experimental findings.
- The 0.7 sec sentence calculation omits steps per sentence: at 12 steps/sentence, 100k sentences means 1.2M projections, or ~8.76 sec at 137k/sec *if* the projection definition matches. End-to-end realistic timing must also include encoding, scheduling, memory operations and host communication.
- The proposed 'zero-overhead' pruning and homeostasis claims require measuring HBM read/write traffic and row compaction under online loads.
- A role-swap wired by evaluation scaffolding is not autonomous syntax acquisition; semantic grounding is not established by synthetic random vectors.

## Repository integration targets

- `neural_assemblies/tests/`: native semantic and state-machine conformance tests.
- `crates/na-kernels/`: PyO3 Rust optimization, only after standalone correctness gate.
- `research/notes/multimodal/`: preregistrations, results, known failures, seeds and explicit evidence.
- `examples/`: inspectable demos separated from publishable experiments.
- `docs/`: scope/limitations of model, numerical contract and provenance.

## Evidence discipline

A result moves through `proposed -> implemented -> parity_validated -> benchmarked -> replicated -> supported`, or to `rejected` with retained counterevidence. All results must identify repository commit, package version, full invocation, machine/CPU/GPU, seed manifest, input digests, and provenance. CI on external GitHub Actions is not required: local `pytest` and `cargo test` gates must be reproducible.

## Next decisive milestone

**M1:** native `Brain` parity on explicit seeded graph with genuine multiplicative plasticity + refraction options, with executable repeated-presentation and read-only checks. Until this passes, additive CSR and multiplicative standalone fibers are **different models** and cannot be used interchangeably for scientific claims.