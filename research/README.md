# Research Organization

Research work lives here.

The installable package lives under `neural_assemblies/`. The material here is
for scientific questions, experiments, results, claims, and paper scaffolding
that should not be confused with released package behavior.

## Principle

Organize research by questions and evidence, not by hoped-for papers.

Papers should emerge from validated claims. They should not be the structure
that everything else bends around.

## Research Layers

### Broad tracker

- `open_questions.md`
  The wide inventory of active, incomplete, or still-uncertain questions.

### Curated question set

- `core_questions/index.json`
  Machine-readable inventory of the smaller set of question directories that
  already map cleanly onto evidence.
- `core_questions/`
  Per-question directories with `hypothesis.md`, `theoretical_basis.md`,
  `experiments.md`, `results.md`, `analysis.md`, and `conclusions.md`.

### Experiment and suite inventory

- `registry.json`
  Canonical map of experiment families, code paths, result directories, and
  recommended entry points.
- `experiments/`
  Experimental code, manifests, infrastructure, and suite-level notes.

### Results and claims

- `results/`
  Result artifacts produced by experiments.
- `claims/index.json`
  Inventory of formalized claims and claim-ready evidence summaries.
- `claims/`
  Claim documents and supporting material.

### Literature inventory

- `literature/index.json`
  Canonical Assembly Calculus bibliography with per-paper implementation status,
  package module mapping, and gap list.
- `literature/README.md`
  How to use the literature index when implementing paper features.
- [../docs/literature.md](../docs/literature.md)
  Human-readable field map and parity matrix.

### Papers

- `papers/`
  LaTeX infrastructure and paper drafting space that should only be used after
  claims are bounded and evidence-backed.

## Typical Workflow

1. Start with `open_questions.md`.
2. Check whether the topic already exists in `core_questions/index.json`.
3. If it is mature enough, create or update a curated question directory.
4. Register or update the relevant experiment suite in `registry.json`.
5. Run experiments and store outputs under `results/`.
6. Promote bounded evidence into `claims/`.
7. Draft papers from validated claims.

## Current State

### Curated questions

The curated question set includes:

- `Q01` assembly stability
- `Q03` scaling laws
- `Q20` competition and distinctiveness
- `Q22` N400 as global pre-k-WTA energy

These live under `core_questions/` and are indexed in
`core_questions/index.json`.

### Active suites

The suite registry tracks:

- `applications`
- `biological_validation`
- `distinctiveness`
- `information_theory`
- `infrastructure`
- `primitives`
- `stability`
- `vocab`

### Claims status

The claims inventory distinguishes between fully formalized claims and
claim-ready evidence summaries. See `claims/index.json` for the exact status.

## Quality Rules

- Questions may be exploratory.
- Experiments must be reproducible enough to rerun.
- Results should state caveats, not just wins.
- Claims should only say what the evidence supports.
- Papers should be downstream of claims, not upstream of them.

## Validators

Use the repo tooling to check the research indexes:

```bash
uv run python research/literature/validate_index.py
uv run python research/experiments/infrastructure/validate_registry.py
uv run python research/claims/validate_index.py
uv run python research/core_questions/validate_index.py
```

## Where To Start

List the supported shared-runner commands before choosing a study:

```bash
python -m research.runner --list
```

The listing is registry-only and does not import experiment modules or load GPU
extensions. Each command then owns its protocol-specific options while the
shared runner owns tags, seed identities, provenance, and immutable storage.

If you are new to this tree:

1. Read `open_questions.md`.
2. Read `core_questions/index.json`.
3. Read `claims/index.json`.
4. Read `GETTING_STARTED.md`.

That order gives you the broad tracker, the curated subset, the claim
inventory, and the practical workflow.


## Source identity

[The shared runner](runner.py) records `git_commit`, `source_sha256`, and
`source_inventory` before measurement and rejects a changed identity afterward.
`source-inputs-v3` hashes Git-discovered tracked and nonignored untracked code,
compiler headers, build scripts/configuration, Lean/Dafny specifications, the
Lean toolchain pin and manifest, and JSON contracts under `neural_assemblies/ir`.
The inventory policy identifier is also included in the digest. Older records
without `source_inventory` used the narrower original suffix inventory; their
hashes must not be interpreted as covering these additional inputs.

Results JSON is excluded so publishing observations does not invalidate its own
run. Registrations and explicitly declared `input_artifacts` are hashed separately.
Other data/configuration JSON must be declared as an input artifact. This is a
start/end repository-content guard, not a hermetic execution certificate: it does
not capture installed binaries, ignored sources, external
data, or changes made and reverted between the two checks. A completed record
remains UNJUDGED (or VOID for smoke), never automatically adopted evidence.

The constructed negative in
[test_research_runner.py](../neural_assemblies/tests/test_research_runner.py)
changes a Lean file during measurement and requires a retained failure record
with no completed results artifact. Separate cases cover added/modified compiler
and verification inputs and show that emitting results leaves identity unchanged.


## Environment identity

[The shared fingerprint](../neural_assemblies/core/environment.py) covers exact current
values of `ASSEMBLIES_*`, `NEURAL_ASSEMBLIES_*`, and `EMERGENT_*`. Run schema 2
requires an `environment` record naming `repository-environment-v1` and mapping
variable names to SHA-256 digests. No raw environment values are persisted. The
runner compares this snapshot after measurement and retains failure if it changed.
The evidence validator checks its structure; schema 1 remains readable as historical
records that did not require environment identity.

The parser training cache consumes the same fingerprint, excluding only its cache
location (`ASSEMBLIES_BACKBONE_CACHE`) and separately keyed calibration mode
(`EMERGENT_ERP_FAST`). Backend switches under `NEURAL_ASSEMBLIES_*` now invalidate
both memory and disk reuse. Experiment records exclude neither of those settings.

This fingerprint establishes equality, not reconstruction or secrecy of low-entropy
settings. Reproduction still requires explicit resolved protocol parameters.
It describes the current process environment, not values previously captured at
module import, external thread/CUDA settings, installed binary versions, or changes
reverted before the final check. Schema 7 records a discriminated execution
profile for primary Brain models and bespoke hashed organs. Area-local
mixed-engine routing remains open. This guard does
not substitute for those executable contracts.

Version 3 adds Windows `.cmd` build scripts to the source inventory. Earlier
version 2 records remain historical records with that coverage limitation.


<a id="study-checkout"></a>

## Study checkout location

Run a study from a worktree pinned at a commit and placed beside the
repository, not under the system temp directory:

```bash
git worktree add ../assemblies-runs-<date> <commit>
```

The runner writes the study's record into the checkout it runs from, and
Windows Storage Sense empties `%TEMP%` on its own schedule. Two pinned-run
worktrees there lost every tracked file in September 2026. The runner
therefore refuses a study (not a smoke run) whose checkout is under the temp
directory (`research.runner.refuse_temporary_checkout`).

<a id="one-device-job"></a>

## One device job

Studies contend for the card and its memory limit, and a smoke run
scheduled beside a study slowed both and had to be killed by PID
(docs/onboarding.md). The runner enforces the rule: a run whose engine puts
work on the device (`torch_sparse`, `cuda_implicit`, `cupy_sparse`, every
hashed organ, both aligners) takes a machine-wide exclusive lock
(`%TEMP%/assemblies-device-job.lock`, or `ASSEMBLIES_DEVICE_LOCK`) before
reserving its tag and holds it until the record is written. A second device
job is refused immediately with the holder's pid, engine and tag; it is not
queued, because a queued job would run against a checkout that may have
moved. CPU engines, the vendored reference and computed baselines never take
the lock. The lock is shared across worktrees and environments on the same
machine, which is what "one GPU job at a time" means.

## Recoverable source

The [runner](runner.py) writes schema 10 records for every execution kind (Brain,
organ, alignment, computed baseline and vendored reference; schemas 7 to 9 are the
historical forms before the observation policy joined the record, see
`neural_assemblies/ir/VERIFICATION.md#contract-execution-kinds` and
`#contract-observation-policy`) with its source archive captured before calling measurement. The archive preserves exact checkout bytes, including mixed
line endings and Git-discovered nonignored untracked source. Its `source/` members
use the same inventory and ordering as `source_sha256`; `script` and `registration`
preserve the separately hashed entry point and preregistration. ZIP timestamps are
fixed. The run record binds the archive by SHA-256. Existing tags remain reserved
if capture fails, and measurement does not start.

**Storage (since 2026-09-30).** The archive is assembled in memory and never
written as a file. Its members are stored once each, by SHA-256, in
`research/results/source-store/objects/` ([source_store.py](source_store.py)),
and the run keeps a `source.manifest.json` listing them in archive order with
the archive digest the record binds. Consecutive runs share almost all of their
source, and a per-run zip is an opaque binary git cannot deduplicate: 71 such
archives took 313 MB of the tree while their distinct content was 35.5 MB. All
71 were migrated, and each zip was removed only after the store rebuilt it
byte-for-byte (the archive is a deterministic function of the member names and
bytes). A run may still carry a legacy `source.zip`; the validator accepts
exactly one of the two forms. Store objects are marked `-text` in
`.gitattributes`, since a line-ending conversion would change their names.
`validate_source_archive(..., deep=True)` also rebuilds the archive and
requires the recorded digest.

[Archive validation](source_archive.py) recomputes the inventory digest and both
individual digests from archived bytes, rejects duplicate or unsafe member names,
and never extracts or executes code. The runner also validates the archive before
publishing completion; the evidence validator checks it for every schema 3 through 10 record.
Schema 1 and 2 records remain readable without an archive. Their historical byte
recovery gaps are not repaired by this change.

This captures repository source, not a hermetic execution environment or all data.
Schemas 4 through 10 additionally preserve every declared repository input artifact under
`inputs/`, bound to its separately recorded SHA-256. Input aliases resolve to one
repository-relative name; duplicates fail before reservation. Missing, extra or
changed archived inputs invalidate the archive even if its ZIP digest is updated.
Schema 3 archives do not guarantee input recovery; their separate input hashes
remain readable without retroactively claiming the bytes were captured. Later
checkout changes do not change captured bytes. The evidence graph still checks
that referenced repository paths exist; archive integrity and a dangling graph
edge are distinct checks. Mutation during measurement prevents completed output.
Undeclared datasets, installed binaries and external dependencies are not bundled. Keep those limits distinct
from the scientific pass conditions. The archive is evidence to inspect, not a
promise that executing it elsewhere reproduces a study.


<a id="raw-evidence-attachments"></a>
## Raw evidence attachments

Run schemas 5 through 7 separate compact indexed conclusions from large raw observations.
An experiment may return `ExperimentOutput(observations, json_attachments)`. The
runner validates every value as strict finite JSON, encodes it deterministically,
compresses it with gzip at a fixed timestamp, and writes only safe sibling names
ending in `.json.gz`. `results.json` records for each attachment its media type,
encoding, compressed and decoded byte counts, and SHA-256 digests of both forms.
It is written only after every attachment succeeds.

The evidence validator requires the completed directory's exact file inventory,
checks both sizes and digests, decompresses and parses every attachment with the
same duplicate-key and finite-number rules as ordinary evidence, and never executes
its contents. Missing, extra, renamed, malformed or modified sidecars invalidate the
whole result. `load_json_attachment` returns content only after that full validation.
Schemas 1 through 4 remain readable with their historical inline observations.

Attachments reduce checkout size and review noise; they do not weaken retention.
Summaries must identify the attachment and its domain-specific format, counts and
scientific verdict. Compression does not turn a large opaque payload into a useful
measurement, establish adoption, or replace domain-level recomputation from raw data.


### Legacy result storage

`ExperimentResult.save` now uses the runner's strict exclusive JSON writer. Existing
paths raise FileExistsError; arrays and unsupported objects must be represented
explicitly by their producing experiment, and nonfinite numbers are rejected before
creating output. Do not stringify a failed statistic: record an explicit undefined
value and its reason. The historical noise study uses null plus its existing
`degenerate` reason and `significant: false`. Loading rejects duplicate keys and
nonfinite numeric encodings. These safeguards do not add the missing run provenance
of an unmigrated experiment or turn its completion into scientific adoption.


## Historical experiment parameter files

The historical merge, association, noise, projection, scaling and phase adapters share `--parameters`.
Use a repository-relative UTF-8 JSON object containing only the protocol settings
you want to replace. Unspecified settings come from the selected smoke/full defaults;
arrays replace entire grids. For example, to change phase evaluation to one round:

```bash
python -m research.runner historical-phase --smoke --seeds 1 2 3 --tag phase-one-round-UNIQUE --parameters research/experiments/configs/phase_one_round.json
```

The file is archived alongside the fully resolved parameters. The runner checks
`expected_input_digests` against the entire captured input inventory before creating
a run, so the bytes parsed by the adapter must be the bytes captured. Unknown keys
and attempts to set seeds, tag or engine in the file fail; use their CLI flags.
Invalid domain values fail in the producer before trials. Smoke remains VOID and
full historical runs UNADOPTED. Customizing a grid does not preregister it: write
its hypotheses and bars before collecting scientific evidence.
