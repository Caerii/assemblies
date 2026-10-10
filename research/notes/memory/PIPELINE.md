# How a memory study goes from a question to the paper

One map of the refraction-memory programme's pipeline. For each step it gives what happens, which
module owns it, and what it leaves behind. The rules the pipeline enforces come from the
pre-registration's preface ([prereg/A00.md](prereg/A00.md)). Bars are fixed before data. A failed bar
is recorded as failed and never moved. Every number a registration saw beforehand is disclosed.

```
 probe ─► specify ─► register ─► implement ─► run (pinned) ─► collect ─► evaluate ─► record ─► register a claim ─► notebook, paper
```

| # | Step | Owner | Leaves behind |
|---|---|---|---|
| 0 | Probe | `research/notes/memory/probes/<date>/` | a script, a log, a README row |
| 1 | Specify | `research/experiments/memory_lib/` | a `Registration` object |
| 2 | Register | `research/amend.py` `Prereg.register` | `prereg/A<N>.md`, `ORDER`, the rebuilt PREREG file |
| 3 | Implement | `research/experiments/memory_<name>.py`, `research/runner.py` | the study module, its test, its runner key |
| 4 | Run | `research/runner.py` in a pinned worktree | `results/runs/<protocol>/<tag>/` |
| 5 | Collect | `research/collect.py`, `research/evidence.py` | the run directory and log in this checkout |
| 6 | Evaluate | the module's `evaluate`, `memory_lib` bars | the verdict of every bar and the numbers that decided it |
| 7 | Record | `research/amend.py` `Prereg.record`, `.scorecard` | the result section and the scorecard rows |
| 8 | Claim | `research/amend.py`, `neural_assemblies/theory_claims/` | a register entry, its citation, `docs/register.md` |
| 9 | Write up | by hand | the notebook section and the manuscript's propositions |

## 0. Probe: look before registering

A probe is a short, disclosed look at whether a question is worth registering and where its bars
should sit. Probes live in `probes/<date>/`, one script and one log each, listed in that folder's
README. They run only on **probe brains 900 to 999**, which no registered study may use. Every
number a probe produced appears in the registration under "Seen before registering". A probe is not
evidence, and no claim cites one.

## 1. Specify: one object that says everything

`research/experiments/memory_lib/` is the memory-study library. A registration is assembled from
its parts:

- `Cell.of(n, k, p)` sets the operating point. τ follows Amendment 41's measured rule (n/k up to
  about 50; n/k / 2 from about 100; refused in between), and β = θ(n, k, p). It also checks the
  regime.
- `Plan` and `Store` (`stores.py`) say what is written and read: sequences, uses per word, the walk
  salt, plain or comparator writes, and replay, sleep and downscaling, all on the fast paths in
  `memory_fast/`.
- `Bar(id, name, statement, checks)` holds `Check(reading, op, threshold)` (`bars.py`). Bars are
  data, judged by the **confidence bound** of the ensemble over brains (`how="mean"` is the older
  reading).
- `Registration` (`spec.py`) gathers the amendment number, preamble, what was seen, protocol,
  cells, brains, bars and interpretation. `problems()` checks its cells and brains against the
  **ledger** (`ledger.py`). The ledger is read from the registered modules themselves, and since
  Amendment 37 it requires new cells and new brains, outside the probe range.

## 2. Register: before any data

`research.amend.Prereg(root).register(N, reg.registration_text(date))` writes `prereg/A<N>.md`,
appends it to `ORDER` and rebuilds `PREREG_refraction_memory.md`.

**The built file is never edited directly.** Its sources live in `prereg/`, and a test checks that
the file equals their concatenation. `Prereg.write_index()` regenerates `prereg/README.md`.

Commit the registration. That commit is the one the run is pinned to.

## 3. Implement: the study module

Each study module is `research/experiments/memory_<name>.py` and follows the same shape:

- `plan` gives the cells;
- `measure` / `experiment` produces the observations;
- `evaluate` turns observations into bars;
- `main` calls `run_experiment(...)` with the protocol, the registration path and the parameters.

Its docstring names the amendment, and the ledger reads that docstring. The module also needs a key
in `research/runner.py`'s `EXPERIMENTS`.

Its test, `neural_assemblies/tests/test_memory_<name>.py`, checks three things:

- `evaluate` on synthetic observations, with bars set off the knife edge;
- novelty through `ledger.check_new`;
- the `--smoke` path.

The test is also listed in `.github/workflows/research-contracts.yml`.

## 4. Run: from a pinned worktree

The runner (`research/runner.py`) validates the arguments, takes the machine's one-GPU device lock,
and archives every source input into the content-addressed store (`research/source_store.py`).
After the run it hashes them again. **If any source changed while the run was going, the run is
void.** So studies run from a worktree pinned at the registration commit, beside the repository
(never under a temp directory, which the runner refuses):

    git worktree add --detach C:/Github/assemblies-runs-<date> <registration commit>
    python -m research.runner <key> --registration research/notes/memory/PREREG_refraction_memory.md \
        --tag <name>-<date> --seeds ...

`--smoke` checks the API only. Its observations are void as evidence.

## 5. Collect and validate

    python -m research.collect --clean --worktree W --log x.log:NAME RUN...
    python -m research.evidence validate research/results/runs/<protocol>/<tag>

`collect` copies the run directory and the source objects its manifest names, refuses to overwrite
a different copy, and clears them from the worktree. `evidence validate` checks the record against
its archived source (`research/source_archive.py`).

## 6. Evaluate

The module's `evaluate(observations)` gives the **registered** verdicts. A registration built with
`memory_lib` is also judged by `Registration.judge(observations)`, which reads each bar by its
confidence bound. That gives the deciding number for every bar.

An engine change that claims to be exact must pass `python -m research.replay`: it reruns a
recorded study and requires byte-identical observations.

## 7. Record

- `Prereg.record(N, reg.result_text(...))` appends the result to `prereg/A<N>.md`.
- `Prereg.scorecard(reg.scorecard_rows(...))` adds one row per bar.

Both rebuild the PREREG file in its own line endings. Write the result as the bars came out, with
the failures included.

## 8. Register a claim

A result the programme stands on becomes a claim in the theory register, which is built from
`neural_assemblies/theory_claims/`, one module per programme:

- `amend.theory_result(...)` generates the `Result(...)` block (status MEASURED, its evidence ref,
  its sensitivity checks).
- `amend.add_claim("neural_assemblies/theory_claims", block, before_id)` inserts it into the module
  that holds the anchor claim.
- `amend.allow_citation` lists it in `test_theory_citations.py`.
- `amend.render_register()` regenerates `docs/register.md`, whose freshness a test checks.

Cite claims by ID, never by paraphrase.

## 9. Write up

These two are edited by hand, because they hold prose and argument:

- the notebook: `research/theory/assembly_statmech.tex`;
- the manuscript: `research/papers/drafts/sequence_budget/main.tex`, which has the propositions,
  the methods table and the "What failed" list.

Both cite the register IDs and the PREREG anchors (`amend.anchor(heading)`).
