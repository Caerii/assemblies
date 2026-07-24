# Upstream reference clones

Shallow git clones of author reference implementations for literature parity work.
**Not vendored into the package** — compare against these, then pin goldens in
`research/literature/parity/golden/`.

## Sync

```powershell
uv run python research/literature/sync_reference_repos.py
```

Set `REF_ROOT` to override the clone directory (default: this folder).

## Repositories

| Directory | Upstream | Papers / role |
|-----------|----------|----------------|
| [`mdabagia-nemo/`](mdabagia-nemo/) | [mdabagia/nemo](https://github.com/mdabagia/nemo) | [dabagia.org/nemo](https://dabagia.org/nemo/) Jekyll site source — coin, sequences, assemblies pages |
| [`mdabagia-learning-with-assemblies/`](mdabagia-learning-with-assemblies/) | [learning-with-assemblies](https://github.com/mdabagia/learning-with-assemblies) | **COLT 2022** MNIST / halfspace / stimulus-class notebooks |
| [`mdabagia-SantoshHebbian/`](mdabagia-SantoshHebbian/) | [SantoshHebbian](https://github.com/mdabagia/SantoshHebbian) | Assembly convergence heuristics (Vempala lab); ITCS scaling research |
| [`mdabagia-NeuralTuringMachine/`](mdabagia-NeuralTuringMachine/) | [NeuralTuringMachine](https://github.com/mdabagia/NeuralTuringMachine) | PyTorch NTM (Graves 2014) — **not AC**; archived for completeness |
| [`dmitropolsky-assemblies/`](dmitropolsky-assemblies/) | [dmitropolsky/assemblies](https://github.com/dmitropolsky/assemblies) | Canonical Python AC: parser, learner, simulations |

Machine-readable map: [`../research/literature/reference/manifest.json`](../research/literature/reference/manifest.json).

**Live site ↔ repo ↔ package:** [`../research/literature/reference/nemo_site.json`](../research/literature/reference/nemo_site.json) maps [dabagia.org/nemo](https://dabagia.org/nemo/) pages to claim rows and figure assets.

## Environment variables

| Variable | Default | Used by |
|----------|---------|---------|
| `REF_ROOT` | `.reference/` | `sync_reference_repos.py` |
| `REF_ASSEMBLIES_PATH` | `.reference/dmitropolsky-assemblies` | `test_cross_repo_parity.py` |
| `REF_NEMO_PATH` | `.reference/mdabagia-nemo` | future coin/sequence cross-checks |
| `REF_COLT2022_PATH` | `.reference/mdabagia-learning-with-assemblies` | MNIST notebook parity |

## What we pull from each (priority)

### mdabagia/nemo — **high**

- `brain.py` — `RandomChoiceArea.flip`, `FSMNetwork`, `MarkovNetwork` (softmax + noise coin, not k-split)
- `nemo-demo.ipynb` — Markov estimation, trigram babbler (Fig 8 analog)
- `docs/coinflipping.markdown` — dabagia.org coin-flipping page text + figure URLs

**Gap vs this repo:** `neural_assemblies` uses proportional k-split in `RandomChoiceArea.flip`; paper/repo reference uses weight softmax + input noise.

### learning-with-assemblies — **high**

- `MNIST.ipynb` — hierarchical assembly-per-class (COLT22-E04 target)
- `Stimulus-Classes.ipynb`, `Halfspace.ipynb` — separable distribution protocols
- `brain.py` / `nemo.py` — shared NEMO engine variant

**Gap vs this repo:** toy 10-class golden only; full MNIST still needs notebook port or pinned run.

### SantoshHebbian — **medium**

- `heuristics.py`, `calc_converge_value.py`, `assembly.py` — convergence round estimates
- Useful for ITCS19-E02 / PNAS scaling sweeps, not yet wired to parity tests

### NeuralTuringMachine — **low**

- Unrelated to assembly calculus reproduction matrix; kept as author archive only.

## Still missing (not in mdabagia GitHub)

| Item | Where to get it |
|------|-----------------|
| Sequences paper full sim repo | Mostly in `dmitropolsky/assemblies` + this package; no separate dabagia sequences repo |
| TACL / NALOMA parsers | `dmitropolsky/assemblies` (`parser.py`, `recursive_parser.py`) |
| NEMO 2025 language acquisition | `mitropolsky2025` → this repo `EmergentParser` |
| Hoff E%-WTA continuous engine | Paper only; partial `EPercentPolicy` in package |
| PNAS PDF (paywall) | [PMC7459898](https://pmc.ncbi.nlm.nih.gov/articles/PMC7459898/) |
