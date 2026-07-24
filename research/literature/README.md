# Assembly Calculus Literature Inventory

Canonical bibliography and implementation-parity map for the entire AC field.

## Files

| File | Role |
|------|------|
| [index.json](index.json) | Machine-readable paper list, mechanisms, implementation status, gaps |
| [REPRODUCTION_MATRIX.md](REPRODUCTION_MATRIX.md) | **Claim → test → gap** matrix (Donoho reproducibility target) |
| [reproduction_matrix.json](reproduction_matrix.json) | Base claim rows (CI-parity focused) |
| [reproduction_matrix_supplement.json](reproduction_matrix_supplement.json) | Pass-2 exhaustive audit (configs, legacy, extended claims) |
| [parity/configs/](parity/configs/) | Per-paper YAML parameter registries |
| [reproduce.py](reproduce.py) | One-command runner for pinned claims |
| [reference/manifest.json](reference/manifest.json) | Upstream clone map (mdabagia, dmitropolsky, …) |
| [reference/nemo_site.json](reference/nemo_site.json) | [dabagia.org/nemo](https://dabagia.org/nemo/) pages → papers → parity status |
| [sync_reference_repos.py](sync_reference_repos.py) | Clone/update `.reference/` repos |
| [../../.reference/README.md](../../.reference/README.md) | Human guide to reference clones |
| [../../docs/literature.md](../../docs/literature.md) | Human-readable field guide and parity matrix |
| [../../docs/references.md](../../docs/references.md) | Curated bibliography with links |
| [../papers/_shared_assets/bibliography/references.bib](../papers/_shared_assets/bibliography/references.bib) | BibTeX for paper writing |

## Status Legend

| Status | Meaning |
|--------|---------|
| `implemented` | Maintained `neural_assemblies/` code + tests |
| `partial` | Some mechanisms; paper protocol incomplete |
| `legacy` | Only under `legacy/` or unmaintained paths |
| `research` | Experiments only; not a package claim |
| `not_started` | No meaningful code yet |

## How To Use

1. Find the paper in [index.json](index.json).
2. Check **all** claims in [reproduction_matrix.json](reproduction_matrix.json) + [reproduction_matrix_supplement.json](reproduction_matrix_supplement.json).
3. Read hyperparameters in [parity/configs/](parity/configs/).
4. Implement in `neural_assemblies/`; add parity test; flip row to `pinned`.
5. Run `uv run python research/literature/validate_matrix.py`.
6. Sync author repos: `uv run python research/literature/sync_reference_repos.py`.

Plans for closing gaps live in [../plans/PRIORITIES_AND_GAPS.md](../plans/PRIORITIES_AND_GAPS.md).
