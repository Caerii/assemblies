# Documentation

Use these docs as a map of the repository, not as a substitute for the tests or
research artifacts.

## Main Guides

| Document | Read it for |
|----------|-------------|
| [api.md](api.md) | Imports, public objects, compatibility shims, and common commands. |
| [architecture.md](architecture.md) | Runtime layers, engines, automata helpers, language modules, and archive layout. |
| [scientific_status.md](scientific_status.md) | Which claims are tested in the package, which are experimental, and which belong to the literature. |
| [supported_surfaces.md](supported_surfaces.md) | What is maintained as package code, what is compatibility code, and what is research-only. |
| [project_context.md](project_context.md) | Project history, authorship, collaboration context, and research motivation. |
| [references.md](references.md) | Papers and background literature behind the package and research program. |
| [literature.md](literature.md) | Full AC field map, implementation parity matrix, and gap priorities. |
| [packaging.md](packaging.md) | Release workflow for the `neural-assemblies` PyPI package. |
| [contributing.md](contributing.md) | Contributor setup and expected checks. |
| [documentation_style.md](documentation_style.md) | Writing standards for docs in this repo. |
| [register.md](register.md) | Every adopted result with its evidence and caveats, rendered from `neural_assemblies/theory.py`. |
| [../research/notes/README.md](../research/notes/README.md) | The reading map for the registrations and design notes: what each line concluded, which file to open first. |
| [../research/experiments/README.md](../research/experiments/README.md) | The active experiment scripts by line, with typical runs; results are in [../research/results/](../research/results/README.md). |

## Package Sections

- [core](../neural_assemblies/core/README.md)
- [compute](../neural_assemblies/compute/README.md)
- [assembly_calculus](../neural_assemblies/assembly_calculus/)
- [simulation](../neural_assemblies/simulation/README.md)
- [language](../neural_assemblies/language/README.md)
- [lexicon](../neural_assemblies/lexicon/README.md)
- [nemo](../neural_assemblies/nemo/README.md)
- [viz](../neural_assemblies/viz/README.md)

## Research

The research tree has its own workflow:

- [research/README.md](../research/README.md)
- [research/literature/index.json](../research/literature/index.json)
- [research/claims/index.json](../research/claims/index.json)
- [research/core_questions/index.json](../research/core_questions/index.json)

Package docs should not overrule those artifacts. If a scientific statement
depends on a particular experiment, link to the experiment or the indexed claim.
