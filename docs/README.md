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
| [packaging.md](packaging.md) | Release workflow for the `neural-assemblies` PyPI package. |
| [contributing.md](contributing.md) | Contributor setup and expected checks. |
| [documentation_style.md](documentation_style.md) | Writing standards for docs in this repo. |

## Package Sections

- [core](../neural_assemblies/core/README.md)
- [compute](../neural_assemblies/compute/README.md)
- [assembly_calculus](../neural_assemblies/assembly_calculus/)
- [simulation](../neural_assemblies/simulation/README.md)
- [language](../neural_assemblies/language/README.md)
- [lexicon](../neural_assemblies/lexicon/README.md)
- [nemo](../neural_assemblies/nemo/README.md)

## Research

The research tree has its own workflow:

- [research/README.md](../research/README.md)
- [research/claims/index.json](../research/claims/index.json)
- [research/core_questions/index.json](../research/core_questions/index.json)

Package docs should not overrule those artifacts. If a scientific statement
depends on a particular experiment, link to the experiment or the indexed claim.
