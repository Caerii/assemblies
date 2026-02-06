# Papers Organization

Papers are constructed from **validated research** in `../claims/`. This directory provides LaTeX infrastructure for rapid iteration and high-quality typesetting.

## 🎯 Philosophy

**Papers emerge from research, not the other way around.** We don't create paper structures until we have clear, validated claims to present.

## 📁 Directory Structure

```
papers/
├── _latex_infrastructure/     # Fast compilation & rendering tools
│   ├── preambles/              # Shared LaTeX preambles
│   ├── styles/                 # Custom LaTeX styles
│   ├── templates/              # Paper templates for different venues
│   ├── build_tools/            # Compilation scripts
│   └── README.md               # LaTeX workflow documentation
│
├── _shared_assets/             # Reusable components
│   ├── figures/                # All figures (with source files)
│   ├── tables/                 # All tables (LaTeX and data)
│   ├── equations/              # Reusable equation blocks
│   ├── bibliography/           # Master bibliography
│   └── code_listings/          # Code snippets for papers
│
└── drafts/                     # Active paper drafts
    └── [paper_name]/           # One folder per paper
        ├── main.tex
        ├── sections/
        ├── figures/            # Paper-specific figures (symlinks to shared)
        ├── Makefile            # Fast compilation
        └── notes/
```

## ⚡ Fast LaTeX Workflow

### Prerequisites

You'll need:
- **LaTeX Distribution**: TeX Live (recommended) or MiKTeX
- **Editor**: VSCode with LaTeX Workshop extension (best for real-time preview)
- **Build Tools**: Make (optional, for Makefiles)

### Real-Time Preview Setup

The LaTeX infrastructure supports:
- **Auto-compilation on save**
- **Real-time PDF preview**
- **Fast incremental builds**
- **Equation rendering without full compile**

See `_latex_infrastructure/README.md` for detailed setup.

## 🎨 Creating a New Paper

### Step 1: Identify Claims

Before creating a paper, identify which claims from `../claims/` you're presenting.

```bash
# List all validated claims
ls ../claims/

# Review claims to include
cat ../claims/C01_*/claim.md
```

### Step 2: Choose Template

Select appropriate template based on venue:
- `neurips`: NeurIPS conference format
- `nature`: Nature journal format
- `plos`: PLOS journals format
- `arxiv`: arXiv preprint format
- `custom`: Your custom format

### Step 3: Create Paper Structure

```bash
cd drafts
mkdir paper_name
cd paper_name

# Copy template
cp ../../_latex_infrastructure/templates/neurips/main.tex .
cp -r ../../_latex_infrastructure/templates/neurips/sections .

# Create Makefile for fast builds
cp ../../_latex_infrastructure/build_tools/Makefile .
```

### Step 4: Start Writing

Open in VSCode with LaTeX Workshop for real-time preview:
```bash
code main.tex
```

## 📚 Shared Assets Management

### Figures

Store figures in `_shared_assets/figures/` with:
- **Source files**: `.svg`, `.py` (matplotlib scripts), `.tikz`
- **Compiled versions**: `.pdf`, `.png`
- **Metadata**: `figure_catalog.md`

**Usage in papers:**
```latex
\includegraphics{../../_shared_assets/figures/performance_scaling.pdf}
```

### Tables

Store tables in `_shared_assets/tables/` with:
- **Data files**: `.csv`, `.json`
- **LaTeX tables**: `.tex`
- **Generation scripts**: `.py`

**Usage in papers:**
```latex
\input{../../_shared_assets/tables/benchmark_results.tex}
```

### Equations

Store reusable equation blocks in `_shared_assets/equations/`:
- `assembly_dynamics.tex`
- `complexity_bounds.tex`
- `information_theory.tex`

**Usage in papers:**
```latex
\input{../../_shared_assets/equations/assembly_dynamics.tex}
```

### Bibliography

Maintain master bibliography in `_shared_assets/bibliography/`:
- `master.bib`: All references
- `assembly_calculus.bib`: Domain-specific
- `ml_systems.bib`: ML/systems references
- `neuroscience.bib`: Neuroscience references

**Usage in papers:**
```latex
\bibliography{../../_shared_assets/bibliography/master}
```

## 🚀 Compilation Workflow

### Quick Compile (Development)

```bash
# In paper directory
make quick
```

Compiles without bibliography for fast iteration.

### Full Compile (Final)

```bash
make full
```

Compiles with bibliography, cross-references, and all links.

### Watch Mode (Real-Time)

```bash
make watch
```

Automatically recompiles on file changes.

### Clean Build

```bash
make clean
make full
```

## 📝 Paper Lifecycle

### 1. Draft Phase
- Active writing
- Frequent iteration
- Shared with collaborators
- Location: `drafts/[paper_name]/`

### 2. Submission Phase
- Polished version
- Ready for submission
- Version controlled
- Location: `drafts/[paper_name]/submission/`

### 3. Published Phase
- Accepted version
- Archived
- Location: `published/[year]/[venue]/[paper_name]/`

## 🎯 Best Practices

### Version Control

- Commit frequently during writing
- Use meaningful commit messages: "Add scaling analysis section"
- Tag important versions: `git tag v1.0-submission`

### File Organization

- One sentence per line (makes diffs cleaner)
- Keep sections in separate files
- Use consistent naming: `01_introduction.tex`, `02_methods.tex`

### Equations

- Label all equations: `\label{eq:assembly_stability}`
- Use semantic names: `eq:main_theorem` not `eq:equation1`
- Store complex derivations in appendix

### Figures

- High resolution (300 DPI minimum)
- Vector format when possible (PDF, SVG)
- Consistent styling across all figures
- Always include source files

### Tables

- Use booktabs package for professional tables
- Store data separately from presentation
- Include scripts to regenerate tables

## 🔧 LaTeX Infrastructure

See `_latex_infrastructure/README.md` for:
- Custom style files
- Shared preambles
- Build system documentation
- Troubleshooting guide
- Editor setup instructions

## 📊 Current Papers

### In Progress
*To be populated when papers are started*

### Submitted
*To be populated when papers are submitted*

### Published
*To be populated when papers are accepted*

---

**Remember**: Don't create a paper until you have validated claims to present. Quality over quantity.
