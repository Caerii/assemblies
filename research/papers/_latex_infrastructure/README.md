# LaTeX Infrastructure for Fast Iteration

This directory provides tools and templates for **rapid LaTeX development** with real-time preview and equation rendering.

## 🎯 Goals

- **Fast compilation**: Incremental builds in < 1 second
- **Real-time preview**: See changes immediately
- **Equation rendering**: Test math without full compile
- **Consistent styling**: Shared preambles and styles
- **Easy debugging**: Clear error messages

## 📁 Structure

```
_latex_infrastructure/
├── preambles/          # Shared LaTeX preambles
├── styles/             # Custom .sty files
├── templates/          # Paper templates by venue
├── build_tools/        # Makefiles and build scripts
└── README.md           # This file
```

## ⚡ Quick Start

### 1. Install Prerequisites

#### Windows (Recommended)

```powershell
# Install TeX Live (comprehensive)
# Download from: https://tug.org/texlive/acquire-netinstall.html

# Or install MiKTeX (lighter weight)
# Download from: https://miktex.org/download

# Install VSCode with LaTeX Workshop extension
code --install-extension James-Yu.latex-workshop
```

#### Alternative: Overleaf

If you prefer browser-based editing, upload the template to Overleaf for real-time preview.

### 2. Configure VSCode

Create `.vscode/settings.json` in your paper folder:

```json
{
  "latex-workshop.latex.autoBuild.run": "onSave",
  "latex-workshop.latex.recipe.default": "first",
  "latex-workshop.view.pdf.viewer": "tab",
  "latex-workshop.latex.clean.enabled": true,
  "latex-workshop.latex.clean.fileTypes": [
    "*.aux", "*.bbl", "*.blg", "*.idx", "*.ind", 
    "*.lof", "*.lot", "*.out", "*.toc", "*.acn", 
    "*.acr", "*.alg", "*.glg", "*.glo", "*.gls", 
    "*.fls", "*.log", "*.fdb_latexmk", "*.synctex.gz"
  ],
  "latex-workshop.latex.recipes": [
    {
      "name": "Quick Build (PDFLaTeX)",
      "tools": ["pdflatex"]
    },
    {
      "name": "Full Build (PDFLaTeX + BibTeX)",
      "tools": ["pdflatex", "bibtex", "pdflatex", "pdflatex"]
    }
  ],
  "latex-workshop.latex.tools": [
    {
      "name": "pdflatex",
      "command": "pdflatex",
      "args": [
        "-synctex=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "%DOC%"
      ]
    },
    {
      "name": "bibtex",
      "command": "bibtex",
      "args": ["%DOCFILE%"]
    }
  ]
}
```

### 3. Start Writing

1. Create paper from template
2. Open `main.tex` in VSCode
3. Edit and save
4. PDF updates automatically in preview pane

## 📝 Shared Preambles

### `standard_preamble.tex`

Common packages and settings for all papers:
```latex
% Math packages
\usepackage{amsmath, amssymb, amsthm}
\usepackage{mathtools}

% Graphics and figures
\usepackage{graphicx}
\usepackage{tikz}
\usepackage{pgfplots}

% Tables
\usepackage{booktabs}
\usepackage{multirow}

% Algorithms
\usepackage{algorithm}
\usepackage{algorithmic}

% Bibliography
\usepackage{natbib}

% Hyperlinks
\usepackage{hyperref}
\usepackage{cleveref}
```

### `neuroscience_preamble.tex`

Additional packages for neuroscience papers:
```latex
% Neuroscience-specific notation
\newcommand{\assembly}[1]{\mathcal{A}_{#1}}
\newcommand{\neurons}[1]{\mathcal{N}_{#1}}
\newcommand{\sparsity}{\rho}
\newcommand{\activation}[1]{a_{#1}}
```

### `algorithms_preamble.tex`

Additional packages for algorithmic papers:
```latex
% Complexity notation
\newcommand{\bigO}{\mathcal{O}}
\newcommand{\bigOmega}{\Omega}
\newcommand{\bigTheta}{\Theta}

% Algorithm environments
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newtheorem{corollary}{Corollary}
```

## 🎨 Templates

### Available Templates

1. **NeurIPS** (`templates/neurips/`)
   - 8-page conference format
   - Includes style file
   - Example paper structure

2. **Nature** (`templates/nature/`)
   - Nature journal format
   - High-quality figure requirements
   - Strict length limits

3. **PLOS Computational Biology** (`templates/plos_compbio/`)
   - Open access journal
   - Extended methods sections
   - Supplementary materials

4. **arXiv** (`templates/arxiv/`)
   - Preprint format
   - No page limits
   - Flexible structure

5. **Custom** (`templates/custom/`)
   - Your preferred styling
   - Customizable templates

### Template Structure

Each template includes:
```
templates/[venue]/
├── main.tex              # Main document
├── [venue].sty           # Style file (if needed)
├── sections/             # Section templates
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── methods.tex
│   ├── results.tex
│   └── discussion.tex
├── Makefile              # Build automation
└── README.md             # Template-specific notes
```

## 🔧 Build Tools

### Makefile (Cross-platform)

Basic Makefile for any paper:

```makefile
# Paper name (change this)
PAPER = main

# Quick build (no bibliography)
quick:
	pdflatex $(PAPER).tex

# Full build (with bibliography)
full:
	pdflatex $(PAPER).tex
	bibtex $(PAPER)
	pdflatex $(PAPER).tex
	pdflatex $(PAPER).tex

# Watch mode (requires latexmk)
watch:
	latexmk -pvc -pdf $(PAPER).tex

# Clean auxiliary files
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz

# Clean everything including PDF
cleanall: clean
	rm -f $(PAPER).pdf

.PHONY: quick full watch clean cleanall
```

### PowerShell Build Script

For Windows users without Make:

```powershell
# build.ps1
param(
    [string]$Mode = "quick"
)

$PAPER = "main"

switch ($Mode) {
    "quick" {
        pdflatex "$PAPER.tex"
    }
    "full" {
        pdflatex "$PAPER.tex"
        bibtex $PAPER
        pdflatex "$PAPER.tex"
        pdflatex "$PAPER.tex"
    }
    "clean" {
        Remove-Item *.aux, *.bbl, *.blg, *.log, *.out, *.toc, *.synctex.gz -ErrorAction SilentlyContinue
    }
}
```

Usage:
```powershell
.\build.ps1 -Mode quick
.\build.ps1 -Mode full
.\build.ps1 -Mode clean
```

## 🎯 Fast Equation Testing

### Standalone Equation File

Create `test_equation.tex`:

```latex
\documentclass[border=2pt]{standalone}
\usepackage{amsmath, amssymb}

% Import your custom commands
\input{../../_latex_infrastructure/preambles/neuroscience_preamble.tex}

\begin{document}
\begin{equation}
    % Test your equation here
    \assembly{t+1} = \text{TopK}\left(\sum_{i} w_i \cdot \activation{i}^t\right)
\end{equation}
\end{document}
```

Compile quickly:
```bash
pdflatex test_equation.tex
```

Opens in < 0.5 seconds!

## 🐛 Troubleshooting

### Compilation Errors

**Error: Missing package**
```bash
# Install missing package (TeX Live)
tlmgr install <package-name>

# Or let MiKTeX auto-install
```

**Error: Bibliography not found**
```bash
# Make sure .bib file path is correct
# Run full build, not quick build
make full
```

**Error: Figure not found**
```bash
# Check relative path to figures
# Make sure figures are compiled (PDF/PNG)
```

### Performance Issues

**Slow compilation**
- Use `\includeonly{sections/intro}` to compile only specific sections
- Enable draft mode: `\documentclass[draft]{article}`
- Use PDF figures instead of generating with TikZ during drafts

**Large PDF size**
- Compress figures before including
- Use `\pdfcompresslevel=9` in preamble
- Convert PNG to PDF with compression

## 📚 Best Practices

### Equation Workflow

1. **Test equations standalone** in `test_equation.tex`
2. **Save reusable equations** in `../_shared_assets/equations/`
3. **Use semantic labels**: `\label{eq:assembly_stability}`
4. **Define custom commands** for repeated notation

### Figure Workflow

1. **Create source files** (Python, TikZ, SVG)
2. **Generate PDFs** at high resolution
3. **Store in shared assets** with metadata
4. **Include with relative paths** in papers

### Table Workflow

1. **Store data** in CSV/JSON
2. **Generate LaTeX tables** with scripts
3. **Save in shared assets**
4. **Input in papers** with `\input{}`

### Bibliography Workflow

1. **Maintain master BibTeX file**
2. **Use consistent keys**: `papadimitriou2020brain`
3. **Include DOIs** for all references
4. **Organize by topic** in separate .bib files

## 🚀 Advanced Tips

### Real-Time Equation Preview

Install Math Preview extension in VSCode:
```bash
code --install-extension mathpresso.math-preview
```

See equations rendered without compiling!

### Git Integration

Create `.gitignore`:
```
*.aux
*.bbl
*.blg
*.log
*.out
*.toc
*.synctex.gz
*.fdb_latexmk
*.fls
*.pdf  # Usually gitignore PDFs, regenerate from source
```

### Collaborative Writing

Use Overleaf for collaboration:
1. Export template to ZIP
2. Upload to Overleaf
3. Share with collaborators
4. Sync back to git when ready

## 📖 Resources

- **LaTeX Workshop Docs**: https://github.com/James-Yu/LaTeX-Workshop/wiki
- **Overleaf Tutorials**: https://www.overleaf.com/learn
- **TeX StackExchange**: https://tex.stackexchange.com/
- **TikZ Examples**: https://texample.net/tikz/

---

**Now you can iterate quickly and render equations in real-time!**
