# Simple Paper Template

A minimal, clean template for academic papers with all the infrastructure set up for fast iteration.

## Quick Start

### Option 1: VSCode with LaTeX Workshop (Recommended)

1. **Open the template folder** in VSCode
2. **Open `main.tex`**
3. **Edit and save** - PDF updates automatically!

### Option 2: Command Line

#### Windows (PowerShell)
```powershell
# Quick compile (fast iteration)
..\..\..\build_tools\build.ps1 -Mode quick

# Full compile (with bibliography)
..\..\..\build_tools\build.ps1 -Mode full

# Watch mode (auto-rebuild)
..\..\..\build_tools\build.ps1 -Mode watch
```

#### macOS/Linux
```bash
# Quick compile
make quick

# Full compile  
make full

# Watch mode
make watch

# Clean auxiliary files
make clean
```

## What's Included

- ✅ Standard LaTeX packages (math, figures, algorithms)
- ✅ Neuroscience-specific notation (assemblies, neurons, dynamics)
- ✅ Example equations, algorithms, figures, tables
- ✅ Bibliography setup with example references
- ✅ Appendix template for proofs and extended results
- ✅ Clean, professional styling

## Customization

### Change Paper Metadata

Edit in `main.tex`:
```latex
\title{Your Paper Title Here}
\author{Your Name...}
\date{\today}
```

### Add Figures

1. **Create figure** (Python, TikZ, etc.)
2. **Save to shared assets**: `../../../../_shared_assets/figures/my_figure.pdf`
3. **Include in paper**:
```latex
\includegraphics[width=0.8\textwidth]{../../../_shared_assets/figures/my_figure.pdf}
```

### Add References

1. **Add to** `../../../_shared_assets/bibliography/references.bib`
2. **Cite in paper**: `\citep{authorYEARkeyword}`
3. **Compile with full build** to update bibliography

### Use Custom Notation

Already defined in `neuroscience_preamble.tex`:
- `\assembly{i}` → Assembly in area i
- `\neurons{i}` → Neurons in area i  
- `\activation{i}{t}` → Activation at time t
- `\TopK` → Top-K selection operator
- And many more...

See `../../../_latex_infrastructure/preambles/neuroscience_preamble.tex` for full list.

## File Structure

```
simple/
├── main.tex              # Your paper (edit this!)
├── README.md             # This file
└── [generated files]     # .aux, .log, .pdf, etc.
```

## Tips for Fast Iteration

### 1. Use Quick Builds During Writing
```bash
make quick    # Fast, no bibliography
```

### 2. Test Equations Standalone

Create `test_eq.tex`:
```latex
\documentclass[border=2pt]{standalone}
\input{../../../_latex_infrastructure/preambles/standard_preamble.tex}
\input{../../../_latex_infrastructure/preambles/neuroscience_preamble.tex}

\begin{document}
\[
    \assembly{t+1} = \TopK\left(\sum_{j} w_j \cdot \activation{j}{t}, k\right)
\]
\end{document}
```

Compile in < 0.5 seconds!

### 3. One Sentence Per Line

Makes git diffs cleaner:
```latex
% Good
This is sentence one.
This is sentence two.
This is sentence three.

% Less good
This is sentence one. This is sentence two. This is sentence three.
```

### 4. Comment Out Sections

Speed up compilation:
```latex
% \input{sections/related_work.tex}  % Skip this section for now
```

## Troubleshooting

### PDF Won't Update

1. Clean and rebuild:
```bash
make clean
make full
```

2. Check for errors in `.log` file

### Bibliography Not Showing

1. Make sure you have at least one `\cite{}` command
2. Run full build (not quick)
3. Check `.bib` file is in correct location

### Equation Errors

1. Check matching braces: `\left(` needs `\right)`
2. Use `\text{}` for text in math mode
3. Escape special characters: `\&`, `\%`, `\$`

### Figure Not Found

1. Check path is correct (use forward slashes even on Windows)
2. Make sure figure exists and has correct extension
3. Try absolute path temporarily for debugging

## Next Steps

1. **Copy this template** to `../../../drafts/my_paper/`
2. **Edit `main.tex`** with your content
3. **Add figures** to shared assets
4. **Write your paper!**

Happy writing! 🎉
