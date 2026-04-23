# Getting Started with Research Organization

Welcome to your research-first workspace! This guide will help you start using this structure effectively.

## 🎯 Philosophy

**Research First, Papers Second**

Don't organize by papers you want to write. Organize by:
1. **Questions** you want to answer
2. **Experiments** you need to run
3. **Evidence** you're collecting
4. **Claims** you can defend

Papers emerge naturally when you have validated claims.

---

## 📁 Directory Overview

```
research/
├── open_questions.md           ← START HERE: Track all questions
├── core_questions/             ← Investigate specific questions
├── experiments/                ← Design and run experiments
├── results/                    ← Analyze experimental data
├── claims/                     ← Validated claims ready for publication
└── papers/                     ← Construct papers from claims
    ├── _latex_infrastructure/  ← Fast LaTeX compilation
    ├── _shared_assets/         ← Figures, tables, bibliography
    └── drafts/                 ← Active paper writing
```

---

## 🚀 Quick Start

### Step 1: Identify Your Questions

Start by reading:

- `open_questions.md` for the broad tracker
- `core_questions/index.json` for the already curated questions
- `claims/index.json` for the current claim/evidence inventory

Then edit `open_questions.md` and list anything new you want to investigate:

```markdown
### Q01: My First Question
**Status:** Not Started
**Hypothesis:** [What are you claiming?]
**Why It Matters:** [Why is this important?]
```

### Step 2: Choose a Question to Investigate

First check whether the question already has a curated directory in
`core_questions/`.

If not, pick one question (start with something tractable):

```bash
cd core_questions
mkdir Q01_my_question
cd Q01_my_question
```

### Step 3: Document Your Hypothesis

Create `hypothesis.md`:
```markdown
# Hypothesis: Assembly Stability

## Central Claim
Sparse neural assemblies converge to stable attractors in O(log N) steps.

## Background
[Why do you think this is true?]

## Predictions
1. Convergence time should scale logarithmically
2. Stability should depend on connectivity threshold
3. ...

## How to Test
[Experimental design]
```

### Step 4: Run Experiments

Design experiments in `experiments/`:
```bash
cd ../../experiments
mkdir E01_convergence_test
```

Link experiments to questions in your documentation.

### Step 5: Analyze Results

Store analyzed results in `results/`:
```bash
cd ../results  
mkdir R01_convergence_analysis
```

### Step 6: Make Claims

When you have solid evidence, create a claim in `claims/`:
```bash
cd ../claims
mkdir C01_logarithmic_convergence
```

Then update `claims/index.json` so the repo reflects whether the result is now
a formalized claim or still only a claim-ready evidence summary.

### Step 7: Write Papers

**Only now** do you start writing papers. Copy claims into `papers/drafts/`.

---

## 📝 LaTeX Workflow

### One-Time Setup

1. **Install LaTeX:**
   - Windows: [MiKTeX](https://miktex.org/) or [TeX Live](https://tug.org/texlive/)
   - macOS: `brew install --cask mactex`
   - Linux: `sudo apt-get install texlive-full`

2. **Install VSCode LaTeX Workshop:**
   ```bash
   code --install-extension James-Yu.latex-workshop
   ```

3. **Configure VSCode:**
   Copy settings from `papers/_latex_infrastructure/README.md`

### Creating a Paper

1. **Copy template to drafts:**
   ```bash
   cd papers/drafts
   cp -r ../_latex_infrastructure/templates/simple my_paper
   cd my_paper
   ```

2. **Open in VSCode:**
   ```bash
   code main.tex
   ```

3. **Edit and save** - PDF updates automatically!

### Fast Iteration Workflow

```bash
# Quick compile (during writing)
make quick

# Full compile (before submission)
make full

# Watch mode (auto-rebuild)
make watch

# Windows (PowerShell)
.\build.ps1 -Mode quick
```

### Testing Equations

Create `test_eq.tex`:
```latex
\documentclass[border=2pt]{standalone}
\input{../../_latex_infrastructure/preambles/standard_preamble.tex}
\input{../../_latex_infrastructure/preambles/neuroscience_preamble.tex}

\begin{document}
\[
    \text{Your equation here}
\]
\end{document}
```

Compile in < 0.5 seconds!

---

## 🎨 Using Shared Assets

### Figures

1. **Create figure** (Python, TikZ, etc.)
2. **Save source** in `papers/_shared_assets/figures/source/`
3. **Generate PDF** in `papers/_shared_assets/figures/`
4. **Include in papers:**
   ```latex
   \includegraphics{../../_shared_assets/figures/my_figure.pdf}
   ```

### Tables

1. **Store data** in `papers/_shared_assets/tables/data/`
2. **Generate LaTeX** in `papers/_shared_assets/tables/`
3. **Include in papers:**
   ```latex
   \input{../../_shared_assets/tables/my_table.tex}
   ```

### Bibliography

Add references to `papers/_shared_assets/bibliography/references.bib`:
```bibtex
@article{author2025paper,
  title={Paper Title},
  author={Author, Name},
  journal={Journal Name},
  year={2025}
}
```

Cite in papers:
```latex
\citep{author2025paper}
```

---

## 🔬 Best Practices

### For Questions

- ✅ One question per directory
- ✅ Write hypothesis before experiments
- ✅ Document null results (they're valuable!)
- ✅ Link to related questions

### For Experiments

- ✅ Version control all code
- ✅ Document parameters completely
- ✅ Make reproducible
- ✅ Store raw data separately from analysis

### For Claims

- ✅ Only claim what you can defend
- ✅ Document evidence clearly
- ✅ State limitations explicitly
- ✅ Link to underlying questions and experiments

### For Papers

- ✅ One sentence per line (cleaner diffs)
- ✅ Compile often (quick builds)
- ✅ Commit frequently
- ✅ Test standalone equations first

---

## 📊 Tracking Progress

### Update These Regularly:

1. **`open_questions.md`** - Track all questions and status
2. **`core_questions/index.json`** - Track which questions are curated
3. **Individual question files** - Document progress
4. **Git commits** - Commit experiments and analysis
5. **Claims directory** - Add validated claims

### Status Workflow:

```
Question → Experiment → Results → Analysis → Claim → Paper
   ↓           ↓          ↓         ↓         ↓       ↓
Not Started  Running   Collected  Done   Validated  Published
```

---

## 🎯 Example Workflow

Let's say you want to investigate assembly stability:

1. **Add to `open_questions.md`:**
   ```markdown
   ### Q01: Assembly Stability
   **Status:** In Progress
   **Hypothesis:** Assemblies converge in O(log N) steps
   ```

2. **Create question directory only if it is not already curated:**
   ```bash
   cd core_questions
   mkdir Q01_assembly_stability
   ```

3. **Write hypothesis:**
   - What exactly are you claiming?
   - Why should this be true?
   - How will you test it?

4. **Design experiments:**
   ```bash
   cd ../experiments
   mkdir E01_convergence_scaling
   # Write experiment protocol
   ```

5. **Run experiments and collect data:**
   ```python
   # Your experiment code
   # Save results to ../results/
   ```

6. **Analyze results:**
   ```bash
   cd ../results
   mkdir R01_convergence_analysis
   # Create figures, tables, statistical analysis
   ```

7. **Draw conclusions:**
   - Update `core_questions/Q01_assembly_stability/conclusions.md`
   - If validated, create claim

8. **Create claim:**
   ```bash
   cd ../claims
   mkdir C01_logarithmic_convergence
   # Document claim, evidence, limitations
   ```

9. **Write paper (only now!):**
   ```bash
   cd ../papers/drafts
   cp -r ../_latex_infrastructure/templates/simple stability_paper
   # Pull in claims, figures, tables
   ```

---

## 🚨 Common Mistakes to Avoid

❌ **Don't:** Create paper structure before knowing what you'll write  
✅ **Do:** Investigate questions first, papers emerge naturally

❌ **Don't:** Skip documenting null results  
✅ **Do:** Document everything (null results teach us too)

❌ **Don't:** Make claims beyond your evidence  
✅ **Do:** State limitations explicitly

❌ **Don't:** Store figures only in papers  
✅ **Do:** Keep source files in shared assets

❌ **Don't:** Have a single monolithic bibliography  
✅ **Do:** Use topic-specific .bib files

---

## 💡 Tips for Success

1. **Start small**: Pick one tractable question
2. **Document early**: Write hypothesis before coding
3. **Iterate fast**: Use quick LaTeX builds
4. **Version control**: Commit experiments and analysis
5. **Stay organized**: Update status documents regularly
6. **Be honest**: Document limitations and null results

---

## 📚 Further Reading

- **Research workflow:** `README.md`
- **Curated question index:** `core_questions/index.json`
- **LaTeX infrastructure:** `papers/_latex_infrastructure/README.md`
- **Paper organization:** `papers/README.md`
- **Question tracking:** `open_questions.md`

---

## 🎉 You're Ready!

1. Read `open_questions.md`
2. Pick a question to investigate
3. Create hypothesis and experiment design
4. Run experiments and analyze results
5. Make validated claims
6. Write papers from claims

**Start with questions, end with papers. Good luck!** 🚀
