# Research Organization

This directory contains the **research-first** organization of the neural assembly simulation project. Papers emerge from completed research, not the other way around.

## 🎯 Philosophy

**Research First, Papers Second**: We organize by scientific questions and hypotheses, not by intended publications. Papers are created only when we have clear, validated claims to make.

## 📁 Directory Structure

### Core Research (`core_questions/`)

Each subdirectory represents a **fundamental research question**. Questions drive everything.

**Template for each question:**
```
core_questions/
└── Q01_assembly_stability/
    ├── hypothesis.md              # What exactly are we claiming?
    ├── theoretical_basis.md       # Mathematical/theoretical foundation
    ├── experiments.md             # How do we test this?
    ├── results.md                 # What have we found?
    ├── analysis.md                # Interpretation of results
    └── conclusions.md             # What can we claim?
```

**Current Questions:**
- *To be populated as research develops*

### Experiments (`experiments/`)

Detailed experimental designs and protocols. Each experiment tests specific hypotheses.

**Template:**
```
experiments/
└── E01_[experiment_name]/
    ├── design.md                  # Experimental design
    ├── protocol.md                # Detailed protocol
    ├── parameters.json            # All parameters
    ├── code/                      # Experiment-specific code
    └── raw_results/               # Raw data
```

### Results (`results/`)

Analyzed results from experiments. This is where raw data becomes insights.

**Template:**
```
results/
└── R01_[result_name]/
    ├── summary.md                 # High-level summary
    ├── data/                      # Processed data
    ├── figures/                   # Generated figures
    ├── tables/                    # Generated tables
    └── analysis_notebooks/        # Jupyter notebooks, etc.
```

### Claims (`claims/`)

**What can we actually claim based on the evidence?** This is the bridge between research and papers.

**Template:**
```
claims/
└── C01_[claim_name]/
    ├── claim.md                   # The claim itself
    ├── evidence.md                # Supporting evidence
    ├── limitations.md             # What we can't claim
    ├── related_questions.md       # What questions this addresses
    └── suitable_venues.md         # Where this could be published
```

Current repo note: the directory also contains a lightweight
`claims/index.json` inventory so the current state of formalized claims versus
claim-ready evidence summaries is explicit even before every claim is promoted
to a full `CXX_*` folder.

### Papers (`papers/`)

Papers are constructed from validated claims. Built last, not first.

See `papers/README.md` for detailed paper organization.

## 🔄 Workflow

1. **Ask a question** → Create in `core_questions/`
2. **Design experiments** → Create in `experiments/`
3. **Run experiments** → Generate data
4. **Analyze results** → Create in `results/`
5. **Make claims** → Create in `claims/`
6. **Write papers** → Construct from claims in `papers/`

## 🚨 Important Principles

### What Goes Where?

- **Hypothesis not yet tested?** → `core_questions/`
- **Testing methodology?** → `experiments/`
- **Raw or analyzed data?** → `results/`
- **Can you defend this claim?** → `claims/`
- **Ready to submit?** → `papers/`

### Quality Gates

- ✅ **Questions**: Can be speculative, exploratory
- ✅ **Experiments**: Must be reproducible, well-documented
- ✅ **Results**: Must be validated, statistically sound
- ✅ **Claims**: Must be defensible, evidence-based
- ✅ **Papers**: Must meet publication standards

## 📊 Current Status

### Questions Under Investigation
- See `open_questions.md` and `core_questions/README.md`. The question-driven
  structure exists, but the per-question directories are not yet fully curated.
- The canonical suite inventory now lives in `research/registry.json`, which
  maps experiment families to code paths, result directories, and recommended
  entry points.

### Active Experiments
- Active suites currently include `applications`, `biological_validation`,
  `distinctiveness`, `information_theory`, `primitives`, `stability`, `vocab`,
  and `infrastructure`.
- See `experiments/README.md` and the corresponding `results/` subdirectories
  for current artifacts.

### Validated Claims
- See `claims/index.json` for the current inventory.
- Current indexed status:
  - `1` formalized claim
  - `6` claim-ready evidence summaries

### Papers in Progress

Paper scaffolding lives under `research/papers/`, but the primary source of
truth for experiment coverage should be the registry and results directories.

## 🎯 Next Steps

1. Keep `research/registry.json` current as suites are added or renamed
2. Use `research/experiments/MANIFEST_TEMPLATE.json` for new experiments
3. Validate the suite map with `uv run python research/experiments/infrastructure/validate_registry.py`
4. Populate `core_questions/` with the most important fundamental questions
5. Promote evidence summaries into formalized claims as they become bounded and defensible
6. Construct papers from validated claims

---

**Remember**: Science is about asking good questions and finding honest answers, not about filling publication quotas.
