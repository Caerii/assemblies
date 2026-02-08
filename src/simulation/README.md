# Simulation — Runners and experiments

The **simulation** module provides ready-made runners and plotting helpers for assembly-calculus experiments: projection, association, merge, pattern completion, density, and Turing-style simulations.

## What’s here

| Component | File | Role |
|-----------|------|------|
| **project_sim** | `projection_simulator.py` | Project stimulus → area, recur; return weights / convergence |
| **association_sim** | `association_simulator.py` | Associate A,B → C; recovery and chain experiments |
| **merge_sim** | `merge_simulator.py` | Merge A,B → C; composition quality |
| **pattern_com** | `pattern_completion.py` | Pattern completion from partial input (alpha, iterations) |
| **density_sim** | `density_simulator.py` | Density / overlap experiments |
| **advanced_simulations** | `advanced_simulations.py` | Reciprocal projection, separate assemblies |
| **turing_simulations** | `turing_simulations.py` | Larger-k and tape-erase style runs (Turing-machine ideas) |
| **plotting_utils** | `plotting_utils.py` | Plot projection, merge, association, overlap, density |

## Quick use

```python
from src.simulation import project_sim, merge_sim, association_sim, pattern_com
from src.simulation import plot_project_sim, plot_merge_sim

# Projection: stimulus → area, recur
weights = project_sim(n=100_000, k=317, p=0.01, beta=0.05, t=50)

# Merge A,B → C
a_w, b_w, c_w = merge_sim(n=100_000, k=317, p=0.01, beta=0.05)

# Association A,B → C
result = association_sim(...)

# Pattern completion (partial assembly)
weights, winners = pattern_com(alpha=0.5, comp_iter=5)
```

## Relation to research

- **Experiments** — Scripts and protocols live under [research/experiments](../../research/experiments/).
- **Results** — Analyzed results and writeups are in [research/results](../../research/results/).

## See also

- [src/core](core/README.md) — Brain and engines used by these sims.
- [research/README.md](../../research/README.md) — How experiments and results are organized.
