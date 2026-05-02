# Simulation

`neural_assemblies.simulation` contains runnable helpers for common assembly
experiments and plots.

## Exports

| Export | Use it for |
|--------|------------|
| `project_sim`, `project_beta_sim`, `assembly_only_sim` | Projection and recurrence studies. |
| `association_sim`, `association_grand_sim` | Association experiments. |
| `merge_sim`, `merge_beta_sim` | Merge experiments. |
| `pattern_com`, `pattern_com_repeated` | Pattern completion experiments. |
| `density`, `density_sim` | Density and overlap experiments. |
| `fixed_assembly_recip_proj`, `fixed_assembly_merge`, `separate` | Structured assembly experiments. |
| `larger_k`, `turing_erase` | Exploratory Turing-style simulations. |
| plotting helpers | Figures for simulation runs. |

## Example

```python
from neural_assemblies.simulation import merge_sim, project_sim

weights = project_sim(n=100_000, k=317, p=0.01, beta=0.05, t=50)
a_w, b_w, c_w = merge_sim(n=100_000, k=317, p=0.01, beta=0.05)
```

The Turing-style helpers are simulations. Cite the sequence-computation papers
for theoretical claims.

## See Also

- [Research guide](../../research/README.md)
- [Scientific status](../../docs/scientific_status.md)
