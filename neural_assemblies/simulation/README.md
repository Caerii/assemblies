# Simulation

The `neural_assemblies.simulation` package contains runnable experiment helpers
and plotting utilities built on the package runtime.

## Main Exports

| Export | Role |
|--------|------|
| `project_sim`, `project_beta_sim`, `assembly_only_sim` | Projection and recurrence studies. |
| `association_sim`, `association_grand_sim` | Association experiments. |
| `merge_sim`, `merge_beta_sim` | Merge experiments. |
| `pattern_com`, `pattern_com_repeated` | Pattern completion experiments. |
| `density`, `density_sim` | Density and overlap experiments. |
| `fixed_assembly_recip_proj`, `fixed_assembly_merge`, `separate` | Additional structured experiments. |
| `larger_k`, `turing_erase` | Exploratory Turing-style simulation helpers. |
| plotting helpers | Plotting utilities for the above runs. |

## Example

```python
from neural_assemblies.simulation import merge_sim, project_sim

weights = project_sim(n=100_000, k=317, p=0.01, beta=0.05, t=50)
a_w, b_w, c_w = merge_sim(n=100_000, k=317, p=0.01, beta=0.05)
```

The Turing-style helpers are exploratory simulations. They should not be read
as a package-level proof artifact.

## See Also

- [../../research/README.md](../../research/README.md)
- [../../docs/scientific_status.md](../../docs/scientific_status.md)
