"""
Minimal example: Brain + projection + assembly calculus (merge).

Run from repo root: uv run python examples/01_basic_assembly_calculus.py
Or after pip install neural-assemblies: python examples/01_basic_assembly_calculus.py
"""
from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus import project, merge

def main():
    b = Brain(p=0.05, save_winners=True, seed=42, engine="numpy_sparse")
    b.add_stimulus("s1", 80)
    b.add_stimulus("s2", 80)
    b.add_area("A1", n=5000, k=80, beta=0.08)
    b.add_area("A2", n=5000, k=80, beta=0.08)
    b.add_area("B", n=5000, k=80, beta=0.08)

    # Project each stimulus into its own area
    a1 = project(b, "s1", "A1", rounds=8)
    a2 = project(b, "s2", "A2", rounds=8)
    print(f"Source assembly sizes: {len(a1)}, {len(a2)}")

    # Merge A1 and A2 into B (conjunctive assembly)
    a_merged = merge(b, "A1", "A2", "B", rounds=5)
    print(f"Merged assembly size: {len(a_merged)}")
    print(f"Merged assembly area: {a_merged.area}")
    print("Done.")

if __name__ == "__main__":
    main()
