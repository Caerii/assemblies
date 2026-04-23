# Theoretical Basis

This question sits directly on the recurrent projection picture in
Papadimitriou et al., "Brain Computation by Assemblies of Neurons" (PNAS 2020).
The relevant ingredients are:

- sparse winner selection
- Hebbian strengthening on active pathways
- repeated re-projection into the same cortical area

In this repo, "stability" is operationalized as overlap between a trained
assembly and the winners produced by autonomous self-projection after the
stimulus has been removed.

Important qualification:

- The literature motivates why recurrent assembly dynamics can support stable
  representations, but the repo's current evidence is still implementation- and
  regime-specific.
- Stronger statements about global attractors, capacity, or universal
  convergence rates remain separate research questions.
