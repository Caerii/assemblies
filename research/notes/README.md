# Research notes: a reading map

This directory holds registrations (`PREREG_*`), design notes (`DESIGN_*`),
audits and mechanism write-ups. A registration states its bars before the
data and then records the result under them, so most files read as a log.
This map says where each active line stands and which file to open first.

The register of adopted results is `neural_assemblies/theory.py`; cite its
IDs (for example `[[REFRACTION-ANTI-MERGING]]`) rather than a note.

## Conventions

- **Bars before data.** Every `PREREG_*` file lists its hypotheses and pass
  conditions before the run. Results are appended under the same labels,
  with PASS or FAIL and the numbers.
- **Amendments are labelled.** A change registered before its run is an
  *Amendment* or *Addendum*; anything decided after seeing data is marked
  *post hoc* and never adopted on its own.
- **Distributions, never bare means.** Twenty or more brains per cell on the
  hashed substrate; means carry a 95% interval; ceilings are read from a
  curve with a bracket, not from one grid point.
- **Failed bars are kept.** A failed prediction is committed with its
  numbers and, where possible, the mechanism the failure located.

## The refracted memory (adopted)

A recurrent k-WTA area refracted at half beta, read with its bias masked,
is an associative memory with about 25 times the Hebbian capacity.

1. [PREREG_refraction_memory.md](PREREG_refraction_memory.md): the
   registration, six amendments, and the adoption. Read the *Status*
   block at the top first.
2. [PREREG_refraction_capacity.md](PREREG_refraction_capacity.md): the
   earlier study whose re-measurement, after a selector defect was fixed,
   produced the question.
3. [PREREG_capacity_scaling.md](PREREG_capacity_scaling.md) and
   [PREREG_substrate_ceiling.md](PREREG_substrate_ceiling.md): the Hebbian
   capacity harness the memory study runs on.

The unit is `neural_assemblies/core/torch_engine/_memory.py`
(`AssemblyMemory`).

## The sequence organ (ported, exact)

The refracted-arc transition machine and the induced-state transducer, on
the hashed substrate at width.

1. [DESIGN_sequence_port.md](DESIGN_sequence_port.md): what was ported,
   the four gates, and their results, including the finding that the
   numpy engine's sampler produced the earlier short horizons.
2. [PREREG_s5_cliff_anatomy.md](PREREG_s5_cliff_anatomy.md): the soft
   transition census, eight addenda, and the result that the organ is
   exact when trained just below the weight clip. Read its *Status* block
   first.
3. [PREREG_seq_a3_transducer.md](PREREG_seq_a3_transducer.md): the
   transducer at width, and why its induced state carries no information
   on this corpus (Amendments 2 and 3).
4. [PREREG_seq_a1_fsm_parity.md](PREREG_seq_a1_fsm_parity.md),
   [PREREG_s5_word_problem.md](PREREG_s5_word_problem.md),
   [the_arc_is_a_conjunction_and_the_state_drifts.md](the_arc_is_a_conjunction_and_the_state_drifts.md):
   the numpy-era studies the port reproduces or corrects.
5. [PREREG_state_refraction.md](PREREG_state_refraction.md): closed at
   width by its own gate.

The units are `_arc_core.py` (`HashedArcCore`), `_hashed_fsm.py`
(`HashedArcFSM`) and `_hashed_transducer.py` (`HashedTransducer`) under
`neural_assemblies/core/torch_engine/`.

## The aligner and word capacity

Cross-situational word learning on the hashed substrate.

1. [DESIGN_hashed_aligner.md](DESIGN_hashed_aligner.md) then
   [DESIGN_present_only.md](DESIGN_present_only.md): the port and the
   present-only kernel that carries it.
2. [PREREG_word_capacity.md](PREREG_word_capacity.md): capacity scales
   with the lexicon's size, not with n/k.

## The substrate itself

- [DESIGN_dense_floor.md](DESIGN_dense_floor.md),
  [DESIGN_present_only.md](DESIGN_present_only.md): the two kernel layouts,
  one per density regime, and their measured floors.
- [PREREG_substrate_c_homeostasis.md](PREREG_substrate_c_homeostasis.md),
  [PREREG_theorem_regime.md](PREREG_theorem_regime.md),
  [PREREG_crosstalk_mechanism.md](PREREG_crosstalk_mechanism.md): the
  substrate's regime conditions.

## Three lessons that recur

- **Materialize before measuring.** The numpy engine's sampled areas
  produced a false horizon, a five-fold inflated soft-transition rate, and
  every derailment in the sequence studies. The hashed substrate equals
  the materialized engine and is the reference for sequence work.
- **A drive-replay gate cannot see a selection defect.** The selector's
  sign bug passed every replay gate; add a selection gate on drives that
  go negative.
- **Refraction is two tools.** In a store with disjoint items its strength
  is a switch across a wide plateau. In a conjunction whose neurons are
  shared across contexts its strength is pinned at beta by two opposite
  constraints, and the free lever is the potentiated gain, kept below the
  weight clip.
