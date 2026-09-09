# Research notes

The `PREREG_*` files are registrations. Each one lists its hypotheses and
pass conditions, then appends the results under the same labels. Later
changes are marked as amendments when they were registered before a run and
as post hoc when they were decided after seeing data. The `DESIGN_*` files
describe substrate work and the gates it passed. Adopted results are
recorded in `neural_assemblies/theory.py`, and other documents cite those
IDs.

The long notes begin with a quoted Status paragraph that summarises the
outcome and names the sections to read.

## Refracted memory

A recurrent k-WTA area refracted at half beta, with its bias masked at
readout, stores about 0.40 (n/k)² assemblies in regime. The Hebbian control
stores about one twenty-fifth of that. Items remain distinct after the area
fills, and they overlap at chance, so refraction prevents merging during
writing and does not orthogonalize the stored set. Strength between 0.3 and
0.6 beta gives the same ceiling. Ending each item's write when its winner
set repeats raises the ceiling by 24 to 34 percent.

Registration: [PREREG_refraction_memory.md](PREREG_refraction_memory.md).
The question came from [PREREG_refraction_capacity.md](PREREG_refraction_capacity.md),
whose first answer was wrong because of a selector bug. The class is
`AssemblyMemory` in `core/torch_engine/_memory.py`.

## Sequence organ

The refracted-arc transition machine, ported to the hashed substrate, runs
2000 random digits without an error on 40 of 40 brains at p = 0.3 and
p = 0.4. The earlier five-seed numpy result had one short horizon at
p = 0.3. That seed runs the full 2000 steps once its arc area is
materialized, so the short horizon came from the sampler.

The soft transitions measured in the S5 census are ties between the target
block's least-connected neuron and the most-connected neuron outside the
block. Their rate is the same across three groups of order 60, rises with
the state area's size, and does not depend on the group's Cayley graph.
Training for 20 to 24 presentations instead of 15 removes them: 0 soft
pairs in 84,000 across 500 organs. The arc's refraction strength has to stay
at beta. Lower values collapse the arc onto the state conjunct, and higher
values relocate its members before training finishes.

Read [DESIGN_sequence_port.md](DESIGN_sequence_port.md), then
[PREREG_s5_cliff_anatomy.md](PREREG_s5_cliff_anatomy.md). The numpy studies
these correct are [PREREG_seq_a1_fsm_parity.md](PREREG_seq_a1_fsm_parity.md),
[PREREG_s5_word_problem.md](PREREG_s5_word_problem.md) and
[the_arc_is_a_conjunction_and_the_state_drifts.md](the_arc_is_a_conjunction_and_the_state_drifts.md).

## Transducer

The induced-state transducer
([PREREG_seq_a3_transducer.md](PREREG_seq_a3_transducer.md)) beats the
recurrent accumulator of #14 by 0.10 MRR, and its state does not collapse.
Scoring with the state area empty gives the same MRR, so the state
contributes no information beyond the current word. Amendment 2 shows that
lowering the arc's strength makes the organ worse. Amendment 3 computes the
corpus's oracle ceiling: a state that knew the generating grammar's phase
would add 0.019 MRR over a bigram. The corpus cannot show a state effect
larger than that, and further work on this organ needs a corpus with a
larger oracle gap.
[PREREG_state_refraction.md](PREREG_state_refraction.md) addressed a state
collapse that occurs only on the sampled numpy engine, and its gate closed
it.

## Aligner

Word capacity in the cross-situational learner scales with the lexicon's n
and does not change with k or with the anchor
([PREREG_word_capacity.md](PREREG_word_capacity.md)). The lexicon has no
recurrent fiber, so the refracted-memory result does not apply to it. The
port is described in [DESIGN_hashed_aligner.md](DESIGN_hashed_aligner.md)
and [DESIGN_present_only.md](DESIGN_present_only.md).

## Substrate

There are two kernel layouts: dense int16 counts for connectivity above
about ten percent ([DESIGN_dense_floor.md](DESIGN_dense_floor.md)) and
present-only lists below it
([DESIGN_present_only.md](DESIGN_present_only.md)). The regime conditions
the theorems require are measured in
[PREREG_theorem_regime.md](PREREG_theorem_regime.md),
[PREREG_substrate_c_homeostasis.md](PREREG_substrate_c_homeostasis.md) and
[PREREG_crosstalk_mechanism.md](PREREG_crosstalk_mechanism.md).

## Practices these notes established

- Materialize an area, or use the hashed substrate, before measuring
  sequence dynamics. The sampled numpy engine produced a false horizon, a
  soft-transition rate five times too high, and every derailment in the
  earlier sequence results.
- Gate a selector on drives that go negative. The k-WTA sign bug passed the
  drive-replay gates because those gates replay recorded winners.
- Leave refraction strength on a conjunction at beta
  ([PREREG_s5_cliff_anatomy.md](PREREG_s5_cliff_anatomy.md), Addendum 6).
- Report at least three seeds. `ensemble_from_values` refuses fewer.
- Record failed bars with their numbers. Four of the six predictions
  registered in the week of 2026-09-05 failed, and each failure identified
  the mechanism that replaced it.
