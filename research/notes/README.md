# Research notes

These are lab notebooks, not documentation. Every `PREREG_*` file writes its
pass conditions down before the data arrive and then reports against them,
so the files are long, dated, and wrong in the middle more often than not.
That is deliberate. If you want what we currently believe, read the register
in `neural_assemblies/theory.py` and cite its IDs. If you want to know why we
believe it, or why we stopped believing something, read the note.

Each long note opens with a quoted **Status** block. Read that, then the
amendments it names, and skip the rest unless you are checking our work.

## The four lines, and where I stand on each

**The refracted memory is the strongest result in this repo.** Refract a
recurrent k-WTA area at half beta, read it with the bias masked, and it
stores about 0.4 (n/k)² assemblies, twenty-five times what Hebbian
plasticity alone manages, with every item still distinct when the area is
full. It is a Willshaw store with a homeostat bolted on, and it does
continual learning without replay. I would build on it. Start with
[PREREG_refraction_memory.md](PREREG_refraction_memory.md); the earlier
[PREREG_refraction_capacity.md](PREREG_refraction_capacity.md) is where the
question came from, after a selector bug made its first answer wrong. The
unit is `AssemblyMemory` in `core/torch_engine/_memory.py`.

**The sequence organ is exact, and its failures are arithmetic.** The
refracted-arc machine runs thousands of steps without an error on the
explicit substrate. Its rare soft transitions are collisions between a
target block's weakest neuron and the area's best-connected outsider, a
Binomial tail that has nothing to do with the group's structure, and they
vanish when you stop training just below the weight clip. Refraction
strength is not a knob here: it is pinned at beta by two opposite
constraints, and sweeping it in either direction breaks the organ. Read
[DESIGN_sequence_port.md](DESIGN_sequence_port.md) for the port and its
gates, then [PREREG_s5_cliff_anatomy.md](PREREG_s5_cliff_anatomy.md) for the
anatomy. The numpy-era studies it corrects are
[PREREG_seq_a1_fsm_parity.md](PREREG_seq_a1_fsm_parity.md),
[PREREG_s5_word_problem.md](PREREG_s5_word_problem.md) and
[the_arc_is_a_conjunction_and_the_state_drifts.md](the_arc_is_a_conjunction_and_the_state_drifts.md).

**The transducer is an honest null.** Its induced state is distinct,
deterministic, and useless: it carries nothing the current word does not.
That is not a tuning problem. A state that is unique per prefix is a hash,
not a memory, and nothing in assembly calculus as built here merges
contexts by what they predict. On this corpus even a perfect state would
add 0.02 MRR over a bigram, so stop tuning the organ and change the
corpus. [PREREG_seq_a3_transducer.md](PREREG_seq_a3_transducer.md) has the
whole arc, including the two amendments that closed it.
[PREREG_state_refraction.md](PREREG_state_refraction.md) was built to
rescue a collapse that does not exist on the explicit substrate; its own
gate closed it.

**The aligner works, and its capacity is the lexicon's size.** Word
capacity scales with n and ignores n/k and the anchor. It is not a
recurrent memory, so the refraction result does not apply to it; I tried
to make that analogy and it does not survive reading the code.
[DESIGN_hashed_aligner.md](DESIGN_hashed_aligner.md),
[DESIGN_present_only.md](DESIGN_present_only.md), then
[PREREG_word_capacity.md](PREREG_word_capacity.md).

## The substrate

Two kernel layouts, one per density regime, and no third:
[DESIGN_dense_floor.md](DESIGN_dense_floor.md) for dense int16 counts above
about ten percent connectivity, [DESIGN_present_only.md](DESIGN_present_only.md)
for present-only lists below it. The regime conditions the theorems need
are measured in [PREREG_theorem_regime.md](PREREG_theorem_regime.md),
[PREREG_substrate_c_homeostasis.md](PREREG_substrate_c_homeostasis.md) and
[PREREG_crosstalk_mechanism.md](PREREG_crosstalk_mechanism.md).

## Rules I would not break again

- **Do not measure sequence dynamics on a sampled area.** The numpy engine's
  sampler gave us a false horizon, a five-fold inflated soft-transition
  rate, and every derailment we ever recorded. Materialize, or use the
  hashed substrate, which equals the materialized engine.
- **Do not gate a selector on replayed winners.** The k-WTA sign bug passed
  every drive-replay gate we had. Gate selection on drives that go negative.
- **Do not sweep refraction strength on a conjunction.** Below beta the arc
  collapses onto the state; above it the members relocate. The lever is the
  potentiated gain, kept under the clip.
- **Do not report a mean from fewer than three seeds.** The code refuses,
  and it is right to.
- **Keep the failed bars.** Four of six registered predictions failed this
  week, and each failure located a mechanism within the hour. A note that
  only records passes is a press release.
