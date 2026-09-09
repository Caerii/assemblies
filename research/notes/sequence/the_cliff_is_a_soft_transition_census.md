# The cliff is a census of soft transitions — and the horizon was never a time constant

**Task #87, S5 word-problem line. Studies: `seq_s5_word_problem.py` (aeb17ce,
6276685), `seq_s5_cliff_anatomy.py` (79399a3), `seq_s5_soft_census.py`
(this note). One retraction en route (0dd57f4).**

## The finding

A trained word-problem organ's single-step map has two populations:

* **~99.5% of transitions are EXACT**: cued with the precise stored block,
  they emit the precise successor block, all 70 neurons.
* **~0.25–0.6% are SOFT**: correct label, 69/70 neurons — exactly ONE
  intruder, every time, on every organ measured (min overlap 0.9857
  uniformly).

Sequence behaviour follows with nothing left over. **The first step at which
a trajectory leaves the lattice equals the first step its word's true path
visits a soft pair — 40/40 seeds, zero parameters, both directions** (seeds
whose paths avoid the soft set never deviate at all). After the visit, the
one-neuron deviation is amplified or corrected by the arc: 6/18 recover
fully, the rest wander sub-threshold (up to ~330 steps, readout still
correct) and then derail into the absorbing off-lattice regime.

So the exact-trajectory "decay" at L=500 was a first-hitting-time
distribution, and the apparent solvability gap was a PAIR-COUNT effect: S5
has twice the transitions of the order-60 groups at the same per-pair defect
rate (0.58% vs 0.25–0.58%), hence more soft pairs per organ, hence shorter
horizons. Non-solvability costs nothing measurable per transition.
[[SEQ-EXACT-RECOVERY]] is repaired, not overturned: exactness is real, and
its boundary is now a measured, static object.

## The route, kept honest

1. Per-step readout showed `step@500 == first_bad/500`: a cliff, not decay.
2. H-defect (wrong-LABEL transitions) was pre-registered at 65% and
   FALSIFIED: the label census found zero defects on 40 organs.
3. An "undeclared memory" claim was committed and RETRACTED within the hour:
   the census margins read 1.0000, constant, everywhere — the dead-probe
   signature this repo already documents. `probe()` restores winners on
   exit; the census had snapped after `fsm.run`'s internal probe returned,
   measuring the same residue 240 times. `diagnostics.verify_probe` exists
   for exactly this and was not used.
4. The live census (snapped INSIDE the probe) found the soft pairs, and the
   zero-parameter prediction closed 40/40.

## What is open, and registered as next

* **The intruder.** Always one neuron. Is it the same neuron across an
  organ's soft pairs (a state-area hub)? These organs run `norm_init=False`
  (parity substrate), which leaves hub formation unchecked — same mechanism
  family as the A3 toy's feed-forward-fiber collapse. Intervention
  prediction: `norm_init=True` (or per-fiber normalization) drives the soft
  rate to ~0 and unbounds the horizon. Needs its own prereg; it changes the
  substrate, so it is a re-measurement of the whole S5 table.
* **Recover vs derail.** What separates the 6 recoveries from the 12
  derailments — where the intruder sits relative to the next arc
  conjunction? The metastable wanderings (112, 264, 331 steps sub-threshold)
  say the basin structure is rich.
* **Solvability at power.** With soft pairs explained, the honest
  solvability comparison is per-pair defect rate and correction dynamics on
  a REPAIRED substrate, not exact@L.
