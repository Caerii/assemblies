# DESIGN: the masked readout as a mode

Built 2026-09-09.

A refracted area can be read in two ways. With the bias subtracted, as
the k-WTA ranks it during writing, a half-cue recall of a stored assembly
reads chance. With the bias masked, the stored assembly returns; the
refracted memory holds about 25 times the Hebbian ceiling only when read
that way (`[[REFRACTION-ANTI-MERGING]]`). Until now the masked read
existed as a per-call option on the hashed area and as a save-zero-restore
of the bias in the numpy mirror script. Any organ that both writes and
reads a refracted store needs the switch as a mode, and nothing in the
substrate flips it on its own.

## What was built

- The numpy area state carries `masked_readout`. `project_into` skips the
  bias on a read (plasticity off) when the flag is set; a write always
  sees and charges the bias, because the bias is what keeps items apart
  while they are written. `Brain.set_masked_readout(area, on)` sets it.
- `HashedArea.masked_readout` is the same flag; a frozen projection reads
  it as the default for `mask_bias`, and a per-call `mask_bias` still
  overrides.
- `AssemblyMemory` sets the flag on its area when refracted and reads
  through it. The numpy mirror (`refraction_memory_numpy.py`) uses the
  primitive instead of editing the bias array.

## Gate

The refracted parity test (`test_hashed_substrate_parity.py`) gains a
masked case: after a refracted training trace, a frozen half-cue
projection with the flag on gives the engine a snapshot equal to the
hashed raw drive to a relative 5e-6; with the flag off it equals the net
drive; and a plastic projection with the flag on still ranks the net
drive. Passes. The memory unit's three parity tests pass unchanged.

## What this is not

It is not a decision about when to switch. A working organ that stores
and recalls will need something to set the mode at the right moments,
and that remains an open design question; this note makes the mode
exist so the question can be asked in code.
