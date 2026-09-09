# The dead fiber was a growth ratchet — expansion was recruitment-gated, and one quiet episode after source growth killed a fiber forever

**Task #151, defect-hunt unit (registered `4ce17d6`, D0–D4 signatures
pre-stated). Fix at `53e9808`. Seeds 45 (defect) + 42 (control), n=10⁵,
L0G4, Brown slice — byte-identical construction to the adoption gate.**

## Verdicts

- **D0 REPRODUCTION — exact.** Seed 45: own-drive 0.0 on 45/46 PL
  words, the one served word at 0.678. Deterministic since the
  one-seeding-path fix.
- **D1 CENSUS — door (d), the index-space/growth door.** Seed 45's
  NOUN_CORE→NUMBER_PL froze at **180 rows against a source of 18,641
  materialized neurons** (extent 49, cols-with-mass 49, top-30 column
  mass share 0.937). Seed 42's control: 24,176 rows, extent 1,600,
  top-30 share 0.022. Every zero word had **all 30 core rows out of the
  fiber's row space**; extent_desync read 0 in both seeds (the COLUMN
  instrument cannot see a ROW freeze — the desync family's blind side).
- **D2 THE SERVED WORD — prediction CONFIRMED.** `checkers`, episode 3,
  IS the first-trained PL word: its columns and rows were materialized
  through the fiber before the freeze.
- **D3 WRITER/READER — clean.** The PL label image lives entirely on
  the 49 frozen columns (0 out of range, mass 2,359): reader and
  writer share the lookup; the zero is in the rows.
- **D4 CONTROL — instruments read healthy on seed 42.**

## The mechanism (a ratchet, all three teeth confirmed in code)

1. `_expand_connectomes` returns early when the target recruits no
   first-time winner — row/col growth lived ONLY there.
2. The drive slice drops out-of-range rows silently
   (`src_w[src_w < conn.weights.shape[0]]`), and the plasticity write
   drops them the same way.
3. Zero drive recruits nobody → no recruitment, no growth → zero
   drive, forever. k-WTA keeps returning k winners throughout
   ([[silent-no-op-dead-fibers]]).

Seed-dependence is which side of the coin the SECOND PL episode lands
on: NOUN_CORE has grown past the fiber's rows by then in EVERY seed
(episode order is corpus-fixed); a seed whose episode-21 word shares at
least one in-range core row gets nonzero baseline drive, recruits, and
heals; seed 45's drew none. Every seed passes through the danger
window — one in twelve died in it.

## The fix (`53e9808`)

Stale coverage rides the EXISTING deferred-init repair — the
empty-block path's next-round semantics, extended: after any projection
that names a fiber, its initialised region covers `(src.w, tgt.w)`
(`_ensure_area_block_coverage`, mirroring `_expand_connectomes`'
mechanics; content-addressed init makes the late fill value-identical
to what the recruit path would have written). Gated off under
`read_only()` — growth is the channel that contract closes. Regression
tests pin the frozen state directly and FAIL on the pre-fix engine
(power verified); healthy-fiber no-op pinned.

## Healing (2 seeds; predictions pre-stated)

- **P-heal-1 CONFIRMED**: seed 45 reads 0/46 exact zeros; fiber
  [20,952 × 1,920], extent 1,905, top-30 share 0.018 — healthy
  geometry.
- **P-heal-3 CONFIRMED**: seed 42 shifts (extent 1,600 → 1,932, served
  word changes) and stays healthy — the danger-window drive is now
  real for everyone, so numbers move.
- **P-heal-2 EXCEEDED, flagged**: both seeds read PL E≥2 = **1.000**
  and PL E=1 = **0.93 / 0.89** — the E=1 stratum was 0.500 ± 0.223
  pre-fix. If this holds at 10 seeds, the "E=1 words sit at COLT22's
  margin" account was partly THIS defect: a 1-exposure word whose
  single episode fell in a lag window had its entire Hebbian write
  silently dropped. The margin arithmetic predicted ~0.55 and matched —
  a correct-looking number produced by a mechanism the theory did not
  contain. Two seeds decide nothing ([[ensemble-not-realization]]);
  the re-registered adoption gate decides.

## What is re-opened

The pre-fix scale-cell numbers (PL E≥2 0.779, PL E=1 0.500, the E=2
step function) were measured on an engine that dropped an unknown
fraction of rare-class writes. The exposure LAW (E is the currency)
stands on the synthetic side and in theory; its measured EXCHANGE RATE
on Brown is stale. The adoption gate re-run (in flight) re-prices it.
