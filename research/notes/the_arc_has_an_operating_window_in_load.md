# The arc has an operating point, and it is a WINDOW in load

A2 result. Pre-registered in `research/experiments/seq_a2_word_order_fsm.py`
(commit dae5f46), which was committed before any Brain-side numbers were read.
Follow-up: `seq_a2_refraction_load.py`.

## #92's task, re-run

| arm | correct | across-STATE | across-MOOD |
|---|---|---|---|
| multi-mood, refraction on | **10/10** | **0.001** | **0.014** |
| single-mood (the R3 floor) | 2/10 | 0.040 | -- |
| null, refraction off | 0/10 | 0.948 | 0.964 |

P-ORDER, P-CONJ2 and P-NULL pass; **P-FLOOR fails**.

Two things to read here. First, **the null arm reproduces #92**: without
refraction the arc reads 0.948 across states and 0.964 across moods, against
their measured 0.90-0.99 and 0.48-0.97, and scores 0/10. We have reproduced
their failure exactly and then removed it. Second, the multi-mood arc holds
BOTH separations at 0.001 and 0.014, so
`arc_conjunction_has_no_operating_point.md` is overturned in its own task, not
merely on mod-3.

## The floor failed, and the reason is not what #92 thought

Ruled out, in order:

* **refraction itself** -- ablating it makes the arm WORSE (0/10, arc collapsed
  to 0.988). Refraction is not the problem.
* **conjunct exposure** -- lowering the over-exposed mood fiber's beta, the
  exact lever #92 swept, rescues nothing: 0/10, 2/10, 0/10, 1/10, 0/10 at beta
  0.1, 0.05, 0.02, 0.01, 0.0. Their "no interior optimum" reproduces even with
  refraction present.
* **the state area's regime** -- `regime_audit` reports both areas above floor
  throughout.

What was left is CONVERGENCE, and it is measurable directly. The arc assembly
for one (state, symbol) pair never settles in the single-mood arm:

    presentations   5 vs 10   10 vs 15   5 vs 15
    single-mood      0.514      0.600     0.286
    multi-mood       0.629      0.957     0.600

A teacher-forced write smeared across assemblies that keep moving cannot
dominate at test. That is why recovery is poor from STEP ONE (0.44-0.67) rather
than degrading along the sequence.

## The mechanism, and the window

Refraction separates by pushing winners off neurons that have already fired.
That needs something to push AGAINST. With 3 conjunctions in a 5000-neuron area
there is always fresh unused space, so the assembly wanders into it instead of
tiling; with enough conjunctions competing, the assemblies fill the space and
lock.

Sweeping arc SIZE at fixed content, so load `M*k/n` is the only thing moving:

    load     single-mood (3 conj)    multi-mood (9 conj)
    0.04           1/10                    --
    0.10           7/10                    --
    0.13            --                    9/10
    0.21          10/10                    --
    0.32            --                   10/10
    0.42          10/10                    --
    0.60          10/10                    --
    0.63            --                   10/10
    1.26            --                   10/10
    1.80            --                    1/10

**Both bounds are real.** Below about 0.2 the assemblies never converge. Above
about 1.15 they cannot fit: that is `alpha* ~ 1.15`, the critical load measured
independently in this repo, and multi-mood collapses at 1.80 while still
working at 1.26.

So the arc DOES have an operating point. It is a window in load, bounded below
by refraction convergence and above by capacity, and #92 was sitting below it
-- their arc ran at n=1000 with few conjunctions, which is exactly the
under-loaded corner.

P-FLOOR passes at any arc sized to its content: 10/10 from load 0.21 upward.
The bar is recorded as failed at the registered parameters, because it was, and
the parameters were the error.

## What this changes

An arc area must be **sized to the number of conjunctions it will hold**, and
"bigger is safer" is wrong in both directions. This is a design rule with a
number attached, and it is the second time a sequence result has turned on an
operating window rather than a monotone knob -- the first being beta.

It also means an under-loaded area is a NEW silent-failure mode to add to the
list: every local diagnostic looks healthy (the conjunction is clean at 0.021,
every area is in-regime, refraction is active and charging) while the organ
fails, because the thing that is wrong is that the assemblies never stopped
moving. `regime_audit` cannot see it; assembly stability across training can.
