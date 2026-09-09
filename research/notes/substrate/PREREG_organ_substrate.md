# PREREG: is the word-problem arc's assembly DECAY the substrate-C merger?

Registered before data. Experiment: `research/experiments/seq_organ_substrate.py`.

## The prediction, and why it is sharp

[[arc-training-is-not-batchable]] recorded a measured and unexplained shape:
the Z60 arc's identical-assembly fraction across presentations climbs to a peak
around presentation 5-8 and then **degrades**.

    pres  2:  11/120 identical    pres  8:  68/120 (0.567)
    pres  3:   0/120              pres 10:  39/120 (0.325)
    pres  5:  69/120 (0.575)      pres 12:  14/120 (0.117)   <- degrading

Three properties of that measurement are exactly the conditions
`PREREG_substrate_ceiling.md` has now characterized:

1. it was taken with **`synaptic_scaling=True` and `norm_init` OFF** -- substrate
   C alone, whose assemblies merge at 6-16x the chance floor;
2. the arc was **SATURATED** (w = 19999/20000) throughout -- the same rows/n ~ 1
   regime where `norm_init` alone loses a quarter of its assemblies to exact
   duplicates and substrate C loses far more;
3. it was **one seed**.

And substrate C is *worse* the sparser the fiber (6.0x chance at p=0.5 against
15.9x at p=0.124), while this organ runs a p=0.05 brain.

**So the hypothesis is that the decay is not a fact about deep training at all.
It is the merger, on a real organ.** If so, `norm_init` alongside
`synaptic_scaling` should remove it -- the same composition that took pairwise
overlap from 7.5x chance to the floor and roughly doubled the M-ceiling.

This is a transfer test. A substrate result that does not move a real organ is
a fact about toys.

## Design

Z60 word problem, `seq_s5_word_problem.build`, arm `trained`.

**`organ_p=0.5`, not the default 0.40.** k=70 gives kp = 35.0 against the
floor 3 ln(20000) = 29.7. The default 0.40 gives kp = 28.0, which is BELOW
floor -- the violation `regime_audit` printed on every run of this organ for
weeks. A null measured there is not evidence about the mechanism.

`w_max=None` (unclamped), `presentations=15` (the registered protocol),
refraction at the organ default, seeds 42-45.

Four substrate arms, the same four the ceiling study used:

    NONE  norm_init=False  synaptic_scaling=False   <- the organ's CURRENT default
    B     norm_init=True   synaptic_scaling=False
    C     norm_init=False  synaptic_scaling=True    <- what the decay was measured on
    G     norm_init=True   synaptic_scaling=True

## Readouts

Stability comes from `saved_winners` via `_stability_from_saved`, imported
from `seq_refraction_stability` rather than re-implemented.

* **identical fraction** per presentation; its PEAK, its TERMINAL value, and
  `decay = terminal / peak` -- the shape is the phenomenon, so a single
  terminal number would not reproduce it.
* **distinct fraction of the 120 arc assemblies** at the final presentation.
  This is the merger readout, and it is the one the original measurement did
  not have. Mean overlap is nearly blind to partial collapse: 256 items on ~58
  distinct assemblies still reads at the chance floor.
* **mean pairwise overlap across transitions**, in units of chance (k/n_arc =
  70/20000 = 0.0035).
* **trajectory accuracy** at L=100, because a substrate that stabilizes
  assemblies and does not help the task has not helped.

## Bars

**O1 (regime).** Every cell has kp >= 3 ln n_arc, asserted in code.

**O2 (the defect reproduces).** Substrate C's `decay` is below 0.80 -- the
arc's stability really does fall away from its own peak. If O2 FAILS the
phenomenon is not present at these settings and O3-O5 say nothing; report that
and stop.

**O3 (THE CLAIM).** `paired_delta(G terminal, C terminal)` beats 0.

**O4 (merger on the organ).** G's arc distinct-fraction has CI-low >= 0.90
while C's has CI-high < 0.90.

**O5 (does it matter).** `paired_delta(G accuracy, C accuracy)` beats 0.

**O6 (against the organ's CURRENT default).** `paired_delta(G terminal, NONE
terminal)` beats 0. O3 could pass while G is merely undoing damage C caused;
O6 asks whether the composition beats what the organ ships with today.

All bars judged on CONFIDENCE BOUNDS, orderings as PAIRED per-seed deltas.
A delta straddling zero is INCONCLUSIVE, reported as such.

## Disclosure: overlap with PREREG_refraction_stability R5

That note's R5 compares `synaptic_scaling` ON against OFF at equal refraction
with `norm_init=False` -- which is exactly this study's NONE-vs-C contrast, at
the organ default refraction. **This study will therefore answer R5 as a side
effect, before the refraction study runs.** Recording that in advance so the
result cannot later be presented as an independent confirmation. Whatever the
NONE-vs-C contrast shows here will be written into the refraction note, and R5
must not be re-scored as new evidence when that study runs.

The two studies remain distinct questions: refraction varies
`refracted_strength` and holds the substrate fixed; this varies the substrate
and holds refraction at the organ default.

## What each outcome means

* O2 passes, O3+O4 pass -> the arc's decay IS the merger, and it is fixable.
  The batchability precondition may then be reachable, which would reopen
  [[arc-training-is-not-batchable]] -- but batching would still need its own
  registration and must not be smuggled in as a perf change.
* O2 passes, O3 fails -> the decay is real and NOT the merger. The substrate
  account does not transfer to this organ; say so plainly and leave
  [[arc-training-is-not-batchable]] standing.
* O3 passes, O5 fails -> stability without capability. Report as stability,
  never as a task result.
* O6 fails -> the composition is not better than what the organ already does;
  no default change should be proposed on this evidence.

---

## Amendment 1 (post-O2, pre-follow-up): the precondition FAILED, and one knob is uncontrolled

**O2 failed. The study cannot answer its stated question.** Recorded before the
follow-up runs.

    identical-assembly fraction per presentation, mean of seeds 42-45
      arm   p02  p04  p06  p08  p10  p12  p14  p15
      NONE  0.00 0.00 0.34 0.56 0.75 0.73 0.73 0.84
      B     0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02
      C     0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
      G     0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00

      arm         peak     terminal          decay   distinct   acc          fill
      NONE  0.852+-0.041 0.840+-0.059  0.985+-0.047  1.000  1.000+-0.000  1.000
      B     0.017+-0.011 0.017+-0.011  1.000+-0.000  1.000  0.897+-0.326  0.437
      C     0.010+-0.017 0.000+-0.000  undefined 1/4 1.000  0.035+-0.009  1.000
      G     0.000+-0.000 0.000+-0.000  undefined 4/4 1.000  0.028+-0.008  0.690

**There is no climb-then-degrade shape here to explain.** Substrate C never
reaches a peak at all (0.010 +/- 0.017), and the arm that DOES show a shape is
NONE, which climbs to 0.85 and does not decay (0.985). The phenomenon this
study was built to explain is absent at these settings, so O3-O6 say nothing
about it and are not reported as if they did.

**`decay` is UNDEFINED when the arm never produced a single identical assembly
(peak = 0)** -- 1/4 seeds for C, 4/4 for G. That is a more total failure than
any decay, not "no decay", so it is neither mapped to 1.0 nor filtered out; O2
was judged on `peak - terminal` instead, which is always defined. Same claim,
same direction, defined everywhere. This is an aggregation change forced by an
undefined statistic, made explicit because undefinedness here correlates with
the arm being worst -- exactly the direction that would bias the result.

### The collateral result, which is larger than the registered one

**Every normalization arm destroys this organ.** Trajectory accuracy at L=100:
NONE 1.000, B 0.897 +/- 0.326, C 0.035, G 0.028. Z60 has 60 states, so chance
is 0.017: **C and G are at chance.** The organ's shipped default is the only
arm that works.

This directly contradicts the direction I expected, and it is the answer to the
question that motivated the study -- just not the one I registered.

### Reading NONE's 0.84 with the right instrument

NONE's high identical-assembly fraction is very likely NOT convergence.
[[stability-is-arithmetic-degeneracy]] measured that the un-normalized arm's
drive landscape is arithmetically degenerate (737 distinct drives over 19,999
columns, 77.9% integer at every depth) and that normalization's float
renormaliser shatters it into ~18k distinct values -- and that the shattering IS
the drift. NONE is also at fill = 1.000, fully materialized, where that
degeneracy is maximal.

So "identical fraction" is measuring tie-degeneracy on the NONE arm and
convergence on none of them. **Accuracy is the readout that survives this
objection**, and accuracy says the same thing more plainly.

### The uncontrolled knob, and the follow-up

`w_max=None` (unclamped weights) was carried over from
`seq_refraction_stability` and is NOT the organ's registered default. It is the
one setting this study changed relative to the protocol under which the organ
normally runs, and unclamped weights interact with exactly the thing under test
-- a normalizer that divides by accumulated mass.

**Registered now, before running:** the same four arms at `organ_p=0.5` (still
IN REGIME, kp=35.0 vs floor 29.7) crossed with `w_max` in {None, the organ
default}. Nothing else changes.

* **O7.** At the organ's default `w_max`, NONE's accuracy stays above 0.90 --
  i.e. the baseline is not a `w_max=None` artifact.
* **O8.** At the organ's default `w_max`, C's and G's accuracy remain below
  0.50. If they RECOVER, then this amendment's collateral result is a fact
  about unclamped weights and not about normalization, and must be reported
  that way.
* **O9.** At either `w_max`, C shows the climb-then-degrade shape
  (`peak - terminal` > 0 on the confidence bound). This is O2 retried at the
  only setting this study is known to have changed; if it fails here too, the
  decay recorded in [[arc-training-is-not-batchable]] does not reproduce on
  this engine at all, and that memory needs narrowing to the configuration it
  was measured in (torch_sparse, one seed).

### Disclosed result: PREREG_refraction_stability R5

As disclosed in advance, NONE-vs-C at the organ default refraction IS that
note's R5. **R5 PASSES decisively**: scaling OFF is more stable than ON, 0.840
vs 0.000, paired delta +0.840 +/- 0.059. This must not be re-scored as new
evidence when the refraction study runs.

---

## Amendment 2: results. The registered question is unanswerable here, and the collateral result needs a fairness caveat

### O7/O8/O9, and why the w_max control came back byte-identical

    arm      wmax        peak      terminal    distinct        acc     fill
    NONE     none   0.852+-0.041 0.840+-0.059  1.000    1.000+-0.000  1.000
    NONE  default   0.852+-0.041 0.840+-0.059  1.000    1.000+-0.000  1.000
    B        none   0.017+-0.011 0.017+-0.011  1.000    0.897+-0.326  0.437
    B     default   0.017+-0.011 0.017+-0.011  1.000    0.897+-0.326  0.437
    C        none   0.010+-0.017 0.000+-0.000  1.000    0.035+-0.009  1.000
    C     default   0.010+-0.017 0.000+-0.000  1.000    0.035+-0.009  1.000
    G        none   0.000+-0.000 0.000+-0.000  1.000    0.028+-0.008  0.690
    G     default   0.000+-0.000 0.000+-0.000  1.000    0.028+-0.008  0.690

Every pair is **byte-identical**. That is not a dormant knob: `DEFAULT_W_MAX`
is 20.0 and the knob really is passed through. The clamp simply never binds --
plasticity here is multiplicative, so the largest weight after T presentations
is `(1+beta)^T = 1.1^15 = 4.18`, comfortably under 20.

So the control is **stronger** than a difference would have been: `w_max` is
excluded as an explanation by construction, not by a null.

    PASS  O7  NONE accuracy at default w_max 1.000+-0.000, CI-low 1.000 > 0.90
    PASS  O8  C 0.035+-0.009 and G 0.028+-0.008, both CI-high < 0.50
    FAIL  O9  C peak 0.010+-0.017; no climb-then-degrade at either w_max

### O9 FAILS: the recorded arc decay does not reproduce

[[arc-training-is-not-batchable]] recorded substrate C climbing to 0.575 by
presentation 5 and degrading to 0.117 by 12. Here substrate C **never reaches a
peak at all** (0.010 +/- 0.017) at either `w_max`, over four seeds.

That measurement was taken on **torch_sparse with one seed**; this is
numpy_sparse with four. The engines are ensemble-equivalent, not bit-identical,
but a shape that goes from 0.575 to 0.010 is not an ensemble difference. **That
memory must be narrowed to the configuration it was measured in**, and the
"deep training destabilises conjunctions under homeostasis" reading it carried
is not supported here.

The batchability conclusion it drew is UNAFFECTED and still holds: the
identical-assembly fraction never reaches 1.0 on any arm, so no suffix of
training is exactly batchable. If anything this strengthens it -- three of four
arms sit at 0.00.

### The collateral result, and the fairness caveat it needs

Trajectory accuracy at L=100, chance = 1/60 = 0.017:

    NONE  1.000+-0.000     B  0.897+-0.326     C  0.035+-0.009     G  0.028+-0.008

**C and G are at chance.** The organ's shipped default is the only arm that
works, and B is one bad seed away from unreliable (+/-0.326 over four seeds).

**This is NOT a fair test of normalization, and must not be reported as one.**
`build()` passes `norm_init=False` explicitly, against the `Brain` default of
True, and every one of this organ's parameters -- k=70, `organ_p`,
`TARGET_LOAD=0.42`, the arc sizing -- was chosen under that setting. Turning a
normalizer on changes the drive scale every one of those was tuned against, so
the organ is simply outside its tuned regime. The fill numbers say so directly:
NONE materializes the whole arc (1.000) while B materializes 0.437 of it. That
is a different recruitment dynamics, not a worse substrate.

What this DOES establish: **no default change is warranted on the ceiling
study's evidence.** A substrate that doubles the M-ceiling of a bare recurrent
area cannot be dropped into a tuned feedforward-conjunction organ and expected
to work. A fair test would re-tune k, `organ_p` and `TARGET_LOAD` under
normalization, which is a different study.

### Why the toy result should not have been expected to transfer

The ceiling study measured ONE area with SELF-RECURRENCE -- the definition of
an assembly, and the setting where a merger is the failure mode. The arc is a
feedforward CONJUNCTION (state x symbol), read out through disjoint neuron-ID
blocks. `PREREG_theorem_regime.md` already found homeostasis had no job on a
feedforward organ; this is the same boundary from the other side.

Also worth stating: NONE's distinct fraction is 1.000 and its arc-assembly
overlap reads 0.00x chance. **There is no merger on this organ to fix.** The
defect the composition repairs is not present, so there was nothing for it to
buy -- which is the cleanest explanation of the whole result and was visible in
the first table.

### Disclosed result: PREREG_refraction_stability R5

As disclosed before the run: NONE-vs-C at the organ default refraction IS that
note's R5. **R5 PASSES decisively** -- scaling OFF is more stable than ON,
0.840 vs 0.000, paired delta +0.840 +/- 0.059. Written into the refraction note;
it must not be re-scored there as independent evidence.

### Bar summary

    PASS  O1  regime asserted for every cell (kp 35.0 vs floor 29.7)
    FAIL  O2  the decay does not reproduce -- precondition, so O3-O6 are void
    ----  O3  void (INCONCLUSIVE: both arms at 0.000)
    ----  O4  void (no merger present: every arm distinct 1.000)
    ----  O5  void (INCONCLUSIVE)
    ----  O6  void as registered, though the direction is unambiguous
    PASS  O7  the baseline is not a w_max artifact
    PASS  O8  normalization breaks the organ at both w_max
    FAIL  O9  no climb-then-degrade shape at either w_max
