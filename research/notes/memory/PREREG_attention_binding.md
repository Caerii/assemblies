# Registration: what a NON-COMPOUNDING plasticity buys the calculus

## Why this exists

Every plasticity rule in this repository compounds. `w *= 1 + beta`, clipped at
`w_max`, with no way to retract. Three of its own recorded findings are
consequences of that and not of anything else:

- Hebbian mass follows FREQUENCY, so representations track base rates
- the clip arithmetic `c* = ln(w_max max(1,kp)/base)/ln(1+beta)` bounds how long
  training may usefully run
- refraction, the ANTI-compounding force, turned out to dominate the whole
  sequence line ([[SEQ-INTEGER-ARC-LOAD]], [[ARC-CONJUNCT-EXPOSURE]])

`AttentionArea`, ported from mdabagia/nemo and checked against the clone bit for
bit (`test_attention_area_parity.py`), is the first rule here that does NOT
compound:

    w[pre, post] = (1 + p) * (w[pre, post] > 0)      idempotent, weights binary
    w -= change                                       and exactly reversible

This registration asks what that buys, in terms the calculus cares about:
COMPOSITION and FREQUENCY. Both are properties of the rule rather than of any
task, so both should be exact rather than approximate -- which makes them easy
to refute.

## What runs

One area, `n = 512`, `cap = 16`, `density = 0.30`, `plasticity = 0.25`. Twenty
seeds. Two rules compared on the SAME connectome per seed:

    MULTIPLY   w *= (1 + p), never released -- what every other operation here does
    SET        w = (1 + p)(w > 0), released after each read -- the ported rule

**Composition.** Bind a pair, read it, perform `d` OTHER bindings, read the
first pair again. `d` in 0, 1, 2, 4, 8, 16, 32, 64. The measure is the overlap
of the re-read with the original read.

**Frequency.** Bind a rare pair ONCE and a frequent pair `c` times, `c` in
1, 2, 4, 8, 16, then read through a cue overlapping BOTH queries equally so the
two keys compete. The measure is the ratio of mean drive into the frequent key
over the rare key.

## Bars

- **AT-1, the exploratory result reproduces at twenty seeds.** Under SET,
  composition overlap is 1.000 at every depth; under MULTIPLY it is below 0.95
  by depth 64.
- **AT-2, composition under SET is EXACT, not merely good.** Overlap is
  identically 1.000 at every depth on every seed. A single seed at 0.999 fails
  this, and should: the substrate is restored by construction, so anything
  below exact means the release is not exact.
- **AT-3, SET is exactly frequency blind.** The frequent-over-rare drive ratio
  is identical at `c = 1` and `c = 16` on every seed, to floating-point
  equality.
- **AT-4, MULTIPLY is not.** Its ratio at `c = 16` is at least 3x its ratio at
  `c = 1`. Without this the tie in AT-3 could be a dead instrument rather than
  a property.
- **AT-5, the two rules share a connectome.** At `c = 1` and `d = 0` the two
  arms agree, so every later difference is the rule and not the draw.

## What each outcome means, stated in advance

**All five pass:** the calculus has had exactly one kind of plasticity, and it
is the kind that cannot express attention -- under it, attending to a pair is
indistinguishable from learning it, and querying repeatedly is
indistinguishable from believing harder. Composition depth becomes unbounded
where it was finite.

**AT-2 fails while AT-1 passes:** the release is approximately but not exactly
inverse, which would contradict the parity test and mean the port has a defect
the reference does not.

**AT-4 fails:** the comparison is not measuring frequency at all at these
parameters -- the cue or the counts are wrong -- and AT-3's tie says nothing.
The fix is the design, not the claim.

## Scope, stated plainly

This measures PROPERTIES OF A RULE on one area with random supports. It is not
a demonstration that attention does anything useful, and no task is involved.
The reference's own attention protocol, and whether transient binding composes
with projection and merge as the calculus's other operations do, are separate
questions this does not touch.

## An execution-kind gap, recorded before the result

This study does not run through `research.runner` and holds no artifact, and
the reason is a real gap rather than haste. The runner admits five execution
kinds -- BRAIN, ORGAN, ALIGNMENT, BASELINE, REFERENCE -- and a plasticity-RULE
study on a bare area with random supports is none of them. No Brain and no
organ runs; it is not an aligner; a substrate DOES run so it is not a
substrate-free baseline, and `BaselineSemantics` is corpus/scoring/tie-break for
language baselines besides; and the port is ours, not vendored, so it is not a
declared REFERENCE either.

Admitting a sixth kind is a schema migration, the same class of work as adding
a field to `OrganSemantics`, and forcing a bad fit to avoid it would put a false
semantic profile in a stored record. The results below are therefore recorded
INLINE with their bars evaluated -- the category the evidence audit already
recognises as "inline results, no retained artifact" -- and the migration is
added to the backlog rather than done in passing.

## Result (2026-09-13): both properties are EXACT, at twenty seeds

Seeds 42..61, `n = 512`, `cap = 16`, `density = 0.30`, `plasticity = 0.25`.

    PASS AT-1   PASS AT-2   PASS AT-3   PASS AT-4   PASS AT-5

    COMPOSITION -- overlap of the first readout after d other bindings
    rule            0       1       2       4       8      16      32      64
    set         1.000   1.000   1.000   1.000   1.000   1.000   1.000   1.000
    multiply    1.000   1.000   0.991   0.988   0.966   0.953   0.903   0.844

    FREQUENCY -- frequent/rare drive ratio after c repetitions
    rule            1       2       4       8      16
    set         0.968   0.968   0.968   0.968   0.968
    multiply    0.968   1.097   1.454   2.824  12.849

**Both exactness bars hold PER SEED, not on average.** AT-2 requires overlap
identically 1.000 on every seed at every depth, and AT-3 requires the
frequent-over-rare ratio identical at `c = 1` and `c = 16` on every seed. A
single seed at 0.999, or a ratio differing in the last digit, fails either --
and neither does.

AT-5 confirms the two arms share a connectome (they agree at `c = 1, d = 0`), so
every difference above is the RULE and not the draw. AT-4 rules out the
alternative reading of AT-3's tie: the multiply arm moves by 13.27x over the
same counts, so the instrument is alive and the tie is a property.

The `0.968` baseline is the random connectome's asymmetry between the two key
supports. It never moves under the set rule, which is the point; it is not
expected to be 1.000.

### What this establishes

The calculus has had exactly ONE kind of plasticity, and it is the kind that
cannot express attention. Under a compounding rule, attending to a pair is
indistinguishable from learning it, and querying a pair repeatedly is
indistinguishable from believing it harder. The non-compounding rule separates
those, and the separation is exact rather than approximate:

- composition depth goes from FINITE (0.844 by 64 bindings) to UNBOUNDED
- frequency sensitivity goes from 13x to NONE
- the clip arithmetic stops applying, since weights are binary

### What it does NOT establish, stated plainly

Properties of a rule on one area with random supports. **No task is involved**
and nothing here shows attention DOES anything useful. Whether transient
binding composes with `project`, `associate` and `merge` as the calculus's other
operations do is untouched, and so is the reference's own attention protocol.
Those are the next questions, and they need a task before they mean anything.
