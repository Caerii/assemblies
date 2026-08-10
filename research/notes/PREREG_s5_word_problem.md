# PREREG: the S5 word problem on an assembly organ

Registered before implementing or running.

## The claim being tested, and why this benchmark

Merrill, Petty & Sabharwal, *The Illusion of State in State-Space Models*
(arXiv:2404.08819v3), prove that S4 and Mamba-style SSMs lie in L-uniform TC0,
because their recurrence is a parallel prefix scan over constant matrix powers
(S4) or over DIAGONAL, i.e. scalar, products (S6/Mamba). Anything in TC0 cannot
solve an NC1-hard problem, and by Barrington (1989) the word problem of any
finite NON-SOLVABLE group is NC1-complete. S5 and A5 are non-solvable. Their
Figure 3 measures the consequence: Mamba needs depth growing with sequence
length on A5.

The same paper names two escapes. Theorem 5.1: a NONLINEAR recurrence
`h_i = sgn(A h_{i-1} + B x_i)` solves any regular language in one layer.
Theorem 5.2: INPUT-DEPENDENT full transition matrices do the same.

The A1 organ is both:

    arc_t = kWTA( W_sym . sigma_t  +  W_state . q_{t-1} )
    q_t   = kWTA( W_arc . arc_t )

k-WTA in place of `sgn`, and the symbol selects a full non-diagonal transition.
So the organ is on the NC1 side of their boundary BY CONSTRUCTION, and running
S5 is not a fishing trip -- it is the measurement that turns a structural
argument into evidence.

WHY A1 IS NOT ALREADY THAT EVIDENCE. A1 ran mod-3. Z3 is ABELIAN, its word
problem is in TC0, and it is precisely the easy case the paper says SSMs
handle. Nothing measured here so far demonstrates anything an SSM cannot do.

## The design

Four groups, chosen to match the paper's own comparison set so the arms mean
the same thing, and matched in ORDER so size is not the variable:

| group | order | class | word problem |
| --- | ---: | --- | --- |
| Z60 | 60 | abelian | TC0 -- easy for SSMs |
| A4 x Z5 | 60 | solvable, non-abelian | TC0 |
| A5 | 60 | NON-SOLVABLE | NC1-complete |
| S5 | 120 | NON-SOLVABLE | NC1-complete |

Each is an FSM in the form `NemoArcFSM` already takes: states are group
elements, symbols are a generating set, and the transition is right
multiplication. The organ is UNCHANGED -- no new mechanism is introduced for
this, which is the point.

Sequence lengths L in {10, 50, 100, 500}. The organ is CONSTANT SIZE across
all of them; only the number of steps changes. That is the axis on which the
paper's models need growing depth.

## Parameters

From A1's closed configuration and FIXED: k=70, beta=0.10, organ_p=0.40,
refracted_strength=0.10, 15 presentations, norm_init=False, engine
numpy_sparse, seeds 42..51.

`n_state` is set per group to `2 * |G| * k`, since state assemblies are
disjoint neuron-ID blocks and |G| differs across arms.

### Amendment 1, made BEFORE the first run: `n_arc` matches LOAD, not size

The paragraph above originally said `n_arc` follows A1 (5000). Working the
arithmetic before running shows that would sink every arm for a reason that
has nothing to do with the question. The arc's load is `M * k / n_arc` where
`M` is the number of distinct (state, symbol) conjunctions, here exactly
`|G| * |generators|`:

    A5, Z60, A4xZ5   M = 120   load at n_arc=5000 = 1.68
    S5               M = 240   load at n_arc=5000 = 3.36

[[REFRACTION-NEEDS-LOAD]] puts the operating window at roughly 0.2 to 1.15;
above it the assemblies do not fit. Every arm would fail, the non-solvable
arms slightly worse for having more states, and the result would read as a
solvability effect. That is the confound the equal-order design exists to
remove.

So `n_arc` is set per group to `round(M * k / 0.42)`, holding the arc at the
middle of its window for every arm. This is matching REGIME rather than
matching a number, and it is the choice that keeps solvability the only
variable. Declared here, before any data, because deciding it afterwards would
be indistinguishable from tuning. The achieved load is reported per group and
must come out at 0.42 for all four; if it does not, the sizing is wrong and
says so.

## Hypotheses

**S1.** A5 exact-trajectory accuracy at L=100 is >= 9/10 seeds.
*Prediction: PASSES.* A1 closed mod-3 at 10/10 and ran 2000 steps with zero
errors; the state alphabet here is larger but still given.

**S2 (the length claim).** Accuracy at L=500 is indistinguishable from L=10 on
every group. *Prediction: PASSES*, from [[SEQ-EXACT-RECOVERY]] -- k-WTA is
superattracting, so per-step error does not accumulate.

**S3 (the money bar).** Accuracy does not depend on SOLVABILITY: the A5 and S5
arms are indistinguishable from the Z60 and A4xZ5 arms.
*Prediction: PASSES, and this is the whole result.* It is the exact contrast
where Mamba degrades. If assemblies are flat across the four and SSMs are not,
the complexity argument has an empirical shadow.

**S4 (controls, must ALL be at chance).** Untrained; beta=0; and a
SHUFFLED-TABLE arm trained on a permuted transition table and tested against
the true one. *Prediction: PASSES.*

## Committed in advance

1. Controls run FIRST. The original mod-3 golden certified a DICTIONARY LOOKUP
   as a neural result -- it passed untrained, at zero presentations, and at
   beta=0 -- and was retracted. Any arm that a degenerate machine also passes
   is not evidence, so the controls gate the real arms rather than decorate
   them.
2. Every reported trajectory is read from the state assembly by nearest
   overlap. `NemoArcFSM.step` does this since the retraction; the test asserts
   it rather than trusting it.
3. `regime_audit` is reported per group. A negative result is not interpreted
   until every area clears kp >= 3 ln n ([[SEQ-REGIME]]) -- A1 spent a day on a
   5/10 that was a state area below its own floor.
4. The full L curve is reported for every group. No best cell.
5. WHAT THIS WOULD AND WOULD NOT SHOW. It would show a constant-size assembly
   organ solving an NC1-complete word problem at lengths where constant-depth
   models provably cannot. It would NOT show that assemblies LEARN such a
   machine: the alphabet is given and the transitions are teacher-forced, which
   is exactly the gap [[SEQ-STATE-CODE-EMERGENT]] records and A3 measures. Any
   write-up says so in the same breath as the result.
