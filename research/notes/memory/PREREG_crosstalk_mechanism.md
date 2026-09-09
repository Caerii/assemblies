# PREREG: is the (n/k)^2 capacity law second-order CROSSTALK?

Registered before running. [[CAP-RATIO]] established that M* is a function of
n/k alone (constant to +/-4% while n varies 4x); the exponent ~2.2 was measured
with NO mechanism and is therefore not quotable. This registers a mechanism
and its falsifiable predictions.

## The mechanism

Half-cue retrieval of assembly B fails when foreign trained drive out-competes
own trained drive at the k-WTA bar. Under a cue of B:

* OWN drive to a member of B:   ~ k * p * (g - 1)   (trained recurrent edges)
* FOREIGN trained drive reaches a neuron j only through SECOND-ORDER overlap:
  j belongs to ~ M k/n other assemblies; each such assembly C shares ~ k^2/n
  members with the cue; each shared member drives j through a trained edge:

      foreign ~ M * (k/n) * (k^2/n) * p * (g - 1)  =  M k^3/n^2 * p * (g-1)

Failure at foreign/own ~ const gives  M* ~ const * (n/k)^2  -- which is exactly
why `M* (k/n)^2` is the invariant CS1 found, and why n and k never appear
separately.

## The predictions, which the existing data cannot have fitted

Everything measured so far ran at p = 0.5, beta = 0.10. The crosstalk ratio
CANCELS both p and (g-1), so:

**X1 (p-invariance).** At n=4000, k=100 (n/k = 40), T=8, beta=0.10, w_max=20,
16 brains, arm B, p in {0.3, 0.5, 0.7} -- all IN REGIME (kp = 30/50/70 vs
floor 24.9). Prediction: M* equal across p within bracket resolution.
PASS: max/min M* <= 1.25 with all three brackets overlapping or adjacent.
FAIL: M* trends with p by more than that; the direction names the missing
term (e.g. M* rising with p implicates the UNTRAINED background, which scales
as sqrt(k p) and is NOT p-invariant).

**X2 (beta-invariance).** Same cell at p=0.5, beta in {0.05, 0.10, 0.20}.
Same bar. Caveat stated now: beta also controls CONVERGENCE
([[SEQ-BETA-WINDOW]]); at beta=0.05 with T=8 the assemblies may not converge,
which reads as LOW M* for a reason outside the mechanism. If the beta=0.05 arm
fails the distinctness/rank1 gates at SMALL M (M << predicted M*), that arm is
reported UNINFORMATIVE rather than counted against X2.

**X3 (the constant).** The mechanism predicts C = M*(k/n)^2 matches the
measured ~0.0155-0.020 via foreign/own ~ const at the bar; it does NOT predict
the residual 0.19-power drift of C with n/k. X3 is explicitly NOT claimed --
the open residual stays open, and passing X1+X2 does not close it.

## Interpretation, stated now

* X1 and X2 pass -> crosstalk is ADOPTED as the mechanism of the square law;
  CAP-RATIO's caveat is upgraded from "no mechanism" to "second-order
  crosstalk, p- and beta-invariance verified"; the exponent residual stays
  open and unquotable.
* X1 fails -> the mechanism is wrong or incomplete; the sign of dM*/dp is
  reported and the mechanism is NOT adopted.
* X2 alone fails outside the convergence caveat -> (g-1) does not cancel,
  pointing at the w_max clip or count-depth effects; reported, not adopted.

---

## Result (2026-08-25): X1 FAILS, X2 FAILS -- crosstalk is REFUTED, informatively

All cells n=4000, k=100 (n/k = 40), T=8, w_max=20, 16 brains, arm B, exact
path. Baseline (p=0.5, beta=0.10): M* = 23.5 [20, 24), fill 0.53 -- the
C ~ 0.0155 calibration predicted 24.8.

**X1 (p-invariance): FAIL.**

    p=0.3  M* = 28.8  [28, 32)  fill 0.64
    p=0.5  M* = 23.5  [20, 24)  fill 0.53
    p=0.7  M* = 17.6  [16, 20)  fill 0.41

max/min = 1.64 against the 1.25 bar, monotone FALLING in p (~ p^-0.6 over this
range). The crosstalk ratio is p-free, so this is not a partial miss; the
mechanism as stated is wrong. Note the direction also rules out the untrained-
background term named in the FAIL clause: background fluctuation ~
sqrt(k(1-p)/p)/n FALLS with p, which would push M* UP with p. Something that
STRENGTHENS with p is binding.

**X2 (beta-invariance): FAIL, outside the convergence caveat.**

    beta=0.05  M* >= 48 (censored high: curve never crossed; grid to 48)
    beta=0.10  M* = 23.5
    beta=0.20  M* =  8.0 (pairwise 4.6x chance already at M=8)

The caveat anticipated beta=0.05 might read LOW through non-convergence; it
reads HIGH (rank1 1.000 at M=8..48). M* ~ beta^-1.3 over this range. (g-1)
does not cancel.

**Per the registered interpretation: the mechanism is NOT adopted.** The
residual direction worth registering next -- as hypothesis, not claim -- is
FORMATION-side interference: during training, recurrence pulls a new assembly
toward stored ones through shared trained edges, a pull that strengthens with
both beta and p; consistent with [[beta-opposes-capacity-and-depth]] and
[[recurrence-is-the-collapse-channel]], and with the cliff (attractor merging
is catastrophic, not gradual).

**Consequence for CAP-RATIO, applied to the register:** "M* is a function of
n/k alone" was established AT FIXED p = 0.5, beta = 0.10 and is now known to
be conditional on both: M* moved 1.6x across p in [0.3, 0.7] and ~6x across
beta in [0.05, 0.20] at fixed n/k. The claim's preconditions are sharpened
accordingly; CS1 (n-invariance at fixed k/n ratio AND fixed p, beta) stands.
