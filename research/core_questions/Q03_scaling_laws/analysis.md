# Analysis

The old tracker language treated this question as if a precise logarithmic law
had already been nailed down. The result summary is more cautious, and that
more cautious reading is the right one.

What the evidence supports:

- larger networks in the tested regime often converge faster
- the trend is consistent with logarithmic scaling
- persistence reflects two opposing forces:
  - larger networks help
  - decreasing `k/n` makes autonomous maintenance harder

What remains open:

- a derivation for the observed scaling
- whether the trend survives across other sparsity schedules
- whether the `n=2000` behavior is noise, a finite-size effect, or a sign of a
  nearby phase boundary
