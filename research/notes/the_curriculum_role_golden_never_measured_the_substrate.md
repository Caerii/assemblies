# The NEMO-2025 curriculum role-probe golden is substrate-invariant

## The broken claim

`nemo2025_curriculum.json` asserts `role_probe_accuracy_min = 1.0`. Cold, it
delivers **0.6667**.

**Not a regression I introduced.** Verified in a worktree at `6ecfa15` — the
commit this session started from — with `ASSEMBLIES_BACKBONE_CACHE=0`: the same
0.6667. It reproduces identically warm and cold.

## Why it stayed green

The golden was recorded **2026-06-23, before `norm_init` became the default
substrate**, and the failure was already observed on 2026-07-31. In between, CI
stayed green because **warm backbones deserialize instead of training** — a
golden can rot for weeks while the suite passes
(`backbone-fingerprint-gap`). It surfaced now only because adding a file moved
the source fingerprint and forced a cold retrain.

## The hypothesis, and the clean negative

Role probes ask whether a word is recoverable from its role, and this session
moved exactly that quantity: β 0.10→0.05 and `phon_weight` 1→6 took role ret@6
from 0.806/0.733 to 0.974/0.964. So the obvious hypothesis was that the golden
broke when role binding degraded.

| arm | role acc | word order | sent acc |
|---|---|---|---|
| golden as recorded | 0.6667 | True | 0.838 |
| β=0.05 | 0.6667 | True | 0.838 |
| phon_weight=6 | 0.6667 | True | 0.838 |
| β=0.05 + phon_weight=6 | 0.6667 | True | 0.838 |

**Refuted.** And the manner of refutation is the finding.

## Every arm is byte-identical, which is the real result

Not "similar" — **identical**, to the last digit, including `sent_acc` 0.838 and
the per-role breakdown (AGENT precision 1.0 / recall 0.5; PATIENT precision 0.5
/ recall 1.0). Two parameters that demonstrably reshape the substrate change
*nothing at all* here.

That is the silent-no-op signature. `evaluate_roles` calls `parser.parse(words)`
and reads `result["roles"]`, and that assignment is **substrate-invariant** —
so this metric was never measuring the assembly calculus. The systematic
confusion (one AGENT read as PATIENT, deterministically) is the signature of a
positional or symbolic rule, not of neural retrieval.

This lines up with what the repo already knows: role exclusivity is enforced
symbolically (`mutual-inhibition-prefers-untrained`), and #33 is open on whether
the symbolic role route can be retired for the neural one.

## What follows

1. **The golden's 1.0 → 0.667 was not caused by `norm_init`.** A substrate
   change cannot move a substrate-invariant metric. Something in the symbolic
   path changed; that is where to look.
2. **Do NOT re-record at 0.667.** Lowering a threshold until it passes converts
   a broken claim into a passing test, which is what the parity programme exists
   to prevent. The value is not the problem — the metric is.
3. **A parity golden whose value cannot respond to the model is not a parity
   golden.** Before the threshold is touched, `evaluate_roles` needs a readout
   that reads the role lexicons, which is #121's job.

## Limits

One seed (42), the golden's own. The denominator is **3** — a single probe item
is 0.333 — so this metric could not resolve a small effect even if it were
substrate-sensitive. The identical-across-arms result is what carries the
conclusion, not the 0.667 itself. I did not trace which line in `parse()`
assigns the roles; that is the next step and it is #33/#121 territory.
