# What the papers actually prescribe — the label is WHERE and WHEN, never WHAT, and our operating point violates their margin condition

**Task #151, literature unit (paper-parity process: claims mined from
the PDFs with fitz, quoted with page numbers, checked against our
measured numbers before recommending anything). Papers read:
mitropolsky2025_acquisition.pdf (deep), dabagia2022_colt.pdf (deep),
scanned for morphology terms: zero hits — the number/tense arc is
genuinely beyond the literature, which is why its failures had no
paper to warn us.**

## Finding 1 — the papers NEVER fire a label stimulus

Our `train_number` co-fires `number_SG`/`number_PL` label stimuli with
the word into the value area. **No paper does anything like this.**

- Acquisition 2025 (p9): roles are "represented as assemblies in the
  corresponding ROLE areas, with strong synaptic connections to the
  corresponding assemblies in LEX1 and LEX2." The role-area CONTENT is
  the scene's word — a WORD-specific assembly — and the role label is
  expressed by WHICH area the scene routes it to. Supervision is
  routing, content is the word.
- COLT22 (Algorithm 1, p6): "The only form of supervision required is
  that all training samples from a given class are presented
  consecutively" — with an inhibition reset between classes. The label
  is WHEN; the class assembly's neurons are selected by the CLASS'S OWN
  STIMULI (extreme columns w.r.t. the class input), then boosted.

Our label stimulus is the deviation that produced everything the
attribution unit measured: winners selected by a third party (the
label), so images are class attractors that are TYPICAL w.r.t. the
word — and every readout downstream fights that fact. In the papers'
designs the trained winners are selected by the SAME drive recall
uses, so own = boosted-extreme vs other = unboosted-extreme: the
extreme-value statistics are common-mode and CANCEL, and the Hebbian
boost alone decides. Their MI device is calibrated BY CONSTRUCTION;
ours was not.

## Finding 2 — COLT22's margin condition predicts our n-collapse

Remark 2 (p7): plasticity must satisfy beta >~ c·sqrt(2 ln(n/k)/(kp))
(r~1). The comparison at our operating points, effective boost
(1+.05)^(3 exposures × 5 plastic rounds) ≈ 2.08 vs the extreme-value
factor sqrt(2 ln(n/k)/kp):

| n, k, p | kp | extreme factor | our boost | verdict |
|---|---|---|---|---|
| 3000, 30, .05 | 1.5 | 2.48 | 2.08 | below margin — marginal |
| 10000, 30, .05 | 1.5 | 2.78 | 2.08 | below — fails (measured!) |
| 10000, 100, .05 | 5.0 | 1.36 | 2.08 | comfortable |

The theory's sufficient condition says our k=30 regime (kp=1.5, an
E-series inheritance from the slow-test scale, NOT a paper parameter)
cannot certify recall at any n we ran, and worsens with n — exactly
the own/other crossing we measured (1.12 → 0.53). The papers run
k=100, kp=5, where the boost clears the margin. This is
[[kp-decides-whether-beta-helps]] and [[norm-init-stability-threshold]]
with the constant finally attached.

## Finding 3 — what the papers do NOT cover (our genuine frontier)

- Inflectional morphology: absent from all ten papers.
- Class imbalance: COLT22 analyzes symmetric few-shot (O(log k) per
  class, block-presented); interleaved Zipfian 90/10 streams are
  outside its theory. The acquisition paper's own input-mix constraint
  (>50% transitive for word order) shows they KNOW distribution mix
  gates learnability — they just never met ours.
- Homeostatic scaling: not in NEMO at all. Our E-series scaling was an
  addition; the graduation measured where it breaks (class imbalance).

## The prescription (registered design for the next unit)

Bring value-area training back inside the papers' calibrated regime,
2×2 on the Brown substrate, seeds 42–46, n ∈ {3000, 10000}:

- **Axis 1 — label-stimulus share ∈ {current, 0 (routing-only)}**:
  teacher keeps ROUTING the projection (the split already does this);
  the WORD drives the area. Prediction: word-specific images (image ∩
  class-attractor falls from 0.998), MI recall calibrates itself, no
  custom readout needed.
- **Axis 2 — value-area drive kp ∈ {1.5 (k=30), ~5 (k=100 value areas
  or fiber-p raised)}**: the margin condition's lever. Prediction:
  at kp≈5 the boost clears the extreme factor and recall becomes
  n-robust; at kp=1.5 even routing-only stays marginal.

Bars: plain-MI balanced ≥ 0.70 with SG ≥ 0.60 at BOTH n in the
(routing-only, kp≈5) cell; image ∩ class-attractor < 0.5 there (the
mechanism check, not just the score); E14 merging counter must not
regress; synthetic #149-gate guard non-negative. The exposure law
stays the floor: 1-exposure forms are at chance in every arm of every
paper too — COLT22 needs O(log k) examples, and that is level 2 of the
five-level law, not a defect.
