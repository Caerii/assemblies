# The P600 needed the WHERE channel (#121)

**Registered:** `research/experiments/erp_pathway_vs_area_binding.py` +
`role_binding_deficit` (erp/adapters.py), both at ece605f
**Artifacts:** `erp_pathway_vs_area_binding_results.json` / `.log` (10 seeds)
**Figures:** `research/figures/fig_121_{dissociation,deficits}.{png,pdf}`

## Question

The shipped P600 (`anchored_p600_live`, a drive-channel energy deficit)
is measured to be an AREA-IDENTITY readout: on the pathway-only contrast
(two nouns, same core area, one bound into ROLE_PATIENT during training
and one not) it reads AUC 0.515 -- chance -- while `input_drive`'s own
contract says drive measures HOW MUCH and can never see WHERE. #52 closed
with the binding channel discriminating stored role bindings at
0.97-1.000. Can a `read_binding`-based deficit see the pathway learning
the drive channel cannot?

## Verdict: YES. Both bars pass; the adapter is ADOPTED as the ERP
## role-integration readout (per the pre-stated decision rule).

10 seeds, three arms x 10 zero-corpus-frequency items each, both
channels paired on the same parsers, AUC mean +/- sd:

| contrast                            | binding (WHERE) | drive (HOW MUCH) |
|-------------------------------------|-----------------|------------------|
| PATHWAY only (agent_only vs patient)| **0.916 ± 0.026** | 0.800 ± 0.072  |
| SHIPPED (verb_object vs patient)    | **1.000 ± 0.000** | 0.944 ± 0.048  |
| AREA only (verb_object vs agent_only)| 0.846 ± 0.067   | **0.947 ± 0.066**|

- **P-PATHWAY: PASS** (bar 0.75; every seed >= 0.865; binding > drive in
  10/10 paired seeds). The WHERE channel sees word-level binding history.
- **P-SHIPPED: PASS at ceiling.** Fake-perfect audit run: the 1.000 is
  genuine non-overlap, not degeneracy -- patient-trained deficits are
  graded 0.000-0.267 and land on the word's OWN stored binding 89/100;
  verb deficits are graded 0.833-0.967 (never a constant, never
  self-landing). The distributions simply do not touch.
- **P-DRIVE: FAIL (reported as measured).** The registered replication
  band [0.40, 0.65] around the control's 0.515 did not hold -- the drive
  channel now reads 0.800 ± 0.072 on the same contrast. The control was
  measured before the growth-ratchet engine fix (53e9808), which repaired
  exactly the class of starved fiber the agent-only arm depends on;
  attribution is NOT claimed here (nothing in this design separates the
  engine fix from other changes since), but "the drive channel's chance
  reading was partly an artifact of dropped writes" is now the standing
  hypothesis, and the control's 0.515 should not be quoted without this
  caveat.
- **#120's inversion did NOT reappear** (pathway AUC 0.92, not 0.35-0.46);
  the landing census shows no dominant-attractor capture (top landing in
  any arm: 13/100).

## The structure in the deficits (fig_121_deficits)

Three cleanly ordered levels, pooled over 100 items/arm:

- patient-trained nouns: 0.038 mean -- recall lands on the word's own
  stored binding.
- agent-only nouns: 0.663 mean but BIMODAL (sd 0.309, min 0.000): a
  minority of never-bound nouns land squarely on a stored binding. That
  is graded generalization through shared core structure, not noise --
  and it is precisely the kind of item an ERP account would call a
  "plausible but unattested" filler.
- verbs in the object slot: 0.906 mean, floor 0.833 -- the cross-area
  untrained pathway delivers essentially no steering.

## What each channel is FOR (the paired dissociation)

Binding wins on pathway (0.916 vs 0.800); drive wins on area identity
(0.947 vs 0.846). Neither dominates: drive reads which AREA the word
lives in (the categorial violation), binding reads whether THIS word was
ever integrated HERE (the selectional/experiential violation). The
composite P600 should eventually carry both -- but flipping the shipped
composite moves every published magnitude, so per the decision rule that
flip is its OWN A/B unit, not a rider on this one.

## Scope

`role_binding_deficit` is landed, tested in both directions (bound <
unbound through the production `ops.bind` protocol; undefined escapes
fire on absent preconditions), and is the sanctioned role-integration
readout for #28/#118 follow-ups. The stale crowding excuse in
`anchored_p600_live` is retired in place.
