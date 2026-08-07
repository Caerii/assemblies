# Role binding reads back INVERTED — unresolved, do not cite

**Status: an unresolved measurement, not a finding.** Recorded because the
result is surprising in a specific direction and the next steps are concrete.

## The question

`erp_pathway_vs_area_control.py` showed the P600 cannot distinguish a noun bound
into `ROLE_PATIENT` from one never bound there. Two explanations with very
different blast radii: the readout washes it out (an ERP problem), or the
binding barely happened (a **role-mechanism** problem touching every role result
in the repo).

## The probe

For each of 71 trained nouns, hand `input_drive` the word's **stored** core
assembly and read the drive it delivers into `ROLE_PATIENT`. Single target, so
the divisor is constant across words and cancels in a rank statistic — this
matters, see `input_drive_normalization_is_disabled.md`.

`source_assemblies` is used rather than projecting `phon → NOUN_CORE`, because
`read_only()` freezes winners and would have made every word read the same
assembly — a manufactured null indistinguishable from "binding is dormant".

Groups the parser supplies itself: 36 `patient_bound`, 29 `agent_bound`
(bound into `ROLE_AGENT` only), 6 `unbound`.

## The result

AUC on drive into `ROLE_PATIENT`, seeds 11 / 12 / 42:

| contrast | AUC |
|---|---|
| patient_bound above agent_bound | 0.4583 / 0.3482 / 0.3578 |
| patient_bound above unbound | 0.4630 / 0.3935 / 0.4861 |

**Consistently below 0.5.** Nouns bound into `ROLE_PATIENT` deliver *less* drive
there than nouns never bound there.

That is neither of the pre-registered outcomes. "Dormant" predicts ~0.5;
"readable" predicts >0.5. An inverted, seed-consistent result says something
structural is happening that the design does not model.

## Why I am not calling it a finding

Three things are unaccounted for:

1. **Absolute drive levels flip across seeds.** Seed 11: mean → PATIENT 12.07,
   → AGENT 5.26. Seed 42: → PATIENT 4.44, → AGENT 14.80. The same groups on the
   same protocol swap which area is "hot". Whatever sets the overall level is
   not the binding, and may be swamping it.
2. **The range is enormous** — 0.013 to 26.7 — so a few words dominate any mean,
   and the AUC is carried by rank orderings I have not inspected per item.
3. **`unbound` is n = 6.** Too small to anchor anything.

## What it does rule out

**The crowding explanation I offered is not supported.** `NOUN_CORE` pairwise
assembly overlap measures **0.126–0.132** across seeds — the noun assemblies are
largely distinct. So "per-word pathway strength is invisible because the source
assemblies overlap" is refuted for the source side, and #52's crowding figure
(0.15–0.22) describes the role areas rather than the core areas.

## Next steps, in order

1. **Verify the probe before believing the number.** Confirm `activate_assembly`
   puts the stored assembly's neurons into the source area in the right index
   space (`Assembly.winners` are NEURON IDS, `Area.winners` are COMPACT indices
   — [[two-index-spaces-compact-vs-neuron-id]]). If that mapping is wrong, the
   activation is effectively random with respect to binding and every number
   above is void.
2. **Ask whether `bind()` changed any weight at all.** Snapshot the
   `NOUN_CORE → ROLE_PATIENT` connectome rows for a word's assembly before and
   after a `bind()` call. That is a direct question about the mechanism and does
   not depend on any readout.
3. **Check `norm_init` in-degree normalisation as a candidate inverter.** A role
   neuron that received many bindings gets normalised down, which could make a
   well-bound target deliver *less* per-candidate drive. That would be a real
   and interesting mechanism rather than a bug, and it is testable by comparing
   `norm_init=True/False`.

Until at least (1) is done, nothing here should be cited in either direction.
