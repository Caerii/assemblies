# substrate

The engines and kernels: drive semantics, pricing, sampling versus materialization, the hashed substrate and its layouts.
The reading map is [../README.md](../README.md).

- [DESIGN_dense_cross_fiber.md](DESIGN_dense_cross_fiber.md): DESIGN: the dense cross fiber -- one launch per drive, one per write, no store walk
- [DESIGN_dense_floor.md](DESIGN_dense_floor.md): DESIGN: the dense formulation to its bandwidth floor
- [DESIGN_exact_drive.md](DESIGN_exact_drive.md): Design: compute the drive, never store the substrate (task #85)
- [DESIGN_gpu_hashed_drive.md](DESIGN_gpu_hashed_drive.md): DESIGN: generate the connectome, don't fetch it — the GPU order of magnitude
- [DESIGN_present_only.md](DESIGN_present_only.md): DESIGN: the present-only fiber -- store what exists, one warp per brain
- [DESIGN_virtual_connectome.md](DESIGN_virtual_connectome.md): DESIGN: stop storing the connectome
- [FINDING_torch_pricing_exposed.md](FINDING_torch_pricing_exposed.md): The corrected hash EXPOSED a torch pricing divergence
- [PREREG_bar_tie.md](PREREG_bar_tie.md): PREREG: are the S5 soft defects k-WTA BAR TIES?
- [PREREG_drive_semantics_v2.md](PREREG_drive_semantics_v2.md): PREREG: drive semantics v2 — the decomposition becomes the definition
- [PREREG_organ_substrate.md](PREREG_organ_substrate.md): PREREG: is the word-problem arc's assembly DECAY the substrate-C merger?
- [PREREG_substrate_c_homeostasis.md](PREREG_substrate_c_homeostasis.md): PREREG: substrate C — per-round homeostasis, the theorems' actual hypothesis
- [PREREG_theorem_regime.md](PREREG_theorem_regime.md): PREREG: the theorem regime — homeostasis where its preconditions actually hold
- [a_beta_zero_null_does_not_hold_the_connectome_still.md](a_beta_zero_null_does_not_hold_the_connectome_still.md): `norm_init` does not invert trained drive — and a beta=0 null is not a null
- [candidate_sampler_ground_truth.md](candidate_sampler_ground_truth.md): The candidate sampler, measured against an engine that does not sample
- [canonical_refactor_plan.md](canonical_refactor_plan.md): One canonical way: the refactor plan
- [engine_pricing_unification.md](engine_pricing_unification.md): The k-WTA pricing law was implemented twice, and the copies diverged
- [epercent_wta_was_measuring_recruitment.md](epercent_wta_was_measuring_recruitment.md): E%-WTA's "emergent assembly size" was the recruited pool
- [exact_drive_equivalences.md](exact_drive_equivalences.md): Why the fast paths compute the same thing — proof sketches
- [graded_similarity_and_sampler_load.md](graded_similarity_and_sampler_load.md): Graded similarity is real, and the sampler destroys it by MERGING
- [input_drive_normalization_is_disabled.md](input_drive_normalization_is_disabled.md): `input_drive` divides by the alias, so its normalization silently does nothing
- [kwta_amplifies_input_overlap.md](kwta_amplifies_input_overlap.md): k-WTA amplifies input overlap — the law, and it predicts the parser
- [materialization_semantics.md](materialization_semantics.md): An area's size and a fiber's column extent are one invariant
- [the_bimodality_was_27_coin_flips.md](the_bimodality_was_27_coin_flips.md): The bimodality was 27 coin flips — the paper regime's residual is the exposure law, priced exactly by COLT22's margin
- [w_alias_back_catalogue.md](w_alias_back_catalogue.md): Back-catalogue audit: did the `area.w` alias void any standing result?
