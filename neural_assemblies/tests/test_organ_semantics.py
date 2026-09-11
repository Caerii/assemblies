"""Hashed organ choices are typed and rejected before GPU allocation."""

from dataclasses import replace

import pytest

from neural_assemblies import (
    AlignerSemantics,
    AlignmentStore,
    NormalizationMode,
    OrganSemantics,
    StateCode,
    StimulusDriveLaw,
    TieBreakRule,
    ExecutionSemantics,
    describe_assembly_memory,
    describe_hashed_arc_fsm,
    describe_hashed_aligner,
    describe_hashed_transducer,
)


def test_aligner_profile_separates_anchor_and_cross_fiber_semantics():
    semantics = describe_hashed_aligner(p=.2, rounds_word=3)

    assert semantics.anchor_gain == 5.0
    assert semantics.anchor_plasticity.value == "none"
    assert semantics.anchor_normalization is NormalizationMode.INVERSE_INDEGREE
    assert semantics.cross_normalization is (
        NormalizationMode.INVERSE_INDEGREE_WITH_COLUMN_SCALING
    )
    assert semantics.cross_store is AlignmentStore.PRESENT_ONLY
    assert semantics.rounds_per_pair == 3
    assert AlignerSemantics.normalize(semantics.to_dict()) == semantics
    assert describe_hashed_aligner(store="csr").cross_store is (
        AlignmentStore.DENSE_COUNTS
    )


@pytest.mark.parametrize("constructor", ["hashed", "scheduled"])
def test_aligner_mismatch_rejects_before_cuda_loading(constructor):
    required = describe_hashed_aligner(p=.2, rounds_word=3)
    if constructor == "hashed":
        from neural_assemblies.core.torch_engine._hashed_aligner import HashedAligner

        construct = lambda: HashedAligner(  # noqa: E731
            [1], ["word"], ["feature"], n=20, k=2, feat_n=20, feat_k=2,
            p=.2, rounds_word=2, device="unusable-device",
            aligner_semantics=required,
        )
    else:
        from neural_assemblies.core.torch_engine._scheduled_aligner import ScheduledAligner

        construct = lambda: ScheduledAligner(  # noqa: E731
            [1], n=20, k=2, feat_n=20, feat_k=2, n_words=1,
            n_features=1, p=.2, rounds_word=2, device="unusable-device",
            aligner_semantics=required,
        )
    with pytest.raises(ValueError, match="aligner_semantics mismatch"):
        construct()


@pytest.mark.parametrize("kwargs", [
    {"store": "typo"}, {"store": "present", "w_max": 20.},
    {"scaling": True, "w_max": 20., "store": "dense"},
    {"p": 0.}, {"rounds_word": 0}, {"tie_jitter": float("nan")},
])
def test_invalid_aligner_relation_rejects_during_description(kwargs):
    with pytest.raises(ValueError):
        describe_hashed_aligner(**kwargs)


def test_memory_profiles_distinguish_dense_and_scaling_fibers():
    dense = describe_assembly_memory(norm_init=True, synaptic_scaling=False)
    scaling = describe_assembly_memory(norm_init=True, synaptic_scaling=True)

    assert dense.substrate.normalization is NormalizationMode.INVERSE_INDEGREE
    assert scaling.substrate.normalization is (
        NormalizationMode.INVERSE_INDEGREE_WITH_COLUMN_SCALING
    )
    assert dense.substrate.stimulus_drive is (
        StimulusDriveLaw.ZERO_OR_SIZE_AFFERENT_COUNT
    )
    assert dense != scaling
    assert describe_assembly_memory(gate=True).convergence_gate


def test_fsm_and_transducer_defaults_are_semantically_distinct():
    fsm = describe_hashed_arc_fsm()
    transducer = describe_hashed_transducer()

    assert fsm.substrate.stimulus_drive is (
        StimulusDriveLaw.FIXED_BERNOULLI_AFFERENT_COUNT
    )
    assert fsm.substrate.default_tie_break is TieBreakRule.LOWEST_NEURON_ID
    assert fsm.state_code is StateCode.ASSIGNED_BLOCKS
    assert transducer.substrate.stimulus_drive is (
        StimulusDriveLaw.ZERO_OR_SIZE_AFFERENT_COUNT
    )
    assert transducer.substrate.default_tie_break is (
        TieBreakRule.DETERMINISTIC_HASH_JITTER
    )
    assert transducer.state_code is StateCode.INDUCED_ASSEMBLY
    assert describe_hashed_transducer(successor_gain=.25) != transducer


def test_organ_semantics_wire_is_strict_and_canonical():
    expected = describe_hashed_transducer(state_mode="copy", predict_gain=1.0)
    wire = expected.to_dict()

    assert OrganSemantics.normalize(wire) == expected
    with pytest.raises(ValueError, match="unknown"):
        OrganSemantics.normalize({**wire, "approximate": True})
    with pytest.raises(ValueError, match="missing"):
        OrganSemantics.normalize({k: v for k, v in wire.items() if k != "state_code"})


def test_execution_profile_collection_is_immutable():
    execution = ExecutionSemantics("organ", {"default": describe_hashed_arc_fsm()})
    with pytest.raises(TypeError):
        execution.profiles["other"] = describe_hashed_arc_fsm()


def test_tie_rule_cannot_disagree_with_jitter():
    valid = describe_hashed_arc_fsm()
    wrong_substrate = replace(
        valid.substrate,
        default_tie_break=TieBreakRule.DETERMINISTIC_HASH_JITTER,
    )
    with pytest.raises(ValueError, match="describe different rules"):
        replace(valid, substrate=wrong_substrate)


@pytest.mark.parametrize("organ", ["memory", "fsm", "transducer"])
def test_required_semantics_mismatch_rejects_before_device_allocation(organ):
    if organ == "memory":
        from neural_assemblies.core.torch_engine._memory import AssemblyMemory

        required = describe_assembly_memory(norm_init=False)
        construct = lambda: AssemblyMemory(  # noqa: E731
            [1], 20, 2, .2, norm_init=True, device="unusable-device",
            organ_semantics=required,
        )
    elif organ == "fsm":
        from neural_assemblies.core.torch_engine._hashed_fsm import HashedArcFSM

        required = describe_hashed_arc_fsm(zero_or_size=False)
        construct = lambda: HashedArcFSM(  # noqa: E731
            [1], ["a"], ["x"], [("a", "x", "a")], n_arc=20, k=2,
            p=.2, zero_or_size=True, device="unusable-device",
            organ_semantics=required,
        )
    else:
        from neural_assemblies.core.torch_engine._hashed_transducer import HashedTransducer

        required = describe_hashed_transducer(state_mode="copy")
        construct = lambda: HashedTransducer(  # noqa: E731
            [1], ["x"], n=20, k=2, p=.2, state_mode="induced",
            device="unusable-device", organ_semantics=required,
        )

    with pytest.raises(ValueError, match="organ_semantics mismatch"):
        construct()


def test_invalid_semantic_switches_reject_during_description():
    with pytest.raises(ValueError, match="column scaling requires"):
        describe_assembly_memory(norm_init=False, synaptic_scaling=True)
    with pytest.raises(ValueError, match="state_mode"):
        describe_hashed_transducer(state_mode="unknown")
    with pytest.raises(ValueError, match="tie_jitter"):
        describe_hashed_arc_fsm(tie_jitter=float("nan"))
