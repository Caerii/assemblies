"""`ErpProtocol` must make an arm a VALUE, not a moment in time.

The defect it replaces: `ERP_EXPECTED_SLOT` is read three call levels below any
caller, so experiments A/B by mutating `os.environ` and restoring it. That makes
"which arm produced this number" a property of WHEN you looked. These tests pin
the properties that fix it -- derivation is an expression, the value is frozen,
and reading the environment happens in exactly one place.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from neural_assemblies.assembly_calculus.emergent.evaluation.erp.protocol import (
    DEFAULT_PROTOCOL,
    ErpProtocol,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.runner import _check_engine_identity
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.calibration import calibrate_erp_thresholds


class TestTheShippedDefault:

    def test_expected_slot_is_on_and_afferent_energy_is_off(self):
        """`expected_slot` was ADOPTED on 2026-08-06 (#108).

        It is the only thing that area-matches the grammatical/violation
        contrast: `structural_role_area` dispatches on the OBSERVED category,
        so a category violation probes VP while its control probes
        ROLE_PATIENT -- in every frame set, at every seed. Area-matched FRAMES
        cannot fix that, because both shipped sets already put the critical
        word in object position and the dispatch still splits them.

        It was off on the claim that it "inverts on freshly-trained parsers".
        REFUTED: cold, 10 seeds, `disk_hits=0 trained_fresh=10`, AUC
        0.9056 -> 0.7167, above chance on every seed. It shrinks the effect,
        which is what removing a confound does.

        `afferent_energy` stays off: rejected on measurement (AUC 0.000, zero
        seed variance) -- though that rejection was itself measured while the
        arms probed different areas, so it is worth re-measuring now.
        """
        assert DEFAULT_PROTOCOL.expected_slot is True
        assert DEFAULT_PROTOCOL.afferent_energy is False

    def test_the_default_describes_itself_rather_than_saying_nothing(self):
        """A study that recorded no flags must not look like one that recorded
        nothing at all -- and the description must name what was ON, not the
        delta from a default that changes over time."""
        assert DEFAULT_PROTOCOL.describe() == "expected_slot;context_reset=construction"
        assert ErpProtocol(expected_slot=False).describe() == "no-flags;context_reset=construction"


class TestAnArmIsAValue:

    def test_deriving_an_arm_is_an_expression(self):
        """Stated in BOTH directions, so it does not silently become vacuous.

        This used to derive `expected_slot=True` from a False default. When the
        default flipped, `base.with_(expected_slot=True)` still passed while
        asserting nothing -- base and arm agreed. Deriving the control arm off
        the shipped default is the version that keeps testing something.
        """
        base = ErpProtocol()
        control = base.with_(expected_slot=False)
        assert control.expected_slot is False
        assert base.expected_slot is True, "deriving must not mutate the base"

        candidate = control.with_(expected_slot=True)
        assert candidate.expected_slot is True
        assert control.expected_slot is False, "deriving must not mutate the base"

    def test_the_protocol_is_frozen(self):
        """A mutable protocol is the environment variable with extra steps."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            ErpProtocol().expected_slot = True        # type: ignore[misc]

    def test_describe_names_every_active_choice(self):
        d = ErpProtocol(expected_slot=True, afferent_energy=True).describe()
        assert "expected_slot" in d and "afferent_energy" in d

    def test_engine_identity_is_part_of_a_declared_protocol(self):
        p = ErpProtocol(engine_name="numpy_exact")
        assert p.describe().endswith(";engine=numpy_exact")
        _check_engine_identity(SimpleNamespace(engine_name="numpy_exact"), p)
        with pytest.raises(ValueError, match="engine"):
            _check_engine_identity(SimpleNamespace(engine_name="numpy_sparse"), p)

    def test_calibration_rejects_a_mismatched_engine_before_sampling(self):
        with pytest.raises(ValueError, match="requires engine"):
            calibrate_erp_thresholds(
                SimpleNamespace(engine_name="numpy_sparse"),
                ensure_prediction=False,
                protocol=ErpProtocol(engine_name="numpy_exact"),
            )


class TestTheOneEnvironmentAdapter:

    def test_reads_the_legacy_variables(self):
        p = ErpProtocol.from_environment(
            {"ERP_EXPECTED_SLOT": "1", "ERP_AFFERENT_ENERGY": "on"})
        assert p.expected_slot and p.afferent_energy

    def test_absent_variables_give_the_shipped_default(self):
        assert ErpProtocol.from_environment({}) == DEFAULT_PROTOCOL

    @pytest.mark.parametrize("raw", ["0", "false", "off", "", "  ", "no"])
    def test_falsey_spellings_do_not_enable(self, raw):
        """`ERP_EXPECTED_SLOT=0` must DISABLE.

        An earlier version of this flag defaulted to on and treated anything
        other than "0"/"false"/"off" as enabled, so the two spellings of the
        same intent behaved differently. Pin the whole set.
        """
        assert not ErpProtocol.from_environment({"ERP_EXPECTED_SLOT": raw}).expected_slot

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "on", "Yes"])
    def test_truthy_spellings_enable(self, raw):
        assert ErpProtocol.from_environment({"ERP_EXPECTED_SLOT": raw}).expected_slot

    def test_adapter_does_not_read_the_real_environment_when_given_one(self):
        """Injectable so a test never depends on ambient process state.

        A config reader that can only read `os.environ` forces tests to mutate
        it -- reintroducing exactly the global-state problem being removed.
        """
        p = ErpProtocol.from_environment({"ERP_EXPECTED_SLOT": "1"})
        assert p.expected_slot is True
