"""`ErpProtocol` must make an arm a VALUE, not a moment in time.

The defect it replaces: `ERP_EXPECTED_SLOT` is read three call levels below any
caller, so experiments A/B by mutating `os.environ` and restoring it. That makes
"which arm produced this number" a property of WHEN you looked. These tests pin
the properties that fix it -- derivation is an expression, the value is frozen,
and reading the environment happens in exactly one place.
"""
from __future__ import annotations

import dataclasses

import pytest

from neural_assemblies.assembly_calculus.emergent.evaluation.erp.protocol import (
    DEFAULT_PROTOCOL,
    ErpProtocol,
)


class TestTheShippedDefault:

    def test_both_measurement_flags_are_off(self):
        """`expected_slot` inverts on fresh parsers; `afferent_energy` was
        rejected on measurement (AUC 0.000, zero variance, four seeds)."""
        assert DEFAULT_PROTOCOL.expected_slot is False
        assert DEFAULT_PROTOCOL.afferent_energy is False

    def test_default_describes_itself_as_default_not_as_nothing(self):
        """A study that recorded no deviations must not look like one that
        recorded nothing at all."""
        assert DEFAULT_PROTOCOL.describe() == "default"


class TestAnArmIsAValue:

    def test_deriving_an_arm_is_an_expression(self):
        base = ErpProtocol()
        arm = base.with_(expected_slot=True)
        assert arm.expected_slot is True
        assert base.expected_slot is False, "deriving must not mutate the base"

    def test_the_protocol_is_frozen(self):
        """A mutable protocol is the environment variable with extra steps."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            ErpProtocol().expected_slot = True        # type: ignore[misc]

    def test_describe_names_every_active_choice(self):
        d = ErpProtocol(expected_slot=True, afferent_energy=True).describe()
        assert "expected_slot" in d and "afferent_energy" in d


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
