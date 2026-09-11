"""Every result citation in the codebase must resolve, and extensions must say so.

A citation register is only worth having if it cannot drift from the code that
cites it. These are the two ways it would rot: a docstring citing an ID nobody
defined, and an EXTENSION quietly losing the caveat that makes it readable as
an extension rather than as an established fact.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace

from neural_assemblies import theory


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)


class TestTheoryCitations(unittest.TestCase):
    def test_every_citation_resolves(self):
        missing = theory.unresolved_citations(ROOT)
        self.assertEqual(
            missing, {},
            "docstrings cite results that are not in neural_assemblies/theory.py; "
            "add the result rather than removing the citation")

    def test_research_notes_citations_resolve(self):
        """Notes and experiments cite the same register as the library."""
        research = os.path.join(REPO, "research")
        if not os.path.isdir(research):
            self.skipTest("no research tree")
        self.assertEqual(theory.unresolved_citations(research), {})

    def test_cite_rejects_unknown_ids(self):
        with self.assertRaises(KeyError):
            theory.cite("NOT-A-RESULT")

    def test_every_extension_carries_a_caveat(self):
        """An extension without a caveat reads as established fact.

        The whole point of the status is to make the unproven step visible at
        the point of use, which a bare claim does not do.
        """
        for result in theory.extensions():
            self.assertTrue(
                result.caveat.strip(),
                f"{result.id} is an EXTENSION with no caveat saying what is "
                f"unproven or what would falsify it")

    def test_measured_results_name_their_evidence(self):
        """A MEASURED claim must point at the experiment that measured it."""
        for result in theory.RESULTS.values():
            if result.status == theory.Status.MEASURED:
                self.assertTrue(
                    result.evidence,
                    f"{result.id} is MEASURED but names no evidence")

    def test_cross_references_between_results_resolve(self):
        """Results citing other results must resolve too."""
        for result in theory.RESULTS.values():
            text = " ".join([result.claim, result.caveat,
                             *result.preconditions, *result.evidence])
            for rid in theory.CITATION.findall(text):
                self.assertIn(rid, theory.RESULTS,
                              f"{result.id} cites unknown result {rid}")

    def test_measured_results_state_engine_or_explicit_provenance_gap(self):
        for result in theory.RESULTS.values():
            if result.status == theory.Status.MEASURED:
                self.assertTrue(result.engine.strip(), f"{result.id} has no engine provenance")

    def test_typed_evidence_references_resolve(self):
        errors = theory.evidence_reference_errors(REPO)
        self.assertEqual(errors, [], "\n".join(errors))

    def test_typed_evidence_rejects_dangling_and_unsafe_paths(self):
        original = theory._RESULTS
        try:
            theory._RESULTS = [theory.Result(
                id="BROKEN-EVIDENCE", status=theory.Status.MEASURED,
                claim="fixture", source="fixture", evidence=("fixture",), engine="fixture",
                evidence_refs=(theory.EvidenceRef("../outside.json", "artifact"),))]
            errors = theory.evidence_reference_errors(REPO)
            self.assertTrue(any("unsafe evidence path" in error for error in errors))
            theory._RESULTS = [theory.Result(
                id="BROKEN-EVIDENCE", status=theory.Status.MEASURED,
                claim="fixture", source="fixture", evidence=("fixture",), engine="fixture",
                evidence_refs=(theory.EvidenceRef("research/results/missing.json", "artifact"),))]
            errors = theory.evidence_reference_errors(REPO)
            self.assertTrue(any("dangling evidence path" in error for error in errors))
        finally:
            theory._RESULTS = original

    def test_retained_sensitivity_accepts_a_moving_control(self):
        original = theory._RESULTS
        try:
            with tempfile.TemporaryDirectory() as root:
                artifact = os.path.join(root, "results.json")
                with open(artifact, "w", encoding="utf-8") as handle:
                    json.dump({"seeds": [1, 2, 3],
                               "treatment": [0.7, 0.8, 0.9],
                               "control": [0.1, 0.2, 0.3]}, handle)
                theory._RESULTS = [theory.Result(
                    id="MOVING-PROBE", status=theory.Status.MEASURED,
                    claim="fixture", source="fixture", evidence=("fixture",),
                    engine="fixture",
                    evidence_refs=(theory.EvidenceRef("results.json", "artifact"),),
                    sensitivity_checks=(theory.SensitivityCheck(
                        artifact="results.json", sample_path="seeds",
                        treatment_path="treatment",
                        control_path="control", relation="all-greater",
                        minimum_effect=0.5, mechanism="constructed control"),))]
                self.assertEqual(theory.evidence_reference_errors(root), [])
        finally:
            theory._RESULTS = original

    def test_retained_sensitivity_rejects_a_dead_probe(self):
        original = theory._RESULTS
        try:
            with tempfile.TemporaryDirectory() as root:
                artifact = os.path.join(root, "results.json")
                with open(artifact, "w", encoding="utf-8") as handle:
                    json.dump({"seeds": [1, 2, 3],
                               "treatment": [1.0, 1.0, 1.0],
                               "control": [1.0, 1.0, 1.0]}, handle)
                theory._RESULTS = [theory.Result(
                    id="DEAD-PROBE", status=theory.Status.MEASURED,
                    claim="fixture", source="fixture", evidence=("fixture",),
                    engine="fixture",
                    evidence_refs=(theory.EvidenceRef("results.json", "artifact"),),
                    sensitivity_checks=(theory.SensitivityCheck(
                        artifact="results.json", sample_path="seeds",
                        treatment_path="treatment",
                        control_path="control", relation="all-different",
                        minimum_effect=0.01, mechanism="constructed null"),))]
                errors = theory.evidence_reference_errors(root)
                self.assertTrue(any("minimum retained effect is 0" in error
                                    for error in errors), errors)
        finally:
            theory._RESULTS = original

    def test_retained_sensitivity_rejects_duplicate_sample_identity(self):
        original = theory._RESULTS
        try:
            with tempfile.TemporaryDirectory() as root:
                artifact = os.path.join(root, "results.json")
                with open(artifact, "w", encoding="utf-8") as handle:
                    json.dump({"seeds": [1, 1, 2], "treatment": [3, 4, 5],
                               "control": [0, 0, 0]}, handle)
                theory._RESULTS = [theory.Result(
                    id="DUPLICATE-SEED", status=theory.Status.MEASURED,
                    claim="fixture", source="fixture", evidence=("fixture",),
                    engine="fixture",
                    evidence_refs=(theory.EvidenceRef("results.json", "artifact"),),
                    sensitivity_checks=(theory.SensitivityCheck(
                        artifact="results.json", sample_path="seeds",
                        treatment_path="treatment", control_path="control",
                        relation="all-greater", minimum_effect=1,
                        mechanism="constructed control"),))]
                errors = theory.evidence_reference_errors(root)
                self.assertTrue(any("sample identities are empty or duplicated"
                                    in error for error in errors), errors)
        finally:
            theory._RESULTS = original

    def test_sensitivity_paths_use_rfc6901_escaped_object_keys(self):
        document = {"cells": {"B/100/10": {"rank1": [0.7, 0.8, 0.9]}}}
        self.assertEqual(
            theory._json_values(document, "cells/B~1100~110/rank1"),
            [0.7, 0.8, 0.9],
        )
        with self.assertRaisesRegex(ValueError, "RFC 6901"):
            theory._json_values(document, "cells/B~2100/rank1")

    def test_real_measured_sensitivity_checks_remain_active(self):
        checked = {result.id for result in theory.RESULTS.values()
                   if result.sensitivity_checks}
        self.assertEqual(checked, {
            "RATE-HETEROGENEITY", "REFRACTION-ANTI-MERGING",
            "SEQ-TEMPORAL-CARRY",
        })

    def test_refraction_sensitivity_fails_its_constructed_true_negative(self):
        result = theory.cite("REFRACTION-ANTI-MERGING")
        check = result.sensitivity_checks[0]
        dead = replace(check, treatment_path=check.control_path)
        errors = theory._sensitivity_errors(result, dead, REPO)
        self.assertTrue(any("minimum retained effect is 0" in error
                            for error in errors), errors)


if __name__ == "__main__":
    unittest.main()


def test_register_rendering_is_current():
    """docs/register.md is the rendering of theory.py; regenerate it with
    `python -m neural_assemblies.theory --render > docs/register.md`."""
    import os
    from neural_assemblies import theory
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(here, "docs", "register.md")
    with open(path, encoding="utf-8") as fh:
        on_disk = fh.read().replace("\r\n", "\n")
    assert on_disk == theory.render_markdown(), (
        "docs/register.md is stale: run `python -m neural_assemblies.theory "
        "--render > docs/register.md`")
