"""Every result citation in the codebase must resolve, and extensions must say so.

A citation register is only worth having if it cannot drift from the code that
cites it. These are the two ways it would rot: a docstring citing an ID nobody
defined, and an EXTENSION quietly losing the caveat that makes it readable as
an extension rather than as an established fact.
"""

from __future__ import annotations

import os
import unittest

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


if __name__ == "__main__":
    unittest.main()
