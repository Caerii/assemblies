"""Parsing the same sentence twice does not give the same answer.

#100 / #80. Pinned as a KNOWN DEFECT, not asserted away, because the fix is a
change to what parsing MEANS and has not been measured yet.

MEASURED. One 5-word sentence of KNOWN words, on an ALREADY-TRAINED parser,
parsed four times::

    after pass0  materialized +663   connectome CHANGED
    after pass1  materialized + 67   connectome CHANGED
    after pass2  materialized + 26   connectome CHANGED
    after pass3  materialized + 10   connectome CHANGED

with the first-word P600 reading 0.434, 0.988, 0.9868, 0.9865 -- a 2x jump
between the first read and the second, then a drift that never converges.

So every reported ERP number depends on how many times, and in what order,
sentences were parsed before it. This is NOT the probe-contamination problem
(#100's first half): isolating the ERP probes leaves it untouched, because the
recruitment happens in `_advance_incremental_word`, inside the outer `frozen()`
that `runner.py` wraps the whole parse in.

THE FIX IS KNOWN AND MEASURED, and deliberately not applied here. Running the
parse advance under `read_only()` makes it idempotent from the second read
(+0 growth, bit-identical probes). But it would mean a word with no existing
assembly can no longer form one mid-sentence, and whether the
grammatical/violation separation survives that is unmeasured. See
research/notes/erp_probe_isolation.md.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

from neural_assemblies.assembly_calculus.emergent.evaluation.erp.runner import (
    run_incremental_erp_probes,
)

SENT = ["the", "cat", "chase", "the", "mouse"]


def _materialized(brain):
    total = 0
    for name in brain.areas:
        eng = brain._engine_for(brain.areas[name])
        total += int(eng.materialized_count(name) or 0)
    return total


def _probe(parser):
    out = run_incremental_erp_probes(parser, SENT, apply_calibration=False)
    probes = out[1] if isinstance(out, tuple) else out
    return [(round(float(p.n400), 6), round(float(p.p600), 6)) for p in probes]


class TestParseIsNotIdempotent:

    def test_a_parse_grows_a_trained_parser(self, forked_parser):
        """The mechanism, pinned directly."""
        parser = forked_parser("SENTENCES", seed=42)
        before = _materialized(parser.brain)
        _probe(parser)
        assert _materialized(parser.brain) > before, (
            "a parse no longer recruits -- if this is intended, evaluation may "
            "have become idempotent and the xfail below should now XPASS")

    @pytest.mark.xfail(strict=True, reason=(
        "KNOWN DEFECT, pinned deliberately: parsing recruits, so repeated "
        "parses of the SAME sentence return different ERP values (first-word "
        "P600 0.434 -> 0.988 -> 0.9868 -> 0.9865). Every ERP number therefore "
        "depends on read order. Fix is measured but unadopted -- see "
        "research/notes/erp_probe_isolation.md"))
    def test_repeated_parses_agree(self, forked_parser):
        parser = forked_parser("SENTENCES", seed=42)
        runs = [_probe(parser) for _ in range(3)]
        assert runs[0] == runs[1] == runs[2]

    def test_a_fully_read_only_parse_IS_idempotent(self, forked_parser):
        """The fix works -- which is why the defect above is a choice, not a
        limitation. An outer read_only() subsumes runner.py's inner frozen(),
        so this needs no production edit to demonstrate.

        The FIRST pass is excluded: `read_only()` deliberately exempts areas
        still below `k` materialized neurons, since there is nothing there to
        select from, so one warm-up pass can still grow.
        """
        parser = forked_parser("SENTENCES", seed=42)
        with parser.brain.read_only():
            _probe(parser)                       # warm-up, may still grow
            before = _materialized(parser.brain)
            a, b = _probe(parser), _probe(parser)
        assert a == b, f"read-only parses still disagree:\n{a}\n{b}"
        assert _materialized(parser.brain) == before, (
            "a read-only parse still recruited")


class TestForkIsIsolatedFromTheSharedParser:
    """#103. `fork()` must clone a PRISTINE snapshot, not the live cache entry.

    THE DEFECT. `ParserCache.get()` hands out a shared parser and its contract
    said that was "only safe to read". Reading is not safe -- parsing recruits
    (see above). So a test that merely PARSED through `sentences_parser` grew
    the shared object, and every later `fork()` cloned the grown version.

    MEASURED consequence: seed 42 read Cohen's d = -0.26, the separation
    INVERTED, against +1.80 from a pristine parser; and two suite tests passed
    or failed depending on which other tests had run first.
    """

    def test_abusing_the_shared_parser_does_not_move_later_forks(self):
        from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (
            get_parser_cache,
        )
        cache = get_parser_cache()
        before = _materialized(cache.fork("SENTENCES", seed=42).brain)
        shared = cache.get("SENTENCES", seed=42)
        for _ in range(3):
            _probe(shared)
        assert _materialized(shared.brain) > before, (
            "the shared parser did not grow -- this test cannot detect the "
            "leak it exists for; check that parsing still recruits")
        after = _materialized(cache.fork("SENTENCES", seed=42).brain)
        assert after == before, (
            f"fork inherited {after - before} neurons of growth from the "
            f"shared cache entry -- ParserCacheEntry.pristine is not being "
            f"used, so evaluation order leaks between tests again")
