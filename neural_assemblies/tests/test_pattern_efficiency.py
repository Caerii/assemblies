"""The pattern-efficiency study's instruments, on the CPU.

PREREG_refraction_memory.md Amendment 9 judges the refracted memory against
idealised memories on the same circuit. Its bars read two statistics -- the
potentiated fraction and PAIR SHARING -- and a rank-1 recall. Each must read
what it claims before a GPU run is spent on it: pair sharing must read ~1x its
expectation on independent patterns (PE-0's calibration), the count builder
must equal a brute-force count, balanced allocation must actually balance, and
the bar evaluator must FAIL when the real memory is replaced by the random one.
"""
import itertools

import pytest

from research.experiments import memory_pattern_efficiency as pe

pytestmark = pytest.mark.requires_torch


def _torch():
    import torch
    return torch


def test_counts_equal_brute_force():
    torch = _torch()
    pats = torch.tensor([[0, 1, 2], [1, 2, 3], [0, 3, 4]])
    N = pe.pattern_counts(pats, 5)
    brute = torch.zeros(5, 5)
    for row in pats.tolist():
        for i, j in itertools.product(row, row):
            brute[i, j] += 1
    assert torch.equal(N, brute)


def test_pair_sharing_reads_one_on_independent_patterns():
    torch = _torch()
    n, k, M = 400, 12, 600
    pats = pe.ideal_patterns(n, k, M, 1, False, 11, "cpu")[:, 0]
    present = torch.ones(n, n, dtype=torch.bool)
    _, share = pe.pattern_stats(pe.pattern_counts(pats, n), present, M, k)
    assert 0.9 <= share / pe.pair_sharing_expected(n, k, M) <= 1.1


def test_pair_sharing_sees_reuse():
    """Items built from a few recurring pairs share far more than chance."""
    torch = _torch()
    n, k, M = 400, 12, 200
    base = torch.arange(k)
    pats = torch.stack([torch.cat([base[:6], torch.randperm(n - k)[:6] + k]) for _ in range(M)])
    present = torch.ones(n, n, dtype=torch.bool)
    _, share = pe.pattern_stats(pe.pattern_counts(pats, n), present, M, k)
    assert share / pe.pair_sharing_expected(n, k, M) > 5


def test_balanced_allocation_balances():
    torch = _torch()
    n, k, M = 300, 10, 90
    pats = pe.ideal_patterns(n, k, M, 2, True, 3, "cpu")
    for b in range(2):
        use = torch.bincount(pats[:, b].reshape(-1), minlength=n)
        assert int(use.max()) - int(use.min()) <= 1


def test_disjoint_memory_recalls_every_item():
    torch = _torch()
    n, k, M = 200, 10, 20
    pats = torch.arange(n).view(M, k)
    present = torch.ones(n, n, dtype=torch.bool)
    tab = torch.tensor([1.0] + [1.1 ** c for c in range(1, 128)]).clamp(max=20.0)
    W = pe.weights((pe.pattern_counts(pats, n) * 5), present, torch.ones(n), tab)
    hit, own = pe.rank1(W, pats, list(range(M)), k)
    assert hit == 1.0 and own == 1.0


def test_windows_contain_the_anchors_and_ceilings_have_room():
    for nk, anchor in pe.ANCHORS.items():
        grids = pe.plan([nk])[0]["grids"]
        assert min(grids["real"]) <= anchor <= max(grids["real"])
        assert max(grids["random"]) >= 4 * anchor
        assert max(grids["balanced"]) >= 8 * anchor


def _variant(points, share_ratio, pot, n, k):
    ensembles, expected = {}, {}
    for M, r in points:
        exp = pe.pair_sharing_expected(n, k, M)
        ensembles[str(M)] = {"rank1": {"mean": r}, "own": {"mean": r},
                             "potentiated": {"mean": pot},
                             "pair_sharing": {"mean": share_ratio * exp}}
        expected[str(M)] = exp
    curve = [(M, r) for M, r in points]
    return {"ensembles": ensembles, "ceiling": pe._ceiling(curve),
            "pair_sharing_expected": expected}


def _observations(real_scale):
    cells = {}
    for n, k in pe.IN_REGIME:
        a = (n / k) ** 2
        grid = [int(a * f) for f in (0.2, 0.3, 0.45, 0.7, 1.0, 1.5, 2.2, 3.3)]

        def curve(m_star):
            return [(M, 1.0 if M < m_star else 0.0) for M in grid]
        real = 0.7 * a * real_scale
        variants = {
            "real": _variant(curve(real), 2.0 if real_scale < 1 else 1.0, 0.5, n, k),
            "clean": _variant(curve(real * 0.8), 2.0, 0.3, n, k),
            "random": _variant(curve(0.7 * a), 1.0, 0.5, n, k),
            "random_cm2": _variant(curve(0.7 * a), 1.0, 0.5, n, k),
            "random_cp2": _variant(curve(0.7 * a), 1.0, 0.5, n, k),
            "balanced": _variant(curve(2.2 * a), 0.9, 0.7, n, k),
        }
        readings = {"real": {str(M): {"rank1_module": [e["rank1"]["mean"]] * 3}
                             for M, e in variants["real"]["ensembles"].items()}}
        cells[f"{n}/{k}"] = {"n": n, "k": k, "variants": variants, "readings": readings}
    return {"cells": cells}


def test_evaluator_judges_a_constructed_pass_and_refuses_the_substitution():
    anchors = {nk: None for nk in pe.IN_REGIME}
    obs = _observations(0.5)
    for key, cell in obs["cells"].items():
        anchors[(cell["n"], cell["k"])] = cell["variants"]["real"]["ceiling"]["m_star"]
    passing = pe.evaluate(obs, anchors=anchors)["bars"]
    assert passing["PE-1"] and passing["PE-2"] and passing["PE-4"] and passing["PE-5"]
    # TRUE NEGATIVE: the random memory substituted for the real one
    swapped = pe.evaluate(_observations(1.0), anchors={nk: 0.7 * (nk[0] / nk[1]) ** 2
                                                       for nk in pe.IN_REGIME})["bars"]
    assert not swapped["PE-2"] and not swapped["PE-3"]
