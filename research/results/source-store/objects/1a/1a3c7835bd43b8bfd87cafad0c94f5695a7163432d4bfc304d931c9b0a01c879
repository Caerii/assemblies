"""The hierarchy study's bars (PREREG_refraction_memory.md Amendment 33), on
synthetic records: a hierarchy that replays every plan at light load, needs
its links, keeps plans sharing a run apart, and breaks first at the links in
a way a larger pair of areas relieves, must pass; a heavy-load failure at the
chunk area rather than the links fails H4."""
import pytest

from research.experiments import memory_hierarchy as mh

pytestmark = pytest.mark.requires_torch

SEEDS = list(range(20))


def _ens(v):
    return {"keys": SEEDS, "values": [v] * 20, "mean": v}


def _obs(*, heavy_starts=(0.7, 0.9), heavy_chain=0.96, light=1.0):
    cells = {}
    for ci, (ns, nc) in enumerate(mh.CELLS):
        links = {}
        for a, b in mh.LOADS:
            for link in mh.LINKS:
                if link == 0:
                    w, st, ch = 0.0, 0.0, 1.0
                elif (a, b) == mh.LOADS[0]:
                    w, st, ch = light, 1.0, 1.0
                elif (a, b) == mh.LOADS[2]:
                    w, st, ch = 0.3, heavy_starts[ci], heavy_chain
                else:
                    w, st, ch = 0.95, 1.0, 1.0
                links[f"{a}/{b}/{link}"] = {str(i): {"order": [0], "whole": _ens(w), "starts": _ens(st),
                                                     "chain": _ens(ch)} for i in range(8)}
        cells[f"{ns}/{nc}"] = {"n_s": ns, "n_c": nc, "links": links}
    return {"cells": cells}


def test_a_hierarchy_that_breaks_at_its_links_passes():
    bars = mh.evaluate(_obs())["bars"]
    assert all(bars.values()), bars


def test_a_light_load_failure_fails_h1():
    assert not mh.evaluate(_obs(light=0.5))["bars"]["H1"]


def test_a_break_at_the_chunk_area_fails_h4():
    assert not mh.evaluate(_obs(heavy_starts=(0.95, 0.97), heavy_chain=0.5))["bars"]["H4"]


def test_the_shared_run_plans_share_their_second_and_third_chunks():
    a, b = mh.plans(8, 5, 6)[-2:]
    assert a[1:3] == b[1:3] and a[0] != b[0] and a[3:] != b[3:]
