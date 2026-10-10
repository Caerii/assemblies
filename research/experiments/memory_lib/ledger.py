"""The ledger of registered memory studies: which cells and which brains each one used.

Read from the registered modules themselves -- every one names its amendment in its docstring
("Registered in PREREG_refraction_memory.md, Amendment N") and keeps CELLS, SEEDS and, for the
sleep studies, REFERENCE_SEEDS -- so it cannot drift from what ran. It replaces the hand-kept
unions of earlier cells that each registration's test used to carry.

THE RULE, since Amendment 37: a registration runs at cells no earlier registration used, on
brains (seeds, reference seeds included) no earlier one used, outside the range kept for smokes
and exploratory probes (900-999). Before 37 the studies reused cells by design -- ladders of n
and k over shared cells -- and the rule does not apply to them.
"""
from __future__ import annotations

import importlib
import pathlib
import re
from dataclasses import dataclass

#: the first amendment held to the new-cells, new-brains rule
NEW_FROM = 37
#: brains kept for VOID smokes (900-902) and exploratory probes (960-999): never registered
PROBE_BRAINS = range(900, 1000)

_HERE = pathlib.Path(__file__).resolve().parent.parent
_AMENDMENT = re.compile(r"Amendments? (\d+)")


@dataclass(frozen=True)
class Entry:
    amendment: int
    module: str
    cells: tuple            # (n, k, p) per cell; p None where a module kept (n, k) only
    seeds: tuple
    reference_seeds: tuple

    @property
    def brains(self):
        return set(self.seeds) | set(self.reference_seeds)


def _cells(raw):
    out = []
    for c in raw or ():
        c = tuple(c)
        out.append((int(c[0]), int(c[1]), float(c[2]) if len(c) > 2 else None))
    return tuple(out)


def entries():
    """every registered memory module with its amendment, cells and brains, in amendment order"""
    found = []
    for path in sorted(_HERE.glob("memory_*.py")):
        head = path.read_text(encoding="utf-8")[:1500]
        m = _AMENDMENT.search(head)
        if not m:
            continue
        mod = importlib.import_module(f"research.experiments.{path.stem}")
        cells = getattr(mod, "CELLS", None)
        if cells is None:
            continue
        found.append(Entry(int(m.group(1)), path.stem, _cells(cells),
                           tuple(getattr(mod, "SEEDS", ()) or ()),
                           tuple(getattr(mod, "REFERENCE_SEEDS", ()) or ())))
    return sorted(found, key=lambda e: (e.amendment, e.module))


def used_before(amendment, ledger=None):
    """(cells, brains) of every registration before ``amendment``"""
    cells, brains = set(), set()
    for e in ledger if ledger is not None else entries():
        if e.amendment < amendment:
            cells |= {c[:2] + ((c[2],) if c[2] is not None else ()) for c in e.cells}
            brains |= e.brains
    return cells, brains


def check_new(cells, seeds, reference_seeds=(), amendment=None, ledger=None):
    """the problems with a registration's cells and brains under the rule (empty: none).
    ``cells`` (n, k, p[, tau]) tuples or Cells; ``amendment`` None means after every one so far."""
    ledger = entries() if ledger is None else ledger
    if amendment is None:
        amendment = max(e.amendment for e in ledger) + 1
    used_cells, used_brains = used_before(amendment, ledger)
    problems = []
    for c in cells:
        t = c.as_tuple() if hasattr(c, "as_tuple") else tuple(c)
        if (t[0], t[1], float(t[2])) in used_cells:
            problems.append(f"cell {t[:3]} was used before Amendment {amendment}")
    brains = set(seeds) | set(reference_seeds)
    if set(seeds) & set(reference_seeds):
        problems.append("subject and reference brains overlap")
    reused = sorted(brains & used_brains)
    if reused:
        problems.append(f"brains {reused[:5]}{'...' if len(reused) > 5 else ''} were used before")
    probe = sorted(b for b in brains if b in PROBE_BRAINS)
    if probe:
        problems.append(f"brains {probe[:5]} are kept for smokes and probes (900-999)")
    return problems


def next_brains(count, ledger=None):
    """the next ``count`` brains no registration has used, above every one so far and outside
    the probe range"""
    ledger = entries() if ledger is None else ledger
    start = max(max(e.brains) for e in ledger if e.brains) + 1
    if start in PROBE_BRAINS or start + count - 1 in PROBE_BRAINS:
        start = max(start, PROBE_BRAINS.stop)
    return tuple(range(start, start + count))
