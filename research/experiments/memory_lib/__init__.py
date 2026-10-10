"""The memory-study library: what every refraction-memory registration needs, written once.

WHY. By Amendment 55 the memory studies had re-implemented the same pieces in each module -- the
cell and its recovery time, the store and sleep loops, the "cells and brains are new" checks, the
bars -- fourteen copies of the store loop alone, and every fix had to find all of them. A new
registration is now a SPECIFICATION over these pieces; the registered modules (Amendments 9-55)
stay exactly as they ran, the record of what was run.

    cells     a Cell (n, k, p) with its recovery time by Amendment 41's rule, its learning rate
              theta(n, k, p), and the regime check k p >= 3 ln n
    ledger    every registered memory study's cells and brains, read from the modules themselves;
              the rule since Amendment 37 (new cells, new brains) checked in one place
    bars      a bar as DATA: what it reads, how it compares, and how it is judged -- by the
              CONFIDENCE BOUND of an ensemble over brains (neural_assemblies.diagnostics), not by
              a bare mean -- and the text the registration states, generated from the same object
    spec      a Registration: the text it registers, its bars, its result section and scorecard
              rows, rendered from one object (research/amend.py writes them into the documents)

The fast paths (replay, sleep, writing, packed counts) are research.experiments.memory_fast.
"""
from __future__ import annotations

from .cells import Cell
from .ledger import Entry, entries, check_new, next_brains, PROBE_BRAINS, NEW_FROM
from .bars import Bar, Check, Reading, brains, delta, below, scalar, every, evaluate
from .spec import Registration

__all__ = ["Cell", "Entry", "entries", "check_new", "next_brains", "PROBE_BRAINS", "NEW_FROM",
           "Bar", "Check", "Reading", "brains", "delta", "below", "scalar", "every", "evaluate", "Registration"]
