"""The memory-study library: what every refraction-memory registration needs, written once.

WHY. By Amendment 55 the memory studies had re-implemented the same pieces in each module -- the
cell and its recovery time, the store and sleep loops, the "cells and brains are new" checks, the
bars -- fourteen copies of the store loop alone, and every fix had to find all of them. A new
registration is now a SPECIFICATION over these pieces.

The VOCABULARY every study shares -- the model's constants, the laws that size a study, how a
memory is read -- had grown up inside particular studies: forty studies took the weight ceiling
from Amendment 9's module, thirty-four the learning rate from Amendment 17's, and the refraction
strength was defined three times. It is owned here, and a study reads it as lib.NAME:

    model     W_MAX, STRENGTH; profile(beta) and profile_name(beta), the semantics a run records
    laws      theta(n, k, p), the learning-rate scale; above_floor, the regime; unit, the load unit
    readout   ROUNDS, RECALL_SAMPLE, MEASUREMENT_SEED, sample_for; the criteria HALF_BAR,
              COMPLETE and MATCH; overlap(a, b)
    walks     LENGTH and RHO of the reuse studies' sequences
    seeding   seeds_for and to_i32: brain seeds to device seeds

The registered modules (Amendments 9-55) re-export the names they used to define, and compute
exactly what they ran: the move was proved name by name -- every function's body, with each global
it reads resolved to the object it reaches, unchanged -- and checked by replaying recorded runs
(research.replay).

The pieces a registration is built from:

    cells     a Cell (n, k, p) with its recovery time by Amendment 41's rule, its learning rate
              theta(n, k, p), and the regime check k p >= 3 ln n
    ledger    every registered memory study's cells and brains, read from the modules themselves;
              the rule since Amendment 37 (new cells, new brains) checked in one place
    bars      a bar as DATA: what it reads, how it compares, and how it is judged -- by the
              CONFIDENCE BOUND of an ensemble over brains (neural_assemblies.diagnostics), not by
              a bare mean -- and the text the registration states, generated from the same object
    spec      a Registration: the text it registers, its bars, its result section and scorecard
              rows, rendered from one object (research/amend.py writes them into the documents)
    stores    a Plan (the sequences a study writes) and a Store (the memory written, on the fast
              paths): replay, sleep, downscaling; the two registered sleep gates

The fast paths (replay, sleep, writing, packed counts) are research.experiments.memory_fast.
"""
from __future__ import annotations

from .model import W_MAX, STRENGTH, profile, profile_name
from .laws import theta, above_floor, unit
from .readout import (ROUNDS, RECALL_SAMPLE, MEASUREMENT_SEED, sample_for, HALF_BAR, COMPLETE, MATCH,
                      overlap)
from .walks import LENGTH, RHO
from .seeding import seeds_for, to_i32
from .cells import Cell
from .ledger import Entry, entries, check_new, next_brains, PROBE_BRAINS, NEW_FROM
from .bars import Bar, Check, Reading, brains, delta, below, scalar, every, evaluate
from .spec import Registration
from .stores import Plan, Store, birth_setpoint, reference_median

__all__ = ["Cell", "Entry", "entries", "check_new", "next_brains", "PROBE_BRAINS", "NEW_FROM",
           "Bar", "Check", "Reading", "brains", "delta", "below", "scalar", "every", "evaluate", "Registration",
           "Plan", "Store", "birth_setpoint", "reference_median"]
