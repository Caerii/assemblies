"""The theory register's claims, by programme, in the register's order (neural_assemblies/theory.py
concatenates them; docs/register.md renders them). A new claim goes into its programme's module
(research.amend.add_claim finds the place)."""
from __future__ import annotations

from . import sequences, sequence_organ, refraction, sequence_memory, capacity, extensions, algebra

ALL = [*sequences.CLAIMS, *sequence_organ.CLAIMS, *refraction.CLAIMS, *sequence_memory.CLAIMS, *capacity.CLAIMS, *extensions.CLAIMS, *algebra.CLAIMS]
