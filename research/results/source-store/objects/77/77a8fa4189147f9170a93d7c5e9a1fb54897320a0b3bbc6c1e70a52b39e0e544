"""
NEMO Archive
============

ARCHIVED FILES - DO NOT USE IN PRODUCTION

These files are kept for reference but have been superseded by
the modular core/ and language/ modules.

Archived files:
- brain_v1.py: Original brain/area implementation
- kernels_v1.py: Original projection kernels
- hierarchical_v1.py: Original hierarchical brain (hardcoded grammar)
- hierarchical_fast_v1.py: Speed-optimized version (hardcoded grammar)
- hierarchical_full_v1.py: Full 26-area version (hardcoded grammar)
- scaling_study_v1.py: Performance scaling analysis
- linguistic_extensions_v1.py: Feature-based selectional restrictions
- test_generation_v1.py: Tests for old system

This list is exhaustive, and worth keeping that way: it previously omitted
three of the files actually present, one of which (emergent_learner_v1.py) was
a BYTE-IDENTICAL copy of nemo/language/emergent_learner.py rather than an
earlier version of anything. An index that does not match the directory is how
a duplicate hides in an archive.

That note called the copied-from file "live". It was not: commit 84bda12 split
it into the nemo/language/emergent/ package and left the original in place,
imported by nothing, for months. It was deleted on 2026-08-05 -- the third copy
of the same class to be found this way. Two of the three were found by looking
for an index that disagreed with the directory; the third by asking who
imports it. Ask both.

Do NOT bulk-delete this directory. hierarchical_full_v1.py is cited evidence in
an OPEN audit item -- docs/claim_audit.md asks whether any reported six-order
generation result came from it, and that cannot be answered once it is gone.

Why archived:
- Grammar was hardcoded (SVO/SOV explicit)
- Selectional restrictions were feature-based, not learned
- Too much code duplication
- Not scientifically valuable (didn't test emergent learning)

Use instead:
- nemo.core: Minimal brain/area/kernel components
- nemo.language: Learner and generator that learn from data
"""

