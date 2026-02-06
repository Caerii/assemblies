"""
NEMO Archive
============

ARCHIVED FILES - DO NOT USE IN PRODUCTION

These files are kept for reference but have been superseded by
the modular core/ and language/ modules.

Archived files:
- hierarchical_v1.py: Original hierarchical brain (hardcoded grammar)
- hierarchical_fast_v1.py: Speed-optimized version (hardcoded grammar)  
- hierarchical_full_v1.py: Full 26-area version (hardcoded grammar)
- scaling_study_v1.py: Performance scaling analysis
- linguistic_extensions_v1.py: Feature-based selectional restrictions
- test_generation_v1.py: Tests for old system

Why archived:
- Grammar was hardcoded (SVO/SOV explicit)
- Selectional restrictions were feature-based, not learned
- Too much code duplication
- Not scientifically valuable (didn't test emergent learning)

Use instead:
- nemo.core: Minimal brain/area/kernel components
- nemo.language: Learner and generator that learn from data
"""

