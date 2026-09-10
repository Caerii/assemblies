# Claims

Use this directory for statements the research tree can defend.

> **Status (2026-09-09).** The register in `neural_assemblies/theory.py`
> ([../../docs/register.md](../../docs/register.md)) is the claim layer for
> everything measured since mid-2026; new claims go there with a
> registration behind them. Of the six evidence summaries indexed below
> (February 2026), the Phase 3 audit in
> [../plans/SOUNDNESS_PROGRAM.md](../plans/SOUNDNESS_PROGRAM.md) voided the
> reactivation and retrieval protocols as dead probes; treat the remaining
> summaries as leads, not evidence, until re-measured under a registered bar.

The canonical inventory is [index.json](index.json). It separates two states:

- `formalized_claim`
  A claim document exists with evidence, limits, and falsification criteria.
- `evidence_summary`
  A result looks strong enough to support a future claim, but the exact claim
  and limitations still need to be written.

Do not cite an `evidence_summary` as a finished claim. Promote it only after
the claim, evidence, and limitations are explicit.

Validate the inventory with:

```bash
uv run python research/claims/validate_index.py
```
