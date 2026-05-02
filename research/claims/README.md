# Claims

Use this directory for statements the research tree can defend.

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
