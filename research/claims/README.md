# Claims Index

This directory now has two layers:

- **Formalized claims**: standalone claim documents with evidence,
  limitations, and falsification criteria.
- **Claim-ready evidence summaries**: result summaries that appear strong enough
  to support future formal claims, but have not yet been promoted to full claim
  documents.

The canonical index is [index.json](index.json). It is intentionally explicit
about status so the repo does not blur together:

- evidence that exists,
- claims that have been written down,
- and hypotheses that are still only aspirations.

## Status values

- `formalized_claim`
  A claim document exists in `research/claims/` and is the main source of truth.
- `evidence_summary`
  A results summary exists and looks claim-ready, but limitations and exact
  claim wording still need to be formalized.

## Rule

Do not cite an `evidence_summary` entry as if it were already a polished claim.
Promote it to `formalized_claim` only after writing the claim, evidence, and
limitations explicitly.
