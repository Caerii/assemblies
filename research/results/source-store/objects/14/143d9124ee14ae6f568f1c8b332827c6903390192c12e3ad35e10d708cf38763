"""New runner evidence must be valid and navigable from its own registration."""

from research.evidence import audit_history, validate_active_evidence_graph


def test_tracked_runner_evidence_has_no_dangling_edges():
    audit = audit_history()
    errors = validate_active_evidence_graph(audit=audit)
    assert not errors, "\n".join(errors)
    # Keep the preregistration assertion in the same audit pass. Both checks
    # traverse the complete tracked repository; running them as separate tests
    # paid the filesystem walk twice without adding independent coverage.
    assert audit['preregistrations_without_resolved_result_links'] == []
