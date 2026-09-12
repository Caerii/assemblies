"""New runner evidence must be valid and navigable from its own registration."""

from research.evidence import audit_history, validate_active_evidence_graph


def test_tracked_runner_evidence_has_no_dangling_edges():
    errors = validate_active_evidence_graph()
    assert not errors, "\n".join(errors)


def test_every_nonpending_preregistration_has_a_result_edge():
    audit = audit_history()
    assert audit['preregistrations_without_resolved_result_links'] == []
