from research import evidence


def test_evidence_check_combines_maintained_graph_and_specification_gate(monkeypatch, capsys):
    monkeypatch.setattr(evidence, "validate_active_evidence_graph", lambda: ["artifact error"])
    monkeypatch.setattr(evidence, "specification_links", lambda: ([{"from": "x", "to": "y"}], ["spec error"]))

    assert evidence.main(["check"]) == 1
    payload = capsys.readouterr().out
    assert '"valid_maintained_graph": false' in payload
    assert "artifact error" in payload
    assert "spec error" in payload


def test_evidence_check_succeeds_when_both_boundaries_are_clean(monkeypatch, capsys):
    monkeypatch.setattr(evidence, "validate_active_evidence_graph", lambda: [])
    monkeypatch.setattr(evidence, "specification_links", lambda: ([], []))

    assert evidence.main(["check"]) == 0
    assert '"valid_maintained_graph": true' in capsys.readouterr().out
