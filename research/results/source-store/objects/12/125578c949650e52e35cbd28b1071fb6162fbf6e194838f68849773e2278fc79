"""Shared negative controls for role parsing experiments."""

from collections.abc import Sequence

def role_guards(parser, probes: Sequence[tuple[str, dict[str, str | None]]]) -> dict:
    """Run the invariant role probes shared by scaling/gain studies."""
    out = {"roles_ok": 0, "roles_total": 0}
    for text, expected in probes:
        roles, _diag = parser.parse_roles_by_reconstruction(text.split())
        for word, wanted in expected.items():
            out["roles_total"] += 1
            out["roles_ok"] += roles.get(word) == wanted
    _r1, first = parser.parse_roles_by_reconstruction(
        "the dog chases the cat".split())
    _r2, second = parser.parse_roles_by_reconstruction(
        "the cat is chased by the dog".split())
    out["c1_identical"] = bool(first["winners"]) and first["winners"] == second["winners"]
    _r3, third = parser.parse_roles_by_reconstruction(
        "the child enters the mouse".split())
    _r4, fourth = parser.parse_roles_by_reconstruction(
        "the child is entered by the mouse".split())
    common = set(third["winners"]) & set(fourth["winners"])
    out["c2"] = (
        sum(
            len(set(third["winners"][role]) & set(fourth["winners"][role]))
            / max(1, len(third["winners"][role]))
            for role in common
        ) / len(common)
        if common else None
    )
    return out
