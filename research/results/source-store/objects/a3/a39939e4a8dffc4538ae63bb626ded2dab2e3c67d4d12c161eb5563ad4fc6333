from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_INDEX_PATH = REPO_ROOT / "research" / "core_questions" / "index.json"
CLAIMS_INDEX_PATH = REPO_ROOT / "research" / "claims" / "index.json"

REQUIRED_QUESTION_FILES = {
    "hypothesis.md",
    "theoretical_basis.md",
    "experiments.md",
    "results.md",
    "analysis.md",
    "conclusions.md",
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    core_index = _load_json(CORE_INDEX_PATH)
    claims_index = _load_json(CLAIMS_INDEX_PATH)

    claim_ids = {entry["id"] for entry in claims_index.get("entries", [])}

    if core_index.get("version") != 1:
        raise SystemExit("core_questions/index.json must declare version 1")

    entries = core_index.get("entries", [])
    if not entries:
        raise SystemExit("core_questions/index.json must contain at least one entry")

    seen_ids: set[str] = set()

    for entry in entries:
        for field in (
            "id",
            "status",
            "title",
            "question_directory",
            "open_question_refs",
            "claim_index_refs",
            "experiment_scripts",
            "result_artifacts",
            "notes",
        ):
            if field not in entry:
                raise SystemExit(f"Missing field '{field}' in core question entry")

        entry_id = entry["id"]
        if entry_id in seen_ids:
            raise SystemExit(f"Duplicate core question id: {entry_id}")
        seen_ids.add(entry_id)

        if entry["status"] != "curated_question":
            raise SystemExit(
                f"Unsupported status for {entry_id}: {entry['status']}"
            )

        question_dir = REPO_ROOT / entry["question_directory"]
        if not question_dir.is_dir():
            raise SystemExit(f"Missing question directory: {question_dir}")

        missing_files = sorted(
            filename
            for filename in REQUIRED_QUESTION_FILES
            if not (question_dir / filename).is_file()
        )
        if missing_files:
            raise SystemExit(
                f"{entry_id} is missing required question files: {missing_files}"
            )

        for rel_path in entry["experiment_scripts"] + entry["result_artifacts"]:
            path = REPO_ROOT / rel_path
            if not path.exists():
                raise SystemExit(f"{entry_id} references missing artifact: {rel_path}")

        for claim_ref in entry["claim_index_refs"]:
            if claim_ref not in claim_ids:
                raise SystemExit(
                    f"{entry_id} references unknown claims index id: {claim_ref}"
                )

    print(
        f"Core question index valid: {len(entries)} curated questions checked."
    )


if __name__ == "__main__":
    main()
