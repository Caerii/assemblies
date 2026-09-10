"""Source guards ignore generated copies without exempting tracked code."""
import subprocess

from neural_assemblies.tests._source_scan import python_sources


def test_git_inventory_includes_new_and_tracked_ignored_source(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("build/\n.venv/\n")
    (tmp_path / "new.py").write_text("pass\n")
    build = tmp_path / "build"
    build.mkdir()
    (build / "generated.py").write_text("pass\n")
    (build / "tracked.py").write_text("pass\n")
    subprocess.run(["git", "add", "-f", "build/tracked.py"], cwd=tmp_path, check=True)
    assert [p.relative_to(tmp_path).as_posix() for p in python_sources(tmp_path)] == [
        "build/tracked.py", "new.py",
    ]
