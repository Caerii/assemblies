from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_historical_scripts_are_archived_under_legacy():
    archived_paths = {
        "scripts/overlap_sim.py": "legacy/scripts/simulations/overlap_sim.py",
        "scripts/turing_sim.py": "legacy/scripts/simulations/turing_sim.py",
        "scripts/run_turing_sim.py": "legacy/scripts/simulations/run_turing_sim.py",
        "scripts/project.py": "legacy/scripts/simulations/project.py",
        "scripts/build_cuda_simple.py": "legacy/scripts/tooling/build_cuda_simple.py",
        "scripts/animator.py": "legacy/scripts/visualization/animator.py",
    }

    for old_path, archived_path in archived_paths.items():
        assert not (REPO_ROOT / old_path).exists()
        assert (REPO_ROOT / archived_path).exists()
