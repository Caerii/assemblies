"""``Brain(seed=)`` must give the same answer in a DIFFERENT PROCESS.

WHY A SUBPROCESS.  This is the fourth time this codebase has shipped a
reproducibility bug that no in-process test could possibly catch, because each
one was invisible *within* a single interpreter and only appeared across two:

  * the global RNG leaking between ``Brain`` constructions -- two brains built
    in one process were correlated, so a same-process comparison looked fine;
  * ``hash()``-derived pair seeds -- stable within a run, different between runs
    under Python's per-process hash randomization;
  * CUDA's synapse init not being Bernoulli, giving near-constant in-degree;
  * the torch candidate samplers drawing from torch's PROCESS-GLOBAL stream
    while ignoring the ``rng`` threaded into them (see ``_device_rng``).

In every case a same-process assertion passes, because the global stream
advances identically when you replay the same calls. The bug lives in the
starting point of that stream, which is not the caller's to control. So the
test has to fork.

Measured before the fix, one cell (k=100, p=0.05, beta=0.05, 15 rounds,
norm_init, n_src=1000 -> n_tgt=10000): ``w`` = 484, 514, 506 on three separate
invocations at a fixed seed. That is far above the noise floor of anything we
would want to conclude from a cross-engine comparison.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest


def _has_torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


ENGINES = ["numpy_sparse"] + (["torch_sparse"] if _has_torch_cuda() else [])

# Deliberately a configuration where candidates compete closely with incumbents
# and land right at the k-WTA cut -- that is where a stray draw changes the
# winner set. A configuration with a comfortable margin would pass while broken.
_PROBE = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, r"{repo}")
    from neural_assemblies import Brain

    b = Brain(p=0.05, save_winners=True, seed=7, engine="{engine}",
              norm_init=True)
    b.add_stimulus("s", 100)
    b.add_area("SRC", 1000, 100, 0.05)
    b.add_area("TGT", 10000, 100, 0.05)
    for _ in range(10):
        b.project({{"s": ["SRC"]}}, {{}})
    for _ in range(15):
        b.project({{"s": ["SRC"]}}, {{"SRC": ["TGT"]}})

    eng = b._engine_for(b.areas["TGT"])
    print("RESULT" + json.dumps({{
        "w": int(eng._areas["TGT"].w),
        "winners": sorted(int(x) for x in b.areas["TGT"].winners),
    }}))
""")


def _run_in_subprocess(engine, repo_root, hashseed):
    env = dict(os.environ)
    # Vary PYTHONHASHSEED between the two runs on purpose: a seeding path that
    # derives from hash() is reproducible only when this happens to match, and
    # that defect has shipped here before.
    env["PYTHONHASHSEED"] = str(hashseed)
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(repo=repo_root, engine=engine)],
        capture_output=True, text=True, env=env, timeout=900,
    )
    if out.returncode != 0:
        pytest.fail(f"{engine} probe failed:\n{out.stdout}\n{out.stderr}")
    for line in out.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[len("RESULT"):])
    pytest.fail(f"{engine} probe produced no RESULT line:\n{out.stdout}")


@pytest.mark.slow
@pytest.mark.parametrize("engine", ENGINES)
def test_same_seed_same_winners_across_processes(engine, pytestconfig):
    """Two fresh interpreters, one seed, identical winners.

    Asserts the WINNER SET, not just ``w``. A count can match by coincidence
    while the identities differ, and it is the identities every downstream
    overlap metric is computed from.
    """
    repo_root = str(pytestconfig.rootpath)
    a = _run_in_subprocess(engine, repo_root, hashseed=0)
    b = _run_in_subprocess(engine, repo_root, hashseed=12345)

    assert a["w"] == b["w"], (
        f"{engine}: recruitment count differs across processes at one seed -- "
        f"w={a['w']} vs {b['w']}. Some draw is reading a process-global RNG "
        f"instead of the seeded stream.")
    assert a["winners"] == b["winners"], (
        f"{engine}: same w={a['w']} but different winner IDENTITIES across "
        f"processes -- {len(set(a['winners']) ^ set(b['winners']))} of "
        f"{len(a['winners'])} differ. Every overlap metric downstream is "
        f"computed from these.")
