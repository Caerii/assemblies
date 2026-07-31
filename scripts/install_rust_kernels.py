"""Copy the built `na-kernels` library into site-packages as an extension.

Deliberately NOT maturin: the repo's own build backend is setuptools, and
adding a second one to `pyproject.toml` would make `pip install -e .` depend on
a Rust toolchain being present. The kernels are optional, so the build stays
optional too -- `cargo build` then this script, and nothing about the ordinary
install path changes.

    cargo build --release --manifest-path crates/Cargo.toml -p na-kernels
    python scripts/install_rust_kernels.py

See docs/RUST_KERNELS.md.
"""

import shutil
import site
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "crates" / "target" / "release"

# One built library, three platform spellings.
CANDIDATES = ("na_kernels.dll", "libna_kernels.so", "libna_kernels.dylib")


def main() -> int:
    built = next((RELEASE / c for c in CANDIDATES if (RELEASE / c).exists()), None)
    if built is None:
        print(f"No built library in {RELEASE}.\n"
              f"Run: cargo build --release "
              f"--manifest-path crates/Cargo.toml -p na-kernels", file=sys.stderr)
        return 1

    # EXT_SUFFIX rather than a hardcoded ".pyd"/".so": it carries the abi3 tag
    # when there is one, and getting it wrong yields a file Python silently
    # will not import.
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or (
        ".pyd" if sys.platform == "win32" else ".so")
    packages = site.getsitepackages()
    target = Path(packages[-1]) / f"na_kernels{suffix}"

    shutil.copy2(built, target)
    print(f"{built.name} -> {target}")

    # Import it for real. A copied-but-unloadable library (wrong Python ABI,
    # missing runtime) is the failure this script exists to surface, and it
    # would otherwise show up much later as "the accelerator is just never on".
    sys.path.insert(0, str(target.parent))
    try:
        import na_kernels
    except Exception as exc:                      # pragma: no cover
        print(f"copied, but the extension does not import: {exc}", file=sys.stderr)
        return 1
    fns = [f for f in dir(na_kernels) if not f.startswith("_")]
    print(f"loaded, exporting: {', '.join(fns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
