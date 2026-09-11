#!/usr/bin/env python3
"""Refuse regeneration of the retracted DIRECT overlap golden."""

from neural_assemblies.exceptions import RetractedProtocol


def main() -> None:
    raise RetractedProtocol(
        "direct2026_pearl is historical evidence only. Its overlap readout "
        "survived wiping the learned fiber; register a new protocol with a "
        "synaptic-asymmetry readout and a wipe negative control."
    )


if __name__ == "__main__":
    main()
