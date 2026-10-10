"""A registration as ONE specification: the text it registers, the bars it judges, and the result
and scorecard it records, all rendered from the same object -- so the registered sentence, the
evaluation and the recorded verdict cannot drift apart.

    reg = Registration(amendment=56, title=..., cells=(Cell.of(...), ...), seeds=..., bars=(...), ...)
    reg.registration_text(date)          the "## Amendment N (date, before running)" section
    reg.problems()                       the ledger's objections to its cells and brains (empty: none)
    reg.judge(observations)              every bar, judged by its confidence bound
    reg.result_text(judged, ...)         the "### Amendment N result" section, bar lines computed
    reg.scorecard_rows(judged)           the scorecard's rows for this amendment

The prose a scientist must write -- why the question, what the probe saw, what each outcome
would mean, the reading of the result -- stays prose, passed in; everything that can be computed
is. research/amend.py applies these renderings to the documents.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field

from .bars import Bar, evaluate
from .ledger import check_new

WIDTH = 78


def _wrap(text, indent="", hang=None):
    hang = indent if hang is None else hang
    out = []
    for para in text.strip().split("\n\n"):
        out.append(textwrap.fill(" ".join(para.split()), WIDTH, initial_indent=indent,
                                 subsequent_indent=hang, break_on_hyphens=False))
    return "\n\n".join(out)


def _span(seeds):
    return f"{min(seeds)} to {max(seeds)}"


@dataclass(frozen=True)
class Registration:
    amendment: int
    title: str
    preamble: str                       # why: the question and what earlier amendments found
    seen: str                           # **Seen before registering.** -- probes and smokes, disclosed
    protocol: str                       # what runs, prose
    module: str                         # e.g. "memory_setpoint_sleep"
    runner_key: str                     # e.g. "setpoint_sleep"
    tag: str                            # the run's tag
    cells: tuple
    seeds: tuple
    bars: tuple                         # Bar objects
    reported: str = ""                  # Reported, not judged: ...
    risk: str = ""                      # ### Risk, stated now (optional)
    interpretation: tuple = field(default_factory=tuple)   # bullet sentences, stated now
    reference_seeds: tuple = ()

    # ------------------------------------------------------------ before the run
    def problems(self, ledger=None):
        return check_new(self.cells, self.seeds, self.reference_seeds, self.amendment, ledger)

    def _bar_ids(self):
        ids = [b.id for b in self.bars]
        return f"{ids[0]} to {ids[-1]}" if len(ids) > 1 else ids[0]

    def registration_text(self, date):
        brains = f"subject seeds {_span(self.seeds)}"
        if self.reference_seeds:
            brains += f", reference brains {_span(self.reference_seeds)}"
        cells = " and ".join(c.describe() for c in self.cells)
        parts = [
            f"## Amendment {self.amendment} ({date}, before running): {self.title}",
            _wrap(self.preamble),
            _wrap("**Seen before registering.** " + self.seen),
            "### Protocol",
            _wrap(f"`research/experiments/{self.module}.py` (`python -m research.runner {self.runner_key}`), "
                  f"{brains} (all new), one run from a worktree pinned at the commit registering this "
                  f"amendment. Cells {cells}. " + self.protocol),
            "    python -m research.runner " + self.runner_key + " \\\n"
            "        --registration research/notes/memory/PREREG_refraction_memory.md \\\n"
            f"        --tag {self.tag} --seeds {min(self.seeds)} ... {max(self.seeds)}",
            "### Bars",
            _wrap("Each bar holds at every cell. A mean over brains is judged by its 95% confidence "
                  "bound (the lower bound for a floor, the upper for a ceiling), not by the bare mean."),
            "\n".join(_wrap(f"{b.name}. {b.statement}", f"    {b.id}  ", "        ") for b in self.bars),
        ]
        if self.reported:
            parts.append(_wrap("Reported, not judged: " + self.reported))
        if self.risk:
            parts += ["### Risk, stated now", _wrap(self.risk)]
        if self.interpretation:
            parts += ["### Interpretation, stated now",
                      "\n".join(_wrap(i, "* ", "  ") for i in self.interpretation
                                + ("A failed bar is recorded as failed and not moved.",))]
        parts.append(f"The run is UNJUDGED until {self._bar_ids()} are evaluated and recorded below.")
        return "\n\n".join(parts) + "\n"

    # ------------------------------------------------------------ after the run
    def judge(self, observations, how="bound"):
        """every bar at every registered cell; observations["cells"] keyed "n/k/p" """
        cells = {c.key: observations["cells"][c.key] for c in self.cells}
        return evaluate(self.bars, cells, how)

    @staticmethod
    def _value(r):
        if "low" in r:
            return f"{r['mean']:.3f} +/- {r['ci']:.3f}"
        if "worst" in r:
            return f"{r['failing']} failing" if r["failing"] else "all"
        v = r["value"]
        return f"{v:g}" if isinstance(v, int) else f"{v:.4g}"

    def _values(self, judged, bar_id):
        cells = judged[bar_id]["cells"]
        return "; ".join(", ".join(self._value(r) for r in rs) for rs in cells.values())

    def bar_lines(self, judged):
        width = max(len(b.name) for b in self.bars) + 2
        return "\n".join(f"    {b.id}  {b.name.ljust(width)}{'PASS' if judged[b.id]['pass'] else 'FAIL'}  "
                         f"{self._values(judged, b.id)}" for b in self.bars)

    def result_text(self, judged, date, commit, run_path, log_path, summary, reading):
        brains = f"subject seeds {_span(self.seeds)}"
        if self.reference_seeds:
            brains += f",\nreference brains {_span(self.reference_seeds)}"
        head = (f"One run from a worktree pinned at {commit}, {brains}:\n"
                f"[record]({run_path}),\n[log]({log_path}).")
        return "\n\n".join([f"### Amendment {self.amendment} result ({date})", head + " " + summary.strip(),
                            self.bar_lines(judged), _wrap("**Reading.** " + reading)]) + "\n"

    def scorecard_rows(self, judged, short):
        """``short`` {bar id: (row label, registered threshold text)}"""
        return "".join(f"| {b.id} {short[b.id][0]} (A{self.amendment}) | {short[b.id][1]} | "
                       f"{'PASS' if judged[b.id]['pass'] else 'FAIL'} | {self._values(judged, b.id)} |\n"
                       for b in self.bars)


__all__ = ["Registration", "Bar"]
