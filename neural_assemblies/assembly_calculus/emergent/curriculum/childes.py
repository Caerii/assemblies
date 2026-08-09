"""CHILDES CHAT (.cha) reading for the real-corpus graduation (#30/#150).

Scope, deliberately narrow: extract ADULT child-directed utterances as
clean word lists, with the %mor tier carried along WHEN IT ALIGNS. This
is the corpus level of the five-level law (research/notes/
production_configuration.md); everything linguistic downstream -- what
counts as a teacher signal, which words are groundable -- stays in the
experiment where it can be registered.

HONESTY RULES built in:
  * %mor alignment is notoriously fragile (clitics, retracings, omitted
    words each shift it). When the main tier and %mor disagree on token
    count, `mor` is None for that utterance and the reader COUNTS it --
    a misaligned teacher signal is worse than none (the
    writer-and-reader-must-share-the-lookup lesson, corpus edition).
  * No silent repair of unintelligible material: xxx/yyy/www tokens drop
    the TOKEN, not the utterance, and the drop is visible in `stats`.
  * The reader never guesses morphology from surface forms -- that is
    the substrate's job; %mor is the annotation the papers' teacher
    assumed into existence.

CHAT reference: MacWhinney, The CHILDES Project (2000), talkbank.org.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

#: Speakers that count as child-directed input (the child's own
#: productions are the OUTPUT distribution, not the input one).
ADULT_SPEAKERS = ("MOT", "FAT", "GRA", "GRF", "GRM", "INV", "ADU", "TEA")

#: Main-tier material that is not a word: events (&=laughs), pauses
#: ((.), (..)), best-guess markers, terminators kept for later use.
_UNINTELLIGIBLE = {"xxx", "yyy", "www"}
_TOKEN_DROP = re.compile(
    r"""^(
        &[=~+-]?\S* |          # events, fillers, phonological fragments
        \(\.+\) |              # pauses
        \+\S* |                # utterance-level codes (+..., +//.)
        [.!?;:,] |             # bare terminators/punctuation
        0\S*                   # omitted-word codes (0det, 0is)
    )$""",
    re.VERBOSE,
)
#: In-token cleanup: parenthesized completions "(be)cause" -> "because",
#: special-form markers "word@f" -> "word", quoting/linking underscores.
_PAREN = re.compile(r"\(([^)]*)\)")
_AT_SUFFIX = re.compile(r"@[\w:$]+$")


@dataclass(frozen=True)
class Utterance:
    speaker: str
    words: Tuple[str, ...]
    #: %mor tokens aligned 1:1 with `words`, or None when alignment failed.
    mor: Optional[Tuple[str, ...]]


@dataclass
class ChaStats:
    utterances_total: int = 0
    utterances_kept: int = 0
    mor_misaligned: int = 0
    unintelligible_tokens: int = 0
    by_speaker: Counter = field(default_factory=Counter)


def _clean_main_token(tok: str) -> Optional[str]:
    """One main-tier token -> a word, or None if it is not a word."""
    # Scoped/bracketed material was removed at the line level.
    if not tok or _TOKEN_DROP.match(tok):
        return None
    if tok in _UNINTELLIGIBLE:
        return None
    tok = _PAREN.sub(r"\1", tok)         # (be)cause -> because
    tok = _AT_SUFFIX.sub("", tok)        # word@f    -> word
    tok = tok.strip("<>„“\"'^")
    tok = tok.replace("_", " ")          # compound linking
    tok = tok.lower()
    if not tok or not any(c.isalpha() for c in tok):
        return None
    return tok


def _strip_brackets(line: str) -> str:
    """Remove [...] code groups and <...> retracing scopes.

    `<the the> [/] the dog` reads "the dog": the scoped material belongs
    to the retracing code that follows it, so both go.
    """
    out = re.sub(r"<[^>]*>\s*\[[^\]]*\]", " ", line)  # scoped + its code
    out = re.sub(r"\[[^\]]*\]", " ", out)             # remaining codes
    return out


def _split_mor(mor_line: str) -> List[str]:
    """%mor tokens, terminators dropped (they pair with punctuation the
    main-tier cleaner also drops)."""
    toks = []
    for t in mor_line.split():
        if t in {".", "!", "?", "+...", "+/.", "+//."} or t.startswith("+"):
            continue
        toks.append(t)
    return toks


def read_cha(text: str,
             speakers: Tuple[str, ...] = ADULT_SPEAKERS,
             ) -> Tuple[List[Utterance], ChaStats]:
    """Parse one CHAT transcript into adult utterances + honest counters."""
    stats = ChaStats()
    utterances: List[Utterance] = []

    # Physical lines -> logical lines (continuations start with a tab).
    logical: List[str] = []
    for raw in text.splitlines():
        if raw.startswith("\t") and logical:
            logical[-1] += " " + raw.strip()
        else:
            logical.append(raw.rstrip("\n"))

    current: Optional[Tuple[str, List[str], int]] = None  # spk, words, dropped
    for line in logical:
        if line.startswith("*"):
            if current is not None:
                spk, words, _ = current
                utterances.append(Utterance(spk, tuple(words), None))
            spk, _, rest = line.partition(":")
            spk = spk[1:].strip()
            stats.utterances_total += 1
            stats.by_speaker[spk] += 1
            if spk not in speakers:
                current = None
                continue
            raw_toks = _strip_brackets(rest).split()
            stats.unintelligible_tokens += sum(
                t in _UNINTELLIGIBLE for t in raw_toks)
            words = [w for t in raw_toks
                     if (w := _clean_main_token(t)) is not None]
            current = (spk, words, 0)
        elif line.startswith("%mor:") and current is not None:
            spk, words, _ = current
            mor = _split_mor(line.partition(":")[2])
            if words and len(mor) == len(words):
                utterances.append(Utterance(spk, tuple(words), tuple(mor)))
            else:
                if words:
                    stats.mor_misaligned += 1
                    utterances.append(Utterance(spk, tuple(words), None))
            current = None
        # @headers and other %tiers: ignored.
    if current is not None:
        spk, words, _ = current
        if words:
            utterances.append(Utterance(spk, tuple(words), None))

    utterances = [u for u in utterances if u.words]
    stats.utterances_kept = len(utterances)
    return utterances, stats


def read_childesdb_jsonl(text: str) -> Tuple[List[Utterance], ChaStats]:
    """childes-db export (fetch_childes_brown.py) -> the same Utterance
    stream read_cha produces.

    REGISTERED DEVIATION (#150, stated in the fetch script): TalkBank's
    raw-CHAT downloads went behind account auth, so data arrives as
    childes-db token rows. Alignment is guaranteed by construction there,
    so the honesty counter transposes: `mor_misaligned` counts
    utterances whose token rows lacked part-of-speech (mor comes through
    as null), the no-teacher case of this format.
    """
    import json as _json

    stats = ChaStats()
    utterances: List[Utterance] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = _json.loads(line)
        stats.utterances_total += 1
        stats.by_speaker[row["speaker"]] += 1
        words = tuple(row["words"])
        if not words:
            continue
        mor = row.get("mor")
        if mor is None:
            stats.mor_misaligned += 1
            utterances.append(Utterance(row["speaker"], words, None))
        else:
            utterances.append(Utterance(row["speaker"], words, tuple(mor)))
    stats.utterances_kept = len(utterances)
    return utterances, stats


# ---------------------------------------------------------------------------
# Corpus-statistics probes the graduation's bars are written against.
# ---------------------------------------------------------------------------

def frequency_spectrum(utterances: List[Utterance]) -> Counter:
    """Token frequency by surface form -- the Zipf check reads this."""
    c: Counter = Counter()
    for u in utterances:
        c.update(u.words)
    return c


def mor_number_teacher(utterance: Utterance) -> Dict[str, str]:
    """word -> "PL"|"SG" for NOUNS, from %mor (n|dog-PL). Empty when the
    utterance has no aligned %mor -- the honest no-teacher case."""
    out: Dict[str, str] = {}
    if utterance.mor is None:
        return out
    for word, mor in zip(utterance.words, utterance.mor):
        head = mor.split("~", 1)[0]          # clitic host
        if not head.startswith(("n|", "n:")):
            continue
        out[word] = "PL" if "-PL" in head else "SG"
    return out
