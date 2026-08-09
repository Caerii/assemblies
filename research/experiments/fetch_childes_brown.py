"""#150 data acquisition: Brown (Eng-NA) adult utterances from childes-db.

WHY THIS CHANNEL. TalkBank's direct .cha downloads went behind account
authentication (checked 2026-08-09; the ?f=zip endpoints return an auth
modal). childes-db (Sanchez, Meylan, Braginsky, MacDonald, Yurovsky &
Frank 2019, Behavior Research Methods) is the project's own licensed
redistribution: a public read-only MySQL whose connection info is
published at langcog.github.io/childes-db-website/childes-db.json and
consumed by their childesr/childespy clients. This script is that
client, minus the R dependency.

REGISTERED DEVIATION from the #150 registration (stated, per the
process): data arrives as childes-db TOKEN ROWS, not raw CHAT, so the
"%mor alignment" honesty rule transposes -- alignment is guaranteed by
construction (tokens are keyed to utterances), and the no-teacher case
becomes "part_of_speech is empty for some token". The census bars C1-C3
and Phase-1 bars F1-F3 are unchanged; they never referenced the format.

Output (git-ignored, TalkBank/childes-db terms -- cite, don't commit):
  data/childes/brown_adult.jsonl   one utterance per line:
      {"speaker": role, "words": [...], "mor": [...] | null}
      mor tokens synthesized as pos|stem[-SUFFIX] to match the CHAT
      reader's mor_number_teacher contract (n|dog-PL).
  data/childes/brown_adult.meta.json   provenance + counts.
"""
from __future__ import annotations

import json
import os
import urllib.request

import pymysql
import pymysql.cursors

DB_INFO_URL = "https://langcog.github.io/childes-db-website/childes-db.json"
CORPUS_ID = 60          # Brown, collection Eng-NA (looked up by name)
ADULT_ROLES = ("Mother", "Father", "Grandmother", "Grandfather",
               "Investigator", "Adult", "Teacher")
SKIP_GLOSS = {"", "xxx", "yyy", "www"}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                        "data", "childes")
OUT_JSONL = os.path.join(DATA_DIR, "brown_adult.jsonl")
OUT_META = os.path.join(DATA_DIR, "brown_adult.meta.json")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    info = json.load(urllib.request.urlopen(DB_INFO_URL))
    print(f"childes-db host={info['host']} version={info['current']}")
    conn = pymysql.connect(
        host=info["host"], user=info["user"], password=info["password"],
        database=info["current"], connect_timeout=30,
        cursorclass=pymysql.cursors.SSCursor)
    cur = conn.cursor()
    roles = ",".join(f"'{r}'" for r in ADULT_ROLES)
    cur.execute(
        f"SELECT utterance_id, token_order, gloss, stem, part_of_speech,"
        f" suffix, speaker_role FROM token"
        f" WHERE corpus_id=%s AND speaker_role IN ({roles})"
        f" ORDER BY utterance_id, token_order", (CORPUS_ID,))

    n_utts = n_tokens = n_no_pos = 0
    by_role: dict = {}
    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        cur_id = None
        words, mor, role, missing_pos = [], [], None, False

        def flush():
            nonlocal n_utts, n_no_pos
            if not words:
                return
            n_utts += 1
            by_role[role] = by_role.get(role, 0) + 1
            m = None
            if not missing_pos:
                m = mor
            else:
                n_no_pos += 1
            out.write(json.dumps({"speaker": role, "words": words,
                                  "mor": m}) + "\n")

        for uid, _order, gloss, stem, pos, suffix, srole in cur:
            if uid != cur_id:
                flush()
                cur_id, words, mor, role, missing_pos = uid, [], [], srole, False
            g = (gloss or "").strip().lower()
            if g in SKIP_GLOSS:
                continue
            n_tokens += 1
            words.append(g)
            if not (pos or "").strip():
                missing_pos = True
                mor.append("")
            else:
                m = f"{pos}|{(stem or g)}"
                if (suffix or "").strip():
                    m += f"-{suffix.strip().upper()}"
                mor.append(m)
        flush()

    meta = {
        "source": "childes-db (public read-only MySQL), corpus Brown "
                  "(Eng-NA), corpus_id 60",
        "db_version": info["current"],
        "fetched": "2026-08-09",
        "citation": "Sanchez et al. 2019, Behav Res Methods; "
                    "Brown 1973 via CHILDES (MacWhinney 2000)",
        "adult_roles": ADULT_ROLES,
        "utterances": n_utts,
        "tokens": n_tokens,
        "utterances_without_full_pos": n_no_pos,
        "by_role": by_role,
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
