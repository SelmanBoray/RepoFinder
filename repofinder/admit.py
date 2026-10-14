"""The phrase heuristic for "the answer admits that what was asked is missing".

An answer *admits the answer is missing* when it says that the code a question
asks about is not in the code excerpts it was given, or not in the repository.
For a question whose answer lives in another library that is the right answer;
for an answerable question it is a false admission. The eval
(``eval/run_eval.py``) measures both.

This module is the local, rules-only way to tell: :func:`admits_missing`
looks for one of :data:`ADMIT_PHRASES`. It misfires both ways: "the route
decorator is in scaffold.py; add_url_rule is not in the excerpts" contains a
phrase but gives the location (a false positive), and wordings outside the list
are missed. :func:`repofinder.decisions.answer_admits_missing` asks Jev
instead and falls back to this heuristic with the local rules
(:meth:`repofinder.decisions.RulesDecisions.admits`, which reports
:data:`ADMIT_YES` / :data:`ADMIT_NO` as its probability).

Standard library only.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# HEURISTIC: an answer "admits the answer is missing" when its text contains one of
# these phrases (compared after normalize_text: case-folded, accents and the
# Turkish dotted/dotless i folded to plain "i", curly apostrophes made straight).
# It can give false positives ("... is not shown here, but X does ...") and false
# negatives (other wordings). Read the stored answer texts before trusting the
# number, and extend the list when a correct admission is missed.
ADMIT_PHRASES: dict[str, tuple[str, ...]] = {
    "en": (
        "not in the excerpts",
        "not in these excerpts",
        "not in the provided excerpts",
        "not in the provided code",
        "excerpts do not",
        "excerpts don't",
        "excerpts does not",
        "not shown",
        "not included",
        "not contained",
        "not present in",
        "do not contain",
        "does not contain",
        "don't contain",
        "doesn't contain",
        "do not include",
        "does not include",
        "do not show",
        "does not show",
        "don't show",
        "doesn't show",
        "not part of this repo",
        "not part of the repo",
        "not implemented in this repo",
        "not implemented in flask",
        "outside this repo",
        "outside the repo",
        "not in this repo",
        "not in the repo",
    ),
    "tr": (
        "alıntılarda yok",
        "alıntılarda bulunmuyor",
        "alıntılarda görünmüyor",
        "alıntılarda yer almıyor",
        "alıntılar içermiyor",
        "görünmüyor",
        "yer almıyor",
        "bulunmuyor",
        "içermiyor",
        "gösterilmiyor",
        "bu depoda değil",
        "depoda yok",
        "bu repoda değil",
        "repoda yok",
        "kod tabanında yok",
    ),
}

# The probability the rules report for the heuristic's verdict: a phrase was
# found / none was found. Not calibrated; it only keeps the verdict on the same
# 0..1 scale as Jev's, on either side of a 0.5 threshold.
ADMIT_YES = 0.9
ADMIT_NO = 0.1


def normalize_text(text: str) -> str:
    """Case-folded, accents removed, Turkish ı/İ/I folded to "i", curly
    apostrophes made straight, whitespace runs collapsed to one space."""
    text = text.replace("İ", "i").replace("I", "i").replace("ı", "i")
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    for curly in ("’", "‘", "ʼ"):
        text = text.replace(curly, "'")
    return re.sub(r"\s+", " ", text)


_ADMIT_NORMALIZED = tuple(
    (phrase, normalize_text(phrase)) for phrases in ADMIT_PHRASES.values() for phrase in phrases
)


def admits_missing(text: str) -> Optional[str]:
    """The first :data:`ADMIT_PHRASES` entry found in ``text``, else None. A heuristic."""
    norm = normalize_text(text or "")
    for phrase, needle in _ADMIT_NORMALIZED:
        if needle in norm:
            return phrase
    return None
