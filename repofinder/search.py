"""Lexical search over a :class:`RepoIndex`: BM25 plus a few code-aware boosts.

Every symbol and every file becomes one bag-of-tokens document:

* symbol: name, qualname, signature, docstring and path components;
* file: path components, the names of the file's symbols and its summary.

A query is tokenized the same way (plus a list of natural-language filler words
such as "where" or "implemented" that are dropped), scored with Okapi BM25 and
then multiplied by boosts. Every boost that fires is named in :attr:`Hit.why`.

Pure Python, standard library only. No stemming.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from .graph import is_test_path, module_names
from .models import RepoIndex, Symbol

K1 = 1.5
B = 0.75
NAME_BOOST = 3.0       # symbol name equals a query word
QUALNAME_BOOST = 2.0   # dotted/slashed query term matches the qualname, module or path
FILE_NAME_BOOST = 2.0  # query word equals a file's stem (package dir name for __init__.py)
CLASS_BOOST = 1.2
TEST_PENALTY = 0.5
NESTED_PENALTY = 0.5   # closures ("outer.<locals>.inner") are rarely what a reader wants
MAX_JOIN = 3           # adjacent query words joined into identifiers: "test client" -> test_client

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
# "Flask.run", "flask/app.py", "sansio.scaffold.Scaffold.route"
_COMPOUND_RE = re.compile(r"[A-Za-z0-9_]+(?:[./][A-Za-z0-9_]+)+")
_LOCALS_RE = re.compile(r"\.?<locals>")

# Dropped wherever they stand alone. Parts of a compound identifier are kept
# ("url_for" -> "url", "for", "url_for").
STOPWORDS = frozenset(
    {
        # Python noise that appears in nearly every signature
        "self", "cls", "def", "class", "return", "returns", "async", "await", "lambda",
        "none", "true", "false", "pass", "args", "kwargs", "elif", "else", "if", "try",
        "except", "finally", "raise", "yield", "global", "nonlocal", "del", "assert",
        "break", "continue", "while", "for", "in", "is", "not", "and", "or", "as",
        "with", "from", "py",
        # reST/Sphinx markup in docstrings
        "meth", "func", "attr", "obj", "param", "rtype", "versionadded", "versionchanged",
        # English function words
        "a", "an", "the", "of", "to", "on", "at", "by", "no", "nor", "are", "was", "were",
        "be", "been", "being", "am", "it", "its", "this", "that", "these", "those", "there",
        "here", "what", "which", "who", "whom", "whose", "when", "why", "how", "do", "does",
        "did", "doing", "can", "could", "should", "would", "will", "shall", "may", "might",
        "must", "has", "have", "had", "having", "we", "our", "you", "your", "they", "them",
        "their", "he", "she", "his", "her", "me", "my", "then", "than", "so", "such", "too",
        "very", "just", "also", "only", "into", "onto", "over", "under", "about", "above",
        "below", "all", "any", "some", "each", "both", "either", "neither", "other",
        "others", "more", "most", "much", "many", "few", "same", "own", "again", "once",
        "but", "yet", "via", "per", "etc", "eg", "ie",
    }
)

# Extra words dropped from queries only: they describe the question, not the code.
QUERY_FILLER = frozenset(
    {
        "where", "implemented", "implement", "implements", "implementation",
        "implementations", "defined", "define", "defines", "definition", "definitions",
        "located", "locate", "location", "find", "code", "logic", "feature", "features",
        "happen", "happens", "happening", "handled", "file", "files", "function",
        "functions", "method", "methods", "module", "modules", "show", "tell", "look",
        "see", "please", "want", "need", "know", "live", "lives", "responsible", "thing",
        "things", "stuff", "repo", "repository", "codebase",
    }
)

_TEST_WORDS = frozenset({"test", "tests", "testing"})


def tokenize(text: str) -> list[str]:
    """Split identifiers and prose into lowercase search tokens.

    snake_case, camelCase, dots and slashes are split; tokens shorter than two
    characters and stand-alone :data:`STOPWORDS` are dropped. A compound
    identifier also yields itself: ``url_for`` -> ``url``, ``for``, ``url_for``;
    ``getURL`` -> ``get``, ``url``, ``geturl``.
    """
    return _tokenize(text, STOPWORDS)


def _tokenize(text: str, stop: frozenset[str]) -> list[str]:
    out: list[str] = []
    for word in _WORD_RE.findall(text):
        pieces = [p.lower() for chunk in word.split("_") if chunk for p in _CAMEL_RE.findall(chunk)]
        if len(pieces) > 1:
            out.extend(p for p in pieces if len(p) >= 2)
            whole = word.strip("_").lower()
            if len(whole) >= 2:
                out.append(whole)
        elif pieces:
            piece = pieces[0]
            if len(piece) >= 2 and piece not in stop:
                out.append(piece)
    return out


def _path_text(path: str) -> str:
    """Path without its extension, for tokenizing (``src/flask/app.py`` -> ``src/flask/app``)."""
    head, _, tail = path.rpartition("/")
    stem = tail.rsplit(".", 1)[0] if "." in tail else tail
    return f"{head}/{stem}" if head else stem


def _file_stem(path: str) -> str:
    """``pkg/mod.py`` -> ``mod``; ``pkg/sub/__init__.py`` -> ``sub``."""
    parts = path.split("/")
    stem = parts[-1].rsplit(".", 1)[0]
    if stem == "__init__" and len(parts) > 1:
        stem = parts[-2]
    return stem.lower()


@dataclass
class Hit:
    path: str
    symbol_id: Optional[str]   # None for file-level hits
    start_line: int            # 1-based; for file-level hits 1
    end_line: int              # for file-level hits the file's n_lines
    score: float
    kind: str                  # "symbol" | "file"
    why: str                   # short human explanation of what matched


@dataclass
class _Doc:
    kind: str                  # "symbol" | "file"
    path: str
    symbol: Optional[Symbol]
    end_line: int
    length: int
    name: str                  # lowercase symbol name, or file stem
    qualname: str              # lowercase qualname without <locals>; "" for files
    dotted: str                # ".module.qualname." lowercase, dot-delimited; "" if unknown
    slashed: str               # "/path" lowercase
    is_test: bool
    is_class: bool
    is_nested: bool


class SearchIndex:
    """BM25 index over the symbols and files of one repository."""

    def __init__(self, index: RepoIndex, summaries: Optional[dict[str, str]] = None) -> None:
        self.index = index
        summaries = summaries or {}
        modules = module_names(f.path for f in index.files)

        self._docs: list[_Doc] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._df: Counter[str] = Counter()

        names_by_file: dict[str, list[str]] = {}
        path_tokens: dict[str, list[str]] = {}
        for record in index.files:
            path_tokens[record.path] = tokenize(_path_text(record.path))

        for sym in index.symbols:
            names_by_file.setdefault(sym.path, []).append(sym.name)
            qual = _LOCALS_RE.sub("", sym.qualname)
            module = modules.get(sym.path)
            tokens = tokenize(sym.name)
            tokens += tokenize(qual)
            tokens += tokenize(sym.signature)
            if sym.docstring:
                tokens += tokenize(sym.docstring)
            tokens += path_tokens.get(sym.path) or tokenize(_path_text(sym.path))
            self._add(
                tokens,
                _Doc(
                    kind="symbol",
                    path=sym.path,
                    symbol=sym,
                    end_line=sym.end_line,
                    length=len(tokens),
                    name=sym.name.lower(),
                    qualname=qual.lower(),
                    dotted=f".{module}.{qual}.".lower() if module else f".{qual}.".lower(),
                    slashed="/" + sym.path.lower(),
                    is_test=is_test_path(sym.path),
                    is_class=sym.kind == "class",
                    is_nested="<locals>" in sym.qualname,
                ),
            )

        for record in index.files:
            tokens = list(path_tokens[record.path])
            for name in names_by_file.get(record.path, []):
                tokens += tokenize(name)
            summary = summaries.get(record.path)
            if summary:
                tokens += tokenize(summary)
            module = modules.get(record.path)
            self._add(
                tokens,
                _Doc(
                    kind="file",
                    path=record.path,
                    symbol=None,
                    end_line=max(record.n_lines, 1),
                    length=len(tokens),
                    name=_file_stem(record.path),
                    qualname="",
                    dotted=f".{module}.".lower() if module else "",
                    slashed="/" + record.path.lower(),
                    is_test=is_test_path(record.path),
                    is_class=False,
                    is_nested=False,
                ),
            )

        # Symbols and files have very different lengths: normalize each kind by its own mean.
        self._avgdl: dict[str, float] = {}
        for kind in ("symbol", "file"):
            lengths = [d.length for d in self._docs if d.kind == kind]
            self._avgdl[kind] = (sum(lengths) / len(lengths)) if lengths else 1.0
        n_docs = len(self._docs)
        self._idf = {
            tok: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)) for tok, df in self._df.items()
        }

    def _add(self, tokens: list[str], doc: _Doc) -> None:
        doc_id = len(self._docs)
        self._docs.append(doc)
        for tok, tf in Counter(tokens).items():
            self._postings.setdefault(tok, []).append((doc_id, tf))
            self._df[tok] += 1

    # ------------------------------------------------------------------ query

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """Top ``k`` hits for ``query``: score desc, then path, then start line."""
        if k <= 0:
            return []
        raw_words = [w.lower() for w in _WORD_RE.findall(query)]
        joined = self._joined_identifiers(raw_words)
        drop = STOPWORDS | QUERY_FILLER
        q_tokens = list(dict.fromkeys(_tokenize(query, drop) + joined))
        if not q_tokens:  # only filler words, e.g. a function literally named "find"
            drop = STOPWORDS
            q_tokens = list(dict.fromkeys(_tokenize(query, drop)))
        if not q_tokens:
            return []
        words = (set(raw_words) - drop) | set(joined)
        compounds = [c.lower() for c in _COMPOUND_RE.findall(query)]
        mentions_tests = bool(_TEST_WORDS & set(q_tokens))

        # BM25 accumulation over the postings of the query tokens only.
        scores: dict[int, float] = {}
        matched: dict[int, list[tuple[float, str]]] = {}
        for tok in q_tokens:
            postings = self._postings.get(tok)
            if not postings:
                continue
            idf = self._idf[tok]
            for doc_id, tf in postings:
                doc = self._docs[doc_id]
                norm = K1 * (1.0 - B + B * doc.length / self._avgdl[doc.kind])
                part = idf * tf * (K1 + 1.0) / (tf + norm)
                scores[doc_id] = scores.get(doc_id, 0.0) + part
                matched.setdefault(doc_id, []).append((part, tok))

        hits: list[Hit] = []
        for doc_id, base in scores.items():
            doc = self._docs[doc_id]
            score = base
            why: list[str] = []
            if doc.kind == "symbol":
                if doc.name in words:
                    score *= NAME_BOOST
                    why.append(f"name match: {doc.symbol.name}")  # type: ignore[union-attr]
                term = _compound_match(doc, compounds)
                if term is not None:
                    score *= QUALNAME_BOOST
                    why.append(f"qualname/path match: {term}")
                if doc.is_class:
                    score *= CLASS_BOOST
                    why.append("class boost")
                if doc.is_nested:
                    score *= NESTED_PENALTY
                    why.append(f"nested x{NESTED_PENALTY:g}")
            else:
                if doc.name in words:
                    score *= FILE_NAME_BOOST
                    why.append(f"path match: {doc.name}")
                else:
                    term = _compound_match(doc, compounds)
                    if term is not None:
                        score *= FILE_NAME_BOOST
                        why.append(f"path match: {term}")
            if doc.is_test and not mentions_tests:
                score *= TEST_PENALTY
                why.append(f"test file x{TEST_PENALTY:g}")
            toks = [t for _part, t in sorted(matched[doc_id], key=lambda pt: (-pt[0], pt[1]))]
            why.append("bm25: " + ", ".join(toks))

            sym = doc.symbol
            hits.append(
                Hit(
                    path=doc.path,
                    symbol_id=sym.id if sym is not None else None,
                    start_line=sym.start_line if sym is not None else 1,
                    end_line=doc.end_line,
                    score=score,
                    kind=doc.kind,
                    why="; ".join(why),
                )
            )

        hits.sort(key=lambda h: (-h.score, h.path, h.start_line, h.kind, h.symbol_id or ""))
        return hits[:k]

    def _joined_identifiers(self, words: list[str]) -> list[str]:
        """Runs of 2..MAX_JOIN adjacent query words that, joined with ``_`` or nothing,
        form a whole identifier present in the index (``url for`` -> ``url_for``)."""
        out: list[str] = []
        for size in range(2, MAX_JOIN + 1):
            for i in range(len(words) - size + 1):
                run = words[i : i + size]
                for candidate in ("_".join(run), "".join(run)):
                    if candidate in self._postings and candidate not in out:
                        out.append(candidate)
        return out


def _compound_match(doc: _Doc, compounds: list[str]) -> Optional[str]:
    """First dotted/slashed query term that names this document, else None.

    ``Flask.run`` matches qualname ``Flask.run`` and dotted ``flask.app.Flask.run``
    on dot boundaries (not ``Flask.run_simple``); ``flask/app.py`` or ``app.py``
    matches the path on ``/`` boundaries.
    """
    for term in compounds:
        if doc.qualname and term == doc.qualname:
            return term
        if doc.dotted and f".{term}." in doc.dotted:
            return term
        at = doc.slashed.find("/" + term)
        if at >= 0:
            end = at + 1 + len(term)
            if end == len(doc.slashed) or doc.slashed[end] in "/.":
                return term
    return None
