"""Lexical search over a :class:`RepoIndex`: BM25 plus a few code-aware boosts.

Every symbol and every file becomes one bag-of-tokens document:

* symbol: name, qualname, signature, docstring and path components;
* file: path components, the names of the file's symbols and its summary.

Every word is also indexed under a light English stem (:func:`stem`:
``signing`` -> ``sign``, ``loaded`` -> ``load``, ``builder`` -> ``build``), next
to the word itself.

A query is analysed into *units* (see :meth:`SearchIndex._analyze`):

* its content words. Natural-language filler ("where", "implemented"; Turkish
  "nerede", "tanımlı", ...) is dropped. In a query that looks Turkish, a word
  that is not in the index vocabulary is translated through a small
  stem-prefix lexicon ("kaydediliyor" -> register, "uygulamaya" -> app,
  application) and matched by the translation's stem;
* runs of adjacent words that, joined, form an identifier present in the index
  ("app context" -> ``app_context``/``appcontext``; with common abbreviations,
  "python file" -> ``pyfile``).

Each unit weighs its IDF (a joined run only what it adds over its words). The
units' tokens, their stems and abbreviations are scored with Okapi BM25 and the
score is multiplied by boosts:

* name: ``NAME_BOOST ** (coverage * part_coverage ** NAME_PART_EXP)``, where
  *coverage* is the share of the query's unit weight that the symbol name
  accounts for and *part_coverage* the share of the name's own parts
  (snake/camel pieces) found in the query. A name that is the whole query gets
  the full boost; a common word that happens to be a name ("loads", "session")
  in a longer question gets little; ``teardown_appcontext`` for "app context
  teardown" gets much;
* owner: the enclosing class is named in the query too (``make_response`` "on
  the Flask application object" -> ``Flask.make_response``), scaled by how
  strongly the name itself matched;
* qualname/path: a dotted or slashed query term names the symbol or file;
* class, kind ("which class ..."), test-file and nested-function adjustments.

Every boost that fires is named in :attr:`Hit.why`. Pure Python, standard
library only.
"""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Callable, Optional

from .graph import is_test_path, module_names
from .models import RepoIndex, Symbol

K1 = 1.5
B = 0.75
NAME_BOOST = 3.0       # symbol name is the whole query; scaled down by coverage (see module doc)
NAME_PART_EXP = 0.5    # weight of the name's own part coverage in the name boost exponent
OWNER_BOOST = 1.5      # the enclosing class is named too; exponent scaled by the name's
QUALNAME_BOOST = 2.0   # dotted/slashed query term matches the qualname, module or path
FILE_NAME_BOOST = 2.0  # file stem (package dir name for __init__.py) is the whole query
CLASS_BOOST = 1.2
KIND_BOOST = 3.0       # the query asks for a class ("which class ...") and this is one
TEST_PENALTY = 0.5
NESTED_PENALTY = 0.5   # closures ("outer.<locals>.inner") are rarely what a reader wants
MAX_JOIN = 3           # adjacent query words joined into identifiers: "test client" -> test_client

_WORD_RE = re.compile(r"\w+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
# "Flask.run", "flask/app.py", "sansio.scaffold.Scaffold.route"
_COMPOUND_RE = re.compile(r"[A-Za-z0-9_]+(?:[./][A-Za-z0-9_]+)+")
_LOCALS_RE = re.compile(r"\.?<locals>")
_CLASS_BASED_RE = re.compile(r"(?i)\bclass-based\b")

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
# ("find" is not one: find_best_app, _find_error_handler.)
QUERY_FILLER = frozenset(
    {
        "where", "implemented", "implement", "implements", "implementation",
        "implementations", "defined", "define", "defines", "definition", "definitions",
        "located", "locate", "location", "code", "logic", "feature", "features",
        "happen", "happens", "happening", "handled", "file", "files", "function",
        "functions", "method", "methods", "module", "modules", "show", "tell", "look",
        "see", "please", "want", "need", "know", "live", "lives", "responsible", "thing",
        "things", "stuff", "repo", "repository", "codebase",
    }
)

# A query word from this set lifts the test-file penalty, unless it is part of a
# joined identifier ("the test client" -> test_client is not a request for tests).
_TEST_WORDS = frozenset({"test", "tests", "testing", "pytest", "conftest", "fixture", "fixtures"})
_CLASS_WORDS = frozenset(
    {"class", "classes", "sınıf", "sınıfı", "sınıfını", "sınıflar", "sınıfları"}
)

# Common abbreviations in identifiers, looked up by a query word or its stem.
_ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "application": ("app",), "configuration": ("config", "conf", "cfg"),
    "environment": ("env", "environ"), "context": ("ctx",), "request": ("req",),
    "response": ("resp",), "python": ("py",), "directory": ("dir",),
    "information": ("info",), "initialize": ("init",), "initialization": ("init",),
    "argument": ("arg", "args"), "parameter": ("param", "params"), "message": ("msg",),
    "object": ("obj",), "error": ("err",), "exception": ("exc",), "function": ("func", "fn"),
    "variable": ("var",), "temporary": ("tmp", "temp"), "number": ("num",),
    "string": ("str",), "dictionary": ("dict",), "integer": ("int",), "boolean": ("bool",),
    "attribute": ("attr",), "database": ("db",), "authentication": ("auth",),
    "authorization": ("auth",), "certificate": ("cert",), "specification": ("spec",),
    "template": ("tmpl",), "callback": ("cb",), "utility": ("util", "utils"),
    "administrator": ("admin",), "implementation": ("impl",), "reference": ("ref",),
    "previous": ("prev",), "source": ("src",), "destination": ("dst", "dest"),
    "length": ("len",), "index": ("idx",), "address": ("addr",), "document": ("doc",),
    "package": ("pkg",), "library": ("lib",), "value": ("val",), "command": ("cmd",),
    "password": ("pwd",),
}

# Turkish. Function words are dropped outright. The lexicon maps a stem prefix
# to English words (the longest stem the word starts with wins; an empty tuple
# means "filler"). Both apply only to a query that looks Turkish and only to
# words outside the index vocabulary, so an English or code word is never
# translated ("rotate" does not become "route", "silent" not "delete").
_TR_FILLER = frozenset(
    {
        "ne", "mi", "mı", "mu", "mü", "ve", "veya", "ile", "bu", "şu", "o", "bir", "da",
        "de", "ki", "ya", "için", "gibi", "var", "yok", "ise", "hem", "çok", "en", "daha",
        "kadar", "ler", "lar", "leri", "ları", "nin", "nın", "nun", "nün", "in", "ın",
        "un", "ün", "yi", "yı", "yu", "yü", "ye", "den", "dan", "ten", "tan", "deki",
        "daki", "dir", "dır", "tir", "tır", "nasıl", "neden", "niye", "niçin", "hangi",
        "hangisi", "orada", "burada", "şurada", "içinde", "olarak",
    }
)
_TR_LEXICON: dict[str, tuple[str, ...]] = {
    # question words and words that only describe the question
    "nere": (), "tanım": (), "fonksiyon": (), "metod": (), "metot": (), "modül": (),
    "kod": (), "dosya": (), "kullan": (), "yapıl": (), "edil": (), "ediyor": (),
    "olan": (), "oluyor": (), "gel": (), "döndür": (), "dönüş": (), "göster": (),
    "sınıf": ("class",),
    # code vocabulary
    "uygulama": ("app", "application"), "uygula": ("apply",), "fabrika": ("factory",),
    "örnek": ("example",), "öğretici": ("tutorial",), "şablon": ("template",),
    "değişken": ("variable",), "değer": ("value",), "otomatik": ("automatic", "auto"),
    "nesne": ("object",), "metin": ("text", "string", "str"), "metn": ("text", "string", "str"),
    "yanıt": ("response",), "cevap": ("response",), "istek": ("request",),
    "oturum": ("session",), "çerez": ("cookie",), "imza": ("sign", "signature"),
    "kayd": ("register",), "kayıt": ("register",), "ekle": ("add",), "eklenti": ("extension",),
    "çağır": ("call",), "çağr": ("call",), "bul": ("find",), "çalıştır": ("run",),
    "çalış": ("run",), "çevir": ("convert",), "çevri": ("convert",),
    "dönüştür": ("convert",),
    "hata": ("error",), "istisna": ("exception",), "işleyici": ("handler",),
    "işle": ("handle", "process"), "yönlendir": ("redirect",), "rota": ("route",),
    "adres": ("url", "address"), "ayar": ("config", "setting"),
    "yapılandır": ("config", "configure"), "ortam": ("environment",), "önek": ("prefix",),
    "komut": ("command",), "parola": ("password",), "şifre": ("password",),
    "kullanıcı": ("user",), "veritabanı": ("database",), "bağlam": ("context",),
    "statik": ("static",), "tarih": ("date",), "zaman": ("time",), "sayı": ("number",),
    "liste": ("list",), "sözlük": ("dict",), "dizin": ("directory",), "klasör": ("folder",),
    "başlat": ("start", "init"), "başla": ("start",), "kapat": ("close",), "oku": ("read",),
    "yaz": ("write",), "sorgu": ("query",), "eşleş": ("match",), "eşle": ("match",),
    "derle": ("compile",), "ayrıştır": ("parse",), "doğrula": ("validate", "verify"),
    "giriş": ("login",), "çıkış": ("logout",), "görünüm": ("view",), "olay": ("event",),
    "sinyal": ("signal",), "önbellek": ("cache",), "başlık": ("header",), "gövde": ("body",),
    "durum": ("status",), "sunucu": ("server",), "istemci": ("client",), "yol": ("path",),
    "argüman": ("argument",), "parametre": ("parameter",), "önce": ("before",),
    "sonra": ("after",), "temiz": ("cleanup",), "seç": ("select", "choose"),
    "sağlayıcı": ("provider",), "serileştir": ("serialize",), "gizli": ("secret",),
    "anahtar": ("key",), "gönder": ("send",), "yükle": ("load",), "oluştur": ("create",),
    "yarat": ("create",), "sil": ("delete", "remove"), "güncelle": ("update",),
    "bağlantı": ("connection", "link"), "içerik": ("content",), "biçim": ("format",),
    "doğrulama": ("validation", "auth"),
}
_TR_MIN_STEM = 3
# A query is treated as Turkish when it has a Turkish letter or one of these words.
_TR_LETTERS = frozenset("çğıöşüÇĞİÖŞÜ")
_TR_HINTS = frozenset({"nerede", "nereden", "nasil", "hangi", "neden", "icin", "ile", "veya"})
# Final consonants that soften before a vowel suffix: örnek -> örneği, kayıt -> kaydı.
_TR_SOFTEN = {"k": "ğ", "t": "d", "p": "b", "ç": "c"}


def tokenize(text: str) -> list[str]:
    """Split identifiers and prose into lowercase search tokens.

    snake_case, camelCase, dots and slashes are split; tokens shorter than two
    characters and stand-alone :data:`STOPWORDS` are dropped. A compound
    identifier also yields itself: ``url_for`` -> ``url``, ``for``, ``url_for``;
    ``getURL`` -> ``get``, ``url``, ``geturl``. Non-ASCII words ("tanımlı") are
    kept whole.
    """
    return _tokenize(text, STOPWORDS)


def _lower(text: str) -> str:
    return text.replace("İ", "i").lower()


def _pieces(word: str) -> list[str]:
    """Lowercase snake/camel parts of one word (any length)."""
    if word.isascii():
        return [p.lower() for chunk in word.split("_") if chunk for p in _CAMEL_RE.findall(chunk)]
    return [_lower(chunk) for chunk in word.split("_") if chunk]


def _tokenize(text: str, stop: frozenset[str]) -> list[str]:
    out: list[str] = []
    for word in _WORD_RE.findall(text):
        pieces = _pieces(word)
        if len(pieces) > 1:
            out.extend(p for p in pieces if len(p) >= 2)
            whole = _lower(word.strip("_"))
            if len(whole) >= 2:
                out.append(whole)
        elif pieces:
            piece = pieces[0]
            if len(piece) >= 2 and piece not in stop:
                out.append(piece)
    return out


_VOWELS = frozenset("aeiouy")


def stem(word: str) -> str:
    """A light English stemmer: ``signing``/``signed``/``signs`` -> ``sign``,
    ``builder`` -> ``build``, ``registered`` and ``register`` -> ``regist``.

    It only has to give the inflections of one word the same key; the key need
    not be a word. Words shorter than four letters, tokens with digits or
    underscores and non-ASCII words are returned unchanged. ``-er`` is only
    removed from words of seven letters or more ("parser", "server" stay).
    """
    w = word
    if len(w) < 4 or not w.isalpha() or not w.isascii():
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if w.endswith("ing") and len(w) >= 6 and _VOWELS & set(w[:-3]):
        w = w[:-3]
    elif w.endswith("ed") and len(w) >= 5 and _VOWELS & set(w[:-2]):
        w = w[:-2]
    elif w.endswith("ed") and len(w) == 4:
        w = w[:-1]  # used -> use
    if w.endswith("er") and len(w) >= 7:
        w = w[:-2]
    if len(w) >= 4 and w[-1] == w[-2] and w[-1] not in _VOWELS and w[-1] not in "lsz":
        w = w[:-1]  # running -> runn -> run
    if w.endswith("e") and len(w) >= 4:
        w = w[:-1]
    return w


def _name_parts(name: str) -> tuple[str, ...]:
    parts = tuple(p for p in _pieces(name) if len(p) >= 2)
    core = name.strip("_")
    return parts or ((_lower(core),) if core else ())


def _whole(name: str) -> str:
    """The whole-identifier token of a name, as :func:`tokenize` emits it."""
    core = name.strip("_")
    pieces = _pieces(core)
    if len(pieces) > 1:
        return _lower(core)
    return pieces[0] if pieces else ""


def _is_compound(word: str) -> bool:
    return "_" in word.strip("_") or len(_pieces(word)) > 1


def _with_stems(parts: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((p, stem(p)) for p in parts)


def _abbreviation_table() -> dict[str, tuple[str, ...]]:
    table: dict[str, tuple[str, ...]] = {}
    for word, abbrs in _ABBREVIATIONS.items():
        table[word] = abbrs
        table.setdefault(stem(word), abbrs)
    return table


_ABBR = _abbreviation_table()


def _abbreviations(word: str) -> tuple[str, ...]:
    return _ABBR.get(word) or _ABBR.get(stem(word)) or ()


def _tr_lexicon() -> dict[str, tuple[str, ...]]:
    table = dict(_TR_LEXICON)
    for stem_, english in _TR_LEXICON.items():
        soft = _TR_SOFTEN.get(stem_[-1])
        if soft:
            table.setdefault(stem_[:-1] + soft, english)
    return table


_TR = _tr_lexicon()


def _translate_tr(word: str) -> Optional[tuple[str, ...]]:
    """English words for a Turkish ``word`` (empty: filler), or None if unknown."""
    if word in _TR_FILLER:
        return ()
    for end in range(len(word), _TR_MIN_STEM - 1, -1):
        hit = _TR.get(word[:end])
        if hit is not None:
            return hit
    return None


def _looks_turkish(query: str, words: list[str]) -> bool:
    return any(ch in _TR_LETTERS for ch in query) or any(w in _TR_HINTS for w in words)


def _path_text(path: str) -> str:
    """Path without its extension, for tokenizing (``src/flask/app.py`` -> ``src/flask/app``)."""
    head, _, tail = path.rpartition("/")
    base = tail.rsplit(".", 1)[0] if "." in tail else tail
    return f"{head}/{base}" if head else base


def _file_stem(path: str) -> str:
    """``pkg/mod.py`` -> ``mod``; ``pkg/sub/__init__.py`` -> ``sub``."""
    parts = path.split("/")
    base = parts[-1].rsplit(".", 1)[0]
    if base == "__init__" and len(parts) > 1:
        base = parts[-2]
    return base.lower()


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
    name: str                  # display name: symbol name, or file stem
    qualname: str              # lowercase qualname without <locals>; "" for files
    dotted: str                # ".module.qualname." lowercase, dot-delimited; "" if unknown
    slashed: str               # "/path" lowercase
    is_test: bool
    is_class: bool
    is_nested: bool
    whole: str                 # whole-name token: "url_for", "flaskclient"
    parts: tuple[tuple[str, str], ...]          # (part, stem) of the name
    owner: str = ""                             # enclosing class/function name
    owner_whole: str = ""
    owner_parts: tuple[tuple[str, str], ...] = ()


@dataclass
class _Unit:
    text: str                  # the query word(s) as written, lowercase
    weight: float


@dataclass
class _Query:
    tokens: list[str]                  # BM25 tokens
    labels: dict[str, str]             # BM25 token -> the query word it stands for
    units: list[_Unit]
    keys: dict[str, frozenset[int]]    # name part / stem / abbreviation -> units it covers
    total: float                       # sum of unit weights
    compounds: list[str]               # dotted/slashed terms, lowercase
    mentions_tests: bool
    wants_class: bool


class SearchIndex:
    """BM25 index over the symbols and files of one repository."""

    def __init__(self, index: RepoIndex, summaries: Optional[dict[str, str]] = None) -> None:
        self.index = index
        summaries = summaries or {}
        modules = module_names(f.path for f in index.files)

        self._docs: list[_Doc] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._df: Counter[str] = Counter()
        self._stems: dict[str, str] = {}

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
            owner = qual.rsplit(".", 2)[-2] if "." in qual else ""
            self._add(
                tokens,
                _Doc(
                    kind="symbol",
                    path=sym.path,
                    symbol=sym,
                    end_line=sym.end_line,
                    length=0,
                    name=sym.name,
                    qualname=qual.lower(),
                    dotted=f".{module}.{qual}.".lower() if module else f".{qual}.".lower(),
                    slashed="/" + sym.path.lower(),
                    is_test=is_test_path(sym.path),
                    is_class=sym.kind == "class",
                    is_nested="<locals>" in sym.qualname,
                    whole=_whole(sym.name),
                    parts=_with_stems(_name_parts(sym.name)),
                    owner=owner,
                    owner_whole=_whole(owner) if owner else "",
                    owner_parts=_with_stems(_name_parts(owner)) if owner else (),
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
            file_stem = _file_stem(record.path)
            self._add(
                tokens,
                _Doc(
                    kind="file",
                    path=record.path,
                    symbol=None,
                    end_line=max(record.n_lines, 1),
                    length=0,
                    name=file_stem,
                    qualname="",
                    dotted=f".{module}.".lower() if module else "",
                    slashed="/" + record.path.lower(),
                    is_test=is_test_path(record.path),
                    is_class=False,
                    is_nested=False,
                    whole=_whole(file_stem),
                    parts=_with_stems(_name_parts(file_stem)),
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
        """Index ``tokens`` plus the stem of every simple token whose stem differs."""
        stems = []
        for tok in tokens:
            if "_" not in tok:
                s = self._stems.get(tok)
                if s is None:
                    s = self._stems[tok] = stem(tok)
                if s != tok:
                    stems.append(s)
        tokens = tokens + stems
        doc.length = len(tokens)
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
        q = self._analyze(query)
        if not q.tokens:
            return []

        # BM25 accumulation over the postings of the query tokens only.
        scores: dict[int, float] = {}
        for tok in q.tokens:
            postings = self._postings.get(tok)
            if not postings:
                continue
            idf = self._idf[tok]
            for doc_id, tf in postings:
                doc = self._docs[doc_id]
                norm = K1 * (1.0 - B + B * doc.length / self._avgdl[doc.kind])
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (K1 + 1.0) / (tf + norm)

        scored = [(base * self._boost(self._docs[d], q, None), d) for d, base in scores.items()]
        top = heapq.nsmallest(k, scored, key=lambda sd: (-sd[0], *self._order(sd[1])))

        # Explanations are built only for the hits that are returned.
        hits: list[Hit] = []
        for score, doc_id in top:
            doc = self._docs[doc_id]
            why: list[str] = []
            self._boost(doc, q, why)
            why.append("bm25: " + self._bm25_terms(doc_id, q))
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
        return hits

    def _order(self, doc_id: int) -> tuple[str, int, str, str]:
        doc = self._docs[doc_id]
        sym = doc.symbol
        if sym is None:
            return (doc.path, 1, doc.kind, "")
        return (doc.path, sym.start_line, doc.kind, sym.id)

    # ----------------------------------------------------------------- boosts

    def _boost(self, doc: _Doc, q: _Query, why: Optional[list[str]]) -> float:
        """Product of the boosts that apply to ``doc``; explanations go to ``why`` if given."""
        f = 1.0
        if doc.kind == "symbol":
            name_f, name_exp, name_units = self._name_factor(doc, q, NAME_BOOST, "name match", why)
            f *= name_f
            if name_exp > 0.0 and doc.owner_parts:
                owner_units, owner_cov = self._cover(doc.owner_whole, doc.owner_parts, q)
                if owner_units - name_units:
                    g = OWNER_BOOST ** (owner_cov * name_exp)
                    f *= g
                    if why is not None:
                        why.append(f"owner match: {doc.owner} x{g:.2f}")
            term = _compound_match(doc, q.compounds)
            if term is not None:
                f *= QUALNAME_BOOST
                if why is not None:
                    why.append(f"qualname/path match: {term}")
            if doc.is_class:
                f *= CLASS_BOOST
                if why is not None:
                    why.append("class boost")
                if q.wants_class:
                    f *= KIND_BOOST
                    if why is not None:
                        why.append(f"class asked x{KIND_BOOST:g}")
            if doc.is_nested:
                f *= NESTED_PENALTY
                if why is not None:
                    why.append(f"nested x{NESTED_PENALTY:g}")
        else:
            name_f, _exp, _units = self._name_factor(doc, q, FILE_NAME_BOOST, "path match", why)
            f *= name_f
            if name_f == 1.0:
                term = _compound_match(doc, q.compounds)
                if term is not None:
                    f *= FILE_NAME_BOOST
                    if why is not None:
                        why.append(f"path match: {term}")
        if doc.is_test and not q.mentions_tests:
            f *= TEST_PENALTY
            if why is not None:
                why.append(f"test file x{TEST_PENALTY:g}")
        return f

    @staticmethod
    def _cover(
        whole: str, parts: tuple[tuple[str, str], ...], q: _Query
    ) -> tuple[frozenset[int], float]:
        """Units a name covers, and the share of the name's parts found in the query."""
        units = q.keys.get(whole) if whole else None
        if units is not None:
            return units, 1.0
        if not parts:
            return frozenset(), 0.0
        covered: set[int] = set()
        n = 0
        for part, part_stem in parts:
            u = q.keys.get(part)
            if u is None and part_stem != part:
                u = q.keys.get(part_stem)
            if u is not None:
                n += 1
                covered |= u
        return frozenset(covered), n / len(parts)

    def _name_factor(
        self, doc: _Doc, q: _Query, boost: float, label: str, why: Optional[list[str]]
    ) -> tuple[float, float, frozenset[int]]:
        """``(boost ** exponent, exponent, covered units)`` for the name of ``doc``."""
        units, part_cov = self._cover(doc.whole, doc.parts, q)
        if not units or q.total <= 0.0:
            return 1.0, 0.0, units
        coverage = sum(q.units[u].weight for u in units) / q.total
        exponent = coverage * part_cov ** NAME_PART_EXP
        if exponent <= 0.0:
            return 1.0, 0.0, units
        factor = boost ** exponent
        if why is not None:
            if exponent >= 0.999:
                why.append(f"{label}: {doc.name}")
            else:
                covered = ", ".join(sorted({q.units[u].text for u in units}))
                why.append(f"{label}: {doc.name} x{factor:.2f} ({covered})")
        return factor, exponent, units

    def _bm25_terms(self, doc_id: int, q: _Query) -> str:
        """Matched query words, strongest first.

        ``word~`` marks a word that matched only through its stem or abbreviation.
        """
        doc = self._docs[doc_id]
        norm = K1 * (1.0 - B + B * doc.length / self._avgdl[doc.kind])
        groups: dict[str, list] = {}
        for tok in q.tokens:
            tf = self._tf(tok, doc_id)
            if not tf:
                continue
            label = q.labels.get(tok, tok)
            group = groups.setdefault(label, [0.0, False])
            group[0] += self._idf[tok] * tf * (K1 + 1.0) / (tf + norm)
            group[1] = group[1] or tok == label
        ordered = sorted(groups.items(), key=lambda kv: (-kv[1][0], kv[0]))
        return ", ".join(label if exact else label + "~" for label, (_s, exact) in ordered)

    def _tf(self, tok: str, doc_id: int) -> int:
        """Term frequency of ``tok`` in ``doc_id`` (postings are in doc-id order)."""
        postings = self._postings.get(tok)
        if not postings:
            return 0
        lo, hi = 0, len(postings)
        while lo < hi:
            mid = (lo + hi) // 2
            if postings[mid][0] < doc_id:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(postings) and postings[lo][0] == doc_id:
            return postings[lo][1]
        return 0

    # --------------------------------------------------------------- analysis

    def _analyze(self, query: str) -> _Query:
        q = self._analyze_with(query, STOPWORDS | QUERY_FILLER)
        if not q.tokens:  # only filler words, e.g. a function literally named "find"
            q = self._analyze_with(query, STOPWORDS)
        return q

    def _analyze_with(self, query: str, drop: frozenset[str]) -> _Query:
        raw = [_lower(w).strip("_") for w in _WORD_RE.findall(query)]
        turkish = _looks_turkish(query, raw)
        units: list[_Unit] = []
        keys: dict[str, set[int]] = {}
        labels: dict[str, str] = {}
        tokens: list[str] = []
        spellings: list[list[str]] = []      # per raw word: forms tried when joining
        word_unit: list[Optional[int]] = []  # per raw word: its unit, if any
        seen: dict[str, tuple[Optional[int], list[str]]] = {}

        def key(k: str, uid: Optional[int]) -> None:
            bucket = keys.setdefault(k, set())
            if uid is not None:
                bucket.add(uid)

        def emit(tok: str, label: str) -> None:
            tokens.append(tok)
            labels.setdefault(tok, label)

        for w in raw:
            if w not in seen:
                seen[w] = self._add_word(w, drop, turkish, units, key, emit)
            uid, forms = seen[w]
            word_unit.append(uid)
            spellings.append(forms)

        # Adjacent words joined into identifiers present in the index. Each run is
        # a unit of its own that weighs what its idf adds over its words' weights,
        # and is covered (with its words) by a name equal to or containing it.
        consumed: set[int] = set()
        for size in range(2, MAX_JOIN + 1):
            for i in range(len(raw) - size + 1):
                options = spellings[i : i + size]
                if any(not o for o in options):
                    continue
                found: list[str] = []
                for combo in product(*options):
                    for candidate in ("_".join(combo), "".join(combo)):
                        if candidate in self._postings and candidate not in found:
                            found.append(candidate)
                if not found:
                    continue
                members = {word_unit[j] for j in range(i, i + size)} - {None}
                excess = max(self._idf[c] for c in found) - sum(units[m].weight for m in members)
                jid = len(units)
                units.append(_Unit(" ".join(raw[i : i + size]), max(0.0, excess)))
                for candidate in found:
                    for uid in members | {jid}:
                        key(candidate, uid)
                    emit(candidate, candidate)
                consumed.update(range(i, i + size))

        return _Query(
            tokens=list(dict.fromkeys(tokens)),
            labels=labels,
            units=units,
            keys={k: frozenset(v) for k, v in keys.items()},
            total=sum(u.weight for u in units),
            compounds=[c.lower() for c in _COMPOUND_RE.findall(query)],
            mentions_tests=any(w in _TEST_WORDS for i, w in enumerate(raw) if i not in consumed),
            wants_class=bool(_CLASS_WORDS & set(raw)) and not _CLASS_BASED_RE.search(query),
        )

    def _add_word(
        self,
        w: str,
        drop: frozenset[str],
        turkish: bool,
        units: list[_Unit],
        key: Callable[[str, Optional[int]], None],
        emit: Callable[[str, str], None],
    ) -> tuple[Optional[int], list[str]]:
        """Turn one query word into a unit (or nothing). Returns its unit id and the
        spellings tried when joining it with its neighbours."""
        spellings = ([w] + [a for a in _abbreviations(w) if a != w]) if w else []
        forms = self._forms(w, drop, turkish) if len(w) >= 2 else None
        if forms is None:  # stopword, filler or too short: may still complete a name's parts
            if len(w) >= 2:
                key(w, None)
            return None, spellings
        uid = len(units)
        translated = forms != [w]
        unit_tokens: list[str] = []
        for form in forms:
            if _is_compound(form):
                for tok in _tokenize(form, drop):
                    unit_tokens.append(tok)
                    emit(tok, tok)
                key(_whole(form), uid)
                continue
            s = stem(form)
            if not translated:
                unit_tokens.append(form)
                emit(form, form)
            if s != form or translated:
                # A translation's inflection is arbitrary: it matches by its stem only.
                unit_tokens.append(s)
                emit(s, form)
            key(form, uid)
            key(s, uid)
            for a in _abbreviations(form):
                key(a, uid)
                if a in self._postings and a not in STOPWORDS:
                    unit_tokens.append(a)
                    emit(a, form)
        if translated:
            spellings += [f for f in forms if f != w and not _is_compound(f)]
        weight = max((self._idf[t] for t in unit_tokens if t in self._idf), default=0.0)
        units.append(_Unit(w, weight))
        return uid, spellings

    def _forms(self, word: str, drop: frozenset[str], turkish: bool) -> Optional[list[str]]:
        """The search forms of one query word, or None when it is dropped."""
        if _is_compound(word):
            return [word]
        if word in drop:
            return None
        if turkish and word not in self._postings:
            english = _translate_tr(word)
            if english is not None:
                forms = [e for e in english if e not in drop]
                return forms or None
        return [word]


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
