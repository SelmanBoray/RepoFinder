"""Lexical search over a :class:`RepoIndex`: BM25F plus a few code-aware boosts.

Every symbol and every file becomes one document with up to two fields:

* symbol metadata: name, qualname, signature, docstring (JSDoc for
  TypeScript) and path components;
* symbol body (functions and methods only): the tokens of its source lines,
  read from the checkout (``index.root``; no checkout, no body). Words that
  only the code says ("priority", "Unity", "retryable") are found this way;
* file: path components, the names of the file's symbols and its summary.

Each field is length-normalized by its own mean (BM25F), so a long body does
not dilute a name match; a body occurrence weighs :data:`BODY_WEIGHT`.
Document frequencies count both fields; the vocabulary that decides Turkish
translation and joined identifiers is the metadata field only.

Every word is also indexed under a light English stem (:func:`stem`:
``signing`` -> ``sign``, ``loaded`` -> ``load``, ``builder`` -> ``build``), next
to the word itself.

A query is analysed into *units* (see :meth:`SearchIndex._analyze`):

* its content words. Natural-language filler ("where", "implemented"; Turkish
  "nerede", "tanımlı", ...) is dropped. In a query that looks Turkish, a word
  that is not in the index vocabulary is translated through a small
  stem-prefix lexicon ("kaydediliyor" -> register, "uygulamaya" -> app,
  application; two-word phrases such as "zaman aşımı" -> timeout) and matched
  by the translation's stem. Turkish typed in ASCII ("govdesi", "basligina")
  is matched too: words and lexicon keys are compared ASCII-folded;
* runs of adjacent words that, joined, form an identifier present in the index
  ("app context" -> ``app_context``/``appcontext``; with common abbreviations,
  "python file" -> ``pyfile``).

Each unit weighs its IDF (a joined run only what it adds over its words). The
units' tokens, their stems and abbreviations are scored with Okapi BM25 and the
score is multiplied by boosts. A token derived from a query word (its stem or
abbreviation) never weighs more than the word itself: in the requests repo
"request" is in nearly every document and its abbreviation "req" in four, so
"req" would otherwise outweigh the word the user wrote. Boosts:

* name: ``NAME_BOOST ** (coverage * part_coverage ** NAME_PART_EXP)``, where
  *coverage* is the share of the query's unit weight that the symbol name
  accounts for and *part_coverage* the share of the name's own parts
  (snake/camel pieces) found in the query. A name that is the whole query gets
  the full boost; a common word that happens to be a name ("loads", "session")
  in a longer question gets little; ``teardown_appcontext`` for "app context
  teardown" gets much. In test code only a full name match counts (tests are
  named after what they test), unless the query asks about tests;
* explicit identifier: a name the query writes as code (backticked, dotted
  ``Session.request``/``requests.get``, snake_case or camelCase) gets the full
  name boost when it is the only identifier named, however common its words
  are; the qualifier of a dotted name ("Session" in ``Session.request``) is not
  a name match by itself. A Rust path is its dotted form: ``Searcher::new`` is
  ``Searcher.new``, ``crate::printer::Standard`` is ``printer.Standard``;
* owner: the enclosing class is named in the query too (``make_response`` "on
  the Flask application object" -> ``Flask.make_response``), scaled by how
  strongly the name itself matched;
* qualname/path: a dotted or slashed query term names the symbol or file
  (``requests.get`` also names ``get`` in module ``requests.api``);
* class, kind ("which class ..."), test-code and nested-function adjustments
  (test code: a test file, or a Rust test inline in a source file);
* type declarations: a TypeScript ``interface`` or ``type`` alias is a data
  shape, rarely what "where is X done" asks for: x :data:`TYPE_PENALTY` unless
  the query mentions types or interfaces;
* auxiliary directories: code under ``examples/``, ``example/``, ``doc(s)/``,
  ``benchmark(s)/``, ``benches/``, ``scripts/`` or ``experiments/`` at the
  repository root or beside a package manifest (``packages/flame/example/``
  next to its ``pubspec.yaml``; ``src/foo/scripts/`` is product code)
  illustrates or maintains the product: x :data:`AUX_PENALTY` unless the
  query asks for examples, tutorials, docs, benchmarks or scripts ("örnek").
  A file the manifest declares as shipped (``package.json`` ``bin``/``main``,
  ``[project.scripts]``, Cargo ``[[bin]]``) is exempt; a test there gets the
  stronger of the two penalties, not both (see :class:`_AuxDirs`).

Query words keep their case for one decision: a lowerCamel word
(``ipcRenderer``) is code and splits into its parts like an identifier; an
UpperCamel word ("JavaScript", a class name) stays whole. Private class
members (``#name``) match by their bare name. In a Turkish query, Turkish
function words ("ve", "mi", "bu") are dropped even when some identifier makes
them index words, unless they are written as code.

Every boost that fires is named in :attr:`Hit.why`. Pure Python, standard
library only.
"""

from __future__ import annotations

import heapq
import json
import math
import os
import posixpath
import re
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Optional

from .graph import PATH_MODULE_EXTENSIONS, is_test_path, module_names, path_module_name, split_source_extension
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
TYPE_PENALTY = 0.7     # TypeScript interface/type alias declarations, unless the query asks for a type
AUX_PENALTY = 0.5      # examples/docs/benchmarks/scripts dirs (see _AuxDirs), unless the query asks for them
BODY_WEIGHT = 1.0      # BM25F weight of the body field (function/method source) vs. the metadata field
BODY_MAX_BYTES = 2 * 1024 * 1024  # files larger than this are not read for bodies
NESTED_PENALTY = 0.5   # closures ("outer.<locals>.inner") are rarely what a reader wants
MAX_JOIN = 3           # adjacent query words joined into identifiers: "test client" -> test_client

_WORD_RE = re.compile(r"\w+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
# "Flask.run", "flask/app.py", "sansio.scaffold.Scaffold.route"
_COMPOUND_RE = re.compile(r"[A-Za-z0-9_]+(?:[./][A-Za-z0-9_]+)+")
_LOCALS_RE = re.compile(r"\.?<locals>")
# Explicit identifiers: a backticked name, a dotted name ("Session.request",
# "requests.get(url)"), a snake_case or camelCase word.
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_DOTTED_RE = re.compile(r"(?<![\w./])[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+(?![\w/])")
_CAMEL_HUMP_RE = re.compile(r"[a-z0-9][A-Z]|[A-Z]{2}[a-z]")
# A Rust path in a query ("Searcher::new", "grep_searcher::Searcher",
# "crate::printer::Standard"): read as the dotted name it stands for, the way
# index qualnames are written (see :func:`_dot_rust_paths`). A path does not
# start inside a token, after a word character, ".", "/" or ":": in the pytest
# node id "tests/test_app.py::TestApp::test_run" no "::" is a Rust path.
_RUST_PATH_RE = re.compile(r"(?<![\w./:])[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+")
# A "::" at the edge of a word: trailing ("Searcher::"), lone (" :: "),
# leading ("::std") or before a turbofish ("collect::<").
_EDGE_PATH_SEPARATOR_RE = re.compile(r"(?<!\w)::|::(?!\w)")
# Leading path segments that only say where a Rust path starts.
_RUST_PATH_ROOTS = frozenset({"crate", "self", "super"})
# A dotted query term ending in one of these is a file name ("app.py",
# "index.ts"), not an identifier.
_FILE_EXTENSION_WORDS = frozenset({"py", "pyi", "ts", "tsx", "mts", "cts", "js", "jsx", "mjs", "cjs", "dart"})
# The other source extensions (Go, Rust, Java) are also common member names
# (``CameraTarget.go``, ``router.go``, ``wg.Go``): a term ending in one is a
# file name only when the index has a file of that name (``main.go``,
# ``lib.rs``, ``Gson.java``; see :func:`_explicit_identifiers`).
_INDEXED_FILE_EXTENSION_WORDS = frozenset(ext[1:] for ext in PATH_MODULE_EXTENSIONS) - _FILE_EXTENSION_WORDS
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

# A query word from this set lifts the test-code penalty, unless it is part of a
# joined identifier ("the test client" -> test_client is not a request for tests).
_TEST_WORDS = frozenset({"test", "tests", "testing", "pytest", "conftest", "fixture", "fixtures"})
_CLASS_WORDS = frozenset(
    {"class", "classes", "sınıf", "sınıfı", "sınıfını", "sınıflar", "sınıfları"}
)
# A query word from this set lifts the type-declaration penalty (TypeScript
# ``interface``/``type``), unless it is part of a joined identifier.
_TYPE_WORDS = frozenset(
    {
        "interface", "interfaces", "type", "types", "typed", "typing",
        "arayüz", "arayüzü", "arayüzleri", "tip", "tipi", "tür", "türü",
    }
)
_TYPE_KINDS = frozenset({"interface", "type"})

# Auxiliary directories (see :class:`_AuxDirs`): code that illustrates,
# documents, measures or maintains the product rather than being it.
# ``benches`` is Cargo's benchmark directory, ``experiments`` holds REA's
# throw-away measurement scripts.
_AUX_DIR_NAMES = frozenset(
    {"example", "examples", "doc", "docs", "benchmark", "benchmarks", "benches", "script", "scripts", "experiments"}
)
# A directory with one of these files is a package root: an auxiliary directory
# counts there and at the repository root only (``packages/flame/example/``
# beside ``packages/flame/pubspec.yaml``, not ``src/foo/scripts/``).
_MANIFEST_NAMES = (
    "pubspec.yaml", "package.json", "Cargo.toml", "go.mod", "pom.xml", "build.gradle",
    "build.gradle.kts", "pyproject.toml", "setup.py",
)
# A larger manifest is not read (REA's package.json, with a long "files" list, is 25 KB).
_MANIFEST_MAX_BYTES = 256 * 1024
# At most this many strings are taken from one JSON field (``exports`` maps).
_MANIFEST_MAX_STRINGS = 10_000
# The manifests that can declare a shipped file (see :func:`_shipped_paths`).
_SHIPPING_MANIFESTS = frozenset({"package.json", "pyproject.toml", "setup.py", "Cargo.toml"})
# A query word from this set (ASCII-folded) or starting with one of the Turkish
# prefixes lifts the auxiliary-directory penalty, unless it is part of a joined
# identifier. "for example" and the Turkish adverb "örneğin" (for instance) do
# not ask for examples.
_AUX_WORDS = frozenset(
    {
        "example", "examples", "tutorial", "tutorials", "sample", "samples", "demo", "demos",
        "doc", "docs", "documentation", "benchmark", "benchmarks", "bench", "benches",
        "script", "scripts", "experiment", "experiments",
    }
)
_AUX_TR_PREFIXES = ("ornek", "orneg", "ogretici", "belge", "dokuman", "betik", "betig")
_AUX_NOT_ASKED = frozenset({"ornegin"})

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
    "password": ("pwd",), "typescript": ("ts",), "javascript": ("js",),
}

# Turkish. Function words are dropped outright. The lexicon maps a stem prefix
# to English words (the longest stem the word starts with wins; an empty tuple
# means "filler"). Both apply only to a query that looks Turkish and only to
# words outside the index vocabulary, so an English or code word is never
# translated ("rotate" does not become "route", "silent" not "delete"). Words
# and keys are compared ASCII-folded (:func:`_fold`), so Turkish typed without
# its letters ("govdesi", "istegi", "basligina") matches too.
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
# Only unambiguous mappings: "bağlantı" is a connection here far more often
# than a link, and "işle" (handle, process, operate, ...) has no one sense.
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
    "yönlendir": ("redirect",), "rota": ("route",),
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
    "bağlantı": ("connection",), "içerik": ("content",), "biçim": ("format",),
    "doğrulama": ("validation", "auth"), "birleştir": ("merge",), "birleşti": ("merge",),
    "süreç": ("process",), "değiştir": ("change", "modify"), "aday": ("candidate",),
    "incele": ("inspect",), "seçenek": ("option",), "satır": ("line",), "ajan": ("agent",),
    "hedef": ("target",), "enjekte": ("inject",), "senaryo": ("scenario",), "göreli": ("relative",),
    "bağlan": ("connect", "attach"), "alanlar": ("field",), "oyun": ("game",),
}
# Two-word phrases: (first stem, second stem) -> English. The first word is
# then filler and the second carries the translation ("zaman aşımı" -> timeout,
# not time + exceed).
_TR_PHRASES: dict[tuple[str, str], tuple[str, ...]] = {
    ("zaman", "aşım"): ("timeout",), ("yeniden", "dene"): ("retry",),
    ("tekrar", "dene"): ("retry",), ("kimlik", "bilgi"): ("credential",),
    ("içe", "aktar"): ("import",), ("dışa", "aktar"): ("export",),
}
_TR_MIN_STEM = 3
# A query is treated as Turkish when it has a Turkish letter or one of these
# words (ASCII-folded).
_TR_LETTERS = frozenset("çğıöşüÇĞİÖŞÜ")
_TR_HINTS = frozenset(
    {
        "nerede", "nereden", "nasil", "hangi", "hangisi", "neden", "niye", "nicin", "icin",
        "icinde", "ile", "veya", "gibi", "kadar", "olarak", "orada", "burada", "tanimli",
    }
)
_TR_FOLD = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
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


def _bare(name: str) -> str:
    """A name without its outer underscores and JavaScript sigils:
    ``__init__`` -> ``init``, ``#selectAutomatically`` (a private class
    member) -> ``selectAutomatically``, ``$el`` -> ``el``."""
    return name.strip("_#$")


def _name_parts(name: str) -> tuple[str, ...]:
    parts = tuple(p for p in _pieces(name) if len(p) >= 2)
    core = _bare(name)
    return parts or ((_lower(core),) if core else ())


def _whole(name: str) -> str:
    """The whole-identifier token of a name, as :func:`tokenize` emits it."""
    core = _bare(name)
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


def _fold(word: str) -> str:
    """Turkish letters to their ASCII look-alikes: ``gövdesi`` -> ``govdesi``."""
    return word.translate(_TR_FOLD)


def _tr_lexicon() -> dict[str, tuple[str, ...]]:
    """The lexicon keyed by ASCII-folded stems, softened stems included."""
    table: dict[str, tuple[str, ...]] = {}
    for stem_, english in _TR_LEXICON.items():
        table.setdefault(_fold(stem_), english)
    for stem_, english in _TR_LEXICON.items():
        soft = _TR_SOFTEN.get(stem_[-1])
        if soft:
            table.setdefault(_fold(stem_[:-1] + soft), english)
    return table


_TR = _tr_lexicon()
_TR_PHRASES_FOLDED = {(_fold(a), _fold(b)): english for (a, b), english in _TR_PHRASES.items()}
_TR_FILLER_FOLDED = frozenset(_fold(w) for w in _TR_FILLER)
_CLASS_WORDS_FOLDED = frozenset(_fold(w) for w in _CLASS_WORDS)
_TYPE_WORDS_FOLDED = frozenset(_fold(w) for w in _TYPE_WORDS)


def _translate_tr(word: str) -> Optional[tuple[str, ...]]:
    """English words for a Turkish ``word`` (empty: filler), or None if unknown."""
    word = _fold(word)
    if word in _TR_FILLER_FOLDED:
        return ()
    for end in range(len(word), _TR_MIN_STEM - 1, -1):
        hit = _TR.get(word[:end])
        if hit is not None:
            return hit
    return None


def _tr_phrases(words: list[str]) -> dict[int, tuple[str, ...]]:
    """Positions of the words of Turkish phrases -> their translation (the
    first word of a phrase -> ``()``, filler)."""
    folded = [_fold(w) for w in words]
    out: dict[int, tuple[str, ...]] = {}
    for i in range(len(folded) - 1):
        for (first, second), english in _TR_PHRASES_FOLDED.items():
            if folded[i].startswith(first) and folded[i + 1].startswith(second):
                out[i], out[i + 1] = (), english
    return out


def _looks_turkish(query: str, words: list[str]) -> bool:
    return any(ch in _TR_LETTERS for ch in query) or any(_fold(w) in _TR_HINTS for w in words)


def _strip_extension(path: str) -> str:
    """``path`` without its source extension (``.py``, ``.ts``, ``.d.ts``, ...,
    see :func:`graph.split_source_extension`); any other extension is cut at
    its last dot."""
    stem, ext = split_source_extension(path)
    if ext:
        return stem
    head, _, tail = path.rpartition("/")
    base = tail.rsplit(".", 1)[0] if "." in tail else tail
    return f"{head}/{base}" if head else base


def _path_text(path: str) -> str:
    """Path without its extension, for tokenizing (``src/flask/app.py`` -> ``src/flask/app``)."""
    return _strip_extension(path)


def _file_stem(path: str) -> str:
    """``pkg/mod.py`` -> ``mod``; ``pkg/sub/__init__.py`` -> ``sub``;
    ``src/util.ts`` -> ``util``; ``src/util/index.ts`` -> ``util`` (an
    ``index`` file names its directory, as ``__init__.py`` does);
    ``src/printer/mod.rs`` -> ``printer`` (so does a Rust ``mod.rs``), while
    ``src/main.rs`` and ``src/lib.rs`` keep their stem. Files named by path
    follow :func:`graph.path_module_name` (``pkg/index.go`` stays ``index``)."""
    module = path_module_name(path)
    if module is not None:
        return module.rsplit("/", 1)[-1].lower()
    parts = _strip_extension(path).split("/")
    base = parts[-1]
    if len(parts) > 1 and base == "__init__":
        base = parts[-2]
    return base.lower()


# Symbols whose source lines are indexed as their body. A class body is the
# sum of its methods (each indexed on its own); an interface or type alias is
# a list of fields, which would make declarations match like implementations.
_BODY_KINDS = frozenset({"function", "method"})


class _SourceLines:
    """Reads checkout files once, as lines split on ``\\n`` (the indexer's
    line numbers). A missing root or file gives no text."""

    def __init__(self, root: str) -> None:
        self.root: Optional[Path] = None
        try:
            if root and Path(root).is_dir():
                self.root = Path(root).resolve()
        except OSError:
            pass
        self._lines: dict[str, list[str]] = {}

    def text(self, path: str, start: int, end: int) -> str:
        if self.root is None:
            return ""
        lines = self._lines.get(path)
        if lines is None:
            lines = self._lines[path] = self._read(self.root, path)
        return "\n".join(lines[max(start, 1) - 1 : end])

    @staticmethod
    def _read(root: Path, path: str) -> list[str]:
        full = (root / path).resolve()
        if not full.is_relative_to(root):
            return []
        try:
            with open(full, "rb") as handle:
                raw = handle.read(BODY_MAX_BYTES + 1)
        except OSError:
            return []
        if len(raw) > BODY_MAX_BYTES:
            return []
        return raw.decode("utf-8", errors="replace").lstrip("\ufeff").split("\n")


class _AuxDirs:
    """Which files lie in an auxiliary directory (:data:`_AUX_DIR_NAMES`).

    Such a directory counts at the repository root (``examples/``,
    ``scripts/``) or directly beside a package manifest
    (:data:`_MANIFEST_NAMES`: ``packages/flame/example/`` next to
    ``packages/flame/pubspec.yaml``); elsewhere (``src/foo/scripts/``) it is
    product code. Manifests are not indexed, so they are looked up in the
    checkout, once per directory that holds a candidate; without a checkout
    only the root rule applies.

    A file the manifest beside the directory declares as shipped is exempt
    (:func:`_shipped_paths`): it is part of the package, wherever it lives.
    """

    def __init__(self, root: str) -> None:
        self.root: Optional[Path] = None
        try:
            if root and Path(root).is_dir():
                self.root = Path(root).resolve()
        except OSError:
            pass
        self._manifests: dict[str, tuple[str, ...]] = {}
        self._shipped: dict[str, frozenset[str]] = {}

    def is_aux(self, path: str) -> bool:
        parts = path.split("/")
        for i, part in enumerate(parts[:-1]):
            if part.lower() not in _AUX_DIR_NAMES:
                continue
            home = "/".join(parts[:i])
            if i and not self._manifests_in(home):
                continue
            # The outermost auxiliary directory decides: a shipped file is
            # exempt, anything else under it is auxiliary.
            return path not in self._shipped_from(home)
        return False

    def _dir(self, rel: str) -> Optional[Path]:
        """The checkout directory ``rel`` (a lexical join: listing names is
        harmless; a manifest is resolved before it is read)."""
        if self.root is None:
            return None
        parts = rel.split("/") if rel else []
        if any(p in ("", ".", "..") or ":" in p for p in parts):
            return None
        return self.root.joinpath(*parts)

    def _manifests_in(self, rel: str) -> tuple[str, ...]:
        found = self._manifests.get(rel)
        if found is None:
            directory = self._dir(rel)
            present: set[str] = set()
            if directory is not None:
                try:
                    with os.scandir(directory) as entries:
                        present = {e.name for e in entries if e.name in _MANIFEST_NAMES and e.is_file()}
                except OSError:
                    pass
            found = self._manifests[rel] = tuple(n for n in _MANIFEST_NAMES if n in present)
        return found

    def _shipped_from(self, rel: str) -> frozenset[str]:
        shipped = self._shipped.get(rel)
        if shipped is None:
            directory = self._dir(rel)
            shipped = frozenset()
            if directory is not None:
                prefix = f"{rel}/" if rel else ""
                found: set[str] = set()
                for name in self._manifests_in(rel):
                    if name not in _SHIPPING_MANIFESTS:
                        continue
                    try:
                        manifest = (directory / name).resolve()
                    except (OSError, RuntimeError, ValueError):  # a link loop, a bad name
                        continue
                    if manifest.is_relative_to(self.root):  # not through a link out of the checkout
                        found.update(_shipped_paths(manifest))
                shipped = frozenset(prefix + p for p in found)
            self._shipped[rel] = shipped
        return shipped


def _shipped_paths(manifest: Path) -> set[str]:
    """Paths (relative to the manifest's directory) that a manifest declares
    as shipped. Only explicit file declarations count:

    * ``package.json``: ``bin``, ``main``, ``module``, ``browser``, ``types``,
      the targets of ``exports`` and the plain file entries of ``files`` (a
      directory or glob entry ships a whole tree, which says nothing about one
      file);
    * ``pyproject.toml``: the modules of ``[project.scripts]``,
      ``[project.gui-scripts]`` and ``[tool.poetry.scripts]`` (``pkg.cli:main``
      -> ``pkg/cli.py``, ``pkg/cli/__init__.py``, also under ``src/``) and
      ``[tool.setuptools] script-files``;
    * ``setup.py``: ``scripts=[...]`` and ``console_scripts`` modules;
    * ``Cargo.toml``: the ``path`` of ``[lib]`` and ``[[bin]]`` (``[[example]]``
      and ``[[bench]]`` do not ship).

    TOML manifests are read by line (:func:`_toml_strings`), so the result is
    the same on every Python version. Repositories are untrusted: a manifest
    larger than :data:`_MANIFEST_MAX_BYTES`, unreadable, undecodable, too
    deeply nested or of an unexpected shape declares nothing, and reading it
    never raises.
    """
    if manifest.name not in _SHIPPING_MANIFESTS:
        return set()
    try:
        with open(manifest, "rb") as handle:
            raw = handle.read(_MANIFEST_MAX_BYTES + 1)
        if len(raw) > _MANIFEST_MAX_BYTES:
            return set()
        return _declared_paths(manifest.name, raw.decode("utf-8", errors="replace").lstrip("\ufeff"))
    except Exception:  # noqa: BLE001 - OSError, RecursionError, odd shapes: no exemption
        return set()


def _declared_paths(name: str, text: str) -> set[str]:
    """The shipped paths of manifest ``name`` with contents ``text`` (see :func:`_shipped_paths`)."""
    files: list[str] = []
    modules: list[str] = []
    if name == "package.json":
        data = json.loads(text)
        if isinstance(data, dict):
            for field in ("bin", "main", "module", "browser", "types", "exports"):
                files.extend(_json_strings(data.get(field)))
            entries = data.get("files")
            if isinstance(entries, list):
                files.extend(e for e in entries if isinstance(e, str))
    elif name == "pyproject.toml":
        for key, values in _toml_strings(text).items():
            if (len(key) == 3 and key[:2] in (("project", "scripts"), ("project", "gui-scripts"))) or (
                len(key) == 4 and key[:3] == ("tool", "poetry", "scripts")
            ):
                modules.extend(v.split(":", 1)[0] for v in values)
            elif key == ("tool", "setuptools", "script-files"):
                files.extend(values)
    elif name == "setup.py":
        listed = re.search(r"\bscripts\s*=\s*\[", text)
        if listed:
            end = text.find("]", listed.end())
            if end >= 0:
                files.extend(re.findall(r"['\"]([^'\"\n]+)['\"]", text[listed.end() : end]))
        modules.extend(re.findall(r"['\"][\w.-]+\s*=\s*([\w.]+)\s*:", text))
    elif name == "Cargo.toml":
        toml = _toml_strings(text)
        files.extend(toml.get(("lib", "path"), ()))
        files.extend(toml.get(("bin", "path"), ()))
    out: set[str] = set()
    for f in files:
        if any(ch in f for ch in "*?[!"):
            continue
        norm = posixpath.normpath(f.replace("\\", "/"))
        if norm and not norm.startswith(("..", "/")) and norm != ".":
            out.add(norm)
    for module in modules:
        module = module.strip()
        if re.fullmatch(r"\w+(?:\.\w+)*", module):
            base = module.replace(".", "/")
            for root in ("", "src/"):
                out.add(f"{root}{base}.py")
                out.add(f"{root}{base}/__init__.py")
    return out


def _json_strings(value: object) -> list[str]:
    """Every string inside a JSON value (``bin`` maps, nested ``exports``),
    at most :data:`_MANIFEST_MAX_STRINGS`; iterative, so depth does not matter."""
    out: list[str] = []
    stack = [value]
    while stack and len(out) < _MANIFEST_MAX_STRINGS:
        item = stack.pop()
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            stack.extend(reversed(list(item.values())))
        elif isinstance(item, list):
            stack.extend(reversed(item))
    return out


# A line-based reader for the few TOML keys of :func:`_shipped_paths`; not a
# TOML parser (compare ``parser_rust._parse_manifest``). Each line is split
# into tokens once, left to right, by anchored matches that consume what they
# match, so reading takes linear time whatever the manifest holds.
_TOML_STRING_RE = re.compile(r""""((?:[^"\\\n]|\\.)*)"|'([^'\n]*)'""")
_TOML_BARE_RE = re.compile(r"[A-Za-z0-9_-]+")
_Token = tuple[str, str]  # ("s", string value) | ("k", bare word) | ("p", one other character)


def _toml_unescape(text: str) -> str:
    return re.sub(r"\\(.)", r"\1", text)


def _toml_tokens(line: str) -> list[_Token]:
    """The tokens of one line, whitespace skipped. A comment ends the line, and
    so does an unterminated (or multi-line) string."""
    out: list[_Token] = []
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        if ch in " \t\r":
            i += 1
        elif ch in "\"'":
            m = _TOML_STRING_RE.match(line, i)
            if m is None:
                break
            out.append(("s", _toml_unescape(m.group(1)) if m.group(1) is not None else m.group(2)))
            i = m.end()
        elif ch == "#":
            break
        else:
            m = _TOML_BARE_RE.match(line, i)
            if m is not None:
                out.append(("k", m.group()))
                i = m.end()
            else:
                out.append(("p", ch))
                i += 1
    return out


def _toml_key_at(tokens: list[_Token], i: int) -> tuple[tuple[str, ...], int]:
    """The dotted key (``tool."poetry".scripts``) that starts at ``tokens[i]``:
    its parts and the index after it; no parts if none starts there."""
    parts: list[str] = []
    while i < len(tokens) and tokens[i][0] in ("k", "s"):
        parts.append(tokens[i][1])
        if i + 2 < len(tokens) and tokens[i + 1] == ("p", ".") and tokens[i + 2][0] in ("k", "s"):
            i += 2
        else:
            i += 1
            break
    return tuple(parts), i


def _toml_header(tokens: list[_Token]) -> Optional[tuple[str, ...]]:
    """The key of a ``[table]`` or ``[[array]]`` header line, else None."""
    depth = 0
    while depth < 2 and depth < len(tokens) and tokens[depth] == ("p", "["):
        depth += 1
    if not depth:
        return None
    key, i = _toml_key_at(tokens, depth)
    if key and tokens[i:] == [("p", "]")] * depth:
        return key
    return None


def _toml_value(tokens: list[_Token], strings: list[str]) -> int:
    """Append the strings of a value's tokens to ``strings``; return their
    bracket balance (``[``/``{`` minus ``]``/``}``)."""
    balance = 0
    for kind, text in tokens:
        if kind == "s":
            strings.append(text)
        elif kind == "p" and text in "[{":
            balance += 1
        elif kind == "p" and text in "]}":
            balance -= 1
    return balance


def _toml_inline_pairs(tokens: list[_Token]) -> list[tuple[tuple[str, ...], str]]:
    """``key = "string"`` pairs directly inside the inline table that
    ``tokens`` open with ``{`` (nested tables and arrays are skipped)."""
    pairs: list[tuple[tuple[str, ...], str]] = []
    depth = 0
    i, n = 0, len(tokens)
    while i < n:
        kind, text = tokens[i]
        if kind == "p" and text in "[{":
            depth += 1
            i += 1
            continue
        if kind == "p" and text in "]}":
            depth -= 1
            if depth <= 0:
                break
            i += 1
            continue
        if depth != 1 or (kind == "p" and text == ","):
            i += 1
            continue
        key, j = _toml_key_at(tokens, i)
        if (
            key
            and j + 1 < n
            and tokens[j] == ("p", "=")
            and tokens[j + 1][0] == "s"
            and (j + 2 == n or tokens[j + 2] in (("p", ","), ("p", "}")))
        ):
            pairs.append((key, tokens[j + 1][1]))
            i = j + 2
        else:
            i = max(j, i + 1)
    return pairs


def _toml_strings(text: str) -> dict[tuple[str, ...], list[str]]:
    """Full dotted key -> the strings assigned to it, by line.

    ``[table]`` and ``[[array]]`` headers set the key prefix (every
    ``[[bin]]`` adds to ``("bin", "path")``); dotted and quoted keys are
    split; a value array may span lines; a one-line inline table
    (``gui-scripts = { gui = "pkg.gui:main" }``) gives one key per string
    pair. Multi-line strings are skipped, so a header inside one is no header.
    Nested inline tables and arrays of inline tables are not read.
    """
    out: dict[tuple[str, ...], list[str]] = {}
    table: tuple[str, ...] = ()
    open_array: Optional[tuple[tuple[str, ...], int]] = None  # key and bracket balance of a value spanning lines
    open_string: Optional[str] = None  # delimiter of a multi-line string being skipped
    for line in text.splitlines():
        if open_array is not None:
            key, balance = open_array
            balance += _toml_value(_toml_tokens(line), out[key])
            open_array = (key, balance) if balance > 0 else None
            continue
        if open_string is not None:
            if line.count(open_string) % 2:
                open_string = None
            continue
        opened = next((d for d in ('"""', "'''") if line.count(d) % 2), None)
        if opened is not None:
            open_string = opened
            continue
        tokens = _toml_tokens(line)
        header = _toml_header(tokens)
        if header is not None:
            table = header
            continue
        name, i = _toml_key_at(tokens, 0)
        if not name or i >= len(tokens) or tokens[i] != ("p", "="):
            continue
        key = table + name
        value = tokens[i + 1 :]
        if value[:1] == [("p", "{")]:
            for pair_key, pair_value in _toml_inline_pairs(value):
                out.setdefault(key + pair_key, []).append(pair_value)
            continue
        balance = _toml_value(value, out.setdefault(key, []))
        if balance > 0:
            open_array = (key, balance)
    return out


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
    length: int                # metadata tokens (stems included)
    name: str                  # display name: symbol name, or file stem
    qualname: str              # lowercase qualname without <locals>; "" for files
    dotted: str                # ".module.qualname." lowercase, dot-delimited; "" if unknown
    slashed: str               # "/path" lowercase
    is_test: bool
    is_class: bool
    is_type: bool              # a TypeScript interface or type alias
    is_nested: bool
    whole: str                 # whole-name token: "url_for", "flaskclient"
    parts: tuple[tuple[str, str], ...]          # (part, stem) of the name
    owner: str = ""                             # enclosing class/function name
    owner_whole: str = ""
    owner_parts: tuple[tuple[str, str], ...] = ()
    body_length: int = 0                        # body tokens (stems included)
    is_aux: bool = False                        # in an examples/docs/benchmarks/scripts dir (see _AuxDirs)


@dataclass
class _Unit:
    text: str                  # the query word(s) as written, lowercase
    weight: float


@dataclass
class _Explicit:
    text: str                   # the identifier, lowercase ("session.request")
    name: str                   # its last part as written, outer underscores stripped
    qualifiers: frozenset[int]  # units of the parts before the name ("session")


@dataclass
class _Query:
    tokens: list[str]                  # BM25 tokens
    labels: dict[str, str]             # BM25 token -> the query word it stands for
    idf: dict[str, float]              # BM25 token -> its IDF for this query (see _analyze_with)
    units: list[_Unit]
    keys: dict[str, frozenset[int]]    # name part / stem / abbreviation -> units it covers
    total: float                       # sum of unit weights
    compounds: list[str]               # dotted/slashed terms, lowercase
    explicit: list[_Explicit]          # identifiers the query names outright
    qualifiers: frozenset[int]         # units only written as the qualifier of a dotted identifier
    mentions_tests: bool
    wants_class: bool
    wants_type: bool                   # the query mentions types/interfaces (no type penalty)
    wants_aux: bool = False            # the query asks for examples/docs/scripts (no auxiliary-dir penalty)


class SearchIndex:
    """BM25 index over the symbols and files of one repository."""

    def __init__(self, index: RepoIndex, summaries: Optional[dict[str, str]] = None) -> None:
        self.index = index
        summaries = summaries or {}
        modules = module_names(f.path for f in index.files)

        self._docs: list[_Doc] = []
        # Metadata postings (name, qualname, signature, docstring, path; for a
        # file its path, symbol names and summary) are also the vocabulary
        # (translation, joined identifiers); body postings only add to BM25.
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._body_postings: dict[str, list[tuple[int, int]]] = {}
        self._df: Counter[str] = Counter()
        self._stems: dict[str, str] = {}
        source = _SourceLines(index.root) if BODY_WEIGHT > 0.0 else None

        names_by_file: dict[str, list[str]] = {}
        path_tokens: dict[str, list[str]] = {}
        aux_dirs = _AuxDirs(index.root)
        aux_paths: dict[str, bool] = {}
        for record in index.files:
            path_tokens[record.path] = tokenize(_path_text(record.path))
            aux_paths[record.path] = aux_dirs.is_aux(record.path)
        # Lowercase names of the indexed files whose extension is also a common
        # member name ("main.go"): such a query term is a file name only then.
        self._file_names = frozenset(
            name
            for name in (record.path.rsplit("/", 1)[-1].lower() for record in index.files)
            if name.rpartition(".")[2] in _INDEXED_FILE_EXTENSION_WORDS
        )

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
            body: list[str] = []
            if source is not None and sym.kind in _BODY_KINDS:
                text = source.text(sym.path, sym.start_line, sym.end_line)
                if text:
                    body = tokenize(text)
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
                    is_type=sym.kind in _TYPE_KINDS,
                    is_nested="<locals>" in sym.qualname,
                    whole=_whole(sym.name),
                    parts=_with_stems(_name_parts(sym.name)),
                    owner=owner,
                    owner_whole=_whole(owner) if owner else "",
                    owner_parts=_with_stems(_name_parts(owner)) if owner else (),
                    is_aux=aux_paths[sym.path] if sym.path in aux_paths else aux_dirs.is_aux(sym.path),
                ),
                body,
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
                    is_type=False,
                    is_nested=False,
                    whole=_whole(file_stem),
                    parts=_with_stems(_name_parts(file_stem)),
                    is_aux=aux_paths[record.path],
                ),
            )

        # Symbols and files have very different lengths: normalize each kind by
        # its own mean, and a body by the mean body length (BM25F: each field
        # is normalized by its own length, so a long body does not dilute the
        # name and signature matches).
        avgdl: dict[str, float] = {}
        for kind in ("symbol", "file"):
            lengths = [d.length for d in self._docs if d.kind == kind]
            avgdl[kind] = (sum(lengths) / len(lengths)) if lengths else 1.0
        bodies = [d.body_length for d in self._docs if d.body_length]
        avg_body = (sum(bodies) / len(bodies)) if bodies else 1.0
        self._norm = [1.0 - B + B * d.length / avgdl[d.kind] for d in self._docs]
        self._body_norm = [
            (1.0 - B + B * d.body_length / avg_body) / BODY_WEIGHT if d.body_length else 0.0
            for d in self._docs
        ]
        n_docs = len(self._docs)
        self._idf = {
            tok: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)) for tok, df in self._df.items()
        }
        self._idf_unseen = math.log(1.0 + (n_docs + 0.5) / 0.5)  # a word no document has

    def _add(self, tokens: list[str], doc: _Doc, body: Optional[list[str]] = None) -> None:
        """Index ``tokens`` and ``body`` (the tokens of the symbol's source
        lines), each plus the stem of every simple token whose stem differs."""
        tokens = tokens + self._stems_of(tokens)
        body = body + self._stems_of(body) if body else []
        doc.length = len(tokens)
        doc.body_length = len(body)
        doc_id = len(self._docs)
        self._docs.append(doc)
        counted = Counter(tokens)
        for tok, tf in counted.items():
            self._postings.setdefault(tok, []).append((doc_id, tf))
            self._df[tok] += 1
        for tok, tf in Counter(body).items():
            self._body_postings.setdefault(tok, []).append((doc_id, tf))
            if tok not in counted:
                self._df[tok] += 1

    def _stems_of(self, tokens: list[str]) -> list[str]:
        stems = []
        for tok in tokens:
            if "_" not in tok:
                s = self._stems.get(tok)
                if s is None:
                    s = self._stems[tok] = stem(tok)
                if s != tok:
                    stems.append(s)
        return stems

    # ------------------------------------------------------------------ query

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """Top ``k`` hits for ``query``: score desc, then path, then start line."""
        if k <= 0:
            return []
        q = self._analyze(query)
        if not q.tokens:
            return []

        # BM25F accumulation over the postings of the query tokens only.
        scores: dict[int, float] = {}
        for tok in q.tokens:
            idf = q.idf[tok]
            for doc_id, tf in self._term_frequencies(tok):
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (K1 + 1.0) / (tf + K1)

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

    def _term_frequencies(self, tok: str) -> list[tuple[int, float]]:
        """``(doc_id, tf)`` for ``tok``, each field's count divided by its
        length normalization and a body count weighted by BODY_WEIGHT (BM25F)."""
        norm = self._norm
        merged = {doc_id: tf / norm[doc_id] for doc_id, tf in self._postings.get(tok) or ()}
        body = self._body_postings.get(tok)
        if body:
            body_norm = self._body_norm
            for doc_id, tf in body:
                merged[doc_id] = merged.get(doc_id, 0.0) + tf / body_norm[doc_id]
        return list(merged.items())

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
            explicit = _explicit_match(doc, q.explicit)
            if explicit is not None:
                name_f, name_exp, name_units = self._explicit_factor(doc, q, explicit, why)
            else:
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
            if doc.is_type and not q.wants_type:
                f *= TYPE_PENALTY
                if why is not None:
                    why.append(f"type declaration x{TYPE_PENALTY:g}")
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
                why.append(f"test code x{TEST_PENALTY:g}")
        if doc.is_aux and not q.wants_aux:
            # A test of an example is penalised once, by the stronger of the
            # two penalties: both say "not product code".
            tested = doc.is_test and not q.mentions_tests
            g = min(1.0, AUX_PENALTY / TEST_PENALTY) if tested else AUX_PENALTY
            f *= g
            if why is not None and g < 1.0:
                why.append(f"auxiliary dir x{g:.2g}")
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
        # "Session" in "Session.request" says whose method is asked for: it
        # qualifies the name, it is not a name asked for itself.
        units = units - q.qualifiers
        if not units or q.total <= 0.0:
            return 1.0, 0.0, units
        coverage = sum(q.units[u].weight for u in units) / q.total
        exponent = coverage * part_cov ** NAME_PART_EXP
        if exponent <= 0.0:
            return 1.0, 0.0, units
        if exponent < 0.999 and doc.is_test and not q.mentions_tests:
            # Tests are named after what they test ("test_env_cert_bundles"): a
            # partial name match there says nothing the test penalty should undo.
            return 1.0, 0.0, units
        factor = boost ** exponent
        if why is not None:
            if exponent >= 0.999:
                why.append(f"{label}: {doc.name}")
            else:
                covered = ", ".join(sorted({q.units[u].text for u in units}))
                why.append(f"{label}: {doc.name} x{factor:.2f} ({covered})")
        return factor, exponent, units

    def _explicit_factor(
        self, doc: _Doc, q: _Query, explicit: _Explicit, why: Optional[list[str]]
    ) -> tuple[float, float, frozenset[int]]:
        """Name boost of a symbol the query names outright (see :func:`_explicit_match`).

        The only identifier a query names is what it asks about: full boost,
        however common its words are (``Session.request`` in the requests
        repo). A query that names several (```request```, ```session``` and
        ```g``` in templates) asks about something around them: the usual
        coverage decides, with the identifier's qualifiers counted as covered.
        """
        units, _part_cov = self._cover(doc.whole, doc.parts, q)
        units = units | explicit.qualifiers
        covered = sum(q.units[u].weight for u in units) / q.total if q.total > 0.0 else 0.0
        exponent = 1.0 if len(q.explicit) == 1 else min(covered, 1.0)
        factor = NAME_BOOST ** exponent
        if why is not None:
            if covered >= 0.999:
                why.append(f"name match: {doc.name}")
            elif exponent >= 0.999:
                why.append(f"name match: {doc.name} (explicit {explicit.text})")
            else:
                why.append(f"name match: {doc.name} x{factor:.2f} (explicit {explicit.text})")
        return factor, exponent, units

    def _bm25_terms(self, doc_id: int, q: _Query) -> str:
        """Matched query words, strongest first; after ``body:`` the words
        found only in the symbol's body.

        ``word~`` marks a word that matched only through its stem or abbreviation.
        """
        groups: dict[str, list] = {}
        for tok in q.tokens:
            meta = self._tf(tok, doc_id)
            body = self._tf(tok, doc_id, self._body_postings)
            if not meta and not body:
                continue
            tf = meta / self._norm[doc_id] + (body / self._body_norm[doc_id] if body else 0.0)
            label = q.labels.get(tok, tok)
            group = groups.setdefault(label, [0.0, False, False])
            group[0] += q.idf[tok] * tf * (K1 + 1.0) / (tf + K1)
            group[1] = group[1] or tok == label
            group[2] = group[2] or meta > 0
        ordered = sorted(groups.items(), key=lambda kv: (-kv[1][0], kv[0]))

        def words(in_meta: bool) -> str:
            return ", ".join(
                label if exact else label + "~" for label, (_s, exact, meta) in ordered if meta == in_meta
            )

        text, body = words(True), words(False)
        if body:
            return f"{text}; body: {body}" if text else f"body: {body}"
        return text

    def _tf(self, tok: str, doc_id: int, store: Optional[dict[str, list[tuple[int, int]]]] = None) -> int:
        """Term frequency of ``tok`` in ``doc_id`` in the metadata postings (or
        in ``store``); postings are in doc-id order."""
        postings = (self._postings if store is None else store).get(tok)
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
        query = _dot_rust_paths(query)
        written = [w.strip("_") for w in _WORD_RE.findall(query)]
        raw = [_lower(w) for w in written]
        turkish = _looks_turkish(query, raw)
        if turkish:
            # Turkish function words ("ve", "mi", "bu") are dropped even when
            # some identifier piece makes them index words, unless written as code.
            coded = {_lower(w) for span in _BACKTICK_RE.findall(query) for w in _WORD_RE.findall(span)}
            drop = drop | frozenset(w for w in raw if _fold(w) in _TR_FILLER_FOLDED and w not in coded)
        units: list[_Unit] = []
        keys: dict[str, set[int]] = {}
        labels: dict[str, str] = {}
        idf: dict[str, float] = {}
        tokens: list[str] = []
        spellings: list[list[str]] = []      # per raw word: forms tried when joining
        word_unit: list[Optional[int]] = []  # per raw word: its unit, if any
        seen: dict[str, tuple[Optional[int], list[str]]] = {}
        phrases = _tr_phrases(raw) if turkish else {}
        seen_phrase: dict[tuple[str, tuple[str, ...]], tuple[Optional[int], list[str]]] = {}

        def key(k: str, uid: Optional[int]) -> None:
            bucket = keys.setdefault(k, set())
            if uid is not None:
                bucket.add(uid)

        def emit(tok: str, label: str, cap: Optional[float] = None) -> None:
            # A token derived from a query word (its stem or abbreviation) is
            # never worth more than the word itself: "request" is in every
            # document of the requests repo, its abbreviation "req" in four.
            tokens.append(tok)
            labels.setdefault(tok, label)
            value = self._idf.get(tok, 0.0)
            if cap is not None:
                value = min(value, cap)
            idf[tok] = max(idf.get(tok, 0.0), value)

        for i, w in enumerate(raw):
            phrase = phrases.get(i) if w not in self._postings else None
            if phrase is not None:
                if (w, phrase) not in seen_phrase:
                    seen_phrase[w, phrase] = self._add_word(w, drop, turkish, units, key, emit, phrase, written[i])
                uid, forms = seen_phrase[w, phrase]
            else:
                if w not in seen:
                    seen[w] = self._add_word(w, drop, turkish, units, key, emit, None, written[i])
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
                if not members:
                    continue  # only function words: "may be" is not ``maybe``
                excess = max(self._idf[c] for c in found) - sum(units[m].weight for m in members)
                jid = len(units)
                units.append(_Unit(" ".join(raw[i : i + size]), max(0.0, excess)))
                for candidate in found:
                    for uid in members | {jid}:
                        key(candidate, uid)
                    emit(candidate, candidate)
                consumed.update(range(i, i + size))

        # A word written only as a qualifier ("session" in "Session.request")
        # does not make a name match by itself.
        explicit: list[_Explicit] = []
        qualifier_words: Counter[str] = Counter()
        for parts in _explicit_identifiers(query, self._file_names):
            lowered = [_lower(p) for p in parts]
            qualifier_words.update(lowered[:-1])
            quals = frozenset(seen[w][0] for w in lowered[:-1] if w in seen and seen[w][0] is not None)
            explicit.append(_Explicit(".".join(lowered), parts[-1], quals))
        word_count = Counter(raw)
        qualifiers = frozenset(
            uid
            for w, (uid, _forms) in seen.items()
            if uid is not None and qualifier_words[w] >= word_count[w]
        )
        return _Query(
            tokens=list(dict.fromkeys(tokens)),
            labels=labels,
            idf=idf,
            units=units,
            keys={k: frozenset(v) for k, v in keys.items()},
            total=sum(u.weight for u in units),
            compounds=[c.lower() for c in _COMPOUND_RE.findall(query)],
            explicit=explicit,
            qualifiers=qualifiers,
            mentions_tests=any(w in _TEST_WORDS for i, w in enumerate(raw) if i not in consumed),
            wants_class=bool(_CLASS_WORDS_FOLDED & {_fold(w) for w in raw})
            and not _CLASS_BASED_RE.search(query),
            wants_type=any(_fold(w) in _TYPE_WORDS_FOLDED for i, w in enumerate(raw) if i not in consumed),
            wants_aux=_asks_for_aux(raw, consumed),
        )

    def _add_word(
        self,
        w: str,
        drop: frozenset[str],
        turkish: bool,
        units: list[_Unit],
        key: Callable[[str, Optional[int]], None],
        emit: Callable[..., None],
        phrase: Optional[tuple[str, ...]] = None,
        written: str = "",
    ) -> tuple[Optional[int], list[str]]:
        """Turn one query word into a unit (or nothing). Returns its unit id and the
        spellings tried when joining it with its neighbours. ``phrase`` is the
        word's translation as part of a Turkish phrase; ``written`` the word as
        written (case kept), so that a camelCase word splits as code does."""
        spellings = ([w] + [a for a in _abbreviations(w) if a != w]) if w else []
        if phrase is not None:
            forms: Optional[list[str]] = [e for e in phrase if e not in drop] or None
        else:
            forms = self._forms(w, drop, turkish, written) if len(w) >= 2 else None
        if forms is None:  # stopword, filler or too short: may still complete a name's parts
            if len(w) >= 2:
                key(w, None)
            return None, spellings
        uid = len(units)
        translated = forms != [w] and forms != [written]
        unit_tokens: list[tuple[str, Optional[float]]] = []  # (token, IDF cap)

        def add(tok: str, label: str, cap: Optional[float] = None) -> None:
            unit_tokens.append((tok, cap))
            emit(tok, label, cap)

        for form in forms:
            if _is_compound(form):
                for tok in _tokenize(form, drop):
                    add(tok, tok)
                key(_whole(form), uid)
                continue
            s = stem(form)
            # Derived tokens are capped at the IDF of the word they come from
            # (a word no document has is not a cap: then the derived form is
            # the only way to match it).
            cap = self._idf.get(form, self._idf_unseen)
            if not translated:
                add(form, form)
            if s != form or translated:
                # A translation's inflection is arbitrary: it matches by its stem only.
                add(s, form, None if translated else cap)
            key(form, uid)
            key(s, uid)
            for a in _abbreviations(form):
                key(a, uid)
                if a in self._postings and a not in STOPWORDS:
                    add(a, form, cap)
        if translated:
            spellings += [f for f in forms if f != w and not _is_compound(f)]
        weight = max(
            (min(self._idf[t], cap if cap is not None else math.inf) for t, cap in unit_tokens if t in self._idf),
            default=0.0,
        )
        units.append(_Unit(w, weight))
        return uid, spellings

    def _forms(self, word: str, drop: frozenset[str], turkish: bool, written: str = "") -> Optional[list[str]]:
        """The search forms of one query word, or None when it is dropped.

        A compound word is searched as an identifier: its parts and its whole.
        ``written`` is the word with its case, so a lowerCamel word
        (``ipcRenderer``, code by its shape) splits into ``ipc``, ``renderer``
        and ``ipcrenderer`` exactly as the same word in code does. An
        UpperCamel word stays whole: in prose it is a product name
        ("JavaScript" is not "java" + "script") or a class the query names,
        which the explicit-identifier match handles."""
        if _is_compound(word):
            return [word]
        if written[:1].islower() and written.isascii() and _is_compound(written):
            return [written]
        if word in drop:
            return None
        if turkish and word not in self._postings:
            english = _translate_tr(word)
            if english is not None:
                forms = [e for e in english if e not in drop]
                return forms or None
        return [word]


def _asks_for_aux(words: list[str], consumed: set[int]) -> bool:
    """The query asks for examples, tutorials, docs, benchmarks or scripts
    (:data:`_AUX_WORDS`; Turkish "örnek" and its inflections, "öğretici",
    "belge", "doküman", "betik"), outside joined identifiers. "for example"
    and the adverb "örneğin" (for instance) are not such a request."""
    for i, w in enumerate(words):
        if i in consumed:
            continue
        folded = _fold(w)
        if folded in _AUX_WORDS:
            if folded.startswith("example") and i and words[i - 1] == "for":
                continue
            return True
        if folded not in _AUX_NOT_ASKED and folded.startswith(_AUX_TR_PREFIXES):
            return True
    return False


def _dot_rust_paths(query: str) -> str:
    """``query`` with its Rust paths written with dots, as index qualnames are.

    ``Searcher::new`` -> ``Searcher.new``, ``grep_searcher::Searcher`` ->
    ``grep_searcher.Searcher``; leading ``crate``, ``self`` and ``super``
    segments only say where the path starts and are dropped
    (``crate::printer::Standard`` -> ``printer.Standard``). A ``::`` at the
    edge of a word (trailing ``Searcher::``, lone, leading ``::std``, before a
    turbofish ``<``) becomes a space first. A ``::`` inside a token that is
    not a path stays as written: a pytest node id
    (``tests/test_app.py::test_run``) reads as it always did. A query without
    ``::`` is returned unchanged. Applying this twice changes nothing more.
    """
    if "::" not in query:
        return query

    def dotted(match: re.Match[str]) -> str:
        segments = match.group().split("::")
        while len(segments) > 1 and segments[0] in _RUST_PATH_ROOTS:
            del segments[0]
        return ".".join(segments)

    return _RUST_PATH_RE.sub(dotted, _EDGE_PATH_SEPARATOR_RE.sub(" ", query))


def _explicit_identifiers(query: str, file_names: frozenset[str] = frozenset()) -> list[tuple[str, ...]]:
    """Identifiers the query names outright, as dot-separated parts (outer
    underscores stripped, case kept).

    A backticked identifier (```abort```, ```Session.request```), a dotted name
    outside a path (``requests.get(url)``; not a file name such as ``app.py``
    or ``index.ts``, not "e.g.") and a
    snake_case or camelCase word (``url_for``, ``CaseInsensitiveDict``). Plain
    words, even backticked ones in a phrase (```flask run```), are not. A Rust
    path counts as its dotted form (``Searcher::new`` is ``Searcher.new``, see
    :func:`_dot_rust_paths`). A dotted name ending in ``go``, ``rs`` or
    ``java`` (:data:`_INDEXED_FILE_EXTENSION_WORDS`) is a file name only when
    it ends in one of ``file_names``, the lowercase names of such indexed files
    (``main.go`` in a Go repository; ``CameraTarget.go`` elsewhere is a method).
    """
    query = _dot_rust_paths(query)
    found: list[str] = []
    for span in _BACKTICK_RE.findall(query):
        span = span.strip()
        if span.endswith("()"):
            span = span[:-2]
        if _IDENTIFIER_RE.fullmatch(span):
            found.append(span)
    for term in _DOTTED_RE.findall(query):
        parts = term.split(".")
        # Not a file name ("app.py", "index.ts") nor an abbreviation ("e.g").
        if all(len(p) >= 2 for p in parts) and not _is_file_name(parts, file_names):
            found.append(term)
    for word in _WORD_RE.findall(query):
        if word.isascii() and ("_" in word.strip("_") or _CAMEL_HUMP_RE.search(word)):
            found.append(word)
    out: list[tuple[str, ...]] = []
    for name in found:
        parts = tuple(p.strip("_") for p in name.split("."))
        if all(parts) and parts not in out:
            out.append(parts)
    return out


def _is_file_name(parts: list[str], file_names: frozenset[str]) -> bool:
    """The dotted term ``parts`` is a file name: it ends in a source
    extension of :data:`_FILE_EXTENSION_WORDS`, or ends in one of
    ``file_names`` (see :func:`_explicit_identifiers`)."""
    ext = parts[-1].lower()
    if ext in _FILE_EXTENSION_WORDS:
        return True
    if ext not in _INDEXED_FILE_EXTENSION_WORDS or not file_names:
        return False
    return any(".".join(parts[i:]).lower() in file_names for i in range(len(parts) - 1))


def _explicit_match(doc: _Doc, explicit: list[_Explicit]) -> Optional[_Explicit]:
    """The explicit identifier that names this symbol, else None.

    The name must match as written, case included (```request``` is not the
    class ``Request``); a dotted identifier names only the symbols it
    qualifies (see :func:`_compound_match`).
    """
    if not explicit or doc.symbol is None:
        return None
    name = _bare(doc.symbol.name)
    for e in explicit:
        if e.name != name:
            continue
        if "." not in e.text or _compound_match(doc, [e.text]) is not None:
            return e
    return None


def _compound_match(doc: _Doc, compounds: list[str]) -> Optional[str]:
    """First dotted/slashed query term that names this document, else None.

    ``Flask.run`` matches qualname ``Flask.run`` and dotted ``flask.app.Flask.run``
    on dot boundaries (not ``Flask.run_simple``); so does ``flask.Flask.run``, a
    qualname prefixed by some of its module's parts (``requests.get`` names
    ``get`` in ``requests.api``, the way a package re-exports it, not
    ``Session.get``). ``flask/app.py`` or ``app.py`` matches the path on ``/``
    boundaries.
    """
    for term in compounds:
        if doc.qualname and term == doc.qualname:
            return term
        if doc.dotted and f".{term}." in doc.dotted:
            return term
        if doc.qualname and doc.dotted and "/" not in term and _module_qualified(term, doc):
            return term
        at = doc.slashed.find("/" + term)
        if at >= 0:
            end = at + 1 + len(term)
            if end == len(doc.slashed) or doc.slashed[end] in "/.":
                return term
    return None


def _module_qualified(term: str, doc: _Doc) -> bool:
    """``term`` is ``<some module parts>.<qualname>``, the parts in order."""
    if not term.endswith("." + doc.qualname):
        return False
    prefix = term[: -len(doc.qualname) - 1].split(".")
    module = iter(doc.dotted.strip(".")[: -len(doc.qualname) - 1].split("."))
    return all(part in module for part in prefix)
