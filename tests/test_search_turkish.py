"""Turkish query translation: where a lexicon stem ends and a suffix begins."""

from __future__ import annotations

from typing import Optional

import pytest

from repofinder.models import FileRecord, RepoIndex, Symbol
from repofinder.search import SearchIndex, _translate_tr


def sym(path: str, qualname: str, kind: str = "function", doc: Optional[str] = None) -> Symbol:
    name = qualname.rsplit(".", 1)[-1]
    parent = qualname.rsplit(".", 1)[0] if "." in qualname else None
    signature = f"class {name}" if kind == "class" else f"def {name}()"
    return Symbol(
        id=f"{path}::{qualname}", path=path, name=name, qualname=qualname, kind=kind,
        start_line=1, end_line=2, signature=signature, docstring=doc, parent=parent,
    )


def make_index(symbols: list[Symbol]) -> RepoIndex:
    paths = sorted({s.path for s in symbols})
    files = [FileRecord(p, "python", 100, 50, True) for p in paths]
    return RepoIndex(name="t", root="does-not-exist", source="t", files=files, symbols=symbols)


# --------------------------------------------------------------------------- #
# stem boundary: a key matches only before an inflection
# --------------------------------------------------------------------------- #


def test_oncelik_is_not_read_as_once() -> None:
    # "öncelik" (priority) starts with "önce" (before), but -lik makes another word.
    assert _translate_tr("öncelik") != _translate_tr("önce")
    assert "before" not in (_translate_tr("öncelik") or ())
    assert "before" not in (_translate_tr("oncelik") or ())  # typed without Turkish letters


def test_oncelik_query_finds_priority_not_before() -> None:
    index = make_index(
        [
            sym("pkg/queue.py", "sort_by_priority", doc="Order the jobs."),
            sym("pkg/hooks.py", "run_before_hooks", doc="Call the hooks first."),
        ]
    )
    hits = SearchIndex(index).search("İşlerin önceliği nerede belirleniyor?")
    assert hits and hits[0].symbol_id == "pkg/queue.py::sort_by_priority"
    assert "pkg/hooks.py::run_before_hooks" not in [h.symbol_id for h in hits]


@pytest.mark.parametrize(
    "word, english",
    [
        # old words, inflected
        ("uygulamaya", "app"),          # dative
        ("kaydediliyor", "register"),   # softened kayıt + passive progressive
        ("isteği", "request"),          # softened final consonant: istek -> isteğ-
        ("istegi", "request"),          # the same, typed in ASCII
        ("başlığına", "header"),        # softened + possessive + dative
        ("oturumlar", "session"),       # plural
        ("önbelleğe", "cache"),         # softened + dative
        ("önceden", "before"),          # ablative of önce still means before
        ("silinir", "delete"),          # passive aorist
        ("listeleniyor", "list"),       # verb-making -le keeps the meaning
        ("listeliyor", "list"),         # -le + progressive -iyor
        ("yüklüyor", "load"),           # yükle + -iyor narrows the final vowel
        # new words, inflected
        ("önceliği", "priority"),
        ("öncelikleri", "priority"),
        ("çerezleri", "cookie"),
        ("serileştiriliyor", "serialize"),
        ("ayrıştırılıyor", "parse"),
        ("ayrıştırıcı", "parser"),      # agent noun with a key of its own
    ],
)
def test_inflected_forms_translate(word: str, english: str) -> None:
    assert english in (_translate_tr(word) or ()), _translate_tr(word)


@pytest.mark.parametrize(
    "word, base_english",
    [
        ("hatasız", "error"),      # -siz: without errors
        ("yolcu", "path"),         # -cu: passenger, not a path
        ("silici", "delete"),      # -(y)ici agent noun, no key of its own
        ("sınırlı", "limit"),      # -lı adjective
        ("güvenlikçi", "security"),
        ("dalga", "branch"),       # "dal" + -ga is another root (wave)
        ("gösterge", "pointer"),
    ],
)
def test_derived_words_do_not_inherit_the_base_meaning(word: str, base_english: str) -> None:
    assert base_english not in (_translate_tr(word) or ()), _translate_tr(word)


def test_words_with_their_own_key_beat_the_base() -> None:
    assert _translate_tr("kullanıcı") == ("user",)
    assert _translate_tr("kodları") == ()                   # plural of kod stays filler
    assert _translate_tr("şifrele") == ("encrypt",)         # not password
    assert _translate_tr("şifreler") == ("password",)       # plural of şifre


@pytest.mark.parametrize(
    "word, english",
    [
        # passive -l + aorist -ir / -ince: not the derivation -li
        ("çevrilir", "convert"),
        ("çevrilirken", "convert"),
        ("çevrilince", "convert"),
        ("çevrilirse", "convert"),
        ("çevrilip", "convert"),
        # conditional -sa/-se + plural -lar/-ler: not the derivation -sal/-sel
        ("ekleseler", "add"),
        ("silseler", "delete"),
        ("bulsalar", "find"),
        # -la + conditional / optative: not the derivation -laş
        ("sorgulasalar", "query"),
        ("sorgulasın", "query"),
        # negative progressive of a verb whose verbal noun is a key of its own
        ("uygulamıyor", "apply"),
        ("uyguluyor", "apply"),
        ("doğrulamıyor", "validate"),
    ],
)
def test_inflection_sequences_that_look_like_derivations(word: str, english: str) -> None:
    assert english in (_translate_tr(word) or ()), _translate_tr(word)


def test_uygulamiyor_is_not_the_app() -> None:
    assert "app" not in (_translate_tr("uygulamıyor") or ())


def test_kod_with_la_is_the_instrumental_case() -> None:
    # "hangi kodla açılıyor": with which code. Filler like "kod", as before.
    assert _translate_tr("kodla") == ()
    assert _translate_tr("kodlanıyor") == ()


def test_ver_guards_keep_veri_plurals_unknown() -> None:
    # Forms of "ver" (give) are filler ...
    for word in ("verilir", "verilen", "verildi", "verilmiş", "verilecek", "verilebilir",
                 "verilse", "verir", "veriyor", "verilince"):
        assert _translate_tr(word) == (), word
    # ... but "veri" (data) + -ler is not swallowed by them.
    for word in ("veriler", "verileri", "verilerde", "verilerin"):
        assert _translate_tr(word) is None, (word, _translate_tr(word))


def test_turkish_phrases_with_inflection() -> None:
    index = make_index(
        [
            sym("pkg/net.py", "Client.timeout", "method", doc="Seconds to wait."),
            sym("pkg/clock.py", "current_time", doc="The current time."),
            sym("pkg/work.py", "Thread", "class", doc="Runs a target."),
            sym("pkg/work.py", "Job", "class", doc="A piece of work."),
            sym("pkg/web.py", "middleware", doc="Wrap the app."),
        ]
    )
    search = SearchIndex(index)
    hits = search.search("zaman aşımına nerede bakılıyor?")
    assert hits[0].symbol_id == "pkg/net.py::Client.timeout"
    assert "pkg/clock.py::current_time" not in [h.symbol_id for h in hits]


# --------------------------------------------------------------------------- #
# lexicon sanity and English queries
# --------------------------------------------------------------------------- #


def test_every_lexicon_key_translates_to_its_own_entry() -> None:
    from repofinder.search import _TR_LEXICON, _TR_MIN_STEM

    for key, english in _TR_LEXICON.items():
        assert len(key) >= _TR_MIN_STEM, key
        assert _translate_tr(key) == english, (key, _translate_tr(key))


def test_english_queries_are_not_translated() -> None:
    index = make_index(
        [
            sym("pkg/queue.py", "sort_by_priority", doc="Order the jobs."),
            sym("pkg/net.py", "Client.timeout", "method", doc="Seconds to wait."),
            sym("pkg/git.py", "checkout_branch", doc="Switch branch."),
        ]
    )
    search = SearchIndex(index)
    # "rota", "yol", "sil" are Turkish keys, but an English query is never translated.
    q = search._analyze("where is the rota yol sil timeout")
    assert "route" not in q.tokens and "path" not in q.tokens and "delet" not in q.tokens
    assert search.search("where is the request timeout set")[0].symbol_id == "pkg/net.py::Client.timeout"
    assert search.search("image rotation") == []
