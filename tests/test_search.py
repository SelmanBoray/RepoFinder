"""Tests for repofinder.search: tokenizer, BM25 ranking, boosts, explanations, Flask targets."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pytest

from repofinder.ingest import build_index
from repofinder.models import FileRecord, RepoIndex, Symbol
from repofinder.search import Hit, SearchIndex, stem, tokenize

FLASK_ROOT = Path(
    r"C:\Users\Selman\AppData\Local\Temp\claude\C--\2893ddbc-306e-4b5d-94d8-359c5612cb51"
    r"\scratchpad\repos\flask"
)
needs_flask = pytest.mark.skipif(not (FLASK_ROOT / "src" / "flask").is_dir(), reason="Flask checkout absent")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def sym(
    path: str,
    qualname: str,
    kind: str = "function",
    *,
    start: int = 1,
    end: Optional[int] = None,
    signature: Optional[str] = None,
    doc: Optional[str] = None,
) -> Symbol:
    name = qualname.rsplit(".", 1)[-1]
    parent = qualname.rsplit(".", 1)[0] if "." in qualname else None
    if signature is None:
        signature = f"class {name}" if kind == "class" else f"def {name}()"
    return Symbol(
        id=f"{path}::{qualname}", path=path, name=name, qualname=qualname, kind=kind,
        start_line=start, end_line=end if end is not None else start + 1,
        signature=signature, docstring=doc, parent=parent,
    )


def make_index(symbols: list[Symbol], extra_files: tuple[str, ...] = ()) -> RepoIndex:
    paths = sorted({s.path for s in symbols} | set(extra_files))
    files = [FileRecord(p, "python", 100, 50, True) for p in paths]
    return RepoIndex(name="t", root="does-not-exist", source="t", files=files, symbols=symbols)


def ids(hits: list[Hit]) -> list[Optional[str]]:
    return [h.symbol_id or h.path for h in hits]


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #


def test_tokenize_splits_identifiers_and_keeps_the_whole() -> None:
    assert tokenize("url_for") == ["url", "for", "url_for"]
    assert tokenize("getURL") == ["get", "url", "geturl"]
    assert tokenize("HTTPServerError") == ["http", "server", "error", "httpservererror"]
    assert tokenize("_make_timedelta") == ["make", "timedelta", "make_timedelta"]
    assert tokenize("Flask.run") == ["flask", "run"]
    assert tokenize("src/flask/sansio/scaffold") == ["src", "flask", "sansio", "scaffold"]


def test_tokenize_drops_short_tokens_and_stopwords() -> None:
    assert tokenize("def f(self, x): return the value") == ["value"]
    assert tokenize("Where IS the Route?") == ["where", "route"]  # "where" is query filler, not a stopword
    assert tokenize("__init__") == ["init"]
    assert tokenize("") == []


def test_tokenize_keeps_stopword_parts_of_compound_identifiers() -> None:
    assert tokenize("is_json") == ["is", "json", "is_json"]
    assert tokenize("for") == []  # alone it is a stopword


def test_tokenize_keeps_non_ascii_words_whole() -> None:
    # The old ASCII-only tokenizer cut "tanımlı" into "tan", "ml".
    assert tokenize("Şablonlarda `request` tanımlı") == ["şablonlarda", "request", "tanımlı"]
    assert tokenize("İstek") == ["istek"]  # Turkish dotted capital I


@pytest.mark.parametrize(
    "words, key",
    [
        (("sign", "signs", "signed", "signing"), "sign"),
        (("load", "loads", "loaded", "loading"), "load"),
        (("build", "builds", "builder", "building"), "build"),
        (("register", "registers", "registered", "registering"), "regist"),
        (("handle", "handler", "handlers", "handled"), "handl"),
        (("run", "runs", "running"), "run"),
        (("class", "classes"), "class"),
        (("query", "queries"), "query"),
    ],
)
def test_stem_maps_inflections_to_one_key(words: tuple[str, ...], key: str) -> None:
    assert {stem(w) for w in words} == {key}


def test_stem_leaves_short_and_special_tokens_alone() -> None:
    assert stem("parser") == "parser"  # -er is only removed from words of 7+ letters
    assert stem("user") == "user"
    assert stem("status") == "status"
    assert stem("url_for") == "url_for"
    assert stem("utf8") == "utf8"
    assert stem("tanımlı") == "tanımlı"


# --------------------------------------------------------------------------- #
# ranking and boosts
# --------------------------------------------------------------------------- #


def test_bm25_ranks_more_specific_documents_first() -> None:
    index = make_index(
        [
            sym("pkg/a.py", "load_config", doc="Load the configuration from a TOML file."),
            sym("pkg/b.py", "load_data", doc="Load data rows."),
            sym("pkg/c.py", "save_data", doc="Save data rows."),
        ]
    )
    search = SearchIndex(index)
    hits = search.search("configuration", k=5)
    assert hits[0].symbol_id == "pkg/a.py::load_config"
    # "configuration" is abbreviated "config" in identifiers: half of the name
    # load_config is in the query, so a partial name boost fires on top of BM25.
    assert hits[0].why.startswith("name match: load_config x")
    assert hits[0].why.endswith(" (configuration); bm25: configuration")
    assert all(h.score > 0 for h in hits)
    # A word that names nothing is explained by BM25 alone.
    hits = search.search("toml", k=5)
    assert [h.symbol_id for h in hits if h.kind == "symbol"] == ["pkg/a.py::load_config"]
    assert hits[0].why == "bm25: toml"


def test_name_match_boost_beats_docstring_mentions() -> None:
    index = make_index(
        [
            sym("pkg/helpers.py", "url_for", doc="Build a URL."),
            sym("pkg/other.py", "make_link", doc="Like url_for, url_for and url_for but for links."),
        ]
    )
    hits = SearchIndex(index).search("url_for")
    assert ids(hits)[0] == "pkg/helpers.py::url_for"
    assert hits[0].why.startswith("name match: url_for; bm25: ")
    assert "name match" not in hits[1].why


def test_dotted_query_matches_qualname_on_boundaries() -> None:
    index = make_index(
        [
            sym("pkg/app.py", "Flask", "class", start=1, end=100),
            sym("pkg/app.py", "Flask.run", "method", start=10, end=20),
            sym("pkg/app.py", "Flask.run_simple", "method", start=30, end=40),
            sym("pkg/other.py", "Runner.run", "method", start=5, end=9),
        ]
    )
    hits = SearchIndex(index).search("Flask.run")
    assert hits[0].symbol_id == "pkg/app.py::Flask.run"
    assert "qualname/path match: flask.run" in hits[0].why
    by_id = {h.symbol_id: h for h in hits}
    assert "qualname/path match" not in by_id["pkg/app.py::Flask.run_simple"].why
    assert "qualname/path match" not in by_id["pkg/other.py::Runner.run"].why


def test_module_path_term_matches_symbols_and_file() -> None:
    index = make_index(
        [sym("src/pkg/core.py", "start"), sym("src/pkg/extra.py", "start")]
    )
    hits = SearchIndex(index).search("pkg/core.py start")
    assert hits[0].symbol_id == "src/pkg/core.py::start"
    assert "qualname/path match: pkg/core.py" in hits[0].why
    dotted = SearchIndex(index).search("pkg.core.start")  # dotted module + qualname
    assert dotted[0].symbol_id == "src/pkg/core.py::start"


def test_class_boost_breaks_otherwise_equal_scores() -> None:
    index = make_index(
        [
            sym("pkg/a.py", "Session", "function", signature="Session"),
            sym("pkg/b.py", "Session", "class", signature="Session"),
        ]
    )
    hits = SearchIndex(index).search("session")
    assert hits[0].symbol_id == "pkg/b.py::Session"
    assert "class boost" in hits[0].why
    assert "class boost" not in hits[1].why
    assert hits[0].score == pytest.approx(hits[1].score * 1.2)


@pytest.mark.parametrize("test_path", ["tests/test_core.py", "pkg/test_core.py", "pkg/conftest.py"])
def test_test_files_are_penalized_unless_the_query_mentions_tests(test_path: str) -> None:
    index = make_index([sym("pkg/core.py", "parse_header"), sym(test_path, "parse_header")])
    search = SearchIndex(index)
    hits = search.search("parse_header")
    assert hits[0].path == "pkg/core.py"
    assert hits[1].path == test_path
    assert "test file x0.5" in hits[1].why
    assert hits[1].score < hits[0].score

    hits = search.search("tests for parse_header")
    assert all("test file" not in h.why for h in hits)


def test_nested_functions_rank_below_their_owner() -> None:
    index = make_index(
        [
            sym("pkg/scaffold.py", "Scaffold.route", "method", start=10, end=30),
            sym("pkg/scaffold.py", "Scaffold.route.<locals>.decorator", start=20, end=25),
            sym("pkg/views.py", "Thing.decorator", "method", start=1, end=5),
        ]
    )
    hits = [h for h in SearchIndex(index).search("route") if h.kind == "symbol"]
    assert ids(hits)[:2] == ["pkg/scaffold.py::Scaffold.route", "pkg/scaffold.py::Scaffold.route.<locals>.decorator"]
    assert "nested x0.5" in hits[1].why
    # The "<locals>" marker itself is not a search term.
    assert SearchIndex(index).search("locals") == []


def test_adjacent_words_join_into_identifiers() -> None:
    index = make_index(
        [
            sym("pkg/app.py", "App.test_client", "method", doc="Create a client."),
            sym("pkg/app.py", "App.errorhandler", "method", doc="Register a function."),
            sym("pkg/client.py", "Client", "class", doc="A test client for tests."),
        ]
    )
    search = SearchIndex(index)
    assert search.search("test client")[0].symbol_id == "pkg/app.py::App.test_client"
    top = search.search("where is the error handler")[0]
    assert top.symbol_id == "pkg/app.py::App.errorhandler"
    assert top.why.startswith("name match: errorhandler")


def test_file_hits_and_summaries() -> None:
    index = make_index([sym("pkg/sessions.py", "SessionMixin", "class")], extra_files=("pkg/other.py",))
    summaries = {"pkg/other.py": "Signs cookies with itsdangerous."}
    search = SearchIndex(index, summaries=summaries)

    hits = search.search("cookies")
    assert len(hits) == 1
    hit = hits[0]
    assert (hit.kind, hit.path, hit.symbol_id, hit.start_line, hit.end_line) == ("file", "pkg/other.py", None, 1, 50)

    hits = search.search("sessions")
    file_hit = next(h for h in hits if h.kind == "file")
    assert file_hit.path == "pkg/sessions.py"
    assert "path match: sessions" in file_hit.why


def test_ordering_is_deterministic_on_ties() -> None:
    index = make_index(
        [
            sym("pkg/b.py", "render", start=5),
            sym("pkg/a.py", "render", start=9),
            sym("pkg/a.py", "render", start=3),  # duplicate id, different line (overload)
        ]
    )
    hits = SearchIndex(index).search("render", k=3)
    assert [(h.path, h.start_line) for h in hits] == [("pkg/a.py", 3), ("pkg/a.py", 9), ("pkg/b.py", 5)]
    assert len({h.score for h in hits}) == 1


def test_empty_and_filler_queries() -> None:
    index = make_index([sym("pkg/a.py", "find"), sym("pkg/a.py", "build_thing", start=5)])
    search = SearchIndex(index)
    assert search.search("") == []
    assert search.search("where is the") == []
    assert search.search("build_thing", k=0) == []
    # A query made only of filler words still finds a symbol with that name.
    assert search.search("find")[0].symbol_id == "pkg/a.py::find"
    # Filler is dropped from natural-language questions.
    assert search.search("where is build thing implemented")[0].symbol_id == "pkg/a.py::build_thing"


def test_k_limits_results(sample_index: RepoIndex) -> None:
    search = SearchIndex(sample_index)
    assert len(search.search("engine", k=2)) == 2
    hits = search.search("engine", k=50)
    assert hits[0].symbol_id == "mypkg/core.py::Engine"
    assert len(hits) == len({(h.path, h.start_line, h.kind) for h in hits})


# --------------------------------------------------------------------------- #
# query analysis: coverage-scaled name boost, owner, stems, kind, Turkish
# --------------------------------------------------------------------------- #


def test_inflections_match_through_stems_and_are_marked() -> None:
    index = make_index(
        [
            sym("pkg/sessions.py", "Interface.get_signing_serializer", "method"),
            sym("pkg/ctx.py", "Context.push", "method", doc="Bind the context."),
        ]
    )
    hit = SearchIndex(index).search("how is the cookie signed")[0]
    assert hit.symbol_id == "pkg/sessions.py::Interface.get_signing_serializer"
    # Only the stem matched: the query word is shown with "~".
    assert hit.why.endswith("bm25: signed~")
    assert "name match: get_signing_serializer x" in hit.why


def test_a_common_word_that_is_a_name_does_not_win_a_longer_question() -> None:
    index = make_index(
        [
            sym("pkg/json.py", "loads", doc="Deserialize data as JSON."),
            sym("pkg/provider.py", "Provider.loads", "method", doc="Deserialize data as JSON."),
            sym("pkg/config.py", "Config", "class", start=1, end=100, doc="A dict that holds the app config."),
            sym("pkg/config.py", "Config.from_pyfile", "method", start=10, end=20,
                doc="Updates the values in the config from a Python file."),
            sym("pkg/config.py", "Config.from_mapping", "method", start=30, end=40,
                doc="Updates the config like update()."),
        ]
    )
    hits = SearchIndex(index).search("Which method loads the app config from a Python file?")
    assert hits[0].symbol_id == "pkg/config.py::Config.from_pyfile"
    # "python file" joins (through the abbreviation py) into the name part "pyfile".
    assert "name match: from_pyfile x" in hits[0].why and "python file" in hits[0].why
    loads = next(h for h in hits if h.symbol_id == "pkg/json.py::loads")
    assert "name match: loads x" in loads.why  # a scaled-down boost, not the full one
    # Asked alone, the word is the whole query and gets the full boost.
    alone = SearchIndex(index).search("loads")
    assert alone[0].why.startswith("name match: loads; ")


def test_a_name_made_of_query_words_beats_a_shorter_exact_name() -> None:
    index = make_index(
        [
            sym("pkg/app.py", "App.app_context", "method", start=1, doc="Create an app context."),
            sym("pkg/app.py", "App.teardown_appcontext", "method", start=10,
                doc="Registers a function to call when the app context is popped."),
            sym("pkg/ctx.py", "AppContext", "class", doc="The app context."),
        ]
    )
    query = "Where do I register cleanup code that should run on app context teardown?"
    hit = SearchIndex(index).search(query)[0]
    assert hit.symbol_id == "pkg/app.py::App.teardown_appcontext"
    assert "name match: teardown_appcontext x" in hit.why and "teardown" in hit.why


def test_naming_the_class_too_picks_its_method() -> None:
    index = make_index(
        [
            sym("pkg/helpers.py", "make_response", doc="Make a response object."),
            sym("pkg/app.py", "Flask.make_response", "method", doc="Convert a view's return value to a response."),
        ]
    )
    hits = SearchIndex(index).search("Where is make_response implemented on the Flask application object?")
    assert hits[0].symbol_id == "pkg/app.py::Flask.make_response"
    assert "owner match: Flask x" in hits[0].why
    assert "owner match" not in hits[1].why


def test_which_class_prefers_classes() -> None:
    index = make_index(
        [
            sym("pkg/app.py", "Flask.test_client", "method", doc="Create a test client for the app."),
            sym("pkg/testing.py", "FlaskClient", "class",
                doc="Works like a regular test client. Created by app.test_client()."),
        ]
    )
    search = SearchIndex(index)
    top = search.search("Which class implements the test client?")[0]
    assert top.symbol_id == "pkg/testing.py::FlaskClient"
    assert "class asked x3" in top.why
    # Without the question for a class, the method named like the query wins.
    assert search.search("test client")[0].symbol_id == "pkg/app.py::Flask.test_client"
    # "class-based" is not a question for a class.
    assert search.search("class-based test client")[0].symbol_id == "pkg/app.py::Flask.test_client"


def test_test_word_inside_a_joined_identifier_keeps_the_test_penalty() -> None:
    index = make_index(
        [
            sym("pkg/app.py", "App.test_client", "method", doc="Create a client."),
            sym("tests/conftest.py", "client", doc="A client fixture."),
        ]
    )
    search = SearchIndex(index)
    by_path = {h.path: h for h in search.search("test client")}
    assert "test file x0.5" in by_path["tests/conftest.py"].why  # "test client" -> test_client
    by_path = {h.path: h for h in search.search("tests for the test client")}
    assert "test file" not in by_path["tests/conftest.py"].why


def test_find_is_a_content_word() -> None:
    index = make_index(
        [
            sym("pkg/app.py", "App._find_error_handler", "method", start=1,
                doc="Return a registered error handler for an exception."),
            sym("pkg/app.py", "App.handle_exception", "method", start=10,
                doc="Handle an exception that has no error handler."),
        ]
    )
    hit = SearchIndex(index).search("How does the app find the error handler for an exception?")[0]
    assert hit.symbol_id == "pkg/app.py::App._find_error_handler"


def test_turkish_filler_is_dropped_so_the_identifier_is_the_whole_query() -> None:
    index = make_index(
        [sym("pkg/helpers.py", "abort"), sym("pkg/app.py", "App.make_aborter", "method", doc="Create the abort object.")]
    )
    hit = SearchIndex(index).search("`abort` fonksiyonu nerede tanımlı?")[0]
    assert hit.symbol_id == "pkg/helpers.py::abort"
    assert hit.why == "name match: abort; bm25: abort"


def test_turkish_words_are_translated() -> None:
    index = make_index(
        [
            sym("pkg/app.py", "App.register_blueprint", "method", doc="Register a blueprint on the application."),
            sym("pkg/blueprints.py", "Blueprint", "class", doc="Represents a blueprint."),
            sym("pkg/wrappers.py", "Request.blueprint", "method", doc="The name of the current blueprint."),
        ]
    )
    hit = SearchIndex(index).search("Blueprint'ler uygulamaya nasıl kaydediliyor?")[0]
    assert hit.symbol_id == "pkg/app.py::App.register_blueprint"
    assert "kaydediliyor" in hit.why  # the Turkish word the name covers


def test_words_in_the_index_are_never_translated() -> None:
    index = make_index(
        [
            sym("pkg/config.py", "from_file", signature="def from_file(filename, silent=False)"),
            sym("pkg/store.py", "delete_entry"),
        ]
    )
    search = SearchIndex(index)
    # "silent" starts with the Turkish stem "sil" (delete) but is an index word.
    assert ids(search.search("silent")) == ["pkg/config.py::from_file"]
    # "silinir" is not: it is translated to delete/remove.
    assert search.search("kayıt nasıl silinir")[0].symbol_id == "pkg/store.py::delete_entry"


def test_only_turkish_looking_queries_are_translated() -> None:
    index = make_index(
        [
            sym("pkg/app.py", "App.route", "method", doc="Register a URL rule."),
            sym("pkg/wrappers.py", "Request", "class", doc="The request object."),
        ]
    )
    search = SearchIndex(index)
    # An English query: "rotation" is not read as the Turkish "rota" (route).
    assert search.search("image rotation") == []
    # "nerede" marks the query as Turkish.
    assert search.search("rota nerede")[0].symbol_id == "pkg/app.py::App.route"
    # Softened final consonant: istek -> isteği.
    assert search.search("isteği nerede")[0].symbol_id == "pkg/wrappers.py::Request"


# --------------------------------------------------------------------------- #
# Flask integration
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def flask_index() -> RepoIndex:
    return build_index(FLASK_ROOT, "flask", "https://github.com/pallets/flask")


@needs_flask
def test_flask_url_for(flask_index: RepoIndex) -> None:
    hits = SearchIndex(flask_index).search("url_for", k=3)
    assert "src/flask/helpers.py::url_for" in ids(hits)


@needs_flask
@pytest.mark.parametrize("query", ["route decorator", "where is the route decorator implemented"])
def test_flask_route_decorator(flask_index: RepoIndex, query: str) -> None:
    hits = SearchIndex(flask_index).search(query, k=5)
    route_ids = {
        "src/flask/sansio/scaffold.py::Scaffold.route",
        "src/flask/app.py::Flask.route",
        "src/flask/blueprints.py::Blueprint.route",
    }
    assert route_ids & set(ids(hits)), ids(hits)


@needs_flask
def test_flask_dotted_identifier(flask_index: RepoIndex) -> None:
    hits = SearchIndex(flask_index).search("Flask.run", k=5)
    assert hits[0].symbol_id == "src/flask/app.py::Flask.run"


@needs_flask
@pytest.mark.parametrize(
    "query, expected",
    [
        ("Which method loads the app config from a Python file?", "src/flask/config.py::Config.from_pyfile"),
        ("Where is make_response implemented on the Flask application object?",
         "src/flask/app.py::Flask.make_response"),
        ("Where do I register cleanup code that should run on app context teardown?",
         "src/flask/sansio/app.py::App.teardown_appcontext"),
        ("Which class implements the test client?", "src/flask/testing.py::FlaskClient"),
        ("`abort` fonksiyonu nerede tanımlı?", "src/flask/helpers.py::abort"),
        ("Blueprint'ler uygulamaya nasıl kaydediliyor?", "src/flask/sansio/app.py::App.register_blueprint"),
    ],
)
def test_flask_question_top_hit(flask_index: RepoIndex, query: str, expected: str) -> None:
    assert SearchIndex(flask_index).search(query, k=1)[0].symbol_id == expected


@needs_flask
def test_flask_performance(flask_index: RepoIndex) -> None:
    builds = []
    for _ in range(3):
        start = time.perf_counter()
        search = SearchIndex(flask_index)
        builds.append(time.perf_counter() - start)
    queries = []
    for query in ["url_for", "where is the route decorator implemented", "test client"] * 3:
        start = time.perf_counter()
        search.search(query)
        queries.append(time.perf_counter() - start)
    assert min(builds) < 0.5
    assert max(queries) < 0.05
