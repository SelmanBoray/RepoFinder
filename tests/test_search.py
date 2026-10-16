"""Tests for repofinder.search: tokenizer, BM25 ranking, boosts, explanations, Flask targets."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pytest

from repofinder.ingest import build_index
from repofinder.models import FileRecord, RepoIndex, Symbol
from repofinder.search import (
    Hit,
    SearchIndex,
    _abbreviations,
    _explicit_identifiers,
    _file_stem,
    _path_text,
    _whole,
    stem,
    tokenize,
)

from conftest import FLASK_ROOT, needs_flask  # noqa: E402


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
# v3: derived-token IDF cap, explicit identifiers, test names, Turkish ASCII
# --------------------------------------------------------------------------- #


def _request_heavy_index() -> RepoIndex:
    """A small requests-like repo: "request" is in almost every document."""
    return make_index(
        [
            sym("src/requests/cookies.py", "MockRequest.get_origin_req_host", "method", start=10,
                doc="Return the origin host."),
            sym("src/requests/sessions.py", "Session", "class", start=1, end=400,
                doc="A Requests session. Provides cookie persistence and connection pooling. "
                    "Basic usage: s = requests.Session(); s.get(url). A session sends a request "
                    "and keeps the session cookies of the session across requests."),
            sym("src/requests/sessions.py", "Session.request", "method", start=100, end=200,
                signature="def request(self, method, url, params=None, data=None, headers=None, "
                          "cookies=None, files=None, auth=None, timeout=None)",
                doc="Constructs a Request, prepares it and sends it. Returns a Response object. "
                    "method: method for the new Request object. url: URL for the new Request "
                    "object. params: dictionary to send in the query string. data: body to "
                    "attach. headers: dictionary of HTTP headers to send. cookies: dict or "
                    "CookieJar to send. files: dictionary of files for multipart encoding."),
            sym("src/requests/sessions.py", "Session.prepare_request", "method", start=50, end=90,
                doc="Constructs a PreparedRequest for transmission from a Request."),
            sym("src/requests/models.py", "Request", "class", start=1, end=80,
                doc="A user-created Request object, used to prepare a PreparedRequest."),
            sym("src/requests/api.py", "request", start=1, end=20, doc="Constructs and sends a Request."),
            sym("src/requests/api.py", "get", start=30, end=40, doc="Sends a GET request."),
            sym("src/requests/sessions.py", "Session.get", "method", start=210, end=220,
                doc="Sends a GET request. Returns a Response object."),
            sym("src/requests/structures.py", "LookupDict.get", "method", start=5, end=8,
                doc="Get a request status by name."),
            sym("src/requests/hooks.py", "dispatch_hook", doc="Dispatches a hook for a request."),
            sym("src/requests/auth.py", "HTTPBasicAuth", "class", doc="Attaches Basic Authentication to a request."),
        ]
    )


def test_a_derived_token_never_outweighs_the_word_it_comes_from() -> None:
    search = SearchIndex(_request_heavy_index())
    # "req" (the abbreviation of "request") is in one document, "request" in
    # nearly all: the rare abbreviation must not weigh more than the common word.
    q = search._analyze("request")
    assert q.idf["req"] <= q.idf["request"]
    assert q.units[0].weight == pytest.approx(search._idf["request"])
    order = ids(search.search("request", k=10))
    # Names that are the word itself come before a name that only has "req".
    origin = order.index("src/requests/cookies.py::MockRequest.get_origin_req_host")
    assert origin > order.index("src/requests/api.py::request")
    assert origin > order.index("src/requests/models.py::Request")
    # A word no document has is no cap: its abbreviation is then the only match.
    index = make_index([sym("pkg/conf.py", "load_cfg", doc="Read the settings.")])
    hit = SearchIndex(index).search("configuration")[0]
    assert hit.symbol_id == "pkg/conf.py::load_cfg"
    assert hit.why.endswith("bm25: configuration~")


def test_an_explicit_dotted_identifier_gets_the_full_name_boost() -> None:
    search = SearchIndex(_request_heavy_index())
    hits = search.search("Where is `Session.request` defined?")
    assert hits[0].symbol_id == "src/requests/sessions.py::Session.request"
    assert hits[0].why.startswith("name match: request; ")  # the full boost
    assert "qualname/path match: session.request" in hits[0].why
    # "Session" only qualifies the name: the class itself is not a name match.
    session = next(h for h in hits if h.symbol_id == "src/requests/sessions.py::Session")
    assert "name match" not in session.why
    # More words around it: still the full boost, because the query names it.
    hit = search.search("Which cookies and files does Session.request send?")[0]
    assert hit.symbol_id == "src/requests/sessions.py::Session.request"
    assert hit.why.startswith("name match: request (explicit session.request); ")


def test_a_package_qualified_identifier_names_the_reexported_function() -> None:
    search = SearchIndex(_request_heavy_index())
    hits = search.search("Where is the top-level `requests.get` function defined?")
    assert hits[0].symbol_id == "src/requests/api.py::get"
    assert "qualname/path match: requests.get" in hits[0].why
    by_id = {h.symbol_id: h for h in hits}
    # Session.get and LookupDict.get are named get, but not requests.get.
    for other in ("src/requests/sessions.py::Session.get", "src/requests/structures.py::LookupDict.get"):
        assert "explicit" not in by_id[other].why and "qualname/path" not in by_id[other].why


def test_explicit_identifiers_match_case_and_share_the_query() -> None:
    index = make_index(
        [
            sym("pkg/ctx.py", "Context.request", "method", doc="The current request."),
            sym("pkg/wrappers.py", "Request", "class", doc="The request object."),
            sym("pkg/app.py", "App.update_template_context", "method",
                doc="Inject request, session and g into the template context."),
        ]
    )
    search = SearchIndex(index)
    by_id = {h.symbol_id: h for h in search.search("Where does `request` come from?")}
    assert by_id["pkg/ctx.py::Context.request"].why.startswith("name match: request")
    assert "explicit" not in by_id["pkg/wrappers.py::Request"].why  # `request` is not the class Request
    # Several identifiers: none of them is the question by itself, coverage decides.
    hit = next(
        h for h in search.search("templates: where do `request`, `session` and `g` come from?")
        if h.symbol_id == "pkg/ctx.py::Context.request"
    )
    assert "name match: request x" in hit.why


def test_a_partial_name_match_does_not_lift_a_test_over_the_code() -> None:
    index = make_index(
        [
            sym("src/pkg/sessions.py", "Session.merge_environment_settings", "method",
                doc="Check the environment and merge it with some settings. Uses a custom CA bundle."),
            sym("tests/test_pkg.py", "TestPkg.test_env_cert_bundles", "method",
                doc="Bundles from the environment."),
        ]
    )
    search = SearchIndex(index)
    hits = search.search("How can I point the client at a custom CA bundle with an environment variable?")
    assert hits[0].symbol_id == "src/pkg/sessions.py::Session.merge_environment_settings"
    test_hit = next(h for h in hits if h.path == "tests/test_pkg.py")
    assert "name match" not in test_hit.why and "test file x0.5" in test_hit.why
    # Asked about tests, the test's name counts again.
    test_hit = next(h for h in search.search("tests for environment cert bundles") if h.path == "tests/test_pkg.py")
    assert "name match: test_env_cert_bundles" in test_hit.why


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Istek govdesi nerede olusturuluyor?", "pkg/models.py::PreparedRequest.prepare_body"),
        ("Cookie basligina nasil yaziliyor?", "pkg/cookies.py::get_cookie_header"),
        ("Hangi sinif oturum tutuyor?", "pkg/store.py::SessionStore"),  # "sınıf": a class is asked for
    ],
)
def test_turkish_typed_without_turkish_letters_is_translated(query: str, expected: str) -> None:
    index = make_index(
        [
            sym("pkg/models.py", "PreparedRequest.prepare_body", "method", doc="Prepares the given HTTP body data."),
            sym("pkg/models.py", "PreparedRequest.prepare_url", "method", doc="Prepares the given HTTP URL."),
            sym("pkg/cookies.py", "get_cookie_header", doc="Produce an appropriate Cookie header string."),
            sym("pkg/cookies.py", "morsel_to_cookie",
                doc="Convert a cookie Morsel into a cookie: the cookie name, the cookie value."),
            sym("pkg/sessions.py", "session", doc="Returns a session."),
            sym("pkg/store.py", "SessionStore", "class", doc="Keeps a session."),
        ]
    )
    hit = SearchIndex(index).search(query)[0]
    assert hit.symbol_id == expected, hit.why


def test_ambiguous_turkish_words_are_not_translated_to_their_rare_sense() -> None:
    index = make_index(
        [
            sym("pkg/models.py", "Response.links", "method", doc="Returns the parsed header links of the response."),
            sym("pkg/adapters.py", "Adapter.get_connection", "method", doc="Returns a connection for the given URL."),
            sym("pkg/hooks.py", "handle_hook", doc="Handle a hook."),
        ]
    )
    search = SearchIndex(index)
    hits = search.search("Bağlantı nerede açılıyor?")
    assert hits[0].symbol_id == "pkg/adapters.py::Adapter.get_connection"
    assert "pkg/models.py::Response.links" not in ids(hits)  # bağlantı is not "link"
    assert search.search("istek nasıl işleniyor?") == []  # işle: no single sense, not "handle"


def test_turkish_phrases_translate_as_one() -> None:
    index = make_index(
        [
            sym("pkg/exceptions.py", "Timeout", "class", doc="The request timed out."),
            sym("pkg/clock.py", "current_time", doc="The current time."),
            sym("pkg/adapters.py", "Adapter.max_retries", "method", doc="Retry configuration."),
            sym("pkg/cookies.py", "merge_cookies", doc="Add cookies to a cookie jar."),
        ]
    )
    search = SearchIndex(index)
    hits = search.search("Zaman aşımı nerede tanımlı?")
    assert hits[0].symbol_id == "pkg/exceptions.py::Timeout"
    assert "pkg/clock.py::current_time" not in ids(hits)  # "zaman aşımı" is not time + something
    assert search.search("Istek nasil yeniden deneniyor?")[0].symbol_id == "pkg/adapters.py::Adapter.max_retries"
    assert search.search("tekrar deneme nerede")[0].symbol_id == "pkg/adapters.py::Adapter.max_retries"
    assert search.search("Cookie'ler nasıl birleştiriliyor?")[0].symbol_id == "pkg/cookies.py::merge_cookies"


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


# --------------------------------------------------------------------------- #
# TypeScript paths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path, expected",
    [
        ("pkg/mod.py", "mod"),
        ("pkg/sub/__init__.py", "sub"),
        ("src/util.ts", "util"),
        ("src/util/index.ts", "util"),       # an index file is named after its directory
        ("src/ui/index.tsx", "ui"),
        ("index.ts", "index"),
        ("src/types.d.ts", "types"),
        ("src/a.test.ts", "a.test"),
        ("lib/Run.mts", "run"),
    ],
)
def test_file_stem_strips_any_registered_extension(path: str, expected: str) -> None:
    assert _file_stem(path) == expected


def test_path_text_strips_typescript_extensions() -> None:
    assert _path_text("src/flask/app.py") == "src/flask/app"
    assert _path_text("src/util/index.ts") == "src/util/index"
    assert _path_text("src/types.d.ts") == "src/types"
    assert _path_text("src/View.tsx") == "src/View"


@pytest.mark.parametrize(
    "query",
    ["where is foo.ts loaded", "what does index.tsx render", "main.mts", "cli.js and loader.mjs", "app.py"],
)
def test_file_names_are_not_explicit_identifiers(query: str) -> None:
    assert _explicit_identifiers(query) == []


def test_dotted_identifier_next_to_a_typescript_file_name() -> None:
    assert _explicit_identifiers("what does Router.dispatch do in router.ts") == [("Router", "dispatch")]
    assert _abbreviations("typescript") == ("ts",)


def test_typescript_index_file_name_and_test_penalty() -> None:
    index = make_index(
        [
            sym("src/parser/index.ts", "parseSource", signature="export function parseSource(text: string): Ast"),
            sym("src/parser/parse.test.ts", "parseSource", signature="function parseSource(): void"),
            sym("src/render.ts", "render", signature="export function render(ast: Ast): string"),
        ]
    )
    search = SearchIndex(index)
    hits = search.search("parseSource", k=5)
    assert hits[0].symbol_id == "src/parser/index.ts::parseSource"
    test_hit = next(h for h in hits if h.path == "src/parser/parse.test.ts")
    assert "test file x0.5" in test_hit.why
    file_hit = next(h for h in search.search("parser", k=10) if h.kind == "file" and h.path == "src/parser/index.ts")
    assert "path match: parser" in file_hit.why


# --------------------------------------------------------------------------- #
# v4: TypeScript repositories (type penalty, body field, camelCase, #private,
# Turkish filler and vocabulary)
# --------------------------------------------------------------------------- #

_TS_REGISTRY = """\
/** What the registry chose and why. */
export interface ProviderChoice {
  readonly provider: string;
  readonly reason: string;
}

export type ProviderChoiceResult = ProviderChoice | undefined;

/** The policy of the snapshot cache. */
export interface SnapshotPolicy {
  readonly maxEntries: number;
}

/** Build the policy of the snapshot cache. */
export function snapshotPolicy(maxEntries: number): SnapshotPolicy {
  return { maxEntries };
}

export class ProviderRegistry {
  /** Pick the provider for a target. */
  choose(target: string): Promise<ProviderChoice> {
    return this.#selectAutomatically(target);
  }

  async #selectAutomatically(target: string): Promise<ProviderChoice> {
    // Ghidra comes before Hopper: a fixed priority order.
    const ranked = ["ghidra", "hopper"];
    return { provider: ranked[0], reason: target };
  }
}
"""

_TS_ELECTRON = """\
/** Pair each renderer call with the main-process listener of its channel. */
export const classifyIpcPairings = (records: readonly string[]): string[] => {
  return records.filter((channel) => channel.length > 0);
};

export const loadJavaBridge = (): string => "jar";

export function scriptRunner(): void {}
"""


@pytest.fixture()
def ts_repo(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "application").mkdir(parents=True)
    (src / "application" / "ProviderRegistry.ts").write_text(_TS_REGISTRY, encoding="utf-8")
    (src / "electron").mkdir()
    (src / "electron" / "ipc.ts").write_text(_TS_ELECTRON, encoding="utf-8")
    return tmp_path


def _ts_search(root: Path) -> SearchIndex:
    return SearchIndex(build_index(root, "ts", str(root)))


def test_type_declarations_rank_below_the_implementation(ts_repo: Path) -> None:
    search = _ts_search(ts_repo)
    hits = search.search("snapshot policy", k=5)
    order = ids(hits)
    implementation = order.index("src/application/ProviderRegistry.ts::snapshotPolicy")
    declaration = order.index("src/application/ProviderRegistry.ts::SnapshotPolicy")
    assert implementation < declaration
    assert "type declaration x0.7" in hits[declaration].why
    assert "type declaration" not in hits[implementation].why
    # Asking for the interface lifts the penalty.
    asked = search.search("snapshot policy interface", k=5)
    hit = next(h for h in asked if h.symbol_id == "src/application/ProviderRegistry.ts::SnapshotPolicy")
    assert "type declaration" not in hit.why
    # A type alias is a declaration too.
    alias = next(h for h in search.search("provider choice result", k=5)
                 if h.symbol_id == "src/application/ProviderRegistry.ts::ProviderChoiceResult")
    assert "type declaration x0.7" in alias.why


def test_function_bodies_are_searched(ts_repo: Path) -> None:
    hits = _ts_search(ts_repo).search("priority order between ghidra and hopper", k=5)
    assert hits[0].symbol_id == "src/application/ProviderRegistry.ts::ProviderRegistry.#selectAutomatically"
    assert hits[0].why.endswith("body: ghidra, hopper, order, priority")
    # A class is not indexed by its body (its methods are).
    assert "src/application/ProviderRegistry.ts::ProviderRegistry" not in ids(hits)


def test_no_checkout_means_no_body(ts_repo: Path) -> None:
    index = build_index(ts_repo, "ts", str(ts_repo))
    index.root = str(ts_repo / "gone")
    assert SearchIndex(index).search("priority order between ghidra and hopper", k=5) == []


def test_a_long_body_does_not_dilute_a_name_match(tmp_path: Path) -> None:
    filler = "\n".join(f"  const step{i} = compute(value, {i});" for i in range(200))
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.ts").write_text(
        f"export function refreshToken(value: string): void {{\n{filler}\n}}\n"
        "export function helper(): string {\n  return 'refresh the token';\n}\n",
        encoding="utf-8",
    )
    hits = _ts_search(tmp_path).search("refresh token", k=3)
    assert hits[0].symbol_id == "src/a.ts::refreshToken"


def test_lower_camel_query_words_split_like_code(ts_repo: Path) -> None:
    search = _ts_search(ts_repo)
    tokens = search._analyze("match an ipcRenderer call").tokens
    assert {"ipc", "renderer", "ipcrenderer"} <= set(tokens)
    assert search.search("ipcRenderer pairings", k=1)[0].symbol_id == "src/electron/ipc.ts::classifyIpcPairings"
    # An UpperCamel word stays whole: "JavaScript" is not "java" + "script".
    tokens = search._analyze("JavaScript bundles").tokens
    assert "javascript" in tokens and "java" not in tokens and "script" not in tokens


def test_private_members_match_by_their_bare_name(ts_repo: Path) -> None:
    assert _whole("#selectAutomatically") == "selectautomatically"
    assert _whole("$fooBar") == "foobar"
    search = _ts_search(ts_repo)
    target = "src/application/ProviderRegistry.ts::ProviderRegistry.#selectAutomatically"
    hit = search.search("select automatically", k=1)[0]
    assert hit.symbol_id == target
    assert "name match: #selectAutomatically;" in hit.why  # the whole name, no partial factor
    hit = search.search("where is `selectAutomatically`", k=1)[0]
    assert hit.symbol_id == target


def test_turkish_function_words_are_dropped_even_when_indexed() -> None:
    index = make_index(
        [
            sym("pkg/flags.ts", "veFlag", doc="Vector extension flag."),
            sym("pkg/session.ts", "openSession", doc="Open a session with a cookie."),
        ]
    )
    search = SearchIndex(index)
    units = [u.text for u in search._analyze("oturum ve çerez nerede açılıyor").units]
    assert "ve" not in units
    assert search.search("oturum ve çerez nerede", k=1)[0].symbol_id == "pkg/session.ts::openSession"
    # Written as code it is kept, and an English query keeps it anyway.
    assert "ve" in [u.text for u in search._analyze("`ve` nerede kullanılıyor").units]
    assert "ve" in [u.text for u in search._analyze("where is the ve flag").units]


@pytest.mark.parametrize(
    "query, english, expected",
    [
        ("kimlik bilgisi nerede tutuluyor", "credential", "pkg/redact.ts::dropCredential"),
        ("adaylar nerede", "candidate", "pkg/match.ts::pickCandidate"),
        ("süreç nerede başlatılıyor", "process", "pkg/process.ts::spawnProcess"),
        ("senaryo nerede", "scenario", "pkg/process.ts::runScenario"),
        ("dosyalar nerede içe aktarılıyor", "import", "pkg/har.ts::importArchive"),
    ],
)
def test_turkish_code_vocabulary(query: str, english: str, expected: str) -> None:
    index = make_index(
        [
            sym("pkg/redact.ts", "dropCredential", doc="Remove a secret."),
            sym("pkg/match.ts", "pickCandidate", doc="Choose one."),
            sym("pkg/process.ts", "spawnProcess", doc="Start a child."),
            sym("pkg/process.ts", "runScenario", doc="Replay the steps."),
            sym("pkg/har.ts", "importArchive", doc="Read a capture."),
            sym("pkg/identity.ts", "identityInfo", doc="Who the user is."),
        ]
    )
    search = SearchIndex(index)
    assert stem(english) in search._analyze(query).tokens
    assert search.search(query, k=1)[0].symbol_id == expected


def test_function_words_alone_never_join_into_an_identifier() -> None:
    index = make_index([sym("pkg/util.ts", "maybe"), sym("pkg/cache.ts", "cacheAnswer", doc="Cache an answer.")])
    q = SearchIndex(index)._analyze("which answers may be cached")
    assert "may be" not in [u.text for u in q.units]
    assert "maybe" not in q.tokens
