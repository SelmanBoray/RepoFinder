"""Tests for repofinder.parser: symbols, imports, relative + in-repo resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from repofinder.ingest import build_index
from repofinder.models import ImportEdge, RepoIndex, Symbol
from repofinder.parser import (
    DOCSTRING_MAX_CHARS,
    nested_module_roots,
    parse_file,
    resolve_imports,
)


def _by_qualname(symbols: list[Symbol], path: str) -> dict[str, list[Symbol]]:
    out: dict[str, list[Symbol]] = {}
    for s in symbols:
        if s.path == path:
            out.setdefault(s.qualname, []).append(s)
    return out


def _one(index: RepoIndex, path: str, qualname: str) -> Symbol:
    matches = _by_qualname(index.symbols, path)[qualname]
    assert len(matches) == 1, matches
    return matches[0]


def _edges(index: RepoIndex, path: str) -> list[tuple[str, list[str], int, str | None]]:
    return [(e.dst_module, e.names, e.line, e.dst_path) for e in index.imports if e.src_path == path]


# --------------------------------------------------------------------------- #
# Symbols in the sample repository
# --------------------------------------------------------------------------- #


def test_module_level_function(sample_index: RepoIndex) -> None:
    s = _one(sample_index, "mypkg/core.py", "run_engine")
    assert s == Symbol(
        id="mypkg/core.py::run_engine",
        path="mypkg/core.py",
        name="run_engine",
        qualname="run_engine",
        kind="function",
        start_line=13,
        end_line=30,
        # multi-line header, comment inside the parameter list is dropped
        signature='def run_engine(engine: "Engine", *, retries: int = 3,) -> bool',
        docstring="Run the engine once.\n\nReturns True on success.",
        parent=None,
    )


def test_class_and_methods(sample_index: RepoIndex) -> None:
    engine = _one(sample_index, "mypkg/core.py", "Engine")
    assert engine.kind == "class"
    assert (engine.start_line, engine.end_line) == (33, 63)
    assert engine.signature == "class Engine(Base)"
    # a comment before the docstring does not hide it
    assert engine.docstring == "An engine that does work."
    assert engine.parent is None

    init = _one(sample_index, "mypkg/core.py", "Engine.__init__")
    assert init.kind == "method"
    assert init.name == "__init__"
    assert init.parent == "Engine"
    assert (init.start_line, init.end_line) == (37, 38)
    assert init.signature == "def __init__(self, name: str) -> None"
    assert init.docstring is None


def test_nested_functions_and_classes(sample_index: RepoIndex) -> None:
    attempt = _one(sample_index, "mypkg/core.py", "run_engine.<locals>.attempt")
    assert attempt.kind == "function"
    assert attempt.name == "attempt"
    assert attempt.parent == "run_engine"
    assert (attempt.start_line, attempt.end_line) == (23, 25)
    assert attempt.docstring == "Single attempt."

    result = _one(sample_index, "mypkg/core.py", "run_engine.<locals>.Result")
    assert result.kind == "class"
    assert result.parent == "run_engine"

    on_done = _one(sample_index, "mypkg/core.py", "Engine.start.<locals>.on_done")
    assert on_done.kind == "function"  # nested in a method: function, not method
    assert on_done.parent == "Engine.start"

    deeper = _one(sample_index, "mypkg/core.py", "Engine.start.<locals>.on_done.<locals>.deeper")
    assert deeper.parent == "Engine.start.<locals>.on_done"
    assert (deeper.start_line, deeper.end_line) == (56, 57)


def test_async_method(sample_index: RepoIndex) -> None:
    start = _one(sample_index, "mypkg/core.py", "Engine.start")
    assert start.kind == "method"
    assert start.signature == "async def start(self, timeout: Optional[float] = None)"
    assert start.docstring == "Start asynchronously."
    assert (start.start_line, start.end_line) == (53, 59)


def test_decorated_definitions_include_decorators(sample_index: RepoIndex) -> None:
    # property getter + setter share a qualname (and therefore an id)
    getter, setter = _by_qualname(sample_index.symbols, "mypkg/core.py")["Engine.label"]
    assert (getter.start_line, getter.end_line) == (40, 42)
    assert getter.signature == "def label(self) -> str"
    assert (setter.start_line, setter.end_line) == (44, 46)
    assert getter.id == setter.id == "mypkg/core.py::Engine.label"

    build = _one(sample_index, "mypkg/core.py", "Engine.build")
    assert build.start_line == 48  # first of two decorators
    assert build.end_line == 51
    assert build.signature == "def build(cls)"  # decorators are not part of the signature

    user = _one(sample_index, "mypkg/sub/models.py", "User")
    assert (user.start_line, user.end_line) == (16, 27)
    assert user.signature == "class User(Base)"


def test_method_inside_conditional_in_class_body(sample_index: RepoIndex) -> None:
    hook = _one(sample_index, "mypkg/core.py", "Engine.platform_hook")
    assert hook.kind == "method"
    assert hook.parent == "Engine"


def test_nested_class_in_class(sample_index: RepoIndex) -> None:
    meta = _one(sample_index, "mypkg/sub/models.py", "User.Meta")
    assert meta.kind == "class"
    assert meta.parent == "User"
    describe = _one(sample_index, "mypkg/sub/models.py", "User.Meta.describe")
    assert describe.kind == "method"
    assert describe.parent == "User.Meta"


def test_symbol_ids_and_line_ranges_point_at_source(sample_index: RepoIndex, sample_repo: Path) -> None:
    for s in sample_index.symbols:
        assert s.id == f"{s.path}::{s.qualname}"
        assert s.qualname.split(".")[-1] == s.name
        lines = (sample_repo / s.path).read_text(encoding="utf-8").splitlines()
        assert 1 <= s.start_line <= s.end_line <= len(lines)
        first = lines[s.start_line - 1].strip()
        keyword = "class" if s.kind == "class" else "def"
        assert first.startswith("@") or f"{keyword} {s.name}" in first, (s, first)
        assert lines[s.end_line - 1].strip(), f"{s.id} ends on a blank line"


def test_symbols_sorted_by_path_and_line(sample_index: RepoIndex) -> None:
    keys = [(s.path, s.start_line) for s in sample_index.symbols]
    assert keys == sorted(keys)


# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #


def test_import_extraction(sample_index: RepoIndex) -> None:
    assert _edges(sample_index, "mypkg/core.py") == [
        ("__future__", ["annotations"], 2, None),
        ("os", [], 4, None),
        ("os.path", [], 5, None),  # alias dropped, one edge per dotted name
        ("sys", [], 5, None),
        ("typing", ["Optional"], 6, None),
        ("mypkg.sub", ["models"], 8, "mypkg/sub/models.py"),  # pkg/name.py exists
        ("mypkg.sub.models", ["User", "Base"], 9, "mypkg/sub/models.py"),  # original names
        ("mypkg.sub.models", ["*"], 10, "mypkg/sub/models.py"),  # relative + star
    ]


def test_relative_imports_are_resolved_to_absolute_modules(sample_index: RepoIndex) -> None:
    assert _edges(sample_index, "mypkg/__init__.py") == [
        ("mypkg.core", ["Engine", "run_engine"], 3, "mypkg/core.py"),
        ("mypkg", ["core"], 4, "mypkg/core.py"),
    ]
    assert _edges(sample_index, "mypkg/sub/__init__.py") == [
        ("mypkg.sub", ["models"], 2, "mypkg/sub/models.py"),
        ("mypkg.sub.models", ["Base"], 3, "mypkg/sub/models.py"),
        ("mypkg.core", ["Engine"], 4, "mypkg/core.py"),
    ]
    assert _edges(sample_index, "mypkg/sub/models.py") == [
        ("json", [], 2, None),
        ("dataclasses", ["dataclass"], 3, None),
        ("mypkg", ["core"], 5, "mypkg/core.py"),
        ("mypkg.core", ["run_engine"], 6, "mypkg/core.py"),
    ]


def test_relative_import_edge_cases(make_repo) -> None:
    root = make_repo(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": (
                "from .. import up\n"           # climbs above the top-level package
                "from ...far import x\n"        # way above
                "from . import sibling\n"
            ),
            "scripts/tools/run.py": "from .helpers import h\nfrom .. import common\n",
            "src/app/main.py": "from . import cfg\n",
            "top.py": "from . import thing\n",
        }
    )
    _, _, edges = parse_file(root, root / "pkg/mod.py")
    assert [e.dst_module for e in edges] == ["..", "...far", "pkg"]

    # Not inside a package: best effort from directory names.
    _, _, edges = parse_file(root, Path("scripts/tools/run.py"))
    assert [e.dst_module for e in edges] == ["scripts.tools.helpers", "scripts"]
    _, _, edges = parse_file(root, root / "src/app/main.py")
    assert [e.dst_module for e in edges] == ["app"]
    _, _, edges = parse_file(root, root / "top.py")
    assert [e.dst_module for e in edges] == ["."]


def test_conditional_and_nested_imports_are_found(make_repo) -> None:
    root = make_repo(
        {
            "m.py": (
                "try:\n"
                "    import ujson as json\n"
                "except ImportError:\n"
                "    import json\n"
                "def f():\n"
                "    from collections import (\n"
                "        OrderedDict,\n"
                "        defaultdict as dd,\n"
                "    )\n"
            )
        }
    )
    _, _, edges = parse_file(root, root / "m.py")
    assert [(e.dst_module, e.names, e.line) for e in edges] == [
        ("ujson", [], 2),
        ("json", [], 4),
        ("collections", ["OrderedDict", "defaultdict"], 6),
    ]


# --------------------------------------------------------------------------- #
# In-repo resolution
# --------------------------------------------------------------------------- #


def _resolved(root: Path) -> dict[tuple[str, str], str | None]:
    index = build_index(root, "r", str(root))
    return {(e.src_path, e.dst_module): e.dst_path for e in index.imports}


def test_resolution_src_layout(make_repo) -> None:
    root = make_repo(
        {
            "src/proj/__init__.py": "from .core import run\n",
            "src/proj/core.py": "import os\n",
            "src/proj/sub/__init__.py": "",
            "tests/test_core.py": (
                "import proj.core\n"
                "from proj import sub\n"
                "from proj.sub import missing_attr\n"
                "import proj.sub.not_a_module\n"
                "import pytest\n"
            ),
        }
    )
    got = _resolved(root)
    assert got[("src/proj/__init__.py", "proj.core")] == "src/proj/core.py"
    assert got[("src/proj/core.py", "os")] is None
    assert got[("tests/test_core.py", "proj.core")] == "src/proj/core.py"
    assert got[("tests/test_core.py", "proj")] == "src/proj/sub/__init__.py"  # from pkg import subpkg
    assert got[("tests/test_core.py", "proj.sub")] == "src/proj/sub/__init__.py"
    # longest importable prefix
    assert got[("tests/test_core.py", "proj.sub.not_a_module")] == "src/proj/sub/__init__.py"
    assert got[("tests/test_core.py", "pytest")] is None


def test_resolution_lib_layout_and_no_stdlib_shadowing(make_repo) -> None:
    root = make_repo(
        {
            "lib/pkg/__init__.py": "",
            "lib/pkg/m.py": "import json\nimport logging\n",
            "mypkg/__init__.py": "",
            "mypkg/json.py": "",
            "mypkg/logging.py": "",
            "main.py": "import pkg.m\nimport mypkg.json\nimport json\n",
        }
    )
    got = _resolved(root)
    assert got[("main.py", "pkg.m")] == "lib/pkg/m.py"
    assert got[("main.py", "mypkg.json")] == "mypkg/json.py"
    # mypkg/json.py must not capture the stdlib module
    assert got[("main.py", "json")] is None
    assert got[("lib/pkg/m.py", "json")] is None
    assert got[("lib/pkg/m.py", "logging")] is None


def test_relative_imports_in_a_nested_package_resolve_by_path(make_repo) -> None:
    # The package is not directly under a global import root.
    root = make_repo(
        {
            "examples/demo/app/__init__.py": "",
            "examples/demo/app/db.py": "def get_db(): pass\n",
            "examples/demo/app/auth.py": (
                "from .db import get_db\n"
                "from .sub import views\n"       # from .pkg import name -> pkg/name.py
                "from .missing import thing\n"   # falls back to the longest existing prefix
            ),
            "examples/demo/app/sub/__init__.py": "",
            "examples/demo/app/sub/views.py": (
                "from ..db import get_db\n"      # two levels
                "from .. import auth\n"
                "from ...outside import x\n"     # climbs above the top-level package
            ),
        }
    )
    got = _resolved(root)
    # dst_module keeps its meaning (absolute name from the package); only dst_path improves.
    assert got == {
        ("examples/demo/app/auth.py", "app.db"): "examples/demo/app/db.py",
        ("examples/demo/app/auth.py", "app.sub"): "examples/demo/app/sub/views.py",
        ("examples/demo/app/auth.py", "app.missing"): "examples/demo/app/__init__.py",
        ("examples/demo/app/sub/views.py", "app.db"): "examples/demo/app/db.py",
        ("examples/demo/app/sub/views.py", "app"): "examples/demo/app/auth.py",
        ("examples/demo/app/sub/views.py", "outside"): None,
    }


def test_nested_project_roots_resolve_absolute_imports_in_scope(make_repo) -> None:
    files = {
        "src/flask/__init__.py": "",
        "examples/tutorial/flaskr/__init__.py": (
            "from flaskr import db\nfrom flaskr.auth import login_required\nfrom flask import Flask\n"
        ),
        "examples/tutorial/flaskr/db.py": "import sqlite3\n",
        "examples/tutorial/flaskr/auth.py": "from flaskr.db import get_db\n",
        "examples/tutorial/tests/conftest.py": "from flaskr import create_app\nfrom flaskr.db import init_db\n",
        "examples/celery/src/task_app/__init__.py": "from task_app import views\n",
        "examples/celery/src/task_app/views.py": "",
        "examples/celery/make_celery.py": "from task_app import create_app\n",
        "tests/test_apps/blueprintapp/__init__.py": "from blueprintapp.apps.admin import admin\n",
        "tests/test_apps/blueprintapp/apps/__init__.py": "",
        "tests/test_apps/blueprintapp/apps/admin/__init__.py": "",
        # outside every nested scope: these stay unresolved
        "tests/test_blueprints.py": "from blueprintapp import app\n",
        "docs/conf.py": "import flaskr\n",
    }
    root = make_repo(files)
    assert nested_module_roots(sorted(files)) == {
        "examples/celery/src/": "examples/celery/",  # a src/ root serves its parent project
        "examples/tutorial/": "examples/tutorial/",
        "tests/test_apps/": "tests/test_apps/",
    }
    got = _resolved(root)
    tutorial = "examples/tutorial/flaskr/"
    assert got[(tutorial + "__init__.py", "flaskr")] == tutorial + "db.py"
    assert got[(tutorial + "__init__.py", "flaskr.auth")] == tutorial + "auth.py"
    assert got[(tutorial + "__init__.py", "flask")] == "src/flask/__init__.py"  # global still works
    assert got[(tutorial + "auth.py", "flaskr.db")] == tutorial + "db.py"
    assert got[("examples/tutorial/tests/conftest.py", "flaskr")] == tutorial + "__init__.py"
    assert got[("examples/tutorial/tests/conftest.py", "flaskr.db")] == tutorial + "db.py"
    assert got[("examples/celery/src/task_app/__init__.py", "task_app")] == (
        "examples/celery/src/task_app/views.py"
    )
    assert got[("examples/celery/make_celery.py", "task_app")] == (
        "examples/celery/src/task_app/__init__.py"
    )
    assert got[("tests/test_apps/blueprintapp/__init__.py", "blueprintapp.apps.admin")] == (
        "tests/test_apps/blueprintapp/apps/admin/__init__.py"
    )
    assert got[("tests/test_blueprints.py", "blueprintapp")] is None
    assert got[("docs/conf.py", "flaskr")] is None


def test_nested_roots_do_not_shadow_stdlib(make_repo) -> None:
    root = make_repo(
        {
            "main.py": "import json\n",
            "examples/demo/json.py": "",                 # loose module named like stdlib
            "examples/demo/app/__init__.py": "",
            "examples/demo/app/json.py": "",             # nested pkg/json.py
            "examples/demo/app/auth.py": "import json\nfrom . import json as j\n",
            "vendor/thirdparty/json/__init__.py": "from . import decoder\nfrom .decoder import x\n",
            "vendor/thirdparty/json/decoder.py": "",
            "vendor/thirdparty/tool.py": "import json\n",
        }
    )
    got = _resolved(root)
    assert got[("main.py", "json")] is None  # outside every nested root
    assert got[("examples/demo/app/auth.py", "json")] is None  # in scope, still stdlib
    assert got[("examples/demo/app/auth.py", "app")] == "examples/demo/app/json.py"  # relative
    assert got[("vendor/thirdparty/tool.py", "json")] is None
    # ...but a stdlib-named package still resolves its own relative imports
    assert got[("vendor/thirdparty/json/__init__.py", "json")] == "vendor/thirdparty/json/decoder.py"
    assert got[("vendor/thirdparty/json/__init__.py", "json.decoder")] == (
        "vendor/thirdparty/json/decoder.py"
    )


def test_resolve_imports_works_on_a_handmade_index() -> None:
    from repofinder.models import FileRecord

    index = RepoIndex(name="x", root="/x", source="/x")
    index.files = [
        FileRecord("a/__init__.py", "python", 0, 0, True),
        FileRecord("a/b.py", "python", 0, 0, True),
    ]
    index.imports = [
        ImportEdge("main.py", "a", None, ["b"], 1),
        ImportEdge("main.py", "a", None, ["c"], 2),
        ImportEdge("main.py", "a.b", None, [], 3),
        ImportEdge("main.py", "requests", None, [], 4),
        ImportEdge("main.py", "..rel", None, ["z"], 5),
    ]
    resolve_imports(index)
    assert [e.dst_path for e in index.imports] == ["a/b.py", "a/__init__.py", "a/b.py", None, None]


# --------------------------------------------------------------------------- #
# Syntax errors and odd inputs
# --------------------------------------------------------------------------- #


def test_syntax_error_file_is_flagged_but_parsed(sample_index: RepoIndex) -> None:
    flags = {f.path: f.parse_ok for f in sample_index.files}
    assert flags["mypkg/broken.py"] is False
    assert all(ok for path, ok in flags.items() if path != "mypkg/broken.py")

    symbols = _by_qualname(sample_index.symbols, "mypkg/broken.py")
    assert (symbols["before"][0].start_line, symbols["before"][0].end_line) == (4, 5)
    assert symbols["After"][0].kind == "class"
    assert symbols["After.method"][0].kind == "method"


def test_docstring_variants(make_repo) -> None:
    long_doc = "x" * (DOCSTRING_MAX_CHARS + 50)
    root = make_repo(
        {
            "d.py": (
                f'def long():\n    """{long_doc}"""\n'
                'def raw():\n    r"""Raw \\d doc."""\n'
                "def single():\n    'single quoted'\n"
                'def concat():\n    "part one, " "part two"\n'
                'def fstring():\n    f"not a {doc}"\n'
                'def bytes_doc():\n    b"not a doc"\n'
                'def not_first():\n    x = 1\n    "not a doc"\n'
                'def empty():\n    """   """\n'
                "class Indented:\n"
                '    """Summary line.\n'
                "\n"
                "        Indented detail.\n"
                '    """\n'
            )
        }
    )
    _, symbols, _ = parse_file(root, root / "d.py")
    docs = {s.name: s.docstring for s in symbols}
    assert docs["long"] == "x" * DOCSTRING_MAX_CHARS
    assert docs["raw"] == "Raw \\d doc."
    assert docs["single"] == "single quoted"
    assert docs["concat"] == "part one, part two"
    assert docs["fstring"] is None
    assert docs["bytes_doc"] is None
    assert docs["not_first"] is None
    assert docs["empty"] is None
    assert docs["Indented"] == "Summary line.\n\nIndented detail."  # dedented like inspect.cleandoc


def test_signature_variants(make_repo) -> None:
    root = make_repo(
        {
            "s.py": (
                "class Meta(type, metaclass=abc.ABCMeta):\n    pass\n"
                "class Multi(\n    Base,\n    Mixin,\n):\n    pass\n"
                "def generic[T](x: T) -> T: return x\n"
                "async   def   spaced( a ,  b ):\n    pass\n"
                "def default_with_colon(f=lambda: 1, d={'k': 1}) -> dict:\n    pass\n"
            )
        }
    )
    _, symbols, _ = parse_file(root, root / "s.py")
    sigs = {s.name: s.signature for s in symbols}
    assert sigs["Meta"] == "class Meta(type, metaclass=abc.ABCMeta)"
    assert sigs["Multi"] == "class Multi(Base, Mixin,)"
    assert sigs["generic"] == "def generic[T](x: T) -> T"
    assert sigs["spaced"] == "async def spaced(a , b)"
    assert sigs["default_with_colon"] == "def default_with_colon(f=lambda: 1, d={'k': 1}) -> dict"


def test_crlf_bom_and_invalid_utf8(make_repo) -> None:
    root = make_repo(
        {
            "crlf.py": b'\xef\xbb\xbfdef f(a,\r\n      b):\r\n    """Doc\r\n\r\n    more."""\r\n    return 1\r\n',
            "latin.py": b"# caf\xe9\ndef g():\n    return '\xff'\n",
            "empty.py": b"",
        }
    )
    record, symbols, _ = parse_file(root, root / "crlf.py")
    assert record.parse_ok is True
    assert record.n_lines == 6
    assert record.size == len((root / "crlf.py").read_bytes())
    (f,) = symbols
    assert (f.start_line, f.end_line) == (1, 6)
    assert f.signature == "def f(a, b)"
    assert f.docstring == "Doc\n\nmore."

    record, symbols, _ = parse_file(root, root / "latin.py")
    assert [s.name for s in symbols] == ["g"]
    assert record.n_lines == 3

    record, symbols, imports = parse_file(root, root / "empty.py")
    assert (record.n_lines, record.size, record.parse_ok) == (0, 0, True)
    assert symbols == [] and imports == []


def test_parse_file_record_fields(sample_repo: Path) -> None:
    record, _, _ = parse_file(sample_repo, sample_repo / "mypkg" / "core.py")
    raw = (sample_repo / "mypkg" / "core.py").read_bytes()
    assert record.path == "mypkg/core.py"
    assert record.language == "python"
    assert record.size == len(raw)
    assert record.n_lines == 63
    assert record.parse_ok is True


def test_parse_file_rejects_unknown_extension(make_repo) -> None:
    root = make_repo({"notes.txt": "hello"})
    with pytest.raises(ValueError):
        parse_file(root, root / "notes.txt")


def test_large_file_line_numbers(make_repo) -> None:
    """Rows/columns above 256 must be read safely (py-tree-sitter 0.26.0 Point bug)."""
    body = "".join(f"def f{i}(a, b):\n    return a + b\n\n" for i in range(150))
    long_line = "x = '" + "y" * 400 + "'\n"
    root = make_repo({"big.py": long_line + body + "class Last:\n    " + long_line})
    for _ in range(5):  # repeat: heap corruption does not always crash on first use
        record, symbols, _ = parse_file(root, root / "big.py")
    assert record.n_lines == 1 + 150 * 3 + 2
    assert len(symbols) == 151
    assert (symbols[149].name, symbols[149].start_line, symbols[149].end_line) == ("f149", 449, 450)
    last = symbols[-1]
    assert (last.name, last.start_line, last.end_line) == ("Last", 452, 453)
