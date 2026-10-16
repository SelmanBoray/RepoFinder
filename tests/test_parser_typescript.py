"""Tests for the TypeScript/TSX side of repofinder.parser.

Fixtures live in tests/fixtures/ts/ (data, not tests). The last test parses a
real TypeScript repository (REA) when a checkout is present.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from repofinder.models import FileRecord, ImportEdge, RepoIndex, Symbol
from repofinder.parser import (
    SIGNATURE_MAX_CHARS,
    extensions_for,
    language_for_path,
    parse_file,
    resolve_imports,
    resolve_ts_specifier,
)

TS_FIXTURES = Path(__file__).parent / "fixtures" / "ts"
REA_ROOT = Path(
    r"C:\Users\Selman\AppData\Local\Temp\claude\C--\4162a3fb-b459-4996-adbe-33b24ab53ad0"
    r"\scratchpad\repos\rea"
)


def _parse(name: str) -> tuple[FileRecord, list[Symbol], list[ImportEdge]]:
    return parse_file(TS_FIXTURES, TS_FIXTURES / name)


def _symbols(name: str) -> dict[str, list[Symbol]]:
    out: dict[str, list[Symbol]] = {}
    for s in _parse(name)[1]:
        out.setdefault(s.qualname, []).append(s)
    return out


def _one(symbols: dict[str, list[Symbol]], qualname: str) -> Symbol:
    matches = symbols[qualname]
    assert len(matches) == 1, matches
    return matches[0]


def _shape(s: Symbol) -> tuple[str, int, int, str | None]:
    return (s.kind, s.start_line, s.end_line, s.parent)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_typescript_languages_are_registered() -> None:
    assert extensions_for(("typescript",)) == (".ts", ".mts", ".cts")
    assert extensions_for(("tsx",)) == (".tsx",)
    assert language_for_path("src/a.ts").name == "typescript"
    assert language_for_path("src/a.mts").name == "typescript"
    assert language_for_path("src/a.cts").name == "typescript"
    assert language_for_path("types/env.d.ts").name == "typescript"  # declaration files too
    assert language_for_path("src/App.tsx").name == "tsx"
    assert language_for_path("src/a.js") is None  # JavaScript is not indexed yet
    assert language_for_path("src/a.py").name == "python"


# --------------------------------------------------------------------------- #
# Functions, arrow functions, nesting, JSDoc
# --------------------------------------------------------------------------- #


def test_function_declaration() -> None:
    add = _one(_symbols("module.ts"), "add")
    assert add == Symbol(
        id="module.ts::add",
        path="module.ts",
        name="add",
        qualname="add",
        kind="function",
        start_line=11,
        end_line=13,
        signature="function add(a: number, b: number): number",
        # first paragraph only: the second paragraph and @tags are dropped,
        # and the file-header comment above is not attached to anything
        docstring="Add two numbers.",
        parent=None,
    )


def test_multiline_header_drops_comments() -> None:
    fetch = _one(_symbols("module.ts"), "fetchUser")
    assert (fetch.kind, fetch.start_line, fetch.end_line) == ("function", 16, 25)
    assert fetch.signature == (
        "async function fetchUser(id: string, opts?: { cache: boolean },): Promise<User | undefined>"
    )
    assert fetch.docstring == "Fetch a user by id."


def test_nested_functions_use_locals() -> None:
    symbols = _symbols("module.ts")
    normalize = _one(symbols, "fetchUser.<locals>.normalize")
    assert _shape(normalize) == ("function", 20, 23, "fetchUser")
    assert normalize.name == "normalize"
    trim = _one(symbols, "fetchUser.<locals>.normalize.<locals>.trim")
    assert _shape(trim) == ("function", 21, 21, "fetchUser.<locals>.normalize")
    assert trim.signature == "const trim = (s: string) =>"


def test_variable_functions() -> None:
    symbols = _symbols("module.ts")
    double = _one(symbols, "double")
    assert _shape(double) == ("function", 32, 32, None)
    assert double.signature == "const double = (x: number): number =>"
    assert double.docstring == "Doubles a value."

    legacy = _one(symbols, "legacy")  # named function expression: the variable names it
    assert _shape(legacy) == ("function", 36, 38, None)
    assert legacy.signature == "const legacy = function legacyImpl(value: any)"
    assert legacy.docstring == "Legacy wrapper."  # a line comment in between is skipped

    handler = _one(symbols, "handler")
    assert handler.signature == "let handler = async ({ id }: { id: string }) =>"
    assert handler.docstring is None  # /* ... */ is not JSDoc

    # Several declarators: the first starts with its declaration, the others on their own line.
    first, second = _one(symbols, "first"), _one(symbols, "second")
    assert (first.start_line, first.end_line, first.signature) == (64, 64, "var first = () =>")
    assert (second.start_line, second.end_line, second.signature) == (65, 67, "var second = function ()")

    assert "LIMIT" not in symbols  # a plain constant is not a symbol


def test_generators_overloads_and_conditional_definitions() -> None:
    symbols = _symbols("module.ts")
    assert _one(symbols, "counter").signature == "function* counter(): Generator<number>"

    # Overload signatures collapse into the implementation.
    parse = _one(symbols, "parse")
    assert (parse.start_line, parse.end_line) == (49, 51)
    assert parse.signature == "function parse(input: string | number): number"

    assert _shape(_one(symbols, "debugLog")) == ("function", 54, 56, None)
    assert _one(symbols, "detached").docstring is None  # two blank lines detach the comment


def test_symbols_point_at_source() -> None:
    for path in sorted(TS_FIXTURES.iterdir()):
        _, symbols, _ = parse_file(TS_FIXTURES, path)
        lines = path.read_text(encoding="utf-8").splitlines()
        for s in symbols:
            assert s.id == f"{s.path}::{s.qualname}"
            assert s.qualname.split(".")[-1] == s.name
            assert 1 <= s.start_line <= s.end_line <= len(lines), s
            assert any(s.name in line for line in lines[s.start_line - 1 : s.end_line]), s
            assert lines[s.end_line - 1].strip(), f"{s.id} ends on a blank line"
            assert "\n" not in s.signature and len(s.signature) <= SIGNATURE_MAX_CHARS


def test_signature_is_clipped(make_repo) -> None:
    params = ", ".join(f"parameter{i}: number" for i in range(30))
    root = make_repo({"long.ts": f"export function long({params}): void {{}}\n"})
    _, symbols, _ = parse_file(root, root / "long.ts")
    (long,) = symbols
    assert len(long.signature) == SIGNATURE_MAX_CHARS
    assert long.signature.startswith("function long(parameter0: number, ")


# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #


def test_class_members() -> None:
    symbols = _symbols("classes.ts")
    service = _one(symbols, "UserService")
    assert _shape(service) == ("class", 4, 48, None)  # decorator line included
    assert service.signature == "class UserService extends BaseService implements Disposable"
    assert service.docstring == "A service with every kind of member."  # above the decorator

    ctor = _one(symbols, "UserService.constructor")
    assert _shape(ctor) == ("method", 9, 11, "UserService")
    assert ctor.signature == "constructor(private readonly http: Http)"

    getter, setter = symbols["UserService.name"]  # like a Python property: one id
    assert (getter.start_line, getter.end_line, getter.signature) == (14, 16, "get name(): string")
    assert getter.docstring == "Current user name."
    assert (setter.start_line, setter.end_line, setter.signature) == (18, 20, "set name(value: string)")
    assert getter.id == setter.id == "classes.ts::UserService.name"

    load = _one(symbols, "UserService.load")
    assert _shape(load) == ("method", 23, 28, "UserService")  # from the first decorator
    assert load.signature == "async load(id: string): Promise<User>"  # decorators excluded
    assert load.docstring == "Load a user."
    nested = _one(symbols, "UserService.load.<locals>.parse")
    assert _shape(nested) == ("function", 26, 26, "UserService.load")  # function, not method

    assert _one(symbols, "UserService.create").signature == "static create(): UserService"

    on_click = _one(symbols, "UserService.onClick")  # arrow-function property
    assert _shape(on_click) == ("method", 35, 37, "UserService")
    assert on_click.signature == "onClick = (event: MouseEvent): void =>"
    assert on_click.docstring == "Arrow property keeps `this`."

    assert _shape(_one(symbols, "UserService.#secret")) == ("method", 39, 41, "UserService")
    assert _one(symbols, "UserService.literal-name").signature == '["literal-name"](): void'

    # Plain properties and [expression]-named members are not symbols.
    for absent in ("UserService.instances", "UserService.cache"):
        assert absent not in symbols
    assert not any("iterator" in q for q in symbols)


def test_abstract_class_and_class_expression() -> None:
    symbols = _symbols("classes.ts")
    shape = _one(symbols, "Shape")
    assert _shape(shape) == ("class", 50, 55, None)
    assert shape.signature == "abstract class Shape"
    area = _one(symbols, "Shape.area")
    assert _shape(area) == ("method", 51, 51, "Shape")
    assert area.signature == "abstract area(): number"
    assert _shape(_one(symbols, "Shape.describe")) == ("method", 52, 54, "Shape")

    point = _one(symbols, "Point")
    assert _shape(point) == ("class", 57, 61, None)
    assert point.signature == "const Point = class"
    assert _shape(_one(symbols, "Point.norm")) == ("method", 58, 60, "Point")


# --------------------------------------------------------------------------- #
# Namespaces, interfaces, type aliases, enums, declarations
# --------------------------------------------------------------------------- #


def test_nested_namespaces() -> None:
    symbols = _symbols("namespaces.ts")
    geometry = _one(symbols, "Geometry")
    assert _shape(geometry) == ("namespace", 2, 20, None)
    assert geometry.signature == "namespace Geometry"
    assert geometry.docstring == "Geometry helpers."
    assert _shape(_one(symbols, "Geometry.distance")) == ("function", 5, 7, "Geometry")
    assert _shape(_one(symbols, "Geometry.Shapes")) == ("namespace", 9, 19, "Geometry")
    assert _shape(_one(symbols, "Geometry.Shapes.Circle")) == ("class", 10, 14, "Geometry.Shapes")
    assert _shape(_one(symbols, "Geometry.Shapes.Circle.area")) == (
        "method", 11, 13, "Geometry.Shapes.Circle",
    )
    assert _shape(_one(symbols, "Geometry.Shapes.Sized")) == ("interface", 16, 18, "Geometry.Shapes")
    assert "Geometry.ORIGIN" not in symbols

    # A dotted namespace name is one symbol named after its last part.
    config = _one(symbols, "App.Config")
    assert (config.name, config.kind, config.parent) == ("Config", "namespace", "App")
    assert _shape(_one(symbols, "App.Config.load")) == ("function", 23, 23, "App.Config")

    legacy = _one(symbols, "Legacy")
    assert (legacy.kind, legacy.signature) == ("namespace", "module Legacy")
    assert _shape(_one(symbols, "Legacy.helper")) == ("function", 27, 27, "Legacy")

    # declare module "pkg" / declare global: no symbol of their own, members at top level.
    assert _shape(_one(symbols, "patched")) == ("function", 31, 31, None)
    assert _one(symbols, "patched").signature == "function patched(): void"
    assert _shape(_one(symbols, "Window")) == ("interface", 35, 37, None)


def test_interfaces_types_and_enums() -> None:
    symbols = _symbols("types.ts")
    user_id = _one(symbols, "UserId")
    assert _shape(user_id) == ("type", 2, 2, None)
    assert user_id.signature == "type UserId = string"
    assert user_id.docstring == "Identifier of a user."

    result = _one(symbols, "Result")
    assert _shape(result) == ("type", 4, 6, None)
    assert result.signature == "type Result<T, E = Error> = | { ok: true; value: T } | { ok: false; error: E }"

    user = _one(symbols, "User")
    assert _shape(user) == ("interface", 11, 15, None)
    assert user.signature == "interface User extends Entity<UserId>"
    assert user.docstring == "A user record."
    assert "User.greet" not in symbols  # interface members are part of the interface

    assert _one(symbols, "Internal").signature == "interface Internal"

    level = _one(symbols, "Level")
    assert _shape(level) == ("enum", 20, 23, None)
    assert (level.signature, level.docstring) == ("enum Level", "Log levels.")
    assert _one(symbols, "Flag").signature == "const enum Flag"


def test_declarations_without_bodies() -> None:
    symbols = _symbols("types.ts")
    legacy = _one(symbols, "legacyGlobal")
    assert _shape(legacy) == ("function", 29, 29, None)
    assert legacy.signature == "function legacyGlobal(x: number): void"

    remote = _one(symbols, "Remote")
    assert _shape(remote) == ("class", 31, 35, None)
    connect = _one(symbols, "Remote.connect")  # repeated overload signatures: the first one
    assert (connect.start_line, connect.signature) == (32, "connect(url: string): Promise<void>")
    assert _one(symbols, "Remote.close").kind == "method"


def test_declaration_file() -> None:
    record, symbols, _ = _parse("env.d.ts")
    assert (record.language, record.parse_ok) == ("typescript", True)
    assert [(s.qualname, s.kind, s.start_line) for s in symbols] == [
        ("setup", "function", 1),
        ("Env", "namespace", 6),
        ("Env.get", "function", 7),
    ]
    assert symbols[0].signature == "function setup(options?: object): void"


# --------------------------------------------------------------------------- #
# TSX, .mts, syntax errors
# --------------------------------------------------------------------------- #


def test_tsx_components() -> None:
    record, symbols, imports = _parse("App.tsx")
    assert (record.language, record.parse_ok, record.n_lines) == ("tsx", True, 19)
    by_name = {s.qualname: s for s in symbols}
    assert list(by_name) == ["Btn", "Counter", "Counter.<locals>.increment", "App"]
    btn = by_name["Btn"]
    assert _shape(btn) == ("function", 4, 4, None)
    assert btn.signature == "const Btn = ({ label }: { label: string }) =>"
    assert btn.docstring == "A button."
    assert _shape(by_name["Counter"]) == ("function", 6, 10, None)
    assert _shape(by_name["Counter.<locals>.increment"]) == ("function", 8, 8, "Counter")
    app = by_name["App"]
    assert _shape(app) == ("function", 13, 19, None)  # export default function App()
    assert (app.signature, app.docstring) == ("function App()", "The application root.")
    assert [(e.dst_module, e.names, e.line) for e in imports] == [("react", ["useState"], 1)]


def test_tsx_dynamic_import_and_require(make_repo) -> None:
    root = make_repo(
        {
            "routes.tsx": (
                'const Page = lazy(() => import("./Page"));\n'
                'const styles = require("./styles.css");\n'
                "export const Route = () => <Page />;\n"
            )
        }
    )
    record, symbols, imports = parse_file(root, root / "routes.tsx")
    assert record.language == "tsx"
    assert [s.qualname for s in symbols] == ["Route"]  # `lazy(...)` is a call, not a function
    assert [(e.dst_module, e.names, e.line) for e in imports] == [
        ("./Page", [], 1),
        ("./styles.css", [], 2),
    ]


def test_anonymous_default_exports_are_named_default(make_repo) -> None:
    root = make_repo(
        {
            "fn.ts": "export default function () {}\n",
            "arrow.tsx": "export default () => <div />;\n",
            "cls.ts": "/** Doc. */\nexport default class {\n  m() {}\n}\n",
        }
    )
    sigs = {}
    for name in ("fn.ts", "arrow.tsx", "cls.ts"):
        _, symbols, _ = parse_file(root, root / name)
        sigs[name] = [(s.qualname, s.kind, s.signature, s.docstring) for s in symbols]
    assert sigs["fn.ts"] == [("default", "function", "export default function ()", None)]
    assert sigs["arrow.tsx"] == [("default", "function", "export default () =>", None)]
    assert sigs["cls.ts"] == [
        ("default", "class", "export default class", "Doc."),
        ("default.m", "method", "m()", None),
    ]


def test_mts_module() -> None:
    record, symbols, imports = _parse("esm.mts")
    assert (record.path, record.language, record.parse_ok) == ("esm.mts", "typescript", True)
    assert [(s.qualname, s.kind, s.start_line, s.end_line) for s in symbols] == [
        ("readConfig", "function", 5, 7),
        ("Loader", "class", 9, 11),
        ("Loader.load", "method", 10, 10),
    ]
    assert symbols[0].docstring == "Read a config file."
    assert symbols[2].signature == "load = async (path: string) =>"
    assert [(e.dst_module, e.names) for e in imports] == [
        ("node:fs/promises", ["readFile"]),
        ("./helper.mjs", ["helper"]),
    ]


def test_syntax_error_is_flagged_and_symbols_recovered() -> None:
    record, symbols, _ = _parse("broken.ts")
    assert record.parse_ok is False
    by_name = {s.qualname: s for s in symbols}
    assert (by_name["before"].start_line, by_name["before"].end_line) == (1, 3)
    assert by_name["After"].kind == "class"
    assert _shape(by_name["After.method"]) == ("method", 10, 10, "After")


def test_crlf_bom_and_empty_file(make_repo) -> None:
    root = make_repo(
        {
            "crlf.ts": b"\xef\xbb\xbf/** Doc\r\n * line two\r\n */\r\nexport function f(\r\n  a: number,\r\n) {\r\n}\r\n",
            "empty.ts": b"",
        }
    )
    record, symbols, _ = parse_file(root, root / "crlf.ts")
    assert record.parse_ok is True
    (f,) = symbols
    assert (f.start_line, f.end_line, f.signature) == (4, 7, "function f(a: number,)")
    assert f.docstring == "Doc line two"  # the first paragraph, joined into one line
    record, symbols, imports = parse_file(root, root / "empty.ts")
    assert (record.n_lines, record.parse_ok, symbols, imports) == (0, True, [], [])


# --------------------------------------------------------------------------- #
# Imports and resolution
# --------------------------------------------------------------------------- #


def test_import_extraction() -> None:
    _, symbols, imports = _parse("imports.ts")
    assert [s.qualname for s in symbols] == ["lazy"]
    assert [(e.dst_module, e.names, e.line, e.dst_path) for e in imports] == [
        ("./a", ["default"], 1, None),
        ("../shared/b.js", ["alpha", "beta"], 2, None),  # original names, not aliases
        ("some-pkg", ["*"], 3, None),
        ("./types", ["TypeOnly"], 4, None),  # import type
        ("./lib", ["default", "Named", "other"], 5, None),  # inline `type` modifier
        ("./polyfill", [], 6, None),  # side effect
        ("./legacy", [], 7, None),  # import x = require()
        ("./y", ["reexported"], 8, None),  # export {x} from
        ("./z", ["*"], 9, None),  # export * from
        ("./zz.mjs", ["*"], 10, None),  # export * as ns from
        ("./widgets", ["default", "helper"], 11, None),
        ("./cjs-module", [], 12, None),  # require()
        ("lodash", ["pick", "omit"], 13, None),  # destructured require()
        ("./lazy-mod.js", [], 16, None),  # dynamic import()
        ("./features", ["feature"], 17, None),  # destructured await import()
        ("./side-effect", [], 19, None),  # a template with ${} (line 18) is not a string
    ]
    assert all(e.src_path == "imports.ts" for e in imports)


_FAKE_REPO = [
    "src/app/imports.ts",
    "src/app/a.ts",
    "src/shared/b.ts",            # "../shared/b.js": the .js -> .ts rewrite
    "src/app/types/index.ts",     # "./types": directory index
    "src/app/lib.tsx",            # "./lib": .tsx extension
    "src/app/polyfill.d.ts",      # "./polyfill": declaration file
    "src/app/legacy.cts",
    "src/app/y.ts",
    "src/app/y/index.ts",         # a file beats a directory index
    "src/app/z.mts",
    "src/app/zz.mts",             # "./zz.mjs": the .mjs -> .mts rewrite
    "src/app/widgets/index.tsx",
    "src/app/cjs-module.ts",
    "src/app/lazy-mod.ts",
    "src/app/features.ts",
    "pkg/mod.py",
    "main.py",
]


def test_resolution_against_repo_files() -> None:
    _, _, imports = _parse("imports.ts")
    for edge in imports:
        edge.src_path = "src/app/imports.ts"
    index = RepoIndex(name="r", root="/r", source="/r")
    index.files = [
        FileRecord(p, language_for_path(p).name, 0, 0, True) for p in _FAKE_REPO
    ] + [FileRecord("pkg/__init__.py", "python", 0, 0, True)]
    py_edges = [
        ImportEdge("main.py", "pkg.mod", None, [], 1),
        ImportEdge("main.py", "requests", None, [], 2),
    ]
    index.imports = imports + py_edges
    resolve_imports(index)
    assert {e.dst_module: e.dst_path for e in imports} == {
        "./a": "src/app/a.ts",
        "../shared/b.js": "src/shared/b.ts",
        "some-pkg": None,
        "./types": "src/app/types/index.ts",
        "./lib": "src/app/lib.tsx",
        "./polyfill": "src/app/polyfill.d.ts",
        "./legacy": "src/app/legacy.cts",
        "./y": "src/app/y.ts",
        "./z": "src/app/z.mts",
        "./zz.mjs": "src/app/zz.mts",
        "./widgets": "src/app/widgets/index.tsx",
        "./cjs-module": "src/app/cjs-module.ts",
        "lodash": None,
        "./lazy-mod.js": "src/app/lazy-mod.ts",
        "./features": "src/app/features.ts",
        "./side-effect": None,  # no such file
    }
    # Python edges in the same index still go through the Python resolver.
    assert [e.dst_path for e in py_edges] == ["pkg/mod.py", None]


def test_resolve_ts_specifier_order_and_limits() -> None:
    known = {
        "src/a.ts",
        "src/a.tsx",
        "src/a.js",
        "src/b.tsx",
        "src/c/index.tsx",
        "src/c/index.ts",
        "src/explicit.ts",
        "src/index.ts",
        "index.ts",
        "src/types.d.ts",
        "src/legacy.cts",
    }
    resolve = lambda spec: resolve_ts_specifier("src/sub/../file.ts", spec, known)  # noqa: E731
    assert resolve("./a") == "src/a.ts"  # .ts before .tsx
    assert resolve("./a.js") == "src/a.js"  # as written, when that file is known
    assert resolve("./b.js") == "src/b.tsx"  # .js -> .tsx when there is no .ts
    assert resolve("./c") == "src/c/index.ts"  # index.ts before index.tsx
    assert resolve("./c/") == "src/c/index.ts"
    assert resolve("./explicit.ts") == "src/explicit.ts"  # allowImportingTsExtensions
    assert resolve("./types") == "src/types.d.ts"
    assert resolve("./legacy.cjs") == "src/legacy.cts"
    assert resolve(".") == "src/index.ts"
    assert resolve("..") == "index.ts"
    assert resolve("../..") is None  # leaves the repository
    assert resolve("./missing") is None
    assert resolve("react") is None  # bare package
    assert resolve("@scope/pkg/a") is None
    assert resolve("/abs/a") is None
    assert resolve_ts_specifier("src/deep/x.ts", "../a", known) == "src/a.ts"


# --------------------------------------------------------------------------- #
# Real repository smoke test
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.skipif(not (REA_ROOT / "src").is_dir(), reason="REA checkout not present")
def test_rea_smoke(capsys) -> None:
    paths = []
    for dirpath, dirnames, filenames in os.walk(REA_ROOT / "src"):
        dirnames[:] = [d for d in dirnames if d != "node_modules" and not d.startswith(".")]
        paths.extend(Path(dirpath, f) for f in filenames if f.endswith(".ts"))
    paths.sort()

    index = RepoIndex(name="rea", root=str(REA_ROOT), source=str(REA_ROOT))
    start = time.perf_counter()
    for path in paths:
        record, symbols, imports = parse_file(REA_ROOT, path)
        index.files.append(record)
        index.symbols.extend(symbols)
        index.imports.extend(imports)
    parse_seconds = time.perf_counter() - start
    resolve_imports(index)
    total_seconds = time.perf_counter() - start

    n_files = len(index.files)
    failures = sum(1 for f in index.files if not f.parse_ok)
    relative = [e for e in index.imports if e.dst_module.startswith(".")]
    resolved = sum(1 for e in relative if e.dst_path is not None)
    with capsys.disabled():
        print(
            f"\nREA src/**/*.ts: {n_files} files, {sum(f.n_lines for f in index.files)} lines, "
            f"{len(index.symbols)} symbols, {failures} parse failures "
            f"(parse_ok {100 * (n_files - failures) / n_files:.1f}%), "
            f"{len(index.imports)} imports ({resolved}/{len(relative)} relative resolved), "
            f"{parse_seconds:.2f}s parse, {total_seconds:.2f}s total"
        )
    assert n_files > 1000
    assert len(index.symbols) > n_files
    assert failures / n_files < 0.05
    assert resolved / len(relative) > 0.9
