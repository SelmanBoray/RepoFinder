"""Tests for the JavaScript side of repofinder.parser (.js, .jsx, .mjs, .cjs).

Written from the contract, not from the implementation. Shared constructs follow the TypeScript
conventions (see test_parser_typescript.py); the CommonJS/prototype patterns are JavaScript-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from repofinder.ingest import build_index
from repofinder.models import FileRecord, ImportEdge, RepoIndex, Symbol
from repofinder.parser import (
    extensions_for,
    language_for_path,
    parse_file,
    resolve_imports,
    resolve_ts_specifier,
)


def _parse(make_repo, name: str, source: str):
    root = make_repo({name: source})
    return parse_file(root, root / name)


def _symbols(make_repo, name: str, source: str) -> dict[str, list[Symbol]]:
    out: dict[str, list[Symbol]] = {}
    for s in _parse(make_repo, name, source)[1]:
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


def test_javascript_language_is_registered() -> None:
    for path in ("src/a.js", "src/a.jsx", "src/a.mjs", "src/a.cjs"):
        spec = language_for_path(path)
        assert spec is not None, path
        assert spec.name == "javascript", path
    assert set(extensions_for(("javascript",))) == {".js", ".jsx", ".mjs", ".cjs"}
    # neighbours are unaffected
    assert language_for_path("src/a.ts").name == "typescript"
    assert language_for_path("src/a.tsx").name == "tsx"
    assert language_for_path("src/a.py").name == "python"


@pytest.mark.parametrize("name", ["a.js", "a.jsx", "a.mjs", "a.cjs"])
def test_record_language_for_every_extension(make_repo, name) -> None:
    record, symbols, _ = _parse(make_repo, name, "function f() {\n  return 1;\n}\n")
    assert (record.path, record.language, record.parse_ok) == (name, "javascript", True)
    assert [(s.qualname, s.kind) for s in symbols] == [("f", "function")]


# --------------------------------------------------------------------------- #
# Functions, arrow functions, JSDoc
# --------------------------------------------------------------------------- #


_FUNCTIONS = """\
// File header, not attached to anything.

/**
 * Add two numbers.
 *
 * Longer description that is dropped.
 * @param {number} a
 */
function add(a, b) {
  return a + b;
}

/** Doubles a value. */
const double = (x) => x * 2;

/** Legacy wrapper. */
// a line comment in between is skipped
const legacy = function legacyImpl(value) {
  return value;
};

/* not jsdoc */
let handler = async ({ id }) => {
  return id;
};

const LIMIT = 10;

function* counter() {
  yield 1;
}

async function outer() {
  function inner() {
    return 1;
  }
  return inner();
}
"""


def test_function_declaration_and_jsdoc(make_repo) -> None:
    symbols = _symbols(make_repo, "m.js", _FUNCTIONS)
    add = _one(symbols, "add")
    assert (add.id, add.path, add.name, add.qualname) == ("m.js::add", "m.js", "add", "add")
    assert _shape(add) == ("function", 9, 11, None)
    assert add.docstring == "Add two numbers."  # first paragraph only, @tags dropped
    assert "add" in add.signature


def test_variable_functions(make_repo) -> None:
    symbols = _symbols(make_repo, "m.js", _FUNCTIONS)
    double = _one(symbols, "double")
    assert _shape(double) == ("function", 14, 14, None)
    assert double.docstring == "Doubles a value."

    legacy = _one(symbols, "legacy")  # named function expression: the variable names it
    assert _shape(legacy) == ("function", 18, 20, None)
    assert legacy.docstring == "Legacy wrapper."

    handler = _one(symbols, "handler")
    assert _shape(handler) == ("function", 23, 25, None)
    assert handler.docstring is None  # /* ... */ is not JSDoc

    assert "LIMIT" not in symbols  # a plain constant is not a symbol


def test_generators_and_nested_functions(make_repo) -> None:
    symbols = _symbols(make_repo, "m.js", _FUNCTIONS)
    assert _shape(_one(symbols, "counter")) == ("function", 29, 31, None)
    outer = _one(symbols, "outer")
    assert _shape(outer) == ("function", 33, 38, None)
    inner = _one(symbols, "outer.<locals>.inner")
    assert _shape(inner) == ("function", 34, 36, "outer")
    assert inner.name == "inner"


def test_symbols_point_at_source(make_repo) -> None:
    source = _FUNCTIONS
    lines = source.splitlines()
    for s in _parse(make_repo, "m.js", source)[1]:
        assert s.id == f"{s.path}::{s.qualname}"
        assert s.qualname.split(".")[-1] == s.name
        assert 1 <= s.start_line <= s.end_line <= len(lines), s
        assert any(s.name in line for line in lines[s.start_line - 1 : s.end_line]), s
        assert lines[s.end_line - 1].strip(), f"{s.id} ends on a blank line"


# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #


_CLASSES = """\
/** A widget with every kind of member. */
class Widget extends Base {
  count = 0;

  constructor(name) {
    super();
    this.name = name;
  }

  /** Current name. */
  get label() {
    return this.name;
  }

  set label(value) {
    this.name = value;
  }

  /** Load it. */
  async load(id) {
    const parse = (s) => s;
    return parse(id);
  }

  static create() {
    return new Widget("x");
  }

  /** Arrow property keeps this. */
  onClick = (event) => {
    this.count += 1;
  };

  static make = function () {
    return 1;
  };

  #secret() {
    return 1;
  }
}

const Point = class {
  norm() {
    return 0;
  }
};
"""


def test_class_members(make_repo) -> None:
    symbols = _symbols(make_repo, "c.js", _CLASSES)
    widget = _one(symbols, "Widget")
    assert _shape(widget) == ("class", 2, 41, None)
    assert widget.docstring == "A widget with every kind of member."

    ctor = _one(symbols, "Widget.constructor")
    assert _shape(ctor) == ("method", 5, 8, "Widget")

    getter, setter = symbols["Widget.label"]  # like a Python property: one id
    assert (getter.kind, getter.start_line, getter.end_line) == ("method", 11, 13)
    assert (setter.kind, setter.start_line, setter.end_line) == ("method", 15, 17)
    assert getter.docstring == "Current name."
    assert getter.id == setter.id == "c.js::Widget.label"

    load = _one(symbols, "Widget.load")
    assert _shape(load) == ("method", 20, 23, "Widget")
    assert load.docstring == "Load it."
    nested = _one(symbols, "Widget.load.<locals>.parse")
    assert _shape(nested) == ("function", 21, 21, "Widget.load")  # function, not method

    create = _one(symbols, "Widget.create")
    assert _shape(create) == ("method", 25, 27, "Widget")

    assert _shape(_one(symbols, "Widget.#secret")) == ("method", 38, 40, "Widget")

    # Plain fields are not symbols.
    assert "Widget.count" not in symbols


def test_class_fields_holding_functions(make_repo) -> None:
    """Same kind TS gives a public arrow-function field ("method")."""
    symbols = _symbols(make_repo, "c.js", _CLASSES)
    on_click = _one(symbols, "Widget.onClick")
    assert _shape(on_click) == ("method", 30, 32, "Widget")
    assert on_click.name == "onClick"
    assert on_click.docstring == "Arrow property keeps this."

    make = _one(symbols, "Widget.make")  # static field holding a function expression
    assert _shape(make) == ("method", 34, 36, "Widget")
    assert make.name == "make"


def test_class_expression(make_repo) -> None:
    symbols = _symbols(make_repo, "c.js", _CLASSES)
    point = _one(symbols, "Point")
    assert _shape(point) == ("class", 43, 47, None)
    assert _shape(_one(symbols, "Point.norm")) == ("method", 44, 46, "Point")


def test_anonymous_default_exports_are_named_default(make_repo) -> None:
    root = make_repo(
        {
            "fn.js": "export default function () {}\n",
            "arrow.js": "export default () => 1;\n",
            "cls.js": "/** Doc. */\nexport default class {\n  m() {}\n}\n",
        }
    )
    got = {}
    for name in ("fn.js", "arrow.js", "cls.js"):
        _, symbols, _ = parse_file(root, root / name)
        got[name] = [(s.qualname, s.kind) for s in symbols]
    assert got["fn.js"] == [("default", "function")]
    assert got["arrow.js"] == [("default", "function")]
    assert got["cls.js"] == [("default", "class"), ("default.m", "method")]


# --------------------------------------------------------------------------- #
# Export forms
# --------------------------------------------------------------------------- #


def test_export_forms(make_repo) -> None:
    source = """\
/** Named function. */
export function a() {
  return 1;
}

export const b = () => 2;

export async function c() {}

export class D {
  m() {}
}

/** Default one. */
export default function e() {}
"""
    symbols = _symbols(make_repo, "e.mjs", source)
    assert _shape(_one(symbols, "a")) == ("function", 2, 4, None)
    assert _one(symbols, "a").docstring == "Named function."
    assert _shape(_one(symbols, "b")) == ("function", 6, 6, None)
    assert _shape(_one(symbols, "c")) == ("function", 8, 8, None)
    assert _shape(_one(symbols, "D")) == ("class", 10, 12, None)
    assert _shape(_one(symbols, "D.m")) == ("method", 11, 11, "D")
    e = _one(symbols, "e")
    assert (e.kind, e.start_line, e.end_line) == ("function", 15, 15)
    assert e.docstring == "Default one."


# --------------------------------------------------------------------------- #
# JSX
# --------------------------------------------------------------------------- #


_JSX = "function App() {\n  return <div className=\"x\">{name}</div>;\n}\n"


@pytest.mark.parametrize("name", ["App.jsx", "App.js"])
def test_jsx_parses(make_repo, name) -> None:
    record, symbols, _ = _parse(make_repo, name, _JSX)
    assert (record.language, record.parse_ok) == ("javascript", True)
    (app,) = symbols
    assert (app.qualname, app.kind) == ("App", "function")
    assert (app.start_line, app.end_line) == (1, 3)


def test_jsx_arrow_component_and_import(make_repo) -> None:
    source = (
        'import React, { useState } from "react";\n'
        "export const Btn = ({ label }) => <button>{label}</button>;\n"
    )
    record, symbols, imports = _parse(make_repo, "Btn.jsx", source)
    assert record.parse_ok is True
    assert [(s.qualname, s.kind) for s in symbols] == [("Btn", "function")]
    assert [(e.dst_module, e.names) for e in imports] == [("react", ["default", "useState"])]


# --------------------------------------------------------------------------- #
# CommonJS / prototype patterns (JavaScript only)
# --------------------------------------------------------------------------- #


_COMMONJS = """\
function Animal() {}

/** Speak up. */
Animal.prototype.speak = function () {
  return "...";
};

Animal.prototype.run = () => {
  return 1;
};

/**
 * Helper f.
 */
exports.f = function () {
  return 1;
};

module.exports.g = () => {
  return 2;
};

/** The handler. */
app.handle = function handle(req, res) {
  res.end();
};

module.exports = function createApp() {
  return {};
};
"""


def test_prototype_methods(make_repo) -> None:
    symbols = _symbols(make_repo, "cjs.js", _COMMONJS)
    speak = _one(symbols, "Animal.speak")
    assert _shape(speak) == ("method", 4, 6, "Animal")
    assert speak.name == "speak"
    assert speak.docstring == "Speak up."
    run = _one(symbols, "Animal.run")  # arrow too
    assert _shape(run) == ("method", 8, 10, "Animal")


def test_exports_assignments(make_repo) -> None:
    symbols = _symbols(make_repo, "cjs.js", _COMMONJS)
    f = _one(symbols, "f")
    assert _shape(f) == ("function", 15, 17, None)
    assert f.name == "f"
    assert f.docstring == "Helper f."  # JSDoc above the statement
    g = _one(symbols, "g")
    assert _shape(g) == ("function", 19, 21, None)
    assert g.name == "g"


def test_object_member_assignment(make_repo) -> None:
    symbols = _symbols(make_repo, "cjs.js", _COMMONJS)
    handle = _one(symbols, "app.handle")
    assert _shape(handle) == ("function", 24, 26, None)
    assert handle.name == "handle"
    assert handle.docstring == "The handler."


def test_module_exports_function(make_repo) -> None:
    symbols = _symbols(make_repo, "cjs.js", _COMMONJS)
    create = _one(symbols, "createApp")
    assert _shape(create) == ("function", 28, 30, None)
    assert create.name == "createApp"


def test_anonymous_module_exports_is_not_a_symbol(make_repo) -> None:
    _, symbols, _ = _parse(make_repo, "anon.js", "module.exports = function () {\n  return 1;\n};\n")
    assert symbols == []


def test_commonjs_symbol_ids(make_repo) -> None:
    _, symbols, _ = _parse(make_repo, "cjs.js", _COMMONJS)
    ids = {s.id for s in symbols}
    assert {
        "cjs.js::Animal",
        "cjs.js::Animal.speak",
        "cjs.js::f",
        "cjs.js::g",
        "cjs.js::app.handle",
        "cjs.js::createApp",
    } <= ids


# --------------------------------------------------------------------------- #
# Syntax errors
# --------------------------------------------------------------------------- #


def test_syntax_error_is_flagged_and_symbols_recovered(make_repo) -> None:
    source = "function before() {\n  return 1;\n}\n\nconst x = ;\n"
    record, symbols, _ = _parse(make_repo, "broken.js", source)
    assert record.parse_ok is False
    by_name = {s.qualname: s for s in symbols}
    assert (by_name["before"].start_line, by_name["before"].end_line) == (1, 3)


# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #


_IMPORTS = """\
import a from "./a";
import { alpha as al, beta } from "../shared/b.js";
import * as ns from "some-pkg";
import "./polyfill";
import d, { named } from "./lib";
export { reexported } from "./y";
export * from "./z";
export * as zz from "./zz.mjs";
const cjs = require("./cjs-module");
const { pick, omit } = require("lodash");
const lazy = () => import("./lazy-mod.js");

async function load() {
  const { feature } = await import("./features");
  return feature;
}
"""


def test_import_extraction(make_repo) -> None:
    _, symbols, imports = _parse(make_repo, "imports.js", _IMPORTS)
    assert [s.qualname for s in symbols] == ["lazy", "load"]
    got = [(e.dst_module, e.names, e.line) for e in imports]
    assert got == [
        ("./a", ["default"], 1),
        ("../shared/b.js", ["alpha", "beta"], 2),  # original names, not aliases
        ("some-pkg", ["*"], 3),
        ("./polyfill", [], 4),
        ("./lib", ["default", "named"], 5),
        ("./y", ["reexported"], 6),
        ("./z", ["*"], 7),
        ("./zz.mjs", ["*"], 8),
        ("./cjs-module", [], 9),  # require()
        ("lodash", ["pick", "omit"], 10),  # destructured require()
        ("./lazy-mod.js", [], 11),  # dynamic import()
        ("./features", ["feature"], 14),  # destructured await import()
    ]
    assert all(e.src_path == "imports.js" and e.dst_path is None for e in imports)


def test_cjs_require_in_cjs_file(make_repo) -> None:
    _, _, imports = _parse(make_repo, "x.cjs", 'const fs = require("node:fs");\nconst b = require("./b");\n')
    assert [(e.dst_module, e.line) for e in imports] == [("node:fs", 1), ("./b", 2)]


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def test_resolve_js_specifiers() -> None:
    assert resolve_ts_specifier("src/a.js", "./b", {"src/b.js"}) == "src/b.js"
    assert resolve_ts_specifier("src/a.js", "./lib", {"src/lib/index.js"}) == "src/lib/index.js"
    assert resolve_ts_specifier("src/a.ts", "./legacy", {"src/legacy.js"}) == "src/legacy.js"
    assert resolve_ts_specifier("src/main.ts", "./a", {"src/a.ts", "src/a.js"}) == "src/a.ts"
    assert resolve_ts_specifier("src/a.js", "./b.cjs", {"src/b.cjs"}) == "src/b.cjs"
    assert resolve_ts_specifier("src/a.js", "lodash", {"src/b.js", "lodash.js"}) is None


def test_resolve_js_extension_and_index_order() -> None:
    def resolve(spec: str, known: set[str]) -> str | None:
        return resolve_ts_specifier("src/a.js", spec, known)

    assert resolve("./b", {"src/b.jsx", "src/b.mjs", "src/b.cjs"}) == "src/b.jsx"
    assert resolve("./b", {"src/b.mjs", "src/b.cjs"}) == "src/b.mjs"
    assert resolve("./b", {"src/b.cjs"}) == "src/b.cjs"
    assert resolve("./b", {"src/b.js", "src/b.ts"}) == "src/b.ts"  # TS candidates stay first
    assert resolve("./b", {"src/b.js", "src/b/index.js"}) == "src/b.js"  # a file beats an index
    assert resolve("./d", {"src/d/index.jsx", "src/d/index.mjs"}) == "src/d/index.jsx"
    assert resolve("./d", {"src/d/index.mjs", "src/d/index.cjs"}) == "src/d/index.mjs"
    assert resolve("./d", {"src/d/index.cjs"}) == "src/d/index.cjs"
    assert resolve("./d", {"src/d/index.js", "src/d/index.ts"}) == "src/d/index.ts"
    assert resolve("./missing", {"src/b.js"}) is None
    assert resolve("../../x", {"x.js"}) is None  # leaves the repository


def test_resolve_imports_on_index() -> None:
    files = ["src/a.js", "src/b.js", "src/lib/index.js", "src/legacy.mjs"]
    index = RepoIndex(name="r", root="/r", source="/r")
    index.files = [FileRecord(p, language_for_path(p).name, 0, 0, True) for p in files]
    edges = [
        ImportEdge("src/a.js", "./b", None, [], 1),
        ImportEdge("src/a.js", "./lib", None, [], 2),
        ImportEdge("src/a.js", "./legacy", None, [], 3),
        ImportEdge("src/a.js", "lodash", None, [], 4),
    ]
    index.imports = edges
    resolve_imports(index)
    assert [e.dst_path for e in edges] == ["src/b.js", "src/lib/index.js", "src/legacy.mjs", None]


def test_end_to_end_require_resolution(make_repo) -> None:
    root = make_repo(
        {
            "a.js": 'const b = require("./b");\nconst _ = require("lodash");\nfunction run() { return b; }\n',
            "b.js": "module.exports = function helper() { return 1; };\n",
        }
    )
    index = build_index(root, "r", str(root))
    assert {f.path: f.language for f in index.files} == {"a.js": "javascript", "b.js": "javascript"}
    by_module = {e.dst_module: e for e in index.imports if e.src_path == "a.js"}
    assert by_module["./b"].dst_path == "b.js"
    assert by_module["lodash"].dst_path is None
    assert {s.id for s in index.symbols} >= {"a.js::run", "b.js::helper"}


def test_file_candidates_beat_directory_indexes() -> None:
    def resolve(known: set[str]) -> str | None:
        return resolve_ts_specifier("src/a.js", "./b", known)

    assert resolve({"src/b.js", "src/b/index.ts"}) == "src/b.js"  # JS file beats a TS index
    assert resolve({"src/b/index.ts"}) == "src/b/index.ts"
    assert resolve({"src/b/index.js", "src/b/index.ts"}) == "src/b/index.ts"  # TS index before JS index
