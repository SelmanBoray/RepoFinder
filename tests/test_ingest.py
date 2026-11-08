"""Tests for repofinder.ingest: source detection, fetching, file walking, indexing."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from repofinder import ingest, parser
from repofinder.ingest import (
    MAX_FILE_BYTES,
    IngestError,
    build_index,
    fetch_repo,
    iter_source_files,
    repo_name_from_source,
)

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


# --------------------------------------------------------------------------- #
# repo_name_from_source
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "source",
    [
        "https://github.com/pallets/flask",
        "https://github.com/pallets/flask/",
        "https://github.com/pallets/flask.git",
        "https://github.com/pallets/flask/tree/main",
        "https://github.com/pallets/flask/tree/feature/x/y",
        "http://www.github.com/pallets/flask",
        "github.com/pallets/flask",
        "git@github.com:pallets/flask.git",
    ],
)
def test_repo_name_from_github_urls(source: str) -> None:
    assert repo_name_from_source(source) == "flask"


def test_repo_name_keeps_dots_in_name() -> None:
    assert repo_name_from_source("https://github.com/owner/my.repo.git") == "my.repo"


def test_repo_name_from_local_path(tmp_path: Path) -> None:
    project = tmp_path / "some-project"
    project.mkdir()
    assert repo_name_from_source(str(project)) == "some-project"
    assert repo_name_from_source(str(project) + "/") == "some-project"


# --------------------------------------------------------------------------- #
# fetch_repo
# --------------------------------------------------------------------------- #


def test_fetch_local_directory_is_returned_resolved(sample_repo: Path, tmp_path: Path) -> None:
    got = fetch_repo(str(sample_repo), tmp_path / "dest")
    assert got == sample_repo.resolve()
    assert not (tmp_path / "dest").exists()  # nothing copied


def test_fetch_rejects_unknown_sources(tmp_path: Path) -> None:
    with pytest.raises(IngestError):
        fetch_repo(str(tmp_path / "does-not-exist"), tmp_path)
    with pytest.raises(IngestError):
        fetch_repo("https://gitlab.com/owner/name", tmp_path)


def test_fetch_clone_command_and_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict]] = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="fatal: repository not found")

    monkeypatch.setattr(ingest.subprocess, "run", fake_run)
    with pytest.raises(IngestError, match="repository not found"):
        fetch_repo("https://github.com/owner/name/tree/dev", tmp_path / "repos")

    (cmd, kwargs), = calls
    assert cmd == [
        "git", "clone", "--depth", "1", "--",
        "https://github.com/owner/name.git", str(tmp_path / "repos" / "name"),
    ]
    assert not kwargs.get("shell", False)
    assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"


def test_fetch_clone_success_returns_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).mkdir(parents=True)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ingest.subprocess, "run", fake_run)
    got = fetch_repo("https://github.com/owner/name.git", tmp_path / "repos")
    assert got == (tmp_path / "repos" / "name").resolve()


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@needs_git
def test_fetch_reuses_existing_checkout_of_same_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "repos" / "flask"
    target.mkdir(parents=True)
    _git("init", "-q", cwd=target)
    _git("remote", "add", "origin", "https://github.com/pallets/flask", cwd=target)

    real_run = subprocess.run

    def guarded_run(cmd, **kwargs):
        assert "clone" not in cmd, "must not clone again"
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(ingest.subprocess, "run", guarded_run)
    assert fetch_repo("https://github.com/pallets/flask.git", tmp_path / "repos") == target.resolve()

    # A checkout of a different repository with the same name is not reused.
    with pytest.raises(IngestError, match="checkout of"):
        fetch_repo("https://github.com/someone-else/flask", tmp_path / "repos")


def test_fetch_refuses_non_empty_non_git_target(tmp_path: Path) -> None:
    target = tmp_path / "repos" / "name"
    target.mkdir(parents=True)
    (target / "file.txt").write_text("x")
    with pytest.raises(IngestError, match="not a git checkout"):
        fetch_repo("https://github.com/owner/name", tmp_path / "repos")


# --------------------------------------------------------------------------- #
# iter_source_files
# --------------------------------------------------------------------------- #


def test_iter_source_files_skip_rules(make_repo) -> None:
    skipped_dirs = [
        ".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".tox",
        ".nox", ".eggs", "build", "dist", "site-packages", ".hidden", ".github",
    ]
    files: dict[str, str | bytes] = {f"{d}/x.py": "x = 1\n" for d in skipped_dirs}
    files.update(
        {
            "lib/python3.12/site-packages/dep/mod.py": "x = 1\n",  # nested skip dir
            "pkg/build/gen.py": "x = 1\n",
            "pkg/__init__.py": "",
            "pkg/sub/mod.py": "x = 1\n",
            "pkg/a.py": "x = 1\n",
            "pkg.py": "x = 1\n",
            "tests/test_pkg.py": "def test(): pass\n",  # tests are kept
            "README.md": "# readme\n",
            "setup.cfg": "[metadata]\n",
            "stub.pyi": "x: int\n",
            "big.py": b"#" * (MAX_FILE_BYTES + 1),
            "exactly_limit.py": b"#" * MAX_FILE_BYTES,
        }
    )
    root = make_repo(files)
    got = [p.relative_to(root).as_posix() for p in iter_source_files(root)]
    assert got == [
        "exactly_limit.py",
        "pkg.py",
        "pkg/__init__.py",
        "pkg/a.py",
        "pkg/sub/mod.py",
        "tests/test_pkg.py",
    ]
    assert all(p.is_absolute() for p in iter_source_files(root.resolve()))


def test_iter_source_files_unknown_language(make_repo) -> None:
    root = make_repo({"a.py": ""})
    with pytest.raises(ValueError, match="unsupported language"):
        list(iter_source_files(root, languages=("cobol",)))


# --------------------------------------------------------------------------- #
# build_index
# --------------------------------------------------------------------------- #


def test_build_index_on_sample_repo(sample_repo: Path) -> None:
    index = build_index(sample_repo, "sample_repo", "local-sample")
    assert index.name == "sample_repo"
    assert index.source == "local-sample"
    assert index.root == str(sample_repo.resolve())
    assert [f.path for f in index.files] == [
        "mypkg/__init__.py",
        "mypkg/broken.py",
        "mypkg/core.py",
        "mypkg/sub/__init__.py",
        "mypkg/sub/models.py",
    ]
    assert len(index.symbols) == 21
    assert {s.kind for s in index.symbols} == {"function", "class", "method"}
    assert [(s.path, s.start_line) for s in index.symbols] == sorted(
        (s.path, s.start_line) for s in index.symbols
    )
    assert [(e.src_path, e.line) for e in index.imports] == sorted(
        (e.src_path, e.line) for e in index.imports
    )
    assert len(index.imports) == 17
    assert sum(e.dst_path is not None for e in index.imports) == 10


def test_build_index_survives_a_file_that_cannot_be_parsed(
    make_repo, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_repo({"good.py": "def ok():\n    pass\n", "bad.py": "def boom():\n    pass\n"})
    real_parse = parser.parse_file

    def flaky_parse(root_arg, path):
        if Path(path).name == "bad.py":
            raise RuntimeError("simulated parser crash")
        return real_parse(root_arg, path)

    monkeypatch.setattr(parser, "parse_file", flaky_parse)
    index = build_index(root, "r", str(root))
    assert [(f.path, f.parse_ok) for f in index.files] == [("bad.py", False), ("good.py", True)]
    bad = index.files[0]
    assert bad.language == "python" and bad.n_lines == 2 and bad.size > 0
    assert [s.name for s in index.symbols] == ["ok"]


def test_build_index_handles_undecodable_and_broken_files(make_repo) -> None:
    root = make_repo(
        {
            "binary.py": b"\x00\xff\xfe garbage \x80\x81\n",
            "broken.py": "def f(:\n",
            "ok.py": "class A:\n    pass\n",
        }
    )
    index = build_index(root, "r", str(root))
    flags = {f.path: f.parse_ok for f in index.files}
    assert flags["ok.py"] is True
    assert flags["broken.py"] is False
    assert "binary.py" in flags


# --------------------------------------------------------------------------- #
# TypeScript repositories: ignore rules, default languages, size limit
# --------------------------------------------------------------------------- #


def _stub_language(name: str, extensions: tuple[str, ...]) -> parser.LanguageSpec:
    """A registered language that is never parsed (these tests only walk files)."""
    return parser.LanguageSpec(name=name, extensions=extensions, grammar=lambda: None,
                               extractor=lambda source, tree: ([], []))


@pytest.fixture
def ts_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(parser.LANGUAGES, "typescript", _stub_language("typescript", (".ts", ".mts", ".cts")))
    monkeypatch.setitem(parser.LANGUAGES, "tsx", _stub_language("tsx", (".tsx",)))
    monkeypatch.setitem(parser.LANGUAGES, "javascript", _stub_language("javascript", (".js", ".mjs", ".cjs")))


def test_ts_repo_skip_rules(make_repo, ts_registered) -> None:
    skipped_dirs = [
        "node_modules", "dist", "build", "out", ".next", "coverage", "third_party", "licenses",
        "vendor",  # next to package.json: installed dependencies
    ]
    files: dict[str, str | bytes] = {f"{d}/x.ts": "export const x = 1;\n" for d in skipped_dirs}
    files.update(
        {
            "packages/app/node_modules/dep/index.ts": "export {};\n",   # nested skip dirs
            "packages/app/dist/index.js": "module.exports = {};\n",
            "src/index.ts": "export {};\n",
            "src/view.tsx": "export const V = () => null;\n",
            "src/lib.mts": "export {};\n",
            "src/legacy.cts": "export {};\n",
            "src/app.js": "export {};\n",
            "src/bundle.min.js": "!function(){}();\n",                   # minified bundles
            "src/vendor.min.mjs": "export{};\n",
            "src/__tests__/app.test.ts": "test('x', () => {});\n",       # tests are kept
            "tests/boundary/a.test.ts": "test('a', () => {});\n",
            "website/src/main.ts": "export {};\n",                       # so are website/ and experiments/
            "experiments/probe.mts": "export {};\n",
            "tools/gen.py": "x = 1\n",
            "README.md": "# readme\n",
            "package.json": "{}\n",
        }
    )
    root = make_repo(files)
    scan = ingest.scan_source_files(root)
    got = [p.relative_to(root).as_posix() for p in scan.files]
    assert got == sorted(
        [
            "experiments/probe.mts",
            "src/__tests__/app.test.ts",
            "src/app.js",
            "src/index.ts",
            "src/legacy.cts",
            "src/lib.mts",
            "src/view.tsx",
            "tests/boundary/a.test.ts",
            "tools/gen.py",
            "website/src/main.ts",
        ]
    )
    assert scan.minified == ["src/bundle.min.js", "src/vendor.min.mjs"]
    assert scan.too_large == []
    assert list(iter_source_files(root)) == scan.files
    assert {"node_modules", "dist", "build", "out", ".next", "coverage", "third_party", "licenses"} <= ingest.SKIP_DIRS
    assert not {"tests", "website", "experiments", "src", "docs", "examples"} & ingest.SKIP_DIRS


def test_vendor_is_skipped_only_next_to_a_package_manifest(make_repo) -> None:
    root = make_repo(
        {
            "vendor/thirdparty/json/__init__.py": "",          # a Python project's vendored code: indexed
            "web/package.json": "{}",
            "web/vendor/dep.py": "x = 1\n",                    # npm-style dependencies: skipped
            "svc/go.mod": "module svc\n",
            "svc/vendor/lib/x.py": "x = 1\n",
            "svc/tools/vendor/keep.py": "x = 1\n",            # no manifest next to this one
        }
    )
    got = [p.relative_to(root).as_posix() for p in iter_source_files(root)]
    assert got == ["svc/tools/vendor/keep.py", "vendor/thirdparty/json/__init__.py"]


def test_default_languages_are_every_registered_language(make_repo, ts_registered) -> None:
    root = make_repo({"a.py": "x = 1\n", "b.ts": "let b = 1;\n", "c.tsx": "", "d.md": ""})
    assert ingest.default_languages() == tuple(parser.LANGUAGES)
    assert {"python", "typescript", "tsx"} <= set(ingest.default_languages())
    assert [p.name for p in iter_source_files(root)] == ["a.py", "b.ts", "c.tsx"]
    assert [p.name for p in iter_source_files(root, languages=("python",))] == ["a.py"]
    assert [p.name for p in iter_source_files(root, languages=("typescript",))] == ["b.ts"]


def test_files_over_the_size_limit_are_counted_not_recorded(make_repo, caplog) -> None:
    assert MAX_FILE_BYTES == 2 * 1024 * 1024
    root = make_repo(
        {
            "big.py": b"#" * (MAX_FILE_BYTES + 1),
            "pkg/huge.py": b"#" * (MAX_FILE_BYTES + 10),
            "ok.py": "x = 1\n",
        }
    )
    assert ingest.scan_source_files(root).too_large == ["big.py", "pkg/huge.py"]
    with caplog.at_level(logging.WARNING, logger="repofinder.ingest"):
        index = build_index(root, "r", str(root))
    assert [f.path for f in index.files] == ["ok.py"]
    assert "skipped 2 files larger than 2 MiB: big.py, pkg/huge.py" in caplog.text


def test_dart_repo_skip_rules(make_repo, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(parser.LANGUAGES, "dart", _stub_language("dart", (".dart",)))
    generated = ["a.g.dart", "a.freezed.dart", "a.mocks.dart", "a.gr.dart", "a.pb.dart"]
    files: dict[str, str | bytes] = {f"lib/src/{n}": "// generated\n" for n in generated}
    files.update(
        {
            ".dart_tool/package_config/x.dart": "void x() {}\n",   # dot directory
            "build/app/outputs/gen.dart": "void g() {}\n",          # Flutter build output
            "lib/main.dart": "void main() {}\n",
            "lib/src/model.dart": "class M {}\n",
            "lib/src/graph.dart": "class G {}\n",                  # "g.dart" is not a generated suffix
            "test/widget_test.dart": "void main() {}\n",
            "integration_test/app_test.dart": "void main() {}\n",
            "android/app/src/main/Gen.kt": "class Gen\n",          # platform folders hold no .dart
            "ios/Runner/AppDelegate.swift": "import UIKit\n",
            "pubspec.yaml": "name: x\n",
        }
    )
    root = make_repo(files)
    got = [p.relative_to(root).as_posix() for p in iter_source_files(root)]
    assert got == [
        "integration_test/app_test.dart",
        "lib/main.dart",
        "lib/src/graph.dart",
        "lib/src/model.dart",
        "test/widget_test.dart",
    ]
    scan = ingest.scan_source_files(root)
    assert scan.minified == sorted(f"lib/src/{n}" for n in generated)


# --------------------------------------------------------------------------- #
# Go / Rust / Java / JavaScript repositories: generated code, build output
# --------------------------------------------------------------------------- #


@pytest.fixture
def polyglot_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(parser.LANGUAGES, "go", _stub_language("go", (".go",)))
    monkeypatch.setitem(parser.LANGUAGES, "rust", _stub_language("rust", (".rs",)))
    monkeypatch.setitem(parser.LANGUAGES, "java", _stub_language("java", (".java",)))
    monkeypatch.setitem(parser.LANGUAGES, "typescript", _stub_language("typescript", (".ts", ".mts", ".cts")))
    monkeypatch.setitem(parser.LANGUAGES, "tsx", _stub_language("tsx", (".tsx",)))
    monkeypatch.setitem(parser.LANGUAGES, "javascript", _stub_language("javascript", (".js", ".jsx", ".mjs", ".cjs")))


def _kept(root: Path) -> list[str]:
    return [p.relative_to(root).as_posix() for p in iter_source_files(root)]


def test_target_is_skipped_next_to_cargo_toml(make_repo, polyglot_registered) -> None:
    root = make_repo(
        {
            "Cargo.toml": "[package]\n",
            "src/main.rs": "fn main() {}\n",
            "target/debug/build/x/out/gen.rs": "fn g() {}\n",
        }
    )
    assert _kept(root) == ["src/main.rs"]


def test_target_is_skipped_next_to_pom_xml(make_repo, polyglot_registered) -> None:
    root = make_repo(
        {
            "pom.xml": "<project/>\n",
            "src/main/java/A.java": "class A {}\n",
            "target/generated-sources/B.java": "class B {}\n",
        }
    )
    assert _kept(root) == ["src/main/java/A.java"]


def test_target_without_a_manifest_next_to_it_is_walked(make_repo, polyglot_registered) -> None:
    root = make_repo({"src/target/t.rs": "fn t() {}\n"})
    assert _kept(root) == ["src/target/t.rs"]


def test_target_manifest_must_be_in_the_same_directory(make_repo, polyglot_registered) -> None:
    root = make_repo(
        {
            "Cargo.toml": "[workspace]\n",
            "crates/a/target/t.rs": "fn t() {}\n",     # manifest is two levels up, not next to target/
        }
    )
    assert _kept(root) == ["crates/a/target/t.rs"]


def test_generated_go_files_are_skipped(make_repo, polyglot_registered) -> None:
    root = make_repo(
        {
            "api/api.pb.go": "package api\n",
            "api/api.pb.gw.go": "package api\n",
            "apis/zz_generated.deepcopy.go": "package apis\n",
            "pkg/stringer_string.go": (
                "// Copyright 2020 The Authors.\n"
                "// Licensed under the Apache License, Version 2.0.\n"
                "\n"
                "// Code generated by stringer; DO NOT EDIT.\n"
                "\n"
                "package x\n"
            ),
            "pkg/late.go": (
                "package x\n"
                "\n"
                "// Code generated by hand; DO NOT EDIT.\n"
                "func F() {}\n"
            ),
            "cmd/server.go": "package main\n\nfunc main() {}\n",
        }
    )
    assert _kept(root) == ["cmd/server.go", "pkg/late.go"]


def test_javascript_compiled_next_to_typescript_is_skipped(make_repo, polyglot_registered) -> None:
    root = make_repo(
        {
            "src/a.ts": "export {};\n",
            "src/a.js": "exports.a = 1;\n",
            "src/c.tsx": "export const C = 1;\n",
            "src/c.js": "exports.c = 1;\n",
            "src/d.tsx": "export const D = 1;\n",
            "src/d.jsx": "exports.d = 1;\n",
            "src/e.mts": "export {};\n",
            "src/e.mjs": "export {};\n",
            "src/f.cts": "export {};\n",
            "src/f.cjs": "exports.f = 1;\n",
        }
    )
    assert _kept(root) == ["src/a.ts", "src/c.tsx", "src/d.tsx", "src/e.mts", "src/f.cts"]


def test_javascript_is_kept_without_a_typescript_source_beside_it(make_repo, polyglot_registered) -> None:
    root = make_repo(
        {
            "index.js": "module.exports = {};\n",
            "index.d.ts": "export {};\n",            # a declaration file is not the source of index.js
            "index.d.cts": "export {};\n",
            "src/b.js": "exports.b = 1;\n",
            "lib/b.ts": "export {};\n",              # different directory
            "src/app.js": "exports.app = 1;\n",
        }
    )
    assert _kept(root) == ["index.d.cts", "index.d.ts", "index.js", "lib/b.ts", "src/app.js", "src/b.js"]


def test_vendor_and_minified_rules_still_apply_to_go_and_js(make_repo, polyglot_registered) -> None:
    root = make_repo(
        {
            "go.mod": "module m\n",
            "vendor/dep/dep.go": "package dep\n",
            "main.go": "package main\n",
            "web/vendor/keep.js": "exports.k = 1;\n",
            "web/app.min.js": "!function(){}();\n",
        }
    )
    assert _kept(root) == ["main.go", "web/vendor/keep.js"]
    assert ingest.scan_source_files(root).minified == ["web/app.min.js"]


def test_go_symlink_is_never_opened(make_repo, polyglot_registered, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_repo({"real.go": "package main\n", "other.go": "package main\n"})
    link = root / "link.go"
    try:
        link.symlink_to(root / "real.go")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks cannot be created here")
    opened: list[str] = []
    real_open = open

    def recorder(file, *args, **kwargs):
        opened.append(Path(file).name)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(ingest, "open", recorder, raising=False)
    got = _kept(root)
    assert "link.go" not in got
    assert got == ["other.go", "real.go"]
    assert "link.go" not in opened


def test_go_bom_header_is_recognised_as_generated(make_repo, polyglot_registered) -> None:
    bom = b"\xef\xbb\xbf"
    root = make_repo(
        {
            "gen.go": bom + b"// Code generated by x; DO NOT EDIT.\n\npackage p\n",
            "tmpl.go": bom + (
                b"package main\n\n"
                b"const tmpl = `\n"
                b"// Code generated by x; DO NOT EDIT.\n"
                b"`\n"
            ),
        }
    )
    assert _kept(root) == ["tmpl.go"]


def test_js_is_kept_when_its_typescript_sibling_is_too_large(
    make_repo, polyglot_registered, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest, "MAX_FILE_BYTES", 100)
    root = make_repo({"src/foo.ts": "// " + "x" * 200 + "\n", "src/foo.js": "exports.f = 1;\n"})
    scan = ingest.scan_source_files(root)
    assert [p.relative_to(root).as_posix() for p in scan.files] == ["src/foo.js"]
    assert scan.too_large == ["src/foo.ts"]
