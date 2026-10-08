"""Command-line interface: ``repofinder index|map|ask|summarize|status|search``.

Results go to stdout; progress spinners and errors go to stderr. Exit codes:
0 success, 1 failure, 2 no usable LLM backend (the message says how to set one up).
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Iterator, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from . import __version__, pipeline
from .ingest import IngestError
from .llm import LLM, Citation, LLMConfigError, LLMError
from .store import StoreError

app = typer.Typer(
    name="repofinder",
    help="Map a repository's architecture and find where features are implemented, with file:line citations.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

SourceArg = Annotated[str, typer.Argument(help="GitHub URL (https://github.com/owner/name) or a local directory.")]
DataDirOpt = Annotated[
    Optional[Path],
    typer.Option("--data-dir", help="Where indexes are kept (default: $REPOFINDER_HOME or ~/.repofinder).",
                 file_okay=False),
]


def _out() -> Console:
    return Console(highlight=False, soft_wrap=True)


def _err() -> Console:
    return Console(stderr=True, highlight=False, soft_wrap=True)


def _make_llm() -> LLM:
    """The LLM used by ``ask`` and ``summarize`` (tests replace this)."""
    return LLM()


@contextlib.contextmanager
def _errors() -> Iterator[None]:
    """Turn expected failures into a message on stderr and an exit code."""
    try:
        yield
    except LLMConfigError as exc:  # includes BackendUnavailable
        _err().print(str(exc), markup=False)
        raise typer.Exit(2)
    except LLMError as exc:
        _err().print(f"LLM call failed: {exc}", markup=False)
        raise typer.Exit(1)
    except (IngestError, StoreError) as exc:
        _err().print(f"Error: {exc}", markup=False)
        raise typer.Exit(1)


def _open(source: str, data_dir: Optional[Path], refresh: bool = False, quiet: bool = False) -> pipeline.Workspace:
    err = _err()
    with err.status(f"Opening {source} (cloning and indexing on first use)…", spinner="dots"):
        ws = pipeline.open_workspace(source, data_dir=data_dir, refresh=refresh)
    if ws.built and not quiet:
        err.print(f"Indexed {ws.name}: {_index_stats(ws)}.", markup=False)
    return ws


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def _index_stats(ws: pipeline.Workspace) -> str:
    idx = ws.index
    failures = sum(1 for f in idx.files if not f.parse_ok)
    return " · ".join([_plural(len(idx.files), "file"), _plural(len(idx.symbols), "symbol"),
                       _plural(len(idx.imports), "import"), _plural(failures, "parse failure")])


def _backend(llm: LLM) -> str:
    """Detect the backend now, so a missing one fails before any work."""
    return llm.backend_name


def source_ranges(citations: list[Citation]) -> list[tuple[str, int, int]]:
    """Citations for the "Sources" list: ranges inside another cited range of
    the same file are left out; first-appearance order is kept."""
    ranges = [(c.path, c.start_line, c.end_line) for c in citations]
    kept = []
    for i, (path, start, end) in enumerate(ranges):
        inside = any(
            j != i and p == path and s <= start and end <= e and ((s, e) != (start, end) or j < i)
            for j, (p, s, e) in enumerate(ranges)
        )
        if not inside:
            kept.append((path, start, end))
    return kept


def _usage_line(llm: LLM, elapsed_s: Optional[float] = None) -> str:
    u = llm.usage
    parts = []
    if elapsed_s is not None:
        parts.append(f"{elapsed_s:.1f} s")
    parts.append(f"{u.calls} call{'s' if u.calls != 1 else ''}")
    parts.append(f"{u.input_tokens:,} input + {u.output_tokens:,} output tokens")
    parts.append(f"{u.cache_read_tokens:,} cache read")
    return " · ".join(parts)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def _version(value: bool) -> None:
    if value:
        typer.echo(f"repofinder {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True,
                                          help="Show the version and exit.")] = False,
) -> None:
    """RepoFinder: an architecture map and cited answers to "where is X implemented?"."""


@app.command()
def index(
    source: SourceArg,
    refresh: Annotated[bool, typer.Option("--refresh", help="Rebuild the index even if one exists.")] = False,
    data_dir: DataDirOpt = None,
) -> None:
    """Clone (for a URL) and index a repository."""
    with _errors():
        started = time.perf_counter()
        ws = _open(source, data_dir, refresh=refresh, quiet=True)
        with ws:
            verb = "Indexed" if ws.built else "Loaded the existing index of"
            out = _out()
            out.print(f"{verb} {ws.name} in {time.perf_counter() - started:.1f} s: {_index_stats(ws)}",
                      markup=False)
            out.print(f"Checkout: {ws.root}", markup=False)
            out.print(f"Data dir: {ws.data_dir}", markup=False)
            if not ws.built:
                out.print("Use --refresh to rebuild it.", markup=False)


@app.command("map")
def map_(
    source: SourceArg,
    out_file: Annotated[Optional[Path], typer.Option("--out", help="Write the Markdown map to this file.",
                                                     dir_okay=False)] = None,
    max_modules: Annotated[int, typer.Option("--max-modules", min=1, help="Rows in the core-modules table.")] = 40,
    data_dir: DataDirOpt = None,
) -> None:
    """Print the architecture map (Markdown)."""
    with _errors():
        with _open(source, data_dir) as ws:
            text = pipeline.architecture_map(ws, max_modules=max_modules)
        if out_file is not None:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(text, encoding="utf-8")
            _err().print(f"Wrote {out_file}", markup=False)
        else:
            typer.echo(text, nl=False)


@app.command()
def ask(
    source: SourceArg,
    question: Annotated[str, typer.Argument(help="Your question, in any language.")],
    k: Annotated[int, typer.Option("-k", "--top-k", min=1, help="Search hits turned into context.")] = pipeline.DEFAULT_K,
    show_context: Annotated[bool, typer.Option("--show-context", help="List the code excerpts sent to the model.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print one JSON object instead of formatted text.")] = False,
    data_dir: DataDirOpt = None,
) -> None:
    """Answer a question about the code, citing path:start-end."""
    if not question.strip():
        _err().print("The question is empty.")
        raise typer.Exit(1)
    with _errors():
        llm = _make_llm()
        backend = _backend(llm)
        with _open(source, data_dir, quiet=as_json) as ws:
            if as_json:
                result = pipeline.ask(ws, question, llm=llm, k=k)
            else:
                out = _out()
                out.print(f"Backend: {backend} · model {llm.smart_model}", markup=False, style="dim")
                with _err().status("Searching and asking Claude…", spinner="dots"):
                    result = pipeline.ask(ws, question, llm=llm, k=k)
            repo = ws.name

    if as_json:
        payload = {
            "question": question,
            "repo": repo,
            "answer": result.answer.text,
            "citations": [asdict(c) for c in result.answer.citations],
            "hits": [asdict(h) for h in result.hits],
            "chunks": [{"path": c.path, "start_line": c.start_line, "end_line": c.end_line, "label": c.label,
                        "chars": len(c.text)} for c in result.chunks],
            "model": result.answer.model,
            "backend": backend,
            "usage": asdict(llm.usage),
            "elapsed_s": round(result.elapsed_s, 3),
        }
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if show_context:
        total = sum(len(c.text) for c in result.chunks)
        out.print(f"Context: {len(result.chunks)} excerpts, {total:,} characters", markup=False, style="bold")
        for c in result.chunks:
            n = c.end_line - c.start_line + 1
            out.print(f"  {c.path}:{c.start_line}-{c.end_line}  {c.label or ''}  ({n} lines)", markup=False)
        out.print()
    out.print(Markdown(result.answer.text), soft_wrap=False)
    out.print()
    out.print("Sources", style="bold")
    sources = source_ranges(result.answer.citations)
    if sources:
        for path, start, end in sources:
            out.print(f"  {path}:{start}" + (f"-{end}" if end != start else ""), markup=False)
    else:
        out.print("  (the answer cites no lines from the excerpts)", markup=False)
    out.print()
    out.print(_usage_line(llm, result.elapsed_s), markup=False, style="dim")


@app.command()
def summarize(
    source: SourceArg,
    limit: Annotated[Optional[int], typer.Option("--limit", min=0, help="Summarize at most N files in this run.")] = None,
    all_files: Annotated[bool, typer.Option("--all", help="Re-summarize files that already have a summary.")] = False,
    data_dir: DataDirOpt = None,
) -> None:
    """Write per-file summaries and the repository overview (uses the LLM)."""
    with _errors():
        llm = _make_llm()
        backend = _backend(llm)
        out = _out()
        with _open(source, data_dir) as ws:
            out.print(f"Backend: {backend} · models {llm.fast_model} (files), {llm.smart_model} (overview)",
                      markup=False, style="dim")
            err = _err()
            progress = Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
                                TimeElapsedColumn(), TextColumn("{task.fields[path]}"),
                                console=err, transient=True, disable=not err.is_terminal)
            with progress:
                task = progress.add_task("Summarizing", total=None, path="")

                def on_progress(done: int, total: int, path: str) -> None:
                    progress.update(task, completed=done, total=total, path=path)

                started = time.perf_counter()
                result = pipeline.summarize(ws, llm, limit=limit, only_missing=not all_files,
                                            on_progress=on_progress)
            elapsed = time.perf_counter() - started
            n_summaries = len(ws.summaries())
            n_files = len(ws.index.files)

        out.print(f"Summarized {_plural(result.done, 'file')} ({result.failed} failed, {result.skipped} skipped). "
                  f"{n_summaries} of {_plural(n_files, 'file')} have a summary; {result.remaining} still need one.",
                  markup=False)
        if result.stopped_early:
            out.print(f"Stopped after {pipeline.MAX_CONSECUTIVE_FAILURES} failures in a row "
                      f"(often a usage limit). Last error: {result.error}", markup=False, style="yellow")
        elif result.error:
            out.print(f"Last error: {result.error}", markup=False, style="yellow")
        if result.repo_summary_written:
            out.print("Repository overview written.", markup=False)
        out.print(_usage_line(llm, elapsed), markup=False, style="dim")
        if result.stopped_early:
            raise typer.Exit(1)


@app.command()
def status(source: SourceArg, data_dir: DataDirOpt = None) -> None:
    """Show what is indexed and summarized, and which LLM backend would be used."""
    with _errors():
        with _open(source, data_dir) as ws:
            n_files = len(ws.index.files)
            stats = _index_stats(ws)
            usable = len(ws.summaries())
            failed = len(ws.failed_summaries())
            overview = ws.repo_summary() is not None
            built_at = ws.store.get_meta(pipeline.META_BUILT_AT) or "unknown"
            stale = pipeline.stale_files(ws)
            name, src, root, data = ws.name, ws.source, ws.root, ws.data_dir
        try:
            backend = LLM().backend_name
        except LLMConfigError as exc:
            backend = f"none ({str(exc).splitlines()[0]})"

    rows = [
        ("Repository", name),
        ("Source", src),
        ("Checkout", str(root)),
        ("Data dir", str(data)),
        ("Index", f"{stats} · built {built_at}"),
        ("Changed", f"{_plural(len(stale), 'file')} changed on disk since indexing"
                    + (" (run `repofinder index --refresh`)" if stale else "")),
        ("Summaries", f"{usable} of {_plural(n_files, 'file')}" + (f" ({failed} failed)" if failed else "")),
        ("Overview", "yes" if overview else "no (run `repofinder summarize`)"),
        ("Backend", backend),
    ]
    out = _out()
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        out.print(f"{label:<{width}}  {value}", markup=False)


@app.command()
def search(
    source: SourceArg,
    query: Annotated[str, typer.Argument(help="Search words or a question.")],
    k: Annotated[int, typer.Option("-k", "--top-k", min=1, help="Number of hits.")] = 10,
    data_dir: DataDirOpt = None,
) -> None:
    """Show the raw search hits for a query (no LLM; for debugging retrieval)."""
    with _errors():
        with _open(source, data_dir) as ws:
            hits = ws.search_index().search(query, k=k)
    out = _out()
    if not hits:
        out.print("No hits.", markup=False)
        return
    width = len(str(len(hits)))
    for rank, hit in enumerate(hits, start=1):
        name = hit.symbol_id.split("::", 1)[-1] if hit.symbol_id else "(file)"
        out.print(f"{rank:>{width}}. {hit.score:7.2f}  {hit.kind:<6}  {hit.path}:{hit.start_line}-{hit.end_line}"
                  f"  {name}", markup=False)
        out.print(f"{'':>{width}}  {'':7}  {hit.why}", markup=False, style="dim")


def main() -> None:
    """Console-script entry point."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
