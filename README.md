# RepoFinder

RepoFinder helps you find your way around a large open-source project before
you contribute to it. Give it a GitHub repository and it draws an architecture
map and answers "where is feature X implemented?" with `path:start-end`
citations into the code.

## Install

Requires Python 3.10+ and `git` on `PATH`.

```powershell
git clone <this repository> RepoFinder
cd RepoFinder
python -m venv .venv
.venv\Scripts\python -m pip install -e .
```

On macOS / Linux use `.venv/bin/python`. The install puts a `repofinder`
command in the virtual environment (`.venv\Scripts\repofinder.exe` on Windows);
activate the environment or call it by that path.

## LLM access: two options

`index`, `map` and `search` work offline. `ask`, `chat` and `summarize` need Claude,
which RepoFinder can reach in two ways. Pick one.

**A. Anthropic API key** (pay as you go). Create a key at
<https://console.anthropic.com> and make it visible to RepoFinder:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."          # PowerShell
set ANTHROPIC_API_KEY=sk-ant-...                # cmd.exe
export ANTHROPIC_API_KEY=sk-ant-...             # macOS / Linux
```

or put the line `ANTHROPIC_API_KEY=sk-ant-...` in a `.env` file in the
directory you run RepoFinder from.

**B. Your Claude subscription, through Claude Code.** Install Claude Code
(<https://claude.com/claude-code>), sign in once with `claude auth login` (or
start `claude` and type `/login`), and make sure the `claude` command is on your
`PATH`. Nothing else is needed: RepoFinder
runs `claude -p` for each model call, which uses your subscription's limits
instead of API billing.

**Automatic detection.** When an API key (or other Anthropic SDK credentials)
is found, option A is used; otherwise RepoFinder looks for the `claude`
command. To choose explicitly, set `REPOFINDER_BACKEND=anthropic` or
`REPOFINDER_BACKEND=claude-code`. `repofinder status <repo>` shows which one
would be used. If neither is available, the LLM commands print setup
instructions and exit with code 2.

Models default to `claude-sonnet-5-5` for per-file summaries and
`claude-opus-5-5` for answers and the overview; override them with
`REPOFINDER_MODEL_FAST` and `REPOFINDER_MODEL_SMART`.

### Decisions (optional, Jev)

Chat mode makes a few small decisions on every turn before Claude writes the
reply: what kind of message it is (a "where is" question, an explanation, a
follow-up, small talk), which search hit answers it, and whether the excerpts
already shown cover a follow-up. By default built-in English and Turkish rules
make them, offline and free; the rules never reorder search results.
TypeSafe's Jev model, which returns typed choices with probabilities rather
than text, can make them instead:

```powershell
.venv\Scripts\python -m pip install -e ".[jev]"     # installs typesafe-sdk
$env:OPENROUTER_API_KEY = "sk-or-..."               # through OpenRouter, or
$env:TYPESAFE_API_KEY = "..."                       # through TypeSafe's own API
$env:REPOFINDER_DECISIONS = "jev"                   # "rules" or "off" (the default) use the rules
```

The keys can also go in the `.env` file. When both are set, the OpenRouter key
is used (model `~typesafe/jev-latest`). Jev bills input only (about $0.042 per
million tokens; output is free), and each request is kept under its 32k-token
context. `REPOFINDER_DECISIONS=jev` without a key or without `typesafe-sdk` is
a configuration error that prints these setup steps.

## Quick start (Flask)

```powershell
repofinder index https://github.com/pallets/flask
repofinder map   https://github.com/pallets/flask --out flask-map.md
repofinder ask   https://github.com/pallets/flask "Where is the route decorator implemented?"
```

`index` makes a shallow clone and builds the index (under a second for Flask,
plus the clone).
Every other command indexes on first use too, so `ask` alone also works. A local
directory works anywhere a URL does, and is indexed in place without copying.

`ask` prints the answer, a **Sources** list and the time and tokens used:

```
The route decorator is implemented in Scaffold.route in
src/flask/sansio/scaffold.py:343-373.
...
Sources
  src/flask/sansio/scaffold.py:343-373
  src/flask/sansio/scaffold.py:461-463

7.6 s · 1 call · 4,936 input + 656 output tokens · 567 cache read
```

Questions can be asked in any language. The answer comes back in the language
of the question.

Optionally, run `repofinder summarize <repo>` to have Claude write a short
summary of every file and an overview of the repository. Summaries make the map
richer, help the search, and the overview gives `ask` some orientation. It makes
one model call per file (about 80 for Flask), so try `--limit 10` first.

## Commands

| Command | What it does |
| --- | --- |
| `repofinder index <source> [--refresh]` | Clone (for a URL) and index. `--refresh` rebuilds an existing index and keeps the summaries of files that still exist. |
| `repofinder map <source> [--out FILE] [--max-modules N]` | Architecture map in Markdown: entry points, core modules by fan-in, package layout, hot spots and where to start. |
| `repofinder ask <source> "question" [-k N] [--show-context] [--json]` | Answer with citations. `-k` sets how many search hits become context (default 8); `--show-context` lists the excerpts sent to Claude; `--json` prints `answer`, `citations`, `hits`, `chunks`, `model`, `backend`, `usage` and `elapsed_s`. |
| `repofinder chat <source> [-k N] [--decisions jev\|rules] [--json]` | A conversation about the code, one message per line from stdin (see [Chat mode](#chat-mode)). `-k` sets the search hits per search (default 8); `--decisions` picks the decision layer for this run. `/context` lists the excerpts of the last search, `/route` shows how the last message was routed, `/help` lists the commands, `/quit` or the end of input leaves. `--json` prints one JSON object per line for every turn (`route`, `answer`, `citations`, `reused_context`, `widened`, `usage`, `decisions`, ...). The `you>` prompt is shown only when you type in a terminal, so piped output stays clean. |
| `repofinder summarize <source> [--limit N] [--all]` | Per-file summaries, then the repository overview. Only files without a summary are done unless `--all`. Stops after 3 failures in a row (usually a usage limit); finished work is kept. |
| `repofinder status <source>` | Counts of files, symbols and imports, summaries, overview present, files changed since indexing, LLM backend, data directory. |
| `repofinder search <source> "query" [-k N]` | The raw search hits with their scores and why they matched. No LLM; useful when an answer misses something. |

Every command accepts `--data-dir PATH`. Indexes live in `~/.repofinder` by
default (or in `$REPOFINDER_HOME`): `repos/<name>/index.db`, plus
`repos/<name>/checkout/<name>/` for cloned repositories. Write the map with
`--out` rather than `>` in Windows PowerShell 5, whose redirection re-encodes
the output.

### Chat mode

`repofinder chat <repo>` holds a conversation about one repository: the
decision layer decides, Claude talks. Every message is routed first. A "where
is" question (*locate*) or a "how does it work" question (*explain*) runs a
search, puts the excerpts the decision layer judges most relevant first and,
when it judges none of them relevant, searches once more with three times as
many hits. A *follow-up* reuses the previous excerpts when the decision layer
says they cover it, and otherwise searches again for the previous question
together with the new message. Small talk (*chat*) and questions about
something the repository leaves to another library (*outside*) get a short
reply without a search. Claude (`REPOFINDER_MODEL_SMART`) writes every reply
with the same `path:start-end` citation rules as `ask` and sees the last six
messages of the conversation. After each reply a dim line shows the route,
whether the excerpts were reused or the search was widened, the time, the
tokens and the decision calls.

The decisions are made by the built-in rules unless Jev is set up, as described
in [Decisions (optional, Jev)](#decisions-optional-jev); with the rules, chat
makes no model calls besides Claude's reply.

## How it works

Tree-sitter parses every Python file into classes, functions, methods and
imports, which are stored in a SQLite database together with an import graph
(fan-in, fan-out, entry points). A question is answered by a lexical search
(BM25 over symbol names, signatures, docstrings, paths and file summaries) that
returns the best symbols and files. A symbol's name boost is scaled by how much
of the question its name covers, so a common word that happens to be a name
does not outrank the real answer; a method also gains when the question names
its class. Words are matched through a light English stem and common
identifier abbreviations ("python file" finds `from_pyfile`), and Turkish
questions are supported by dropping Turkish filler words and translating the
rest through a small lexicon. The selected symbols' exact source lines become
excerpts (whole functions and classes, long ones in windows of about 100 lines,
the class header for methods), about 36,000 characters in all. Claude answers
from those excerpts only and cites `path:start-end`; the Sources list keeps
only citations that fall inside the excerpts.

## Limitations

- Python only for now (other languages need a parser registration).
- Retrieval is lexical, not semantic: the question has to share words with
  the code (identifiers, docstrings, file names). If the answer looks wrong,
  `repofinder search` shows what was retrieved; rephrasing with the names you
  expect, or asking in English, usually helps.
- Turkish support is limited to a small built-in lexicon of common code words;
  Turkish written without Turkish letters ("nasil", "kaydediliyor" typed as
  ASCII) only partly works.
- The answer only knows the retrieved excerpts. It says so when they do not
  contain the answer, and suggests where to look next.
- Two repositories with the same name share a data directory; opening one
  replaces the other's index.

## Development

```powershell
.venv\Scripts\python -m pip install -e ".[dev]"
$env:REPOFINDER_SKIP_LIVE = "1"; .venv\Scripts\python -m pytest -q
```

`REPOFINDER_SKIP_LIVE=1` skips the few tests that call the real `claude` CLI.
Everything else runs offline with fake backends. `tests/fixtures/sample_repo/`
is test data, not a test package (one of its modules has a deliberate syntax
error).

The code is in `repofinder/`: `ingest` and `parser` (clone, walk, tree-sitter),
`store` (SQLite), `search` (BM25), `graph` and `mapper` (import graph, map),
`llm`, `prompts` and `backends` (Claude), `chunker` (hits to excerpts),
`pipeline` (the API the CLI and the evaluation call), `chat` and `decisions`
(conversation mode and its routing choices) and `cli`.

Notes for contributors:

- `Symbol.id` (`path::qualname`) is not unique: a property getter and setter,
  `@overload` stubs and conditional redefinitions share one. Key on
  `(id, start_line)`.
- `FileRecord.parse_ok=False` means tree-sitter reported a syntax error; the
  file's symbols may be incomplete.
- Read tree-sitter points by index (`node.start_point[0]`), not `.row` or
  `.column`: with py-tree-sitter 0.26.0 those corrupt memory for values above
  256 (details in `parser.py`).
- Cloned repositories are untrusted data. RepoFinder only reads them and never
  runs their code.

## License

MIT © Selman Boray Asiltürk (license file to be added).
