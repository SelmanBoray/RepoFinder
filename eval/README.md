# RepoFinder evaluation

We do not say "it works" without a measurement. This folder is how we measure
RepoFinder's retrieval and answers: a hand-checked golden set of questions with
their exact answer locations, and a runner that scores the product against it.

| File | What it is |
| --- | --- |
| `golden/flask.json` | 30 questions about Flask at commit `d086db8`, every location checked by reading the file |
| `golden/httpx.json` | 20 questions about httpx at commit `b5addb6`, written blind (first held-out set), every location checked with `ast` |
| `golden/rea.json` | 20 questions about the TypeScript repository morluto/rea at commit `fe2a398`, written blind to `search.py` (first TypeScript set, the tuning set for search v4); 104 locations verified with tree-sitter and the TypeScript compiler |
| `golden/chat_routes.json` | 24 conversations, 73 user turns, each turn labelled with its route (locate, followup, explain, chat, outside); written blind to the routing code |
| `golden/chat_routes_rea.json` | Chat routing conversations over the REA repository, same format and route labels as `golden/chat_routes.json` |
| `run_eval.py` | The runner: validates the golden set, calls the pipeline, scores, writes a report |
| `route_eval.py` | Routing runner: feeds each turn the previous turns as history, calls `decisions.route_message` (`--decisions jev\|rules`), scores the route per turn |
| `search_baseline.py` | The same scoring with `repofinder.search` alone (no pipeline, no LLM) |
| `rank_analysis.py` | Error analysis for search: the rank of every expected location in the top 100 with the competitors above it, to classify why a question missed |
| `results/` | JSON reports. `flask-search-baseline-d086db8-k8.json` is the search-only baseline |

The runner uses the standard library plus `repofinder.admit` (the phrase
heuristic, itself standard library only); the pipeline and the decisions layer
are imported only when a run needs them. Tests: `tests/test_eval.py`.

## Running

From the repository root (PowerShell):

```powershell
# validate and print the questions
.venv\Scripts\python eval\run_eval.py eval\golden\flask.json --list

# validate against a local checkout too (paths, line ranges, symbols, commit)
.venv\Scripts\python eval\run_eval.py eval\golden\flask.json --source C:\path\to\flask --list

# retrieval only: no LLM, no cost (the default mode)
.venv\Scripts\python eval\run_eval.py eval\golden\flask.json --source C:\path\to\flask --mode retrieval

# answers: ONE LLM CALL PER QUESTION. Start with a few ids.
.venv\Scripts\python eval\run_eval.py eval\golden\flask.json --source C:\path\to\flask --mode both --ids q01,q13,q28

# search-only baseline (needs a local checkout)
.venv\Scripts\python -I eval\search_baseline.py eval\golden\flask.json --source C:\path\to\flask

# judge the answers of a stored report again (no pipeline, no LLM answer calls)
.venv\Scripts\python eval\run_eval.py --rejudge eval\results\flask-both-haiku55-searchv2-d086db8-k8.json --decisions jev --out rejudged.json
```

Options: `--k` (hits/chunks per question, default 8), `--ids q01,q05`,
`--out PATH` (default `eval/results/<golden>-<mode>-<timestamp>.json`),
`--data-dir PATH` (passed to `open_workspace`), `--source` (default: `repo.source`
in the golden file, the GitHub URL), `--decisions jev|rules` (the decision layer:
the pipeline's rerank and the admit judge, see `admits_missing` below),
`--candidates N` (hits shown to the decision layer).

`--rejudge REPORT` takes a stored answer- or both-mode report instead of a golden
file and runs no pipeline and no LLM answer call. It judges every stored
`answer_text` again with `--decisions` (without it: Jev when a key is set, else
the rules), rewrites the `admits_*` fields of each answer, recomputes the summary
and writes a new report to `--out` (default: next to `REPORT`, named
`<report>-rejudged-<judge>-<timestamp>.json`; `REPORT` itself is never
overwritten). `meta` gains `rejudged_from`, `judge` and `rejudged_at`. It prints
the usual table and a comparison per question: id, category, the phrase
heuristic's verdict and phrase, the judge's probability and verdict, `DIFF` where
the two disagree, and the first 90 characters of the answer. With
`--decisions rules` the stored heuristic verdicts come back unchanged. A golden
file, `--list`, `--source`, `--mode`, `--ids`, `--data-dir` and `--candidates`
cannot be combined with it.

Exit codes: **0 whatever the scores are** (this is a measurement, not a gate);
**2** if the golden file is invalid or an `--ids` entry does not exist, or the
`--rejudge` report cannot be read or has no answers; **1** if
the pipeline cannot be set up (`repofinder.pipeline` missing, `open_workspace`
raised), or if the admit judge fails for good during `--rejudge` (no key, a
rejected key, no credit). A question whose `retrieve`/`ask` call raises is recorded with its error,
scored as a miss, and the run continues.

The runner calls the product only through this contract:

```python
from repofinder.pipeline import open_workspace, retrieve, ask
ws = open_workspace(source, refresh=False)          # data_dir=... only when --data-dir is given
chunks, hits = retrieve(ws, question, k=k)           # retrieval mode
result = ask(ws, question, k=k)                      # answer and both modes
# result.answer.text / .citations / .model / .input_tokens / .output_tokens / .cache_read_tokens
# result.hits, result.chunks, result.elapsed_s
```

In `both` mode only `ask` is called and retrieval is scored on `result.hits` and
`result.chunks`, so each question costs exactly one LLM call. `k` is always passed
by keyword (`ask`'s third positional parameter is `llm`).

**Use a local checkout at the golden commit.** With the GitHub URL as source the
pipeline clones the current default branch, whose line numbers differ from the
golden set's. When the workspace has a `root`, the runner reads its commit from
the `.git` files, stores it in `meta.checkout` and prints a warning on a mismatch.
The workspace is closed at the end when it has `close()`.

Console output uses the system code page. When you pipe it to a file or another
program and Turkish letters come out wrong, set `$env:PYTHONIOENCODING="utf-8"`.
The JSON report is always UTF-8.

## Reading the report

The Markdown table has one row per question:

| Column | Meaning |
| --- | --- |
| `hit@1`, `hit@3` | `1` if a strict match is the first hit / in the first three hits; `-` for not-in-repo |
| `chunk_recall` | `1` if the retrieved chunks cover an expected location (all of them when `accept_any` is false) |
| `cited_correct` | `1` if the answer cites an expected location (answer modes only) |
| `cited_wrong` | citations that hit no expected, pointer or related location |
| `extra` | `all@k` (flow: every expected location in the top k), `ptr@k` (not-in-repo: the pointer is in the top k), `all_req` (flow: every expected location cited), `admit` / `ptr_cited` (not-in-repo), `false-admit` (the answer said "not found" for an answerable question; `admit` and `false-admit` are the `admits_missing` verdict, from the Jev judge when it is set up, else from the phrase heuristic), `ERROR` |
| `top-1 hit` | `path:line qualname` of the first hit, `(file)` for a file-level hit |

Then the summary: overall rates (as `rate (yes/n)`, only over the questions where
the metric applies), MRR, the loose rates, token totals and timings, and a table
per category and per language. The not-in-repo line names who gave the
`admits_missing` verdicts, e.g. `admits_missing 0.67 (2/3) (judge: jev)` or
`(judge: rules (phrase heuristic))` (`judge: jev 28, rules 2` when some judge
calls failed and fell back to the heuristic; reports written before the judge
say `(phrase heuristic)`), and the next line counts the answers where the judge
and the phrase heuristic disagree: `admit judge vs phrase heuristic: N/M differ`.
With a decision layer the judge calls are part of the `decisions:` line
(`N calls (M admit judge)`).

The JSON report holds everything: `meta` (source, mode, k, constants, validation
warnings), `summary`, and per question the top hits with their match kind and the
search `why`, the chunk ranges and coverage, the citations with a verdict each,
and the **full answer text**. To understand a failure, open the JSON, find the
question and read `top_hits`, `chunks` and `answer_text` before changing anything.

### How a retrieved range matches an expected location

A hit (or citation) matches a golden location only on the **same path**. Then:

| Match | When | Counts as |
| --- | --- | --- |
| `symbol` | the hit's qualname (after `::` in `symbol_id`) equals the location's `symbol` | strict |
| `overlap` | the line ranges overlap | strict |
| `broad` | it overlaps but contains the location and is much larger (> 4x its span and > 60 lines), e.g. the whole `Flask` class for `Flask.run` | loose only |
| `file` | a file-level hit on the right file | loose only |

The headline numbers (`hit@1`, `hit@3`, `hit@k`, MRR, `cited_correct`) are strict.
The `loose_*` numbers also count `broad` and `file`: they say "the right
neighbourhood". A big gap between the two means the search finds the right file
or class but not the right function. The thresholds are `BROAD_FACTOR` and
`BROAD_MIN_SPAN` in `run_eval.py`.

### Metric definitions

Retrieval (no LLM):

- `hit@n`: a strict match among the first `min(n, k)` hits. MRR: mean of 1/rank of
  the first strict match (0 when none in the top k), over questions that have
  expected locations.
- `chunk_recall`: the union of the chunks' line ranges on a location's path covers
  at least half of its lines (`CHUNK_COVER_MIN`). With `accept_any: false`, every
  expected location must be covered. Per-location coverage is in
  `chunk_coverage`.
- `all@k` (flow): every expected location has a strict match in the top k.
- `pointer@k`, `pointer_chunk_recall` (not-in-repo): the same for `pointer`.

Answers (one LLM call per question):

- `cited_correct`: some citation strictly matches an expected location.
- `cited_wrong`: citations that overlap no expected, pointer or related location.
  `Answer.citations` only keeps citations inside the retrieved chunks
  (`llm.parse_citations`), so a citation to lines the model never saw is dropped
  before scoring and cannot be counted here.
- `all_required` (`accept_any: false`): every expected location is cited.
- `admits_missing`: the answer says that what was asked is not in the code
  excerpts it was given, or not in the repository. It is the target behaviour for
  not-in-repo questions; on answerable questions it is reported as a false
  admission. **The verdict comes from the Jev judge when Jev is set up**
  (`--decisions jev`, or a key with the default detection):
  `repofinder.decisions.answer_admits_missing` asks Jev a yes/no question
  (`ADMIT_QUESTION`: YES only when the answer's main claim is that the code is
  missing, NO when it gives the location even if it remarks that some
  neighbouring detail is not in the excerpts), and the verdict is P(yes) >= 0.5.
  **Otherwise it comes from the phrase heuristic**: the answer text contains a
  phrase from `ADMIT_PHRASES` in `repofinder/admit.py` (English and Turkish,
  e.g. "not in the excerpts", "do not contain", "alıntılarda yok",
  "görünmüyor"), compared case-insensitively with Turkish ı/İ folded to i. The
  heuristic misfires both ways ("X is in scaffold.py; Y is not in the excerpts"
  counts as an admission): read the answer texts. A judge call that fails keeps
  the heuristic's verdict for that answer. Per answer the report stores
  `admits_missing`, `admits_judge` (`jev`, `rules`), `admits_probability`,
  `admits_calls`, `admits_cost_usd`, and always the heuristic's
  `admits_phrase` and `admits_heuristic` for comparison (plus
  `admits_judge_error` after a fallback).
- `pointer_cited` (not-in-repo): a citation strictly matches a `pointer` location.
- Tokens (`input_tokens`, `output_tokens`, `cache_read_tokens`) and `elapsed_s`
  per question; `pipeline_elapsed_s` is what `ask` reported.

## The golden set format

```json
{
  "repo": {"name": "flask", "source": "https://github.com/pallets/flask", "commit": "d086db8"},
  "questions": [
    {
      "id": "q01",
      "question": "Where is the route decorator implemented?",
      "lang": "en",
      "category": "locate-symbol",
      "expected": [
        {"path": "src/flask/sansio/scaffold.py", "symbol": "Scaffold.route", "start_line": 343, "end_line": 373}
      ],
      "accept_any": true,
      "related": [
        {"path": "src/flask/sansio/app.py", "symbol": "App.add_url_rule", "start_line": 605, "end_line": 662}
      ],
      "notes": "why this is the answer; what a wrong-but-plausible answer looks like"
    }
  ]
}
```

- `lang`: `en` or `tr`.
- `category`: `locate-symbol` (the question names or describes one symbol),
  `locate-feature` (no identifier: "how does Flask sign the session cookie"),
  `flow` (a chain of two or more steps), `config` (allowed, unused so far),
  `not-in-repo` (the code lives in a dependency).
- `expected`: repo-relative POSIX `path`, 1-based inclusive `start_line` and
  `end_line`, optional `symbol` (the parser's qualname, `Class.method`). For a
  symbol use the parser's range, which starts at the first decorator. For
  something that is not a symbol (a class attribute, a few lines inside a method)
  give the exact lines and no `symbol`.
- `accept_any`: `true` when any one expected location is a full answer; `false`
  when all of them are needed (flow questions).
- `pointer` (not-in-repo only, required there): the delegating call sites in this
  repo. `expected` must be `[]`. The right answer says the code is not in the
  excerpts and cites the pointer.
- `related` (optional): locations that are fine to cite but are not the answer,
  for example the function a flow passes through, or the wrapper next to the real
  implementation. They are never required and never counted as wrong.
- `notes`: why this is the answer, and the wrong-but-plausible answers.

### Adding a question

1. Write the question the way a newcomer would ask it, before you look at what
   search returns. **Do not tune questions or expected locations to make the
   search look good**; a question the system fails is a useful one.
2. Find the answer by **opening the file and reading the lines**, at the golden
   commit. Never write locations from memory of the project.
3. Prefer one precise location. List several with `accept_any: true` only when
   each one really answers the question; put helpful-but-not-the-answer places in
   `related`.
4. In `notes`, say why, and name the wrong answer a plausible system would give.
5. Validate against a local checkout and fix every error and warning:

   ```powershell
   .venv\Scripts\python eval\run_eval.py eval\golden\flask.json --source C:\path\to\flask --list
   ```

   This checks that ids are unique, categories and languages are valid, every
   path exists, every range fits its file, each `symbol` is defined by a
   `def`/`class` inside its range and the range starts at it or its decorators,
   and that the checkout is at `repo.commit` (otherwise the line numbers may have
   drifted). `tests/test_eval.py` runs the same check on the Flask checkout.
6. Keep the mix balanced: the test suite pins the Flask set at 30 questions (12
   locate-symbol, 10 locate-feature, 5 flow, 3 not-in-repo; 20 en, 10 tr). Update
   that test deliberately when the set grows.

To move a golden set to a new commit, re-check every location by hand, then
update `repo.commit`. Scores from different commits are not comparable.

## Baseline: search layer only (Flask d086db8, k=8)

`search_baseline.py` on 2026-10-08, before the pipeline and the LLM exist.
Report: `results/flask-search-baseline-d086db8-k8.json`.

| group | hit@1 | hit@3 | hit@8 | MRR |
| --- | --- | --- | --- | --- |
| all answerable (27) | 0.37 | 0.48 | 0.67 | 0.451 |
| locate-symbol (12) | 0.75 | 0.83 | 0.92 | 0.812 |
| locate-feature (10) | 0.10 | 0.30 | 0.50 | 0.206 |
| flow (5) | 0.00 | 0.00 | 0.40 | 0.073 |
| en (18) / tr (9) | 0.33 / 0.44 | 0.50 / 0.44 | 0.72 / 0.56 | 0.443 / 0.467 |

Loose (counting whole-file and whole-class hits): hit@1 0.44, hit@3 0.59, hit@8
0.81. Flow `all@8`: 1/5. Not-in-repo pointer in the top 8: 2/3. In the baseline a
"chunk" is just the hit's own range (whole classes included), so its
`chunk_recall` (0.74) says little about the real chunker.

## Search v2 (2026-10-09)

`search_baseline.py` after the search ranking rework in `repofinder/search.py`
(coverage-scaled name boost, owner-class match, light stemming, abbreviations,
Turkish filler and lexicon). Report: `results/flask-search-v2-d086db8-k8.json`.

| group | hit@1 | hit@3 | hit@8 | MRR |
| --- | --- | --- | --- | --- |
| all answerable (27) | 0.74 | 0.85 | 0.89 | 0.789 |
| locate-symbol (12) | 1.00 | 1.00 | 1.00 | 1.000 |
| locate-feature (10) | 0.60 | 0.70 | 0.80 | 0.648 |
| flow (5) | 0.40 | 0.80 | 0.80 | 0.567 |
| en (18) / tr (9) | 0.78 / 0.67 | 0.83 / 0.89 | 0.89 / 0.89 | 0.813 / 0.741 |

Loose: hit@1 0.78, hit@3 0.89, hit@8 0.93. Flow `all@8`: 1/5 (unchanged).
Not-in-repo pointer in the top 8: 2/3 (unchanged). `chunk_recall` 0.81 (from
0.74). Two secondary numbers went down: pointer `chunk_recall` 1.00 -> 0.67 (q28)
and flow `chunk_recall` 0.40 -> 0.20 (q25). Both come from the whole `Flask` class
(about 1,500 lines) no longer reaching the top 8: in this report its one
pseudo-chunk covered every location inside the class. The ranking parameters
were tuned on these same 30 questions and there is no separate validation set
yet, so these numbers are optimistic for unseen questions and repositories.
An independently written second set (`golden/requests.json`, psf/requests @
611c6162) then showed that v2 did not generalize: on requests it fell below v1
(hit@3 0.65 -> 0.53, hit@8 0.88 -> 0.65, MRR 0.608 -> 0.520); see Search v3.

## Search v3 (2026-10-09)

`search_baseline.py` after four mechanism fixes in `repofinder/search.py`, with
Flask as the tuning set and requests as the validation set. Reports:
`results/flask-search-v3-d086db8-k8.json`, `results/requests-search-v3-611c6162-k8.json`.

* **A. Derived tokens are capped at the IDF of their word.** A stem or
  abbreviation never weighs more than the word the user wrote, in BM25 and in
  the unit weight. In requests "request" is in almost every document (IDF 0.24)
  and its abbreviation "req" in four (IDF 4.87), so `MockRequest.origin_req_host`
  had climbed into the top 8 of seven questions.
* **B. Explicit identifiers.** A name written as code (backticked, dotted
  `Session.request` / `requests.get`, snake_case, camelCase; case-sensitive)
  gets the full name boost when it is the only identifier in the query. The
  qualifier of a dotted name ("Session") is not a name match by itself, and
  `requests.get` also matches `get` in module `requests.api` (package
  re-export). With several identifiers (`request`, `session`, `g`) coverage
  decides as before.
* **C. No partial name boost in test files** (unless the query is about tests):
  long test names (`test_env_cert_bundles`) were undoing the x0.5 test penalty.
* **D. Turkish.** (a) words and lexicon keys are compared ASCII-folded, so
  "govdesi", "istegi", "basligina" are translated; (b) ambiguous mappings
  dropped (`bağlantı` -> link, `işle` -> handle/process); (c) "zaman aşımı" ->
  timeout, "yeniden/tekrar dene" -> retry, "birleştir" -> merge.

| set | version | hit@1 | hit@3 | hit@8 | MRR | pointer@8 |
| --- | --- | --- | --- | --- | --- | --- |
| Flask (27 + 3) | v1 | 0.37 | 0.48 | 0.67 | 0.451 | 2/3 |
| | v2 | 0.74 | 0.85 | 0.89 | 0.789 | 2/3 |
| | v3 | 0.67 | 0.85 | 0.89 | 0.752 | 2/3 |
| requests (17 + 3) | v1 | 0.47 | 0.65 | 0.88 | 0.608 | 2/3 |
| | v2 | 0.47 | 0.53 | 0.65 | 0.520 | 2/3 |
| | v3 | 0.59 | 0.65 | 0.82 | 0.642 | 3/3 |

Cumulative ablation (hit@1 / hit@3 / hit@8, MRR):

| step | Flask | requests |
| --- | --- | --- |
| v2 | 0.74 / 0.85 / 0.89, 0.789 | 0.47 / 0.53 / 0.65, 0.520 |
| + A | 0.67 / 0.85 / 0.89, 0.752 | 0.47 / 0.53 / 0.77, 0.542 |
| + B | 0.67 / 0.85 / 0.89, 0.752 | 0.59 / 0.65 / 0.82, 0.648 |
| + C | 0.67 / 0.85 / 0.89, 0.752 | 0.59 / 0.65 / 0.82, 0.649 |
| + D (= v3) | 0.67 / 0.85 / 0.89, 0.752 | 0.59 / 0.65 / 0.82, 0.642 |

Flask lost two top-1 hits to A (q16, q18: near ties that v2 won through the
inflated "req" and "env" weights; both now rank 2). On requests B fixes q01
`Session.request` (3 -> 1) and q02 `requests.get` (13 -> 1); A brings q05, q09
and q12 back towards v1; C cuts test-file hits in the top 8 over the 20
questions (28 -> 16; v2 had 25, v3 has 11); D finds the not-in-repo pointer of
q20 (retry) and moves q17 from nowhere to 19, but sends q14 from 4 to 8 (the
ASCII words now translate, and "yükle" -> load, "gövde" -> body pull in other
code). Against v1, requests hit@8 is still one question short (0.82 vs 0.88:
q13 is at 10, v1 had 6).

Honesty notes. The four mechanisms come from the error analysis of the
requests set, and the D(c) phrases were added for requests questions (q13,
q17, q20), so requests is no longer a clean held-out set; a third golden set is
needed for the next round. No parameter was tuned on requests. Two choices were
made on Flask: B's full boost only for a single identifier (a full boost for
every identifier sends Flask q20 from 1 to 3, a share of 1/n of it to 2;
requests has no question with several identifiers), and rejecting a "best form, not sum" BM25 variant for a
word's synonyms (Flask hit@3 0.85 -> 0.81; not run on requests).

## Jev rerank (2026-10-10)

`jev_rerank.py` gives the top 8 candidates of search v3 to TypeSafe's Jev
decision model (`~typesafe/jev-latest` through OpenRouter). One `choice`
question picks the best candidate and the list is re-sorted by the returned
probabilities. One call per question. Reports:
`results/flask-jev-rerank-searchv3-d086db8-k8.json`,
`results/requests-jev-rerank-searchv3-611c6162-k8.json`.

Strict, search v3 -> Jev:

| set | hit@1 | hit@3 | hit@8 | MRR |
| --- | --- | --- | --- | --- |
| Flask (27 answerable) | 0.67 -> 0.85 | 0.85 -> 0.89 | 0.89 -> 0.89 | 0.752 -> 0.870 |
| Flask locate-feature (10) | 0.40 -> 0.70 | 0.70 -> 0.80 | 0.80 -> 0.80 | 0.548 -> 0.750 |
| Flask flow (5) | 0.40 -> 0.80 | 0.80 -> 0.80 | 0.80 -> 0.80 | 0.567 -> 0.800 |
| requests (17 answerable) | 0.59 -> 0.82 | 0.65 -> 0.82 | 0.82 -> 0.82 | 0.642 -> 0.824 |
| requests locate-feature (6) | 0.33 -> 0.83 | 0.33 -> 0.83 | 0.83 -> 0.83 | 0.402 -> 0.833 |
| requests flow (3) | 0.33 -> 0.33 | 0.33 -> 0.33 | 0.33 -> 0.33 | 0.333 -> 0.333 |

hit@8 cannot change (the reranker never leaves the top 8), so the whole gain
is ordering inside the top 8. requests was never tuned for Jev, so the gain
carries over to a set it did not see.

Cost and speed: Flask, 30 questions, $0.0041 in total (mean $0.00014 per
question, ~3,150 input tokens; `usage.cost` came back 30/30); requests, 20
questions, $0.0025. Mean 0.5 s per question (max 0.9 s). 50 of 50 calls
succeeded.

Limit: questions whose answer is not in the top 8 at all (Flask q15, q21, q23;
requests q15, q17) did not change; those are the search's job.

Run (`OPENROUTER_API_KEY` set; `--dry-run` works without a key):

```
.venv\Scripts\python -I eval\jev_rerank.py eval\golden\flask.json --source PATH
```

## Jev in the pipeline (2026-10-12)

The Jev rerank is now part of `pipeline.rerank_retrieve` (used by `ask`, `chat`
and `run_eval.py`), so it is measured through the real pipeline instead of the
experiment script. Search fetches `max(k, candidates)` hits; the first
`candidates` (default 16, cap 16) are rendered as cards (`path:start-end`, kind,
qualname, first 40 lines) and sent to Jev in one call with a NONE option. The
chosen order is applied and the top k become the context; NONE keeps the search
order and sets `off_target`; a failed call keeps the search order and records
`decision_error`.

Strict, k=8, search v3, local checkouts at the golden commits:

| set | layer | hit@1 | hit@3 | hit@8 | MRR | cost / run | mean s / q |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Flask (27 scored of 30) | none (search v3) | 0.67 | 0.85 | 0.89 | 0.752 | 0 | 0.006 |
| Flask | Jev, 8 candidates | 0.89 | 0.89 | 0.89 | 0.889 | $0.0035 | 0.46 |
| Flask | Jev, 16 candidates | 0.89 | 0.89 | 0.89 | 0.889 | $0.0060 | 0.46 |
| requests (17 of 20) | none | 0.59 | 0.65 | 0.82 | 0.642 | 0 | 0.006 |
| requests | Jev, 8 candidates | 0.82 | 0.82 | 0.82 | 0.824 | $0.0022 | 0.47 |
| requests | Jev, 16 candidates | 0.88 | 0.88 | 0.88 | 0.882 | $0.0038 | 0.41 |

Reports: `results/flask-pipeline-jev-c8-searchv3-d086db8-k8.json`,
`results/flask-pipeline-jev-c16-searchv3-d086db8-k8.json`,
`results/requests-pipeline-jev-c8-searchv3-611c6162-k8.json`,
`results/requests-pipeline-jev-c16-searchv3-611c6162-k8.json`.

Every question Jev moved went to rank 1: Flask q16 2->1, q18 2->1, q19 7->1,
q22 3->1, q24 2->1, q26 3->1; requests q05 2->1, q09 7->1, q12 7->1, q14 8->1,
and with 16 candidates q13 10->1. No question got worse. hit@1 = hit@3 = hit@8
after the rerank: whenever the answer was among the candidates, Jev put it
first. NONE was chosen only for real misses (Flask q15; requests q13 with 8
candidates) and for questions whose answer is not in the repository (Flask q28,
q29; requests q20 with 8 candidates). The remaining misses (Flask q15, q21,
q23; requests q15, q17) are not in the top 16 at all, so they stay a search
problem. The pipeline cards did slightly better than the experiment script
cards: Flask hit@1 0.85 -> 0.89 with the same 8 candidates.

Why 16 is the default: the same hit@1 on Flask, one more question on requests
(q13), and a cost still under 0.02 cents per question.

Run: `run_eval.py ... --decisions jev|rules --candidates N`. Rows get a
`decision` block; the summary gets `decision_calls`, `decision_cost_usd`,
`reranked`, `off_target` and `decision_errors`; the markdown report gets a
`decisions:` line; the per-question log marks `[rerank]`, `[none]` or
`[dec-err]`.

## httpx test set (2026-10-12)

The third golden set: `golden/httpx.json`, encode/httpx at `b5addb6`, 20
questions (8 locate-symbol, 6 locate-feature, 3 flow, 3 not-in-repo; 12 en, 8
tr, of which 4 ASCII-only Turkish). It was written blind: the author did not
open `search.py`, `decisions.py`, `eval/README.md`, `eval/results` or
`tests/test_search.py`, only the httpx source. All 97 location entries were
verified with `ast` against the commit. This is the first clean held-out set:
Flask is the tuning set, requests is validation (semi-tainted, because the v3
phrase additions were made while looking at it).

Rule: no search parameter or lexicon change may be made from httpx errors. If
that ever happens it must be said here, and httpx is validation from then on.

Strict, k=8, local checkout at `b5addb6`:

| group | layer | hit@1 | hit@3 | hit@8 | MRR |
| --- | --- | --- | --- | --- | --- |
| all answerable (17) | search v3 | 0.29 | 0.35 | 0.53 | 0.345 |
| | Jev, 16 candidates | 0.65 | 0.77 | 0.77 | 0.706 |
| locate-symbol (8) | search v3 | 0.25 | 0.38 | 0.62 | |
| | Jev | 0.50 | 0.62 | 0.62 | |
| locate-feature (6) | search v3 | 0.33 | 0.33 | 0.50 | |
| | Jev | 0.83 | 0.83 | 0.83 | |
| flow (3) | search v3 | 0.33 | 0.33 | 0.33 | |
| | Jev | 0.67 | 1.00 | 1.00 | |
| en (hit@1) | search v3 -> Jev | 0.40 -> 0.60 | | | |
| tr (hit@1) | search v3 -> Jev | 0.14 -> 0.71 | | | |

Other numbers. Search v3: loose hit@8 0.82, chunk_recall 0.47, not-in-repo
pointer@8 0.33 (1/3). Jev: loose hit@1 0.77, loose hit@8 0.88, chunk_recall
0.77, pointer@8 unchanged 0.33. Jev made 20 calls for $0.003419 in total
(about 0.017 cents per question), reranked 16 of 20, chose NONE 4 times, had 0
errors, and took 0.48 s per question on average.

Reports: `results/httpx-search-v3-b5addb6-k8.json`,
`results/httpx-pipeline-jev-c16-searchv3-b5addb6-k8.json`.

Rank of the expected location, search v3 -> Jev (none = not in the top 8 for
search, not chosen for Jev):

| q | search -> Jev | q | search -> Jev |
| --- | --- | --- | --- |
| q01 | 1 -> 1 | q11 | 7 -> 1 |
| q02 | none -> none | q12 | 1 -> 1 |
| q03 | none -> none | q13 | none -> 1 |
| q04 | 7 -> 2 | q14 | 1 -> 1 |
| q05 | 1 -> 1 | q15 | none -> 1 |
| q06 | 3 -> 1 | q16 | none -> 1 |
| q07 | 4 -> 1 | q17 | 1 -> 2 |
| q08 | none -> none | q18-q20 | not in repo (pointer only: q18 at 1) |
| q09 | none -> 1 | | |
| q10 | none -> none | | |

Comparison across the three sets, strict hit@1 / hit@8, search v3 alone, then
with Jev (16 candidates):

| set | search v3 | + Jev c16 |
| --- | --- | --- |
| Flask | 0.67 / 0.89 | 0.89 / 0.89 |
| requests | 0.59 / 0.82 | 0.88 / 0.88 |
| httpx | 0.29 / 0.53 | 0.65 / 0.77 |

Reading. On a repository the search was never tuned on, the lexical layer is
much weaker than on Flask (hit@1 0.29 vs 0.67), and Turkish questions suffer
most (0.14). Jev recovers most of it, because the right answer is usually
within the first 16 hits (13 of 17), and it lifts Turkish to 0.71. The four
remaining misses are retrieval failures: the answer was not in the top 16 at
all.

- q02 `Response.raise_for_status` (`_models.py:794-829`): the search surfaces
  the exception classes.
- q03 `BaseClient._redirect_url` (`_client.py:517-544`): URL helpers win.
- q08 `Request._prepare` (`_models.py:441-460`): Host / Content-Length are not
  in the symbol name.
- q10 `Client._send_handling_redirects` (`_client.py:964-999`): the
  `event_hooks` property wins over the place that calls the hooks.

Honesty: one run each, no reruns. The golden set was written by a model agent
under the blindness rule above. The flow questions use `accept_any=true`,
unlike `requests.json` (`false`), so flow numbers across sets are not directly
comparable.

## Admit judge: Jev vs the phrase heuristic (2026-10-14)

What was measured: whether an answer "admits the answer is missing" (its main
claim is that what was asked is not in the excerpts / the repository), judged by
Jev's yes/no (`decisions.answer_admits_missing`, `ADMIT_QUESTION`, threshold
0.5) versus the phrase heuristic (`repofinder.admit.ADMIT_PHRASES`). No new LLM
answers were generated: the 30 stored Haiku 5.5 answers on Flask (2026-10-10)
were re-judged with `run_eval.py --rejudge
eval/results/flask-both-haiku55-searchv2-d086db8-k8.json --decisions jev`;
report `eval/results/flask-both-haiku55-searchv2-d086db8-k8-rejudged-jev.json`.

Hand labels (reading all 30 answers): the true admissions are q15 and q23
(answerable questions the model failed) and q28 and q29 (not-in-repo). q30 is
borderline (it says hashing is in Werkzeug but points to the `register` view
that calls it) and is accepted either way.

Phrase heuristic: flags 15/30. Ten are false positives (q01, q03, q13, q14,
q16, q17, q18, q21, q25, and q20 as a hedge): the answer gives the right
location and then remarks that a neighbouring detail is not in the excerpts.
0 misses. Reported "false admits on answerable questions": 12/27 (0.44).

Jev judge: flags 6/30. q15 0.75, q23 0.60, q28 0.75, q29 0.87 (all four true
admissions found), plus q14 0.50 (false positive, exactly at the threshold: the
answer gives `Flask.send_static_file` and notes that the route registration is
not shown) and q20 0.74 (ambiguous: it names `update_template_context`, then
says it cannot be sure from these excerpts). q30 0.44 -> no. Everything else is
<= 0.42, and most locate answers sit at 0.03-0.07; q25 0.42 and q26 0.37 are
the other hedged answers. Reported "false admits on answerable questions": 4/27
(0.15); not-in-repo admits_missing 2/3. Judge and heuristic differ on 9/30.
Cost: 30 calls, $0.001008 total (0.0034 cents per answer), 0 errors.

Reading: against the hand labels the heuristic is right on 20/30 and the judge
on 28/30 (29/30 if q20 counts as an admission). The judge's probabilities also
separate clean answers (0.03-0.07) from hedged ones (0.37-0.50), which the
heuristic cannot. The threshold stays at 0.5: raising it to clear q14 would be
tuning on this set. The earlier Haiku numbers (false-admit 12/27) should now be
read as 4/27 under the judge; the phrase-heuristic admit metric over-counted by
8.

Honesty: the hand labels were made by the same person who wrote
`ADMIT_QUESTION`'s criteria, after seeing the heuristic's output. One run; 30
answers from one model on one repository.

## Chat routing set (2026-10-14)

The chat router (`decisions.route_message`) picks one of five routes for each
user message: locate, followup, explain, chat, outside. `golden/chat_routes.json`
checks it; `route_eval.py GOLDEN [--decisions jev|rules] [--ids] [--out]` runs
it. Reports: `results/chat-routes-rules-v1.json`, `results/chat-routes-jev-v1.json`.

The set: 24 conversations (flask 12, requests 12), 73 user turns. Routes by
turn: locate 19, followup 21, explain 12, chat 11, outside 10. Languages by
turn: en 41, tr 15, tr-ascii 17 (6 conversations are ASCII-only Turkish). It was
written blind to `decisions.py`, `chat.py`, `prompts.py` and the tests. Each
conversation has an English `notes` field naming the hard case it carries: a
Turkish follow-up with no shared word after an English question, "it" / "bu" /
"onu" follow-ups, a fresh locate right after an answer, questions about
RepoFinder itself, "thanks, one more: where is X", outside questions phrased
like locate, explain/locate boundary pairs, Turkish greetings, a one-word
"neden?", long messages. The assistant replies in the set are short synthetic
lines that only serve as history. The runner feeds each turn the previous turns
as history.

| route | rules precision / recall | Jev precision / recall |
| --- | --- | --- |
| locate | 0.55 / 0.90 | 0.70 / 1.00 |
| explain | 0.64 / 0.75 | 0.91 / 0.83 |
| followup | 0.75 / 0.71 | 1.00 / 0.91 |
| chat | 0.88 / 0.64 | 1.00 / 1.00 |
| outside | - / 0.00 | 0.80 / 0.40 |

| | rules (offline) | Jev |
| --- | --- | --- |
| accuracy | 0.66 (48/73) | 0.86 (63/73) |
| en / tr / tr-ascii | 0.61 / 0.67 / 0.77 | 0.85 / 0.93 / 0.82 |

The rules never answer outside by design, hence outside recall 0.00. Their main
confusion cells (expected -> predicted): outside -> locate 5, explain 3,
followup 2; chat -> locate 4; followup -> locate 3, explain 2.

Jev: mean p(expected) 0.88 on correct turns, 0.21 on wrong turns. Cost: 73
calls, $0.001937 total (0.0027 cents per turn), 0.37 s per turn, 0 errors.

Jev's 10 errors:
- outside -> locate (six): c04 Werkzeug URL matching (0.72 locate); c11 flaskr
  password hashing (0.77); c15 TLS certificate verification (0.61) and
  connection pool (0.86); c19 requests retry loop (0.71); c24 ssl handshake
  (0.99).
- explain -> outside (one): c07, a long message about before_request.
- followup -> locate / explain (two): c02 "and after_request?" (0.64 locate);
  c13 "proxies parametresiyle ne yapiyo" (0.74 explain).
- explain -> locate (one): c24 custom CA bundle via env var (0.36 vs 0.32).

Reading: the route state (`decisions._route_state`) shows only the recent
conversation and the new message. It does not say which repository this is or
what it delegates, so Jev cannot know that Werkzeug, urllib3 or the ssl module
are outside Flask or requests, and it reads "where is X" literally. Adding the
repository name and a one-line description (the overview's first lines) to the
route state is the obvious next change; when it is tried it must be reported
as tuned on this set, since this is the only routing set. Tried below: it did not
move `outside` recall. The rules' weak spot
is different: no outside at all, and "where" questions about RepoFinder itself
go to locate.

Honesty: one run each. The set was written by a model agent and the labels were
not independently reviewed turn by turn (the orchestrator spot-checked five
conversations). Some labels are debatable by nature (c24 turn 3 is explain vs
followup).

### Repository identity in the route state (2026-10-14, tuned on this set)

What changed: `decisions.route_message(..., repo=RepoInfo(name, description))` puts a
`Repository: <name>` / `About it: <description, 400 chars>` block at the top of the
route state, one sentence was added to ROUTE_INSTRUCTIONS, and the `outside` option
text was rewritten to say "code that belongs to a dependency or another project (a
library the repository imports, the standard library, a service) rather than to this
repository". The wording changes apply in every mode, so `none` is not the old prompt;
the old prompt's result is the v1 run (0.86). The golden file now has a top-level
`repos` map with each repository's first README paragraph as description, and
`route_eval.py --repo-info none|name|full` selects the mode. The product does the same
through `Workspace.repo_info()`: the stored repo summary when present, else
`pipeline.readme_intro` of the checkout.

Results (Jev, 73 turns, one run each; reports
`eval/results/chat-routes-jev-v2-repo-{none,name,full}.json`):

| mode | accuracy | outside recall | outside precision |
|------|----------|----------------|-------------------|
| v1 (old prompt) | 0.86 | 0.40 (4/10) | 0.80 |
| none | 0.85 (62/73) | 0.40 (4/10) | 0.67 |
| name | 0.89 (65/73) | 0.40 (4/10) | 1.00 |
| full | 0.89 (65/73) | 0.40 (4/10) | 1.00 |

Per language in `full`: en 0.88, tr 0.93, tr-ascii 0.88. The three turns that changed
between modes are all expected `explain`: c07 turn 1 (v1 outside, none outside, full
explain), c20 turn 2 (v1 explain, none outside, full explain), c24 turn 3 (v1 locate,
none locate, full explain).

The `outside` turns in `full`: correct with high confidence when the message names the
external project (c08 Jinja compile 0.92, c08 pip 1.00, c12 stdlib json 1.00, c19
urllib3 Retry 0.99). Wrong when the message names the repository itself or no library:
c15 "where does requests verify the server certificate" p(locate) 0.86, c15 connection
pool 0.85, c19 "requests kendisi tekrar deniyor mu" 0.81, c24 ssl handshake 0.99, c04
Werkzeug URL matching 0.55 vs 0.41 outside (borderline; Flask's description says it
"began as a simple wrapper around Werkzeug", which may read as Werkzeug being inside),
c11 flaskr password hashing 0.54 vs 0.45.

Reading: the hypothesis (Jev lacked the repository identity) was wrong for `outside`
recall. Jev already knows Jinja, pip and urllib3 are external; the misses are questions
that attribute the behaviour to the repository itself, which only the search can
refute. In the product those go locate, search, Jev rerank picks NONE, widened search,
and the answer says what the excerpts show, so the cost of this error is one search,
not a wrong answer. The gain is elsewhere: no false `outside` left (the harmful
direction, since `outside` skips the search) and three hedged `explain` turns fixed,
+3 turns overall. `full` and `name` are tied; `full` is kept as the default because the
product has the description anyway and the token cost difference is about 50 tokens
per call.

Honesty: the wording and the block were changed after reading this set's errors, so
0.89 is a tuned number; a fresh routing set is needed to confirm it. One run per mode;
Jev's run-to-run variation on this set is unmeasured (v1 and none differ on two turns
with the same state content except wording).

## Search v4 (2026-10-16): TypeScript, body field, REA as the tuning set

RepoFinder now indexes TypeScript (commit `4baa850`). The first TypeScript set is
`golden/rea.json`: morluto/rea at `fe2a398`, 20 questions (8 locate-symbol, 6
locate-feature, 3 flow, 3 not-in-repo; 12 en, 8 tr, of which 3 ASCII-only Turkish).
It was written blind to `search.py`; all 104 locations were verified with tree-sitter
and the TypeScript compiler. Search v3 on it: hit@1 0.18, hit@3 0.18, hit@8 0.35, MRR
0.205 (`rea-search-v3ts-fe2a398-k8.json`); with Jev, 16 candidates: 0.35 / 0.41 / 0.41
(`rea-pipeline-jev-c16-searchv3ts-fe2a398-k8.json`), where 10 of 17 answers were
outside the top 16, so Jev could not help. Decision: REA became the tuning set for
TypeScript search. httpx stayed held out and was not run until v4 was frozen.

Error analysis (`rank_analysis.py`, rank of every expected location in the top 100
with the competitors above it):

| class | what | cases |
| --- | --- | --- |
| a | test-file pollution | several |
| b | interface/type declarations outranking the implementation | q06, q09, q14, q15 (3 interfaces/types in the top 5) |
| c | tokenization: lowerCamel query words (`ipcRenderer`) not split; `#private` and `$` names | q04 and others |
| d | Turkish mapping: kimlik bilgisi, surec, incele, aday, secenek | several |
| e | query concept only in function bodies, not in names, signatures or docs | 9 questions (priority/auto, Unity/Mono, cacheable, backup) |
| g | other: Turkish filler words indexed; stopword-only joins | few |

Class (f), granularity, had no case.

### Mechanisms, cumulative

Strict, k=8, each row on top of the previous (hit@1 / hit@3 / hit@8, MRR):

| step | REA | Flask | requests |
| --- | --- | --- | --- |
| v3 | .18 / .18 / .35, .205 | .67 / .85 / .89, .752 | .59 / .65 / .82, .642 |
| + TYPE_PENALTY 0.7 for interface/type kinds when the query is not about types | .18 / .18 / .41, .227 | same | same |
| + lowerCamel query words split into parts; `#`/`$`-prefixed names match the bare name (bug fixes) | same (q04 64 -> 13) | same | same |
| + Turkish filler words dropped even when indexed; lexicon +14 words, +3 phrases | .18 / .18 / .47, .242 | same | same |
| + no stopword-only joins (bug fix) | same | same | same |
| + BM25F body field (= v4) | **.41 / .47 / .71, .494** | **.74 / .85 / .93, .812** | **.59 / .765 / .88, .693** |

The body field: BODY_WEIGHT 1.0, per-field length normalization, bodies read from the
checkout (BODY_MAX_BYTES 2 MiB). The body contributes to BM25 only, not to the
vocabulary used for Turkish mapping and joins, and `why` gains a `body: ...` part.
v4 reports: `rea-search-v4-fe2a398-k8.json`, `flask-search-v4-d086db8-k8.json`,
`requests-search-v4-611c6162-k8.json`.

Weight choice: hit@8 is flat for BODY_WEIGHT 0.7 to 2.0 on all three sets. 1.5 raised
hit@3 slightly but dropped REA hit@16 from 0.76 to 0.71 (Jev sees 16), so the untuned
1.0 was kept. Leave-one-out from v4: without TYPE_PENALTY REA hit@16 0.76 -> 0.71;
without the lexicon REA hit@8 0.71 -> 0.59; without the body REA hit@8 0.47.

Rejected:

- Splitting UpperCamel query words (JavaScript -> java + script): REA q14 55 -> >100.
- Not lifting the test penalty on the single word "test": only q13 moved, 44 -> 24.
- The body as a single-field BM25 term, or weighted tf with shared length
  normalization: Flask hit@3 0.85 -> 0.81 at every weight, because long bodies dilute
  the name match of `create_app` and `make_response`.
- df counted on the name, signature and doc fields only: Flask hit@1 0.59-0.63.

Side regressions to watch: Flask not-in-repo pointer@8 2/3 -> 1/3 (q29 4 -> 9);
requests q09 7 -> 9; requests chunk_recall 0.88 -> 0.82 and loose hit@8 1.00 -> 0.88;
REA q06 4 -> 15 (functions whose bodies mention GHIDRA_INSTALL_DIR). SearchIndex build
on REA 0.74 s -> 4.8 s (once per workspace; Flask 0.32 s, requests 0.18 s).

The remaining REA misses q01, q06, q09, q10, q11 are vocabulary gaps ("cacheable" vs
"answerable from snapshot", "backup" vs "copy"): semantic search or LLM query
expansion territory, not a ranking fix.

### Held-out confirmation: httpx, run once after v4 was frozen

httpx search v3 .29 / .35 / .53, MRR .345 -> v4 **.35 / .59 / .77, MRR .475**
(`httpx-search-v4-b5addb6-k8.json`). Loose hit@8 0.77, chunk_recall 0.77, not-in-repo
pointer@8 0.33 -> 1.00. By language: en hit@1 0.40, tr 0.29 (tr hit@8 0.71). This is
the number that shows v4 generalizes; the REA, Flask and requests numbers are tuned or
semi-tuned.

### With Jev, 16 candidates, on v4

Strict hit@1 / hit@3 / hit@8:

| set | v4 + Jev | v3 + Jev | report | calls, cost |
| --- | --- | --- | --- | --- |
| Flask | .85 / .89 / .93 | .89 / .89 / .89 (hit@1 -0.04, hit@8 +0.04) | `flask-pipeline-jev-c16-searchv4-d086db8-k8.json` | 30, $0.0058 |
| requests | .94 / 1.00 / 1.00 | .88 / .88 / .88 | `requests-pipeline-jev-c16-searchv4-611c6162-k8.json` | 20, $0.0040 |
| httpx | .88 / .88 / .88, pointer@8 1.00 | .65 / .77 / .77 | `httpx-pipeline-jev-c16-searchv4-b5addb6-k8.json` | 20, $0.0035 |
| REA | .59 / .77 / .77 | .35 / .41 / .41 | `rea-pipeline-jev-c16-searchv4-fe2a398-k8.json` | 20, $0.0051 |

### Four-set summary (hit@1 / hit@8, strict, k=8)

| set | status | search v3 | search v4 | v3 + Jev | v4 + Jev |
| --- | --- | --- | --- | --- | --- |
| Flask | tuning | .67 / .89 | .74 / .93 | .89 / .89 | .85 / .93 |
| requests | semi-tainted validation (used for the weight choice) | .59 / .82 | .59 / .88 | .88 / .88 | .94 / 1.00 |
| httpx | held out, one run | .29 / .53 | .35 / .77 | .65 / .77 | .88 / .88 |
| REA | tuning | .18 / .35 | .41 / .71 | .35 / .41 | .59 / .77 |

Honesty: the lexicon words were chosen while looking at REA questions, so REA is
optimistic. The body weight was chosen looking at Flask and requests too, so requests
is no longer a clean set. httpx is the only clean set and was run once, after v4 was
frozen. One run each. The agent that implemented v4 was interrupted once by an API
timeout and resumed; the measurements were logged per step.
