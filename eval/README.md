# RepoFinder evaluation

We do not say "it works" without a measurement. This folder is how we measure
RepoFinder's retrieval and answers: a hand-checked golden set of questions with
their exact answer locations, and a runner that scores the product against it.

| File | What it is |
| --- | --- |
| `golden/flask.json` | 30 questions about Flask at commit `d086db8`, every location checked by reading the file |
| `run_eval.py` | The runner: validates the golden set, calls the pipeline, scores, writes a report |
| `search_baseline.py` | The same scoring with `repofinder.search` alone (no pipeline, no LLM) |
| `results/` | JSON reports. `flask-search-baseline-d086db8-k8.json` is the search-only baseline |

The runner uses the standard library only. Tests: `tests/test_eval.py`.

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
```

Options: `--k` (hits/chunks per question, default 8), `--ids q01,q05`,
`--out PATH` (default `eval/results/<golden>-<mode>-<timestamp>.json`),
`--data-dir PATH` (passed to `open_workspace`), `--source` (default: `repo.source`
in the golden file, the GitHub URL).

Exit codes: **0 whatever the scores are** (this is a measurement, not a gate);
**2** if the golden file is invalid or an `--ids` entry does not exist; **1** if
the pipeline cannot be set up (`repofinder.pipeline` missing, `open_workspace`
raised). A question whose `retrieve`/`ask` call raises is recorded with its error,
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
| `extra` | `all@k` (flow: every expected location in the top k), `ptr@k` (not-in-repo: the pointer is in the top k), `all_req` (flow: every expected location cited), `admit` / `ptr_cited` (not-in-repo), `false-admit` (the answer said "not found" for an answerable question), `ERROR` |
| `top-1 hit` | `path:line qualname` of the first hit, `(file)` for a file-level hit |

Then the summary: overall rates (as `rate (yes/n)`, only over the questions where
the metric applies), MRR, the loose rates, token totals and timings, and a table
per category and per language.

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
- `admits_missing`: **a heuristic.** The answer text contains a phrase from
  `ADMIT_PHRASES` in `run_eval.py` (English and Turkish, e.g. "not in the
  excerpts", "do not contain", "alıntılarda yok", "görünmüyor"), compared
  case-insensitively with Turkish ı/İ folded to i. It is the target behaviour for
  not-in-repo questions; on answerable questions it is reported as a false
  admission. It can misfire both ways: read the answer texts and extend the list
  when a correct admission is missed.
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
