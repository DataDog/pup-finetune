# AGENTS.md

Guide for contributors and coding agents working in this repository.

## What this is

`pup-finetune` curates a supervised fine-tuning dataset from your Datadog Agent Observability
traces, fine-tunes an open-weight model on it, evaluates the tuned model against the base model,
and deploys it. It ships as a `pup` extension, so users run it as `pup finetune ...`; the
standalone executable is `pup-finetune`. All Datadog reads and writes go through `pup` and its
OAuth login. The project is in early development: commands and file formats may change.
User-facing docs: `README.md` (install, quickstart), `docs/commands.md` (every command and
flag), `docs/data-format.md` (rows, exclusion codes, dataset files) and the agent skill in
`skills/datadog-finetune/SKILL.md`.

## Commands

```sh
uv sync --all-extras                 # create .venv and install the project, extras + dev tools
uv run pytest                        # tests (no network, no paid calls)
uv run ruff check --fix              # lint
uv run ruff format                   # format (check only: uv run ruff format --check)
uv run mypy src                      # type-check (strict)
uv run python scripts/check_headers.py  # license header on every tracked .py and shell script
uv run python scripts/check_3rdparty.py # LICENSE-3rdparty.csv covers every dependency
uv run python scripts/gen_command_docs.py  # regenerate docs/commands.md (and the exclusion
                                     # table in docs/data-format.md); a test fails when stale
uv run pre-commit install            # run the hooks above on every commit
uv run pup-finetune --help           # run the CLI from the checkout
```

The `hf` extra (`transformers`, `jinja2`, `huggingface-hub`) backs the shared renderer and each
provider SDK has its own extra (`fireworks`, `baseten`); CI installs every extra. CI
(`.github/workflows/ci.yml`) runs the lint, format, type, test, header and third-party license
checks on Ubuntu and macOS with Python 3.12 and 3.13, and smoke-tests the built wheel. When you
add a dependency, add its row to `LICENSE-3rdparty.csv`.

To develop against pup, install the checkout as a pup extension. `bin/pup-finetune` is a small
POSIX sh launcher that runs `uv run --project <checkout> pup-finetune` from any directory, and
pup names the extension `finetune` after the file:

```sh
pup extension install --local bin/pup-finetune --link
pup finetune --help
pup finetune doctor                  # pup version, Datadog login, and other readiness checks
```

The offline dev loop trains end to end on the `fake` provider, with no network or credentials:

```sh
export PUP_FINETUNE_DEV=1
uv run pup-finetune dataset import /tmp/ds --file tests/fixtures/rows/tool_use.jsonl
uv run pup-finetune train --provider fake --model fake-model --dataset /tmp/ds --run-dir /tmp/run
                                     # preview only (exit 2); its next_command adds --confirm
uv run pup-finetune train --provider fake --model fake-model --dataset /tmp/ds --run-dir /tmp/run --confirm
uv run pup-finetune status /tmp/run  # phase, losses, best epoch, worker, promoted model
uv run pup-finetune evaluate /tmp/run          # preview only (exit 2): cases and calls
uv run pup-finetune evaluate /tmp/run --confirm
                                     # replays the test split with base and tuned into
                                     # /tmp/run/eval; rerun to resume; summary.json at the end
uv run pup-finetune deploy /tmp/run  # preview only (exit 2): what is created, how to undeploy
uv run pup-finetune deploy /tmp/run --confirm    # receipt in /tmp/run/deployment.json
uv run pup-finetune undeploy /tmp/run --confirm  # deletes it; the model is kept
```

A complete evaluation is also published to Datadog as two Agent Observability experiments
(`pup-finetune-base-<run id>` and `pup-finetune-tuned-<run id>`) on the dataset `dataset push`
created, at the version it pushed: a root span per test conversation linked to its dataset
record and a child span per replayed action, with the metrics and judge verdicts. It needs the
dataset's push receipt (`push.json`): without one the preview and the result warn and the
evaluation is not published (the local results are complete either way); `--no-publish`
(`--eval-no-publish` for `train --evaluate`) skips it without a warning. Publishing resumes:
`experiments.json` and `published.json` in the evaluation directory record the experiments
created and the spans sent, so a rerun creates and sends nothing twice, and a failed publish
exits 1 with a `next_command` that publishes the rest (a Ctrl-C while publishing leaves a
`cancelled` publication that the rerun continues too). Spans already published keep the
results they were sent with: retrying an evaluation's errors updates only the local results,
so evaluate into a new `--out-dir` for experiments with every result. Spans not indexed within
two minutes (one budget for both experiments) are a warning. Tests serve it with `FakePup` and `FakeExperimentsApi`; nothing reaches Datadog.

`evaluate` on a fake run uses the offline fake judge (`--judge-model fake-judge`, dev mode
only), and the fake's tuned checkpoint calls the tools its training data called, so the
summary shows a positive `tuned_minus_base`. `train ... --evaluate --confirm` evaluates in the
training session right after promotion instead (foreground only; `--eval-out-dir`, which must
be new or empty and is checked before training, `--eval-max-cases-per-trajectory`,
`--eval-judge-model`, `--eval-concurrency`, `--eval-max-tokens`, `--eval-seed` and
`--eval-skip-calibration` configure it). The run's end is recorded before that evaluation
starts, so a crash during it keeps the trained, promoted run; a failed evaluation exits 1
with the trained run and the `evaluate` command to retry it. A cancelled or budget-capped
training is not evaluated in its session (the evaluation would spend the session's time past
the cap). A budget-capped run still ends `completed`, so `evaluate` evaluates it on request;
like a budget-capped run, a cancelled run is evaluated by `evaluate` on request (its best
epoch; one cancelled before any epoch has nothing to evaluate). Evaluating a real run is paid
and needs the judge's key (`FIREWORKS_API_KEY` for the default judge); the preview is free and allowed with
`PUP_READ_ONLY` set, evaluating is not. A changed `--max-tokens`, judge, seed or cap needs a
new `--out-dir`. An evaluation whose pairs ended in errors (a sampler or judge outage) reports
`complete_with_errors` with an error count and a `next_command` that retries them, and exits
1 when a target has no verdict at all; a cancelled one reports `partial`.

To build a dataset from your own Agent Observability traces (Datadog reads through pup, so
both work with `PUP_READ_ONLY` set):

```sh
uv run pup-finetune dataset preview --ml-app my-app --from now-7d   # in memory, writes nothing
uv run pup-finetune dataset pull /tmp/ds --ml-app my-app --from now-7d
                                     # the same flags; rerun to resume (pages cached in /tmp/ds/raw)
uv run pup-finetune dataset show /tmp/ds
uv run pup-finetune dataset sample /tmp/ds --n 20            # rows in brief, to review
uv run pup-finetune dataset label /tmp/ds --file labels.jsonl  # {id, keep, reason} per line
uv run pup-finetune dataset triage --print-template > rubric.md  # Task / Keep / Drop / Examples
uv run pup-finetune dataset triage /tmp/ds --rubric rubric.md --max-rows 20
                                     # preview only (exit 2): rows, judge calls, rubric hash;
                                     # add --confirm to have the judge council label them (paid)
uv run pup-finetune dataset push /tmp/ds --name my-dataset --project pup-finetune-dev
                                     # preview only (exit 2); add --confirm to push the kept rows
```

A relative `--from` or `--to` (`now`, or `now-<n><unit>` with unit `s`, `m`, `h`, `d` or `w`)
is resolved to epoch ms when the command starts. The first pull into a directory freezes that
window together with the expressions as written, and a re-run with the same expressions reads
the frozen window, so a resumed pull reads the same spans.

`dataset push` is a remote write: it is refused with `PUP_READ_ONLY` set. Push test data to
the `pup-finetune-dev` project, never to a shared one.

`dataset triage` (`--rubric FILE` and/or `--rule TEXT`, repeated) has each judge model
(`--judges m1,m2`; by default two Fireworks serverless models, with the Fireworks key) review
each whole row and vote keep or drop: a strict majority of keep votes keeps a row, a tie
drops it, a row missing a vote stays undecided. Votes go to `judgments.jsonl` as they arrive,
so rerunning the same command resumes and retries only failed calls; the rubric hash and
judges are frozen in `triage.json` once votes exist (`--reset-triage` discards them). Decided
rows are labeled `by: judge`; rows the agent or a person labeled are skipped. A label cannot be
withdrawn, so judge labels the current votes do not back (after a reset with `--max-rows`, or
on undecided rows) are reported as `stale_judge_labels`, and the triage stays `incomplete`
until a run over every row replaces them. It is paid, so
it needs `--confirm`, but it writes nothing to Datadog, so `PUP_READ_ONLY` does not block it.
Tests replace the judges' clients (`cli/triage.py`'s `open_clients`) with a fake.

Judges can run on Anthropic or OpenAI too: `--judge-model`, `--eval-judge-model` and each
`--judges` entry take `anthropic/<model>` (`ANTHROPIC_API_KEY`, or the Anthropic SDK's own
login), `openai/<model>` (`OPENAI_API_KEY`), or otherwise a Fireworks model id such as
`accounts/fireworks/models/<model>`, and a triage council can mix them. Anthropic
judges need the `anthropic` extra (`uv sync --extra anthropic`, or `--all-extras`):

```sh
uv run pup-finetune auth login anthropic       # or openai; hidden prompt, OS keychain
uv run pup-finetune auth status                # anthropic-sdk when the SDK signs in itself
uv run pup-finetune evaluate /tmp/run --judge-model anthropic/claude-opus-5-5
                                     # preview names the judge's provider and endpoint
```

Tests never call them: `tests/test_anthropic_client.py` drives the client with a fake SDK
client (and the real SDK over a mock `httpx2` transport), and the OpenAI request policy is
checked over `httpx.MockTransport`. `PUP_FINETUNE_LIVE=1` with `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY` set runs `tests/test_judges_live.py`, one paid judge call per provider.

To see the models a provider can fine-tune (both are read-only, so they work with
`PUP_READ_ONLY` set):

```sh
uv run pup-finetune models list --provider fireworks          # the profiles; needs no API key
uv run pup-finetune models list --provider fireworks --check  # plus a live preflight per model
```

`--check` asks Fireworks about each model (one GET each, nothing paid) and needs the API key:
`FIREWORKS_API_KEY`, or the OS keychain after `pup finetune auth login fireworks`.

Training on Fireworks is paid: it needs the API key and the Training SDK, which comes with the
`fireworks` extra (`uv sync --extra fireworks`, or `--all-extras`). Train only on synthetic
rows while developing:

```sh
uv run pup-finetune train --provider fireworks --model qwen3p8-27b --dataset /tmp/ds \
  --run-dir /tmp/fw-run --max-epochs 1   # preview only (exit 2); add --confirm to train
```

The best epoch is promoted to a Fireworks model (named after the run id; a new best found by
resuming a run that already promoted one gets the id with `-e<epoch>s<step>`) before the
session ends, and `result.json` records it as `promoted_model`. `deploy /tmp/fw-run`
previews an on-demand Fireworks deployment of it (paid per GPU-second while a replica runs,
until `undeploy`); `--confirm` creates it (first validated shape unless `--shape`, at most
one replica, scaling to zero when idle, hot loading off), waits until it is `READY`
(`--timeout-minutes`, default 30), smoke-tests one chat completion and prints the inference
model (`accounts/<account>/deployments/<id>`). Fireworks' own CLI refuses to create
deployments when an AI agent runs it; we use the API, so an agent must pass `--confirm` only
with the user's explicit authorization. Always `undeploy` what you deploy while developing.
`--provider-option max_seq_len=<n>` cuts the profile's context to `n` tokens: the
run records the cut profile, and the preview already leaves longer rows out (`exceeds_context`).
Training runs in the key's own account (Fireworks refuses a key that sees several accounts);
`FIREWORKS_ACCOUNT_ID` only overrides the account lookup, i.e. the account id pup-finetune
records.

Training on Baseten Loops is paid too: `--provider baseten --model qwen3p5-9b` (or
`glm-5p3-flash`) needs the `baseten` extra and a Baseten API key (`BASETEN_API_KEY`, the OS
keychain after `pup finetune auth login baseten`, or the default remote's key in
`~/.trussrc`, read only). Its preview asks Baseten, read only, whether the workspace can train
the model; `--confirm` starts a billed dedicated trainer, deactivated when the session ends.
Baseten has no promotion, so `result.json` records the best sampler checkpoint as
`deployable_checkpoint`; deploying it on Baseten arrives later. `--provider-option
max_seq_len=<n>` (at least 1024) cuts the profile's context as on Fireworks.

Add `--detach` to train in a background worker (output in `/tmp/run/worker.log`) and follow it
with `status /tmp/run --watch`. `--provider-option step_delay_s=0.5 --batch-size 1` makes the
fake slow enough to `cancel /tmp/run --confirm` (or press Ctrl-C in the foreground), and
`--provider-option fail_at_step=9` simulates a crash to `resume /tmp/run --confirm` from. Use a
new `--run-dir` for each `train`.

Environment variables for development and unusual setups:

- `PUP_FINETUNE_PUP` - the pup executable to run (a path, or a name on `PATH`) instead of `pup`.
- `PUP_FINETUNE_KEEP_DD_TOKEN=1` - pass `DD_ACCESS_TOKEN` through to the `pup` processes we
  start. By default it is removed, so each child `pup` loads and refreshes its own stored OAuth
  token instead of reusing the short-lived token pup handed us. Set it only if you
  authenticate pup through environment variables alone.
- `PUP_FINETUNE_DEV=1` - development mode: enables dev-only providers such as the offline
  `fake` provider (`--provider fake`).

The `datadog-finetune` agent skill (`skills/datadog-finetune/SKILL.md`, the one source, in the
layout the skills CLI discovers) ships in the wheel as `pup_finetune/_skills/` through
`[tool.hatch.build.targets.wheel.force-include]`; an editable install reads the repository's
copy. `skill show` prints it; `skill install` runs `npx --yes skills add <bundled dir> --skill
datadog-finetune` (passing on `-a`, `-g` and `-y`, its stdout sent to stderr, and its
environment without the Datadog secrets or any provider key variable, since npx downloads and
runs the skills CLI from npm) when `npx` is on `PATH` and `--dir` is not given, and otherwise
copies it to `.agents/skills/` and `.claude/skills/` under the project (or the home directory
with `-g`). Neither makes a Datadog or provider call, so both work with `PUP_READ_ONLY` set.
When you change a command the skill names, keep the skill in step:
`tests/test_doc_commands.py` checks every `pup finetune` command and flag in it (and in the
README) against the app. `SKILL.md` has no license header: its frontmatter must come first,
and `scripts/check_headers.py` checks only code.

Opt-in test suites: `PUP_FINETUNE_LIVE=1` runs `live` tests (real external services);
`PUP_FINETUNE_TOKENIZER_TESTS=1` runs `tokenizer` tests (downloads real tokenizers, needs the
`hf` extra): `PUP_FINETUNE_TOKENIZER_TESTS=1 uv run pytest -m tokenizer`.

## Layout

Source lives under `src/pup_finetune/`.

- `cli/` - the Typer app; one module per command group. `dataset.py` has `dataset import`,
  `preview`, `pull`, `show`, `sample`, `label` and `push`; `train.py` has `train`, `resume`,
  `status`, `cancel` and the hidden `_worker` a detached run executes (and `train
  --evaluate`); `models.py` has `models list` (with `--check`, a preflight per profile);
  `evaluate.py` has `evaluate`, which only parses its flags and emits what
  `evaluation/runner.py` returns; `deploy.py` has `deploy` and `undeploy` (over
  `deployment.py`); `triage.py` has `dataset triage` (over `curation/triage.py`; its
  `open_clients` builds one chat client per judge provider); `skill.py` has `skill show` and `skill
  install` (its `find_npx` and `run_npx` are what tests replace).
- `skill.py` - the bundled agent skill: `bundled_skills_dir` (the package's `_skills/`, else
  the repository's `skills/`), `npx_argv`, `npx_env` (the environment without
  `datadog/pup.py`'s `SECRET_ENV` and every registered provider key variable) and
  `install_files` (the fallback copy; a different copy is replaced only when asked).
- `deployment.py` - what `deploy` and `undeploy` run, provider-neutral: `deploy_target` (a
  finished run's promoted model, else its deployable checkpoint), `deployment_id_for` (the
  run id made safe, or `--deployment-id`), `execute_deploy` (writes `deployment.json` as
  `REQUESTED`, with the deployment's resource name and what the create sends (`request`),
  from the provider's `on_creating` callback right before the create, removed again if the
  provider refused; then
  from the provider's `on_created` callback, before the wait, on `READY` and with the smoke
  test; a failure once it may exist, a smoke test that raises included, is recorded with an
  undeploy hint; one deployment per run)
  and `execute_undeploy` (idempotent; records `deleted_at`; the model is kept; a `REQUESTED`
  receipt's deployment is read back with `Provider.owns_deployment` and deleted only when it
  is the one requested).
- `runtime.py` - non-secret settings from the environment (`PUP_OUTPUT`, `PUP_READ_ONLY`, ...).
- `errors.py` - expected failures and their exit codes; the `--confirm` and read-only guards.
- `output.py` - the one stdout result (through `pup format` under pup), progress on stderr.
- `storage.py` - atomic writes, `0600` files and `0700` directories, directory locks, frozen
  settings.
- `doctor.py` - the check registry behind `doctor`; modules add checks with `register_check`.
- `credentials.py` - provider API keys (never Datadog's): a `CredentialSpec` registry, resolution
  from the environment variable, then the OS keychain (`keyring`), as `SecretStr`; `store` refuses
  when there is no OS keychain rather than writing a file. `auth login/status/logout` use it.
- `redaction.py` - masks resolved secrets and `Bearer` tokens in error output and logs.
- `concurrency.py` - `DaemonPool`, worker threads for parallel remote calls that never hold
  up the process's exit, so Ctrl-C stops at once instead of waiting for a call in flight.
- `http_common.py` - what the `httpx` clients that send a provider key share: `bearer_key`
  (refuses an empty or malformed key and registers it for redaction), `CLIENT_SOURCE`
  (`pup-finetune/<version>`) and short, redacted failure text.
- `inference.py` - `ChatClient`, an OpenAI-compatible `POST {base_url}/chat/completions` client
  over `httpx` (no SDK): bounded retries with backoff and jitter on 408, 429, 5xx and connection
  errors (honoring a capped `Retry-After`), no retry on other 4xx, 401/403 as
  `ProviderAuthError` with a login hint, 429 after retries as `ProviderCapacityError`, the rest
  as `TransientError` or `ExternalError`; `ChatCompleter` is its protocol (with an optional
  `response_schema`). `RequestPolicy` is what an endpoint takes (`OPENAI_POLICY`:
  `max_completion_tokens`, no temperature, JSON mode). `JudgeConfig` is the judge's provider,
  model, endpoint, credential and sampling settings (`identity()` for resume checks, `spec`
  for the command line).
- `judges.py` - judges on any provider, no SDK import: `parse_judge` reads a spec by prefix
  (`anthropic/` or `openai/`; anything else is a Fireworks model id) with the
  provider's defaults (`PROVIDERS`), `check_available` refuses a missing extra up front,
  `open_client` and `open_clients` (one per provider) build the clients.
- `anthropic_client.py` - `AnthropicChatClient`, the Anthropic judge client over the official
  SDK (the `anthropic` extra), the only module that imports `anthropic` (and its `httpx2`):
  `system` from the leading system message, no sampling parameters, `output_config` format
  and effort, `fallbacks="default"` through the beta API on the models that have it, `stop_reason`
  mapped to `finish_reason`, typed SDK errors mapped to ours. Tests pass a fake SDK client.
- `judge_credentials.py` - registers the `anthropic` and `openai` `CredentialSpec`s (imported
  by the providers package); `anthropic_sdk_login` detects the Anthropic SDK's own login,
  reported as the source `anthropic-sdk`.
- `datadog/` - Datadog access via `pup`. `pup.py` is the client every Datadog call goes
  through: `pup --no-agent --output json ...`, a scrubbed environment, timeouts, and failures as
  `ExternalError`. `spans.py` exports LLM spans through the Export API
  (`POST v2/llm-obs/v1/spans/events/search` via `Pup.api`): `SpanQuery` (the filter, plus
  `settings()` to freeze), `iter_spans` and `iter_span_pages` (cursor pagination, retries of
  timed-out pages, an optional raw page cache that resumes without refetching, `ExportStats`),
  and tolerant `LLMObs*` models that skip and count spans they cannot read (`UnreadableSpan`).
  `datasets.py` has `LLMObsDatasets`: projects (`pup llm-obs projects list/create`), datasets
  (listed and created through `Pup.api` on `v2/llm-obs/v1/<project>/datasets`, since pup
  1.10.7 cannot read the response of `pup llm-obs datasets create`), `records-add --confirm`
  and `records-all`; lists are read page by page, a create whose response is lost is found
  again by name. `jsonapi.py` has the JSON:API helpers both clients share: the media type,
  `private_json_file` (`--file` bodies are `0600` in a private temporary directory, removed
  afterwards) and `object_id`. `experiments.py` publishes an evaluation: `LLMObsExperiments` (`pup llm-obs
  experiments create --file`, `list`, `events list`, and spans plus metrics through `Pup.api`
  on `v2/llm-obs/v1/experiments/<id>/events`) and `ExperimentsPublisher`, the evaluation's
  `Publisher`: one experiment per target on the pushed dataset version, a `replay_trajectory`
  root span per test row (`dataset_record_id` from `push.json`) and a `replay_assistant`
  child per case, metrics stamped at submission, at most 100 spans a request, uuid5 span ids,
  `experiments.json` and `published.json` so a rerun sends nothing twice (a batch whose
  response was lost is sent again with the same span ids), cancellation between requests,
  and a bounded read-back until the events are indexed.
- `data/` - training data. `schema.py` is the canonical row every provider consumes (OpenAI
  chat format plus its source) and the stable `RowErrorCode` reasons a row is rejected for;
  `jsonl.py` reads and writes rows and converts them to the provider form; `exclusions.py`
  records why a row was left out (`excluded.jsonl`), with `ExclusionCode`s for reasons the
  schema cannot see (conversion problems, dataset-level rules);
  `split.py` assigns deterministic, persisted train/validation/test splits per trace;
  `dataset_dir.py` is the dataset directory (`settings.json`, `rows.jsonl`, `excluded.jsonl`,
  `splits.json`, `summary.json`, `raw/`, a pull's page cache, and `push.json`, the receipt
  `dataset push` writes) behind `dataset import`, `dataset pull` and `dataset show`;
  `labels.py` has the keep/drop `Label` and the `labels.jsonl` format (append-only, last label
  per id wins, a reason required to drop; `DatasetDir.kept_rows` leaves dropped rows out of
  training and split exports).
  `convert.py` maps one LLM span to a canonical conversation (`span_conversation`: input then
  output messages, a plain `value` as one message, tool calls and results, tools, `SourceRef`;
  reasoning omitted or preserved), with conversion issues as stable codes; `trajectories.py` has
  `rows_from_spans`, which dedupes spans by id, groups them by trace, collapses conversations
  that are a prefix or duplicate of another (so sub-agents stay their own rows) and yields a
  validated row or an `Exclusion` per kept conversation. `checks.py` has the span-level checks
  that run before conversion, in order, first failure wins (`span_error`; errored calls are kept
  with `--include-errors`); there is no `truncated_content` check, as the Export API gives no
  explicit truncation signal. `exclusions.py` also explains every code (`explain`). `pull.py` is
  what `dataset preview` (in memory) and `dataset pull` (into a dataset, through its `raw/`
  cache, with the settings frozen first) run, a relative time range resolved once (a re-run
  reuses the frozen window, `resolve_window`): export a page at a time, check, convert, drop
  repeated conversations, and stop at `--target-count` rows or `--max-candidates` spans before
  fetching another page. Unreadable items are excluded as `unreadable_span` and count toward
  `--max-candidates` (`SpanQuery.max_spans` caps export items, readable or not); a re-run that reads
  further retires a stored row its trace now shows to be a prefix of a longer conversation
  (`retirements`, `contained_conversation`), and one with lower limits does not add such a row
  back (`contained`); both compare tools and messages as conversion collapses them
  (`comparable_tools` and `comparable_messages` in `trajectories.py`). With `--reasoning preserve`, a stored row a
  re-run reads more reasoning for is updated in place, never losing reasoning it has
  (`reasoning_updates`, `DatasetDir.add(updated=...)`), and a stored row that update makes
  repeat another stored row is retired as `duplicate_conversation` (`repeat_retirements`).
  `push.py` is `dataset push`: `dataset_record`
  (input = the messages before the final assistant message plus tools, expected output = that
  message, metadata = row id, split and source, tags `split:*` and `source:pup-finetune`),
  `plan_push` (what would be sent, from local files only) and `push_dataset`, which sends only
  kept rows in batches of 100 and saves `push.json` (project and dataset ids, version, the
  record id per row, the batch `in_flight`) around every batch, so a re-push sends only new
  rows and an interrupted push resumes, first adopting the records of a batch whose response
  was lost (matched on `metadata.row_id` through `records-all`) so no row is added twice (a
  listing that is truncated or hit pup's page cap, `RecordsListTruncated`, or that holds
  items it cannot read, `RecordsListing.unreadable`, keeps the batch `in_flight` and stops;
  so do rows of it not found within `IN_FLIGHT_SETTLE`, 10 minutes, of `in_flight_at`, since
  a timed-out write may still land); rows dropped after a push are reported as
  `stale_records`, not deleted, and kept rows whose record no longer matches the digest
  `push.json` saved for it (`record_digests`; none in older receipts) as `changed_records`,
  not updated: a push only adds records.
- `curation/` - deciding which accepted rows to train on. `labels.py` has `apply_labels`
  (every id an accepted row, last label per id wins, repeats skipped, appended through
  `DatasetDir.append_labels`); `samples.py` picks a seeded, readable sample of rows for
  review. `triage.py` is the judge council behind `dataset triage`: `Criteria` (rubric and/or
  rules, hashed), `build_messages` (the untrusted-data system prompt with the rubric, the
  whole row JSON-encoded in blocks), `parse_vote` (strict `{"keep", "reason"}`), `Vote`s in
  `judgments.jsonl` (counted only for the row content, rubric and judge they were cast for),
  `decide` (strict majority keeps, a tie drops, a missing vote is undecided), `plan_triage`
  (rows not labeled by the agent or a person, in a fixed pseudo-random order for
  `--max-rows`; settings frozen in `triage.json` once votes exist) and `run_triage` (a thread
  pool, votes appended as they arrive, failed calls reported and retried next run, labels
  `by: judge` for decided rows, judge labels the votes do not back reported as stale; a torn
  last vote line is cut before appending). `rubric_template.md` ships in the wheel. Curation builds on
  `data/`; `data/` never imports `curation/`.
- `training/` - the shared, provider-neutral SFT loop. `datums.py` renders rows into
  `TokenizedExample`s (per turn when a renderer is not prefix-stable) and rejects rows that
  exceed the context or train on nothing; `batching.py` shuffles rows per epoch and packs
  optimizer steps into microbatches; `budget.py` caps trained tokens and wall-clock spend;
  `loop.py` has `run_training` (pipelined steps, validation loss, early stopping, best epoch,
  epoch-end and time-based checkpoints, cancellation, deterministic resume at an epoch and
  step, and the `finalize` hook, whose failure raises `FinalizeFailed` carrying the result);
  `records.py` has its records (`EpochMetrics`, `TrainResult`, `ResumePoint`, `Resume`);
  `promotion.py` has `Promotion`, the `finalize` hook that promotes the best checkpoint
  before the session ends (re-creating its sampler checkpoint when that session expired) or
  records it as the deployable checkpoint, recording each promotion request per checkpoint
  first (a new best after an earlier promotion gets its own model id,
  `model_id_for_checkpoint`, so a model id never holds two checkpoints); `runner.py` has
  `start_training` and `resume_training`, which open a session for a run directory and train
  or resume in it (`start_training` takes an `after_finalize` hook, run in the session after
  a successful promotion with the run's spend guard; its failure only adds a warning);
  `inputs.py` renders a dataset's train and validation rows for a run (and checks, for an
  existing run, that the dataset and renderer did not change); `worker.py` runs a mode
  (`train`, `resume`, `finalize`) in the foreground or a detached worker (`spawn_worker`:
  `python -P -m pup_finetune _worker` in a new session, in the run directory, output in
  `worker.log`, the environment inherited and no secret passed on; the worker notes how it
  ended in `worker.json`), and has the liveness checks behind `status` and `cancel`.
- `runs.py` - the run directory (`run.json`, `state.json`, `epochs.json`, `result.json`,
  `deployment.json`),
  `RunState`, which the loop reports into, and `execute_run`, which holds the directory lock
  while training and always writes `result.json`, also on Ctrl-C or SIGTERM (reported as
  `TrainingTerminated`, a `FinetuneError`). Any error during training leaves the run
  `interrupted`, resumable from its last training checkpoint; a failed promotion leaves it
  `failed` with `resume_mode` `finalize_only`, so a resume only retries the promotion.
  `result.json` records `promoted_model`, `deployable_checkpoint` and `finalize_error`. With
  `signals="cancel"` (what the CLI uses), SIGTERM and the first Ctrl-C cancel gracefully
  instead: stop at a step boundary, checkpoint, promote the best epoch, end `cancelled`; with
  `heartbeat_s`, `state.json` is also saved on a timer so its heartbeat stays fresh.
- `rendering/` - chat rows to model tokens and completions back to messages. `tokens.py` has
  `TokenizedExample`, the provider-neutral, next-token-shifted token and weight arrays;
  `base.py` has the `Renderer` protocol and `RendererError`. `hf.py` has
  `HFChatTemplateRenderer`, the one renderer every provider shares: the model's Hugging Face
  chat template and tokenizer at a pinned revision, one example per trainable assistant turn
  (the prompt tokenized whole, then the completion through its stop token, re-encoded after
  the prompt when a token merges across the boundary), and a per-row-checked one-example fast
  path for families that pin `prefix_stable`. It also has the narrow `ChatTokenizer` seam
  (tests inject `tests/fakes/tokenizer.py`) and the lazily imported `transformers` adapter
  (the `hf` extra). `parsers.py` has the `ResponseParser`s (`Qwen3XmlToolParser`,
  `Qwen3_5XmlToolParser`, `GlmXmlToolParser`, `JsonToolCallParser`) that turn sampled text
  back into a message; `families.py` is the model-family table (`qwen3_8`, `qwen3_5`,
  `glm5_3`), each entry a tokenizer, revision, parser and stop tokens.
- `evaluation/` - replay evaluation, provider-neutral. `cases.py` turns a dataset's kept
  test-split rows into `ReplayCase`s (`cases_from_dataset`, `build_cases`): one per trained
  (`weight` not `0`) assistant message with a visible action, with the recorded prefix and
  tools as input and the recorded message as the reference, capped per trajectory
  (`max_cases_per_trajectory`) with evenly spaced, seeded picks that always keep the final
  turn. Tools are never executed, so replay measures agreement with the recorded behavior, not
  task success. `metrics.py` has the deterministic tool-call metrics (`MetricName`,
  `score_case`, `aggregate`; `None` when a metric does not apply) and `schema_errors`, a
  validator for the common JSON Schema subset tool declarations use. `judge.py` is the LLM
  judge: `judge` asks whether a candidate action is an acceptable substitute for the reference
  (the recorded data JSON-encoded in tagged blocks marked untrusted, reasoning never shown) and
  parses `{"pass", "reason"}` strictly (`parse_verdict`, outside any `<think>` block; anything
  else fails as `unparseable`, or `truncated` when the judge hit `max_tokens`, and a refused
  answer as `refused`); a malformed candidate fails without a call, and a judge outage raises rather
  than failing the case. `calibrate` checks that references pass and `corrupt`ed ones fail on a
  few mixed cases; `check_judge_differs` refuses a judge that is the model being trained or the run's
  tuned model (`model_id_for_run`, and the promoted model once there is one).
  `fake_judge.py` is `FakeJudge`, an offline, deterministic judge for development mode and
  tests (same tools with consistent arguments, or mostly the same words). `replay.py` has
  `run_replay`: every `(target, case)` pair (`base`, `tuned`) generated, scored and judged on
  a thread pool, each generation appended to `generations.jsonl` and each result to
  `results.jsonl` as it completes, so a rerun skips finished pairs; `evaluation.json` is the
  identity a rerun must match (`evaluation_identity`; a mismatch is a `StateError` asking for
  a new `--out-dir`); a sampler or judge failure is an error result, counted apart and out of
  every rate; `cancel` stops new pairs and keeps finished ones (any other error waits for the
  pairs in flight, then propagates). `summarize` writes
  `summary.json`: per-target pass rates (judge and metrics), per-trajectory agreement, the
  paired wins, regressions and ties, and `tuned_minus_base` with a seeded paired bootstrap
  95% CI over trajectories (`paired_bootstrap_ci`), plus the agreement-not-task-success
  `CAVEAT`. A `Publisher` (default `NoopPublisher`) gets a complete evaluation and returns
  what it reports, the outcome's `publication`. `runner.py` is what `evaluate` and `train
  --evaluate` run: `plan_evaluation` (an `EvalPlan`: cases, judge, settings and `Publishing`,
  whether the dataset's push receipt allows publishing), calibration (`prepare_eval_dir`
  stores a passing one), `replay` with the plan's publisher, `replay_result` and
  `evaluation_exit_code`, `plan_standalone` and `StandaloneEvaluation` for `evaluate`, and
  `EvaluateAfterTraining`, the `after_finalize` hook that evaluates in the training session
  and stops, partial, at the run's `--max-spend-usd`.
  The package exports its names lazily, so importing the runner (the CLI does, for every
  command) loads no judge client. `targets.py` makes the two samplers from a session: the live training
  session's best sampler checkpoint, or the best training checkpoint loaded and saved again
  as an `ev...` sampler checkpoint in a new session (`run_samplers`).
- `providers/` - the provider seam. `base.py` has the neutral types (`ModelProfile`,
  `TrainSpec`, `CheckpointRef`, `StepMetrics`, ...) and the `Provider`, `TrainingBackend`,
  `TrainingSession` and `Sampler` protocols; `__init__.py` is the lazy registry
  (`register_provider`, `get_provider`, `known_providers`), which imports a provider only on
  first use. `fake.py` is the offline, deterministic, dev-only reference provider (registered
  with `dev_only=True`, so it needs `PUP_FINETUNE_DEV=1`) that the default tests train
  against; its tuned checkpoints learn, per role of the message before each trained turn,
  whether to call a tool (by position) or answer in text, carried in the checkpoint URI. `options.py` has `parse_options`, which every provider validates its
  `--provider-option`s with (unknown keys refused, string values decoded as JSON). `fireworks/` is the Fireworks provider: its `__init__.py` only registers the
  `FIREWORKS_API_KEY` `CredentialSpec` (the registry imports it, so `auth` and `doctor` see
  the key without loading anything heavy; the validator lists accounts); `models.py` is the
  profile table (`qwen3p8-27b`), each entry naming a rendering family that supplies its
  tokenizer and revision; `api.py` is a small read-only REST client over `httpx` (no SDK;
  `model_info`, `list_accounts`, injectable transport, redacted errors, 401/403 as
  `FireworksAuthError`, 404 as a missing model); `provider.py` has `FireworksProvider`
  (profiles, preflight against `GET /v1/accounts/fireworks/models/<id>`, the family's HF
  renderer, the `max_seq_len` option). `client.py` is the only module that imports the
  Fireworks Training SDK (`fireworks`, `tinker`; the `fireworks` extra), and only on first
  use: it defines the narrow interface the backend uses (`FireworksTraining` -> `service()` ->
  `TrainingService` -> `TrainingRun` and `TokenSampler`, with plain results such as
  `ForwardOutput` and typed `CheckpointRow`s), implements it as `FireworksTrainingClient`
  (training runs in the key's own account; the recorded account id is `FIREWORKS_ACCOUNT_ID`,
  else the key's first account), converts to
  `tinker.Datum` and `tinker.AdamParams`, and maps every SDK failure with `fireworks_error`
  (`ProviderCapacityError`, `ProviderAuthError`, `TransientError`, redacted). `backend.py`
  has `FireworksBackend` (`provider.backend()`, built lazily with a client factory, so the
  SDK loads only when a session opens): one serverless session per `open_session`, always
  ended on exit (in-flight results drained, or abandoned on an error), resume through
  `create_training_client_from_state_with_optimizer`; `FireworksSession` (the validation loss
  computed from `forward`'s per-token logprobs, checkpoint names checked against the
  17-character limit before saving, `promote` picking the newest promotable sampler row
  carrying the checkpoint's name (`promotable_row`) and adopting a model this run already
  promoted whose response was lost (`adopt_since`: the run's request to promote that
  checkpoint, saved in `state.json` under `promotions`, less a minute's clock allowance),
  sampler checkpoints of an ended session refused with a hint); and `FireworksSampler`
  (render, sample greedily to the renderer's stop tokens, parse; only sampling is retried, a
  bounded number of times). `client.py` also wraps
  the SDK's `Fireworks` client for deployments (`FireworksDeployments`,
  `FireworksDeploymentsClient`: `match_shapes`, `create_deployment`, `get_deployment` and
  `delete_deployment` with plain `ShapeVersion` and `DeploymentRecord` results, 404 as gone,
  409 as a taken id). `deploy.py` has `FireworksDeployer` (shape choice, create with hot
  loading off and 0 to 1 replicas, polling every 15 s until `READY`, `DeploymentNotReady`
  for a bad state or timeout, the smoke test through the inference API retried while it
  scales from zero, idempotent delete; injectable clock and sleep), which
  `FireworksProvider(deployer=...)` replaces in tests. `baseten/client.py` is the only module
  that imports the Baseten Loops SDK (`baseten-loops`; the `baseten` extra), and only on first
  use, after refusing the `LOOPS_*` environment overrides: `BasetenHTTP` (Baseten's REST
  routes, among them the checkpoint deployment, instance types, models by name and model
  deletion; `Bearer` auth, no redirects, a 4 MiB cap, fixed messages; 401/403 as
  `ProviderAuthError`, 429 as `ProviderCapacityError`, a 4xx to a create as
  `DeploymentRejected` with Baseten's reason, `TransientError` for a call safe to repeat, a
  create accepted with an unreadable answer or a deployment whose outcome is unknown),
  `loops_service` (a Loops session with the key, base URL and dedicated availability passed
  explicitly), `loops_trainer`, `loops_sampling_client`, `loops_failure` (SDK failures, with
  the same auth, capacity and transient classes) and `to_loops_datum`. `baseten/backend.py`
  has `BasetenBackend` (a key factory resolved when a session opens, the profile's renderer,
  an optional `max_seq_len` cap), `BasetenSession` and `BasetenSampler`: one dedicated Loops
  trainer per session (its context the next power of two over the longest example, at least
  1024; no capacity is a `ProviderCapacityError` with nothing left running), steps as SDK
  futures, `bt://loops:` checkpoints, resume with the optimizer state, no promotion, and
  samplers on their own deployments with the profile's context (render, sample greedily,
  parse with the tools; a completion cut at `max_tokens` is not `format_valid`; only sampling
  is retried, a bounded number of times; `generate` is safe from several threads); leaving a
  session always deactivates its run and samplers. `baseten/__init__.py` only registers the
  `BASETEN_API_KEY` `CredentialSpec`, as Fireworks' does (falling back, read only, to the key
  of the default remote in `~/.trussrc`; the validator lists models). `baseten/provider.py`
  has `BasetenProvider`: the `qwen3p5-9b` and `glm-5p3-flash` profiles (tokenizer pins from
  their families), a read-only `preflight` (the SDK loads, and Loops' capabilities offer and
  enable the model with a limit of at least the `max_seq_len` option, else the profile's
  context), `check`, the family's HF renderer and the `max_seq_len` option (at least 1024; it
  cuts the training profile, so it also sizes the samplers and an evaluation's trainer);
  deployment arrives later. See `docs/architecture.md`.

Tests live under `tests/` and mirror `src/`. Shared fakes live in `tests/fakes/` (import them
as `tests.fakes...`); `tests/fakes/pup.py` has `FakePup`, a fake pup runner that answers by
argument prefix (or computes the answer with a `handler`) and records every call;
`tests/fakes/llmobs.py` has `FakeExportApi`, the Export API search route served by cursor.
`tests/fakes/datasets.py` has `FakeDatasetsApi`, in-memory projects, datasets and records
behind the pup calls `LLMObsDatasets` makes, with failures to inject.
`tests/fakes/experiments.py` has `FakeExperimentsApi`, in-memory experiments and their events
behind the pup calls `LLMObsExperiments` makes (failed and lost creates and event posts,
delayed or capped indexing); it chains to the `api` handler installed before it, so it serves
one `FakePup` with `FakeDatasetsApi`.
`tests/fakes/tokenizer.py` has `FakeChatTokenizer`, a character-level chat tokenizer with a
Qwen3.8-like Python template (optionally merging `\n\n` across the prompt boundary) or a
GLM-5-like one (`glm5_like`).
`tests/fakes/chat_api.py` has `FakeChatApi`, an OpenAI-compatible chat completions endpoint
(scripted responses or errors first, then an answer function) and `matching_judge`.
`tests/fakes/fireworks_api.py` has `FakeFireworksApi`, the Fireworks model and account routes
behind an `httpx.MockTransport`, with statuses and transport errors to inject.
`tests/fakes/fireworks.py` has `FakeFireworksTrainingClient`, the Fireworks Training API
behind the `client.py` interface in memory (sessions, server-renamed checkpoints, promotion
that fails once a session has expired, deterministic losses, canned samples, injectable
capacity, auth, 5xx and connection failures); `FireworksProvider(training_client=...)` trains
against it (see `tests/providers/fireworks/test_fireworks_backend.py` and
`tests/cli/test_train_fireworks.py`).
`tests/fakes/fireworks_deploy.py` has `FakeSdkFireworks`, the SDK's `Fireworks` client as far as
deployments use it (shape matching, deployments that report scripted states, injectable
failures), `FakeClock`, and `fake_deployer`, the provider's `deployer` factory on them and a
`FakeChatApi` (see `tests/providers/fireworks/test_fireworks_deploy.py` and
`tests/cli/test_deploy.py`).
`tests/fakes/baseten.py` has `FakeLoops`, the Loops SDK in memory (sessions, runs, futures,
deterministic losses and logprobs, `bt://loops:` checkpoints with the name limit, expiry,
closed clients and deactivated runs, samplers that answer a fixed completion, injectable
failures), and `BasetenTransport`, Baseten's HTTP routes by `"METHOD /path"`
(`FakeLoops.transport()` also serves its runs, samplers and their deployments);
`capabilities()` builds a Loops capabilities answer.
`tests/fakes/keyring.py` has in-memory, failing and locked keyring backends; every test starts
on the failing one, so none touches the real keychain.
`tests/providers/contract/` has the `RendererContract`, `BackendContract` and
`ProviderContract` suites every provider subclasses (`tests/providers/test_fake_contract.py`
runs them against the fake provider; `tests/providers/fireworks/` against Fireworks on its fakes;
`tests/providers/baseten/` against Baseten on its fake).
Synthetic fixtures live in `tests/fixtures/` (see its README); `rows/` holds canonical rows and
`llmobs/` raw Export API pages.

## Conventions

- Every first-party file starts with the Apache-2.0 license header (see any `.py` file); shell
  scripts and YAML use the same lines with `#`.
- Everything is typed; `mypy --strict` must pass on `src/`.
- Each command writes exactly one JSON result to stdout; progress and logs go to stderr. Wrap
  commands with `run_command` from `cli/_common.py`: return the result mapping and raise a
  `FinetuneError` subclass for expected failures. Return `CommandResult(data, exit_code=1)`
  when the result must still be printed but the command should fail (e.g. `doctor`).
- Exit codes: `0` success, `1` external or runtime failure, `2` usage, validation or a missing
  `--confirm`.
- Paid or remote-changing actions preview by default and run only with `--confirm`. `--confirm`
  is never implied by `PUP_AUTO_APPROVE` or agent mode.
- Honor `PUP_READ_ONLY`: reject every remote write when it is set.
- Only `datadog/pup.py` spawns `pup`. Only `providers/<provider>/client.py` imports a training
  provider's SDK, and only `anthropic_client.py` imports `anthropic` (and `httpx2`).
  Only `rendering/hf.py` imports `transformers`, and only when a tokenizer is first loaded.
- Secrets are never printed, logged or written to disk, and never accepted as a flag value.
- Tests use fakes, not mocks. Default tests make no network or paid calls.
- Fixtures are synthetic only: never commit real trace content or customer data.
- No internal links or internal-only references; this repository will be public.

## Adding a provider

See "How a new provider plugs in" in `docs/architecture.md`.
