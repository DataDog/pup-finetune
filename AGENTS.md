# AGENTS.md

Guide for contributors and coding agents working in this repository.

## What this is

`pup-finetune` curates a supervised fine-tuning dataset from your Datadog Agent Observability
traces, fine-tunes an open-weight model on it, evaluates the tuned model against the base model,
and deploys it. It ships as a `pup` extension, so users run it as `pup finetune ...`; the
standalone executable is `pup-finetune`. All Datadog reads and writes go through `pup` and its
OAuth login. The project is in early development: most commands arrive in later PRs.

## Commands

```sh
uv sync                              # create .venv and install the project + dev tools
uv run pytest                        # tests (no network, no paid calls)
uv run ruff check --fix              # lint
uv run ruff format                   # format (check only: uv run ruff format --check)
uv run mypy src                      # type-check (strict)
uv run python scripts/check_headers.py  # license header on every tracked .py and shell script
uv run python scripts/check_3rdparty.py # LICENSE-3rdparty.csv covers every dependency
uv run pre-commit install            # run the hooks above on every commit
uv run pup-finetune --help           # run the CLI from the checkout
```

CI (`.github/workflows/ci.yml`) runs the lint, format, type, test, header and third-party
license checks on Ubuntu and macOS with Python 3.12 and 3.13, and smoke-tests the built wheel.
When you add a dependency, add its row to `LICENSE-3rdparty.csv`.

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
```

To build a dataset from your own Agent Observability traces (Datadog reads through pup, so
both work with `PUP_READ_ONLY` set):

```sh
uv run pup-finetune dataset preview --ml-app my-app --from now-7d   # in memory, writes nothing
uv run pup-finetune dataset pull /tmp/ds --ml-app my-app --from now-7d
                                     # the same flags; rerun to resume (pages cached in /tmp/ds/raw)
uv run pup-finetune dataset show /tmp/ds
uv run pup-finetune dataset sample /tmp/ds --n 20            # rows in brief, to review
uv run pup-finetune dataset label /tmp/ds --file labels.jsonl  # {id, keep, reason} per line
```

A relative `--from` or `--to` (`now`, or `now-<n><unit>` with unit `s`, `m`, `h`, `d` or `w`)
is resolved to epoch ms when the command starts. The first pull into a directory freezes that
window together with the expressions as written, and a re-run with the same expressions reads
the frozen window, so a resumed pull reads the same spans.

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

Opt-in test suites: `PUP_FINETUNE_LIVE=1` runs `live` tests (real external services);
`PUP_FINETUNE_TOKENIZER_TESTS=1` runs `tokenizer` tests (downloads real tokenizers).

## Layout

Source lives under `src/pup_finetune/`. Packages marked *later* arrive in later PRs.

- `cli/` - the Typer app; one module per command group. `dataset.py` has `dataset import`,
  `preview`, `pull`, `show`, `sample` and `label`; `train.py` has `train`, `resume`, `status`,
  `cancel` and the hidden `_worker` a detached run executes.
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
- `datadog/` - Datadog access via `pup`. `pup.py` is the client every Datadog call goes
  through: `pup --no-agent --output json ...`, a scrubbed environment, timeouts, and failures as
  `ExternalError`. `spans.py` exports LLM spans through the Export API
  (`POST v2/llm-obs/v1/spans/events/search` via `Pup.api`): `SpanQuery` (the filter, plus
  `settings()` to freeze), `iter_spans` and `iter_span_pages` (cursor pagination, retries of
  timed-out pages, an optional raw page cache that resumes without refetching, `ExportStats`),
  and tolerant `LLMObs*` models that skip and count spans they cannot read (`UnreadableSpan`).
  Datasets and experiments arrive later.
- `data/` - training data. `schema.py` is the canonical row every provider consumes (OpenAI
  chat format plus its source) and the stable `RowErrorCode` reasons a row is rejected for;
  `jsonl.py` reads and writes rows and converts them to the provider form; `exclusions.py`
  records why a row was left out (`excluded.jsonl`), with `ExclusionCode`s for reasons the
  schema cannot see (conversion problems, dataset-level rules);
  `split.py` assigns deterministic, persisted train/validation/test splits per trace;
  `dataset_dir.py` is the dataset directory (`settings.json`, `rows.jsonl`, `excluded.jsonl`,
  `splits.json`, `summary.json`, and `raw/`, a pull's page cache) behind `dataset import`,
  `dataset pull` and `dataset show`; `labels.py` has the keep/drop `Label` and the
  `labels.jsonl` format (append-only, last label per id wins, a reason required to drop;
  `DatasetDir.kept_rows` leaves dropped rows out of training and split exports).
  `convert.py` maps one LLM span to a canonical conversation (`span_conversation`: input then
  output messages, tool calls and results, tools, `SourceRef`; reasoning omitted or
  preserved), with conversion issues as stable codes; `trajectories.py` has
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
  back (`contained`); both compare messages as conversion collapses them
  (`comparable_messages` in `trajectories.py`).
- `curation/` - deciding which accepted rows to train on. `labels.py` has `apply_labels`
  (every id an accepted row, last label per id wins, repeats skipped, appended through
  `DatasetDir.append_labels`); `samples.py` picks a seeded, readable sample of rows for
  review. Judge-based triage is *later*. Curation builds on `data/`; `data/` never imports
  `curation/`.
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
  records it as the deployable checkpoint; `runner.py` has `start_training` and
  `resume_training`, which open a session for a run directory and train or resume in it;
  `inputs.py` renders a dataset's train and validation rows for a run (and checks, for an
  existing run, that the dataset and renderer did not change); `worker.py` runs a mode
  (`train`, `resume`, `finalize`) in the foreground or a detached worker (`spawn_worker`:
  `python -P -m pup_finetune _worker` in a new session, in the run directory, output in
  `worker.log`, the environment inherited and no secret passed on; the worker notes how it
  ended in `worker.json`), and has the liveness checks behind `status` and `cancel`.
- `runs.py` - the run directory (`run.json`, `state.json`, `epochs.json`, `result.json`),
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
  `base.py` has the `Renderer` protocol (built on Hugging Face chat templates) and
  `RendererError`. Concrete renderers arrive later.
- `evaluation/` - *later*. Replay cases, deterministic metrics, LLM judge and reporting.
- `providers/` - the provider seam. `base.py` has the neutral types (`ModelProfile`,
  `TrainSpec`, `CheckpointRef`, `StepMetrics`, ...) and the `Provider`, `TrainingBackend`,
  `TrainingSession` and `Sampler` protocols; `__init__.py` is the lazy registry
  (`register_provider`, `get_provider`, `known_providers`), which imports a provider only on
  first use. `fake.py` is the offline, deterministic, dev-only reference provider (registered
  with `dev_only=True`, so it needs `PUP_FINETUNE_DEV=1`) that the default tests train
  against. One package per real provider arrives later. See `docs/architecture.md`.
- `_vendor/` - *later*. Minimal vendored third-party code, with its own license and attribution.

Tests live under `tests/` and mirror `src/`. Shared fakes live in `tests/fakes/` (import them
as `tests.fakes...`); `tests/fakes/pup.py` has `FakePup`, a fake pup runner that answers by
argument prefix (or computes the answer with a `handler`) and records every call;
`tests/fakes/llmobs.py` has `FakeExportApi`, the Export API search route served by cursor.
`tests/fakes/keyring.py` has in-memory, failing and locked keyring backends; every test starts
on the failing one, so none touches the real keychain.
`tests/providers/contract/` has the `RendererContract`, `BackendContract` and
`ProviderContract` suites every provider subclasses (`tests/providers/test_fake_contract.py`
runs them against the fake provider).
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
- Only `datadog/pup.py` spawns `pup`. Only `providers/<provider>/client.py` imports a provider SDK.
- Secrets are never printed, logged or written to disk, and never accepted as a flag value.
- Tests use fakes, not mocks. Default tests make no network or paid calls.
- Fixtures are synthetic only: never commit real trace content or customer data.
- No internal links or internal-only references; this repository will be public.

## Adding a provider

See "How a new provider plugs in" in `docs/architecture.md`.
