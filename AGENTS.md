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

Live `train` and `deploy` from the checkout also need the `hf` extra and the Loops SDK:
`uv sync --extra hf`, then `uv pip install baseten-loops==0.26.0`. A plain `uv sync` removes both
again; the launcher's `uv run` keeps them.

Environment variables for development and unusual setups:

- `PUP_FINETUNE_PUP` - the pup executable to run (a path, or a name on `PATH`) instead of `pup`.
- `PUP_FINETUNE_KEEP_DD_TOKEN=1` - pass `DD_ACCESS_TOKEN` through to the `pup` processes we
  start. By default it is removed, so each child `pup` loads and refreshes its own stored OAuth
  token instead of reusing the short-lived token pup handed us. Set it only if you
  authenticate pup through environment variables alone.

Opt-in test suites: `PUP_FINETUNE_LIVE=1` runs `live` tests (real external services);
`PUP_FINETUNE_TOKENIZER_TESTS=1` runs `tokenizer` tests (downloads real tokenizers).

## Layout

Source lives under `src/pup_finetune/`. Packages marked *later* arrive in later PRs.

- `cli/` - the Typer app; one module per command group (`train.py`; `deploy.py` holds `deploy`
  and `undeploy`).
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
  `ExternalError`. Span export, datasets and experiments arrive later.
- `data/` - training data. `schema.py` is the canonical row every provider consumes (OpenAI
  chat format plus its source) and the stable `RowErrorCode` reasons a row is rejected for;
  `jsonl.py` reads and writes rows and converts them to the provider form. Conversion from
  spans, checks, splits and dataset directories arrive later.
- `curation/` - *later*. Keep/drop labels and judge-based triage.
- `training/` - the shared, provider-neutral LoRA SFT loop (`loop.py`), `SFTSettings`, microbatch
  packing, the spend guard and the run directory (`run.py`: `plan.json`, `epochs.json`,
  `run-state.json`, `result.json`). The detached worker arrives later.
- `rendering/` - turns a canonical row into token datums with per-token loss weights by diffing
  chat-template prefixes (`render.py`). `families.py` pins each model family's tokenizer
  revision; `hf.py` is the only module that imports `transformers` (the `hf` extra).
- `evaluation/` - *later*. Replay cases, deterministic metrics, LLM judge and reporting.
- `providers/` - the `Provider` protocol, its requests and the lazy registry (`__init__.py`), and
  one package per provider. `baseten/`: `registration.py` (credential spec and `doctor` checks;
  cheap, imported with the registry; its key check is the one Baseten call outside `client.py`),
  `models.py` (supported models), `client.py` (all other Baseten HTTP, and the only module that
  imports the Loops SDK, which is optional and not declared because it has no published license),
  `training.py`, `deploy.py` and `provider.py`.
- `_vendor/` - *later*. Minimal vendored third-party code, with its own license and attribution.

Tests live under `tests/` and mirror `src/`. Shared fakes live in `tests/fakes/` (import them
as `tests.fakes...`); `tests/fakes/pup.py` has `FakePup`, a fake pup runner that answers by
argument prefix and records every call. `tests/fakes/keyring.py` has in-memory, failing and locked
keyring backends; every test starts on the failing one, so none touches the real keychain.
`tests/fakes/tokenizer.py` has `FakeChatTokenizer`, `tests/fakes/training.py` a scripted
`FakeBackend`, and `tests/fakes/baseten.py` `FakeLoops` (the SDK) and an `httpx` mock transport.
Synthetic fixtures live in `tests/fixtures/` (see its README); `rows/` holds canonical rows.

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

1. Add `providers/<name>/` whose `PROVIDER` implements `providers.Provider`, and add `<name>` to
   `_PROVIDERS` in `providers/__init__.py`.
2. Register its credential spec and `doctor` checks in a cheap `registration.py` that
   `providers/__init__.py` imports; load everything else lazily inside the provider's methods.
3. Keep every SDK import and remote call in `providers/<name>/client.py` (only the key check in
   `registration.py` makes its own cheap call), and test it with fakes under `tests/fakes/`. See
   `providers/baseten/` for a complete example.
