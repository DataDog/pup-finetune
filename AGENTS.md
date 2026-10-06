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
uv run pre-commit install            # run the hooks above on every commit
uv run pup-finetune --help           # run the CLI from the checkout
```

Once `bin/pup-finetune` exists, install the checkout as a pup extension:

```sh
pup extension install --local bin/pup-finetune --link
pup finetune --help
```

Opt-in test suites: `PUP_FINETUNE_LIVE=1` runs `live` tests (real external services);
`PUP_FINETUNE_TOKENIZER_TESTS=1` runs `tokenizer` tests (downloads real tokenizers).

## Layout

Source lives under `src/pup_finetune/`. Packages marked *later* arrive in later PRs.

- `cli/` - the Typer app; one module per command group.
- `datadog/` - *later*. Datadog access via `pup` (span export, datasets, experiments).
- `data/` - *later*. Canonical row schema, conversion from spans, checks, splits, dataset dirs.
- `curation/` - *later*. Keep/drop labels and judge-based triage.
- `training/` - *later*. The shared, provider-neutral SFT loop, run state and detached worker.
- `rendering/` - *later*. The `Renderer` protocol that turns chat rows into model tokens.
- `evaluation/` - *later*. Replay cases, deterministic metrics, LLM judge and reporting.
- `providers/` - *later*. Provider protocols, the lazy registry and one package per provider.
- `_vendor/` - *later*. Minimal vendored third-party code, with its own license and attribution.

Tests live under `tests/` and mirror `src/`.

## Conventions

- Every first-party file starts with the Apache-2.0 license header (see any `.py` file); shell
  scripts and YAML use the same lines with `#`.
- Everything is typed; `mypy --strict` must pass on `src/`.
- Each command writes exactly one JSON result to stdout; progress and logs go to stderr.
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

See `CONTRIBUTING.md` (the "Adding a provider" section arrives in a later PR).
