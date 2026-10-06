# Contributing

Thanks for your interest in `pup-finetune`! Bug reports, feature requests and pull requests are
all welcome. For a security issue, follow [SECURITY.md](SECURITY.md) instead of opening an issue.

## Setup

You need Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                      # create .venv and install the project and dev tools
uv run pre-commit install    # run the checks below on every commit
```

## Checks

Every pull request must pass:

```sh
uv run ruff check                       # lint
uv run ruff format --check              # formatting
uv run mypy src                         # strict type-check
uv run pytest                           # tests (no network, no paid calls)
uv run python scripts/check_headers.py  # license headers
uv run python scripts/check_3rdparty.py # LICENSE-3rdparty.csv covers every dependency
```

## Pull requests

- Keep pull requests small and focused on one change. Open an issue first for larger changes.
- Add or update tests for every behavior change. Tests mirror the `src/` layout: tests for
  `src/pup_finetune/cli/` live under `tests/cli/`. Use fakes rather than mocks, and
  make no network or paid calls in the default suite.
- Start every new Python file and shell script with the license header used throughout the
  repository (see any `.py` file). `scripts/check_headers.py` enforces it.
- Do not add links to internal or private resources; this repository is public.
- Use synthetic data only in tests, fixtures and examples. Never commit real traces, credentials
  or customer data.
- Update the documentation when you change user-facing behavior.

## Adding a provider

See the "Adding a provider" section of [AGENTS.md](AGENTS.md).

## License

By contributing, you agree that your contributions are licensed under the repository's
[LICENSE](LICENSE).
