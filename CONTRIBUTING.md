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

A training provider is a small package behind the seam described in
[docs/architecture.md](docs/architecture.md#how-a-new-provider-plugs-in). The offline `fake`
provider, [`src/pup_finetune/providers/fake.py`](src/pup_finetune/providers/fake.py), implements
every protocol in one file and is the reference to read first. A provider is done when it
ticks every box:

- [ ] **Package.** Put the code under `src/pup_finetune/providers/<name>/`. Only `client.py`
      imports the provider's SDK; every other module talks to that client.
- [ ] **Protocols.** Implement `Provider`, `TrainingBackend` (with its `TrainingSession`) and
      `Sampler` from `providers/base.py`. For the `Renderer`, reuse the shared Hugging Face
      chat-template renderer of the model family; only a new model family needs a new one.
- [ ] **Registration.** Register the provider lazily in `providers/__init__.py` with
      `register_provider("<name>", "pup_finetune.providers.<name>:<Name>Provider")`, so
      listing providers imports nothing.
- [ ] **Credentials.** Add a `CredentialSpec` for the provider's API key to the registry in
      `credentials.py`, so `auth login`, `auth status` and `doctor` cover it.
- [ ] **Install.** Add a `<name>` extra to `pyproject.toml` for the SDK, with its rows in
      `LICENSE-3rdparty.csv`.
- [ ] **Contract suites.** Subclass `RendererContract`, `BackendContract` and
      `ProviderContract` from `tests/providers/contract/` and provide the fixtures each one
      declares (`renderer_factory` for the renderer suite), backed by a fake of your SDK
      client. `tests/providers/test_fake_contract.py` shows how.
- [ ] **Live tests.** Add tests against the real service marked `@pytest.mark.live`; they run
      only with `PUP_FINETUNE_LIVE=1`, so the default suite stays offline and free.

Take Baseten as the worked example. `providers/baseten/client.py` is the only module that
imports the Baseten Loops SDK. The backend's `open_session` returns a context manager that
deactivates the Loops session on exit, whether training finished, stopped early or failed, so
an interrupted run never leaves a session billing. It converts each `TokenizedExample` to the
SDK's datum type, maps `AdamConfig` onto the SDK's optimizer parameters, sizes the trainer
from the `seed` and `max_seq_len` it is opened with, and reports the provider's
checkpoint-name limit as `max_checkpoint_name_len` and its capabilities as
`supports_resume` and `supports_promotion`. The models Baseten serves use
the same renderers as on any other provider. `BASETEN_API_KEY` gets a `CredentialSpec`, the SDK
goes in a `baseten` extra, and the three contract suites run against a fake Loops client in
the default suite and against Baseten itself under `PUP_FINETUNE_LIVE=1`.

## License

By contributing, you agree that your contributions are licensed under the repository's
[LICENSE](LICENSE).
