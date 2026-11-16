# Contributing

Thanks for your interest in `pup-finetune`! Bug reports, feature requests and pull requests are
all welcome. For a security issue, follow [SECURITY.md](SECURITY.md) instead of opening an issue.

## Setup

You need Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --all-extras         # create .venv and install the project, its extras and dev tools
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

After you change a command's help or flags, regenerate the command reference with
`uv run python scripts/gen_command_docs.py`; a test fails when `docs/commands.md` is stale.

## Sign your commits

The `main` branch accepts only signed commits, so sign every commit in your pull request. If you
don't sign commits yet, follow GitHub's guide to
[signing commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits).

## Pull requests

- Keep pull requests small and focused on one change. For a new feature or a larger change,
  start with an RFC (see the next section).
- Add or update tests for every behavior change. Tests mirror the `src/` layout: tests for
  `src/pup_finetune/cli/` live under `tests/cli/`. Use fakes rather than mocks, and
  make no network or paid calls in the default suite.
- Start every new Python file and shell script with the license header used throughout the
  repository (see any `.py` file). `scripts/check_headers.py` enforces it.
- Do not add links to internal or private resources; this repository is public.
- Use synthetic data only in tests, fixtures and examples. Never commit real traces, credentials
  or customer data.
- Update the documentation when you change user-facing behavior.

## New features and larger changes

Get a new feature or a larger change accepted before you write the code. This saves you from
building something that is already in progress another way, or that is out of scope.

1. Open an issue with the RFC template, titled `[RFC] <your proposal>`. Explain the problem,
   who it affects, and how you plan to solve it.
2. A maintainer labels the issue `rfc/discussion` and discusses the proposal with you.
3. If the proposal is accepted, the issue is labeled `rfc/approved`. Open your pull request and
   link it to the issue.
4. If the proposal is declined, a maintainer explains why, labels the issue `rfc/rejected` and
   closes it.

A bug fix needs an RFC only when the fix is large or changes behavior that others rely on.
Maintainers may close a feature pull request that has no approved RFC.

## AI code assistants

You can use AI code assistants to contribute. The rest of this guide still applies, including
the RFC process for new features. The change is yours: understand the code you submit, review
it, and test it. Maintainers may close a pull request that looks generated without review or
tests.

Point your assistant at [AGENTS.md](AGENTS.md). It describes the project's layout, conventions
and checks. If an assistant helped with a pull request, name it in the pull request
description.

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
      chat-template renderer (`rendering/hf.py`) of the model family from
      `rendering/families.py`; only a new model family needs a new table entry (plus a
      response parser when its output format is new).
- [ ] **Registration.** Register the provider lazily in `providers/__init__.py` with
      `register_provider("<name>", "pup_finetune.providers.<name>.provider:<Name>Provider")`,
      so listing providers imports no provider implementation; keep the implementation in
      `provider.py` and the package `__init__.py` light. Name its `--provider-option` keys
      with `options=(...)` (`train --help` lists them, and a test holds them to the
      provider's options model), and its profiles' aliases with `models=(...)` (`train`
      infers `--provider` from them, and a test holds them to the provider's profiles).
      Say what it needs: `label=`, `credential=` (its `CredentialSpec`), `packages=`
      (`Package`s: the SDK and the renderer, by step), `refuses=` (`RefusedFlag`s for flags
      it cannot honor) and `deploys=False` if it cannot deploy yet. `doctor` and the
      previews read these, so nothing fails only after `--confirm`.
- [ ] **Credentials.** Add a `CredentialSpec` for the provider's API key (with
      `needed_for`, what `doctor` and `auth status` say it is for) to the registry in
      `credentials.py`, from the package's `__init__.py` (keep it free of heavy imports), and
      import the package at the bottom of `providers/__init__.py` with
      `from pup_finetune.providers import <name> as _<name>  # noqa: E402, F401`, so
      `auth login`, `auth status` and `doctor` cover it.
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
the same renderers as on any other provider, and `preflight` asks Baseten, read only, whether
the workspace can train the model. Without promotion, `deploy` serves the run's deployable
checkpoint, and reads a model back by name before it adopts or deletes one, since Baseten
assigns the ids. `BASETEN_API_KEY` gets a `CredentialSpec` from the
package's `__init__.py` (falling back, read only, to the key in `~/.trussrc`), the SDK goes in
a `baseten` extra, and the three contract suites run against a fake Loops client in the
default suite and against Baseten itself under `PUP_FINETUNE_LIVE=1`.

## License

By contributing, you agree that your contributions are licensed under the repository's
[LICENSE](LICENSE).
