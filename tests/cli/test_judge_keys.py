# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""A preview that would call a paid judge or compare model without a key fails instead.

Its ``--confirm`` would fail on the missing key, so the preview names the key and how to log
in (exit 2) and offers no ``--confirm``. A key in the environment, in the keychain or, for
Anthropic, the SDK's own login is enough, checked offline. The fake judge and compare model
need none. Every test runs offline, in development mode, on the fake provider.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import keyring.core
import pytest
from typer.testing import CliRunner, Result

from pup_finetune import credentials, judges
from pup_finetune.cli import app
from pup_finetune.errors import ExternalError
from pup_finetune.evaluation.fake_judge import FakeJudge
from pup_finetune.inference import ChatCompleter, ChatCompletion, JudgeConfig
from tests.conftest import JUDGE_KEYS
from tests.fakes.keyring import LockedKeyring, MemoryKeyring

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"
PAID = "openai/gpt-5.5"

runner = CliRunner()


@pytest.fixture(autouse=True)
def _dev_mode_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")
    for name in JUDGE_KEYS:
        monkeypatch.delenv(name, raising=False)


def invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def output(result: Result) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.stdout)
    return data


def refused(result: Result, provider: str, needed_by: str) -> None:
    """``result`` failed on the missing key, named with its login hint, offering nothing."""
    assert result.exit_code == 2, result.stderr
    assert result.stdout == ""
    assert f"Error [usage]: no {provider} API key found for the {needed_by}\n" in result.stderr
    assert f"pup finetune auth login {provider}" in result.stderr


def offered(result: Result) -> dict[str, Any]:
    """``result`` is a preview whose next command is its ``--confirm``."""
    assert result.exit_code == 2, result.stderr
    data = output(result)
    assert data["next_command"].endswith(" --confirm")
    return data


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ds"
    assert invoke("dataset", "import", str(path), "--file", str(ROWS)).exit_code == 0
    return path


def train_args(dataset: Path, run: Path, *extra: str) -> list[str]:
    return [
        "train", "--model", "fake-model", "--dataset", str(dataset), "--run-dir", str(run),
        "--max-epochs", "1", *extra,
    ]  # fmt: skip


@pytest.fixture
def run_dir(dataset: Path, tmp_path: Path) -> Path:
    run = tmp_path / "run"
    trained = invoke(*train_args(dataset, run, "--confirm"))
    assert trained.exit_code == 0, trained.stderr
    return run


def test_evaluate_needs_the_judge_s_key(run_dir: Path) -> None:
    refused(invoke("evaluate", str(run_dir), "--judge-model", PAID), "openai", f"judge {PAID}")
    assert not (run_dir / "eval").exists()


def test_a_key_in_the_keychain_is_enough(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    keychain = MemoryKeyring()
    keychain.set_password("pup-finetune", "openai", "sk-test-key")
    monkeypatch.setattr(keyring.core, "_keyring_backend", keychain)
    offered(invoke("evaluate", str(run_dir), "--judge-model", PAID))


def test_the_anthropic_sdk_s_own_login_is_enough(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("anthropic")
    judge = "anthropic/claude-opus-5-5"
    refused(invoke("evaluate", str(run_dir), "--judge-model", judge), "anthropic", f"judge {judge}")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "an-sdk-token")
    offered(invoke("evaluate", str(run_dir), "--judge-model", judge))


def test_evaluate_needs_the_compare_model_s_key(run_dir: Path) -> None:
    assert invoke("evaluate", str(run_dir), "--no-publish", "--confirm").exit_code == 0
    compare = invoke("evaluate", str(run_dir), "--compare-model", PAID, "--no-publish")
    refused(compare, "openai", f"compare model {PAID}")
    assert not (run_dir / "eval" / "compare.json").exists()
    # The offline judge and compare model need no key.
    offered(invoke("evaluate", str(run_dir), "--compare-model", "fake-compare", "--no-publish"))


def test_train_evaluate_needs_the_judge_s_key(dataset: Path, tmp_path: Path) -> None:
    run = tmp_path / "run"
    result = invoke(*train_args(dataset, run, "--evaluate", "--eval-judge-model", PAID))
    refused(result, "openai", f"judge {PAID}")
    assert not run.exists()


def test_benchmark_needs_the_judge_s_key(dataset: Path) -> None:
    result = invoke("benchmark", str(dataset), "--models", "fake-model", "--judge-model", PAID)
    refused(result, "openai", f"judge {PAID}")


def test_triage_needs_each_judge_s_key(dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    council = ["dataset", "triage", str(dataset), "--rule", "stay on topic"]
    result = invoke(*council, "--judges", f"deepseek-v4p1-flash,{PAID}")
    refused(result, "fireworks", "judge accounts/fireworks/models/deepseek-v4p1-flash")
    monkeypatch.setenv("FIREWORKS_API_KEY", JUDGE_KEYS["FIREWORKS_API_KEY"])
    refused(invoke(*council, "--judges", f"deepseek-v4p1-flash,{PAID}"), "openai", f"judge {PAID}")
    offered(invoke(*council, "--judges", "deepseek-v4p1-flash"))


# --- a run with nothing left to ask a model never opens its client --------------------------


class Voter:
    """A triage judge that keeps every row, or with ``down`` fails every call."""

    def __init__(self, *, down: bool = False) -> None:
        self.down = down

    def complete(self, messages: Sequence[Mapping[str, Any]], **options: Any) -> ChatCompletion:
        if self.down:
            raise ExternalError("the judge endpoint is unreachable", source="judge")
        return ChatCompletion(message={"role": "assistant", "content": KEEP})


KEEP = '{"keep": true, "reason": "on topic"}'


def serve(monkeypatch: pytest.MonkeyPatch, answer: Callable[[str], ChatCompleter]) -> list[str]:
    """Serve every judge client with ``answer(provider)``; the providers opened, in order. A
    provider without a key refuses to open, as its real client does."""
    opened: list[str] = []

    @contextmanager
    def open_client(config: JudgeConfig, **options: Any) -> Iterator[ChatCompleter]:
        if credentials.key_source(config.credential) is None:
            raise credentials.missing_key_error(config.credential)
        opened.append(config.provider)
        yield answer(config.provider)

    monkeypatch.setattr(judges, "open_client", open_client)
    return opened


def test_a_complete_evaluation_reruns_without_the_judge_s_key(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = ["evaluate", str(run_dir), "--judge-model", PAID, "--no-publish", "--confirm"]
    monkeypatch.setenv("OPENAI_API_KEY", JUDGE_KEYS["OPENAI_API_KEY"])
    assert serve(monkeypatch, lambda provider: FakeJudge()) == []
    first = invoke(*command)
    assert first.exit_code == 0, first.stderr
    monkeypatch.delenv("OPENAI_API_KEY")
    opened = serve(monkeypatch, lambda provider: FakeJudge())
    again = invoke(*command)
    assert again.exit_code == 0, again.stderr
    assert output(again)["status"] == "complete"
    assert opened == []


def test_a_complete_benchmark_reruns_without_the_judge_s_key(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = ["benchmark", str(dataset), "--models", "fake-model", "--judge-model", PAID]
    monkeypatch.setenv("OPENAI_API_KEY", JUDGE_KEYS["OPENAI_API_KEY"])
    serve(monkeypatch, lambda provider: FakeJudge())
    first = invoke(*command, "--confirm")
    assert first.exit_code == 0, first.stderr
    monkeypatch.delenv("OPENAI_API_KEY")
    opened = serve(monkeypatch, lambda provider: FakeJudge())
    again = invoke(*command, "--confirm")
    assert again.exit_code == 0, again.stderr
    assert opened == []


def test_a_resumed_council_opens_only_the_judges_with_votes_left(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    council = ["dataset", "triage", str(dataset), "--rule", "stay on topic"]
    council += ["--judges", f"deepseek-v4p1-flash,{PAID}"]
    for name in ("FIREWORKS_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, JUDGE_KEYS[name])
    serve(monkeypatch, lambda provider: Voter(down=provider == "openai"))
    first = invoke(*council, "--confirm")
    assert output(first)["status"] == "incomplete", first.stderr
    # Only OpenAI votes are left, so the Fireworks key is no longer needed.
    monkeypatch.delenv("FIREWORKS_API_KEY")
    opened = serve(monkeypatch, lambda provider: Voter())
    offered(invoke(*council))
    resumed = invoke(*council, "--confirm")
    assert resumed.exit_code == 0, resumed.stderr
    assert output(resumed)["status"] == "complete"
    assert opened == ["openai"]


def test_a_read_only_preview_still_fails_on_a_locked_keychain(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "1")
    monkeypatch.setattr(keyring.core, "_keyring_backend", LockedKeyring())
    result = invoke("evaluate", str(run_dir), "--judge-model", PAID)
    assert result.exit_code == 1, result.stderr
    assert "Error [external]" in result.stderr


def test_a_read_only_preview_warns_of_the_missing_key(
    run_dir: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A read-only preview offers no --confirm, so the key it would need is only a warning.
    monkeypatch.setenv("PUP_READ_ONLY", "1")
    previews = (
        ["evaluate", str(run_dir), "--judge-model", PAID],
        ["benchmark", str(dataset), "--models", "fake-model", "--judge-model", PAID],
    )
    for args in previews:
        result = invoke(*args)
        assert result.exit_code == 2, result.stderr
        data = output(result)
        assert data["next_command"] is None
        assert "read_only" in data["preview"]
        missing = f"no openai API key found for the judge {PAID}: "
        (warning,) = (text for text in data["preview"]["warnings"] if text.startswith(missing))
        assert "pup finetune auth login openai" in warning


def test_a_preview_refuses_a_malformed_judge_key(
    run_dir: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test key")
    for args in (
        ["evaluate", str(run_dir), "--judge-model", PAID],
        ["benchmark", str(dataset), "--models", "fake-model", "--judge-model", PAID],
    ):
        result = invoke(*args)
        assert result.exit_code == 2, result.stderr
        assert "Error [usage]: the openai API key is malformed" in result.stderr
        assert result.stdout == ""


def test_a_malformed_judge_key_fails_before_anything_is_generated(
    run_dir: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test key")
    runs = (
        (
            ["evaluate", str(run_dir), "--judge-model", PAID, "--no-publish"],
            run_dir / "eval" / "generations.jsonl",
        ),
        (
            ["benchmark", str(dataset), "--models", "fake-model", "--judge-model", PAID],
            dataset / "benchmark" / "fake-model" / "generations.jsonl",
        ),
    )
    for args, generations in runs:
        result = invoke(*args, "--skip-calibration", "--confirm")
        assert result.exit_code == 2, result.stderr
        assert "Error [usage]: the openai API key is malformed" in result.stderr
        assert not generations.exists()
