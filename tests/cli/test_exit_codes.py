# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""One exit-code rule, checked on common mistakes made through the CLI.

Exit 2 when the user can fix it by changing the command or the directory; exit 1 when
something outside failed or local state is broken. Each failure's stderr starts with
``Error [<code>]: `` and stdout stays empty (a preview's JSON aside). Every case runs offline,
on the fake provider, the fake judge, a fake pup or the in-memory Fireworks.
"""

from __future__ import annotations

import contextlib
import json
import shutil
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune.cli import app
from pup_finetune.cli import dataset as dataset_cli
from pup_finetune.cli import train as train_module
from pup_finetune.datadog.spans import EXPORT_PATH
from pup_finetune.errors import ERROR_CODES
from pup_finetune.providers.base import Provider
from pup_finetune.providers.fireworks.provider import FireworksProvider
from pup_finetune.runtime import Runtime
from pup_finetune.storage import DirectoryLock, atomic_write_json
from pup_finetune.training import inputs as inputs_module
from tests.fakes.fireworks import FakeFireworksTrainingClient
from tests.fakes.fireworks_api import API_KEY, FakeFireworksApi
from tests.fakes.pup import FakePup
from tests.fakes.tokenizer import FakeChatTokenizer

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"

runner = CliRunner()


@dataclass(frozen=True)
class World:
    """A dataset, a completed fake run (evaluated once) and a scratch directory, per test."""

    dataset: Path
    run: Path
    tmp: Path
    monkeypatch: pytest.MonkeyPatch
    held: contextlib.ExitStack
    """What the case holds while the command runs (a lock), released after the test."""

    def train(self, *extra: str, run_dir: Path | None = None) -> list[str]:
        """``train`` on the fake provider into a new run directory, with ``extra``."""
        target = run_dir if run_dir is not None else self.tmp / "new-run"
        return [
            "train",
            *("--provider", "fake", "--model", "fake-model"),
            *("--dataset", str(self.dataset), "--run-dir", str(target)),
            *("--max-epochs", "1"),
            *extra,
        ]

    def set_state(self, **fields: Any) -> None:
        path = self.run / "state.json"
        atomic_write_json(path, {**json.loads(path.read_text()), **fields})


def _invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Path, Path]]:
    """A dataset and a completed, evaluated fake run, built once for the module."""
    root = tmp_path_factory.mktemp("exit-codes")
    dataset, run = root / "ds", root / "run"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PUP_FINETUNE_DEV", "1")
        imported = _invoke("dataset", "import", str(dataset), "--file", str(ROWS))
        assert imported.exit_code == 0, imported.stderr
        trained = _invoke(
            "train",
            *("--provider", "fake", "--model", "fake-model"),
            *("--dataset", str(dataset), "--run-dir", str(run), "--max-epochs", "1"),
            "--confirm",
        )
        assert trained.exit_code == 0, trained.stderr
        evaluated = _invoke("evaluate", str(run), "--no-publish", "--confirm")
        assert evaluated.exit_code == 0, evaluated.stderr
    yield dataset, run


@pytest.fixture
def world(
    template: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[World]:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")
    dataset, run = template
    copy = tmp_path / "run"
    shutil.copytree(run, copy)  # cases may change the run; the dataset is only read
    with contextlib.ExitStack() as held:
        yield World(dataset=dataset, run=copy, tmp=tmp_path, monkeypatch=monkeypatch, held=held)


# --- the mistakes -------------------------------------------------------------------------


def _wrong_judge_settings(w: World) -> list[str]:
    return ["evaluate", str(w.run), "--no-publish", "--max-tokens", "64"]


def _still_training(command: str) -> Callable[[World], list[str]]:
    def setup(w: World) -> list[str]:
        w.set_state(phase="training")
        return [command, str(w.run)]

    return setup


def _read_only_deploy(w: World) -> list[str]:
    w.monkeypatch.setenv("PUP_READ_ONLY", "1")
    return ["deploy", str(w.run), "--confirm"]


def _corrupt_state(w: World) -> list[str]:
    (w.run / "state.json").write_text("{not json")
    return ["status", str(w.run)]


def _locked_dataset(w: World) -> list[str]:
    copy = w.tmp / "ds"
    shutil.copytree(w.dataset, copy)
    # As if another process held it: the lock is not reentrant, even in-process.
    w.held.enter_context(DirectoryLock(copy))
    return ["dataset", "import", str(copy), "--file", str(ROWS)]


def _pup_fails(w: World) -> list[str]:
    fake = FakePup().on("api", EXPORT_PATH, returncode=1, stderr="Error: 500 Internal Error")
    w.monkeypatch.setattr(dataset_cli, "Pup", fake.client)
    return ["dataset", "preview", "--ml-app", "synthetic-app"]


def _pup_interrupted(w: World) -> list[str]:
    """Ctrl-C while pup answers: what a command that has no graceful stop of its own gets."""

    def interrupt(_call: object) -> object:
        raise KeyboardInterrupt

    fake = FakePup().on("api", EXPORT_PATH, handler=interrupt)
    w.monkeypatch.setattr(dataset_cli, "Pup", fake.client)
    return ["dataset", "preview", "--ml-app", "synthetic-app"]


def _fireworks(failure: str) -> Callable[[World], list[str]]:
    """Train on the in-memory Fireworks, whose session start fails with ``failure``."""

    def setup(w: World) -> list[str]:
        fw = FakeFireworksTrainingClient()
        fw.fail("create_lora_training_client", failure)  # type: ignore[arg-type]

        def get_provider(name: str, runtime: Runtime) -> Provider:
            return FireworksProvider(
                runtime,
                env={"FIREWORKS_API_KEY": API_KEY},
                transport=FakeFireworksApi().transport,
                tokenizer=FakeChatTokenizer(),
                training_client=lambda _key: fw,
            )

        w.monkeypatch.setattr(train_module, "get_provider", get_provider)
        w.monkeypatch.setattr(inputs_module, "get_provider", get_provider)
        return [
            "train",
            *("--provider", "fireworks", "--model", "qwen3p8-27b"),
            *("--dataset", str(w.dataset), "--run-dir", str(w.tmp / "fw-run")),
            *("--max-epochs", "1", "--confirm"),
        ]

    return setup


Setup = Callable[[World], Sequence[str]]

MISTAKES: list[tuple[str, Setup, int, str]] = [
    # Usage: Click's errors, in our format.
    ("train without its options", lambda w: ["train"], 2, "usage"),
    ("a mistyped option", lambda w: ["train", "--conf"], 2, "usage"),
    ("a mistyped command", lambda w: ["evalute"], 2, "usage"),
    ("a malformed value", lambda w: w.train("--max-epochs", "many"), 2, "usage"),
    ("a flag without the one it needs", lambda w: w.train("--eval-max-tokens", "9"), 2, "usage"),
    # A preview is exit 2 too: nothing ran.
    ("train without --confirm", lambda w: w.train(), 2, "confirmation_required"),
    ("a remote write under PUP_READ_ONLY", _read_only_deploy, 2, "read_only"),
    # The wrong, missing or used directory.
    ("status of a missing run", lambda w: ["status", str(w.tmp / "nope")], 2, "not_a_run_dir"),
    ("evaluate a dataset", lambda w: ["evaluate", str(w.dataset)], 2, "not_a_run_dir"),
    (
        "show a missing dataset",
        lambda w: ["dataset", "show", str(w.tmp / "nope")],
        2,
        "not_a_dataset_dir",
    ),
    ("sample a run", lambda w: ["dataset", "sample", str(w.run)], 2, "not_a_dataset_dir"),
    ("train into a used run directory", lambda w: w.train(run_dir=w.run), 2, "run_dir_not_empty"),
    ("train into a dataset", lambda w: w.train(run_dir=w.dataset), 2, "run_dir_not_empty"),
    # Frozen settings.
    (
        "import with another seed",
        lambda w: ["dataset", "import", str(w.dataset), "--file", str(ROWS), "--seed", "7"],
        2,
        "settings_conflict",
    ),
    ("evaluate again with other settings", _wrong_judge_settings, 2, "settings_conflict"),
    # The wrong phase.
    ("resume a completed run", lambda w: ["resume", str(w.run), "--confirm"], 2, "wrong_phase"),
    ("cancel a completed run", lambda w: ["cancel", str(w.run), "--confirm"], 2, "wrong_phase"),
    ("undeploy what was never deployed", lambda w: ["undeploy", str(w.run)], 2, "wrong_phase"),
    ("deploy a run still training", _still_training("deploy"), 2, "wrong_phase"),
    ("evaluate a run still training", _still_training("evaluate"), 2, "wrong_phase"),
    # Something outside failed, or local state is broken: exit 1.
    ("a dataset another process is using", _locked_dataset, 1, "lock_busy"),
    ("an unreadable state.json", _corrupt_state, 1, "state"),
    ("pup fails", _pup_fails, 1, "external"),
    (
        "the provider crashes mid-run",
        lambda w: w.train("--provider-option", "fail_at_step=1", "--confirm"),
        1,
        "external",
    ),
    ("the provider has no capacity", _fireworks("capacity"), 1, "provider_capacity"),
    ("the provider rejects the key", _fireworks("auth"), 1, "provider_auth"),
    ("the provider drops the connection", _fireworks("connection"), 1, "transient"),
    ("Ctrl-C", _pup_interrupted, 1, "interrupted"),
]


@pytest.mark.parametrize(
    ("setup", "exit_code", "code"),
    [pytest.param(setup, exit_code, code, id=name) for name, setup, exit_code, code in MISTAKES],
)
def test_a_mistake_exits_by_the_rule_with_its_code(
    world: World, setup: Setup, exit_code: int, code: str
) -> None:
    result = _invoke(*setup(world))
    assert result.exit_code == exit_code, result.stderr
    assert ERROR_CODES[code][0] == exit_code  # the documented exit code of the code
    if code == "confirmation_required":
        assert json.loads(result.stdout)["status"] == "confirmation_required"
        assert result.stderr.startswith("Preview only, nothing was run")
        return
    assert result.stdout == ""
    # Progress may come first (training reports its start); the error line is the only one.
    errors = [line for line in result.stderr.splitlines() if line.startswith("Error ")]
    assert len(errors) == 1, result.stderr
    assert errors[0].startswith(f"Error [{code}]: "), result.stderr


def test_every_missing_required_option_is_named_at_once(world: World) -> None:
    result = _invoke("train", "--provider", "fake")
    assert result.stderr == (
        "Error [usage]: train needs --dataset, --model and --run-dir\n"
        "Hint: see `pup-finetune help train`\n"
    )
