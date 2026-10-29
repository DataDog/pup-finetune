# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Every paid or remote-changing command, through the CLI, against the one approval rule.

``--confirm`` is the user's approval: without it a command only previews. ``PUP_READ_ONLY``
refuses the ``--confirm`` of a paid run or a remote write, and a read-only preview offers no
``--confirm``, unless nothing the command does leaves the machine (the offline fake provider
and judge). ``dataset triage`` writes nothing to Datadog, so read-only mode does not refuse
it. Each command runs offline and not where it has an offline mode, on the in-memory
Fireworks, a fake pup and a fake judge client, so nothing reaches the network.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune import judges
from pup_finetune.cli import app
from pup_finetune.cli import dataset as dataset_cli
from pup_finetune.cli import deploy as deploy_module
from pup_finetune.cli import train as train_module
from pup_finetune.cli import triage as triage_cli
from pup_finetune.data.push import PUSH_FILE
from pup_finetune.evaluation.fake_judge import FakeJudge
from pup_finetune.inference import ChatCompleter, ChatCompletion, JudgeConfig
from pup_finetune.providers import get_provider as real_get_provider
from pup_finetune.providers.base import Provider
from pup_finetune.providers.fireworks.provider import FireworksProvider
from pup_finetune.runs import RunDir
from pup_finetune.runtime import Runtime
from pup_finetune.training import inputs as inputs_module
from tests.conftest import JUDGE_KEYS
from tests.fakes.chat_api import FakeChatApi
from tests.fakes.datasets import FakeDatasetsApi
from tests.fakes.fireworks import FakeFireworksTrainingClient
from tests.fakes.fireworks_api import API_KEY, FakeFireworksApi
from tests.fakes.fireworks_deploy import FakeClock, FakeSdkFireworks, fake_deployer
from tests.fakes.pup import FakePup
from tests.fakes.pushed import mark_pushed
from tests.fakes.tokenizer import FakeChatTokenizer

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"
PAID_JUDGE = "openai/gpt-5.5"

runner = CliRunner()


def _invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def _ok(*args: str) -> None:
    result = _invoke(*args)
    assert result.exit_code == 0, result.stderr


class Keeper:
    """A triage judge that keeps every row."""

    def complete(self, messages: Sequence[Mapping[str, Any]], **options: Any) -> ChatCompletion:
        return ChatCompletion(message={"role": "assistant", "content": KEEP})


KEEP = '{"keep": true, "reason": "on topic"}'


class Fireworks:
    """The in-memory Fireworks: training (with failures to inject) and deployments."""

    def __init__(self) -> None:
        self.training = FakeFireworksTrainingClient()
        self.sdk = FakeSdkFireworks()
        self.chat = FakeChatApi(api_key=API_KEY)
        self.clock = FakeClock()

    def provider(self, runtime: Runtime) -> FireworksProvider:
        return FireworksProvider(
            runtime,
            env={"FIREWORKS_API_KEY": API_KEY},
            transport=FakeFireworksApi().transport,
            tokenizer=FakeChatTokenizer(),
            training_client=lambda _key: self.training,
            deployer=fake_deployer(self.sdk, self.chat, self.clock),
        )


@dataclass(frozen=True)
class World:
    """Per test: copies of a pushed dataset, an unpushed one, a completed fake run and an
    interrupted one, a scratch directory, and the fakes paid commands reach."""

    dataset: Path
    unpushed: Path
    run: Path
    interrupted: Path
    tmp: Path
    fireworks: Fireworks

    def train(self, provider: str, model: str, run_dir: Path, *extra: str) -> list[str]:
        return [
            "train",
            *("--provider", provider, "--model", model),
            *("--dataset", str(self.dataset), "--run-dir", str(run_dir)),
            *("--max-epochs", "1", "--batch-size", "2"),
            *extra,
        ]

    def fireworks_run(self) -> Path:
        """A run trained on the in-memory Fireworks; a failed training exits 1."""
        run_dir = self.tmp / "fw-run"
        result = _invoke(*self.train("fireworks", "qwen3p8-27b", run_dir, "--confirm"))
        assert result.exit_code in (0, 1), result.stderr
        return run_dir


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """A dataset, a completed fake run and an interrupted one, built once for the module."""
    root = tmp_path_factory.mktemp("approval-gate")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PUP_FINETUNE_DEV", "1")
        _ok("dataset", "import", str(root / "ds"), "--file", str(ROWS))
        shutil.copytree(root / "ds", root / "unpushed")
        mark_pushed(root / "ds")
        fake = ["train", "--model", "fake-model", "--dataset", str(root / "ds")]
        _ok(*fake, "--run-dir", str(root / "run"), "--max-epochs", "1", "--confirm")
        crashed = _invoke(
            *fake,
            *("--run-dir", str(root / "interrupted")),
            *("--provider-option", "fail_at_step=2", "--confirm"),
        )
        assert crashed.exit_code == 1, crashed.stderr
    yield root


@pytest.fixture
def world(template: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")
    monkeypatch.setenv("OPENAI_API_KEY", JUDGE_KEYS["OPENAI_API_KEY"])
    for name in ("ds", "unpushed", "run", "interrupted"):
        shutil.copytree(template / name, tmp_path / name)
    fireworks = Fireworks()

    def get_provider(name: str, runtime: Runtime) -> Provider:
        if name == "fireworks":
            return fireworks.provider(runtime)
        return real_get_provider(name, runtime)

    for module in (train_module, inputs_module, deploy_module):
        monkeypatch.setattr(module, "get_provider", get_provider)
    pup = FakePup()
    FakeDatasetsApi().install(pup)
    monkeypatch.setattr(dataset_cli, "Pup", pup.client)
    return World(
        dataset=tmp_path / "ds",
        unpushed=tmp_path / "unpushed",
        run=tmp_path / "run",
        interrupted=tmp_path / "interrupted",
        tmp=tmp_path,
        fireworks=fireworks,
    )


@pytest.fixture
def judge_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer every paid judge here: an evaluation's as the offline judge does, triage's
    keeping every row."""

    @contextmanager
    def open_client(config: JudgeConfig, **options: Any) -> Iterator[ChatCompleter]:
        yield FakeJudge()

    @contextmanager
    def open_clients(specs: Sequence[str]) -> Iterator[Mapping[str, ChatCompleter]]:
        yield {"openai": Keeper()}

    monkeypatch.setattr(judges, "open_client", open_client)
    monkeypatch.setattr(triage_cli, "open_clients", open_clients)


# --- the gated commands ------------------------------------------------------------------


@dataclass(frozen=True)
class Gated:
    """A gated command: ``setup`` prepares what it acts on (read-only off) and returns its
    words without ``--confirm``; ``ran`` reads back whether ``--confirm`` did the work."""

    name: str
    setup: Callable[[World], list[str]]
    ran: Callable[[World], bool]
    offline: bool | None
    """Whether nothing it does leaves the machine; ``None`` when that never happens."""
    read_only_refuses: bool = True


def _phase(run_dir: Path) -> str:
    return str(json.loads((run_dir / "state.json").read_text())["phase"])


def _deployed(run_dir: Path) -> bool:
    receipt = RunDir(run_dir).read_deployment()
    return receipt is not None and receipt.get("state") == "READY" and not receipt["deleted_at"]


def _deleted(run_dir: Path) -> bool:
    receipt = RunDir(run_dir).read_deployment()
    return receipt is not None and bool(receipt["deleted_at"])


def _deploy_then(run_dir: Path) -> list[str]:
    _ok("deploy", str(run_dir), "--confirm")
    return ["undeploy", str(run_dir)]


def _crashed_fireworks_run(w: World) -> list[str]:
    w.fireworks.training.fail("forward_backward", "server", after=1)
    run_dir = w.fireworks_run()
    assert _phase(run_dir) == "interrupted"
    return ["resume", str(run_dir)]


def _triage(w: World) -> list[str]:
    return ["dataset", "triage", str(w.dataset), "--rule", "stay on topic", "--judges", PAID_JUDGE]


GATED: list[Gated] = [
    Gated(
        "train",
        lambda w: w.train("fake", "fake-model", w.tmp / "new-run"),
        lambda w: (w.tmp / "new-run" / "result.json").exists(),
        offline=True,
    ),
    Gated(
        "train",
        lambda w: w.train("fireworks", "qwen3p8-27b", w.tmp / "new-run"),
        lambda w: (w.tmp / "new-run" / "result.json").exists(),
        offline=False,
    ),
    Gated(
        "resume",
        lambda w: ["resume", str(w.interrupted)],
        lambda w: _phase(w.interrupted) == "completed",
        offline=True,
    ),
    Gated(
        "resume",
        _crashed_fireworks_run,
        lambda w: _phase(w.tmp / "fw-run") == "completed",
        offline=False,
    ),
    Gated(
        "evaluate",
        lambda w: ["evaluate", str(w.run), "--no-publish"],
        lambda w: (w.run / "eval" / "summary.json").exists(),
        offline=True,
    ),
    Gated(
        "evaluate",
        lambda w: ["evaluate", str(w.run), "--no-publish", "--judge-model", PAID_JUDGE],
        lambda w: (w.run / "eval" / "summary.json").exists(),
        offline=False,
    ),
    Gated(
        "benchmark",
        lambda w: ["benchmark", str(w.dataset), "--models", "fake-model"],
        lambda w: (w.dataset / "benchmark" / "fake-model").exists(),
        offline=True,
    ),
    Gated(
        "benchmark",
        lambda w: [
            *("benchmark", str(w.dataset)),
            *("--models", "fake-model", "--judge-model", PAID_JUDGE),
        ],
        lambda w: (w.dataset / "benchmark" / "fake-model").exists(),
        offline=False,
    ),
    Gated(
        "deploy",
        lambda w: ["deploy", str(w.run)],
        lambda w: _deployed(w.run),
        offline=True,
    ),
    Gated(
        "deploy",
        lambda w: ["deploy", str(w.fireworks_run())],
        lambda w: _deployed(w.tmp / "fw-run"),
        offline=False,
    ),
    Gated(
        "undeploy",
        lambda w: _deploy_then(w.run),
        lambda w: _deleted(w.run),
        offline=True,
    ),
    Gated(
        "undeploy",
        lambda w: _deploy_then(w.fireworks_run()),
        lambda w: _deleted(w.tmp / "fw-run"),
        offline=False,
    ),
    Gated(
        "dataset push",
        lambda w: ["dataset", "push", str(w.unpushed), "--name", "synthetic-gate"],
        lambda w: (w.unpushed / PUSH_FILE).exists(),
        offline=None,
    ),
    Gated(
        "dataset triage",
        _triage,
        lambda w: (w.dataset / "labels.jsonl").exists(),
        offline=None,
        read_only_refuses=False,
    ),
]


def _id(gated: Gated) -> str:
    if gated.offline is None:
        return gated.name
    return f"{gated.name} ({'offline' if gated.offline else 'paid'})"


@pytest.mark.usefixtures("judge_clients")
@pytest.mark.parametrize("read_only", [False, True], ids=["writable", "read-only"])
@pytest.mark.parametrize("confirm", [False, True], ids=["preview", "confirm"])
@pytest.mark.parametrize("gated", GATED, ids=_id)
def test_a_gated_command_runs_only_when_approved_and_allowed(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
    gated: Gated,
    confirm: bool,
    read_only: bool,
) -> None:
    words = gated.setup(world)
    if read_only:
        monkeypatch.setenv("PUP_READ_ONLY", "1")
    refused = read_only and gated.read_only_refuses and not gated.offline

    result = _invoke(*words, *(["--confirm"] if confirm else []))

    if not confirm:
        assert result.exit_code == 2, result.stderr
        data = json.loads(result.stdout)
        assert data["status"] == "confirmation_required"
        # A preview never offers a --confirm read-only mode would refuse.
        offered = data["next_command"]
        assert (offered is None) is refused, offered
        if not refused:
            assert offered.endswith(" --confirm")
        assert ("read_only" in data["preview"]) is refused
        if gated.read_only_refuses:
            assert data["preview"]["offline"] is bool(gated.offline)
        assert not gated.ran(world)
    elif refused:
        assert result.exit_code == 2, result.stderr
        assert result.stdout == ""
        errors = [line for line in result.stderr.splitlines() if line.startswith("Error ")]
        assert len(errors) == 1, result.stderr
        assert errors[0].startswith("Error [read_only]: "), result.stderr
        assert not gated.ran(world)
    else:
        assert result.exit_code == 0, result.stderr
        assert gated.ran(world)
