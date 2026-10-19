# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Fewer required flags without hidden state: ``train`` defaults ``--run-dir`` to
``runs/<run id>`` and infers ``--provider`` from ``--model``, a bare run id names
``runs/<id>``, ``status`` alone shows the newest run, ``runs`` lists them, and a new
``runs/`` or ``datasets/`` gets a ``.gitignore``. Offline: the fake provider, in a temporary
working directory."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune import providers
from pup_finetune.cli import app
from pup_finetune.runs import RunDir
from pup_finetune.storage import atomic_write_json

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"
GITIGNORE_NOTE = "created runs/.gitignore so trace-derived files stay out of git"

runner = CliRunner()


@pytest.fixture(autouse=True)
def _dev_mode_in_a_new_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")
    monkeypatch.chdir(tmp_path)


def invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def output(result: Result) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.stdout)
    return data


def follow(next_command: str) -> Result:
    words = shlex.split(next_command)
    assert words[:2] == ["pup", "finetune"]
    return invoke(*words[2:])


@pytest.fixture
def dataset() -> Path:
    result = invoke("dataset", "import", "ds", "--file", str(ROWS))
    assert result.exit_code == 0, result.stderr
    return Path("ds")


def created_at(run_id: str, when: str) -> None:
    """Record that the run ``run_id`` under ``runs/`` was created at ``when``, so tests of
    the order runs are listed in never depend on two runs created in the same second."""
    path = Path("runs") / run_id / "run.json"
    manifest = json.loads(path.read_text())
    atomic_write_json(path, {**manifest, "created_at": when})


def train(*extra: str) -> dict[str, Any]:
    """Preview ``train --dataset ds --model fake-model`` with ``extra``, then run its
    next_command; the trained run's result."""
    preview = invoke("train", "--dataset", "ds", "--model", "fake-model", *extra)
    assert preview.exit_code == 2, preview.stderr
    result = follow(output(preview)["next_command"])
    assert result.exit_code == 0, result.stderr
    return output(result)


# --- the default run directory and the inferred provider ---------------------------------


def test_train_needs_only_a_dataset_and_a_model(dataset: Path) -> None:
    result = invoke("train", "--dataset", str(dataset), "--model", "fake-model")
    assert result.exit_code == 2, result.stderr
    preview = output(result)["preview"]
    run_id = preview["run_id"]
    assert preview["run_dir"] == f"runs/{run_id}"
    assert preview["provider"] == "fake"
    assert preview["provider_source"] == "inferred from --model fake-model"
    # The next command pins what was inferred or defaulted, so it trains what was previewed.
    words = shlex.split(output(result)["next_command"])
    assert words[words.index("--provider") + 1] == "fake"
    assert words[words.index("--run-dir") + 1] == f"runs/{run_id}"
    assert words[words.index("--run-id") + 1] == run_id
    assert f"train fake-model on fake in runs/{run_id}" in result.stderr
    # The preview writes nothing.
    assert not Path("runs").exists()


def test_pinning_the_run_directory_and_id_changes_only_where_it_runs(dataset: Path) -> None:
    first = output(invoke("train", "--dataset", str(dataset), "--model", "fake-model"))
    pinned = output(
        invoke(
            "train",
            "--dataset",
            str(dataset),
            "--model",
            "fake-model",
            "--provider",
            "fake",
            "--run-dir",
            "elsewhere",
            "--run-id",
            "other-id",
        )
    )
    for key in ("rows", "settings", "action"):
        assert first["preview"][key] == pinned["preview"][key], key
    assert pinned["preview"]["provider_source"] == "--provider"
    assert pinned["preview"]["run_dir"] == "elsewhere"


def test_the_first_run_creates_runs_with_a_gitignore_and_says_so_once(dataset: Path) -> None:
    del dataset
    first = train()
    assert first["status"] == "completed"
    assert first["run_dir"] == f"runs/{first['run_id']}"
    assert first["notes"] == [GITIGNORE_NOTE]
    assert GITIGNORE_NOTE not in first["warnings"]
    assert Path("runs/.gitignore").read_text() == "*\n"
    # runs/ exists now: the next run says nothing about it.
    second = train()
    assert second["notes"] == []


def test_an_existing_runs_directory_is_left_as_it_is(dataset: Path) -> None:
    del dataset
    Path("runs").mkdir()
    result = train()
    assert result["notes"] == []
    assert not Path("runs/.gitignore").exists()


def test_a_detached_run_notes_the_gitignore_too(dataset: Path) -> None:
    preview = invoke("train", "--dataset", str(dataset), "--model", "fake-model", "--detach")
    started = follow(output(preview)["next_command"])
    assert started.exit_code == 0, started.stderr
    data = output(started)
    assert data["status"] == "started"
    assert data["notes"] == [GITIGNORE_NOTE]
    assert data["warnings"] == []
    watched = invoke("status", data["run_id"], "--watch", "--interval", "0.1", "--timeout", "60")
    assert watched.exit_code == 0, watched.stderr
    assert output(watched)["phase"] == "completed", watched.stderr


@pytest.mark.parametrize("name", ["fake-model", "fake/tokenizer"])
def test_a_name_offered_by_two_providers_needs_provider(
    dataset: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    providers.register_provider(
        "other", "pkg.other:Provider", models={"fake-model": ("fake-model", "fake/tokenizer")}
    )
    result = invoke("train", "--dataset", str(dataset), "--model", name)
    assert result.exit_code == 2
    assert result.stdout == ""
    assert f"Error [usage]: {name} is offered by fake and other; pass --provider" in (result.stderr)
    # Naming the provider settles it.
    named = invoke("train", "--dataset", str(dataset), "--model", name, "--provider", "fake")
    assert named.exit_code == 2
    assert output(named)["status"] == "confirmation_required"


@pytest.mark.parametrize("name", ["fake/fake-model", "fake/tokenizer"])
def test_the_provider_id_or_tokenizer_selects_a_model_and_the_rerun_pins_its_alias(
    dataset: Path, name: str
) -> None:
    preview = output(invoke("train", "--dataset", str(dataset), "--model", name))
    assert preview["preview"]["provider_source"] == f"inferred from --model {name}"
    words = shlex.split(preview["next_command"])
    assert words[words.index("--model") + 1] == "fake-model"
    assert name not in words
    started = follow(preview["next_command"])
    assert started.exit_code == 0, started.stderr
    assert (
        json.loads(Path(output(started)["run_dir"], "run.json").read_text())["profile"]["alias"]
        == "fake-model"
    )


def test_an_unknown_model_lists_the_known_aliases(dataset: Path) -> None:
    result = invoke("train", "--dataset", str(dataset), "--model", "nope")
    assert result.exit_code == 2
    assert "Error [usage]: unknown model 'nope'" in result.stderr
    assert "fake-model (fake)" in result.stderr
    assert "qwen3p8-27b, glm-5p3-flash, kimi-k3" in result.stderr
    assert "muse-glimmer-30b (fireworks)" in result.stderr
    assert "models list`" in result.stderr


def test_a_development_provider_is_not_inferred_outside_development_mode(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PUP_FINETUNE_DEV")
    result = invoke("train", "--dataset", str(dataset), "--model", "fake-model")
    assert result.exit_code == 2
    assert "Error [usage]: unknown model 'fake-model'" in result.stderr
    assert "(fake)" not in result.stderr


# --- an occupied run directory -----------------------------------------------------------


def test_an_occupied_default_run_directory_asks_for_another_run_id(dataset: Path) -> None:
    del dataset
    train("--run-id", "taken")
    result = invoke("train", "--dataset", "ds", "--model", "fake-model", "--run-id", "taken")
    assert result.exit_code == 2
    assert result.stderr.splitlines()[-1] == (
        "Hint: see it with `pup finetune status runs/taken`; to train again, pass another "
        "--run-id (or omit it)"
    )


def test_train_evaluate_notes_the_gitignore_too(dataset: Path) -> None:
    del dataset
    result = train("--evaluate", "--eval-judge-model", "fake-judge")
    assert result["notes"] == [GITIGNORE_NOTE]
    assert GITIGNORE_NOTE not in result["warnings"]
    assert result["evaluation"]["status"] == "complete"


def test_an_occupied_run_directory_says_what_it_holds_and_how_to_go_on(dataset: Path) -> None:
    del dataset
    trained = train("--run-dir", "run1")
    result = invoke("train", "--dataset", "ds", "--model", "fake-model", "--run-dir", "run1")
    assert result.exit_code == 2
    assert result.stderr.splitlines() == [
        f"Error [run_dir_not_empty]: run1 already holds run {trained['run_id']} "
        "(completed, best epoch 5)",
        "Hint: see it with `pup finetune status run1`; to train again, omit --run-dir or "
        "pass a new one",
    ]


# --- referring to runs -------------------------------------------------------------------


def test_a_bare_run_id_names_its_directory_under_runs(dataset: Path) -> None:
    del dataset
    run_id = train()["run_id"]
    status = invoke("status", run_id)
    assert status.exit_code == 0, status.stderr
    assert output(status)["run_dir"] == f"runs/{run_id}"
    assert output(status)["run_dir_source"] == "run id"
    assert f"run id {run_id}: using runs/{run_id}" in status.stderr
    assert output(invoke("status", f"runs/{run_id}"))["run_dir_source"] == "given"
    # Every command that takes a run takes the id, and its next command names the directory.
    evaluate = invoke("evaluate", run_id)
    assert evaluate.exit_code == 2, evaluate.stderr
    assert output(evaluate)["next_command"].startswith(f"pup finetune evaluate runs/{run_id} ")
    assert output(evaluate)["preview"]["run_dir_source"] == "run id"
    for command in ("resume", "cancel", "deploy", "undeploy"):
        result = invoke(command, run_id)
        assert f"run id {run_id}: using runs/{run_id}" in result.stderr, command
        assert "run.json does not exist" not in result.stderr, command
    deploy = output(invoke("deploy", run_id))
    assert deploy["preview"]["run_dir_source"] == "run id"
    # Its next_command names the directory; confirmed by the id, the result says so.
    assert shlex.split(deploy["next_command"])[3] == f"runs/{run_id}"
    deployed = output(invoke("deploy", run_id, "--confirm"))
    assert deployed["run_dir_source"] == "run id"


def test_an_existing_path_wins_over_a_run_id(dataset: Path) -> None:
    del dataset
    train("--run-id", "same", "--run-dir", "same")
    train("--run-id", "same")  # runs/same
    status = invoke("status", "same")
    assert status.exit_code == 0, status.stderr
    assert output(status)["run_dir"] == "same"
    assert "using" not in status.stderr


def test_a_path_that_shadows_a_run_id_names_the_run(dataset: Path) -> None:
    del dataset
    train("--run-id", "same")  # runs/same
    Path("same").mkdir()  # a directory here, not a run
    result = invoke("status", "same")
    assert result.exit_code == 2
    assert "Error [not_a_run_dir]: same is not a training run (it has no run.json)" in (
        result.stderr
    )
    assert "pass runs/same" in result.stderr


def test_an_unknown_run_points_at_the_listing() -> None:
    result = invoke("status", "ft-nothing")
    assert result.exit_code == 2
    assert "Error [not_a_run_dir]: no such run directory or run id: ft-nothing" in result.stderr
    assert "`pup finetune runs`" in result.stderr


def test_status_alone_shows_the_newest_run(dataset: Path) -> None:
    del dataset
    train("--run-id", "newer")
    train("--run-id", "older")  # trained last, but recorded as created first
    created_at("older", "2026-10-01T10:00:00Z")
    created_at("newer", "2026-10-01T11:00:00Z")
    result = invoke("status")
    assert result.exit_code == 0, result.stderr
    assert output(result)["run_id"] == "newer"
    assert output(result)["run_dir_source"] == "newest under ./runs"
    assert "the newest run under runs/: runs/newer" in result.stderr


def test_status_alone_without_runs_is_an_error() -> None:
    result = invoke("status")
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Error [not_a_run_dir]: no run under runs/ to show" in result.stderr


# --- runs ---------------------------------------------------------------------------------


def test_runs_lists_the_runs_newest_first(dataset: Path) -> None:
    del dataset
    train("--run-id", "newer")
    train("--run-id", "older")
    created_at("older", "2026-10-01T10:00:00Z")
    created_at("newer", "2026-10-01T11:00:00Z")
    receipt = RunDir(Path("runs/older")).deployment_path
    atomic_write_json(receipt, {"state": "READY", "deleted_at": None})
    Path("runs/.cache").mkdir()  # not named like a run: left out, silently
    result = invoke("runs")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["dir"] == "runs"
    assert data["count"] == 2
    assert [run["run_id"] for run in data["runs"]] == ["newer", "older"]
    newer, older = data["runs"]
    assert set(newer) == {
        "run_id",
        "phase",
        "model",
        "provider",
        "best_epoch",
        "promoted_model",
        "deployment",
        "run_dir",
        "created_at",
    }
    assert newer["phase"] == "completed"
    assert newer["model"] == "fake-model"
    assert newer["provider"] == "fake"
    assert newer["best_epoch"] == 5
    assert newer["promoted_model"] == "newer"
    assert newer["deployment"] is None
    assert newer["run_dir"] == "runs/newer"
    assert older["deployment"] == "READY"
    assert data["warnings"] == []
    assert data["next_command"] == "pup finetune status runs/newer"
    assert "2 runs under runs; newest newer (completed, fake-model)" in result.stderr


def test_runs_names_run_id_directories_without_run_json(dataset: Path) -> None:
    del dataset
    train("--run-id", "good")
    Path("runs/ft-20261001000000-abcdef").mkdir()
    data = output(invoke("runs"))
    assert [run["run_id"] for run in data["runs"]] == ["good"]
    assert data["warnings"] == [
        "runs/ft-20261001000000-abcdef has no run.json, so it is not listed (a run whose "
        "start failed, or not a run)"
    ]


def test_runs_says_what_is_malformed(dataset: Path) -> None:
    del dataset
    train("--run-id", "good")
    train("--run-id", "nospec")
    train("--run-id", "badmodel")
    manifest = json.loads(Path("runs/nospec/run.json").read_text())
    del manifest["spec"]
    atomic_write_json(Path("runs/nospec/run.json"), manifest)
    result = json.loads(Path("runs/badmodel/result.json").read_text())
    atomic_write_json(Path("runs/badmodel/result.json"), {**result, "promoted_model": "oops"})
    data = output(invoke("runs"))
    assert sorted(run["run_id"] for run in data["runs"]) == ["badmodel", "good"]
    assert "runs/nospec cannot be read: run.json has no spec" in data["warnings"]
    assert (
        "runs/badmodel: result.json's promoted_model is not a model (an object with a "
        "model_id), so none is shown" in data["warnings"]
    )
    assert (
        next(run for run in data["runs"] if run["run_id"] == "badmodel")["promoted_model"] is None
    )


def test_runs_reports_a_run_it_cannot_read(dataset: Path) -> None:
    del dataset
    train("--run-id", "good")
    Path("runs/broken").mkdir()
    Path("runs/broken/run.json").write_text("{not json")
    data = output(invoke("runs"))
    assert [run["run_id"] for run in data["runs"]] == ["good"]
    assert len(data["warnings"]) == 1
    assert data["warnings"][0].startswith("runs/broken cannot be read")


def test_runs_with_none_suggests_training() -> None:
    result = invoke("runs")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["runs"] == []
    assert data["warnings"] == []
    assert data["next_command"] is None
    assert [step["command"] for step in data["next_steps"]] == [
        "pup finetune train --provider <PROVIDER> --dataset <DIR> --model <MODEL>"
    ]


def test_runs_reads_another_directory(dataset: Path) -> None:
    del dataset
    train("--run-id", "kept", "--run-dir", "old/kept")
    data = output(invoke("runs", "--dir", "old"))
    assert data["dir"] == "old"
    assert [run["run_dir"] for run in data["runs"]] == ["old/kept"]
    missing = output(invoke("runs", "--dir", "missing"))
    assert missing["warnings"] == ["missing is not a directory"]


# --- datasets/ -----------------------------------------------------------------------------


def test_the_first_dataset_under_datasets_gets_a_gitignore() -> None:
    first = invoke("dataset", "import", "datasets/one", "--file", str(ROWS))
    assert first.exit_code == 0, first.stderr
    note = "created datasets/.gitignore so trace-derived files stay out of git"
    assert output(first)["notes"] == [note]
    assert f"note: {note}" in first.stderr
    assert Path("datasets/.gitignore").read_text() == "*\n"
    second = invoke("dataset", "import", "datasets/two", "--file", str(ROWS))
    assert output(second)["notes"] == []
    # Elsewhere, nothing is added.
    other = invoke("dataset", "import", "mine/ds", "--file", str(ROWS))
    assert not Path("mine/.gitignore").exists()
    assert output(other)["notes"] == []
    assert all("gitignore" not in text for text in output(other)["warnings"])
