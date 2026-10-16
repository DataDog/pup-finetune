# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The default judges follow the credentials that resolve offline: Anthropic, then OpenAI,
then Fireworks. A frozen judge (``evaluation.json``, ``triage.json``) outlives a new key.

Every key here is a fake one, in the environment or an in-memory keychain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import keyring.core
import pytest
from typer.testing import CliRunner, Result

from pup_finetune import credentials, readiness
from pup_finetune.cli import app
from pup_finetune.curation.triage import SETTINGS_FILE, default_council
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.evaluation import runner as runner_module
from pup_finetune.judge_models import (
    DEFAULT_JUDGE_MODEL,
    EVALUATE_DEFAULTS,
    FIREWORKS_JUDGES,
    TRIAGE_DEFAULTS,
)
from pup_finetune.judges import default_judges
from pup_finetune.runtime import Runtime
from tests.fakes.keyring import CRASHED_HOLDING, CrashingKeyring, LockedKeyring, MemoryKeyring
from tests.fakes.setup import ALL_PACKAGES, installed_packages

ROWS = Path(__file__).resolve().parent / "fixtures" / "rows" / "tool_use.jsonl"
ANTHROPIC_KEY = "sk-ant-synthetic-0000-test-key"
OPENAI_KEY = "sk-synthetic-openai-0000-test-key"

runner = CliRunner()

HAIKU, SONNET = "anthropic/claude-haiku-5-5", "anthropic/claude-sonnet-5-5"
MINI, ASTRA = "openai/gpt-5.4-mini", "openai/gpt-6-astra"


def invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def output(result: Result) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.stdout)
    return data


@pytest.mark.parametrize(
    ("env", "evaluate", "triage"),
    [
        ({}, (DEFAULT_JUDGE_MODEL,), FIREWORKS_JUDGES),
        ({"ANTHROPIC_API_KEY": ANTHROPIC_KEY}, (HAIKU,), (HAIKU, SONNET)),
        ({"OPENAI_API_KEY": OPENAI_KEY}, (MINI,), (MINI, ASTRA)),
        (
            {"ANTHROPIC_API_KEY": ANTHROPIC_KEY, "OPENAI_API_KEY": OPENAI_KEY},
            (HAIKU,),
            (HAIKU, MINI),
        ),
    ],
)
def test_the_defaults_follow_the_keys_in_the_environment(
    env: dict[str, str], evaluate: tuple[str, ...], triage: tuple[str, ...]
) -> None:
    assert default_judges(EVALUATE_DEFAULTS, env=env)[0] == evaluate
    assert default_judges(TRIAGE_DEFAULTS, env=env)[0] == triage


def test_each_source_is_named() -> None:
    env = {"ANTHROPIC_API_KEY": ANTHROPIC_KEY, "OPENAI_API_KEY": OPENAI_KEY}
    assert default_judges(TRIAGE_DEFAULTS, env=env)[1] == (
        "default: Anthropic credential (ANTHROPIC_API_KEY) and OpenAI credential (OPENAI_API_KEY)"
    )
    assert default_judges(TRIAGE_DEFAULTS, env={})[1] == (
        "default: no Anthropic or OpenAI credential found"
    )


def test_a_key_in_the_keychain_and_the_anthropic_sdk_login_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = MemoryKeyring()
    memory.set_password("pup-finetune", "openai", OPENAI_KEY)
    monkeypatch.setattr(keyring.core, "_keyring_backend", memory)
    judges, source = default_judges(EVALUATE_DEFAULTS, env={})
    assert (judges, source) == ((MINI,), "default: OpenAI credential (keychain)")
    sdk = {"ANTHROPIC_AUTH_TOKEN": "synthetic-token"}
    assert default_judges(EVALUATE_DEFAULTS, env=sdk) == (
        (HAIKU,),
        "default: Anthropic credential (Anthropic SDK login)",
    )


@pytest.mark.parametrize("backend", [LockedKeyring, CrashingKeyring])
def test_a_keychain_that_cannot_be_read_falls_through_to_the_next_default_as_in_doctor(
    monkeypatch: pytest.MonkeyPatch, backend: type[MemoryKeyring | LockedKeyring]
) -> None:
    monkeypatch.setattr(keyring.core, "_keyring_backend", backend())
    assert default_judges(EVALUATE_DEFAULTS, env={})[0] == (DEFAULT_JUDGE_MODEL,)
    assert default_judges(TRIAGE_DEFAULTS, env={"OPENAI_API_KEY": OPENAI_KEY})[0] == (MINI, ASTRA)
    assert not readiness.key_check(credentials.get_spec("anthropic"), required=False).ok
    fireworks = readiness.key_check(credentials.get_spec("fireworks"), required=False)
    ready = readiness.ready_map([*readiness.judge_checks(Runtime()), fireworks], Runtime())
    assert ready["dataset triage"]["blocked_by"] == ["fireworks:key"]


def test_an_anthropic_key_without_its_sdk_is_passed_over_and_said_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed_packages(monkeypatch, ALL_PACKAGES - {"anthropic"})
    unused = (
        "; the Anthropic credential is not used, as Anthropic judges need the 'anthropic' "
        "package, which is not installed"
    )
    env = {"ANTHROPIC_API_KEY": ANTHROPIC_KEY}
    assert default_judges(EVALUATE_DEFAULTS, env=env) == (
        (DEFAULT_JUDGE_MODEL,),
        "default: no Anthropic or OpenAI judge available" + unused,
    )
    both = {**env, "OPENAI_API_KEY": OPENAI_KEY}
    assert default_judges(TRIAGE_DEFAULTS, env=both) == (
        (MINI, ASTRA),
        "default: OpenAI credential (OPENAI_API_KEY)" + unused,
    )


def test_an_evaluate_preview_without_the_anthropic_sdk_keeps_the_fireworks_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed_packages(monkeypatch, ALL_PACKAGES - {"anthropic"})
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
    config, source = runner_module.resolve_judge(None, provider="fireworks", runtime=Runtime())
    assert config.spec == DEFAULT_JUDGE_MODEL
    assert source.startswith("default: no Anthropic or OpenAI judge available")


def test_evaluate_resolves_the_default_and_keeps_a_frozen_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
    config, source = runner_module.resolve_judge(None, provider="fireworks", runtime=Runtime())
    assert (config.spec, source) == (HAIKU, "default: Anthropic credential (ANTHROPIC_API_KEY)")
    frozen, source = runner_module.resolve_judge(
        None, provider="fireworks", runtime=Runtime(), stored=DEFAULT_JUDGE_MODEL
    )
    assert (frozen.spec, source) == (DEFAULT_JUDGE_MODEL, "evaluation directory")
    given, source = runner_module.resolve_judge(
        MINI, provider="fireworks", runtime=Runtime(), stored=DEFAULT_JUDGE_MODEL
    )
    assert (given.spec, source) == (MINI, "given")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ds"
    assert invoke("dataset", "import", str(path), "--file", str(ROWS)).exit_code == 0
    return path


def test_a_triage_preview_names_the_default_council_and_where_it_came_from(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    preview = output(invoke("dataset", "triage", str(dataset), "--rule", "stay on topic"))
    assert [judge["model"] for judge in preview["preview"]["judges"]] == [MINI, ASTRA]
    assert preview["preview"]["judges_source"] == "default: OpenAI credential (OPENAI_API_KEY)"
    assert f"--judges {MINI},{ASTRA} --confirm" in preview["next_command"]


@pytest.mark.usefixtures("judge_keys")
def test_a_triage_preview_on_a_crashing_keychain_names_the_fireworks_council(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(keyring.core, "_keyring_backend", CrashingKeyring())
    result = invoke("dataset", "triage", str(dataset), "--rule", "stay on topic")
    assert result.exit_code == 2, result.stderr
    preview = output(result)["preview"]
    assert [judge["model"] for judge in preview["judges"]] == list(FIREWORKS_JUDGES)
    assert preview["judges_source"] == "default: no Anthropic or OpenAI credential found"
    assert CRASHED_HOLDING not in result.stdout + result.stderr


@pytest.mark.usefixtures("judge_keys")
def test_a_triage_rerun_keeps_the_council_triage_json_froze(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (dataset / SETTINGS_FILE).write_text(json.dumps({"judges": list(FIREWORKS_JUDGES)}))
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
    assert default_council(DatasetDir(dataset)) == (FIREWORKS_JUDGES, "triage.json")
    assert default_council(DatasetDir(dataset), reset=True)[0] == (HAIKU, SONNET)
    preview = output(invoke("dataset", "triage", str(dataset), "--rule", "x"))
    assert preview["preview"]["judges_source"] == "triage.json"
