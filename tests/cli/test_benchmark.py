# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`benchmark` end to end on the offline fake provider and fake judge; nothing leaves the
machine."""

from __future__ import annotations

import dataclasses
import json
import shlex
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune import providers
from pup_finetune.cli import app
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.evaluation import benchmark as benchmark_module
from pup_finetune.inference import ChatCompletion
from pup_finetune.providers import fake as fake_module
from pup_finetune.providers.base import ModelProfile, Sampler
from pup_finetune.providers.fake import FakeRenderer
from pup_finetune.rendering.base import ChatMessage, RendererError, RendererErrorCode, ToolSpec
from tests.fakes.rows import import_with_opening_assistant_test_row, import_with_refused_test_row

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"

runner = CliRunner()


@pytest.fixture(autouse=True)
def _dev_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")


def invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def output(result: Result) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.stdout)
    return data


def rerun(next_command: str) -> Result:
    words = shlex.split(next_command)
    assert words[:2] == ["pup", "finetune"]
    return invoke(*words[2:])


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ds"
    result = invoke("dataset", "import", str(path), "--file", str(ROWS))
    assert result.exit_code == 0, result.stderr
    return path


def test_preview_lists_the_calls_and_runs_nothing(dataset: Path) -> None:
    result = invoke("benchmark", str(dataset), "--models", "fake-model")

    assert result.exit_code == 2, result.stderr
    data = output(result)
    preview = data["preview"]
    assert preview["models"][0]["model"] == "fake-model"
    assert preview["generation_calls"] == preview["cases"]
    assert preview["judge_calls"] == preview["cases"] + preview["calibration_calls"]
    assert "target_pass_rate" in preview["defaults_used"]
    assert preview["data_recipients"]
    assert data["next_command"].endswith("--judge-model fake-judge --confirm")
    assert not (dataset / "benchmark").exists()


def test_confirm_ranks_recommends_and_leads_to_training(dataset: Path) -> None:
    preview = output(invoke("benchmark", str(dataset), "--models", "fake-model"))

    result = rerun(preview["next_command"])

    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "complete"
    [entry] = data["leaderboard"]
    assert entry["model"] == "fake-model"
    assert entry["cases"] == preview["preview"]["cases"]
    assert data["recommendation"]["verdict"] == "fine_tune"
    assert data["recommendation"]["model"] == "fake-model"
    assert data["next_command"] == (
        f"pup finetune train --provider fake --dataset {dataset} --model fake-model"
    )
    assert (dataset / "benchmark" / "benchmark.json").is_file()
    assert (dataset / "benchmark" / "fake-model" / "results.jsonl").is_file()
    assert "recommended: fine-tune fake-model" in result.stderr


def test_rerun_reuses_every_result(dataset: Path) -> None:
    args = ("benchmark", str(dataset), "--models", "fake-model", "--confirm")
    assert invoke(*args).exit_code == 0
    generations = dataset / "benchmark" / "fake-model" / "generations.jsonl"
    before = generations.read_text()

    preview = output(invoke("benchmark", str(dataset), "--models", "fake-model"))["preview"]
    again = invoke(*args)

    assert preview["generation_calls"] == 0
    assert preview["calibration_calls"] == 0
    assert again.exit_code == 0, again.stderr
    assert generations.read_text() == before


def test_a_target_the_base_model_meets_needs_no_training(dataset: Path) -> None:
    result = invoke(
        "benchmark", str(dataset), "--models", "fake-model", "--target-pass-rate", "0", "--confirm"
    )

    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["recommendation"]["verdict"] == "base_meets_target"
    assert data["next_command"] is None
    assert data["next_steps"][0]["command"].startswith("pup finetune train --provider fake")


def test_other_settings_in_the_same_out_dir_are_refused_at_preview(dataset: Path) -> None:
    assert invoke("benchmark", str(dataset), "--models", "fake-model", "--confirm").exit_code == 0

    result = invoke("benchmark", str(dataset), "--models", "fake-model", "--max-tokens", "16")

    assert result.exit_code == 2
    assert "settings_conflict" in result.stderr
    assert "--out-dir" in result.stderr


@pytest.mark.parametrize(
    ("models", "message"),
    [("fake-model,fake-model", "twice"), ("no-such-model", "unknown model"), (",", "no model")],
)
def test_bad_models_are_usage_errors(dataset: Path, models: str, message: str) -> None:
    result = invoke("benchmark", str(dataset), "--models", models)

    assert result.exit_code == 2
    assert message in result.stderr


def test_a_whole_entry_that_names_a_model_is_never_split(dataset: Path) -> None:
    result = invoke("benchmark", str(dataset), "--models", "fake/tokenizer")

    assert result.exit_code == 2, result.stderr
    [model] = output(result)["preview"]["models"]
    assert (model["model"], model["provider"]) == ("fake-model", "fake")


def test_a_provider_prefix_takes_only_that_provider_s_models(dataset: Path) -> None:
    result = invoke("benchmark", str(dataset), "--models", "fake/no-such-model")

    assert result.exit_code == 2
    assert "fake offers no model 'no-such-model'" in result.stderr
    assert "models list --provider fake" in result.stderr


def test_a_model_several_providers_offer_needs_its_provider_named(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    providers.register_provider(
        "other", "pkg.other:Provider", models={"fake-model": ("fake-model",)}
    )

    ambiguous = invoke("benchmark", str(dataset), "--models", "fake-model")
    assert ambiguous.exit_code == 2
    assert "fake-model is offered by fake and other" in ambiguous.stderr
    assert "name one of fake/fake-model, other/fake-model" in ambiguous.stderr

    named = output(invoke("benchmark", str(dataset), "--models", "fake/fake-model"))
    assert named["preview"]["models"][0]["model"] == "fake/fake-model"
    assert "--models fake/fake-model " in named["next_command"]
    result = rerun(named["next_command"])
    assert result.exit_code == 0, result.stderr
    assert output(result)["status"] == "complete"


def test_the_hint_for_a_model_several_providers_offer_names_their_aliases(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    providers.register_provider(
        "other", "pkg.other:Provider", models={"other-model": ("other-model", "fake/tokenizer")}
    )

    result = invoke("benchmark", str(dataset), "--models", "FAKE/TOKENIZER")

    assert result.exit_code == 2
    assert "name one of fake/fake-model, other/other-model" in result.stderr


def test_one_model_on_two_providers_is_named_by_provider_everywhere(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    providers.register_provider(
        "twin",
        "pup_finetune.providers.fake:FakeProvider",
        dev_only=True,
        models={"fake-model": ("fake-model",)},
        offline=True,
    )

    preview = output(
        invoke("benchmark", str(dataset), "--models", "fake/fake-model,twin/fake-model")
    )
    models = [(m["model"], m["provider"]) for m in preview["preview"]["models"]]
    assert models == [("fake/fake-model", "fake"), ("twin/fake-model", "twin")]
    assert "--models fake/fake-model,twin/fake-model " in preview["next_command"]

    result = rerun(preview["next_command"])
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert sorted(data["files"]["models"]) == ["fake/fake-model", "twin/fake-model"]
    for model in ("fake", "twin"):
        assert (dataset / "benchmark" / model / "fake-model" / "results.jsonl").exists()
    train = data["next_command"] or data["next_steps"][0]["command"]
    assert "--model fake-model" in train
    assert f"--provider {data['recommendation']['model'].split('/')[0]}" in train


def test_a_model_whose_session_fails_is_an_error_and_exits_1(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @contextmanager
    def no_capacity(plan: Any, candidate: Any) -> Iterator[Sampler]:
        raise ExternalError("no capacity for a sampler", source="fake")
        yield  # pragma: no cover

    monkeypatch.setattr(benchmark_module, "_base_sampler", no_capacity)

    result = invoke("benchmark", str(dataset), "--models", "fake-model", "--confirm")

    assert result.exit_code == 1
    data = output(result)
    assert data["status"] == "complete_with_errors"
    [entry] = data["leaderboard"]
    assert entry["status"] == "error"
    assert entry["error"]["message"].endswith("no capacity for a sampler")
    assert data["recommendation"]["verdict"] == "no_result"
    assert data["next_command"].startswith("pup finetune benchmark")


class AlwaysPass:
    """A judge that passes every action, a corrupted one too, so calibration fails."""

    def complete(self, messages: Any, **kwargs: Any) -> ChatCompletion:
        return ChatCompletion(
            message={"role": "assistant", "content": '{"pass": true, "reason": "fine"}'}
        )


def test_a_judge_that_fails_calibration_stops_before_any_generation(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @contextmanager
    def always_pass(config: Any, **options: Any) -> Iterator[AlwaysPass]:
        yield AlwaysPass()

    monkeypatch.setattr(benchmark_module, "open_judge_client", always_pass)

    result = invoke("benchmark", str(dataset), "--models", "fake-model", "--confirm")

    assert result.exit_code == 1
    data = output(result)
    assert data["status"] == "calibration_failed"
    assert "--judge-model" in data["hint"]
    assert not (dataset / "benchmark" / "fake-model").exists()


def test_read_only_allows_an_offline_benchmark(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "1")

    result = invoke("benchmark", str(dataset), "--models", "fake-model", "--confirm")

    assert result.exit_code == 0, result.stderr


def test_a_run_directory_is_not_a_dataset(dataset: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    trained = invoke(
        "train", "--model", "fake-model", "--dataset", str(dataset), "--run-dir", str(run_dir),
        "--max-epochs", "1", "--confirm",
    )  # fmt: skip
    assert trained.exit_code == 0, trained.stderr

    result = invoke("benchmark", str(run_dir), "--models", "fake-model")

    assert result.exit_code == 2
    assert "not a dataset" in result.stderr


def test_a_case_a_model_cannot_replay_is_left_out_for_it(tmp_path: Path) -> None:
    dataset, refused = import_with_refused_test_row(tmp_path)
    preview = output(invoke("benchmark", str(dataset), "--models", "fake-model"))["preview"]
    [model] = preview["models"]
    excluded = model["excluded_cases"]
    assert excluded > 0
    assert model["excluded_by_reason"] == {"unsupported_message": excluded}
    assert model["cases"] == preview["common_cases"] == preview["cases"] - excluded
    assert any("test cases are left out for fake-model, " in w for w in preview["warnings"])

    result = rerun(
        output(invoke("benchmark", str(dataset), "--models", "fake-model"))["next_command"]
    )
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "complete"
    assert data["leaderboard"][0]["excluded_cases"] == excluded
    results = (dataset / "benchmark" / "fake-model" / "results.jsonl").read_text()
    assert refused not in results


def test_a_prompt_refused_while_sampling_never_stops_the_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A renderer whose check misses what it then refuses, while sizing the session too.
    monkeypatch.setattr(FakeRenderer, "prompt_refusal", lambda self, messages, tools: None)
    dataset, _ = import_with_refused_test_row(tmp_path)

    result = invoke("benchmark", str(dataset), "--models", "fake-model", "--confirm")

    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "complete_with_errors"
    [entry] = data["leaderboard"]
    assert entry["errors_unrenderable"] == entry["errors"] > 0
    assert entry["cases"] > 0
    assert "which a retry cannot fix" in data["hint"]
    assert "benchmark" not in (data["next_command"] or "")


def test_the_preview_says_when_a_tokenizer_runs_repository_code(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = dataclasses.replace(fake_module.FAKE_PROFILE, trust_remote_code=True)
    monkeypatch.setattr(fake_module, "FAKE_PROFILE", remote)

    preview = output(invoke("benchmark", str(dataset), "--models", "fake-model"))["preview"]

    notice = "the fake-model tokenizer runs Python code from the Hugging Face repository"
    assert any(warning.startswith(notice) for warning in preview["warnings"])


class _ToolShyRenderer(FakeRenderer):
    """The fake renderer, refusing every prompt that ends with a tool result."""

    def prompt_refusal(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> RendererError | None:
        if messages and messages[-1].get("role") == "tool":
            return RendererError(RendererErrorCode.UNSUPPORTED_MESSAGE, "no turn after a tool")
        return super().prompt_refusal(messages, tools)


class ToolShyProvider(fake_module.FakeProvider):
    """The fake provider with :class:`_ToolShyRenderer`."""

    def renderer(self, profile: ModelProfile) -> FakeRenderer:
        return _ToolShyRenderer(profile)


def register_fake_model(monkeypatch: pytest.MonkeyPatch, **targets: str) -> None:
    """Register each provider of ``targets`` (name to target), offering ``fake-model``."""
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    for name, target in targets.items():
        providers.register_provider(
            name, target, dev_only=True, models={"fake-model": ("fake-model",)}, offline=True
        )


def test_adding_a_model_reuses_the_finished_ones_and_ranks_on_the_shared_cases(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_fake_model(
        monkeypatch,
        twin="pup_finetune.providers.fake:FakeProvider",
        shy="tests.cli.test_benchmark:ToolShyProvider",
    )
    two = "fake/fake-model,twin/fake-model"
    first = rerun(output(invoke("benchmark", str(dataset), "--models", two))["next_command"])
    assert first.exit_code == 0, first.stderr
    finished = {
        model: (dataset / "benchmark" / model / "results.jsonl").read_bytes()
        for model in two.split(",")
    }

    three = f"{two},shy/fake-model"
    preview = output(invoke("benchmark", str(dataset), "--models", three))
    by_model = {model["model"]: model for model in preview["preview"]["models"]}
    shy = by_model["shy/fake-model"]
    assert 0 < shy["excluded_cases"] < preview["preview"]["cases"]
    assert shy["excluded_by_reason"] == {"unsupported_message": shy["excluded_cases"]}
    assert preview["preview"]["common_cases"] == shy["cases"]
    for model in finished:
        assert by_model[model]["already_done"] == preview["preview"]["cases"]
        assert by_model[model]["generation_calls"] == 0

    opened: list[str] = []
    base_sampler = benchmark_module._base_sampler

    def recording(plan: Any, candidate: Any) -> Any:
        opened.append(candidate.target)
        return base_sampler(plan, candidate)

    monkeypatch.setattr(benchmark_module, "_base_sampler", recording)
    result = rerun(preview["next_command"])

    assert result.exit_code == 0, result.stderr
    assert opened == ["shy/fake-model"]
    for model, results in finished.items():
        assert (dataset / "benchmark" / model / "results.jsonl").read_bytes() == results
    data = output(result)
    assert data["status"] == "complete"
    assert data["common_cases"] == shy["cases"]
    gaps = [entry["gap_to_best"] for entry in data["leaderboard"][1:]]
    assert [gap["n_cases"] for gap in gaps] == [shy["cases"]] * 2


def test_a_changed_renderer_conflicts_with_the_finished_results(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview = output(invoke("benchmark", str(dataset), "--models", "fake-model"))
    assert rerun(preview["next_command"]).exit_code == 0
    identity = FakeRenderer.identity
    monkeypatch.setattr(
        FakeRenderer, "identity", lambda self: {**identity(self), "parser_version": "2"}
    )

    result = invoke("benchmark", str(dataset), "--models", "fake-model")

    assert result.exit_code == 2
    assert "Error [settings_conflict]" in result.stderr
    assert "changed: rendering" in result.stderr


def test_a_turn_with_no_message_before_it_is_left_out_for_every_model(tmp_path: Path) -> None:
    dataset, opened = import_with_opening_assistant_test_row(tmp_path)

    preview = output(
        invoke("benchmark", str(dataset), "--models", "fake-model", "--max-cases", "10")
    )

    [model] = preview["preview"]["models"]
    assert model["excluded_by_reason"] == {"empty_prompt": 1}
    result = rerun(preview["next_command"])
    assert result.exit_code == 0, result.stderr
    results = (dataset / "benchmark" / "fake-model" / "results.jsonl").read_text()
    assert f'"{opened}:0"' not in results
    assert f'"{opened}:' in results


def test_the_preview_fails_without_the_key_of_a_provider_it_generates_with(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    providers.register_provider(
        "keyed",
        "pup_finetune.providers.fake:FakeProvider",
        dev_only=True,
        models={"fake-model": ("fake-model",)},
        credential="fireworks",
        offline=True,
    )
    models = "fake/fake-model,keyed/fake-model"

    result = invoke("benchmark", str(dataset), "--models", models)

    assert result.exit_code == 2
    assert "no fireworks API key found for generating with keyed/fake-model" in result.stderr
    assert "--confirm" not in result.stdout

    monkeypatch.setenv("PUP_READ_ONLY", "1")
    read_only = output(invoke("benchmark", str(dataset), "--models", models))
    assert read_only["next_command"] is None
    assert any(
        warning.startswith("no fireworks API key found for generating with keyed/fake-model")
        for warning in read_only["preview"]["warnings"]
    )


def test_the_retry_hint_names_only_the_models_a_rerun_can_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_fake_model(monkeypatch, twin="pup_finetune.providers.fake:FakeProvider")
    # The fake renderer's check misses what it then refuses while sampling: final errors.
    monkeypatch.setattr(FakeRenderer, "prompt_refusal", lambda self, messages, tools: None)
    dataset, _ = import_with_refused_test_row(tmp_path)
    base_sampler = benchmark_module._base_sampler

    @contextmanager
    def twin_has_no_capacity(plan: Any, candidate: Any) -> Iterator[Sampler]:
        if candidate.provider == "twin":
            raise ExternalError("no capacity for a sampler", source="fake")
        with base_sampler(plan, candidate) as sampler:
            yield sampler

    monkeypatch.setattr(benchmark_module, "_base_sampler", twin_has_no_capacity)
    models = "fake/fake-model,twin/fake-model"

    data = output(invoke("benchmark", str(dataset), "--models", models, "--confirm"))

    statuses = {entry["model"]: entry["status"] for entry in data["leaderboard"]}
    assert statuses == {"fake/fake-model": "complete_with_errors", "twin/fake-model": "error"}
    assert data["hint"].startswith("twin/fake-model has no result for every case")
    assert data["next_command"].startswith("pup finetune benchmark")


def test_a_model_whose_stop_tokens_do_not_resolve_fails_the_preview(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unresolved(self: FakeRenderer) -> list[int]:
        raise UsageError("'<|end|>' is not a single token of the fake/tokenizer tokenizer")

    monkeypatch.setattr(FakeRenderer, "stop_token_ids", property(unresolved))

    result = invoke("benchmark", str(dataset), "--models", "fake-model")

    assert result.exit_code == 2
    assert "is not a single token" in result.stderr
    assert "--confirm" not in result.stdout
