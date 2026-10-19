# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Evaluator specs: parsing, identities, and what an evaluation directory freezes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from pup_finetune.errors import SettingsConflict, UsageError
from pup_finetune.evaluation.evaluators import (
    CheckEvaluator,
    JudgeEvaluator,
    attach_evaluators,
    load_spec,
    resolve_evaluators,
    stored_evaluators,
)
from pup_finetune.evaluation.fake_judge import fake_judge_config
from pup_finetune.runtime import Runtime

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "evaluators"
JUDGE = fake_judge_config()


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> Runtime:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")
    return Runtime.from_env()


@pytest.fixture
def spec_dir(tmp_path: Path) -> Path:
    target = tmp_path / "spec"
    shutil.copytree(FIXTURES, target)
    return target


def write(directory: Path, text: str) -> Path:
    path = directory / "evals.toml"
    path.write_text(text)
    return path


CHECK = '[[evaluator]]\nname = "no_card_numbers"\ntype = "check"\nfile = "checks.py"\n'
JUDGE_ITEM = '[[evaluator]]\nname = "polite"\ntype = "judge"\nrubric = "Be kind."\n'


def test_the_fixture_spec_loads_with_paths_from_the_spec(spec_dir: Path, runtime: Runtime) -> None:
    found = load_spec(spec_dir / "evals.toml", judge=JUDGE, runtime=runtime)
    assert found.headline == "all"
    assert found.names == ["polite", "no_card_numbers", "text_when_expected"]
    polite, card, text = found.evaluators
    assert isinstance(polite, JudgeEvaluator) and polite.config == JUDGE
    assert [example.expected_pass for example in polite.examples] == [True, False]
    assert polite.examples[0].case.id == "example-1"
    assert isinstance(card, CheckEvaluator) and card.file == spec_dir / "checks.py"
    assert isinstance(text, CheckEvaluator) and text.function == "answers_in_text"
    assert found.spec == spec_dir / "evals.toml"


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (CHECK + "colour = 'red'\n", r"evaluator\[0\]\.colour: Extra inputs"),
        ("unknown = 1\n" + CHECK, r"unknown: Extra inputs"),
        (CHECK + CHECK, "'no_card_numbers' is given twice"),
        (CHECK.replace("no_card_numbers", "judge_pass"), "'judge_pass' is reserved"),
        (CHECK.replace("no_card_numbers", "all"), "'all' is reserved"),
        (CHECK.replace("no_card_numbers", "tool_name_match"), "is reserved"),
        (CHECK.replace("no_card_numbers", "Bad-Name"), "must be lowercase"),
        (CHECK.replace("checks.py", "missing.py"), "cannot be read"),
        (CHECK + 'function = "nope"\n', "no public function 'nope'"),
        ('headline = "polite"\n' + CHECK, "headline 'polite' names no evaluator"),
        ("headline = 1\n", "headline: Input should be a valid string"),
        (JUDGE_ITEM.replace('rubric = "Be kind."', 'rubric = ""'), "rubric"),
    ],
)
def test_a_bad_spec_is_a_usage_error_naming_the_key(
    spec_dir: Path, runtime: Runtime, text: str, match: str
) -> None:
    with pytest.raises(UsageError, match=match):
        load_spec(write(spec_dir, text), judge=JUDGE, runtime=runtime)


def test_a_toml_error_reports_its_line(spec_dir: Path, runtime: Runtime) -> None:
    with pytest.raises(UsageError, match=r"not valid TOML: .*line 5"):
        load_spec(write(spec_dir, CHECK + "name = \n"), judge=JUDGE, runtime=runtime)


@pytest.mark.parametrize(
    "lines",
    [
        ['{"conversation": [{"role": "user", "content": "hi"}], "candidate": "x", "pass": true}'],
        ['{"conversation": [], "candidate": "x", "pass": true}'],
        ['{"conversation": [{"role": "user", "content": "hi"}], "candidate": "x", "pass": "yes"}'],
    ],
)
def test_examples_need_both_verdicts_and_valid_lines(
    spec_dir: Path, runtime: Runtime, lines: list[str]
) -> None:
    (spec_dir / "examples.jsonl").write_text("\n".join(lines) + "\n")
    text = JUDGE_ITEM + 'examples = "examples.jsonl"\n'
    with pytest.raises(UsageError, match=r"examples\.jsonl"):
        load_spec(write(spec_dir, text), judge=JUDGE, runtime=runtime)


def test_the_identity_is_the_rubric_model_and_file_not_the_paths_or_examples(
    spec_dir: Path, runtime: Runtime, tmp_path: Path
) -> None:
    first = load_spec(spec_dir / "evals.toml", judge=JUDGE, runtime=runtime)
    moved = tmp_path / "moved"
    shutil.copytree(spec_dir, moved)
    (moved / "polite-examples.jsonl").write_text((moved / "polite-examples.jsonl").read_text() * 2)
    second = load_spec(moved / "evals.toml", judge=JUDGE, runtime=runtime)
    assert [e.identity() for e in first.evaluators] == [e.identity() for e in second.evaluators]
    (moved / "checks.py").write_text((moved / "checks.py").read_text() + "\n")
    third = load_spec(moved / "evals.toml", judge=JUDGE, runtime=runtime)
    assert first.evaluators[1].identity() != third.evaluators[1].identity()
    assert set(first.evaluators[0].identity()) == {"name", "type", "rubric_sha256", "model"}


def test_attaching_only_grows_and_takes_a_new_headline(
    spec_dir: Path, runtime: Runtime, tmp_path: Path
) -> None:
    path = tmp_path / "eval" / "evaluators.json"
    path.parent.mkdir()
    one = load_spec(write(spec_dir, JUDGE_ITEM), judge=JUDGE, runtime=runtime)
    attach_evaluators(path, one)
    first = path.read_bytes()
    attach_evaluators(path, one)
    assert path.read_bytes() == first
    both = load_spec(
        write(spec_dir, 'headline = "no_card_numbers"\n' + JUDGE_ITEM + CHECK),
        judge=JUDGE,
        runtime=runtime,
    )
    attach_evaluators(path, both)
    stored = json.loads(path.read_text())
    assert [item["name"] for item in stored["evaluators"]] == ["polite", "no_card_numbers"]
    assert stored["headline"] == "no_card_numbers"
    assert stored["evaluators"][0]["rubric"] == "Be kind."


@pytest.mark.parametrize(
    ("text", "match", "hint"),
    [
        (JUDGE_ITEM.replace("Be kind.", "Be very kind."), "polite changed", "keep polite as"),
        (JUDGE_ITEM + 'model = "glm-5p3-flash"\n', "polite changed", "keep polite as"),
        (CHECK, "leaves out", "add polite back"),
    ],
)
def test_a_changed_or_dropped_evaluator_is_a_conflict_with_a_free_out_dir(
    spec_dir: Path, runtime: Runtime, tmp_path: Path, text: str, match: str, hint: str
) -> None:
    path = tmp_path / "eval" / "evaluators.json"
    path.parent.mkdir()
    attach_evaluators(path, load_spec(write(spec_dir, JUDGE_ITEM), judge=JUDGE, runtime=runtime))
    with pytest.raises(SettingsConflict, match=match) as caught:
        attach_evaluators(path, load_spec(write(spec_dir, text), judge=JUDGE, runtime=runtime))
    assert hint in str(caught.value.hint)
    assert "--out-dir" in str(caught.value.hint) and "eval-2" in str(caught.value.hint)


def test_attaching_again_from_a_moved_spec_records_its_files(
    spec_dir: Path, runtime: Runtime, tmp_path: Path
) -> None:
    path = tmp_path / "eval" / "evaluators.json"
    path.parent.mkdir()
    attach_evaluators(path, load_spec(write(spec_dir, CHECK), judge=JUDGE, runtime=runtime))
    moved = tmp_path / "moved"
    spec_dir.rename(moved)
    attach_evaluators(path, load_spec(moved / "evals.toml", judge=JUDGE, runtime=runtime))
    stored = stored_evaluators(path)
    assert stored is not None and stored["evaluators"][0]["file"] == str(moved / "checks.py")
    found = resolve_evaluators(None, stored, judge=JUDGE, runtime=runtime)
    assert found is not None and found.checks[0].module is not None


def test_a_changed_check_file_is_a_conflict(
    spec_dir: Path, runtime: Runtime, tmp_path: Path
) -> None:
    path = tmp_path / "eval" / "evaluators.json"
    path.parent.mkdir()
    attach_evaluators(path, load_spec(write(spec_dir, CHECK), judge=JUDGE, runtime=runtime))
    (spec_dir / "checks.py").write_text((spec_dir / "checks.py").read_text() + "# edited\n")
    with pytest.raises(SettingsConflict, match="file_sha256"):
        attach_evaluators(path, load_spec(write(spec_dir, CHECK), judge=JUDGE, runtime=runtime))


def test_stored_evaluators_resolve_again_and_load_a_check_only_with_work_left(
    spec_dir: Path, runtime: Runtime, tmp_path: Path
) -> None:
    path = tmp_path / "eval" / "evaluators.json"
    path.parent.mkdir()
    attach_evaluators(
        path, load_spec(write(spec_dir, JUDGE_ITEM + CHECK), judge=JUDGE, runtime=runtime)
    )
    stored = stored_evaluators(path)
    (spec_dir / "checks.py").write_text("raise RuntimeError('must not run')\n")

    def resolve(pending: bool) -> Any:
        return resolve_evaluators(
            None, stored, judge=JUDGE, runtime=runtime, pending=lambda name: pending
        )

    done = resolve(False)
    assert done.source == "evaluation directory"
    assert done.names == ["polite", "no_card_numbers"]
    assert done.evaluators[0].rubric == "Be kind."
    assert done.checks[0].module is None
    with pytest.raises(UsageError, match="changed since it was attached") as caught:
        resolve(True)
    assert caught.value.hint == (
        "restore it (a new version goes in a new file, under another name), or pass a new --out-dir"
    )
    (spec_dir / "checks.py").unlink()
    with pytest.raises(UsageError, match="is missing since it was attached"):
        resolve(True)
    assert resolve_evaluators(None, None, judge=JUDGE, runtime=runtime) is None
