# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""scripts/gen_command_docs.py: the committed command reference and exclusion table are current."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import typer
import typer.main

from pup_finetune.cli import app
from pup_finetune.data.exclusions import EXPLANATIONS

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gen_command_docs.py"


def _load_script() -> ModuleType:
    # scripts/ is not a package (and is not shipped in the wheel), so load the file by path.
    spec = importlib.util.spec_from_file_location("gen_command_docs", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_script()

_FIX = "run `uv run python scripts/gen_command_docs.py` and commit the result"


def test_docs_commands_md_is_up_to_date() -> None:
    assert gen.COMMANDS_DOC.read_text(encoding="utf-8") == gen.render_commands(), _FIX


def test_the_exclusion_table_in_data_format_md_is_up_to_date() -> None:
    text = gen.DATA_FORMAT_DOC.read_text(encoding="utf-8")
    assert gen.with_exclusions(text) == text, _FIX


def test_the_table_of_what_each_step_needs_in_the_readme_is_up_to_date() -> None:
    text = gen.README.read_text(encoding="utf-8")
    assert gen.with_requirements(text) == text, _FIX


def test_the_table_of_what_each_step_needs_comes_from_the_registries() -> None:
    from pup_finetune import credentials
    from pup_finetune.providers import registered_providers

    table = gen.render_requirements()
    for spec in credentials.all_specs():
        assert f"`{spec.env_var}`" in table, spec.provider
    for name in registered_providers():
        assert f"`train` on {name.capitalize()}" in table or f"on {name.capitalize()} |" in table
    assert "Fake" not in table  # development-only providers stay out of the README
    assert gen.requirement_rows() == [
        ("`dataset preview`, `dataset pull`", "yes", "", ""),
        ("`dataset import`, `show`, `sample`, `label`; `status`, `runs`", "", "", ""),
        ("`dataset triage`", "", "`FIREWORKS_API_KEY`", ""),
        ("`dataset push`", "yes", "", ""),
        ("`train`, `evaluate` on Fireworks", "", "`FIREWORKS_API_KEY`", "`fireworks`, `hf`"),
        ("`deploy`, `undeploy` on Fireworks", "", "`FIREWORKS_API_KEY`", "`fireworks`"),
        ("`train` on Baseten", "", "`BASETEN_API_KEY`", "`baseten`, `hf`"),
        # Evaluating needs the default judge's key too.
        (
            "`evaluate` on Baseten",
            "",
            "`BASETEN_API_KEY`, `FIREWORKS_API_KEY`",
            "`baseten`, `hf`",
        ),
        ("`anthropic/<model>` judges", "", "`ANTHROPIC_API_KEY`", "`anthropic`"),
        ("`openai/<model>` judges", "", "`OPENAI_API_KEY`", ""),
    ]


def test_check_mode_reports_nothing_stale() -> None:
    assert gen.stale() == []
    assert gen.main(["--check"]) == 0


@pytest.mark.parametrize(
    "env",
    [{"GITHUB_ACTIONS": "true"}, {"COLUMNS": "40"}, {"FORCE_COLOR": "1", "TERM": "xterm"}],
    ids=["github-actions", "narrow-terminal", "forced-color"],
)
def test_the_output_does_not_depend_on_the_terminal(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = gen.render_commands()
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    rendered = gen.render_commands()
    assert rendered == baseline
    assert "\x1b[" not in rendered


def test_every_visible_command_and_option_is_documented() -> None:
    command = typer.main.get_command(app)
    ctx = typer.Context(command, info_name="pup finetune")
    text = gen.render_commands()
    commands = list(gen.iter_commands(command, ctx))
    assert {name for name, _ in commands} >= {"train", "dataset pull", "skill install"}
    assert "_worker" not in {name for name, _ in commands}
    for name, leaf in commands:
        assert f"## `pup finetune {name}`" in text
        for param in leaf.params:
            if param.param_type_name == "option" and not param.hidden:
                assert f"`{param.opts[0]}`" in text


def test_every_exclusion_code_is_explained_in_data_format_md() -> None:
    text = gen.DATA_FORMAT_DOC.read_text(encoding="utf-8")
    for code in EXPLANATIONS:
        assert f"| `{code.value}` |" in text


def test_a_stale_file_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stale_doc = tmp_path / "commands.md"
    stale_doc.write_text("old\n")
    monkeypatch.setattr(gen, "COMMANDS_DOC", stale_doc)
    assert gen.stale() == [stale_doc]


def test_the_docs_keep_what_help_cuts_and_every_example_and_safety_line() -> None:
    from pup_finetune.cli._presentation import DOCS

    text = " ".join(gen.render_commands().split())
    for name, doc in DOCS.items():
        assert " ".join(doc.description.split()) in text, name
        if doc.more:  # cut from --help at \f, kept here
            assert " ".join(gen.md(doc.more).split()) in text, name
        for example in doc.examples:
            assert example in text, (name, example)
        assert f"Safety: {gen.md(doc.safety.footer())}." in text, name
        row = " | ".join(gen.md(cell, cell=True) for cell in doc.safety.table_row())
        assert f"| `pup finetune {name}` | {row} |" in text, name


def test_a_default_written_into_the_help_goes_to_the_default_column() -> None:
    text = gen.render_commands()
    assert "| `--run-id` | `ID` | generated | Run and model id. |" in text
    assert "[dim]" not in text
    assert "\f" not in text
