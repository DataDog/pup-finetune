# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The bundled `datadog-finetune` skill: where it is found, its frontmatter, and installing it."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Callable
from pathlib import Path

import pytest

from pup_finetune import skill
from pup_finetune.errors import ERROR_CODES, StateError, UsageError

REPO_SKILL = Path(__file__).resolve().parents[1] / "skills" / "datadog-finetune" / "SKILL.md"


def frontmatter(text: str) -> dict[str, str]:
    """The ``key: value`` lines between the leading ``---`` fences (plain YAML scalars)."""
    lines = text.splitlines()
    assert lines[0] == "---", "the frontmatter must be the first thing in SKILL.md"
    end = lines.index("---", 1)
    fields: dict[str, str] = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(": ")
        assert sep, f"not a single-line `key: value` field: {line!r}"
        fields[key] = value
    return fields


def test_the_skill_is_found_and_is_the_repository_file() -> None:
    # An editable install (the test run) has no packaged copy, so the repo's file is used.
    assert skill.skill_path().read_bytes() == REPO_SKILL.read_bytes()
    assert skill.skill_text() == REPO_SKILL.read_text(encoding="utf-8")
    assert skill.bundled_skills_dir().name in {skill.PACKAGED_DIR, skill.REPO_DIR}


def test_frontmatter_has_a_valid_name_and_description() -> None:
    fields = frontmatter(skill.skill_text())
    assert set(fields) == {"name", "description"}
    assert fields["name"] == skill.SKILL_NAME == REPO_SKILL.parent.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", fields["name"])
    description = fields["description"]
    assert 50 <= len(description) <= 1024
    # A plain YAML scalar: nothing that would end it early or change its type.
    assert ": " not in description
    assert " #" not in description
    assert description[0].isalpha()
    for trigger in ("Agent Observability", "ml_app", "pup finetune", "train", "evaluate"):
        assert trigger in description


def test_the_body_has_the_rules_steps_and_debugging_table() -> None:
    text = skill.skill_text()
    assert "## Rules for every step" in text
    for step in range(8):
        assert re.search(rf"^## Step {step}\b", text, re.MULTILINE), f"step {step} is missing"
    sections = re.split(r"^## ", text, flags=re.MULTILINE)
    for section in sections:
        if section.startswith("Step "):
            assert "\nCheck:\n" in section, f"no Check: list in {section.splitlines()[0]!r}"
    assert "## End state" in text
    assert "| Symptom | Cause | Fix |" in text
    assert "pup finetune auth login <PROVIDER>" in text
    assert "untrusted data" in text


def _section(text: str, heading: str) -> str:
    """The body of the ``## heading`` section of ``text``."""
    return text.split(f"\n## {heading}\n", 1)[1].split("\n## ", 1)[0]


def test_the_body_opens_with_the_flow_and_where_to_start() -> None:
    text = skill.skill_text()
    assert text.index("## The flow") < text.index("## Rules for every step")
    assert "| The user has | Start at |" in _section(text, "The flow")


def test_there_are_about_eight_short_rules() -> None:
    rules = re.findall(
        r"^\d+\. (.*?)(?=^\d+\. |\Z)",
        _section(skill.skill_text(), "Rules for every step"),
        re.MULTILINE | re.DOTALL,
    )
    assert 6 <= len(rules) <= 9, rules
    for rule in rules:
        rule = rule.split("\n\n", 1)[0]
        assert len(rule.split()) <= 30, rule


def test_the_preview_and_status_templates_are_there() -> None:
    text = skill.skill_text()
    preview = _section(text, "Presenting a preview")
    for line in ("Runs:", "Sends to:", "Cost:", "Command:", "Run it?"):
        assert line in preview, line
    status = _section(text, "Status -> do this")
    for key in ("`phase`", "`worker.state`", "`publication.status`", "complete_with_errors"):
        assert key in status, key


def test_commands_use_runnable_examples_not_shell_variables_or_lowercase_placeholders() -> None:
    # An agent copies these; its shell state does not persist between calls, and a lowercase
    # `<run_dir>` reads like a real argument. `<UPPER>` marks a value to fill in.
    text = skill.skill_text()
    for line in text.splitlines():
        if "pup finetune" not in line and "pup llm-obs" not in line:
            continue
        assert not re.search(r"\$[A-Za-z_{]", line), line
        for command in re.findall(r"pup (?:finetune|llm-obs)[^`|]*", line):
            assert not re.search(r"<[a-z][a-z_ ]*>", command), command


@pytest.mark.parametrize(
    "phrase",
    [
        # The safety rules.
        "Never add `--confirm` without the user's explicit authorization",
        "`PUP_AUTO_APPROVE`",
        "are not authorization",
        "Never ask for API keys in chat",
        "Sending traces to a model provider needs consent",
        "Respect read-only mode",
        "Do not change `PUP_READ_ONLY` yourself",
        "Deploying needs the user's explicit authorization",
        "per GPU-second",
    ],
)
def test_the_body_keeps_each_safety_rule(phrase: str) -> None:
    assert phrase in skill.skill_text()


def test_the_debugging_table_covers_each_failure() -> None:
    # Keyed on the error codes the error line shows, never on class names that no output has.
    table = skill.skill_text().split("## Debugging", 1)[1]
    rows = [line for line in table.splitlines() if line.startswith("| ") and "---" not in line]
    for term in (
        "datadog_auth",  # Datadog auth
        "provider key",  # provider auth
        "matches no spans",  # empty pulls
        "excluded_by_code",  # exclusions
        "Error [read_only]",  # read-only mode
        "Error [provider_capacity]",  # capacity
        "Error [provider_auth]",  # a rejected key
        "worker.state",  # a stale or exited worker
        "Error [wrong_phase]",  # resume, evaluate or deploy in the wrong phase
        "Error [settings_conflict]",  # frozen settings, evaluation identity
        "[not_a_run_dir]",  # the wrong directory
        "calibration",  # judge calibration
        "push.json",  # publishing without a push receipt
        "times out",  # deployment timeouts
    ):
        assert any(term in row for row in rows), f"no debugging row mentions {term!r}"


def test_the_skill_lists_every_error_code_itself() -> None:
    # The installed skill ships alone, without docs/commands.md: its own table is the list.
    table = skill.skill_text().split("## Debugging", 1)[1]
    assert "docs/commands.md" not in skill.skill_text()
    for code, (exit_code, meaning) in ERROR_CODES.items():
        assert f"| `{code}` | `{exit_code}` | {meaning} |" in table, code


def test_the_skill_never_types_a_confirm_for_a_previewed_command() -> None:
    # The agent runs the preview's next_command, which pins what was previewed, rather than
    # typing --confirm itself.
    text = skill.skill_text()
    for command in ("train", "resume", "dataset triage", "dataset push"):
        typed = re.findall(rf"pup finetune {command} [^`\n]*--confirm", text)
        assert typed == [], typed


MAX_SKILL_LINES = 320
"""The most lines SKILL.md may have: an agent loads all of it into its context."""


def test_the_skill_stays_short() -> None:
    lines = len(skill.skill_text().splitlines())
    assert lines <= MAX_SKILL_LINES, f"SKILL.md has {lines} lines, over {MAX_SKILL_LINES}"


def test_the_skill_never_types_a_confirm_for_undeploy_or_deploy() -> None:
    # Deleting or creating a deployment is the user's to approve from the preview.
    text = skill.skill_text()
    for command in ("deploy", "undeploy"):
        assert re.findall(rf"pup finetune {command} [^`\n]*--confirm", text) == [], command


def test_the_skill_keeps_data_out_of_the_checkout() -> None:
    text = skill.skill_text()
    assert "./ds" not in text
    assert "datasets/my-agent" in text


def _fake_files(root: Path) -> Callable[[str], Path]:
    return lambda package: root / package


def test_bundled_skills_dir_prefers_the_packaged_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packaged = tmp_path / "pup_finetune" / skill.PACKAGED_DIR
    (packaged / skill.SKILL_NAME).mkdir(parents=True)
    (packaged / skill.SKILL_NAME / skill.SKILL_FILE).write_text("packaged\n")
    monkeypatch.setattr(skill.importlib.resources, "files", _fake_files(tmp_path))
    assert skill.bundled_skills_dir() == packaged
    assert skill.skill_text() == "packaged\n"


def test_bundled_skills_dir_falls_back_to_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(skill.importlib.resources, "files", _fake_files(tmp_path))
    assert skill.bundled_skills_dir() == REPO_SKILL.parent.parent


def test_bundled_skills_dir_without_either_copy_is_a_state_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(skill.importlib.resources, "files", _fake_files(tmp_path))
    monkeypatch.setattr(skill, "__file__", str(tmp_path / "src" / "pup_finetune" / "skill.py"))
    with pytest.raises(StateError, match="missing from this install"):
        skill.bundled_skills_dir()


def test_npx_env_drops_datadog_secrets_and_provider_keys() -> None:
    environ = {
        "PATH": "/usr/bin",
        "HOME": "/home/someone",
        "DD_ACCESS_TOKEN": "synthetic",
        "DD_API_KEY": "synthetic",
        "DD_APP_KEY": "synthetic",
        "DD_APPLICATION_KEY": "synthetic",
        "FIREWORKS_API_KEY": "synthetic",
        "ANTHROPIC_API_KEY": "synthetic",
        "ANTHROPIC_AUTH_TOKEN": "synthetic",
        "ANTHROPIC_IDENTITY_TOKEN": "synthetic",
        "BASETEN_API_KEY": "synthetic",
        "OPENAI_API_KEY": "synthetic",
    }
    assert skill.npx_env(environ) == {"PATH": "/usr/bin", "HOME": "/home/someone"}
    assert "DD_API_KEY" in environ  # the parent's mapping is left alone


def test_npx_argv_installs_only_this_skill_and_passes_options_on() -> None:
    source = Path("/opt/skills")
    assert skill.npx_argv("npx", source) == [
        "npx",
        "--yes",
        "skills",
        "add",
        "/opt/skills",
        "--skill",
        "datadog-finetune",
    ]
    argv = skill.npx_argv("npx", source, agents=["claude-code", "codex"], global_=True, yes=True)
    assert argv[7:] == ["--global", "--yes", "--agent", "claude-code", "--agent", "codex"]


def test_install_files_writes_both_locations(tmp_path: Path) -> None:
    installed = skill.install_files(tmp_path, text="skill v1\n")
    assert [entry.to_json() for entry in installed] == [
        {"path": str(tmp_path / ".agents/skills/datadog-finetune/SKILL.md"), "state": "created"},
        {"path": str(tmp_path / ".claude/skills/datadog-finetune/SKILL.md"), "state": "created"},
    ]
    for entry in installed:
        assert entry.path.read_text() == "skill v1\n"
        assert stat.S_IMODE(entry.path.stat().st_mode) == 0o644
    again = skill.install_files(tmp_path, text="skill v1\n")
    assert [entry.state for entry in again] == ["unchanged", "unchanged"]


def test_install_files_replaces_a_different_copy_only_when_asked(tmp_path: Path) -> None:
    skill.install_files(tmp_path, text="skill v1\n")
    claude = tmp_path / ".claude/skills/datadog-finetune/SKILL.md"
    claude.write_text("edited locally\n")
    with pytest.raises(UsageError, match="different copy") as raised:
        skill.install_files(tmp_path, text="skill v2\n")
    assert str(claude) in str(raised.value)
    # Nothing was written: the copy that matched v1 is still v1.
    assert (tmp_path / ".agents/skills/datadog-finetune/SKILL.md").read_text() == "skill v1\n"
    states = [
        entry.state for entry in skill.install_files(tmp_path, text="skill v2\n", replace=True)
    ]
    assert states == ["updated", "updated"]
    assert claude.read_text() == "skill v2\n"


def test_install_files_that_cannot_read_a_target_is_a_state_error(tmp_path: Path) -> None:
    (tmp_path / ".agents/skills/datadog-finetune/SKILL.md").mkdir(parents=True)
    with pytest.raises(StateError, match="cannot read"):
        skill.install_files(tmp_path, text="skill v1\n")


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_install_files_that_cannot_write_a_target_is_a_state_error(tmp_path: Path) -> None:
    locked = tmp_path / ".claude/skills/datadog-finetune"
    locked.mkdir(parents=True)
    locked.chmod(0o500)
    try:
        with pytest.raises(StateError, match="cannot write"):
            skill.install_files(tmp_path, text="skill v1\n")
    finally:
        locked.chmod(0o700)
