# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune skill`: show or install the `datadog-finetune` agent skill."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer

from pup_finetune import skill
from pup_finetune.cli._common import run_command
from pup_finetune.cli._presentation import SET_UP, CommandDoc, Safety, describe, group, local
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.output import info, warn

skill_app = group(
    "skill",
    short="Show or install the datadog-finetune agent skill.",
    description=(
        f"The `{skill.SKILL_NAME}` agent skill teaches a coding agent the curate, train, "
        "evaluate and deploy workflow, with its rules: preview before paying, never confirm "
        "without the user's authorization, and never ask for API keys in chat."
    ),
)

SHOW_DOC = CommandDoc(
    path="skill show",
    short="Print the skill's path and content.",
    description=(
        f"Print where the bundled `{skill.SKILL_NAME}` skill is and its SKILL.md content, "
        "for an agent to read or a person to copy."
    ),
    examples=("pup finetune skill show",),
    safety=local("prints the skill"),
)
INSTALL_DOC = CommandDoc(
    path="skill install",
    short="Install the skill for your coding agents.",
    description=describe(
        f"Install the `{skill.SKILL_NAME}` skill: with npx on PATH and no --dir, through "
        "`npx skills add` for the agents you choose (it may prompt); otherwise by copying "
        "SKILL.md into the project directory (--dir) or, with --global, your home directory.",
        "The npx path runs `npx --yes skills add <bundled skills directory> --skill "
        f"{skill.SKILL_NAME}`, passing on --agent, --global and --yes. The copy goes to "
        f".agents/skills/{skill.SKILL_NAME}/ and .claude/skills/{skill.SKILL_NAME}/; a "
        "different copy already there is replaced only with --yes. npx runs without your "
        "secrets in its environment.",
    ),
    examples=(
        "pup finetune skill install",
        "pup finetune skill install -a claude-code -g -y",
        "pup finetune skill install --dir .",
    ),
    safety=Safety(
        data="writes skill files; npx fetches the skills CLI from npm",
        recipients="nowhere: npx gets no secrets",
        without="installs the skill",
    ),
)

NpxRunner = Callable[[Sequence[str], Mapping[str, str]], int]
"""Runs the ``npx skills add`` command in an environment and returns its exit code."""


def run_npx(argv: Sequence[str], env: Mapping[str, str]) -> int:
    """Run ``argv`` in ``env`` with the terminal's stdin and stderr, and its stdout on stderr.

    The skills CLI may prompt, so it keeps the terminal; its stdout goes to stderr so this
    command's stdout still holds only its one JSON result. Tests replace it with a fake.
    """
    return subprocess.run(  # noqa: S603 - a fixed argv, no shell
        list(argv), stdout=sys.stderr, env=dict(env), check=False
    ).returncode


def find_npx() -> str | None:
    """The ``npx`` executable on ``PATH``, if any. Tests replace it."""
    return shutil.which("npx")


@skill_app.command(**SHOW_DOC.typer())
@run_command
def show() -> dict[str, Any]:
    path = skill.skill_path()
    return {
        "name": skill.SKILL_NAME,
        "path": str(path),
        "content": skill.skill_text(),
    }


@skill_app.command(**INSTALL_DOC.typer())
@run_command
def install(
    agents: Annotated[
        list[str] | None,
        typer.Option(
            "--agent",
            "-a",
            metavar="AGENT",
            help="An agent to install for (e.g. claude-code); repeatable.",
            show_default=False,
        ),
    ] = None,
    global_: Annotated[
        bool,
        typer.Option("--global", "-g", help="Install for your user instead of this project."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip the prompts; without npx, replace a different copy.",
        ),
    ] = False,
    directory: Annotated[
        Path | None,
        typer.Option(
            "--dir",
            metavar="DIR",
            help="Copy the skill into this project directory, without npx.",
            file_okay=False,
            show_default=False,
        ),
    ] = None,
) -> dict[str, Any]:
    if global_ and directory is not None:
        raise UsageError("pass --global or --dir, not both")
    source = skill.bundled_skills_dir()
    npx = None if directory is not None else find_npx()
    if npx is not None:
        return _install_with_npx(npx, source, agents or [], global_=global_, yes=yes)
    base = Path.home() if global_ else (directory or Path.cwd())
    if agents:
        warn("--agent is used only by the skills CLI (npx); the copy goes to both directories")
    installed = skill.install_files(base, text=skill.skill_text(), replace=yes)
    for entry in installed:
        info(f"{entry.state}: {entry.path}")
    return {
        "name": skill.SKILL_NAME,
        "method": "copy",
        "source": str(source / skill.SKILL_NAME / skill.SKILL_FILE),
        "base": str(base),
        "installed": [entry.to_json() for entry in installed],
    }


def _install_with_npx(
    npx: str, source: Path, agents: Sequence[str], *, global_: bool, yes: bool
) -> dict[str, Any]:
    argv = skill.npx_argv(npx, source, agents=agents, global_=global_, yes=yes)
    info(f"running: {shlex.join(argv)}")
    try:
        code = run_npx(argv, skill.npx_env(os.environ))
    except OSError as exc:
        raise ExternalError(
            f"could not run npx: {exc}",
            source="npx",
            hint="pass --dir . to copy the skill into this project instead",
        ) from exc
    if code != 0:
        raise ExternalError(
            f"`npx skills add` exited {code}",
            source="npx",
            hint="see its output above, or pass --dir . to copy the skill into this "
            "project instead",
        )
    return {
        "name": skill.SKILL_NAME,
        "method": "npx",
        "source": str(source),
        "command": shlex.join(argv),
        "agents": list(agents),
        "global": global_,
        "installed": [],
        "note": "the skills CLI chose where to install; its output is on stderr",
    }


def register(app: typer.Typer) -> None:
    """Add the `skill` command group to ``app``."""
    app.add_typer(skill_app, name="skill", rich_help_panel=SET_UP)
