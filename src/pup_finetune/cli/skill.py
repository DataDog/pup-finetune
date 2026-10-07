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
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.output import info, warn

_LOCAL_ONLY = "Reads and writes local files only, so it is allowed with PUP_READ_ONLY set."
_NO_REMOTE_CALLS = (
    "Makes no Datadog or provider call, so it is allowed with PUP_READ_ONLY set; the npx "
    "path downloads the skills CLI from npm and runs it, without your secrets in its "
    "environment."
)

skill_app = typer.Typer(
    name="skill",
    short_help="Show or install the datadog-finetune agent skill.",
    help=(
        f"The `{skill.SKILL_NAME}` agent skill teaches a coding agent the curate, train, "
        "evaluate and deploy workflow, with its rules: preview before paying, never confirm "
        "without the user's authorization, and never ask for API keys in chat. "
        "Its commands make no Datadog or provider call, so they are allowed with "
        "PUP_READ_ONLY set."
    ),
    no_args_is_help=True,
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


@skill_app.command(
    "show",
    short_help="Print the skill's path and content.",
    help=(
        f"Print where the bundled `{skill.SKILL_NAME}` skill is and its SKILL.md content, "
        f"for an agent to read or a person to copy. {_LOCAL_ONLY}"
    ),
)
@run_command
def show() -> dict[str, Any]:
    path = skill.skill_path()
    return {
        "name": skill.SKILL_NAME,
        "path": str(path),
        "content": skill.skill_text(),
    }


@skill_app.command(
    "install",
    short_help="Install the skill for your coding agents.",
    help=(
        f"Install the `{skill.SKILL_NAME}` skill. When npx is on PATH and --dir is not given, "
        "runs `npx --yes skills add <bundled skills directory> --skill "
        f"{skill.SKILL_NAME}`, passing on --agent, --global and --yes, so the skills CLI "
        "installs it for the agents you choose (it may prompt). Otherwise copies SKILL.md to "
        f".agents/skills/{skill.SKILL_NAME}/ and .claude/skills/{skill.SKILL_NAME}/ under "
        "the project directory (--dir, default: the current directory) or, with --global, "
        "your home directory; a different copy already there is replaced only with --yes. "
        f"{_NO_REMOTE_CALLS}"
    ),
)
@run_command
def install(
    agents: Annotated[
        list[str] | None,
        typer.Option(
            "--agent",
            "-a",
            help="An agent to install for, as the skills CLI names it (e.g. claude-code); "
            "repeat for more.",
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
            help="Skip the skills CLI's prompts; without npx, replace a different copy.",
        ),
    ] = False,
    directory: Annotated[
        Path | None,
        typer.Option(
            "--dir",
            help="Copy the skill into this project directory instead of running npx.",
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
    app.add_typer(skill_app, name="skill")
