# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The ``datadog-finetune`` agent skill: where it ships and how it is installed.

The one source is ``skills/datadog-finetune/SKILL.md`` at the repository root (the layout the
``skills`` CLI discovers, so ``npx skills add DataDog/pup-finetune`` finds it). The wheel
force-includes it as ``pup_finetune/_skills/datadog-finetune/SKILL.md``; an editable install
has no copy in the package, so the repository's file is used instead.

``skill install`` hands the bundled directory to ``npx skills add`` when ``npx`` exists
(:func:`npx_argv`), so the skills CLI places it for every agent it knows. Without ``npx``, or
with an explicit directory, :func:`install_files` copies the file into the two locations most
agents read: ``.agents/skills/`` and ``.claude/skills/``.
"""

from __future__ import annotations

import importlib.resources
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pup_finetune import credentials
from pup_finetune import providers as _providers  # noqa: F401 - registers provider API keys
from pup_finetune.datadog.pup import SECRET_ENV
from pup_finetune.errors import StateError, UsageError
from pup_finetune.storage import atomic_write_text

SKILL_NAME: Final = "datadog-finetune"
SKILL_FILE: Final = "SKILL.md"
PACKAGED_DIR: Final = "_skills"
"""The skills directory inside the installed package (see ``pyproject.toml``)."""
REPO_DIR: Final = "skills"
"""The skills directory at the repository root, the source of truth."""
INSTALL_ROOTS: Final[tuple[str, ...]] = (".agents/skills", ".claude/skills")
"""Where the fallback install writes the skill, relative to the project or home directory."""

_FILE_MODE: Final = 0o644
"""A skill holds no secret, so installed copies are readable like any other project file."""

InstallState = Literal["created", "updated", "unchanged"]


def bundled_skills_dir() -> Path:
    """The directory holding ``datadog-finetune/SKILL.md``: the package's, else the repo's.

    Raises :class:`StateError` when neither exists (a broken install).
    """
    packaged = importlib.resources.files("pup_finetune").joinpath(PACKAGED_DIR)
    if isinstance(packaged, Path) and (packaged / SKILL_NAME / SKILL_FILE).is_file():
        return packaged
    # An editable install imports from src/, where the wheel's force-include does not apply.
    repo = Path(__file__).resolve().parents[2] / REPO_DIR
    if (repo / SKILL_NAME / SKILL_FILE).is_file():
        return repo
    raise StateError(
        f"the {SKILL_NAME} skill is missing from this install",
        hint="reinstall pup-finetune",
    )


def skill_path() -> Path:
    """The bundled ``SKILL.md``."""
    return bundled_skills_dir() / SKILL_NAME / SKILL_FILE


def skill_text() -> str:
    """The bundled ``SKILL.md``'s content."""
    return skill_path().read_text(encoding="utf-8")


def npx_argv(
    npx: str,
    source: Path,
    *,
    agents: Sequence[str] = (),
    global_: bool = False,
    yes: bool = False,
) -> list[str]:
    """The ``npx skills add`` command that installs the skill from ``source``.

    ``npx --yes`` only lets npx fetch the skills CLI without asking; the CLI's own ``-y``
    (skip its prompts) is passed only when ``yes`` is set.
    """
    argv = [npx, "--yes", "skills", "add", str(source), "--skill", SKILL_NAME]
    if global_:
        argv.append("--global")
    if yes:
        argv.append("--yes")
    for agent in agents:
        argv.extend(["--agent", agent])
    return argv


def npx_env(environ: Mapping[str, str]) -> dict[str, str]:
    """The environment for ``npx skills add``: a copy of ``environ`` without any secret.

    npx downloads and runs a package from npm, so it gets neither the Datadog secrets pup may
    hold (:data:`~pup_finetune.datadog.pup.SECRET_ENV`) nor any provider API key variable.
    """
    secret = set(SECRET_ENV) | {spec.env_var for spec in credentials.all_specs()}
    return {name: value for name, value in environ.items() if name not in secret}


@dataclass(frozen=True, slots=True)
class InstalledFile:
    """One copy the fallback install wrote, or found already up to date."""

    path: Path
    state: InstallState

    def to_json(self) -> dict[str, str]:
        return {"path": str(self.path), "state": self.state}


def install_targets(base: Path) -> list[Path]:
    """The ``SKILL.md`` paths the fallback install writes under ``base``."""
    return [base / root / SKILL_NAME / SKILL_FILE for root in INSTALL_ROOTS]


def install_files(base: Path, *, text: str, replace: bool = False) -> list[InstalledFile]:
    """Copy ``text`` to every :func:`install_targets` path under ``base``.

    A copy that differs from ``text`` (an older version, or one edited locally) is replaced
    only with ``replace``; otherwise nothing is written and :class:`UsageError` names it.
    A file that cannot be read or written raises :class:`StateError`.
    """
    targets = install_targets(base)
    current: dict[Path, str | None] = {}
    for target in targets:
        try:
            current[target] = target.read_text(encoding="utf-8")
        except FileNotFoundError:
            current[target] = None
        except (OSError, UnicodeDecodeError) as exc:
            raise StateError(f"cannot read {target}: {exc}") from exc
    differing = [path for path, old in current.items() if old is not None and old != text]
    if differing and not replace:
        raise UsageError(
            "a different copy of the skill is already installed at "
            + ", ".join(str(path) for path in differing),
            hint="pass -y to replace it with this version",
        )
    installed: list[InstalledFile] = []
    for target in targets:
        old = current[target]
        if old == text:
            installed.append(InstalledFile(target, "unchanged"))
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(target, text)
            target.chmod(_FILE_MODE)
        except OSError as exc:
            raise StateError(f"cannot write {target}: {exc}") from exc
        installed.append(InstalledFile(target, "created" if old is None else "updated"))
    return installed
