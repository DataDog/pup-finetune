# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The working directory's ``runs/`` and ``datasets/``: where runs go by default, how a
bare run id names one, and the ``.gitignore`` that keeps both out of git.

``train`` without ``--run-dir`` trains in ``runs/<run id>`` (:func:`default_run_dir`), and
every command that takes a run directory also takes a bare run id (:func:`find_run_dir`): a
path that exists wins, then ``runs/<id>``. :func:`run_dirs` lists the runs under a directory,
newest first, and :func:`latest_run_dir` is what ``status`` shows when it is given no run.

Both directories hold trace-derived content, and the working directory may be a git
repository: when a command creates ``runs/`` or ``datasets/`` (:func:`ignore_new_parent`), it
writes a ``.gitignore`` holding ``*`` there, and the command's result says so in its
``notes``; one that exists without a ``.gitignore`` in a git work tree that does not ignore
it gets a warning instead.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from pup_finetune.display import display_path
from pup_finetune.errors import FinetuneError
from pup_finetune.runs import RUN_FILE, RunDir, is_run_id
from pup_finetune.storage import atomic_write_text

__all__ = [
    "DATASETS_DIR",
    "IGNORED_DIRS",
    "RUNS_DIR",
    "FoundRun",
    "ParentSetup",
    "RunsListing",
    "default_run_dir",
    "find_run_dir",
    "ignore_new_parent",
    "latest_run_dir",
    "list_runs",
    "run_dirs",
]

RUNS_DIR: Final = Path("runs")
"""Where runs go by default, relative to the working directory."""

DATASETS_DIR: Final = Path("datasets")
"""Where ``dataset preview`` suggests pulling into, relative to the working directory."""

IGNORED_DIRS: Final = (RUNS_DIR, DATASETS_DIR)
"""The directories that get a ``.gitignore`` when a command creates them."""

GITIGNORE: Final = ".gitignore"


def default_run_dir(run_id: str) -> Path:
    """Where ``train`` trains without ``--run-dir``: ``runs/<run id>``."""
    return RUNS_DIR / run_id


def _under_cwd(directory: Path) -> Path:
    return Path(os.getcwd()) / directory


@dataclass(frozen=True, slots=True)
class ParentSetup:
    """What :func:`ignore_new_parent` did and found: the ``notes`` and ``warnings`` a result
    shows (each at most one)."""

    notes: list[str]
    warnings: list[str]


def _git_work_tree(directory: Path) -> Path | None:
    """The top of the git work tree ``directory`` is in (the first ancestor holding a
    ``.git`` directory or file), or ``None``. Reads the file system only."""
    for candidate in (directory, *directory.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _ignored_by_git(directory: Path, top: Path) -> bool:
    """Whether git ignores ``directory`` (``git check-ignore``); ``False`` when git says it
    does not, or cannot say (git missing, or failing), so the caller warns."""
    git = shutil.which("git")
    if git is None:
        return False
    env = {key: value for key, value in os.environ.items() if key not in _GIT_ENV}
    try:
        done = subprocess.run(  # noqa: S603 - git, with fixed arguments
            [git, "-C", str(top), "check-ignore", "-q", "--", str(directory) + os.sep],
            capture_output=True,
            timeout=5,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


_GIT_ENV: Final = frozenset({"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"})
"""Variables that would point git at another repository than the working directory's."""


def _create_ignored(parent: Path) -> None:
    """Create ``parent`` (``0700``) with a ``.gitignore`` holding ``*``, atomically: both are
    made in a temporary directory beside it, which is then renamed into place, so ``parent``
    never exists without its ``.gitignore``. One created meanwhile by another process is
    kept (and the temporary directory removed)."""
    staging = Path(tempfile.mkdtemp(prefix=f".{parent.name}.", dir=parent.parent))
    try:
        staging.chmod(0o700)
        atomic_write_text(staging / GITIGNORE, "*\n")
        try:
            staging.rename(parent)
        except OSError:
            if not parent.is_dir():
                raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def ignore_new_parent(path: Path) -> ParentSetup:
    """Create the parent of ``path`` when it is a missing ``./runs`` or ``./datasets``
    (:data:`IGNORED_DIRS`), with a ``.gitignore`` holding ``*`` (atomically,
    :func:`_create_ignored`); a note says it did.

    A parent that exists is left as it is: it was the user's to set up. When it has no
    ``.gitignore`` and lies in a git work tree that does not ignore it, one warning says so
    (each time a run or dataset is created in it, never for reading one). Only a direct
    child counts (``runs/<id>``, ``datasets/<name>``).
    """
    parent = Path(os.path.abspath(path)).parent
    for directory in IGNORED_DIRS:
        if parent != _under_cwd(directory):
            continue
        ignore = parent / GITIGNORE
        shown = display_path(ignore)
        if not parent.exists():
            _create_ignored(parent)
            return ParentSetup([f"created {shown} so trace-derived files stay out of git"], [])
        top = _git_work_tree(parent)
        if ignore.exists() or top is None or _ignored_by_git(parent, top):
            return ParentSetup([], [])
        return ParentSetup(
            [],
            [
                f"{display_path(parent)}/ is in a git work tree and git does not ignore it, "
                "but it holds trace-derived files: add a .gitignore holding * to it, or "
                f"ignore /{directory}/ in the repository's .gitignore"
            ],
        )
    return ParentSetup([], [])


def find_run_dir(given: Path) -> tuple[Path, bool]:
    """The run directory ``given`` names, and whether it was read as a run id.

    A path that exists is that path. Otherwise a bare run id (one word that
    :func:`~pup_finetune.runs.is_run_id` accepts) names ``runs/<id>`` when that exists.
    Anything else is returned as given, so the command's own error names it.
    """
    if given.exists():
        return given, False
    text = os.fspath(given)
    if os.sep in text or (os.altsep and os.altsep in text) or not is_run_id(text):
        return given, False
    candidate = RUNS_DIR / text
    if (candidate / RUN_FILE).is_file():
        return candidate, True
    return given, False


def _age(run: RunDir) -> tuple[str, int]:
    """How new ``run`` is: when ``run.json`` says it was created (ISO 8601 in UTC, to the
    second, so it sorts as text; ``""`` when it cannot be read), then when ``run.json`` was
    written, for runs created in the same second."""
    try:
        created = str(run.manifest().get("created_at") or "")
    except FinetuneError:
        created = ""
    try:
        written = run.run_path.stat().st_mtime_ns
    except OSError:
        written = 0
    return created, written


def run_dirs(root: Path = RUNS_DIR) -> list[RunDir]:
    """The runs directly under ``root`` (directories holding ``run.json``), newest first
    (:func:`_age`, then by name); none when ``root`` does not exist."""
    if not root.is_dir():
        return []
    found = [RunDir(entry) for entry in root.iterdir() if (entry / RUN_FILE).is_file()]
    found.sort(key=lambda run: run.path.name, reverse=True)
    found.sort(key=_age, reverse=True)
    return found


def latest_run_dir(root: Path = RUNS_DIR) -> Path | None:
    """The most recently created run under ``root``, or ``None`` when there is none."""
    found = run_dirs(root)
    return found[0].path if found else None


@dataclass(frozen=True, slots=True)
class FoundRun:
    """One run of a :class:`RunsListing`, as ``runs`` prints it."""

    run_id: str
    phase: str
    model: str | None
    provider: str | None
    best_epoch: int | None
    promoted_model: str | None
    deployment: str | None
    run_dir: str
    created_at: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "model": self.model,
            "provider": self.provider,
            "best_epoch": self.best_epoch,
            "promoted_model": self.promoted_model,
            "deployment": self.deployment,
            "run_dir": self.run_dir,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class RunsListing:
    """The runs under a directory, newest first, and the ones that could not be read."""

    runs: list[FoundRun]
    warnings: list[str]


def _deployment_state(run: RunDir) -> str | None:
    """The run's deployment as its receipt records it (``READY``, ``REQUESTED``, ...),
    ``DELETED`` once undeployed, ``None`` when it was never deployed."""
    receipt = run.read_deployment()
    if receipt is None:
        return None
    if receipt.get("deleted_at"):
        return "DELETED"
    state = receipt.get("state")
    return str(state) if state is not None else None


class _Unreadable(Exception):
    """Why a run cannot be listed, in words."""


def _required(data: Any, key: str, where: str) -> Any:
    if not isinstance(data, dict) or key not in data:
        raise _Unreadable(f"{where} has no {key}")
    return data[key]


def _found(run: RunDir) -> tuple[FoundRun, list[str]]:
    """``run`` as ``runs`` lists it, and what it read in passing that is malformed."""
    manifest = run.manifest()
    for key in ("run_id", "provider"):
        _required(manifest, key, RUN_FILE)
    profile = _required(manifest, "profile", RUN_FILE)
    _required(profile, "alias", f"{RUN_FILE}'s profile")
    spec = _required(manifest, "spec", RUN_FILE)
    _required(spec, "max_epochs", f"{RUN_FILE}'s spec")
    summary = run.status()
    notes: list[str] = []
    best = summary.get("best") or {}
    promoted = summary.get("promoted_model")
    model_id = promoted.get("model_id") if isinstance(promoted, dict) else None
    if promoted is not None and not (isinstance(promoted, dict) and model_id):
        notes.append(
            f"{display_path(run.path)}: result.json's promoted_model is not a model "
            "(an object with a model_id), so none is shown"
        )
    epoch = best.get("epoch") if isinstance(best, dict) else None
    found = FoundRun(
        run_id=str(manifest.get("run_id")),
        phase=str(summary.get("phase")),
        model=summary.get("model"),
        provider=summary.get("provider"),
        best_epoch=epoch if isinstance(epoch, int) else None,
        promoted_model=str(model_id) if model_id else None,
        deployment=_deployment_state(run),
        run_dir=display_path(run.path),
        created_at=manifest.get("created_at"),
    )
    return found, notes


def _not_runs(root: Path) -> list[str]:
    """Warnings for the directories under ``root`` named like a run id that hold no
    ``run.json``: not listed, so they are named."""
    found = []
    for entry in sorted(root.iterdir(), key=lambda path: path.name):
        if entry.is_dir() and is_run_id(entry.name) and not (entry / RUN_FILE).exists():
            found.append(
                f"{display_path(entry)} has no {RUN_FILE}, so it is not listed (a run whose "
                "start failed, or not a run)"
            )
    return found


def list_runs(root: Path = RUNS_DIR) -> RunsListing:
    """Every run under ``root`` (:func:`run_dirs`), read from its files; a run that cannot
    be read is left out with a warning naming it and why, and so is a directory named like
    a run id without ``run.json``."""
    runs: list[FoundRun] = []
    warnings: list[str] = []
    for run in run_dirs(root):
        try:
            found, notes = _found(run)
        except _Unreadable as exc:
            warnings.append(f"{display_path(run.path)} cannot be read: {exc}")
            continue
        except (FinetuneError, KeyError, TypeError, AttributeError) as exc:
            warnings.append(f"{display_path(run.path)} cannot be read: {exc}")
            continue
        runs.append(found)
        warnings.extend(notes)
    if root.is_dir():
        warnings.extend(_not_runs(root))
    return RunsListing(runs, warnings)
