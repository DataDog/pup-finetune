# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Python checks the user writes: a file of plain functions, loaded and called here.

:func:`load_checks` runs the file as a module (its sha256 names it, so an edited file is a
new module) and returns a :class:`CheckModule`; :func:`call` runs one check and reads its
answer: ``True`` passes, ``False`` fails, ``(False, "why")`` fails with a reason and
``None`` means the check does not apply. Checks run in this process, with the user's
privileges, one at a time (a lock), on deep copies of their arguments. Anything else they
return, and anything they raise, is a :class:`CheckError` naming the check.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final

from pup_finetune.data.schema import lone_surrogate
from pup_finetune.display import display_path
from pup_finetune.errors import UsageError
from pup_finetune.redaction import redact

__all__ = ["REASON_LIMIT", "CheckError", "CheckModule", "CheckVerdict", "call", "load_checks"]

REASON_LIMIT: Final = 500
_LOCK: Final = threading.Lock()

Check = Callable[..., Any]


class CheckError(Exception):
    """A check raised, or returned something other than its contract allows."""

    def __init__(self, check: str, message: str) -> None:
        super().__init__(f"check {check}: {message}")
        self.check = check
        self.message = message


@dataclass(frozen=True, slots=True)
class CheckVerdict:
    """A check's answer: ``passed`` ``None`` when it does not apply."""

    passed: bool | None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class CheckModule:
    """A loaded check file: its path, the sha256 of its bytes and the module it ran as."""

    path: Path
    sha256: str
    module: ModuleType

    def functions(self) -> list[tuple[str, Check]]:
        """The public functions the file defines (not imports), in definition order."""
        return [
            (name, value)
            for name, value in vars(self.module).items()
            if inspect.isfunction(value)
            and value.__module__ == self.module.__name__
            and not name.startswith("_")
        ]

    def function(self, name: str, arity: int) -> Check:
        """The function ``name``, which must take ``arity`` positional arguments."""
        found = dict(self.functions()).get(name)
        if found is None:
            raise UsageError(
                f"{display_path(self.path)} defines no public function {name!r}",
                hint="name a function the file defines (not one it imports)",
            )
        try:
            inspect.signature(found).bind(*([None] * arity))
        except TypeError:
            raise UsageError(
                f"the check {name} in {display_path(self.path)} must take {arity} "
                f"argument{'' if arity == 1 else 's'}",
            ) from None
        return found


def load_checks(path: Path) -> CheckModule:
    """Run the check file at ``path`` as a module; a :class:`UsageError` when it is missing
    or cannot be loaded."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise UsageError(
            f"the check file {display_path(path)} cannot be read: {exc.strerror or exc}"
        ) from None
    sha256 = hashlib.sha256(data).hexdigest()
    name = f"pup_finetune_user_checks_{sha256[:12]}"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise UsageError(f"{display_path(path)} is not a Python file")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            del sys.modules[name]
            first = (str(exc).splitlines() or [""])[0]
            raise UsageError(
                f"{display_path(path)} could not be loaded: {type(exc).__name__}: {redact(first)}"
            ) from None
    return CheckModule(path=path, sha256=sha256, module=module)


def call(check: str, function: Check, *args: Any) -> CheckVerdict:
    """Run ``function`` (the check ``check``) on deep copies of ``args``; its verdict."""
    with _LOCK:
        try:
            answer = function(*copy.deepcopy(args))
        except Exception as exc:
            raise CheckError(check, redact(f"{type(exc).__name__}: {exc}")) from None
    if answer is None or isinstance(answer, bool):
        return CheckVerdict(answer)
    if (
        isinstance(answer, tuple)
        and len(answer) == 2
        and isinstance(answer[0], bool)
        and isinstance(answer[1], str)
    ):
        if (surrogate := lone_surrogate(answer[1])) is not None:
            raise CheckError(
                check,
                f"returned a reason with a lone surrogate ({surrogate}); UTF-8 cannot encode it",
            )
        return CheckVerdict(answer[0], answer[1].strip()[:REASON_LIMIT])
    raise CheckError(
        check,
        f"returned {type(answer).__name__}, not True, False, None or (bool, reason)",
    )
