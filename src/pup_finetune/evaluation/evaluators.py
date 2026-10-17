# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The user's own evaluators, from ``evaluate --evaluators FILE`` (a TOML spec).

A spec lists LLM-judge evaluators (:class:`JudgeEvaluator`: a rubric, an optional judge
model and optional graded examples that calibrate it) and Python check evaluators
(:class:`CheckEvaluator`: a function in a local file, loaded by
:mod:`pup_finetune.user_checks`), and a ``headline``: ``judge`` (the built-in judge), an
evaluator's name, or ``all`` (the built-in judge and every evaluator pass). Paths are
relative to the spec.

An evaluation directory freezes its evaluators by name in ``evaluators.json``
(:func:`attach_evaluators`): a judge by its rubric's sha256 and its model, a check by its
function and its file's sha256. The list only grows: a rerun may add evaluators, which score
the stored answers, but a dropped or changed one is a
:class:`~pup_finetune.errors.SettingsConflict`. The headline is not frozen.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from pup_finetune.display import display_path, quoted
from pup_finetune.errors import SettingsConflict, StateError, UsageError
from pup_finetune.evaluation.cases import ReplayCase
from pup_finetune.evaluation.judge import CUT_OFF_REASON, MALFORMED_REASON, Example, judge
from pup_finetune.evaluation.metrics import MetricName
from pup_finetune.inference import ChatCompleter, JudgeConfig
from pup_finetune.providers.base import Generation
from pup_finetune.runtime import Runtime
from pup_finetune.storage import (
    atomic_write_json,
    canonical_json,
    finite_json,
    free_sibling,
    identity_changes,
    read_json,
)
from pup_finetune.user_checks import CheckModule, call, load_checks

__all__ = [
    "ALL",
    "EVALUATORS_FILE",
    "JUDGE",
    "CheckEvaluator",
    "Evaluator",
    "EvaluatorSet",
    "JudgeEvaluator",
    "Scorer",
    "attach_evaluators",
    "evaluators_conflict",
    "load_spec",
    "resolve_evaluators",
    "stored_evaluators",
]

EVALUATORS_FILE: Final = "evaluators.json"
JUDGE: Final = "judge"
"""The headline of the built-in judge."""
ALL: Final = "all"
"""The composite headline: the built-in judge and every evaluator pass (or do not apply)."""
NAME: Final = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
RESERVED: Final = frozenset(
    {JUDGE, ALL, "judge_pass", "trajectory_agreement", "completed_actions", *MetricName}
)

Scorer = Callable[[ReplayCase, Generation], dict[str, Any]]
"""Scores one generation of a case: a ``results.jsonl`` evaluator entry; raises on an
outage or a :class:`~pup_finetune.user_checks.CheckError`."""


def _entry(status: str, passed: bool | None, reason: str, **rest: Any) -> dict[str, Any]:
    return {
        "status": status,
        "passed": passed,
        "reason": reason,
        "raw": rest.get("raw"),
        "no_verdict": rest.get("no_verdict"),
        "error": None,
    }


def _malformed(generation: Generation) -> dict[str, Any]:
    return _entry(
        "ok", False, CUT_OFF_REASON if generation.stop_reason == "length" else MALFORMED_REASON
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class JudgeEvaluator:
    """An LLM judge that grades each answer by ``rubric``, calibrated on ``examples``."""

    name: str
    rubric: str
    config: JudgeConfig
    examples: tuple[Example, ...] = ()

    @property
    def rubric_sha256(self) -> str:
        return hashlib.sha256(self.rubric.encode("utf-8")).hexdigest()

    def identity(self) -> dict[str, Any]:
        """What freezes it: a changed rubric or model is another evaluator."""
        return {
            "name": self.name,
            "type": "judge",
            "rubric_sha256": self.rubric_sha256,
            "model": self.config.identity(),
        }

    def record(self) -> dict[str, Any]:
        return {**self.identity(), "rubric": self.rubric}

    def scorer(self, client: ChatCompleter) -> Scorer:
        def score(case: ReplayCase, generation: Generation) -> dict[str, Any]:
            verdict = judge(
                case,
                generation.message,
                client,
                self.config,
                format_valid=generation.format_valid,
                cut_off=generation.stop_reason == "length",
                rubric=self.rubric,
            )
            return _entry(
                "ok", verdict.passed, verdict.reason, raw=verdict.raw, no_verdict=verdict.error
            )

        return score


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckEvaluator:
    """A Python check, ``function(case, candidate)`` in ``file``; ``module`` is loaded only
    when it has cases to score."""

    name: str
    file: Path
    function: str
    file_sha256: str
    module: CheckModule | None = None

    def identity(self) -> dict[str, Any]:
        """What freezes it: the whole file, since its helpers decide too."""
        return {
            "name": self.name,
            "type": "check",
            "function": self.function,
            "file_sha256": self.file_sha256,
        }

    def record(self) -> dict[str, Any]:
        return {**self.identity(), "file": str(self.file)}

    def loaded(self) -> CheckEvaluator:
        """This check with its module loaded; a :class:`UsageError` when its file is gone
        or changed since it was frozen."""
        if self.module is not None:
            return self
        try:
            data = self.file.read_bytes()
        except OSError:
            data = None
        if data is None or hashlib.sha256(data).hexdigest() != self.file_sha256:
            what = "is missing" if data is None else "changed"
            raise UsageError(
                f"the check file {display_path(self.file)} of the evaluator {self.name} {what} "
                "since it was attached, and it has cases left to score",
                hint="restore it (a new version goes in a new file, under another name), or "
                "pass a new --out-dir",
            )
        module = load_checks(self.file)
        module.function(self.function, 2)
        return CheckEvaluator(
            name=self.name,
            file=self.file,
            function=self.function,
            file_sha256=self.file_sha256,
            module=module,
        )

    def check(self, case: ReplayCase, candidate: Mapping[str, Any]) -> dict[str, Any]:
        """The check's entry for ``candidate``; raises its
        :class:`~pup_finetune.user_checks.CheckError`."""
        if self.module is None:
            raise StateError(f"the check {self.name} was not loaded")
        verdict = call(self.name, self.module.function(self.function, 2), case.to_json(), candidate)
        if verdict.passed is None:
            return _entry("not_applicable", None, verdict.reason)
        return _entry("ok", verdict.passed, verdict.reason)

    def scorer(self) -> Scorer:
        def score(case: ReplayCase, generation: Generation) -> dict[str, Any]:
            if not generation.format_valid:
                return _malformed(generation)
            return self.check(case, generation.message)

        return score


Evaluator = JudgeEvaluator | CheckEvaluator
Source = Literal["flag", "evaluation directory"]


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluatorSet:
    """The evaluators of an evaluation, their headline, and where they came from."""

    evaluators: tuple[Evaluator, ...]
    headline: str
    spec: Path | None
    source: Source

    @property
    def names(self) -> list[str]:
        return [evaluator.name for evaluator in self.evaluators]

    @property
    def judges(self) -> list[JudgeEvaluator]:
        return [e for e in self.evaluators if isinstance(e, JudgeEvaluator)]

    @property
    def checks(self) -> list[CheckEvaluator]:
        return [e for e in self.evaluators if isinstance(e, CheckEvaluator)]

    @property
    def offline(self) -> bool:
        return all(judge.config.provider == "fake" for judge in self.judges)

    @property
    def pinned(self) -> Path | None:
        """The spec a command repeats: only one ``--evaluators`` loaded. A stored set
        continues without it, and its file may since list evaluators no preview showed."""
        return self.spec if self.source == "flag" else None

    def record(self) -> dict[str, Any]:
        """``evaluators.json`` for these evaluators alone."""
        return {
            "evaluators": [evaluator.record() for evaluator in self.evaluators],
            "headline": self.headline,
            "spec": None if self.spec is None else str(self.spec),
        }

    def scorers(self, clients: Mapping[str, ChatCompleter]) -> dict[str, Scorer]:
        """Each evaluator's scorer, its judge on the client of its provider in
        ``clients``; one whose client is not open raises when called."""
        found: dict[str, Scorer] = {}
        for evaluator in self.evaluators:
            if isinstance(evaluator, CheckEvaluator):
                found[evaluator.name] = evaluator.scorer()
            elif (client := clients.get(evaluator.config.provider)) is not None:
                found[evaluator.name] = evaluator.scorer(client)
            else:
                found[evaluator.name] = _closed(evaluator.name)
        return found


def _closed(name: str) -> Scorer:
    def score(case: ReplayCase, generation: Generation) -> dict[str, Any]:
        raise StateError(f"the judge of the evaluator {name} is not open")

    return score


# --- the spec -----------------------------------------------------------------------------


class _JudgeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["judge"]
    model: str | None = None
    rubric: str = Field(min_length=1)
    examples: str | None = None


class _CheckSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["check"]
    file: str
    function: str | None = None


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headline: str = JUDGE
    evaluator: list[Annotated[_JudgeSpec | _CheckSpec, Field(discriminator="type")]] = Field(
        min_length=1
    )


class _ExampleLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation: list[dict[str, Any]] = Field(min_length=1)
    candidate: str | dict[str, Any]
    passed: StrictBool = Field(alias="pass")
    tools: list[dict[str, Any]] = Field(default_factory=list)
    reference: dict[str, Any] | None = None


def _problem(error: ValidationError) -> str:
    """The first problem of ``error``, at its key: ``evaluator[0].colour: ...``."""
    first = error.errors()[0]
    loc = list(first["loc"])
    if len(loc) > 2 and loc[0] == "evaluator" and loc[2] in ("judge", "check"):
        del loc[2]
    where = "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in loc)
    return f"{where.lstrip('.') or 'the spec'}: {first['msg']}"


def _judge_config(spec: str, runtime: Runtime) -> JudgeConfig:
    from pup_finetune.evaluation.fake_judge import FAKE_JUDGE_MODEL, fake_judge_config
    from pup_finetune.judges import check_available, parse_judge

    if spec.strip() == FAKE_JUDGE_MODEL:
        if not runtime.dev_mode:
            raise UsageError(
                "the fake judge is for development only",
                hint="set PUP_FINETUNE_DEV=1, or name another model for the evaluator",
            )
        return fake_judge_config()
    config = parse_judge(spec, what="evaluator judge")
    check_available(config, what="evaluator judge")
    return config


def load_spec(path: Path, *, judge: JudgeConfig, runtime: Runtime) -> EvaluatorSet:
    """The evaluators the spec at ``path`` lists, every check loaded; a judge without a
    ``model`` is ``judge``. A :class:`UsageError` names whatever is wrong."""
    path = Path(path)
    shown = display_path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise UsageError(f"the evaluators file {shown} cannot be read: {exc.strerror}") from None
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise UsageError(f"{shown} is not valid TOML: {exc}") from None
    try:
        spec = _Spec.model_validate(data)
    except ValidationError as exc:
        raise UsageError(f"{shown}: {_problem(exc)}") from None
    evaluators: list[Evaluator] = []
    for index, item in enumerate(spec.evaluator):
        where = f"{shown}: evaluator[{index}].name"
        if not NAME.match(item.name):
            raise UsageError(
                f"{where} {item.name!r} must be lowercase letters, digits and _, starting "
                "with a letter, at most 40 characters"
            )
        if item.name in RESERVED:
            raise UsageError(f"{where} {item.name!r} is reserved", hint="pick another name")
        if item.name in {evaluator.name for evaluator in evaluators}:
            raise UsageError(f"{where} {item.name!r} is given twice")
        evaluators.append(_evaluator(item, path.parent, judge=judge, runtime=runtime))
    names = {evaluator.name for evaluator in evaluators}
    if spec.headline not in names | {JUDGE, ALL}:
        raise UsageError(
            f"{shown}: headline {spec.headline!r} names no evaluator",
            hint=f"use {JUDGE!r}, {ALL!r} or the name of an evaluator in the file",
        )
    return EvaluatorSet(
        evaluators=tuple(evaluators),
        headline=spec.headline,
        spec=path.absolute(),
        source="flag",
    )


def _evaluator(
    item: _JudgeSpec | _CheckSpec, base: Path, *, judge: JudgeConfig, runtime: Runtime
) -> Evaluator:
    if isinstance(item, _CheckSpec):
        file = (base / item.file).absolute()
        module = load_checks(file)
        function = item.function or item.name
        module.function(function, 2)
        return CheckEvaluator(
            name=item.name, file=file, function=function, file_sha256=module.sha256, module=module
        )
    config = judge if item.model is None else _judge_config(item.model, runtime)
    examples = () if item.examples is None else _examples(base / item.examples)
    return JudgeEvaluator(
        name=item.name, rubric=item.rubric.strip(), config=config, examples=examples
    )


def _examples(path: Path) -> tuple[Example, ...]:
    """The graded answers in the JSON Lines file at ``path``, as synthetic cases."""
    shown = display_path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise UsageError(f"the examples file {shown} cannot be read: {exc}") from None
    found: list[Example] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = _ExampleLine.model_validate(json.loads(line))
        except ValueError as exc:
            problem = _problem(exc) if isinstance(exc, ValidationError) else str(exc)
            raise UsageError(f"{shown} line {number}: {problem}") from None
        candidate = item.candidate
        message = (
            {"role": "assistant", "content": candidate} if isinstance(candidate, str) else candidate
        )
        case_id = f"example-{len(found) + 1}"
        case = ReplayCase(
            id=case_id,
            row_id=case_id,
            trace_id=None,
            message_index=len(item.conversation),
            prefix=tuple(item.conversation),
            tools=tuple(item.tools),
            reference=item.reference or {"role": "assistant", "content": ""},
        )
        found.append(Example(case, message, item.passed))
    if not ({True, False} <= {example.expected_pass for example in found}):
        raise UsageError(
            f'{shown} needs at least one example with "pass": true and one with '
            '"pass": false, so the judge is checked both ways'
        )
    return tuple(found)


# --- the directory's record ---------------------------------------------------------------


def stored_evaluators(path: Path) -> dict[str, Any] | None:
    """The evaluators attached at ``path`` (an evaluation directory's ``evaluators.json``),
    or ``None`` when none are."""
    if not path.exists():
        return None
    stored = read_json(path)
    if not isinstance(stored, dict) or not isinstance(stored.get("evaluators"), list):
        raise StateError(f"{path}: expected a JSON object with a list of evaluators")
    return stored


def resolve_evaluators(
    spec: Path | None,
    stored: Mapping[str, Any] | None,
    *,
    judge: JudgeConfig,
    runtime: Runtime,
    pending: Callable[[str], bool] = lambda name: True,
) -> EvaluatorSet | None:
    """The evaluators of an evaluation: the spec ``--evaluators`` names, else those
    ``stored`` (``evaluators.json``) records, else ``None``.

    Stored judges are resolved from their stored model as it is called now; stored checks
    are loaded only when ``pending`` says they have cases left to score, from their stored
    file, which must be unchanged.
    """
    if spec is not None:
        return load_spec(spec, judge=judge, runtime=runtime)
    if stored is None:
        return None
    found: list[Evaluator] = []
    try:
        for item in stored["evaluators"]:
            name = str(item["name"])
            if item["type"] == "judge":
                model = JudgeConfig.model_validate(item["model"]).spec
                config = _judge_config(model, runtime)
                found.append(JudgeEvaluator(name=name, rubric=str(item["rubric"]), config=config))
                continue
            check = CheckEvaluator(
                name=name,
                file=Path(str(item["file"])),
                function=str(item["function"]),
                file_sha256=str(item["file_sha256"]),
            )
            found.append(check.loaded() if pending(name) else check)
    except (KeyError, TypeError, ValueError) as exc:
        raise StateError(f"{EVALUATORS_FILE} holds an evaluator it cannot read: {exc}") from None
    spec_path = stored.get("spec")
    return EvaluatorSet(
        evaluators=tuple(found),
        headline=str(stored.get("headline") or JUDGE),
        spec=Path(spec_path) if isinstance(spec_path, str) else None,
        source="evaluation directory",
    )


def _identity(item: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key not in ("rubric", "file")}


def evaluators_conflict(
    stored: Mapping[str, Any] | None, wanted: EvaluatorSet, *, out_dir: Path
) -> SettingsConflict | None:
    """Why ``wanted`` cannot score in ``out_dir``, which has ``stored`` attached: an
    evaluator it drops or changes, with a free ``--out-dir``; ``None`` when it can."""
    if stored is None:
        return None
    current = {evaluator.name: evaluator for evaluator in wanted.evaluators}
    fresh = quoted(free_sibling(out_dir))
    for item in stored.get("evaluators") or []:
        name = item.get("name")
        evaluator = current.get(name)
        if evaluator is None:
            return SettingsConflict(
                f"{display_path(out_dir)} has the evaluator {name}, which "
                f"{display_path(wanted.spec or out_dir)} leaves out, and an attached "
                "evaluator is never dropped",
                hint=f"add {name} back to the spec, or pass --out-dir {fresh}",
            )
        before, now = _identity(item), finite_json(evaluator.identity())
        if canonical_json(before) != canonical_json(now):
            return SettingsConflict(
                f"the evaluator {name} changed since it was attached to "
                f"{display_path(out_dir)} (changed: {', '.join(identity_changes(before, now))})",
                hint=f"keep {name} as it was attached and add the new version under another "
                "name, which scores the stored answers (a check in a new file, since its whole "
                f"file is frozen), or pass --out-dir {fresh}",
            )
    return None


def attach_evaluators(path: Path, wanted: EvaluatorSet | None) -> None:
    """Attach ``wanted`` at ``path`` (an evaluation directory's ``evaluators.json``): add
    the evaluators it has not, and set its headline; raise :func:`evaluators_conflict`
    otherwise. ``None`` (an evaluation planned without any) raises when some are attached.
    Call it under the directory's lock."""
    stored = stored_evaluators(path)
    if wanted is None:
        if stored is not None and stored["evaluators"]:
            raise SettingsConflict(
                f"{path.parent} has evaluators this evaluation was planned without",
                hint="run the command again: it continues them too",
            )
        return
    conflict = evaluators_conflict(stored, wanted, out_dir=path.parent)
    if conflict is not None:
        raise conflict
    current = {evaluator.name: evaluator.record() for evaluator in wanted.evaluators}
    # The identity is unchanged, so its file and rubric follow the spec now attaching it.
    kept = [current.get(item.get("name"), item) for item in (stored or {}).get("evaluators", [])]
    known = {item.get("name") for item in kept}
    record = {
        **wanted.record(),
        "evaluators": [*kept, *(e.record() for e in wanted.evaluators if e.name not in known)],
    }
    if stored is None or canonical_json(stored) != canonical_json(finite_json(record)):
        atomic_write_json(path, finite_json(record))


def scored(entries: object, name: str) -> bool:
    """Whether a result's ``evaluators`` hold a final entry for ``name``."""
    entry = entries.get(name) if isinstance(entries, Mapping) else None
    return isinstance(entry, Mapping) and entry.get("status") in ("ok", "not_applicable")


def unscored(entries: object, names: Sequence[str]) -> list[str]:
    """The ``names`` a result's ``evaluators`` hold no final entry for."""
    return [name for name in names if not scored(entries, name)]
