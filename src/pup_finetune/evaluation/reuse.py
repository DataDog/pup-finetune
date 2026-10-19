# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Base-model results shared by the evaluations and benchmarks of one dataset.

A dataset directory keeps a content-addressed store, ``replay_cache/`` (``0700``, files
``0600``): ``generations.jsonl``, one base answer per :func:`generation_key`, and
``verdicts.jsonl``, one judge verdict per :func:`verdict_key`. Both are append-only, the first
line of a key wins and a torn line is skipped, so deleting the store only costs new
generations.

A run reads the store once, when it plans (:meth:`ReuseIndex.load`), so its preview and its
execution see the same snapshot. :func:`plan_target` says what a target takes from it,
:func:`import_target` copies that into the run's evaluation directory before the replay, and
:func:`export_target` adds the run's own base answers and real verdicts afterwards, only
from a directory whose answers were all sampled under the inputs they are keyed by
(:func:`stamp`). Only base models are shared: they sample greedily from pinned weights, so the
same input gets the same answer, and a verdict is reused only for the same judge request.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, NamedTuple

from pup_finetune.display import display_path
from pup_finetune.errors import FinetuneError
from pup_finetune.evaluation.cases import ReplayCase
from pup_finetune.evaluation.judge import VERDICT_SCHEMA, build_messages
from pup_finetune.evaluation.metrics import score_case
from pup_finetune.evaluation.replay import EvalDir, PairKey
from pup_finetune.inference import JudgeConfig
from pup_finetune.providers.base import ModelProfile
from pup_finetune.redaction import redact
from pup_finetune.storage import (
    DirectoryLock,
    append_jsonl,
    atomic_write_json,
    canonical_json,
    finite_json,
    make_private_dir,
    parse_jsonl_line,
    read_json,
)

__all__ = [
    "CACHE_DIR_NAME",
    "REUSE_VERSION",
    "ReuseIndex",
    "TargetReuse",
    "Wanted",
    "export_target",
    "generation_key",
    "import_target",
    "key_inputs",
    "not_shared",
    "plan_target",
    "stamp",
    "stamped",
    "store_path",
    "verdict_key",
]

REUSE_VERSION: Final = "1"
"""Bumped when sampling or parsing changes in a way the renderer identity does not cover."""
CACHE_DIR_NAME: Final = "replay_cache"
GENERATIONS_FILE: Final = "generations.jsonl"
VERDICTS_FILE: Final = "verdicts.jsonl"
STAMP_FILE: Final = "reuse.json"

_GENERATION_FIELDS: Final = (
    "message",
    "format_valid",
    "prompt_tokens",
    "completion_tokens",
    "latency_s",
    "stop_reason",
    "sampled_at",
)


def store_path(dataset_dir: Path) -> Path:
    """The store of the dataset in ``dataset_dir``."""
    return Path(dataset_dir) / CACHE_DIR_NAME


def _digest(payload: Mapping[str, Any]) -> str:
    text = canonical_json(finite_json(dict(payload)), ensure_ascii=True)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def key_inputs(
    provider: str, profile: ModelProfile, rendering: Mapping[str, Any], max_tokens: int
) -> dict[str, Any]:
    """What a base answer is sampled under: every input of its :func:`generation_key` but
    the case."""
    return {
        "v": REUSE_VERSION,
        "provider": provider,
        "base_model": profile.base_model,
        "max_seq_len": profile.max_seq_len,
        "rendering": dict(rendering),
        "max_tokens": max_tokens,
    }


def generation_key(inputs: Mapping[str, Any], case: ReplayCase) -> str:
    """The key of ``case``'s base answer sampled under ``inputs`` (:func:`key_inputs`):
    everything the base model is sent and sampled with, and nothing that only names the case
    (its id, split or reference)."""
    return _digest({**inputs, "prefix": list(case.prefix), "tools": list(case.tools)})


def verdict_key(judge: JudgeConfig, case: ReplayCase, message: Mapping[str, Any]) -> str:
    """The key of the judge's verdict on ``message`` for ``case``: the judge's settings and
    the exact request it is sent."""
    return _digest(
        {
            "v": REUSE_VERSION,
            "judge": judge.identity(),
            "schema": VERDICT_SCHEMA,
            "request": build_messages(case, message),
        }
    )


def _lines(path: Path) -> Iterator[dict[str, Any]]:
    """The JSON objects of a store file, read as a stream; a torn line is skipped."""
    if not path.exists():
        return
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                row, problem = parse_jsonl_line(line)
                if problem is None:
                    yield row


def _key(row: Mapping[str, Any]) -> str | None:
    key = row.get("key")
    return key if isinstance(key, str) else None


def _answer(record: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The message of a well-formed answer, the only kind a judge is asked about."""
    if record is None or record.get("format_valid") is not True:
        return None
    message = record.get("message")
    return message if isinstance(message, Mapping) else None


class Wanted(NamedTuple):
    """What one target of a plan looks up: its cases, each one's generation key, and the
    answers its directory already holds (whose verdicts the store may have too)."""

    target: str
    cases: Sequence[ReplayCase]
    keys: Mapping[str, str]
    generations: Mapping[PairKey, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ReuseIndex:
    """The store's lines a plan can use, read once: answers by generation key and verdicts
    by verdict key. Empty when the store is missing; empty with a warning when it cannot be
    read."""

    store: Path
    generations: dict[str, dict[str, Any]] = field(default_factory=dict)
    verdicts: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @classmethod
    def load(cls, dataset_dir: Path, judge: JudgeConfig, wanted: Sequence[Wanted]) -> ReuseIndex:
        """Two passes over the store: the answers ``wanted`` needs, then the verdicts of
        those answers and of the answers the targets' directories hold."""
        store = store_path(dataset_dir)
        needed = {key for want in wanted for key in want.keys.values()}
        generations: dict[str, dict[str, Any]] = {}
        verdicts: dict[str, dict[str, Any]] = {}
        try:
            for row in _lines(store / GENERATIONS_FILE):
                key = _key(row)
                if key in needed and key not in generations and _has_message(row):
                    generations[key] = row
            judged = set()
            for want in wanted:
                for case in want.cases:
                    own = want.generations.get((want.target, case.id))
                    message = _answer(own or generations.get(want.keys.get(case.id, "")))
                    if message is not None:
                        judged.add(verdict_key(judge, case, message))
            for row in _lines(store / VERDICTS_FILE) if judged else ():
                key = _key(row)
                if key in judged and key not in verdicts and isinstance(row.get("judgment"), dict):
                    verdicts[key] = row
        except OSError as exc:
            reason = redact(exc.strerror or str(exc))
            return cls(
                store,
                warnings=(
                    f"base results of earlier runs were not reused: {display_path(store)} "
                    f"cannot be read ({reason})",
                ),
            )
        return cls(store, generations, verdicts)


def _has_message(row: Mapping[str, Any]) -> bool:
    return isinstance(row.get("message"), dict) and isinstance(row.get("format_valid"), bool)


@dataclass(frozen=True, slots=True)
class TargetReuse:
    """What one target takes from the store, by case id: answers, and verdicts on answers
    (taken or its own) judged by ``judge``."""

    target: str
    judge: JudgeConfig
    generations: dict[str, dict[str, Any]]
    verdicts: dict[str, dict[str, Any]]

    def counts(self) -> dict[str, int]:
        return {"generations": len(self.generations), "verdicts": len(self.verdicts)}

    @property
    def oldest_sampled_at(self) -> str | None:
        stamps = [
            str(row["sampled_at"]) for row in self.generations.values() if row.get("sampled_at")
        ]
        return min(stamps, default=None)

    @property
    def sources(self) -> list[str]:
        """The commands that made the answers taken, e.g. ``evaluate ft-...``, once each."""
        found: list[str] = []
        for row in self.generations.values():
            source = row.get("source")
            source = source if isinstance(source, Mapping) else {}
            name = " ".join(
                str(part) for part in (source.get("command"), source.get("run_id")) if part
            )
            if name and name not in found:
                found.append(name)
        return found


def plan_target(
    index: ReuseIndex,
    target: str,
    cases: Sequence[ReplayCase],
    keys: Mapping[str, str],
    judge: JudgeConfig,
    generations: Mapping[PairKey, Mapping[str, Any]],
    results: Mapping[PairKey, Mapping[str, Any]],
) -> TargetReuse:
    """What ``target`` takes from ``index`` for its pairs without an ``ok`` result in a
    directory holding ``generations`` and ``results``: the answer, unless the directory has
    one, then the verdict on that answer. Once :func:`import_target` copied it, nothing."""
    taken: dict[str, dict[str, Any]] = {}
    judged: dict[str, dict[str, Any]] = {}
    for case in cases:
        pair = (target, case.id)
        if results.get(pair, {}).get("status") == "ok":
            continue
        record = generations.get(pair)
        if record is None:
            record = index.generations.get(keys.get(case.id, ""))
            if record is None:
                continue
            taken[case.id] = record
        message = _answer(record)
        verdict = None if message is None else index.verdicts.get(verdict_key(judge, case, message))
        if verdict is not None:
            judged[case.id] = verdict
    return TargetReuse(target=target, judge=judge, generations=taken, verdicts=judged)


def import_target(
    eval_dir: EvalDir, reuse: TargetReuse, cases: Sequence[ReplayCase]
) -> dict[str, int]:
    """Copy ``reuse`` into ``eval_dir``, whose lock the caller holds; the counts copied.

    An answer keeps its ``sampled_at`` and gets ``reused_from``; a verdict becomes an ``ok``
    result with ``reused`` and metrics computed here. A pair the directory filled meanwhile
    is skipped.
    """
    generations, results = eval_dir.read_generations(), eval_dir.read_results()
    copied = {"generations": 0, "verdicts": 0}
    for case in cases:
        pair = (reuse.target, case.id)
        if results.get(pair, {}).get("status") == "ok":
            continue
        record = generations.get(pair)
        stored = reuse.generations.get(case.id)
        if record is None and stored is not None:
            record = {
                "target": reuse.target,
                "case_id": case.id,
                **{name: stored.get(name) for name in _GENERATION_FIELDS},
                "reused_from": stored.get("source"),
            }
            eval_dir.append_generation(record)
            copied["generations"] += 1
        verdict = reuse.verdicts.get(case.id)
        message = _answer(record)
        if verdict is None or message is None:
            continue
        if verdict.get("key") != verdict_key(reuse.judge, case, message):
            continue
        metrics = score_case(dict(message), case.reference, case.tools, format_valid=True)
        eval_dir.append_result(
            {
                "target": reuse.target,
                "case_id": case.id,
                "row_id": case.row_id,
                "trace_id": case.trace_id,
                "message_index": case.message_index,
                "status": "ok",
                "error": None,
                "metrics": dict(metrics),
                "judgment": verdict["judgment"],
                "judged_at": verdict.get("judged_at"),
                "reused": True,
            }
        )
        copied["verdicts"] += 1
    return copied


def stamp(eval_dir: EvalDir, inputs: Mapping[str, Any]) -> None:
    """Record in ``eval_dir`` (its lock held) the ``inputs`` its base answers are sampled
    under, while it holds none: a stamp names its first answers' inputs, and a directory made
    before stamps gets none."""
    if not eval_dir.read_generations():
        atomic_write_json(eval_dir.path / STAMP_FILE, finite_json(dict(inputs)))


def stamped(eval_dir: EvalDir) -> bool:
    """Whether ``eval_dir`` has a :func:`stamp`."""
    return (eval_dir.path / STAMP_FILE).exists()


def _made_under(eval_dir: EvalDir, inputs: Mapping[str, Any], *, legacy: bool) -> bool:
    """Whether ``eval_dir``'s answers were all sampled under ``inputs`` (see
    :func:`export_target`)."""
    path = eval_dir.path / STAMP_FILE
    if path.exists():
        stamped = read_json(path)
    elif legacy:
        stamped = {**inputs, "v": "1"}
    else:
        return False
    return canonical_json(stamped) == canonical_json(finite_json(dict(inputs)))


def not_shared(exc: FinetuneError | OSError, rerun: str) -> str:
    """The warning of a run whose base results could not be added to the store; ``rerun``
    is the command that adds them."""
    reason = exc.message if isinstance(exc, FinetuneError) else exc.strerror or str(exc)
    return f"base results were not shared: {redact(reason)}; run `{rerun}` to share them"


def export_target(
    dataset_dir: Path,
    eval_dir: EvalDir,
    target: str,
    cases: Sequence[ReplayCase],
    inputs: Mapping[str, Any],
    judge: JudgeConfig,
    source: Mapping[str, str | None],
    rerun: str,
    *,
    legacy: bool = False,
) -> str | None:
    """Add ``target``'s own answers in ``eval_dir`` (not those it reused), keyed under
    ``inputs``, and its real verdicts (no error, a judge answer kept) to the store; a key it
    has is skipped. ``rerun`` is the command that shares them when this cannot.

    Only a directory whose answers were all sampled under ``inputs`` shares: its
    :func:`stamp` holds them, or, with ``legacy`` (an identity that pins every input but the
    version), it has no stamp and ``inputs`` are version 1's. Takes the store's lock once:
    when it is busy or the store cannot be written, nothing is written and the warning is
    returned.
    """
    store = store_path(dataset_dir)
    generations, results = eval_dir.read_generations(), eval_dir.read_results()
    try:
        if not _made_under(eval_dir, inputs, legacy=legacy):
            return None
        make_private_dir(store)
        with DirectoryLock(store):
            answers = {_key(row) for row in _lines(store / GENERATIONS_FILE)}
            judged = {_key(row) for row in _lines(store / VERDICTS_FILE)}
            new_answers: list[dict[str, Any]] = []
            new_verdicts: list[dict[str, Any]] = []
            for case in cases:
                record = generations.get((target, case.id))
                if record is None or not _has_message(record):
                    continue
                key = generation_key(inputs, case)
                if "reused_from" not in record and key not in answers:
                    answers.add(key)
                    fields = {name: record.get(name) for name in _GENERATION_FIELDS}
                    new_answers.append({"key": key, **fields, "source": dict(source)})
                result = results.get((target, case.id)) or {}
                judgment = result.get("judgment")
                message = _answer(record)
                if (
                    message is None
                    or result.get("status") != "ok"
                    or not isinstance(judgment, Mapping)
                    or judgment.get("error") is not None
                    or judgment.get("raw") is None
                ):
                    continue
                key = verdict_key(judge, case, message)
                if key not in judged:
                    judged.add(key)
                    new_verdicts.append(
                        {
                            "key": key,
                            "judgment": dict(judgment),
                            "judged_at": result.get("judged_at"),
                        }
                    )
            if new_answers:
                append_jsonl(store / GENERATIONS_FILE, new_answers, ensure_ascii=True)
            if new_verdicts:
                append_jsonl(store / VERDICTS_FILE, new_verdicts, ensure_ascii=True)
    except (FinetuneError, OSError) as exc:
        return not_shared(exc, rerun)
    return None
