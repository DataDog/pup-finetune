# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Publish a replay evaluation as Agent Observability experiments, through :class:`Pup`.

:class:`ExperimentsPublisher` is the :class:`~pup_finetune.evaluation.replay.Publisher` of
an evaluation whose dataset was pushed (``push.json``, from ``dataset push``). For each
target (``base`` and ``tuned``) it creates one experiment on the pushed dataset and version
(``pup llm-obs experiments create --file``), then submits the evaluation to it as spans and
metrics (``pup api v2/llm-obs/v1/experiments/<id>/events``):

- one root span per test-split row, ``replay_trajectory``, linked to the row's dataset record
  (``dataset_record_id``), with the record's input and expected output and, as its output,
  the steps replayed; its metrics are ``trajectory_agreement`` (the mean judge pass of the
  row's cases) and ``completed_actions`` (the cases judged);
- one child span per case, ``replay_assistant``, in the root's trace, with the prefix and
  tools in, the generated message out and the recorded message expected; its metrics are a
  boolean per deterministic metric that applies and ``judge_pass``, with the judge's reason.
  A case that ended in an error is an ``error`` span with its message and no metric, as it
  counts in no rate of the local summary.

Metrics are stamped with the time they are submitted, since the API refuses timestamps more
than a day old. At most :data:`SPANS_PER_REQUEST` spans go in one request, with their
metrics, and a row's spans stay together where they fit.

Publishing resumes. Span and trace ids are derived (uuid5) from the evaluation's identity,
the target and the row or case, so they are the same in every run of one evaluation. The
evaluation directory keeps, both ``0600``:

- ``experiments.json`` - the experiment of each target (``experiment_id``, ``name``), saved
  as soon as it is created and reused afterwards, so a rerun never creates another (an entry
  for another dataset, after the dataset was pushed again, or one that cannot be read is
  replaced by a new experiment, with a warning). Before the create, the target's entry is a
  pending one (``pending_nonce``, ``dataset_id``): the nonce, random, goes in the
  experiment's description marker, so a rerun after a crash between the create and the save
  looks for that experiment and adopts it, or creates it with the same nonce. The nonce
  keeps another evaluation directory of the same evaluation from adopting it;
- ``published.json`` - the spans submitted to each experiment (span id to a digest of what was
  sent), saved after every request, so a rerun sends only the spans not sent yet. A span sent
  with an earlier result (an error since retried) is not sent again; it is reported instead.

A span is recorded only once its request was answered with a 2xx. An events request whose
response is lost (a timeout after the API stored it) is therefore not recorded, and a rerun
sends its batch again, with the same deterministic span ids. A cancel (the first Ctrl-C or
SIGTERM) stops between requests and while waiting for the read-back, leaving a
``cancelled`` publication that a rerun continues.

A create whose response is lost is looked up by the marker in the experiment's description
before it fails, as it may have gone through; only the first page of the dataset's
experiments is searched (``pup llm-obs experiments list`` takes no cursor). After
submitting, the experiments' events are counted (``pup llm-obs experiments events list``)
until every span is indexed or :data:`READBACK_TIMEOUT_S` passes, one budget for both; spans
not indexed yet are a warning, not a failure. Files passed with ``--file`` are written
``0600`` in a private temporary directory and removed afterwards. The spans carry the
conversations already pushed to the dataset, and nothing else.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote

from pup_finetune.data.push import SOURCE_TAG, PushState
from pup_finetune.datadog.jsonapi import JSON_API, object_id, private_json_file
from pup_finetune.datadog.pup import Pup
from pup_finetune.errors import ExternalError, StateError, require_writable
from pup_finetune.evaluation.cases import ReplayCase
from pup_finetune.evaluation.replay import METRICS_VERSION, TARGETS, PairKey
from pup_finetune.redaction import redact
from pup_finetune.runtime import Runtime
from pup_finetune.storage import atomic_write_json, canonical_json, finite_json, read_json

__all__ = [
    "CHILD_SPAN_NAME",
    "EXPERIMENTS_FILE",
    "PUBLISHED_FILE",
    "READBACK_POLL_S",
    "READBACK_TIMEOUT_S",
    "ROOT_SPAN_NAME",
    "SPANS_PER_REQUEST",
    "CreatedExperiment",
    "ExperimentsPublisher",
    "LLMObsExperiments",
    "PublishSettings",
    "SpanGroup",
    "build_span_groups",
    "experiment_name",
]

EXPERIMENTS_FILE: Final = "experiments.json"
PUBLISHED_FILE: Final = "published.json"
SPANS_PER_REQUEST: Final = 100
READBACK_TIMEOUT_S: Final = 120.0
READBACK_POLL_S: Final = 10.0
ROOT_SPAN_NAME: Final = "replay_trajectory"
CHILD_SPAN_NAME: Final = "replay_assistant"
NO_PARENT: Final = "undefined"
"""The ``parent_id`` of a root span."""

_SOURCE: Final = "Agent Observability"
_NAMESPACE: Final = uuid.uuid5(uuid.NAMESPACE_URL, "pup-finetune:replay-evaluation")
_UINT63: Final = (1 << 63) - 1

Clock = Callable[[], datetime]
"""The current time, timezone-aware."""
Sleep = Callable[[float], None]
Progress = Callable[[str], None]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def experiment_name(target: str, run_id: str) -> str:
    """The name the experiment of ``target`` is created with (the API may make it unique)."""
    return f"pup-finetune-{target}-{run_id}"


# --- the API ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CreatedExperiment:
    """An experiment: its id and its name (the run name the API gave it)."""

    id: str
    name: str


class LLMObsExperiments:
    """Agent Observability experiments and their events, read and written through ``pup``."""

    def __init__(self, pup: Pup) -> None:
        self._pup = pup

    def create(self, attributes: Mapping[str, Any], *, marker: str) -> CreatedExperiment:
        """Create an experiment with ``attributes``; returns its id and name.

        When the call fails or its response cannot be read, the dataset's experiments are
        listed and one whose description contains ``marker`` is adopted: the create may have
        gone through all the same.
        """
        body = {"data": {"type": "experiments", "attributes": dict(attributes)}}
        dataset_id = str(attributes["dataset_id"])
        hint = _lost_create_hint(str(attributes.get("name")))
        try:
            with private_json_file(body) as path:
                data = self._pup.run_json(["llm-obs", "experiments", "create", "--file", str(path)])
        except ExternalError as original:
            try:
                found = self.find(dataset_id, marker)
            except ExternalError:
                raise original from None  # The create failure, not the lookup's, explains it.
            if found is None:
                raise ExternalError(
                    original.message, source=original.source, hint=original.hint or hint
                ) from original
            return found
        created = _experiment(data.get("data") if isinstance(data, dict) else None)
        if created is not None:
            return created
        found = self.find(dataset_id, marker)
        if found is None:
            raise ExternalError(
                f"creating experiment {attributes.get('name')!r} returned no id, and no "
                "experiment it created was found afterwards",
                source=_SOURCE,
                hint=hint,
            )
        return found

    def find(self, dataset_id: str, marker: str) -> CreatedExperiment | None:
        """The experiment of the dataset whose description contains ``marker``, or ``None``."""
        data = self._pup.run_json(
            ["llm-obs", "experiments", "list", "--filter-dataset-id", dataset_id]
        )
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise ExternalError("the experiments list response has no data list", source=_SOURCE)
        for item in items:
            attributes = item.get("attributes") if isinstance(item, dict) else None
            description = attributes.get("description") if isinstance(attributes, dict) else None
            if isinstance(description, str) and marker in description:
                found = _experiment(item)
                if found is not None:
                    return found
        return None

    def submit_events(
        self,
        experiment_id: str,
        *,
        spans: Sequence[Mapping[str, Any]],
        metrics: Sequence[Mapping[str, Any]],
        tags: Sequence[str],
    ) -> None:
        """Submit ``spans`` and ``metrics`` to the experiment (an empty 2xx on success)."""
        body = {
            "data": {
                "type": "experiments",
                "attributes": {
                    "tags": list(tags),
                    "spans": [dict(span) for span in spans],
                    "metrics": [dict(metric) for metric in metrics],
                },
            }
        }
        self._pup.api("POST", _events_path(experiment_id), body, content_type=JSON_API)

    def count_events(self, experiment_id: str) -> int:
        """How many events of the experiment are indexed (``total_matching``).

        pup prints ``{"events": [...], "total_matching": N, ...}`` (checked live); an
        enveloped ``{"data": {...}}`` answer is read too.
        """
        data = self._pup.run_json(
            ["llm-obs", "experiments", "events", "list", experiment_id, "--limit", "1"]
        )
        total = data.get("total_matching") if isinstance(data, dict) else None
        if total is None and isinstance(data, dict) and isinstance(data.get("data"), dict):
            total = data["data"].get("total_matching")
        if type(total) is not int:
            raise ExternalError(
                "the experiment events list response has no total_matching count", source=_SOURCE
            )
        return total


def _lost_create_hint(name: str) -> str:
    return (
        f"the create may have gone through all the same, and only the first page of the "
        f"dataset's experiments was searched for it: look for an experiment named {name!r} on "
        "the dataset before running the command again, which creates another when it cannot "
        "find it"
    )


def _pending_nonce(entry: Mapping[str, Any]) -> str | None:
    """The nonce of a pending ``experiments.json`` entry (a create not known to be done)."""
    nonce = entry.get("pending_nonce")
    return nonce if isinstance(nonce, str) and nonce else None


def _events_path(experiment_id: str) -> str:
    return f"v2/llm-obs/v1/experiments/{quote(experiment_id, safe='')}/events"


def _experiment(item: Any) -> CreatedExperiment | None:
    identifier = object_id({"data": item})
    if identifier is None:
        return None
    attributes = item.get("attributes")
    name = attributes.get("name") if isinstance(attributes, dict) else None
    return CreatedExperiment(identifier, name if isinstance(name, str) and name else identifier)


# --- spans and metrics --------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PublishSettings:
    """What the experiments of one evaluation record, and the pushed dataset they attach to.

    ``eval_key`` is a digest of the evaluation's identity: it seeds every span id, and the
    experiments' descriptions carry it. ``records`` maps each test row id to its dataset
    record (``input``, ``expected_output``, ``metadata``), as pushed.
    """

    eval_key: str
    run_id: str
    provider: str
    base_model: str
    tuned_checkpoint: str
    judge_model: str
    max_tokens: int
    push: PushState
    records: Mapping[str, Mapping[str, Any]]

    @property
    def dataset_id(self) -> str:
        if self.push.dataset_id is None:
            raise StateError("the push receipt has no dataset id: run `dataset push` again")
        return self.push.dataset_id

    def tags(self, target: str) -> list[str]:
        return [SOURCE_TAG, f"target:{target}", f"run_id:{self.run_id}"]

    def marker(self, target: str, nonce: str) -> str:
        """What the description of ``target``'s experiment carries, to find it again.

        ``nonce`` is the one the evaluation directory recorded for the create, so another
        directory evaluating the same run never finds this directory's experiment.
        """
        return f"Evaluation key: {self.eval_key}-{target}-{nonce}."

    def experiment_attributes(self, target: str, nonce: str) -> dict[str, Any]:
        """The body attributes of ``target``'s experiment create, marked with ``nonce``."""
        model = self.base_model if target == "base" else f"{self.base_model}, tuned"
        attributes: dict[str, Any] = {
            "name": experiment_name(target, self.run_id),
            "description": (
                f"Replay evaluation by pup-finetune of run {self.run_id}: the {target} model "
                f"({model}) redoes each recorded assistant action of the test split. "
                f"{self.marker(target, nonce)}"
            ),
            "dataset_id": self.dataset_id,
            "project_id": self.push.project_id,
            "config": {
                "provider": self.provider,
                "base_model": self.base_model,
                "target": target,
                "run_id": self.run_id,
                "checkpoint": self.tuned_checkpoint if target == "tuned" else None,
                "judge_model": self.judge_model,
                "max_tokens": self.max_tokens,
                "metrics_version": METRICS_VERSION,
            },
            "metadata": {"tags": self.tags(target)},
            "ensure_unique": True,
            "run_count": 1,
        }
        if self.push.dataset_version is not None:
            attributes["dataset_version"] = self.push.dataset_version
        return attributes


@dataclass(frozen=True, slots=True)
class SpanGroup:
    """One row's spans (its root first, then a child per case) and their metrics.

    Metrics are templates without ``timestamp_ms``: it is stamped when they are submitted.
    """

    spans: list[dict[str, Any]]
    metrics: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    """Metrics by the span id they belong to."""

    def digest(self, span: Mapping[str, Any]) -> str:
        """A sha256 over ``span`` and its metrics: what was sent for it, minus timestamps."""
        payload = canonical_json({"span": span, "metrics": self.metrics.get(span["span_id"], [])})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ids(eval_key: str, name: str) -> tuple[str, str]:
    """The span id (decimal, below 2**63) and trace id (32 hex digits) for ``name``.

    Names say what they name: ``root:<target>:<row id>`` for a row's root span and
    ``case:<target>:<case id>`` for a case's span, so a row id that looks like a case id
    (``row-a:1``) cannot take the id of another row's case.
    """
    namespace = uuid.uuid5(_NAMESPACE, eval_key)
    derived = uuid.uuid5(namespace, name)
    span_id = derived.int & _UINT63 or 1
    return str(span_id), uuid.uuid5(namespace, f"{name}:trace").hex


def _ns(text: Any) -> int | None:
    """An ISO 8601 time as nanoseconds since the epoch, or ``None``."""
    if not isinstance(text, str):
        return None
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return int(moment.timestamp()) * 1_000_000_000 + moment.microsecond * 1_000


def _timing(
    generation: Mapping[str, Any] | None, result: Mapping[str, Any], fallback_ns: int
) -> tuple[int, int]:
    """``(start_ns, duration)`` of a case: its generation, ending when it was sampled."""
    if generation is not None:
        end = _ns(generation.get("sampled_at")) or fallback_ns
        latency = generation.get("latency_s")
        duration = max(int(float(latency) * 1e9), 0) if isinstance(latency, int | float) else 0
        return end - duration, duration
    return _ns(result.get("judged_at")) or fallback_ns, 0


def _metric(
    span: Mapping[str, Any], label: str, kind: str, value: Any, tags: Sequence[str]
) -> dict[str, Any]:
    return {
        "metric_source": "custom",
        "span_id": span["span_id"],
        "trace_id": span["trace_id"],
        "metric_type": kind,
        "label": label,
        f"{kind}_value": value,
        "tags": list(tags),
    }


def _judge(result: Mapping[str, Any]) -> tuple[bool | None, str | None]:
    judgment = result.get("judgment")
    if not isinstance(judgment, Mapping):
        return None, None
    reason = judgment.get("reason")
    return bool(judgment.get("passed")), reason if isinstance(reason, str) else None


def _child(
    settings: PublishSettings,
    target: str,
    case: ReplayCase,
    root: Mapping[str, Any],
    generation: Mapping[str, Any] | None,
    result: Mapping[str, Any],
    tags: list[str],
    fallback_ns: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """The span of ``case``, its metrics and its step in the root's output."""
    span_id, _ = _ids(settings.eval_key, f"case:{target}:{case.id}")
    start_ns, duration = _timing(generation, result, fallback_ns)
    ok = result.get("status") == "ok"
    source = settings.records[case.row_id].get("metadata") or {}
    meta: dict[str, Any] = {
        "span": {"kind": "llm"},
        "parent_experiment_span_id": root["span_id"],
        "input": {"messages": list(case.prefix), "tools": list(case.tools)},
        "output": None if generation is None else generation.get("message"),
        "expected_output": case.reference,
        "metadata": {
            "case_id": case.id,
            "message_index": case.message_index,
            "source_trace_id": case.trace_id,
            "source_span_id": source.get("span_id"),
        },
    }
    error = result.get("error")
    if not ok:
        detail = error if isinstance(error, Mapping) else {}
        meta["error"] = {
            "message": redact(str(detail.get("message") or "the case ended in an error")),
            "type": str(detail.get("type") or "Error"),
        }
    span = {
        "span_id": span_id,
        "trace_id": root["trace_id"],
        "parent_id": root["span_id"],
        "name": CHILD_SPAN_NAME,
        "start_ns": start_ns,
        "duration": duration,
        "status": "ok" if ok else "error",
        "tags": tags,
        "meta": meta,
    }
    step: dict[str, Any] = {
        "case_id": case.id,
        "message_index": case.message_index,
        "status": "ok" if ok else "error",
        "judge_pass": None,
    }
    metrics: list[dict[str, Any]] = []
    if ok:
        values = result.get("metrics")
        for name, value in sorted((values if isinstance(values, Mapping) else {}).items()):
            step[name] = value
            if isinstance(value, bool):
                metrics.append(_metric(span, name, "boolean", value, settings.tags(target)))
        passed, reason = _judge(result)
        step["judge_pass"] = passed
        if passed is not None:
            judge_metric = _metric(span, "judge_pass", "boolean", passed, settings.tags(target))
            if reason:
                judge_metric["reasoning"] = reason
            metrics.append(judge_metric)
    return span, metrics, step


def _row_group(
    settings: PublishSettings,
    target: str,
    experiment_id: str,
    row_id: str,
    cases: Sequence[ReplayCase],
    generations: Mapping[PairKey, Mapping[str, Any]],
    results: Mapping[PairKey, Mapping[str, Any]],
    fallback_ns: int,
) -> SpanGroup:
    record = settings.records[row_id]
    record_id = settings.push.record_ids[row_id]
    span_id, trace_id = _ids(settings.eval_key, f"root:{target}:{row_id}")
    tags = [
        *settings.tags(target),
        f"experiment_id:{experiment_id}",
        f"dataset_id:{settings.dataset_id}",
        f"dataset_record_id:{record_id}",
    ]
    root: dict[str, Any] = {"span_id": span_id, "trace_id": trace_id}
    children: list[dict[str, Any]] = []
    metrics: dict[str, list[dict[str, Any]]] = {}
    steps: list[dict[str, Any]] = []
    for case in cases:
        key = (target, case.id)
        child, child_metrics, step = _child(
            settings,
            target,
            case,
            root,
            generations.get(key),
            results[key],
            tags,
            fallback_ns,
        )
        children.append(child)
        metrics[child["span_id"]] = child_metrics
        steps.append(step)
    judged = [step for step in steps if step["status"] == "ok"]
    start_ns = min(child["start_ns"] for child in children)
    end_ns = max(child["start_ns"] + child["duration"] for child in children)
    root.update(
        {
            "parent_id": NO_PARENT,
            "name": ROOT_SPAN_NAME,
            "dataset_record_id": record_id,
            "start_ns": start_ns,
            "duration": end_ns - start_ns,
            "status": "ok" if judged else "error",
            "tags": tags,
            "meta": {
                "span": {"kind": "experiment"},
                "input": record.get("input"),
                "expected_output": record.get("expected_output"),
                "output": {
                    "steps": steps,
                    "completed_actions": len(judged),
                    "total_actions": len(steps),
                    "status": "complete" if len(judged) == len(steps) else "partial",
                },
                "metadata": {"row_id": row_id, "run_id": settings.run_id, "target": target},
            },
        }
    )
    root_metrics: list[dict[str, Any]] = []
    if judged:
        agreement = sum(bool(step["judge_pass"]) for step in judged) / len(judged)
        short = settings.tags(target)
        root_metrics.append(_metric(root, "trajectory_agreement", "score", agreement, short))
        root_metrics.append(_metric(root, "completed_actions", "score", len(judged), short))
    else:
        root["meta"]["error"] = {
            "message": "every case of the row ended in an error",
            "type": "ReplayError",
        }
    metrics[span_id] = root_metrics
    return SpanGroup(spans=[root, *children], metrics=finite_json(metrics))


def build_span_groups(
    settings: PublishSettings,
    target: str,
    experiment_id: str,
    cases: Sequence[ReplayCase],
    generations: Mapping[PairKey, Mapping[str, Any]],
    results: Mapping[PairKey, Mapping[str, Any]],
    *,
    fallback_ns: int,
) -> list[SpanGroup]:
    """The spans and metrics of ``target``'s experiment: one :class:`SpanGroup` per row.

    Every case must have a result. ``fallback_ns`` times a case whose records hold no time.
    """
    by_row: dict[str, list[ReplayCase]] = {}
    for case in cases:
        by_row.setdefault(case.row_id, []).append(case)
    return [
        _row_group(
            settings, target, experiment_id, row_id, row_cases, generations, results, fallback_ns
        )
        for row_id, row_cases in by_row.items()
    ]


def _batches(groups: Sequence[SpanGroup], size: int) -> list[list[tuple[SpanGroup, int]]]:
    """``(group, span index)`` pairs in batches of at most ``size``, keeping groups whole
    unless one alone is bigger."""
    batches: list[list[tuple[SpanGroup, int]]] = []
    current: list[tuple[SpanGroup, int]] = []
    for group in groups:
        items = [(group, index) for index in range(len(group.spans))]
        if current and len(current) + len(items) > size:
            batches.append(current)
            current = []
        while len(items) > size:
            batches.append(items[:size])
            items = items[size:]
        current.extend(items)
    if current:
        batches.append(current)
    return batches


# --- the publisher ------------------------------------------------------------------------


class _PublishCancelled(Exception):
    """The cancel event was set: stop before the next request."""


class ExperimentsPublisher:
    """Publishes an evaluation as one experiment per target; see the module docstring.

    ``publish`` returns what to report (``status`` ``published``, ``failed`` or
    ``cancelled``, the dataset, each target's experiment and spans, and warnings). A failed
    call is reported, not raised: the local results are complete, and running the evaluation
    again publishes the rest. ``cancel``, when set, stops publishing between requests and
    the read-back while it waits; ``sleep`` defaults to waiting on it.
    """

    def __init__(
        self,
        pup: Pup,
        settings: PublishSettings,
        *,
        out_dir: Path,
        runtime: Runtime,
        clock: Clock = _utc_now,
        sleep: Sleep | None = None,
        cancel: threading.Event | None = None,
        readback_timeout_s: float = READBACK_TIMEOUT_S,
        poll_s: float = READBACK_POLL_S,
        batch_size: int = SPANS_PER_REQUEST,
        progress: Progress | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.api = LLMObsExperiments(pup)
        self.settings = settings
        self.out_dir = Path(out_dir)
        self.runtime = runtime
        self.clock = clock
        self.cancel = cancel if cancel is not None else threading.Event()
        self.sleep: Sleep = sleep if sleep is not None else self._wait
        self.readback_timeout_s = readback_timeout_s
        self.poll_s = poll_s
        self.batch_size = batch_size
        self.progress = progress or (lambda _message: None)

    @property
    def experiments_path(self) -> Path:
        return self.out_dir / EXPERIMENTS_FILE

    @property
    def published_path(self) -> Path:
        return self.out_dir / PUBLISHED_FILE

    def _wait(self, seconds: float) -> None:
        self.cancel.wait(seconds)

    def publish(
        self,
        *,
        cases: Sequence[ReplayCase],
        generations: Mapping[PairKey, Mapping[str, Any]],
        results: Mapping[PairKey, Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        del summary
        require_writable(self.runtime, "publishing the evaluation as experiments")
        settings = self.settings
        report: dict[str, Any] = {
            "status": "published",
            "project_id": settings.push.project_id,
            "dataset_id": settings.dataset_id,
            "dataset_version": settings.push.dataset_version,
            "experiments": {},
            "warnings": [],
        }
        rerun = (
            "run the same evaluate command again to publish the rest: "
            f"{EXPERIMENTS_FILE} and {PUBLISHED_FILE} in the evaluation directory record "
            "what was created and sent"
        )
        try:
            experiments = {
                target: self._experiment(target, report["warnings"]) for target in TARGETS
            }
            for target, experiment in experiments.items():
                report["experiments"][target] = self._submit(
                    target, experiment, cases, generations, results, report["warnings"]
                )
        except _PublishCancelled:
            report["status"] = "cancelled"
            report["hint"] = rerun
            return report
        except ExternalError as exc:
            report["status"] = "failed"
            report["error"] = {
                "message": redact(exc.message),
                "source": exc.source,
                "hint": None if exc.hint is None else redact(exc.hint),
            }
            report["hint"] = report["error"]["hint"] or rerun
            return report
        self._read_back(report["experiments"], report["warnings"])
        return report

    def _check_cancel(self) -> None:
        if self.cancel.is_set():
            raise _PublishCancelled

    # --- experiments ----------------------------------------------------------------------

    def _read_experiments(self) -> dict[str, Any]:
        try:
            data = read_json(self.experiments_path)
        except FileNotFoundError:
            return {}
        if not isinstance(data, dict):
            raise StateError(f"{self.experiments_path}: expected a JSON object")
        return dict(data)

    def _experiment(self, target: str, warnings: list[str]) -> CreatedExperiment:
        """The experiment of ``target``: the one ``experiments.json`` records, or a new one.

        A pending entry on the pushed dataset is a create that may have gone through before
        the process died: its experiment is looked up by the entry's nonce and adopted, or
        created with that nonce when it is not found.
        """
        self._check_cancel()
        stored = self._read_experiments()
        entry = stored.get(target)
        dataset_id = self.settings.dataset_id
        nonce: str | None = None
        pending = _pending_nonce(entry) if isinstance(entry, dict) else None
        if isinstance(entry, dict) and pending is not None:
            if entry.get("dataset_id") == dataset_id:
                nonce = pending
            else:
                warnings.append(
                    f"{EXPERIMENTS_FILE} records a {target} experiment create on the dataset "
                    f"{entry.get('dataset_id')!r}, not the pushed dataset {dataset_id!r} (the "
                    f"dataset was pushed again), that may have gone through; a new {target} "
                    "experiment is created and recorded in its place, and one that create "
                    "made, if any, is left as it is"
                )
        elif entry is not None:
            identifier = entry.get("experiment_id") if isinstance(entry, dict) else None
            readable = isinstance(identifier, str) and bool(identifier)
            if readable and isinstance(entry, dict) and entry.get("dataset_id") == dataset_id:
                name = entry.get("name")
                return CreatedExperiment(
                    str(identifier), name if isinstance(name, str) else str(identifier)
                )
            if readable and isinstance(entry, dict):
                why = (
                    f"it is on the dataset {entry.get('dataset_id')!r}, not the pushed "
                    f"dataset {dataset_id!r} (the dataset was pushed again)"
                )
            else:
                why = "its entry cannot be read"
            warnings.append(
                f"{EXPERIMENTS_FILE} records a {target} experiment"
                + (f" {identifier!r}" if readable else "")
                + f" that is not reused, as {why}; a new {target} experiment is created and "
                "recorded in its place, and the old one is left as it is"
            )
        created: CreatedExperiment | None = None
        if nonce is not None:
            self.progress(f"looking for the {target} experiment an earlier run may have created")
            created = self.api.find(dataset_id, self.settings.marker(target, nonce))
        else:
            nonce = secrets.token_hex(8)
            stored[target] = {
                "pending_nonce": nonce,
                "dataset_id": dataset_id,
                "requested_at": self._now(),
            }
            atomic_write_json(self.experiments_path, stored)
        if created is None:
            attributes = self.settings.experiment_attributes(target, nonce)
            self.progress(f"creating the {target} experiment {attributes['name']!r}")
            created = self.api.create(attributes, marker=self.settings.marker(target, nonce))
        stored[target] = {
            "experiment_id": created.id,
            "name": created.name,
            "dataset_id": dataset_id,
            "created_at": self._now(),
        }
        atomic_write_json(self.experiments_path, stored)
        return created

    def _now(self) -> str:
        return self.clock().astimezone(UTC).isoformat(timespec="seconds")

    # --- events ---------------------------------------------------------------------------

    def _read_published(self) -> dict[str, dict[str, str]]:
        try:
            data = read_json(self.published_path)
        except FileNotFoundError:
            return {}
        if not isinstance(data, dict):
            raise StateError(f"{self.published_path}: expected a JSON object")
        return {
            key: {str(span): str(digest) for span, digest in value.items()}
            for key, value in data.items()
            if isinstance(value, dict)
        }

    def _submit(
        self,
        target: str,
        experiment: CreatedExperiment,
        cases: Sequence[ReplayCase],
        generations: Mapping[PairKey, Mapping[str, Any]],
        results: Mapping[PairKey, Mapping[str, Any]],
        warnings: list[str],
    ) -> dict[str, Any]:
        """Submit the spans of ``target`` its experiment does not hold yet; its report."""
        fallback_ns = int(self.clock().timestamp() * 1e9)
        groups = build_span_groups(
            self.settings,
            target,
            experiment.id,
            cases,
            generations,
            results,
            fallback_ns=fallback_ns,
        )
        published = self._read_published()
        sent = published.setdefault(experiment.id, {})
        todo: list[SpanGroup] = []
        stale = 0
        total = 0
        for group in groups:
            keep = []
            for span in group.spans:
                total += 1
                previous = sent.get(span["span_id"])
                if previous is None:
                    keep.append(span)
                elif previous != group.digest(span):
                    stale += 1
            if keep:
                todo.append(SpanGroup(spans=keep, metrics=group.metrics))
        if stale:
            warnings.append(
                f"{stale} span(s) of the {target} experiment were published with an earlier "
                "result (a case since retried) and are not sent again; evaluate into a new "
                "--out-dir to publish new experiments with every result"
            )
        submitted = 0
        pending = sum(len(group.spans) for group in todo)
        tags = self.settings.tags(target)
        for batch in _batches(todo, self.batch_size):
            self._check_cancel()
            spans = [group.spans[index] for group, index in batch]
            now_ms = int(self.clock().timestamp() * 1000)
            metrics = [
                {**metric, "timestamp_ms": now_ms, "experiment_id": experiment.id}
                for group, index in batch
                for metric in group.metrics.get(group.spans[index]["span_id"], [])
            ]
            self.api.submit_events(experiment.id, spans=spans, metrics=metrics, tags=tags)
            for group, index in batch:
                sent[group.spans[index]["span_id"]] = group.digest(group.spans[index])
            atomic_write_json(self.published_path, published)
            submitted += len(spans)
            self.progress(
                f"published {submitted}/{pending} new spans to the {target} experiment "
                f"{experiment.name!r}"
            )
        return {
            "experiment_id": experiment.id,
            "name": experiment.name,
            "spans": total,
            "submitted": submitted,
            "already_published": total - submitted,
        }

    def _read_back(self, entries: Mapping[str, dict[str, Any]], warnings: list[str]) -> None:
        """Count every experiment's events until all its spans are indexed, or the one
        timeout of all of them passes (or ``cancel`` is set)."""
        deadline = self.clock().timestamp() + self.readback_timeout_s
        for entry in entries.values():
            entry["events_indexed"] = None
            entry["indexed"] = False
        waiting = dict(entries)
        while waiting:
            for target, entry in list(waiting.items()):
                try:
                    seen = self.api.count_events(str(entry["experiment_id"]))
                except ExternalError as exc:
                    warnings.append(
                        f"could not count the events of the {target} experiment: "
                        f"{redact(exc.message)}"
                    )
                    del waiting[target]
                    continue
                entry["events_indexed"] = seen
                if seen >= int(entry["spans"]):
                    entry["indexed"] = True
                    del waiting[target]
            if not waiting:
                return
            if self.cancel.is_set():
                for target, entry in waiting.items():
                    warnings.append(
                        f"stopped waiting for the events of the {target} experiment to be "
                        f"indexed (cancelled): it shows {entry['events_indexed']} of its "
                        f"{entry['spans']} spans so far"
                    )
                return
            if self.clock().timestamp() + self.poll_s > deadline:
                break
            self.sleep(self.poll_s)
        for target, entry in waiting.items():
            warnings.append(
                f"the {target} experiment shows {entry['events_indexed']} of its "
                f"{entry['spans']} spans after {self.readback_timeout_s:g}s; Datadog may still "
                "be indexing the rest"
            )
