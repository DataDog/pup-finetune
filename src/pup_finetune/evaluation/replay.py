# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Resumable replay: generate every case with the base and the tuned model, score, judge.

:func:`run_replay` replays each :class:`~pup_finetune.evaluation.cases.ReplayCase` with two
targets, ``base`` and ``tuned``, each a :class:`~pup_finetune.providers.base.Sampler`. Every
``(target, case)`` pair is generated once, scored with the deterministic metrics and judged,
up to ``concurrency`` pairs at a time. The evaluation directory (``--out-dir``, ``0700``)
holds, all ``0600``:

- ``evaluation.json`` - the evaluation's identity (:func:`evaluation_identity`): the run,
  the tuned weights, the base model, the judge, a hash of the cases, ``max_tokens``, the
  rendering, the metrics version, the seed and the per-trajectory cap. A later run in the
  directory must have the same identity, so results made under other settings are never
  mixed in (:class:`~pup_finetune.errors.StateError`: use a new ``--out-dir``).
- ``generations.jsonl`` - one line per generated pair, appended as soon as it completes.
- ``results.jsonl`` - one line per judged pair (metrics and judgment), or per pair that
  failed (``status`` ``error``), appended as soon as it is known. The last line of a pair
  wins.
- ``summary.json`` - :func:`summarize`, rewritten at the end of every run, partial or not.

A rerun skips the pairs whose result is ``ok``, reuses a stored generation for a pair that
still needs a judgment, and retries the pairs that failed. A generation that fails after the
sampler's own retries, or a judge call that fails after the client's (an outage, not a
verdict), is recorded as an error result and the run goes on; errors are counted apart and
never count in a pass rate. Setting ``cancel`` (the CLI sets it on Ctrl-C or SIGTERM) stops
new pairs from starting; the ones in flight finish and are kept. A second Ctrl-C stops at
once: the pairs in flight are abandoned, and nothing they finish later is written, since
the directory's lock is released and the samplers are closed once :func:`run_replay`
returns. The workers are daemon threads, so the process can exit without waiting for a call
still blocked in a provider.

Like the metrics and the judge, every score measures agreement with the recorded behavior
on held-out conversations, not task success: tools were never executed (:data:`CAVEAT`).

Once every pair has a result, the evaluation goes to a :class:`Publisher`, which reports it
elsewhere (:class:`~pup_finetune.datadog.experiments.ExperimentsPublisher` publishes it as
Agent Observability experiments); :class:`NoopPublisher`, the default, publishes nothing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
import threading
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, Protocol, get_args

from pup_finetune.concurrency import DaemonPool
from pup_finetune.errors import StateError
from pup_finetune.evaluation.cases import ReplayCase
from pup_finetune.evaluation.judge import Judgment
from pup_finetune.evaluation.metrics import aggregate, score_case
from pup_finetune.inference import JudgeConfig
from pup_finetune.providers.base import CheckpointRef, Generation, Sampler
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
    "BOOTSTRAP_RESAMPLES",
    "CAVEAT",
    "METRICS_VERSION",
    "TARGETS",
    "EvalDir",
    "JudgeFn",
    "NoopPublisher",
    "PairKey",
    "Publisher",
    "ReplaySummary",
    "Target",
    "cases_sha256",
    "evaluation_identity",
    "paired_bootstrap_ci",
    "remaining_work",
    "run_replay",
    "summarize",
]

Target = Literal["base", "tuned"]
TARGETS: tuple[Target, ...] = get_args(Target)

METRICS_VERSION: Final = "1"
"""Bumped whenever a metric or the summary's math changes meaning; part of the identity."""

CAVEAT: Final = (
    "Scores measure agreement with recorded behavior on held-out conversations; tools were "
    "not executed, so they do not measure task success."
)

BOOTSTRAP_RESAMPLES: Final = 2000
MIN_CLUSTERS_FOR_CI: Final = 2
"""The fewest trajectories with a paired case for which a confidence interval is reported."""

IDENTITY_FILE: Final = "evaluation.json"
GENERATIONS_FILE: Final = "generations.jsonl"
RESULTS_FILE: Final = "results.jsonl"
SUMMARY_FILE: Final = "summary.json"
CALIBRATION_FILE: Final = "calibration.json"

_POLL_S: Final = 0.1

JudgeFn = Callable[[ReplayCase, Generation], Judgment]
"""Judges one generation of a case; raises when the judge cannot be reached (an outage)."""

PairKey = tuple[str, str]
"""``(target, case id)``."""

ProgressSink = Callable[[dict[str, Any]], None]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- identity -----------------------------------------------------------------------------


def cases_sha256(cases: Iterable[ReplayCase]) -> str:
    """A sha256 over the cases as JSON, in order: any change to what is replayed changes it."""
    payload = canonical_json([case.to_json() for case in cases])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluation_identity(
    *,
    run_id: str,
    tuned_checkpoint: CheckpointRef,
    base_model: str,
    judge: JudgeConfig,
    cases: Sequence[ReplayCase],
    max_tokens: int,
    rendering: Mapping[str, Any],
    seed: int,
    max_cases_per_trajectory: int | None,
) -> dict[str, Any]:
    """What ``evaluation.json`` records: everything that changes a generation or a score.

    ``tuned_checkpoint`` is the training checkpoint whose weights the tuned target samples
    (the sampler checkpoint made from it differs per session, the weights do not), and
    ``rendering`` the run's renderer identity.
    """
    return {
        "run_id": run_id,
        "tuned_checkpoint": dataclasses.asdict(tuned_checkpoint),
        "base_model": base_model,
        "judge": judge.identity(),
        "cases_sha256": cases_sha256(cases),
        "n_cases": len(cases),
        "max_tokens": max_tokens,
        "rendering": dict(rendering),
        "metrics_version": METRICS_VERSION,
        "seed": seed,
        "max_cases_per_trajectory": max_cases_per_trajectory,
    }


# --- the evaluation directory -------------------------------------------------------------


class EvalDir:
    """The files of one evaluation under :attr:`path`; see the module docstring.

    Constructing one touches nothing. Appends are serialized, so worker threads may call
    :meth:`append_generation` and :meth:`append_result` concurrently.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._append_lock = threading.Lock()

    @property
    def identity_path(self) -> Path:
        return self.path / IDENTITY_FILE

    @property
    def generations_path(self) -> Path:
        return self.path / GENERATIONS_FILE

    @property
    def results_path(self) -> Path:
        return self.path / RESULTS_FILE

    @property
    def summary_path(self) -> Path:
        return self.path / SUMMARY_FILE

    @property
    def calibration_path(self) -> Path:
        return self.path / CALIBRATION_FILE

    def lock(self) -> DirectoryLock:
        """The directory's lock: one evaluation writes in a directory at a time."""
        return DirectoryLock(self.path)

    def create(self) -> None:
        """Create the directory if it is missing, and make it ``0700`` either way: it holds
        generations and judge answers derived from trace data."""
        make_private_dir(self.path)

    def has_work(self) -> bool:
        """Whether any generation or result is stored: work an identity change would orphan."""
        return any(
            path.exists() and path.stat().st_size > 0
            for path in (self.generations_path, self.results_path)
        )

    def check_identity(self, identity: Mapping[str, Any]) -> None:
        """Write ``evaluation.json`` the first time; later, raise unless it is ``identity``.

        Raises :class:`StateError` naming what changed, with a hint to use a new
        ``--out-dir``. A directory with no generation or result yet (e.g. only a failed
        calibration) takes the new identity instead, and forgets its calibration.
        """
        wanted = finite_json(dict(identity))
        if not self.identity_path.exists():
            atomic_write_json(self.identity_path, wanted)
            return
        stored = read_json(self.identity_path)
        if not isinstance(stored, dict):
            raise StateError(f"{self.identity_path}: expected a JSON object")
        if canonical_json(stored) == canonical_json(wanted):
            return
        if not self.has_work():
            self.calibration_path.unlink(missing_ok=True)
            atomic_write_json(self.identity_path, wanted)
            return
        changed = sorted(
            key
            for key in set(stored) | set(wanted)
            if canonical_json(stored.get(key)) != canonical_json(wanted.get(key))
        )
        raise StateError(
            f"{self.path} holds an evaluation with other settings "
            f"(changed: {', '.join(changed)}), so its results cannot be reused",
            hint="pass a new --out-dir to evaluate with these settings",
        )

    def stored_identity(self) -> dict[str, Any] | None:
        """``evaluation.json``, or ``None`` before the first evaluation in the directory."""
        if not self.identity_path.exists():
            return None
        stored = read_json(self.identity_path)
        return stored if isinstance(stored, dict) else None

    def read_generations(self) -> dict[PairKey, dict[str, Any]]:
        """The stored generation of each pair (the last line of a pair wins)."""
        return _by_pair(_read_lines(self.generations_path))

    def read_results(self) -> dict[PairKey, dict[str, Any]]:
        """The last result of each pair, ``ok`` or ``error``."""
        return _by_pair(_read_lines(self.results_path))

    # Generations hold model output, which a tool call parser can decode into a lone
    # surrogate (JSON's ``\ud800`` escape); UTF-8 cannot encode one, so these files are
    # written ASCII-only, as ``\u`` escapes that read back to the same strings.

    def append_generation(self, record: Mapping[str, Any]) -> None:
        with self._append_lock:
            append_jsonl(self.generations_path, [finite_json(dict(record))], ensure_ascii=True)

    def append_result(self, record: Mapping[str, Any]) -> None:
        with self._append_lock:
            append_jsonl(self.results_path, [finite_json(dict(record))], ensure_ascii=True)

    def read_calibration(self) -> dict[str, Any] | None:
        if not self.calibration_path.exists():
            return None
        stored = read_json(self.calibration_path)
        return stored if isinstance(stored, dict) else None

    def write_calibration(self, report: Mapping[str, Any]) -> None:
        atomic_write_json(self.calibration_path, finite_json(dict(report)))

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        atomic_write_json(self.summary_path, finite_json(dict(summary)))


def remaining_work(eval_dir: EvalDir, cases: Sequence[ReplayCase]) -> dict[str, int]:
    """What a run in ``eval_dir`` would still do: ``{"done", "generate", "judge"}`` pairs.

    ``done`` pairs have an ``ok`` result; every other pair is judged, and generated too
    unless a generation is stored. Reads only, without the lock: for previews.
    """
    generations, results = eval_dir.read_generations(), eval_dir.read_results()
    done = generate = judge = 0
    for case in cases:
        for target in TARGETS:
            key = (target, case.id)
            if results.get(key, {}).get("status") == "ok":
                done += 1
                continue
            judge += 1
            generate += key not in generations
    return {"done": done, "generate": generate, "judge": judge}


def _read_lines(path: Path) -> list[dict[str, Any]]:
    """The JSON objects of a JSON Lines file; a torn or unreadable line is skipped.

    A torn line can only be the last write of a process that died mid-append; its pair is
    simply done again.
    """
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            row, problem = parse_jsonl_line(line)
            if problem is None:
                rows.append(row)
    return rows


def _by_pair(rows: Iterable[Mapping[str, Any]]) -> dict[PairKey, dict[str, Any]]:
    found: dict[PairKey, dict[str, Any]] = {}
    for row in rows:
        target, case_id = row.get("target"), row.get("case_id")
        if isinstance(target, str) and isinstance(case_id, str):
            found[(target, case_id)] = dict(row)
    return found


# --- publishing ---------------------------------------------------------------------------


class Publisher(Protocol):
    """Receives a complete evaluation, e.g. to report it to Datadog. Called once per run.

    Called with the directory's lock held. Returns what to report of it (strict JSON), or
    ``None`` for nothing; :attr:`ReplaySummary.publication` carries it.
    """

    def publish(
        self,
        *,
        cases: Sequence[ReplayCase],
        generations: Mapping[PairKey, Mapping[str, Any]],
        results: Mapping[PairKey, Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> dict[str, Any] | None: ...


class NoopPublisher:
    """The default :class:`Publisher`: publishes nothing."""

    def publish(
        self,
        *,
        cases: Sequence[ReplayCase],
        generations: Mapping[PairKey, Mapping[str, Any]],
        results: Mapping[PairKey, Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        del cases, generations, results, summary
        return None


# --- replay -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    """How a :func:`run_replay` ended: ``summary.json``, the pairs left and whether it was cut.

    ``pending`` counts the pairs without a result (cancelled before they ran); ``cancelled``
    says ``cancel`` was set. A complete evaluation has neither. ``publication`` is what the
    publisher reported, ``None`` when it reported nothing or the evaluation is incomplete.
    """

    summary: dict[str, Any]
    pending: int
    cancelled: bool
    out_dir: Path
    publication: dict[str, Any] | None = None

    @property
    def complete(self) -> bool:
        return self.pending == 0 and not self.cancelled


def _error_json(stage: str, error: BaseException) -> dict[str, Any]:
    return {"stage": stage, "type": type(error).__name__, "message": redact(str(error))}


def _generation_record(target: str, case: ReplayCase, generation: Generation) -> dict[str, Any]:
    return {
        "target": target,
        "case_id": case.id,
        "message": generation.message,
        "format_valid": generation.format_valid,
        "prompt_tokens": generation.prompt_tokens,
        "completion_tokens": generation.completion_tokens,
        "latency_s": generation.latency_s,
        "stop_reason": generation.stop_reason,
        "sampled_at": _now(),
    }


def _generation_from(record: Mapping[str, Any]) -> Generation:
    return Generation(
        message=dict(record["message"]),
        format_valid=bool(record["format_valid"]),
        prompt_tokens=int(record.get("prompt_tokens") or 0),
        completion_tokens=int(record.get("completion_tokens") or 0),
        latency_s=float(record.get("latency_s") or 0.0),
        stop_reason=record.get("stop_reason"),
    )


def _result_record(
    target: str,
    case: ReplayCase,
    *,
    metrics: Mapping[str, bool | None] | None = None,
    judgment: Judgment | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "target": target,
        "case_id": case.id,
        "row_id": case.row_id,
        "trace_id": case.trace_id,
        "message_index": case.message_index,
        "status": "error" if error is not None else "ok",
        "error": error,
        "metrics": None if metrics is None else dict(metrics),
        "judgment": None if judgment is None else judgment.to_json(),
        "judged_at": _now(),
    }


class _Replay:
    """One :func:`run_replay`: the pairs to do, the workers' task, and what it collected."""

    def __init__(
        self,
        eval_dir: EvalDir,
        samplers: Mapping[str, Sampler],
        judge_fn: JudgeFn,
        *,
        max_tokens: int,
        cancel: threading.Event,
    ) -> None:
        self.eval_dir = eval_dir
        self.samplers = samplers
        self.judge_fn = judge_fn
        self.max_tokens = max_tokens
        self.cancel = cancel
        self.generations = eval_dir.read_generations()
        self.results = eval_dir.read_results()
        self._writes = threading.Lock()
        self._closed = False

    def close_writes(self) -> None:
        """Stop the workers' writes, once an append in progress is done (a second Ctrl-C).

        A worker that finishes later writes nothing: the directory's lock is about to be
        released, and its sampler closed.
        """
        with self._writes:
            self._closed = True

    def _append(
        self, append: Callable[[Mapping[str, Any]], None], record: Mapping[str, Any]
    ) -> bool:
        """``append(record)`` unless writes are closed; whether it was written."""
        with self._writes:
            if self._closed:
                return False
            append(record)
            return True

    def todo(self, cases: Sequence[ReplayCase]) -> list[tuple[Target, ReplayCase]]:
        return [
            (target, case)
            for case in cases
            for target in TARGETS
            if self.results.get((target, case.id), {}).get("status") != "ok"
        ]

    def pair(self, target: Target, case: ReplayCase) -> dict[str, Any] | None:
        """Generate (unless stored), score and judge one pair; ``None`` when cancelled first.

        Failures of the sampler or the judge become the pair's error result; anything else
        (e.g. a failed write) propagates.
        """
        if self.cancel.is_set():
            return None
        key = (target, case.id)
        record = self.generations.get(key)
        if record is None:
            try:
                generation = self.samplers[target].generate(
                    [dict(message) for message in case.prefix],
                    [dict(tool) for tool in case.tools],
                    max_tokens=self.max_tokens,
                )
            except Exception as exc:
                result = _result_record(target, case, error=_error_json("generation", exc))
                return result if self._append(self.eval_dir.append_result, result) else None
            record = _generation_record(target, case, generation)
            if not self._append(self.eval_dir.append_generation, record):
                return None
        else:
            generation = _generation_from(record)
        if self.cancel.is_set():
            return {"generation": record}
        metrics = score_case(
            generation.message, case.reference, case.tools, format_valid=generation.format_valid
        )
        try:
            judgment = self.judge_fn(case, generation)
        except Exception as exc:
            result = _result_record(target, case, metrics=metrics, error=_error_json("judge", exc))
        else:
            result = _result_record(target, case, metrics=metrics, judgment=judgment)
        if not self._append(self.eval_dir.append_result, result):
            return None
        return {**result, "generation": record}

    def collect(self, target: Target, case: ReplayCase, outcome: dict[str, Any] | None) -> None:
        if outcome is None:
            return
        generation = outcome.pop("generation", None)
        if generation is not None:
            self.generations[(target, case.id)] = generation
        if "status" in outcome:
            self.results[(target, case.id)] = outcome


def run_replay(
    cases: Sequence[ReplayCase],
    samplers: Mapping[str, Sampler],
    judge_fn: JudgeFn,
    *,
    out_dir: Path,
    identity: Mapping[str, Any],
    concurrency: int = 4,
    cancel: threading.Event | None = None,
    publisher: Publisher | None = None,
    on_progress: ProgressSink | None = None,
) -> ReplaySummary:
    """Replay ``cases`` with the ``base`` and ``tuned`` samplers into ``out_dir``.

    ``identity`` is :func:`evaluation_identity` for these cases (checked against
    ``evaluation.json``, see :meth:`EvalDir.check_identity`); each generation is capped at
    its ``max_tokens``, and its ``seed`` seeds the summary's bootstrap. A target's sampler
    is needed only while one of its pairs still needs a generation, so a rerun with nothing
    left to generate may pass fewer. Holds the directory's lock throughout. ``on_progress``
    receives ``{"done", "total", "target", "case_id", "status"}`` per finished pair.
    ``publisher`` gets the evaluation once every pair has a result, and what it reports is
    the outcome's ``publication``. Returns the summary also written to ``summary.json``.
    """
    max_tokens, seed = identity.get("max_tokens"), identity.get("seed")
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError(f"the identity's max_tokens must be at least 1, got {max_tokens!r}")
    if not isinstance(seed, int):
        raise ValueError(f"the identity's seed must be an integer, got {seed!r}")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("case ids must be unique")
    if identity.get("cases_sha256") != cases_sha256(cases):
        raise ValueError("identity does not describe these cases")
    cancel = cancel if cancel is not None else threading.Event()
    eval_dir = EvalDir(out_dir)
    eval_dir.create()
    with eval_dir.lock():
        eval_dir.check_identity(identity)
        replay = _Replay(eval_dir, samplers, judge_fn, max_tokens=max_tokens, cancel=cancel)
        todo = replay.todo(cases)
        missing = sorted(
            {
                target
                for target, case in todo
                if (target, case.id) not in replay.generations and target not in samplers
            }
        )
        if missing:
            raise ValueError(f"no sampler for {', '.join(missing)}, which has cases to generate")
        _run_pool(replay, todo, concurrency=concurrency, on_progress=on_progress)
        summary = summarize(cases, replay.results, replay.generations, seed=seed)
        eval_dir.write_summary(summary)
        pending = sum(
            1 for case in cases for target in TARGETS if (target, case.id) not in replay.results
        )
        outcome = ReplaySummary(
            summary=summary, pending=pending, cancelled=cancel.is_set(), out_dir=eval_dir.path
        )
        if outcome.complete:
            publication = (publisher or NoopPublisher()).publish(
                cases=cases,
                generations=replay.generations,
                results=replay.results,
                summary=summary,
            )
            if publication is not None:
                outcome = dataclasses.replace(outcome, publication=finite_json(publication))
    return outcome


def _run_pool(
    replay: _Replay,
    todo: Sequence[tuple[Target, ReplayCase]],
    *,
    concurrency: int,
    on_progress: ProgressSink | None,
) -> None:
    """Run ``todo`` on ``concurrency`` threads until done or ``replay.cancel`` is set.

    On cancel, the pairs not started are dropped and the ones in flight finish. An interrupt
    in this thread (a second Ctrl-C) sets cancel, closes the workers' writes (waiting only
    for an append in progress) and returns at once, so nothing is written after the
    directory's lock is released; the workers are daemon threads, so a call blocked in a
    provider does not hold up the process's exit either. Any other error (e.g. a failed
    write) also cancels the pairs not started, but waits for the ones in flight before it
    propagates, so none outlives the directory's lock or its sampler.
    """
    if not todo:
        return
    pool = DaemonPool(concurrency, name="replay")
    futures: dict[Future[dict[str, Any] | None], tuple[Target, ReplayCase]] = {}
    done_count = 0
    interrupted = False

    def finish(future: Future[dict[str, Any] | None]) -> None:
        nonlocal done_count
        target, case = futures[future]
        outcome = future.result()
        replay.collect(target, case, outcome)
        if outcome is not None and "status" in outcome:
            done_count += 1
            if on_progress is not None:
                on_progress(
                    {
                        "done": done_count,
                        "total": len(todo),
                        "target": target,
                        "case_id": case.id,
                        "status": outcome["status"],
                    }
                )

    try:
        for target, case in todo:
            futures[pool.submit(replay.pair, target, case)] = (target, case)
        pending: set[Future[dict[str, Any] | None]] = set(futures)
        while pending and not replay.cancel.is_set():
            done, pending = wait(pending, timeout=_POLL_S, return_when=FIRST_COMPLETED)
            for future in done:
                finish(future)
        for future in pending:
            future.cancel()
        for future in pending:
            if not future.cancelled():
                finish(future)  # in flight when cancelled: wait for it and keep it
    except KeyboardInterrupt:
        interrupted = True
        replay.cancel.set()
        replay.close_writes()
        raise
    except BaseException:
        replay.cancel.set()
        raise
    finally:
        pool.shutdown(wait=not interrupted)


# --- the summary --------------------------------------------------------------------------


def _quantile(values: Sequence[float], q: float) -> float:
    """The ``q`` quantile of sorted ``values``, interpolating linearly between ranks."""
    position = q * (len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def paired_bootstrap_ci(
    clusters: Sequence[Sequence[float]],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
    level: float = 0.95,
) -> tuple[float, float] | None:
    """A percentile bootstrap CI of the mean paired difference, resampling whole clusters.

    Each cluster holds one trajectory's per-case differences (tuned minus base, each ``-1``,
    ``0`` or ``1``). A resample draws as many clusters as there are, with replacement, and
    takes the mean over all their cases, so the cases of one trajectory move together.
    Seeded with ``seed``: the same input gives the same interval. ``None`` with fewer than
    :data:`MIN_CLUSTERS_FOR_CI` non-empty clusters.
    """
    kept = [list(cluster) for cluster in clusters if cluster]
    if len(kept) < MIN_CLUSTERS_FOR_CI or resamples < 1:
        return None
    sums = [sum(cluster) for cluster in kept]
    counts = [len(cluster) for cluster in kept]
    rng = random.Random(seed)  # noqa: S311 - a reproducible resample, not crypto
    size = len(kept)
    stats: list[float] = []
    for _ in range(resamples):
        picks = [rng.randrange(size) for _ in range(size)]
        stats.append(sum(sums[i] for i in picks) / sum(counts[i] for i in picks))
    stats.sort()
    tail = (1.0 - level) / 2.0
    return _quantile(stats, tail), _quantile(stats, 1.0 - tail)


def _rate(passed: int, n: int) -> dict[str, Any]:
    return {"rate": passed / n if n else None, "passed": passed, "n": n}


def _passed(result: Mapping[str, Any]) -> bool:
    judgment = result.get("judgment")
    return bool(judgment.get("passed")) if isinstance(judgment, Mapping) else False


def _target_summary(
    target: str,
    cases: Sequence[ReplayCase],
    results: Mapping[PairKey, Mapping[str, Any]],
    generations: Mapping[PairKey, Mapping[str, Any]],
) -> dict[str, Any]:
    ok: list[Mapping[str, Any]] = []
    errors: Counter[str] = Counter()
    too_long = 0
    pending = 0
    for case in cases:
        result = results.get((target, case.id))
        if result is None:
            pending += 1
        elif result.get("status") == "ok":
            ok.append(result)
        else:
            error = result.get("error")
            stage = error.get("stage") if isinstance(error, Mapping) else None
            errors[str(stage or "unknown")] += 1
            if isinstance(error, Mapping) and error.get("type") == "ContextLimitError":
                too_long += 1
    unanswered: Counter[str] = Counter()
    for result in ok:
        judgment = result.get("judgment")
        flaw = judgment.get("error") if isinstance(judgment, Mapping) else None
        if flaw is not None:
            unanswered[str(flaw)] += 1
    by_trajectory: dict[str, list[bool]] = defaultdict(list)
    for result in ok:
        by_trajectory[str(result.get("row_id"))].append(_passed(result))
    means = [sum(values) / len(values) for values in by_trajectory.values()]
    tokens = {"prompt_tokens": 0, "completion_tokens": 0, "generations": 0}
    for case in cases:
        generation = generations.get((target, case.id))
        if generation is not None:
            tokens["prompt_tokens"] += int(generation.get("prompt_tokens") or 0)
            tokens["completion_tokens"] += int(generation.get("completion_tokens") or 0)
            tokens["generations"] += 1
    return {
        "completed": len(ok),
        "errors": sum(errors.values()),
        "errors_by_stage": dict(sorted(errors.items())),
        "errors_too_long": too_long,
        "pending": pending,
        "judge_pass": _rate(sum(_passed(result) for result in ok), len(ok)),
        "judge_no_verdict": sum(unanswered.values()),
        "judge_no_verdict_by_error": dict(sorted(unanswered.items())),
        "metrics": aggregate(result.get("metrics") or {} for result in ok),
        "trajectory_agreement": {
            "mean": sum(means) / len(means) if means else None,
            "n_trajectories": len(means),
        },
        "tokens": tokens,
    }


def summarize(
    cases: Sequence[ReplayCase],
    results: Mapping[PairKey, Mapping[str, Any]],
    generations: Mapping[PairKey, Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    """``summary.json`` for ``cases`` from the last result of each pair; strict JSON.

    Per target: the judge pass rate and every metric's (:func:`~pup_finetune.evaluation.
    metrics.aggregate`) over its ``ok`` results, the mean over trajectories of each
    trajectory's judge pass rate, the errors by stage, the judge answers that held no
    verdict (``judge_no_verdict``, by :attr:`~pup_finetune.evaluation.judge.Judgment.error`:
    they count as failures, as the judge module decides) and the generated tokens. ``paired``
    compares the judge verdicts of the cases both targets have an ``ok`` result for (wins:
    only tuned passes; regressions: only base passes; ties: the same verdict), and counts the
    cases where both or one target failed with an error, or one is still pending.
    ``tuned_minus_base`` is the mean paired difference of the judge pass (``null`` without
    paired cases) with a 95% paired bootstrap CI over trajectories
    (:func:`paired_bootstrap_ci`, ``null`` when too few).
    """
    targets = {target: _target_summary(target, cases, results, generations) for target in TARGETS}
    paired: Counter[str] = Counter(
        {"wins": 0, "regressions": 0, "ties": 0, "both_error": 0, "one_error": 0, "pending": 0}
    )
    clusters: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        base, tuned = results.get(("base", case.id)), results.get(("tuned", case.id))
        if base is None or tuned is None:
            paired["pending"] += 1
            continue
        base_ok, tuned_ok = base.get("status") == "ok", tuned.get("status") == "ok"
        if not base_ok and not tuned_ok:
            paired["both_error"] += 1
            continue
        if not (base_ok and tuned_ok):
            paired["one_error"] += 1
            continue
        difference = int(_passed(tuned)) - int(_passed(base))
        paired["wins" if difference > 0 else "regressions" if difference < 0 else "ties"] += 1
        clusters[case.row_id].append(float(difference))
    differences = [value for cluster in clusters.values() for value in cluster]
    interval = paired_bootstrap_ci(list(clusters.values()), seed=seed)
    complete = paired["pending"] == 0
    return finite_json(
        {
            "status": "complete" if complete else "partial",
            "n_cases": len(cases),
            "n_trajectories": len({case.row_id for case in cases}),
            "targets": targets,
            "paired": {"n": len(differences), **paired},
            "tuned_minus_base": {
                "judge_pass_rate": sum(differences) / len(differences) if differences else None,
                "ci95": None if interval is None else list(interval),
                "n_cases": len(differences),
                "n_trajectories": len(clusters),
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": seed,
            },
            "errors": {target: targets[target]["errors"] for target in TARGETS},
            "tokens": {target: targets[target]["tokens"] for target in TARGETS},
            "metrics_version": METRICS_VERSION,
            "caveat": CAVEAT,
        }
    )
