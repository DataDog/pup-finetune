# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Plan, calibrate, replay and publish one evaluation: what ``evaluate`` and ``train
--evaluate`` run.

:func:`plan_evaluation` turns a run and its dataset into an :class:`EvalPlan`: the cases of
the test split, the judge, the evaluation directory and whether the result can be published
(:class:`Publishing`). :func:`plan_standalone` plans the ``evaluate`` command for a completed
run, and :meth:`StandaloneEvaluation.execute` runs it: it calibrates the judge (unless
``--skip-calibration``; a failed calibration stops before any generation), replays every case
(:func:`~pup_finetune.evaluation.replay.run_replay`) in a session of its own and returns what
the command emits. :class:`EvaluateAfterTraining` is the ``after_finalize`` hook of ``train
--evaluate``, which replays in the training session before it closes.

Once every case has a result, the evaluation is published as two Agent Observability
experiments on the pushed dataset (:mod:`pup_finetune.datadog.experiments`) unless publishing
is off (``--no-publish``) or the dataset has no push receipt, which is a warning: the local
results are complete either way.

The evaluation modules this drives (and through them the judge's HTTP client) are imported
only when a function here runs, so the CLI can import this module for every command.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import shlex
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.data.push import (
    PushState,
    dataset_record,
    push_path,
    read_push_state,
    record_digest,
)
from pup_finetune.datadog.pup import Pup
from pup_finetune.errors import FinetuneError, StateError, UsageError
from pup_finetune.output import info
from pup_finetune.providers import get_provider
from pup_finetune.providers.base import (
    CheckpointRef,
    Generation,
    ModelProfile,
    ModelRef,
    Provider,
    Sampler,
    TrainingSession,
)
from pup_finetune.runs import Phase, RunDir, dataset_content_hash, error_json, model_id_for_run
from pup_finetune.runs import cancel_on_signals as _cancel_on_signals
from pup_finetune.runtime import Runtime
from pup_finetune.storage import LOCK_FILE_NAME, canonical_json, finite_json
from pup_finetune.training.records import StoppedReason, TrainResult

if TYPE_CHECKING:
    from pup_finetune.evaluation.cases import ReplayCase
    from pup_finetune.evaluation.judge import Judgment
    from pup_finetune.evaluation.replay import Publisher, ReplaySummary
    from pup_finetune.inference import ChatCompleter, JudgeConfig

__all__ = [
    "CALIBRATION_CASES",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_MAX_CASES_PER_TRAJECTORY",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_OUT_DIR_NAME",
    "DEFAULT_SEED",
    "EVALUATED_STOPS",
    "FINISHED_PHASES",
    "SPEND_POLL_S",
    "EvalPlan",
    "EvalSettings",
    "EvaluateAfterTraining",
    "Publishing",
    "StandaloneEvaluation",
    "calibration_failed",
    "cancel_on_signals",
    "evaluation_exit_code",
    "finished_result",
    "open_judge_client",
    "plan_evaluation",
    "plan_standalone",
    "prepare_eval_dir",
    "replay",
    "replay_result",
    "require_new_eval_dir",
    "resolve_judge",
    "run_calibration",
]

DEFAULT_OUT_DIR_NAME: Final = "eval"
DEFAULT_MAX_CASES_PER_TRAJECTORY: Final = 2
DEFAULT_CONCURRENCY: Final = 4
DEFAULT_MAX_TOKENS: Final = 1024
DEFAULT_SEED: Final = 0
CALIBRATION_CASES: Final = 4
"""How many cases :func:`~pup_finetune.evaluation.judge.calibrate` judges, three times each
(the reference and two corruptions of it)."""

EVALUATED_STOPS: Final[frozenset[StoppedReason]] = frozenset({"completed", "early_stopped"})
"""How a training must have stopped for ``train --evaluate`` to evaluate it.

A cancelled run was stopped on request, and a ``budget_exhausted`` one reached the spend
cap, which the training session's time counts against: evaluating in that session would
spend past it. Both are skipped with the ``evaluate`` command to run instead, which
evaluates any finished run (:data:`FINISHED_PHASES`: a budget-capped or cancelled one too)
with a best epoch in a session of its own, on request.

A run that stopped otherwise is evaluated in its session still under ``--max-spend-usd``
(its reserve included): it is skipped the same way when the cap is already reached, and an
evaluation that reaches it stops as if cancelled, partial, with the ``evaluate`` command to
continue it.
"""

FINISHED_PHASES: Final[frozenset[Phase]] = frozenset({"completed", "cancelled"})
"""The run phases ``evaluate`` evaluates: a cancelled run still promoted its best epoch."""

_CANCEL_NOTICE: Final = (
    b"cancel requested: finishing the generations in flight, then writing the summary"
)


def _q(value: object) -> str:
    return shlex.quote(str(value))


# --- publishing ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Publishing:
    """Whether and where an evaluation is published as Agent Observability experiments.

    ``enabled`` is false with ``--no-publish``. ``skipped`` says why an enabled evaluation
    cannot be published (no push receipt, an unfinished push, test rows the receipt does not
    list, test rows that changed after they were pushed); ``None`` when it can. ``records``
    holds each test row's dataset record, as pushed.
    """

    enabled: bool
    dataset_dir: Path
    push: PushState | None = None
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped: str | None = None

    @property
    def ready(self) -> bool:
        return self.enabled and self.skipped is None and self.push is not None

    def status(self) -> dict[str, Any]:
        """What a preview, or a result without a publication, says of publishing."""
        if not self.enabled:
            return {"status": "disabled"}
        push = self.push
        if self.skipped is not None or push is None:
            return {"status": "skipped", "reason": self.skipped}
        return {
            "status": "planned",
            "project_id": push.project_id,
            "dataset_id": push.dataset_id,
            "dataset_version": push.dataset_version,
        }


def _push_hint(dataset_dir: Path) -> str:
    return (
        f"run `pup finetune dataset push {_q(dataset_dir)} --name NAME` first to publish the "
        "evaluation as experiments, or pass --no-publish"
    )


def plan_publishing(dataset: DatasetDir, *, enabled: bool) -> Publishing:
    """What publishing an evaluation of ``dataset``'s test split needs, read from local files.

    Raises :class:`StateError` when ``push.json`` cannot be read, with a hint to push again
    or pass ``--no-publish``.
    """
    if not enabled:
        return Publishing(enabled=False, dataset_dir=dataset.path)
    try:
        push = read_push_state(dataset)
    except StateError as exc:
        raise StateError(exc.message, hint=_push_hint(dataset.path)) from exc
    if push is None:
        return Publishing(
            enabled=True,
            dataset_dir=dataset.path,
            skipped=f"{dataset.path} has no push receipt ({push_path(dataset).name}), and "
            "experiments attach to the pushed dataset; " + _push_hint(dataset.path),
        )
    if push.dataset_id is None or push.in_flight:
        return Publishing(
            enabled=True,
            dataset_dir=dataset.path,
            push=push,
            skipped=f"the last push of {dataset.path} did not finish; " + _push_hint(dataset.path),
        )
    rows = dataset.rows_for_split("test")
    missing = [row.id for row in rows if row.id not in push.record_ids]
    if missing:
        return Publishing(
            enabled=True,
            dataset_dir=dataset.path,
            push=push,
            skipped=f"{len(missing)} test row(s) of {dataset.path} were not pushed (e.g. "
            f"{missing[0]}); " + _push_hint(dataset.path),
        )
    records = {row.id: dataset_record(row, "test") for row in rows}
    # A row pushed by a receipt that predates digests has none and counts as unchanged.
    changed = [
        row_id
        for row_id, record in records.items()
        if (sent := push.record_digests.get(row_id)) is not None and sent != record_digest(record)
    ]
    if changed:
        return Publishing(
            enabled=True,
            dataset_dir=dataset.path,
            push=push,
            skipped=f"{len(changed)} test row(s) of {dataset.path} changed after they were "
            f"pushed (e.g. {changed[0]}), so their remote records hold the content pushed "
            f"earlier, not what is evaluated; to publish, move {push_path(dataset).name} aside "
            "and push to a new --name, or pass --no-publish",
        )
    return Publishing(enabled=True, dataset_dir=dataset.path, push=push, records=records)


# --- the plan -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalSettings:
    """The evaluation flags, as given (``judge_model`` ``None`` for the default)."""

    out_dir: Path
    max_cases_per_trajectory: int = DEFAULT_MAX_CASES_PER_TRAJECTORY
    judge_model: str | None = None
    concurrency: int = DEFAULT_CONCURRENCY
    max_tokens: int = DEFAULT_MAX_TOKENS
    seed: int = DEFAULT_SEED
    skip_calibration: bool = False
    publish: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalPlan:
    """What an evaluation of one run replays, with which judge, into which directory."""

    run_id: str
    run_dir: Path
    provider: str
    profile: ModelProfile
    rendering: dict[str, Any]
    cases: list[ReplayCase]
    judge: JudgeConfig
    settings: EvalSettings
    publishing: Publishing
    runtime: Runtime

    @property
    def out_dir(self) -> Path:
        return self.settings.out_dir

    def identity(self, tuned_checkpoint: CheckpointRef) -> dict[str, Any]:
        from pup_finetune.evaluation.replay import evaluation_identity

        return evaluation_identity(
            run_id=self.run_id,
            tuned_checkpoint=tuned_checkpoint,
            base_model=self.profile.base_model,
            judge=self.judge,
            cases=self.cases,
            max_tokens=self.settings.max_tokens,
            rendering=self.rendering,
            seed=self.settings.seed,
            max_cases_per_trajectory=self.settings.max_cases_per_trajectory,
        )

    def calibration_calls(self) -> int:
        if self.settings.skip_calibration:
            return 0
        return 3 * min(CALIBRATION_CASES, len(self.cases))

    def evaluate_command(self) -> list[str]:
        """The standalone ``evaluate`` command for this plan, every setting pinned."""
        settings = self.settings
        words = ["pup", "finetune", "evaluate", str(self.run_dir)]
        words += ["--out-dir", str(settings.out_dir)]
        words += ["--max-cases-per-trajectory", str(settings.max_cases_per_trajectory)]
        words += ["--judge-model", self.judge.spec]
        words += ["--concurrency", str(settings.concurrency)]
        words += ["--max-tokens", str(settings.max_tokens), "--seed", str(settings.seed)]
        if settings.skip_calibration:
            words.append("--skip-calibration")
        if not settings.publish:
            words.append("--no-publish")
        return words

    def train_flags(self) -> list[str]:
        """The ``train --evaluate`` flags for this plan, every setting pinned."""
        settings = self.settings
        words = ["--evaluate", "--eval-out-dir", str(settings.out_dir)]
        words += ["--eval-max-cases-per-trajectory", str(settings.max_cases_per_trajectory)]
        words += ["--eval-judge-model", self.judge.spec]
        words += ["--eval-concurrency", str(settings.concurrency)]
        words += ["--eval-max-tokens", str(settings.max_tokens)]
        words += ["--eval-seed", str(settings.seed)]
        if settings.skip_calibration:
            words.append("--eval-skip-calibration")
        if not settings.publish:
            words.append("--eval-no-publish")
        return words

    def publisher(
        self, tuned_checkpoint: CheckpointRef, *, cancel: threading.Event | None = None
    ) -> Publisher | None:
        """The publisher of this evaluation, or ``None`` when it is not published.

        ``cancel``, when set, stops publishing between requests (a ``cancelled`` publication).
        """
        from pup_finetune.datadog.experiments import ExperimentsPublisher, PublishSettings

        publishing = self.publishing
        if not publishing.ready or publishing.push is None:
            return None
        identity = canonical_json(finite_json(self.identity(tuned_checkpoint)))
        settings = PublishSettings(
            eval_key=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32],
            run_id=self.run_id,
            provider=self.provider,
            base_model=self.profile.base_model,
            tuned_checkpoint=tuned_checkpoint.uri,
            judge_model=self.judge.spec,
            max_tokens=self.settings.max_tokens,
            push=publishing.push,
            records=publishing.records,
        )
        return ExperimentsPublisher(
            Pup(),
            settings,
            out_dir=self.out_dir,
            runtime=self.runtime,
            cancel=cancel,
            progress=info,
        )

    def preview(self, *, tuned: str, tuned_checkpoint: CheckpointRef | None) -> dict[str, Any]:
        """The calls the evaluation makes. With ``tuned_checkpoint`` (a standalone run), what
        ``--out-dir`` already holds under the same settings is left out of the counts."""
        from pup_finetune.datadog.experiments import experiment_name
        from pup_finetune.evaluation.replay import CAVEAT, TARGETS, EvalDir, remaining_work

        settings = self.settings
        warnings: list[str] = []
        pairs = len(TARGETS) * len(self.cases)
        remaining = {"done": 0, "generate": pairs, "judge": pairs}
        calibration = self.calibration_calls()
        eval_dir = EvalDir(self.out_dir)
        stored = eval_dir.stored_identity() if tuned_checkpoint is not None else None
        if tuned_checkpoint is not None and stored is not None:
            if finite_json(self.identity(tuned_checkpoint)) == stored:
                remaining = remaining_work(eval_dir, self.cases)
                if (eval_dir.read_calibration() or {}).get("ok"):
                    calibration = 0
            elif eval_dir.has_work():
                warnings.append(
                    f"{self.out_dir} holds an evaluation with other settings: "
                    "--confirm will refuse; pass a new --out-dir"
                )
        from pup_finetune.judges import destination as judge_destination

        trajectories = len({case.row_id for case in self.cases})
        judged_at = (
            "the offline fake judge (nothing is sent)"
            if self.judge.provider == "fake"
            else judge_destination(self.judge)
        )
        destination = (
            f"the test split's conversations are sent to {self.provider} to generate and "
            f"to {judged_at} to judge"
        )
        publish = self.publishing.status()
        if publish["status"] == "skipped":
            warnings.append(f"the evaluation will not be published: {publish['reason']}")
        elif publish["status"] == "planned":
            publish["experiments"] = [experiment_name(t, self.run_id) for t in TARGETS]
            publish["spans_per_experiment"] = trajectories + len(self.cases)
            destination += (
                ", and the cases with the generated answers and their scores are published to "
                "Datadog as experiments on the pushed dataset"
            )
        return {
            "run_id": self.run_id,
            "out_dir": str(self.out_dir),
            "cases": len(self.cases),
            "trajectories": trajectories,
            "targets": {"base": self.profile.base_model, "tuned": tuned},
            "judge_model": self.judge.spec,
            "judge_provider": self.judge.provider,
            "judge_endpoint": self.judge.base_url,
            "already_done": remaining["done"],
            "generation_calls": remaining["generate"],
            "judge_calls": remaining["judge"] + calibration,
            "calibration_calls": calibration,
            "max_generated_tokens": remaining["generate"] * settings.max_tokens,
            "max_tokens": settings.max_tokens,
            "concurrency": settings.concurrency,
            "seed": settings.seed,
            "max_cases_per_trajectory": settings.max_cases_per_trajectory,
            "publish": publish,
            "data_destination": destination,
            "caveat": CAVEAT,
            "warnings": warnings,
        }


def resolve_judge(judge_model: str | None, *, provider: str, runtime: Runtime) -> JudgeConfig:
    """The judge for ``--judge-model``: the default judge, or for a ``fake`` run the fake one,
    or the judge the spec names (:func:`pup_finetune.judges.parse_judge`).

    :data:`~pup_finetune.evaluation.fake_judge.FAKE_JUDGE_MODEL` is the offline judge, for
    development mode only. Raises :class:`UsageError` for an empty spec or an empty model
    after a provider prefix, and when a package the judge needs is not installed.
    """
    from pup_finetune.evaluation.fake_judge import FAKE_JUDGE_MODEL, fake_judge_config
    from pup_finetune.inference import DEFAULT_JUDGE_MODEL
    from pup_finetune.judges import check_available, parse_judge

    if judge_model is None and provider == "fake":
        judge_model = FAKE_JUDGE_MODEL
    if judge_model == FAKE_JUDGE_MODEL:
        if not runtime.dev_mode:
            raise UsageError(
                "the fake judge is for development only",
                hint="set PUP_FINETUNE_DEV=1, or pass another --judge-model",
            )
        return fake_judge_config()
    config = parse_judge(DEFAULT_JUDGE_MODEL if judge_model is None else judge_model)
    check_available(config)
    return config


def plan_evaluation(
    runtime: Runtime,
    *,
    run_id: str,
    run_dir: Path,
    provider: str,
    profile: ModelProfile,
    dataset: DatasetDir,
    rendering: Mapping[str, Any],
    settings: EvalSettings,
    promoted: ModelRef | None = None,
) -> EvalPlan:
    """The evaluation of the run ``run_id`` on ``dataset``'s test split.

    Raises :class:`UsageError` when the test split has no case, or the judge is the model
    being trained or the run's tuned model: the model id the run promotes to
    (:func:`~pup_finetune.runs.model_id_for_run`) or, once known, the ``promoted`` model.
    """
    from pup_finetune.evaluation.cases import cases_from_dataset
    from pup_finetune.evaluation.judge import check_judge_differs

    judge_config = resolve_judge(settings.judge_model, provider=provider, runtime=runtime)
    tuned = [model_id_for_run(run_id)]
    if promoted is not None:
        tuned += [promoted.model_id, promoted.uri]
    check_judge_differs(judge_config, profile.base_model, tuned)
    cases = cases_from_dataset(
        dataset,
        split="test",
        max_cases_per_trajectory=settings.max_cases_per_trajectory,
        seed=settings.seed,
    )
    if not cases:
        rows = len(dataset.rows_for_split("test"))
        raise UsageError(
            f"nothing to evaluate: the test split of {dataset.path} has {rows} kept rows and "
            "no assistant turn to replay",
            hint="import more rows so the test split holds some conversations",
        )
    return EvalPlan(
        run_id=run_id,
        run_dir=run_dir,
        provider=provider,
        profile=profile,
        rendering=dict(rendering),
        cases=cases,
        judge=judge_config,
        settings=settings,
        publishing=plan_publishing(dataset, enabled=settings.publish),
        runtime=runtime,
    )


# --- the judge ----------------------------------------------------------------------------


@contextmanager
def open_judge_client(config: JudgeConfig) -> Iterator[ChatCompleter]:
    """The client for ``config``'s judge: the offline fake judge, or the provider's client
    (:func:`pup_finetune.judges.open_client`) with the key of its credential. Closed on
    exit."""
    from pup_finetune.evaluation.fake_judge import FakeJudge
    from pup_finetune.judges import open_client

    if config.provider == "fake":
        yield FakeJudge()
        return
    with open_client(config) as client:
        yield client


def run_calibration(plan: EvalPlan, client: ChatCompleter) -> dict[str, Any] | None:
    """The calibration report of ``plan``'s judge, or ``None`` with ``--skip-calibration``."""
    from pup_finetune.evaluation.judge import calibrate

    if plan.settings.skip_calibration:
        return None
    info(f"calibrating the judge {plan.judge.spec} on {plan.calibration_calls() // 3} cases")
    report = calibrate(plan.cases, client, plan.judge, n=CALIBRATION_CASES).to_json()
    return {**report, "judge": plan.judge.identity()}


def calibration_failed(plan: EvalPlan, report: Mapping[str, Any]) -> dict[str, Any]:
    """What a command emits (exit 1, see :func:`evaluation_exit_code`) when calibration
    failed: nothing was generated."""
    return {
        "status": "calibration_failed",
        "run_id": plan.run_id,
        "out_dir": str(plan.out_dir),
        "judge_model": plan.judge.spec,
        "calibration": dict(report),
        "hint": "the judge passed a corrupted action or failed a recorded one, so its "
        "verdicts cannot be trusted; pick another --judge-model (or rerun with "
        "--skip-calibration to evaluate anyway)",
    }


def require_new_eval_dir(out_dir: Path, run_dir: Path) -> None:
    """For ``train --evaluate``, before anything is trained: ``out_dir`` must be new.

    It must be missing or empty (apart from a lock file), and not the run directory. A new
    run has a new run id, so an earlier evaluation in ``out_dir`` could never be continued,
    and the run directory is locked while it trains. Raises :class:`UsageError`.
    """
    hint = "pass a new --eval-out-dir (the default is RUN_DIR/eval)"
    if out_dir.resolve() == run_dir.resolve():
        raise UsageError(f"--eval-out-dir {out_dir} is the run directory", hint=hint)
    if not out_dir.exists():
        return
    if not out_dir.is_dir():
        raise UsageError(f"--eval-out-dir {out_dir} exists and is not a directory", hint=hint)
    if any(entry.name != LOCK_FILE_NAME for entry in out_dir.iterdir()):
        raise UsageError(
            f"--eval-out-dir {out_dir} is not empty, and a new run cannot continue the "
            "evaluation of another",
            hint=hint,
        )


def prepare_eval_dir(
    plan: EvalPlan,
    tuned_checkpoint: CheckpointRef,
    client: ChatCompleter,
    *,
    calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Create ``--out-dir``, check its identity and settle the calibration; the report.

    A calibration that passed is stored and reused by later runs in the directory. Unless
    one is stored or given in ``calibration``, the judge is calibrated now (see
    :func:`run_calibration`). ``None`` with ``--skip-calibration``.
    """
    from pup_finetune.evaluation.replay import EvalDir

    eval_dir = EvalDir(plan.out_dir)
    eval_dir.create()
    with eval_dir.lock():
        eval_dir.check_identity(plan.identity(tuned_checkpoint))
        if plan.settings.skip_calibration:
            return None
        stored = eval_dir.read_calibration()
        if stored is not None and stored.get("ok"):
            return stored
        report = dict(calibration) if calibration is not None else run_calibration(plan, client)
        if report is not None:
            eval_dir.write_calibration(report)
        return report


# --- the replay ---------------------------------------------------------------------------


def cancel_on_signals(cancel: threading.Event) -> contextlib.AbstractContextManager[None]:
    """Make SIGTERM and the first Ctrl-C set ``cancel``; a second Ctrl-C stops at once.

    See :func:`pup_finetune.runs.cancel_on_signals`; main thread only.
    """
    return _cancel_on_signals(cancel, _CANCEL_NOTICE)


def _progress(event: Mapping[str, Any]) -> None:
    info(
        f"evaluated {event['done']}/{event['total']}: {event['target']} {event['case_id']} "
        f"({event['status']})"
    )


def replay(
    plan: EvalPlan,
    samplers: Mapping[str, Sampler],
    client: ChatCompleter,
    tuned_checkpoint: CheckpointRef,
    *,
    cancel: threading.Event,
) -> ReplaySummary:
    """:func:`~pup_finetune.evaluation.replay.run_replay` of ``plan`` with ``client`` judging,
    published by :meth:`EvalPlan.publisher` once complete."""
    from pup_finetune.evaluation.judge import judge
    from pup_finetune.evaluation.replay import run_replay

    config = plan.judge

    def judge_fn(case: ReplayCase, generation: Generation) -> Judgment:
        return judge(case, generation.message, client, config, format_valid=generation.format_valid)

    return run_replay(
        plan.cases,
        samplers,
        judge_fn,
        out_dir=plan.out_dir,
        identity=plan.identity(tuned_checkpoint),
        concurrency=plan.settings.concurrency,
        cancel=cancel,
        publisher=plan.publisher(tuned_checkpoint, cancel=cancel),
        on_progress=_progress,
    )


_SENT: Final = frozenset({"published", "failed", "cancelled"})
"""Publication statuses after which some spans may have been sent."""


def _publication(plan: EvalPlan, outcome: ReplaySummary) -> dict[str, Any]:
    if outcome.publication is not None:
        return dict(outcome.publication)
    status = plan.publishing.status()
    if status["status"] == "planned" and not outcome.complete:
        return {
            "status": "pending",
            "reason": "the evaluation is published once every case has a result; run "
            "next_command to finish it",
        }
    return status


def replay_result(
    plan: EvalPlan,
    outcome: ReplaySummary,
    *,
    calibration: Mapping[str, Any] | None,
    tuned_sampler: CheckpointRef | None,
) -> dict[str, Any]:
    """What a finished (or cancelled) evaluation emits.

    ``status`` is ``complete``, ``complete_with_errors`` (every pair has a result, but some
    are errors, which a rerun of ``next_command`` retries) or ``partial`` (cancelled).
    ``publication`` says how the evaluation was published (``published``, ``failed`` or
    ``cancelled``, which a rerun of ``next_command`` continues, ``pending``, ``skipped`` or
    ``disabled``). Spans already published keep the results they were sent with, so after
    a retry of the errors the experiments still show them as errors.
    """
    summary = outcome.summary
    errors = sum(int(count) for count in summary["errors"].values())
    status = "partial"
    if outcome.complete:
        status = "complete_with_errors" if errors else "complete"
    warnings: list[str] = []
    hint = None
    publication = _publication(plan, outcome)
    if errors:
        hint = (
            f"{errors} case(s) ended in an error, left out of every rate (see results.jsonl "
            "in out_dir); run next_command to retry them"
        )
        too_long = sum(
            int(target.get("errors_too_long", 0)) for target in summary["targets"].values()
        )
        if too_long:
            hint += (
                f". {too_long} of them are longer than the judge's context window, which a "
                "retry cannot fix: evaluate into a new --out-dir with a --judge-model that has "
                "a larger context window"
            )
        if publication["status"] in _SENT:
            hint += (
                ". Spans already published keep the results they were sent with, these errors "
                "included: a retry updates the local results only, so evaluate into a new "
                "--out-dir to publish experiments with every result"
            )
    no_verdict = sum(int(target["judge_no_verdict"]) for target in summary["targets"].values())
    if no_verdict:
        warnings.append(
            f"{no_verdict} judge answer(s) held no verdict (unparseable, cut off at the "
            "judge's max_tokens, or refused by the judge's provider) and count as failures; "
            "see judge_no_verdict per target"
        )
    warnings.extend(str(warning) for warning in publication.get("warnings") or [])
    if publication["status"] == "skipped":
        warnings.append(f"the evaluation was not published: {publication['reason']}")
    elif publication["status"] == "failed":
        error = publication["error"]
        then = f"{error['hint']}, then run" if error.get("hint") else "run"
        warnings.append(
            f"publishing the evaluation failed: {error['message']}; {then} next_command to "
            "publish the rest"
        )
    elif publication["status"] == "cancelled":
        warnings.append(
            "publishing the evaluation was cancelled; run next_command to publish the rest"
        )
    retry = status != "complete" or publication["status"] in {"failed", "cancelled"}
    return finite_json(
        {
            "status": status,
            "run_id": plan.run_id,
            "out_dir": str(outcome.out_dir),
            "summary": summary,
            "pending": outcome.pending,
            "cancelled": outcome.cancelled,
            "errors": errors,
            "publication": publication,
            "warnings": warnings,
            "hint": hint,
            "calibration": None
            if calibration is None
            else {"ok": calibration.get("ok"), "cases": calibration.get("cases")},
            "tuned_checkpoint": None
            if tuned_sampler is None
            else dataclasses.asdict(tuned_sampler),
            "next_command": shlex.join(plan.evaluate_command()) if retry else None,
        }
    )


def evaluation_exit_code(result: Mapping[str, Any]) -> int:
    """1 when an evaluation's result says it failed: it raised (``failed``), its judge failed
    calibration, its publishing failed, or every pair of a target ended in an error, so there
    is nothing to compare; otherwise 0."""
    if result.get("status") in {"failed", "calibration_failed"}:
        return 1
    publication = result.get("publication")
    if isinstance(publication, Mapping) and publication.get("status") == "failed":
        return 1
    if result.get("status") != "complete_with_errors":
        return 0
    targets = result["summary"]["targets"].values()
    return 1 if any(int(target["completed"]) == 0 for target in targets) else 0


SPEND_POLL_S: Final = 0.5
"""How often :class:`EvaluateAfterTraining` checks the spend cap (and a cancel) while it
evaluates."""


class EvaluateAfterTraining:
    """The ``after_finalize`` hook of ``train --evaluate``: replay in the training session.

    ``outcome`` is what to report: the evaluation's result, or why it was skipped or failed.
    A failure is also raised, for the run's warnings. With the run's spend guard (see
    :data:`~pup_finetune.training.runner.AfterFinalize`), the evaluation is skipped when the
    cap is already reached, and stopped like a cancel, every ``poll_s``, once it is.
    """

    def __init__(
        self,
        plan: EvalPlan,
        client: ChatCompleter,
        *,
        calibration: Mapping[str, Any] | None,
        max_checkpoint_name_len: int,
        poll_s: float | None = None,
    ) -> None:
        self.plan = plan
        self.client = client
        self.calibration = calibration
        self.max_checkpoint_name_len = max_checkpoint_name_len
        self.poll_s = SPEND_POLL_S if poll_s is None else poll_s
        self.outcome: dict[str, Any] | None = None

    def __call__(
        self,
        session: TrainingSession,
        result: TrainResult,
        cancel: threading.Event,
        spend_guard: Callable[[], str | None] | None = None,
    ) -> None:
        next_command = shlex.join(self.plan.evaluate_command())
        spent = None if spend_guard is None else spend_guard()
        if result.stopped_reason not in EVALUATED_STOPS or spent is not None:
            why = {
                "cancelled": "it was cancelled",
                "budget_exhausted": "it reached its spend cap, which evaluating in its session "
                "would spend past",
            }.get(result.stopped_reason, f"it stopped: {result.stopped_reason}")
            if result.stopped_reason in EVALUATED_STOPS:
                why = f"its spend cap is reached ({spent})"
            self.outcome = {
                "status": "skipped",
                "reason": f"the run was not evaluated because {why}; next_command evaluates "
                "it in a session of its own",
                "next_command": next_command,
            }
            return
        try:
            with self._capped(cancel, spend_guard) as (stop, spend_stop):
                self._evaluate(session, result, stop)
            if spend_stop and self.outcome is not None and self.outcome["status"] == "partial":
                reason = (
                    f"the evaluation stopped at the run's spend cap ({spend_stop[0]}); "
                    "next_command continues it in a session of its own"
                )
                self.outcome["reason"] = reason
                self.outcome["warnings"] = [*self.outcome.get("warnings", []), reason]
        except BaseException as exc:
            self.outcome = {
                "status": "failed",
                "error": error_json(exc),
                "out_dir": str(self.plan.out_dir),
                "next_command": next_command,
            }
            raise

    def _evaluate(
        self, session: TrainingSession, result: TrainResult, stop: threading.Event
    ) -> None:
        from pup_finetune.evaluation.targets import best_training_checkpoint, session_samplers

        training = best_training_checkpoint(result)
        report = prepare_eval_dir(self.plan, training, self.client, calibration=self.calibration)
        info(f"evaluating the run {self.plan.run_id} in its training session")
        with session_samplers(
            session,
            result,
            run_id=self.plan.run_id,
            max_checkpoint_name_len=self.max_checkpoint_name_len,
        ) as targets:
            outcome = replay(self.plan, targets.samplers, self.client, training, cancel=stop)
        self.outcome = replay_result(
            self.plan, outcome, calibration=report, tuned_sampler=targets.tuned_checkpoint
        )

    @contextmanager
    def _capped(
        self, cancel: threading.Event, spend_guard: Callable[[], str | None] | None
    ) -> Iterator[tuple[threading.Event, list[str]]]:
        """An event set on ``cancel`` or once ``spend_guard`` says the cap is reached, and
        the guard's reason (in a list, empty until it fires)."""
        if spend_guard is None:
            yield cancel, []
            return
        stop = threading.Event()
        done = threading.Event()
        reasons: list[str] = []

        def watch() -> None:
            while not done.wait(self.poll_s):
                if cancel.is_set():
                    stop.set()
                    return
                why = spend_guard()
                if why is not None:
                    reasons.append(why)
                    info(f"stopping the evaluation at the run's spend cap: {why}")
                    stop.set()
                    return

        watcher = threading.Thread(target=watch, name="eval-spend-cap", daemon=True)
        watcher.start()
        try:
            yield stop, reasons
        finally:
            done.set()
            watcher.join()


# --- evaluate -----------------------------------------------------------------------------


def finished_result(run: RunDir) -> TrainResult:
    """The training result of a completed or cancelled run (:data:`FINISHED_PHASES`).

    Raises :class:`UsageError` for another phase. A cancelled run that finished no epoch has
    nothing to evaluate either, which
    :func:`~pup_finetune.evaluation.targets.best_training_checkpoint` refuses.
    """
    state = run.load_state()
    if state.phase not in FINISHED_PHASES:
        raise UsageError(
            f"the run in {run.path} is {state.phase}, not completed or cancelled, so there is "
            "nothing to evaluate yet",
            hint=f"see `pup finetune status {_q(run.path)}`",
        )
    stored = run.read_result()
    if stored is None:
        raise StateError(f"{run.result_path} does not exist")
    return TrainResult.from_json(stored)


def _check_run_inputs(run: RunDir, manifest: Mapping[str, Any], dataset: DatasetDir) -> None:
    """The dataset must hold the rows the run trained on, so the test split stays held out."""
    dataset.settings()  # a clear error when it is not a dataset directory any more
    if dataset_content_hash(dataset) != manifest.get("dataset_sha256"):
        raise StateError(
            f"the dataset in {dataset.path} changed since the run in {run.path} was trained, "
            "so its test split may no longer be held out",
            hint="restore the dataset the run trained on, or train a new run",
        )


def _check_renderer(run: RunDir, manifest: Mapping[str, Any], provider: Provider) -> None:
    provider.validate_options(run.spec().provider_options)
    profile = run.profile()
    if provider.renderer(profile).identity() != manifest.get("rendering"):
        raise StateError(
            f"the {profile.alias} renderer changed since the run in {run.path} was trained, "
            "so the tuned model would be prompted differently from how it learned",
            hint="evaluate with the version of pup-finetune the run was trained with",
        )


@dataclass(frozen=True, slots=True)
class StandaloneEvaluation:
    """The ``evaluate`` command for one finished run: its plan and what it samples."""

    plan: EvalPlan
    provider: Provider
    run: RunDir
    result: TrainResult
    training: CheckpointRef
    """The best epoch's training checkpoint: the weights the tuned target samples."""

    def preview(self) -> dict[str, Any]:
        tuned = f"epoch {self.result.best_epoch} of run {self.plan.run_id} ({self.training.name})"
        preview = self.plan.preview(tuned=tuned, tuned_checkpoint=self.training)
        preview["run_dir"] = str(self.plan.run_dir)
        return preview

    def execute(self) -> dict[str, Any]:
        """Calibrate, replay and publish; what the command emits (see
        :func:`evaluation_exit_code` for its exit code).

        Ctrl-C or SIGTERM stops it gracefully (a ``partial`` result); a second Ctrl-C stops
        at once with :class:`FinetuneError`.
        """
        plan = self.plan
        cancel = threading.Event()
        with open_judge_client(plan.judge) as client:
            report = prepare_eval_dir(plan, self.training, client)
            if report is not None and not report.get("ok"):
                return calibration_failed(plan, report)
            try:
                with cancel_on_signals(cancel):
                    return self._replay(client, cancel, report)
            except KeyboardInterrupt:
                raise FinetuneError(
                    f"evaluation stopped at once (Ctrl-C); what finished is kept in {plan.out_dir}",
                    hint=f"continue it with `{shlex.join(plan.evaluate_command())}`",
                ) from None

    def _replay(
        self, client: ChatCompleter, cancel: threading.Event, report: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        from pup_finetune.evaluation.replay import EvalDir, remaining_work
        from pup_finetune.evaluation.targets import run_samplers

        plan = self.plan
        if remaining_work(EvalDir(plan.out_dir), plan.cases)["generate"] == 0:
            info(f"nothing left to generate in {plan.out_dir}: judging and summarizing only")
            outcome = replay(plan, {}, client, self.training, cancel=cancel)
            return replay_result(plan, outcome, calibration=report, tuned_sampler=None)
        info(
            f"evaluating run {plan.run_id}: {len(plan.cases)} cases, base and tuned (Ctrl-C stops)"
        )
        with run_samplers(self.provider, self.run, self.result) as targets:
            outcome = replay(plan, targets.samplers, client, self.training, cancel=cancel)
        return replay_result(
            plan, outcome, calibration=report, tuned_sampler=targets.tuned_checkpoint
        )


def plan_standalone(
    runtime: Runtime, run_dir: Path, settings: EvalSettings
) -> StandaloneEvaluation:
    """The evaluation of the finished run in ``run_dir``, checked against its inputs.

    Raises :class:`UsageError` when the run is neither completed nor cancelled, and
    :class:`StateError` when it has no best epoch or its dataset or renderer changed since it
    was trained.
    """
    from pup_finetune.evaluation.targets import best_training_checkpoint

    run = RunDir(run_dir)
    manifest = run.manifest()
    result = finished_result(run)
    training = best_training_checkpoint(result)
    provider = get_provider(str(manifest["provider"]), runtime)
    dataset = DatasetDir(Path(str(manifest["dataset_dir"])))
    _check_run_inputs(run, manifest, dataset)
    _check_renderer(run, manifest, provider)
    plan = plan_evaluation(
        runtime,
        run_id=run.run_id,
        run_dir=run_dir,
        provider=provider.name,
        profile=run.profile(),
        dataset=dataset,
        rendering=manifest.get("rendering") or {},
        settings=settings,
        promoted=result.promoted_model,
    )
    return StandaloneEvaluation(
        plan=plan, provider=provider, run=run, result=result, training=training
    )
