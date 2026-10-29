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
experiments on the pushed dataset (:mod:`pup_finetune.datadog.experiments`), three with a
compare target, unless publishing is off (``--no-publish``) or the dataset has no push
receipt, which is a warning: the local results are complete either way.

``evaluate --compare-model`` adds a third target, the compare model
(:mod:`pup_finetune.evaluation.compare`), to the same evaluation directory under an identity
of its own (``compare.json``), so it can be added to a complete evaluation: only its pairs
are generated and judged, and no training session opens when base and tuned are done.
``evaluate --evaluators`` adds the user's own judges and checks
(:mod:`pup_finetune.evaluation.evaluators`) the same way: added to a complete evaluation,
they score the stored answers alone.

The evaluation modules this drives (and through them the judge's HTTP client) are imported
only when a function here runs, so the CLI can import this module for every command.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import os
import shlex
import threading
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from pup_finetune.billing import BillingPart, billing_fields
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.data.push import (
    PUSH_FILE,
    PushState,
    dataset_record,
    push_path,
    read_push_state,
    record_digest,
)
from pup_finetune.datadog.pup import Pup
from pup_finetune.display import display_path, joined, quoted
from pup_finetune.errors import (
    FinetuneError,
    Interrupted,
    SettingsConflict,
    StateError,
    UsageError,
    WrongPhase,
)
from pup_finetune.next_steps import PROGRAM, command, step, template
from pup_finetune.output import info
from pup_finetune.progress import EvaluationProgress
from pup_finetune.providers import (
    get_provider,
    provider_info,
    require_packages,
    sampling_billing,
)
from pup_finetune.providers.base import (
    CheckpointRef,
    Generation,
    ModelProfile,
    ModelRef,
    Provider,
    Sampler,
    TrainingSession,
)
from pup_finetune.redaction import redact
from pup_finetune.runs import (
    Phase,
    RunDir,
    dataset_content_hash,
    error_json,
    model_id_for_run,
    utcnow,
)
from pup_finetune.runs import cancel_on_signals as _cancel_on_signals
from pup_finetune.runtime import Runtime
from pup_finetune.storage import (
    LOCK_FILE_NAME,
    canonical_json,
    finite_json,
    free_sibling,
    identity_changes,
)
from pup_finetune.training.records import StoppedReason, TrainResult

if TYPE_CHECKING:
    from pup_finetune.evaluation.cases import Exclusion, ReplayCase
    from pup_finetune.evaluation.compare import CompareTarget
    from pup_finetune.evaluation.evaluators import CheckEvaluator, EvaluatorSet, JudgeEvaluator
    from pup_finetune.evaluation.judge import Judgment
    from pup_finetune.evaluation.replay import EvalDir, PairKey, Publisher, ReplaySummary
    from pup_finetune.evaluation.reuse import ReuseIndex, TargetReuse
    from pup_finetune.inference import ChatCompleter, ChatCompletion, JudgeConfig

__all__ = [
    "CALIBRATION_CASES",
    "CALIBRATION_FAILURES_SHOWN",
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
    "EvalWork",
    "EvaluateAfterTraining",
    "EvaluatorsPreview",
    "Publishing",
    "RecordedEvaluation",
    "StandaloneEvaluation",
    "busy_judges",
    "calibrate_evaluators",
    "calibration_failed",
    "calibration_line",
    "cancel_on_signals",
    "check_hint",
    "deploy_command",
    "evaluate_words",
    "evaluation_exit_code",
    "evaluator_calibration_calls",
    "evaluator_errors",
    "evaluator_fix",
    "evaluators_preview",
    "finished_result",
    "headline",
    "next_after_training",
    "no_reuse_hint",
    "open_evaluator_clients",
    "open_judge_client",
    "plan_evaluation",
    "plan_standalone",
    "prepare_eval_dir",
    "record_evaluation",
    "recorded_evaluation",
    "replay",
    "replay_result",
    "require_new_eval_dir",
    "resolve_judge",
    "run_calibration",
    "scoring_warning",
    "share_base",
    "uncalibrated",
]

DEFAULT_OUT_DIR_NAME: Final = "eval"
DEFAULT_MAX_CASES_PER_TRAJECTORY: Final = 2
DEFAULT_CONCURRENCY: Final = 4
DEFAULT_MAX_TOKENS: Final = 4096
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
    return quoted(value)


def no_reuse_hint(stored: Mapping[str, Any], wanted: Mapping[str, Any]) -> str:
    """The way to continue a directory that only ``--no-reuse`` keeps apart, else ``""``."""
    if identity_changes(stored, wanted) == ["reuse"] and stored.get("reuse") is False:
        return "it was made with --no-reuse: add --no-reuse to continue it, or "
    return ""


# --- publishing ---------------------------------------------------------------------------

Remedy = Literal["push", "push_again", "new_dataset"]


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
    remedy: Remedy | None = None
    """What makes a skipped evaluation publishable: ``push`` the dataset (it never was),
    ``push_again`` to the same dataset (the last push did not finish, or left test rows
    out), or push to a ``new_dataset`` (test rows changed after they were pushed)."""

    @property
    def ready(self) -> bool:
        return self.enabled and self.skipped is None and self.push is not None

    def push_step(self) -> dict[str, str] | None:
        """The ``dataset push`` that makes a skipped evaluation publishable, as a step."""
        push = self.push
        if self.remedy == "push_again" and push is not None:
            return step(
                command(
                    "dataset",
                    "push",
                    self.dataset_dir,
                    "--name",
                    push.dataset_name,
                    "--project",
                    push.project,
                ),
                "finish pushing the dataset's rows, the test split's included, to the dataset "
                "the evaluation publishes on (a remote write; it previews first)",
            )
        if self.remedy == "push":
            why = "push the dataset, so the evaluation can be published as experiments on it"
        elif self.remedy == "new_dataset":
            why = (
                f"after moving {PUSH_FILE} aside, push the rows as they are now to a new "
                "dataset, since test rows changed after they were pushed"
            )
        else:
            return None
        return step(
            template("dataset", "push", self.dataset_dir, "--name", "<NAME>"),
            f"{why} (a remote write; it previews first)",
        )

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


NO_PUBLISH: Final = "--no-publish"
"""``evaluate``'s flag that skips publishing; ``train --evaluate`` spells it
:data:`EVAL_NO_PUBLISH`."""
EVAL_NO_PUBLISH: Final = "--eval-no-publish"


def _push_hint(dataset_dir: Path, no_publish: str) -> str:
    return (
        f"run `pup finetune dataset push {_q(dataset_dir)} --name <NAME>` to publish the "
        f"evaluation as experiments, or pass {no_publish}"
    )


def plan_publishing(
    dataset: DatasetDir, *, enabled: bool, no_publish: str = NO_PUBLISH
) -> Publishing:
    """What publishing an evaluation of ``dataset``'s test split needs, read from local files.

    ``no_publish`` is how the command running it spells the flag that skips publishing
    (:data:`NO_PUBLISH`, or :data:`EVAL_NO_PUBLISH` for ``train --evaluate``), which the
    reasons and hints name. Raises :class:`StateError` when ``push.json`` cannot be read,
    with a hint to push again or pass that flag.
    """
    if not enabled:
        return Publishing(enabled=False, dataset_dir=dataset.path)
    try:
        push = read_push_state(dataset)
    except StateError as exc:
        # The same class (a settings conflict stays one), with the hint for this command.
        raise type(exc)(exc.message, hint=_push_hint(dataset.path, no_publish)) from exc
    if push is None:
        return Publishing(
            enabled=True,
            dataset_dir=dataset.path,
            skipped=f"{display_path(dataset.path)} has no push receipt ({push_path(dataset).name}), and "
            "experiments attach to the pushed dataset; " + _push_hint(dataset.path, no_publish),
            remedy="push",
        )
    if push.dataset_id is None or push.in_flight:
        return Publishing(
            enabled=True,
            dataset_dir=dataset.path,
            push=push,
            skipped=f"the last push of {display_path(dataset.path)} did not finish; "
            + _push_hint(dataset.path, no_publish),
            remedy="push_again",
        )
    rows = dataset.rows_for_split("test")
    missing = [row.id for row in rows if row.id not in push.record_ids]
    if missing:
        return Publishing(
            enabled=True,
            dataset_dir=dataset.path,
            push=push,
            skipped=f"{len(missing)} test row(s) of {display_path(dataset.path)} were not pushed (e.g. "
            f"{missing[0]}); " + _push_hint(dataset.path, no_publish),
            remedy="push_again",
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
            skipped=f"{len(changed)} test row(s) of {display_path(dataset.path)} changed after they were "
            f"pushed (e.g. {changed[0]}), so their remote records hold the content pushed "
            f"earlier, not what is evaluated; to publish, move {push_path(dataset).name} aside "
            f"and push to a new --name, or pass {no_publish}",
            remedy="new_dataset",
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
    compare_model: str | None = None
    reuse: bool = True
    evaluators: Path | None = None


def evaluate_words(
    run_dir: Path,
    settings: EvalSettings,
    judge_spec: str,
    *,
    pin: bool,
    compare_spec: str | None = None,
    evaluators: Path | None = None,
) -> list[str]:
    """The ``evaluate`` command (without ``--confirm``) for the run in ``run_dir`` with
    ``settings`` and the judge ``judge_spec``: the judge always, and the other settings that
    differ from the defaults, or every one of them with ``pin``. ``compare_spec``, the
    compare model, and the ``evaluators`` spec (while it exists) are repeated whenever there
    is one: like the judge, they decide where the data goes."""
    defaults = EvalSettings(out_dir=run_dir / DEFAULT_OUT_DIR_NAME)
    words = [*PROGRAM, "evaluate", display_path(run_dir)]
    if pin or settings.out_dir != defaults.out_dir:
        words += ["--out-dir", display_path(settings.out_dir)]
    if pin or settings.max_cases_per_trajectory != defaults.max_cases_per_trajectory:
        words += ["--max-cases", str(settings.max_cases_per_trajectory)]
    words += ["--judge-model", judge_spec]
    if compare_spec is not None:
        words += ["--compare-model", compare_spec]
    if evaluators is not None and evaluators.is_file():
        words += ["--evaluators", display_path(evaluators)]
    if pin or settings.concurrency != defaults.concurrency:
        words += ["--concurrency", str(settings.concurrency)]
    if pin or settings.max_tokens != defaults.max_tokens:
        words += ["--max-tokens", str(settings.max_tokens)]
    if pin or settings.seed != defaults.seed:
        words += ["--seed", str(settings.seed)]
    if settings.skip_calibration:
        words.append("--skip-calibration")
    if not settings.publish:
        words.append("--no-publish")
    if not settings.reuse:
        words.append("--no-reuse")
    return words


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalWork:
    """What an evaluation would still do (:meth:`EvalPlan.work`)."""

    by_target: dict[str, dict[str, Any]]
    """Each target's ``{"done", "generate", "judge"}`` pairs, and with evaluators the pairs
    each still scores (``"evaluators": {name: n}``)."""
    calibration: int
    """The judge calls calibration makes: none once one passed in the directory."""
    probe: bool
    """Whether the compare model answers one call before it is attached."""
    reused: dict[str, TargetReuse] = field(default_factory=dict)
    """What each target takes from the dataset's store, already left out of
    :attr:`by_target`."""
    evaluator_calibration: dict[str, int] = field(default_factory=dict)
    """The example calls each of the user's judges still makes to calibrate."""

    def evaluator_calls(self, name: str) -> int:
        """The pairs the evaluator ``name`` still scores, over every target."""
        return sum(
            int(counts.get("evaluators", {}).get(name, 0)) for counts in self.by_target.values()
        )

    @property
    def judge_calls(self) -> int:
        return sum(int(counts["judge"]) for counts in self.by_target.values()) + self.calibration


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
    judge_source: str
    """Where the judge came from (see :func:`resolve_judge`)."""
    judge_warnings: tuple[str, ...] = ()
    """What the preview warns about the default judge, such as a malformed key passed over."""
    settings: EvalSettings
    publishing: Publishing
    runtime: Runtime
    standalone: bool = False
    """``evaluate`` (not ``train --evaluate``): its commands repeat only the settings that
    differ from the defaults, and the judge (see :meth:`evaluate_command`)."""
    compare: CompareTarget | None = None
    """The compare model: ``--compare-model``'s, else the one the directory has attached."""
    excluded: Mapping[str, Exclusion] = field(default_factory=dict)
    """The test cases left out of :attr:`cases` because the run's model cannot replay them,
    by case id (:func:`~pup_finetune.evaluation.cases.exclusions`)."""
    base_keys: Mapping[str, str] = field(default_factory=dict)
    """Each case's base answer key in the dataset's store, by case id."""
    reuse: ReuseIndex | None = None
    """What the dataset's store holds for this plan, read once when it was planned; ``None``
    with ``--no-reuse``."""
    evaluators: EvaluatorSet | None = None
    """The user's own evaluators: ``--evaluators``, else those the directory has attached."""
    compare_skips: Mapping[str, str] = field(init=False)
    """The cases the compare model's API refuses, by case id, with the reason; none
    without a compare model (:func:`~pup_finetune.evaluation.compare.compare_skips`)."""
    replays: dict[str, list[ReplayCase]] = field(init=False)
    """The cases each target replays: every one for base and tuned, and for the compare
    model those its API takes (:attr:`compare_skips`)."""

    def __post_init__(self) -> None:
        from pup_finetune.evaluation.compare import compare_skips
        from pup_finetune.evaluation.replay import COMPARE

        skips = (
            {} if self.compare is None else compare_skips(self.cases, self.compare.config.provider)
        )
        replays = dict.fromkeys(self.targets, self.cases)
        if self.compare is not None:
            replays[COMPARE] = [case for case in self.cases if case.id not in skips]
        object.__setattr__(self, "compare_skips", skips)
        object.__setattr__(self, "replays", replays)

    @property
    def targets(self) -> tuple[str, ...]:
        """The targets replayed: base and tuned, and the compare model when there is one."""
        from pup_finetune.evaluation.replay import COMPARE, TARGETS

        return (*TARGETS, COMPARE) if self.compare is not None else TARGETS

    @property
    def compare_identity(self) -> dict[str, Any] | None:
        """``compare.json`` for this plan's compare target, ``None`` without one."""
        return None if self.compare is None else self.compare.identity()

    @property
    def out_dir(self) -> Path:
        return self.settings.out_dir

    @property
    def offline(self) -> bool:
        """Whether nothing it does leaves the machine: the fake provider generates, the fake
        judge judges and nothing is published. ``PUP_READ_ONLY`` allows such a run."""
        return (
            provider_info(self.provider, self.runtime).offline
            and self.judge.offline
            and (self.compare is None or self.compare.offline)
            and (self.evaluators is None or self.evaluators.offline)
            and not (self.publishing.ready)
        )

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
            reuse=self.settings.reuse,
        )

    @property
    def base_inputs(self) -> dict[str, Any]:
        """What base answers are sampled under, for their keys in the dataset's store."""
        from pup_finetune.evaluation.reuse import key_inputs

        return key_inputs(self.provider, self.profile, self.rendering, self.settings.max_tokens)

    def base_reuse(self, eval_dir: EvalDir | None) -> TargetReuse | None:
        """What base takes from the store for the pairs ``eval_dir`` has not done (every
        pair without one); ``None`` with ``--no-reuse``."""
        from pup_finetune.evaluation.reuse import plan_target

        if self.reuse is None:
            return None
        generations = {} if eval_dir is None else eval_dir.read_generations()
        results = {} if eval_dir is None else eval_dir.read_results()
        return plan_target(
            self.reuse,
            "base",
            self.replays["base"],
            self.base_keys,
            self.judge,
            generations,
            results,
        )

    def calibration_calls(self) -> int:
        if self.settings.skip_calibration:
            return 0
        return 3 * min(CALIBRATION_CASES, len(self.cases))

    def evaluate_command(self, *, pin: bool | None = None, evaluators: bool = False) -> list[str]:
        """The standalone ``evaluate`` command for this plan, without ``--confirm``.

        For ``evaluate`` it repeats the settings that differ from the defaults, and always
        the judge, which decides where the data goes; the identity check of
        ``evaluation.json`` refuses any drift. After ``train --evaluate`` (or with ``pin``)
        every setting is pinned, so the command continues the evaluation that run started.
        Only the preview's command repeats ``--evaluators`` (with ``evaluators``): a run
        attaches every evaluator in it, and later commands continue them without the flag.
        """
        return evaluate_words(
            self.run_dir,
            self.settings,
            self.judge.spec,
            pin=not self.standalone if pin is None else pin,
            compare_spec=None if self.compare is None else self.compare.spec,
            evaluators=self.evaluators_spec if evaluators else None,
        )

    @property
    def evaluators_spec(self) -> Path | None:
        return None if self.evaluators is None else self.evaluators.pinned

    @property
    def evaluator_names(self) -> list[str]:
        return [] if self.evaluators is None else self.evaluators.names

    @property
    def default_out_dir(self) -> Path:
        return self.run_dir / DEFAULT_OUT_DIR_NAME

    def train_flags(self) -> list[str]:
        """The ``train --evaluate`` flags for this plan: ``--evaluate``, the judge, and the
        settings that differ from the defaults."""
        settings, defaults = self.settings, EvalSettings(out_dir=self.default_out_dir)
        words = ["--evaluate"]
        if settings.out_dir != defaults.out_dir:
            words += ["--eval-out-dir", display_path(settings.out_dir)]
        if settings.max_cases_per_trajectory != defaults.max_cases_per_trajectory:
            words += ["--eval-max-cases", str(settings.max_cases_per_trajectory)]
        words += ["--eval-judge-model", self.judge.spec]
        if settings.concurrency != defaults.concurrency:
            words += ["--eval-concurrency", str(settings.concurrency)]
        if settings.max_tokens != defaults.max_tokens:
            words += ["--eval-max-tokens", str(settings.max_tokens)]
        if settings.seed != defaults.seed:
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
            compare=None if self.compare is None else self.compare.config,
            replays=self.replays,
            evaluators=tuple(self.evaluator_names),
            headline=None if self.evaluators is None else self.evaluators.headline,
        )
        return ExperimentsPublisher(
            Pup(),
            settings,
            out_dir=self.out_dir,
            runtime=self.runtime,
            cancel=cancel,
            progress=info,
        )

    def recipients(self, *, trained: bool = True, compared: bool = True) -> list[str]:
        """Who receives the evaluation's data, one entry per kind of data, e.g.
        ``test split -> fireworks (generate), Anthropic api.anthropic.com (judge)``: the
        training provider only when base or tuned has a case left to generate (``trained``),
        and the compare model only when it has (``compared``)."""
        from pup_finetune.judges import recipient

        sent = [f"{self.provider} (generate)"] if trained else []
        if self.compare is not None and compared:
            sent.append(self.compare.recipient())
        sent.append(
            "the offline fake judge (judge; nothing is sent)"
            if self.judge.offline
            else f"{recipient(self.judge)} (judge)"
        )
        found = [f"test split -> {', '.join(sent)}"]
        if self.publishing.ready:
            found.append("test cases, generated answers and scores -> Datadog (experiments)")
        return found

    def exclusion_warnings(self) -> list[str]:
        from pup_finetune.evaluation.cases import exclusion_warning

        total = len(self.cases) + len(self.excluded)
        warning = exclusion_warning(self.excluded, cases=total, model=self.profile.alias)
        return [] if warning is None else [warning]

    def _refuse_another_compare(self, eval_dir: EvalDir) -> None:
        """Raise the conflict --confirm would raise for another compare model than the one
        ``eval_dir`` keeps: fail the preview now, never offer that --confirm."""
        from pup_finetune.evaluation.compare import compare_conflict, stored_compare

        if self.compare_identity is None:
            return
        conflict = compare_conflict(
            stored_compare(eval_dir.compare_path), self.compare_identity, out_dir=self.out_dir
        )
        if conflict is not None:
            raise conflict

    def _refuse_other_evaluators(self, eval_dir: EvalDir) -> None:
        """Raise the conflict --confirm would raise for evaluators that drop or change one
        ``eval_dir`` keeps."""
        from pup_finetune.evaluation.evaluators import evaluators_conflict, stored_evaluators

        if self.evaluators is None:
            return
        stored = stored_evaluators(eval_dir.evaluators_path)
        if conflict := evaluators_conflict(stored, self.evaluators, out_dir=self.out_dir):
            raise conflict

    def work(self, tuned_checkpoint: CheckpointRef | None) -> EvalWork:
        """What the evaluation would still do: everything, or with ``tuned_checkpoint`` (a
        standalone run) less what ``--out-dir`` already holds under the same settings. The
        preview counts it and execution opens the judge only when it has a judge call left.

        Raises the :class:`SettingsConflict` ``--confirm`` would raise for a directory that
        holds work under other settings, or another compare model.
        """
        from pup_finetune.evaluation.replay import EvalDir, remaining_work

        names = self.evaluator_names
        by_target: dict[str, Any] = {
            target: {
                "done": 0,
                "generate": len(cases),
                "judge": len(cases),
                **({"evaluators": dict.fromkeys(names, len(cases))} if names else {}),
            }
            for target, cases in self.replays.items()
        }
        calibration = self.calibration_calls()
        eval_dir = EvalDir(self.out_dir)
        wanted = None if tuned_checkpoint is None else finite_json(self.identity(tuned_checkpoint))
        stored = eval_dir.stored_identity() if wanted is not None else None
        # The run probes a compare model right before it writes compare.json, which it does
        # when there is none, or when check_identity drops the one there.
        probe = self.compare is not None and (
            not eval_dir.compare_path.exists() or (wanted is not None and eval_dir.replaces(wanted))
        )
        reusing = wanted is not None and wanted == stored
        if wanted is not None and stored is not None and not reusing and eval_dir.has_work():
            # --confirm would refuse (EvalDir.check_identity): fail now, never offer it.
            raise SettingsConflict(
                f"{self.out_dir} holds an evaluation with other settings (changed: "
                f"{', '.join(identity_changes(stored, wanted))}), so its results cannot be "
                "reused",
                hint=f"{no_reuse_hint(stored, wanted)}evaluate into a new directory: pass "
                f"--out-dir {_q(free_sibling(self.out_dir))}",
            )
        if reusing:
            self._refuse_another_compare(eval_dir)
            self._refuse_other_evaluators(eval_dir)
            by_target = remaining_work(eval_dir, self.replays, names)
            if (eval_dir.read_calibration() or {}).get("ok"):
                calibration = 0
        reused = {}
        base = self.base_reuse(eval_dir if reusing else None)
        if base is not None:
            reused["base"] = base
            by_target["base"]["generate"] -= len(base.generations)
            by_target["base"]["judge"] -= len(base.verdicts)
        examples = evaluator_calibration_calls(
            self.evaluators, eval_dir if reusing else None, skip=self.settings.skip_calibration
        )
        return EvalWork(
            by_target=by_target,
            calibration=calibration,
            probe=probe,
            reused=reused,
            evaluator_calibration=examples,
        )

    def preview(self, *, tuned: str, tuned_checkpoint: CheckpointRef | None) -> dict[str, Any]:
        """The calls the evaluation makes. With ``tuned_checkpoint`` (a standalone run), what
        ``--out-dir`` already holds under the same settings is left out of the counts."""
        from pup_finetune.datadog.experiments import experiment_name
        from pup_finetune.evaluation.cases import excluded_by_reason
        from pup_finetune.evaluation.compare import require_sent, skipped_warning
        from pup_finetune.evaluation.replay import CAVEAT, COMPARE, TARGETS, EvalDir

        settings = self.settings
        warnings = [*self.judge_warnings, *self.exclusion_warnings()]
        if (remote := self.profile.remote_code_warning()) is not None:
            warnings.append(remote)
        warnings += [] if self.reuse is None else self.reuse.warnings
        targets = self.targets
        replays = self.replays
        skips = Counter(self.compare_skips.values())
        if self.compare is not None:
            # --confirm would attach a model that answers nothing: fail now, never offer it.
            require_sent(skips, spec=self.compare.spec, cases=len(self.cases))
            if skipped := skipped_warning(skips, spec=self.compare.spec, cases=len(self.cases)):
                warnings.append(skipped)
        work = self.work(tuned_checkpoint)
        by_target = {target: dict(counts) for target, counts in work.by_target.items()}
        calibration = work.calibration
        from pup_finetune.judges import destination as judge_destination

        if work.probe:
            by_target[COMPARE]["generate"] += 1
        remaining = {
            key: sum(counts[key] for counts in by_target.values())
            for key in ("done", "generate", "judge")
        }
        scoring = [name for name in self.evaluator_names if work.evaluator_calls(name)]
        trained = sum(by_target[target]["generate"] for target in targets if target != COMPARE)
        compared = by_target.get(COMPARE, {}).get("generate", 0)
        trajectories = len({case.row_id for case in self.cases})
        judged_at = (
            "the offline fake judge (nothing is sent)"
            if self.judge.offline
            else judge_destination(self.judge)
        )
        generators = [self.provider] if trained else []
        if self.compare is not None and compared:
            generators.append(self.compare.destination())
        generate = f"{' and to '.join(generators)} to generate and to " if generators else ""
        destination = f"the test split's conversations are sent to {generate}{judged_at} to judge"
        publish = self.publishing.status()
        if publish["status"] == "skipped":
            warnings.append(f"the evaluation will not be published: {publish['reason']}")
        elif publish["status"] == "planned":
            publish["experiments"] = [experiment_name(t, self.run_id) for t in targets]
            publish["spans_per_experiment"] = {
                target: len({case.row_id for case in cases}) + len(cases)
                for target, cases in replays.items()
            }
            destination += (
                ", and the cases with the generated answers and their scores are published to "
                "Datadog as experiments on the pushed dataset"
            )
        from pup_finetune.judges import billing as judge_billing
        from pup_finetune.judges import require_key
        from pup_finetune.providers.base import SamplingUsage

        judge_calls = remaining["judge"] + calibration
        # --confirm would fail on a key nothing provides: fail now, never offer it. A
        # read-only preview offers none, so it only warns.
        soft = warnings if self.runtime.read_only else None
        if judge_calls:
            require_key(self.judge, warnings=soft)
        if self.compare is not None and compared:
            require_key(self.compare.config, what="compare model", warnings=soft)
        evaluated = None
        if (evaluators := self.evaluators) is not None:
            for judge in busy_judges(evaluators, work):
                require_key(judge.config, what=f"evaluator {judge.name}", warnings=soft)
            eval_dir = EvalDir(self.out_dir)
            evaluated = evaluators_preview(
                evaluators,
                work,
                self.cases,
                attached=_attached(eval_dir),
                reports=_reports(eval_dir),
                calibrate=not settings.skip_calibration,
            )
            warnings.extend(evaluated.warnings)
        judge_name = "the fake judge" if self.judge.provider == "fake" else self.judge.spec
        # Only what generates bills or is named: answers already stored, or taken from the
        # dataset's store, are reused, so attaching a compare model never mentions the
        # training provider.
        generating = tuple(target for target in TARGETS if by_target[target]["generate"])
        models = [f"{_GENERATING[generating]} on {self.provider}"] if generating else []
        base = work.reused.get("base")
        taken = {"generations": 0, "verdicts": 0} if base is None else base.counts()
        from_store = any(taken.values())
        reused = [
            target
            for target in TARGETS
            if not by_target[target]["generate"] and not (target == "base" and from_store)
        ]
        billing: list[BillingPart] = []
        if trained:
            billing.append(
                sampling_billing(
                    self.provider,
                    self.runtime,
                    SamplingUsage(calls=trained, max_tokens_per_call=settings.max_tokens),
                )
            )
        if self.compare is not None and compared:
            models.append(
                f"{self.compare.spec} (after one call that checks it answers)"
                if work.probe
                else self.compare.spec
            )
            billing.append(self.compare.billing(compared))
        elif self.compare is not None:
            reused.append("compare")
        billing.append(judge_billing(self.judge, judge_calls))
        if evaluated is not None:
            billing.extend(evaluated.billing)
        steps = [_store_step(taken)] if from_store else []
        if reused:
            steps.append(f"reuse the {joined(reused)} answers already generated")
        if models:
            steps.append(f"generate each with {' and with '.join(models)}")
        cases = f"{_cases(len(self.cases))} from {trajectories} test conversations"
        score = f", and score each with {joined(scoring)}" if scoring else ""
        if models or remaining["judge"]:
            each = "each new answer" if from_store else "each answer"
            action = (
                f"evaluate {cases}: {', '.join(steps)}, and judge {each} with {judge_name}{score}"
            )
        elif scoring:
            calibrate = f"calibrate {judge_name} first, then " if calibration else ""
            answers = joined([*reused] or ["stored"])
            action = (
                f"{calibrate}score {cases} with {joined(scoring)}, reusing the {answers} answers "
                "already generated and judged"
            )
        else:
            calibrate = f"calibrate {judge_name} first, then " if calibration else ""
            action = f"{calibrate}summarize {cases}: every answer is already generated and judged"
        if publish["status"] == "planned":
            action += f", then publish {_COUNTS[len(targets)]} experiments to Datadog"
        target_names: dict[str, str] = {"base": self.profile.base_model, "tuned": tuned}
        if self.compare is not None:
            target_names[COMPARE] = self.compare.spec
        compare_max = 0 if self.compare is None else self.compare.config.max_tokens
        return {
            "action": action,
            **billing_fields(billing),
            "run_id": self.run_id,
            "out_dir": display_path(self.out_dir),
            "rows": {"test": trajectories},
            "cases": len(self.cases),
            "excluded_cases": len(self.excluded),
            "excluded_by_reason": excluded_by_reason(self.excluded),
            "trajectories": trajectories,
            "targets": target_names,
            **(
                {
                    "compare": {
                        **self.compare.describe(),
                        "skipped": sum(skips.values()),
                        "skipped_by_reason": dict(sorted(skips.items())),
                    }
                }
                if self.compare is not None
                else {}
            ),
            "judge_model": self.judge.spec,
            "judge_source": self.judge_source,
            "judge_provider": self.judge.provider,
            "judge_endpoint": self.judge.base_url,
            **({} if evaluated is None else {"evaluators": evaluated.block}),
            "already_done": remaining["done"],
            "generation_calls": remaining["generate"],
            "generation_calls_by_target": {
                target: counts["generate"] for target, counts in by_target.items()
            },
            "judge_calls": remaining["judge"] + calibration,
            "calibration_calls": calibration,
            "max_generated_tokens": trained * settings.max_tokens + compared * compare_max,
            "reuse": self._reuse_block(work),
            "settings": {
                "judge_model": self.judge.spec,
                **({"compare_model": self.compare.spec} if self.compare is not None else {}),
                "max_cases": settings.max_cases_per_trajectory,
                "max_tokens": settings.max_tokens,
                "concurrency": settings.concurrency,
                "seed": settings.seed,
                "calibration": not settings.skip_calibration,
                "publish": settings.publish,
                "reuse": settings.reuse,
                **(
                    {}
                    if self.evaluators is None
                    else {
                        "evaluators": _spec_word(self.evaluators),
                        "headline": self.evaluators.headline,
                    }
                ),
            },
            "publish": publish,
            "data_destination": destination,
            "data_recipients": [
                *self.recipients(trained=bool(trained), compared=bool(compared)),
                *([] if evaluated is None else evaluated.recipients),
            ],
            "caveat": CAVEAT,
            **({"notes": evaluated.notes} if evaluated is not None and evaluated.notes else {}),
            "warnings": warnings,
        }

    def _reuse_block(self, work: EvalWork) -> dict[str, Any]:
        """The preview's ``reuse``: what each target takes from the dataset's store, from
        which runs, and the calls and samplers that saves."""
        if self.reuse is None:
            return {"status": "off"}
        by_target: dict[str, Any] = {}
        for target, taken in work.reused.items():
            counts = taken.counts()
            if any(counts.values()):
                by_target[target] = {
                    **counts,
                    "oldest_sampled_at": taken.oldest_sampled_at,
                    "from": taken.sources,
                }
        answers = sum(counts["generations"] for counts in by_target.values())
        left = work.by_target
        return {
            "status": "on",
            "store": display_path(self.reuse.store),
            "by_target": by_target,
            "saved": {
                "generation_calls": answers,
                "judge_calls": sum(counts["verdicts"] for counts in by_target.values()),
                "max_generated_tokens": answers * self.settings.max_tokens,
                "samplers": int(
                    bool(by_target.get("base", {}).get("generations"))
                    and not left["base"]["generate"]
                    and bool(left["tuned"]["generate"])
                ),
            },
        }


_COUNTS: Final = {2: "two", 3: "three"}
_GENERATING: Final = {
    ("base", "tuned"): "the base and the tuned model",
    ("base",): "the base model",
    ("tuned",): "the tuned model",
}


def _cases(count: int) -> str:
    return f"{count} case{'' if count == 1 else 's'}"


def _store_step(taken: Mapping[str, int]) -> str:
    """The preview's step for what base takes from the dataset's store."""
    answers, judged = taken["generations"], taken["verdicts"]
    if not answers:
        return (
            f"reuse {judged} judgment{'' if judged == 1 else 's'} of the base model's answers "
            "made earlier on this dataset"
        )
    noted = f" ({judged} already judged)" if judged else ""
    plural = "" if answers == 1 else "s"
    return f"reuse the base model's {answers} answer{plural}{noted} made earlier on this dataset"


def _spec_word(evaluators: EvaluatorSet) -> str | None:
    return None if evaluators.spec is None else display_path(evaluators.spec)


def _attached(eval_dir: EvalDir) -> set[str]:
    from pup_finetune.evaluation.evaluators import stored_evaluators

    stored = stored_evaluators(eval_dir.evaluators_path) or {}
    return {str(item.get("name")) for item in stored.get("evaluators") or []}


def busy_judges(evaluators: EvaluatorSet, work: EvalWork) -> list[JudgeEvaluator]:
    """The user's judges with a call left, to score or to calibrate."""
    return [
        judge
        for judge in evaluators.judges
        if work.evaluator_calls(judge.name) or work.evaluator_calibration.get(judge.name)
    ]


def _calibrated(report: object, judge: JudgeEvaluator) -> bool:
    return (
        isinstance(report, Mapping)
        and bool(report.get("ok"))
        and canonical_json(report.get("identity")) == canonical_json(finite_json(judge.identity()))
    )


def _reports(eval_dir: EvalDir | None) -> Mapping[str, Any]:
    if eval_dir is None:
        return {}
    found = (eval_dir.read_calibration() or {}).get("evaluators")
    return found if isinstance(found, Mapping) else {}


def evaluator_calibration_calls(
    evaluators: EvaluatorSet | None, eval_dir: EvalDir | None, *, skip: bool
) -> dict[str, int]:
    """The example calls each of the user's judges still makes to calibrate, by name: none
    with ``skip``, without examples, or once one passed for it in ``eval_dir``."""
    if evaluators is None or skip:
        return {}
    reports = _reports(eval_dir)
    return {
        judge.name: len(judge.examples)
        for judge in evaluators.judges
        if judge.examples and not _calibrated(reports.get(judge.name), judge)
    }


@dataclass(frozen=True, slots=True)
class EvaluatorsPreview:
    """What the user's evaluators add to a preview."""

    block: dict[str, Any]
    billing: list[BillingPart]
    recipients: list[str]
    notes: list[str]
    warnings: list[str]


def uncalibrated(
    evaluators: EvaluatorSet, reports: Mapping[str, Any], *, calibrate: bool
) -> list[str]:
    """A warning for each of the user's judges without examples nor a calibration that
    passed (``reports``, by name); none when calibration is skipped."""
    return [
        f"evaluator {judge.name} is not calibrated: add examples (answers you graded) to its spec"
        for judge in evaluators.judges
        if calibrate and not judge.examples and not _calibrated(reports.get(judge.name), judge)
    ]


def evaluators_preview(
    evaluators: EvaluatorSet,
    work: EvalWork,
    cases: Sequence[ReplayCase],
    *,
    attached: set[str],
    reports: Mapping[str, Any],
    calibrate: bool,
) -> EvaluatorsPreview:
    """The preview's ``evaluators`` block (``attached`` names those frozen already) and
    what they bill, receive and warn. Every check with cases left runs on each case's
    reference first (free, local): one that raises or answers outside its contract fails
    the preview."""
    from pup_finetune.evaluation.evaluators import CheckEvaluator
    from pup_finetune.judges import billing, recipient

    items: list[dict[str, Any]] = []
    calls: dict[str, tuple[JudgeConfig, int]] = {}
    recipients: list[str] = []
    files: list[str] = []
    for evaluator in evaluators.evaluators:
        left = work.evaluator_calls(evaluator.name)
        if isinstance(evaluator, CheckEvaluator):
            rate = None
            if left:
                rate = _self_test(evaluator, cases)
                if (shown := display_path(evaluator.file)) not in files:
                    files.append(shown)
            items.append(
                {
                    "name": evaluator.name,
                    "type": "check",
                    "file": display_path(evaluator.file),
                    "function": evaluator.function,
                    "file_sha256": evaluator.file_sha256[:8],
                    "attached": evaluator.name in attached,
                    "reference_pass_rate": rate,
                }
            )
            continue
        examples = work.evaluator_calibration.get(evaluator.name, 0)
        config = evaluator.config
        if left + examples:
            before = calls.get(config.spec, (config, 0))[1]
            calls[config.spec] = (config, before + left + examples)
            recipients.append(
                f"test cases and answers -> the offline fake judge (evaluator {evaluator.name}; "
                "nothing is sent)"
                if config.offline
                else f"test cases and answers -> {recipient(config)} (evaluator {evaluator.name})"
            )
        items.append(
            {
                "name": evaluator.name,
                "type": "judge",
                "model": config.spec,
                "provider": config.provider,
                "rubric_sha256": evaluator.rubric_sha256[:8],
                "attached": evaluator.name in attached,
                "calls": left,
                "calibration": "examples" if evaluator.examples else "none",
                "calibration_calls": examples,
            }
        )
    notes = [
        f"{name} is Python you wrote: its checks run in this process with your privileges (this "
        "preview ran them on every reference); nothing is sent"
        for name in files
    ]
    return EvaluatorsPreview(
        block={
            "spec": _spec_word(evaluators),
            "source": evaluators.source,
            "headline": evaluators.headline,
            "items": items,
        },
        billing=[billing(config, count, what="evaluator") for config, count in calls.values()],
        recipients=recipients,
        notes=notes,
        warnings=uncalibrated(evaluators, reports, calibrate=calibrate),
    )


def _self_test(check: CheckEvaluator, cases: Sequence[ReplayCase]) -> float | None:
    """``check``'s pass rate on the references of ``cases`` (``None`` when it never
    applies); a :class:`UsageError` naming the case when it fails on one."""
    from pup_finetune.user_checks import CheckError

    passed = applied = 0
    for case in cases:
        try:
            entry = check.loaded().check(case, case.reference)
        except CheckError as exc:
            raise UsageError(
                f"the check {check.name} failed on the reference of case {case.id}: {exc.message}",
                hint="fix it to return True, False, None or (bool, reason) for every answer; "
                "this preview ran it, free, before anything is paid",
            ) from None
        if entry["status"] == "ok":
            applied += 1
            passed += bool(entry["passed"])
    return passed / applied if applied else None


JUDGE_GIVEN: Final = "given"
JUDGE_FROZEN: Final = "evaluation directory"
JUDGE_FAKE_RUN: Final = "fake run"
"""Where a judge came from (``judge_source``) when it is not a default by credential: the
flag, the evaluation the output directory holds, or the fake judge of a fake run."""


def resolve_judge(
    judge_model: str | None,
    *,
    provider: str,
    runtime: Runtime,
    stored: str | None = None,
    warnings: list[str] | None = None,
) -> tuple[JudgeConfig, str]:
    """The judge for ``--judge-model`` and where it came from (``judge_source``): the judge
    the spec names (:func:`pup_finetune.judges.parse_judge`); without one, ``stored`` (the
    judge an evaluation directory froze, so a rerun keeps it), for a ``fake`` run the fake
    judge, else the default by the credentials that resolve
    (:data:`~pup_finetune.judge_models.EVALUATE_DEFAULTS`), whose warnings go to ``warnings``.

    :data:`~pup_finetune.evaluation.fake_judge.FAKE_JUDGE_MODEL` is the offline judge, for
    development mode only. Raises :class:`UsageError` for an empty spec or an empty model
    after a provider prefix, and when a package the judge needs is not installed.
    """
    from pup_finetune.evaluation.fake_judge import FAKE_JUDGE_MODEL, fake_judge_config
    from pup_finetune.judge_models import EVALUATE_DEFAULTS
    from pup_finetune.judges import check_available, default_judges, parse_judge

    source = JUDGE_GIVEN
    if judge_model is None and stored is not None:
        judge_model, source = stored, JUDGE_FROZEN
    if judge_model is None and provider == "fake":
        judge_model, source = FAKE_JUDGE_MODEL, JUDGE_FAKE_RUN
    if judge_model == FAKE_JUDGE_MODEL:
        if not runtime.dev_mode:
            raise UsageError(
                "the fake judge is for development only",
                hint="set PUP_FINETUNE_DEV=1, or pass another --judge-model",
            )
        return fake_judge_config(), source
    if judge_model is None:
        (judge_model,), source = default_judges(EVALUATE_DEFAULTS, warnings=warnings)
    config = parse_judge(judge_model)
    check_available(config)
    return config, source


def _stored(eval_dir: EvalDir) -> str | None:
    """The judge ``eval_dir`` froze: its ``evaluation.json``'s, once it holds work (before,
    another identity replaces it; see :meth:`~pup_finetune.evaluation.replay.EvalDir.replaces`)."""
    return stored_judge(eval_dir.stored_identity()) if eval_dir.has_work() else None


def stored_judge(identity: Mapping[str, Any] | None) -> str | None:
    """The judge spec an ``evaluation.json`` (``identity``) froze, or ``None``."""
    from pup_finetune.inference import JudgeConfig

    judge = identity.get("judge") if identity is not None else None
    if not isinstance(judge, Mapping):
        return None
    try:
        return JudgeConfig.model_validate(judge).spec
    except ValueError:
        return None


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
    no_publish: str = NO_PUBLISH,
) -> EvalPlan:
    """The evaluation of the run ``run_id`` on ``dataset``'s test split.

    ``no_publish`` is the running command's flag that skips publishing (see
    :func:`plan_publishing`). Unless ``settings.reuse`` is off, the dataset's store of base
    results is read here, once (:mod:`~pup_finetune.evaluation.reuse`).

    Raises :class:`UsageError` when the test split has no case, or the judge is the model
    being trained or the run's tuned model: the model id the run promotes to
    (:func:`~pup_finetune.runs.model_id_for_run`) or, once known, the ``promoted`` model.
    """
    from pup_finetune.evaluation.cases import cases_from_dataset, exclusions
    from pup_finetune.evaluation.compare import resolve_compare, stored_compare
    from pup_finetune.evaluation.evaluators import resolve_evaluators, stored_evaluators
    from pup_finetune.evaluation.judge import check_judge_differs
    from pup_finetune.evaluation.replay import EvalDir, remaining_work
    from pup_finetune.evaluation.reuse import ReuseIndex, Wanted, generation_key, key_inputs

    judge_warnings: list[str] = []
    judge_config, judge_source = resolve_judge(
        settings.judge_model,
        provider=provider,
        runtime=runtime,
        stored=_stored(EvalDir(settings.out_dir)),
        warnings=judge_warnings,
    )
    compare = resolve_compare(
        settings.compare_model,
        stored_compare(EvalDir(settings.out_dir).compare_path),
        runtime=runtime,
    )
    tuned = [model_id_for_run(run_id)]
    if promoted is not None:
        tuned += [promoted.model_id, promoted.uri]
    check_judge_differs(
        judge_config, profile.base_model, tuned, compare=None if compare is None else compare.config
    )
    cases = cases_from_dataset(
        dataset,
        split="test",
        max_cases_per_trajectory=settings.max_cases_per_trajectory,
        seed=settings.seed,
    )
    if not cases:
        rows = len(dataset.rows_for_split("test"))
        raise UsageError(
            f"nothing to evaluate: the test split of {display_path(dataset.path)} has {rows} kept rows and "
            "no assistant turn to replay",
            hint="import more rows so the test split holds some conversations",
        )
    excluded = exclusions(cases, get_provider(provider, runtime).renderer(profile))
    if len(excluded) == len(cases):
        first = next(iter(excluded.values()))
        raise UsageError(
            f"nothing to evaluate: {profile.alias} cannot replay any of the {len(cases)} test "
            f"cases ({first.code}: {first.detail})",
            hint="import rows whose test conversations the model can replay",
        )
    replayed = [case for case in cases if case.id not in excluded]
    inputs = key_inputs(provider, profile, rendering, settings.max_tokens)
    keys = {case.id: generation_key(inputs, case) for case in replayed}
    reuse = None
    if settings.reuse:
        own = EvalDir(settings.out_dir).read_generations()
        reuse = ReuseIndex.load(dataset.path, judge_config, [Wanted("base", replayed, keys, own)])
    plan = EvalPlan(
        run_id=run_id,
        run_dir=run_dir,
        provider=provider,
        profile=profile,
        rendering=dict(rendering),
        cases=replayed,
        excluded=excluded,
        judge=judge_config,
        judge_source=judge_source,
        judge_warnings=tuple(judge_warnings),
        settings=settings,
        publishing=plan_publishing(dataset, enabled=settings.publish, no_publish=no_publish),
        runtime=runtime,
        compare=compare,
        base_keys=keys,
        reuse=reuse,
    )
    eval_dir = EvalDir(settings.out_dir)

    def pending(name: str) -> bool:
        left = remaining_work(eval_dir, plan.replays, [name])
        return any(counts["evaluators"][name] for counts in left.values())

    evaluators = resolve_evaluators(
        settings.evaluators,
        stored_evaluators(eval_dir.evaluators_path),
        judge=judge_config,
        runtime=runtime,
        pending=pending,
    )
    for judge in [] if evaluators is None else evaluators.judges:
        try:
            check_judge_differs(
                judge.config,
                profile.base_model,
                tuned,
                compare=None if compare is None else compare.config,
            )
        except UsageError as exc:
            raise UsageError(
                f"evaluator {judge.name}: {exc.message}",
                hint="name another model for it in the spec: a model should not grade its own "
                "outputs",
            ) from None
    return plan if evaluators is None else dataclasses.replace(plan, evaluators=evaluators)


# --- the judge ----------------------------------------------------------------------------


class _NoJudgeCall:
    """The judge of a run with no judge call left (:attr:`EvalWork.judge_calls`)."""

    def complete(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        raise AssertionError("the judge was called with no judge call left")


@contextmanager
def open_judge_client(config: JudgeConfig, *, calls: int) -> Iterator[ChatCompleter]:
    """The client for ``config``'s judge, opened now, before anything is generated, with
    the key of its credential (:func:`pup_finetune.judges.open_client`), so a malformed key,
    a missing package or an unreadable keychain fails before anything is paid; the offline
    fake judge; or, with no ``calls`` left, none, so a rerun with nothing left to judge
    needs no key. Closed on exit."""
    from pup_finetune.evaluation.fake_judge import FakeJudge
    from pup_finetune.judges import open_client

    if config.provider == "fake":
        yield FakeJudge()
        return
    if not calls:
        yield _NoJudgeCall()
        return
    with open_client(config) as client:
        yield client


@contextmanager
def open_evaluator_clients(
    judges: Sequence[JudgeEvaluator],
) -> Iterator[dict[str, ChatCompleter]]:
    """One client per provider of ``judges`` (the user's judges with a call left, to score
    or to calibrate), by provider, opened now like the judge's (the fake judge offline);
    none without any, so a run that only adds checks needs no key. Closed on exit."""
    from pup_finetune.evaluation.fake_judge import FakeJudge
    from pup_finetune.judges import open_clients

    busy = [judge.config for judge in judges]
    with open_clients([config for config in busy if not config.offline]) as clients:
        found: dict[str, ChatCompleter] = {str(name): client for name, client in clients.items()}
        if any(config.provider == "fake" for config in busy):
            found["fake"] = FakeJudge()
        yield found


def calibrate_evaluators(
    evaluators: EvaluatorSet | None,
    eval_dir: EvalDir,
    clients: Mapping[str, ChatCompleter],
    *,
    skip: bool,
) -> dict[str, Any] | None:
    """Calibrate each of the user's judges on its examples, unless one passed for it in
    ``eval_dir`` (its lock held), keeping each report in its ``calibration.json`` under
    ``evaluators``; the first failed report, with ``evaluator``, its judge's name."""
    from pup_finetune.evaluation.judge import calibrate_examples

    due = evaluator_calibration_calls(evaluators, eval_dir, skip=skip)
    for judge in [] if evaluators is None else evaluators.judges:
        if not due.get(judge.name):
            continue
        info(f"calibrating the evaluator {judge.name} on {len(judge.examples)} examples")
        found = calibrate_examples(
            judge.examples, clients[judge.config.provider], judge.config, judge.rubric
        )
        report = {**found.to_json(), "identity": judge.identity()}
        info(calibration_line(report))
        stored = eval_dir.read_calibration() or {}
        reports = {**_reports(eval_dir), judge.name: report}
        eval_dir.write_calibration({**stored, "evaluators": reports})
        if not found.ok:
            return {**report, "evaluator": judge.name}
    return None


def run_calibration(plan: EvalPlan, client: ChatCompleter) -> dict[str, Any] | None:
    """The calibration report of ``plan``'s judge, or ``None`` with ``--skip-calibration``."""
    from pup_finetune.evaluation.judge import calibrate

    if plan.settings.skip_calibration:
        return None
    info(f"calibrating the judge {plan.judge.spec} on {plan.calibration_calls() // 3} cases")
    report = calibrate(plan.cases, client, plan.judge, n=CALIBRATION_CASES).to_json()
    info(calibration_line(report))
    return {**report, "judge": plan.judge.identity()}


CALIBRATION_FAILURES_SHOWN: Final = 3
"""How many failed checks the calibration line names."""


def calibration_line(report: Mapping[str, Any]) -> str:
    """The progress line for a calibration report: ``calibration passed: 12/12 checks``, or
    the checks it failed."""
    checks = [check for check in report.get("checks") or [] if isinstance(check, Mapping)]
    passed = sum(1 for check in checks if check.get("ok"))
    if report.get("ok"):
        return f"calibration passed: {passed}/{len(checks)} checks"
    failed = []
    for check in checks:
        if check.get("ok"):
            continue
        judgment = check.get("judgment") or {}
        if judgment.get("error"):
            got = f"no verdict ({judgment['error']})"
        else:
            got = "judged pass" if judgment.get("passed") else "judged fail"
        what = check.get("kind")
        if check.get("corruption"):
            what = f"{what} ({check['corruption']})"
        failed.append(f"{check.get('case_id')} {what}: {got}")
    shown = "; ".join(failed[:CALIBRATION_FAILURES_SHOWN])
    if len(failed) > CALIBRATION_FAILURES_SHOWN:
        shown += f"; +{len(failed) - CALIBRATION_FAILURES_SHOWN} more"
    return f"calibration failed: {passed}/{len(checks)} checks passed; {shown}"


def calibration_failed(plan: EvalPlan, report: Mapping[str, Any]) -> dict[str, Any]:
    """What a command emits (exit 1, see :func:`evaluation_exit_code`) when calibration
    failed: nothing was generated. The hint spells the flags of the command that ran it
    (``evaluate``, or ``train --evaluate``, whose calibration runs before training)."""
    judge_flag, skip_flag = (
        ("--judge-model", "--skip-calibration")
        if plan.standalone
        else ("--eval-judge-model", "--eval-skip-calibration")
    )
    if (name := report.get("evaluator")) is not None:
        return {
            "status": "calibration_failed",
            "run_id": plan.run_id,
            "out_dir": display_path(plan.out_dir),
            "judge_model": plan.judge.spec,
            "evaluator": name,
            "calibration": dict(report),
            "hint": f"the evaluator {name} disagreed with answers you graded, so its verdicts "
            f"cannot be trusted; {evaluator_fix(name, [plan.out_dir], plan.out_dir)} (or rerun "
            f"with {skip_flag} to evaluate anyway)",
        }
    return {
        "status": "calibration_failed",
        "run_id": plan.run_id,
        "out_dir": display_path(plan.out_dir),
        "judge_model": plan.judge.spec,
        "calibration": dict(report),
        "hint": "the judge passed a corrupted action or failed a recorded one, so its "
        f"verdicts cannot be trusted; pick another {judge_flag} (or rerun with "
        f"{skip_flag} to evaluate anyway)",
    }


def evaluator_fix(name: str, dirs: Sequence[Path], out_dir: Path) -> str:
    """How to fix the evaluator ``name`` that failed calibration: its examples, and its
    rubric or model only in a free ``--out-dir`` once one of ``dirs`` has it attached."""
    from pup_finetune.evaluation.replay import EvalDir

    if any(name in _attached(EvalDir(directory)) for directory in dirs):
        return (
            f"fix its examples, or pass --out-dir {quoted(free_sibling(out_dir))} to change its "
            "rubric or model, frozen since it was attached"
        )
    return "fix its rubric or examples, or pick another model for it"


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
    probe: Callable[[], None] | None = None,
    evaluator_clients: Mapping[str, ChatCompleter] | None = None,
) -> dict[str, Any] | None:
    """Create ``--out-dir``, check its identity, attach the compare target, settle the
    calibration and attach the user's evaluators once their own calibration passed; the
    report (a failed one of an evaluator's names it as ``evaluator``).

    ``probe`` checks that the compare model answers right before it is attached, so a model
    that cannot fails before ``compare.json`` freezes it. A calibration that passed is stored
    and reused by later runs in the directory. Unless one is stored or given in
    ``calibration``, the judge is calibrated now (see :func:`run_calibration`). ``None`` with
    ``--skip-calibration``. Unless calibration failed, the base answers and verdicts the plan
    takes from the dataset's store are copied in last.
    """
    from pup_finetune.evaluation.compare import attach_compare
    from pup_finetune.evaluation.evaluators import attach_evaluators
    from pup_finetune.evaluation.replay import EvalDir

    eval_dir = EvalDir(plan.out_dir)
    eval_dir.create()
    with eval_dir.lock():
        eval_dir.check_identity(plan.identity(tuned_checkpoint))
        attach_compare(eval_dir.compare_path, plan.compare_identity, probe=probe)
        plan._refuse_other_evaluators(eval_dir)
        report = _settle_calibration(plan, eval_dir, client, calibration)
        # Imported work freezes the identity, so a failed calibration's retry with another
        # judge must find none.
        if report is not None and not report.get("ok"):
            return report
        failed = calibrate_evaluators(
            plan.evaluators,
            eval_dir,
            evaluator_clients or {},
            skip=plan.settings.skip_calibration,
        )
        if failed is not None:
            return failed
        attach_evaluators(eval_dir.evaluators_path, plan.evaluators)
        _import_base(plan, eval_dir)
        return report


def _settle_calibration(
    plan: EvalPlan,
    eval_dir: EvalDir,
    client: ChatCompleter,
    calibration: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """The calibration :func:`prepare_eval_dir` settles, with ``eval_dir``'s lock held."""
    if plan.settings.skip_calibration:
        return None
    stored = eval_dir.read_calibration()
    if stored is not None and stored.get("ok"):
        return stored
    report = dict(calibration) if calibration is not None else run_calibration(plan, client)
    if report is not None:
        kept = (stored or {}).get("evaluators")
        eval_dir.write_calibration({**report, **({"evaluators": kept} if kept else {})})
    return report


def _import_base(plan: EvalPlan, eval_dir: EvalDir) -> None:
    """Stamp ``eval_dir`` (its lock held) with what base answers are sampled under, then copy
    in what base takes from the dataset's store."""
    from pup_finetune.evaluation.reuse import import_target, stamp

    stamp(eval_dir, plan.base_inputs)
    reuse = plan.base_reuse(eval_dir)
    if reuse is None or plan.reuse is None or not any(reuse.counts().values()):
        return
    copied = import_target(eval_dir, reuse, plan.replays["base"])
    info(
        f"reusing {copied['generations']} base answers ({copied['verdicts']} judged) from "
        f"{display_path(plan.reuse.store)}"
    )


def share_base(plan: EvalPlan) -> str | None:
    """Add the base answers and verdicts the evaluation made to its dataset's store; the
    warning when they could not be (see :func:`~pup_finetune.evaluation.reuse.export_target`)."""
    from pup_finetune.evaluation.replay import EvalDir
    from pup_finetune.evaluation.reuse import export_target

    return export_target(
        plan.publishing.dataset_dir,
        EvalDir(plan.out_dir),
        "base",
        plan.replays["base"],
        plan.base_inputs,
        plan.judge,
        {"command": "evaluate", "run_id": plan.run_id},
        f"{shlex.join(plan.evaluate_command())} --confirm",
        legacy=True,
    )


# --- the replay ---------------------------------------------------------------------------


def cancel_on_signals(cancel: threading.Event) -> contextlib.AbstractContextManager[None]:
    """Make SIGTERM and the first Ctrl-C set ``cancel``; a second Ctrl-C stops at once.

    See :func:`pup_finetune.runs.cancel_on_signals`; main thread only.
    """
    return _cancel_on_signals(cancel, _CANCEL_NOTICE)


def replay(
    plan: EvalPlan,
    samplers: Mapping[str, Sampler],
    client: ChatCompleter,
    tuned_checkpoint: CheckpointRef,
    *,
    cancel: threading.Event,
    evaluator_clients: Mapping[str, ChatCompleter] | None = None,
) -> ReplaySummary:
    """:func:`~pup_finetune.evaluation.replay.run_replay` of ``plan`` with ``client`` judging
    and the user's judges on ``evaluator_clients``, published by :meth:`EvalPlan.publisher`
    once complete."""
    from pup_finetune.evaluation.judge import judge
    from pup_finetune.evaluation.replay import run_replay

    config = plan.judge

    def judge_fn(case: ReplayCase, generation: Generation) -> Judgment:
        return judge(
            case,
            generation.message,
            client,
            config,
            format_valid=generation.format_valid,
            cut_off=generation.stop_reason == "length",
        )

    return run_replay(
        plan.cases,
        samplers,
        judge_fn,
        out_dir=plan.out_dir,
        identity=plan.identity(tuned_checkpoint),
        concurrency=plan.settings.concurrency,
        cancel=cancel,
        publisher=plan.publisher(tuned_checkpoint, cancel=cancel),
        on_progress=EvaluationProgress(),
        replays=plan.replays,
        compare=plan.compare_identity,
        evaluators=plan.evaluators,
        scorers=None
        if plan.evaluators is None
        else plan.evaluators.scorers(evaluator_clients or {}),
    )


_SENT: Final = frozenset({"published", "failed", "cancelled"})
"""Publication statuses after which some spans may have been sent."""


def _publication(plan: EvalPlan, outcome: ReplaySummary) -> dict[str, Any]:
    if outcome.publication is not None:
        return dict(outcome.publication)
    status = plan.publishing.status()
    if status["status"] == "planned" and not outcome.complete:
        published = (
            "rows finished before the last live publish are published; the rest are"
            if outcome.link
            else "it is"
        )
        return {
            "status": "pending",
            "reason": f"{published} published once every case has a result; run "
            "next_command to finish it",
            "comparison_url": outcome.link,
        }
    return status


def replay_result(
    plan: EvalPlan,
    outcome: ReplaySummary,
    *,
    calibration: Mapping[str, Any] | None,
    tuned_sampler: CheckpointRef | None,
    shared: str | None = None,
) -> dict[str, Any]:
    """What a finished (or cancelled) evaluation emits; ``shared`` is :func:`share_base`'s
    warning.

    ``status`` is ``complete``, ``complete_with_errors`` (every pair has a result, but some
    are errors, which a rerun of ``next_command`` retries) or ``partial`` (cancelled).
    ``publication`` says how the evaluation was published (``published``, ``failed`` or
    ``cancelled``, which a rerun of ``next_command`` continues, ``pending``, ``skipped`` or
    ``disabled``). Spans already published keep the results they were sent with, so after
    a retry of the errors the experiments still show them as errors.
    """
    from pup_finetune.evaluation.compare import skipped_warning
    from pup_finetune.evaluation.replay import COMPARE, EvalDir, truncation_warning

    summary = outcome.summary
    errors = sum(int(count) for count in summary["errors"].values())
    scoring = sum(
        int(target["evaluators"][name]["errors"])
        for target in summary["targets"].values()
        for name in plan.evaluator_names
    )
    status = "partial"
    if outcome.complete:
        status = "complete_with_errors" if errors or scoring else "complete"
    warnings = plan.exclusion_warnings()
    if shared is not None:
        warnings.append(shared)
    hint = None
    publication = _publication(plan, outcome)
    unrenderable = sum(int(target["errors_unrenderable"]) for target in summary["targets"].values())
    failed = (
        {}
        if plan.evaluators is None
        else evaluator_errors(
            EvalDir(plan.out_dir).read_results(), plan.replays, plan.evaluator_names
        )
    )
    checks = {check.name for check in plan.evaluators.checks} if plan.evaluators else set()
    rescored = errors > unrenderable or any(failed[name]["errors"] for name in set(failed) - checks)
    if missing := scoring - unrenderable * len(plan.evaluator_names):
        warnings.append(scoring_warning(missing, retried=rescored))
    if plan.evaluators is not None:
        reports = _reports(EvalDir(plan.out_dir))
        calibrate = not plan.settings.skip_calibration
        warnings += uncalibrated(plan.evaluators, reports, calibrate=calibrate)
    if errors:
        hint = (
            f"{errors} case(s) ended in an error, left out of every rate (see the file "
            "files.results names)"
        )
        if errors > unrenderable:
            hint += "; run next_command to retry them"
        too_long = {
            name: target.get("errors_too_long_by_stage") or {}
            for name, target in summary["targets"].items()
        }
        judged = sum(int(stages.get("judge", 0)) for stages in too_long.values())
        compared = int(too_long.get(COMPARE, {}).get("generation", 0))
        if judged:
            hint += (
                f". {judged} of them are longer than the judge's context window, which a "
                "retry cannot fix: evaluate into a new --out-dir with a --judge-model that has "
                "a larger context window"
            )
        if compared:
            hint += (
                f". {compared} of them are longer than the compare model's context window, "
                "which a retry cannot fix: compare with another --compare-model in a new "
                "--out-dir"
            )
        if unrenderable:
            hint += (
                f". {unrenderable} of them have a prompt the model's chat format cannot "
                "continue, which a retry cannot fix, so next_command never retries them"
            )
        if publication["status"] in _SENT:
            hint += (
                ". Spans already published keep the results they were sent with, these errors "
                "included: a retry updates the local results only, so evaluate into a new "
                "--out-dir to publish experiments with every result"
            )
    if broken := check_hint(failed, checks, plan.out_dir):
        hint = broken if hint is None else f"{hint}. {broken}"
    if plan.compare is not None:
        skips = Counter(plan.compare_skips.values())
        skipped = skipped_warning(skips, spec=plan.compare.spec, cases=len(plan.cases))
        if skipped is not None:
            warnings.append(skipped)
    for name, target in summary["targets"].items():
        truncated = int(target.get("truncated") or 0)
        if truncated and plan.compare is not None and name == COMPARE:
            warnings.append(
                f"{truncated} of the compare model's generations reached its limit of "
                f"{plan.compare.config.max_tokens} tokens, thinking included, and were cut "
                "off; they count as failures"
            )
        elif truncated:
            warnings.append(
                truncation_warning(
                    truncated, f"the {name} model's", plan.settings.max_tokens, "evaluate"
                )
            )
    no_verdict = sum(int(target["judge_no_verdict"]) for target in summary["targets"].values())
    if no_verdict:
        warnings.append(
            f"{no_verdict} judge answer(s) held no verdict (unparseable, cut off at the "
            "judge's max_tokens, or refused by the judge's provider) and count as failures; "
            "see judge_no_verdict per target in the file files.summary names"
        )
    warnings.extend(str(warning) for warning in publication.get("warnings") or [])
    if publication["status"] == "skipped":
        # Skipping publishing is moot once the evaluation ran: only the push is offered.
        reason = str(publication["reason"])
        for flag in (NO_PUBLISH, EVAL_NO_PUBLISH):
            reason = reason.removesuffix(f", or pass {flag}")
        publication = {**publication, "reason": reason}
        warnings.append(f"the evaluation was not published: {reason}")
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
    retry = status == "partial" or rescored or publication["status"] in {"failed", "cancelled"}
    return finite_json(
        {
            "status": status,
            "run_id": plan.run_id,
            "out_dir": display_path(outcome.out_dir),
            "judge_model": plan.judge.spec,
            "compare_model": None if plan.compare is None else plan.compare.spec,
            "summary": summary,
            "pending": outcome.pending,
            "cancelled": outcome.cancelled,
            "errors": errors,
            **({} if plan.evaluators is None else {"evaluator_errors": failed}),
            "publication": publication,
            "warnings": warnings,
            "hint": hint,
            "calibration": None
            if calibration is None
            else {"ok": calibration.get("ok"), "cases": calibration.get("cases")},
            "tuned_checkpoint": None
            if tuned_sampler is None
            else dataclasses.asdict(tuned_sampler),
            **_replay_next(
                plan,
                finished=_finished(status, summary),
                publication=publication,
                retry=retry,
            ),
        }
    )


def scoring_warning(count: int, *, retried: bool = True) -> str:
    """The warning for ``count`` evaluator entries an error left without a score; one that
    ``next_command`` does not retry points to the hint."""
    return (
        f"{count} evaluator score(s) are missing after an error (an evaluator's judge outage, "
        "a check that raised, or the pair's own error), left out of their rates; "
        + ("run next_command to retry them" if retried else "the hint says how to fix them")
    )


def evaluator_errors(
    results: Mapping[PairKey, Mapping[str, Any]],
    replays: Mapping[str, Sequence[ReplayCase]],
    names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Each of the evaluators ``names``' entries in ``results`` that ended in an error over
    the pairs of ``replays``, with the first error (``{"stage", "type", "message"}``)."""
    found: dict[str, dict[str, Any]] = {name: {"errors": 0, "first_error": None} for name in names}
    for target, cases in replays.items():
        for case in cases:
            entries = (results.get((target, case.id)) or {}).get("evaluators")
            for name, counted in found.items():
                entry = entries.get(name) if isinstance(entries, Mapping) else None
                if isinstance(entry, Mapping) and entry.get("status") == "error":
                    counted["errors"] += 1
                    counted["first_error"] = counted["first_error"] or entry.get("error")
    return found


def check_hint(found: Mapping[str, Mapping[str, Any]], checks: set[str], out_dir: Path) -> str:
    """The hint for the ``checks`` that raised in ``found``, which a retry cannot fix since
    each runs its frozen file again; empty when none raised."""
    raised = [
        f"the check {name} raised on {entry['errors']} answer(s): "
        + str((entry["first_error"] or {}).get("message")).removeprefix(f"check {name}: ")
        for name, entry in found.items()
        if name in checks and entry["errors"]
    ]
    if not raised:
        return ""
    return (
        f"{'; '.join(raised)}; a retry runs the same frozen file, so fix the check in a new "
        f"file under a new evaluator name, or pass --out-dir {quoted(free_sibling(out_dir))}"
    )


def _replay_next(
    plan: EvalPlan, *, finished: bool, publication: Mapping[str, Any], retry: bool
) -> dict[str, Any]:
    """``next_command`` and ``next_steps`` after an evaluation.

    One that stopped, has errors or whose publishing failed is continued by the evaluate
    command (a free preview first). A complete one is deployed (a free preview), unless the
    run has a deployment already. One that could not be published for want of a push has no
    next command: its steps push the dataset, then rerun the evaluation (a complete
    evaluation's rerun reuses every result, so it makes no new model calls), or deploy it
    unpublished.
    """
    rerun = plan.evaluate_command()
    pushing = (
        _push_first(plan.publishing, shlex.join(rerun), plan.run_dir, complete=finished)
        if publication.get("status") == "skipped"
        else []
    )
    steps = list(pushing)
    if finished and plan.compare is None:
        compared = evaluate_words(
            plan.run_dir,
            plan.settings,
            plan.judge.spec,
            pin=not plan.standalone,
            compare_spec="<MODEL>",
        )
        steps.append(
            step(
                template(*compared[len(PROGRAM) :]),
                "compare the tuned model with another model, such as the one your app runs "
                "today, on the same cases and judge: base and tuned results are reused, so "
                "only the compare model's answers and their judgments are paid for (it "
                "previews first)",
            )
        )
    if finished and plan.evaluators is None:
        steps.append(
            step(
                template(*rerun[len(PROGRAM) :], "--evaluators", "<EVALUATORS_FILE>"),
                "score the stored answers with your own LLM judges and Python checks; nothing "
                "is generated again, so only those judges' calls are paid (it previews first)",
            )
        )
    if retry:
        next_command: str | None = shlex.join(rerun)
    else:
        next_command = None if pushing else deploy_command(plan.run_dir)
    return {"next_command": next_command, "next_steps": steps}


def _push_first(
    publishing: Publishing, rerun: str, run_dir: Path, *, complete: bool
) -> list[dict[str, str]]:
    """The steps of an evaluation the missing push kept unpublished, after it and in
    ``status`` alike: push the dataset, then rerun the evaluation (``rerun`` with
    ``--confirm``), or, once it is complete, deploy it unpublished while the run has no
    deployment. None when ``publishing`` needs no push."""
    push = publishing.push_step()
    if push is None:
        return []
    if complete:
        why = (
            "then rerun the evaluation to publish it: every result is reused, so it makes no "
            "new model calls (it writes to Datadog: ask the user)"
        )
    else:
        why = (
            "then rerun the evaluation: it finishes the cases left and publishes it (paid: "
            "ask the user)"
        )
    steps = [push, step(f"{rerun} --confirm", why)]
    if complete and (deploy := deploy_command(run_dir)) is not None:
        steps.append(step(deploy, "or deploy the tuned model unpublished (a free preview)"))
    return steps


def deploy_command(run_dir: Path) -> str | None:
    """``deploy`` for the run (a free preview), or ``None`` while it is deployed already.

    An unreadable ``deployment.json`` still gets ``deploy``, which reports it, rather than
    hiding a deployment that may be live.
    """
    try:
        receipt = RunDir(run_dir).read_deployment()
    except FinetuneError:
        return command("deploy", run_dir)
    if receipt is not None and not receipt.get("deleted_at"):
        return None
    return command("deploy", run_dir)


def evaluation_exit_code(result: Mapping[str, Any]) -> int:
    """1 when an evaluation's result says it failed: it raised (``failed``), its judge failed
    calibration, its publishing failed, or every pair of a target ended in an error (its
    skipped cases aside), so there is nothing to compare; otherwise 0."""
    if result.get("status") in {"failed", "calibration_failed"}:
        return 1
    publication = result.get("publication")
    if isinstance(publication, Mapping) and publication.get("status") == "failed":
        return 1
    if result.get("status") != "complete_with_errors":
        return 0
    targets = result["summary"]["targets"].values()
    return 1 if any(not target["completed"] and target["errors"] for target in targets) else 0


# --- the run's record of its evaluations --------------------------------------------------

RECORD_STATUSES: Final = frozenset(
    {"running", "complete", "complete_with_errors", "partial", "skipped", "failed"}
)
"""How a recorded evaluation stands: ``running`` from the start of its replay until its
result is recorded (so a crash leaves it), then its result's ``status``."""


def _ci(value: object) -> list[Any] | None:
    return list(value) if isinstance(value, list | tuple) else None


def headline(summary: Mapping[str, Any]) -> dict[str, Any]:
    """An evaluation summary's headline: the judge pass rate per target on its own judged
    cases (``judged_cases``), tuned minus base with its 95% CI beside both pass rates on the
    cases it pairs (``common_pass_rate`` on ``common_cases``), the paired verdicts and the
    cases; with a compare target, tuned minus compare too."""
    from pup_finetune.evaluation.replay import COMPARE

    targets: Mapping[str, Any] = summary.get("targets") or {}
    difference: Mapping[str, Any] = summary.get("tuned_minus_base") or {}
    paired: Mapping[str, Any] = summary.get("paired") or {}
    verdicts = ("wins", "regressions", "ties")

    def judged(name: str) -> Mapping[str, Any]:
        found = (targets.get(name) or {}).get("judge_pass")
        return found if isinstance(found, Mapping) else {}

    names = ("base", "tuned", COMPARE) if COMPARE in targets else ("base", "tuned")
    found: dict[str, Any] = {
        "judge_pass_rate": {
            **{name: judged(name).get("rate") for name in names},
            "judged_cases": {name: judged(name).get("n") for name in names},
            "common_pass_rate": difference.get("common_pass_rate"),
            "common_cases": difference.get("n_cases"),
            "delta": difference.get("judge_pass_rate"),
            "ci95": _ci(difference.get("ci95")),
        },
        "paired": {key: paired.get(key, 0) for key in verdicts},
    }
    versus = summary.get("tuned_minus_compare")
    if isinstance(versus, Mapping):
        against: Mapping[str, Any] = versus.get("paired") or {}
        found["tuned_minus_compare"] = {
            "common_pass_rate": versus.get("common_pass_rate"),
            "common_cases": versus.get("n_cases"),
            "delta": versus.get("judge_pass_rate"),
            "ci95": _ci(versus.get("ci95")),
            **{key: against.get(key, 0) for key in verdicts},
            "skipped": int((targets.get(COMPARE) or {}).get("skipped") or 0),
        }
    evaluators = summary.get("evaluators")
    if isinstance(evaluators, Mapping):
        found["evaluator"] = evaluators.get("headline")
        found["evaluators"] = _evaluator_rates(targets, evaluators)
    found |= {"cases": summary.get("n_cases"), "trajectories": summary.get("n_trajectories")}
    truncated = {
        name: int(target.get("truncated") or 0)
        for name, target in targets.items()
        if isinstance(target, Mapping) and target.get("truncated")
    }
    if truncated:
        found["truncated"] = truncated
    return found


def _evaluator_rates(targets: Mapping[str, Any], evaluators: Mapping[str, Any]) -> dict[str, Any]:
    """Each evaluator's and ``all``'s pass rate per target on its own scored cases
    (``scored_cases``), and tuned minus base (and minus compare) with its CI beside both
    pass rates on the ``n`` cases paired."""
    from pup_finetune.evaluation.evaluators import ALL

    found: dict[str, Any] = {}
    for name in [*(evaluators.get("names") or []), ALL]:
        entry: dict[str, Any] = {}
        scored: dict[str, Any] = {}
        for target, values in targets.items():
            rate = ((values.get("evaluators") or {}).get(name) or {}).get("pass") or {}
            entry[target] = rate.get("rate")
            scored[target] = rate.get("n")
        base = (evaluators.get("tuned_minus_base") or {}).get(name) or {}
        entry |= {
            "scored_cases": scored,
            "common_pass_rate": base.get("common_pass_rate"),
            "delta": base.get("delta"),
            "ci95": _ci(base.get("ci95")),
            "n": base.get("n_cases"),
        }
        versus = (evaluators.get("tuned_minus_compare") or {}).get(name)
        if isinstance(versus, Mapping):
            entry["tuned_minus_compare"] = {
                "common_pass_rate": versus.get("common_pass_rate"),
                "delta": versus.get("delta"),
                "ci95": _ci(versus.get("ci95")),
                "n": versus.get("n_cases"),
            }
        found[name] = entry
    return found


def _out_dir_key(out_dir: Path, run_dir: Path) -> str:
    """How ``evaluation.json`` names an evaluation directory: relative to the run when it is
    inside it (so the record survives moving the run), else absolute."""
    absolute = Path(os.path.abspath(out_dir))
    try:
        return absolute.relative_to(Path(os.path.abspath(run_dir))).as_posix()
    except ValueError:
        return str(absolute)


def _settings_record(plan: EvalPlan) -> dict[str, Any]:
    settings = plan.settings
    return {
        "out_dir": _out_dir_key(settings.out_dir, plan.run_dir),
        "max_cases": settings.max_cases_per_trajectory,
        "judge_model": plan.judge.spec,
        "concurrency": settings.concurrency,
        "max_tokens": settings.max_tokens,
        "seed": settings.seed,
        "skip_calibration": settings.skip_calibration,
        "publish": settings.publish,
        "reuse": settings.reuse,
        **({"compare_model": plan.compare.spec} if plan.compare is not None else {}),
    }


def _is_complete(status: str, publication: str | None, summary: object) -> bool:
    """Whether an evaluation is done: :func:`_finished`, and its publishing neither failed nor
    was cancelled (either is what a rerun continues)."""
    return _finished(status, summary) and publication not in {"failed", "cancelled", "pending"}


def _finished(status: str, summary: object) -> bool:
    """Whether every pair is judged, without an error a retry could fix: a rerun makes no
    model call."""
    return status == "complete" or (status == "complete_with_errors" and _errors_final(summary))


def _errors_final(summary: object) -> bool:
    """Whether every error in ``summary`` is a prompt the renderer refused, which a rerun
    never retries: no other pair error, and no evaluator error on an answer (an evaluator
    counts each pair error as its own too)."""
    if not isinstance(summary, Mapping):
        return False
    return all(
        target["errors"] == target["errors_unrenderable"]
        and all(
            entry["errors"] == target["errors"]
            for entry in (target.get("evaluators") or {}).values()
        )
        for target in summary["targets"].values()
    )


def record_evaluation(plan: EvalPlan, outcome: Mapping[str, Any]) -> None:
    """Record ``outcome`` (an evaluation's result, or ``{"status": "running"}``) in the run's
    ``evaluation.json`` (:data:`~pup_finetune.runs.EVALUATIONS_FILE`), under its directory,
    with every setting, so ``status`` can name the command that continues it or, once it is
    complete, the next step.

    Never raises: the evaluation itself is decided already. A record that cannot be written
    is a warning in ``outcome`` (when it has a ``warnings`` list).
    """
    status = str(outcome.get("status"))
    if status not in RECORD_STATUSES:
        return
    publication = outcome.get("publication")
    published = str(publication.get("status")) if isinstance(publication, Mapping) else None
    summary = outcome.get("summary")
    settings = _settings_record(plan)
    entry = {
        "settings": settings,
        "status": status,
        "complete": _is_complete(status, published, summary),
        "publication": published,
        "headline": headline(summary) if isinstance(summary, Mapping) else None,
        "recorded_at": utcnow().isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    run = RunDir(plan.run_dir)
    try:
        record = run.read_evaluations() or {}
        evaluations = record.get("evaluations")
        found = dict(evaluations) if isinstance(evaluations, Mapping) else {}
        key = str(settings["out_dir"])
        found[key] = entry
        run.write_evaluations({"latest": key, "evaluations": found})
    except (FinetuneError, OSError) as exc:
        warnings = outcome.get("warnings")
        if isinstance(warnings, list):
            warnings.append(
                f"could not record the evaluation in {display_path(run.evaluations_path)}: "
                f"{redact(str(exc))}"
            )


@dataclass(frozen=True, slots=True)
class RecordedEvaluation:
    """An evaluation of a run, as its ``evaluation.json`` records it."""

    out_dir: Path
    status: str
    complete: bool
    publication: str | None
    headline: dict[str, Any] | None
    settings: EvalSettings
    judge_model: str

    def command(self, run_dir: Path) -> str:
        """The ``evaluate`` command that continues it (a free preview), every setting
        pinned, so its directory's identity check accepts it."""
        words = evaluate_words(
            run_dir,
            self.settings,
            self.judge_model,
            pin=True,
            compare_spec=self.settings.compare_model,
        )
        return shlex.join(words)


def recorded_evaluation(run_dir: Path) -> RecordedEvaluation | None:
    """The evaluation ``status`` reports for the run in ``run_dir``: the newest complete one,
    else the newest one; ``None`` when none is recorded or the record cannot be read."""
    try:
        record = RunDir(run_dir).read_evaluations()
    except FinetuneError:
        return None
    evaluations = record.get("evaluations") if record else None
    if not isinstance(evaluations, Mapping):
        return None
    latest = record.get("latest") if record else None
    entries: list[RecordedEvaluation] = []
    for _, entry in sorted(evaluations.items(), key=lambda item: _age(item, latest)):
        found = _recorded(run_dir, entry)
        if found is not None:
            entries.append(found)
    if not entries:
        return None
    complete = [entry for entry in entries if entry.complete]
    return (complete or entries)[-1]


def _age(item: tuple[str, object], latest: object) -> tuple[str, bool]:
    """Orders ``evaluation.json``'s entries (its keys are sorted) oldest first, ``latest``
    last of those recorded in the same second."""
    key, entry = item
    recorded_at = entry.get("recorded_at") if isinstance(entry, Mapping) else None
    return str(recorded_at or ""), key == latest


def _recorded(run_dir: Path, entry: object) -> RecordedEvaluation | None:
    if not isinstance(entry, Mapping) or not isinstance(entry.get("settings"), Mapping):
        return None
    settings: Mapping[str, Any] = entry["settings"]
    try:
        out_dir = Path(str(settings["out_dir"]))
        out_dir = out_dir if out_dir.is_absolute() else run_dir / out_dir
        values = EvalSettings(
            out_dir=out_dir,
            max_cases_per_trajectory=int(settings["max_cases"]),
            judge_model=str(settings["judge_model"]),
            concurrency=int(settings["concurrency"]),
            max_tokens=int(settings["max_tokens"]),
            seed=int(settings["seed"]),
            skip_calibration=bool(settings.get("skip_calibration")),
            publish=bool(settings.get("publish", True)),
            compare_model=str(settings["compare_model"]) if settings.get("compare_model") else None,
            reuse=bool(settings.get("reuse", True)),
        )
    except (KeyError, TypeError, ValueError):
        return None
    found_headline = entry.get("headline")
    publication = entry.get("publication")
    return RecordedEvaluation(
        out_dir=out_dir,
        status=str(entry.get("status")),
        complete=bool(entry.get("complete")),
        publication=str(publication) if publication is not None else None,
        headline=dict(found_headline) if isinstance(found_headline, Mapping) else None,
        settings=values,
        judge_model=str(settings["judge_model"]),
    )


def next_after_training(run_dir: Path) -> dict[str, Any]:
    """``next_command`` and ``next_steps`` for a trained run with a best epoch: ``evaluate``
    (a free preview) when it was never evaluated; the recorded evaluation's own command while
    it is not complete (every setting pinned, so it continues that evaluation); once it is,
    ``deploy`` (a free preview, ``None`` while the run is deployed), unless the missing push
    kept it unpublished (:func:`_unpublished_next`)."""
    found = recorded_evaluation(run_dir)
    if found is None:
        return {"next_command": command("evaluate", run_dir), "next_steps": []}
    if not found.complete:
        return {"next_command": found.command(run_dir), "next_steps": []}
    if found.publication == "skipped":
        return _unpublished_next(run_dir, found)
    return {"next_command": deploy_command(run_dir), "next_steps": []}


def _unpublished_next(run_dir: Path, found: RecordedEvaluation) -> dict[str, Any]:
    """What follows a complete evaluation the missing push kept unpublished, read from the
    run and its dataset as they are now: while the dataset still needs a push, no next
    command and the push-first steps the evaluation gave (:func:`_push_first`, without its
    compare suggestion); once it is pushed, the evaluation's own command, which previews
    publishing it, or deploying it unpublished. A dataset or push receipt it cannot read
    gets no next command, since that command's preview would fail on it, and a step that
    says why."""
    rerun = found.command(run_dir)
    try:
        dataset = DatasetDir(Path(str(RunDir(run_dir).manifest()["dataset_dir"])))
        publishing = plan_publishing(dataset, enabled=True)
    except (FinetuneError, KeyError) as exc:
        why = exc.message if isinstance(exc, FinetuneError) else "the run names no dataset"
        return {
            "next_command": None,
            "next_steps": [step(rerun, f"rerun it once its dataset can be read: {why}")],
        }
    pushing = _push_first(publishing, rerun, run_dir, complete=True)
    if pushing:
        return {"next_command": None, "next_steps": pushing}
    deploy = deploy_command(run_dir)
    unpublished = "or deploy the tuned model unpublished (a free preview)"
    return {
        "next_command": rerun,
        "next_steps": [] if deploy is None else [step(deploy, unpublished)],
    }


SPEND_POLL_S: Final = 0.5
"""How often :class:`EvaluateAfterTraining` checks the spend cap (and a cancel) while it
evaluates."""


def stop_cause(result: TrainResult) -> str:
    """Why training stopped, in words, naming the cap for a ``budget_exhausted`` run: the
    token budget (``--max-train-tokens``) or the spend cap (``--max-spend-usd``), from the
    warning the loop recorded when it stopped."""
    if result.stopped_reason == "cancelled":
        return "it was cancelled"
    if result.stopped_reason != "budget_exhausted":
        return f"it stopped: {result.stopped_reason}"
    said = " ".join(result.warnings)
    if "max_train_tokens" in said:
        return "it reached its token budget, --max-train-tokens"
    if "max_spend_usd" in said:
        return "it reached its spend cap, --max-spend-usd"
    return "it reached its budget"


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
        if result.best_epoch is None:
            # Nothing to evaluate, here or later: no command would change that.
            self.outcome = {
                "status": "skipped",
                "reason": f"the run was not evaluated because no epoch finished, so there is "
                f"no checkpoint to evaluate: {stop_cause(result)}",
                "next_command": None,
            }
            return
        if result.stopped_reason not in EVALUATED_STOPS or spent is not None:
            why = stop_cause(result)
            if why.endswith("--max-spend-usd"):
                why += ", which evaluating in its session would spend past"
            elif result.stopped_reason == "budget_exhausted":
                why += " (a budget-capped run is evaluated on request)"
            if result.stopped_reason in EVALUATED_STOPS:
                why = f"its spend cap is reached ({spent})"
            self.outcome = {
                "status": "skipped",
                "reason": f"the run was not evaluated because {why}; next_command evaluates "
                "it in a session of its own",
                "next_command": next_command,
            }
            record_evaluation(self.plan, self.outcome)
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
                "out_dir": display_path(self.plan.out_dir),
                "next_command": next_command,
            }
            record_evaluation(self.plan, self.outcome)
            raise
        if self.outcome is not None:
            record_evaluation(self.plan, self.outcome)

    def _evaluate(
        self, session: TrainingSession, result: TrainResult, stop: threading.Event
    ) -> None:
        from pup_finetune.evaluation.targets import best_training_checkpoint, session_samplers

        training = best_training_checkpoint(result)
        report = prepare_eval_dir(self.plan, training, self.client, calibration=self.calibration)
        info(f"evaluating the run {self.plan.run_id} in its training session")
        record_evaluation(self.plan, {"status": "running"})
        with session_samplers(
            session,
            result,
            run_id=self.plan.run_id,
            max_checkpoint_name_len=self.max_checkpoint_name_len,
            targets=_generating(self.plan),
        ) as targets:
            outcome = replay(self.plan, targets.samplers, self.client, training, cancel=stop)
        self.outcome = replay_result(
            self.plan,
            outcome,
            calibration=report,
            tuned_sampler=targets.tuned_checkpoint,
            shared=share_base(self.plan),
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

    Raises :class:`WrongPhase` for another phase. A cancelled run that finished no epoch has
    nothing to evaluate either, which
    :func:`~pup_finetune.evaluation.targets.best_training_checkpoint` refuses.
    """
    state = run.load_state()
    if state.phase not in FINISHED_PHASES:
        raise WrongPhase(
            f"the run in {display_path(run.path)} is {state.phase}, not completed or cancelled, so there is "
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
        raise SettingsConflict(
            f"the dataset in {display_path(dataset.path)} changed since the run in "
            f"{display_path(run.path)} was trained, "
            "so its test split may no longer be held out",
            hint="restore the dataset the run trained on, or train a new run",
        )


def _check_renderer(run: RunDir, manifest: Mapping[str, Any], provider: Provider) -> None:
    provider.validate_options(run.spec().provider_options)
    profile = run.profile()
    if provider.renderer(profile).identity() != manifest.get("rendering"):
        raise SettingsConflict(
            f"the {profile.alias} renderer changed since the run in {display_path(run.path)} was trained, "
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
        preview["run_dir"] = display_path(self.plan.run_dir)
        return preview

    def execute(self) -> dict[str, Any]:
        """Calibrate, replay and publish; what the command emits (see
        :func:`evaluation_exit_code` for its exit code).

        Ctrl-C or SIGTERM stops it gracefully (a ``partial`` result); a second Ctrl-C stops
        at once with :class:`FinetuneError`.
        """
        from pup_finetune.evaluation.compare import probe

        plan = self.plan
        cancel = threading.Event()
        work = plan.work(self.training)
        with ExitStack() as stack:
            client = stack.enter_context(open_judge_client(plan.judge, calls=work.judge_calls))
            busy = [] if plan.evaluators is None else busy_judges(plan.evaluators, work)
            scoring = stack.enter_context(open_evaluator_clients(busy))
            compare, target = self._open_compare(stack, work), plan.compare
            check = (
                None
                if compare is None or target is None
                else lambda: probe(compare, target, plan.cases)
            )
            report = prepare_eval_dir(
                plan, self.training, client, probe=check, evaluator_clients=scoring
            )
            if report is not None and not report.get("ok"):
                return calibration_failed(plan, report)
            record_evaluation(plan, {"status": "running"})
            try:
                with cancel_on_signals(cancel):
                    result = self._replay(client, compare, cancel, report, scoring)
                record_evaluation(plan, result)
                return result
            except KeyboardInterrupt as exc:
                raise Interrupted(
                    f"evaluation stopped at once (Ctrl-C); what finished is kept in {display_path(plan.out_dir)}",
                    hint=f"preview what is left with `{shlex.join(plan.evaluate_command())}`; "
                    "its next_command continues it",
                    interrupt=exc,
                ) from None

    def _open_compare(self, stack: ExitStack, work: EvalWork) -> Sampler | None:
        """The compare model's sampler when it has cases left to generate, opened on
        ``stack`` before anything is written or any training session opens."""
        from pup_finetune.evaluation.compare import open_compare_sampler
        from pup_finetune.evaluation.replay import COMPARE

        plan = self.plan
        if plan.compare is None or not work.by_target[COMPARE]["generate"]:
            return None
        return stack.enter_context(
            open_compare_sampler(plan.compare, plan.cases, seed=plan.settings.seed)
        )

    def _replay(
        self,
        client: ChatCompleter,
        compare: Sampler | None,
        cancel: threading.Event,
        report: Mapping[str, Any] | None,
        scoring: Mapping[str, ChatCompleter],
    ) -> dict[str, Any]:
        """Replay with the samplers of the targets that have cases left to generate: a
        training session only for base and tuned, and ``compare``, already open, for the
        compare model; the user's judges on ``scoring``."""
        from pup_finetune.evaluation.replay import COMPARE
        from pup_finetune.evaluation.targets import run_samplers

        plan = self.plan
        samplers: dict[str, Sampler] = {}
        tuned_sampler = None
        with ExitStack() as stack:
            if need := _generating(plan):
                targets = stack.enter_context(
                    run_samplers(self.provider, self.run, self.result, targets=need)
                )
                samplers |= targets.samplers
                tuned_sampler = targets.tuned_checkpoint
            if compare is not None:
                samplers[COMPARE] = compare
            if samplers:
                info(
                    f"evaluating run {plan.run_id}: {len(plan.cases)} cases, "
                    f"{joined(list(samplers))} (Ctrl-C stops)"
                )
            else:
                info(
                    f"nothing left to generate in {display_path(plan.out_dir)}: judging and "
                    "summarizing only"
                )
            outcome = replay(
                plan, samplers, client, self.training, cancel=cancel, evaluator_clients=scoring
            )
        return replay_result(
            plan,
            outcome,
            calibration=report,
            tuned_sampler=tuned_sampler,
            shared=share_base(plan),
        )


def _generating(plan: EvalPlan) -> list[str]:
    """The targets of the training session (base, tuned) with a case left to generate in
    the evaluation directory, read after what was copied from the dataset's store."""
    from pup_finetune.evaluation.replay import TARGETS, EvalDir, remaining_work

    left = remaining_work(EvalDir(plan.out_dir), plan.replays)
    return [target for target in TARGETS if left[target]["generate"]]


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
    # A missing SDK or renderer fails the preview, not the evaluation after --confirm.
    require_packages(str(manifest["provider"]), runtime, "evaluate")
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
    plan = dataclasses.replace(plan, standalone=True)
    return StandaloneEvaluation(
        plan=plan, provider=provider, run=run, result=result, training=training
    )
