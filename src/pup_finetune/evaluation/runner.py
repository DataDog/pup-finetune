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
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

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
from pup_finetune.display import display_path, quoted
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
from pup_finetune.providers import get_provider, require_packages
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
from pup_finetune.storage import LOCK_FILE_NAME, canonical_json, finite_json
from pup_finetune.training.records import StoppedReason, TrainResult

if TYPE_CHECKING:
    from pup_finetune.evaluation.cases import ReplayCase
    from pup_finetune.evaluation.compare import CompareTarget
    from pup_finetune.evaluation.judge import Judgment
    from pup_finetune.evaluation.replay import Publisher, ReplaySummary
    from pup_finetune.inference import ChatCompleter, JudgeConfig

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
    "EvaluateAfterTraining",
    "Publishing",
    "RecordedEvaluation",
    "StandaloneEvaluation",
    "calibration_failed",
    "calibration_line",
    "cancel_on_signals",
    "deploy_command",
    "evaluate_words",
    "evaluation_exit_code",
    "finished_result",
    "headline",
    "next_after_training",
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


def _free_dir(path: Path) -> Path:
    """``path`` with ``-2``, ``-3``, ... added: the first that does not exist or is empty."""
    n = 2
    while True:
        candidate = path.with_name(f"{path.name}-{n}")
        if not candidate.exists() or (candidate.is_dir() and not any(candidate.iterdir())):
            return candidate
        n += 1


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


def evaluate_words(
    run_dir: Path,
    settings: EvalSettings,
    judge_spec: str,
    *,
    pin: bool,
    compare_spec: str | None = None,
) -> list[str]:
    """The ``evaluate`` command (without ``--confirm``) for the run in ``run_dir`` with
    ``settings`` and the judge ``judge_spec``: the judge always, and the other settings that
    differ from the defaults, or every one of them with ``pin``. ``compare_spec``, the
    compare model, is repeated whenever there is one: like the judge, it decides where the
    data goes."""
    defaults = EvalSettings(out_dir=run_dir / DEFAULT_OUT_DIR_NAME)
    words = [*PROGRAM, "evaluate", display_path(run_dir)]
    if pin or settings.out_dir != defaults.out_dir:
        words += ["--out-dir", display_path(settings.out_dir)]
    if pin or settings.max_cases_per_trajectory != defaults.max_cases_per_trajectory:
        words += ["--max-cases", str(settings.max_cases_per_trajectory)]
    words += ["--judge-model", judge_spec]
    if compare_spec is not None:
        words += ["--compare-model", compare_spec]
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
    return words


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
    standalone: bool = False
    """``evaluate`` (not ``train --evaluate``): its commands repeat only the settings that
    differ from the defaults, and the judge (see :meth:`evaluate_command`)."""
    compare: CompareTarget | None = None
    """The compare model: ``--compare-model``'s, else the one the directory has attached."""

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
            self.provider == "fake"
            and self.judge.provider == "fake"
            and (self.compare is None or self.compare.offline)
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
        )

    def calibration_calls(self) -> int:
        if self.settings.skip_calibration:
            return 0
        return 3 * min(CALIBRATION_CASES, len(self.cases))

    def evaluate_command(self, *, pin: bool | None = None) -> list[str]:
        """The standalone ``evaluate`` command for this plan, without ``--confirm``.

        For ``evaluate`` it repeats the settings that differ from the defaults, and always
        the judge, which decides where the data goes; the identity check of
        ``evaluation.json`` refuses any drift. After ``train --evaluate`` (or with ``pin``)
        every setting is pinned, so the command continues the evaluation that run started.
        """
        return evaluate_words(
            self.run_dir,
            self.settings,
            self.judge.spec,
            pin=not self.standalone if pin is None else pin,
            compare_spec=None if self.compare is None else self.compare.spec,
        )

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
            targets=self.targets,
            compare_model=None if self.compare is None else self.compare.spec,
            compare_max_tokens=None if self.compare is None else self.compare.config.max_tokens,
        )
        return ExperimentsPublisher(
            Pup(),
            settings,
            out_dir=self.out_dir,
            runtime=self.runtime,
            cancel=cancel,
            progress=info,
        )

    def recipients(self) -> list[str]:
        """Who receives the evaluation's data, one entry per kind of data, e.g.
        ``test split -> fireworks (generate), Anthropic api.anthropic.com (judge)``."""
        from pup_finetune.judges import recipient

        judge = (
            "the offline fake judge (judge; nothing is sent)"
            if self.judge.provider == "fake"
            else f"{recipient(self.judge)} (judge)"
        )
        compare = "" if self.compare is None else f", {self.compare.recipient()}"
        found = [f"test split -> {self.provider} (generate){compare}, {judge}"]
        if self.publishing.ready:
            found.append("test cases, generated answers and scores -> Datadog (experiments)")
        return found

    def _check_compare(self, eval_dir: Any, tuned_checkpoint: CheckpointRef) -> None:
        """Fail the preview, as ``--confirm`` would (:meth:`EvalDir.check_compare`), when the
        directory has attached another compare model, never offering that ``--confirm``."""
        stored = eval_dir.stored_compare()
        wanted = self.compare_identity
        if stored is None or wanted is None or finite_json(wanted) == stored:
            return
        raise SettingsConflict(
            f"{self.out_dir} compares with {stored.get('spec')}, not {wanted['spec']}, and a "
            "compare target never changes once attached",
            hint="leave --compare-model out to continue the attached one, or compare with "
            f"another model in a new directory: pass --out-dir {_q(_free_dir(self.out_dir))}",
        )

    def preview(self, *, tuned: str, tuned_checkpoint: CheckpointRef | None) -> dict[str, Any]:
        """The calls the evaluation makes. With ``tuned_checkpoint`` (a standalone run), what
        ``--out-dir`` already holds under the same settings is left out of the counts."""
        from pup_finetune.datadog.experiments import experiment_name
        from pup_finetune.evaluation.replay import (
            CAVEAT,
            COMPARE,
            EvalDir,
            identity_changes,
            remaining_work,
        )

        settings = self.settings
        warnings: list[str] = []
        targets = self.targets
        by_target = {
            target: {"done": 0, "generate": len(self.cases), "judge": len(self.cases)}
            for target in targets
        }
        calibration = self.calibration_calls()
        eval_dir = EvalDir(self.out_dir)
        stored = eval_dir.stored_identity() if tuned_checkpoint is not None else None
        if tuned_checkpoint is not None and stored is not None:
            wanted = finite_json(self.identity(tuned_checkpoint))
            if wanted == stored:
                self._check_compare(eval_dir, tuned_checkpoint)
                by_target = {
                    target: remaining_work(eval_dir, self.cases, (target,)) for target in targets
                }
                if (eval_dir.read_calibration() or {}).get("ok"):
                    calibration = 0
            elif eval_dir.has_work():
                # --confirm would refuse (EvalDir.check_identity): fail now, never offer it.
                raise SettingsConflict(
                    f"{self.out_dir} holds an evaluation with other settings (changed: "
                    f"{', '.join(identity_changes(stored, wanted))}), so its results cannot be "
                    "reused",
                    hint="evaluate into a new directory: pass --out-dir "
                    f"{_q(_free_dir(self.out_dir))}",
                )
        from pup_finetune.judges import destination as judge_destination

        remaining = {
            key: sum(counts[key] for counts in by_target.values())
            for key in ("done", "generate", "judge")
        }
        trained = sum(by_target[target]["generate"] for target in targets if target != COMPARE)
        compared = by_target.get(COMPARE, {}).get("generate", 0)
        trajectories = len({case.row_id for case in self.cases})
        judged_at = (
            "the offline fake judge (nothing is sent)"
            if self.judge.provider == "fake"
            else judge_destination(self.judge)
        )
        compare_to = "" if self.compare is None else f", to {self.compare.destination()}"
        destination = (
            f"the test split's conversations are sent to {self.provider} to generate"
            f"{compare_to} and to {judged_at} to judge"
        )
        publish = self.publishing.status()
        if publish["status"] == "skipped":
            warnings.append(f"the evaluation will not be published: {publish['reason']}")
        elif publish["status"] == "planned":
            publish["experiments"] = [experiment_name(t, self.run_id) for t in targets]
            publish["spans_per_experiment"] = trajectories + len(self.cases)
            destination += (
                ", and the cases with the generated answers and their scores are published to "
                "Datadog as experiments on the pushed dataset"
            )
        from pup_finetune.judges import billing as judge_billing
        from pup_finetune.providers.base import SamplingUsage

        judge_calls = remaining["judge"] + calibration
        sampling = get_provider(self.provider, self.runtime).sampling_billing(
            SamplingUsage(calls=trained, max_tokens_per_call=settings.max_tokens)
        )
        judge_name = "the fake judge" if self.judge.provider == "fake" else self.judge.spec
        models = "the base and the tuned model on " + self.provider
        if self.compare is not None:
            models += f" and with {self.compare.spec}"
        action = (
            f"evaluate {_cases(len(self.cases))} from {trajectories} test conversations: "
            f"generate each with {models}, and judge each answer with {judge_name}"
        )
        if publish["status"] == "planned":
            action += f", then publish {_COUNTS[len(targets)]} experiments to Datadog"
        billing = [sampling]
        if self.compare is not None:
            billing.append(self.compare.billing(compared))
        billing.append(judge_billing(self.judge, judge_calls))
        target_names: dict[str, str] = {"base": self.profile.base_model, "tuned": tuned}
        if self.compare is not None:
            target_names[COMPARE] = self.compare.spec
        compare_max = 0 if self.compare is None else self.compare.config.max_tokens
        return {
            "action": action,
            "billing": "; ".join(billing),
            "run_id": self.run_id,
            "out_dir": display_path(self.out_dir),
            "rows": {"test": trajectories},
            "cases": len(self.cases),
            "trajectories": trajectories,
            "targets": target_names,
            **({"compare": self.compare.describe()} if self.compare is not None else {}),
            "judge_model": self.judge.spec,
            "judge_provider": self.judge.provider,
            "judge_endpoint": self.judge.base_url,
            "already_done": remaining["done"],
            "generation_calls": remaining["generate"],
            "generation_calls_by_target": {
                target: counts["generate"] for target, counts in by_target.items()
            },
            "judge_calls": remaining["judge"] + calibration,
            "calibration_calls": calibration,
            "max_generated_tokens": trained * settings.max_tokens + compared * compare_max,
            "settings": {
                "judge_model": self.judge.spec,
                **({"compare_model": self.compare.spec} if self.compare is not None else {}),
                "max_cases": settings.max_cases_per_trajectory,
                "max_tokens": settings.max_tokens,
                "concurrency": settings.concurrency,
                "seed": settings.seed,
                "calibration": not settings.skip_calibration,
                "publish": settings.publish,
            },
            "publish": publish,
            "data_destination": destination,
            "data_recipients": self.recipients(),
            "caveat": CAVEAT,
            "warnings": warnings,
        }


_COUNTS: Final = {2: "two", 3: "three"}


def _cases(count: int) -> str:
    return f"{count} case{'' if count == 1 else 's'}"


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
    no_publish: str = NO_PUBLISH,
) -> EvalPlan:
    """The evaluation of the run ``run_id`` on ``dataset``'s test split.

    ``no_publish`` is the running command's flag that skips publishing (see
    :func:`plan_publishing`).

    Raises :class:`UsageError` when the test split has no case, or the judge is the model
    being trained or the run's tuned model: the model id the run promotes to
    (:func:`~pup_finetune.runs.model_id_for_run`) or, once known, the ``promoted`` model.
    """
    from pup_finetune.evaluation.cases import cases_from_dataset
    from pup_finetune.evaluation.compare import resolve_compare
    from pup_finetune.evaluation.judge import check_judge_differs
    from pup_finetune.evaluation.replay import EvalDir

    judge_config = resolve_judge(settings.judge_model, provider=provider, runtime=runtime)
    compare = resolve_compare(
        settings.compare_model, EvalDir(settings.out_dir).stored_compare(), runtime=runtime
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
    return EvalPlan(
        run_id=run_id,
        run_dir=run_dir,
        provider=provider,
        profile=profile,
        rendering=dict(rendering),
        cases=cases,
        judge=judge_config,
        settings=settings,
        publishing=plan_publishing(dataset, enabled=settings.publish, no_publish=no_publish),
        runtime=runtime,
        compare=compare,
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
        eval_dir.check_compare(plan.compare_identity)
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
        targets=plan.targets,
        compare=plan.compare_identity,
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
) -> dict[str, Any]:
    """What a finished (or cancelled) evaluation emits.

    ``status`` is ``complete``, ``complete_with_errors`` (every pair has a result, but some
    are errors, which a rerun of ``next_command`` retries) or ``partial`` (cancelled).
    ``publication`` says how the evaluation was published (``published``, ``failed`` or
    ``cancelled``, which a rerun of ``next_command`` continues, ``pending``, ``skipped`` or
    ``disabled``). Spans already published keep the results they were sent with, so after
    a retry of the errors the experiments still show them as errors.
    """
    from pup_finetune.evaluation.replay import COMPARE

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
            f"{errors} case(s) ended in an error, left out of every rate (see the file "
            "files.results names); run next_command to retry them"
        )
        too_long = sum(
            int(target.get("errors_too_long", 0)) for target in summary["targets"].values()
        )
        if too_long:
            hint += (
                f". {too_long} of them are longer than the judge's context window (or the "
                "compare model's), which a retry cannot fix: evaluate into a new --out-dir "
                "with a --judge-model that has a larger context window"
            )
        if publication["status"] in _SENT:
            hint += (
                ". Spans already published keep the results they were sent with, these errors "
                "included: a retry updates the local results only, so evaluate into a new "
                "--out-dir to publish experiments with every result"
            )
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
                f"{truncated} of the {name} model's generations reached --max-tokens "
                f"({plan.settings.max_tokens}) and were cut off; they count as failures, so "
                "evaluate into a new --out-dir with a higher --max-tokens to compare the "
                "models on their answers rather than their length"
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
    retry = status != "complete" or publication["status"] in {"failed", "cancelled"}
    return finite_json(
        {
            "status": status,
            "run_id": plan.run_id,
            "out_dir": display_path(outcome.out_dir),
            "judge_model": plan.judge.spec,
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
            **_replay_next(plan, status=status, publication=publication, retry=retry),
        }
    )


def _replay_next(
    plan: EvalPlan, *, status: str, publication: Mapping[str, Any], retry: bool
) -> dict[str, Any]:
    """``next_command`` and ``next_steps`` after an evaluation.

    One that stopped, has errors or whose publishing failed is continued by the evaluate
    command (a free preview first). A complete one is deployed (a free preview), unless the
    run has a deployment already, whether or not it was published: publishing is optional.
    One that could not be published also gets the steps to publish it: push the dataset,
    then rerun the evaluation; a complete evaluation's rerun reuses every result, so it
    makes no new model calls.
    """
    steps: list[dict[str, str]] = []
    rerun = plan.evaluate_command()
    if publication["status"] == "skipped" and (push := plan.publishing.push_step()) is not None:
        steps.append(push)
        if status == "complete":
            why = (
                "then rerun the evaluation to publish it: every result is reused, so it "
                "makes no new model calls (it writes to Datadog: ask the user)"
            )
        else:
            why = (
                "then rerun the evaluation: it finishes the cases left and publishes it "
                "(paid: ask the user)"
            )
        steps.append(step(shlex.join([*rerun, "--confirm"]), why))
    if status == "complete" and plan.compare is None:
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
    if retry:
        next_command: str | None = shlex.join(rerun)
    else:
        next_command = deploy_command(plan.run_dir)
    return {"next_command": next_command, "next_steps": steps}


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


# --- the run's record of its evaluations --------------------------------------------------

RECORD_STATUSES: Final = frozenset(
    {"running", "complete", "complete_with_errors", "partial", "skipped", "failed"}
)
"""How a recorded evaluation stands: ``running`` from the start of its replay until its
result is recorded (so a crash leaves it), then its result's ``status``."""


def _ci(value: object) -> list[Any] | None:
    return list(value) if isinstance(value, list | tuple) else None


def headline(summary: Mapping[str, Any]) -> dict[str, Any]:
    """An evaluation summary's headline: the judge pass rate per target, tuned minus base
    with its 95% CI, the paired verdicts and the cases; with a compare target, tuned minus
    compare too."""
    from pup_finetune.evaluation.replay import COMPARE

    targets: Mapping[str, Any] = summary.get("targets") or {}
    difference: Mapping[str, Any] = summary.get("tuned_minus_base") or {}
    paired: Mapping[str, Any] = summary.get("paired") or {}
    verdicts = ("wins", "regressions", "ties")

    def rate(name: str) -> Any:
        found = (targets.get(name) or {}).get("judge_pass")
        return found.get("rate") if isinstance(found, Mapping) else None

    rates = {"base": rate("base"), "tuned": rate("tuned")}
    if COMPARE in targets:
        rates[COMPARE] = rate(COMPARE)
    found: dict[str, Any] = {
        "judge_pass_rate": {
            **rates,
            "delta": difference.get("judge_pass_rate"),
            "ci95": _ci(difference.get("ci95")),
        },
        "paired": {key: paired.get(key, 0) for key in verdicts},
    }
    versus = summary.get("tuned_minus_compare")
    if isinstance(versus, Mapping):
        against: Mapping[str, Any] = versus.get("paired") or {}
        found["tuned_minus_compare"] = {
            "delta": versus.get("judge_pass_rate"),
            "ci95": _ci(versus.get("ci95")),
            **{key: against.get(key, 0) for key in verdicts},
        }
    found |= {"cases": summary.get("n_cases"), "trajectories": summary.get("n_trajectories")}
    truncated = {
        name: int(target.get("truncated") or 0)
        for name, target in targets.items()
        if isinstance(target, Mapping) and target.get("truncated")
    }
    if truncated:
        found["truncated"] = truncated
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
        **({"compare_model": plan.compare.spec} if plan.compare is not None else {}),
    }


def _is_complete(status: str, publication: str | None) -> bool:
    """Whether an evaluation is done: every pair judged without error, and its publishing
    neither failed nor was cancelled (either is what a rerun continues)."""
    return status == "complete" and publication not in {"failed", "cancelled", "pending"}


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
        "complete": _is_complete(status, published),
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
        found.pop(key, None)  # the newest last
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
    entries: list[RecordedEvaluation] = []
    for entry in evaluations.values():
        found = _recorded(run_dir, entry)
        if found is not None:
            entries.append(found)
    if not entries:
        return None
    complete = [entry for entry in entries if entry.complete]
    return (complete or entries)[-1]


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


def next_after_training(run_dir: Path) -> str | None:
    """What follows a trained run with a best epoch: ``evaluate`` (a free preview) when it
    was never evaluated; the recorded evaluation's own command while it is not complete
    (every setting pinned, so it continues that evaluation); ``deploy`` (a free preview)
    once it is, or ``None`` while the run is deployed."""
    found = recorded_evaluation(run_dir)
    if found is None:
        return command("evaluate", run_dir)
    if not found.complete:
        return found.command(run_dir)
    return deploy_command(run_dir)


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
        plan = self.plan
        cancel = threading.Event()
        with ExitStack() as stack:
            client = stack.enter_context(open_judge_client(plan.judge))
            compare = self._open_compare(stack)
            report = prepare_eval_dir(plan, self.training, client)
            if report is not None and not report.get("ok"):
                return calibration_failed(plan, report)
            record_evaluation(plan, {"status": "running"})
            try:
                with cancel_on_signals(cancel):
                    result = self._replay(client, compare, cancel, report)
                record_evaluation(plan, result)
                return result
            except KeyboardInterrupt:
                raise Interrupted(
                    f"evaluation stopped at once (Ctrl-C); what finished is kept in {display_path(plan.out_dir)}",
                    hint=f"preview what is left with `{shlex.join(plan.evaluate_command())}`; "
                    "its next_command continues it",
                ) from None

    def _open_compare(self, stack: ExitStack) -> Sampler | None:
        """The compare model's sampler when it has cases left to generate, opened on
        ``stack`` before anything is written or any training session opens. A compare model
        not attached yet is probed first (:func:`~pup_finetune.evaluation.compare.probe`)."""
        from pup_finetune.evaluation.compare import open_compare_sampler, probe
        from pup_finetune.evaluation.replay import COMPARE, EvalDir, remaining_work

        plan = self.plan
        eval_dir = EvalDir(plan.out_dir)
        if plan.compare is None or not remaining_work(eval_dir, plan.cases, (COMPARE,))["generate"]:
            return None
        sampler = stack.enter_context(
            open_compare_sampler(plan.compare, plan.cases, seed=plan.settings.seed)
        )
        if not eval_dir.compare_path.exists():
            info(f"checking that the compare model {plan.compare.spec} answers")
            probe(sampler)
        return sampler

    def _replay(
        self,
        client: ChatCompleter,
        compare: Sampler | None,
        cancel: threading.Event,
        report: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Replay with the samplers of the targets that have cases left to generate: a
        training session only for base and tuned, and ``compare``, already open, for the
        compare model."""
        from pup_finetune.evaluation.replay import COMPARE, TARGETS, EvalDir, remaining_work
        from pup_finetune.evaluation.targets import run_samplers

        plan = self.plan
        trained = remaining_work(EvalDir(plan.out_dir), plan.cases, TARGETS)["generate"]
        samplers: dict[str, Sampler] = {}
        tuned_sampler = None
        with ExitStack() as stack:
            if trained:
                targets = stack.enter_context(run_samplers(self.provider, self.run, self.result))
                samplers |= targets.samplers
                tuned_sampler = targets.tuned_checkpoint
            if compare is not None:
                samplers[COMPARE] = compare
            if samplers:
                info(
                    f"evaluating run {plan.run_id}: {len(plan.cases)} cases, "
                    f"{' and '.join(samplers)} (Ctrl-C stops)"
                )
            else:
                info(
                    f"nothing left to generate in {display_path(plan.out_dir)}: judging and "
                    "summarizing only"
                )
            outcome = replay(plan, samplers, client, self.training, cancel=cancel)
        return replay_result(plan, outcome, calibration=report, tuned_sampler=tuned_sampler)


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
