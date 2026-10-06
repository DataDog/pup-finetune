# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune evaluate`: replay a run's held-out test cases with the base and tuned model.

Evaluating is paid (it generates with both models and calls a judge) and creates remote
sampling resources, so without ``--confirm`` it previews the calls it would make (free, and
allowed with ``PUP_READ_ONLY`` set) and prints the exact command that runs it. With
``--confirm`` it calibrates the judge (unless ``--skip-calibration``; a failed calibration
stops before any generation, with exit code 1 and the report), then replays every case
(:func:`~pup_finetune.evaluation.replay.run_replay`) and emits the summary. Ctrl-C or
SIGTERM stops it gracefully: the generations in flight finish, everything done is kept, and
running the command again continues where it stopped.

The planning, calibration and replay helpers here are shared with ``train --evaluate``,
which evaluates in the training session before it closes. They import
:mod:`pup_finetune.evaluation` (and through it the HTTP client of the judge) only when they
run, so loading the CLI for any other command stays light.
"""

from __future__ import annotations

import contextlib
import dataclasses
import shlex
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final

import typer

from pup_finetune.cli._common import CommandResult, get_runtime, run_command
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.errors import (
    FinetuneError,
    StateError,
    UsageError,
    require_confirm,
    require_writable,
)
from pup_finetune.output import info
from pup_finetune.providers import get_provider
from pup_finetune.providers.base import (
    CheckpointRef,
    Generation,
    ModelProfile,
    Provider,
    Sampler,
    TrainingSession,
)
from pup_finetune.runs import RunDir, dataset_content_hash, error_json
from pup_finetune.runs import cancel_on_signals as _cancel_on_signals
from pup_finetune.runtime import Runtime
from pup_finetune.storage import LOCK_FILE_NAME, finite_json
from pup_finetune.training.records import StoppedReason, TrainResult

if TYPE_CHECKING:
    from pup_finetune.evaluation.cases import ReplayCase
    from pup_finetune.evaluation.judge import Judgment
    from pup_finetune.evaluation.replay import ReplaySummary
    from pup_finetune.inference import ChatCompleter, JudgeConfig

DEFAULT_OUT_DIR_NAME: Final = "eval"
DEFAULT_MAX_CASES_PER_TRAJECTORY: Final = 2
DEFAULT_CONCURRENCY: Final = 4
DEFAULT_MAX_TOKENS: Final = 1024
DEFAULT_SEED: Final = 0
CALIBRATION_CASES: Final = 4
"""How many cases :func:`~pup_finetune.evaluation.judge.calibrate` judges, twice each."""

EVALUATED_STOPS: Final[frozenset[StoppedReason]] = frozenset({"completed", "early_stopped"})
"""How a training must have stopped for ``train --evaluate`` to evaluate it.

A cancelled run was stopped on request, and a ``budget_exhausted`` one reached the spend
cap, which the training session's time counts against: evaluating in that session would
spend past it. Both are skipped with the ``evaluate`` command to run instead, which
evaluates any completed run (a budget-capped one too) in a session of its own, on request.
"""

_CANCEL_NOTICE: Final = (
    b"cancel requested: finishing the generations in flight, then writing the summary"
)


def _q(value: object) -> str:
    return shlex.quote(str(value))


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
        return 2 * min(CALIBRATION_CASES, len(self.cases))

    def evaluate_command(self) -> list[str]:
        """The standalone ``evaluate`` command for this plan, every setting pinned."""
        settings = self.settings
        words = ["pup", "finetune", "evaluate", str(self.run_dir)]
        words += ["--out-dir", str(settings.out_dir)]
        words += ["--max-cases-per-trajectory", str(settings.max_cases_per_trajectory)]
        words += ["--judge-model", self.judge.model]
        words += ["--concurrency", str(settings.concurrency)]
        words += ["--max-tokens", str(settings.max_tokens), "--seed", str(settings.seed)]
        if settings.skip_calibration:
            words.append("--skip-calibration")
        return words

    def train_flags(self) -> list[str]:
        """The ``train --evaluate`` flags for this plan, every setting pinned."""
        settings = self.settings
        words = ["--evaluate", "--eval-out-dir", str(settings.out_dir)]
        words += ["--eval-max-cases-per-trajectory", str(settings.max_cases_per_trajectory)]
        words += ["--eval-judge-model", self.judge.model]
        words += ["--eval-concurrency", str(settings.concurrency)]
        words += ["--eval-max-tokens", str(settings.max_tokens)]
        words += ["--eval-seed", str(settings.seed)]
        if settings.skip_calibration:
            words.append("--eval-skip-calibration")
        return words

    def preview(self, *, tuned: str, tuned_checkpoint: CheckpointRef | None) -> dict[str, Any]:
        """The calls the evaluation makes. With ``tuned_checkpoint`` (a standalone run), what
        ``--out-dir`` already holds under the same settings is left out of the counts."""
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
        return {
            "run_id": self.run_id,
            "out_dir": str(self.out_dir),
            "cases": len(self.cases),
            "trajectories": len({case.row_id for case in self.cases}),
            "targets": {"base": self.profile.base_model, "tuned": tuned},
            "judge_model": self.judge.model,
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
            "data_destination": (
                f"the test split's conversations are sent to {self.provider} to generate and "
                f"to {self.judge.base_url} to judge"
            ),
            "caveat": CAVEAT,
            "warnings": warnings,
        }


def resolve_judge(judge_model: str | None, *, provider: str, runtime: Runtime) -> JudgeConfig:
    """The judge for ``--judge-model``: the default judge, or for a ``fake`` run the fake one.

    :data:`~pup_finetune.evaluation.fake_judge.FAKE_JUDGE_MODEL` is the offline judge, for
    development mode only.
    """
    from pup_finetune.evaluation.fake_judge import FAKE_JUDGE_MODEL, fake_judge_config
    from pup_finetune.inference import JudgeConfig

    if judge_model is None and provider == "fake":
        judge_model = FAKE_JUDGE_MODEL
    if judge_model == FAKE_JUDGE_MODEL:
        if not runtime.dev_mode:
            raise UsageError(
                "the fake judge is for development only",
                hint="set PUP_FINETUNE_DEV=1, or pass another --judge-model",
            )
        return fake_judge_config()
    return JudgeConfig() if judge_model is None else JudgeConfig(model=judge_model)


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
) -> EvalPlan:
    """The evaluation of the run ``run_id`` on ``dataset``'s test split.

    Raises :class:`UsageError` when the test split has no case, or the judge is the model
    being trained.
    """
    from pup_finetune.evaluation.cases import cases_from_dataset
    from pup_finetune.evaluation.judge import check_judge_differs

    judge_config = resolve_judge(settings.judge_model, provider=provider, runtime=runtime)
    check_judge_differs(judge_config, profile.base_model)
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
    )


# --- the judge ----------------------------------------------------------------------------


@contextmanager
def open_judge_client(config: JudgeConfig) -> Iterator[ChatCompleter]:
    """The client for ``config``'s judge: the offline fake judge, or an OpenAI-compatible
    endpoint with the key of its credential. Closed on exit."""
    from pup_finetune.evaluation.fake_judge import FAKE_JUDGE_MODEL, FakeJudge
    from pup_finetune.inference import ChatClient

    if config.model == FAKE_JUDGE_MODEL:
        yield FakeJudge()
        return
    with ChatClient.for_judge(config) as client:
        yield client


def run_calibration(plan: EvalPlan, client: ChatCompleter) -> dict[str, Any] | None:
    """The calibration report of ``plan``'s judge, or ``None`` with ``--skip-calibration``."""
    from pup_finetune.evaluation.judge import calibrate

    if plan.settings.skip_calibration:
        return None
    info(f"calibrating the judge {plan.judge.model} on {plan.calibration_calls() // 2} cases")
    report = calibrate(plan.cases, client, plan.judge, n=CALIBRATION_CASES).to_json()
    return {**report, "judge": plan.judge.identity()}


def calibration_failed(plan: EvalPlan, report: Mapping[str, Any]) -> CommandResult:
    """What a command emits (exit 1) when calibration failed: nothing was generated."""
    return CommandResult(
        {
            "status": "calibration_failed",
            "run_id": plan.run_id,
            "out_dir": str(plan.out_dir),
            "judge_model": plan.judge.model,
            "calibration": dict(report),
            "hint": "the judge passed a corrupted action or failed a recorded one, so its "
            "verdicts cannot be trusted; pick another --judge-model (or rerun with "
            "--skip-calibration to evaluate anyway)",
        },
        exit_code=1,
    )


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
    """:func:`~pup_finetune.evaluation.replay.run_replay` of ``plan`` with ``client`` judging."""
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
        on_progress=_progress,
    )


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
    """
    summary = outcome.summary
    errors = sum(int(count) for count in summary["errors"].values())
    status = "partial"
    if outcome.complete:
        status = "complete_with_errors" if errors else "complete"
    warnings: list[str] = []
    hint = None
    if errors:
        hint = (
            f"{errors} case(s) ended in an error, left out of every rate (see results.jsonl "
            "in out_dir); run next_command to retry them"
        )
    no_verdict = sum(int(target["judge_no_verdict"]) for target in summary["targets"].values())
    if no_verdict:
        warnings.append(
            f"{no_verdict} judge answer(s) held no verdict (unparseable, or cut off at the "
            "judge's max_tokens) and count as failures; see judge_no_verdict per target"
        )
    return finite_json(
        {
            "status": status,
            "run_id": plan.run_id,
            "out_dir": str(outcome.out_dir),
            "summary": summary,
            "pending": outcome.pending,
            "cancelled": outcome.cancelled,
            "errors": errors,
            "warnings": warnings,
            "hint": hint,
            "calibration": None
            if calibration is None
            else {"ok": calibration.get("ok"), "cases": calibration.get("cases")},
            "tuned_checkpoint": None
            if tuned_sampler is None
            else dataclasses.asdict(tuned_sampler),
            "next_command": None if status == "complete" else shlex.join(plan.evaluate_command()),
        }
    )


def evaluation_exit_code(result: Mapping[str, Any]) -> int:
    """1 when an evaluation's result says it failed: it raised (``failed``), or every pair
    of a target ended in an error, so there is nothing to compare; otherwise 0."""
    if result.get("status") == "failed":
        return 1
    if result.get("status") != "complete_with_errors":
        return 0
    targets = result["summary"]["targets"].values()
    return 1 if any(int(target["completed"]) == 0 for target in targets) else 0


class EvaluateAfterTraining:
    """The ``after_finalize`` hook of ``train --evaluate``: replay in the training session.

    ``outcome`` is what to report: the evaluation's result, or why it was skipped or failed.
    A failure is also raised, for the run's warnings.
    """

    def __init__(
        self,
        plan: EvalPlan,
        client: ChatCompleter,
        *,
        calibration: Mapping[str, Any] | None,
        max_checkpoint_name_len: int,
    ) -> None:
        self.plan = plan
        self.client = client
        self.calibration = calibration
        self.max_checkpoint_name_len = max_checkpoint_name_len
        self.outcome: dict[str, Any] | None = None

    def __call__(
        self, session: TrainingSession, result: TrainResult, cancel: threading.Event
    ) -> None:
        from pup_finetune.evaluation.targets import best_training_checkpoint, session_samplers

        next_command = shlex.join(self.plan.evaluate_command())
        if result.stopped_reason not in EVALUATED_STOPS:
            why = {
                "cancelled": "it was cancelled",
                "budget_exhausted": "it reached its spend cap, which evaluating in its session "
                "would spend past",
            }.get(result.stopped_reason, f"it stopped: {result.stopped_reason}")
            self.outcome = {
                "status": "skipped",
                "reason": f"the run was not evaluated because {why}; next_command evaluates "
                "it in a session of its own",
                "next_command": next_command,
            }
            return
        try:
            training = best_training_checkpoint(result)
            report = prepare_eval_dir(
                self.plan, training, self.client, calibration=self.calibration
            )
            info(f"evaluating the run {self.plan.run_id} in its training session")
            with session_samplers(
                session,
                result,
                run_id=self.plan.run_id,
                max_checkpoint_name_len=self.max_checkpoint_name_len,
            ) as targets:
                outcome = replay(self.plan, targets.samplers, self.client, training, cancel=cancel)
            self.outcome = replay_result(
                self.plan, outcome, calibration=report, tuned_sampler=targets.tuned_checkpoint
            )
        except BaseException as exc:
            self.outcome = {
                "status": "failed",
                "error": error_json(exc),
                "out_dir": str(self.plan.out_dir),
                "next_command": next_command,
            }
            raise


# --- evaluate -----------------------------------------------------------------------------


def completed_result(run: RunDir) -> TrainResult:
    """The training result of a completed run; raises :class:`UsageError` for another phase."""
    state = run.load_state()
    if state.phase != "completed":
        raise UsageError(
            f"the run in {run.path} is {state.phase}, not completed, so there is nothing to "
            "evaluate yet",
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


RunDirArgument = Annotated[
    Path, typer.Argument(metavar="RUN_DIR", help="A completed run directory.", show_default=False)
]


@run_command
def evaluate(
    run_dir: RunDirArgument,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out-dir",
            help="Where the evaluation is written and resumed from (default: RUN_DIR/eval).",
            show_default=False,
        ),
    ] = None,
    max_cases_per_trajectory: Annotated[
        int,
        typer.Option(min=1, help="Most assistant turns replayed per test conversation."),
    ] = DEFAULT_MAX_CASES_PER_TRAJECTORY,
    judge_model: Annotated[
        str | None,
        typer.Option(
            "--judge-model",
            help="The judge model (default: a Fireworks serverless model; the offline fake "
            "judge for a fake run).",
            show_default=False,
        ),
    ] = None,
    concurrency: Annotated[
        int, typer.Option(min=1, max=64, help="Cases generated and judged at once.")
    ] = DEFAULT_CONCURRENCY,
    max_tokens: Annotated[
        int, typer.Option(min=1, help="Most tokens each generation may produce.")
    ] = DEFAULT_MAX_TOKENS,
    seed: Annotated[
        int, typer.Option(min=0, help="Seed for picking cases and the bootstrap.")
    ] = DEFAULT_SEED,
    skip_calibration: Annotated[
        bool,
        typer.Option("--skip-calibration", help="Do not check the judge on known answers first."),
    ] = False,
    confirm: Annotated[
        bool, typer.Option("--confirm", help="Run it (paid); without it, only preview the calls.")
    ] = False,
) -> dict[str, Any] | CommandResult:
    from pup_finetune.evaluation.targets import best_training_checkpoint

    runtime = get_runtime()
    run = RunDir(run_dir)
    manifest = run.manifest()
    result = completed_result(run)
    training = best_training_checkpoint(result)
    provider_name = str(manifest["provider"])
    provider = get_provider(provider_name, runtime)
    dataset = DatasetDir(Path(str(manifest["dataset_dir"])))
    _check_run_inputs(run, manifest, dataset)
    _check_renderer(run, manifest, provider)
    settings = EvalSettings(
        out_dir=out_dir if out_dir is not None else run_dir / DEFAULT_OUT_DIR_NAME,
        max_cases_per_trajectory=max_cases_per_trajectory,
        judge_model=judge_model,
        concurrency=concurrency,
        max_tokens=max_tokens,
        seed=seed,
        skip_calibration=skip_calibration,
    )
    plan = plan_evaluation(
        runtime,
        run_id=run.run_id,
        run_dir=run_dir,
        provider=provider.name,
        profile=run.profile(),
        dataset=dataset,
        rendering=manifest.get("rendering") or {},
        settings=settings,
    )
    tuned = f"epoch {result.best_epoch} of run {plan.run_id} ({training.name})"
    preview = plan.preview(tuned=tuned, tuned_checkpoint=training)
    preview["run_dir"] = str(run_dir)
    if confirm:
        require_writable(runtime, f"evaluating on {provider.name}")
    require_confirm(confirm, preview, shlex.join([*plan.evaluate_command(), "--confirm"]))

    cancel = threading.Event()
    with open_judge_client(plan.judge) as client:
        report = prepare_eval_dir(plan, training, client)
        if report is not None and not report.get("ok"):
            return calibration_failed(plan, report)
        try:
            with cancel_on_signals(cancel):
                data = _replay_run(plan, provider, run, result, training, client, cancel, report)
        except KeyboardInterrupt:
            raise FinetuneError(
                f"evaluation stopped at once (Ctrl-C); what finished is kept in {plan.out_dir}",
                hint=f"continue it with `{shlex.join(plan.evaluate_command())}`",
            ) from None
    code = evaluation_exit_code(data)
    return CommandResult(data, exit_code=code) if code else data


def _replay_run(
    plan: EvalPlan,
    provider: Provider,
    run: RunDir,
    result: TrainResult,
    training: CheckpointRef,
    client: ChatCompleter,
    cancel: threading.Event,
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from pup_finetune.evaluation.replay import EvalDir, remaining_work
    from pup_finetune.evaluation.targets import run_samplers

    if remaining_work(EvalDir(plan.out_dir), plan.cases)["generate"] == 0:
        info(f"nothing left to generate in {plan.out_dir}: judging and summarizing only")
        outcome = replay(plan, {}, client, training, cancel=cancel)
        return replay_result(plan, outcome, calibration=report, tuned_sampler=None)
    info(f"evaluating run {plan.run_id}: {len(plan.cases)} cases, base and tuned (Ctrl-C stops)")
    with run_samplers(provider, run, result) as targets:
        outcome = replay(plan, targets.samplers, client, training, cancel=cancel)
    return replay_result(plan, outcome, calibration=report, tuned_sampler=targets.tuned_checkpoint)


def register(app: typer.Typer) -> None:
    """Add `evaluate` to ``app``."""
    app.command(
        "evaluate",
        short_help="Replay held-out conversations with the base and tuned model (paid).",
        help=(
            "Replay every held-out test conversation of a completed run, one recorded "
            "assistant turn at a time, with the base model and the run's best checkpoint; "
            "score each answer with deterministic tool-call metrics and an LLM judge; and "
            "compare the two (pass rates, wins and regressions, and the tuned-minus-base "
            "delta with a bootstrap confidence interval). Tools are not executed, so the "
            "scores measure agreement with the recorded behavior, not task success. "
            "Without --confirm, previews the calls it would make. With --confirm, "
            "calibrates the judge first, then evaluates; rerun the same command to continue "
            "an evaluation that stopped. The preview is allowed with PUP_READ_ONLY set; "
            "evaluating is not."
        ),
    )(evaluate)
