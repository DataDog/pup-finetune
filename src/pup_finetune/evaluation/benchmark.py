# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Benchmark base models on a dataset's test split, to choose the one to fine-tune.

:func:`plan_benchmark` turns a dataset and the models to compare into a
:class:`BenchmarkPlan`: the same replay cases ``evaluate`` uses, the judge and the output
directory. :func:`execute` calibrates the judge once, then replays the cases with each base
model in turn (:func:`~pup_finetune.evaluation.replay.run_replay`, with the model as its one
target), in a session of its own that ends before the next model starts, into
``OUT/<model>``; a rerun resumes every model where it stopped. A model is named by its alias,
or ``PROVIDER/ALIAS`` when another provider offers that alias too (:attr:`Candidate.target`),
so one model can be benchmarked on both providers. A model whose session cannot
start is recorded as an error and the others still run.

Each model replays the cases it can (:func:`~pup_finetune.evaluation.cases.exclusions`) and
its directory is identified by those cases and its renderer, never by the other models, so
adding or dropping a model reuses every finished one. :func:`leaderboard` ranks the models on
the cases every ranked model has a result for, by judge pass rate, with a 95% bootstrap CI over
trajectories and each one's paired gap to the best; :func:`recommend` names the model to
fine-tune, or says the best base model already reaches ``--target-pass-rate``. Like
``evaluate``, the scores measure agreement with the recorded answers, not task success.
"""

from __future__ import annotations

import shlex
import statistics
import threading
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.display import display_path
from pup_finetune.errors import ExternalError, SettingsConflict, UsageError
from pup_finetune.evaluation.runner import (
    CALIBRATION_CASES,
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_CASES_PER_TRAJECTORY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SEED,
    calibration_line,
    cancel_on_signals,
    open_judge_client,
    resolve_judge,
    stored_judge,
)
from pup_finetune.next_steps import PROGRAM, command, step
from pup_finetune.output import info
from pup_finetune.progress import EvaluationProgress
from pup_finetune.providers import (
    get_provider,
    provider_info,
    providers_for_model,
    require_packages,
)
from pup_finetune.providers import require_key as require_provider_key
from pup_finetune.providers.base import (
    Generation,
    LoraConfig,
    ModelProfile,
    Sampler,
    SamplingUsage,
)
from pup_finetune.redaction import redact
from pup_finetune.rendering.base import Renderer, RendererError
from pup_finetune.runs import error_json
from pup_finetune.runtime import Runtime
from pup_finetune.storage import (
    DirectoryLock,
    atomic_write_json,
    canonical_json,
    finite_json,
    identity_changes,
)

if TYPE_CHECKING:
    from pup_finetune.evaluation.cases import Exclusion, ReplayCase
    from pup_finetune.evaluation.judge import Judgment
    from pup_finetune.inference import ChatCompleter, JudgeConfig

__all__ = [
    "DEFAULT_OUT_DIR_NAME",
    "DEFAULT_TARGET_PASS_RATE",
    "LOW_PASS_RATE",
    "BenchmarkPlan",
    "BenchmarkSettings",
    "Candidate",
    "execute",
    "exit_code",
    "leaderboard",
    "plan_benchmark",
    "recommend",
]

DEFAULT_OUT_DIR_NAME: Final = "benchmark"
DEFAULT_TARGET_PASS_RATE: Final = 0.9
LOW_PASS_RATE: Final = 0.25
"""Below this, the best base model is far from the recorded answers: a warning."""
MIN_TRAJECTORIES: Final = 30
"""Fewer test conversations than this is a warning: the intervals are too wide to choose."""
RESULT_FILE: Final = "benchmark.json"


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkSettings:
    """The benchmark flags, as given (``judge_model`` ``None`` for the default)."""

    out_dir: Path
    max_cases_per_trajectory: int = DEFAULT_MAX_CASES_PER_TRAJECTORY
    judge_model: str | None = None
    concurrency: int = DEFAULT_CONCURRENCY
    max_tokens: int = DEFAULT_MAX_TOKENS
    seed: int = DEFAULT_SEED
    skip_calibration: bool = False
    target_pass_rate: float = DEFAULT_TARGET_PASS_RATE


@dataclass(frozen=True, slots=True)
class Candidate:
    """One base model to benchmark: its provider, profile and renderer identity, and the
    test cases it cannot replay."""

    provider: str
    profile: ModelProfile
    rendering: Mapping[str, str]
    """Its renderer's :meth:`~pup_finetune.rendering.base.Renderer.identity`."""
    excluded: Mapping[str, Exclusion]
    """The test cases it cannot replay, by case id."""
    shared: bool = False
    """Whether another provider offers the profile's alias too."""

    @property
    def target(self) -> str:
        """How the benchmark names the model: its alias, or ``PROVIDER/ALIAS`` when another
        provider offers that alias too. It is the ``--models`` entry, the replay target, the
        model's directory under ``--out-dir`` and its name in the result."""
        return f"{self.provider}/{self.profile.alias}" if self.shared else self.profile.alias


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkPlan:
    """What a benchmark replays, with which models and judge, into which directory."""

    dataset_dir: Path
    candidates: tuple[Candidate, ...]
    cases: list[ReplayCase]
    judge: JudgeConfig
    judge_source: str
    """Where the judge came from (see :func:`~pup_finetune.evaluation.runner.resolve_judge`)."""
    settings: BenchmarkSettings
    runtime: Runtime

    @property
    def out_dir(self) -> Path:
        return self.settings.out_dir

    @property
    def offline(self) -> bool:
        """Whether nothing leaves the machine: the fake provider generates, the fake judge
        judges. ``PUP_READ_ONLY`` allows such a run."""
        return self.judge.provider == "fake" and all(
            candidate.provider == "fake" for candidate in self.candidates
        )

    @property
    def trajectories(self) -> int:
        return len({case.row_id for case in self.cases})

    def cases_for(self, candidate: Candidate) -> list[ReplayCase]:
        """The cases ``candidate`` replays: every test case it can."""
        return [case for case in self.cases if case.id not in candidate.excluded]

    def common_cases(self) -> list[ReplayCase]:
        """The cases every model replays: at most what the models are ranked on."""
        return [
            case
            for case in self.cases
            if not any(case.id in candidate.excluded for candidate in self.candidates)
        ]

    def model_dir(self, candidate: Candidate) -> Path:
        return self.out_dir / candidate.target

    def identity(self, candidate: Candidate) -> dict[str, Any]:
        """What ``OUT/<model>/evaluation.json`` records: everything that changes a generation
        or a score of that model."""
        from pup_finetune.evaluation.replay import METRICS_VERSION, cases_sha256

        cases = self.cases_for(candidate)
        return {
            "benchmark": True,
            "provider": candidate.provider,
            "base_model": candidate.profile.base_model,
            "rendering": dict(candidate.rendering),
            "judge": self.judge.identity(),
            "cases_sha256": cases_sha256(cases),
            "n_cases": len(cases),
            "max_tokens": self.settings.max_tokens,
            "metrics_version": METRICS_VERSION,
            "seed": self.settings.seed,
            "max_cases_per_trajectory": self.settings.max_cases_per_trajectory,
        }

    def remaining(self, candidate: Candidate) -> dict[str, int]:
        """What a run would still do for ``candidate``: ``{"done", "generate", "judge"}``.

        Raises :class:`SettingsConflict` when its directory holds work done under other
        settings, which ``--confirm`` would refuse.
        """
        from pup_finetune.evaluation.replay import EvalDir, remaining_work

        eval_dir = EvalDir(self.model_dir(candidate))
        stored = eval_dir.stored_identity()
        wanted = finite_json(self.identity(candidate))
        cases = self.cases_for(candidate)
        if stored is not None and canonical_json(stored) != canonical_json(wanted):
            if eval_dir.has_work():
                raise SettingsConflict(
                    f"{display_path(eval_dir.path)} holds a benchmark with other settings "
                    f"(changed: {', '.join(identity_changes(stored, wanted))}), so its "
                    "results cannot be reused",
                    hint="benchmark into a new directory: pass a new --out-dir",
                )
            return {"done": 0, "generate": len(cases), "judge": len(cases)}
        return remaining_work(eval_dir, {candidate.target: cases})[candidate.target]

    def split_warnings(self) -> list[str]:
        """A warning per model that leaves cases out, and one when the test split is too
        small for the intervals to separate models."""
        from pup_finetune.evaluation.cases import exclusion_warning

        found = (
            exclusion_warning(candidate.excluded, cases=len(self.cases), model=candidate.target)
            for candidate in self.candidates
        )
        warnings = [warning for warning in found if warning is not None]
        if self.trajectories < MIN_TRAJECTORIES:
            warnings.append(
                f"the test split holds only {self.trajectories} conversations, so the "
                "confidence intervals are wide: pull more rows for a firmer choice"
            )
        return warnings

    def judge_calls(self) -> int:
        """The judge calls a run would still make: calibration's and every candidate's.
        The preview counts them and execution opens the judge only when one is left."""
        remaining = sum(self.remaining(candidate)["judge"] for candidate in self.candidates)
        return remaining + self.calibration_calls()

    def calibration_calls(self) -> int:
        if self.settings.skip_calibration or self._calibrated():
            return 0
        return 3 * min(CALIBRATION_CASES, len(self.calibration_cases()))

    def calibration_cases(self) -> list[ReplayCase]:
        """The cases some model replays, which the judge is calibrated on."""
        return [
            case
            for case in self.cases
            if any(case.id not in candidate.excluded for candidate in self.candidates)
        ]

    def _calibrated(self) -> bool:
        from pup_finetune.evaluation.replay import EvalDir

        stored = EvalDir(self.out_dir).read_calibration() or {}
        return bool(stored.get("ok")) and canonical_json(stored.get("judge")) == canonical_json(
            finite_json(self.judge.identity())
        )

    def command(self) -> list[str]:
        """The ``benchmark`` command for this plan, without ``--confirm``: the models and the
        judge always, the other settings when they differ from the defaults."""
        settings = self.settings
        defaults = BenchmarkSettings(out_dir=self.dataset_dir / DEFAULT_OUT_DIR_NAME)
        words = [*PROGRAM, "benchmark", display_path(self.dataset_dir)]
        words += ["--models", ",".join(candidate.target for candidate in self.candidates)]
        if settings.out_dir != defaults.out_dir:
            words += ["--out-dir", display_path(settings.out_dir)]
        if settings.max_cases_per_trajectory != defaults.max_cases_per_trajectory:
            words += ["--max-cases", str(settings.max_cases_per_trajectory)]
        words += ["--judge-model", self.judge.spec]
        if settings.concurrency != defaults.concurrency:
            words += ["--concurrency", str(settings.concurrency)]
        if settings.max_tokens != defaults.max_tokens:
            words += ["--max-tokens", str(settings.max_tokens)]
        if settings.seed != defaults.seed:
            words += ["--seed", str(settings.seed)]
        if settings.target_pass_rate != defaults.target_pass_rate:
            words += ["--target-pass-rate", str(settings.target_pass_rate)]
        if settings.skip_calibration:
            words.append("--skip-calibration")
        return words

    def recipients(self) -> list[str]:
        from pup_finetune.judges import recipient

        judge = (
            "the offline fake judge (judge; nothing is sent)"
            if self.judge.provider == "fake"
            else f"{recipient(self.judge)} (judge)"
        )
        providers = sorted({candidate.provider for candidate in self.candidates})
        return [f"test split -> {', '.join(providers)} (generate), {judge}"]

    def preview(self) -> dict[str, Any]:
        """The calls the benchmark makes; work ``--out-dir`` already holds is left out."""
        from pup_finetune.evaluation.cases import excluded_by_reason
        from pup_finetune.evaluation.replay import CAVEAT
        from pup_finetune.judges import billing as judge_billing
        from pup_finetune.judges import destination, require_key

        settings = self.settings
        models: list[dict[str, Any]] = []
        generate = judge_calls = done = 0
        calls_by_provider: dict[str, int] = defaultdict(int)
        for candidate in self.candidates:
            remaining = self.remaining(candidate)
            generate += remaining["generate"]
            judge_calls += remaining["judge"]
            done += remaining["done"]
            calls_by_provider[candidate.provider] += remaining["generate"]
            models.append(
                {
                    "model": candidate.target,
                    "provider": candidate.provider,
                    "base_model": candidate.profile.base_model,
                    "cases": len(self.cases_for(candidate)),
                    "excluded_cases": len(candidate.excluded),
                    "excluded_by_reason": excluded_by_reason(candidate.excluded),
                    "already_done": remaining["done"],
                    "generation_calls": remaining["generate"],
                }
            )
        calibration = self.calibration_calls()
        warnings = self.split_warnings()
        warnings += [
            remote
            for candidate in self.candidates
            if (remote := candidate.profile.remote_code_warning()) is not None
        ]
        # --confirm would fail on a key nothing provides: fail now, never offer it. A
        # read-only preview offers none, so it only warns.
        soft = warnings if self.runtime.read_only else None
        for model in models:
            if model["generation_calls"]:
                needed_by = f"generating with {model['model']}"
                require_provider_key(
                    model["provider"], self.runtime, needed_by=needed_by, warnings=soft
                )
        if judge_calls + calibration:
            require_key(self.judge, warnings=soft)
        billing = [
            get_provider(name, self.runtime).sampling_billing(
                SamplingUsage(calls=calls, max_tokens_per_call=settings.max_tokens)
            )
            for name, calls in calls_by_provider.items()
        ]
        billing.append(judge_billing(self.judge, judge_calls + calibration))
        judge_name = "the fake judge" if self.judge.provider == "fake" else self.judge.spec
        judged_at = (
            "the offline fake judge (nothing is sent)"
            if self.judge.provider == "fake"
            else destination(self.judge)
        )
        names = ", ".join(candidate.target for candidate in self.candidates)
        return {
            "action": f"benchmark {_models(len(self.candidates))} ({names}) on "
            f"{len(self.cases)} cases from {self.trajectories} test conversations: generate "
            "each with every model that can replay it, one model at a time in a session of "
            f"its own, and judge each answer with {judge_name}",
            "billing": "; ".join(billing),
            "dataset": display_path(self.dataset_dir),
            "out_dir": display_path(self.out_dir),
            "rows": {"test": self.trajectories},
            "cases": len(self.cases),
            "common_cases": len(self.common_cases()),
            "trajectories": self.trajectories,
            "models": models,
            "judge_model": self.judge.spec,
            "judge_source": self.judge_source,
            "judge_provider": self.judge.provider,
            "judge_endpoint": self.judge.base_url,
            "already_done": done,
            "generation_calls": generate,
            "judge_calls": judge_calls + calibration,
            "calibration_calls": calibration,
            "max_generated_tokens": generate * settings.max_tokens,
            "settings": {
                "models": [candidate.target for candidate in self.candidates],
                "judge_model": self.judge.spec,
                "max_cases": settings.max_cases_per_trajectory,
                "max_tokens": settings.max_tokens,
                "concurrency": settings.concurrency,
                "seed": settings.seed,
                "calibration": not settings.skip_calibration,
                "target_pass_rate": settings.target_pass_rate,
            },
            "data_destination": "the test split's conversations are sent to "
            f"{', '.join(sorted(calls_by_provider))} to generate and to {judged_at} to judge",
            "data_recipients": self.recipients(),
            "caveat": CAVEAT,
            "warnings": warnings,
        }


def _models(count: int) -> str:
    return f"{count} base model{'' if count == 1 else 's'}"


def plan_benchmark(
    runtime: Runtime,
    dataset_dir: Path,
    models: Sequence[tuple[str, str]],
    settings: BenchmarkSettings,
) -> BenchmarkPlan:
    """The benchmark of ``models`` (``(alias, provider)`` pairs) on the test split of the
    dataset in ``dataset_dir``.

    Raises :class:`UsageError` for fewer than one model, a model given twice, a missing
    package, a judge that is one of the models, a test split with no case, a model that can
    replay none of its cases, or models that share none.
    """
    from pup_finetune.curation.pipeline import require_in_datadog
    from pup_finetune.evaluation.cases import build_cases, exclusions
    from pup_finetune.evaluation.judge import check_judge_differs

    dataset = DatasetDir(dataset_dir)
    by_split = dataset.rows_by_split()
    require_in_datadog(
        dataset,
        by_split,
        "benchmark",
        directory=dataset_dir,
        offline=all(provider_info(provider, runtime).offline for _, provider in models),
    )
    cases = build_cases(
        by_split["test"],
        max_cases_per_trajectory=settings.max_cases_per_trajectory,
        seed=settings.seed,
        split="test",
    )
    if not cases:
        rows = len(by_split["test"])
        raise UsageError(
            f"nothing to benchmark: the test split of {display_path(dataset.path)} has {rows} "
            "kept rows and no assistant turn to replay",
            hint="import more rows so the test split holds some conversations",
        )
    candidates: list[Candidate] = []
    for alias, provider_name in models:
        require_packages(provider_name, runtime, "evaluate")
        provider = get_provider(provider_name, runtime)
        profile = provider.resolve_model(alias)
        renderer = provider.renderer(profile)
        _resolve_stop_tokens(renderer)
        candidate = Candidate(
            provider=provider_name,
            profile=profile,
            rendering=renderer.identity(),
            excluded=exclusions(cases, renderer),
            shared=providers_for_model(profile.alias, runtime) != [provider_name],
        )
        if any(other.target == candidate.target for other in candidates):
            raise UsageError(f"--models names {candidate.target} twice")
        if len(candidate.excluded) == len(cases):
            first = next(iter(candidate.excluded.values()))
            raise UsageError(
                f"nothing to benchmark with {candidate.target}: it cannot replay any of the "
                f"{len(cases)} test cases ({first.code}: {first.detail})",
                hint="benchmark without that model",
            )
        candidates.append(candidate)
    if not candidates:
        raise UsageError("--models names no model", hint="pass --models MODEL,MODEL")
    offline = all(candidate.provider == "fake" for candidate in candidates)
    judge, judge_source = resolve_judge(
        settings.judge_model,
        provider="fake" if offline else candidates[0].provider,
        runtime=runtime,
        stored=_stored_judge(settings.out_dir, candidates),
    )
    for candidate in candidates:
        check_judge_differs(judge, candidate.profile.base_model)
    plan = BenchmarkPlan(
        dataset_dir=dataset_dir,
        candidates=tuple(candidates),
        cases=cases,
        judge=judge,
        judge_source=judge_source,
        settings=settings,
        runtime=runtime,
    )
    if not plan.common_cases():
        raise UsageError(
            "the models share no test case they can all replay, so they cannot be ranked on "
            "the same cases",
            hint="benchmark the models that leave cases out on their own",
        )
    return plan


def _resolve_stop_tokens(renderer: Renderer) -> None:
    """Resolve ``renderer``'s stop tokens, and with them its parser's control tokens, which
    raises for a family where one is not a single token of its tokenizer: the preview fails
    then, before any paid call, instead of the model's session."""
    renderer.stop_token_ids  # noqa: B018


def _stored_judge(out_dir: Path, candidates: Sequence[Candidate]) -> str | None:
    """The judge a benchmark in ``out_dir`` froze: the ``evaluation.json`` of a model with
    results. A rerun without ``--judge-model`` keeps it."""
    from pup_finetune.evaluation.replay import EvalDir

    for candidate in candidates:
        model_dir = EvalDir(out_dir / candidate.target)
        found = stored_judge(model_dir.stored_identity()) if model_dir.has_work() else None
        if found is not None:
            return found
    return None


# --- running ------------------------------------------------------------------------------


@contextmanager
def _base_sampler(plan: BenchmarkPlan, candidate: Candidate) -> Iterator[Sampler]:
    """A sampler of ``candidate``'s base model, in a session of its own; both end on exit.

    The session's context fits the longest case and its generation, not the profile's whole
    context: a provider provisions a smaller trainer for it, which starts sooner.
    """
    provider = get_provider(candidate.provider, plan.runtime)
    provider.validate_options({})
    profile = candidate.profile
    renderer = provider.renderer(profile)
    longest = max(_prompt_length(renderer, case) for case in plan.cases_for(candidate))
    with provider.backend().open_session(
        profile,
        LoraConfig(rank=profile.default_lora_rank),
        seed=plan.settings.seed,
        max_seq_len=min(profile.max_seq_len, longest + plan.settings.max_tokens),
    ) as session:
        sampler = session.sampler(None)
        try:
            yield sampler
        finally:
            try:
                sampler.close()
            except Exception as exc:  # its work is done: only note it
                info(f"closing the {candidate.target} sampler failed: {redact(str(exc))}")


def _prompt_length(renderer: Renderer, case: ReplayCase) -> int:
    """The tokens of ``case``'s prompt; ``0`` for one the renderer refuses, whose pair then
    records the refusal instead of stopping the benchmark."""
    try:
        prompt = renderer.generation_prompt_tokens(
            [dict(m) for m in case.prefix], [dict(t) for t in case.tools]
        )
    except RendererError:
        return 0
    return len(prompt)


def _calibration(plan: BenchmarkPlan, client: ChatCompleter) -> dict[str, Any] | None:
    """The judge's calibration report, reused from ``--out-dir`` when one passed there for
    this judge; ``None`` with ``--skip-calibration``."""
    from pup_finetune.evaluation.judge import calibrate
    from pup_finetune.evaluation.replay import EvalDir

    if plan.settings.skip_calibration:
        return None
    store = EvalDir(plan.out_dir)
    if plan._calibrated():
        return store.read_calibration()
    info(f"calibrating the judge {plan.judge.spec} on {plan.calibration_calls() // 3} cases")
    report = {
        **calibrate(plan.calibration_cases(), client, plan.judge, n=CALIBRATION_CASES).to_json(),
        "judge": plan.judge.identity(),
    }
    info(calibration_line(report))
    store.write_calibration(report)
    return report


def _replay_model(
    plan: BenchmarkPlan,
    candidate: Candidate,
    client: ChatCompleter,
    cancel: threading.Event,
) -> dict[str, Any] | None:
    """Replay every case with ``candidate``; the error that kept its session from starting,
    or ``None``."""
    from pup_finetune.evaluation.judge import judge
    from pup_finetune.evaluation.replay import run_replay

    remaining = plan.remaining(candidate)
    if not remaining["judge"]:
        return None
    config = plan.judge
    cases = plan.cases_for(candidate)

    def judge_fn(case: ReplayCase, generation: Generation) -> Judgment:
        return judge(
            case,
            generation.message,
            client,
            config,
            format_valid=generation.format_valid,
            cut_off=generation.stop_reason == "length",
        )

    def replay(samplers: Mapping[str, Sampler]) -> None:
        run_replay(
            cases,
            samplers,
            judge_fn,
            out_dir=plan.model_dir(candidate),
            identity=plan.identity(candidate),
            concurrency=plan.settings.concurrency,
            cancel=cancel,
            on_progress=EvaluationProgress(),
            replays={candidate.target: cases},
        )

    info(f"benchmarking {candidate.target} on {candidate.provider}")
    if not remaining["generate"]:
        replay({})
        return None
    try:
        with _base_sampler(plan, candidate) as sampler:
            replay({candidate.target: sampler})
    except ExternalError as exc:
        info(f"{candidate.target} could not be benchmarked: {redact(exc.message)}")
        return error_json(exc)
    return None


def execute(plan: BenchmarkPlan) -> dict[str, Any]:
    """Calibrate the judge, replay the cases with every model in turn, rank them and
    recommend one; writes ``benchmark.json`` and returns what the command emits."""
    from pup_finetune.evaluation.replay import EvalDir

    cancel = threading.Event()
    errors: dict[str, dict[str, Any]] = {}
    EvalDir(plan.out_dir).create()
    with DirectoryLock(plan.out_dir), cancel_on_signals(cancel):
        calls = plan.judge_calls()
        with open_judge_client(plan.judge, calls=calls) as client:
            calibration = _calibration(plan, client)
            if calibration is not None and not calibration.get("ok"):
                return _calibration_failed(plan, calibration)
            for candidate in plan.candidates:
                if cancel.is_set():
                    break
                error = _replay_model(plan, candidate, client, cancel)
                if error is not None:
                    errors[candidate.target] = error
    board = leaderboard(plan, errors)
    result = _result(plan, board, calibration=calibration, cancelled=cancel.is_set())
    atomic_write_json(plan.out_dir / RESULT_FILE, finite_json(result))
    return result


def _calibration_failed(plan: BenchmarkPlan, report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "calibration_failed",
        "out_dir": display_path(plan.out_dir),
        "judge_model": plan.judge.spec,
        "calibration": dict(report),
        "hint": "the judge passed a corrupted action or failed a recorded one, so its "
        "verdicts cannot be trusted; pick another --judge-model (or rerun with "
        "--skip-calibration to benchmark anyway)",
        "next_command": None,
    }


# --- ranking ------------------------------------------------------------------------------


def _passed(result: Mapping[str, Any]) -> bool:
    judgment = result.get("judgment")
    return bool(judgment.get("passed")) if isinstance(judgment, Mapping) else False


def _ok_passes(plan: BenchmarkPlan, candidate: Candidate) -> dict[str, bool]:
    """Whether each case with an ``ok`` result passed the judge, by case id."""
    from pup_finetune.evaluation.replay import EvalDir

    results = EvalDir(plan.model_dir(candidate)).read_results()
    found: dict[str, bool] = {}
    for case in plan.cases_for(candidate):
        result = results.get((candidate.target, case.id))
        if result is not None and result.get("status") == "ok":
            found[case.id] = _passed(result)
    return found


def _passes(plan: BenchmarkPlan) -> dict[str, dict[str, bool]]:
    return {candidate.target: _ok_passes(plan, candidate) for candidate in plan.candidates}


def _ranked_on(passes: Mapping[str, Mapping[str, bool]]) -> set[str]:
    """The cases every model with a judged case has an ``ok`` result for: what the models are
    ranked on, so each is compared on the same cases."""
    judged = [set(found) for found in passes.values() if found]
    return set.intersection(*judged) if judged else set()


def _by_trajectory(cases: Sequence[ReplayCase], values: Mapping[str, float]) -> list[list[float]]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        if case.id in values:
            clusters[case.row_id].append(values[case.id])
    return list(clusters.values())


def _entry(
    plan: BenchmarkPlan,
    candidate: Candidate,
    error: Mapping[str, Any] | None,
    passes: Mapping[str, bool],
) -> dict[str, Any]:
    from pup_finetune.evaluation.bootstrap import paired_bootstrap_ci
    from pup_finetune.evaluation.cases import excluded_by_reason
    from pup_finetune.evaluation.replay import EvalDir, summarize

    eval_dir = EvalDir(plan.model_dir(candidate))
    results, generations = eval_dir.read_results(), eval_dir.read_generations()
    cases = plan.cases_for(candidate)
    target = summarize(
        cases, results, generations, seed=plan.settings.seed, replays={candidate.target: cases}
    )["targets"][candidate.target]
    interval = paired_bootstrap_ci(
        _by_trajectory(cases, {key: float(value) for key, value in passes.items()}),
        seed=plan.settings.seed,
    )
    latencies = [
        float(record["latency_s"])
        for (name, _), record in generations.items()
        if name == candidate.target and isinstance(record.get("latency_s"), int | float)
    ]
    if error is not None:
        status = "error"
    elif target["pending"]:
        status = "partial"
    else:
        status = "complete_with_errors" if target["errors"] else "complete"
    return {
        "model": candidate.target,
        "provider": candidate.provider,
        "base_model": candidate.profile.base_model,
        "status": status,
        "judge_pass_rate": sum(passes.values()) / len(passes) if passes else None,
        "ci95": None if interval is None else list(interval),
        "cases": target["completed"],
        "excluded_cases": len(candidate.excluded),
        "excluded_by_reason": excluded_by_reason(candidate.excluded),
        "errors": target["errors"],
        "errors_unrenderable": target["errors_unrenderable"],
        "pending": target["pending"],
        "latency_p50_s": statistics.median(latencies) if latencies else None,
        "gap_to_best": None,
        "within_noise_of_best": None,
        "error": None if error is None else dict(error),
        "out_dir": display_path(eval_dir.path),
    }


def leaderboard(
    plan: BenchmarkPlan, errors: Mapping[str, Mapping[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Every model, best judge pass rate first (models without one last, in the order
    given). The rates, their 95% bootstrap CIs over trajectories and each model's gap to the
    best (the mean paired difference, with its CI) are over the same cases: those every model
    with a judged case has an ``ok`` result for. A model is within noise of the best when that
    CI reaches 0 or there are too few cases for one."""
    from pup_finetune.evaluation.bootstrap import paired_bootstrap_ci

    errors = errors or {}
    passes = _passes(plan)
    common = _ranked_on(passes)
    shared = {
        target: {case_id: passed for case_id, passed in found.items() if case_id in common}
        for target, found in passes.items()
    }
    entries = [
        _entry(plan, candidate, errors.get(candidate.target), shared[candidate.target])
        for candidate in plan.candidates
    ]
    order = sorted(
        range(len(entries)),
        key=lambda i: (
            entries[i]["judge_pass_rate"] is None,
            -(entries[i]["judge_pass_rate"] or 0),
        ),
    )
    board = [entries[i] for i in order]
    if not board or board[0]["judge_pass_rate"] is None:
        return board
    best = shared[board[0]["model"]]
    for entry in board[1:]:
        if entry["judge_pass_rate"] is None:
            continue
        other = shared[entry["model"]]
        differences = {case_id: float(best[case_id]) - float(other[case_id]) for case_id in common}
        interval = paired_bootstrap_ci(
            _by_trajectory(plan.cases, differences), seed=plan.settings.seed
        )
        entry["gap_to_best"] = {
            "judge_pass_rate": sum(differences.values()) / len(differences),
            "ci95": None if interval is None else list(interval),
            "n_cases": len(differences),
        }
        entry["within_noise_of_best"] = interval is None or interval[0] <= 0
    return board


def _pct(rate: float) -> str:
    return f"{round(rate * 100)}%"


def recommend(board: Sequence[Mapping[str, Any]], *, target_pass_rate: float) -> dict[str, Any]:
    """The model to fine-tune, from a :func:`leaderboard`.

    ``verdict`` is ``fine_tune`` (the best model is below ``target_pass_rate``: fine-tune
    it), ``base_meets_target`` (the best base model already reaches it, so fine-tuning may
    add little) or ``no_result`` (no model has a judged case). Models within noise of the
    best are named too: any of them is as good a start, so the cheapest to serve wins.
    """
    scored = [entry for entry in board if entry.get("judge_pass_rate") is not None]
    if not scored:
        return {
            "verdict": "no_result",
            "model": None,
            "reason": "no model has a judged case yet, so there is nothing to compare",
        }
    best = scored[0]
    rate = float(best["judge_pass_rate"])
    close = [str(entry["model"]) for entry in scored[1:] if entry.get("within_noise_of_best")]
    if rate >= target_pass_rate:
        verdict = "base_meets_target"
        reason = (
            f"{best['model']} already matches the recorded answers on {_pct(rate)} of the "
            f"cases, at or above the {_pct(target_pass_rate)} target, so fine-tuning may add "
            "little: check whether the remaining misses matter before training"
        )
    else:
        verdict = "fine_tune"
        reason = (
            f"{best['model']} matches the recorded answers best ({_pct(rate)} of the cases), "
            f"below the {_pct(target_pass_rate)} target: fine-tune it"
        )
    if close:
        reason += (
            f"; {', '.join(close)} {'is' if len(close) == 1 else 'are'} within noise of it, "
            "so the cheapest to serve is as good a start"
        )
    return {
        "verdict": verdict,
        "model": best["model"],
        "provider": best["provider"],
        "judge_pass_rate": rate,
        "target_pass_rate": target_pass_rate,
        "within_noise": close,
        "reason": reason,
    }


def _result(
    plan: BenchmarkPlan,
    board: list[dict[str, Any]],
    *,
    calibration: Mapping[str, Any] | None,
    cancelled: bool,
) -> dict[str, Any]:
    from pup_finetune.evaluation.replay import CAVEAT

    recommendation = recommend(board, target_pass_rate=plan.settings.target_pass_rate)
    statuses = {entry["status"] for entry in board}
    if cancelled or "partial" in statuses:
        status = "partial"
    elif statuses == {"complete"}:
        status = "complete"
    else:
        status = "complete_with_errors"
    warnings: list[str] = []
    hint = None
    unrenderable = sum(int(entry["errors_unrenderable"]) for entry in board)
    failed = [
        entry["model"]
        for entry in board
        if entry["error"] is not None
        or entry["pending"]
        or entry["errors"] > entry["errors_unrenderable"]
    ]
    if failed:
        hint = (
            f"{', '.join(failed)} {'has' if len(failed) == 1 else 'have'} no result for every "
            "case (see each entry's errors, error and pending); run next_command to continue, "
            "which reuses every finished result"
        )
    if unrenderable:
        hint = (f"{hint}. " if hint else "") + (
            f"{unrenderable} case(s) have a prompt a model's chat format cannot continue, "
            "which a retry cannot fix, so next_command never retries them"
        )
    rate = recommendation.get("judge_pass_rate")
    if isinstance(rate, float) and rate < LOW_PASS_RATE:
        warnings.append(
            f"the best base model matches only {_pct(rate)} of the recorded answers: "
            "fine-tuning may not close a gap this wide, so add rows or try a larger base "
            "model too"
        )
    warnings += plan.split_warnings()
    return finite_json(
        {
            "status": status,
            "recommendation": recommendation,
            "leaderboard": board,
            "dataset": display_path(plan.dataset_dir),
            "out_dir": display_path(plan.out_dir),
            "judge_model": plan.judge.spec,
            "cases": len(plan.cases),
            "common_cases": len(_ranked_on(_passes(plan))),
            "trajectories": plan.trajectories,
            "cancelled": cancelled,
            "calibration": None
            if calibration is None
            else {"ok": calibration.get("ok"), "cases": calibration.get("cases")},
            "caveat": CAVEAT,
            "warnings": warnings,
            "hint": hint,
            "files": {
                "result": display_path(plan.out_dir / RESULT_FILE),
                "models": {entry["model"]: entry["out_dir"] for entry in board},
            },
            **_next(plan, recommendation, retry=bool(failed)),
        }
    )


def _next(plan: BenchmarkPlan, recommendation: Mapping[str, Any], *, retry: bool) -> dict[str, Any]:
    """An unfinished benchmark that a rerun can advance is continued (a free preview first).
    A finished one leads to the training preview of the recommended model, which is a step
    instead when the base model already meets the target."""
    rerun = shlex.join(plan.command())
    model = recommendation.get("model")
    chosen = next((c for c in plan.candidates if c.target == model), None)
    train = (
        None
        if chosen is None
        else command(
            "train",
            "--provider",
            chosen.provider,
            "--dataset",
            plan.dataset_dir,
            "--model",
            chosen.profile.alias,
        )
    )
    if retry:
        return {"next_command": rerun, "next_steps": []}
    if train is None:
        return {"next_command": None, "next_steps": []}
    if recommendation.get("verdict") == "fine_tune":
        return {"next_command": train, "next_steps": []}
    return {
        "next_command": None,
        "next_steps": [
            step(train, "fine-tune the best base model anyway (free preview; training is paid)")
        ],
    }


def exit_code(result: Mapping[str, Any]) -> int:
    """1 when the judge failed calibration, or a benchmark that ran every model left one
    with no result at all (its session failed, or every case ended in an error); else 0."""
    if result.get("status") == "calibration_failed":
        return 1
    if result.get("status") != "complete_with_errors":
        return 0
    return 1 if any(entry.get("cases") == 0 for entry in result.get("leaderboard") or []) else 0
