# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Benchmark base models on a dataset's test split, to choose the one to fine-tune.

:func:`plan_benchmark` turns a dataset and the models to compare into a
:class:`BenchmarkPlan`: the same replay cases ``evaluate`` uses (with ``--max-rows``, those of
a seeded sample of the test conversations), the judge and the output directory.
:func:`execute` calibrates the judge once, then replays the cases with each base
model in turn (:func:`~pup_finetune.evaluation.replay.run_replay`, with the model as its one
target), in a session of its own that ends before the next model starts, into
``OUT/<model>``; a rerun resumes every model where it stopped. A model is named by its alias,
or ``PROVIDER/ALIAS`` when another provider offers that alias too (:attr:`Candidate.target`),
so one model can be benchmarked on both providers. A model whose session cannot start, or
whose renderer cannot load, is recorded as an error and the others still run. Base answers
and verdicts are shared with the dataset's other evaluations and benchmarks
(:mod:`~pup_finetune.evaluation.reuse`): a model with every answer reused opens no session.

Each model replays the cases it can (:func:`~pup_finetune.evaluation.cases.exclusions`) and
its directory is identified by those cases and its renderer, never by the other models, so
adding or dropping a model reuses every finished one. With ``--evaluators``, each model's
directory scores the user's own judges and checks too
(:mod:`~pup_finetune.evaluation.evaluators`), and their headline ranks the models.

:func:`leaderboard` gives each model its own headline pass rate (the judge's by default), with
a 95% bootstrap CI over trajectories, and ranks the finished ones on the cases the headline
scored for every one of them, with each one's paired gap to the best; a model still running is
listed unranked. :func:`recommend` names the model to fine-tune, or says the best base model
already reaches ``--target-pass-rate`` on its own cases. Like ``evaluate``, the scores measure
agreement with the recorded answers, not task success.
"""

from __future__ import annotations

import hashlib
import shlex
import statistics
import threading
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.display import display_path, joined
from pup_finetune.errors import ExternalError, SettingsConflict, UsageError
from pup_finetune.evaluation.runner import (
    CALIBRATION_CASES,
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_CASES_PER_TRAJECTORY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SEED,
    EvalWork,
    busy_judges,
    calibrate_evaluators,
    calibration_line,
    cancel_on_signals,
    check_hint,
    evaluator_calibration_calls,
    evaluator_errors,
    evaluator_fix,
    evaluators_preview,
    no_reuse_hint,
    open_evaluator_clients,
    open_judge_client,
    resolve_judge,
    scoring_warning,
    stored_judge,
)
from pup_finetune.next_steps import PROGRAM, command, step, template
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
    from pup_finetune.evaluation.evaluators import EvaluatorSet
    from pup_finetune.evaluation.judge import Judgment
    from pup_finetune.evaluation.replay import EvalDir, PairKey
    from pup_finetune.evaluation.reuse import ReuseIndex, TargetReuse
    from pup_finetune.inference import ChatCompleter, JudgeConfig

__all__ = [
    "DEFAULT_OUT_DIR_NAME",
    "DEFAULT_TARGET_PASS_RATE",
    "LOW_PASS_RATE",
    "TARGET_COVERAGE",
    "BenchmarkPlan",
    "BenchmarkSettings",
    "Candidate",
    "Leaderboard",
    "execute",
    "exit_code",
    "leaderboard",
    "plan_benchmark",
    "recommend",
    "sample_conversations",
]

DEFAULT_OUT_DIR_NAME: Final = "benchmark"
DEFAULT_TARGET_PASS_RATE: Final = 0.9
LOW_PASS_RATE: Final = 0.25
"""Below this, the best base model is far from the recorded answers: a warning."""
MIN_TRAJECTORIES: Final = 30
"""Fewer test conversations than this is a warning: the intervals are too wide to choose."""
RESULT_FILE: Final = "benchmark.json"
TARGET_COVERAGE: Final = 0.9
"""The share of the test cases with a prompt that the cases a model's headline scored must
cover before its own headline pass rate can show it meets ``--target-pass-rate``: a rate on
fewer says too little about the rest of the agent's conversations to call fine-tuning
unneeded."""


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkSettings:
    """The benchmark flags, as given (``judge_model`` ``None`` for the default)."""

    out_dir: Path
    max_cases_per_trajectory: int = DEFAULT_MAX_CASES_PER_TRAJECTORY
    max_rows: int | None = None
    judge_model: str | None = None
    concurrency: int = DEFAULT_CONCURRENCY
    max_tokens: int = DEFAULT_MAX_TOKENS
    seed: int = DEFAULT_SEED
    skip_calibration: bool = False
    target_pass_rate: float = DEFAULT_TARGET_PASS_RATE
    reuse: bool = True
    evaluators: Path | None = None


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
    unavailable: ExternalError | None = None
    """Why its renderer could not load, which its run records as its error. Its identity
    needs the renderer, so its directory is not checked against it and its results are not
    scored; planning still reads the judge and evaluators it froze, and which of them have
    work left, as for every model. Its :attr:`excluded` are only the cases with no prompt."""

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
    sampled_from: int | None = None
    """How many test conversations ``--max-rows`` sampled from; ``None`` when it cut none."""
    inputs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    """What each model's answers are sampled under, by model, from its full renderer
    identity; read only when the store exists, and not for a model whose renderer cannot
    load."""
    keys: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    """Each model's answer key per case in the dataset's store, by model and case id, from
    :attr:`inputs`."""
    reuse: ReuseIndex | None = None
    """What the dataset's store holds for this plan, read once when it was planned; ``None``
    with ``--no-reuse``."""
    evaluators: EvaluatorSet | None = None
    """The user's own evaluators: ``--evaluators``, else those the models' directories have
    attached."""

    @property
    def out_dir(self) -> Path:
        return self.settings.out_dir

    @property
    def offline(self) -> bool:
        """Whether nothing leaves the machine: the fake provider generates, the fake judge
        judges. ``PUP_READ_ONLY`` allows such a run."""
        return (
            self.judge.provider == "fake"
            and all(candidate.provider == "fake" for candidate in self.candidates)
            and (self.evaluators is None or self.evaluators.offline)
        )

    @property
    def headline(self) -> str:
        """What ranks the models: the built-in judge, an evaluator or ``all``."""
        return "judge" if self.evaluators is None else self.evaluators.headline

    @property
    def evaluator_names(self) -> list[str]:
        return [] if self.evaluators is None else self.evaluators.names

    @property
    def headline_evaluators(self) -> list[str]:
        """The evaluators the headline needs: none for the judge, every one for ``all``."""
        from pup_finetune.evaluation.evaluators import ALL, JUDGE

        if self.headline == JUDGE:
            return []
        return self.evaluator_names if self.headline == ALL else [self.headline]

    @property
    def trajectories(self) -> int:
        return len({case.row_id for case in self.cases})

    def cases_for(self, candidate: Candidate) -> list[ReplayCase]:
        """The cases ``candidate`` replays: every test case it can."""
        return [case for case in self.cases if case.id not in candidate.excluded]

    def replayed_by_every_model(self) -> list[ReplayCase]:
        """The cases every model replays: the most the finished models can be compared on."""
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
            # Only when it sampled, so one that cut nothing resumes without --max-rows.
            **({} if self.sampled_from is None else {"max_rows": self.settings.max_rows}),
            **({} if self.settings.reuse else {"reuse": False}),
        }

    def _matching_dir(self, candidate: Candidate) -> EvalDir | None:
        """``candidate``'s directory when what it holds counts for this plan; ``None`` when
        ``--confirm`` would start it again (another identity, no work), or when its renderer
        cannot load, so its identity cannot be checked.

        Raises :class:`SettingsConflict` when it holds work done under other settings, or
        evaluators these drop or change, which ``--confirm`` would refuse.
        """
        from pup_finetune.evaluation.evaluators import evaluators_conflict, stored_evaluators
        from pup_finetune.evaluation.replay import EvalDir

        if candidate.unavailable is not None:
            return None
        eval_dir = EvalDir(self.model_dir(candidate))
        stored = eval_dir.stored_identity()
        wanted = finite_json(self.identity(candidate))
        if stored is None or canonical_json(stored) == canonical_json(wanted):
            if self.evaluators is not None and (
                conflict := evaluators_conflict(
                    stored_evaluators(eval_dir.evaluators_path),
                    self.evaluators,
                    out_dir=eval_dir.path,
                    flag_dir=self.out_dir,
                )
            ):
                raise conflict
            return eval_dir
        if eval_dir.has_work():
            raise SettingsConflict(
                f"{display_path(eval_dir.path)} holds a benchmark with other settings "
                f"(changed: {', '.join(identity_changes(stored, wanted))}), so its "
                "results cannot be reused",
                hint=f"{no_reuse_hint(stored, wanted)}benchmark into a new directory: pass a "
                "new --out-dir",
            )
        return None

    def reused(self, candidate: Candidate) -> TargetReuse | None:
        """What ``candidate`` takes from the dataset's store for the pairs its directory has
        not done; ``None`` with ``--no-reuse``."""
        from pup_finetune.evaluation.reuse import plan_target

        if self.reuse is None:
            return None
        eval_dir = self._matching_dir(candidate)
        return plan_target(
            self.reuse,
            candidate.target,
            self.cases_for(candidate),
            self.keys.get(candidate.target, {}),
            self.judge,
            {} if eval_dir is None else eval_dir.read_generations(),
            {} if eval_dir is None else eval_dir.read_results(),
        )

    def remaining(self, candidate: Candidate) -> dict[str, Any]:
        """What a run would still do for ``candidate``: ``{"done", "generate", "judge"}``,
        less what it takes from the dataset's store, and with evaluators the cases each
        still scores (``"evaluators": {name: n}``).

        Raises :class:`SettingsConflict` when its directory holds work done under other
        settings, which ``--confirm`` would refuse.
        """
        from pup_finetune.evaluation.replay import remaining_work

        eval_dir = self._matching_dir(candidate)
        names = self.evaluator_names
        cases = self.cases_for(candidate)
        if eval_dir is None:
            found: dict[str, Any] = {
                "done": 0,
                "generate": len(cases),
                "judge": len(cases),
                **({"evaluators": dict.fromkeys(names, len(cases))} if names else {}),
            }
        else:
            work = remaining_work(eval_dir, {candidate.target: cases}, names)
            found = work[candidate.target]
        taken = self.reused(candidate)
        if taken is not None:
            found["generate"] -= len(taken.generations)
            found["judge"] -= len(taken.verdicts)
        return found

    def split_warnings(self) -> list[str]:
        """A warning per model that leaves cases out, and one when the test split (or the
        ``--max-rows`` sample of it) is too small for the intervals to separate models."""
        from pup_finetune.evaluation.cases import exclusion_warning

        found = (
            exclusion_warning(candidate.excluded, cases=len(self.cases), model=candidate.target)
            for candidate in self.candidates
        )
        warnings = [warning for warning in found if warning is not None]
        if self.trajectories >= MIN_TRAJECTORIES:
            return warnings
        if self.sampled_from is not None:
            warnings.append(
                f"--max-rows {self.settings.max_rows} replays only {self.trajectories} of the "
                f"{self.sampled_from} test conversations, so the confidence intervals are wide: "
                "raise --max-rows for a firmer choice"
            )
        else:
            warnings.append(
                f"the test split holds only {self.trajectories} conversations, so the "
                "confidence intervals are wide: pull more rows for a firmer choice"
            )
        return warnings

    def judge_calls(self) -> int:
        """The judge calls a run would still make: calibration's and every candidate's.
        The preview counts them and execution opens the judge only when one is left."""
        remaining = sum(int(self.remaining(c)["judge"]) for c in self.candidates)
        return remaining + self.calibration_calls()

    def work(self) -> EvalWork:
        """What the user's evaluators still do: each candidate's cases to score, and each
        judge's example calls to calibrate in ``--out-dir``."""
        from pup_finetune.evaluation.replay import EvalDir

        return EvalWork(
            by_target={c.target: self.remaining(c) for c in self.candidates},
            calibration=0,
            probe=False,
            evaluator_calibration=evaluator_calibration_calls(
                self.evaluators, EvalDir(self.out_dir), skip=self.settings.skip_calibration
            ),
        )

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

    def command(self, *, evaluators: bool = True) -> list[str]:
        """The ``benchmark`` command for this plan, without ``--confirm``: the models and the
        judge always, the other settings when they differ from the defaults. Without
        ``evaluators`` it leaves out ``--evaluators``, which a run attached already."""
        settings = self.settings
        defaults = BenchmarkSettings(out_dir=self.dataset_dir / DEFAULT_OUT_DIR_NAME)
        words = [*PROGRAM, "benchmark", display_path(self.dataset_dir)]
        words += ["--models", ",".join(candidate.target for candidate in self.candidates)]
        if settings.out_dir != defaults.out_dir:
            words += ["--out-dir", display_path(settings.out_dir)]
        if settings.max_cases_per_trajectory != defaults.max_cases_per_trajectory:
            words += ["--max-cases", str(settings.max_cases_per_trajectory)]
        if settings.max_rows is not None:
            words += ["--max-rows", str(settings.max_rows)]
        words += ["--judge-model", self.judge.spec]
        spec = None if self.evaluators is None or not evaluators else self.evaluators.pinned
        if spec is not None and spec.is_file():
            words += ["--evaluators", display_path(spec)]
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
        if not settings.reuse:
            words.append("--no-reuse")
        return words

    def recipients(self, providers: Sequence[str]) -> list[str]:
        """Who receives the test split: the ``providers`` of the models with a case left to
        generate, and the judge."""
        from pup_finetune.judges import recipient

        judge = (
            "the offline fake judge (judge; nothing is sent)"
            if self.judge.provider == "fake"
            else f"{recipient(self.judge)} (judge)"
        )
        generate = f"{', '.join(providers)} (generate), " if providers else ""
        return [f"test split -> {generate}{judge}"]

    def preview(self) -> dict[str, Any]:
        """The calls the benchmark makes; work ``--out-dir`` already holds is left out."""
        from pup_finetune.evaluation.cases import excluded_by_reason
        from pup_finetune.evaluation.evaluators import stored_evaluators
        from pup_finetune.evaluation.replay import CAVEAT, EvalDir
        from pup_finetune.judges import billing as judge_billing
        from pup_finetune.judges import destination, require_key

        settings = self.settings
        models: list[dict[str, Any]] = []
        generate = judge_calls = done = 0
        calls_by_provider: dict[str, int] = defaultdict(int)
        names: list[str] = []
        taken: list[tuple[TargetReuse, int]] = []
        for candidate in self.candidates:
            remaining = self.remaining(candidate)
            generate += remaining["generate"]
            judge_calls += remaining["judge"]
            done += remaining["done"]
            if remaining["generate"]:
                calls_by_provider[candidate.provider] += remaining["generate"]
            reused = self.reused(candidate)
            if reused is not None:
                taken.append((reused, remaining["generate"]))
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
                    "reused": {"generations": 0, "verdicts": 0}
                    if reused is None
                    else reused.counts(),
                }
            )
            idle = "" if remaining["generate"] else " (no session for it)"
            names.append(f"{candidate.target}{idle}")
        calibration = self.calibration_calls()
        warnings = [*self.split_warnings(), *(() if self.reuse is None else self.reuse.warnings)]
        for candidate in self.candidates:
            if (remote := candidate.profile.remote_code_warning()) is not None:
                warnings.append(remote)
            if candidate.unavailable is not None:
                warnings.append(
                    f"the renderer of {candidate.target} cannot load "
                    f"({redact(candidate.unavailable.message)}), so --confirm records it as an "
                    "error unless it loads by then; the other models still run"
                )
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
        evaluated = None
        scoring: list[str] = []
        if (evaluators := self.evaluators) is not None:
            work = self.work()
            scoring = [name for name in evaluators.names if work.evaluator_calls(name)]
            for judge in busy_judges(evaluators, work):
                require_key(judge.config, what=f"evaluator {judge.name}", warnings=soft)
            attached = {
                str(item.get("name"))
                for candidate in self.candidates
                for item in (
                    stored_evaluators(EvalDir(self.model_dir(candidate)).evaluators_path) or {}
                ).get("evaluators", [])
            }
            reports = (EvalDir(self.out_dir).read_calibration() or {}).get("evaluators") or {}
            evaluated = evaluators_preview(
                evaluators,
                work,
                self.cases,
                attached=attached,
                reports=reports,
                calibrate=not settings.skip_calibration,
            )
            billing += evaluated.billing
            warnings += evaluated.warnings
        judge_name = "the fake judge" if self.judge.provider == "fake" else self.judge.spec
        judged_at = (
            "the offline fake judge (nothing is sent)"
            if self.judge.provider == "fake"
            else destination(self.judge)
        )
        sampled = "" if self.sampled_from is None else f" (of {self.sampled_from}, --max-rows)"
        providers = sorted(calls_by_provider)
        sent = f"{', '.join(providers)} to generate and to " if providers else ""
        return {
            "action": f"benchmark {_models(len(self.candidates))} ({', '.join(names)}) on "
            f"{len(self.cases)} cases from {self.trajectories} test conversations{sampled}: "
            + (
                "generate each with every model that can replay it, one model at a time in a "
                f"session of its own, and judge each answer with {judge_name}"
                if generate
                else "every answer is already generated, so no session opens; judge any answer "
                f"left with {judge_name}"
            )
            + (f", and score each with {joined(scoring)}" if scoring else ""),
            "billing": "; ".join(billing),
            "dataset": display_path(self.dataset_dir),
            "out_dir": display_path(self.out_dir),
            "rows": {
                "test": self.trajectories,
                "test_total": self.sampled_from or self.trajectories,
            },
            "cases": len(self.cases),
            "replayed_by_every_model": len(self.replayed_by_every_model()),
            "trajectories": self.trajectories,
            "models": models,
            "judge_model": self.judge.spec,
            "judge_source": self.judge_source,
            "judge_provider": self.judge.provider,
            "judge_endpoint": self.judge.base_url,
            **({} if evaluated is None else {"evaluators": evaluated.block}),
            "already_done": done,
            "generation_calls": generate,
            "judge_calls": judge_calls + calibration,
            "calibration_calls": calibration,
            "max_generated_tokens": generate * settings.max_tokens,
            "reuse": self._reuse_block(taken),
            "settings": {
                "models": [candidate.target for candidate in self.candidates],
                "judge_model": self.judge.spec,
                "max_cases": settings.max_cases_per_trajectory,
                "max_rows": settings.max_rows,
                "max_tokens": settings.max_tokens,
                "concurrency": settings.concurrency,
                "seed": settings.seed,
                "calibration": not settings.skip_calibration,
                "target_pass_rate": settings.target_pass_rate,
                "reuse": settings.reuse,
                **(
                    {}
                    if evaluated is None
                    else {"evaluators": evaluated.block["spec"], "headline": self.headline}
                ),
            },
            "data_destination": f"the test split's conversations are sent to {sent}{judged_at} "
            "to judge",
            "data_recipients": [
                *self.recipients(providers),
                *([] if evaluated is None else evaluated.recipients),
            ],
            "caveat": CAVEAT,
            **({"notes": evaluated.notes} if evaluated is not None and evaluated.notes else {}),
            "warnings": list(dict.fromkeys(warnings)),
        }

    def _reuse_block(self, taken: Sequence[tuple[TargetReuse, int]]) -> dict[str, Any]:
        """The preview's ``reuse``: the store, its oldest answer reused, and the calls and
        sessions that saves; ``taken`` is each model's reuse and generations left."""
        if self.reuse is None:
            return {"status": "off"}
        answers = sum(len(reused.generations) for reused, _ in taken)
        stamps = [reused.oldest_sampled_at for reused, _ in taken if reused.oldest_sampled_at]
        return {
            "status": "on",
            "store": display_path(self.reuse.store),
            "oldest_sampled_at": min(stamps, default=None),
            "saved": {
                "generation_calls": answers,
                "judge_calls": sum(len(reused.verdicts) for reused, _ in taken),
                "max_generated_tokens": answers * self.settings.max_tokens,
                "sessions": sum(1 for reused, left in taken if reused.generations and not left),
            },
        }


def _models(count: int) -> str:
    return f"{count} base model{'' if count == 1 else 's'}"


def sample_conversations(
    cases: Sequence[ReplayCase], max_rows: int, *, seed: int
) -> list[ReplayCase]:
    """The cases of at most ``max_rows`` of the conversations ``cases`` come from, in order.

    The conversations are ranked by a seeded hash of their row ids, so a seed always picks
    the same ones, and a larger ``max_rows`` keeps those it picked before.
    """

    def rank(row_id: str) -> str:
        return hashlib.sha256(f"benchmark:{seed}:{row_id}".encode()).hexdigest()

    kept = set(sorted({case.row_id for case in cases}, key=rank)[:max_rows])
    return [case for case in cases if case.row_id in kept]


def plan_benchmark(
    runtime: Runtime,
    dataset_dir: Path,
    models: Sequence[tuple[str, str]],
    settings: BenchmarkSettings,
) -> BenchmarkPlan:
    """The benchmark of ``models`` (``(alias, provider)`` pairs) on the test split of the
    dataset in ``dataset_dir``.

    Raises :class:`UsageError` for fewer than one model, a model given twice, a missing
    package, a judge that is one of the models, a test split with no case or none with a
    prompt, a model that can replay none of its cases, or models that share none. With
    ``max_rows``, the cases are those of :func:`sample_conversations`, and each model replays
    those of them it can. Each model's renderer is loaded here for its identity and the cases
    it refuses; a model whose renderer cannot load is planned as an error
    (:attr:`Candidate.unavailable`). Unless ``settings.reuse`` is off, the dataset's store of
    base results is read here, once.
    """
    from pup_finetune.curation.pipeline import require_in_datadog
    from pup_finetune.evaluation.cases import EMPTY_PROMPT, build_cases
    from pup_finetune.evaluation.evaluators import resolve_evaluators, stored_evaluators
    from pup_finetune.evaluation.judge import check_judge_differs
    from pup_finetune.evaluation.replay import EvalDir, remaining_work
    from pup_finetune.evaluation.reuse import ReuseIndex, Wanted, generation_key, store_path

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
    conversations = len({case.row_id for case in cases})
    sampled_from = None
    if settings.max_rows is not None and settings.max_rows < conversations:
        cases = sample_conversations(cases, settings.max_rows, seed=settings.seed)
        sampled_from = conversations
    candidates: list[Candidate] = []
    for alias, provider_name in models:
        require_packages(provider_name, runtime, "evaluate")
        candidate = _candidate(runtime, provider_name, alias, cases)
        if any(other.target == candidate.target for other in candidates):
            raise UsageError(f"--models names {candidate.target} twice")
        if len(candidate.excluded) == len(cases):
            refused = [
                exclusion
                for exclusion in candidate.excluded.values()
                if exclusion.code != EMPTY_PROMPT
            ]
            if not refused:
                raise UsageError(
                    f"nothing to benchmark: the test split of {display_path(dataset.path)} has "
                    f"{len(cases)} test cases, and none has a message before its recorded turn "
                    f"({EMPTY_PROMPT})",
                    hint="import rows whose test conversations have a message before each "
                    "assistant turn",
                )
            first = refused[0]
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
        sampled_from=sampled_from,
    )
    if not plan.replayed_by_every_model():
        raise UsageError(
            "the models share no test case they can all replay, so they cannot be ranked on "
            "the same cases",
            hint="benchmark the models that leave cases out on their own",
        )
    model_dirs = [EvalDir(plan.model_dir(candidate)) for candidate in candidates]
    records = {
        d.evaluators_path: r
        for d in model_dirs
        if (r := stored_evaluators(d.evaluators_path)) is not None
    }
    # The newest breaks a tie, so --models order never picks an older record.
    newest = max(
        records, key=lambda p: (len(records[p]["evaluators"]), p.stat().st_mtime), default=None
    )
    stored = None if newest is None else records[newest]

    def pending(name: str) -> bool:
        return any(
            remaining_work(d, {c.target: plan.cases_for(c)}, [name])[c.target]["evaluators"][name]
            for d, c in zip(model_dirs, candidates, strict=True)
        )

    evaluators = resolve_evaluators(
        settings.evaluators, stored, judge=judge, runtime=runtime, pending=pending
    )
    for evaluator in [] if evaluators is None else evaluators.judges:
        for candidate in candidates:
            try:
                check_judge_differs(evaluator.config, candidate.profile.base_model)
            except UsageError as exc:
                raise UsageError(
                    f"evaluator {evaluator.name}: {exc.message}",
                    hint="name another model for it in the spec: a model should not grade its "
                    "own outputs",
                ) from None
    plan = replace(plan, evaluators=evaluators)
    if not settings.reuse:
        return plan
    if not store_path(dataset_dir).exists():
        return replace(plan, reuse=ReuseIndex(store_path(dataset_dir)))
    available = [candidate for candidate in candidates if candidate.unavailable is None]
    inputs = {candidate.target: _inputs(plan, candidate) for candidate in available}
    keys = {
        c.target: {case.id: generation_key(inputs[c.target], case) for case in plan.cases_for(c)}
        for c in available
    }
    wanted = [
        Wanted(
            c.target,
            plan.cases_for(c),
            keys[c.target],
            EvalDir(plan.model_dir(c)).read_generations(),
        )
        for c in available
    ]
    return replace(
        plan, inputs=inputs, keys=keys, reuse=ReuseIndex.load(dataset_dir, judge, wanted)
    )


def _candidate(
    runtime: Runtime, provider_name: str, alias: str, cases: Sequence[ReplayCase]
) -> Candidate:
    """``alias`` on ``provider_name``, with its renderer's identity and the ``cases`` it
    refuses; :attr:`Candidate.unavailable` when its renderer cannot load."""
    from pup_finetune.evaluation.cases import exclusions

    provider = get_provider(provider_name, runtime)
    profile = provider.resolve_model(alias)
    shared = any(name != provider_name for name in providers_for_model(profile.alias, runtime))
    try:
        renderer = provider.renderer(profile)
        renderer.check_tokens()
        rendering = renderer.identity()
    except ExternalError as exc:
        return Candidate(
            provider=provider_name,
            profile=profile,
            rendering={},
            excluded=exclusions(cases, None),
            shared=shared,
            unavailable=exc,
        )
    return Candidate(
        provider=provider_name,
        profile=profile,
        rendering=rendering,
        excluded=exclusions(cases, renderer),
        shared=shared,
    )


def _inputs(plan: BenchmarkPlan, candidate: Candidate) -> dict[str, Any]:
    """What ``candidate``'s answers are sampled under, from its renderer identity."""
    from pup_finetune.evaluation.reuse import key_inputs

    return key_inputs(
        candidate.provider, candidate.profile, candidate.rendering, plan.settings.max_tokens
    )


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
    kept = (store.read_calibration() or {}).get("evaluators")
    store.write_calibration({**report, **({"evaluators": kept} if kept else {})})
    return report


def _prepare(plan: BenchmarkPlan, candidate: Candidate, inputs: Mapping[str, Any]) -> None:
    """Stamp ``candidate``'s directory with the ``inputs`` its answers are sampled under,
    then copy in what it takes from the dataset's store."""
    from pup_finetune.evaluation.replay import EvalDir
    from pup_finetune.evaluation.reuse import import_target, stamp

    eval_dir = EvalDir(plan.model_dir(candidate))
    eval_dir.create()
    with eval_dir.lock():
        eval_dir.check_identity(plan.identity(candidate))
        stamp(eval_dir, inputs)
        taken = plan.reused(candidate)
        if plan.reuse is None or taken is None or not any(taken.counts().values()):
            return
        copied = import_target(eval_dir, taken, plan.cases_for(candidate))
    info(
        f"reusing {copied['generations']} answers of {candidate.target} "
        f"({copied['verdicts']} judged) from {display_path(plan.reuse.store)}"
    )


def _attach(plan: BenchmarkPlan, candidate: Candidate) -> None:
    """Attach the ``--evaluators`` spec to ``candidate``'s directory, which has nothing left
    to replay, so its headline and files are recorded as a replay would."""
    from pup_finetune.evaluation.evaluators import attach_evaluators
    from pup_finetune.evaluation.replay import EvalDir

    if plan.evaluators is None or plan.evaluators.source != "flag":
        return
    eval_dir = EvalDir(plan.model_dir(candidate))
    with eval_dir.lock():
        eval_dir.check_identity(plan.identity(candidate))
        attach_evaluators(eval_dir.evaluators_path, plan.evaluators)


def _share(
    plan: BenchmarkPlan,
    candidate: Candidate,
    inputs: Mapping[str, Any] | None,
    warnings: list[str],
) -> None:
    """Add ``candidate``'s own answers and verdicts, sampled under ``inputs`` (made now when
    ``None``), to the dataset's store; a warning when they could not be. A directory made
    before stamps shares nothing."""
    from pup_finetune.evaluation.replay import EvalDir
    from pup_finetune.evaluation.reuse import export_target, stamped

    eval_dir = EvalDir(plan.model_dir(candidate))
    if not eval_dir.has_work() or not stamped(eval_dir):
        return
    warning = export_target(
        plan.dataset_dir,
        eval_dir,
        candidate.target,
        plan.cases_for(candidate),
        inputs if inputs is not None else _inputs(plan, candidate),
        plan.judge,
        {"command": "benchmark", "run_id": None},
        f"{shlex.join(plan.command(evaluators=False))} --confirm",
    )
    if warning is not None and warning not in warnings:
        warnings.append(warning)


def _replay_model(
    plan: BenchmarkPlan,
    candidate: Candidate,
    client: ChatCompleter,
    cancel: threading.Event,
    warnings: list[str],
    scoring: Mapping[str, ChatCompleter],
) -> dict[str, Any] | None:
    """Replay every case ``candidate`` can (after copying what it takes from the dataset's
    store), then share its answers; the error that kept its renderer from loading or its
    session from starting, or ``None``."""
    from pup_finetune.evaluation.judge import judge
    from pup_finetune.evaluation.replay import run_replay

    if candidate.unavailable is not None:
        info(
            f"{candidate.target} could not be benchmarked: {redact(candidate.unavailable.message)}"
        )
        return error_json(candidate.unavailable)
    inputs = plan.inputs.get(candidate.target)
    if inputs is not None:
        _prepare(plan, candidate, inputs)
    remaining = plan.remaining(candidate)
    if not remaining["judge"] and not any(remaining.get("evaluators", {}).values()):
        # A rerun of a complete model shares what an earlier one could not.
        _attach(plan, candidate)
        _share(plan, candidate, inputs, warnings)
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
            evaluators=plan.evaluators,
            scorers=None if plan.evaluators is None else plan.evaluators.scorers(scoring),
        )

    info(f"benchmarking {candidate.target} on {candidate.provider}")
    if not remaining["generate"]:
        replay({})
    else:
        try:
            if inputs is None:
                inputs = _inputs(plan, candidate)
                _prepare(plan, candidate, inputs)
            with _base_sampler(plan, candidate) as sampler:
                replay({candidate.target: sampler})
        except ExternalError as exc:
            info(f"{candidate.target} could not be benchmarked: {redact(exc.message)}")
            return error_json(exc)
    _share(plan, candidate, inputs, warnings)
    return None


def execute(plan: BenchmarkPlan) -> dict[str, Any]:
    """Calibrate the judge, replay the cases with every model in turn, rank them and
    recommend one; writes ``benchmark.json`` and returns what the command emits."""
    from pup_finetune.evaluation.replay import EvalDir

    cancel = threading.Event()
    errors: dict[str, dict[str, Any]] = {}
    shared: list[str] = []
    EvalDir(plan.out_dir).create()
    with DirectoryLock(plan.out_dir), cancel_on_signals(cancel):
        calls = plan.judge_calls()
        busy = [] if plan.evaluators is None else busy_judges(plan.evaluators, plan.work())
        with (
            open_judge_client(plan.judge, calls=calls) as client,
            open_evaluator_clients(busy) as scoring,
        ):
            calibration = _calibration(plan, client)
            if calibration is not None and not calibration.get("ok"):
                return _calibration_failed(plan, calibration)
            failed = calibrate_evaluators(
                plan.evaluators,
                EvalDir(plan.out_dir),
                scoring,
                skip=plan.settings.skip_calibration,
            )
            if failed is not None:
                return _calibration_failed(plan, failed)
            for candidate in plan.candidates:
                if cancel.is_set():
                    break
                error = _replay_model(plan, candidate, client, cancel, shared, scoring)
                if error is not None:
                    errors[candidate.target] = error
    board = leaderboard(plan, errors)
    result = _result(plan, board, calibration=calibration, cancelled=cancel.is_set(), shared=shared)
    atomic_write_json(plan.out_dir / RESULT_FILE, finite_json(result))
    return result


def _calibration_failed(plan: BenchmarkPlan, report: Mapping[str, Any]) -> dict[str, Any]:
    name = report.get("evaluator")
    dirs = [plan.model_dir(candidate) for candidate in plan.candidates]
    hint = (
        "the judge passed a corrupted action or failed a recorded one, so its verdicts cannot "
        "be trusted; pick another --judge-model"
        if name is None
        else f"the evaluator {name} disagreed with answers you graded, so its verdicts cannot "
        f"be trusted; {evaluator_fix(str(name), dirs, plan.out_dir)}"
    )
    return {
        "status": "calibration_failed",
        "out_dir": display_path(plan.out_dir),
        "judge_model": plan.judge.spec,
        **({} if name is None else {"evaluator": name}),
        "calibration": dict(report),
        "hint": f"{hint} (or rerun with --skip-calibration to benchmark anyway)",
        "next_command": None,
    }


# --- ranking ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Leaderboard:
    """Every model's entry, ranked ones first, best first, then the unranked ones in the order
    given; :attr:`common_cases`, the size of the set the ranked ones are compared on; and
    :attr:`rerun`, whether a rerun has work left for some model."""

    entries: list[dict[str, Any]]
    common_cases: int
    """How many cases the headline scored for every finished model it scored a case for."""
    rerun: bool = False
    """Whether a model is unfinished, or has an evaluator score left that a rerun retries."""
    raised: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Each check's answers it raised on over every model, with the first error, as
    :func:`~pup_finetune.evaluation.runner.evaluator_errors` counts them."""


def _headline_passes(
    plan: BenchmarkPlan, candidate: Candidate, results: Mapping[PairKey, Mapping[str, Any]]
) -> dict[str, bool]:
    """Whether each of ``candidate``'s cases the headline scored in ``results`` passed it, by
    case id: the judge's verdict on an ``ok`` result, or the evaluator's (or ``all``'s) where
    it applies."""
    from pup_finetune.evaluation.replay import outcome

    found: dict[str, bool] = {}
    for case in plan.cases_for(candidate):
        result = results.get((candidate.target, case.id))
        verdict = outcome(result, plan.headline, plan.evaluator_names)
        if verdict in ("pass", "fail"):
            found[case.id] = verdict == "pass"
    return found


def _by_trajectory(cases: Sequence[ReplayCase], values: Mapping[str, float]) -> list[list[float]]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        if case.id in values:
            clusters[case.row_id].append(values[case.id])
    return list(clusters.values())


def _entry(
    plan: BenchmarkPlan, candidate: Candidate, error: Mapping[str, Any] | None
) -> tuple[dict[str, Any], dict[str, bool], bool, dict[str, dict[str, Any]]]:
    """``candidate``'s entry, whether each case its headline scored passed, whether a rerun
    has work left for it, and each check's answers it raised on. A check that raised is not
    work left: a rerun runs the same frozen file. A model whose renderer cannot load is an
    error, its results not read."""
    from pup_finetune.evaluation.bootstrap import paired_bootstrap_ci
    from pup_finetune.evaluation.cases import EMPTY_PROMPT, excluded_by_reason
    from pup_finetune.evaluation.replay import EvalDir, remaining_work, summarize

    eval_dir = EvalDir(plan.model_dir(candidate))
    cases = plan.cases_for(candidate)
    names = plan.evaluator_names
    results: dict[PairKey, dict[str, Any]] = {}
    generations: dict[PairKey, dict[str, Any]] = {}
    if candidate.unavailable is not None:
        error = error or error_json(candidate.unavailable)
    else:
        results, generations = eval_dir.read_results(), eval_dir.read_generations()
    target = summarize(
        cases,
        results,
        generations,
        seed=plan.settings.seed,
        replays={candidate.target: cases},
        evaluators=plan.evaluators,
    )["targets"][candidate.target]
    passes = _headline_passes(plan, candidate, results)
    interval = paired_bootstrap_ci(
        _by_trajectory(cases, {key: float(value) for key, value in passes.items()}),
        seed=plan.settings.seed,
    )
    latencies = [
        float(record["latency_s"])
        for (name, _), record in generations.items()
        if name == candidate.target and isinstance(record.get("latency_s"), int | float)
    ]
    unparseable = 0
    for case in cases:
        record = generations.get((candidate.target, case.id))
        if record is not None and record.get("stop_reason") != "length":
            unparseable += record.get("format_valid") is False
    scoring = sum(int(target["evaluators"][name]["errors"]) for name in names)
    checks = [] if plan.evaluators is None else [check.name for check in plan.evaluators.checks]
    raised = evaluator_errors(results, {candidate.target: cases}, checks)
    finished = settled = False
    if error is None:
        left = remaining_work(eval_dir, {candidate.target: cases}, names)[candidate.target]
        scores = {
            name: count - (raised[name]["errors"] if name in raised else 0)
            for name, count in left.get("evaluators", {}).items()
        }
        finished = not left["judge"] and not any(scores[name] for name in plan.headline_evaluators)
        settled = finished and not any(scores.values())
    if error is not None:
        status = "error"
    elif target["pending"]:
        status = "partial"
    else:
        status = "complete_with_errors" if target["errors"] or scoring else "complete"
    excluded = excluded_by_reason(candidate.excluded)
    prompted = len(plan.cases) - excluded.get(EMPTY_PROMPT, 0)
    entry = {
        "model": candidate.target,
        "provider": candidate.provider,
        "base_model": candidate.profile.base_model,
        "status": status,
        "finished": finished,
        "rank": None,
        "judge_pass_rate": target["judge_pass"]["rate"],
        "headline_pass_rate": sum(passes.values()) / len(passes) if passes else None,
        **(
            {}
            if plan.evaluators is None
            else {
                "evaluators": {
                    name: counts["pass"]["rate"] for name, counts in target["evaluators"].items()
                },
                "evaluator_errors": scoring,
            }
        ),
        "ci95": None if interval is None else list(interval),
        "judged_cases": target["completed"],
        "headline_cases": len(passes),
        "coverage": len(passes) / prompted,
        "common_pass_rate": None,
        "gap_to_best": None,
        "within_noise_of_best": None,
        "cases": len(cases),
        "excluded_cases": len(candidate.excluded),
        "excluded_by_reason": excluded,
        "errors": target["errors"],
        "errors_unrenderable": target["errors_unrenderable"],
        "pending": target["pending"],
        "truncated": target["truncated"],
        "unparseable": unparseable,
        "reused": target["reused"],
        "latency_p50_s": statistics.median(latencies) if latencies else None,
        "error": None if error is None else dict(error),
        "out_dir": display_path(eval_dir.path),
    }
    return entry, passes, not settled, raised


def leaderboard(
    plan: BenchmarkPlan, errors: Mapping[str, Mapping[str, Any]] | None = None
) -> Leaderboard:
    """Every model, with its headline pass rate (the judge's by default) and 95% bootstrap CI
    over trajectories on its own cases, ranked when it is finished (no session error, nothing
    pending, no error a rerun retries, no score the headline still needs) and the headline
    scored one of its cases.

    The ranked models are compared on the cases the headline scored for every one of them, so
    a model still running never narrows the comparison: each one's headline pass rate there
    (``common_pass_rate``) orders them, and each one's gap to the best (the mean paired
    difference, with its CI) is over those cases. A model is within noise of the best when that
    CI reaches 0 or there are too few cases for one."""
    from pup_finetune.evaluation.bootstrap import paired_bootstrap_ci

    errors = errors or {}
    scored = [
        _entry(plan, candidate, errors.get(candidate.target)) for candidate in plan.candidates
    ]
    compared = [(entry, passes) for entry, passes, _, _ in scored if entry["finished"] and passes]
    common = set.intersection(*(set(passes) for _, passes in compared)) if compared else set()
    ranked: list[tuple[dict[str, Any], dict[str, bool]]] = []
    if common:
        for entry, passes in compared:
            entry["common_pass_rate"] = sum(passes[case_id] for case_id in common) / len(common)
        ranked = sorted(compared, key=lambda item: -item[0]["common_pass_rate"])
        best = ranked[0][1]
        for rank, (entry, passes) in enumerate(ranked, start=1):
            entry["rank"] = rank
            if rank == 1:
                continue
            differences = {
                case_id: float(best[case_id]) - float(passes[case_id]) for case_id in common
            }
            interval = paired_bootstrap_ci(
                _by_trajectory(plan.cases, differences), seed=plan.settings.seed
            )
            gap = sum(differences.values()) / len(differences)
            entry["gap_to_best"] = {
                "pass_rate": gap,
                **({"judge_pass_rate": gap} if plan.headline == "judge" else {}),
                "ci95": None if interval is None else list(interval),
                "n_cases": len(differences),
            }
            entry["within_noise_of_best"] = interval is None or interval[0] <= 0
    entries = [entry for entry, _ in ranked]
    entries += [entry for entry, _, _, _ in scored if entry["rank"] is None]
    raised: dict[str, dict[str, Any]] = {}
    for _, _, _, found in scored:
        for name, counted in found.items():
            total = raised.setdefault(name, {"errors": 0, "first_error": None})
            total["errors"] += counted["errors"]
            total["first_error"] = total["first_error"] or counted["first_error"]
    return Leaderboard(
        entries,
        common_cases=len(common),
        rerun=any(rerun for _, _, rerun, _ in scored),
        raised=raised,
    )


def _pct(rate: float) -> str:
    return f"{round(rate * 100)}%"


def recommend(
    board: Sequence[Mapping[str, Any]], *, target_pass_rate: float, headline: str = "judge"
) -> dict[str, Any]:
    """The model to fine-tune, from a :attr:`Leaderboard.entries` ranked by ``headline``: the
    best ranked one.

    ``verdict`` is ``base_meets_target`` when that model's own headline pass rate reaches
    ``target_pass_rate`` and the cases its headline scored cover at least
    :data:`TARGET_COVERAGE` of the test cases with a prompt (fine-tuning may add little),
    ``fine_tune`` otherwise, or ``no_result`` when no model is ranked. The reason quotes the
    cases the rate is over. Only finished models are ranked, so a model still running changes
    neither the choice nor the verdict; the reason says the results are partial then. Models
    within noise of the best are named too: any of them is as good a start, so the cheapest to
    serve wins.
    """
    ranked = [entry for entry in board if entry.get("rank") is not None]
    unfinished = [str(entry["model"]) for entry in board if not entry.get("finished")]
    partial = (
        f"; results are partial: {joined(unfinished)} can still advance, and a rerun may "
        "change this"
        if unfinished
        else ""
    )
    which = "the judge and every evaluator" if headline == "all" else f"the evaluator {headline}"
    case = "judged case" if headline == "judge" else f"case {which} scored"
    if not ranked:
        if any(e.get("finished") and e.get("headline_pass_rate") is not None for e in board):
            reason = f"the finished models share no {case}, so none can be ranked"
        elif any(entry.get("headline_pass_rate") is not None for entry in board):
            reason = f"no finished model has a {case}, so none is ranked yet"
        else:
            pending = " yet" if unfinished else ""
            reason = f"no model has a {case}{pending}, so there is nothing to compare"
        return {"verdict": "no_result", "model": None, "reason": reason + partial}
    best = ranked[0]
    rate, coverage = float(best["headline_pass_rate"]), float(best["coverage"])
    close = [str(entry["model"]) for entry in ranked[1:] if entry.get("within_noise_of_best")]
    scored = best["headline_cases"]
    judged = (
        f"its {scored} judged cases"
        if headline == "judge"
        else f"the {scored} cases {'they' if headline == 'all' else 'it'} scored"
    )
    own = (
        f"{best['model']} ranks first and matches {_pct(rate)} of the recorded answers on {judged}"
        if headline == "judge"
        else f"{best['model']} ranks first and passes {which} on {_pct(rate)} of {judged}"
    )
    if rate < target_pass_rate:
        verdict = "fine_tune"
        reason = f"{own}, below the {_pct(target_pass_rate)} target: fine-tune it"
    elif coverage < TARGET_COVERAGE:
        verdict = "fine_tune"
        reason = (
            f"{own}, but they cover only {_pct(coverage)} of the test cases, under the "
            f"{_pct(TARGET_COVERAGE)} needed to say it meets the {_pct(target_pass_rate)} "
            "target: fine-tune it"
        )
    else:
        verdict = "base_meets_target"
        reason = (
            f"{own}, at or above the {_pct(target_pass_rate)} target, so fine-tuning may add "
            "little: check whether the remaining misses matter before training"
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
        "headline": headline,
        "pass_rate": rate,
        "judge_pass_rate": best["judge_pass_rate"],
        "coverage": coverage,
        "target_pass_rate": target_pass_rate,
        "within_noise": close,
        "reason": reason + partial,
    }


def _result(
    plan: BenchmarkPlan,
    board: Leaderboard,
    *,
    calibration: Mapping[str, Any] | None,
    cancelled: bool,
    shared: Sequence[str] = (),
) -> dict[str, Any]:
    from pup_finetune.evaluation.replay import CAVEAT, truncation_warning

    entries = board.entries
    recommendation = recommend(
        entries, target_pass_rate=plan.settings.target_pass_rate, headline=plan.headline
    )
    statuses = {entry["status"] for entry in entries}
    if cancelled or "partial" in statuses:
        status = "partial"
    elif statuses == {"complete"}:
        status = "complete"
    else:
        status = "complete_with_errors"
    warnings: list[str] = []
    hint = None
    unrenderable = sum(int(entry["errors_unrenderable"]) for entry in entries)
    failed = [entry["model"] for entry in entries if not entry["finished"]]
    if failed:
        keys = "errors, evaluator_errors" if plan.evaluators is not None else "errors"
        hint = (
            f"{', '.join(failed)} {'has' if len(failed) == 1 else 'have'} no result for every "
            f"case (see each entry's {keys}, error and pending); run next_command to continue, "
            "which reuses every finished result"
        )
    if unrenderable:
        hint = (f"{hint}. " if hint else "") + (
            f"{unrenderable} case(s) have a prompt a model's chat format cannot continue, "
            "which a retry cannot fix, so next_command never retries them"
        )
    if broken := check_hint(board.raised, set(board.raised), plan.out_dir):
        hint = broken if hint is None else f"{hint}. {broken}"
    scoring = sum(int(entry.get("evaluator_errors") or 0) for entry in entries)
    if missing := scoring - unrenderable * len(plan.evaluator_names):
        warnings.append(scoring_warning(missing, retried=board.rerun))
    rate = recommendation.get("judge_pass_rate")
    if isinstance(rate, float) and rate < LOW_PASS_RATE:
        warnings.append(
            f"the best base model matches only {_pct(rate)} of the recorded answers: "
            "fine-tuning may not close a gap this wide, so add rows or try a larger base "
            "model too"
        )
    for entry in entries:
        if entry["truncated"]:
            warnings.append(
                truncation_warning(
                    entry["truncated"],
                    f"{entry['model']}'s",
                    plan.settings.max_tokens,
                    "benchmark",
                )
            )
        if entry["unparseable"]:
            warnings.append(
                f"{entry['unparseable']} of {entry['model']}'s answers could not be "
                "parsed (format_valid false in the generations.jsonl of "
                f"{entry['out_dir']}) and count as failures"
            )
    warnings += plan.split_warnings()
    warnings += shared
    return finite_json(
        {
            "status": status,
            "recommendation": recommendation,
            "leaderboard": entries,
            "dataset": display_path(plan.dataset_dir),
            "out_dir": display_path(plan.out_dir),
            "judge_model": plan.judge.spec,
            "cases": len(plan.cases),
            "common_cases": board.common_cases,
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
                "models": {entry["model"]: entry["out_dir"] for entry in entries},
            },
            **_next(plan, recommendation, entries, retry=board.rerun),
        }
    )


def _next(
    plan: BenchmarkPlan,
    recommendation: Mapping[str, Any],
    board: Sequence[Mapping[str, Any]],
    *,
    retry: bool,
) -> dict[str, Any]:
    """An unfinished benchmark that a rerun can advance is continued (a free preview first).
    A finished one leads to the training preview of the recommended model, which is a step
    instead when the base model already meets the target or most of its failures were cut
    off at ``--max-tokens``. Answers cut off add a step that benchmarks again with twice the
    ``--max-tokens``."""
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
    steps = []
    if any(entry["truncated"] for entry in board):
        longer = replace(
            plan.settings,
            out_dir=plan.dataset_dir / DEFAULT_OUT_DIR_NAME,
            max_tokens=2 * plan.settings.max_tokens,
        )
        words = replace(plan, settings=longer).command()[len(PROGRAM) :]
        steps.append(
            step(
                template(*words, "--out-dir", "<NEW_DIR>"),
                "compare the models on their answers, not their length (free preview; "
                "benchmarking is paid)",
            )
        )
    if retry:
        return {"next_command": rerun, "next_steps": steps}
    if train is None:
        return {"next_command": None, "next_steps": steps}
    best = board[0]
    failed = best["judged_cases"] - round((best["judge_pass_rate"] or 0) * best["judged_cases"])
    if recommendation.get("verdict") == "fine_tune" and 2 * best["truncated"] <= failed:
        return {"next_command": train, "next_steps": steps}
    steps.append(
        step(train, "fine-tune the best base model anyway (free preview; training is paid)")
    )
    return {"next_command": None, "next_steps": steps}


def exit_code(result: Mapping[str, Any]) -> int:
    """1 when the judge failed calibration, or a benchmark that ran every model left one
    with no result at all (its session failed, or every case ended in an error); else 0."""
    if result.get("status") == "calibration_failed":
        return 1
    if result.get("status") != "complete_with_errors":
        return 0
    entries = result.get("leaderboard") or []
    return 1 if any(entry.get("judged_cases") == 0 for entry in entries) else 0
