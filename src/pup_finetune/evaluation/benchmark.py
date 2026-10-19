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
model in turn (:func:`~pup_finetune.evaluation.replay.run_replay`, with the model alias as
its one target), in a session of its own that ends before the next model starts, into
``OUT/<model>``; a rerun resumes every model where it stopped. A model whose session cannot
start is recorded as an error and the others still run. Base answers and verdicts are shared
with the dataset's other evaluations and benchmarks (:mod:`~pup_finetune.evaluation.reuse`):
a model with every answer reused opens no session.

:func:`leaderboard` ranks the models by judge pass rate, with a 95% bootstrap CI over
trajectories and each one's paired gap to the best; :func:`recommend` names the model to
fine-tune, or says the best base model already reaches ``--target-pass-rate``. Like
``evaluate``, the scores measure agreement with the recorded answers, not task success.
"""

from __future__ import annotations

import hashlib
import shlex
import statistics
import threading
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.display import display_path
from pup_finetune.errors import ExternalError, FinetuneError, SettingsConflict, UsageError
from pup_finetune.evaluation.runner import (
    CALIBRATION_CASES,
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_CASES_PER_TRAJECTORY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SEED,
    calibration_line,
    cancel_on_signals,
    no_reuse_hint,
    open_judge_client,
    resolve_judge,
    stored_judge,
)
from pup_finetune.next_steps import PROGRAM, command, step, template
from pup_finetune.output import info
from pup_finetune.progress import EvaluationProgress
from pup_finetune.providers import get_provider, provider_info, require_packages
from pup_finetune.providers.base import (
    Generation,
    LoraConfig,
    ModelProfile,
    Sampler,
    SamplingUsage,
)
from pup_finetune.redaction import redact
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
    from pup_finetune.evaluation.cases import ReplayCase
    from pup_finetune.evaluation.judge import Judgment
    from pup_finetune.evaluation.replay import EvalDir
    from pup_finetune.evaluation.reuse import ReuseIndex, TargetReuse
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
    "sample_conversations",
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
    max_rows: int | None = None
    judge_model: str | None = None
    concurrency: int = DEFAULT_CONCURRENCY
    max_tokens: int = DEFAULT_MAX_TOKENS
    seed: int = DEFAULT_SEED
    skip_calibration: bool = False
    target_pass_rate: float = DEFAULT_TARGET_PASS_RATE
    reuse: bool = True


@dataclass(frozen=True, slots=True)
class Candidate:
    """One base model to benchmark: its provider and profile."""

    provider: str
    profile: ModelProfile

    @property
    def target(self) -> str:
        """The replay target name, and the name of its directory under ``--out-dir``."""
        return self.profile.alias


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

    def model_dir(self, candidate: Candidate) -> Path:
        return self.out_dir / candidate.target

    def identity(self, candidate: Candidate) -> dict[str, Any]:
        """What ``OUT/<model>/evaluation.json`` records: everything that changes a generation
        or a score of that model."""
        from pup_finetune.evaluation.replay import METRICS_VERSION, cases_sha256

        profile = candidate.profile
        return {
            "benchmark": True,
            "provider": candidate.provider,
            "base_model": profile.base_model,
            "rendering": {
                "renderer": profile.renderer,
                "tokenizer_model": profile.tokenizer_model,
                "tokenizer_revision": profile.tokenizer_revision,
            },
            "judge": self.judge.identity(),
            "cases_sha256": cases_sha256(self.cases),
            "n_cases": len(self.cases),
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
        ``--confirm`` would start it again (another identity, no work).

        Raises :class:`SettingsConflict` when it holds work done under other settings, which
        ``--confirm`` would refuse.
        """
        from pup_finetune.evaluation.replay import EvalDir

        eval_dir = EvalDir(self.model_dir(candidate))
        stored = eval_dir.stored_identity()
        wanted = finite_json(self.identity(candidate))
        if stored is None or canonical_json(stored) == canonical_json(wanted):
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
            self.cases,
            self.keys.get(candidate.target, {}),
            self.judge,
            {} if eval_dir is None else eval_dir.read_generations(),
            {} if eval_dir is None else eval_dir.read_results(),
        )

    def remaining(self, candidate: Candidate) -> dict[str, int]:
        """What a run would still do for ``candidate``: ``{"done", "generate", "judge"}``,
        less what it takes from the dataset's store.

        Raises :class:`SettingsConflict` when its directory holds work done under other
        settings, which ``--confirm`` would refuse.
        """
        from pup_finetune.evaluation.replay import remaining_work

        eval_dir = self._matching_dir(candidate)
        if eval_dir is None:
            found = {"done": 0, "generate": len(self.cases), "judge": len(self.cases)}
        else:
            found = remaining_work(eval_dir, {candidate.target: self.cases})[candidate.target]
        taken = self.reused(candidate)
        if taken is not None:
            found["generate"] -= len(taken.generations)
            found["judge"] -= len(taken.verdicts)
        return found

    def split_warnings(self) -> list[str]:
        """A warning when the test split is too small for the intervals to separate models."""
        if self.trajectories >= MIN_TRAJECTORIES:
            return []
        if self.sampled_from is not None:
            return [
                f"--max-rows {self.settings.max_rows} replays only {self.trajectories} of the "
                f"{self.sampled_from} test conversations, so the confidence intervals are wide: "
                "raise --max-rows for a firmer choice"
            ]
        return [
            f"the test split holds only {self.trajectories} conversations, so the confidence "
            "intervals are wide: pull more rows for a firmer choice"
        ]

    def judge_calls(self) -> int:
        """The judge calls a run would still make: calibration's and every candidate's.
        The preview counts them and execution opens the judge only when one is left."""
        remaining = sum(self.remaining(candidate)["judge"] for candidate in self.candidates)
        return remaining + self.calibration_calls()

    def calibration_calls(self) -> int:
        if self.settings.skip_calibration or self._calibrated():
            return 0
        return 3 * min(CALIBRATION_CASES, len(self.cases))

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
        if settings.max_rows is not None:
            words += ["--max-rows", str(settings.max_rows)]
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
        from pup_finetune.evaluation.replay import CAVEAT
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
        if judge_calls + calibration:
            # --confirm would fail on a key nothing provides: fail now, never offer it. A
            # read-only preview offers none, so it only warns.
            require_key(self.judge, warnings=warnings if self.runtime.read_only else None)
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
        sampled = "" if self.sampled_from is None else f" (of {self.sampled_from}, --max-rows)"
        providers = sorted(calls_by_provider)
        sent = f"{', '.join(providers)} to generate and to " if providers else ""
        return {
            "action": f"benchmark {_models(len(self.candidates))} ({', '.join(names)}) on "
            f"{len(self.cases)} cases from {self.trajectories} test conversations{sampled}: "
            + (
                "generate each with every model, one model at a time in a session of its own, "
                f"and judge each answer with {judge_name}"
                if generate
                else "every answer is already generated, so no session opens; judge any answer "
                f"left with {judge_name}"
            ),
            "billing": "; ".join(billing),
            "dataset": display_path(self.dataset_dir),
            "out_dir": display_path(self.out_dir),
            "rows": {
                "test": self.trajectories,
                "test_total": self.sampled_from or self.trajectories,
            },
            "cases": len(self.cases),
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
            },
            "data_destination": f"the test split's conversations are sent to {sent}{judged_at} "
            "to judge",
            "data_recipients": self.recipients(providers),
            "caveat": CAVEAT,
            "warnings": warnings,
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
    package, a judge that is one of the models, or a test split with no case. With
    ``max_rows``, the cases are those of :func:`sample_conversations`. Unless
    ``settings.reuse`` is off, the dataset's store of base results is read here, once: each
    model's renderer is loaded for the keys only when the store exists, and a model whose
    renderer cannot load reuses nothing (its run records the error).
    """
    from pup_finetune.curation.pipeline import require_in_datadog
    from pup_finetune.evaluation.cases import build_cases
    from pup_finetune.evaluation.judge import check_judge_differs
    from pup_finetune.evaluation.replay import EvalDir
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
    candidates: list[Candidate] = []
    for alias, provider_name in models:
        require_packages(provider_name, runtime, "evaluate")
        profile = get_provider(provider_name, runtime).resolve_model(alias)
        if any(candidate.target == profile.alias for candidate in candidates):
            raise UsageError(f"--models names {profile.alias} twice")
        candidates.append(Candidate(provider=provider_name, profile=profile))
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
    if not settings.reuse:
        return plan
    if not store_path(dataset_dir).exists():
        return replace(plan, reuse=ReuseIndex(store_path(dataset_dir)))
    inputs: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        with suppress(ExternalError):
            inputs[candidate.target] = _inputs(plan, candidate)
    keys = {
        target: {case.id: generation_key(found, case) for case in cases}
        for target, found in inputs.items()
    }
    wanted = [
        Wanted(c.target, cases, keys[c.target], EvalDir(plan.model_dir(c)).read_generations())
        for c in candidates
        if c.target in keys
    ]
    return replace(
        plan, inputs=inputs, keys=keys, reuse=ReuseIndex.load(dataset_dir, judge, wanted)
    )


def _inputs(plan: BenchmarkPlan, candidate: Candidate) -> dict[str, Any]:
    """What ``candidate``'s answers are sampled under, from its full renderer identity, which
    loads its renderer."""
    from pup_finetune.evaluation.reuse import key_inputs

    provider = get_provider(candidate.provider, plan.runtime)
    provider.validate_options({})
    rendering = provider.renderer(candidate.profile).identity()
    return key_inputs(candidate.provider, candidate.profile, rendering, plan.settings.max_tokens)


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
    longest = max(
        len(
            renderer.generation_prompt_tokens(
                [dict(m) for m in case.prefix], [dict(t) for t in case.tools]
            )
        )
        for case in plan.cases
    )
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
        **calibrate(plan.cases, client, plan.judge, n=CALIBRATION_CASES).to_json(),
        "judge": plan.judge.identity(),
    }
    info(calibration_line(report))
    store.write_calibration(report)
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
        copied = import_target(eval_dir, taken, plan.cases)
    info(
        f"reusing {copied['generations']} answers of {candidate.target} "
        f"({copied['verdicts']} judged) from {display_path(plan.reuse.store)}"
    )


def _share(
    plan: BenchmarkPlan,
    candidate: Candidate,
    inputs: Mapping[str, Any] | None,
    warnings: list[str],
) -> None:
    """Add ``candidate``'s own answers and verdicts, sampled under ``inputs`` (loaded now
    when ``None``), to the dataset's store; a warning when they could not be. A directory
    made before stamps shares nothing, so its renderer is not loaded."""
    from pup_finetune.evaluation.replay import EvalDir
    from pup_finetune.evaluation.reuse import export_target, not_shared, stamped

    eval_dir = EvalDir(plan.model_dir(candidate))
    if not eval_dir.has_work() or not stamped(eval_dir):
        return
    rerun = f"{shlex.join(plan.command())} --confirm"
    try:
        inputs = inputs if inputs is not None else _inputs(plan, candidate)
    except (FinetuneError, OSError) as exc:
        warning: str | None = not_shared(exc, rerun)
    else:
        warning = export_target(
            plan.dataset_dir,
            eval_dir,
            candidate.target,
            plan.cases,
            inputs,
            plan.judge,
            {"command": "benchmark", "run_id": None},
            rerun,
        )
    if warning is not None and warning not in warnings:
        warnings.append(warning)


def _replay_model(
    plan: BenchmarkPlan,
    candidate: Candidate,
    client: ChatCompleter,
    cancel: threading.Event,
    warnings: list[str],
) -> dict[str, Any] | None:
    """Replay every case with ``candidate`` (after copying what it takes from the dataset's
    store), then share its answers; the error that kept its session from starting, or
    ``None``."""
    from pup_finetune.evaluation.judge import judge
    from pup_finetune.evaluation.replay import run_replay

    inputs = plan.inputs.get(candidate.target)
    if inputs is not None:
        _prepare(plan, candidate, inputs)
    remaining = plan.remaining(candidate)
    if not remaining["judge"]:
        # A rerun of a complete model shares what an earlier one could not.
        _share(plan, candidate, inputs, warnings)
        return None
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

    def replay(samplers: Mapping[str, Sampler]) -> None:
        run_replay(
            plan.cases,
            samplers,
            judge_fn,
            out_dir=plan.model_dir(candidate),
            identity=plan.identity(candidate),
            concurrency=plan.settings.concurrency,
            cancel=cancel,
            on_progress=EvaluationProgress(),
            replays={candidate.target: plan.cases},
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
        with open_judge_client(plan.judge, calls=calls) as client:
            calibration = _calibration(plan, client)
            if calibration is not None and not calibration.get("ok"):
                return _calibration_failed(plan, calibration)
            for candidate in plan.candidates:
                if cancel.is_set():
                    break
                error = _replay_model(plan, candidate, client, cancel, shared)
                if error is not None:
                    errors[candidate.target] = error
    board = leaderboard(plan, errors)
    result = _result(plan, board, calibration=calibration, cancelled=cancel.is_set(), shared=shared)
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
    for case in plan.cases:
        result = results.get((candidate.target, case.id))
        if result is not None and result.get("status") == "ok":
            found[case.id] = _passed(result)
    return found


def _by_trajectory(cases: Sequence[ReplayCase], values: Mapping[str, float]) -> list[list[float]]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        if case.id in values:
            clusters[case.row_id].append(values[case.id])
    return list(clusters.values())


def _entry(
    plan: BenchmarkPlan, candidate: Candidate, error: Mapping[str, Any] | None
) -> dict[str, Any]:
    from pup_finetune.evaluation.bootstrap import paired_bootstrap_ci
    from pup_finetune.evaluation.replay import EvalDir, summarize

    eval_dir = EvalDir(plan.model_dir(candidate))
    results, generations = eval_dir.read_results(), eval_dir.read_generations()
    target = summarize(
        plan.cases,
        results,
        generations,
        seed=plan.settings.seed,
        replays={candidate.target: plan.cases},
    )["targets"][candidate.target]
    passes = _ok_passes(plan, candidate)
    interval = paired_bootstrap_ci(
        _by_trajectory(plan.cases, {key: float(value) for key, value in passes.items()}),
        seed=plan.settings.seed,
    )
    latencies = [
        float(record["latency_s"])
        for (name, _), record in generations.items()
        if name == candidate.target and isinstance(record.get("latency_s"), int | float)
    ]
    unparseable = 0
    for case in plan.cases:
        record = generations.get((candidate.target, case.id))
        if record is not None and record.get("stop_reason") != "length":
            unparseable += record.get("format_valid") is False
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
        "judge_pass_rate": target["judge_pass"]["rate"],
        "ci95": None if interval is None else list(interval),
        "cases": target["completed"],
        "errors": target["errors"],
        "pending": target["pending"],
        "truncated": target["truncated"],
        "unparseable": unparseable,
        "reused": target["reused"],
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
    given), each with its gap to the best: the mean paired difference over the cases both
    passed through the judge, with a 95% bootstrap CI over trajectories. A model is within
    noise of the best when that CI reaches 0 or there are too few cases for one."""
    from pup_finetune.evaluation.bootstrap import paired_bootstrap_ci

    errors = errors or {}
    entries = [
        _entry(plan, candidate, errors.get(candidate.target)) for candidate in plan.candidates
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
    by_target = {candidate.target: candidate for candidate in plan.candidates}
    best = _ok_passes(plan, by_target[board[0]["model"]])
    for entry in board[1:]:
        if entry["judge_pass_rate"] is None:
            continue
        other = _ok_passes(plan, by_target[entry["model"]])
        differences = {
            case_id: float(best[case_id]) - float(other[case_id])
            for case_id in best.keys() & other.keys()
        }
        interval = paired_bootstrap_ci(
            _by_trajectory(plan.cases, differences), seed=plan.settings.seed
        )
        entry["gap_to_best"] = {
            "judge_pass_rate": sum(differences.values()) / len(differences)
            if differences
            else None,
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
    shared: Sequence[str] = (),
) -> dict[str, Any]:
    from pup_finetune.evaluation.replay import CAVEAT, truncation_warning

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
    failed = [entry["model"] for entry in board if entry["status"] != "complete"]
    if status != "complete":
        hint = (
            f"{', '.join(failed)} {'has' if len(failed) == 1 else 'have'} no result for every "
            "case (see each entry's errors, error and pending); run next_command to continue, "
            "which reuses every finished result"
        )
    rate = recommendation.get("judge_pass_rate")
    if isinstance(rate, float) and rate < LOW_PASS_RATE:
        warnings.append(
            f"the best base model matches only {_pct(rate)} of the recorded answers: "
            "fine-tuning may not close a gap this wide, so add rows or try a larger base "
            "model too"
        )
    for entry in board:
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
            "leaderboard": board,
            "dataset": display_path(plan.dataset_dir),
            "out_dir": display_path(plan.out_dir),
            "judge_model": plan.judge.spec,
            "cases": len(plan.cases),
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
            **_next(plan, status, recommendation, board),
        }
    )


def _next(
    plan: BenchmarkPlan,
    status: str,
    recommendation: Mapping[str, Any],
    board: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """An unfinished benchmark is continued (a free preview first). A finished one leads to
    the training preview of the recommended model, which is a step instead when the base
    model already meets the target or most of its failures were cut off at ``--max-tokens``.
    Answers cut off add a step that benchmarks again with twice the ``--max-tokens``."""
    rerun = shlex.join(plan.command())
    model = recommendation.get("model")
    train = (
        command("train", "--dataset", plan.dataset_dir, "--model", model)
        if isinstance(model, str)
        else None
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
    if status != "complete":
        return {"next_command": rerun, "next_steps": steps}
    if train is None:
        return {"next_command": None, "next_steps": steps}
    best = board[0]
    failed = best["cases"] - round(best["judge_pass_rate"] * best["cases"])
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
    return 1 if any(entry.get("cases") == 0 for entry in result.get("leaderboard") or []) else 0
