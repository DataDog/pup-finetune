# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune dataset triage`: a judge council labels a dataset's rows against a rubric.

Judging is paid (one call per row and judge), so without ``--confirm`` it only previews the
calls. It reads and writes local files and calls the judges' inference APIs (Fireworks,
Anthropic or OpenAI, by each judge's prefix), with no Datadog or other remote write, so
``PUP_READ_ONLY`` does not block it; ``--confirm`` is still required, and never implied by
``PUP_AUTO_APPROVE``. The council, the votes file and the labels live in
:mod:`pup_finetune.curation.triage`. ``--agreement`` reviews the rows the agent or a person
labeled instead, labels nothing, and reports how often the council agrees with them.
``--checks`` runs the user's Python row checks first; with ``--checks`` alone no judge is
called, so it labels at once, free and local like ``dataset label``, with no ``--confirm``.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from pup_finetune.cli._common import CommandResult, current_context, run_command, with_settings
from pup_finetune.cli._dirs import require_dataset_dir
from pup_finetune.cli._flags import flag_name, given, require_alone, require_enablers
from pup_finetune.cli._presentation import CommandDoc, Safety, default_note, describe
from pup_finetune.cli.dataset import dataset_next, merge_next
from pup_finetune.curation import triage
from pup_finetune.curation.pipeline import pipeline_state
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.display import display_path
from pup_finetune.errors import UsageError, require_confirm
from pup_finetune.judge_models import BY_KEY
from pup_finetune.next_steps import PROGRAM, step, template
from pup_finetune.output import info
from pup_finetune.user_checks import load_checks

if TYPE_CHECKING:
    from pup_finetune.inference import ChatCompleter, JudgeProvider


@contextmanager
def open_clients(judges: Sequence[str]) -> Iterator[Mapping[JudgeProvider, ChatCompleter]]:
    """One chat client per provider of ``judges``, each with its provider's key, opened up
    front and closed together (:func:`pup_finetune.judges.open_clients`).

    Tests replace it with a fake.
    """
    from pup_finetune.judges import open_clients as open_judge_clients

    with open_judge_clients(triage.judge_config(spec) for spec in judges) as clients:
        yield clients


def _check_available(judges: Sequence[str]) -> None:
    """A :class:`UsageError` before the preview when a judge needs a missing package."""
    from pup_finetune.judges import check_available

    for spec in judges:
        check_available(triage.judge_config(spec))


TRIAGE_NEEDS: dict[str, str] = {"allow_template_rubric": "rubric"}
"""``dataset triage``'s options that need another one (see :mod:`pup_finetune.cli._flags`)."""

RUBRIC = "Rubric"
JUDGES = "Judges"
RUN = "Run"
"""``dataset triage``'s option panels."""

DOC = CommandDoc(
    path="dataset triage",
    short="Label rows with checks and judge models (judges paid; preview first).",
    description=describe(
        "Have each judge model vote keep or drop on each whole row of DIR against --rubric "
        "and/or --rule; a strict majority of keep votes keeps a row. Without --confirm, "
        "previews the calls and where the rows go. Rerun the same command to resume; start "
        "with --max-rows 20 to calibrate the rubric.",
        "A judge reviews every turn of a row and its tools, since training imitates every "
        "assistant turn. A tie drops a row; a row missing a vote (a failed call) stays "
        "undecided and unlabeled, unless the votes cast already drop it (one drop of two "
        "judges, two of three), which also skips its other calls. Decisions are recorded as "
        "labels by judge; rows the agent or a person labeled are not reviewed. Votes are "
        "appended to DIR/judgments.jsonl as they arrive, so a rerun retries only failed "
        "calls. The rubric and the judges are frozen in DIR/triage.json once votes exist: "
        "change them in a new directory or with --reset-triage. Row content is sent to each "
        "judge's provider, Anthropic, OpenAI or Fireworks serverless, with that provider's "
        "API key; the preview names every one and where the council came from "
        "(judges_source). Without --judges, the council is the one triage.json froze, else "
        "anthropic/claude-haiku-5-5 and openai/gpt-5.4-mini when both credentials resolve, "
        "anthropic/claude-haiku-5-5 and anthropic/claude-sonnet-5-5 with only Anthropic's, "
        "openai/gpt-5.4-mini and openai/gpt-6-astra with only OpenAI's, else Fireworks' "
        "deepseek-v4p1-flash and glm-5p3-flash. --agreement measures the council against rows you "
        "labeled first; --rows FILE (one row id per line) reviews only those rows in either "
        "mode, and is not frozen; --print-template prints a rubric to start from. --checks "
        "FILE runs every public function of a Python file on each row reviewed first, free "
        "and in this process: a row one fails is labeled drop by: check and never sent to a "
        "judge (with --agreement, the checks are compared with your labels apart). --checks "
        "alone calls no judge and labels at once, without --confirm; without --checks, rows "
        "a check labeled are skipped.",
    ),
    examples=(
        "pup finetune dataset triage --print-template > rubric.md",
        "pup finetune dataset triage datasets/my-agent --rubric rubric.md --max-rows 20",
        "pup finetune dataset triage datasets/my-agent --rubric rubric.md --agreement",
        "pup finetune dataset triage datasets/my-agent --checks checks.py",
    ),
    safety=Safety(
        data="sends rows to each judge's provider",
        recipients="each judge's provider: Fireworks, Anthropic or OpenAI",
        without="previews the judge calls (--checks alone labels at once, free)",
        confirm="runs the judges and labels rows (paid)",
        paid=True,
    ),
)


@run_command
def triage_rows(
    directory: Annotated[
        Path | None,
        typer.Argument(metavar="DIR", help="The dataset directory.", show_default=False),
    ] = None,
    rubric: Annotated[
        Path | None,
        typer.Option(
            "--rubric",
            metavar="FILE",
            help="A rubric file (see --print-template).",
            exists=True,
            dir_okay=False,
            readable=True,
            show_default=False,
            rich_help_panel=RUBRIC,
        ),
    ] = None,
    rule: Annotated[
        list[str] | None,
        typer.Option(
            "--rule",
            metavar="TEXT",
            help="A rule to judge by; repeatable.",
            show_default=False,
            rich_help_panel=RUBRIC,
        ),
    ] = None,
    checks: Annotated[
        Path | None,
        typer.Option(
            "--checks",
            metavar="FILE",
            help="Python row checks, run first (free).",
            exists=True,
            dir_okay=False,
            readable=True,
            show_default=False,
            rich_help_panel=RUBRIC,
        ),
    ] = None,
    allow_template_rubric: Annotated[
        bool,
        typer.Option(
            "--allow-template-rubric",
            help="With --rubric: allow the template's Task text.",
            rich_help_panel=RUBRIC,
        ),
    ] = False,
    print_template: Annotated[
        bool,
        typer.Option(
            "--print-template",
            help="Print a rubric template and exit.",
            rich_help_panel=RUBRIC,
        ),
    ] = False,
    judges: Annotated[
        str | None,
        typer.Option(
            "--judges",
            metavar="M1,M2",
            help="Aliases (glm-5p3-flash), anthropic/MODEL, openai/MODEL or Fireworks ids. "
            f"{default_note(BY_KEY)}",
            show_default=False,
            rich_help_panel=JUDGES,
        ),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option(
            min=1, max=64, metavar="N", help="Judge calls at once (1-64).", rich_help_panel=JUDGES
        ),
    ] = triage.DEFAULT_CONCURRENCY,
    max_rows: Annotated[
        int | None,
        typer.Option(
            "--max-rows",
            min=1,
            metavar="N",
            help="Review a fixed sample of at most N rows.",
            show_default=False,
            rich_help_panel=RUN,
        ),
    ] = None,
    rows: Annotated[
        Path | None,
        typer.Option(
            "--rows",
            metavar="FILE",
            help="Review only these row ids, one per line.",
            exists=True,
            dir_okay=False,
            readable=True,
            show_default=False,
            rich_help_panel=RUN,
        ),
    ] = None,
    agreement: Annotated[
        bool,
        typer.Option(
            "--agreement",
            help="Measure the council against your labels.",
            rich_help_panel=RUN,
        ),
    ] = False,
    reset_triage: Annotated[
        bool,
        typer.Option(
            "--reset-triage",
            help="Discard the votes, rubric and judges first.",
            rich_help_panel=RUN,
        ),
    ] = False,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help="Run the judges (paid); without it, only preview.",
            rich_help_panel=RUN,
        ),
    ] = False,
) -> dict[str, Any] | CommandResult | None:
    require_alone("print_template")
    require_enablers(TRIAGE_NEEDS)
    if print_template:
        typer.echo(triage.rubric_template(), nl=False)
        return None
    if directory is None:
        raise UsageError("pass the dataset directory DIR", hint="or --print-template")
    rules = rule or []
    checks_only = checks is not None and rubric is None and not rules
    if checks_only:
        _refuse_judge_flags()
    rubric_text = rubric.read_text(encoding="utf-8") if rubric is not None else None
    criteria = None if checks_only else triage.Criteria.build(rubric_text, rules)
    template_task = None
    if rubric is not None and rubric_text is not None and triage.has_template_task(rubric_text):
        template_task = (
            f"{rubric} still has the template's Task text; describe what the agent is for first"
        )
    template_hint = (
        f"edit the Task section of {rubric}, or pass --allow-template-rubric to judge by it anyway"
    )
    if template_task is not None and confirm and not allow_template_rubric:
        raise UsageError(template_task, hint=template_hint)
    require_dataset_dir(directory)
    dataset = DatasetDir(directory)
    module = load_checks(checks) if checks is not None else None
    council: tuple[str, ...] = ()
    council_source = ""
    if judges is not None:
        council, council_source = triage.parse_judges(judges), triage.COUNCIL_GIVEN
    elif not checks_only:
        council, council_source = triage.default_council(dataset, reset=reset_triage)
    _check_available(council)
    plan = triage.plan_triage(
        dataset,
        criteria,
        council,
        max_rows=max_rows,
        only=_row_ids(rows) if rows is not None else None,
        reset=reset_triage,
        agreement=agreement,
        checks=module,
    )
    labeling = [*PROGRAM, "dataset", "triage", str(directory)]
    if rubric is not None:
        labeling += ["--rubric", str(rubric)]
    for text in rules:
        labeling += ["--rule", text]
    if checks is not None:
        labeling += ["--checks", str(checks)]
    if not checks_only:
        labeling += ["--judges", ",".join(council)]
    if concurrency != triage.DEFAULT_CONCURRENCY:
        labeling += ["--concurrency", str(concurrency)]
    if allow_template_rubric and template_task is not None:
        labeling.append("--allow-template-rubric")
    every_row = [*labeling, "--agreement"] if agreement else list(labeling)
    args = list(every_row)
    if rows is not None:
        args += ["--rows", str(rows)]
    if max_rows is not None:
        args += ["--max-rows", str(max_rows)]
        labeling += ["--max-rows", str(max_rows)]
    rerun = [*args, "--reset-triage"] if reset_triage else args
    if criteria is None:
        return _checks_result(dataset, directory, plan, triage.apply_check_labels(plan), args)
    settings = {
        "rows_file": str(rows) if rows is not None else None,
        "max_rows": max_rows,
        "concurrency": concurrency,
        "reset_triage": reset_triage,
    }
    ctx = current_context()
    defaults_used = [
        name
        for name, param in (
            ("judges", "judges"),
            ("max_rows", "max_rows"),
            ("concurrency", "concurrency"),
        )
        if not (ctx is not None and given(ctx, param))
    ]
    planned = plan.preview()
    planned["judges_source"] = council_source
    planned["rubric_file"] = str(rubric) if rubric is not None else None
    preview = with_settings(planned, settings, defaults_used)
    refused = None
    if template_task is not None:
        preview["warnings"] = [*preview.get("warnings", []), template_task]
        if not allow_template_rubric:
            refused = "--confirm is refused while the rubric's Task section is the template's"
            # In the result too, not only on stderr: an agent reads why and what to change.
            preview["refused"] = refused
            preview["refused_hint"] = template_hint
    hint = template_hint if refused is not None else None
    if refused is None and not plan.rows and not plan.checked:
        among = "row --rows names" if rows is not None else "accepted row"
        refused = "there is no row to review: " + (
            f"every {among} is labeled by you, the agent or a check"
            if plan.skipped_checked
            else f"every {among} is labeled by you or the agent"
            if plan.skipped_labeled
            else "the dataset has no accepted rows"
        )
        preview["refused"] = refused
    require_confirm(confirm, preview, shlex.join([*rerun, "--confirm"]), refused=refused, hint=hint)
    info(
        f"triage: {len(plan.pending)} judge calls for {len(plan.rows)} rows "
        f"({plan.votes_reused} votes reused)"
    )
    # Only the judges with votes left: a resumed council needs no other's key.
    with open_clients(plan.pending_judges) as clients:
        report = triage.run_triage(plan, clients, concurrency=concurrency, progress=info)
    if report.agreement is not None:
        return _agreement_result(dataset, plan, report, report.agreement, criteria, args, labeling)
    counts = report.counts()
    summary = dataset.summary()
    complete = report.complete
    data: dict[str, Any] = {
        "dataset_dir": display_path(dataset.path),
        "status": "complete" if complete else "incomplete",
        "rows_reviewed": len(plan.rows),
        "rows_skipped_labeled": plan.skipped_labeled + report.skipped_labeled,
        **({"rows_skipped_checked": plan.skipped_checked} if plan.skipped_checked else {}),
        **({"checks": _checks(plan, report)} if plan.checks is not None else {}),
        "kept": counts["keep"],
        "dropped": counts["drop"],
        "undecided": counts["undecided"],
        "too_long": report.too_long,
        "top_drop_reasons": report.top_drop_reasons(),
        "judges": list(plan.judges),
        "rubric_hash": criteria.sha256[: triage.RUBRIC_HASH_SHOWN],
        "calls_made": report.calls_made,
        "votes_cast": report.votes_cast,
        "calls_failed": len(report.failures),
        "calls_skipped": report.calls_skipped,
        "failures": triage.failures_shown(report.failures),
        "failure_counts": report.failure_counts(),
        "stale_judge_labels": len(report.stale_labels),
        "stale_judge_label_ids": list(report.stale_labels[: triage.FAILURES_SHOWN]),
        "labels": {
            "applied": report.labels.applied if report.labels else 0,
            "unchanged": report.labels.unchanged if report.labels else 0,
        },
        "dataset": {
            "accepted": summary["accepted"],
            "kept": summary["kept"],
            "dropped": summary["dropped"],
        },
        "judgments_file": display_path(dataset.path / triage.JUDGMENTS_FILE),
    }
    if complete:
        merge_next(data, dataset_next(dataset, directory, summary=summary))
        if checks is not None:
            finish = template("dataset", "triage", directory, "--rubric", "<RUBRIC_FILE>")
            data["next_steps"] = [
                step(
                    shlex.join(every_row),
                    "finish the triage over every row with the same rubric, judges and checks, "
                    "reusing the votes cast (paid; it previews first)",
                )
                if entry["command"] == finish
                else entry
                for entry in data["next_steps"]
            ]
        data["hint"] = (
            "review a sample of the decisions with the user (`dataset sample "
            "--include-dropped`) before training; correct any with `dataset label`"
        )
        return data
    hints: list[str] = []
    if report.failures:
        hints.append(
            "some judge calls produced no vote; rerun the same command to retry only those "
            "(the votes so far are kept)"
        )
    sampled = {row.id for row in plan.rows}
    if any(row_id not in sampled for row_id in report.stale_labels):
        args = every_row
    if report.stale_labels:
        hints.append(
            f"{len(report.stale_labels)} rows keep a judge label the current votes do not "
            "back (from a discarded rubric, outside --max-rows or --rows, or left undecided); run "
            "the next command to judge them again, or relabel them with `dataset label`"
        )
    state = pipeline_state(dataset, summary)
    data["pipeline"] = state.to_json()
    if state.warnings:
        data["warnings"] = list(state.warnings)
    data["next_command"] = shlex.join([*args, "--confirm"])
    data["hint"] = "; ".join(hints)
    return CommandResult(data, exit_code=1) if report.failures else data


def _refuse_judge_flags() -> None:
    """With ``--checks`` alone, a :class:`UsageError` for an option only judges use."""
    ctx = current_context()
    for name in ("judges", "concurrency", "reset_triage"):
        if ctx is not None and given(ctx, name):
            flag = flag_name(ctx.command, name)
            raise UsageError(
                f"{flag} needs --rubric or --rule (a run with --checks alone calls no judge)",
                hint=f"pass --rubric FILE or --rule TEXT too, or leave {flag} out",
            )


def _checks(plan: triage.TriagePlan, report: triage.TriageReport) -> dict[str, Any]:
    """The plan's ``checks`` block, with the check labels ``report`` recorded."""
    labels = report.check_labels
    return {
        **plan.checks_block(),
        "labels": {
            "applied": labels.applied if labels else 0,
            "unchanged": labels.unchanged if labels else 0,
        },
    }


def _checks_result(
    dataset: DatasetDir,
    directory: Path,
    plan: triage.TriagePlan,
    report: triage.TriageReport,
    args: list[str],
) -> dict[str, Any]:
    """The result of a run with ``--checks`` alone, which labeled the rows they fail."""
    summary = dataset.summary()
    stale = plan.stale_check_labels
    data: dict[str, Any] = {
        "dataset_dir": display_path(dataset.path),
        "mode": "checks",
        "rows_skipped_labeled": plan.skipped_labeled + report.skipped_labeled,
        "checks": _checks(plan, report),
        "stale_check_labels": len(stale),
        "stale_check_label_ids": stale[: triage.FAILURES_SHOWN],
        "dataset": {
            "accepted": summary["accepted"],
            "kept": summary["kept"],
            "dropped": summary["dropped"],
        },
    }
    if plan.references is not None:
        data["agreement"] = {"checks": plan.check_agreement()}
    merge_next(data, dataset_next(dataset, directory, summary=summary))
    if plan.rows:
        council = template(*args[len(PROGRAM) :], "--rubric", "<RUBRIC_FILE>")
        why = (
            "measure the judge council against your labels on the rows the checks pass"
            if plan.references is not None
            else "have the judge council review the rows the checks pass"
        )
        # It pins --checks, so it stands for the first council step the dataset's state offers.
        first = template("dataset", "triage", directory, "--rubric", "<RUBRIC_FILE>", "--max-rows")
        data["next_steps"] = [
            step(council, f"{why} (paid; it previews first)"),
            *(s for s in data["next_steps"] if not s["command"].startswith(f"{first} ")),
        ]
    if stale:
        data["hint"] = (
            f"{len(stale)} rows keep a check label every check now passes, and a label is never "
            "withdrawn: relabel them with `dataset label`, or add --rubric to have the council "
            "judge them"
        )
    return data


def _row_ids(path: Path) -> frozenset[str]:
    """The row ids in ``path``, one per line; blank lines are ignored."""
    return frozenset(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def _agreement_result(
    dataset: DatasetDir,
    plan: triage.TriagePlan,
    report: triage.TriageReport,
    agreement: triage.Agreement,
    criteria: triage.Criteria,
    args: list[str],
    labeling: list[str],
) -> dict[str, Any] | CommandResult:
    """The result of an ``--agreement`` run: the council against the user's labels.
    ``labeling`` is the triage that labels the rest: ``args`` without ``--agreement`` and
    without the ``--rows FILE`` that chose the labeled rows to compare."""
    data: dict[str, Any] = {
        "dataset_dir": display_path(dataset.path),
        "mode": "agreement",
        "status": "complete" if report.complete else "incomplete",
        "rows_reviewed": len(plan.rows),
        "agreement": agreement.to_json(),
        **({"checks": _checks(plan, report)} if plan.checks is not None else {}),
        "judges": list(plan.judges),
        "rubric_hash": criteria.sha256[: triage.RUBRIC_HASH_SHOWN],
        "calls_made": report.calls_made,
        "votes_cast": report.votes_cast,
        "calls_failed": len(report.failures),
        "calls_skipped": report.calls_skipped,
        "failures": triage.failures_shown(report.failures),
        "failure_counts": report.failure_counts(),
        "judgments_file": display_path(dataset.path / triage.JUDGMENTS_FILE),
    }
    if plan.checks is not None:
        data["agreement"]["checks"] = plan.check_agreement()
    if (noisy := triage.noisy_agreement(len(plan.rows))) is not None:
        data["warnings"] = [noisy]
    if report.failures:
        data["next_command"] = shlex.join([*args, "--confirm"])
        data["hint"] = (
            "some judge calls produced no vote; rerun the same command to retry only those "
            "(the votes so far are kept)"
        )
        return CommandResult(data, exit_code=1)
    data["next_command"] = shlex.join([*labeling, "--confirm"])
    data["next_steps"] = [
        step(
            shlex.join([*args, "--reset-triage"]),
            "after sharpening the rubric, measure again: --reset-triage discards every vote "
            "(it previews first)",
        )
    ]
    data["hint"] = (
        "go through the disagreements with the user: where the labels are right, sharpen the "
        "rubric and measure again (a changed rubric needs --reset-triage, which discards "
        "every vote); once the council agrees closely enough, drop --agreement to label the "
        "rest"
    )
    return data


def register(dataset_app: typer.Typer) -> None:
    """Add `triage` to the `dataset` command group."""
    dataset_app.command(**DOC.typer())(triage_rows)
