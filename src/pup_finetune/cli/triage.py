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
"""

from __future__ import annotations

import shlex
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from pup_finetune.cli._common import CommandResult, run_command
from pup_finetune.cli._dirs import require_dataset_dir
from pup_finetune.cli._flags import require_alone, require_enablers
from pup_finetune.cli.dataset import dataset_next
from pup_finetune.curation import triage
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.errors import UsageError, require_confirm
from pup_finetune.next_steps import PROGRAM, step
from pup_finetune.output import info

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
            help="A rubric file (Markdown; see --print-template).",
            exists=True,
            dir_okay=False,
            readable=True,
            show_default=False,
        ),
    ] = None,
    rule: Annotated[
        list[str] | None,
        typer.Option(
            "--rule",
            metavar="TEXT",
            help="A rule to judge by; repeat for more.",
            show_default=False,
        ),
    ] = None,
    judges: Annotated[
        str | None,
        typer.Option(
            "--judges",
            metavar="M1,M2",
            help="Comma-separated judge models, each anthropic/<model> (ANTHROPIC_API_KEY), "
            "openai/<model> (OPENAI_API_KEY), or otherwise a Fireworks model id such as "
            "accounts/fireworks/models/<model> (FIREWORKS_API_KEY); providers can be mixed "
            f"(default: {','.join(triage.DEFAULT_JUDGES)}).",
            show_default=False,
        ),
    ] = None,
    concurrency: Annotated[
        int, typer.Option(min=1, max=64, help="Judge calls made at once.")
    ] = triage.DEFAULT_CONCURRENCY,
    max_rows: Annotated[
        int | None,
        typer.Option(
            "--max-rows",
            min=1,
            help="Review at most this many rows (a fixed sample, e.g. 20 to calibrate the "
            "rubric first; a larger run later reuses its votes).",
            show_default=False,
        ),
    ] = None,
    agreement: Annotated[
        bool,
        typer.Option(
            "--agreement",
            help="Review the rows you or the agent labeled instead, label nothing, and report "
            "how often the council agrees with those labels (rate, kappa, each judge, the "
            "rows they disagree on).",
        ),
    ] = False,
    reset_triage: Annotated[
        bool,
        typer.Option(
            "--reset-triage",
            help="Discard every vote (even an unreadable judgments.jsonl) and the frozen "
            "rubric and judges first.",
        ),
    ] = False,
    print_template: Annotated[
        bool,
        typer.Option(
            "--print-template",
            help="Print a rubric template (Task / Keep / Drop / Examples) and exit; it takes "
            "nothing else.",
        ),
    ] = False,
    allow_template_rubric: Annotated[
        bool,
        typer.Option(
            "--allow-template-rubric",
            help="With --rubric: judge by a rubric whose Task section is still the "
            "template's (refused otherwise: the judges would not know what the agent is for).",
        ),
    ] = False,
    confirm: Annotated[
        bool, typer.Option("--confirm", help="Run the judges (paid); without it, only preview.")
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
    rubric_text = rubric.read_text(encoding="utf-8") if rubric is not None else None
    criteria = triage.Criteria.build(rubric_text, rules)
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
    council = triage.parse_judges(judges)
    _check_available(council)
    require_dataset_dir(directory)
    dataset = DatasetDir(directory)
    plan = triage.plan_triage(
        dataset, criteria, council, max_rows=max_rows, reset=reset_triage, agreement=agreement
    )
    args = [*PROGRAM, "dataset", "triage", str(directory)]
    if rubric is not None:
        args += ["--rubric", str(rubric)]
    for text in rules:
        args += ["--rule", text]
    if judges is not None:
        args += ["--judges", ",".join(council)]
    if concurrency != triage.DEFAULT_CONCURRENCY:
        args += ["--concurrency", str(concurrency)]
    if agreement:
        args.append("--agreement")
    if allow_template_rubric and template_task is not None:
        args.append("--allow-template-rubric")
    every_row = list(args)
    if max_rows is not None:
        args += ["--max-rows", str(max_rows)]
    rerun = [*args, "--reset-triage"] if reset_triage else args
    preview = plan.preview()
    refused = None
    if template_task is not None:
        preview["warnings"] = [*preview.get("warnings", []), template_task]
        if not allow_template_rubric:
            refused = "--confirm is refused while the rubric's Task section is the template's"
    require_confirm(
        confirm,
        preview,
        shlex.join([*rerun, "--confirm"]),
        refused=refused,
        hint=template_hint if refused is not None else None,
    )
    info(
        f"triage: {len(plan.pending)} judge calls for {len(plan.rows)} rows "
        f"({plan.votes_reused} votes reused)"
    )
    with open_clients(plan.judges) as clients:
        report = triage.run_triage(plan, clients, concurrency=concurrency, progress=info)
    if report.agreement is not None:
        return _agreement_result(dataset, plan, report, report.agreement, criteria, args)
    counts = report.counts()
    summary = dataset.summary()
    complete = report.complete
    data: dict[str, Any] = {
        "dataset_dir": str(dataset.path.resolve()),
        "status": "complete" if complete else "incomplete",
        "rows_reviewed": len(plan.rows),
        "rows_skipped_labeled": plan.skipped_labeled + report.skipped_labeled,
        "kept": counts["keep"],
        "dropped": counts["drop"],
        "undecided": counts["undecided"],
        "too_long": report.too_long,
        "top_drop_reasons": report.top_drop_reasons(),
        "judges": list(plan.judges),
        "rubric_sha256": criteria.sha256,
        "calls_made": report.calls_made,
        "votes_cast": report.votes_cast,
        "calls_failed": len(report.failures),
        "calls_skipped_too_long": report.calls_skipped,
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
        "judgments_file": str((dataset.path / triage.JUDGMENTS_FILE).resolve()),
    }
    if complete:
        data.update(dataset_next(dataset, directory, accepted=summary["accepted"]))
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
        # Rows outside the --max-rows sample are judged only by a run over every row.
        args = every_row
    if report.stale_labels:
        hints.append(
            f"{len(report.stale_labels)} rows keep a judge label the current votes do not "
            "back (from a discarded rubric, outside --max-rows, or left undecided); run "
            "the next command to judge them again, or relabel them with `dataset label`"
        )
    data["next_command"] = shlex.join([*args, "--confirm"])
    data["hint"] = "; ".join(hints)
    return CommandResult(data, exit_code=1) if report.failures else data


def _agreement_result(
    dataset: DatasetDir,
    plan: triage.TriagePlan,
    report: triage.TriageReport,
    agreement: triage.Agreement,
    criteria: triage.Criteria,
    args: list[str],
) -> dict[str, Any] | CommandResult:
    """The result of an ``--agreement`` run: the council against the user's labels."""
    data: dict[str, Any] = {
        "dataset_dir": str(dataset.path.resolve()),
        "mode": "agreement",
        "status": "complete" if report.complete else "incomplete",
        "rows_reviewed": len(plan.rows),
        "agreement": agreement.to_json(),
        "judges": list(plan.judges),
        "rubric_sha256": criteria.sha256,
        "calls_made": report.calls_made,
        "votes_cast": report.votes_cast,
        "calls_failed": len(report.failures),
        "calls_skipped_too_long": report.calls_skipped,
        "failures": triage.failures_shown(report.failures),
        "failure_counts": report.failure_counts(),
        "judgments_file": str((dataset.path / triage.JUDGMENTS_FILE).resolve()),
    }
    without = [word for word in args if word != "--agreement"]
    if report.failures:
        data["next_command"] = shlex.join([*args, "--confirm"])
        data["hint"] = (
            "some judge calls produced no vote; rerun the same command to retry only those "
            "(the votes so far are kept)"
        )
        return CommandResult(data, exit_code=1)
    data["next_command"] = shlex.join([*without, "--confirm"])
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
    dataset_app.command(
        "triage",
        short_help="Label rows with a council of judge models and a rubric (paid).",
        help=(
            "Have each judge model review each whole row of DIR (every turn and its tools, "
            "since training imitates every assistant turn) against --rubric FILE and/or "
            "--rule TEXT, and vote keep or drop. A row is kept only on a strict majority of "
            "keep votes; a tie drops it; a row missing a vote (a failed call) stays "
            "undecided and unlabeled. Decisions are recorded as labels by judge; rows the "
            "agent or a person labeled are not reviewed. Votes are appended to "
            "DIR/judgments.jsonl as they arrive, so rerunning the same command resumes and "
            "retries only failed calls. The rubric and the judges are frozen in "
            "DIR/triage.json once votes exist: change them in a new directory or with "
            "--reset-triage. Row content is sent to each judge's provider: Fireworks "
            "serverless (the default judges), Anthropic or OpenAI, with that provider's API "
            "key; the preview names every one. Without --confirm, previews the calls. "
            "Paid, but no Datadog write: allowed with PUP_READ_ONLY set. Start with "
            "--max-rows 20 to calibrate the rubric, or --agreement to measure the council "
            "against rows you labeled first. --print-template prints a rubric template."
        ),
    )(triage_rows)
