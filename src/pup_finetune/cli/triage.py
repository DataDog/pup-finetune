# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune dataset triage`: a judge council labels a dataset's rows against a rubric.

Judging is paid (one call per row and judge), so without ``--confirm`` it only previews the
calls. It reads and writes local files and calls a serverless inference API, with no
Datadog or other remote write, so ``PUP_READ_ONLY`` does not block it; ``--confirm`` is
still required, and never implied by ``PUP_AUTO_APPROVE``. The council, the votes file and
the labels live in :mod:`pup_finetune.curation.triage`.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from pup_finetune.cli._common import CommandResult, run_command
from pup_finetune.curation import triage
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.errors import UsageError, require_confirm
from pup_finetune.output import info

if TYPE_CHECKING:
    from pup_finetune.inference import ChatCompleter


def judges_base_url() -> str:
    """Where the judges are served: the default judge endpoint (Fireworks serverless)."""
    from pup_finetune.inference import JudgeConfig

    return JudgeConfig().base_url


@contextmanager
def open_client() -> Iterator[ChatCompleter]:
    """The judges' chat client: Fireworks serverless inference with the Fireworks key.

    Tests replace it with a fake.
    """
    from pup_finetune.inference import ChatClient, JudgeConfig

    with ChatClient.for_judge(JudgeConfig(base_url=judges_base_url())) as client:
        yield client


def _q(value: object) -> str:
    return shlex.quote(str(value))


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
            help=f"Comma-separated judge models (default: {','.join(triage.DEFAULT_JUDGES)}).",
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
    reset_triage: Annotated[
        bool,
        typer.Option(
            "--reset-triage",
            help="Discard every vote and the frozen rubric and judges first.",
        ),
    ] = False,
    print_template: Annotated[
        bool,
        typer.Option(
            "--print-template",
            help="Print a rubric template (Task / Keep / Drop / Examples) and exit.",
        ),
    ] = False,
    confirm: Annotated[
        bool, typer.Option("--confirm", help="Run the judges (paid); without it, only preview.")
    ] = False,
) -> dict[str, Any] | CommandResult | None:
    if print_template:
        typer.echo(triage.rubric_template(), nl=False)
        return None
    if directory is None:
        raise UsageError("pass the dataset directory DIR", hint="or --print-template")
    rules = rule or []
    criteria = triage.Criteria.build(
        rubric.read_text(encoding="utf-8") if rubric is not None else None, rules
    )
    council = triage.parse_judges(judges)
    dataset = DatasetDir(directory)
    plan = triage.plan_triage(dataset, criteria, council, max_rows=max_rows, reset=reset_triage)
    args = ["pup", "finetune", "dataset", "triage", str(directory)]
    if rubric is not None:
        args += ["--rubric", str(rubric)]
    for text in rules:
        args += ["--rule", text]
    if judges is not None:
        args += ["--judges", ",".join(council)]
    if concurrency != triage.DEFAULT_CONCURRENCY:
        args += ["--concurrency", str(concurrency)]
    every_row = list(args)
    if max_rows is not None:
        args += ["--max-rows", str(max_rows)]
    rerun = [*args, "--reset-triage"] if reset_triage else args
    preview = plan.preview(base_url=judges_base_url())
    require_confirm(confirm, preview, shlex.join([*rerun, "--confirm"]))
    info(
        f"triage: {len(plan.pending)} judge calls for {len(plan.rows)} rows "
        f"({plan.calls - len(plan.pending)} votes reused)"
    )
    with open_client() as client:
        report = triage.run_triage(plan, client, concurrency=concurrency, progress=info)
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
        "top_drop_reasons": report.top_drop_reasons(),
        "judges": list(plan.judges),
        "rubric_sha256": criteria.sha256,
        "calls_made": report.calls_made,
        "votes_cast": report.votes_cast,
        "calls_failed": len(report.failures),
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
    sample = f"pup finetune dataset sample {_q(directory)}"
    if complete:
        data["next_command"] = sample
        data["next_commands"] = [
            sample,
            f"pup finetune dataset push {_q(directory)} --name ...",
            f"pup finetune train --dataset {_q(directory)} ...",
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
            "--reset-triage. Row content is sent to the judges' inference API (Fireworks "
            "serverless, with the Fireworks API key). Without --confirm, previews the calls. "
            "Paid, but no Datadog write: allowed with PUP_READ_ONLY set. Start with "
            "--max-rows 20 to calibrate the rubric. --print-template prints a rubric "
            "template."
        ),
    )(triage_rows)
