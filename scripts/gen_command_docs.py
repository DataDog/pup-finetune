#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Generate the command reference and the exclusion-code table from the code.

Usage: uv run python scripts/gen_command_docs.py [--check]

Writes ``docs/commands.md`` from the Typer app (every visible command, its full help,
arguments, options, examples and safety, the safety table built from every command's
``Safety`` record in ``pup_finetune.cli._presentation``, and the exit-code rule and error
codes from ``pup_finetune.errors.ERROR_CODES``), the exclusion-code table between
the markers in ``docs/data-format.md`` from ``pup_finetune.data.exclusions``, and the table
of what each step needs between the markers in ``README.md`` from what ``doctor`` checks
(``pup_finetune.readiness``, the provider and credential registries), and the table of
supported models there from the registered providers' model profiles. With ``--check``,
writes nothing and exits 1 when any of them is out of date.

The output is deterministic: it is built from the commands' parameters, not from rendered
``--help`` (which depends on the terminal width, color settings and ``GITHUB_ACTIONS``).
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMANDS_DOC = REPO_ROOT / "docs" / "commands.md"
DATA_FORMAT_DOC = REPO_ROOT / "docs" / "data-format.md"
README = REPO_ROOT / "README.md"

PROG = "pup finetune"
EXCLUSIONS_BEGIN = "<!-- BEGIN GENERATED: exclusion codes (scripts/gen_command_docs.py) -->"
EXCLUSIONS_END = "<!-- END GENERATED: exclusion codes -->"
REQUIREMENTS_BEGIN = "<!-- BEGIN GENERATED: what each step needs (scripts/gen_command_docs.py) -->"
REQUIREMENTS_END = "<!-- END GENERATED: what each step needs -->"
MODELS_BEGIN = "<!-- BEGIN GENERATED: supported models (scripts/gen_command_docs.py) -->"
MODELS_END = "<!-- END GENERATED: supported models -->"

_DEFAULT_NOTE = re.compile(r"\s*\[dim\]\\\[default: (?P<default>.*?)\]\[/dim\]")
"""A default written into an option's help (``_presentation.default_note``)."""

_TYPE_NAMES = {
    "STR": "TEXT",
    "STRING": "TEXT",
    "INT": "INTEGER",
    "FLOAT": "FLOAT",
    "BOOL": "BOOLEAN",
}


def iter_commands(command: Any, ctx: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, Any]]:
    """Every visible leaf command under ``command``, as (``"group name"``, command)."""
    if hasattr(command, "list_commands"):
        for name in command.list_commands(ctx):
            sub = command.get_command(ctx, name)
            if sub is not None and not sub.hidden:
                yield from iter_commands(sub, ctx, (*path, name))
    elif path:
        yield " ".join(path), command


def ordered(commands: Sequence[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """``commands`` in workflow order (``COMMAND_ORDER``); any not listed follow, by name."""
    from pup_finetune.cli._presentation import COMMAND_ORDER

    rank = {name: index for index, name in enumerate(COMMAND_ORDER)}
    return sorted(commands, key=lambda item: (rank.get(item[0], len(COMMAND_ORDER)), item[0]))


def split_default(help_text: str) -> tuple[str, str | None]:
    """``help_text`` without the default written into it, and that default (or ``None``)."""
    found = _DEFAULT_NOTE.search(help_text)
    if found is None:
        return help_text, None
    rest = (help_text[: found.start()] + help_text[found.end() :]).strip()
    return rest, found["default"].replace("\\[", "[")


def md(text: str, *, cell: bool = False) -> str:
    """``text`` safe in Markdown prose (or a table ``cell``); code spans are kept as written."""
    parts = text.split("`")
    for index in range(0, len(parts), 2):  # outside code spans
        parts[index] = (
            parts[index]
            .replace("\\", "\\\\")
            .replace("*", "\\*")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
    joined = "`".join(parts)
    return joined.replace("|", "\\|") if cell else joined


def paragraphs(text: str) -> list[str]:
    """``text`` as paragraphs, each on one line."""
    return [" ".join(block.split()) for block in re.split(r"\n\s*\n", text.strip()) if block]


def value_of(param: Any) -> str:
    """What an option takes, e.g. ``N (x>=1)`` or ``omit|preserve``; empty for flags."""
    if getattr(param, "is_flag", False):
        return ""
    kind = param.type
    if hasattr(kind, "choices"):
        value = "|".join(str(choice) for choice in kind.choices)
    elif param.metavar:
        value = str(param.metavar)
    elif hasattr(kind, "file_okay") and hasattr(kind, "dir_okay"):
        value = "FILE" if not kind.dir_okay else "DIR" if not kind.file_okay else "PATH"
    else:
        name = str(getattr(kind, "name", "") or "").upper()
        if name.endswith(" RANGE"):
            name = name.removesuffix(" RANGE")
        value = _TYPE_NAMES.get(name, name or "VALUE")
    if hasattr(kind, "min") and hasattr(kind, "max"):
        value += f" ({_range(kind)})"
    if param.multiple:
        value += ", repeatable"
    return value


def _range(kind: Any) -> str:
    low = None if kind.min is None else f"{kind.min}{'<' if kind.min_open else '<='}"
    high = None if kind.max is None else f"{'<' if kind.max_open else '<='}{kind.max}"
    if low is not None and high is not None:
        return f"{low}x{high}"
    if low is not None:
        return f"x{'>' if kind.min_open else '>='}{kind.min}"
    return f"x{high}"


def default_of(param: Any) -> str:
    if param.required:
        return "required"
    written = split_default(param.help or "")[1]
    if written is not None:
        return md(written, cell=True)
    if getattr(param, "is_flag", False) or not getattr(param, "show_default", True):
        return ""
    default = param.default
    if default is None or callable(default) or default == () or default == []:
        return ""
    return f"`{default}`"


def usage(name: str, command: Any) -> str:
    words = [PROG, name]
    if any(param.param_type_name == "option" for param in command.params):
        words.append("[OPTIONS]")
    for param in command.params:
        if param.param_type_name != "argument":
            continue
        metavar = str(param.metavar or param.name.upper())
        word = metavar if param.required else f"[{metavar}]"
        words.append(f"{word}..." if param.nargs == -1 else word)
    return " ".join(words)


def render_command(name: str, command: Any) -> list[str]:
    from pup_finetune.cli._presentation import DOCS

    lines = [f"## `{PROG} {name}`", ""]
    for paragraph in paragraphs(command.help or command.short_help or ""):
        lines += [md(paragraph), ""]
    lines += ["```", usage(name, command), "```", ""]
    arguments = [p for p in command.params if p.param_type_name == "argument"]
    options = [p for p in command.params if p.param_type_name == "option" and not p.hidden]
    if arguments:
        lines += ["| Argument | Description |", "| --- | --- |"]
        for param in arguments:
            metavar = str(param.metavar or param.name.upper())
            help_text, shown_default = split_default(param.help or "")
            required = ""
            if not param.required:
                required = " Optional"
                required += f"; default: {md(shown_default, cell=True)}." if shown_default else "."
            lines.append(f"| `{metavar}` | {md(help_text, cell=True)}{required} |")
        lines.append("")
    if options:
        lines += ["| Option | Value | Default | Description |", "| --- | --- | --- | --- |"]
        shown_panel = None
        for param in options:
            panel = getattr(param, "rich_help_panel", None)
            if panel and panel != shown_panel:  # the option panels of --help, as rows
                lines.append(f"| **{md(panel, cell=True)}** | | | |")
                shown_panel = panel
            names = ", ".join(f"`{opt}`" for opt in [*param.opts, *param.secondary_opts])
            value = value_of(param)
            cell = f"`{value}`".replace("|", "\\|") if value else ""
            help_text = md(split_default(param.help or "")[0], cell=True)
            lines.append(f"| {names} | {cell} | {default_of(param)} | {help_text} |")
        lines.append("")
    doc = DOCS.get(name)
    if doc is not None:
        lines += ["Examples:", "", "```sh", *doc.examples, "```", ""]
        lines += [f"Safety: {md(doc.safety.footer())}.", ""]
    return lines


def render_safety_table(names: Sequence[str]) -> list[str]:
    """The table of what every command costs and where its data goes."""
    from pup_finetune.cli._presentation import DOCS

    lines = [
        "| Command | Without `--confirm` | With `--confirm` | Data goes to | `PUP_READ_ONLY` |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in names:
        doc = DOCS.get(name)
        if doc is None:
            continue
        cells = [md(cell, cell=True) for cell in doc.safety.table_row()]
        lines.append(f"| `{PROG} {name}` | {' | '.join(cells)} |")
    return lines


def render_errors() -> list[str]:
    """The exit-code rule, the error line and its codes, and the outputs that are not JSON."""
    from pup_finetune.errors import ERROR_CODES

    lines = [
        "## Exit codes and errors",
        "",
        "- `0`: success.",
        "- `2`: you can fix it by changing the command or the directory: a usage or "
        "validation error, a preview that needs `--confirm`, a paid run or a remote write "
        "while `PUP_READ_ONLY` is set, a wrong, missing or non-empty directory, a directory that "
        "records other settings, or a run in the wrong phase (still training, not completed, "
        "already deployed).",
        "- `1`: something outside failed or local state is broken: pup, Datadog or a "
        "provider failed, a local file is unreadable or inconsistent, another process holds "
        "the directory's lock, or Ctrl-C or SIGTERM stopped the command.",
        "",
        "An error leaves stdout empty and prints two lines on stderr: "
        "`Error [<code>]: <message>`, then `Hint: <what to do>` when there is a hint, for "
        "example `Error [run_dir_not_empty]: run1 already holds run ft-... (completed)`. "
        "The code is stable and listed below; the message is for people and may change. "
        "Usage errors that Click finds (an unknown command or option, a missing or malformed "
        "value) print the same way with the code `usage`, name every missing required option "
        "at once (`Error [usage]: train needs --dataset and --model`), keep Click's "
        "suggestions (`Did you mean ...`, `Possible options: ...`) and point to "
        f"`{PROG} help <command>`. A preview is not a failure: it prints its JSON on stdout "
        '(`"status": "confirmation_required"`, with `next_command`), then on stderr a card '
        "of at most seven lines generated from it (`Preview only, nothing was run: ...`, then "
        "the data, settings, recipients, who bills it and the cost; a longer card drops the "
        "settings first, never the cost or a refusal), its warnings and `Next: <command>`, and "
        "exits `2`; its code is `confirmation_required`. A command that returns a result with "
        "a nonzero exit, such as `doctor` when a required check fails, still prints the result "
        "on stdout. A run records the error that "
        "stopped it, with the same `code`, under `error` in `state.json`.",
        "",
        "| Code | Exit | Meaning |",
        "| --- | --- | --- |",
    ]
    for code, (exit_code, meaning) in ERROR_CODES.items():
        lines.append(f"| `{code}` | `{exit_code}` | {md(meaning, cell=True)} |")
    lines += [
        "",
        "These print text on stdout instead of one JSON document: `help` (alone or with a "
        "command), `--help` or `-h` after any command, `--version`, "
        "`dataset triage --print-template`, and a group run without a command "
        f"(`{PROG}`, `{PROG} dataset`, `{PROG} auth`, ...), which prints the group's help "
        "and exits `2`.",
        "",
    ]
    return lines


def summary_of(command: Any) -> str:
    """The command's one-line summary: its short help, else its help's first sentence."""
    if command.short_help:
        return " ".join(command.short_help.split())
    first = paragraphs(command.help or "")
    return first[0] if first else ""


def render_commands() -> str:
    """The content of ``docs/commands.md``."""
    import typer
    import typer.main

    from pup_finetune.cli import app

    root = typer.main.get_command(app)
    ctx = typer.Context(root, info_name=PROG)
    commands = ordered(list(iter_commands(root, ctx)))
    lines = [
        "<!-- Generated by scripts/gen_command_docs.py from the CLI. Do not edit by hand. -->",
        "",
        "# Command reference",
        "",
        f"Every `{PROG}` command, with its arguments and options. `{PROG}` builds a training "
        "dataset from your Datadog Agent Observability traces, fine-tunes a model on it, "
        "evaluates it and deploys it. Each command prints one JSON "
        "result on stdout and its progress on stderr; "
        "[Exit codes and errors](#exit-codes-and-errors) has the exit-code rule, the error "
        "codes and the few outputs that are not JSON. Paid and remote-changing commands "
        "preview unless you pass `--confirm`. A result's `next_command` is a command that runs exactly as written "
        "(after your yes, when it ends in `--confirm`), or `null`; it never holds a "
        "placeholder or `...`. Alternatives, and commands that need a value only you can "
        'supply, are in `next_steps`, a list of `{"command", "why"}` whose commands mark '
        "such a value as `<UPPER_CASE>`, for example `--name <NAME>`. "
        "Run `pup finetune help <command>` for the same text in your terminal "
        "(under a coding agent, Pup answers `--help` after `pup finetune` with its own help).",
        "",
        "Outside Pup, run the same commands as `pup-finetune <command>`.",
        "",
        *render_errors(),
        "## Commands",
        "",
    ]
    for name, command in commands:
        anchor = re.sub(r"[^a-z0-9 -]", "", f"{PROG} {name}".lower()).replace(" ", "-")
        summary = md(summary_of(command))
        lines.append(f"- [`{PROG} {name}`](#{anchor}): {summary}")
    lines += [
        "",
        "## Safety",
        "",
        "What each command does without and with `--confirm`, where its data goes, and what "
        "`PUP_READ_ONLY` allows. `--confirm` is never implied by `PUP_AUTO_APPROVE` or agent "
        "mode. Every preview runs with `PUP_READ_ONLY` set; `--confirm` is refused where the "
        "table says so: training, evaluating and remote writes (`dataset triage` writes "
        "nothing remote, so its paid `--confirm` is allowed, and a training or evaluation that "
        "stays offline, on the fake provider and judge, is allowed too). Each command's help "
        "ends with "
        "the same facts in one line.",
        "",
        *render_safety_table([name for name, _ in commands]),
        "",
    ]
    for name, command in commands:
        lines += render_command(name, command)
    return "\n".join(lines).rstrip("\n") + "\n"


def render_exclusions() -> str:
    """The exclusion-code table that goes between the markers in ``docs/data-format.md``."""
    from pup_finetune.data.exclusions import EXPLANATIONS, ExclusionCode
    from pup_finetune.data.schema import RowErrorCode

    lines = ["| Code | Kind | Why the row was left out |", "| --- | --- | --- |"]
    for code in [*RowErrorCode, *ExclusionCode]:
        kind = "schema" if isinstance(code, RowErrorCode) else "dataset"
        explanation = EXPLANATIONS[code]
        text = md(explanation[0].upper() + explanation[1:], cell=True)
        lines.append(f"| `{code.value}` | {kind} | {text}. |")
    return "\n".join(lines)


def between(text: str, begin: str, end: str, content: str, *, name: str) -> str:
    """``text`` with what lies between the markers ``begin`` and ``end`` replaced."""
    start, stop = text.find(begin), text.find(end)
    if start < 0 or stop < start:
        raise ValueError(f"{name} lacks the markers {begin!r} and {end!r}")
    return f"{text[: start + len(begin)]}\n{content}\n{text[stop:]}"


def with_exclusions(text: str) -> str:
    """``text`` (``docs/data-format.md``) with its generated table replaced."""
    return between(
        text, EXCLUSIONS_BEGIN, EXCLUSIONS_END, render_exclusions(), name=DATA_FORMAT_DOC.name
    )


_DATASET_STEPS = (
    "dataset pull",
    "dataset show",
    "dataset push",
)
"""The dataset steps of :func:`~pup_finetune.readiness.ready_map`, in workflow order, but
for ``dataset triage``, whose only need is the judges."""


def one_of(items: Sequence[str]) -> str:
    """``items`` as a choice: ``one of a, b or c``."""
    return items[0] if len(items) == 1 else f"one of {', '.join(items[:-1])} or {items[-1]}"


def requirement_rows() -> list[tuple[str, str, str, str]]:
    """What each step needs, as (step, Datadog login, API key, extra) rows.

    Read from what ``doctor`` checks: every check fails but the judge-only providers'
    (Anthropic, OpenAI), so each step of :func:`~pup_finetune.readiness.ready_map` is
    blocked by exactly what it needs besides its judges (deploying on each provider that
    deploys included). The judges take one key, of any judge provider, so they get a row of
    their own. Development-only providers are left out.
    """
    from pup_finetune import credentials
    from pup_finetune.doctor import CheckResult
    from pup_finetune.judges import JUDGE_PACKAGES, PROVIDERS
    from pup_finetune.providers import Package, provider_infos
    from pup_finetune.readiness import DATADOG_CHECKS, ready_map
    from pup_finetune.runtime import Runtime

    runtime = Runtime()  # not development mode, not read-only
    infos = provider_infos(runtime)
    packages: dict[str, Package] = {
        f"{info.name}:{package.key}": package for info in infos for package in info.packages
    }
    packages |= {f"{owner}:{package.key}": package for owner, package in JUDGE_PACKAGES.items()}
    keys = {f"{spec.provider}:key": spec.env_var for spec in credentials.all_specs()}
    provider_keys = {info.credential for info in infos}
    judges_met = {name for name in keys if name.partition(":")[0] not in provider_keys}
    judges_met |= {f"{owner}:{package.key}" for owner, package in JUDGE_PACKAGES.items()}
    checks = [
        CheckResult(name=name, status="pass" if name in judges_met else "fail", detail="")
        for name in [*DATADOG_CHECKS, *keys, *packages]
    ]
    steps = ready_map(checks, runtime)

    def cells(blocked_by: Sequence[str]) -> tuple[str, str, str]:
        datadog = "yes" if any(name in DATADOG_CHECKS for name in blocked_by) else ""
        env_vars = [keys[name] for name in blocked_by if name in keys]
        extras = dict.fromkeys(packages[name].extra for name in blocked_by if name in packages)
        return (
            datadog,
            ", ".join(f"`{var}`" for var in env_vars),
            ", ".join(f"`{extra}`" for extra in extras),
        )

    def rows_of(
        labeled: Sequence[tuple[str, Sequence[str]]], where: str = ""
    ) -> list[tuple[str, str, str, str]]:
        """``labeled`` steps as rows (on ``where``), adjacent ones with the same needs in one."""
        merged: list[tuple[list[str], tuple[str, str, str]]] = []
        for label, blocked_by in labeled:
            needs = cells(blocked_by)
            if merged and merged[-1][1] == needs:
                merged[-1][0].append(label)
            else:
                merged.append(([label], needs))
        suffix = f" on {where}" if where else ""
        return [(", ".join(labels) + suffix, *needs) for labels, needs in merged]

    labels = {
        "dataset pull": "`dataset pull`, `fetch`",
        "dataset show": "`dataset import`, `show`, `sample`, `label`; `status`, `runs`",
    }
    rows = rows_of(
        [(labels.get(name, f"`{name}`"), steps[name]["blocked_by"]) for name in _DATASET_STEPS]
    )
    for info in infos:
        labeled: list[tuple[str, Sequence[str]]] = [
            ("`train`", steps[f"train on {info.name}"]["blocked_by"]),
            ("`evaluate`, `benchmark`", steps[f"evaluate on {info.name}"]["blocked_by"]),
        ]
        if info.deploys:
            labeled.append(("`deploy`, `undeploy`", steps[f"deploy on {info.name}"]["blocked_by"]))
        rows += rows_of(labeled, info.label)
    judges = sorted(PROVIDERS.values(), key=lambda judge: keys[f"{judge.credential}:key"])
    rows.append(
        (
            "Judges for `dataset triage`, `evaluate` and `benchmark`",
            "",
            one_of([f"`{keys[f'{judge.credential}:key']}`" for judge in judges]),
            ", ".join(
                f"`{JUDGE_PACKAGES[judge.credential].extra}` for {judge.name}"
                for judge in judges
                if judge.credential in JUDGE_PACKAGES
            ),
        )
    )
    return rows


def render_requirements() -> str:
    """The table of what each step needs that goes between the markers in ``README.md``."""
    lines = [
        "| Step | Datadog login | API key | Extra |",
        "| --- | --- | --- | --- |",
    ]
    for step, datadog, key, extra in requirement_rows():
        lines.append(f"| {step} | {datadog or '-'} | {key or '-'} | {extra or '-'} |")
    return "\n".join(lines)


def with_requirements(text: str) -> str:
    """``text`` (``README.md``) with its generated table replaced."""
    return between(
        text, REQUIREMENTS_BEGIN, REQUIREMENTS_END, render_requirements(), name=README.name
    )


def render_models() -> str:
    """The table of supported models that goes between the markers in ``README.md``.

    Read from the registry's model profiles (each alias with its names, the Hugging Face name
    last), one row per Hugging Face model, in the order the providers (in registration order)
    list them, with a column per provider. Development-only providers are left out.
    """
    from pup_finetune.providers import provider_infos
    from pup_finetune.runtime import Runtime

    infos = provider_infos(Runtime())  # not development mode
    aliases: dict[str, dict[str, None]] = {}
    offered: dict[str, set[str]] = {}
    for info in infos:
        for alias, names in info.models.items():
            aliases.setdefault(names[-1], {})[alias] = None
            offered.setdefault(names[-1], set()).add(info.name)
    lines = [
        "| Model | `--model` | " + " | ".join(info.label for info in infos) + " |",
        "| --- | --- | " + " | ".join("---" for _ in infos) + " |",
    ]
    for model, model_aliases in aliases.items():
        names = ", ".join(f"`{alias}`" for alias in model_aliases)
        cells = " | ".join("yes" if info.name in offered[model] else "-" for info in infos)
        lines.append(f"| `{model}` | {names} | {cells} |")
    return "\n".join(lines)


def with_models(text: str) -> str:
    """``text`` (``README.md``) with its generated table of supported models replaced."""
    return between(text, MODELS_BEGIN, MODELS_END, render_models(), name=README.name)


def expected() -> dict[Path, str]:
    """Each generated file and the content it should have."""
    return {
        COMMANDS_DOC: render_commands(),
        DATA_FORMAT_DOC: with_exclusions(DATA_FORMAT_DOC.read_text(encoding="utf-8")),
        README: with_models(with_requirements(README.read_text(encoding="utf-8"))),
    }


def stale() -> list[Path]:
    """The generated files whose content differs from :func:`expected`."""
    out = []
    for path, content in expected().items():
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current != content:
            out.append(path)
    return out


def main(argv: Sequence[str] = ()) -> int:
    if list(argv) == ["--check"]:
        paths = stale()
        for path in paths:
            print(
                f"{path.relative_to(REPO_ROOT)} is out of date; run "
                "`uv run python scripts/gen_command_docs.py`",
                file=sys.stderr,
            )
        return 1 if paths else 0
    if argv:
        print(__doc__, file=sys.stderr)
        return 2
    for path, content in expected().items():
        path.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
