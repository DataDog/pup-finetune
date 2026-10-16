# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The compare target of ``evaluate --compare-model``: another model, replayed like base and tuned.

The compare model is named as a judge is (:func:`~pup_finetune.judges.parse_judge`:
``anthropic/<model>``, ``openai/<model>``, or a Fireworks model id or alias), so it runs on
the same clients, keys and endpoints. Frontier models think before they answer, and
thinking counts against the token limit, so the compare target gets
:data:`COMPARE_MAX_TOKENS` rather than ``--max-tokens``, and the provider's lowest reasoning
effort, as a judge does. Both are recorded in the evaluation directory's ``compare.json``
(:meth:`CompareTarget.identity`), frozen when the target is first attached.

:class:`ChatSampler` makes any :class:`~pup_finetune.inference.ChatCompleter` a
:class:`~pup_finetune.providers.base.Sampler`; :class:`FakeCompareSampler` is the offline
compare model of development mode (:data:`FAKE_COMPARE_MODEL`).

Some recorded prefixes are requests a chat API refuses every time (:func:`refused_prefix`,
by provider: :data:`REFUSED_SHAPES`), so a retry could never answer them. Those cases are
skipped for the compare target (:func:`compare_skips`): never sent, counted per reason, and
left out of its rates and of tuned minus compare.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import takewhile
from pathlib import Path
from typing import Any, Final, Literal

from pup_finetune.anthropic_messages import anthropic_messages
from pup_finetune.data.split import split_position
from pup_finetune.display import joined, quoted
from pup_finetune.errors import SettingsConflict, StateError, UsageError
from pup_finetune.evaluation.cases import ReplayCase
from pup_finetune.inference import ChatCompleter, JudgeConfig
from pup_finetune.output import info
from pup_finetune.providers.base import Generation, Sampler
from pup_finetune.rendering.base import ChatMessage, ToolSpec
from pup_finetune.runtime import Runtime
from pup_finetune.storage import (
    atomic_write_json,
    canonical_json,
    finite_json,
    free_sibling,
    identity_changes,
    read_json,
)

__all__ = [
    "COMPARE_MAX_TOKENS",
    "COMPARE_TIMEOUT_S",
    "FAKE_COMPARE_MODEL",
    "REFUSED_SHAPES",
    "SKIP_REASONS",
    "ChatSampler",
    "CompareTarget",
    "FakeCompareSampler",
    "SkipReason",
    "attach_compare",
    "compare_conflict",
    "compare_skips",
    "open_compare_sampler",
    "probe",
    "refused_prefix",
    "require_sent",
    "resolve_compare",
    "skipped_warning",
    "stored_compare",
]

COMPARE_MAX_TOKENS: Final = 16000
"""The compare model's limit per answer, thinking included."""
COMPARE_TIMEOUT_S: Final = 600.0
"""Each attempt's limit: a frontier model writing up to :data:`COMPARE_MAX_TOKENS` takes far
longer than a judge's verdict."""
FAKE_COMPARE_MODEL: Final = "fake-compare"
"""The offline compare model, for development mode only."""
FAKE_COMPARE_AGREEMENT: Final = 0.9
"""The share of cases the fake compare model answers with the reference."""
FAKE_COMPARE_ANSWER: Final = "I cannot decide from the recorded context."
PROBE_PROMPT: Final = "Reply with the word OK."
PROBE_TOOL: Final[ToolSpec] = {
    "type": "function",
    "function": {
        "name": "noop",
        "description": "Does nothing.",
        "parameters": {"type": "object", "properties": {}},
    },
}

CompareSource = Literal["flag", "evaluation directory"]

SkipReason = Literal[
    "no_messages", "unanswered_tool_call", "orphan_tool_result", "ends_on_assistant"
]
"""Why a compare model's API refuses a case's prefix on every try."""
SKIP_REASONS: Final[Mapping[str, str]] = {
    "no_messages": "have nothing to send before the recorded action",
    "unanswered_tool_call": "hold a tool call with no tool result right after it",
    "orphan_tool_result": "hold a tool result that answers no tool call right before it",
    "ends_on_assistant": "end on an assistant turn, a prefill the model refuses",
}
"""Each reason as it reads after a count of cases."""
REFUSED_SHAPES: Final[Mapping[str, tuple[SkipReason, ...]]] = {
    "openai": ("no_messages", "unanswered_tool_call", "orphan_tool_result"),
    "anthropic": ("no_messages", "unanswered_tool_call", "orphan_tool_result", "ends_on_assistant"),
}
"""The prefixes each compare provider's API refuses, in the order a prefix is checked. Any
other provider (Fireworks, the offline fake) takes every prefix."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CompareTarget:
    """The compare model, resolved: a chat model's :class:`~pup_finetune.inference.JudgeConfig`
    with :data:`COMPARE_MAX_TOKENS` and the provider's lowest reasoning effort, and where it
    came from (``--compare-model``, or the evaluation directory's ``compare.json``)."""

    config: JudgeConfig
    source: CompareSource

    @property
    def spec(self) -> str:
        return self.config.spec

    @property
    def offline(self) -> bool:
        return self.config.provider == "fake"

    def identity(self) -> dict[str, Any]:
        """``compare.json``: the spec, and the model as it is called."""
        return {"spec": self.spec, "model": self.config.identity()}

    def recipient(self) -> str:
        """The compare model as a ``data_recipients`` entry names it."""
        from pup_finetune.judges import recipient

        if self.offline:
            return "the offline fake compare model (compare; nothing is sent)"
        return f"{recipient(self.config)} (compare model)"

    def destination(self) -> str:
        from pup_finetune.judges import destination

        if self.offline:
            return "the offline fake compare model (nothing is sent)"
        return f"{self.spec} on {destination(self.config)}"

    def billing(self, calls: int) -> str:
        """What ``calls`` compare-model calls bill, in one sentence for a preview."""
        from pup_finetune.judges import billing

        return billing(self.config, calls, what="compare-model")

    def describe(self) -> dict[str, Any]:
        """The preview's ``compare`` block."""
        return {
            "model": self.spec,
            "provider": self.config.provider,
            "endpoint": self.config.base_url,
            "max_tokens": self.config.max_tokens,
            "reasoning_effort": self.config.reasoning_effort,
            "source": self.source,
        }


def resolve_compare(
    spec: str | None, stored: Mapping[str, Any] | None, *, runtime: Runtime
) -> CompareTarget | None:
    """The compare target: ``spec`` (``--compare-model``), else the spec ``stored``
    (``compare.json``) records, else ``None``.

    Both are resolved alike: :data:`FAKE_COMPARE_MODEL` needs development mode, and a package
    the model needs must be installed (:func:`~pup_finetune.judges.check_available`). The
    stored spec is resolved as it is called now, never from the stored endpoint or
    credential; when that differs from what is stored, or ``spec`` names another model,
    replaying into the directory is refused later (:func:`compare_conflict`).
    """
    source: CompareSource = "flag"
    if spec is None:
        if stored is None:
            return None
        spec, source = stored.get("spec"), "evaluation directory"
        if not isinstance(spec, str):
            raise StateError("compare.json does not name its compare model (spec)")
    if spec.strip() == FAKE_COMPARE_MODEL:
        if not runtime.dev_mode:
            raise UsageError(
                "the fake compare model is for development only",
                hint="set PUP_FINETUNE_DEV=1, or pass another --compare-model",
            )
        config = JudgeConfig(
            provider="fake",
            model=FAKE_COMPARE_MODEL,
            base_url="offline",
            credential="none",
            temperature=None,
            max_tokens=COMPARE_MAX_TOKENS,
        )
        return CompareTarget(config=config, source=source)
    from pup_finetune.judges import check_available, parse_judge

    parsed = parse_judge(spec, what="compare model")
    config = parsed.model_copy(update={"max_tokens": COMPARE_MAX_TOKENS})
    check_available(config, what="compare model")
    return CompareTarget(config=config, source=source)


def refused_prefix(messages: Sequence[Mapping[str, Any]], provider: str) -> SkipReason | None:
    """Why ``provider``'s chat API refuses a request of ``messages`` (a case's prefix, in
    OpenAI form) every time, or ``None`` when it takes it.

    Each API refuses an empty conversation, a tool call whose results do not come in the
    turn right after it, before anything else, and a tool result that answers no call of
    the turn right before it. OpenAI's chat completions take the messages as they are, the
    tool results of one call a turn. The Anthropic Messages API is judged on the turns its
    client sends (:func:`~pup_finetune.anthropic_messages.anthropic_messages`: system
    messages joined, consecutive turns of one role merged), and also refuses a conversation
    that ends on an assistant turn: current models take no prefill.
    """
    shapes = REFUSED_SHAPES.get(provider, ())
    turns = anthropic_messages(messages)[1] if provider == "anthropic" else _chat_turns(messages)
    unanswered, orphaned = _tool_pairing(turns)
    found: dict[SkipReason, bool] = {
        "no_messages": not turns,
        "unanswered_tool_call": unanswered,
        "orphan_tool_result": orphaned,
        "ends_on_assistant": bool(turns) and turns[-1]["role"] == "assistant",
    }
    return next((reason for reason in shapes if found[reason]), None)


def compare_skips(cases: Sequence[ReplayCase], provider: str) -> dict[str, SkipReason]:
    """The cases whose prefix ``provider``'s API refuses (:func:`refused_prefix`), by case
    id: the compare target never sends them."""
    return {
        case.id: reason
        for case in cases
        if (reason := refused_prefix(case.prefix, provider)) is not None
    }


def skipped_warning(by_reason: Mapping[str, int], *, spec: str, cases: int) -> str | None:
    """What a preview and a result warn about the compare model's skipped cases (counts
    ``by_reason`` of ``cases``), or ``None`` when none is skipped."""
    total = sum(by_reason.values())
    if not total:
        return None
    return (
        f"{total} of {cases} cases are not sent to the compare model {spec}, whose API refuses "
        f"them every time ({_reasons(by_reason)}); they count in none of its rates, nor in "
        "tuned minus compare"
    )


def require_sent(by_reason: Mapping[str, int], *, spec: str, cases: int) -> None:
    """Raise a :class:`UsageError` when the compare model ``spec`` would be sent none of
    ``cases``, its API refusing every one (counts ``by_reason``): nothing would compare."""
    if sum(by_reason.values()) < cases:
        return
    raise UsageError(
        f"the compare model {spec} would be sent none of the {cases} cases, as its API refuses "
        f"every one ({_reasons(by_reason)})",
        hint="compare with a model on Fireworks, which takes every conversation, or evaluate "
        "without --compare-model",
    )


def _reasons(by_reason: Mapping[str, int]) -> str:
    return joined(
        [f"{count} {SKIP_REASONS.get(reason, reason)}" for reason, count in by_reason.items()]
    )


def _chat_turns(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``messages`` as chat completions take them, as turns of the blocks
    :func:`~pup_finetune.anthropic_messages.anthropic_messages` makes: a message a turn, but
    consecutive tool results one turn, the results of the calls before them."""
    turns: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role"))
        if role == "tool":
            result = {"type": "tool_result", "tool_use_id": str(message.get("tool_call_id"))}
            if turns and turns[-1]["role"] == "tool":
                turns[-1]["content"].append(result)
            else:
                turns.append({"role": role, "content": [result]})
            continue
        calls = message.get("tool_calls") if role == "assistant" else None
        calls = calls if isinstance(calls, Sequence) else ()
        uses = [
            {"type": "tool_use", "id": str(call.get("id"))}
            for call in calls
            if isinstance(call, Mapping)
        ]
        turns.append({"role": role, "content": uses})
    return turns


def _tool_pairing(turns: Sequence[Mapping[str, Any]]) -> tuple[bool, bool]:
    """Whether ``turns`` hold a tool call whose result is not among the tool results that
    open the turn right after it, and whether they hold a tool result that is not one of
    those."""
    unanswered = orphaned = False
    calls: set[str] = set()
    for turn in turns:
        content = turn["content"]
        blocks: list[Mapping[str, Any]] = content if isinstance(content, list) else []
        results = [block for block in blocks if block.get("type") == "tool_result"]
        opening = list(takewhile(lambda block: block.get("type") == "tool_result", blocks))
        answered = {str(block.get("tool_use_id")) for block in opening}
        unanswered |= not calls <= answered
        orphaned |= len(opening) < len(results) or not answered <= calls
        calls = {str(block.get("id")) for block in blocks if block.get("type") == "tool_use"}
    return unanswered or bool(calls), orphaned


def stored_compare(path: Path) -> dict[str, Any] | None:
    """The compare target attached at ``path`` (an evaluation directory's ``compare.json``),
    or ``None`` when none is."""
    if not path.exists():
        return None
    stored = read_json(path)
    if not isinstance(stored, dict):
        raise StateError(f"{path}: expected a JSON object")
    return stored


def compare_conflict(
    stored: Mapping[str, Any] | None, wanted: Mapping[str, Any], *, out_dir: Path
) -> SettingsConflict | None:
    """Why the compare target ``wanted`` cannot replay into ``out_dir``, which has
    ``stored`` attached (both :meth:`CompareTarget.identity`), naming what changed and a
    free ``--out-dir``; ``None`` when it can (none attached yet, or the same one)."""
    wanted = finite_json(dict(wanted))
    if stored is None or canonical_json(stored) == canonical_json(wanted):
        return None
    changed = identity_changes(stored.get("model") or {}, wanted.get("model") or {})
    attached, spec = stored.get("spec"), wanted.get("spec")
    other = f"not {spec}" if attached != spec else "called as it was then"
    fresh = quoted(free_sibling(out_dir))
    return SettingsConflict(
        f"{out_dir} compares with {attached}, {other} (changed: {', '.join(changed) or 'spec'}), "
        "and a compare target never changes once attached",
        hint=f"leave --compare-model out to continue the attached one, or compare with {spec} "
        f"in a new directory: pass --out-dir {fresh}"
        if attached != spec
        else f"compare with {spec} as it is called now in a new directory: pass --out-dir {fresh}",
    )


def attach_compare(
    path: Path, identity: Mapping[str, Any] | None, *, probe: Callable[[], None] | None = None
) -> None:
    """Attach the compare target ``identity`` at ``path`` (an evaluation directory's
    ``compare.json``) the first time, once ``probe`` (when given) returned; later, raise
    :func:`compare_conflict` unless it is the one attached. ``None`` (an evaluation planned
    without one) raises when one is attached, as another process may have done since it was
    planned. Call it under the directory's lock.

    A different compare target is a :class:`SettingsConflict` even before its first pair: an
    experiment may have been created for it already.
    """
    stored = stored_compare(path)
    if identity is None:
        if stored is not None:
            raise SettingsConflict(
                f"{path.parent} has a compare target ({stored.get('spec')}) this evaluation "
                "was planned without",
                hint="run the command again: it continues the compare target too",
            )
        return
    if stored is None:
        if probe is not None:
            probe()
        atomic_write_json(path, finite_json(dict(identity)))
        return
    conflict = compare_conflict(stored, identity, out_dir=path.parent)
    if conflict is not None:
        raise conflict


class ChatSampler:
    """A :class:`~pup_finetune.providers.base.Sampler` over a chat completions client.

    The prefix (without its recorded ``reasoning_content``, which the compare model never
    wrote) and tools go in OpenAI form, with the config's temperature and reasoning effort.
    The answer keeps its text and tool calls. ``stop_reason`` is the completion's
    ``finish_reason``; ``length`` (cut off at ``max_tokens``) and a tool call
    whose arguments are not a JSON object make it not ``format_valid``. A refusal is an
    answer like any other, judged as one. A failed call raises, which replay records as the
    pair's generation error.
    """

    def __init__(
        self,
        client: ChatCompleter,
        config: JudgeConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.config = config
        self.clock = clock

    def generate(
        self, messages: list[ChatMessage], tools: list[ToolSpec], *, max_tokens: int
    ) -> Generation:
        start = self.clock()
        completion = self.client.complete(
            [
                {k: v for k, v in message.items() if k != "reasoning_content"}
                for message in messages
            ],
            model=self.config.model,
            tools=tools or None,
            temperature=self.config.temperature,
            max_tokens=max_tokens,
            reasoning_effort=self.config.reasoning_effort,
        )
        latency = self.clock() - start
        message, calls_valid = _assistant(completion.message)
        usage = completion.usage
        return Generation(
            message=message,
            format_valid=calls_valid and completion.finish_reason != "length",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_s=latency,
            stop_reason=completion.finish_reason,
        )

    def close(self) -> None:
        """Nothing to release: the client's owner closes it."""


def _assistant(raw: Mapping[str, Any]) -> tuple[ChatMessage, bool]:
    """An answer as an assistant message (its text and tool calls), and whether every tool
    call names a function and has arguments that are a JSON object."""
    content = raw.get("content")
    message: ChatMessage = {
        "role": "assistant",
        "content": content if isinstance(content, str) else "",
    }
    valid = True
    calls: list[dict[str, Any]] = []
    found = raw.get("tool_calls")
    for index, call in enumerate(found if isinstance(found, list) else []):
        function = call.get("function") if isinstance(call, Mapping) else None
        if not isinstance(function, Mapping):
            valid = False
            continue
        name, arguments = function.get("name"), function.get("arguments")
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else None
        except ValueError:
            parsed = None
        valid = valid and isinstance(name, str) and bool(name) and isinstance(parsed, dict)
        calls.append(
            {
                "id": str(call.get("id") or f"call_{index}"),
                "type": "function",
                "function": {"name": str(name or ""), "arguments": str(arguments or "")},
            }
        )
    if calls:
        message["tool_calls"] = calls
    return message, valid


class FakeCompareSampler:
    """The offline compare model of development mode: it answers a case's reference for a
    seeded :data:`FAKE_COMPARE_AGREEMENT` of the cases, and :data:`FAKE_COMPARE_ANSWER` for
    the rest. A case is found by its prefix and tools."""

    def __init__(self, cases: Sequence[ReplayCase], *, seed: int) -> None:
        self._answers: dict[str, ChatMessage] = {}
        for case in cases:
            known = split_position(f"fake-compare:{case.id}", seed=seed) < FAKE_COMPARE_AGREEMENT
            answer = {
                key: value for key, value in case.reference.items() if key != "reasoning_content"
            }
            self._answers[_prompt_key(case.prefix, case.tools)] = (
                answer if known else {"role": "assistant", "content": FAKE_COMPARE_ANSWER}
            )

    def generate(
        self, messages: list[ChatMessage], tools: list[ToolSpec], *, max_tokens: int
    ) -> Generation:
        answer = self._answers.get(
            _prompt_key(messages, tools), {"role": "assistant", "content": FAKE_COMPARE_ANSWER}
        )
        return Generation(
            message=dict(answer),
            format_valid=True,
            prompt_tokens=len(canonical_json(messages)) // 4,
            completion_tokens=len(canonical_json(answer)) // 4,
            latency_s=0.0,
            stop_reason="stop",
        )

    def close(self) -> None:
        pass


def _prompt_key(messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json([list(messages), list(tools)])


def probe(sampler: Sampler, target: CompareTarget, cases: Sequence[ReplayCase]) -> None:
    """One short call to the compare model before it is attached, so a model that cannot
    answer (a mistyped name, a rejected key) fails the command before ``compare.json``
    freezes it. It is sent as replay calls it, with the target's ``max_tokens`` and reasoning
    effort, and a tool (:data:`PROBE_TOOL`) when any of ``cases`` has tools, so a model that
    refuses those fails here too. Any answer passes, a cut-off one included; a failure raises
    the provider's error."""
    info(f"checking that the compare model {target.spec} answers")
    sampler.generate(
        [{"role": "user", "content": PROBE_PROMPT}],
        [PROBE_TOOL] if any(case.tools for case in cases) else [],
        max_tokens=target.config.max_tokens,
    )


@contextmanager
def open_compare_sampler(
    target: CompareTarget, cases: Sequence[ReplayCase], *, seed: int
) -> Iterator[Sampler]:
    """The compare model's sampler: the fake one, or a :class:`ChatSampler` over the client
    :func:`~pup_finetune.judges.open_client` opens for it (with :data:`COMPARE_TIMEOUT_S`
    per attempt, and no server-side fallback to another model), closed on exit."""
    if target.offline:
        yield FakeCompareSampler(cases, seed=seed)
        return
    from pup_finetune.judges import open_client

    with open_client(
        target.config, timeout_s=COMPARE_TIMEOUT_S, fallbacks=False, what="compare model"
    ) as client:
        yield ChatSampler(client, target.config)
