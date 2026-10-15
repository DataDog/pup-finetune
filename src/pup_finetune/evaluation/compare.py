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
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final, Literal

from pup_finetune.data.split import split_position
from pup_finetune.errors import StateError, UsageError
from pup_finetune.evaluation.cases import ReplayCase
from pup_finetune.inference import ChatCompleter, JudgeConfig
from pup_finetune.providers.base import Generation, Sampler
from pup_finetune.rendering.base import ChatMessage, ToolSpec
from pup_finetune.runtime import Runtime
from pup_finetune.storage import canonical_json, finite_json

__all__ = [
    "COMPARE_MAX_TOKENS",
    "COMPARE_TIMEOUT_S",
    "FAKE_COMPARE_MODEL",
    "ChatSampler",
    "CompareTarget",
    "FakeCompareSampler",
    "open_compare_sampler",
    "probe",
    "resolve_compare",
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
PROBE_MAX_TOKENS: Final = 16

CompareSource = Literal["flag", "evaluation directory"]


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

    def identity(self, evaluation: Mapping[str, Any]) -> dict[str, Any]:
        """``compare.json``: the model as it is called, and the sha256 of the evaluation
        identity (``evaluation.json``) it was attached to."""
        payload = canonical_json(finite_json(dict(evaluation))).encode("utf-8")
        return {
            "evaluation_sha256": hashlib.sha256(payload).hexdigest(),
            "spec": self.spec,
            "model": self.config.identity(),
        }

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
    """The compare target: ``spec`` (``--compare-model``), else the one ``stored``
    (``compare.json``) records, else ``None``.

    :data:`FAKE_COMPARE_MODEL` needs development mode; a package the model needs must be
    installed (:func:`~pup_finetune.judges.check_available`). A ``spec`` other than the
    stored one is resolved all the same: replaying it into that directory is refused later.
    """
    from pup_finetune.judges import check_available, parse_judge

    if spec is None:
        if stored is None:
            return None
        try:
            config = JudgeConfig.model_validate(stored.get("model"))
        except ValueError as exc:
            raise StateError(f"compare.json does not describe a compare model: {exc}") from None
        return CompareTarget(config=config, source="evaluation directory")
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
        return CompareTarget(config=config, source="flag")
    parsed = parse_judge(spec)
    config = parsed.model_copy(update={"max_tokens": COMPARE_MAX_TOKENS})
    check_available(config)
    return CompareTarget(config=config, source="flag")


class ChatSampler:
    """A :class:`~pup_finetune.providers.base.Sampler` over a chat completions client.

    The prefix and tools go as they are (OpenAI form), with the config's temperature and
    reasoning effort. The answer keeps its text and tool calls. ``stop_reason`` is the
    completion's ``finish_reason``; ``length`` (cut off at ``max_tokens``) and a tool call
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
            messages,
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


def probe(sampler: Sampler) -> None:
    """One cheap call to the compare model before it is attached, so a model that cannot
    answer (a mistyped name, a rejected key) fails the command before ``compare.json``
    freezes it. Any answer passes, one cut off at :data:`PROBE_MAX_TOKENS` included; a
    failure raises the provider's error."""
    sampler.generate([{"role": "user", "content": PROBE_PROMPT}], [], max_tokens=PROBE_MAX_TOKENS)


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

    with open_client(target.config, timeout_s=COMPARE_TIMEOUT_S, fallbacks=False) as client:
        yield ChatSampler(client, target.config)
