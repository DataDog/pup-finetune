# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""An offline, deterministic judge for development mode and tests: no network, no key.

:class:`FakeJudge` is a :class:`~pup_finetune.inference.ChatCompleter` that reads the
reference and candidate actions back out of the prompt
:func:`~pup_finetune.evaluation.judge.build_messages` wrote, and answers the
``{"pass", "reason"}`` verdict a strict but reasonable judge would:

- a tool-call reference passes a candidate that calls the same tools (as many times each),
  with arguments that contradict none of the reference's: every argument both give has the
  same value, so leaving an argument out is allowed but changing one is not;
- a text reference passes a text candidate (no tool calls) that shares at least half of its
  words with the reference (Jaccard similarity over lowercase words).

So the reference passes and every corruption :func:`~pup_finetune.evaluation.judge.corrupt`
makes fails (a wrong tool, an altered argument, an unsupported claim, a refusal), and
calibration succeeds. The evaluate command uses it for runs of the development-only ``fake``
provider (or when ``--judge-model`` is :data:`FAKE_JUDGE_MODEL`), and only with
``PUP_FINETUNE_DEV=1``.
"""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Final

from pup_finetune.inference import ChatCompletion, JudgeConfig
from pup_finetune.storage import loads_finite

__all__ = ["FAKE_JUDGE_MODEL", "FakeJudge", "fake_judge_config"]

FAKE_JUDGE_MODEL: Final = "fake-judge"
"""The judge model name that selects :class:`FakeJudge` in development mode."""

_BLOCK: Final = re.compile(r"<(reference_action|candidate_action)>\n(.*?)\n</\1>", re.DOTALL)
_WORD: Final = re.compile(r"[a-z0-9]+")
_MIN_OVERLAP: Final = 0.5


def fake_judge_config() -> JudgeConfig:
    """The :class:`~pup_finetune.inference.JudgeConfig` that names :class:`FakeJudge`."""
    return JudgeConfig(model=FAKE_JUDGE_MODEL, base_url="offline://fake-judge", credential="none")


class FakeJudge:
    """The offline judge; see the module docstring. ``calls`` counts the completions answered."""

    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        tools: Sequence[Mapping[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatCompletion:
        with self._lock:
            self.calls += 1
        blocks = _blocks(messages)
        reference = blocks.get("reference_action")
        candidate = blocks.get("candidate_action")
        if reference is None or candidate is None:
            content = '{"pass": false, "reason": "the prompt holds no actions to compare"}'
        else:
            passed, reason = verdict(candidate, reference)
            content = json.dumps({"pass": passed, "reason": reason})
        return ChatCompletion(
            message={"role": "assistant", "content": content},
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model=model,
            finish_reason="stop",
        )

    def close(self) -> None:
        """Nothing to release; here so it closes like a :class:`~pup_finetune.inference.ChatClient`."""


def _blocks(messages: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    text = "\n".join(
        str(message.get("content")) for message in messages if message.get("role") == "user"
    )
    found: dict[str, dict[str, Any]] = {}
    for name, body in _BLOCK.findall(text):
        try:
            value = loads_finite(body)
        except ValueError:
            continue
        if isinstance(value, dict):
            found[name] = value
    return found


def _calls(message: Mapping[str, Any]) -> list[tuple[str, Any]]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return []
    result: list[tuple[str, Any]] = []
    for call in calls:
        function = call.get("function") if isinstance(call, Mapping) else None
        if not isinstance(function, Mapping):
            result.append(("", None))
            continue
        raw = function.get("arguments")
        try:
            arguments = loads_finite(raw) if isinstance(raw, str) else raw
        except ValueError:
            arguments = None
        result.append((str(function.get("name") or ""), arguments))
    return result


def _words(message: Mapping[str, Any]) -> set[str]:
    content = message.get("content")
    return set(_WORD.findall(content.lower())) if isinstance(content, str) else set()


def _contradicts(candidate: object, reference: object) -> bool:
    if not isinstance(candidate, dict):
        return True
    if not isinstance(reference, dict):
        return False
    return any(key in reference and reference[key] != value for key, value in candidate.items())


def verdict(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> tuple[bool, str]:
    """``(passed, reason)`` for ``candidate`` against ``reference`` (the module's rules)."""
    expected, actual = _calls(reference), _calls(candidate)
    if expected:
        if not actual:
            return False, "the reference calls a tool and the candidate does not"
        if Counter(name for name, _ in expected) != Counter(name for name, _ in actual):
            return False, "the candidate calls other tools than the reference"
        by_name: dict[str, list[Any]] = {}
        for name, arguments in expected:
            by_name.setdefault(name, []).append(arguments)
        for name, arguments in actual:
            if _contradicts(arguments, by_name[name].pop(0)):
                return False, f"the candidate passes other arguments to {name}"
        return True, "the candidate calls the same tools with consistent arguments"
    if actual:
        return False, "the reference answers in text and the candidate calls a tool"
    want, got = _words(reference), _words(candidate)
    union = want | got
    overlap = len(want & got) / len(union) if union else 1.0
    if overlap >= _MIN_OVERLAP:
        return True, "the candidate says what the reference says"
    return False, "the candidate says something other than the reference"
