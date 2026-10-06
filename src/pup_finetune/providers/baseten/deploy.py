# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Serve a Baseten training run's best checkpoint behind OpenAI routes, and stop serving it.

``endpoint.json`` in the run directory is the receipt: it is saved before every paid create and
again as soon as Baseten returns IDs, so an interrupted deploy can always be found and stopped.
``dedicated`` deploys the checkpoint as its own Baseten model on the requested GPUs; ``sampler``
serves it from a Loops sampler, which works without BDN volumes.
"""

from __future__ import annotations

import functools
import json
import math
import re
import shlex
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeGuard

from pydantic import SecretStr

from pup_finetune import output
from pup_finetune.errors import ExternalError, FinetuneError, StateError, UsageError
from pup_finetune.providers import DeployRequest
from pup_finetune.providers.baseten.client import (
    CHECKPOINT_URI,
    BasetenHTTP,
    DeploymentRejected,
    LoopsModule,
    load_sampler_checkpoint,
    loops_sdk,
)
from pup_finetune.providers.baseten.models import BasetenModel, get_model
from pup_finetune.rendering import ChatTokenizer, ModelFamily, load_tokenizer, prompt_tokens
from pup_finetune.storage import DirectoryLock, atomic_write_json, read_json
from pup_finetune.training import RunDir

__all__ = ["MIN_SEQ_LEN", "RECEIPT_FILE", "deploy", "deploy_preview", "undeploy"]

RECEIPT_FILE = "endpoint.json"
MIN_SEQ_LEN = 1024
MAX_TIMEOUT_SECONDS = 7200.0
UNDEPLOY_TIMEOUT_SECONDS = 120.0
POLL_SECONDS = 5.0

Sleep = Callable[[float], None]
Clock = Callable[[], float]

_MODES = ("dedicated", "sampler")
_ACCELERATOR = re.compile(r"[A-Z][A-Z0-9_]*:[1-9][0-9]*")
_ENDPOINT_ID = re.compile(r"[a-z0-9]{1,64}")
_RESOURCE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_READY = frozenset({"ACTIVE", "SCALED_TO_ZERO"})
_FATAL = frozenset(
    {"UNHEALTHY", "DEPLOY_FAILED", "BUILD_FAILED", "BUILD_STOPPED", "FAILED", "DEACTIVATING"}
)
_SAMPLER_FIELDS = ("id", "model_id", "deployment_id", "base_url")
_LOOKUP_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "lookup_number",
        "description": "Look up the requested number. Use this tool when asked to look up a number.",
        "parameters": {
            "type": "object",
            "properties": {"number": {"type": "integer"}},
            "required": ["number"],
            "additionalProperties": False,
        },
    },
}


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_id(value: object, pattern: re.Pattern[str] = _RESOURCE_ID) -> TypeGuard[str]:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _deployment_url(model_id: str, deployment_id: str) -> str:
    return f"https://model-{model_id}.api.baseten.co/deployment/{deployment_id}/sync"


def _reason(exc: BaseException) -> str:
    return exc.message if isinstance(exc, FinetuneError) else type(exc).__name__


def _add_hint(exc: BaseException, hint: str) -> None:
    if isinstance(exc, FinetuneError):
        exc.hint = f"{exc.hint}; {hint}" if exc.hint else hint


@dataclass(frozen=True, slots=True)
class _Target:
    """What a deploy request resolves to, from the run directory alone."""

    run_dir: Path
    mode: str
    model: BasetenModel
    checkpoint: dict[str, str]
    settings: dict[str, Any]
    timeout: float

    @property
    def receipt_path(self) -> Path:
        return self.run_dir / RECEIPT_FILE

    @property
    def served_model(self) -> str:
        """The model name the OpenAI routes serve the checkpoint under."""
        return self.checkpoint["checkpoint_path" if self.mode == "sampler" else "checkpoint_name"]

    @property
    def cleanup_command(self) -> str:
        return f"pup finetune undeploy --run-dir {shlex.quote(str(self.run_dir))} --confirm"


def _target(request: DeployRequest) -> _Target:
    run_dir = Path(request.run_dir).resolve()
    runs = RunDir(run_dir)
    plan, result = runs.read_plan(), runs.read_result()
    if plan.get("provider") != "baseten" or result.get("provider") != "baseten":
        raise UsageError(
            f"{run_dir} is not a Baseten training run",
            hint="pass the --run-dir of a `pup finetune train` run",
        )
    model = get_model(str(plan.get("model")))
    if plan.get("base_model") != model.base_model:
        raise StateError(f"{run_dir} records a base model that {model.alias} does not use")
    uri = result.get("best_sampler_weights_uri")
    match = CHECKPOINT_URI.fullmatch(uri) if isinstance(uri, str) else None
    if match is None or match[1] != result.get("baseten_run_id"):
        raise UsageError(
            f"{run_dir} has no best sampler checkpoint to deploy",
            hint="training saves one whenever validation loss improves; resumable state "
            "cannot be served",
        )
    checkpoint = {
        "checkpoint_path": match[0],
        "run_id": match[1],
        "checkpoint_name": match[2],
        "base_model": model.base_model,
    }
    settings = _settings(request, model)
    return _Target(
        run_dir, request.serving_mode, model, checkpoint, settings, request.timeout_seconds
    )


def _settings(request: DeployRequest, model: BasetenModel) -> dict[str, Any]:
    limit = model.family.max_seq_len
    max_seq_len = limit if request.max_seq_len is None else request.max_seq_len
    if not MIN_SEQ_LEN <= max_seq_len <= limit:
        raise UsageError(
            f"--max-seq-len must be between {MIN_SEQ_LEN} and {limit} for {model.alias}"
        )
    timeout = request.timeout_seconds
    if not (math.isfinite(timeout) and 1 <= timeout <= MAX_TIMEOUT_SECONDS):
        raise UsageError(f"--timeout must be between 1 and {MAX_TIMEOUT_SECONDS:g} seconds")
    if request.serving_mode not in _MODES:
        raise UsageError(
            f"unknown serving mode {request.serving_mode!r}", hint="use dedicated or sampler"
        )
    if request.serving_mode == "sampler":
        if request.accelerator is not None:
            raise UsageError(
                "--accelerator applies only to --serving-mode dedicated",
                hint="a Loops sampler chooses its own GPUs; drop --accelerator",
            )
        return {"max_seq_len": max_seq_len}
    if request.accelerator is None or not _ACCELERATOR.fullmatch(request.accelerator):
        raise UsageError(
            "--serving-mode dedicated needs --accelerator TYPE:COUNT, e.g. H100:1",
            hint="or serve from a Loops sampler with --serving-mode sampler",
        )
    return {"accelerator": request.accelerator, "max_seq_len": max_seq_len}


def _read_receipt(path: Path) -> dict[str, Any] | None:
    try:
        data = read_json(path)
    except FileNotFoundError:
        return None
    if not (
        isinstance(data, dict)
        and data.get("schema_version") == 1
        and data.get("provider") == "baseten"
        and data.get("owned") is True
        and data.get("serving_mode") in _MODES
        and isinstance(data.get("checkpoint"), dict)
        and isinstance(data.get("settings"), dict)
    ):
        raise StateError(f"{path} is not a deployment receipt written by pup-finetune")
    return data


def _new_receipt(target: _Target, **fields: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "provider": "baseten",
        "owned": True,
        "serving_mode": target.mode,
        "checkpoint": dict(target.checkpoint),
        "settings": dict(target.settings),
        "cleanup_command": target.cleanup_command,
        **fields,
    }


def _endpoint_fields(model_id: object, deployment_id: object) -> dict[str, Any]:
    if not (_is_id(model_id, _ENDPOINT_ID) and _is_id(deployment_id, _ENDPOINT_ID)):
        raise ExternalError("Baseten returned no valid deployment IDs", source="baseten")
    url = _deployment_url(model_id, deployment_id)
    return {
        "endpoint": {"model_id": model_id, "deployment_id": deployment_id, "url": url},
        "openai_base_url": f"{url}/v1",
    }


def _unknown_hint(receipt: Mapping[str, Any], path: Path) -> str:
    name = receipt.get("model_name")
    where = f"a model named {name}" if name else f"Loops session {receipt.get('session_id')}"
    return (
        f"look for {where} in the Baseten UI and deactivate what it serves, "
        f"then delete {path} and rerun deploy"
    )


def _receipt_ids(receipt: Mapping[str, Any], path: Path) -> tuple[str, str]:
    """The recorded model and deployment IDs; a receipt without them has an unknown outcome."""
    endpoint = receipt.get("endpoint")
    if isinstance(endpoint, dict):
        model_id, deployment_id = endpoint.get("model_id"), endpoint.get("deployment_id")
        if _is_id(model_id, _ENDPOINT_ID) and _is_id(deployment_id, _ENDPOINT_ID):
            return model_id, deployment_id
    raise StateError(
        f"{path} records no deployment IDs, so an earlier deploy has an unknown outcome",
        hint=_unknown_hint(receipt, path),
    )


class _Deployment:
    """The receipt in ``endpoint.json`` and the Baseten deployment it records."""

    def __init__(
        self, path: Path, data: dict[str, Any], http: BasetenHTTP, *, sleep: Sleep, clock: Clock
    ) -> None:
        self.path = path
        self.data = data
        self.http = http
        self._sleep = sleep
        self._clock = clock

    def save(self, state: str, **fields: Any) -> None:
        self.data.update(state=state, **fields)
        atomic_write_json(self.path, self.data)

    def unknown_hint(self) -> str:
        return _unknown_hint(self.data, self.path)

    def ids(self) -> tuple[str, str]:
        return _receipt_ids(self.data, self.path)

    def status(self) -> str | None:
        """Baseten's status of the recorded deployment; ``None`` when Baseten no longer has it."""
        model_id, deployment_id = self.ids()
        found = self.http.get_deployment(model_id, deployment_id)
        if found is None:
            return None
        status = found.get("status")
        if (
            found.get("id") != deployment_id
            or found.get("model_id") != model_id
            or not isinstance(status, str)
        ):
            raise ExternalError("Baseten described a different deployment", source="baseten")
        return status

    def wait(self, *, timeout: float, inactive: bool = False) -> None:
        deadline = self._clock() + timeout
        while True:
            status = self.status()
            if inactive and status in (None, "INACTIVE"):
                return
            if not inactive:
                if status is None:
                    raise ExternalError("Baseten no longer has the deployment", source="baseten")
                if status in _READY:
                    return
                if status in _FATAL:
                    raise ExternalError(f"the Baseten deployment is {status}", source="baseten")
            remaining = deadline - self._clock()
            if remaining <= 0:
                goal = "inactive" if inactive else "ready"
                raise ExternalError(
                    f"the Baseten deployment was not {goal} after {timeout:g} s (it is {status})",
                    source="baseten",
                )
            self._sleep(min(POLL_SECONDS, remaining))

    def stop(self) -> None:
        """Deactivate the recorded deployment, once, and wait until Baseten reports it inactive."""
        model_id, deployment_id = self.ids()
        try:
            status = self.status()
            if status not in (None, "INACTIVE"):
                self.save("deactivating")
                if status != "DEACTIVATING":
                    self.http.deactivate_deployment(model_id, deployment_id)
                self.wait(timeout=UNDEPLOY_TIMEOUT_SECONDS, inactive=True)
            self.save("inactive")
        except BaseException:
            self.save("cleanup_required")
            raise


def _prior(target: _Target) -> dict[str, Any] | None:
    """The receipt to resume, or ``None`` to create anew; refuses a different live deployment."""
    receipt = _read_receipt(target.receipt_path)
    if receipt is None:
        return None
    same = (
        receipt["serving_mode"] == target.mode
        and receipt["settings"] == target.settings
        and all(receipt["checkpoint"].get(key) == value for key, value in target.checkpoint.items())
    )
    if not same:
        if receipt.get("state") == "inactive":
            return None
        raise StateError(
            f"{target.receipt_path} records a different deployment of this run",
            hint=f"stop it first with `{target.cleanup_command}`, then rerun deploy",
        )
    _receipt_ids(receipt, target.receipt_path)
    return receipt


def _checkpoint_id(http: BasetenHTTP, checkpoint: Mapping[str, str]) -> str:
    """Baseten's id of the run's checkpoint, once Baseten confirms it matches and has synced."""
    record = http.get_checkpoint(checkpoint["checkpoint_path"])
    expected = {
        "run_id": checkpoint["run_id"],
        "checkpoint_id": checkpoint["checkpoint_name"],
        "base_model": checkpoint["base_model"],
        "target": "sampler",
    }
    record_id = record.get("id")
    if any(record.get(key) != value for key, value in expected.items()) or not _is_id(record_id):
        raise ExternalError(
            "Baseten's record of the checkpoint does not match this training run", source="baseten"
        )
    if record.get("sync_status") not in (None, "COMPLETE"):
        raise ExternalError(
            "the checkpoint is still syncing on Baseten",
            source="baseten",
            hint="rerun deploy in a few minutes",
        )
    return record_id


def _instance_type(instances: Sequence[Mapping[str, Any]], accelerator: str) -> str:
    """The first single-node instance with exactly the requested GPUs; never a larger one."""
    gpu, count = accelerator.split(":")
    offered: set[tuple[str, int]] = set()
    for instance in instances:
        gpu_type, gpu_count, instance_id = (
            instance.get("gpuType"),
            instance.get("gpuCount"),
            instance.get("id"),
        )
        if instance.get("nodeCount") != 1 or not (isinstance(gpu_type, str) and _is_int(gpu_count)):
            continue
        # IDs look like "H100:1x26x234"; they only travel as a GraphQL variable.
        if (
            (gpu_type, gpu_count) == (gpu, int(count))
            and isinstance(instance_id, str)
            and instance_id
        ):
            return instance_id
        if gpu_count > 0:
            offered.add((gpu_type, gpu_count))
    nearby = sorted(item for item in offered if item[0] == gpu) or sorted(offered)
    raise UsageError(
        f"Baseten offers no single-node instance with exactly {accelerator}",
        hint="available: " + ", ".join(f"{t}:{n}" for t, n in nearby)
        if nearby
        else "Baseten listed no GPU instances",
    )


def _message(response: Mapping[str, Any], test: str) -> dict[str, Any]:
    choices = response.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        raise ExternalError(f"the {test} smoke test got no message", source="baseten")
    return message


def _visible(message: Mapping[str, Any], test: str) -> str:
    content = message.get("content")
    if not isinstance(content, str) or not content.strip() or message.get("tool_calls"):
        raise ExternalError(f"the {test} smoke test got no visible answer", source="baseten")
    return content


def _is_lookup_seven(call: Mapping[str, Any]) -> bool:
    function = call.get("function")
    call_id = call.get("id")
    if not (
        isinstance(function, dict)
        and isinstance(call_id, str)
        and call_id
        and call.get("type") == "function"
        and function.get("name") == "lookup_number"
    ):
        return False
    try:
        return bool(json.loads(function["arguments"]) == {"number": 7})
    except (KeyError, TypeError, ValueError):
        return False


def _lookup_call(message: Mapping[str, Any]) -> dict[str, Any]:
    calls = message.get("tool_calls")
    call = calls[0] if isinstance(calls, list) and len(calls) == 1 else None
    if isinstance(call, dict) and _is_lookup_seven(call):
        return call
    if not calls and "lookup_number" in str(message.get("content") or ""):
        raise ExternalError(
            "the tool smoke test got the tool call only as text; "
            "the server does not parse tool calls for this model",
            source="baseten",
        )
    raise ExternalError(
        "the tool smoke test did not get one lookup_number call with number 7", source="baseten"
    )


def _served_context(models: Sequence[Mapping[str, Any]], served_model: str) -> int:
    entries = [entry for entry in models if entry.get("id") == served_model]
    if len(entries) != 1:
        raise ExternalError(f"the deployment does not list {served_model}", source="baseten")
    context = entries[0].get("max_model_len")
    parent = entries[0].get("parent")
    if context is None and isinstance(parent, str):
        # vLLM LoRA entries can inherit the context limit of their base model.
        parents = [entry for entry in models if entry.get("id") == parent]
        context = parents[0].get("max_model_len") if len(parents) == 1 else None
    if not _is_int(context):
        raise ExternalError(
            f"the deployment advertises no context for {served_model}", source="baseten"
        )
    return context


def _smoke(http: BasetenHTTP, base_url: str, served_model: str, max_seq_len: int) -> int:
    """Check the OpenAI routes: model listing and context, a text reply, a tool call, its result."""
    context = _served_context(http.list_served_models(base_url), served_model)
    if context < max_seq_len:
        raise ExternalError(
            f"the deployment serves {context} tokens of context, below --max-seq-len {max_seq_len}",
            source="baseten",
            hint=f"undeploy, then deploy again with --max-seq-len {context}"
            if context >= MIN_SEQ_LEN
            else None,
        )

    def chat(
        messages: list[dict[str, Any]], max_tokens: int, *, tools: bool, test: str
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": served_model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if tools:
            body |= {"tools": [_LOOKUP_TOOL], "tool_choice": "auto"}
        return _message(http.chat_completion(base_url, body), test)

    hello = [{"role": "user", "content": "Reply with the word hello."}]
    if "</think>" in _visible(chat(hello, 256, tools=False, test="text"), "text"):
        output.warn("replies include reasoning markup (</think>); clients must strip it")
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": "Use lookup_number to look up number 7. Do not answer without calling it.",
        }
    ]
    reply = chat(messages, 512, tools=True, test="tool")
    call = _lookup_call(reply)
    messages += [
        {"role": "assistant", "content": reply.get("content"), "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": "The value is seven."},
    ]
    _visible(chat(messages, 256, tools=True, test="tool-result"), "tool-result")
    return context


def _open(target: _Target, http: BasetenHTTP, *, sleep: Sleep, clock: Clock) -> _Deployment:
    """The deployment to resume, or an empty one to create (when none, or the prior one is gone)."""
    deployment = _Deployment(
        target.receipt_path, _prior(target) or {}, http, sleep=sleep, clock=clock
    )
    if deployment.data:
        status = deployment.status()
        # A stopped sampler cannot be restarted; a new one is created in a new session.
        if status is None or (target.mode == "sampler" and status == "INACTIVE"):
            deployment.data = {}
    return deployment


def _become_ready(
    deployment: _Deployment, target: _Target, before_smoke: Callable[[], None] | None = None
) -> None:
    deployment.save("waiting")
    output.info(f"waiting up to {target.timeout:g} s for the Baseten deployment to be ready")
    deployment.wait(timeout=target.timeout)
    deployment.save("smoke_testing")
    if before_smoke is not None:
        before_smoke()
    output.info(f"smoke-testing {target.served_model} on its OpenAI routes")
    base_url = f"{_deployment_url(*deployment.ids())}/v1"
    max_seq_len = target.settings["max_seq_len"]
    advertised = _smoke(deployment.http, base_url, target.served_model, max_seq_len)
    deployment.save("ready", advertised_max_seq_len=advertised, openai_base_url=base_url)


def _create_dedicated(deployment: _Deployment, target: _Target) -> None:
    http = deployment.http
    checkpoint_id = _checkpoint_id(http, target.checkpoint)
    accelerator = target.settings["accelerator"]
    instance_type_id = _instance_type(http.instance_types(), accelerator)
    model_name = f"pup-finetune-{uuid.uuid4().hex}"
    deployment.data = _new_receipt(
        target,
        checkpoint={**target.checkpoint, "checkpoint_id": checkpoint_id},
        model_name=model_name,
        served_model=target.served_model,
    )
    deployment.save("creating")
    output.info(f"creating Baseten model {model_name} on {accelerator}")
    try:
        created = http.create_checkpoint_deployment(
            checkpoint_id=checkpoint_id, model_name=model_name, instance_type_id=instance_type_id
        )
        version = created.get("model_version")
        version = version if isinstance(version, dict) else {}
        deployment.save("created", **_endpoint_fields(version.get("model_id"), version.get("id")))
    except DeploymentRejected as exc:
        deployment.path.unlink(missing_ok=True)
        if "bdn volumes" in exc.message.lower():
            exc.hint = (
                "nothing was created; rerun with --serving-mode sampler, "
                "or ask Baseten to enable BDN volumes"
            )
        else:
            exc.hint = "nothing was created; fix the cause and rerun deploy"
        raise
    except BaseException as exc:
        deployment.save("creation_unknown")
        _add_hint(exc, deployment.unknown_hint())
        raise


def _serve_dedicated(deployment: _Deployment, target: _Target) -> None:
    if not deployment.data:
        _create_dedicated(deployment, target)
    try:
        if deployment.status() == "INACTIVE":
            deployment.save("activating")
            deployment.http.activate_deployment(*deployment.ids())
        _become_ready(deployment, target)
    except BaseException as exc:
        deployment.save("needs_attention")
        _add_hint(exc, f"rerun deploy to resume, or stop it with `{target.cleanup_command}`")
        raise


def _create_sampler(deployment: _Deployment, target: _Target) -> None:
    http = deployment.http
    _checkpoint_id(http, target.checkpoint)
    checkpoint_path = target.checkpoint["checkpoint_path"]
    output.info(f"creating a Baseten Loops sampler for {checkpoint_path}")
    session_id = http.create_session()
    deployment.data = _new_receipt(target, session_id=session_id, served_model=target.served_model)
    deployment.save("creating")
    try:
        sampler = http.create_sampler(
            session_id=session_id,
            model_path=checkpoint_path,
            max_seq_len=target.settings["max_seq_len"],
        )
        # Saved as returned, before validation, so cleanup can find whatever Baseten created.
        deployment.save("created", sampler={k: sampler[k] for k in _SAMPLER_FIELDS if k in sampler})
        deployment.save(
            "created", **_endpoint_fields(sampler.get("model_id"), sampler.get("deployment_id"))
        )
    except DeploymentRejected as exc:
        deployment.path.unlink(missing_ok=True)
        exc.hint = "nothing was created; fix the cause and rerun deploy"
        raise
    except BaseException as exc:
        deployment.save("creation_unknown")
        _add_hint(exc, deployment.unknown_hint())
        raise


def _sampler_ids(deployment: _Deployment) -> tuple[str, str]:
    """The recorded sampler and session IDs, once the sampler's URL matches its deployment."""
    sampler = deployment.data.get("sampler")
    sampler = sampler if isinstance(sampler, dict) else {}
    sampler_id, session_id = sampler.get("id"), deployment.data.get("session_id")
    url = _deployment_url(*deployment.ids())
    if (
        not (_is_id(sampler_id) and _is_id(session_id))
        or str(sampler.get("base_url")).rstrip("/") != url
    ):
        raise ExternalError("Baseten described the sampler unexpectedly", source="baseten")
    return sampler_id, session_id


def _serve_sampler(deployment: _Deployment, target: _Target, load: Callable[..., None]) -> None:
    if not deployment.data:
        _create_sampler(deployment, target)
    try:
        sampler_id, session_id = _sampler_ids(deployment)
        model_id, deployment_id = deployment.ids()
        _become_ready(
            deployment,
            target,
            lambda: load(
                base_url=_deployment_url(model_id, deployment_id),
                model_id=model_id,
                deployment_id=deployment_id,
                sampler_id=sampler_id,
                session_id=session_id,
            ),
        )
    except BaseException as exc:
        try:
            deployment.stop()
        except BaseException as cleanup:
            raise ExternalError(
                f"{_reason(exc)}; stopping the sampler failed too ({_reason(cleanup)}), "
                "so it may still bill",
                source="baseten",
                hint=f"stop it with `{target.cleanup_command}`",
            ) from exc
        _add_hint(exc, "the sampler was stopped; rerun deploy to start a new one")
        raise


def deploy_preview(request: DeployRequest) -> dict[str, Any]:
    """What :func:`deploy` would serve, from local files only. Free."""
    target = _target(request)
    resumes = _prior(target) is not None
    what = "dedicated deployment" if target.mode == "dedicated" else "Loops sampler"
    return {
        "provider": "baseten",
        "action": "resume" if resumes else "create",
        "serving_mode": target.mode,
        "run_dir": str(target.run_dir),
        "model": target.model.alias,
        "checkpoint_path": target.checkpoint["checkpoint_path"],
        "accelerator": target.settings.get("accelerator"),
        "max_seq_len": target.settings["max_seq_len"],
        "timeout_seconds": target.timeout,
        "billing": f"the {what} bills per hour until `pup finetune undeploy`",
        "cleanup_command": target.cleanup_command,
    }


def deploy(
    request: DeployRequest,
    *,
    api_key: SecretStr,
    http: BasetenHTTP,
    loops: LoopsModule | None = None,
    tokenizer_loader: Callable[[ModelFamily], ChatTokenizer] = load_tokenizer,
    sleep: Sleep = time.sleep,
    clock: Clock = time.monotonic,
) -> dict[str, Any]:
    """Serve the run's best checkpoint and smoke-test its OpenAI routes. Paid.

    Resumes the deployment ``endpoint.json`` records for the same checkpoint and settings, and
    refuses while it records a different live one. A sampler that fails before it is ready is
    stopped; a dedicated deployment is kept so a rerun can resume it.
    """
    target = _target(request)
    load: Callable[..., None] | None = None
    if target.mode == "sampler":
        # Prepared before anything is paid: the SDK and the tokenizer must be available.
        family = target.model.family
        load = functools.partial(
            load_sampler_checkpoint,
            api_key,
            checkpoint_path=target.checkpoint["checkpoint_path"],
            base_model=target.checkpoint["base_model"],
            prompt=prompt_tokens(
                [{"role": "user", "content": "Hello"}], None, family, tokenizer_loader(family)
            ),
            ready_timeout=target.timeout,
            loops=loops_sdk(loops),
        )
    with DirectoryLock(target.run_dir):
        deployment = _open(target, http, sleep=sleep, clock=clock)
        if load is None:
            _serve_dedicated(deployment, target)
        else:
            _serve_sampler(deployment, target, load)
    receipt = deployment.data
    model_id, deployment_id = deployment.ids()
    output.info(f"OpenAI base URL: {receipt['openai_base_url']}")
    output.info(f"model: {receipt['served_model']}")
    output.info(f"Baseten model {model_id}, deployment {deployment_id}")
    output.info(f"stop billing with: {target.cleanup_command}")
    keys = (
        "state",
        "serving_mode",
        "openai_base_url",
        "served_model",
        "endpoint",
        "advertised_max_seq_len",
        "cleanup_command",
    )
    return {key: receipt.get(key) for key in keys}


def undeploy(
    run_dir: Path,
    *,
    http: BasetenHTTP,
    sleep: Sleep = time.sleep,
    clock: Clock = time.monotonic,
) -> dict[str, Any]:
    """Deactivate only the deployment ``endpoint.json`` records; the checkpoint and receipt stay."""
    run_dir = Path(run_dir).resolve()
    path = run_dir / RECEIPT_FILE
    if not path.is_file():
        raise StateError(f"{run_dir} records no deployment", hint="pass the --run-dir you deployed")
    with DirectoryLock(run_dir):
        deployment = _Deployment(path, _read_receipt(path) or {}, http, sleep=sleep, clock=clock)
        deployment.stop()
    receipt = deployment.data
    output.info(f"the Baseten deployment is inactive; the checkpoint and {path} are kept")
    return {
        "state": receipt["state"],
        "serving_mode": receipt.get("serving_mode"),
        "endpoint": receipt.get("endpoint"),
        "checkpoint_path": receipt["checkpoint"].get("checkpoint_path"),
        "receipt": str(path),
    }
