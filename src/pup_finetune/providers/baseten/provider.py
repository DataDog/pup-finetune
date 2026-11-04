# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The ``baseten`` provider: model profiles, preflight, rendering, options and training.

:class:`BasetenProvider` serves :data:`PROFILES`, checks with Baseten, read only, that the
workspace can train a model before anything is paid (:meth:`~BasetenProvider.preflight`) and
whether it trains it on serverless Loops (:meth:`~BasetenProvider.expected_placement`), from
one read of Loops' capabilities per provider, renders each profile with its model family's
Hugging Face chat template, and trains through :class:`~.backend.BasetenBackend`. It deploys
a run's sampler checkpoint as a dedicated Baseten model, smoke-tests it and deletes it
(:mod:`.deploy`: :meth:`~BasetenProvider.deploy`, :meth:`~BasetenProvider.smoke_test`,
:meth:`~BasetenProvider.undeploy`), and runs a run's training loop in a Baseten training job
for ``train --remote`` (:meth:`~BasetenProvider.remote_jobs`, :mod:`.jobs`).

Building the provider is cheap: the Loops SDK is imported by preflight and when a session
opens, ``transformers`` when a renderer first loads its tokenizer. The API key
(:data:`~pup_finetune.providers.baseten.CREDENTIALS`: ``BASETEN_API_KEY``, the OS keychain,
then ``~/.trussrc``) is resolved on the first API call or when a session opens, never earlier,
and handed to the clients directly, never through the environment.

Provider options (``TrainSpec.provider_options``, from ``--provider-option key=value``) are
listed on :class:`BasetenOptions`; string values are decoded as JSON.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Final, TypeGuard

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from pup_finetune import credentials
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.inference import ChatClient
from pup_finetune.providers.base import (
    DEDICATED,
    SERVERLESS,
    DeploymentRequest,
    DeployOptions,
    Endpoint,
    ModelProfile,
    ModelRef,
    Placement,
    SamplingUsage,
    SmokeTest,
    TrainingUsage,
    find_profile,
    sampling_usage_text,
    training_usage_text,
)
from pup_finetune.providers.baseten import PROVIDER_NAME
from pup_finetune.providers.baseten import deploy as serving
from pup_finetune.providers.baseten.backend import MIN_SEQ_LEN, BasetenBackend, Sleep
from pup_finetune.providers.baseten.client import (
    SOURCE,
    BasetenHTTP,
    LoopsModule,
    loops_sdk,
)
from pup_finetune.providers.options import parse_options
from pup_finetune.rendering.families import (
    GLM5_3,
    KIMI_K3,
    QWEN3_5,
    QWEN3_8,
    ModelFamily,
    get_family,
)
from pup_finetune.rendering.hf import ChatTokenizer, HFChatTemplateRenderer
from pup_finetune.runtime import Runtime

if TYPE_CHECKING:
    import httpx

    from pup_finetune.providers.base import Provider, SupportsPlacement, SupportsRemote
    from pup_finetune.providers.baseten.jobs import BasetenJobs, CliRunner

__all__ = [
    "LOOPS_VERSION",
    "MAX_SEQ_LEN",
    "PROFILES",
    "BasetenOptions",
    "BasetenProvider",
    "offers_serverless",
    "training_limit",
]

LOOPS_VERSION: Final = "0.33.0"
"""The ``baseten-loops`` version the ``baseten`` extra pins and the client is tested with."""


def _profile(
    alias: str,
    base_model: str,
    family: ModelFamily,
    *,
    max_seq_len: int,
    supports_reasoning: bool = False,
    notes: str = "",
) -> ModelProfile:
    return ModelProfile(
        provider=PROVIDER_NAME,
        alias=alias,
        base_model=base_model,
        tokenizer_model=family.tokenizer_model,
        tokenizer_revision=family.revision,
        renderer=family.key,
        max_seq_len=max_seq_len,
        supports_reasoning=supports_reasoning,
        notes=notes,
        trust_remote_code=family.trust_remote_code,
    )


PROFILES: Final[tuple[ModelProfile, ...]] = (
    _profile(
        "qwen3p5-9b",
        "Qwen/Qwen3.5-9B",
        QWEN3_5,
        max_seq_len=131_072,
        supports_reasoning=True,
        notes="Qwen3.5 9B; reasoning is kept after the last user turn only.",
    ),
    _profile(
        "glm-5p3-flash",
        "zai-org/GLM-5.3-Flash",
        GLM5_3,
        max_seq_len=131_072,
        supports_reasoning=True,
        notes="GLM-5.3 Flash; it stops by emitting the next role tag.",
    ),
    _profile(
        "qwen3p8-27b",
        "Qwen/Qwen3.8-27B",
        QWEN3_8,
        max_seq_len=262_144,
        supports_reasoning=True,
        notes="Qwen3.8 27B; reasoning history is preserved in training.",
    ),
    _profile(
        "kimi-k3",
        "moonshotai/Kimi-K3",
        KIMI_K3,
        max_seq_len=131_072,
        supports_reasoning=True,
        notes="Kimi K3; its repository's tokenizer code renders the chat.",
    ),
)
"""Every model we train on Baseten Loops; ``base_model`` is its Loops name and ``max_seq_len``
the context Loops' capabilities report for it, which preflight checks the workspace against.
The tokenizer and its pinned revision come from the model family, so a profile cannot drift
from its renderer."""

MAX_SEQ_LEN: Final = max(profile.max_seq_len for profile in PROFILES)
"""The largest training context of any profile: the bound on the ``max_seq_len`` option."""


class BasetenOptions(BaseModel):
    """The Baseten provider's options, all optional.

    ``max_seq_len`` caps the context below the profile's, in tokens (at least
    :data:`MIN_SEQ_LEN`, the smallest trainer): a run trains with the profile's context cut to
    it (:meth:`BasetenProvider.training_profile`), so a row with a longer example is left out
    when the rows are rendered, and its samplers and the trainer an evaluation opens are sized
    from it; preflight checks it against what Baseten trains the model on in the workspace.
    A training session also refuses to open for a longer sequence length, as a backstop.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    max_seq_len: int | None = Field(default=None, ge=MIN_SEQ_LEN, le=MAX_SEQ_LEN)

    @classmethod
    def parse(cls, options: Mapping[str, Any]) -> BasetenOptions:
        """Validate ``options``, decoding string values as JSON, or raise :class:`UsageError`."""
        return parse_options(cls, options, provider=PROVIDER_NAME, example="max_seq_len=32768")


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _model_entry(capabilities: Mapping[str, Any], base_model: str) -> Mapping[str, Any]:
    """``base_model``'s entry in ``capabilities``; :class:`UsageError` when it has none."""
    models = capabilities.get("supported_models")
    matches = [
        entry
        for entry in (models if isinstance(models, list) else [])
        if isinstance(entry, dict) and entry.get("model_name") == base_model
    ]
    if len(matches) != 1:
        raise UsageError(
            f"Baseten Loops does not offer {base_model} to this workspace",
            hint="choose another --model, or ask Baseten to enable this one",
        )
    return matches[0]


def training_limit(capabilities: Mapping[str, Any], base_model: str) -> int:
    """The longest sequence Baseten Loops trains ``base_model`` on in this workspace, in tokens.

    ``capabilities`` is what ``GET /v1/loops/capabilities`` returned. Raises
    :class:`UsageError` when the model is not offered or not enabled, or its limit is under
    :data:`MIN_SEQ_LEN`, and :class:`ExternalError` when Baseten reports no valid limit.
    """
    entry = _model_entry(capabilities, base_model)
    context = entry.get("max_context_length")
    enabled_limit = entry.get("max_enabled_seq_len", context)
    if entry.get("enabled") is False or enabled_limit == 0:
        found = entry.get("enablement_details")
        details = found if isinstance(found, dict) else {}
        reason = details.get("reason_detail") or details.get("reason")
        raise UsageError(
            f"this Baseten workspace is not enabled to train {base_model}"
            + (f": {reason}" if reason else ""),
            hint=str(details.get("remediation") or "ask Baseten to enable the model"),
        )
    if not (_is_int(context) and _is_int(enabled_limit)):
        raise ExternalError(
            f"Baseten returned an invalid context limit for {base_model}", source=SOURCE
        )
    limit = min(context, enabled_limit)
    if limit < MIN_SEQ_LEN:
        raise UsageError(
            f"Baseten trains {base_model} on at most {limit} tokens in this workspace, under "
            f"the {MIN_SEQ_LEN}-token minimum",
            hint="choose another --model, or ask Baseten to raise the limit",
        )
    return limit


def offers_serverless(capabilities: Mapping[str, Any], base_model: str) -> bool:
    """Whether this workspace can run ``base_model`` on serverless Loops (Baseten's shared
    capacity), from ``capabilities`` as for :func:`training_limit`.

    Baseten reports ``serverless`` only to workspaces in its early access and leaves it out
    for the rest, so a missing flag is ``False``. Raises :class:`UsageError` when the model
    is not offered and :class:`ExternalError` for a flag that is not a boolean.
    """
    flag = _model_entry(capabilities, base_model).get("serverless", False)
    if not isinstance(flag, bool):
        raise ExternalError(
            f"Baseten returned an invalid serverless flag for {base_model}", source=SOURCE
        )
    return flag


class BasetenProvider:
    """The ``baseten`` provider. See the module docstring.

    ``env`` is where the API key is looked up first (default: the process environment).
    ``transport`` replaces the network for Baseten's HTTP routes (a deployment's chat route
    included), ``loops`` the Loops SDK, ``tokenizer`` the Hugging Face tokenizer, ``sleep``
    the wait between polls and retries and ``clock`` the monotonic clock a deploy's waits run
    on, so tests run against the fakes in ``tests/fakes/baseten.py``; ``cli_runner``, ``which``
    and ``job_root`` replace how :meth:`remote_jobs` runs the ``baseten`` CLI, finds
    executables and where the job's files land.
    :meth:`validate_options` also makes the validated options this provider's configuration,
    which :meth:`training_profile` and :meth:`backend` use.
    """

    def __init__(
        self,
        runtime: Runtime,
        *,
        env: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        loops: LoopsModule | None = None,
        tokenizer: ChatTokenizer | None = None,
        sleep: Sleep = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        cli_runner: CliRunner | None = None,
        which: Callable[[str], str | None] | None = None,
        job_root: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.options = BasetenOptions()
        self._env = env
        self._transport = transport
        self._loops = loops
        self._tokenizer = tokenizer
        self._sleep = sleep
        self._clock = clock
        self._cli_runner = cli_runner
        self._which = which
        self._job_root = job_root
        self._api_key: SecretStr | None = None
        self._capabilities: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def models(self) -> list[ModelProfile]:
        return list(PROFILES)

    def resolve_model(self, alias_or_id: str) -> ModelProfile:
        return find_profile(PROFILES, alias_or_id, provider=PROVIDER_NAME)

    def preflight(self, profile: ModelProfile) -> int:
        """Confirm, read only, that this workspace can train ``profile`` on Baseten Loops, and
        return the longest sequence it trains it on (:func:`training_limit`).

        Loads the Loops SDK (refusing the environment overrides the client refuses) and asks
        Baseten's capabilities whether the model is offered and enabled. The limit can be
        under the profile's context (a workspace's default capacity), which ``train`` then
        trains at; a ``max_seq_len`` option over it is refused. Raises :class:`UsageError`, or
        :class:`ExternalError` when Baseten fails. Needs the API key.
        """
        self._own(profile)
        loops_sdk(self._loops)
        limit = training_limit(self._read_capabilities(), profile.base_model)
        max_seq_len = self.options.max_seq_len
        if max_seq_len is not None and limit < max_seq_len:
            raise UsageError(
                f"Baseten trains {profile.base_model} on at most {limit} tokens in this "
                f"workspace, shorter than the {max_seq_len} to train at",
                hint=f"pass --max-seq-len N with N at most {limit}",
            )
        return limit

    def expected_placement(self, profile: ModelProfile) -> Placement:
        """``serverless`` when Loops' capabilities say this workspace can run ``profile`` on
        serverless Loops, else ``dedicated``. From the capabilities :meth:`preflight` read.

        A forecast: no scope is sent, so Baseten still runs a trainer on a dedicated
        deployment when no shared pool fits its LoRA rank and context, and says nothing of
        it; only a sampler reports where it went (see :mod:`.backend`).
        """
        self._own(profile)
        serverless = offers_serverless(self._read_capabilities(), profile.base_model)
        return SERVERLESS if serverless else DEDICATED

    def renderer(self, profile: ModelProfile) -> HFChatTemplateRenderer:
        """The family's chat-template renderer; its tokenizer loads on first use."""
        self._own(profile)
        try:
            family = get_family(profile.renderer)
        except KeyError as exc:
            raise UsageError(
                f"{profile.alias} names unknown renderer {profile.renderer!r}"
            ) from exc
        return family.renderer(tokenizer=self._tokenizer)

    def training_profile(self, profile: ModelProfile) -> ModelProfile:
        """``profile`` with its context cut to the ``max_seq_len`` option, if one is set.

        Raises :class:`UsageError` when the option is over the profile's context.
        """
        self._own(profile)
        cap = self.options.max_seq_len
        if cap is None or cap == profile.max_seq_len:
            return profile
        if cap > profile.max_seq_len:
            raise UsageError(
                f"the max_seq_len provider option of {cap} is over the "
                f"{profile.max_seq_len}-token context of {profile.alias}",
                hint=f"pass --max-seq-len N with N at most {profile.max_seq_len}",
            )
        return dataclasses.replace(profile, max_seq_len=cap)

    # Previews are built before any trainer or sampler exists, so they name both placements.
    def training_billing(self, usage: TrainingUsage) -> str:
        return (
            "Baseten bills a trainer on serverless Loops by the token and one on a dedicated "
            "deployment for its GPUs' time, at its own rates, which pup-finetune does not "
            f"know: it trains on {training_usage_text(usage)}"
        )

    def sampling_billing(self, usage: SamplingUsage) -> str:
        return (
            "Baseten bills a sampler on serverless Loops by the token and one on a dedicated "
            "deployment for its GPU's time, at its own rates, which pup-finetune does not "
            f"know: {sampling_usage_text(usage)}"
        )

    def backend(self) -> BasetenBackend:
        """The training backend. Builds nothing remote; the key is resolved per session.

        The ``max_seq_len`` option caps the sequence length a session opens with (a backstop:
        :meth:`training_profile` already left longer rows out).
        """
        return BasetenBackend(
            self._key,
            renderer=self.renderer,
            max_seq_len=self.options.max_seq_len,
            loops=self._loops,
            transport=self._transport,
            sleep=self._sleep,
        )

    def validate_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        parsed = BasetenOptions.parse(options)
        self.options = parsed
        return parsed.model_dump(exclude_unset=True)

    def deploy(
        self,
        model: ModelRef,
        options: DeployOptions,
        *,
        on_creating: Callable[[DeploymentRequest], None] | None = None,
        on_created: Callable[[Endpoint, str], None] | None = None,
    ) -> Endpoint:
        """Deploy ``model``'s sampler checkpoint as the dedicated Baseten model
        ``options.deployment_id`` and wait until it serves; see :mod:`.deploy`. Needs the API
        key. It bills per GPU-second while a replica runs, until :meth:`undeploy`.
        """
        with self._http() as http:
            return serving.deploy(
                http,
                model,
                options,
                served={profile.base_model for profile in PROFILES},
                on_creating=on_creating,
                on_created=on_created,
                clock=self._clock,
                sleep=self._sleep,
            )

    def deployment_name(self, deployment_id: str) -> str | None:
        """``None``: Baseten names a deployment by ids it assigns when it creates one."""
        return None

    def owns_deployment(self, request: DeploymentRequest) -> bool | None:
        """Whether the Baseten model named ``request.deployment_id`` is the one ``request``
        creates."""
        with self._http() as http:
            return serving.owns_deployment(http, request)

    def smoke_test(self, endpoint: Endpoint) -> SmokeTest:
        """One chat completion to the deployment, retried while it scales up from zero."""
        self._own_endpoint(endpoint)
        return serving.smoke_test(endpoint, chat=self._chat, clock=self._clock, sleep=self._sleep)

    def undeploy(self, endpoint: Endpoint) -> bool:
        """Delete the deployment's Baseten model (idempotent); the checkpoint is kept."""
        self._own_endpoint(endpoint)
        with self._http() as http:
            return serving.undeploy(http, endpoint)

    def remote_jobs(self) -> BasetenJobs:
        """Baseten training jobs that run a run's training loop (``train --remote``); builds
        nothing remote."""
        from pup_finetune.providers.baseten.jobs import JOB_ROOT, BasetenJobs, run_cli

        return BasetenJobs(
            http=self._http,
            key=self._key,
            environ=self._environ(),
            runner=self._cli_runner or run_cli,
            which=self._which or shutil.which,
            root=self._job_root or JOB_ROOT,
        )

    # Internals.

    def _own(self, profile: ModelProfile) -> None:
        if profile.provider != PROVIDER_NAME:
            raise UsageError(
                f"the baseten provider does not serve {profile.provider}/{profile.alias}"
            )

    def _own_endpoint(self, endpoint: Endpoint) -> None:
        if endpoint.provider != PROVIDER_NAME:
            raise UsageError(f"the baseten provider does not serve a {endpoint.provider} endpoint")

    def _environ(self) -> Mapping[str, str]:
        return self._env if self._env is not None else os.environ

    def _key(self) -> SecretStr:
        if self._api_key is None:
            self._api_key = credentials.resolve(PROVIDER_NAME, env=self._environ()).value
        return self._api_key

    def _http(self) -> BasetenHTTP:
        return BasetenHTTP(self._key(), transport=self._transport)

    def _read_capabilities(self) -> dict[str, Any]:
        """Loops' capabilities for this workspace, read once per provider: a ``doctor
        --check`` preflights every model, and ``train`` asks for the limit and the
        placement."""
        if self._capabilities is None:
            with self._http() as http:
                self._capabilities = http.capabilities()
        return self._capabilities

    def _chat(self, base_url: str) -> ChatClient:
        # One attempt per call: the smoke test retries on its own clock while it scales up.
        return ChatClient(
            base_url,
            self._key(),
            credential=PROVIDER_NAME,
            max_attempts=1,
            transport=self._transport,
        )


if TYPE_CHECKING:
    # mypy checks that the provider satisfies the protocol.
    def _conforms(provider: BasetenProvider) -> Provider:
        return provider

    def _runs_remotely(provider: BasetenProvider) -> SupportsRemote:
        return provider

    def _places(provider: BasetenProvider) -> SupportsPlacement:
        return provider
