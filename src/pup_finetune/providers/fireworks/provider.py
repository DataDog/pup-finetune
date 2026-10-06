# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The ``fireworks`` provider: model profiles, preflight, rendering and options.

:class:`FireworksProvider` serves the profiles in :mod:`.models`, checks with the Fireworks
API that a model can be fine-tuned right now (:meth:`FireworksProvider.preflight`), and
renders each profile with its model family's Hugging Face chat template. Training and
deployment on Fireworks land in a later release; until then :meth:`~FireworksProvider.backend`,
:meth:`~FireworksProvider.deploy` and :meth:`~FireworksProvider.undeploy` refuse with a
:class:`UsageError`.

Building the provider is cheap: the HTTP client is imported on the first API call, and
``transformers`` only when a renderer first needs its tokenizer. The API key is resolved
(``FIREWORKS_API_KEY``, then the OS keychain) on the first API call, never earlier.

Provider options (``TrainSpec.provider_options``, from ``--provider-option key=value``) are
listed on :class:`FireworksOptions`; string values are decoded as JSON.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, NoReturn
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from pup_finetune import credentials
from pup_finetune.doctor import error_detail
from pup_finetune.errors import FinetuneError, UsageError
from pup_finetune.providers.base import Check, Endpoint, ModelProfile, ModelRef
from pup_finetune.providers.fireworks import PROVIDER_NAME
from pup_finetune.providers.fireworks.models import PROFILES
from pup_finetune.providers.options import parse_options
from pup_finetune.redaction import redact
from pup_finetune.rendering.families import get_family
from pup_finetune.rendering.hf import INSTALL_HINT, ChatTokenizer, HFChatTemplateRenderer
from pup_finetune.runtime import Runtime

if TYPE_CHECKING:
    import httpx

    from pup_finetune.providers.base import Provider, TrainingBackend
    from pup_finetune.providers.fireworks.api import FireworksApi, ModelInfo

__all__ = ["MAX_SEQ_LEN", "FireworksOptions", "FireworksProvider", "preflight_problems"]

MAX_SEQ_LEN: Final = max(profile.max_seq_len for profile in PROFILES)
"""The largest training context of any profile: the bound on the ``max_seq_len`` option."""

READY: Final = "READY"
_HF_HOSTS: Final = frozenset({"huggingface.co", "www.huggingface.co"})
_LATER_RELEASE: Final = (
    "is not available in this version of pup-finetune; it lands in a later release"
)


class FireworksOptions(BaseModel):
    """The Fireworks provider's options, all optional.

    ``max_seq_len`` caps the training context below the profile's, in tokens; preflight
    checks it against both the profile and what Fireworks reports for the model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    max_seq_len: int | None = Field(default=None, ge=1, le=MAX_SEQ_LEN)

    @classmethod
    def parse(cls, options: Mapping[str, Any]) -> FireworksOptions:
        """Validate ``options``, decoding string values as JSON, or raise :class:`UsageError`."""
        return parse_options(cls, options, provider=PROVIDER_NAME, example="max_seq_len=32768")


def _hf_repo(url: str | None) -> str | None:
    """The repository id in a Hugging Face model URL, e.g. ``Qwen/Qwen3.8-27B``."""
    if not url:
        return None
    parts = urlsplit(url.strip())
    if parts.scheme not in ("https", "http") or parts.hostname not in _HF_HOSTS:
        return None
    return parts.path.strip("/") or None


def preflight_problems(profile: ModelProfile, info: ModelInfo, *, max_seq_len: int) -> list[str]:
    """Why Fireworks' ``info`` says ``profile`` cannot be trained at ``max_seq_len``; empty if ok."""
    problems: list[str] = []
    if info.state != READY:
        problems.append(f"its state is {info.state or 'unknown'}, not {READY}")
    if not info.supervised_lora_tunable:
        problems.append("Fireworks does not offer supervised LoRA tuning for it")
    repo = _hf_repo(info.hugging_face_url)
    if repo is None or repo.casefold() != profile.tokenizer_model.casefold():
        problems.append(
            f"its Hugging Face model is {info.hugging_face_url or 'not reported'}, not "
            f"{profile.tokenizer_model}, the tokenizer the profile renders with"
        )
    limit = info.training_context_length
    if limit is None:
        problems.append("Fireworks does not report its training context length")
    elif limit < max_seq_len:
        problems.append(
            f"its training context is {limit} tokens, shorter than the {max_seq_len} to train at"
        )
    return problems


class FireworksProvider:
    """The ``fireworks`` provider. See the module docstring.

    ``env`` is where the API key is looked up first (default: the process environment);
    ``transport`` replaces the network for the Fireworks API and ``tokenizer`` the Hugging
    Face tokenizer, for tests. :meth:`validate_options` also makes the validated options this
    provider's configuration, which :meth:`preflight` checks against.
    """

    def __init__(
        self,
        runtime: Runtime,
        *,
        env: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        tokenizer: ChatTokenizer | None = None,
    ) -> None:
        self.runtime = runtime
        self.options = FireworksOptions()
        self._env = env
        self._transport = transport
        self._tokenizer = tokenizer
        self._api_key: SecretStr | None = None

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def check(self) -> list[Check]:
        return [self._key_check(), self._renderer_check()]

    def models(self) -> list[ModelProfile]:
        return list(PROFILES)

    def resolve_model(self, alias_or_id: str) -> ModelProfile:
        for profile in PROFILES:
            if alias_or_id in (profile.alias, profile.base_model):
                return profile
        known = ", ".join(profile.alias for profile in PROFILES)
        raise UsageError(f"unknown fireworks model {alias_or_id!r}", hint=f"known models: {known}")

    def preflight(self, profile: ModelProfile) -> None:
        """Confirm with Fireworks that ``profile``'s base model can be LoRA-tuned right now.

        Raises :class:`UsageError` when the model is missing, not ``READY``, not tunable with
        supervised LoRA, backed by a different Hugging Face model than the profile's tokenizer,
        or has a shorter training context than the one to train at (the ``max_seq_len``
        option, else the profile's); :class:`~pup_finetune.errors.ExternalError` when the API
        fails or rejects the key. Needs the API key.
        """
        self._own(profile)
        max_seq_len = self.options.max_seq_len or profile.max_seq_len
        if max_seq_len > profile.max_seq_len:
            raise UsageError(
                f"max_seq_len {max_seq_len} is over the {profile.max_seq_len}-token context "
                f"of {profile.alias}"
            )
        with self._api() as api:
            info = api.model_info(profile.base_model)
        problems = preflight_problems(profile, info, max_seq_len=max_seq_len)
        if problems:
            raise UsageError(
                f"Fireworks cannot fine-tune {profile.base_model} right now: {'; '.join(problems)}",
                hint="check `pup finetune models list --provider fireworks --check`, or choose "
                "another model",
            )

    def renderer(self, profile: ModelProfile) -> HFChatTemplateRenderer:
        """The family's chat-template renderer; its tokenizer loads on first use.

        Raises :class:`UsageError` with an install hint when the ``hf`` extra is missing.
        """
        self._own(profile)
        try:
            family = get_family(profile.renderer)
        except KeyError as exc:
            raise UsageError(
                f"{profile.alias} names unknown renderer {profile.renderer!r}"
            ) from exc
        if self._tokenizer is None and not _transformers_installed():
            raise UsageError(
                "rendering for Fireworks needs Hugging Face transformers, which is not installed",
                hint=INSTALL_HINT,
            )
        return family.renderer(tokenizer=self._tokenizer)

    def backend(self) -> TrainingBackend:
        _later("training on Fireworks")

    def validate_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        parsed = FireworksOptions.parse(options)
        self.options = parsed
        return parsed.model_dump(exclude_unset=True)

    def deploy(self, model: ModelRef, options: Mapping[str, Any]) -> Endpoint:
        _later("deploying on Fireworks")

    def undeploy(self, endpoint: Endpoint) -> None:
        _later("removing a Fireworks deployment")

    # Internals.

    def _own(self, profile: ModelProfile) -> None:
        if profile.provider != PROVIDER_NAME:
            raise UsageError(
                f"the fireworks provider does not serve {profile.provider}/{profile.alias}"
            )

    def _environ(self) -> Mapping[str, str]:
        return self._env if self._env is not None else os.environ

    def _api(self) -> FireworksApi:
        from pup_finetune.providers.fireworks.api import FireworksApi

        if self._api_key is None:
            self._api_key = credentials.resolve(PROVIDER_NAME, env=self._environ()).value
        return FireworksApi(self._api_key, transport=self._transport)

    def _key_check(self) -> Check:
        name = "fireworks api key"
        try:
            found = credentials.lookup(PROVIDER_NAME, env=self._environ())
        except FinetuneError as exc:
            return Check(name, False, redact(error_detail(exc)))
        except Exception as exc:
            return Check(name, False, redact(f"could not look up the key ({type(exc).__name__})"))
        if found is None:
            hint = credentials.login_hint(credentials.get_spec(PROVIDER_NAME))
            return Check(name, False, f"no Fireworks API key; {hint}")
        return Check(name, True, f"Fireworks API key found (source: {found.source})")

    def _renderer_check(self) -> Check:
        name = "fireworks renderer"
        if self._tokenizer is not None or _transformers_installed():
            return Check(name, True, "Hugging Face transformers is installed")
        return Check(name, False, f"Hugging Face transformers is not installed; {INSTALL_HINT}")


def _transformers_installed() -> bool:
    """Whether ``transformers`` can be imported, without importing it."""
    try:
        return importlib.util.find_spec("transformers") is not None
    except (ImportError, ValueError):
        return False


def _later(action: str) -> NoReturn:
    raise UsageError(
        f"{action} {_LATER_RELEASE}",
        hint="meanwhile, check model readiness with "
        "`pup finetune models list --provider fireworks --check`",
    )


if TYPE_CHECKING:
    # mypy checks that the provider satisfies the protocol.
    def _conforms(provider: FireworksProvider) -> Provider:
        return provider
