# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Fireworks provider: preflight, options, rendering and checks, against fakes."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from typing import Any

import keyring.core
import pytest
from pydantic import SecretStr

from pup_finetune import redaction
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.providers import get_provider, known_providers
from pup_finetune.providers.base import DeployOptions, LoraConfig, ModelRef, Provider
from pup_finetune.providers.fake import FAKE_PROFILE
from pup_finetune.providers.fireworks.api import FireworksAuthError
from pup_finetune.providers.fireworks.models import PROFILES
from pup_finetune.providers.fireworks.provider import (
    MAX_SEQ_LEN,
    DeployerFactory,
    FireworksProvider,
    preflight_problems,
)
from pup_finetune.rendering.hf import INSTALL_HINT, HFChatTemplateRenderer
from pup_finetune.runtime import Runtime
from tests.fakes.chat_api import FakeChatApi
from tests.fakes.fireworks import FakeFireworksTrainingClient
from tests.fakes.fireworks_api import API_KEY, FakeFireworksApi
from tests.fakes.fireworks_deploy import FakeClock, FakeSdkFireworks, fake_deployer
from tests.fakes.keyring import MemoryKeyring
from tests.fakes.tokenizer import FakeChatTokenizer
from tests.providers.contract import ProviderContract

PROFILE = PROFILES[0]
ENV = {"FIREWORKS_API_KEY": API_KEY}


@pytest.fixture(autouse=True)
def _clean_secrets() -> Iterator[None]:
    redaction.clear_secrets()
    yield
    redaction.clear_secrets()


@pytest.fixture
def fake() -> FakeFireworksApi:
    return FakeFireworksApi()


def provider_for(
    fake: FakeFireworksApi | None = None,
    env: dict[str, str] | None = None,
    training: FakeFireworksTrainingClient | None = None,
    keys: list[SecretStr] | None = None,
    deployer: DeployerFactory | None = None,
) -> FireworksProvider:
    fw = training or FakeFireworksTrainingClient()

    def training_client(key: SecretStr) -> FakeFireworksTrainingClient:
        if keys is not None:
            keys.append(key)
        return fw

    return FireworksProvider(
        Runtime(),
        env=ENV if env is None else env,
        transport=fake.transport if fake is not None else None,
        tokenizer=FakeChatTokenizer(),
        training_client=training_client,
        deployer=deployer,
    )


# --- the contract ------------------------------------------------------------------------


class TestFireworksProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        deployer = fake_deployer(FakeSdkFireworks(), FakeChatApi(api_key=API_KEY), FakeClock())
        return provider_for(FakeFireworksApi(), deployer=deployer)

    @pytest.fixture
    def model_ref(self, provider: Provider) -> ModelRef:
        return ModelRef(
            provider="fireworks",
            model_id="pft-contract-model",
            uri="accounts/synthetic-team/models/pft-contract-model",
        )

    def test_deploy_runs_on_the_fake_sdk(
        self, provider: Provider, model_ref: ModelRef, deploy_options: DeployOptions
    ) -> None:
        assert isinstance(provider, FireworksProvider)
        endpoint = provider.deploy(model_ref, deploy_options)
        assert endpoint.inference_model == (
            "accounts/synthetic-team/deployments/pft-contract-deployment"
        )
        assert provider.undeploy(endpoint) is True


# --- training ----------------------------------------------------------------------------


def test_the_backend_trains_with_the_resolved_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    keys: list[SecretStr] = []
    fw = FakeFireworksTrainingClient()
    backend = provider_for(training=fw, keys=keys).backend()
    assert keys == []  # building the backend resolves nothing
    with backend.open_session(PROFILE, LoraConfig(rank=8), seed=1, max_seq_len=1024) as session:
        assert session.session_id in fw.sessions
    assert [key.get_secret_value() for key in keys] == [API_KEY]
    assert "FIREWORKS_API_KEY" not in os.environ


def test_opening_a_session_without_a_key_asks_for_one() -> None:
    fw = FakeFireworksTrainingClient()
    backend = provider_for(env={}, training=fw).backend()
    with (
        pytest.raises(UsageError, match="no fireworks API key"),
        backend.open_session(PROFILE, LoraConfig(rank=8), seed=1, max_seq_len=1024),
    ):
        pass
    assert fw.calls == []


def test_the_max_seq_len_option_caps_the_backend() -> None:
    provider = provider_for()
    provider.validate_options({"max_seq_len": "512"})
    backend = provider.backend()
    with (
        pytest.raises(UsageError, match="over the max_seq_len provider option of 512"),
        backend.open_session(PROFILE, LoraConfig(rank=8), seed=1, max_seq_len=513),
    ):
        pass
    with backend.open_session(PROFILE, LoraConfig(rank=8), seed=1, max_seq_len=512):
        pass


def test_the_max_seq_len_option_cuts_the_training_profile() -> None:
    provider = provider_for()
    assert provider.training_profile(PROFILE) is PROFILE  # no option: unchanged
    provider.validate_options({"max_seq_len": "512"})
    capped = provider.training_profile(PROFILE)
    assert capped == dataclasses.replace(PROFILE, max_seq_len=512)
    assert provider.training_profile(capped) == capped  # a resumed run's profile stays put
    provider.validate_options({"max_seq_len": str(PROFILE.max_seq_len)})
    assert provider.training_profile(PROFILE) is PROFILE


def test_a_max_seq_len_option_over_the_profile_cannot_cut_it() -> None:
    provider = provider_for()
    provider.validate_options({"max_seq_len": "4096"})
    small = dataclasses.replace(PROFILE, max_seq_len=2048)
    with pytest.raises(UsageError, match="option of 4096 is over the 2048-token context"):
        provider.training_profile(small)


def test_the_training_profile_of_another_provider_is_refused() -> None:
    from pup_finetune.providers.fake import FAKE_PROFILE

    with pytest.raises(UsageError, match="does not serve"):
        provider_for().training_profile(FAKE_PROFILE)


# --- the registry ------------------------------------------------------------------------


def test_fireworks_is_registered_for_everyone() -> None:
    assert "fireworks" in known_providers(Runtime())
    provider = get_provider("fireworks", Runtime())
    assert isinstance(provider, FireworksProvider)
    assert provider.name == "fireworks"


# --- models ------------------------------------------------------------------------------


def test_models_are_the_profile_table() -> None:
    assert provider_for().models() == list(PROFILES)


def test_resolve_by_alias_base_model_or_tokenizer() -> None:
    provider = provider_for()
    assert provider.resolve_model("qwen3p8-27b") is PROFILE
    assert provider.resolve_model("accounts/fireworks/models/qwen3p8-27b") is PROFILE
    assert provider.resolve_model("Qwen/Qwen3.8-27B") is PROFILE


def test_an_unknown_model_lists_the_known_aliases() -> None:
    with pytest.raises(UsageError, match="unknown fireworks model 'gpt-x'") as caught:
        provider_for().resolve_model("gpt-x")
    assert caught.value.hint.startswith("known models: qwen3p8-27b, glm-5p3-flash;")


# --- preflight ---------------------------------------------------------------------------


def test_preflight_passes_for_a_ready_tunable_model(fake: FakeFireworksApi) -> None:
    provider_for(fake).preflight(PROFILE)
    (request,) = fake.requests
    assert request.url.path == "/v1/accounts/fireworks/models/qwen3p8-27b"


def test_preflight_resolves_the_key_once(fake: FakeFireworksApi) -> None:
    env = dict(ENV)
    provider = provider_for(fake, env=env)
    provider.preflight(PROFILE)
    env.clear()
    provider.preflight(PROFILE)
    assert len(fake.requests) == 2


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"state": "DEPLOYING"}, "its state is DEPLOYING, not READY"),
        ({"state": None}, "its state is unknown, not READY"),
        ({"supervisedLoraTunable": False}, "does not offer supervised LoRA tuning"),
        ({"supervisedLoraTunable": None}, "does not offer supervised LoRA tuning"),
        (
            {"huggingFaceUrl": "https://huggingface.co/Qwen/Qwen3.8-8B"},
            "its Hugging Face model is https://huggingface.co/Qwen/Qwen3.8-8B, not "
            "Qwen/Qwen3.8-27B",
        ),
        (
            {"huggingFaceUrl": "https://example.com/Qwen/Qwen3.8-27B"},
            "its Hugging Face model is https://example.com/Qwen/Qwen3.8-27B",
        ),
        ({"huggingFaceUrl": None}, "its Hugging Face model is not reported"),
        ({"trainingContextLength": 32768}, "its training context is 32768 tokens, shorter"),
        ({"trainingContextLength": None}, "does not report its training context length"),
    ],
    ids=[
        "not-ready",
        "no-state",
        "not-tunable",
        "tunable-unknown",
        "other-hf-model",
        "not-huggingface",
        "no-hf-url",
        "short-context",
        "no-context",
    ],
)
def test_preflight_refuses_an_unusable_model(
    fake: FakeFireworksApi, change: dict[str, Any], problem: str
) -> None:
    model = fake.models["qwen3p8-27b"]
    for key, value in change.items():
        if value is None:
            model.pop(key, None)
        else:
            model[key] = value
    with pytest.raises(UsageError) as caught:
        provider_for(fake).preflight(PROFILE)
    assert caught.value.message.startswith(
        "Fireworks cannot fine-tune accounts/fireworks/models/qwen3p8-27b right now: "
    )
    assert problem in caught.value.message
    assert caught.value.hint is not None


def test_preflight_reports_every_problem_at_once(fake: FakeFireworksApi) -> None:
    fake.models["qwen3p8-27b"].update(state="FAILED", supervisedLoraTunable=False)
    with pytest.raises(UsageError) as caught:
        provider_for(fake).preflight(PROFILE)
    assert "state is FAILED" in caught.value.message
    assert "supervised LoRA" in caught.value.message


def test_preflight_refuses_a_missing_model(fake: FakeFireworksApi) -> None:
    del fake.models["qwen3p8-27b"]
    with pytest.raises(UsageError, match="Fireworks has no model"):
        provider_for(fake).preflight(PROFILE)


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/Qwen/Qwen3.8-27B/",
        "https://huggingface.co/qwen/qwen3.8-27b",
        "https://www.huggingface.co/Qwen/Qwen3.8-27B",
        " https://huggingface.co/Qwen/Qwen3.8-27B ",
    ],
)
def test_equivalent_hugging_face_urls_match(fake: FakeFireworksApi, url: str) -> None:
    fake.models["qwen3p8-27b"]["huggingFaceUrl"] = url
    provider_for(fake).preflight(PROFILE)


def test_a_shorter_max_seq_len_option_fits_a_shorter_context(fake: FakeFireworksApi) -> None:
    fake.models["qwen3p8-27b"]["trainingContextLength"] = 32768
    provider = provider_for(fake)
    provider.validate_options({"max_seq_len": "32768"})
    provider.preflight(PROFILE)
    provider.validate_options({"max_seq_len": "32769"})
    with pytest.raises(UsageError, match="32768 tokens, shorter than the 32769"):
        provider.preflight(PROFILE)


def test_a_max_seq_len_option_over_the_profile_is_refused(fake: FakeFireworksApi) -> None:
    provider = provider_for(fake)
    provider.validate_options({"max_seq_len": "4096"})
    small = dataclasses.replace(PROFILE, max_seq_len=2048)
    with pytest.raises(UsageError, match="over the 2048-token context"):
        provider.preflight(small)
    assert fake.requests == []


def test_preflight_without_a_key_asks_for_one(fake: FakeFireworksApi) -> None:
    with pytest.raises(UsageError, match="no fireworks API key found") as caught:
        provider_for(fake, env={}).preflight(PROFILE)
    assert caught.value.hint is not None
    assert "pup finetune auth login fireworks" in caught.value.hint
    assert fake.requests == []


def test_preflight_reads_the_key_from_the_keychain(
    fake: FakeFireworksApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    keychain = MemoryKeyring()
    keychain.passwords[("pup-finetune", "fireworks")] = API_KEY
    monkeypatch.setattr(keyring.core, "_keyring_backend", keychain)
    provider_for(fake, env={}).preflight(PROFILE)
    assert fake.requests[0].headers["authorization"] == f"Bearer {API_KEY}"
    assert "FIREWORKS_API_KEY" not in os.environ
    assert API_KEY not in os.environ.values()


def test_preflight_with_a_rejected_key_is_an_auth_error(fake: FakeFireworksApi) -> None:
    with pytest.raises(FireworksAuthError) as caught:
        provider_for(fake, env={"FIREWORKS_API_KEY": "fw-synthetic-revoked-key"}).preflight(PROFILE)
    assert "fw-synthetic-revoked-key" not in str(caught.value)


def test_preflight_with_an_api_outage_is_external(fake: FakeFireworksApi) -> None:
    fake.status["/v1/accounts/fireworks/models/qwen3p8-27b"] = 503
    with pytest.raises(ExternalError, match="HTTP 503"):
        provider_for(fake).preflight(PROFILE)


def test_preflight_refuses_another_providers_profile(fake: FakeFireworksApi) -> None:
    with pytest.raises(UsageError, match="does not serve fake/fake-model"):
        provider_for(fake).preflight(FAKE_PROFILE)
    assert fake.requests == []


def test_preflight_problems_is_empty_for_a_good_model() -> None:
    from pup_finetune.providers.fireworks.api import ModelInfo

    info = ModelInfo.model_validate(FakeFireworksApi().models["qwen3p8-27b"])
    assert preflight_problems(PROFILE, info, max_seq_len=PROFILE.max_seq_len) == []


# --- options -----------------------------------------------------------------------------


def test_no_options_normalize_to_none() -> None:
    assert provider_for().validate_options({}) == {}


def test_max_seq_len_is_decoded_and_normalized() -> None:
    provider = provider_for()
    normalized = provider.validate_options({"max_seq_len": "4096"})
    assert normalized == {"max_seq_len": 4096}
    assert provider.validate_options(normalized) == normalized
    assert provider.options.max_seq_len == 4096
    assert provider.validate_options({"max_seq_len": MAX_SEQ_LEN}) == {"max_seq_len": 131072}


@pytest.mark.parametrize("value", ["0", "-1", str(MAX_SEQ_LEN + 1), "4096.5", "true", '"4096"'])
def test_an_invalid_max_seq_len_is_refused(value: str) -> None:
    with pytest.raises(UsageError, match="invalid fireworks provider options: max_seq_len"):
        provider_for().validate_options({"max_seq_len": value})


def test_a_value_that_is_not_json_is_refused() -> None:
    with pytest.raises(UsageError, match="must be JSON"):
        provider_for().validate_options({"max_seq_len": "lots"})


def test_unknown_options_are_refused() -> None:
    with pytest.raises(UsageError, match="unknown fireworks provider option") as caught:
        provider_for().validate_options({"step_delay_s": "1", "max_seq_len": "8"})
    assert "step_delay_s" in caught.value.message
    assert caught.value.hint == "known options: max_seq_len"


# --- rendering ---------------------------------------------------------------------------


def test_the_renderer_is_the_familys() -> None:
    renderer = provider_for().renderer(PROFILE)
    assert isinstance(renderer, HFChatTemplateRenderer)
    assert renderer.name == "qwen3_8"
    assert renderer.prefix_stable is False
    identity = renderer.identity()
    assert identity["tokenizer"] == "Qwen/Qwen3.8-27B"
    assert identity["tokenizer_revision"] == PROFILE.tokenizer_revision


def test_the_renderer_needs_the_hf_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)  # what a missing package looks like
    provider = FireworksProvider(Runtime(), env=ENV)
    with pytest.raises(UsageError, match="needs the Hugging Face transformers package") as caught:
        provider.renderer(PROFILE)
    assert caught.value.hint == INSTALL_HINT


def test_the_renderer_refuses_another_providers_profile() -> None:
    with pytest.raises(UsageError, match="does not serve"):
        provider_for().renderer(FAKE_PROFILE)


def test_building_the_provider_and_renderer_imports_nothing_heavy() -> None:
    code = textwrap.dedent(
        """
        import json, sys
        from pup_finetune.errors import UsageError
        from pup_finetune.providers import get_provider
        from pup_finetune.runtime import Runtime

        provider = get_provider("fireworks", Runtime())
        provider.validate_options({"max_seq_len": "1024"})
        try:
            provider.renderer(provider.resolve_model("qwen3p8-27b"))
        except UsageError:  # no hf extra here: the imports are what this checks
            pass
        heavy = {"transformers", "fireworks", "tinker", "httpx"}
        print(json.dumps(sorted({name.partition(".")[0] for name in sys.modules} & heavy)))
        """
    )
    completed = subprocess.run(  # noqa: S603 - this interpreter, a fixed script
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": "/nonexistent",
            "PYTHON_KEYRING_BACKEND": "keyring.backends.fail.Keyring",
        },
    )
    assert json.loads(completed.stdout) == []


def test_building_the_backend_loads_no_sdk() -> None:
    code = textwrap.dedent(
        """
        import json, sys
        from pup_finetune.providers import get_provider
        from pup_finetune.runtime import Runtime

        backend = get_provider("fireworks", Runtime()).backend()
        assert backend.supports_resume and backend.supports_promotion
        heavy = {"transformers", "fireworks", "tinker"}
        print(json.dumps(sorted({name.partition(".")[0] for name in sys.modules} & heavy)))
        """
    )
    completed = subprocess.run(  # noqa: S603 - this interpreter, a fixed script
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": "/nonexistent",
            "PYTHON_KEYRING_BACKEND": "keyring.backends.fail.Keyring",
        },
    )
    assert json.loads(completed.stdout) == []
