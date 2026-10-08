# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Baseten provider against the Loops fake: the renderer and provider contracts, then
registration, profiles, preflight, checks, the key and options. Deployment is in
test_baseten_deploy.py."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from pup_finetune import redaction
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.providers import get_provider, known_providers
from pup_finetune.providers.base import (
    DeployOptions,
    LoraConfig,
    ModelProfile,
    ModelRef,
    Provider,
)
from pup_finetune.providers.baseten import client
from pup_finetune.providers.baseten.provider import (
    MAX_SEQ_LEN,
    PROFILES,
    BasetenProvider,
    training_limit,
)
from pup_finetune.providers.fake import FAKE_PROFILE
from pup_finetune.rendering.base import Renderer
from pup_finetune.rendering.families import GLM5_3, QWEN3_5
from pup_finetune.rendering.hf import ChatTokenizer
from pup_finetune.runtime import Runtime
from tests.fakes.baseten import Answer, BasetenTransport, FakeLoops, capabilities
from tests.fakes.fireworks_deploy import FakeClock
from tests.fakes.tokenizer import FakeChatTokenizer, glm5_like
from tests.providers.contract import ProviderContract, RendererContract

KEY = SecretStr("synthetic-baseten-key")
ENV = {"BASETEN_API_KEY": KEY.get_secret_value()}
CAPABILITIES_ROUTE = "GET /v1/loops/capabilities"
CAPABILITIES = capabilities(*(profile.base_model for profile in PROFILES))
QWEN = PROFILES[0]
FAKE_TOKENIZERS: dict[str, Callable[[], ChatTokenizer]] = {
    "qwen3_5": lambda: FakeChatTokenizer(merge_newlines=True),
    "glm5_3": lambda: FakeChatTokenizer(template=glm5_like),
}
"""A fake tokenizer per renderer family, with a template like the family's."""


def provider_on(
    loops: FakeLoops, tokenizer: ChatTokenizer | None = None, *, clock: FakeClock | None = None
) -> BasetenProvider:
    """The real provider on the Loops fake, with a key and a workspace offering every model;
    its waits run on ``clock`` (by default, a new one)."""
    time = clock or FakeClock()
    return BasetenProvider(
        Runtime(),
        env=ENV,
        transport=loops.transport({CAPABILITIES_ROUTE: (200, CAPABILITIES)}),
        loops=loops,
        tokenizer=tokenizer or FakeChatTokenizer(),
        sleep=time.sleep,
        clock=time.now,
    )


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for name in (*client._REJECTED_ENV, "LOOPS_BASE_URL", "BASETEN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.trussrc
    yield
    redaction.clear_secrets()


def provider_for(
    loops: FakeLoops | None = None,
    *,
    env: dict[str, str] | None = None,
    caps: Answer = (200, CAPABILITIES),
) -> tuple[BasetenProvider, BasetenTransport]:
    fake = loops or FakeLoops()
    transport = fake.transport({CAPABILITIES_ROUTE: caps})
    provider = BasetenProvider(
        Runtime(),
        env=ENV if env is None else env,
        transport=transport,
        loops=fake,
        tokenizer=FakeChatTokenizer(merge_newlines=True),
        sleep=lambda _: None,
    )
    return provider, transport


# --- the contracts -----------------------------------------------------------------------


class TestBasetenRenderer(RendererContract):
    @pytest.fixture(params=PROFILES, ids=lambda profile: profile.alias)
    def renderer_factory(self, request: pytest.FixtureRequest) -> Callable[[], Renderer]:
        profile: ModelProfile = request.param
        tokenizer = FAKE_TOKENIZERS[profile.renderer]
        return lambda: provider_on(FakeLoops(), tokenizer()).renderer(profile)


class TestBasetenProviderContract(ProviderContract):
    @pytest.fixture
    def loops(self) -> FakeLoops:
        return FakeLoops()

    @pytest.fixture
    def provider(self, loops: FakeLoops) -> Provider:
        return provider_on(loops)

    @pytest.fixture
    def model_ref(self, loops: FakeLoops) -> ModelRef:
        uri = loops.sampler_checkpoint("best", base_model=QWEN.base_model)
        return ModelRef(provider="baseten", model_id="best", uri=uri)

    @pytest.fixture
    def deploy_options(self) -> DeployOptions:
        return DeployOptions(deployment_id="pft-contract-deployment", shape="H100:1")


# --- registration and profiles -----------------------------------------------------------


def test_baseten_is_registered_for_everyone() -> None:
    assert "baseten" in known_providers(Runtime())
    provider = get_provider("baseten", Runtime())
    assert isinstance(provider, BasetenProvider)
    assert provider.name == "baseten"


def test_building_the_provider_loads_no_sdk_tokenizer_or_key(tmp_path: Path) -> None:
    code = textwrap.dedent(
        """
        import json, sys
        from pup_finetune.providers import get_provider
        from pup_finetune.runtime import Runtime

        provider = get_provider("baseten", Runtime())
        provider.validate_options({"max_seq_len": "4096"})
        provider.backend()
        provider.renderer(provider.resolve_model("qwen3p5-9b"))
        print(json.dumps(sorted(
            name for name in ("baseten", "transformers", "keyring") if name in sys.modules
        )))
        """
    )
    env = {key: value for key, value in os.environ.items() if key != "BASETEN_API_KEY"}
    completed = subprocess.run(  # noqa: S603 - this interpreter, a fixed script
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**env, "HOME": str(tmp_path)},
    )
    assert json.loads(completed.stdout) == []


def test_profiles_take_their_tokenizer_from_the_family() -> None:
    by_alias = {profile.alias: profile for profile in PROFILES}
    assert set(by_alias) == {"qwen3p5-9b", "glm-5p3-flash"}
    for alias, base_model, family in [
        ("qwen3p5-9b", "Qwen/Qwen3.5-9B", QWEN3_5),
        ("glm-5p3-flash", "zai-org/GLM-5.3-Flash", GLM5_3),
    ]:
        profile = by_alias[alias]
        assert profile.base_model == base_model
        assert (profile.tokenizer_model, profile.tokenizer_revision, profile.renderer) == (
            family.tokenizer_model,
            family.revision,
            family.key,
        )
        assert profile.max_seq_len == 131_072
    assert by_alias["qwen3p5-9b"].supports_reasoning
    assert not by_alias["glm-5p3-flash"].supports_reasoning


def test_an_unknown_model_lists_the_known_aliases() -> None:
    provider, _ = provider_for()
    with pytest.raises(UsageError, match="unknown baseten model 'no-such-model'") as caught:
        provider.resolve_model("no-such-model")
    assert caught.value.hint == "known models: qwen3p5-9b, glm-5p3-flash"


# --- preflight ---------------------------------------------------------------------------


def test_preflight_only_reads_the_capabilities() -> None:
    loops = FakeLoops()
    provider, transport = provider_for(loops)
    for profile in PROFILES:
        provider.preflight(profile)
    assert transport.calls == [CAPABILITIES_ROUTE] * len(PROFILES)
    assert transport.requests[0].headers["authorization"] == f"Bearer {KEY.get_secret_value()}"
    assert loops.calls == []  # no session, nothing paid


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"model_name": "Qwen/Qwen3.5-27B"}, "does not offer Qwen/Qwen3.5-9B"),
        (
            {
                "enabled": False,
                "enablement_details": {"reason": "quota", "remediation": "contact support"},
            },
            "not enabled to train Qwen/Qwen3.5-9B: quota",
        ),
        ({"max_enabled_seq_len": 0}, "not enabled to train Qwen/Qwen3.5-9B"),
        ({"max_enabled_seq_len": 512}, "at most 512 tokens in this workspace"),
        ({"max_context_length": 1000}, "under the 1024-token minimum"),
    ],
    ids=["not-offered", "disabled", "no-enabled-length", "enabled-too-short", "context-too-short"],
)
def test_preflight_refuses_a_model_the_workspace_cannot_train(
    fields: dict[str, Any], message: str
) -> None:
    provider, _ = provider_for(caps=(200, capabilities(QWEN.base_model, **fields)))
    with pytest.raises(UsageError, match=message) as caught:
        provider.preflight(QWEN)
    assert caught.value.hint


@pytest.mark.parametrize("option", [None, "65536"], ids=["profile", "option"])
def test_preflight_refuses_a_context_over_the_workspace_limit(option: str | None) -> None:
    caps = capabilities(QWEN.base_model, max_enabled_seq_len=32_768)
    provider, _ = provider_for(caps=(200, caps))
    provider.validate_options({} if option is None else {"max_seq_len": option})
    needed = option or str(QWEN.max_seq_len)
    with pytest.raises(UsageError, match=f"at most 32768 tokens .* the {needed} to train at") as e:
        provider.preflight(provider.training_profile(QWEN))
    assert e.value.hint == "pass --max-seq-len N with N at most 32768"
    provider.validate_options({"max_seq_len": "32768"})
    provider.preflight(provider.training_profile(QWEN))
    provider.preflight(QWEN)  # the option, not the uncut profile, is what trains


def test_the_remediation_baseten_gives_is_the_hint() -> None:
    details = {"reason_detail": "not on your plan", "remediation": "upgrade the plan"}
    caps = capabilities(QWEN.base_model, enabled=False, enablement_details=details)
    with pytest.raises(UsageError, match="not on your plan") as caught:
        training_limit(caps, QWEN.base_model)
    assert caught.value.hint == "upgrade the plan"


def test_the_limit_is_the_smaller_of_context_and_enabled_length() -> None:
    caps = capabilities(QWEN.base_model, max_context_length=65_536, max_enabled_seq_len=32_768)
    assert training_limit(caps, QWEN.base_model) == 32_768
    del caps["supported_models"][0]["max_enabled_seq_len"]
    assert training_limit(caps, QWEN.base_model) == 65_536


@pytest.mark.parametrize(
    "caps",
    [
        capabilities(QWEN.base_model, max_context_length=None),
        capabilities(QWEN.base_model, max_enabled_seq_len="131072"),
        capabilities(QWEN.base_model, max_context_length=True),
    ],
    ids=["missing", "string", "bool"],
)
def test_an_invalid_limit_is_baseten_s_fault(caps: dict[str, Any]) -> None:
    with pytest.raises(ExternalError, match="invalid context limit"):
        training_limit(caps, QWEN.base_model)


@pytest.mark.parametrize(
    "caps", [{}, {"supported_models": None}, capabilities(QWEN.base_model, QWEN.base_model)]
)
def test_no_single_offer_is_not_offered(caps: dict[str, Any]) -> None:
    with pytest.raises(UsageError, match="does not offer"):
        training_limit(caps, QWEN.base_model)


def test_preflight_without_a_key_asks_for_one_before_any_request() -> None:
    provider, transport = provider_for(env={})
    with pytest.raises(UsageError, match="no baseten API key found") as caught:
        provider.preflight(QWEN)
    assert caught.value.hint is not None
    assert "pup finetune auth login baseten" in caught.value.hint
    assert transport.calls == []


def test_preflight_refuses_a_loops_override_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOOPS_TEAM", "")
    provider, transport = provider_for()
    with pytest.raises(UsageError, match="LOOPS_TEAM is set"):
        provider.preflight(QWEN)
    assert transport.calls == []


def test_another_provider_s_profile_is_refused() -> None:
    provider, transport = provider_for()
    for call in (provider.preflight, provider.renderer, provider.training_profile):
        with pytest.raises(UsageError, match="does not serve fake/fake-model"):
            call(FAKE_PROFILE)
    with pytest.raises(UsageError, match="names unknown renderer 'nope'"):
        provider.renderer(dataclasses.replace(QWEN, renderer="nope"))
    assert transport.calls == []


# --- the key and the backend -------------------------------------------------------------


def test_the_backend_trains_with_the_key_resolved_once() -> None:
    loops = FakeLoops()
    env = dict(ENV)
    provider, _ = provider_for(loops, env=env)
    provider.preflight(QWEN)
    env.clear()  # resolved once, by preflight
    backend = provider.backend()
    with backend.open_session(QWEN, LoraConfig(rank=8), seed=1, max_seq_len=1024) as session:
        assert session.session_id == "run1"
    assert [kwargs["api_key"] for kwargs in loops.kwargs("ServiceClient")] == [
        KEY.get_secret_value()
    ]
    assert not loops.runs["run1"].active
    assert "BASETEN_API_KEY" not in os.environ


def test_opening_a_session_without_a_key_asks_for_one() -> None:
    loops = FakeLoops()
    provider, transport = provider_for(loops, env={})
    backend = provider.backend()  # building it resolves nothing
    with (
        pytest.raises(UsageError, match="no baseten API key found"),
        backend.open_session(QWEN, LoraConfig(rank=8), seed=1, max_seq_len=1024),
    ):
        pass
    assert loops.calls == []
    assert transport.calls == []


def test_the_key_can_come_from_the_trussrc(tmp_path: Path) -> None:
    (tmp_path / ".trussrc").write_text(
        f"[baseten]\nremote_provider = baseten\napi_key = {KEY.get_secret_value()}\n"
        "remote_url = https://app.baseten.co\n"
    )
    provider, transport = provider_for(env={})
    provider.preflight(QWEN)
    assert transport.requests[0].headers["authorization"] == f"Bearer {KEY.get_secret_value()}"


# --- options -----------------------------------------------------------------------------


def test_no_options_normalize_to_none() -> None:
    provider, _ = provider_for()
    assert provider.validate_options({}) == {}
    assert provider.options.max_seq_len is None


def test_max_seq_len_is_decoded_and_normalized() -> None:
    provider, _ = provider_for()
    normalized = provider.validate_options({"max_seq_len": "4096"})
    assert normalized == {"max_seq_len": 4096}
    assert provider.validate_options(normalized) == normalized
    assert provider.options.max_seq_len == 4096
    assert provider.validate_options({"max_seq_len": MAX_SEQ_LEN}) == {"max_seq_len": 131072}


@pytest.mark.parametrize("value", ["1023", "-1", str(MAX_SEQ_LEN + 1), "4096.5", "true", '"4096"'])
def test_an_invalid_max_seq_len_is_refused(value: str) -> None:
    provider, _ = provider_for()
    with pytest.raises(UsageError, match="invalid baseten provider options: max_seq_len"):
        provider.validate_options({"max_seq_len": value})


def test_a_value_that_is_not_json_is_refused() -> None:
    provider, _ = provider_for()
    with pytest.raises(UsageError, match="must be JSON"):
        provider.validate_options({"max_seq_len": "lots"})


def test_unknown_options_are_refused() -> None:
    provider, _ = provider_for()
    with pytest.raises(UsageError, match=r"unknown baseten provider option\(s\): a, b") as caught:
        provider.validate_options({"b": "1", "a": "2", "max_seq_len": "4096"})
    assert caught.value.hint == "known options: max_seq_len"


def test_the_max_seq_len_option_cuts_the_training_profile() -> None:
    provider, _ = provider_for()
    assert provider.training_profile(QWEN) is QWEN  # no option: unchanged
    provider.validate_options({"max_seq_len": "4096"})
    capped = provider.training_profile(QWEN)
    assert capped == dataclasses.replace(QWEN, max_seq_len=4096)
    assert provider.training_profile(capped) == capped  # a resumed run's profile stays put
    provider.validate_options({"max_seq_len": str(QWEN.max_seq_len)})
    assert provider.training_profile(QWEN) is QWEN


def test_a_max_seq_len_option_over_the_profile_cannot_cut_it() -> None:
    provider, _ = provider_for()
    provider.validate_options({"max_seq_len": "4096"})
    small = dataclasses.replace(QWEN, max_seq_len=2048)
    with pytest.raises(UsageError, match="option of 4096 is over the 2048-token context"):
        provider.training_profile(small)


def test_the_max_seq_len_option_caps_the_backend() -> None:
    loops = FakeLoops()
    provider, _ = provider_for(loops)
    provider.validate_options({"max_seq_len": "2048"})
    backend = provider.backend()
    with (
        pytest.raises(UsageError, match="over the max_seq_len provider option of 2048"),
        backend.open_session(QWEN, LoraConfig(rank=8), seed=1, max_seq_len=2049),
    ):
        pass
    assert loops.calls == []
    with backend.open_session(QWEN, LoraConfig(rank=8), seed=1, max_seq_len=2048):
        pass
