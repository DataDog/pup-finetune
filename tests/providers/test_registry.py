# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from pup_finetune import providers
from pup_finetune.errors import FinetuneError, UsageError
from pup_finetune.providers import get_provider, known_providers, register_provider
from pup_finetune.providers.base import Provider
from pup_finetune.runtime import Runtime

DEV = Runtime(dev_mode=True)
USER = Runtime()

# A minimal provider: enough to satisfy the Provider protocol; no method is called here.
_PROVIDER_SOURCE = textwrap.dedent(
    """
    class StubProvider:
        def __init__(self, runtime):
            self.runtime = runtime

        @property
        def name(self):
            return "stub"

        def check(self): return []
        def models(self): return []
        def resolve_model(self, alias_or_id): raise NotImplementedError
        def preflight(self, profile): raise NotImplementedError
        def renderer(self, profile): raise NotImplementedError
        def backend(self): raise NotImplementedError
        def validate_options(self, options): return dict(options)
        def deploy(self, model, options): raise NotImplementedError
        def undeploy(self, endpoint): raise NotImplementedError

    NOT_A_PROVIDER = lambda runtime: object()
    NOT_CALLABLE = 42
    """
)


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers, "_REGISTRY", {})


@pytest.fixture
def provider_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[[str], str]]:
    """Write a provider module under a fresh name on sys.path; returns its module name."""
    monkeypatch.syspath_prepend(str(tmp_path))
    created: list[str] = []

    def make(source: str = _PROVIDER_SOURCE) -> str:
        name = f"pft_test_provider_{len(created)}_{tmp_path.name}".replace("-", "_")
        (tmp_path / f"{name}.py").write_text(source)
        created.append(name)
        return name

    yield make
    for name in created:
        sys.modules.pop(name, None)


def test_registering_and_listing_imports_nothing(provider_module: Callable[[str], str]) -> None:
    module = provider_module(_PROVIDER_SOURCE)
    register_provider("stub", f"{module}:StubProvider")
    assert known_providers(USER) == ["stub"]
    assert module not in sys.modules

    provider = get_provider("stub", USER)

    assert module in sys.modules
    assert isinstance(provider, Provider)
    assert provider.name == "stub"
    assert provider.runtime is USER  # type: ignore[attr-defined]


def test_get_provider_builds_a_new_provider_each_time(
    provider_module: Callable[[str], str],
) -> None:
    register_provider("stub", f"{provider_module(_PROVIDER_SOURCE)}:StubProvider")
    assert get_provider("stub", USER) is not get_provider("stub", USER)


def test_unknown_provider_lists_the_known_names() -> None:
    register_provider("zeta", "example.zeta:Provider")
    register_provider("alpha", "example.alpha:Provider")
    with pytest.raises(UsageError) as info:
        get_provider("nope", USER)
    assert info.value.message == "unknown provider 'nope'"
    assert info.value.hint == "known providers: alpha, zeta"
    assert info.value.exit_code == 2


def test_unknown_provider_with_none_registered() -> None:
    with pytest.raises(UsageError) as info:
        get_provider("nope", USER)
    assert info.value.hint == "no providers are available"


def test_dev_only_providers_are_hidden_outside_dev_mode() -> None:
    register_provider("real", "example.real:Provider")
    register_provider("toy", "example.toy:Provider", dev_only=True)
    assert known_providers(USER) == ["real"]
    assert known_providers(DEV) == ["real", "toy"]


def test_a_dev_only_provider_is_unknown_outside_dev_mode() -> None:
    register_provider("real", "example.real:Provider")
    register_provider("toy", "example.toy:Provider", dev_only=True)
    with pytest.raises(UsageError) as info:
        get_provider("toy", USER)
    hint = info.value.hint or ""
    assert hint.startswith("known providers: real;")
    assert "PUP_FINETUNE_DEV=1" in hint


def test_a_dev_only_provider_loads_in_dev_mode(provider_module: Callable[[str], str]) -> None:
    register_provider("toy", f"{provider_module(_PROVIDER_SOURCE)}:StubProvider", dev_only=True)
    assert isinstance(get_provider("toy", DEV), Provider)


def test_a_missing_provider_module_is_a_clear_error() -> None:
    register_provider("ghost", "pup_finetune.providers.ghost_never_exists:GhostProvider")
    with pytest.raises(UsageError) as info:
        get_provider("ghost", USER)
    assert "pup_finetune.providers.ghost_never_exists is missing" in info.value.message


def test_a_missing_parent_package_is_a_broken_install() -> None:
    register_provider("orphan", "pft_absent_pkg_xyz.provider:Provider")
    with pytest.raises(UsageError) as info:
        get_provider("orphan", USER)
    assert "pft_absent_pkg_xyz.provider is missing" in info.value.message
    hint = info.value.hint or ""
    assert "reinstall" in hint
    assert "pup-finetune[orphan]" not in hint


def test_a_missing_first_party_submodule_is_a_broken_install(
    provider_module: Callable[[str], str],
) -> None:
    module = provider_module("import pup_finetune.providers.pft_absent_client\n")
    register_provider("partial", f"{module}:Provider")
    with pytest.raises(UsageError) as info:
        get_provider("partial", USER)
    hint = info.value.hint or ""
    assert "reinstall" in hint
    assert "pup-finetune[partial]" not in hint


def test_a_missing_dependency_names_the_package_and_extra(
    provider_module: Callable[[str], str],
) -> None:
    module = provider_module("import pft_sdk_that_is_not_installed\n")
    register_provider("needy", f"{module}:Provider")
    with pytest.raises(UsageError) as info:
        get_provider("needy", USER)
    assert "'pft_sdk_that_is_not_installed'" in info.value.message
    assert "pup-finetune[needy]" in (info.value.hint or "")


def test_a_missing_attribute_is_an_error(provider_module: Callable[[str], str]) -> None:
    register_provider("stub", f"{provider_module(_PROVIDER_SOURCE)}:Missing")
    with pytest.raises(FinetuneError, match="no attribute 'Missing'"):
        get_provider("stub", USER)


@pytest.mark.parametrize(
    ("attr", "match"), [("NOT_CALLABLE", "not callable"), ("NOT_A_PROVIDER", "did not build")]
)
def test_a_target_that_does_not_build_a_provider_is_an_error(
    provider_module: Callable[[str], str], attr: str, match: str
) -> None:
    register_provider("stub", f"{provider_module(_PROVIDER_SOURCE)}:{attr}")
    with pytest.raises(FinetuneError, match=match):
        get_provider("stub", USER)


def test_registering_the_same_entry_twice_is_a_no_op() -> None:
    register_provider("real", "example.real:Provider")
    register_provider("real", "example.real:Provider")
    assert known_providers(USER) == ["real"]


@pytest.mark.parametrize(
    ("target", "dev_only"), [("example.other:Provider", False), ("example.real:Provider", True)]
)
def test_registering_a_different_entry_under_a_taken_name_fails(
    target: str, dev_only: bool
) -> None:
    register_provider("real", "example.real:Provider")
    with pytest.raises(ValueError, match="already registered"):
        register_provider("real", target, dev_only=dev_only)


@pytest.mark.parametrize("target", ["example.real", "example.real:", ":Provider", ""])
def test_register_rejects_malformed_targets(target: str) -> None:
    with pytest.raises(ValueError, match="module:attr"):
        register_provider("real", target)


def test_register_rejects_an_empty_name() -> None:
    with pytest.raises(ValueError, match="needs a name"):
        register_provider("", "example.real:Provider")


def test_importing_the_cli_imports_no_provider_module() -> None:
    # A fresh interpreter, so modules other tests imported do not count.
    code = textwrap.dedent(
        """
        import json, sys
        import pup_finetune.cli
        from pup_finetune import providers
        from pup_finetune.runtime import Runtime

        # A target that would fail to import: listing must not try.
        providers.register_provider(
            "unimportable", "pft_unimportable_xyz.provider:Provider", dev_only=True
        )
        assert "unimportable" in providers.known_providers(Runtime(dev_mode=True))
        targets = sorted(
            entry.target.partition(":")[0] for entry in providers._REGISTRY.values()
        )
        print(json.dumps({
            "provider_modules": sorted(
                name for name in sys.modules
                if name.startswith("pup_finetune.providers.")
            ),
            "targets": targets,
            "loaded_targets": [name for name in targets if name in sys.modules],
            "sdks": sorted(
                name for name in ("fireworks", "transformers", "truss")
                if name in sys.modules
            ),
        }))
        """
    )
    completed = subprocess.run(  # noqa: S603 - our own interpreter and code
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    report = json.loads(completed.stdout)
    # Only the protocols module, never a provider implementation.
    assert set(report["provider_modules"]) <= {"pup_finetune.providers.base"}
    assert report["targets"]
    assert report["loaded_targets"] == []
    assert report["sdks"] == []
