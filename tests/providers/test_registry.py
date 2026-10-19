# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from pup_finetune import providers
from pup_finetune.errors import FinetuneError, UsageError, install_hint
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
        def training_profile(self, profile): return profile
        def training_billing(self, usage): return "nothing"
        def sampling_billing(self, usage): return "nothing"
        def backend(self): raise NotImplementedError
        def validate_options(self, options): return dict(options)
        def deploy(self, model, options, *, on_creating=None, on_created=None):
            raise NotImplementedError
        def deployment_name(self, deployment_id): return None
        def smoke_test(self, endpoint): raise NotImplementedError
        def undeploy(self, endpoint): raise NotImplementedError
        def owns_deployment(self, request): raise NotImplementedError

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


_LAZY_IMPORT_SOURCE = textwrap.dedent(
    """
    def build_sdk(runtime):
        import pft_lazy_sdk_that_is_not_installed

    def build_client(runtime):
        import pup_finetune.providers.pft_absent_lazy_client
    """
)


def test_a_dependency_missing_while_building_names_the_package_and_extra(
    provider_module: Callable[[str], str],
) -> None:
    # Providers may import their SDK lazily, in the factory, rather than with the module.
    register_provider("lazy", f"{provider_module(_LAZY_IMPORT_SOURCE)}:build_sdk")
    with pytest.raises(UsageError) as info:
        get_provider("lazy", USER)
    assert "'pft_lazy_sdk_that_is_not_installed'" in info.value.message
    assert "pup-finetune[lazy]" in (info.value.hint or "")


def test_a_first_party_module_missing_while_building_is_a_broken_install(
    provider_module: Callable[[str], str],
) -> None:
    register_provider("lazy", f"{provider_module(_LAZY_IMPORT_SOURCE)}:build_client")
    with pytest.raises(UsageError) as info:
        get_provider("lazy", USER)
    assert "pup_finetune.providers.pft_absent_lazy_client is missing" in info.value.message
    hint = info.value.hint or ""
    assert "reinstall" in hint
    assert "pup-finetune[lazy]" not in hint


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


def test_any_name_of_a_model_names_the_providers_that_offer_it() -> None:
    register_provider(
        "one", "pkg.one:Provider", models={"shared": ("shared", "org/Shared"), "only": ("only",)}
    )
    register_provider("two", "pkg.two:Provider", models={"two-s": ("two-s", "org/Shared")})
    register_provider("dev", "pkg.dev:Provider", dev_only=True, models={"dev-m": ("dev-m",)})
    assert providers.providers_for_model("only", USER) == ["one"]
    assert providers.providers_for_model("shared", USER) == ["one"]
    assert providers.providers_for_model("org/Shared", USER) == ["one", "two"]
    # A provider id or tokenizer name matches in any case; an alias only exactly.
    assert providers.providers_for_model("ORG/shared", USER) == ["one", "two"]
    assert providers.providers_for_model("Only", USER) == []
    # A development-only provider's models are unknown outside development mode.
    assert providers.providers_for_model("dev-m", USER) == []
    assert providers.providers_for_model("dev-m", DEV) == ["dev"]
    assert providers.providers_for_model("nope", DEV) == []
    assert providers.known_models(USER) == {
        "one": {"shared": ("shared", "org/Shared"), "only": ("only",)},
        "two": {"two-s": ("two-s", "org/Shared")},
    }


@pytest.mark.parametrize(
    ("models", "match"),
    [
        ({"a": ("b", "a")}, "must start with it"),
        ({"a": ()}, "must start with it"),
        ({"a": ("a", "org/A"), "b": ("b", "org/A")}, "'org/A' names two of its models"),
        ({"a": ("a", "org/A"), "b": ("b", "org/a")}, "'org/a' names two of its models"),
        ({"a": ("a", "org/A"), "org/a": ("org/a",)}, "'org/a' names two of its models"),
    ],
)
def test_a_model_table_must_name_each_model_once_alias_first(
    models: dict[str, tuple[str, ...]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        register_provider("bad", "pkg.bad:Provider", models=models)


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


def test_importing_the_cli_and_inferring_a_provider_import_no_provider_module() -> None:
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
        # What `train` infers --provider with: the registry's aliases, nothing imported.
        inferred = providers.providers_for_model("qwen3p8-27b", Runtime())
        targets = sorted(
            entry.target.partition(":")[0] for entry in providers._REGISTRY.values()
        )
        print(json.dumps({
            "inferred": inferred,
            "provider_modules": sorted(
                name for name in sys.modules
                if name.startswith("pup_finetune.providers.")
            ),
            "targets": targets,
            "loaded_targets": [name for name in targets if name in sys.modules],
            "sdks": sorted(
                name for name in ("baseten", "fireworks", "transformers", "truss")
                if name in sys.modules
            ),
        }))
        """
    )
    completed = subprocess.run(  # noqa: S603 - our own interpreter and code
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    report = json.loads(completed.stdout)
    assert report["inferred"] == ["baseten", "fireworks"]
    # Only the protocols module and the light packages that register provider API keys,
    # never a provider implementation.
    assert set(report["provider_modules"]) <= {
        "pup_finetune.providers.base",
        "pup_finetune.providers.baseten",
        "pup_finetune.providers.fireworks",
    }
    assert report["targets"]
    assert report["loaded_targets"] == []
    assert report["sdks"] == []


# --- what a provider needs -----------------------------------------------------------------

SDK = providers.Package(
    key="sdk",
    label="Toy SDK",
    modules=("pft_toy_sdk_xyz",),
    extra="toy",
    steps=("train", "deploy"),
    needed_for="training on toy",
)


def test_a_provider_s_needs_are_read_without_importing_it() -> None:
    register_provider(
        "toy",
        "pft_unimportable_xyz.provider:Provider",
        label="Toy",
        credential="toy",
        packages=(SDK,),
        refuses=(providers.RefusedFlag(field="lora_alpha", flag="--lora-alpha", reason="no"),),
        deploys=False,
    )
    info = providers.provider_info("toy", USER)
    assert (info.label, info.credential, info.deploys, info.offline) == ("Toy", "toy", False, False)
    assert info.packages_for("train") == (SDK,)
    assert info.packages_for("evaluate") == ()
    assert "pft_unimportable_xyz.provider" not in sys.modules


def test_the_label_defaults_to_the_name() -> None:
    register_provider("toy", "example.toy:Provider")
    assert providers.provider_info("toy", USER).label == "Toy"


def test_providers_are_listed_in_registration_order() -> None:
    register_provider("zeta", "example.zeta:Provider")
    register_provider("alpha", "example.alpha:Provider")
    register_provider("dev", "example.dev:Provider", dev_only=True)
    assert providers.registered_providers() == ["zeta", "alpha"]
    assert providers.registered_providers(include_dev=True) == ["zeta", "alpha", "dev"]
    assert [info.name for info in providers.provider_infos(USER)] == ["zeta", "alpha"]
    assert [info.name for info in providers.provider_infos(DEV)] == ["zeta", "alpha", "dev"]
    with pytest.raises(UsageError, match="unknown provider 'dev'"):
        providers.provider_info("dev", USER)


def test_a_missing_package_fails_with_the_install_hint() -> None:
    register_provider("toy", "example.toy:Provider", label="Toy", packages=(SDK,))
    assert providers.missing_packages("toy", USER, "train") == [SDK]
    assert providers.missing_packages("toy", USER, "evaluate") == []
    with pytest.raises(UsageError) as info:
        providers.require_packages("toy", USER, "deploy")
    assert info.value.message == "deploying on Toy needs the Toy SDK, which is not installed"
    assert info.value.hint == install_hint("toy")
    assert info.value.exit_code == 2
    providers.require_packages("toy", USER, "evaluate")  # needs nothing


def test_an_installed_package_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    register_provider("toy", "example.toy:Provider", packages=(SDK,))
    monkeypatch.setattr(providers, "installed", lambda module: module == "pft_toy_sdk_xyz")
    providers.require_packages("toy", USER, "train")


def test_installed_finds_modules_without_importing_them() -> None:
    assert providers.installed("json")
    assert not providers.installed("pft_no_such_module_xyz")
    assert not providers.installed("pft_no_such_module_xyz.sub")
    assert "pft_no_such_module_xyz" not in sys.modules


@pytest.fixture
def packages_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A directory on ``sys.path`` for synthetic packages; their modules are forgotten
    afterwards."""
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def _package(root: Path, name: str, init: str) -> None:
    (root / name).mkdir()
    (root / name / "__init__.py").write_text(init)
    importlib.invalidate_caches()


def test_module_state_tells_missing_from_broken(packages_dir: Path) -> None:
    _package(packages_dir, "pft_ok_pkg_xyz", "")
    (packages_dir / "pft_ok_pkg_xyz" / "sub.py").write_text("")
    _package(packages_dir, "pft_raising_pkg_xyz", "raise RuntimeError('bad build')\n")
    _package(packages_dir, "pft_needy_pkg_xyz", "import pft_absent_dependency_xyz\n")
    try:
        assert providers.module_state("pft_ok_pkg_xyz.sub") == ("installed", "")
        # Missing: the module itself, or a parent package.
        assert providers.module_state("pft_ok_pkg_xyz.nope") == ("missing", "")
        assert providers.module_state("pft_no_such_module_xyz.sub") == ("missing", "")
        # Broken: a parent that raises, or that needs a module that is missing.
        assert providers.module_state("pft_raising_pkg_xyz.sub") == (
            "broken",
            "RuntimeError: bad build",
        )
        assert providers.module_state("pft_needy_pkg_xyz.sub") == (
            "broken",
            "ModuleNotFoundError: No module named 'pft_absent_dependency_xyz'",
        )
        assert not providers.installed("pft_raising_pkg_xyz.sub")
    finally:
        for name in [name for name in sys.modules if name.startswith("pft_")]:
            del sys.modules[name]


def test_a_broken_package_says_it_cannot_be_loaded(packages_dir: Path) -> None:
    _package(packages_dir, "pft_raising_sdk_xyz", "raise RuntimeError('bad build')\n")
    sdk = providers.Package(
        key="sdk",
        label="Toy SDK",
        modules=("pft_raising_sdk_xyz.client",),
        extra="toy",
        steps=("train",),
        needed_for="training on toy",
    )
    register_provider("toy", "example.toy:Provider", label="Toy", packages=(sdk,))
    try:
        with pytest.raises(UsageError) as info:
            providers.require_packages("toy", USER, "train")
    finally:
        for name in [name for name in sys.modules if name.startswith("pft_")]:
            del sys.modules[name]
    assert info.value.message == (
        "training on Toy needs the Toy SDK, which is installed but cannot be loaded "
        "(RuntimeError: bad build)"
    )
    assert info.value.hint == install_hint("toy")


def test_a_refused_flag_fails_and_an_unset_one_passes() -> None:
    register_provider(
        "toy",
        "example.toy:Provider",
        refuses=(
            providers.RefusedFlag(
                field="lora_alpha", flag="--lora-alpha", reason="Toy sets the LoRA alpha itself"
            ),
        ),
    )
    providers.require_supported_flags("toy", USER, {"lora_alpha": None, "lora_rank": 8})
    with pytest.raises(UsageError) as info:
        providers.require_supported_flags("toy", USER, {"lora_alpha": 16})
    assert info.value.message == (
        "--lora-alpha is not supported on toy: Toy sets the LoRA alpha itself"
    )
    assert info.value.hint == "drop --lora-alpha"


def test_a_provider_that_does_not_deploy_is_refused() -> None:
    register_provider("toy", "example.toy:Provider", label="Toy", deploys=False)
    register_provider("real", "example.real:Provider")
    providers.require_deploys("real", USER)
    with pytest.raises(UsageError, match="deploying on Toy is not available"):
        providers.require_deploys("toy", USER)
