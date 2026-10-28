# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import io
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import keyring.core
import pytest
from pydantic import SecretStr

from pup_finetune import credentials, redaction
from pup_finetune import doctor as doctor_registry
from pup_finetune.cli._common import handle_exception
from pup_finetune.credentials import CredentialSpec
from pup_finetune.errors import ConfirmRequired, ExternalError, UsageError
from pup_finetune.output import emit_error
from pup_finetune.runtime import Runtime
from tests.fakes.keyring import (
    OS_KEYCHAIN_FAKES,
    FailKeyring,
    LockedKeyring,
    MemoryKeyring,
    PlaintextFileKeyring,
    StickyKeyring,
    UnvettedKeyring,
    chainer_of,
)

SECRET = "zq9Xsynthetic0000keyVALUEw7Pm"  # noqa: S105 - synthetic test value
SPEC = CredentialSpec(provider="acme", env_var="ACME_API_KEY", description="Acme test provider")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(credentials, "_REGISTRY", {})
    monkeypatch.setattr(doctor_registry, "_REGISTRY", {})
    monkeypatch.delenv("ACME_API_KEY", raising=False)
    redaction.clear_secrets()
    yield
    redaction.clear_secrets()


def files_under(root: Path) -> set[Path]:
    return {path for path in root.rglob("*")}


# --- specs and the registry -------------------------------------------------------------


def test_spec_defaults_the_keychain_account_to_the_provider() -> None:
    assert SPEC.keychain_service == "pup-finetune"
    assert SPEC.keychain_account == "acme"
    assert SPEC.config_reader is None
    assert SPEC.validator is None


def test_spec_keeps_an_explicit_keychain_account() -> None:
    spec = CredentialSpec(provider="acme", env_var="ACME_API_KEY", keychain_account="acme-2")
    assert spec.keychain_account == "acme-2"


@pytest.mark.parametrize(("provider", "env_var"), [("", "X_API_KEY"), ("acme", "")])
def test_spec_requires_a_provider_and_env_var(provider: str, env_var: str) -> None:
    with pytest.raises(ValueError, match="needs"):
        CredentialSpec(provider=provider, env_var=env_var)


def test_nothing_is_registered_by_default() -> None:
    assert credentials.all_specs() == []


def test_register_and_look_up() -> None:
    other = CredentialSpec(provider="beta", env_var="BETA_API_KEY")
    credentials.register(other)
    credentials.register(SPEC)
    assert credentials.get_spec("acme") is SPEC
    assert credentials.all_specs() == [SPEC, other]


def test_register_again_is_a_no_op_but_a_different_spec_is_rejected() -> None:
    credentials.register(SPEC)
    credentials.register(
        CredentialSpec(provider="acme", env_var="ACME_API_KEY", description="Acme test provider")
    )
    credentials.register(SPEC)
    with pytest.raises(ValueError, match="already registered"):
        credentials.register(CredentialSpec(provider="acme", env_var="OTHER_API_KEY"))


def test_unknown_provider_lists_the_registered_ones() -> None:
    credentials.register(SPEC)
    with pytest.raises(UsageError, match="unknown provider 'nope'") as info:
        credentials.get_spec("nope")
    assert info.value.hint == "registered providers: acme"


def test_unknown_provider_with_none_registered() -> None:
    with pytest.raises(UsageError) as info:
        credentials.resolve("nope", env={})
    assert info.value.hint == "no providers are registered"


# --- resolution -------------------------------------------------------------------------


def test_env_beats_keychain() -> None:
    credentials.register(SPEC)
    backend = MemoryKeyring()
    backend.set_password("pup-finetune", "acme", "keychain-value-0000")
    found = credentials.resolve("acme", env={"ACME_API_KEY": SECRET}, keyring_backend=backend)
    assert found.source == "env"
    assert found.value.get_secret_value() == SECRET


def test_keychain_is_used_when_env_is_unset_or_blank() -> None:
    credentials.register(SPEC)
    backend = MemoryKeyring()
    backend.set_password("pup-finetune", "acme", SECRET)
    for env in ({}, {"ACME_API_KEY": "  "}):
        found = credentials.resolve("acme", env=env, keyring_backend=backend)
        assert found.source == "keychain"
        assert found.value.get_secret_value() == SECRET


def test_keychain_values_are_stripped() -> None:
    credentials.register(SPEC)
    backend = MemoryKeyring()
    backend.set_password("pup-finetune", "acme", f"{SECRET}\n")
    found = credentials.resolve("acme", env={}, keyring_backend=backend)
    assert found.source == "keychain"
    assert found.value.get_secret_value() == SECRET


def test_a_blank_keychain_entry_counts_as_missing() -> None:
    credentials.register(SPEC)
    backend = MemoryKeyring()
    backend.set_password("pup-finetune", "acme", "  ")
    with pytest.raises(UsageError, match="no acme API key found") as info:
        credentials.resolve("acme", env={}, keyring_backend=backend)
    assert "pup finetune auth login acme" in (info.value.hint or "")


def test_provider_config_is_the_last_fallback() -> None:
    reads: list[bool] = []

    def reader() -> str:
        reads.append(True)
        return f" {SECRET}\n"

    credentials.register(
        CredentialSpec(provider="acme", env_var="ACME_API_KEY", config_reader=reader)
    )
    found = credentials.resolve("acme", env={}, keyring_backend=MemoryKeyring())
    assert found.source == "provider-config"
    assert found.value.get_secret_value() == SECRET
    # The env var wins without reading the provider's config.
    reads.clear()
    assert credentials.resolve("acme", env={"ACME_API_KEY": "x" * 12}).source == "env"
    assert reads == []


def test_missing_key_raises_usage_error_with_a_login_hint() -> None:
    credentials.register(SPEC)
    with pytest.raises(UsageError, match="no acme API key found") as info:
        credentials.resolve("acme", env={}, keyring_backend=MemoryKeyring())
    assert info.value.hint == (
        "run `pup finetune auth login acme` in your own terminal, or set ACME_API_KEY"
    )


def test_missing_keychain_backend_falls_through_to_the_hint() -> None:
    credentials.register(SPEC)
    with pytest.raises(UsageError, match="no acme API key found"):
        credentials.resolve("acme", env={}, keyring_backend=FailKeyring())
    assert credentials.lookup("acme", env={}, keyring_backend=FailKeyring()) is None


def test_default_backend_is_the_active_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials.register(SPEC)
    backend = MemoryKeyring()
    backend.set_password("pup-finetune", "acme", SECRET)
    monkeypatch.setattr(keyring.core, "_keyring_backend", backend)
    assert credentials.resolve("acme", env={}).source == "keychain"


def test_locked_keychain_is_an_external_error() -> None:
    credentials.register(SPEC)
    with pytest.raises(ExternalError, match="could not read the acme key") as info:
        credentials.resolve("acme", env={}, keyring_backend=LockedKeyring())
    assert info.value.source == "keychain"
    assert "ACME_API_KEY" in (info.value.hint or "")


def test_resolve_never_writes_into_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials.register(SPEC)
    backend = MemoryKeyring()
    backend.set_password("pup-finetune", "acme", SECRET)
    before = dict(os.environ)
    credentials.resolve("acme", keyring_backend=backend)
    assert dict(os.environ) == before
    assert SECRET not in os.environ.values()


def test_resolved_value_is_masked_in_repr() -> None:
    credentials.register(SPEC)
    found = credentials.resolve("acme", env={"ACME_API_KEY": SECRET})
    assert SECRET not in repr(found)
    assert SECRET not in str(found)


def test_resolved_secrets_are_registered_for_redaction() -> None:
    credentials.register(SPEC)
    credentials.resolve("acme", env={"ACME_API_KEY": SECRET})
    assert credentials.redact(f"key={SECRET}") == "key=***"


# --- store and delete -------------------------------------------------------------------


def test_store_writes_the_keychain_and_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    credentials.register(SPEC)
    backend = MemoryKeyring()
    before = files_under(tmp_path)
    credentials.store("acme", SecretStr(SECRET), keyring_backend=backend)
    assert backend.passwords == {("pup-finetune", "acme"): SECRET}
    assert files_under(tmp_path) == before
    assert credentials.redact(SECRET) == "***"


def test_store_refuses_without_an_os_keychain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    credentials.register(SPEC)
    before = files_under(tmp_path)
    with pytest.raises(UsageError, match="no OS keychain") as info:
        credentials.store("acme", SecretStr(SECRET), keyring_backend=FailKeyring())
    assert "set ACME_API_KEY" in (info.value.hint or "")
    assert "plaintext" in (info.value.hint or "")
    assert files_under(tmp_path) == before


def test_store_refuses_the_null_and_empty_chainer_backends() -> None:
    from keyring.backends import chainer, null

    credentials.register(SPEC)
    with pytest.raises(UsageError, match="no OS keychain"):
        credentials.store("acme", SecretStr(SECRET), keyring_backend=null.Keyring())

    class EmptyChainer(chainer.ChainerBackend):
        backends: list[object] = []  # noqa: RUF012

    with pytest.raises(UsageError, match="no OS keychain"):
        credentials.store("acme", SecretStr(SECRET), keyring_backend=EmptyChainer())


def test_store_refuses_a_plaintext_file_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    credentials.register(SPEC)
    plaintext = PlaintextFileKeyring()
    before = files_under(tmp_path)
    for backend in (plaintext, chainer_of(plaintext, MemoryKeyring())):
        with pytest.raises(UsageError, match="no OS keychain"):
            credentials.store("acme", SecretStr(SECRET), keyring_backend=backend)
        with pytest.raises(UsageError, match="no OS keychain"):
            credentials.ensure_storable("acme", keyring_backend=backend)
    assert plaintext.writes == 0
    assert files_under(tmp_path) == before


def test_store_through_a_chainer_vets_the_backend_that_receives_the_key() -> None:
    credentials.register(SPEC)
    memory = MemoryKeyring()
    credentials.store("acme", SecretStr(SECRET), keyring_backend=chainer_of(memory))
    assert memory.passwords == {("pup-finetune", "acme"): SECRET}
    credentials.ensure_storable("acme", keyring_backend=chainer_of(memory))
    with pytest.raises(UsageError, match="no OS keychain"):
        credentials.store("acme", SecretStr(SECRET), keyring_backend=chainer_of(FailKeyring()))
    with pytest.raises(UsageError, match="no OS keychain"):
        credentials.ensure_storable("acme", keyring_backend=chainer_of(FailKeyring()))


def test_preflight_and_store_agree_on_a_chain_that_cannot_write() -> None:
    class ReadOnlyKeyring(MemoryKeyring):
        def set_password(self, service: str, username: str, password: str) -> None:
            raise NotImplementedError

    credentials.register(SPEC)
    memory = MemoryKeyring()
    for backend in (chainer_of(ReadOnlyKeyring()), chainer_of(ReadOnlyKeyring(), memory)):
        with pytest.raises(UsageError, match="no OS keychain"):
            credentials.ensure_storable("acme", keyring_backend=backend)
        with pytest.raises(UsageError, match="no OS keychain"):
            credentials.store("acme", SecretStr(SECRET), keyring_backend=backend)
    assert memory.passwords == {}


def test_store_refuses_a_backend_that_is_not_an_allowlisted_os_keychain() -> None:
    credentials.register(SPEC)
    plugin = UnvettedKeyring()
    memory = MemoryKeyring()
    for backend in (plugin, chainer_of(plugin, memory)):
        with pytest.raises(UsageError, match="no OS keychain"):
            credentials.ensure_storable("acme", keyring_backend=backend)
        with pytest.raises(UsageError, match="no OS keychain"):
            credentials.store("acme", SecretStr(SECRET), keyring_backend=backend)
    assert plugin.passwords == {}
    assert memory.passwords == {}


def test_the_allowlist_names_the_real_os_keychain_backends() -> None:
    from keyring.backends import SecretService, Windows, kwallet, libsecret, macOS

    real = {macOS.Keyring, SecretService.Keyring, libsecret.Keyring, Windows.WinVaultKeyring}
    real |= {kwallet.DBusKeyring, kwallet.DBusKeyringKWallet4}
    # The conftest seam adds the OS keychain fakes; everything else must be a real backend.
    fakes = {credentials._backend_name(fake) for fake in OS_KEYCHAIN_FAKES}
    allowlisted = credentials._OS_KEYCHAIN_BACKENDS - fakes
    assert allowlisted == {credentials._backend_name(backend) for backend in real}


def test_ensure_storable_accepts_an_os_keychain_and_refuses_none() -> None:
    credentials.register(SPEC)
    credentials.ensure_storable("acme", keyring_backend=MemoryKeyring())
    with pytest.raises(UsageError, match="no OS keychain") as info:
        credentials.ensure_storable("acme", keyring_backend=FailKeyring())
    assert "set ACME_API_KEY" in (info.value.hint or "")


def test_store_rejects_an_empty_key() -> None:
    credentials.register(SPEC)
    with pytest.raises(UsageError, match="empty"):
        credentials.store("acme", SecretStr(""), keyring_backend=MemoryKeyring())


def test_store_failure_is_an_external_error() -> None:
    credentials.register(SPEC)
    with pytest.raises(ExternalError, match="could not store"):
        credentials.store("acme", SecretStr(SECRET), keyring_backend=LockedKeyring())


def test_delete() -> None:
    credentials.register(SPEC)
    backend = MemoryKeyring()
    credentials.store("acme", SecretStr(SECRET), keyring_backend=backend)
    assert credentials.delete("acme", keyring_backend=backend) is True
    assert backend.passwords == {}
    assert credentials.delete("acme", keyring_backend=backend) is False
    assert credentials.delete("acme", keyring_backend=FailKeyring()) is False
    with pytest.raises(ExternalError, match="could not delete"):
        credentials.delete("acme", keyring_backend=LockedKeyring())


def test_a_failed_delete_that_keeps_the_key_is_an_external_error() -> None:
    credentials.register(SPEC)
    backend = StickyKeyring()
    assert credentials.delete("acme", keyring_backend=backend) is False
    backend.set_password("pup-finetune", "acme", SECRET)
    with pytest.raises(ExternalError, match="could not delete the acme key") as info:
        credentials.delete("acme", keyring_backend=backend)
    assert info.value.source == "keychain"
    assert "allow keychain access" in (info.value.hint or "")
    assert stored_in(backend) == SECRET


def test_delete_through_a_chainer_removes_the_key_from_a_later_backend() -> None:
    credentials.register(SPEC)
    first, second = MemoryKeyring(), MemoryKeyring()
    second.set_password("pup-finetune", "acme", SECRET)
    chain = chainer_of(first, second)
    assert credentials.delete("acme", keyring_backend=chain) is True
    assert second.passwords == {}
    assert credentials.lookup("acme", env={}, keyring_backend=chain) is None
    assert credentials.delete("acme", keyring_backend=chain) is False


def test_delete_through_a_chainer_removes_every_copy() -> None:
    credentials.register(SPEC)
    first, second = MemoryKeyring(), MemoryKeyring()
    for backend in (first, second):
        backend.set_password("pup-finetune", "acme", SECRET)
    assert credentials.delete("acme", keyring_backend=chainer_of(first, second)) is True
    assert first.passwords == second.passwords == {}


def test_delete_through_a_chainer_still_reports_a_denied_delete() -> None:
    credentials.register(SPEC)
    sticky, memory = StickyKeyring(), MemoryKeyring()
    for backend in (sticky, memory):
        backend.set_password("pup-finetune", "acme", SECRET)
    with pytest.raises(ExternalError, match="could not delete the acme key") as info:
        credentials.delete("acme", keyring_backend=chainer_of(sticky, memory))
    assert "allow keychain access" in (info.value.hint or "")
    assert stored_in(sticky) == SECRET
    assert memory.passwords == {}  # the other copy is still removed


def test_delete_through_an_empty_chain_deletes_nothing() -> None:
    credentials.register(SPEC)
    assert credentials.delete("acme", keyring_backend=chainer_of()) is False
    assert credentials.delete("acme", keyring_backend=chainer_of(FailKeyring())) is False


def stored_in(backend: MemoryKeyring) -> str | None:
    return backend.passwords.get(("pup-finetune", "acme"))


# --- doctor -----------------------------------------------------------------------------


def test_registering_adds_no_doctor_check_and_reports_the_key_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials.register(SPEC)
    # doctor builds its key checks from every registered spec (pup_finetune.readiness).
    assert doctor_registry.run_registered_checks() == []
    assert credentials.key_source("acme") is None

    monkeypatch.setenv("ACME_API_KEY", SECRET)
    assert credentials.key_source("acme") == "env"


@pytest.mark.parametrize("debug", [False, True])
def test_doctor_redacts_failing_check_details(
    debug: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    if debug:
        monkeypatch.setenv("PUP_FINETUNE_DEBUG", "1")
    credentials.register_secret(SECRET)

    def expected() -> doctor_registry.CheckOutcome:
        raise ExternalError(f"provider said {SECRET}", source="acme", hint=f"key {SECRET}")

    def unexpected() -> doctor_registry.CheckOutcome:
        raise RuntimeError(f"crashed with {SECRET}")

    doctor_registry.register_check("expected", expected)
    doctor_registry.register_check("unexpected", unexpected)
    results = doctor_registry.run_registered_checks()
    err = capsys.readouterr().err
    details = " ".join(result.detail for result in results)
    assert SECRET not in details + err
    assert "provider said ***; key ***" in details
    assert "RuntimeError: crashed with ***" in details
    if debug:
        assert "Traceback" in err
        assert "RuntimeError: crashed with ***" in err


def test_doctor_redacts_returned_check_details(capsys: pytest.CaptureFixture[str]) -> None:
    credentials.register_secret(SECRET)
    doctor_registry.register_check(
        "passing", lambda: doctor_registry.CheckOutcome(True, f"using {SECRET}")
    )
    doctor_registry.register_check(
        "failing", lambda: doctor_registry.CheckOutcome(False, "Authorization: Bearer tok3n")
    )
    passing, failing = doctor_registry.run_registered_checks()
    assert passing.detail == "using ***"
    assert failing.detail == "Authorization: Bearer ***"


def test_the_key_source_reports_a_locked_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(keyring.core, "_keyring_backend", LockedKeyring())
    credentials.register(SPEC)
    found = credentials.key_source("acme")
    assert isinstance(found, credentials.KeyUnreadable)
    assert found.detail.startswith(
        "could not read the acme key from the OS keychain (KeyringLocked"
    )
    assert found.fix == f"unlock the keychain and retry, or set {SPEC.env_var}"


# --- redaction --------------------------------------------------------------------------


def test_redact_masks_registered_secrets_and_bearer_tokens() -> None:
    credentials.register_secret(SECRET)
    credentials.register_secret(SecretStr("another-synthetic-secret"))
    text = f"a {SECRET} b another-synthetic-secret c Authorization: Bearer abc.DEF-123_x=="
    assert credentials.redact(text) == "a *** b *** c Authorization: Bearer ***"


def test_redact_masks_the_longest_secret_whole() -> None:
    credentials.register_secret("synthetic-key-0000")
    credentials.register_secret("synthetic-key-0000-extended")
    assert credentials.redact("v=synthetic-key-0000-extended") == "v=***"


def test_redact_never_masks_a_value_too_short_to_be_a_key() -> None:
    credentials.register_secret("finetune")
    assert credentials.redact("pup finetune doctor") == "pup finetune doctor"


def test_redact_leaves_ordinary_text_alone() -> None:
    credentials.register_secret("")
    assert credentials.redact("nothing secret here") == "nothing secret here"


def test_logging_filter_redacts_messages_args_and_exceptions() -> None:
    credentials.register_secret(SECRET)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(credentials.RedactingFilter())
    logger = logging.getLogger("pup_finetune.tests.redaction")
    logger.addHandler(handler)
    logger.propagate = False
    try:
        logger.warning("key is %s", SECRET)
        logger.warning("Bearer tok3n-value")
        try:
            raise RuntimeError(f"failed with {SECRET}")
        except RuntimeError:
            logger.exception("boom")
    finally:
        logger.removeHandler(handler)
    output = stream.getvalue()
    assert SECRET not in output
    assert "tok3n-value" not in output
    assert "key is ***" in output
    assert "RuntimeError: failed with ***" in output


@pytest.fixture
def capture_logger() -> Iterator[tuple[logging.Logger, io.StringIO]]:
    """A logger whose handler is attached after redaction is installed."""
    credentials.install_log_redaction()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("pup_finetune.tests.late")
    logger.addHandler(handler)
    logger.propagate = False
    try:
        yield logger, stream
    finally:
        logger.removeHandler(handler)
        logger.propagate = True


def test_install_log_redaction_covers_handlers_added_later(
    capture_logger: tuple[logging.Logger, io.StringIO],
) -> None:
    logger, stream = capture_logger
    credentials.register_secret(SECRET)
    child = logging.getLogger(f"{logger.name}.child")
    child.warning("key=%s", SECRET)
    try:
        raise RuntimeError(f"failed with {SECRET}")
    except RuntimeError:
        child.exception("boom")
    output = stream.getvalue()
    assert SECRET not in output
    assert "key=***" in output
    assert "RuntimeError: failed with ***" in output


def test_install_log_redaction_is_idempotent() -> None:
    credentials.install_log_redaction()
    make_record = logging.Logger.makeRecord
    credentials.install_log_redaction()
    assert logging.Logger.makeRecord is make_record


def test_install_log_redaction_covers_extra_fields_a_formatter_renders() -> None:
    credentials.install_log_redaction()
    credentials.register_secret(SECRET)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s url=%(url)s"))
    logger = logging.getLogger("pup_finetune.tests.late_extra")
    logger.addHandler(handler)
    logger.propagate = False
    try:
        logger.warning("request failed", extra={"url": f"https://x.test/?key={SECRET}"})
    finally:
        logger.removeHandler(handler)
    output = stream.getvalue()
    assert SECRET not in output
    assert "request failed url=https://x.test/?key=***" in output


def test_logging_filter_redacts_extra_fields_a_formatter_renders() -> None:
    credentials.register_secret(SECRET)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(credentials.RedactingFilter())
    handler.setFormatter(logging.Formatter("%(message)s header=%(header)s count=%(count)d"))
    logger = logging.getLogger("pup_finetune.tests.redaction_extra")
    logger.addHandler(handler)
    logger.propagate = False
    try:
        logger.warning("sent", extra={"header": f"Bearer {SECRET}", "count": 3})
    finally:
        logger.removeHandler(handler)
    output = stream.getvalue()
    assert SECRET not in output
    assert "sent header=Bearer *** count=3" in output


@pytest.mark.parametrize("via", ["install", "filter"])
def test_nested_extra_fields_are_redacted(via: str) -> None:
    credentials.register_secret(SECRET)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    if via == "install":
        credentials.install_log_redaction()
    else:
        handler.addFilter(credentials.RedactingFilter())
    handler.setFormatter(logging.Formatter("%(message)s ctx=%(ctx)s"))
    logger = logging.getLogger(f"pup_finetune.tests.nested_extra_{via}")
    logger.addHandler(handler)
    logger.propagate = False
    ctx = {
        "headers": {"Authorization": f"Bearer {SECRET}"},
        "attempts": [("url", f"https://x.test/?key={SECRET}"), 3],
        SECRET: 1.5,
    }
    try:
        logger.warning("request failed", extra={"ctx": ctx})
    finally:
        logger.removeHandler(handler)
    output = stream.getvalue()
    assert SECRET not in output
    assert "'Authorization': 'Bearer ***'" in output
    assert "('url', 'https://x.test/?key=***'), 3]" in output
    assert "'***': 1.5" in output
    # The caller's own object is left untouched.
    assert ctx["headers"] == {"Authorization": f"Bearer {SECRET}"}


def test_install_log_redaction_covers_the_last_resort_handler(
    capsys: pytest.CaptureFixture[str],
) -> None:
    credentials.install_log_redaction()
    credentials.register_secret(SECRET)
    orphan = logging.Logger("pup_finetune.tests.orphan")  # no parent, so no handlers at all
    assert logging.lastResort is not None
    orphan.warning("leaked %s", SECRET)
    err = capsys.readouterr().err
    assert "leaked ***" in err
    assert SECRET not in err


def test_a_malformed_log_call_neither_raises_nor_leaks(
    capture_logger: tuple[logging.Logger, io.StringIO],
) -> None:
    logger, stream = capture_logger
    credentials.register_secret(SECRET)
    logger.warning("%s and %s", SECRET)  # one argument short
    output = stream.getvalue()
    assert "unformattable args" in output
    assert SECRET not in output


def test_the_filter_survives_a_malformed_log_call() -> None:
    credentials.register_secret(SECRET)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(credentials.RedactingFilter())
    record = logging.LogRecord("x", logging.WARNING, __file__, 1, "%s and %s", (SECRET,), None)
    handler.handle(record)
    assert "unformattable args" in stream.getvalue()
    assert SECRET not in stream.getvalue()


def test_emit_error_redacts_message_and_hint(capsys: pytest.CaptureFixture[str]) -> None:
    credentials.register_secret(SECRET)
    emit_error(UsageError(f"bad key {SECRET}", hint=f"try {SECRET}"), runtime=Runtime())
    captured = capsys.readouterr()
    assert SECRET not in captured.err
    assert captured.err == "Error [usage]: bad key ***\nHint: try ***\n"


@pytest.mark.parametrize("debug", [False, True])
def test_unexpected_error_path_redacts(debug: bool, capsys: pytest.CaptureFixture[str]) -> None:
    credentials.register_secret(SECRET)
    try:
        raise RuntimeError(f"provider said {SECRET}")
    except RuntimeError as exc:
        code = handle_exception(exc, Runtime(debug=debug))
    captured = capsys.readouterr()
    assert code == 1
    assert SECRET not in captured.err
    assert "RuntimeError: provider said ***" in captured.err


def test_emit_error_redacts_a_confirmation_preview(capsys: pytest.CaptureFixture[str]) -> None:
    credentials.register_secret(SECRET)
    error = ConfirmRequired(
        f"would send {SECRET}",
        preview={"key": SECRET},
        rerun=f"pup finetune x --token {SECRET} --confirm",
        hint=f"check {SECRET}",
    )
    emit_error(error, runtime=Runtime())
    captured = capsys.readouterr()
    assert SECRET not in captured.err
    assert SECRET not in captured.out
    assert "Would send ***" in captured.err
    assert "Next: pup finetune x --token *** --confirm" in captured.err
    assert "Hint: check ***" in captured.err
    assert '"key": "***"' in captured.out
    assert '"next_command": "pup finetune x --token *** --confirm"' in captured.out
