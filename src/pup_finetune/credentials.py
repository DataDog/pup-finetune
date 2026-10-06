# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Provider API keys: where they come from, and how they are stored.

Only provider keys live here. Datadog credentials are never touched: every Datadog call goes
through ``pup``, which uses its own login.

A provider registers a :class:`CredentialSpec`. :func:`resolve` then looks for its key in
order: the environment variable, the OS keychain (through ``keyring``: service
``pup-finetune``, account ``<provider>``), and finally the provider CLI's own config, read
only. Keys are returned as :class:`pydantic.SecretStr`, passed explicitly to whatever needs
them, and never copied into ``os.environ``. Every resolved or stored key is registered for
redaction, so it is masked if it ever reaches an error message or a log.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import SecretStr

from pup_finetune.doctor import CheckOutcome, register_check
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.redaction import RedactingFilter, install_log_redaction, redact, register_secret

if TYPE_CHECKING:
    from keyring.backend import KeyringBackend

__all__ = [
    "KEYCHAIN_SERVICE",
    "CredentialSpec",
    "RedactingFilter",
    "ResolvedCredential",
    "Source",
    "all_specs",
    "delete",
    "ensure_storable",
    "get_spec",
    "install_log_redaction",
    "login_hint",
    "lookup",
    "redact",
    "register",
    "register_secret",
    "resolve",
    "store",
]

KEYCHAIN_SERVICE = "pup-finetune"

Source = Literal["env", "keychain", "provider-config"]


@dataclass(frozen=True, slots=True, kw_only=True)
class CredentialSpec:
    """How to find one provider's API key. Fields are keyword-only.

    ``config_reader`` is an optional, read-only fallback to the provider CLI's own stored
    login. ``validator`` checks a key with a cheap provider call before it is stored: it
    returns an account label (or ``None``), raises :class:`UsageError` when the provider
    rejects the key, and :class:`ExternalError` when the provider cannot be reached.
    """

    provider: str
    env_var: str
    description: str = ""
    keychain_service: str = KEYCHAIN_SERVICE
    keychain_account: str = ""
    config_reader: Callable[[], str | None] | None = None
    validator: Callable[[SecretStr], str | None] | None = None

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("a credential spec needs a provider name")
        if not self.env_var:
            raise ValueError(f"credential spec {self.provider!r} needs an env_var")
        if not self.keychain_account:
            object.__setattr__(self, "keychain_account", self.provider)


@dataclass(frozen=True, slots=True)
class ResolvedCredential:
    """A provider key and where it came from. The value's repr is masked."""

    value: SecretStr
    source: Source


_REGISTRY: dict[str, CredentialSpec] = {}


def register(spec: CredentialSpec) -> None:
    """Add a provider's credential spec, and its (optional) ``doctor`` check.

    Registering an equal spec again is a no-op; a different spec under a taken provider name
    raises ``ValueError``. Cheap: nothing is resolved until a command asks.
    """
    existing = _REGISTRY.get(spec.provider)
    if existing is not None:
        if existing != spec:
            raise ValueError(f"credentials for {spec.provider!r} are already registered")
        return
    _REGISTRY[spec.provider] = spec
    register_check(f"credentials:{spec.provider}", _doctor_check(spec), required=False)


def all_specs() -> list[CredentialSpec]:
    """Every registered spec, sorted by provider name."""
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


def get_spec(provider: str) -> CredentialSpec:
    """The spec for ``provider``; a :class:`UsageError` listing the known ones if missing."""
    spec = _REGISTRY.get(provider)
    if spec is None:
        known = ", ".join(sorted(_REGISTRY))
        raise UsageError(
            f"unknown provider {provider!r}",
            hint=f"registered providers: {known}" if known else "no providers are registered",
        )
    return spec


def login_hint(spec: CredentialSpec) -> str:
    """How a user configures ``spec``'s key."""
    return (
        f"run `pup finetune auth login {spec.provider}` in your own terminal, or set {spec.env_var}"
    )


def _backend(keyring_backend: KeyringBackend | None) -> KeyringBackend:
    if keyring_backend is not None:
        return keyring_backend
    import keyring  # deferred: backend discovery is not free, and most commands never need it

    return keyring.get_keyring()


# The keyring backends that keep secrets in an OS keychain, by exact class. Anything else (the
# fail and null backends, keyrings.alt's plaintext files, or any other third-party plugin) never
# receives a key: matching the exact class means a subclass that writes elsewhere is refused too.
_OS_KEYCHAIN_BACKENDS: frozenset[str] = frozenset(
    {
        "keyring.backends.macOS.Keyring",
        "keyring.backends.SecretService.Keyring",
        "keyring.backends.libsecret.Keyring",
        "keyring.backends.kwallet.DBusKeyring",
        "keyring.backends.kwallet.DBusKeyringKWallet4",
        "keyring.backends.Windows.WinVaultKeyring",
    }
)


def _backend_name(backend_type: type[KeyringBackend]) -> str:
    """The name :data:`_OS_KEYCHAIN_BACKENDS` lists ``backend_type`` under."""
    return f"{backend_type.__module__}.{backend_type.__qualname__}"


def _write_targets(backend: KeyringBackend) -> Iterator[KeyringBackend]:
    """The concrete backends a write to ``backend`` could reach, in order.

    A chainer writes to the first of its backends that accepts the write, so it is flattened
    here and the candidate is vetted before it receives the key.
    """
    from keyring.backends import chainer

    if isinstance(backend, chainer.ChainerBackend):
        chained: list[KeyringBackend] = list(backend.backends)
        for inner in chained:
            yield from _write_targets(inner)
    else:
        yield backend


def _is_os_keychain(backend: KeyringBackend) -> bool:
    """Whether ``backend`` is one of the allowlisted OS keychain backends."""
    return _backend_name(type(backend)) in _OS_KEYCHAIN_BACKENDS


def _keychain_target(backend: KeyringBackend) -> KeyringBackend | None:
    """The backend a stored key goes to, or ``None`` when storing must be refused.

    That is the first write target, and only when it is an OS keychain: a key is never passed
    over an unvetted backend to one further down the chain. :func:`ensure_storable` and
    :func:`store` share this, so the preflight and the write always agree.
    """
    first = next(_write_targets(backend), None)
    if first is None or not _is_os_keychain(first):
        return None
    return first


def _no_keychain(spec: CredentialSpec) -> UsageError:
    return UsageError(
        "no OS keychain is available to store the key in",
        hint=f"set {spec.env_var} in the environment instead; keys are never written to "
        "plaintext files",
    )


def ensure_storable(provider: str, *, keyring_backend: KeyringBackend | None = None) -> None:
    """Raise the :class:`UsageError` :func:`store` would, when there is no OS keychain.

    Lets ``auth login`` refuse before it asks for the key or calls the provider.
    """
    spec = get_spec(provider)
    if _keychain_target(_backend(keyring_backend)) is None:
        raise _no_keychain(spec)


def _read_keychain(spec: CredentialSpec, backend: KeyringBackend) -> str | None:
    from keyring.errors import KeyringError, NoKeyringError

    try:
        return backend.get_password(spec.keychain_service, spec.keychain_account)
    except NoKeyringError:
        return None
    except KeyringError as exc:
        raise ExternalError(
            f"could not read the {spec.provider} key from the OS keychain "
            f"({type(exc).__name__}: {redact(str(exc)) or 'no detail'})",
            source="keychain",
            hint=f"unlock the keychain and retry, or set {spec.env_var}",
        ) from exc


def _found(value: str, source: Source) -> ResolvedCredential:
    register_secret(value)
    return ResolvedCredential(SecretStr(value), source)


def lookup(
    provider: str,
    *,
    env: Mapping[str, str] = os.environ,
    keyring_backend: KeyringBackend | None = None,
) -> ResolvedCredential | None:
    """Like :func:`resolve`, but ``None`` when no source has a key."""
    spec = get_spec(provider)
    from_env = env.get(spec.env_var, "").strip()
    if from_env:
        return _found(from_env, "env")
    from_keychain = (_read_keychain(spec, _backend(keyring_backend)) or "").strip()
    if from_keychain:
        return _found(from_keychain, "keychain")
    if spec.config_reader is not None:
        from_config = (spec.config_reader() or "").strip()
        if from_config:
            return _found(from_config, "provider-config")
    return None


def resolve(
    provider: str,
    *,
    env: Mapping[str, str] = os.environ,
    keyring_backend: KeyringBackend | None = None,
) -> ResolvedCredential:
    """``provider``'s key from the environment, then the keychain, then its CLI config.

    Raises :class:`UsageError` with a login hint when none has it. The key is never written
    into ``env``.
    """
    found = lookup(provider, env=env, keyring_backend=keyring_backend)
    if found is None:
        spec = get_spec(provider)
        raise UsageError(f"no {spec.provider} API key found", hint=login_hint(spec))
    return found


def store(
    provider: str,
    value: SecretStr,
    *,
    keyring_backend: KeyringBackend | None = None,
) -> None:
    """Save ``provider``'s key in the OS keychain.

    Refuses with :class:`UsageError` when there is no OS keychain (e.g. headless Linux): the
    key is never written to a plaintext file, so the environment variable is the way there.
    """
    from keyring.errors import KeyringError

    spec = get_spec(provider)
    secret = value.get_secret_value()
    if not secret:
        raise UsageError(f"the {spec.provider} API key is empty")
    register_secret(secret)
    # Write to the vetted backend ourselves rather than through a chainer, so the backend that
    # actually receives the key is the one that was checked.
    target = _keychain_target(_backend(keyring_backend))
    if target is None:
        raise _no_keychain(spec)
    try:
        target.set_password(spec.keychain_service, spec.keychain_account, secret)
    except KeyringError as exc:
        raise ExternalError(
            f"could not store the {spec.provider} key in the OS keychain "
            f"({type(exc).__name__}: {redact(str(exc)) or 'no detail'})",
            source="keychain",
            hint=f"set {spec.env_var} in the environment instead",
        ) from exc


def _delete_from(spec: CredentialSpec, target: KeyringBackend) -> bool:
    """Remove ``spec``'s key from one concrete backend; whether it held one.

    Raises :class:`ExternalError` when the backend holds the key but will not delete it.
    """
    from keyring.errors import KeyringError, NoKeyringError, PasswordDeleteError

    try:
        target.delete_password(spec.keychain_service, spec.keychain_account)
    except (NoKeyringError, NotImplementedError):
        # No keychain here, or a backend that does not support deleting (a chainer skips those).
        return False
    except PasswordDeleteError as exc:
        # Some backends (macOS) also report a denied or failed delete this way, so only
        # call it "nothing stored" when nothing is.
        if not _read_keychain(spec, target):
            return False
        raise ExternalError(
            f"could not delete the {spec.provider} key from the OS keychain "
            f"({type(exc).__name__}: {redact(str(exc)) or 'no detail'})",
            source="keychain",
            hint="allow keychain access or unlock the keychain, then retry",
        ) from exc
    except KeyringError as exc:
        raise ExternalError(
            f"could not delete the {spec.provider} key from the OS keychain "
            f"({type(exc).__name__}: {redact(str(exc)) or 'no detail'})",
            source="keychain",
        ) from exc
    return True


def delete(provider: str, *, keyring_backend: KeyringBackend | None = None) -> bool:
    """Remove ``provider``'s key from the OS keychain; whether there was one to remove.

    A chainer is flattened and the key deleted from every backend that holds it, since
    :func:`lookup` reads through all of them; a chainer's own delete stops at the first
    backend that has no entry. Every backend is tried before the first failure is raised.
    """
    spec = get_spec(provider)
    deleted = False
    failure: ExternalError | None = None
    for target in _write_targets(_backend(keyring_backend)):
        try:
            deleted = _delete_from(spec, target) or deleted
        except ExternalError as exc:
            failure = failure or exc
    if failure is not None:
        raise failure
    return deleted


def _doctor_check(spec: CredentialSpec) -> Callable[[], CheckOutcome]:
    def check() -> CheckOutcome:
        found = lookup(spec.provider)
        if found is None:
            return CheckOutcome(False, f"no {spec.provider} API key; {login_hint(spec)}")
        return CheckOutcome(True, f"{spec.provider} API key found (source: {found.source})")

    return check
