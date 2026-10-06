# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Fake ``keyring`` backends, so tests never read or write the real OS keychain.

- :class:`MemoryKeyring` keeps passwords in a dict, like a working OS keychain.
- :class:`FailKeyring` is keyring's own "no backend available" keyring, as on headless Linux.
- :class:`LockedKeyring` raises :class:`keyring.errors.KeyringLocked` on every operation.
- :class:`StickyKeyring` reports a failed delete as :class:`PasswordDeleteError` and keeps
  the entry, as macOS does when the user denies keychain access.
- :class:`PlaintextFileKeyring` poses as a ``keyrings.alt`` file backend and really writes a
  file under ``$HOME``, so a test can prove it is never used.
- :class:`UnvettedKeyring` works like :class:`MemoryKeyring` but stands for a third-party
  plugin that is not on the OS keychain allowlist, so it must never receive a key.
- :func:`chainer_of` builds a chainer backend over the given backends.

:data:`OS_KEYCHAIN_FAKES` are the fakes that stand in for a real OS keychain; the shared
conftest adds them to the credentials allowlist for every test.
"""

from __future__ import annotations

from pathlib import Path

from keyring.backend import KeyringBackend
from keyring.backends import chainer, fail
from keyring.compat import properties
from keyring.errors import KeyringLocked, PasswordDeleteError


class MemoryKeyring(KeyringBackend):
    """An in-memory keychain: ``passwords[(service, account)] = value``."""

    def __init__(self) -> None:
        super().__init__()
        self.passwords: dict[tuple[str, str], str] = {}

    @properties.classproperty
    def priority(cls) -> float:
        return 1

    def get_password(self, service: str, username: str) -> str | None:
        return self.passwords.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.passwords[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        if self.passwords.pop((service, username), None) is None:
            raise PasswordDeleteError("no such password")


class FailKeyring(fail.Keyring):
    """No OS keychain: every operation raises :class:`keyring.errors.NoKeyringError`."""


class LockedKeyring(KeyringBackend):
    """A keychain that is present but locked."""

    @properties.classproperty
    def priority(cls) -> float:
        return 1

    def get_password(self, service: str, username: str) -> str | None:
        raise KeyringLocked("the keychain is locked")

    def set_password(self, service: str, username: str, password: str) -> None:
        raise KeyringLocked("the keychain is locked")

    def delete_password(self, service: str, username: str) -> None:
        raise KeyringLocked("the keychain is locked")


class StickyKeyring(MemoryKeyring):
    """A keychain whose deletes fail with :class:`PasswordDeleteError`, keeping the entry."""

    def delete_password(self, service: str, username: str) -> None:
        raise PasswordDeleteError("the user denied keychain access")


class PlaintextFileKeyring(KeyringBackend):
    """A plaintext-file backend, identified by module the way ``keyrings.alt`` ones are."""

    __module__ = "keyrings.alt.file"

    def __init__(self) -> None:
        super().__init__()
        self.writes = 0

    @properties.classproperty
    def priority(cls) -> float:
        return 1

    @property
    def path(self) -> Path:
        return Path.home() / "keyring_pass.cfg"

    def get_password(self, service: str, username: str) -> str | None:
        return None

    def set_password(self, service: str, username: str, password: str) -> None:
        self.writes += 1
        self.path.write_text(f"{service}/{username}={password}\n")

    def delete_password(self, service: str, username: str) -> None:
        raise PasswordDeleteError("no such password")


class UnvettedKeyring(MemoryKeyring):
    """A working keychain from a plugin that is not an allowlisted OS keychain backend."""


OS_KEYCHAIN_FAKES: tuple[type[KeyringBackend], ...] = (MemoryKeyring, LockedKeyring, StickyKeyring)


def chainer_of(*backends: KeyringBackend) -> chainer.ChainerBackend:
    """A chainer over exactly ``backends``, in that order."""
    chained = list(backends)

    class FixedChainer(chainer.ChainerBackend):
        @properties.classproperty
        def backends(cls) -> list[KeyringBackend]:
            return chained

    return FixedChainer()
