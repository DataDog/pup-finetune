# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Write a GitHub Release's assets: the wheel, one launcher per platform, and checksums.

Usage: python scripts/release_launcher.py --tag v0.1.0 --wheel dist/pup_finetune-0.1.0-py3-none-any.whl
    --constraints dist/constraints.txt --out-dir dist/release

`pup extension install DataDog/pup-finetune` downloads the launcher named for the user's
platform (Pup's `pup-<name>-<os>-<arch>` convention) and installs it as the `finetune`
extension. Every launcher is the same POSIX shell script, rendered from
`release_launcher.sh.in`: it pins the release's wheel by SHA-256 and the dependency versions
`uv.lock` resolved (`--constraints`, from `uv export`). `checksums.txt` lists the SHA-256 of
every asset. The release's version, normalized as in the wheel's name, is the only line printed
on stdout. Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "release_launcher.sh.in"

PLATFORMS: tuple[str, ...] = (
    "darwin-aarch64",
    "darwin-x86_64",
    "linux-aarch64",
    "linux-x86_64",
)

_TAG = re.compile(r"v(\d+)\.(\d+)\.(\d+)(?:-(a|b|rc)\.?(\d+))?")


class ReleaseError(Exception):
    """A tag, wheel or constraints file this script cannot release."""


def version_of(tag: str) -> str:
    """The PEP 440 version a release tag names, as hatch-vcs writes it into the wheel's name:
    `v1.2.3` is `1.2.3` and `v1.2.3-rc.1` is `1.2.3rc1`."""
    match = _TAG.fullmatch(tag)
    if match is None:
        raise ReleaseError(f"tag {tag!r} is not vX.Y.Z or vX.Y.Z-rc.N (also a.N, b.N)")
    major, minor, patch, pre, number = match.groups()
    return f"{major}.{minor}.{patch}{pre or ''}{number or ''}"


def constraint_lines(text: str) -> list[str]:
    """The requirement lines of `uv export` output, without its comments and blank lines."""
    lines = [line.strip() for line in text.splitlines()]
    kept = [line for line in lines if line and not line.startswith("#")]
    if not kept:
        raise ReleaseError("the constraints file has no requirements")
    if any(line.startswith("-") for line in kept):
        raise ReleaseError("the constraints file has options; export it with --no-emit-project")
    return kept


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(*, tag: str, wheel: str, wheel_sha256: str, constraints: list[str]) -> str:
    """The launcher for one release."""
    values = {
        "@TAG@": tag,
        "@VERSION@": version_of(tag),
        "@WHEEL@": wheel,
        "@WHEEL_SHA256@": wheel_sha256,
        "@CONSTRAINTS@": "\n".join(constraints),
    }
    text = TEMPLATE.read_text(encoding="utf-8")
    for placeholder, value in values.items():
        if placeholder not in text:
            raise ReleaseError(f"the template has no {placeholder}")
        text = text.replace(placeholder, value)
    return text


def write_assets(*, tag: str, wheel: Path, constraints: Path, out_dir: Path) -> list[Path]:
    """Write the wheel, the launchers and `checksums.txt` into a new or empty `out_dir`."""
    version = version_of(tag)
    expected = f"pup_finetune-{version}-py3-none-any.whl"
    if wheel.name != expected:
        raise ReleaseError(f"tag {tag} needs the wheel {expected}, not {wheel.name}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ReleaseError(f"{out_dir} is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    launcher = render(
        tag=tag,
        wheel=wheel.name,
        wheel_sha256=sha256_of(wheel),
        constraints=constraint_lines(constraints.read_text(encoding="utf-8")),
    )
    assets = [Path(shutil.copyfile(wheel, out_dir / wheel.name))]
    for platform in PLATFORMS:
        path = out_dir / f"pup-finetune-{platform}"
        path.write_text(launcher, encoding="utf-8")
        path.chmod(0o755)
        assets.append(path)
    checksums = out_dir / "checksums.txt"
    checksums.write_text(
        "".join(f"{sha256_of(path)}  {path.name}\n" for path in assets), encoding="utf-8"
    )
    return [*assets, checksums]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="the release tag, such as v0.1.0")
    parser.add_argument("--wheel", required=True, type=Path, help="the wheel built from the tag")
    parser.add_argument("--constraints", required=True, type=Path, help="`uv export` output")
    parser.add_argument("--out-dir", required=True, type=Path, help="a new or empty directory")
    args = parser.parse_args(argv)
    try:
        write_assets(
            tag=args.tag, wheel=args.wheel, constraints=args.constraints, out_dir=args.out_dir
        )
    except (ReleaseError, OSError) as exc:
        print(f"release_launcher: {exc}", file=sys.stderr)
        return 1
    print(version_of(args.tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
