# Releasing pup-finetune

A release is a GitHub Release built from a version tag on `main`. Users install it with
`pup extension install DataDog/pup-finetune` and move to a newer one with
`pup extension upgrade finetune`. The release workflow, `.github/workflows/release.yml`, builds
and publishes it, and only a maintainer or an admin can start it, because the org-wide tag
ruleset lets only them push tags.

## Cut a release

1. Check that `main` is green and holds every change the release needs.
2. Choose the version. Use `vX.Y.Z` for a release and `vX.Y.Z-rc.N` for a pre-release, which
   GitHub marks as one and `pup extension install` skips unless you pass `--tag`.
3. Tag `main` and push the tag:

   ```sh
   git fetch origin
   git tag -a v0.1.0 origin/main -m "pup-finetune v0.1.0"
   git push origin v0.1.0
   ```

4. Follow the Release workflow in the repository's Actions tab. When it succeeds, the release
   is public, with notes GitHub generated from the merged pull requests. Edit the notes on the
   release page if they need context.
5. Install the release under another name, so it doesn't replace your development install,
   and check it:

   ```sh
   pup extension install DataDog/pup-finetune --tag v0.1.0 --name finetune-check
   pup finetune-check doctor
   pup extension remove finetune-check
   ```

## What the workflow does

1. It refuses a tag that is not on `main`, or that is not `vX.Y.Z` or `vX.Y.Z-rc.N`.
2. It builds the wheel, which takes its version from the tag, and exports the dependency
   versions `uv.lock` pins.
3. `scripts/release_launcher.py` writes the release assets: the wheel, one launcher per platform
   (`pup-finetune-darwin-aarch64`, `pup-finetune-darwin-x86_64`, `pup-finetune-linux-aarch64`
   and `pup-finetune-linux-x86_64`, which are the same script) and `checksums.txt`.
4. It runs a launcher against the built wheel and checks the version it prints.
5. It records build provenance for every asset, signed with Sigstore.
6. It creates the GitHub Release with the assets.

## How the launcher works

Pup installs the launcher named for the user's platform as the `finetune` extension. The
launcher is a POSIX shell script rendered from `scripts/release_launcher.sh.in`. On its first
run, it downloads the release's wheel into `~/.cache/pup-finetune/<version>/` (or
`$XDG_CACHE_HOME/pup-finetune/<version>/`). On every run, it checks the wheel against the SHA-256
it carries, and it deletes a wheel that does not match. It then runs the wheel with `uv tool run`,
with the Anthropic, Baseten and Fireworks extras, on the Python version in `.python-version`, and
with the dependency versions the release was built with as constraints. uv downloads that Python
when the machine does not have it. Without the pin, uv would pick the newest Python on the
machine, for which a pinned dependency may have no wheel.

The launcher needs `uv` and `curl`, and runs on macOS and Linux. `train --remote` stages
pup-finetune's source and `uv.lock`, so it needs a checkout install, as `CONTRIBUTING.md`
describes.

## Verify a release

Download the assets from the release page, then check their provenance and checksums:

```sh
gh attestation verify pup-finetune-darwin-aarch64 --repo DataDog/pup-finetune
shasum -a 256 -c checksums.txt
```

## Fix a bad release

Treat a published release as permanent, since anyone may have installed it already. Ship the fix
as a new patch version. To stop new installs of a broken release, a maintainer marks it as a
pre-release on its release page, so that `pup extension install` picks the previous release.
