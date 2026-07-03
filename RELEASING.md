# Releasing CVDT

CVDT is published to [PyPI](https://pypi.org/project/cvdt/) automatically by the
[`release.yml`](.github/workflows/release.yml) GitHub Actions workflow. This
document is the authoritative checklist for cutting a release.

## How it works

- **Every push to `main` and every PR** runs the test jobs only (Rust core via
  `cargo test`, the scikit-learn binding via `pytest`). Nothing is published.
- **Pushing a `vX.Y.Z` tag** runs the full pipeline:
  `test` → `check-version` → `build` + `sdist` → `release` (publish to PyPI).
- Publishing uses **PyPI Trusted Publishing (OIDC)** — there is no API token
  stored anywhere. Authentication is minted at publish time from GitHub's
  identity.
- Because the extension is built with `abi3-py38`, a single wheel per
  (OS, arch) covers every CPython >= 3.8.

## One-time setup (already done once per project)

These only need doing once; they're recorded here so the setup can be
reproduced.

1. **PyPI account** with 2FA enabled.
2. **Pending trusted publisher** on PyPI
   (https://pypi.org/manage/account/publishing/):
   - PyPI Project Name: `cvdt`
   - Owner: `AdventuresInDataScience`
   - Repository name: `CVDT`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. **GitHub environment** named `pypi`
   (repo → Settings → Environments). Optionally add a required reviewer to
   gate the publish step with a manual approval.

## Cutting a release

1. Bump the version to the same number in all three places:
   - `pyproject.toml` → `version = "X.Y.Z"`
   - `Cargo.toml` → `version = "X.Y.Z"`
   - `python/cvdt/__init__.py` → `__version__ = "X.Y.Z"`

   The `check-version` job fails the release if the tag and `pyproject.toml` /
   `Cargo.toml` disagree, so those must match exactly. `__init__.py` is not
   gated by CI but should be kept in step so `cvdt.__version__` is correct.
2. Add a dated `X.Y.Z` section to [`CHANGELOG.md`](CHANGELOG.md) describing the
   release (Added / Changed / Fixed), and call out any behaviour changes.
3. Commit and push to `main`:
   ```sh
   git add pyproject.toml Cargo.toml python/cvdt/__init__.py CHANGELOG.md
   git commit -m "Release vX.Y.Z"
   git push
   ```
4. Tag and push the tag:
   ```sh
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
5. Watch the run at
   https://github.com/AdventuresInDataScience/CVDT/actions. If a required
   reviewer is configured, approve the `release` job when it pauses.
6. Confirm the new version at https://pypi.org/project/cvdt/ and, optionally:
   ```sh
   pip install --upgrade cvdt
   ```

## If something goes wrong

- **`check-version` fails** — the tag doesn't match `pyproject.toml` /
  `Cargo.toml`. Delete the bad tag (`git tag -d vX.Y.Z && git push --delete
  origin vX.Y.Z`), fix the versions, and re-tag.
- **Publish fails with an OIDC/permissions error** — recheck that the PyPI
  trusted publisher fields and the GitHub environment name both say `pypi` and
  reference `release.yml`.
- **PyPI rejects a re-upload** — a version can only be published once. Bump to
  a new version; you cannot overwrite an existing release.
