"""Verify that the release tag matches the package version."""

import os
import tomllib
from pathlib import Path


PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"


def main() -> None:
    """Compare RELEASE_TAG with the project version."""
    with PYPROJECT_PATH.open("rb") as pyproject_file:
        project_version = tomllib.load(pyproject_file)["project"]["version"]

    release_tag = os.environ["RELEASE_TAG"]
    if release_tag.removeprefix("v") != project_version:
        raise SystemExit("The release tag does not match the package version.")


if __name__ == "__main__":
    main()
