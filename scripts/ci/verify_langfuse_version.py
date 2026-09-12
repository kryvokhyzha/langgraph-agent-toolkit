"""Verify that the installed Langfuse SDK has the selected major version."""

import sys
from importlib.metadata import version


def verify_langfuse_version(expected_version: str) -> None:
    """Compare the installed SDK major with the selected version."""
    selected_version = version("langfuse")
    expected_major = expected_version.split(".", maxsplit=1)[0]
    selected_major = selected_version.split(".", maxsplit=1)[0]
    if selected_major != expected_major:
        raise AssertionError(f"Langfuse SDK {selected_version} does not match major version {expected_major}.")
    print(f"The extra selected Langfuse SDK {selected_version}")


def main() -> None:
    """Read the selected version and run the check."""
    if len(sys.argv) != 2:
        raise SystemExit("Usage: verify_langfuse_version.py EXPECTED_VERSION")
    verify_langfuse_version(sys.argv[1])


if __name__ == "__main__":
    main()
