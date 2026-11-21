"""Enter the isolated protocol-behavior environment from the root project."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PROJECT = REPOSITORY_ROOT / "analyses" / "protocol_behavior"


def build_command(argv: list[str]) -> list[str]:
    """Build the nested uv command without importing the review-pending scPertEval."""
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("uv is required to run the protocol-behavior workflow")
    return [
        uv,
        "run",
        "--project",
        str(PROTOCOL_PROJECT),
        "--frozen",
        "python",
        "-m",
        "analyses.protocol_behavior.cli",
        *argv,
    ]


def protocol_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    """Keep a caller's root uv environment from contaminating the isolated project."""
    environment = dict(os.environ if source is None else source)
    requested = environment.pop("SCPERTEVAL_PROTOCOL_ENVIRONMENT", None)
    environment.pop("UV_PROJECT_ENVIRONMENT", None)
    environment.pop("VIRTUAL_ENV", None)
    if requested:
        environment["UV_PROJECT_ENVIRONMENT"] = requested
    return environment


def main(argv: list[str] | None = None) -> int:
    """Run protocol behavior through its pinned scPertEval environment."""
    completed = subprocess.run(
        build_command(list(sys.argv[1:] if argv is None else argv)),
        cwd=REPOSITORY_ROOT,
        env=protocol_environment(),
        check=False,
    )
    return completed.returncode
