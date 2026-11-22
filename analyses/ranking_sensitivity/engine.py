"""Pin and validate the evaluation engine used by ranking-sensitivity production."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

DEFAULT_RANKING_CONFIG = Path("configs/evaluation/ranking-sensitivity-atlas-v1.json")


def expected_scperteval_revision(
    project_root: Path,
    config: Path = DEFAULT_RANKING_CONFIG,
) -> str:
    """Return the exact scPertEval revision frozen by the ranking manifest."""
    path = config if config.is_absolute() else project_root / config
    manifest = json.loads(path.read_text())
    return str(manifest["evaluation_engine"]["revision"])


def installed_scperteval_revision() -> str | None:
    """Return the resolved Git commit recorded by the installed direct dependency."""
    distribution = importlib.metadata.distribution("scperteval")
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        return None
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id")


def assert_scperteval_revision(project_root: Path) -> str:
    """Refuse to score with a different evaluator than the frozen manifest."""
    expected = expected_scperteval_revision(project_root)
    observed = installed_scperteval_revision()
    if observed != expected:
        raise RuntimeError(
            "installed scPertEval revision does not match the ranking manifest: "
            f"observed={observed!r}, expected={expected!r}"
        )
    return observed
