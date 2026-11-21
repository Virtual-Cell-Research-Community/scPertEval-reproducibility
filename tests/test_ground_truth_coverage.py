from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analyses.ranking_sensitivity.ground_truth_coverage import (
    pairwise_coverage_by_fold,
    protocol_coverage,
    run_audit,
    summarize_protocol_pairs,
    validate_availability,
)


def availability() -> pd.DataFrame:
    defined = {
        "a": {"u1": True, "u2": True, "u3": False},
        "b": {"u1": False, "u2": True, "u3": True},
        "c": {"u1": True, "u2": True, "u3": True},
    }
    return pd.DataFrame(
        [
            {
                "task": "task",
                "fold": "fold",
                "protocol_id": protocol,
                "evaluation_unit": unit,
                "ground_truth_defined": is_defined,
                "undefined_reason": "" if is_defined else "no_ground_truth_positives",
            }
            for protocol, units in defined.items()
            for unit, is_defined in units.items()
        ]
    )


def config(tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "tasks": [{"id": "task", "folds": {"fold": 3}}],
                "ground_truth_coverage": {
                    "candidate_minimum_common_fraction": 0.75,
                    "candidate_is_frozen": True,
                },
            }
        )
    )
    return path


def test_pairwise_coverage_uses_the_same_units_for_both_protocols() -> None:
    frame = validate_availability(availability())
    individual = protocol_coverage(frame).set_index("protocol_id")
    assert individual.loc["a", "n_defined"] == 2
    assert individual.loc["c", "defined_fraction"] == 1.0

    by_fold = pairwise_coverage_by_fold(frame, 0.9)
    pair = by_fold.set_index(["protocol_a", "protocol_b"]).loc[("a", "b")]
    assert pair["n_total"] == 3
    assert pair["n_common"] == 1
    assert pair["common_fraction"] == pytest.approx(1 / 3)
    assert not bool(pair["meets_candidate_threshold"])

    summary = summarize_protocol_pairs(by_fold, 0.9)
    pair = summary.set_index(["protocol_a", "protocol_b"]).loc[("a", "c")]
    assert pair["n_common"] == 2
    assert pair["minimum_fold_common_fraction"] == pytest.approx(2 / 3)


def test_availability_rejects_model_outputs_and_incomplete_protocol_universes() -> None:
    with_model = availability().assign(model="m1")
    with pytest.raises(ValueError, match="must not contain model outputs"):
        validate_availability(with_model)

    incomplete = availability().drop(index=0)
    with pytest.raises(ValueError, match="do not share a complete unit universe"):
        validate_availability(incomplete)

    with pytest.raises(ValueError, match="protocol manifest mismatch"):
        validate_availability(
            availability(),
            config={
                "tasks": [{"id": "task", "folds": {"fold": 3}}],
                "ground_truth_coverage": {},
            },
            expected_protocols={"task": {"a", "b", "c", "missing"}},
        )


def test_availability_requires_structured_reasons_only_for_undefined_rows() -> None:
    missing_reason = availability()
    missing_reason.loc[missing_reason["ground_truth_defined"].eq(False), "undefined_reason"] = ""
    with pytest.raises(ValueError, match="require an undefined_reason"):
        validate_availability(missing_reason)

    spurious_reason = availability()
    spurious_reason.loc[0, "undefined_reason"] = "not actually undefined"
    with pytest.raises(ValueError, match="must not carry an undefined_reason"):
        validate_availability(spurious_reason)


def test_run_audit_writes_reproducible_source_tables(tmp_path: Path) -> None:
    availability_path = tmp_path / "availability.csv"
    availability().to_csv(availability_path, index=False)
    output = tmp_path / "output"
    summary = run_audit(availability_path, config(tmp_path), output)

    assert summary["candidate_is_frozen"] is True
    assert summary["tasks"]["task"]["n_protocol_pairs"] == 3
    assert (output / "protocol-coverage-by-fold.csv").is_file()
    assert (output / "pairwise-coverage-by-fold.csv").is_file()
    assert (output / "pairwise-coverage-summary.csv").is_file()
    assert (output / "coverage-summary.json").is_file()
