from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cghc_completion_audit", ROOT / "scripts" / "audit_cghc_completion.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def test_requirement_collects_pass_and_failure() -> None:
    records = []
    AUDIT.requirement(records, "pass", "passes", lambda: {"value": 1})

    def fail() -> None:
        raise ValueError("expected")

    AUDIT.requirement(records, "fail", "fails", fail)
    assert records[0]["status"] == "PASS"
    assert records[0]["evidence"] == {"value": 1}
    assert records[1]["status"] == "FAIL"
    assert "expected" in records[1]["error"]


def test_metric_and_baseline_contract_is_fixed() -> None:
    assert len(AUDIT.METRICS) == 6
    assert len(AUDIT.BASELINES) == 6
    assert AUDIT.DATASETS == ("adni", "abcd")
