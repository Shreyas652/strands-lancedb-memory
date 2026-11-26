"""The README metric gate rejects a plausible but unsupported claim."""

import pytest

from scripts.check_reported_numbers import _check, _get


def test_readme_metric_must_match_json_value():
    report = {"summary": {"baseline": {"accuracy": 0.94}}}
    actual = _get(report, "/summary/baseline/accuracy")
    _check("94.0%", actual)
    with pytest.raises(ValueError, match="README says"):
        _check("97.0%", actual)
