"""The README metric gate rejects a plausible but unsupported claim."""

import pytest

from scripts.check_reported_numbers import _check, _get, _reference_from_link


def test_readme_metric_must_match_json_value():
    report = {"summary": {"baseline": {"accuracy": 0.94}}}
    actual = _get(report, "/summary/baseline/accuracy")
    _check("94.0%", actual)
    with pytest.raises(ValueError, match="README says"):
        _check("97.0%", actual)


def test_readme_result_link_must_point_to_its_json_source():
    label = "results/frozen_answers.json#/summary/file:3/accuracy"
    link = (
        f"[`{label}`](https://github.com/Shreyas652/strands-lancedb-memory/"
        "blob/main/results/frozen_answers.json)"
    )
    assert _reference_from_link(link) == label
    with pytest.raises(ValueError, match="does not match"):
        _reference_from_link(link.replace("frozen_answers.json)", "wrong.json)"))
