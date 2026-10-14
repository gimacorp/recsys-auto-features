import json
from copy import deepcopy
from pathlib import Path

import pytest

from recsys_auto.config import validate_config


def config():
    return json.loads(Path("configs/smoke.json").read_text())


@pytest.mark.parametrize(
    "field,value",
    [
        ("seeds", [17, 17]),
        ("manual_days", [7, 30]),
        ("candidate_days", [7]),
        ("search_rounds", 0),
        ("min_improvement", float("nan")),
    ],
)
def test_invalid_configuration_fails_before_training(field, value):
    candidate = deepcopy(config())
    candidate[field] = value
    with pytest.raises(ValueError):
        validate_config(candidate)


@pytest.mark.parametrize(
    "filename", ["smoke.json", "full.json", "external.json", "replication.json"]
)
def test_shipped_configurations_are_valid(filename):
    validate_config(json.loads((Path("configs") / filename).read_text()))


def test_paired_configuration_validates_after_overlay():
    base = json.loads(Path("configs/replication.json").read_text())
    paired = json.loads(Path("configs/paired.json").read_text())
    validate_config({**base, **paired})
    with pytest.raises(ValueError):
        validate_config({**base, **paired, "feature_budget_ratio": float("nan")})
