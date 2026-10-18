import numpy as np
import pandas as pd
import pytest

from aet.pipeline import flat_predictions
from aet.schema import SchemaError, check_training_window, clean_hourly, clean_quarter_hourly, validate_predictions
from tests.synthetic import hourly_from, synthetic_long


@pytest.fixture
def hourly():
    return clean_hourly(hourly_from(synthetic_long("2026-02-01", "2026-02-01")))


def test_flat_predictions_are_compliant(hourly):
    report = validate_predictions(flat_predictions(hourly), hourly)
    assert report["rows"] == len(hourly) * 4 and report["max_abs_mean_error"] == 0.0


@pytest.mark.parametrize("corruption", ["duplicate", "drop", "shift", "nan"])
def test_validator_rejects_non_compliant_outputs(hourly, corruption):
    pred = flat_predictions(hourly)
    if corruption == "duplicate":
        pred = pd.concat([pred, pred.iloc[[0]]])
    elif corruption == "drop":
        pred = pred.iloc[1:]
    elif corruption == "shift":
        pred.loc[0, "predicted_price_eur_mwh"] += 0.001
    else:
        pred.loc[0, "predicted_price_eur_mwh"] = np.nan
    with pytest.raises(SchemaError):
        validate_predictions(pred, hourly)


def test_history_cleaning_is_lenient_but_training_is_strict():
    df = synthetic_long("2026-02-01", "2026-02-01")
    df = pd.concat([df, df.iloc[[0]]])
    with pytest.raises(SchemaError):
        clean_quarter_hourly(df, strict=True)
    assert len(clean_quarter_hourly(df, strict=False)) == 4 * 96


def test_october_2026_cannot_enter_training():
    df = clean_quarter_hourly(synthetic_long("2026-09-30", "2026-10-01"))
    with pytest.raises(SchemaError):
        check_training_window(df)
