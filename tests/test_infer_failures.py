"""Fallbacks require valid hourly references; malformed inputs must not invent output."""
import pandas as pd
import pytest

from aet.infer import run
from aet.pipeline import ShapePipeline
from aet.schema import SchemaError, validate_predictions


def hourly_file(tmp_path):
    frame = pd.DataFrame({"delivery_date": ["2026-09-30"]*2,
                          "market_id": ["ZONE_0"]*2, "hour_index": [1, 2],
                          "hourly_price_eur_mwh": [-12.125, 100.0025]})
    path = tmp_path/"hourly.csv"
    frame.to_csv(path, index=False)
    return path, frame


def test_missing_model_falls_back_with_valid_hourly_input(tmp_path):
    path, hourly = hourly_file(tmp_path)
    out = tmp_path/"prediction.csv"
    report = run(path, None, tmp_path/"absent_model", out)
    assert report["emergency_flat"] is True
    assert report["candidate"] == "unavailable"
    validate_predictions(pd.read_csv(out), hourly)


def test_unreadable_history_is_inside_fallback_boundary(tmp_path, monkeypatch):
    path, hourly = hourly_file(tmp_path)
    class Dummy:
        class config:
            name = "test_model"
    monkeypatch.setattr(ShapePipeline, "load", lambda _: (Dummy(), object()))
    out = tmp_path/"prediction.csv"
    report = run(path, tmp_path/"absent_history.csv", tmp_path/"model", out)
    assert report["emergency_flat"] is True
    assert report["candidate"] == "test_model"
    validate_predictions(pd.read_csv(out), hourly)


def test_malformed_hourly_input_is_rejected_not_fabricated(tmp_path):
    path, hourly = hourly_file(tmp_path)
    hourly.loc[0, "hourly_price_eur_mwh"] = float("nan")
    hourly.to_csv(path, index=False)
    out = tmp_path/"prediction.csv"
    with pytest.raises(SchemaError):
        run(path, None, tmp_path/"absent_model", out)
    assert not out.exists()
