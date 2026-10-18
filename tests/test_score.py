"""tools.score reproduces the official metric: day & zone MAE -> mean over days -> mean over zones."""
import numpy as np
import pandas as pd
import pytest

from tools.score import score


def _truth(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [["2026-09-29", "2026-09-30"], ["ZONE_0", "ZONE_1", "ZONE_2", "ZONE_3"], range(1, 25), range(1, 5)],
        names=["delivery_date", "market_id", "hour_index", "quarter_index"])
    return idx.to_frame(index=False).assign(price_eur_mwh=rng.normal(80, 20, len(idx)))


def test_score_matches_the_official_metric():
    truth = _truth()
    perfect = truth.rename(columns={"price_eur_mwh": "predicted_price_eur_mwh"})
    _, _, overall = score(perfect, truth)
    assert overall["score"] == pytest.approx(0.0, abs=1e-12) and overall["flat"] > 0
    flat = perfect.assign(predicted_price_eur_mwh=truth.groupby(
        ["delivery_date", "market_id", "hour_index"])["price_eur_mwh"].transform("mean"))
    _, _, overall = score(flat, truth)
    assert overall["score"] == pytest.approx(overall["flat"]) and overall["gain_pct"] == pytest.approx(0.0, abs=1e-9)
