"""Unit tests for labelling logic and overnight masking."""

from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pytest

from src.labels import make_labels


def test_make_labels_overnight_nans():
    # Construct 2 days with 7 bars each (09:30 to 15:30)
    day1 = pd.date_range("2024-01-02 09:30", periods=7, freq="1h", tz="America/New_York")
    day2 = pd.date_range("2024-01-03 09:30", periods=7, freq="1h", tz="America/New_York")
    index = day1.append(day2)

    closes = np.linspace(100, 114, 14)
    df = pd.DataFrame({"close": closes}, index=index)

    df_labeled = make_labels(df, label_threshold=0.0010)

    # Bar 6 (last bar of day 1) must have fwd NaN
    assert pd.isna(df_labeled.iloc[6]["fwd"]), "Last bar of day 1 must have NaN fwd return"
    assert pd.isna(df_labeled.iloc[6]["y"]), "Last bar of day 1 must have NaN label"
    assert not df_labeled.iloc[6]["decisive"], "Last bar of day 1 must not be decisive"

    # Bar 13 (last bar of day 2) must have fwd NaN
    assert pd.isna(df_labeled.iloc[13]["fwd"]), "Last bar of day 2 must have NaN fwd return"

    # Bar 0 should have valid forward return
    expected_fwd = (closes[1] / closes[0]) - 1.0
    assert pytest.approx(df_labeled.iloc[0]["fwd"], rel=1e-5) == expected_fwd
    assert df_labeled.iloc[0]["y"] == 1
    assert df_labeled.iloc[0]["decisive"] == True
