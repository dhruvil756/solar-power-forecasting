"""Prediction helpers used by the Streamlit app.

The saved model is an hourly-average lookup table: the average AC output for
each hour of the day (0-23) over the training period. Hours outside 0-23 are
rejected instead of returning a value silently.
"""
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hourly_lookup.joblib"
_lookup = None


def _load():
    """Load the saved lookup table once and reuse it."""
    global _lookup
    if _lookup is None:
        _lookup = joblib.load(MODEL_PATH)
    return _lookup


def predict_power(hour: int) -> float:
    """Return the average AC output for an hour of the day (0-23), from training data."""
    if not 0 <= hour <= 23:
        raise ValueError("hour must be between 0 and 23")
    return float(_load().loc[hour])


def hourly_profile():
    """Return the full 24-hour profile as a pandas Series indexed by hour."""
    return _load().copy()