"""Target labelling and sample weight generation.

Computes forward 1-hour simple return within each session, handles session boundaries
by setting overnight forward returns to NaN, assigns binary labels, and identifies decisive rows.
"""

from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

from configs.config import CONFIG


def make_labels(df_bars: pd.DataFrame, label_threshold: float = None) -> pd.DataFrame:
    """Compute forward returns, binary labels, and decisive mask.

    Parameters
    ----------
    df_bars : pd.DataFrame
        Session-anchored 1h bars with 'close' column and datetime index.
    label_threshold : float, optional
        Threshold for decisive move filter. Defaults to CONFIG['label_threshold'] (0.0010).

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns: 'fwd', 'y', 'decisive'.
    """
    if label_threshold is None:
        label_threshold = CONFIG["label_threshold"]

    df = df_bars.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.tz_convert(ZoneInfo("America/New_York"))

    # Next close and next timestamp
    next_close = df["close"].shift(-1)
    next_time = pd.Series(df.index).shift(-1)
    next_time.index = df.index

    # Identify whether next bar is in the same trading date
    curr_date = pd.Series(df.index.date, index=df.index)
    next_date = pd.Series(next_time.dt.date, index=df.index)
    same_day = curr_date == next_date

    # Simple return: fwd = close[t+1] / close[t] - 1
    fwd_return = (next_close / df["close"]) - 1.0
    # Any bar where next bar is on a different date (or end of dataset) gets NaN
    fwd_return = fwd_return.where(same_day, np.nan)

    df["fwd"] = fwd_return

    # Binary label: y = 1 if fwd > 0 else 0
    # Report count of fwd == 0
    zero_moves = (df["fwd"] == 0.0).sum()
    print(f"make_labels: Total bars: {len(df)}, Labelled bars: {df['fwd'].notna().sum()}, Unlabelled (session close): {df['fwd'].isna().sum()}")
    print(f"make_labels: Flat returns (fwd == 0.0): {zero_moves}")

    df["y"] = np.where(df["fwd"] > 0, 1, 0)
    # Mask y to NaN where fwd is NaN so unlabelled bars aren't accidentally used
    df.loc[df["fwd"].isna(), "y"] = np.nan

    # Decisive mask: |fwd| > label_threshold
    df["decisive"] = (df["fwd"].abs() > label_threshold) & df["fwd"].notna()

    return df


def compute_sample_weights(
    y_train: pd.Series,
    base_weight: float = 1.0,
    class_balance_weights: bool = True
) -> np.ndarray:
    """Compute sample weights for training.

    sample_weight = base_weight * class_weight,
    where class_weight equalizes total weight between Up (1) and Down (0).
    """
    y = np.asarray(y_train)
    weights = np.full(len(y), base_weight, dtype=float)

    if class_balance_weights and len(y) > 0:
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        if n_pos > 0 and n_neg > 0:
            total = len(y)
            w_pos = total / (2.0 * n_pos)
            w_neg = total / (2.0 * n_neg)
            weights[y == 1] *= w_pos
            weights[y == 0] *= w_neg

    return weights


def run_truncation_test(df_raw_gs, df_raw_spy, df_raw_xlf, n_samples: int = 50, seed: int = 42):
    """Verify that features at bar t computed on history up to t match full-dataset features."""
    from src.features import make_features, FEATURES

    print(f"\n--- Running Truncation Test ({n_samples} random cutoffs, seed {seed}) ---")
    rng = np.random.default_rng(seed)

    # Compute full dataset features
    df_full = make_features(df_raw_gs, df_raw_spy, df_raw_xlf, warmup_rows=0)

    # Eligible bars: starting after 60 bars to allow min periods
    eligible_indices = np.arange(60, len(df_full))
    sampled_indices = rng.choice(eligible_indices, size=min(n_samples, len(eligible_indices)), replace=False)

    max_abs_diff = 0.0
    tested = 0

    for idx in sorted(sampled_indices):
        # Truncate raw input exactly at row idx (inclusive)
        gs_cut = df_raw_gs.iloc[: idx + 1]
        spy_cut = df_raw_spy.iloc[: idx + 1]
        xlf_cut = df_raw_xlf.iloc[: idx + 1]

        df_cut = make_features(gs_cut, spy_cut, xlf_cut, warmup_rows=0)

        # Compare row at idx
        row_full = df_full.iloc[idx][FEATURES]
        row_cut = df_cut.iloc[-1][FEATURES]

        diff = (row_full - row_cut).abs().max()
        if np.isnan(diff):
            # If both are NaN it's fine, but check non-NaNs
            both_nan = (row_full.isna() & row_cut.isna()).all()
            if not both_nan:
                raise AssertionError(f"Truncation test NaN mismatch at index {idx} ({df_full.index[idx]})")
        else:
            if diff > max_abs_diff:
                max_abs_diff = diff
            assert diff < 1e-8, f"Look-ahead detected! Difference {diff} at bar {df_full.index[idx]}"
        tested += 1

    print(f"Truncation test PASSED: {tested} points evaluated. Max absolute feature discrepancy: {max_abs_diff:.2e} (<= 1e-8). Zero look-ahead confirmed.")
    return True
