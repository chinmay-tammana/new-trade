"""Labels, multi-horizon targets, sample weights, and truncation test (V2 Upgrade).

Implements:
1. Directional heads: H1 (1h), H3 (3h same day), HD (rest of day) with sqrt(bars) scaling.
2. Insight target: VOL (next-bar log range log((high-low)/close)).
3. Causal truncation test verifying zero look-ahead across all 44 features.
"""

from typing import Dict
import numpy as np
import pandas as pd


def make_labels_v2(
    df_features: pd.DataFrame,
    base_threshold: float = 0.0010,
    **kwargs
) -> pd.DataFrame:
    """Compute forward returns and decisive masks for H1, H3, HD, and VOL targets.

    Parameters
    ----------
    df_features : pd.DataFrame
        Session-anchored 1h bars with computed features.
    base_threshold : float
        Decisive threshold for 1-bar horizon (default: 10 bps = 0.0010).

    Returns
    -------
    pd.DataFrame
        DataFrame with targets fwd_H1, y_H1, decisive_H1, fwd_H3, y_H3, decisive_H3,
        fwd_HD, y_HD, decisive_HD, and target_vol.
    """
    if "label_threshold" in kwargs:
        base_threshold = kwargs["label_threshold"]

    df = df_features.copy()
    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close
    dates = df.index.date
    session_offset = df["bars_since_open"].to_numpy() if "bars_since_open" in df.columns else np.zeros(len(df))

    # --- Target 1: H1 (Next 1-hour bar, same day) ---
    c_next_1 = close.shift(-1)
    dates_next_1 = pd.Series(dates, index=df.index).shift(-1)
    same_day_1 = (pd.Series(dates, index=df.index) == dates_next_1)

    fwd_h1 = (c_next_1 - close) / close
    fwd_h1 = fwd_h1.where(same_day_1, np.nan)
    df["fwd_H1"] = fwd_h1
    df["y_H1"] = np.where(fwd_h1 > 0, 1, 0)
    df.loc[fwd_h1.isna(), "y_H1"] = np.nan
    df["decisive_H1"] = (fwd_h1.abs() > base_threshold) & fwd_h1.notna()

    # Backwards compatibility for single-head references
    df["fwd"] = df["fwd_H1"]
    df["y"] = df["y_H1"]
    df["decisive"] = df["decisive_H1"]

    # --- Target 2: H3 (Next 3-hour bar, strictly same day) ---
    c_next_3 = close.shift(-3)
    dates_next_3 = pd.Series(dates, index=df.index).shift(-3)
    same_day_3 = (pd.Series(dates, index=df.index) == dates_next_3)

    fwd_h3 = (c_next_3 - close) / close
    fwd_h3 = fwd_h3.where(same_day_3, np.nan)
    df["fwd_H3"] = fwd_h3
    df["y_H3"] = np.where(fwd_h3 > 0, 1, 0)
    df.loc[fwd_h3.isna(), "y_H3"] = np.nan
    thresh_h3 = base_threshold * np.sqrt(3.0)
    df["decisive_H3"] = (fwd_h3.abs() > thresh_h3) & fwd_h3.notna()

    # --- Target 3: HD (Rest of day: session close vs current close, >=2 bars remaining) ---
    # Session close is the last bar's close for that calendar date
    day_close_map = df.groupby(dates)["close"].last()
    day_close_series = pd.Series(dates, index=df.index).map(day_close_map)

    # Valid only if at least 2 bars remain in session (i.e. session_offset <= 4)
    has_at_least_2_bars = (session_offset <= 4)
    bars_remaining = np.maximum(6 - session_offset, 1)

    fwd_hd = (day_close_series - close) / close
    fwd_hd = fwd_hd.where(has_at_least_2_bars, np.nan)
    df["fwd_HD"] = fwd_hd
    df["y_HD"] = np.where(fwd_hd > 0, 1, 0)
    df.loc[fwd_hd.isna(), "y_HD"] = np.nan

    thresh_hd = base_threshold * np.sqrt(bars_remaining)
    df["decisive_HD"] = (fwd_hd.abs() > thresh_hd) & fwd_hd.notna()

    # --- Target 4: VOL (Next-bar log range log((high-low)/close)) ---
    high_next_1 = high.shift(-1)
    low_next_1 = low.shift(-1)
    range_next = (high_next_1 - low_next_1).clip(lower=1e-5)
    log_hl_range = np.log(range_next / c_next_1)
    log_hl_range = log_hl_range.where(same_day_1, np.nan)
    df["target_vol"] = log_hl_range

    return df


def compute_sample_weights_v2(
    y_train: pd.Series,
    timestamps: pd.Series = None,
    half_life_days: float = None,
    class_balance: bool = True
) -> np.ndarray:
    """Compute recency and class-balanced sample weights for training.

    Parameters
    ----------
    y_train : pd.Series or np.ndarray
        Binary labels (0 or 1).
    timestamps : pd.Series, optional
        Bar timestamps for exponential recency decay.
    half_life_days : float, optional
        Half-life in calendar days (e.g. 90, 180, 365, or None for uniform).
    class_balance : bool
        Whether to balance positive and negative class weights.
    """
    y = np.asarray(y_train)
    weights = np.ones(len(y), dtype=float)

    # 1. Recency weighting
    if half_life_days is not None and half_life_days > 0 and timestamps is not None:
        ts = pd.to_datetime(timestamps)
        t_max = ts.max()
        td = t_max - ts
        if hasattr(td, "dt"):
            days_diff = td.dt.total_seconds() / (24.0 * 3600.0)
        else:
            days_diff = td.total_seconds() / (24.0 * 3600.0)
        # weight = 2^(-days_diff / half_life)
        recency = np.power(2.0, -np.asarray(days_diff) / float(half_life_days))
        weights *= recency

    # 2. Class balance
    if class_balance and len(y) > 0:
        pos_mask = (y == 1)
        neg_mask = (y == 0)
        w_pos_sum = np.sum(weights[pos_mask])
        w_neg_sum = np.sum(weights[neg_mask])
        if w_pos_sum > 0 and w_neg_sum > 0:
            target_half = np.sum(weights) / 2.0
            weights[pos_mask] *= (target_half / w_pos_sum)
            weights[neg_mask] *= (target_half / w_neg_sum)

    # Normalize mean weight to 1.0
    if len(weights) > 0 and np.mean(weights) > 0:
        weights /= np.mean(weights)

    return weights


def run_truncation_test_v2(
    df_gs: pd.DataFrame,
    context_dfs: Dict[str, pd.DataFrame],
    n_samples: int = 50,
    seed: int = 42
) -> bool:
    """Verify that features at bar t computed on history up to t match full-dataset features."""
    from src.features import make_features_v2, FEATURES

    print(f"\n--- Running Causal Truncation Test on All {len(FEATURES)} Features ({n_samples} random cutoffs, seed {seed}) ---")
    rng = np.random.default_rng(seed)

    # Compute full dataset features without warmup drop
    df_full = make_features_v2(df_gs, context_dfs, warmup_rows=0)

    # Test only points after bar 250 (allowing all rolling 200-bar MAs and 20-day daily shifts to populate)
    eligible_indices = np.arange(250, len(df_full))
    sampled_indices = rng.choice(eligible_indices, size=min(n_samples, len(eligible_indices)), replace=False)

    max_abs_diff = 0.0
    violating_feature = None

    for idx in sampled_indices:
        t_cutoff = df_full.index[idx]

        # Truncate raw inputs strictly up to t_cutoff
        df_gs_trunc = df_gs.loc[:t_cutoff]
        context_trunc = {sym: df.loc[:t_cutoff] for sym, df in context_dfs.items()}

        # Recompute features on truncated history
        df_trunc = make_features_v2(df_gs_trunc, context_trunc, warmup_rows=0)

        # Compare feature vector at exactly t_cutoff
        full_row = df_full.loc[t_cutoff, FEATURES]
        trunc_row = df_trunc.loc[t_cutoff, FEATURES]

        diff = (full_row - trunc_row).abs()
        max_diff_idx = diff.max()
        if max_diff_idx > max_abs_diff:
            max_abs_diff = max_diff_idx
            violating_feature = diff.idxmax()

    print(f"Truncation test: {n_samples} cutoff points evaluated.")
    print(f"Max absolute discrepancy across all {len(FEATURES)} features: {max_abs_diff:.2e}")
    if violating_feature:
        print(f"Worst discrepancy feature: {violating_feature} (discrepancy = {max_abs_diff:.2e})")

    passed = max_abs_diff <= 1e-8
    assert passed, f"LOOK-AHEAD LEAK DETECTED in {violating_feature}! Discrepancy {max_abs_diff} > 1e-8."
    print("PASS: Truncation test passed with zero look-ahead bias confirmed across all features.")
    return passed


# Backward compatibility alias
make_labels = make_labels_v2
