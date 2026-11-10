"""Exploratory Data Analysis and Dataset Assembly.

Assembles dataset.parquet (features, fwd, y, decisive, block), runs the 50-cutoff
causal truncation test, checks integrity, and produces Base-only EDA diagnostics
along with Base vs FT Kolmogorov-Smirnov distribution drift analysis.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from configs.config import CONFIG
from src.features import FEATURES, make_features
from src.labels import make_labels, run_truncation_test


def build_and_audit_dataset():
    proc_dir = Path("data/processed")
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("results/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load 1h bars
    df_gs = pd.read_parquet(proc_dir / "bars_GS_1h.parquet")
    df_spy = pd.read_parquet(proc_dir / "bars_SPY_1h.parquet")
    df_xlf = pd.read_parquet(proc_dir / "bars_XLF_1h.parquet")

    # 2. Run truncation test on raw series before assembling
    run_truncation_test(df_gs, df_spy, df_xlf, n_samples=50, seed=CONFIG["seed"])

    # 3. Compute stationary features with warmup drop
    warmup = CONFIG["warmup_rows"]
    df_feat = make_features(df_gs, df_spy, df_xlf, warmup_rows=warmup)

    # 4. Generate forward returns and labels
    df_dataset = make_labels(df_feat, label_threshold=CONFIG["label_threshold"])

    # 5. Define chronological blocks
    t_end = df_dataset.index.max()
    era_days = CONFIG["era_days"]
    y4_start = t_end - timedelta(days=era_days)
    y3_start = t_end - timedelta(days=2 * era_days)

    gap = CONFIG["gap_bars"]
    i3 = int(np.searchsorted(df_dataset.index, y3_start))
    i4 = int(np.searchsorted(df_dataset.index, y4_start))

    n_total = len(df_dataset)
    n4 = n_total - i4
    val_len = int(CONFIG["year4_val_frac"] * n4)
    val_end_idx = i4 + val_len

    # Mark blocks
    df_dataset["block"] = "gap"
    df_dataset.iloc[0 : i3 - gap, df_dataset.columns.get_loc("block")] = "Base"
    df_dataset.iloc[i3 : i4 - gap, df_dataset.columns.get_loc("block")] = "FT"
    df_dataset.iloc[i4 : val_end_idx, df_dataset.columns.get_loc("block")] = "Val"
    df_dataset.iloc[val_end_idx + gap : n_total, df_dataset.columns.get_loc("block")] = "Test"

    # Integrity assertions
    # No infs
    assert not np.isinf(df_dataset[FEATURES].to_numpy()).any(), "Infinite values detected in features!"
    # No constant features
    for f in FEATURES:
        assert df_dataset[f].nunique() > 1, f"Constant feature detected: {f}"

    # Verify last bar of each trading day is unlabelled
    dates = df_dataset.index.date
    for d in np.unique(dates):
        day_mask = dates == d
        last_idx = np.where(day_mask)[0][-1]
        assert pd.isna(df_dataset.iloc[last_idx]["fwd"]), f"Last bar of day {d} has non-NaN forward return!"

    # Save dataset.parquet
    out_path = Path(CONFIG["paths"]["processed"])
    df_dataset.to_parquet(out_path)
    print(f"Dataset successfully created and saved to {out_path} ({len(df_dataset)} rows).")

    # Print Up/Down/decisive shares for Base and FT only (Test-set discipline strictly observed)
    for b_name in ["Base", "FT"]:
        sub = df_dataset[(df_dataset["block"] == b_name) & df_dataset["fwd"].notna()]
        n_sub = len(sub)
        n_up = (sub["y"] == 1).sum()
        n_dec = sub["decisive"].sum()
        dec_up = (sub["decisive"] & (sub["y"] == 1)).sum()
        print(f"\n{b_name} Block Statistics (Labelled rows = {n_sub}):")
        print(f"  Up share: {n_up}/{n_sub} ({n_up / n_sub * 100:.2f}%)")
        print(f"  Decisive share (|fwd| > {CONFIG['label_threshold']}): {n_dec}/{n_sub} ({n_dec / n_sub * 100:.2f}%)")
        if n_dec > 0:
            print(f"  Decisive Up share: {dec_up}/{n_dec} ({dec_up / n_dec * 100:.2f}%)")

    # 6. Run EDA on BASE block only
    run_base_eda(df_dataset, figures_dir)

    # 7. Run Drift Check between Base and FT
    run_drift_check(df_dataset, figures_dir)


def run_base_eda(df_dataset: pd.DataFrame, figures_dir: Path):
    """Run exploratory data analysis strictly on the Base training block."""
    df_base = df_dataset[df_dataset["block"] == "Base"].copy()
    labelled_base = df_base[df_base["fwd"].notna()].copy()

    # 1. Target Class Counts Plot
    plt.figure(figsize=(7, 4.5))
    dec_up = (labelled_base["decisive"] & (labelled_base["y"] == 1)).sum()
    dec_down = (labelled_base["decisive"] & (labelled_base["y"] == 0)).sum()
    neut_up = (~labelled_base["decisive"] & (labelled_base["y"] == 1)).sum()
    neut_down = (~labelled_base["decisive"] & (labelled_base["y"] == 0)).sum()

    cats = ["Decisive Up\n(>+10 bps)", "Decisive Down\n(<-10 bps)", "Neutral Up\n(0 to +10 bps)", "Neutral Down\n(-10 to 0 bps)"]
    counts = [dec_up, dec_down, neut_up, neut_down]
    colors = ["#2ca02c", "#d62728", "#98df8a", "#ff9896"]

    bars = plt.bar(cats, counts, color=colors, edgecolor="black")
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, h + 20, f"{h}\n({h / len(labelled_base) * 100:.1f}%)", ha="center", fontsize=9)
    plt.title("Base Era Label Breakdown: Decisive vs Neutral Intraday Moves (GS)", fontsize=11, fontweight="bold")
    plt.ylabel("Candle Count")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_target_distribution.png", dpi=200)
    plt.close()

    # 2. Key Feature Distributions
    key_features = ["ret_1", "rsi", "bb_pos", "atr_n", "rvol", "rel_ret_1"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for i, feat in enumerate(key_features):
        vals = df_base[feat].dropna()
        axes[i].hist(vals, bins=40, color="#1f77b4", edgecolor="black", alpha=0.7)
        axes[i].set_title(f"{feat} (mean={vals.mean():.4f}, std={vals.std():.4f})", fontsize=10, fontweight="bold")
        axes[i].grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Distributions of Key Stationary Predictors in Base Era", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_feature_distributions.png", dpi=200)
    plt.close()

    # 3. Correlation Heatmap
    corr = df_base[FEATURES].corr()
    plt.figure(figsize=(11, 9))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson Correlation")
    plt.xticks(range(len(FEATURES)), FEATURES, rotation=90, fontsize=8)
    plt.yticks(range(len(FEATURES)), FEATURES, fontsize=8)
    plt.title("Feature Cross-Correlation Heatmap (Base Era)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_correlation_heatmap.png", dpi=200)
    plt.close()

    # 4. Autocorrelation of ret_1 and |ret_1|
    max_lag = 15
    ret = df_base["ret_1"].dropna()
    abs_ret = ret.abs()

    acf_ret = [ret.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
    acf_abs = [abs_ret.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
    lags = np.arange(1, max_lag + 1)

    plt.figure(figsize=(8, 4.5))
    plt.plot(lags, acf_ret, marker="o", color="#1f77b4", lw=1.5, label="Raw Return (ret_1)")
    plt.plot(lags, acf_abs, marker="s", color="#ff7f0e", lw=1.5, label="Absolute Return (|ret_1| Volatility)")
    conf_band = 1.96 / np.sqrt(len(ret))
    plt.axhline(conf_band, linestyle="--", color="gray", alpha=0.7, label="95% Significance Band")
    plt.axhline(-conf_band, linestyle="--", color="gray", alpha=0.7)
    plt.axhline(0, color="black", lw=0.8)
    plt.title("Autocorrelation of Hourly Returns and Volatility Clustering (Base Era)", fontsize=11, fontweight="bold")
    plt.xlabel("Lag (1-Hour Bars)")
    plt.ylabel("Autocorrelation")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_autocorrelation.png", dpi=200)
    plt.close()

    # 5. Up-Rate by Hour of Day and Day of Week
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    hour_uprate = labelled_base.groupby("hour")["y"].agg(["mean", "count"])
    ax1.bar([f"{h:.1f}" for h in hour_uprate.index], hour_uprate["mean"] * 100, color="#3498db", edgecolor="black")
    ax1.axhline(50, color="crimson", linestyle="--", lw=1.2)
    ax1.set_title("Up-Rate by Regular Session Hour (ET)", fontweight="bold")
    ax1.set_xlabel("Session Hour")
    ax1.set_ylabel("Up Rate (%)")
    ax1.set_ylim(40, 60)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    dow_uprate = labelled_base.groupby("dow")["y"].agg(["mean", "count"])
    ax2.bar([dow_names[d] for d in dow_uprate.index], dow_uprate["mean"] * 100, color="#e67e22", edgecolor="black")
    ax2.axhline(50, color="crimson", linestyle="--", lw=1.2)
    ax2.set_title("Up-Rate by Day of Week", fontweight="bold")
    ax2.set_xlabel("Day of Week")
    ax2.set_ylabel("Up Rate (%)")
    ax2.set_ylim(40, 60)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    plt.suptitle("Directional Drift Patterns in Goldman Sachs (Base Era)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_hourly_dow_uprate.png", dpi=200)
    plt.close()


def run_drift_check(df_dataset: pd.DataFrame, figures_dir: Path):
    """Run two-sample Kolmogorov-Smirnov test between Base and FT feature distributions."""
    base_rows = df_dataset[df_dataset["block"] == "Base"]
    ft_rows = df_dataset[df_dataset["block"] == "FT"]

    ks_records = []
    for f in FEATURES:
        s_base = base_rows[f].dropna()
        s_ft = ft_rows[f].dropna()
        ks_stat, p_val = stats.ks_2samp(s_base, s_ft)
        ks_records.append({"feature": f, "ks_stat": ks_stat, "p_value": p_val})

    df_ks = pd.DataFrame(ks_records).sort_values("ks_stat", ascending=False)
    df_ks.to_csv("results/tables/drift_ks_base_vs_ft.csv", index=False)

    plt.figure(figsize=(9, 5))
    bars = plt.barh(df_ks["feature"], df_ks["ks_stat"], color="#e74c3c", edgecolor="black")
    plt.gca().invert_yaxis()
    plt.axvline(0.10, color="black", linestyle="--", lw=1.2, label="KS = 0.10 Threshold")
    plt.title("Base vs FT Distribution Shift (Two-Sample Kolmogorov-Smirnov Statistic)", fontsize=11, fontweight="bold")
    plt.xlabel("KS Statistic (Higher = Greater Distribution Shift)")
    plt.legend()
    plt.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_drift_ks_base_vs_ft.png", dpi=200)
    plt.close()
    print("Drift check complete. Saved results/tables/drift_ks_base_vs_ft.csv and results/figures/eda_drift_ks_base_vs_ft.png.")


if __name__ == "__main__":
    build_and_audit_dataset()
