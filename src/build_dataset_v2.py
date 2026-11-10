"""Dataset builder and pre-holdout exploratory analysis (V2 Upgrade).

Assembles data/processed/dataset_v2.parquet containing 51 stationary features,
multi-horizon directional targets (H1, H3, HD), insight targets (VOL, BAND, REGIME),
and chronological partition flags (pre_holdout, gap, holdout).
"""

import json
from pathlib import Path
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from configs.config import CONFIG
from src.features import FEATURES, FEATURE_GROUPS, make_features_v2
from src.labels import make_labels_v2, run_truncation_test_v2


def build_v2_dataset():
    proc_dir = Path("data/processed")
    figures_dir = Path("results/figures")
    tables_dir = Path("results/tables")
    logs_dir = Path("results/logs")
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- 1. Loading 8-Year Hourly Bars ---")
    df_gs = pd.read_parquet(proc_dir / "bars_GS_1h.parquet")
    context_symbols = ["SPY", "XLF", "JPM", "MS", "BAC", "TLT", "VIXY"]
    context_dfs = {sym: pd.read_parquet(proc_dir / f"bars_{sym}_1h.parquet") for sym in context_symbols}

    # Verify synchronization
    common_index = df_gs.index
    for sym, c_df in context_dfs.items():
        common_index = common_index.intersection(c_df.index)

    print(f"GS raw bars: {len(df_gs):,} | Synchronized across all 8 symbols: {len(common_index):,}")
    df_gs = df_gs.loc[common_index].sort_index()
    for sym in context_symbols:
        context_dfs[sym] = context_dfs[sym].loc[common_index].sort_index()

    # 2. Causal truncation test
    print("\n--- 2. Verifying Causal Truncation Test ---")
    trunc_pass = run_truncation_test_v2(df_gs, context_dfs, n_samples=50, seed=CONFIG["seed"])
    assert trunc_pass, "Truncation test failed!"

    # 3. Compute stationary features with warmup
    warmup = CONFIG.get("warmup_rows", 200)
    print(f"\n--- 3. Computing All {len(FEATURES)} Features (Warmup rows: {warmup}) ---")
    df_feat = make_features_v2(df_gs, context_dfs, warmup_rows=warmup)
    print(f"Features computed. Total rows remaining: {len(df_feat):,}")

    # 4. Multi-horizon targets and insight heads
    print("\n--- 4. Computing Multi-Horizon Targets & Insight Heads (H1, H3, HD, VOL) ---")
    df_dataset = make_labels_v2(df_feat, base_threshold=CONFIG["label_threshold"])

    # 5. Define Research Holdout boundary and embargo gap
    holdout_start_str = CONFIG.get("research_holdout_start", "2025-10-16")
    holdout_start_ts = pd.Timestamp(holdout_start_str, tz=CONFIG["tz"])
    freeze_date_ts = pd.Timestamp(CONFIG["freeze_date"], tz=CONFIG["tz"]) + pd.Timedelta(hours=16)

    # Bars up to holdout_start_ts
    pre_mask = df_dataset.index < holdout_start_ts
    pre_indices = np.where(pre_mask)[0]
    last_pre_idx = pre_indices[-1]

    gap_bars = CONFIG.get("gap_bars", 8)
    # The 8 bars immediately preceding holdout_start_ts form the embargo gap
    gap_start_idx = last_pre_idx - gap_bars + 1
    
    # Assign partition blocks
    df_dataset["block"] = "pre_holdout"
    df_dataset.iloc[gap_start_idx : last_pre_idx + 1, df_dataset.columns.get_loc("block")] = "gap"
    df_dataset.loc[df_dataset.index >= holdout_start_ts, "block"] = "holdout"

    # Filter strictly up to freeze_date
    df_dataset = df_dataset[df_dataset.index <= freeze_date_ts].copy()

    block_counts = df_dataset["block"].value_counts()
    print("\nPartition Block Breakdown:")
    for b_name, count in block_counts.items():
        print(f"  - {b_name}: {count:,} bars")

    # Save processed dataset
    out_path = Path(CONFIG["paths"]["processed"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_dataset.to_parquet(out_path)
    print(f"\nProcessed dataset saved to {out_path} ({len(df_dataset):,} rows).")

    # Save split metadata
    split_meta = {
        "dataset_start": str(df_dataset.index[0]),
        "dataset_end": str(df_dataset.index[-1]),
        "total_bars": len(df_dataset),
        "pre_holdout_bars": int((df_dataset["block"] == "pre_holdout").sum()),
        "gap_bars": int((df_dataset["block"] == "gap").sum()),
        "holdout_bars": int((df_dataset["block"] == "holdout").sum()),
        "holdout_start": str(holdout_start_ts),
        "freeze_date": str(freeze_date_ts),
        "n_features": len(FEATURES),
        "feature_groups": {grp: len(feats) for grp, feats in FEATURE_GROUPS.items()},
        "frozen_at": pd.Timestamp.now(ZoneInfo(CONFIG["tz"])).isoformat()
    }
    with open("results/logs/split_v2.json", "w") as f:
        json.dump(split_meta, f, indent=2)

    # 6. Pre-Holdout EDA and Feature Drift
    print("\n--- 5. Performing Pre-Holdout EDA & KS Drift Analysis ---")
    run_pre_holdout_eda(df_dataset, figures_dir, tables_dir)

    return df_dataset


def run_pre_holdout_eda(df_dataset: pd.DataFrame, figures_dir: Path, tables_dir: Path):
    """Run exploratory data analysis strictly on pre_holdout data to avoid look-ahead."""
    df_pre = df_dataset[df_dataset["block"] == "pre_holdout"].copy()
    print(f"Pre-Holdout sample size for EDA: {len(df_pre):,} bars")

    # Split pre_holdout into earlier half and later half for KS drift test
    mid = len(df_pre) // 2
    df_early = df_pre.iloc[:mid]
    df_late = df_pre.iloc[mid:]

    drift_records = []
    for feat in FEATURES:
        s1 = df_early[feat].dropna()
        s2 = df_late[feat].dropna()
        ks_stat, p_val = stats.ks_2samp(s1, s2)
        drift_records.append({
            "feature": feat,
            "ks_stat": float(ks_stat),
            "p_value": float(p_val),
            "drift_detected": bool(p_val < 0.01)
        })

    df_drift = pd.DataFrame(drift_records).sort_values("ks_stat", ascending=False)
    df_drift.to_csv(tables_dir / "drift_ks_v2.csv", index=False)
    print(f"KS drift analysis saved to {tables_dir / 'drift_ks_v2.csv'}")

    # Plot top 12 drifted features
    plt.figure(figsize=(10, 6))
    top_drift = df_drift.head(12)
    colors = ["#d9534f" if d else "#337ab7" for d in top_drift["drift_detected"]]
    plt.barh(top_drift["feature"][::-1], top_drift["ks_stat"][::-1], color=colors[::-1])
    plt.title("Kolmogorov-Smirnov Drift (Early vs Late Pre-Holdout Era)", fontsize=11, fontweight="bold")
    plt.xlabel("KS Statistic (Distance)")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_drift_ks_v2.png", dpi=200)
    plt.close()

    # Correlation matrix of selected core features
    core_feats = [
        "ret_1", "ret_3", "rsi", "macd_hist_n", "vol_20", "rvol",
        "spy_ret_1", "xlf_ret_1", "jpm_ret_1", "tlt_ret_1", "vixy_ret_1",
        "realized_vol_1m", "signed_volume_ratio", "efficiency_ratio_20"
    ]
    plt.figure(figsize=(10, 8))
    corr = df_pre[core_feats].corr()
    plt.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.colorbar(label="Pearson Correlation")
    plt.xticks(range(len(core_feats)), core_feats, rotation=45, ha="right", fontsize=9)
    plt.yticks(range(len(core_feats)), core_feats, fontsize=9)
    plt.title("Correlation Matrix of Representative V2 Features (Pre-Holdout)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_correlation_matrix_v2.png", dpi=200)
    plt.close()
    print("Pre-Holdout EDA figures generated successfully.")


if __name__ == "__main__":
    build_v2_dataset()
