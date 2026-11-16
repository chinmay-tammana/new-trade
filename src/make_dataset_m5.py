"""
Compile complete 5-minute feature dataset from processed 5-minute bars.
Saves to data/processed/m5/features_m5.parquet.
"""
import os
import json
import pandas as pd
from configs.config import CONFIG_M5
from src.features_m5 import make_features_m5

PROCESSED_DIR = "data/processed/m5"


def main():
    print("Loading 5-minute bars for GS and context symbols...")
    gs_path = os.path.join(PROCESSED_DIR, "bars_5m_GS.parquet")
    if not os.path.exists(gs_path):
        raise FileNotFoundError(f"GS 5-minute bars not found at {gs_path}")
    
    df_gs = pd.read_parquet(gs_path)
    
    # Read freeze file if exists
    freeze_path = CONFIG_M5["paths"]["data_freeze"]
    if os.path.exists(freeze_path):
        with open(freeze_path, "r") as f:
            freeze_info = json.load(f)
        kept_symbols = freeze_info.get("symbols_kept", ["GS", "SPY", "XLF", "JPM", "MS", "BAC"])
    else:
        kept_symbols = ["GS", "SPY", "XLF", "JPM", "MS", "BAC"]

    context_dfs = {}
    for sym in kept_symbols:
        if sym == "GS":
            continue
        c_path = os.path.join(PROCESSED_DIR, f"bars_5m_{sym}.parquet")
        if os.path.exists(c_path):
            context_dfs[sym] = pd.read_parquet(c_path)
            print(f"Loaded {sym}: {len(context_dfs[sym]):,} bars")

    print(f"\nComputing features and labels on {len(df_gs):,} GS 5-minute bars...")
    df_feat = make_features_m5(df_gs, context_dfs=context_dfs)
    
    out_path = CONFIG_M5["paths"]["processed_features"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_feat.to_parquet(out_path)
    print(f"Saved feature dataset to {out_path} ({len(df_feat):,} rows, {len(df_feat.columns)} columns)")


if __name__ == "__main__":
    main()
