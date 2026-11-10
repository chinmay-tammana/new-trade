"""Chronological split management and validation.

Enforces strict chronological boundaries with 8-bar embargo gaps between blocks:
Base -> gap -> FT -> gap -> Val -> gap -> Test.
Saves block boundaries and index definitions to results/logs/split.json.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

from configs.config import CONFIG


def run_chronological_split():
    dataset_path = Path(CONFIG["paths"]["processed"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset at {dataset_path}")

    df = pd.read_parquet(dataset_path)
    N = len(df)
    t_end = df.index.max()
    era_days = CONFIG["era_days"]
    gap = CONFIG["gap_bars"]

    y4_start = t_end - timedelta(days=era_days)
    y3_start = t_end - timedelta(days=2 * era_days)

    i3 = int(np.searchsorted(df.index, y3_start))
    i4 = int(np.searchsorted(df.index, y4_start))

    n4 = N - i4
    val_len = int(CONFIG["year4_val_frac"] * n4)
    val_end = i4 + val_len

    # Slices in bar positions
    base_slice = (0, i3 - gap)
    ft_slice = (i3, i4 - gap)
    val_slice = (i4, val_end)
    test_slice = (val_end + gap, N)

    df_base = df.iloc[base_slice[0] : base_slice[1]]
    df_ft = df.iloc[ft_slice[0] : ft_slice[1]]
    df_val = df.iloc[val_slice[0] : val_slice[1]]
    df_test = df.iloc[test_slice[0] : test_slice[1]]

    # Assertions
    assert df_base.index.max() < df_ft.index.min(), "Base max timestamp must precede FT min timestamp!"
    assert df_ft.index.max() < df_val.index.min(), "FT max timestamp must precede Val min timestamp!"
    assert df_val.index.max() < df_test.index.min(), "Val max timestamp must precede Test min timestamp!"

    # Gap assertions in bars
    assert (ft_slice[0] - base_slice[1]) >= gap, f"Base->FT gap must be >= {gap} bars!"
    assert (val_slice[0] - ft_slice[1]) >= gap, f"FT->Val gap must be >= {gap} bars!"
    assert (test_slice[0] - val_slice[1]) >= gap, f"Val->Test gap must be >= {gap} bars!"

    print("Chronological split assertions PASSED:")
    print(f"  Base block: [{base_slice[0]}, {base_slice[1]}), {len(df_base)} bars ({df_base['fwd'].notna().sum()} labelled)")
    print(f"    Range: {df_base.index.min()} to {df_base.index.max()}")
    print(f"  FT block:   [{ft_slice[0]}, {ft_slice[1]}), {len(df_ft)} bars ({df_ft['fwd'].notna().sum()} labelled)")
    print(f"    Range: {df_ft.index.min()} to {df_ft.index.max()}")
    print(f"  Val block:  [{val_slice[0]}, {val_slice[1]}), {len(df_val)} bars ({df_val['fwd'].notna().sum()} labelled)")
    print(f"    Range: {df_val.index.min()} to {df_val.index.max()}")
    print(f"  Test block: [{test_slice[0]}, {test_slice[1]}), {len(df_test)} bars ({df_test['fwd'].notna().sum()} labelled)")
    print(f"    Range: {df_test.index.min()} to {df_test.index.max()}")

    # Save to split.json (counts and boundaries only; NO test/val label shares!)
    split_info = {
        "status": "FROZEN",
        "total_bars": N,
        "gap_bars": gap,
        "blocks": {
            "Base": {
                "start_idx": base_slice[0],
                "end_idx": base_slice[1],
                "bar_count": len(df_base),
                "labelled_count": int(df_base["fwd"].notna().sum()),
                "min_timestamp": str(df_base.index.min()),
                "max_timestamp": str(df_base.index.max()),
            },
            "FT": {
                "start_idx": ft_slice[0],
                "end_idx": ft_slice[1],
                "bar_count": len(df_ft),
                "labelled_count": int(df_ft["fwd"].notna().sum()),
                "min_timestamp": str(df_ft.index.min()),
                "max_timestamp": str(df_ft.index.max()),
            },
            "Val": {
                "start_idx": val_slice[0],
                "end_idx": val_slice[1],
                "bar_count": len(df_val),
                "labelled_count": int(df_val["fwd"].notna().sum()),
                "min_timestamp": str(df_val.index.min()),
                "max_timestamp": str(df_val.index.max()),
            },
            "Test": {
                "start_idx": test_slice[0],
                "end_idx": test_slice[1],
                "bar_count": len(df_test),
                "labelled_count": int(df_test["fwd"].notna().sum()),
                "min_timestamp": str(df_test.index.min()),
                "max_timestamp": str(df_test.index.max()),
            },
        },
        "frozen_at": datetime.now().isoformat(),
    }

    out_file = Path(CONFIG["paths"]["split"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Saved frozen split configuration to {out_file}")

    return split_info


if __name__ == "__main__":
    run_chronological_split()
