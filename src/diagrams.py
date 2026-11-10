"""Phase 10: System Architecture Diagram Generator.

Generates results/figures/architecture.png illustrating the end-to-end ML data flow,
leakage boundaries, fit scopes, calibration, and held-out evaluation per IMPACT Lab guidelines.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from configs.config import CONFIG


def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Background color
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # Title & Subtitle
    ax.text(8.0, 9.6, "Goldman Sachs (GS) Directional Trading Indicator: System Architecture",
            ha="center", va="center", fontsize=16, fontweight="bold", color="#1E293B")
    ax.text(8.0, 9.2, "End-to-End Pipeline, Causal Boundaries, In-Fold Preprocessing, and Single-Read Test Discipline",
            ha="center", va="center", fontsize=11, color="#64748B", style="italic")

    # Helper function for drawing boxes
    def draw_box(x, y, w, h, title, subtitle, fill_color, border_color, text_color="#1E293B", badge=""):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                      linewidth=1.5, edgecolor=border_color, facecolor=fill_color)
        ax.add_patch(rect)
        if badge:
            badge_rect = patches.FancyBboxPatch((x + 0.1, y + h - 0.28), len(badge)*0.09 + 0.3, 0.22,
                                                boxstyle="round,pad=0.04,rounding_size=0.06",
                                                linewidth=0, facecolor=border_color)
            ax.add_patch(badge_rect)
            ax.text(x + 0.15, y + h - 0.17, badge, ha="left", va="center", fontsize=7.5, fontweight="bold", color="white")
        
        ax.text(x + w / 2, y + h * 0.58 if subtitle else y + h * 0.5, title,
                ha="center", va="center", fontsize=9.5, fontweight="bold", color=text_color, wrap=True)
        if subtitle:
            ax.text(x + w / 2, y + h * 0.28, subtitle,
                    ha="center", va="center", fontsize=7.5, color="#475569", wrap=True)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#334155", lw=1.5, shrinkA=3, shrinkB=3))
        if label:
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.12, label, ha="center", va="bottom", fontsize=7.5, color="#334155")

    # Column 1: Raw Ingestion & Session Aggregation
    draw_box(0.6, 6.8, 3.2, 1.6, "1. Ingestion: Alpaca SIP",
             "GS, SPY, XLF 1-min raw bars\nOct 2021 - Oct 2025 (4+ yrs)\nyfinance audit: 0.999998 corr",
             "#E0F2FE", "#0284C7", badge="RAW INGESTION")
    
    draw_arrow(2.2, 6.8, 2.2, 5.8)

    draw_box(0.6, 4.2, 3.2, 1.6, "2. Session Aggregation",
             "7 regular-hours 1h candles/day\n9:30-10:30 ... 15:30-16:00 (30m)\nZero zero-vol, 0 gaps in inner join",
             "#E0F2FE", "#0284C7", badge="TIME RESAMPLING")

    draw_arrow(2.2, 4.2, 2.2, 3.2)

    draw_box(0.6, 1.6, 3.2, 1.6, "3. Features & Labels",
             "22 causal, stationary features\nLabel: fwd ret > +10 bps (1), < -10 bps (0)\nOvernight & last-bar NaN (censored)",
             "#E0F2FE", "#0284C7", badge="FEATURE ENGIN.")

    # Arrow to Column 2
    draw_arrow(3.8, 2.4, 4.8, 2.4)

    # Column 2: Chronological Split & Embargoes
    draw_box(4.8, 1.6, 3.2, 6.8, "4. Chronological Split\n(Zero Look-Ahead)",
             "• Base Block: Oct 2021 - Dec 2023\n  [0, 3613), 3096 labelled bars\n  (8-bar embargo)\n"
             "• FT Block: Dec 2023 - Nov 2024\n  [3621, 5353), 1484 labelled bars\n  (8-bar embargo)\n"
             "• Val Block: Nov 2024 - Mar 2025\n  [5361, 6063), 602 labelled bars\n  (8-bar embargo)\n"
             "• Test Block: Mar 2025 - Oct 2025\n  [6071, 7118), 896 labelled bars",
             "#F1F5F9", "#475569", badge="EMBARGOED SPLIT")

    # Column 3: In-Fold Preprocessing & Model Training
    draw_arrow(8.0, 6.6, 8.8, 6.6)
    draw_box(8.8, 5.8, 3.2, 1.6, "5. In-Fold Preprocessing",
             "Winsorizer (1%, 99% quantiles)\n+ StandardScaler\nFitted strictly on Base+FT fold",
             "#FEF3C7", "#D97706", badge="FIT: BASE+FT")

    draw_arrow(10.4, 5.8, 10.4, 4.8)
    draw_box(8.8, 3.2, 3.2, 1.6, "6. XGBoost Model Training",
             "S0: Base Default fit\nS2: Refit on Base+FT with CV params\nWalk-forward CV on FT (4 folds)\nCV Log-Loss: 0.69423",
             "#FEF3C7", "#D97706", badge="FIT: BASE+FT")

    draw_arrow(10.4, 3.2, 10.4, 2.2)
    draw_box(8.8, 0.6, 3.2, 1.6, "7. Probability Calibration\n& Signal Search",
             "Platt Scaling (Log-Loss: 0.68704)\nAsymmetric threshold search:\nd_up = 0.06 (p >= 0.56, Up)\nd_down = None (Down disabled)",
             "#FEF3C7", "#D97706", badge="FIT: VAL BLOCK")

    # Column 4: Frozen Model & Single-Read Test
    draw_arrow(12.0, 1.4, 12.6, 1.4)
    draw_arrow(12.6, 1.4, 12.6, 4.0)
    draw_arrow(12.6, 4.0, 12.6, 6.0)

    draw_box(12.6, 5.6, 3.0, 2.2, "8. Frozen Artifacts",
             "• models/model.joblib\n• models/model.json\n• models/calib.joblib\n• Thresholds: [0.56, 0.00]\n• Commit: model-v1",
             "#ECFDF5", "#059669", badge="FROZEN SYSTEM")

    draw_arrow(14.1, 5.6, 14.1, 4.2)
    draw_box(12.6, 1.8, 3.0, 2.4, "9. Final Evaluation\n(Strict Single-Read)",
             "Evaluated ONCE on Test (896 bars)\n• Coverage: 20.98% (188 Up, 0 Down)\n• Accuracy: 46.28% [39.3, 53.4]%\n"
             "• Drift (Always-Up): 46.28%\n• Outperformance: +0.00%\n• Net Return @ 5bps: -7.24 bps",
             "#FEE2E2", "#DC2626", badge="READ ONCE: TEST")

    # Explanatory bottom caption
    ax.text(8.0, 0.2, "Note: Preprocessing, training, and fine-tuning are fitted strictly prior to the Validation block. "
                      "Validation block is used exclusively for probability calibration and threshold search. "
                      "Test block is read exactly ONCE after all code and hyperparameters are locked.",
            ha="center", va="center", fontsize=8.5, color="#334155", bbox=dict(boxstyle="square,pad=0.4", fc="white", ec="#CBD5E1", lw=1))

    out_path = Path(CONFIG["paths"]["figures"]) / "architecture.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Architecture diagram saved to {out_path}")


if __name__ == "__main__":
    generate_architecture_diagram()
