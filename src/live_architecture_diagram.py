"""Generate live architecture diagram for Goldman Sachs AI Trading Indicator."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Colors
    c_bg = "#f8fafc"
    c_border = "#cbd5e1"
    c_header = "#1e293b"
    c_blue = "#2563eb"
    c_blue_bg = "#eff6ff"
    c_emerald = "#059669"
    c_emerald_bg = "#ecfdf5"
    c_amber = "#d97706"
    c_amber_bg = "#fffbeb"
    c_purple = "#7c3aed"
    c_purple_bg = "#faf5ff"
    c_gray = "#475569"

    # Title
    ax.text(7, 8.5, "Goldman Sachs (GS) V3 Live Indicator Architecture",
            ha="center", va="center", fontsize=18, fontweight="bold", color=c_header)
    ax.text(7, 8.1, "Deterministic 1h Session Engine | Immutable Audit Store | Zero-Trust Web & SSE UI",
            ha="center", va="center", fontsize=11, color=c_gray)

    def draw_box(x, y, w, h, title, subtitle, bg_col, border_col, title_col=c_header):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=bg_col, edgecolor=border_col, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 0.35, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=title_col)
        if subtitle:
            ax.text(x + w/2, y + (h - 0.4)/2, subtitle, ha="center", va="center",
                    fontsize=8.5, color="#334155", multialignment="center")

    def draw_arrow(x1, y1, x2, y2, label=""):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle="->,head_width=4,head_length=6",
                                color="#64748b", linewidth=1.8, mutation_scale=15)
        ax.add_patch(arrow)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.15, label, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="#475569",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#e2e8f0", alpha=0.9))

    # 1. External Ingestion
    draw_box(0.5, 5.0, 3.2, 2.4, "External Feeds",
             "Alpaca REST API (IEX Feed)\n1-minute OHLCV bars\nGS, SPY, XLF, JPM, MS,\nBAC, TLT, VIXY\n+ Alpaca WS (Live Trades)",
             c_blue_bg, c_blue, c_blue)

    # 2. Live Engine & Aggregator
    draw_box(4.7, 5.0, 4.6, 2.4, "Live Engine & Aggregator",
             "NyseCalendar (Half-day / DST aware)\nSession-Anchored 1h Aggregator\nRolling 600-Bar Buffer (Causal Features)\nTimed Fetch (Bar Close + 10s)\nRetry with Backoff (up to 90s)",
             c_emerald_bg, c_emerald, c_emerald)

    # 3. Frozen Models & Verification
    draw_box(10.3, 5.0, 3.2, 2.4, "Frozen Models",
             "model_v2.joblib (H1 XGBoost)\ncalib_v2.joblib (Isotonic)\ncalib_live.json (STRONG 52.04%)\nMODEL_SHA256.json (Startup Lock)\nZero Model Changes Enforced",
             c_purple_bg, c_purple, c_purple)

    # 4. Immutable SQLite Store
    draw_box(4.7, 1.2, 4.6, 2.6, "Immutable Audit Store",
             "SQLite WAL Mode (data/live/live.db)\nSingleInstanceLock (live.lock)\nAppend-Only Signals Table (Abort Triggers)\nCryptographic Hash Chaining (SHA-256)\nOutcomes & 1h Bar Cache",
             c_amber_bg, c_amber, c_amber)

    # 5. Read-Only FastAPI Backend
    draw_box(0.5, 1.2, 3.2, 2.6, "FastAPI Web Server",
             "Read-Only Endpoints (No POST/PUT)\nGET /healthz, /api/status\nGET /api/bars, /api/signals\nGET /api/latest, /api/scoreboard\nSSE Streaming (/api/stream)\nZero Secrets Leaked",
             "#f8fafc", "#94a3b8")

    # 6. Static Frontend Client
    draw_box(10.3, 1.2, 3.2, 2.6, "TradingView Frontend",
             "TradingView Lightweight Charts v4.2.1\nLive GS Candlesticks + Forming Bar\nUp/Down Signal Markers & Confidence\nVolatility / Regime / STRONG Tiers\nParity Fallback / Event Banners",
             "#f8fafc", "#94a3b8")

    # Arrows
    draw_arrow(3.7, 6.2, 4.7, 6.2, "1m Bars")
    draw_arrow(9.3, 6.2, 10.3, 6.2, "Features")
    draw_arrow(10.3, 5.7, 9.3, 5.7, "Signals")
    draw_arrow(7.0, 5.0, 7.0, 3.8, "Append Signal")
    draw_arrow(4.7, 2.5, 3.7, 2.5, "Read State")
    draw_arrow(4.7, 6.8, 2.1, 6.8, "")
    draw_arrow(2.1, 3.8, 2.1, 5.0, "Health / Clock")
    draw_arrow(3.7, 1.8, 4.7, 1.8, "Query DB")
    draw_arrow(3.7, 2.2, 10.3, 2.2, "REST / SSE")

    plt.tight_layout()
    out_path = Path("results/figures/live_architecture.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, facecolor="white")
    plt.close()
    print(f"Saved live architecture diagram: {out_path}")

if __name__ == "__main__":
    generate_architecture_diagram()
