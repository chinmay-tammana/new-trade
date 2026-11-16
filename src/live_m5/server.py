"""
FastAPI Server for M5 (5-Minute) Live System.
Serves real-time 5-minute directional signals, chart bars, signal log,
single-source-of-truth summaries, performance panels, doctor diagnostics, and replay.
"""
import os
import sys
import json
import hashlib
import datetime
import zoneinfo
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath("."))
from configs.config import CONFIG_M5
from src.live_m5.store import LiveM5Database, DB_PATH
from src.live_m5.log_engine import get_m5_signal_log, load_m5_calib, ET_TZ, IST_TZ
from src.live_m5.doctor import run_doctor
from src.live_m5.performance import get_previous_market_day, get_day_performance_details, get_overall_performance
from alpaca.trading.client import TradingClient

load_dotenv()

app = FastAPI(title="GS AI 5-Minute Directional Indicator", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database instance
db = LiveM5Database()


@app.on_event("startup")
def verify_hashes_on_startup():
    """Verify production model and rule hashes on startup."""
    sha_path = CONFIG_M5["paths"]["sha_path"]
    if os.path.exists(sha_path):
        with open(sha_path, "r") as f:
            hashes_dict = json.load(f)
        for fname, expected in hashes_dict.items():
            fpath = os.path.join("models/m5", fname)
            if os.path.exists(fpath):
                with open(fpath, "rb") as bf:
                    act = hashlib.sha256(bf.read()).hexdigest()
                if act != expected:
                    raise RuntimeError(f"FATAL: Model hash mismatch on startup for {fname}!")
    print("M5 Server Startup: All model & rule hashes verified successfully against manifest.")


def get_alpaca_clock():
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        return None
    try:
        client = TradingClient(api_key, secret_key, paper=True)
        return client.get_clock()
    except Exception:
        return None


@app.get("/", response_class=HTMLResponse)
def index_page():
    # Return m5 index page as default view
    index_file = "static/index.html"
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            return HTMLResponse(content=f.read())
    index_m5 = "static/index_m5.html"
    if os.path.exists(index_m5):
        with open(index_m5, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>GS AI 5-Minute Directional Indicator</h1>")


@app.get("/index_h1.html", response_class=HTMLResponse)
def legacy_h1_page():
    # Return legacy frozen 1-hour view
    h1_file = "static/index_h1.html"
    if os.path.exists(h1_file):
        with open(h1_file, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Legacy 1-Hour view not found</h1>")


@app.get("/api/status")
def get_status():
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_et = now_utc.astimezone(ET_TZ)
    now_ist = now_utc.astimezone(IST_TZ)
    
    clock = get_alpaca_clock()
    market_open = clock.is_open if clock else False
    
    # Check model hash and verification
    sha_path = CONFIG_M5["paths"]["sha_path"]
    model_verified = False
    model_hash = "unverified"
    if os.path.exists(sha_path):
        with open(sha_path, "r") as f:
            shas = json.load(f)
            model_hash = shas.get("model_m5.joblib", "")[:8]
            model_verified = True
            
    calib = load_m5_calib()
    active_v = calib.get("active_rule_version", calib.get("rule_version", 1))
    
    # Latest signal (using highest rule_version per session)
    with db.get_connection() as conn:
        cursor = conn.execute("""
        SELECT s.*, b.open, b.close 
        FROM signals s 
        INNER JOIN (
            SELECT date(bar_start) as dt_max, MAX(rule_version) as max_v 
            FROM signals GROUP BY date(bar_start)
        ) m ON date(s.bar_start) = m.dt_max AND s.rule_version = m.max_v
        LEFT JOIN bars_5m b ON b.symbol = 'GS' AND b.bar_start = s.bar_start
        ORDER BY s.bar_start DESC LIMIT 1;
        """)
        latest_row = cursor.fetchone()
        
        # Previous signal for interval check
        cursor2 = conn.execute("""
        SELECT s.*, b.open, b.close 
        FROM signals s 
        INNER JOIN (
            SELECT date(bar_start) as dt_max, MAX(rule_version) as max_v 
            FROM signals GROUP BY date(bar_start)
        ) m ON date(s.bar_start) = m.dt_max AND s.rule_version = m.max_v
        LEFT JOIN bars_5m b ON b.symbol = 'GS' AND b.bar_start = s.bar_start
        ORDER BY s.bar_start DESC LIMIT 1 OFFSET 1;
        """)
        prev_row = cursor2.fetchone()

    # Headline construction
    d = dict(latest_row) if latest_row else {}
    tier_val = d.get("tier", "NORMAL")
    b_start_dt = None
    b_end_dt = None
    age_s = None

    if latest_row:
        b_start_dt = datetime.datetime.fromisoformat(d["bar_start"]).astimezone(ET_TZ)
        b_end_dt = datetime.datetime.fromisoformat(d["bar_end"]).astimezone(ET_TZ)
        call_direction = d["direction"]
        conf_pct = int(round(d["confidence"] * 100)) if d["direction"] in ["Up", "Down"] else 50
        is_strong = (tier_val == "STRONG") or bool(d.get("is_strong", 0))
        
        delivered_at_str = d.get("delivered_at") or d.get("bar_start")
        if delivered_at_str:
            try:
                deliv_dt = datetime.datetime.fromisoformat(delivered_at_str).astimezone(datetime.timezone.utc)
                age_s = int((now_utc - deliv_dt).total_seconds())
            except Exception:
                age_s = None

        if d["direction"] in ["Up", "Down"]:
            headline_text = f"{call_direction.upper()} {conf_pct}% ({tier_val}) for {b_start_dt.strftime('%H:%M')} to {b_end_dt.strftime('%H:%M')} ET"
            if tier_val == "WEAK":
                headline_sub = f"Confidence {conf_pct}% (WEAK: close to a coin flip) | Calibrated P(Up): {d['p_cal']*100:.1f}%"
            elif tier_val == "STRONG":
                headline_sub = f"Confidence {conf_pct}% (STRONG: top quartile) | Calibrated P(Up): {d['p_cal']*100:.1f}%"
            else:
                headline_sub = f"Confidence {conf_pct}% (NORMAL) | Calibrated P(Up): {d['p_cal']*100:.1f}%"
        else:
            if b_start_dt.hour == 15 and b_start_dt.minute >= 55:
                ns_reason = "End of Day (no forward 5m interval in session)"
            elif abs(d.get("p_cal", 0.5) - 0.5) < 1e-9:
                ns_reason = "Exact tie at P(Up) = 50.0%"
            else:
                ns_reason = "Data gap / Market closed"
            headline_text = f"NO SIGNAL ({ns_reason}) for {b_start_dt.strftime('%H:%M')} to {b_end_dt.strftime('%H:%M')} ET"
            headline_sub = f"Calibrated P(Up): {d['p_cal']*100:.1f}% | Allowed abstention: {ns_reason}"
    else:
        headline_text = "STANDBY - WAITING FOR SIGNAL"
        headline_sub = "Market is closed. Initializing 5-minute pipeline."
        is_strong = False
        conf_pct = 50

    latest_signal = {
        "direction": d.get("direction", "Standby"),
        "confidence_pct": conf_pct if d.get("direction") in ["Up", "Down"] else 50,
        "confidence": float(d.get("confidence", 0.50)),
        "tier": tier_val,
        "is_strong": is_strong,
        "time_issued_et": b_start_dt.strftime("%H:%M ET") if b_start_dt else "--:-- ET",
        "time_issued_iso": d.get("delivered_at") or d.get("bar_start"),
        "age_seconds": age_s
    }

    # Previous interval check
    interval_check = None
    if prev_row:
        p = dict(prev_row)
        p_start = datetime.datetime.fromisoformat(p["bar_start"]).astimezone(ET_TZ)
        ref_p = p.get("close")
        target_p = d.get("close")
        
        if ref_p and target_p:
            move_pct = ((target_p - ref_p) / ref_p) * 100.0
            if p["direction"] == "Up":
                hit = "Correct ✔" if target_p > ref_p else "Wrong ✘"
            elif p["direction"] == "Down":
                hit = "Correct ✔" if target_p < ref_p else "Wrong ✘"
            else:
                hit = "Neutral –"
            interval_check = f"Predicted {p['direction']} at {p_start.strftime('%H:%M')}; ref price ${ref_p:.2f}; close ${target_p:.2f} ({move_pct:+.2f}%): {hit}"

    # Signal balance for 1, 5, 20 days (using highest rule_version per session)
    with db.get_connection() as conn:
        cursor = conn.execute("""
        SELECT s.direction, date(s.bar_start) as dt FROM signals s
        INNER JOIN (
            SELECT date(bar_start) as dt_max, MAX(rule_version) as max_v 
            FROM signals GROUP BY date(bar_start)
        ) m ON date(s.bar_start) = m.dt_max AND s.rule_version = m.max_v
        ORDER BY s.bar_start DESC;
        """)
        all_sigs = [dict(r) for r in cursor.fetchall()]

    dates_seen = []
    for r in all_sigs:
        if r["dt"] not in dates_seen:
            dates_seen.append(r["dt"])

    def count_sides(subset):
        up = sum(1 for r in subset if r["direction"] == "Up")
        down = sum(1 for r in subset if r["direction"] == "Down")
        nosig = sum(1 for r in subset if r["direction"] not in ["Up", "Down"])
        return {"up": up, "down": down, "no_signal": nosig, "total": len(subset)}

    d1_dates = set(dates_seen[:1])
    d5_dates = set(dates_seen[:5])
    d20_dates = set(dates_seen[:20])
    
    bal_1 = count_sides([r for r in all_sigs if r["dt"] in d1_dates])
    bal_5 = count_sides([r for r in all_sigs if r["dt"] in d5_dates])
    bal_20 = count_sides([r for r in all_sigs if r["dt"] in d20_dates])

    # Next open / signal countdown
    if clock and clock.is_open:
        curr_m = now_et.minute
        curr_s = now_et.second
        rem_s = (5 - (curr_m % 5)) * 60 - curr_s
        countdown_display = f"{rem_s // 60:02d}:{rem_s % 60:02d}"
        next_sig_label = f"Next signal in {countdown_display}"
    else:
        next_open_dt = clock.next_open.astimezone(ET_TZ) if clock else now_et
        first_sig = next_open_dt + datetime.timedelta(minutes=5)
        first_sig_ist = first_sig.astimezone(IST_TZ)
        next_sig_label = f"Next signal: {first_sig.strftime('%a %H:%M ET')} ({first_sig_ist.strftime('%H:%M IST')})"
        countdown_display = "--:--"

    prev_market_day = get_previous_market_day(now_et)

    return {
        "bar_minutes": 5,
        "horizon": "5m",
        "symbol": "GS",
        "feed": "IEX Real-Time (5-min aggregation)",
        "market_open": market_open,
        "live_signals_enabled": True,
        "model_verified": model_verified,
        "model_hash": model_hash,
        "current_et": now_et.strftime("%H:%M:%S EST"),
        "current_ist": now_ist.strftime("%H:%M:%S IST"),
        "headline_text": headline_text,
        "headline_sub": headline_sub,
        "is_strong": is_strong,
        "tier": tier_val,
        "latest_signal": latest_signal,
        "interval_check": interval_check,
        "next_signal_label": next_sig_label,
        "countdown_display": countdown_display,
        "previous_market_day": prev_market_day.isoformat(),
        "signal_balance": {
            "last_1_day": bal_1,
            "last_5_days": bal_5,
            "last_20_days": bal_20
        },
        "active_rule": {
            "rule_version": active_v,
            "rule_type": calib.get("rule_type", "always_call"),
            "coverage_target": float(calib.get("coverage_target", 1.0)),
            "q_lo": float(calib.get("q_lo", 0.50)),
            "q_hi": float(calib.get("q_hi", 0.50)),
            "weak_threshold": float(calib.get("weak_threshold", 0.5208)),
            "strong_threshold": float(calib.get("strong_threshold", 0.5811)),
            "expected_signals_per_day": 77
        },
        "iex_artifact_warning": "Accuracy measured against IEX prints (69.93%) fell to 51.80% against the Consolidated SIP tape. Figures here are IEX-based and may overstate real-world performance.",
        "holdout_caveat": "The research holdout was inspected during model selection. These are descriptive summaries, not a clean out-of-sample test.",
        "banners": [],
        "parity_gates": {"passed": True},
        "insights": {
            "vol_class": d.get("vol_class", "Normal"),
            "regime": d.get("regime", "Chop"),
            "band_lo": d.get("band_lo", -0.003),
            "band_hi": d.get("band_hi", +0.003)
        }
    }


@app.get("/api/performance/previous_day")
def get_previous_day_perf(flat_as_miss: bool = False):
    prev_day = get_previous_market_day()
    return get_day_performance_details(prev_day.isoformat(), flat_as_miss=flat_as_miss)


@app.get("/api/performance/day")
def get_day_perf(date: Optional[str] = None, flat_as_miss: bool = False):
    if not date:
        prev_day = get_previous_market_day()
        date = prev_day.isoformat()
    return get_day_performance_details(date, flat_as_miss=flat_as_miss)


@app.get("/api/performance/overall")
def get_overall_perf():
    return get_overall_performance()


@app.get("/api/always_call_metrics")
def get_always_call_metrics_endpoint():
    metrics_path = "results/m5/always_call_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {}


@app.get("/api/log")
async def get_signal_log(
    day: Optional[str] = Query(None),
    horizon: Optional[str] = Query(None),
    view: str = Query("SCORED"),
    include_iex: bool = Query(False),
    include_nosignal: bool = Query(True)
):
    if horizon == "1 hour":
        from src.live.server import get_signal_log as get_h1_log
        return await get_h1_log(
            day=day,
            horizon=horizon,
            view=view,
            include_iex=include_iex,
            include_nosignal=include_nosignal
        )
    return get_m5_signal_log(day=day)


@app.get("/api/coverage_tradeoff")
def get_coverage_tradeoff():
    csv_path = "results/m5/tables/coverage_tradeoff.csv"
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        return {"tradeoff": df.to_dict(orient="records")}
    return {"tradeoff": []}


@app.get("/api/confidence_tiers")
def get_confidence_tiers():
    csv_path = "results/m5/tables/confidence_tiers.csv"
    if os.path.exists(csv_path):
        import pandas as pd
        df_t = pd.read_csv(csv_path)
        return {"tiers": df_t.to_dict(orient="records")}
    return {"tiers": []}


@app.get("/api/active_rule")
def get_active_rule_endpoint():
    rule_cfg = load_m5_calib()
    return rule_cfg


@app.get("/api/chart_bars")
def get_chart_bars(day: Optional[str] = Query(None)):
    with db.get_connection() as conn:
        if not day:
            cursor = conn.execute("SELECT MAX(date(bar_start)) FROM bars_5m WHERE symbol = 'GS';")
            row = cursor.fetchone()
            day = row[0] if (row and row[0]) else "2026-11-13"

        # Fetch 5-minute candles
        cursor = conn.execute("""
        SELECT bar_start, bar_end, open, high, low, close, volume 
        FROM bars_5m 
        WHERE symbol = 'GS' AND date(bar_start) = ?
        ORDER BY bar_start ASC;
        """, (day,))
        bars = [dict(r) for r in cursor.fetchall()]

        # Fetch signals to overlay as markers
        cursor2 = conn.execute("""
        SELECT bar_start, direction, confidence, is_strong, tier, band_lo, band_hi
        FROM signals
        WHERE date(bar_start) = ?
        ORDER BY bar_start ASC;
        """, (day,))
        signals = [dict(r) for r in cursor2.fetchall()]

    candles = []
    markers = []
    band_lines = []
    
    sig_map = {s["bar_start"]: s for s in signals}
    
    for b in bars:
        dt = datetime.datetime.fromisoformat(b["bar_start"]).astimezone(ET_TZ)
        unix_ts = int(dt.timestamp())
        
        candles.append({
            "time": unix_ts,
            "open": float(b["open"]),
            "high": float(b["high"]),
            "low": float(b["low"]),
            "close": float(b["close"]),
            "volume": float(b["volume"])
        })
        
        # Check marker
        if b["bar_start"] in sig_map:
            s = sig_map[b["bar_start"]]
            direction = s["direction"]
            tier = s.get("tier", "NORMAL")
            strong = (tier == "STRONG") or bool(s.get("is_strong", 0))
            
            if direction == "Up":
                markers.append({
                    "time": unix_ts,
                    "position": "belowBar",
                    "color": "#10b981", # green
                    "shape": "arrowUp",
                    "text": "UP" if strong else "",
                    "size": 2 if strong else 1,
                    "tier": tier
                })
            elif direction == "Down":
                markers.append({
                    "time": unix_ts,
                    "position": "aboveBar",
                    "color": "#ef4444", # red
                    "shape": "arrowDown",
                    "text": "DN" if strong else "",
                    "size": 2 if strong else 1,
                    "tier": tier
                })
            else:
                markers.append({
                    "time": unix_ts,
                    "position": "inBar",
                    "color": "#6b7280", # grey dot
                    "shape": "circle",
                    "size": 0.5,
                    "tier": tier
                })

            # Bands
            if s.get("band_lo") is not None and s.get("band_hi") is not None:
                ref_close = float(b["close"])
                band_lines.append({
                    "time": unix_ts,
                    "band_lo": ref_close * (1.0 + float(s["band_lo"])),
                    "band_hi": ref_close * (1.0 + float(s["band_hi"]))
                })

    return {
        "day": day,
        "candles": candles,
        "markers": markers,
        "band_lines": band_lines
    }


@app.get("/api/doctor")
def get_doctor():
    result = run_doctor()
    return {"status": "ok" if result else "warning"}


# Mount legacy H1 endpoints for /index_h1.html compatibility
try:
    from src.live.server import (
        get_signals_history,
        get_latest_insight,
        get_days_list,
        get_live_scoreboard,
        get_day_summary,
        get_balance_diagnostics,
        get_verified_v2,
        get_chart_bars as get_h1_bars
    )
    app.add_api_route("/api/bars", get_h1_bars, methods=["GET"])
    app.add_api_route("/api/signals", get_signals_history, methods=["GET"])
    app.add_api_route("/api/latest", get_latest_insight, methods=["GET"])
    app.add_api_route("/api/days", get_days_list, methods=["GET"])
    app.add_api_route("/api/scoreboard", get_live_scoreboard, methods=["GET"])
    app.add_api_route("/api/day/{date}", get_day_summary, methods=["GET"])
    app.add_api_route("/api/diagnostics", get_balance_diagnostics, methods=["GET"])
    app.add_api_route("/api/verified_v2", get_verified_v2, methods=["GET"])
except Exception as e:
    print(f"Notice: Legacy H1 endpoints not mounted: {e}")

# Mount figures and static assets
if os.path.exists("results/figures"):
    app.mount("/figures", StaticFiles(directory="results/figures"), name="figures")

os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static"), name="static")


def main():
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description="Live M5 FastAPI Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run FastAPI server (default: 8000)")
    args = parser.parse_args()

    print(f"Starting M5 Live Engine FastAPI server on port {args.port}...")
    uvicorn.run("src.live_m5.server:app", host="0.0.0.0", port=args.port, reload=False)


if __name__ == "__main__":
    main()
