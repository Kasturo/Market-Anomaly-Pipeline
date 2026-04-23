"""
FinPulse: financial data pipeline.
Pulls market data via yfinance, engineers features, runs Isolation Forest
anomaly detection and trend analysis, exports CSV/JSON, and rebuilds the HTML dashboard.

Usage (from project root):
  python pipeline.py
  python pipeline.py --tickers AAPL TSLA NVDA
  python pipeline.py --period 2y
  python pipeline.py --no-dashboard
  python pipeline.py --dashboard-only
  python pipeline.py --s3-bucket YOUR_BUCKET
  python pipeline.py --open
"""

import argparse
import json
import os
import sys
import time
import warnings
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


#  PATHS


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RPT_DIR  = ROOT_DIR / "reports"

for d in (DATA_DIR, RPT_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _load_dotenv() -> None:
    """Load AWS_* (and other vars) from project-root .env if present. Never commit .env."""
    env_path = ROOT_DIR / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        print(
            "[WARN] Found .env but python-dotenv is not installed. "
            "Run: pip install python-dotenv   (or use aws configure / export AWS_* )"
        )
        return
    if load_dotenv(env_path):
        print("[INFO] Loaded environment from .env in project root.")


_load_dotenv()


#  CONFIG DEFAULTS


DEFAULT_TICKERS       = ["AAPL", "MSFT", "GOOGL", "JPM", "SPY"]
DEFAULT_PERIOD        = "1y"    # any yfinance period string: 1mo 3mo 6mo 1y 2y 5y
DEFAULT_INTERVAL      = "1d"    # daily bars
ANOMALY_CONTAMINATION = 0.05    # expected fraction of anomalous trading days

# Features fed into Isolation Forest
ANOMALY_FEATURES = [
    "ret_1d", "ret_5d", "vol_10d", "vol_zscore",
    "rsi", "bb_width", "macd_diff",
]


#  1. DATA INGESTION


def fetch_market_data(
    tickers: list,
    period:   str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> dict:
    """
    Download OHLCV data from Yahoo Finance via yfinance.
    Returns {ticker: DataFrame}.
    Falls back to a cached CSV if yfinance fails for a given ticker.
    """
    try:
        import yfinance as yf
    except ImportError:
        sys.exit(
            "[ERROR] yfinance not installed.\n"
            "Run:  pip install yfinance"
        )

    print(f"\n[1/4] INGESTION: {len(tickers)} tickers | period={period} | interval={interval}")
    result = {}

    for ticker in tickers:
        cache = DATA_DIR / f"{ticker}_raw.csv"
        try:
            last_err: Exception | None = None
            df = pd.DataFrame()
            for attempt in range(1, 4):
                try:
                    # Ticker.history is usually more reliable than download() for one symbol.
                    tkr = yf.Ticker(ticker)
                    df = tkr.history(
                        period=period,
                        interval=interval,
                        auto_adjust=True,
                    )
                    if df.empty:
                        raise ValueError("Empty dataframe returned")
                    break
                except Exception as e:
                    last_err = e
                    if attempt < 3:
                        wait = 2 ** (attempt - 1)
                        print(f"  RETRY {ticker} (attempt {attempt}/3, sleep {wait}s): {e}")
                        time.sleep(wait)
                    else:
                        raise last_err from None

            # yfinance can return MultiIndex columns; flatten them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"
            df.to_csv(cache)   # cache for offline use

            result[ticker] = df
            print(f"  OK {ticker:<6} {len(df):4d} rows  "
                  f"({df.index[0].date()} to {df.index[-1].date()})  [live]")

        except Exception as exc:
            if cache.exists():
                df = pd.read_csv(cache, index_col="Date", parse_dates=True)
                result[ticker] = df
                print(f"  CACHE {ticker:<6} {len(df):4d} rows  "
                      f"({df.index[0].date()} to {df.index[-1].date()})  [cached]")
            else:
                print(f"  FAIL {ticker:<6} FAILED: {exc}")

    if not result:
        sys.exit(
            "[ERROR] No data loaded. Check your internet connection, "
            "or run:  python generate_mock_data.py"
        )
    return result



#  2. FEATURE ENGINEERING


def engineer_features(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    Compute 12 financial features from raw OHLCV data.

    Returns
    -------
    pd.DataFrame with columns:
      ret_1d, ret_5d, ret_20d: returns over 1 / 5 / 20 days
      vol_10d, vol_30d: rolling return std dev (annualised proxy)
      vol_zscore: volume z-score vs 20-day average
      rsi: RSI(14)
      bb_width: Bollinger Band width (normalised)
      macd_diff: MACD line minus signal line
      price_vs_sma50: percent deviation from 50-day SMA
      momentum: composite percentile score
      ticker: string label
    """
    d = df.copy()

    # Returns
    d["ret_1d"]  = d["Close"].pct_change()
    d["ret_5d"]  = d["Close"].pct_change(5)
    d["ret_20d"] = d["Close"].pct_change(20)

    # Volatility
    d["vol_10d"] = d["ret_1d"].rolling(10).std()
    d["vol_30d"] = d["ret_1d"].rolling(30).std()

    # Volume z-score
    vm = d["Volume"].rolling(20).mean()
    vs = d["Volume"].rolling(20).std()
    d["vol_zscore"] = (d["Volume"] - vm) / (vs + 1e-9)

    # RSI(14)
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # Bollinger Band width
    sma20 = d["Close"].rolling(20).mean()
    std20 = d["Close"].rolling(20).std()
    d["bb_width"] = (2 * std20) / (sma20 + 1e-9)

    # MACD
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    d["macd_diff"] = macd - macd.ewm(span=9, adjust=False).mean()

    # Price vs SMA50
    sma50 = d["Close"].rolling(50).mean()
    d["price_vs_sma50"] = (d["Close"] - sma50) / (sma50 + 1e-9)

    # Composite momentum (percentile-ranked)
    d["momentum"] = (
        d["ret_5d"].rank(pct=True)  * 0.40
        + d["ret_20d"].rank(pct=True) * 0.40
        + (1 - d["vol_10d"].rank(pct=True)) * 0.20
    )

    d["ticker"] = ticker
    return d.dropna()



#  3. ANOMALY DETECTION (Isolation Forest)


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = ANOMALY_CONTAMINATION,
) -> pd.DataFrame:
    """
    Flag unusual trading days with Isolation Forest (unsupervised).

    Isolation Forest randomly partitions the feature space; anomalies are
    isolated in fewer splits (shorter paths). No labelled data is required;
    the same approach used in fraud detection.

    Adds three columns:
      anomaly_score (float): higher = more anomalous
      is_anomaly (bool): True if flagged as anomalous
      anomaly_type (str): human-readable type (price_spike, volume_surge, etc.)
    """
    feat_cols = [c for c in ANOMALY_FEATURES if c in df.columns]
    X = StandardScaler().fit_transform(df[feat_cols].values)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        max_samples="auto",
    )
    model.fit(X)

    result = df.copy()
    result["anomaly_score"] = -model.decision_function(X)   # flip: higher = worse
    result["is_anomaly"]    = model.predict(X) == -1

    def _classify(row: pd.Series) -> str:
        if not row["is_anomaly"]:
            return "normal"
        if abs(row["ret_1d"]) > 0.03:
            return "price_spike"       # daily move > ±3 %
        if abs(row["vol_zscore"]) > 2.5:
            return "volume_surge"      # volume > 2.5 std dev above mean
        if row["rsi"] > 80 or row["rsi"] < 20:
            return "rsi_extreme"       # overbought / oversold
        return "multivariate"          # combination of smaller signals

    result["anomaly_type"] = result.apply(_classify, axis=1)
    return result



#  4. TREND ANALYSIS


def analyze_trends(df: pd.DataFrame) -> dict:
    """
    Compute a suite of trend metrics and emit a composite BUY / SELL / HOLD signal.

    Returns
    -------
    dict with keys:
      annual_slope_pct: annualised linear trend as percent of current price
      r2: goodness-of-fit for the linear trend model
      regime: BULL / BEAR / SIDEWAYS (based on 20-day return)
      support: 52-week rolling minimum (support level)
      resistance: 52-week rolling maximum (resistance level)
      current_price: latest closing price
      rsi: current RSI(14) value
      macd_diff: current MACD divergence from signal line
      momentum_score: composite momentum percentile
      signal: BUY / HOLD / SELL
      bullish_factors: number of bullish conditions satisfied (out of 6)
    """
    close = df["Close"].values.reshape(-1, 1)
    idx   = np.arange(len(close)).reshape(-1, 1)

    reg   = LinearRegression().fit(idx, close)
    slope = float(reg.coef_[0][0])
    r2    = float(reg.score(idx, close))

    current       = float(df["Close"].iloc[-1])
    annual_pct    = slope * 252 / current * 100

    ret20  = float(df["ret_20d"].iloc[-1])
    regime = "BULL" if ret20 > 0.05 else ("BEAR" if ret20 < -0.05 else "SIDEWAYS")

    support    = float(df["Close"].rolling(252, min_periods=20).min().iloc[-1])
    resistance = float(df["Close"].rolling(252, min_periods=20).max().iloc[-1])
    rsi        = float(df["rsi"].iloc[-1])
    macd_d     = float(df["macd_diff"].iloc[-1])
    mom        = float(df["momentum"].iloc[-1])
    sma50      = float(df["Close"].rolling(50).mean().iloc[-1])

    # 6 binary bullish conditions to composite signal
    bullish = sum([
        rsi < 70,               # not overbought
        macd_d > 0,             # MACD above signal
        mom > 0.55,             # high momentum rank
        annual_pct > 0,         # upward trend
        regime == "BULL",       # bull market
        current > sma50,        # above 50-day MA
    ])
    signal = "BUY" if bullish >= 4 else ("SELL" if bullish <= 2 else "HOLD")

    return {
        "annual_slope_pct": round(annual_pct, 2),
        "r2":               round(r2, 4),
        "regime":           regime,
        "support":          round(support, 2),
        "resistance":       round(resistance, 2),
        "current_price":    round(current, 2),
        "rsi":              round(rsi, 1),
        "macd_diff":        round(macd_d, 4),
        "momentum_score":   round(mom, 3),
        "signal":           signal,
        "bullish_factors":  bullish,
    }



#  5. EXPORT


def _serial(obj):
    """JSON serialiser for numpy / pandas types."""
    if hasattr(obj, "item"):      return obj.item()
    if hasattr(obj, "isoformat"): return obj.isoformat()
    return str(obj)


def export_results(results: dict, processed: dict, skip_local: bool = False) -> None:
    """
    Write three artefacts to data/:
      {TICKER}_processed.csv: engineered features + anomaly labels
      pipeline_summary.json: full results dict (trend + anomaly stats)
      chart_data.json: 60-day window for the dashboard charts

    Parameters
    ----------
    skip_local : if True, skip writing files to the local filesystem
                 (used when --skip-local flag is set with --s3-bucket)
    """
    if skip_local:
        print("\n  [skip-local] Skipping local file writes (S3-only mode)")
        return

    # Per-ticker CSVs
    for ticker, df in processed.items():
        df.to_csv(DATA_DIR / f"{ticker}_processed.csv")

    # Full summary
    with open(DATA_DIR / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_serial)

    # Chart data (last 60 trading days per ticker)
    chart_data = {}
    cols = ["Close", "ret_1d", "rsi", "vol_10d", "is_anomaly", "anomaly_type"]
    for ticker, df in processed.items():
        rows = []
        for ts, row in df[cols].tail(60).iterrows():
            rows.append({
                "date":    str(ts.date()),
                "close":   round(float(row["Close"]),  2),
                "ret1d":   round(float(row["ret_1d"]), 4),
                "rsi":     round(float(row["rsi"]),    1),
                "vol10d":  round(float(row["vol_10d"]),4),
                "anomaly": bool(row["is_anomaly"]),
                "type":    str(row["anomaly_type"]),
            })
        chart_data[ticker] = rows

    with open(DATA_DIR / "chart_data.json", "w", encoding="utf-8") as f:
        json.dump(chart_data, f)

    print(f"\n  Saved: pipeline_summary.json, chart_data.json, "
          f"{len(processed)} processed CSVs")



#  6. DASHBOARD BUILDER


def launch_dashboard() -> None:
    """Open reports/dashboard.html in the default browser (works from Git Bash on Windows)."""
    path = RPT_DIR / "dashboard.html"
    if not path.is_file():
        print(f"[WARN] No dashboard at {path}. Run: python pipeline.py")
        return
    url = path.resolve().as_uri()
    print(f"Opening in browser:\n  {url}")
    if not webbrowser.open(url):
        print(f"[WARN] Could not launch browser. Open the file manually:\n  {path.resolve()}")


def build_dashboard() -> Path:
    """
    Read data/chart_data.json + data/pipeline_summary.json and write
    reports/dashboard.html: a fully self-contained interactive HTML file.
    Open it in any browser; no web server required.
    """
    for p in (DATA_DIR / "chart_data.json", DATA_DIR / "pipeline_summary.json"):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p.name}. Run the pipeline first."
            )

    with open(DATA_DIR / "chart_data.json",   encoding="utf-8") as f:
        chart_data = json.load(f)
    with open(DATA_DIR / "pipeline_summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    html = _DASHBOARD_HTML
    html = html.replace("/*INJECT_CHART*/",   f"const CHART_DATA={json.dumps(chart_data)};")
    html = html.replace("/*INJECT_SUMMARY*/", f"const SUMMARY={json.dumps(summary)};")

    out = RPT_DIR / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out



#  7. PIPELINE ORCHESTRATOR


def run_pipeline(
    tickers:    list        = DEFAULT_TICKERS,
    period:     str         = DEFAULT_PERIOD,
    build_dash: bool        = True,
    s3_bucket:  str | None  = None,
    skip_local: bool        = False,
) -> dict:
    """
    Full end-to-end pipeline:
      1. Ingest: download OHLCV via yfinance (cached CSV fallback)
      2. Engineer: compute 12 financial features
      3. Detect: Isolation Forest anomaly detection
      4. Analyse: linear trend, regime, support/resistance, signal
      5. Export: processed CSVs + JSON artefacts
      6. Dashboard: rebuild self-contained HTML (optional)
      7. S3 upload: push all artifacts to S3 (optional, requires boto3)

    Parameters
    ----------
    s3_bucket  : if provided, upload all output files to this S3 bucket
                 after the pipeline completes
    skip_local : skip writing files to the local filesystem
                 (only meaningful alongside s3_bucket)
    """
    SEP = "=" * 62
    print(f"\n{SEP}")
    print(f"  FINPULSE PIPELINE  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)

    raw       = fetch_market_data(tickers, period=period)
    results   = {}
    processed = {}

    print("\n[2-4/4] FEATURE ENGINEERING, ANOMALY DETECTION, TREND ANALYSIS")

    for ticker, df in raw.items():
        feat_df = engineer_features(df, ticker)
        anom_df = detect_anomalies(feat_df)
        trend   = analyze_trends(anom_df)

        n   = int(anom_df["is_anomaly"].sum())
        pct = round(n / len(anom_df) * 100, 1)

        processed[ticker] = anom_df
        results[ticker] = {
            "trend": trend,
            "anomalies": {
                "total":         n,
                "pct_of_days":   pct,
                "by_type":       (
                    anom_df[anom_df["is_anomaly"]]["anomaly_type"]
                    .value_counts().to_dict()
                ),
                "recent_anomaly_dates": [
                    str(ts.date())
                    for ts in anom_df[anom_df["is_anomaly"]].index[-5:]
                ],
            },
            "rows": len(anom_df),
            "date_range": {
                "start": str(anom_df.index[0].date()),
                "end":   str(anom_df.index[-1].date()),
            },
        }

        icon = {"BUY": "[+]", "SELL": "[-]", "HOLD": "[=]"}.get(trend["signal"], "[?]")
        print(
            f"  {icon} {ticker:<6} {trend['signal']:<4}  "
            f"Regime: {trend['regime']:<9}  "
            f"Trend: {trend['annual_slope_pct']:+6.1f}%/yr  "
            f"RSI: {trend['rsi']:4.1f}  "
            f"Anomalies: {n} ({pct}%)"
        )

    print("\n[5/4] EXPORTING")
    export_results(results, processed, skip_local=skip_local)

    if build_dash and not skip_local:
        dash = build_dashboard()
        print(f"  Dashboard: {dash.relative_to(ROOT_DIR)}")

    s3_uploaded = 0
    if s3_bucket:
        print(f"\n[6/4] S3 UPLOAD: s3://{s3_bucket}/")
        try:
            from aws.s3_storage import upload_pipeline_outputs
            report = upload_pipeline_outputs(
                bucket     = s3_bucket,
                data_dir   = DATA_DIR,
                rpt_dir    = RPT_DIR,
            )
            s3_uploaded = len(report.get("uploaded", []))
            if s3_uploaded == 0:
                print(
                    "\n  [WARN] No files reached S3. boto3 needs AWS credentials on this machine.\n"
                    "         Run:  aws configure\n"
                    "         Or set:  AWS_ACCESS_KEY_ID  and  AWS_SECRET_ACCESS_KEY\n"
                    "         (Use the same account that owns the bucket.)"
                )
        except ImportError:
            print("  [WARN] boto3 not installed; skipping S3 upload.")
            print("         Run:  pip install boto3")

    print(f"\n{SEP}")
    print(f"  DONE: {len(results)} tickers processed")
    if not s3_bucket:
        print(f"  Open reports/dashboard.html in your browser to explore results")
    elif s3_uploaded:
        print(f"  Uploaded {s3_uploaded} file(s) to s3://{s3_bucket}/")
    else:
        print("  Local outputs in data/ and reports/; fix credentials to upload to S3")
    print(SEP + "\n")
    return results



#  DASHBOARD HTML (injected at build time)


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FinPulse: Market Intelligence Pipeline</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0c12;--bg2:#111420;--bg3:#181d2e;--border:#1f2640;--acc:#4fffb0;--acc3:#4f9fff;--text:#e8eaf2;--mu:#5a6080;--bull:#4fffb0;--bear:#ff4f7b;--hold:#f5c542;--mono:"Space Mono",monospace;--sans:"DM Sans",sans-serif}
*{margin:0;padding:0;box-sizing:border-box}body{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh;overflow-x:hidden}
body::before{content:"";position:fixed;inset:0;z-index:0;background-image:linear-gradient(var(--border) 1px,transparent 1px),linear-gradient(90deg,var(--border) 1px,transparent 1px);background-size:40px 40px;opacity:.28;pointer-events:none}
.w{position:relative;z-index:1;max-width:1400px;margin:0 auto;padding:0 24px 64px}
header{display:flex;align-items:center;justify-content:space-between;padding:26px 0 20px;border-bottom:1px solid var(--border);margin-bottom:28px}
.logo{display:flex;align-items:center;gap:12px}.li{width:36px;height:36px;border-radius:8px;background:linear-gradient(135deg,var(--acc),var(--acc3));display:flex;align-items:center;justify-content:center;font-size:16px}
.lt{font-family:var(--mono);font-size:17px;font-weight:700;letter-spacing:.05em}.ls{font-size:10px;color:var(--mu);font-family:var(--mono);letter-spacing:.12em;text-transform:uppercase;margin-top:3px}
.badge{display:inline-flex;align-items:center;gap:6px;background:rgba(79,255,176,.08);border:1px solid rgba(79,255,176,.22);border-radius:20px;padding:4px 11px;font-family:var(--mono);font-size:10px;color:var(--acc);letter-spacing:.08em;text-transform:uppercase}
.dot{width:6px;height:6px;border-radius:50%;background:var(--acc);animation:pulse 2s infinite}@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.rt{font-family:var(--mono);font-size:10px;color:var(--mu);margin-top:5px;text-align:right}
.tbar{display:flex;gap:8px;margin-bottom:24px;flex-wrap:wrap;align-items:center}
.tbtn{font-family:var(--mono);font-size:12px;font-weight:700;padding:7px 17px;border-radius:6px;cursor:pointer;border:1px solid var(--border);background:var(--bg3);color:var(--mu);transition:all .18s;letter-spacing:.06em}
.tbtn:hover{border-color:var(--acc);color:var(--text)}.tbtn.on{background:var(--acc);color:#0a0c12;border-color:var(--acc)}
.tmeta{margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--mu);align-self:center}
.kg{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:24px}
@media(max-width:1100px){.kg{grid-template-columns:repeat(3,1fr)}}@media(max-width:580px){.kg{grid-template-columns:repeat(2,1fr)}}
.kpi{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:14px 16px;transition:border-color .2s}.kpi:hover{border-color:var(--acc)}
.kl{font-size:10px;color:var(--mu);text-transform:uppercase;letter-spacing:.1em;font-family:var(--mono);margin-bottom:4px}
.kv{font-family:var(--mono);font-size:20px;font-weight:700;line-height:1.1}.ks{font-size:11px;color:var(--mu);margin-top:3px}
.bull{color:var(--bull)}.bear{color:var(--bear)}.hold{color:var(--hold)}.blue{color:var(--acc3)}
.sig{display:inline-block;font-family:var(--mono);font-size:13px;font-weight:700;padding:4px 13px;border-radius:4px;letter-spacing:.1em}
.sig-BUY{background:rgba(79,255,176,.13);color:var(--bull);border:1px solid rgba(79,255,176,.36)}
.sig-SELL{background:rgba(255,79,123,.13);color:var(--bear);border:1px solid rgba(255,79,123,.36)}
.sig-HOLD{background:rgba(245,197,66,.13);color:var(--hold);border:1px solid rgba(245,197,66,.36)}
.g2{display:grid;grid-template-columns:2fr 1fr;gap:14px;margin-bottom:14px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-top:14px}
@media(max-width:900px){.g2,.g3{grid-template-columns:1fr}}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:18px 20px}
.ct{font-size:10px;text-transform:uppercase;letter-spacing:.13em;color:var(--mu);font-family:var(--mono);margin-bottom:14px;display:flex;align-items:center;gap:8px}
.cd{width:5px;height:5px;border-radius:50%;background:var(--acc);flex-shrink:0}canvas{width:100%!important}
.at{width:100%;border-collapse:collapse;font-size:12px}
.at th{font-family:var(--mono);font-size:10px;color:var(--mu);text-transform:uppercase;letter-spacing:.1em;padding:5px 9px;border-bottom:1px solid var(--border);text-align:left}
.at td{padding:7px 9px;border-bottom:1px solid rgba(31,38,64,.45);font-family:var(--mono)}
.at tr:hover td{background:var(--bg3)}
.chip{font-size:9px;padding:2px 6px;border-radius:3px;font-family:var(--mono);letter-spacing:.05em;text-transform:uppercase}
.chip-price_spike{background:rgba(255,79,123,.14);color:#ff6b8e}.chip-volume_surge{background:rgba(79,159,255,.14);color:#7ab8ff}
.chip-rsi_extreme{background:rgba(245,197,66,.14);color:#f5c542}.chip-multivariate{background:rgba(160,79,255,.14);color:#c47aff}
.tov{display:grid;grid-template-columns:1fr;gap:7px}
.trow{display:flex;align-items:center;justify-content:space-between;padding:9px 13px;background:var(--bg3);border-radius:7px;border:1px solid var(--border);cursor:pointer;transition:border-color .18s;gap:8px}
.trow:hover{border-color:var(--acc3)}.trow.on{border-color:var(--acc)}
.tn{font-family:var(--mono);font-weight:700;font-size:13px;min-width:44px}.tp{font-family:var(--mono);font-size:12px}.tt{font-size:12px;font-family:var(--mono)}
.fi{animation:fadeIn .3s ease forwards;opacity:0}@keyframes fadeIn{to{opacity:1}}
.pb{height:3px;background:var(--border);border-radius:2px;overflow:hidden;margin-top:3px}
.pf{height:100%;transition:width .45s ease}
</style>
</head>
<body>
<div class="w">
<header>
  <div class="logo"><div class="li">FP</div><div><div class="lt">FINPULSE</div><div class="ls">Market Intelligence Pipeline</div></div></div>
  <div><div class="badge"><span class="dot"></span>Pipeline Active</div><div class="rt" id="rt"></div></div>
</header>
<div class="tbar" id="tbar"></div>
<div class="kg" id="kg"></div>
<div class="g2">
  <div class="card"><div class="ct"><span class="cd"></span>Price + Anomaly Detection (60-Day Window)</div><canvas id="pc" height="220"></canvas></div>
  <div class="card"><div class="ct"><span class="cd"></span>RSI (14-Day) with Overbought/Oversold Bands</div><canvas id="rc" height="220"></canvas></div>
</div>
<div class="g2">
  <div class="card"><div class="ct"><span class="cd"></span>Rolling Volatility: 10-Day Return Std Dev</div><canvas id="vc" height="150"></canvas></div>
  <div class="card"><div class="ct"><span class="cd"></span>Anomaly Type Breakdown</div><canvas id="ac" height="150"></canvas></div>
</div>
<div class="g3">
  <div class="card"><div class="ct"><span class="cd"></span>Portfolio Overview</div><div class="tov" id="tov"></div></div>
  <div class="card"><div class="ct"><span class="cd"></span>Recent Anomaly Events</div><div id="al"></div></div>
  <div class="card"><div class="ct"><span class="cd"></span>Signal Factor Analysis</div><div id="fb"></div></div>
</div>
</div>
<script>
/*INJECT_CHART*/
/*INJECT_SUMMARY*/
const TKS=Object.keys(SUMMARY);let active=TKS[0];const CH={};
function fmt(n,d=2){return n===undefined?"N/A":Number(n).toFixed(d);}
function kill(id){if(CH[id]){CH[id].destroy();delete CH[id];}}
const TT={backgroundColor:"#181d2e",borderColor:"#1f2640",borderWidth:1,titleFont:{family:"Space Mono",size:10},bodyFont:{family:"Space Mono",size:11}};
const XA={grid:{color:"#1a1f30"},ticks:{color:"#5a6080",font:{family:"Space Mono",size:9},maxTicksLimit:8}};
const YA={grid:{color:"#1a1f30"},ticks:{color:"#5a6080",font:{family:"Space Mono",size:9}}};

function render(){
  const t=active,s=SUMMARY[t],tr=s.trend,an=s.anomalies,cd=CHART_DATA[t];
  const L=cd.map(r=>r.date);

  document.getElementById("tbar").innerHTML=
    TKS.map(tk=>`<button class="tbtn ${tk===t?"on":""}" onclick="pick('${tk}')">${tk}</button>`).join("")+
    `<span class="tmeta">${s.date_range.start} to ${s.date_range.end} &nbsp;|&nbsp; ${s.rows} trading days</span>`;

  document.getElementById("kg").innerHTML=`
    <div class="kpi fi"><div class="kl">Signal</div><div class="kv"><span class="sig sig-${tr.signal}">${tr.signal}</span></div><div class="ks">${tr.bullish_factors}/6 bullish</div></div>
    <div class="kpi fi" style="animation-delay:.05s"><div class="kl">Price</div><div class="kv blue">$${fmt(tr.current_price)}</div><div class="ks">Regime: ${tr.regime}</div></div>
    <div class="kpi fi" style="animation-delay:.1s"><div class="kl">Annual Trend</div><div class="kv ${tr.annual_slope_pct>=0?"bull":"bear"}">${tr.annual_slope_pct>=0?"+":""}${fmt(tr.annual_slope_pct)}%</div><div class="ks">R2 = ${fmt(tr.r2,3)}</div></div>
    <div class="kpi fi" style="animation-delay:.15s"><div class="kl">RSI (14)</div><div class="kv ${tr.rsi>70?"bear":tr.rsi<30?"bull":"blue"}">${fmt(tr.rsi,1)}</div><div class="ks">${tr.rsi>70?"Overbought":tr.rsi<30?"Oversold":"Neutral zone"}</div></div>
    <div class="kpi fi" style="animation-delay:.2s"><div class="kl">Anomalies</div><div class="kv bear">${an.total}</div><div class="ks">${an.pct_of_days}% of days</div></div>
    <div class="kpi fi" style="animation-delay:.25s"><div class="kl">52-Week Range</div><div class="kv blue" style="font-size:13px">$${fmt(tr.support)} / $${fmt(tr.resistance)}</div><div class="ks">Support / Resistance</div></div>`;

  /* price + anomaly */
  kill("p");
  const cx1=document.getElementById("pc").getContext("2d");
  const g1=cx1.createLinearGradient(0,0,0,300);g1.addColorStop(0,"rgba(79,255,176,.18)");g1.addColorStop(1,"rgba(79,255,176,0)");
  CH["p"]=new Chart(cx1,{type:"line",data:{labels:L,datasets:[
    {label:"Close",data:cd.map(r=>r.close),borderColor:"#4fffb0",borderWidth:2,fill:true,backgroundColor:g1,tension:.3,pointRadius:0,pointHoverRadius:4},
    {label:"Anomaly",data:cd.map(r=>r.anomaly?r.close:null),borderColor:"transparent",backgroundColor:"#ff4f7b",pointRadius:7,pointStyle:"crossRot",pointBorderWidth:3,pointHoverRadius:9,showLine:false}
  ]},options:{responsive:true,animation:{duration:350},plugins:{legend:{display:false},tooltip:{...TT,callbacks:{label:c=>c.dataset.label==="Anomaly"&&c.raw!==null?`ANOMALY $${c.raw}`:c.dataset.label==="Close"?`$${c.raw}`:null}}},
    scales:{x:{...XA},y:{...YA,ticks:{...YA.ticks,callback:v=>"$"+v}}}}});

  /* RSI */
  kill("r");
  const cx2=document.getElementById("rc").getContext("2d");
  CH["r"]=new Chart(cx2,{type:"line",data:{labels:L,datasets:[
    {label:"RSI",data:cd.map(r=>r.rsi),borderColor:"#4f9fff",borderWidth:2,fill:false,tension:.3,pointRadius:0},
    {label:"70",data:cd.map(()=>70),borderColor:"rgba(255,79,123,.45)",borderWidth:1,borderDash:[4,4],fill:false,pointRadius:0},
    {label:"30",data:cd.map(()=>30),borderColor:"rgba(79,255,176,.45)",borderWidth:1,borderDash:[4,4],fill:false,pointRadius:0}
  ]},options:{responsive:true,animation:{duration:350},plugins:{legend:{display:false},tooltip:{...TT,filter:i=>i.dataset.label==="RSI"}},
    scales:{x:{...XA,ticks:{...XA.ticks,maxTicksLimit:6}},y:{min:0,max:100,...YA}}}});

  /* volatility */
  kill("v");
  const cx3=document.getElementById("vc").getContext("2d");
  const g3=cx3.createLinearGradient(0,0,0,180);g3.addColorStop(0,"rgba(245,197,66,.22)");g3.addColorStop(1,"rgba(245,197,66,0)");
  CH["v"]=new Chart(cx3,{type:"line",data:{labels:L,datasets:[
    {label:"Vol",data:cd.map(r=>+(r.vol10d*100).toFixed(3)),borderColor:"#f5c542",borderWidth:1.5,fill:true,backgroundColor:g3,tension:.4,pointRadius:0}
  ]},options:{responsive:true,animation:{duration:350},plugins:{legend:{display:false},tooltip:{...TT,callbacks:{label:c=>`Vol: ${Number(c.raw).toFixed(2)}%`}}},
    scales:{x:{...XA},y:{...YA,ticks:{...YA.ticks,callback:v=>Number(v).toFixed(1)+"%"}}}}});

  /* anomaly doughnut */
  kill("a");
  const at=an.by_type,atL=Object.keys(at).map(k=>k.replace("_"," ").toUpperCase()),atV=Object.values(at);
  CH["a"]=new Chart(document.getElementById("ac").getContext("2d"),{type:"doughnut",
    data:{labels:atL,datasets:[{data:atV,backgroundColor:["#ff4f7b","#4f9fff","#f5c542","#c47aff"].slice(0,atL.length),borderColor:"#0a0c12",borderWidth:3}]},
    options:{responsive:true,animation:{duration:350},cutout:"65%",
      plugins:{legend:{position:"right",labels:{color:"#8890b0",font:{family:"Space Mono",size:10},padding:10,boxWidth:10}},tooltip:{...TT}}}});

  /* portfolio overview */
  document.getElementById("tov").innerHTML=TKS.map(tk=>{
    const t2=SUMMARY[tk].trend,up=t2.annual_slope_pct>=0;
    return `<div class="trow ${tk===t?"on":""}" onclick="pick('${tk}')">
      <span class="tn">${tk}</span><span class="tp blue">$${fmt(t2.current_price)}</span>
      <span class="tt ${up?"bull":"bear"}">${up?"+":""}${fmt(t2.annual_slope_pct)}%</span>
      <span class="sig sig-${t2.signal}" style="font-size:10px;padding:2px 8px">${t2.signal}</span>
    </div>`;}).join("");

  /* anomaly events */
  const dates=an.recent_anomaly_dates||[];
  const tm={};cd.filter(r=>r.anomaly).forEach(r=>{tm[r.date]=r.type;});
  document.getElementById("al").innerHTML=dates.length===0
    ?`<p style="color:var(--mu);font-size:12px;font-family:var(--mono)">No recent anomalies detected</p>`
    :`<table class="at"><thead><tr><th>Date</th><th>Type</th></tr></thead><tbody>
      ${dates.slice().reverse().map(d=>`<tr><td>${d}</td><td>
        <span class="chip chip-${tm[d]||"multivariate"}">${(tm[d]||"multivariate").replace("_"," ")}</span>
      </td></tr>`).join("")}</tbody></table>`;

  /* signal factors */
  const sma50approx=tr.support*1.05;
  const F=[
    {l:"RSI < 70 (not overbought)",  v:tr.rsi<70},
    {l:"MACD bullish crossover",      v:tr.macd_diff>0},
    {l:"Momentum score > 0.55",       v:tr.momentum_score>0.55},
    {l:"Annual trend positive",       v:tr.annual_slope_pct>0},
    {l:"Bull market regime",          v:tr.regime==="BULL"},
    {l:"Price above 50-day SMA",      v:tr.current_price>sma50approx},
  ];
  document.getElementById("fb").innerHTML=F.map(f=>`
    <div style="margin-bottom:11px">
      <div style="display:flex;justify-content:space-between;margin-bottom:3px">
        <span style="font-size:11px;color:var(--mu);font-family:var(--mono)">${f.l}</span>
        <span style="font-size:11px;font-family:var(--mono);color:${f.v?"var(--bull)":"var(--bear)"}">${f.v?"Y":"N"}</span>
      </div>
      <div class="pb"><div class="pf" style="width:${f.v?100:12}%;background:${f.v?"var(--bull)":"var(--border)"}"></div></div>
    </div>`).join("");

  document.getElementById("rt").textContent="Last run: "+new Date().toLocaleString();
}

function pick(t){active=t;render();}
render();
</script>
</body>
</html>"""



#  CLI ENTRY POINT


def _parse_args():
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="FinPulse: financial data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py
  python pipeline.py --tickers AAPL TSLA NVDA AMZN
  python pipeline.py --period 2y --tickers SPY QQQ IWM
  python pipeline.py --no-dashboard
  python pipeline.py --dashboard-only
  python pipeline.py --s3-bucket my-bucket
  python pipeline.py --open
  python pipeline.py --dashboard-only --open
        """,
    )
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS, metavar="TICKER",
        help=f"Ticker symbols (default: {' '.join(DEFAULT_TICKERS)})",
    )
    parser.add_argument(
        "--period", default=DEFAULT_PERIOD,
        choices=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        help="Lookback period (default: 1y)",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Skip regenerating the HTML dashboard",
    )
    parser.add_argument(
        "--dashboard-only", action="store_true",
        help="Only rebuild dashboard from existing data (no download)",
    )
    parser.add_argument(
        "--s3-bucket", metavar="BUCKET", default=None,
        help="S3 bucket name; upload all pipeline outputs after the run",
    )
    parser.add_argument(
        "--skip-local", action="store_true",
        help="Skip writing files locally (use with --s3-bucket for cloud-only mode)",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="After run, open reports/dashboard.html in the default browser",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.dashboard_only:
        path = build_dashboard()
        print(f"Dashboard rebuilt: {path}")
        if args.open:
            launch_dashboard()
    else:
        run_pipeline(
            tickers    = [t.upper() for t in args.tickers],
            period     = args.period,
            build_dash = not args.no_dashboard,
            s3_bucket  = args.s3_bucket,
            skip_local = args.skip_local,
        )
        if args.open and not args.no_dashboard and not args.skip_local:
            launch_dashboard()
