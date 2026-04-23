"""
Financial data pipeline, CSV mode (offline or when yfinance is blocked).
Reads from data/{TICKER}_raw.csv or falls back to yfinance if available.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime
import json, os, warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TICKERS  = ["AAPL", "MSFT", "GOOGL", "JPM", "SPY"]

# Load

def load_data(tickers):
    raw = {}
    for t in tickers:
        path = f"{DATA_DIR}/{t}_raw.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            raw[t] = df
            print(f"  OK {t}: {len(df)} rows (CSV)")
        else:
            try:
                import yfinance as yf
                df = yf.download(t, period="1y", progress=False, auto_adjust=True)
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                raw[t] = df
                print(f"  OK {t}: {len(df)} rows (yfinance)")
            except Exception as e:
                print(f"  FAIL {t}: {e}")
    return raw

# Features

def engineer_features(df, ticker=""):
    d = df.copy()
    d["ret_1d"]  = d["Close"].pct_change()
    d["ret_5d"]  = d["Close"].pct_change(5)
    d["ret_20d"] = d["Close"].pct_change(20)
    d["vol_10d"] = d["ret_1d"].rolling(10).std()
    d["vol_30d"] = d["ret_1d"].rolling(30).std()
    vm = d["Volume"].rolling(20).mean(); vs = d["Volume"].rolling(20).std()
    d["vol_zscore"] = (d["Volume"] - vm) / (vs + 1e-9)
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    sma20 = d["Close"].rolling(20).mean(); std20 = d["Close"].rolling(20).std()
    d["bb_width"] = 2 * std20 / (sma20 + 1e-9)
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    d["macd_diff"] = macd - macd.ewm(span=9, adjust=False).mean()
    sma50 = d["Close"].rolling(50).mean()
    d["price_vs_sma50"] = (d["Close"] - sma50) / (sma50 + 1e-9)
    d["momentum"] = (
        d["ret_5d"].rank(pct=True) * 0.4 +
        d["ret_20d"].rank(pct=True) * 0.4 +
        (1 - d["vol_10d"].rank(pct=True)) * 0.2
    )
    d["ticker"] = ticker
    return d.dropna()

# Anomaly detection

FEAT = ["ret_1d","ret_5d","vol_10d","vol_zscore","rsi","bb_width","macd_diff"]

def detect_anomalies(df, contamination=0.05):
    X = StandardScaler().fit_transform(df[[c for c in FEAT if c in df.columns]])
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iso.fit(X)
    df = df.copy()
    df["anomaly_score"] = -iso.decision_function(X)
    df["is_anomaly"]    = iso.predict(X) == -1
    def classify(r):
        if not r["is_anomaly"]: return "normal"
        if abs(r["ret_1d"]) > 0.03: return "price_spike"
        if abs(r["vol_zscore"]) > 2.5: return "volume_surge"
        if r["rsi"] > 80 or r["rsi"] < 20: return "rsi_extreme"
        return "multivariate"
    df["anomaly_type"] = df.apply(classify, axis=1)
    return df

# Trend analysis

def analyze_trends(df):
    close = df["Close"].values.reshape(-1,1)
    X     = np.arange(len(close)).reshape(-1,1)
    reg   = LinearRegression().fit(X, close)
    slope = float(reg.coef_[0][0])
    r2    = float(reg.score(X, close))
    current = float(df["Close"].iloc[-1])
    annual_slope_pct = slope * 252 / current * 100

    ret20 = float(df["ret_20d"].iloc[-1])
    regime = "BULL" if ret20 > 0.05 else ("BEAR" if ret20 < -0.05 else "SIDEWAYS")

    support    = float(df["Close"].rolling(252, min_periods=20).min().iloc[-1])
    resistance = float(df["Close"].rolling(252, min_periods=20).max().iloc[-1])
    rsi  = float(df["rsi"].iloc[-1])
    macd = float(df["macd_diff"].iloc[-1])
    mom  = float(df["momentum"].iloc[-1])
    sma50 = float(df["Close"].rolling(50).mean().iloc[-1])

    bullish = sum([rsi<70, macd>0, mom>0.55, annual_slope_pct>0,
                   regime=="BULL", current>sma50])
    signal  = "BUY" if bullish>=4 else ("SELL" if bullish<=2 else "HOLD")

    return dict(
        annual_slope_pct=round(annual_slope_pct,2), r2=round(r2,4),
        regime=regime, support=round(support,2), resistance=round(resistance,2),
        current_price=round(current,2), rsi=round(rsi,1),
        macd_diff=round(macd,4), momentum_score=round(mom,3),
        signal=signal, bullish_factors=bullish
    )

# Pipeline

def run_pipeline(tickers=TICKERS):
    print("\n" + "="*60)
    print("  FINANCIAL DATA PIPELINE  |  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("="*60)
    print(f"\n[INGEST] Loading {len(tickers)} tickers")
    raw = load_data(tickers)

    results = {}
    for ticker, df in raw.items():
        feat_df  = engineer_features(df, ticker)
        anom_df  = detect_anomalies(feat_df)
        trend    = analyze_trends(anom_df)
        n_anom   = int(anom_df["is_anomaly"].sum())
        anom_pct = round(n_anom / len(anom_df) * 100, 1)

        # Save processed CSV (include anomaly info)
        anom_df.to_csv(f"{DATA_DIR}/{ticker}_processed.csv")

        results[ticker] = {
            "trend": trend,
            "anomalies": {
                "total": n_anom, "pct_of_days": anom_pct,
                "by_type": anom_df[anom_df["is_anomaly"]]["anomaly_type"]
                           .value_counts().to_dict(),
                "recent_anomaly_dates": [
                    str(d.date()) for d in
                    anom_df[anom_df["is_anomaly"]].index[-5:]
                ],
            },
            "rows": len(anom_df),
            "date_range": {
                "start": str(anom_df.index[0].date()),
                "end":   str(anom_df.index[-1].date()),
            }
        }

        icons = {"BUY": "[+]", "SELL": "[-]", "HOLD": "[=]"}
        sig   = trend["signal"]
        print(f"\n  [{ticker}] {icons.get(sig, '[?]')} {sig}  |  "
              f"Regime: {trend['regime']}  |  "
              f"Trend: {trend['annual_slope_pct']:+.1f}%/yr  |  "
              f"RSI: {trend['rsi']}  |  "
              f"Anomalies: {n_anom} ({anom_pct}%)")

    with open(f"{DATA_DIR}/pipeline_summary.json", "w", encoding="utf-8") as f:
        def serial(o):
            if hasattr(o,"item"): return o.item()
            return str(o)
        json.dump(results, f, indent=2, default=serial)

    print(f"\n[DONE] Results saved to {DATA_DIR}/")
    print("="*60)
    return results

if __name__ == "__main__":
    results = run_pipeline()
