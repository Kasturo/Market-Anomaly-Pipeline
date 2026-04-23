# FinPulse: Full Project Rundown

This document describes the FinPulse repository end to end: purpose, layout, data flow, machine learning, outputs, AWS integration, and how to run everything.

---

## 1. Purpose

FinPulse is a **portfolio-style** market data pipeline. It:

- Downloads daily OHLCV (open, high, low, close, volume) stock data.
- Builds **technical features** used in trading and risk workflows.
- Flags **anomalous days** with an **Isolation Forest** (unsupervised learning).
- Summarises **trend and regime** (bull/bear/sideways) and emits a simple **BUY / HOLD / SELL** style signal from rule-based factors.
- Writes **CSV and JSON** artefacts and a **single-file HTML dashboard** (Chart.js over CDN).
- Optionally **uploads** those artefacts to **Amazon S3** and can be extended with **AWS Lambda**, **EventBridge**, **Glue**, and **Athena**.

It is **not** investment advice. Signals are for demonstration and learning.

---

## 2. Repository layout

| Path | Role |
|------|------|
| `pipeline.py` | Main entry: Yahoo Finance ingestion, features, anomalies, trends, export, dashboard build, optional S3. |
| `pipeline_csv.py` | Same analytics path but reads `data/{TICKER}_raw.csv` first; can fall back to yfinance if a file is missing. |
| `generate_mock_data.py` | Creates synthetic OHLCV CSVs (GBM-style paths plus injected spike days) for offline demos. |
| `aws/s3_storage.py` | Boto3 helpers: upload one file, download one file, bulk-upload pipeline outputs under `raw/`, `processed/`, `json/`, `dashboard/`. |
| `aws/lambda_handler.py` | AWS Lambda handler: calls `run_pipeline`, then uploads to the bucket named in env `S3_BUCKET`. |
| `aws/athena_queries.py` | Named SQL strings plus `run_query()` to execute them in Athena (after Glue has catalogued S3 data). |
| `aws/__init__.py` | Marks `aws` as a package. |
| `data/` | Runtime outputs: `*_raw.csv`, `*_processed.csv`, `pipeline_summary.json`, `chart_data.json`. |
| `reports/dashboard.html` | Generated dashboard; open in a browser (UTF-8). |
| `requirements.txt` | Python dependencies including `boto3`. |
| `README.md` | Quick start and AWS overview. |
| `.gitignore` | Ignores virtualenv, `private`, zip artefacts, etc. Do **not** commit AWS keys. |

---

## 3. End-to-end data flow (`pipeline.py`)

1. **Ingest**  
   For each ticker, `yfinance` downloads daily bars (`period`, `interval` configurable). On failure, if `data/{TICKER}_raw.csv` exists, it is loaded instead. Successful live pulls are cached to that raw CSV path.

2. **Feature engineering** (`engineer_features`)  
   Computes returns (1d, 5d, 20d), rolling volatilities, volume z-score vs 20-day window, RSI(14), Bollinger width, MACD histogram-style `macd_diff`, price vs 50-day SMA, a composite **momentum** score from ranked returns and vol, and a `ticker` label. Rows with NaNs from rolling windows are dropped.

3. **Anomaly detection** (`detect_anomalies`)  
   Standardises columns listed in `ANOMALY_FEATURES`, fits **Isolation Forest** (`contamination` default 5%), assigns `anomaly_score`, `is_anomaly`, and a string `anomaly_type` from simple thresholds on returns, volume z-score, and RSI.

4. **Trend analysis** (`analyze_trends`)  
   Linear regression of close vs time index (annualised slope as percent of last price), R2, regime from 20-day return bands, 52-week min/max as support/resistance proxies, and a **signal** from six boolean checks (RSI, MACD, momentum, slope, regime, price vs SMA50).

5. **Export** (`export_results`)  
   Per-ticker processed CSVs, `pipeline_summary.json` (full nested results), `chart_data.json` (last 60 rows per ticker for charts). Optional `skip_local` skips disk writes (intended for advanced cloud-only setups; normal runs keep it false).

6. **Dashboard** (`build_dashboard`)  
   Injects JSON into the embedded HTML template and writes `reports/dashboard.html` as **UTF-8** (important on Windows).

7. **S3** (optional)  
   If `s3_bucket` is set, `upload_pipeline_outputs` pushes the same files to the bucket using the default boto3 credential chain (env vars, shared credentials file, or IAM role).

---

## 4. CLI reference (`pipeline.py`)

| Flag | Meaning |
|------|---------|
| `--tickers AAPL MSFT ...` | Symbols to process (default five mega-cap / ETF names). |
| `--period` | `1mo` … `5y` for yfinance. |
| `--no-dashboard` | Skip HTML rebuild. |
| `--dashboard-only` | Rebuild dashboard from existing JSON only. |
| `--s3-bucket NAME` | After a successful run, upload artefacts to S3. |
| `--skip-local` | Skip local export writes (use only if you understand the implications). |

Run from the **project root** so `data/` and `reports/` resolve correctly.

---

## 5. CSV pipeline (`pipeline_csv.py`)

- Fixed ticker list in `TICKERS`.
- Loads `data/{TICKER}_raw.csv` when present; otherwise tries yfinance for that symbol.
- Writes `*_processed.csv` and `pipeline_summary.json` only (no chart JSON, no dashboard, no S3 in this script). Use it for **offline** or **minimal** runs.

---

## 6. Mock data (`generate_mock_data.py`)

- CLI: `--tickers`, `--days`, `--anomalies`.
- Writes `data/{TICKER}_raw.csv` with business-day index.
- Per-ticker parameters in `TICKER_PARAMS`; unknown tickers fall back to AAPL-like parameters with a console warning.

---

## 7. AWS components

### S3 layout (after `upload_pipeline_outputs`)

- `raw/{TICKER}_raw.csv`
- `processed/{TICKER}_processed.csv`
- `json/pipeline_summary.json`, `json/chart_data.json`
- `dashboard/dashboard.html`

### Credentials

- **Local:** Install AWS CLI v2 and `aws configure`, or set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` in the **same** environment as `python`.
- **Project `.env`:** Copy `.env.example` to `.env` in the project root (never commit it). `pipeline.py` loads it at startup via `python-dotenv` so S3 works in the same command as live yfinance pulls.
- **Cloud Shell** credentials do not apply to your PC unless you copy keys (prefer IAM user keys on the laptop, or SSO per your org).
- IAM user needs at least `s3:ListBucket` on the bucket and `s3:GetObject` / `s3:PutObject` on `bucket/*`.

### Lambda (`aws/lambda_handler.py`)

- Handler name: `handler` (so Lambda runtime handler string is `lambda_handler.handler` if the file is packaged as `lambda_handler.py` at zip root; adjust packaging to match your layout).
- Environment: `S3_BUCKET` required; optional `TICKERS`, `PERIOD`, `PUBLIC_DASHBOARD`.

### Athena (`aws/athena_queries.py`)

- Requires a Glue database and crawler over the S3 prefixes, Athena workgroup output location, and table names that match the SQL (often `processed` and `pipeline_summary` after crawler naming). Adjust SQL if Glue names columns differently (e.g. string booleans from CSV).

---

## 8. Dashboard behaviour

- Self-contained page except for Chart.js and Google Fonts CDN URLs.
- Ticker tabs, KPI cards, price + anomaly scatter overlay, RSI, volatility, doughnut of anomaly types, portfolio list, recent anomaly table, factor checklist.
- Rebuilt whenever you run a full pipeline (unless `--no-dashboard`).

---

## 9. Dependencies (`requirements.txt`)

- **yfinance:** market data.
- **pandas / numpy:** tables and numerics.
- **scikit-learn:** Isolation Forest, scaler, linear regression.
- **matplotlib / seaborn:** listed for typical data science environments (not all are required for the core path).
- **requests:** pulled in by ecosystem.
- **boto3:** S3 and Athena helpers.

---

## 10. Security and hygiene

- Never commit IAM access keys or a `private` file with secrets. Rotate any key that was pasted into chat or checked into git.
- Add `private`, `.env`, and key material patterns to `.gitignore` (already ignores `private`).

---

## 11. Typical commands

```bash
pip install -r requirements.txt
python pipeline.py
python pipeline.py --s3-bucket your-bucket-name
python generate_mock_data.py
python pipeline_csv.py
python pipeline.py --dashboard-only
```

Windows: open `reports\dashboard.html` in Chrome or Edge after a successful run.

---

## 12. Possible extensions

- Scheduled Lambda + EventBridge.
- Glue/Athena for analyst SQL.
- Stronger backtesting and transaction-cost modelling.
- Database sink instead of flat files.
- Streaming ingestion (Kinesis, etc.) for intraday data.

This file is the **canonical long-form** description of the project; `README.md` stays shorter for first-time readers.
