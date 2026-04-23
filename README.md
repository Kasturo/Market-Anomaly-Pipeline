# FinPulse: Financial Data Pipeline with Anomaly Detection

> Automated market data pipeline featuring machine learning anomaly detection, technical trend analysis, an interactive dashboard, and AWS cloud integration. Built to demonstrate financial domain knowledge, data engineering skills, and cloud architecture.

For a longer technical description of every module and output, see [PROJECT_RUNDOWN.md](PROJECT_RUNDOWN.md).

---

## Project Overview

FinPulse is an end-to-end data pipeline that:
1. **Ingests** OHLCV market data via `yfinance` (or CSV fallback)
2. **Engineers** 12 financial features (RSI, MACD, Bollinger Bands, momentum, etc.)
3. **Detects anomalies** using Isolation Forest (unsupervised ML)
4. **Analyzes trends** with linear regression, regime detection, and composite buy/sell signals
5. **Visualizes results** in an interactive HTML dashboard
6. **Stores outputs** in Amazon S3 and can run on a schedule via AWS Lambda + EventBridge
7. **Queries results** with Amazon Athena (serverless SQL over S3)

---

## Skills Demonstrated (Resume Talking Points)

| Skill | Where Used |
|-------|------------|
| Python data engineering | `pipeline.py`: modular ETL pipeline |
| Feature engineering | RSI, MACD, Bollinger Bands, momentum scoring |
| Unsupervised ML (Isolation Forest) | Anomaly detection on multivariate financial data |
| Statistical modeling | Linear regression for trend slope + R2 |
| Financial domain knowledge | Regime detection, support/resistance, signal generation |
| Data visualization | Interactive Chart.js dashboard |
| Cloud storage (AWS S3) | `aws/s3_storage.py`: artifact upload/download |
| Serverless computing (AWS Lambda) | `aws/lambda_handler.py`: EventBridge-scheduled pipeline |
| Serverless SQL (Amazon Athena) | `aws/athena_queries.py`: 8 pre-built analyst queries |
| Software engineering | Modular, documented, production-style code |

---

## Project Structure

```
financial_pipeline/
  pipeline.py               Main pipeline (yfinance + optional --s3-bucket)
  pipeline_csv.py           CSV-first pipeline
  generate_mock_data.py     Synthetic OHLCV generator
  aws/
    s3_storage.py           S3 helpers (boto3)
    lambda_handler.py       Lambda entry point
    athena_queries.py       Athena SQL library + runner
  data/
    {TICKER}_raw.csv
    {TICKER}_processed.csv
    chart_data.json
    pipeline_summary.json
  reports/
    dashboard.html
  requirements.txt
  README.md
```

---

## How to Run

```bash
# 1. Install dependencies (includes boto3 for AWS)
pip install -r requirements.txt

# 2. Run pipeline (live Yahoo Finance data)
python pipeline.py

# 3. Or CSV / offline mode
python pipeline_csv.py

# 4. Open the dashboard (easiest: let Python open your default browser)
python pipeline.py --dashboard-only --open
# Or macOS/Linux:  open reports/dashboard.html
# Windows CMD:     start reports\dashboard.html
# Git Bash:        cmd //c start reports/dashboard.html

# 5. Run pipeline and upload outputs to S3 (needs AWS credentials)
python pipeline.py --s3-bucket your-finpulse-bucket
```

**Live Yahoo data + S3 together**

1. Install deps: `pip install -r requirements.txt` (includes `python-dotenv`).
2. Copy `.env.example` to `.env`, add your IAM access key, secret, and region (same values you would `export` in Git Bash). Do not commit `.env`.
3. From the project folder run: `python pipeline.py --s3-bucket your-bucket-name`  
   You should see `[INFO] Loaded environment from .env` once, then `OK ... [live]` for each ticker, then S3 upload lines.  
   Alternative: run `aws configure` on Windows so credentials live in `%USERPROFILE%\.aws\` (no `.env` needed).

If a ticker shows `[cached]` instead of `[live]`, Yahoo failed and an older `data/*_raw.csv` was used; check network or wait and retry.

---

## Pipeline Architecture

```
Yahoo Finance or CSV
  -> Data ingestion (yfinance / pandas)
  -> Feature engineering (12 indicators: RSI, MACD, Bollinger, volume z-score, momentum, ...)
       -> Isolation Forest (anomaly_score, is_anomaly, anomaly_type)
       -> Trend analysis (linear slope, regime, support/resistance, BUY/SELL/HOLD)
  -> Exports: pipeline_summary.json, chart_data.json, processed CSVs
  -> dashboard.html (Chart.js) and optional S3 upload / Athena
```

---

## Anomaly Detection: Isolation Forest

Isolation Forest randomly partitions features and measures how quickly a point is isolated. Unusual trading days need fewer splits.

**Features used:**
- 1-day and 5-day returns
- 10-day rolling volatility
- Volume z-score (vs 20-day average)
- RSI (14-day)
- Bollinger Band width
- MACD signal line divergence

**Anomaly types classified:**
- `price_spike`: abs(daily return) > 3%
- `volume_surge`: volume z-score > 2.5 std dev
- `rsi_extreme`: RSI > 80 or < 20
- `multivariate`: combination of moderate signals

---

## Trend Analysis

- **Linear slope** of closing price (annualised as percent of current price)
- **R2 score**: how well the linear model fits
- **Market regime**: BULL / BEAR / SIDEWAYS based on 20-day returns
- **Support/Resistance**: 52-week rolling low/high
- **Composite signal (BUY/SELL/HOLD)**: 6 boolean factors (RSI, MACD, momentum, trend, regime, price vs 50-day SMA)

---

## Connection to Fraud Detection

| Fraud Detection | This Pipeline |
|-----------------|---------------|
| Transaction features | OHLCV to technical indicators |
| Fraud flag (rare class) | Market anomaly (5% contamination) |
| Isolation Forest | Same algorithm |
| Behavioral scoring | Momentum / composite signal |
| Alert classification | Anomaly type labeling |

The same unsupervised pattern (rich features, Isolation Forest, alert-style labels) maps to transaction fraud use cases.

---

## AWS Cloud Deployment

| Pattern | Services Used |
|---------|--------------|
| Cloud data storage | **Amazon S3**: raw CSVs, processed data, dashboard HTML |
| Scheduled execution | **AWS Lambda** + **EventBridge**: daily cron |
| Serverless SQL | **AWS Glue** + **Amazon Athena**: SQL over S3 |

### S3 bucket layout

```
s3://your-finpulse-bucket/
  raw/
  processed/
  json/
  dashboard/
  athena-results/
```

### 1. Configure AWS credentials

```bash
# Option A: environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Option B: AWS CLI (persistent on that machine)
aws configure
```

### 2. Run pipeline and upload

```bash
pip install -r requirements.txt
python pipeline.py --s3-bucket your-finpulse-bucket
```

### 3. Lambda (optional)

See comments in `aws/lambda_handler.py` for packaging, `create-function`, and EventBridge cron examples.

### 4. Athena (optional)

Create a Glue database and crawler over `processed/` and `json/`, then use `aws/athena_queries.py` (`run_query` or CLI `--print-only`).

---

## Extending the Project

- Alpha Vantage or other data APIs
- Amazon Kinesis for streaming ticks
- Amazon Redshift or other warehouses
- LSTM / Prophet for forecasting
- Portfolio metrics and backtesting
- Alerts (email, Slack, SNS)

---

## Git and GitHub (do not leak AWS keys)

- **Ignored:** `.env`, `private`, any `.env.*` except `.env.example` (see `.gitignore`). Put real keys only in `.env` on your machine.
- **Safe to commit:** `.env.example` (empty placeholders), all `.py` files, `README.md`, `PROJECT_RUNDOWN.md`, `requirements.txt`, `aws/` helpers.
- **Your choice:** `data/*.csv`, `data/*.json`, and `reports/dashboard.html` are **not** in `.gitignore` by default (good for a self-contained demo, but files can be large). To keep the repo small, add those paths to `.gitignore` and rely on `generate_mock_data.py` + a fresh `python pipeline.py` run for anyone cloning.

**Before the first push**

1. Confirm nothing secret is tracked: `git status` should not list `.env`.
2. Optional check: `git check-ignore -v .env` (should show the ignore rule).
3. If you ever pasted a key into a file that was committed, **rotate the IAM key** in AWS; deleting the file from git is not enough (history).

**New repo from this folder**

```bash
cd /path/to/Project
git init
git add .
git status
git commit -m "Initial commit: FinPulse pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

---

## Requirements

See `requirements.txt`. Core: `yfinance`, `pandas`, `numpy`, `scikit-learn`, `boto3`.
