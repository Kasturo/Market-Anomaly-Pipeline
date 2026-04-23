# Financial Data Pipeline with Anomaly Detection

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



## Requirements

See `requirements.txt`. Core: `yfinance`, `pandas`, `numpy`, `scikit-learn`, `boto3`.
