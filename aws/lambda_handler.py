"""
aws/lambda_handler.py
=====================
AWS Lambda entry point for the FinPulse pipeline.

Triggered daily by Amazon EventBridge (CloudWatch Events) on a cron schedule.
After running the pipeline it uploads all artifacts to S3, making results
available for Athena queries and public dashboard access.

Architecture
------------
  EventBridge (cron) -> Lambda -> run_pipeline() -> S3 upload
                                      ↓
                              reports/dashboard.html
                              data/*_processed.csv
                              data/pipeline_summary.json
                              data/chart_data.json

Environment Variables (set in Lambda configuration)
----------------------------------------------------
  S3_BUCKET              Required. Destination bucket name.
  TICKERS                Optional. Comma-separated list, e.g. "AAPL,MSFT,TSLA"
                         Defaults to DEFAULT_TICKERS in pipeline.py.
  PERIOD                 Optional. yfinance period string (1mo 3mo 6mo 1y 2y 5y).
                         Defaults to "1y".
  PUBLIC_DASHBOARD       Optional. Set to "true" to make dashboard.html public-read.

Deploy / Update Lambda
----------------------
  # Package the function (run from project root)
  pip install -r requirements.txt -t package/
  cp -r aws pipeline.py generate_mock_data.py package/
  cd package && zip -r ../finpulse.zip . && cd ..

  # Create (first time)
  aws lambda create-function \\
      --function-name finpulse-pipeline \\
      --runtime python3.12 \\
      --role arn:aws:iam::<ACCOUNT_ID>:role/finpulse-lambda-role \\
      --handler lambda_handler.handler \\
      --timeout 300 \\
      --memory-size 512 \\
      --zip-file fileb://finpulse.zip \\
      --environment "Variables={S3_BUCKET=my-finpulse-bucket}"

  # Update code after changes
  aws lambda update-function-code \\
      --function-name finpulse-pipeline \\
      --zip-file fileb://finpulse.zip

  # Schedule: every weekday at 08:00 UTC (before US market open)
  aws events put-rule \\
      --name finpulse-daily \\
      --schedule-expression "cron(0 8 ? * MON-FRI *)" \\
      --state ENABLED

  aws events put-targets \\
      --rule finpulse-daily \\
      --targets "Id=1,Arn=arn:aws:lambda:<REGION>:<ACCOUNT_ID>:function:finpulse-pipeline"

Required IAM Permissions for the Lambda execution role
------------------------------------------------------
  s3:PutObject, s3:GetObject, s3:ListBucket  on the target bucket
  logs:CreateLogGroup, logs:CreateLogStream, logs:PutLogEvents  (CloudWatch)
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Lambda execution environment: project root is on PYTHONPATH
# (the zip packages everything at the top level)
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pipeline import (          # noqa: E402
    DEFAULT_TICKERS,
    DATA_DIR,
    RPT_DIR,
    run_pipeline,
)
from aws.s3_storage import upload_pipeline_outputs  # noqa: E402


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: dict, context) -> dict:
    """
    Lambda entry point.

    EventBridge passes an empty (or scheduled) event dict.
    You can also invoke manually with a custom payload:
      {
        "tickers": ["AAPL", "TSLA", "NVDA"],
        "period":  "6mo"
      }

    Returns
    -------
    dict: {"statusCode": 200|500, "body": {...}}
    """
    start = datetime.now(timezone.utc)
    print(f"[FinPulse Lambda] Invoked at {start.isoformat()}")

    # ── Resolve config from env vars + event payload ──────────────────────
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        return _error("S3_BUCKET environment variable is not set.")

    env_tickers = os.environ.get("TICKERS", "")
    tickers = (
        event.get("tickers")
        or (env_tickers.split(",") if env_tickers else None)
        or DEFAULT_TICKERS
    )
    tickers = [t.strip().upper() for t in tickers if t.strip()]

    period = event.get("period") or os.environ.get("PERIOD", "1y")
    public_dash = os.environ.get("PUBLIC_DASHBOARD", "false").lower() == "true"

    print(f"  Tickers : {tickers}")
    print(f"  Period  : {period}")
    print(f"  Bucket  : {bucket}")

    # ── Run the pipeline ──────────────────────────────────────────────────
    try:
        results = run_pipeline(
            tickers    = tickers,
            period     = period,
            build_dash = True,
        )
    except Exception:
        tb = traceback.format_exc()
        print("[ERROR] Pipeline failed:\n", tb)
        return _error(f"Pipeline raised an exception:\n{tb}")

    # ── Upload all artifacts to S3 ────────────────────────────────────────
    try:
        upload_report = upload_pipeline_outputs(
            bucket               = bucket,
            data_dir             = DATA_DIR,
            rpt_dir              = RPT_DIR,
            make_dashboard_public= public_dash,
        )
    except Exception:
        tb = traceback.format_exc()
        print("[ERROR] S3 upload failed:\n", tb)
        return _error(f"S3 upload raised an exception:\n{tb}")

    # ── Build response ────────────────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    summary = {
        t: {
            "signal":            r["trend"]["signal"],
            "regime":            r["trend"]["regime"],
            "annual_slope_pct":  r["trend"]["annual_slope_pct"],
            "anomaly_count":     r["anomalies"]["total"],
        }
        for t, r in results.items()
    }

    body = {
        "run_at":            start.isoformat(),
        "elapsed_seconds":   round(elapsed, 1),
        "tickers_processed": len(results),
        "s3_bucket":         bucket,
        "files_uploaded":    len(upload_report["uploaded"]),
        "files_skipped":     len(upload_report["skipped"]),
        "ticker_summary":    summary,
    }

    print(f"\n[FinPulse Lambda] Completed in {elapsed:.1f}s")
    print(json.dumps(summary, indent=2))

    return {"statusCode": 200, "body": body}


# ---------------------------------------------------------------------------
# Local test entry point
# ---------------------------------------------------------------------------

def _error(message: str) -> dict:
    print(f"[FinPulse Lambda] ERROR: {message}")
    return {"statusCode": 500, "body": {"error": message}}


if __name__ == "__main__":
    # Quick local smoke test: set S3_BUCKET env var before running
    # python aws/lambda_handler.py
    import pprint
    response = handler({"tickers": ["AAPL", "SPY"], "period": "6mo"}, context=None)
    pprint.pprint(response)
