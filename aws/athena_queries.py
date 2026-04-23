"""
aws/athena_queries.py
=====================
Pre-built Amazon Athena SQL queries for analysing FinPulse pipeline output
stored in S3, plus a helper to submit and poll them programmatically.

This implements the "Serverless Querying Dashboard" AWS pattern:
  S3 (processed CSVs) -> Glue Crawler (auto schema) -> Athena SQL

Setup (one-time, ~5 minutes)
-----------------------------
1. Create a Glue database and crawler pointed at your S3 bucket:

   aws glue create-database --database-input '{"Name":"finpulse"}'

   aws glue create-crawler \\
       --name finpulse-crawler \\
       --role arn:aws:iam::<ACCOUNT_ID>:role/finpulse-glue-role \\
       --database-name finpulse \\
       --targets '{"S3Targets":[{"Path":"s3://<BUCKET>/processed/"},
                                 {"Path":"s3://<BUCKET>/json/"}]}'

   aws glue start-crawler --name finpulse-crawler

2. After the crawler finishes (1-2 min) you'll have two tables in Athena:
   - finpulse.processed   (all *_processed.csv columns)
   - finpulse.pipeline_summary  (from pipeline_summary.json)

3. Set an Athena results bucket:
   aws athena update-work-group \\
       --work-group primary \\
       --configuration-updates \\
         'ResultConfigurationUpdates={OutputLocation=s3://<BUCKET>/athena-results/}'

Usage
-----
  from aws.athena_queries import run_query, QUERIES

  # Run a named query and get results as a list of dicts
  rows = run_query("top_anomalies", database="finpulse",
                   output_bucket="my-finpulse-bucket")
  for row in rows:
      print(row)

  # Or just print the SQL for use in the Athena console
  print(QUERIES["monthly_anomaly_counts"])
"""

from __future__ import annotations

import time
from typing import Any

# ---------------------------------------------------------------------------
# SQL query library
# ---------------------------------------------------------------------------

QUERIES: dict[str, str] = {

    "top_anomalies": """
-- Top 50 most anomalous trading days across all tickers.
-- Uses anomaly_score (higher = more unusual).
SELECT
    ticker,
    date,
    anomaly_type,
    ROUND(CAST(ret_1d AS DOUBLE) * 100, 2)   AS daily_return_pct,
    ROUND(CAST(vol_zscore AS DOUBLE), 2)      AS volume_zscore,
    ROUND(CAST(rsi AS DOUBLE), 1)             AS rsi,
    ROUND(CAST(anomaly_score AS DOUBLE), 4)   AS anomaly_score
FROM processed
WHERE is_anomaly = 'True'
ORDER BY CAST(anomaly_score AS DOUBLE) DESC
LIMIT 50;
""".strip(),

    "monthly_anomaly_counts": """
-- Anomaly frequency by ticker and month.
-- Useful for spotting whether specific calendar periods are consistently volatile.
SELECT
    ticker,
    DATE_FORMAT(DATE_PARSE(date, '%Y-%m-%d'), '%Y-%m') AS month,
    COUNT(*) FILTER (WHERE is_anomaly = 'True')        AS anomaly_days,
    COUNT(*)                                            AS total_trading_days,
    ROUND(
        COUNT(*) FILTER (WHERE is_anomaly = 'True') * 100.0 / COUNT(*), 1
    )                                                   AS anomaly_rate_pct
FROM processed
GROUP BY ticker, DATE_FORMAT(DATE_PARSE(date, '%Y-%m-%d'), '%Y-%m')
ORDER BY ticker, month;
""".strip(),

    "anomaly_type_breakdown": """
-- Distribution of anomaly types per ticker.
-- Helps understand whether unusual days are driven by price, volume, or RSI.
SELECT
    ticker,
    anomaly_type,
    COUNT(*) AS occurrences,
    ROUND(AVG(CAST(ret_1d AS DOUBLE)) * 100, 2) AS avg_daily_return_pct,
    ROUND(AVG(CAST(vol_zscore AS DOUBLE)), 2)    AS avg_volume_zscore
FROM processed
WHERE is_anomaly = 'True'
GROUP BY ticker, anomaly_type
ORDER BY ticker, occurrences DESC;
""".strip(),

    "signal_regime_distribution": """
-- How often each ticker receives BUY / HOLD / SELL signals per market regime.
-- Cross-tab of signal vs regime for quick portfolio overview.
SELECT
    ticker,
    regime,
    signal,
    bullish_factors,
    annual_slope_pct,
    rsi,
    current_price
FROM pipeline_summary
ORDER BY annual_slope_pct DESC;
""".strip(),

    "volatility_ranking": """
-- Rank tickers by average 10-day rolling volatility (annualised proxy).
-- Higher vol_10d: more volatile / riskier asset.
SELECT
    ticker,
    ROUND(AVG(CAST(vol_10d AS DOUBLE)) * 100, 3)  AS avg_vol_10d_pct,
    ROUND(MAX(CAST(vol_10d AS DOUBLE)) * 100, 3)  AS max_vol_10d_pct,
    ROUND(AVG(CAST(rsi AS DOUBLE)), 1)             AS avg_rsi,
    COUNT(*) FILTER (WHERE is_anomaly = 'True')    AS anomaly_count,
    COUNT(*)                                        AS total_days
FROM processed
GROUP BY ticker
ORDER BY avg_vol_10d_pct DESC;
""".strip(),

    "momentum_leaders": """
-- Top momentum days (composite score > 0.7) per ticker in the last 60 days.
-- Identifies periods where price + recent returns align strongly upward.
SELECT
    ticker,
    date,
    ROUND(CAST(momentum AS DOUBLE), 3)           AS momentum_score,
    ROUND(CAST(ret_5d AS DOUBLE)  * 100, 2)      AS ret_5d_pct,
    ROUND(CAST(ret_20d AS DOUBLE) * 100, 2)      AS ret_20d_pct,
    ROUND(CAST(rsi AS DOUBLE), 1)                AS rsi
FROM processed
WHERE CAST(momentum AS DOUBLE) > 0.70
ORDER BY date DESC, momentum_score DESC
LIMIT 100;
""".strip(),

    "support_resistance": """
-- Latest support and resistance levels for each ticker,
-- along with where current price sits within the 52-week range.
SELECT
    ticker,
    current_price,
    support,
    resistance,
    ROUND(
        (current_price - support) * 100.0 / NULLIF(resistance - support, 0), 1
    ) AS position_in_range_pct,
    signal,
    regime
FROM pipeline_summary
ORDER BY position_in_range_pct DESC;
""".strip(),

    "macd_crossovers": """
-- Days where MACD crossed above the signal line (macd_diff flipped from negative to positive).
-- A classic buy-signal trigger in technical analysis.
SELECT
    ticker,
    date,
    ROUND(CAST(macd_diff AS DOUBLE), 4)   AS macd_diff,
    ROUND(CAST(rsi AS DOUBLE), 1)         AS rsi,
    ROUND(CAST(ret_1d AS DOUBLE) * 100, 2) AS daily_return_pct
FROM (
    SELECT *,
        LAG(CAST(macd_diff AS DOUBLE)) OVER (
            PARTITION BY ticker ORDER BY date
        ) AS prev_macd_diff
    FROM processed
)
WHERE prev_macd_diff < 0
  AND CAST(macd_diff AS DOUBLE) >= 0
ORDER BY date DESC
LIMIT 200;
""".strip(),

}


# ---------------------------------------------------------------------------
# Athena execution helper
# ---------------------------------------------------------------------------

def run_query(
    query_name: str,
    database: str,
    output_bucket: str,
    output_prefix: str = "athena-results/",
    poll_interval: float = 2.0,
    timeout: float = 120.0,
    workgroup: str = "primary",
) -> list[dict[str, Any]]:
    """
    Submit a named query from QUERIES to Athena and return results as a list
    of dicts (one per row, keys = column names).

    Parameters
    ----------
    query_name    : key in the QUERIES dict (e.g. "top_anomalies")
    database      : Glue / Athena database name (e.g. "finpulse")
    output_bucket : S3 bucket for Athena result files
    output_prefix : prefix within the bucket (default "athena-results/")
    poll_interval : seconds between status polls (default 2.0)
    timeout       : max seconds to wait before raising TimeoutError (default 120)
    workgroup     : Athena workgroup (default "primary")

    Returns
    -------
    list[dict]: query result rows
    """
    if query_name not in QUERIES:
        available = ", ".join(QUERIES.keys())
        raise ValueError(
            f"Unknown query '{query_name}'. Available: {available}"
        )

    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 is required.  Run:  pip install boto3")

    client = boto3.client("athena")
    sql    = QUERIES[query_name]
    output = f"s3://{output_bucket}/{output_prefix}"

    # Start query
    response = client.start_query_execution(
        QueryString          = sql,
        QueryExecutionContext= {"Database": database},
        ResultConfiguration  = {"OutputLocation": output},
        WorkGroup            = workgroup,
    )
    exec_id = response["QueryExecutionId"]
    print(f"[Athena] '{query_name}' submitted; execution ID: {exec_id}")

    # Poll until done
    elapsed = 0.0
    while elapsed < timeout:
        status = client.get_query_execution(QueryExecutionId=exec_id)
        state  = status["QueryExecution"]["QueryExecutionStatus"]["State"]

        if state == "SUCCEEDED":
            break
        if state in ("FAILED", "CANCELLED"):
            reason = status["QueryExecution"]["QueryExecutionStatus"].get(
                "StateChangeReason", "unknown"
            )
            raise RuntimeError(f"[Athena] Query {state}: {reason}")

        time.sleep(poll_interval)
        elapsed += poll_interval

    if elapsed >= timeout:
        raise TimeoutError(
            f"[Athena] Query '{query_name}' did not complete within {timeout}s"
        )

    # Fetch results
    paginator = client.get_paginator("get_query_results")
    headers: list[str] = []
    rows: list[dict] = []

    for page in paginator.paginate(QueryExecutionId=exec_id):
        result_rows = page["ResultSet"]["Rows"]
        if not headers:
            headers = [c["VarCharValue"] for c in result_rows[0]["Data"]]
            result_rows = result_rows[1:]   # skip header row
        for row in result_rows:
            values = [c.get("VarCharValue", "") for c in row["Data"]]
            rows.append(dict(zip(headers, values)))

    print(f"[Athena] '{query_name}': {len(rows)} rows returned")
    return rows


def print_query(query_name: str) -> None:
    """Print the SQL for a named query (useful for pasting into Athena console)."""
    if query_name not in QUERIES:
        available = ", ".join(QUERIES.keys())
        raise ValueError(f"Unknown query '{query_name}'. Available: {available}")
    print(f"\n-- {query_name}\n")
    print(QUERIES[query_name])
    print()


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json as _json

    parser = argparse.ArgumentParser(
        prog="athena_queries",
        description="Run or print FinPulse Athena queries.",
    )
    parser.add_argument("query", choices=list(QUERIES.keys()), help="Query to run or print")
    parser.add_argument("--print-only", action="store_true",
                        help="Print SQL without executing")
    parser.add_argument("--database",     default="finpulse",
                        help="Glue/Athena database name (default: finpulse)")
    parser.add_argument("--output-bucket", required=False,
                        help="S3 bucket for Athena results (required unless --print-only)")
    args = parser.parse_args()

    if args.print_only:
        print_query(args.query)
    else:
        if not args.output_bucket:
            parser.error("--output-bucket is required when not using --print-only")
        results = run_query(args.query, database=args.database,
                            output_bucket=args.output_bucket)
        print(_json.dumps(results, indent=2))
