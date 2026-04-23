"""
aws/s3_storage.py
=================
Boto3 helpers for pushing FinPulse pipeline artifacts to Amazon S3.

Credentials are resolved automatically in this order (standard boto3 chain):
  1. Environment variables  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
  2. ~/.aws/credentials  (aws configure)
  3. IAM instance / task / Lambda execution role  (when running on AWS)

S3 bucket layout written by upload_pipeline_outputs():
  s3://<bucket>/
    raw/          {TICKER}_raw.csv          (original OHLCV download)
    processed/    {TICKER}_processed.csv    (engineered features + anomaly labels)
    json/         pipeline_summary.json     (trend + anomaly stats per ticker)
                  chart_data.json           (60-day window for the dashboard)
    dashboard/    dashboard.html            (self-contained interactive report)

Usage
-----
  from aws.s3_storage import upload_pipeline_outputs, download_file

  # After running the pipeline locally
  upload_pipeline_outputs("my-finpulse-bucket", DATA_DIR, RPT_DIR)

  # Pull the latest summary for inspection
  download_file("my-finpulse-bucket", "json/pipeline_summary.json", "/tmp/summary.json")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _client():
    """Return a boto3 S3 client (credentials resolved from environment/IAM)."""
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for S3 storage.\n"
            "Run:  pip install boto3"
        )
    return boto3.client("s3")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upload_file(
    local_path: str | Path,
    bucket: str,
    s3_key: str,
    extra_args: Optional[dict] = None,
) -> str:
    """
    Upload a single local file to S3.

    Parameters
    ----------
    local_path : path to the local file
    bucket     : S3 bucket name
    s3_key     : destination key inside the bucket (e.g. "raw/AAPL_raw.csv")
    extra_args : optional dict passed to boto3 upload_file ExtraArgs
                 e.g. {"ContentType": "text/html", "ACL": "public-read"}

    Returns
    -------
    str: the full S3 URI s3://<bucket>/<s3_key>
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    s3 = _client()
    s3.upload_file(
        Filename=str(local_path),
        Bucket=bucket,
        Key=s3_key,
        ExtraArgs=extra_args or {},
    )
    uri = f"s3://{bucket}/{s3_key}"
    logger.info("Uploaded %s to %s", local_path.name, uri)
    return uri


def download_file(
    bucket: str,
    s3_key: str,
    local_path: str | Path,
) -> Path:
    """
    Download a single file from S3 to a local path.

    Parameters
    ----------
    bucket     : S3 bucket name
    s3_key     : source key inside the bucket
    local_path : destination path on the local filesystem

    Returns
    -------
    Path: resolved path to the downloaded file
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    s3 = _client()
    s3.download_file(Bucket=bucket, Key=s3_key, Filename=str(local_path))
    logger.info("Downloaded s3://%s/%s to %s", bucket, s3_key, local_path)
    return local_path


def list_objects(bucket: str, prefix: str = "") -> list[dict]:
    """
    List objects in a bucket under the given prefix.

    Returns a list of dicts with keys: Key, Size, LastModified.
    """
    s3 = _client()
    paginator = s3.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects.append({
                "Key":          obj["Key"],
                "Size":         obj["Size"],
                "LastModified": obj["LastModified"].isoformat(),
            })
    return objects


def upload_pipeline_outputs(
    bucket: str,
    data_dir: str | Path,
    rpt_dir: str | Path,
    make_dashboard_public: bool = False,
) -> dict[str, list[str]]:
    """
    Bulk-upload all pipeline artifacts to S3 after a pipeline run.

    Uploads
    -------
    data_dir/*_raw.csv          to s3://<bucket>/raw/
    data_dir/*_processed.csv    to s3://<bucket>/processed/
    data_dir/*.json             to s3://<bucket>/json/
    rpt_dir/dashboard.html      to s3://<bucket>/dashboard/

    Parameters
    ----------
    bucket               : destination S3 bucket name
    data_dir             : local data/ directory (Path or str)
    rpt_dir              : local reports/ directory (Path or str)
    make_dashboard_public: if True, sets dashboard.html ACL to public-read
                           so it can be opened via the S3 website endpoint

    Returns
    -------
    dict: {"uploaded": [list of S3 URIs], "skipped": [list of missing files]}
    """
    data_dir = Path(data_dir)
    rpt_dir  = Path(rpt_dir)

    uploaded: list[str] = []
    skipped:  list[str] = []

    upload_map = [
        # (glob_pattern,  s3_prefix,   extra_args)
        ("*_raw.csv",       "raw/",       {}),
        ("*_processed.csv", "processed/", {}),
        ("*.json",          "json/",      {}),
    ]

    print("\n[AWS S3] Uploading pipeline outputs to", f"s3://{bucket}/")

    for pattern, prefix, args in upload_map:
        for f in sorted(data_dir.glob(pattern)):
            key = prefix + f.name
            try:
                uri = upload_file(f, bucket, key, extra_args=args or None)
                uploaded.append(uri)
                print(f"  OK {key}")
            except Exception as exc:
                skipped.append(str(f))
                print(f"  FAIL {f.name}: {exc}")

    # Dashboard HTML
    dash = rpt_dir / "dashboard.html"
    if dash.exists():
        dash_args: dict = {"ContentType": "text/html"}
        if make_dashboard_public:
            dash_args["ACL"] = "public-read"
        try:
            uri = upload_file(dash, bucket, "dashboard/dashboard.html", extra_args=dash_args)
            uploaded.append(uri)
            print(f"  OK dashboard/dashboard.html"
                  + (" [public]" if make_dashboard_public else ""))
        except Exception as exc:
            skipped.append(str(dash))
            print(f"  FAIL dashboard.html: {exc}")
    else:
        skipped.append(str(dash))
        print("  SKIP dashboard.html not found (run pipeline first)")

    print(f"\n  {len(uploaded)} file(s) uploaded | {len(skipped)} skipped")
    return {"uploaded": uploaded, "skipped": skipped}
