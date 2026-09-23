"""S3-compatible object storage helpers for Filebase artifact delivery."""

from __future__ import annotations

import os
from pathlib import Path

import boto3
from botocore.client import BaseClient

DEFAULT_FILEBASE_ENDPOINT = "https://s3.filebase.com"
DEFAULT_FILEBASE_REGION = "us-east-1"


def get_object_storage_config() -> dict[str, str]:
    """Read non-secret storage settings and require credentials at runtime."""
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    bucket = os.getenv("FILEBASE_S3_BUCKET")
    if not access_key or not secret_key:
        raise ValueError(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required "
            "for object storage"
        )
    if not bucket:
        raise ValueError("FILEBASE_S3_BUCKET is required for object storage")

    return {
        "endpoint_url": os.getenv("FILEBASE_S3_ENDPOINT", DEFAULT_FILEBASE_ENDPOINT),
        "region_name": os.getenv("FILEBASE_S3_REGION", DEFAULT_FILEBASE_REGION),
        "bucket": bucket,
    }


def get_s3_client() -> BaseClient:
    """Create an S3-compatible client without logging credentials."""
    config = get_object_storage_config()
    return boto3.client(
        "s3",
        endpoint_url=config["endpoint_url"],
        region_name=config["region_name"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def upload_file(local_path: str | Path, object_key: str) -> None:
    """Upload a local file to the configured artifact bucket."""
    path = Path(local_path)
    if not path.is_file():
        raise FileNotFoundError(f"Artifact file does not exist: {path}")
    config = get_object_storage_config()
    get_s3_client().upload_file(str(path), config["bucket"], object_key)


def download_file(object_key: str, local_path: str | Path) -> Path:
    """Download an artifact from the configured bucket."""
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = get_object_storage_config()
    get_s3_client().download_file(config["bucket"], object_key, str(path))
    return path
