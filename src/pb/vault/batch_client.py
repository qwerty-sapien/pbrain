# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Vertex Batch transport layer — shared by SocraticService and scaffold_generator.

Owns ONLY: JSONL line construction, GCS upload, batch job submit/poll, result parse.
Does NOT own: note plans, system instructions, learning profiles, retry manifests.
Callers build request dicts and pass them in.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import structlog

from pb.storage.config import get_config

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def build_batch_line(
    key: str,
    prompt: str,
    system_instruction: str,
    temperature: float = 0.4,
    max_output_tokens: int = 4000,
) -> dict:
    """Build a single Vertex Batch request dict. Caller json-encodes if writing JSONL."""
    return {
        "key": key,
        "request": {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        },
    }


class VertexBatchClient:
    """Thin wrapper over google-genai Vertex Batch + GCS for submit/poll/download."""

    def __init__(self, gcp_project: str, gcs_bucket: str, location: str = "us-central1"):
        self.gcp_project = gcp_project
        self.gcs_bucket = gcs_bucket
        self.location = location
        self._log = structlog.get_logger()

    def _validate_config(self) -> None:
        if not self.gcp_project or not self.gcs_bucket:
            raise ValueError(
                "GCS not configured for Vertex Batch. "
                "Run `python scripts/scaffold_generator.py` once to set up GCS bucket and GCP project, "
                "or pass `--sync` to bypass batch and create the note synchronously."
            )

    def submit(
        self,
        requests: list[dict],
        model: str,
        run_id: str,
        prefix: str = "socratic",
    ) -> str:
        """Upload requests to GCS, create batch job, return job_name."""
        self._validate_config()
        from google import genai as genai_sdk
        from google.genai.types import CreateBatchJobConfig
        from google.cloud import storage as gcs_sdk

        jsonl = "\n".join(json.dumps(r) for r in requests)
        gcs_client = gcs_sdk.Client(project=self.gcp_project)
        bucket = gcs_client.bucket(self.gcs_bucket)
        input_blob = bucket.blob(f"{prefix}/{run_id}/input.jsonl")
        input_blob.upload_from_string(jsonl, content_type="application/jsonl")
        input_uri = f"gs://{self.gcs_bucket}/{prefix}/{run_id}/input.jsonl"
        output_uri_prefix = f"gs://{self.gcs_bucket}/{prefix}/{run_id}/output/"

        genai_client = genai_sdk.Client(
            vertexai=True, project=self.gcp_project, location=self.location
        )
        job = genai_client.batches.create(
            model=model,
            src=input_uri,
            config=CreateBatchJobConfig(dest=output_uri_prefix),
        )
        self._log.info(
            "batch_client.submitted",
            job_name=job.name,
            run_id=run_id,
            prefix=prefix,
            request_count=len(requests),
        )
        return job.name

    def poll_and_download(self, job_name: str) -> tuple[list[dict], list[dict]]:
        """Poll job until terminal state, download results, return (successes, failures).

        success entry shape: {"key": str, "content": str}
        failure entry shape: {"key": str, "error": str}
        """
        self._validate_config()
        from google import genai as genai_sdk
        from google.cloud import storage as gcs_sdk

        cfg = get_config()
        poll_interval = cfg.scaffold.poll_interval_seconds or 90

        genai_client = genai_sdk.Client(
            vertexai=True, project=self.gcp_project, location=self.location
        )
        while True:
            job = genai_client.batches.get(name=job_name)
            if job.state.name in TERMINAL_STATES:
                final_state = job.state.name
                break
            time.sleep(poll_interval)

        # Reconstruct prefix from job_name -> output URI is recorded in job; we use bucket listing
        # The convention: job_name embeds the prefix/run_id; we relist by GCS prefix.
        # For deterministic behaviour, the output_uri is also accessible via job.dest. We parse it.
        try:
            output_uri = job.dest.gcs_uri  # e.g. gs://bucket/socratic/<run_id>/output/
        except AttributeError:
            output_uri = ""
        successes: list[dict] = []
        failures: list[dict] = []
        if not output_uri.startswith("gs://"):
            self._log.warning("batch_client.no_output_uri", job_name=job_name, state=final_state)
            return successes, failures

        # gs://<bucket>/<path>/
        rest = output_uri[len("gs://"):]
        bucket_name, _, path_prefix = rest.partition("/")
        gcs_client = gcs_sdk.Client(project=self.gcp_project)
        bucket = gcs_client.bucket(bucket_name)
        all_output = ""
        for blob in bucket.list_blobs(prefix=path_prefix):
            if blob.name.endswith(".jsonl"):
                all_output += blob.download_as_text() + "\n"

        for raw_line in all_output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = obj.get("key", "")
                if "response" in obj:
                    parts = obj["response"]["candidates"][0]["content"]["parts"]
                    content = "".join(p.get("text", "") for p in parts).strip()
                    successes.append({"key": key, "content": content})
                else:
                    failures.append({"key": key, "error": obj.get("error", "no response field")})
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                failures.append({"key": "", "error": f"malformed: {exc}"})

        self._log.info(
            "batch_client.downloaded",
            job_name=job_name,
            state=final_state,
            successes=len(successes),
            failures=len(failures),
        )
        return successes, failures
