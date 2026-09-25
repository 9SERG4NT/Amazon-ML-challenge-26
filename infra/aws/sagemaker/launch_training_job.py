#!/usr/bin/env python3
"""Run the full entity-resolution pipeline as a SageMaker training job.

Packages code/business_entity_resolution/src with the pinned requirements and the
official submission validator, uploads it to S3, and starts a training job on the
AWS PyTorch CPU container (Python 3.12, the same version as local runs). The job reads
the dataset from S3 and writes:
  s3://<bucket>/experiments/sagemaker/<job>/output/output.tar.gz   submission TSVs + metrics.json
  s3://<bucket>/experiments/sagemaker/<job>/output/model.tar.gz    fold models + metrics.json
The code bundle goes to code/sagemaker/<job>/, which the bucket's lifecycle rule deletes after 7 days.

    python infra/aws/sagemaker/launch_training_job.py --dry-run
    python infra/aws/sagemaker/launch_training_job.py --sample 0.01 --instance ml.m5.xlarge --wait   # smoke run
    python infra/aws/sagemaker/launch_training_job.py --wait                                          # full run

The account needs a quota above 0 for "<instance> for training job usage"; the script
checks before it launches. A full run needs about 30 GB of memory (ml.m5.4xlarge: 64 GB).
"""
from __future__ import annotations

import argparse
import io
import sys
import tarfile
import time
from pathlib import Path

import boto3

REGION = "us-east-1"
ACCOUNT = "125650147728"
BUCKET = f"amazon-ml-challenge-26-sagemaker-{ACCOUNT}"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/service-role/AmazonSageMaker-ExecutionRole-20260925T163023"
IMAGE = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.7.1-cpu-py312-ubuntu22.04-sagemaker"

REPO = Path(__file__).resolve().parents[3]
PROJECT = REPO / "code" / "business_entity_resolution"
VALIDATOR = REPO / "resources" / "student_resource" / "utils" / "validate_submission.py"


def sourcedir_tarball() -> bytes:
    """src/ at the archive root, plus requirements.txt (installed by the container) and the validator."""
    buf = io.BytesIO()
    skip = lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted((PROJECT / "src").iterdir()):
            tar.add(p, arcname=p.name, filter=skip)
        tar.add(PROJECT / "requirements.txt", arcname="requirements.txt")
        if VALIDATOR.exists():
            tar.add(VALIDATOR, arcname="validate_submission.py")
    return buf.getvalue()


def training_quota(instance: str) -> float | None:
    sq = boto3.client("service-quotas", region_name=REGION)
    name = f"{instance} for training job usage"
    for page in sq.get_paginator("list_service_quotas").paginate(ServiceCode="sagemaker"):
        for q in page["Quotas"]:
            if q["QuotaName"] == name:
                return q["Value"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="ml.m5.4xlarge")
    ap.add_argument("--volume-gb", type=int, default=150)
    ap.add_argument("--max-hours", type=float, default=12)
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of the data (use 0.01 for a smoke run)")
    ap.add_argument("--stages", default=None, help="comma-separated pipeline stages (default: all)")
    ap.add_argument("--preset", default=None, help="end-to-end version from ber/versions.py (default: its DEFAULT_PRESET)")
    ap.add_argument("--wait", action="store_true", help="poll until the job finishes")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    quota = training_quota(args.instance)
    print(f"quota '{args.instance} for training job usage': {quota}")
    if not quota:
        print("Blocked: this instance type has no training quota yet. Request one with infra/aws/request_quotas.py "
              "or pick an instance type whose quota is above 0.")
        return 1

    job = f"mlc26-ber-{time.strftime('%Y%m%d-%H%M%S')}"
    prefix = f"experiments/sagemaker/{job}"   # outputs
    code_key = f"code/sagemaker/{job}/sourcedir.tar.gz"  # code/ expires after 7 days (bucket lifecycle rule)
    hp = {"sagemaker_program": "sagemaker_entry.py", "sagemaker_submit_directory": f"s3://{BUCKET}/{code_key}",
          "sagemaker_region": REGION, "sample": str(args.sample)}
    if args.stages:
        hp["stages"] = args.stages
    if args.preset:
        hp["preset"] = args.preset
    tarball = sourcedir_tarball()
    print(f"job {job}: {args.instance}, {args.volume_gb} GB, up to {args.max_hours} h; code {len(tarball) / 1e3:.0f} kB")
    if args.dry_run:
        print("Dry run: nothing uploaded or started.")
        return 0

    boto3.client("s3", region_name=REGION).put_object(Bucket=BUCKET, Key=code_key, Body=tarball)
    sm = boto3.client("sagemaker", region_name=REGION)
    sm.create_training_job(
        TrainingJobName=job,
        RoleArn=ROLE,
        AlgorithmSpecification={"TrainingImage": IMAGE, "TrainingInputMode": "File"},
        HyperParameters=hp,
        InputDataConfig=[{"ChannelName": "dataset", "InputMode": "File", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": f"s3://{BUCKET}/dataset/", "S3DataDistributionType": "FullyReplicated"}}}],
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/{prefix}/output"},
        ResourceConfig={"InstanceType": args.instance, "InstanceCount": 1, "VolumeSizeInGB": args.volume_gb},
        StoppingCondition={"MaxRuntimeInSeconds": int(args.max_hours * 3600)},
        Tags=[{"Key": "Project", "Value": "amazon-ml-challenge-26"}],
    )
    print(f"started. Job:     https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs/{job}")
    print(f"         Outputs: https://{REGION}.console.aws.amazon.com/s3/buckets/{BUCKET}?region={REGION}&prefix={prefix}/")
    if not args.wait:
        return 0
    last = None
    while True:
        d = sm.describe_training_job(TrainingJobName=job)
        status = (d["TrainingJobStatus"], d.get("SecondaryStatus"))
        if status != last:
            print(f"  {time.strftime('%H:%M:%S')} {status[0]} / {status[1]}", flush=True)
            last = status
        if d["TrainingJobStatus"] in ("Completed", "Failed", "Stopped"):
            if d.get("FailureReason"):
                print("failure:", d["FailureReason"])
            print("artifacts:", d.get("ModelArtifacts", {}).get("S3ModelArtifacts"))
            return 0 if d["TrainingJobStatus"] == "Completed" else 1
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
