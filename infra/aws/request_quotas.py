#!/usr/bin/env python3
"""Request the SageMaker quotas the pipeline needs in us-east-1: training-job instances.

A full run peaks at about 45 GB of RAM, so it needs a training-job quota for a large
instance, and new accounts start at 0 for every instance type. Requests already open are
skipped. SageMaker increases for new accounts can go to manual review (status CASE_OPENED);
follow those in the console's Support Center.

    python infra/aws/request_quotas.py            # dry run: current vs wanted
    python infra/aws/request_quotas.py --apply     # file the increases
    python infra/aws/request_quotas.py --status    # requests already filed and their status
"""
from __future__ import annotations

import argparse
import sys

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"

# Matched on the exact QuotaName so we never hardcode a per-instance-type code.
WANTED: dict[str, float] = {
    "ml.m5.4xlarge for training job usage": 1,   # 16 vCPU / 64 GB: a full run
    "ml.m5.xlarge for training job usage": 1,    # 4 vCPU / 16 GB: smoke runs (--sample 0.01)
    "ml.r5.4xlarge for training job usage": 1,   # 16 vCPU / 128 GB: headroom for deeper blocking (more candidates)
}


def all_quotas(sq) -> list[dict]:
    quotas: list[dict] = []
    for page in sq.get_paginator("list_service_quotas").paginate(ServiceCode="sagemaker"):
        quotas.extend(page["Quotas"])
    return quotas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually file the increase requests")
    ap.add_argument("--status", action="store_true", help="show requests already filed")
    args = ap.parse_args()

    sq = boto3.client("service-quotas", region_name=REGION)

    if args.status:
        hist = sq.list_requested_service_quota_change_history(ServiceCode="sagemaker")
        rows = hist.get("RequestedQuotas", [])
        if not rows:
            print(f"No quota increase requests on record for sagemaker in {REGION}.")
            return 0
        for r in sorted(rows, key=lambda x: x["Created"], reverse=True):
            print(f"{r['Status']:<12} {r['DesiredValue']:>8}  {r['QuotaName']}")
        return 0

    by_name = {q["QuotaName"]: q for q in all_quotas(sq)}

    missing = [n for n in WANTED if n not in by_name]
    if missing:
        print("These quota names did not resolve (AWS may have renamed them):", file=sys.stderr)
        for n in missing:
            print(f"  - {n}", file=sys.stderr)

    todo = []
    for name, desired in WANTED.items():
        q = by_name.get(name)
        if q is None:
            continue
        current = q["Value"]
        if not q.get("Adjustable", False):
            print(f"  SKIP  {name}: not adjustable (fixed at {current})")
            continue
        if current >= desired:
            print(f"  OK    {name}: already {current:g} (>= {desired:g})")
            continue
        todo.append((name, q["QuotaCode"], current, desired))

    if not todo:
        print(f"\nNothing to request -- every wanted quota is already satisfied in {REGION}.")
        return 0

    print(f"\n{len(todo)} quota(s) need an increase in {REGION}:")
    for name, code, current, desired in todo:
        print(f"  {code}  {current:g} -> {desired:g}   {name}")

    if not args.apply:
        print("\nDry run. Re-run with --apply to file these requests.")
        return 0

    print()
    failed = 0
    for name, code, _current, desired in todo:
        try:
            r = sq.request_service_quota_increase(
                ServiceCode="sagemaker", QuotaCode=code, DesiredValue=desired
            )
            case = r["RequestedQuota"].get("CaseId") or "(no case id)"
            print(f"  filed   {name} -> {desired:g}   case={case}")
        except ClientError as e:
            code_str = e.response["Error"]["Code"]
            if code_str == "ResourceAlreadyExistsException":
                print(f"  pending {name}: a request is already open")
            else:
                print(f"  FAILED  {name}: {code_str}: {e.response['Error']['Message']}")
                failed += 1

    print(
        "\nSageMaker quota increases are usually approved in minutes to a few hours."
        "\nCheck with: python infra/aws/request_quotas.py --status"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
