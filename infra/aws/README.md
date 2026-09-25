# AWS setup

What exists in AWS for this project, how to run the pipeline there, and what it costs.
Account `125650147728`, region `us-east-1`.

## Resources

| Resource | ID / location | Notes |
|---|---|---|
| S3 bucket | `s3://amazon-ml-challenge-26-sagemaker-125650147728` | [console](https://us-east-1.console.aws.amazon.com/s3/buckets/amazon-ml-challenge-26-sagemaker-125650147728?region=us-east-1&tab=objects) |
| SageMaker domain | `d-tgxrzl4mojbh` (QuickSetupDomain-20260925T163022) | [console](https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/studio/d-tgxrzl4mojbh) · profile `Sumukh`, space `test` |
| SageMaker execution role | `AmazonSageMaker-ExecutionRole-20260925T163023` | reads and writes the bucket (checked by IAM simulation) |
| EC2 runner | `i-033e809bc1d8c21b5`, r7a.2xlarge | [console](https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#InstanceDetails:instanceId=i-033e809bc1d8c21b5) · interim compute, see below |
| EC2 role / security group | `mlc26-ec2-runner` / `sg-02c6cc9396979636b` | bucket read/write + SSM only; no inbound rules |
| Budget | `mlc26-monthly-50usd` | $50/month, email alerts at 50% and 100% |

An older, empty Quick Setup domain `d-d4wvkyftnhlu` exists in eu-north-1; it costs nothing
while it has no apps and can be deleted. The teammate setup notes that mention account
`911797456769` describe a different account.

### Bucket layout

The bucket keeps the data and the results of runs, nothing else:

| Prefix | Contents |
|---|---|
| `dataset/` | the 7 competition TSVs |
| `experiments/` | run logs and `metrics.json` files |
| `predictions/<version>/` | `matching_results.tsv` + `candidate_pairs.tsv` per run |
| `models/<version>/` | trained LightGBM fold models |
| `code/` | short-lived code bundles that the EC2 runner and SageMaker jobs pull; **deleted automatically after 7 days** |

Encryption AES256, all public access blocked, TLS-only bucket policy, incomplete multipart
uploads aborted after 7 days. Versioning is off (deletes are permanent).

## Compute: where the pipeline runs

A full run peaks at about 45 GB of RAM, so it needs a large machine.

**SageMaker training job (target).** `sagemaker/launch_training_job.py` packages
`code/business_entity_resolution/src`, the pinned requirements and the submission validator,
and runs `src/sagemaker_entry.py` on the AWS PyTorch CPU container (Python 3.12). It checks the
instance quota before launching and prints the job's console link.

```bash
python infra/aws/sagemaker/launch_training_job.py --dry-run
python infra/aws/sagemaker/launch_training_job.py --sample 0.01 --instance ml.m5.xlarge --wait   # smoke run
python infra/aws/sagemaker/launch_training_job.py --preset M-v3 --wait                           # full run, ml.m5.4xlarge
```

This needs a training quota above 0 for the instance type. Status on 2026-09-25: requests for
ml.m5.4xlarge, ml.m5.xlarge and ml.g5.2xlarge training (and ml.g5.xlarge processing, Studio
JupyterLab on ml.r5.4xlarge / ml.g5.xlarge) are **under AWS review** (`CASE_OPENED`; follow them
in the console's Support Center). Check with `python infra/aws/request_quotas.py --status`.

**Studio.** The `test` space runs JupyterLab on ml.t3.medium (4 GB RAM, 5 GB disk): fine for
notebooks and looking at results, too small for the pipeline. It has no idle shutdown, so stop
the app from Studio when you are done.

**EC2 runner (interim, until the SageMaker quota is approved).** r7a.2xlarge (8 cores, 64 GB,
about $0.61/hour), Amazon Linux 2023, 200 GB disk. Bootstrap (`ec2/user_data.sh`) installs
Python 3.12 with the pinned dependencies and copies the dataset to `/data/dataset`. It is driven
with SSM Run Command (no SSH, no open ports). A systemd timer **stops** the instance after 60
minutes with load below 0.3, so an idle instance only costs its disk (~$0.50/day); start it again
from the console. Terminate it once SageMaker takes over.

On the instance: Python in `/opt/mlc26/venv`, code unpacked from `code/*.tar.gz` into
`/opt/mlc26/pipe*/`, runs in `/opt/mlc26/runs/`, logs in `/opt/mlc26/logs/`.

## Credentials

The CLI signs in with `aws login` as the root user. The credentials are short-lived (15-minute
rotation, 12-hour maximum) and root has no access keys, but every command runs with unrestricted
privileges. Recommended: create an IAM user with [iam/ml-engineer-policy.json](iam/ml-engineer-policy.json)
plus the AWS-managed `SignInLocalDevelopmentAccess` policy and sign in as that user with
`aws login`, and turn on MFA for the root user.

Local Python (boto3) cannot use `aws login` credentials until you run
`pip install "botocore[crt]"`.

## Quotas

`request_quotas.py` files quota increases by quota name (`--apply`) and shows the status of
requests already filed (`--status`). Approved on 2026-09-25: Studio user profiles 2, domains 2,
running Studio apps 40. Everything else listed above is pending; all other SageMaker training
and processing quotas are 0. The EC2 on-demand standard vCPU quota is 8.
