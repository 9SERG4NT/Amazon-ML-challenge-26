#!/bin/bash
# Create the mlc26 batch runner in the AWS account you are logged in to (`aws login`) and start a preset on it.
#
#   bash infra/aws/ec2/launch_runner.sh [PRESET]      (from the repo root, e.g. in Git Bash; default M-v7)
#
# Creates, or reuses when they exist:
#   - bucket amazon-ml-challenge-26-<account>: private, AES256, TLS only; the 7 dataset TSVs go to dataset/
#   - role and instance profile mlc26-ec2-runner: SSM plus read/write on that bucket
#   - security group mlc26-runner: no inbound rules (the runner is driven through S3 and SSM, never SSH)
#   - an r7a.2xlarge (8 vCPU, 64 GB, 200 GB gp3, about $0.61/h) that stops itself after 60 idle minutes
# The instance bootstraps (user_data.sh), waits for the data, runs the preset and writes to the bucket:
# experiments/logs/<PRESET>.log (every 5 min), predictions/<version key>/, experiments/metrics/, models/<key>/.
set -euo pipefail
export MSYS_NO_PATHCONV=1 AWS_DEFAULT_REGION=us-east-1 AWS_PAGER=""
PRESET=${1:-M-v7}
TYPE=${TYPE:-r7a.2xlarge}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=amazon-ml-challenge-26-$ACCOUNT
ROLE=mlc26-ec2-runner
echo "account $ACCOUNT | bucket $BUCKET | preset $PRESET | $TYPE"

# 1. bucket
aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null || aws s3api create-bucket --bucket "$BUCKET" > /dev/null
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
aws s3api put-bucket-policy --bucket "$BUCKET" --policy "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Sid\":\"TLSOnly\",\
\"Effect\":\"Deny\",\"Principal\":\"*\",\"Action\":\"s3:*\",\"Resource\":[\"arn:aws:s3:::$BUCKET\",\"arn:aws:s3:::$BUCKET/*\"],\
\"Condition\":{\"Bool\":{\"aws:SecureTransport\":\"false\"}}}]}"
echo "bucket ready"

# 2. role and instance profile
NEW_ROLE=0
if ! aws iam get-role --role-name $ROLE > /dev/null 2>&1; then
  aws iam create-role --role-name $ROLE --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' > /dev/null
  aws iam attach-role-policy --role-name $ROLE --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
  aws iam create-instance-profile --instance-profile-name $ROLE > /dev/null
  aws iam add-role-to-instance-profile --instance-profile-name $ROLE --role-name $ROLE
  NEW_ROLE=1
fi
aws iam put-role-policy --role-name $ROLE --policy-name bucket-rw --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[\
{\"Effect\":\"Allow\",\"Action\":\"s3:ListBucket\",\"Resource\":\"arn:aws:s3:::$BUCKET\"},\
{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\"],\"Resource\":\"arn:aws:s3:::$BUCKET/*\"}]}"
echo "role ready"

# 3. security group without inbound rules (the default VPC's)
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=mlc26-runner --query 'SecurityGroups[0].GroupId' --output text)
if [ "$SG" = "None" ]; then
  SG=$(aws ec2 create-security-group --group-name mlc26-runner --description "mlc26 batch runner, no inbound" \
       --query GroupId --output text)
fi
echo "security group $SG"

# 4. the instance; its bootstrap waits for the data, so the upload below overlaps with it
AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
      --query Parameter.Value --output text)
UD=infra/aws/ec2/.user_data.rendered.sh
sed -e "s/^BUCKET=.*/BUCKET=$BUCKET/" -e "s/^PRESET=.*/PRESET=$PRESET/" infra/aws/ec2/user_data.sh > "$UD"
for attempt in 1 2 3 4 5 6; do  # a new instance profile takes a few seconds before EC2 accepts it
  if ID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" \
        --iam-instance-profile Name=$ROLE --security-group-ids "$SG" \
        --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}' \
        --instance-initiated-shutdown-behavior stop --metadata-options HttpTokens=required \
        --user-data "file://$UD" --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=mlc26-runner}]' \
        --query 'Instances[0].InstanceId' --output text); then
    break
  fi
  [ "$NEW_ROLE" = 1 ] && [ $attempt -lt 6 ] || { rm -f "$UD"; exit 1; }
  echo "instance profile not usable yet, retrying in 10 s"; sleep 10
done
rm -f "$UD"
echo "instance $ID launched"

# 5. the data: the 7 TSVs, 2.4 GB
aws s3 cp resources/student_resource/dataset "s3://$BUCKET/dataset/" --recursive --exclude "*" --include "*.tsv"
echo "data uploaded. $ID now bootstraps (~10 min) and runs $PRESET; log: s3://$BUCKET/experiments/logs/$PRESET.log"
