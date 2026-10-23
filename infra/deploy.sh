#!/usr/bin/env bash
# Deploy CrackRuler to AWS: build zip -> upload to an artifact bucket -> CloudFormation.
# Needs AWS credentials with infra/iam-deployer-policy.json (or broader).
#   AWS_REGION   default us-east-1
#   ALARM_EMAIL  optional, subscribes an e-mail to the error alarm
#   ARCH=x86_64 SUFFIX=-x86 STACK=crackruler-x86  -> side-by-side x86 stack for the A/B benchmark
set -euo pipefail
cd "$(dirname "$0")/.."
REGION=${AWS_REGION:-us-east-1}
STACK=${STACK:-crackruler}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ART="crackruler-artifacts-${ACCOUNT}-${REGION}"

ARCH=${ARCH:-arm64}
SUFFIX=${SUFFIX:-}
ZIP=$(ARCH=$ARCH infra/build.sh | tail -1)
# content hash in the key: a rebuilt zip always reaches the function, even without a new commit
KEY="lambda/$(basename "$ZIP" .zip)-$(sha256sum "$ZIP" | cut -c1-10).zip"

if ! aws s3api head-bucket --bucket "$ART" 2>/dev/null; then
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$ART" --region "$REGION" >/dev/null
  else
    aws s3api create-bucket --bucket "$ART" --region "$REGION" \
      --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
  fi
  aws s3api put-public-access-block --bucket "$ART" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  aws s3api put-bucket-lifecycle-configuration --bucket "$ART" --lifecycle-configuration \
    '{"Rules":[{"ID":"old-builds","Status":"Enabled","Filter":{"Prefix":"lambda/"},"Expiration":{"Days":30}}]}'
fi
aws s3 cp --only-show-errors "$ZIP" "s3://$ART/$KEY"

aws cloudformation deploy --region "$REGION" --stack-name "$STACK" \
  --template-file infra/template.yaml --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides CodeBucket="$ART" CodeKey="$KEY" AlarmEmail="${ALARM_EMAIL:-}" \
    Architecture="$ARCH" FunctionSuffix="$SUFFIX" MemorySize="${MEMORY:-512}" \
  --no-fail-on-empty-changeset

URL=$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='EndpointUrl'].OutputValue" --output text)
echo "Endpoint: $URL"
curl -fsS "${URL%/}/api/health" && echo
