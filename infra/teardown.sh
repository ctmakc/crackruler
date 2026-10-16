#!/usr/bin/env bash
# Remove a stack created by deploy.sh (its reports bucket is emptied first).
#   main stack:      infra/teardown.sh
#   A/B x86 twin:    STACK=crackruler-x86 SUFFIX=-x86 infra/teardown.sh   (keeps the shared artifact bucket)
set -euo pipefail
REGION=${AWS_REGION:-us-east-1}
STACK=${STACK:-crackruler}
SUFFIX=${SUFFIX:-}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
aws s3 rm --recursive "s3://crackruler-reports${SUFFIX}-${ACCOUNT}-${REGION}" || true
aws cloudformation delete-stack --region "$REGION" --stack-name "$STACK"
aws cloudformation wait stack-delete-complete --region "$REGION" --stack-name "$STACK"
if [ -z "$SUFFIX" ]; then aws s3 rb --force "s3://crackruler-artifacts-${ACCOUNT}-${REGION}" || true; fi
echo "removed $STACK"
