#!/usr/bin/env bash
# Remove everything the deploy created (reports bucket is emptied first).
set -euo pipefail
REGION=${AWS_REGION:-us-east-1}
STACK=${STACK:-crackruler}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
aws s3 rm --recursive "s3://crackruler-reports-${ACCOUNT}-${REGION}" || true
aws cloudformation delete-stack --region "$REGION" --stack-name "$STACK"
aws cloudformation wait stack-delete-complete --region "$REGION" --stack-name "$STACK"
aws s3 rb --force "s3://crackruler-artifacts-${ACCOUNT}-${REGION}" || true
echo "removed"
