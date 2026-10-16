#!/usr/bin/env bash
# Build the Lambda deployment zip for linux/arm64 (AWS Graviton), Python 3.12.
# No Docker or emulation needed: pip fetches the manylinux aarch64 wheels directly.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
ARCH=${ARCH:-arm64}            # arm64 (Graviton, deployed) or x86_64 (local Lambda-emulator test)
PLAT=$([ "$ARCH" = "arm64" ] && echo aarch64 || echo x86_64)
rm -rf build/pkg && mkdir -p build/pkg dist
$PY -m pip install --quiet --no-compile --platform manylinux_2_28_$PLAT --platform manylinux2014_$PLAT \
    --implementation cp --python-version 3.12 --only-binary=:all: \
    --target build/pkg -r requirements.txt
# trim what Lambda never loads
find build/pkg -type d \( -name tests -o -name __pycache__ \) -prune -exec rm -rf {} +
rm -rf build/pkg/numpy/_core/include build/pkg/cv2/data
cp -r crackruler app models samples build/pkg/
find build/pkg -name '__pycache__' -prune -exec rm -rf {} +
SHA=$(git rev-parse --short HEAD 2>/dev/null || date +%s)
ZIP="dist/crackruler-${SHA}-${ARCH}.zip"
rm -f "$ZIP"
(cd build/pkg && zip -qr9 "../../$ZIP" .)
echo "unzipped: $(du -sh build/pkg | cut -f1)  zip: $(du -h "$ZIP" | cut -f1)"
echo "$ZIP"
