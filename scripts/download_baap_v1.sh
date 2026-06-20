#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-chenyuming052/BAAP}"
TAG="${TAG:-baap-v1.0}"
OUT_DIR="${1:-checkpoints/baap-v1.0}"
BASE_URL="https://github.com/${REPO}/releases/download/${TAG}"

mkdir -p "${OUT_DIR}"

for name in \
  baap-v1-paperbest.ckpt \
  SHA256SUMS
do
  curl -L "${BASE_URL}/${name}" -o "${OUT_DIR}/${name}"
done

(
  cd "${OUT_DIR}"
  sha256sum -c SHA256SUMS
)

echo "BAAP v1 checkpoint downloaded to ${OUT_DIR}/baap-v1-paperbest.ckpt"
