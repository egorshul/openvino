#!/usr/bin/env bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Package the produced model directories into a single compressed archive plus
# a manifest with sizes and sha256 sums.
#
# Usage: ./make_archive.sh [MODELS_DIR] [ARCHIVE_NAME]
set -euo pipefail

MODELS_DIR="${1:-models}"
ARCHIVE="${2:-openvino-2026.1-npu-models-fp16.tar.gz}"

if [[ ! -d "$MODELS_DIR" ]]; then
  echo "No such directory: $MODELS_DIR" >&2
  exit 1
fi

MANIFEST="$MODELS_DIR/MANIFEST.txt"
{
  echo "OpenVINO 2026.1 static / NPU model archive (FP16)"
  echo "generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "openvino: $(python -c 'import openvino,sys; sys.stdout.write(openvino.get_version())' 2>/dev/null || echo unknown)"
  echo
  echo "contents:"
  du -sh "$MODELS_DIR"/*/ 2>/dev/null || true
} > "$MANIFEST"
cat "$MANIFEST"

echo "Creating $ARCHIVE ..."
tar -czf "$ARCHIVE" -C "$(dirname "$MODELS_DIR")" "$(basename "$MODELS_DIR")"
sha256sum "$ARCHIVE" | tee "${ARCHIVE}.sha256"
echo "Archive ready: $ARCHIVE ($(du -sh "$ARCHIVE" | cut -f1))"
