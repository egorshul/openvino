#!/usr/bin/env bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Prepare all three models (SDXL, Whisper large v3, Llama 3.1 8B) as OpenVINO
# 2026.1 FP16 IRs for static / NPU execution, then package them.
#
# Prerequisites:
#   * pip install -r requirements.txt   (inside a fresh venv)
#   * For Llama (gated): accept the license on HuggingFace and
#       export HF_TOKEN=<your token>
#   * Disk: ~60 GB free.  RAM: >=32 GB recommended (Llama 8B FP16 export).
set -euo pipefail
cd "$(dirname "$0")"

OUT_ROOT="${OUT_ROOT:-models}"
mkdir -p "$OUT_ROOT"

# Set MLPERF=1 to align to the MLPerf Inference Closed/Datacenter reference
# (pinned checkpoint commits + GREEDY decoding, no beam search).
MLP="${MLPERF:+--mlperf}"
[[ -n "$MLP" ]] && echo "### MLPerf alignment ENABLED (pinned commits + greedy)"

echo "### 1/3  Stable Diffusion XL (fully static FP16)"
python export_sdxl.py    $MLP --output "$OUT_ROOT/sdxl-base-1.0-ov-fp16-static" --overwrite

echo "### 2/3  Whisper large v3 (FP16, beam_idx)"
python export_whisper.py $MLP --output "$OUT_ROOT/whisper-large-v3-ov-fp16"     --overwrite

echo "### 3/3  Llama 3.1 8B Instruct (stateful FP16, beam_idx)"
if [[ -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]]; then
  echo ">>> HF_TOKEN not set: skipping gated Llama. Set it and re-run export_llm.py."
else
  python export_llm.py   $MLP --output "$OUT_ROOT/llama-3.1-8b-instruct-ov-fp16" --overwrite
fi

echo "### Verifying shapes"
python verify_static.py "$OUT_ROOT"/*-ov-*

echo "### Packaging archive"
./make_archive.sh "$OUT_ROOT"
echo "Done."
