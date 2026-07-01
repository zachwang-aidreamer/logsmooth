#!/usr/bin/env bash
#
# One-line LogSmooth quantization example for Qwen3.5-35B-A13B.
# SPDX-License-Identifier: MIT
#
#   bash examples/run_qwen35_35b_logsmooth.sh
#
# Applies online R1 Hadamard rotation + LogSmooth (weight-side static D_group +
# activation-side dynamic per-channel sidecar) and quantizes attention + MoE
# (shared_expert AND routed experts) to MXFP4, then exports a quark checkpoint.
#
# Override any of these via env, e.g.:
#   MODEL=/path/to/local/qwen35-35b OUTPUT_DIR=./out bash examples/run_qwen35_35b_logsmooth.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3.5-35B-A13B-FP8}"
ROTATION_CONFIG="${ROTATION_CONFIG:-${SCRIPT_DIR}/qwen35_35b_a13b_logsmooth_rotation.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/qwen35_35b_a13b_logsmooth}"
DEVICE="${DEVICE:-cuda}"
DATASET="${DATASET:-pileval}"
NUM_CALIB_DATA="${NUM_CALIB_DATA:-128}"
SEQ_LEN="${SEQ_LEN:-512}"
WEIGHT_FORMAT="${WEIGHT_FORMAT:-real_quantized}"

mkdir -p "${OUTPUT_DIR}"

python "${SCRIPT_DIR}/quantize.py" \
    --model "${MODEL}" \
    --rotation_config "${ROTATION_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --dataset "${DATASET}" \
    --num_calib_data "${NUM_CALIB_DATA}" \
    --seq_len "${SEQ_LEN}" \
    --weight_format "${WEIGHT_FORMAT}"
