#!/usr/bin/env bash
# Benchmark every model of the Stable Diffusion XL pipeline with OpenVINO benchmark_app
# on the CPU device (tuned for ARM Cortex-A75 builds of OpenVINO).
#
# Downloads pre-converted OpenVINO IR models from Hugging Face
# (default: OpenVINO/stable-diffusion-xl-base-1.0-fp16-ov) and runs benchmark_app
# on each inference model of the pipeline:
#   text_encoder, text_encoder_2, unet, vae_encoder, vae_decoder
# (tokenizer/tokenizer_2 are not benchmarked - they require the openvino_tokenizers
#  extension and contribute ~0% of pipeline time)
#
# Usage:
#   ./scripts/arm-cortex-a75/benchmark_sdxl.sh [path/to/benchmark_app]
#
# Tunables (environment variables):
#   BENCHMARK_APP   - path to benchmark_app binary (or first positional argument)
#   MODELS_DIR      - where to store models (default: ./sdxl-ov-models)
#   MODEL_REPO      - HF repo with OV IRs (default: OpenVINO/stable-diffusion-xl-base-1.0-fp16-ov;
#                     for the fastest variant on Cortex-A75 try
#                     OpenVINO/stable-diffusion-xl-base-1.0-int8-ov - int8 uses the
#                     ASIMD dot-product (asimddp) kernels of ACL/KleidiAI)
#   HINT            - latency | throughput (default: latency)
#   INFER_PRECISION - f16 | f32 (default: f16 - Cortex-A75 has native fp16 ASIMD)
#   RESOLUTION      - image resolution (default: 1024, latent = RESOLUTION/8)
#   ONLY            - space-separated subset of models to run (default: all)
#
# Benchmarked shapes correspond to a real SDXL generation at RESOLUTION x RESOLUTION
# with classifier-free guidance (UNet batch = 2).

set -euo pipefail

BENCHMARK_APP="${1:-${BENCHMARK_APP:-benchmark_app}}"
MODELS_DIR="${MODELS_DIR:-$(pwd)/sdxl-ov-models}"
MODEL_REPO="${MODEL_REPO:-OpenVINO/stable-diffusion-xl-base-1.0-fp16-ov}"
HINT="${HINT:-latency}"
INFER_PRECISION="${INFER_PRECISION:-f16}"
RESOLUTION="${RESOLUTION:-1024}"
LATENT=$((RESOLUTION / 8))
ONLY="${ONLY:-text_encoder text_encoder_2 unet vae_encoder vae_decoder}"

if ! command -v "${BENCHMARK_APP}" >/dev/null 2>&1 && [[ ! -x "${BENCHMARK_APP}" ]]; then
    echo "error: benchmark_app not found: ${BENCHMARK_APP}" >&2
    echo "Pass it as the first argument or via BENCHMARK_APP env var." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Download models
# ---------------------------------------------------------------------------
BASE_URL="https://huggingface.co/${MODEL_REPO}/resolve/main"
mkdir -p "${MODELS_DIR}"
for model in text_encoder text_encoder_2 unet vae_encoder vae_decoder; do
    mkdir -p "${MODELS_DIR}/${model}"
    for f in openvino_model.xml openvino_model.bin; do
        dst="${MODELS_DIR}/${model}/${f}"
        if [[ ! -s "${dst}" ]]; then
            echo ">> downloading ${model}/${f}"
            curl -fL --retry 4 --retry-delay 2 -C - "${BASE_URL}/${model}/${f}" -o "${dst}"
        fi
    done
done

# UNet 'text_embeds' [N,1280] and 'time_ids' [N,6] inputs may be exported under
# generated names (Parameter_NNN) - resolve them from the IR by their signature.
resolve_unet_aux_inputs() {
    python3 - "$MODELS_DIR/unet/openvino_model.xml" <<'EOF'
import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
embeds = time_ids = None
for l in root.iter('layer'):
    if l.get('type') != 'Parameter':
        continue
    dims = [d.text for d in l.findall('output/port/dim')]
    if dims[1:] == ['1280']:
        embeds = l.get('name')
    elif dims[1:] == ['6']:
        time_ids = l.get('name')
print(embeds, time_ids)
EOF
}

read -r UNET_EMBEDS UNET_TIME_IDS < <(resolve_unet_aux_inputs)

# ---------------------------------------------------------------------------
# Static shapes of one real SDXL inference step (CFG => UNet batch 2)
# ---------------------------------------------------------------------------
declare -A SHAPES NITER
SHAPES[text_encoder]="input_ids[1,77]"
SHAPES[text_encoder_2]="input_ids[1,77]"
SHAPES[unet]="sample[2,4,${LATENT},${LATENT}],timestep[1],encoder_hidden_states[2,77,2048],${UNET_EMBEDS}[2,1280],${UNET_TIME_IDS}[2,6]"
SHAPES[vae_encoder]="sample[1,3,${RESOLUTION},${RESOLUTION}]"
SHAPES[vae_decoder]="latent_sample[1,4,${LATENT},${LATENT}]"
NITER[text_encoder]="${NITER_OVERRIDE:-10}"
NITER[text_encoder_2]="${NITER_OVERRIDE:-10}"
NITER[unet]="${NITER_OVERRIDE:-3}"
NITER[vae_encoder]="${NITER_OVERRIDE:-3}"
NITER[vae_decoder]="${NITER_OVERRIDE:-3}"

# ---------------------------------------------------------------------------
# Run benchmark_app on every model
# ---------------------------------------------------------------------------
declare -A RESULT
FAILED=0
for model in ${ONLY}; do
    xml="${MODELS_DIR}/${model}/openvino_model.xml"
    echo
    echo "=================================================================="
    echo "== benchmark_app: ${model}  (${SHAPES[$model]})"
    echo "=================================================================="
    if "${BENCHMARK_APP}" \
            -m "${xml}" \
            -d CPU \
            -hint "${HINT}" \
            -infer_precision "${INFER_PRECISION}" \
            -shape "${SHAPES[$model]}" \
            -niter "${NITER[$model]}" \
            -report_type no_counters \
            -report_folder "${MODELS_DIR}/${model}" 2>&1 | tee "${MODELS_DIR}/${model}/benchmark.log"; then
        RESULT[$model]=$(grep -E "Throughput|Median" "${MODELS_DIR}/${model}/benchmark.log" | tr -d ' ' | paste -sd' ' -)
    else
        RESULT[$model]="FAILED"
        FAILED=1
    fi
done

echo
echo "================== SDXL pipeline benchmark summary =================="
echo "OpenVINO build: $("${BENCHMARK_APP}" --help 2>/dev/null | grep -m1 -oE "Build.*" || true)"
echo "Hint: ${HINT}, inference precision: ${INFER_PRECISION}, resolution: ${RESOLUTION}"
for model in ${ONLY}; do
    printf "  %-16s %s\n" "${model}:" "${RESULT[$model]}"
done
exit ${FAILED}
