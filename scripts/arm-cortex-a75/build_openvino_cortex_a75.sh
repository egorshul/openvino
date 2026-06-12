#!/usr/bin/env bash
# Build OpenVINO 2026.1 with maximum performance for ARM Cortex-A75 (e.g. 48-core aarch64 host).
#
# Cortex-A75 capabilities (from lscpu/cpuinfo):
#   armv8.2-a, NEON/ASIMD, fp16 (fphp/asimdhp), dot-product (asimddp),
#   rdm (asimdrdm), rcpc (lrcpc), crc32, atomics. No SVE/SVE2/SME, no i8mm/bf16.
#
# Therefore:
#   * -mcpu=cortex-a75            -> armv8.2-a + fp16 + dotprod + rcpc codegen, tuned for A75
#   * OV_CPU_ARM_TARGET_ARCH=arm64-v8.2-a + OV_CPU_AARCH64_USE_MULTI_ISA=OFF
#                                 -> ARM Compute Library compiled directly with fp16/dotprod
#                                    kernels, without SVE/SME multi-ISA ballast that A75
#                                    can never execute
#   * ENABLE_LTO=ON, Release, TBB threading, ITT profiling off
#
# Works in two modes:
#   * natively on the aarch64 target (gcc >= 8 supports -mcpu=cortex-a75; gcc 11.4 is fine)
#   * cross-compiling from x86_64 (requires gcc-aarch64-linux-gnu, g++-aarch64-linux-gnu)
#
# Requirements: cmake >= 3.20, scons (for ARM Compute Library), ninja (recommended) or make.
#
# Usage:
#   ./scripts/arm-cortex-a75/build_openvino_cortex_a75.sh
#
# Tunables (environment variables):
#   OPENVINO_DIR  - openvino source root   (default: repo root of this script)
#   BUILD_DIR     - build directory        (default: $OPENVINO_DIR/build-cortex-a75)
#   INSTALL_DIR   - install prefix         (default: $OPENVINO_DIR/install-cortex-a75)
#   JOBS          - parallel jobs          (default: nproc)
#   ENABLE_PYTHON - build python bindings  (default: OFF; native build only)
#   EXTRA_FRONTENDS - ON to also build ONNX/TF/TFLite/PyTorch frontends (default: OFF,
#                     only IR frontend is built - enough for benchmark_app with IR models)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENVINO_DIR="${OPENVINO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-${OPENVINO_DIR}/build-cortex-a75}"
INSTALL_DIR="${INSTALL_DIR:-${OPENVINO_DIR}/install-cortex-a75}"
JOBS="${JOBS:-$(nproc)}"
ENABLE_PYTHON="${ENABLE_PYTHON:-OFF}"
EXTRA_FRONTENDS="${EXTRA_FRONTENDS:-OFF}"

A75_FLAGS="-mcpu=cortex-a75"

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"
    -DCMAKE_C_FLAGS="${A75_FLAGS}"
    -DCMAKE_CXX_FLAGS="${A75_FLAGS}"

    # Performance
    -DENABLE_LTO=ON
    -DTHREADING=TBB
    -DENABLE_PROFILING_ITT=OFF

    # ARM Compute Library: compile exactly for armv8.2-a (fp16 + dotprod baked in),
    # do not build runtime multi-ISA dispatch with SVE/SVE2/SME kernels useless on A75
    -DOV_CPU_AARCH64_USE_MULTI_ISA=OFF
    -DOV_CPU_ARM_TARGET_ARCH=arm64-v8.2-a
    -DARM_COMPUTE_SCONS_JOBS="${JOBS}"

    # Only the CPU plugin is relevant on this machine
    -DENABLE_INTEL_GPU=OFF
    -DENABLE_INTEL_NPU=OFF

    # benchmark_app comes from C++ samples
    -DENABLE_SAMPLES=ON

    # Leaner & faster build
    -DENABLE_TESTS=OFF
    -DENABLE_FUNCTIONAL_TESTS=OFF
    -DENABLE_CPPLINT=OFF
    -DENABLE_CLANG_FORMAT=OFF
    -DENABLE_NCC_STYLE=OFF
    -DENABLE_WHEEL=OFF
    -DENABLE_PYTHON="${ENABLE_PYTHON}"

    # IR frontend is all benchmark_app needs for OpenVINO IR (.xml/.bin) models
    -DENABLE_OV_IR_FRONTEND=ON
    -DENABLE_OV_ONNX_FRONTEND="${EXTRA_FRONTENDS}"
    -DENABLE_OV_TF_FRONTEND="${EXTRA_FRONTENDS}"
    -DENABLE_OV_TF_LITE_FRONTEND="${EXTRA_FRONTENDS}"
    -DENABLE_OV_PYTORCH_FRONTEND="${EXTRA_FRONTENDS}"
    -DENABLE_OV_JAX_FRONTEND=OFF
    -DENABLE_OV_PADDLE_FRONTEND=OFF
)

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "== Cross-compiling for aarch64 (Cortex-A75) from $(uname -m) =="
    CMAKE_ARGS+=(-DCMAKE_TOOLCHAIN_FILE="${OPENVINO_DIR}/cmake/arm64.toolchain.cmake")
else
    echo "== Native aarch64 build for Cortex-A75 =="
fi

if command -v ninja >/dev/null 2>&1; then
    CMAKE_ARGS+=(-G Ninja)
fi

echo "Source:  ${OPENVINO_DIR}"
echo "Build:   ${BUILD_DIR}"
echo "Install: ${INSTALL_DIR}"
echo "Jobs:    ${JOBS}"

cmake -S "${OPENVINO_DIR}" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"
cmake --install "${BUILD_DIR}"

echo
echo "Done. Set up the environment with:"
echo "  source ${INSTALL_DIR}/setupvars.sh"
echo "benchmark_app: ${INSTALL_DIR}/samples_bin/benchmark_app (or build samples in-place)"
