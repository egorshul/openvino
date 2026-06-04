#!/usr/bin/env bash
# Cross-compile OpenVINO Runtime for RISC-V 64 (RVV 1.0 capable) with the
# Intel CPU plugin and TBB threading, then run a ResNet-50 inference under QEMU.
#
# Validated on Ubuntu 24.04 (x86-64 host) against OpenVINO releases/2026/1
# using the distro GNU RISC-V cross toolchain (riscv64-linux-gnu-* 13.3.0)
# and qemu-riscv64-static 8.2.2.
#
# Usage:
#   ./scripts/riscv64_cross_build_and_run.sh /path/to/resnet50.xml
#
set -euo pipefail

OV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${OV_ROOT}/build_riscv64}"
INSTALL_DIR="${INSTALL_DIR:-${OV_ROOT}/install_riscv64}"
MODEL_XML="${1:-/tmp/resnet50/resnet50.xml}"
QEMU_LD_PREFIX="${QEMU_LD_PREFIX:-/usr/riscv64-linux-gnu}"

# ---------------------------------------------------------------------------
# 1. Install the cross toolchain, binutils, QEMU and host build tooling.
# ---------------------------------------------------------------------------
install_deps() {
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build pkg-config python3 \
        gcc-riscv64-linux-gnu g++-riscv64-linux-gnu binutils-riscv64-linux-gnu \
        qemu-user-static qemu-user
}

# ---------------------------------------------------------------------------
# 2. Configure. Key choices:
#    * CMAKE_TOOLCHAIN_FILE -> riscv64.linux.toolchain.cmake (distro GNU cross GCC)
#    * THREADING=TBB        -> prebuilt oneapi-tbb-2022.3.0 riscv package is fetched
#    * ENABLE_INTEL_CPU=ON  -> CPU plugin (enabled by default for RISCV64; pinned here)
#    * XBYAK_RISCV_V=ON is selected automatically -> RVV 1.0 JIT kernels.
# ---------------------------------------------------------------------------
configure() {
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE="${OV_ROOT}/cmake/toolchains/riscv64.linux.toolchain.cmake" \
        -DTHREADING=TBB \
        -DENABLE_TBBBIND_2_5=OFF \
        -DENABLE_INTEL_CPU=ON \
        -DENABLE_INTEL_GPU=OFF \
        -DENABLE_INTEL_NPU=OFF \
        -DENABLE_PYTHON=OFF \
        -DENABLE_WHEEL=OFF \
        -DENABLE_SAMPLES=ON \
        -DENABLE_TESTS=OFF \
        -DENABLE_OV_ONNX_FRONTEND=OFF \
        -DENABLE_OV_PADDLE_FRONTEND=OFF \
        -DENABLE_OV_TF_FRONTEND=OFF \
        -DENABLE_OV_TF_LITE_FRONTEND=OFF \
        -DENABLE_OV_PYTORCH_FRONTEND=OFF \
        -DENABLE_OV_JAX_FRONTEND=OFF \
        -DENABLE_INTEL_OPENMP=OFF \
        -DENABLE_STRICT_DEPENDENCIES=OFF \
        -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
        -S "${OV_ROOT}" -B "${BUILD_DIR}"
}

build() {
    cmake --build "${BUILD_DIR}" --parallel "$(nproc)"
    cmake --install "${BUILD_DIR}"
}

# ---------------------------------------------------------------------------
# 3. Run ResNet-50 under QEMU with the CPU plugin emulating RVV 1.0.
# ---------------------------------------------------------------------------
run() {
    local libdir="${INSTALL_DIR}/runtime/lib/riscv64"
    local tbbdir="${INSTALL_DIR}/runtime/3rdparty/tbb/lib"
    local bench="${OV_ROOT}/bin/riscv64/Release/benchmark_app"

    # The prebuilt oneTBB RISC-V package targets T-Head Xuantie boards (e.g. the
    # Lichee Pi 4A / TH1520 C910) and therefore contains XThead custom
    # instructions, so the emulated CPU must expose both RVV 1.0 (used by the
    # CPU plugin JIT) and the XThead extensions (used by libtbb).
    local cpu='rv64,v=true,vext_spec=v1.0'
    cpu+=',xtheadba=true,xtheadbb=true,xtheadbs=true,xtheadcondmov=true'
    cpu+=',xtheadmac=true,xtheadmemidx=true,xtheadmempair=true,xtheadsync=true'
    cpu+=',xtheadcmo=true,xtheadfmemidx=true,xtheadfmv=true'

    QEMU_LD_PREFIX="${QEMU_LD_PREFIX}" \
    LD_LIBRARY_PATH="${libdir}:${tbbdir}" \
    qemu-riscv64-static -cpu "${cpu}" \
        "${bench}" -m "${MODEL_XML}" -d CPU -niter 4 -nstreams 1 -hint none
}

install_deps
configure
build
run
