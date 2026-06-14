#!/usr/bin/env bash
#
# Сборка OpenVINO под ARM Cortex-A75 (aarch64, armv8.2-a + FP16 + DotProd).
# Подробности и обоснование флагов: BUILD_CORTEX_A75_ASTRALINUX.md
#
# Использование:
#   export CC=/opt/gcc-11/bin/gcc-11 CXX=/opt/gcc-11/bin/g++-11   # рекомендуется GCC>=11
#   ./build_cortex_a75.sh                  # сборка с KleidiAI (нужен GCC>=11)
#   ./build_cortex_a75.sh --no-kleidiai    # запасной путь для GCC 8.3 (KleidiAI выключен)
#
set -euo pipefail

# --------------------------------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
INSTALL_DIR="${INSTALL_DIR:-${ROOT_DIR}/install}"
JOBS="${JOBS:-$(nproc)}"

ENABLE_KLEIDIAI=ON
ENABLE_PYTHON="${ENABLE_PYTHON:-ON}"
ENABLE_LTO="${ENABLE_LTO:-OFF}"

for arg in "$@"; do
    case "$arg" in
        --no-kleidiai) ENABLE_KLEIDIAI=OFF ;;
        --lto)         ENABLE_LTO=ON ;;
        --no-python)   ENABLE_PYTHON=OFF ;;
        *) echo "Неизвестный аргумент: $arg"; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------
# Компилятор
CC="${CC:-gcc}"
CXX="${CXX:-g++}"
GCC_VER="$("$CC" -dumpfullversion -dumpversion 2>/dev/null | cut -d. -f1)"

echo "=============================================================="
echo " OpenVINO build for ARM Cortex-A75"
echo "   CC=$CC ($("$CC" --version | head -1))"
echo "   CXX=$CXX"
echo "   KleidiAI=$ENABLE_KLEIDIAI  LTO=$ENABLE_LTO  Python=$ENABLE_PYTHON"
echo "   jobs=$JOBS  build=$BUILD_DIR  install=$INSTALL_DIR"
echo "=============================================================="

# Защита от типичной ошибки: KleidiAI на старом GCC падает на '+i8mm'.
if [[ "$ENABLE_KLEIDIAI" == "ON" && "${GCC_VER:-0}" -lt 11 ]]; then
    echo ""
    echo "ВНИМАНИЕ: KleidiAI требует GCC>=11, а обнаружен GCC ${GCC_VER}.x."
    echo "Сборка KleidiAI упадёт с 'invalid feature modifier in -march=...+i8mm'."
    echo "Варианты:"
    echo "  1) Соберите GCC 11 (см. BUILD_CORTEX_A75_ASTRALINUX.md, раздел 3)"
    echo "     и задайте: export CC=/opt/gcc-11/bin/gcc-11 CXX=/opt/gcc-11/bin/g++-11"
    echo "  2) Запустите с флагом: ./build_cortex_a75.sh --no-kleidiai"
    echo ""
    exit 1
fi

# --------------------------------------------------------------------------
# Флаги тюнинга под Cortex-A75.
#  -mtune=cortex-a75 : только планировщик, ISA не меняет (ACL/KleidiAI задают -march сами).
#  static libstdc++  : самодостаточные бинарники при сборке свежим GCC из /opt.
TUNE_FLAGS="-mtune=cortex-a75"
C_FLAGS="${TUNE_FLAGS} -static-libgcc"
CXX_FLAGS="${TUNE_FLAGS} -static-libstdc++ -static-libgcc"

GENERATOR="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then GENERATOR="Ninja"; fi

cmake -B "$BUILD_DIR" -S "$ROOT_DIR" -G "$GENERATOR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_C_FLAGS="$C_FLAGS" \
  -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
  -DOV_CPU_AARCH64_USE_MULTI_ISA=OFF \
  -DOV_CPU_ARM_TARGET_ARCH=arm64-v8.2-a \
  -DENABLE_KLEIDIAI_FOR_CPU="$ENABLE_KLEIDIAI" \
  -DTHREADING=TBB \
  -DENABLE_LTO="$ENABLE_LTO" \
  -DENABLE_INTEL_GPU=OFF \
  -DENABLE_INTEL_NPU=OFF \
  -DENABLE_OV_TF_FRONTEND=ON \
  -DENABLE_OV_ONNX_FRONTEND=ON \
  -DENABLE_OV_PYTORCH_FRONTEND=ON \
  -DENABLE_OV_PADDLE_FRONTEND=OFF \
  -DENABLE_OV_JAX_FRONTEND=OFF \
  -DENABLE_SAMPLES=ON \
  -DENABLE_TESTS=OFF \
  -DENABLE_PYTHON="$ENABLE_PYTHON" \
  ${ENABLE_PYTHON:+-DPython3_EXECUTABLE="$(command -v python3)"}

cmake --build "$BUILD_DIR" --parallel "$JOBS"
cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR"

echo ""
echo "Готово. Активируйте окружение:  source ${INSTALL_DIR}/setupvars.sh"
