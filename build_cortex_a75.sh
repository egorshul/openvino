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

# Опциональный prefix со СВЕЖИМ binutils (>=2.34), чтобы уже собранный GCC
# использовал ассемблер, понимающий i8mm/bf16. Передаётся в gcc через -B.
#   export BINUTILS_BIN=/opt/gcc-11/bin   (каталог, где лежит новый 'as')
BINUTILS_BIN="${BINUTILS_BIN:-}"
BPREFIX=""
if [[ -n "$BINUTILS_BIN" ]]; then
    BPREFIX="-B${BINUTILS_BIN}"
fi

# Проверка, что ассемблер реально понимает i8mm (нужно для KleidiAI).
# Частая ситуация: GCC обновили до 11, но binutils остался 2.31 (AstraLinux),
# и сборка падает на 'Assembler ... unknown architectural extension i8mm'.
asm_supports_i8mm() {
    echo 'int f(void){return 0;}' | \
        "$CC" $BPREFIX -march=armv8.2-a+i8mm -x c - -c -o /dev/null 2>/dev/null
}

if [[ "$ENABLE_KLEIDIAI" == "ON" ]]; then
    if [[ "${GCC_VER:-0}" -lt 11 ]]; then
        echo ""
        echo "ВНИМАНИЕ: KleidiAI требует GCC>=11, а обнаружен GCC ${GCC_VER}.x."
        echo "Соберите GCC 11 (BUILD_CORTEX_A75_ASTRALINUX.md, раздел 3) либо"
        echo "запустите: ./build_cortex_a75.sh --no-kleidiai"
        echo ""
        exit 1
    fi
    if ! asm_supports_i8mm; then
        echo ""
        echo "ВНИМАНИЕ: ассемблер (binutils) не понимает '+i8mm'/'+bf16'."
        echo "GCC у вас свежий, но 'as' старый (на AstraLinux обычно binutils 2.31)."
        echo "KleidiAI упадёт на 'Assembler ... unknown architectural extension i8mm'."
        echo "Варианты:"
        echo "  1) Соберите binutils>=2.40 (раздел 3.2) и укажите каталог с новым as:"
        echo "       export BINUTILS_BIN=/opt/gcc-11/bin"
        echo "  2) Запустите без KleidiAI:  ./build_cortex_a75.sh --no-kleidiai"
        echo ""
        exit 1
    fi
    echo "Проверка: ассемблер понимает i8mm/bf16 — OK."
fi

# --------------------------------------------------------------------------
# Флаги тюнинга под Cortex-A75.
#  -mtune=cortex-a75 : только планировщик, ISA не меняет (ACL/KleidiAI задают -march сами).
#  static libstdc++  : самодостаточные бинарники при сборке свежим GCC из /opt.
# BPREFIX (-B<dir>) заставляет gcc брать свежий 'as'/'ld' из нового binutils.
TUNE_FLAGS="-mtune=cortex-a75 ${BPREFIX}"
C_FLAGS="${TUNE_FLAGS} -static-libgcc"
CXX_FLAGS="${TUNE_FLAGS} -static-libstdc++ -static-libgcc"
ASM_FLAGS="${BPREFIX}"

GENERATOR="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then GENERATOR="Ninja"; fi

cmake -B "$BUILD_DIR" -S "$ROOT_DIR" -G "$GENERATOR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_C_FLAGS="$C_FLAGS" \
  -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
  -DCMAKE_ASM_FLAGS="$ASM_FLAGS" \
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
