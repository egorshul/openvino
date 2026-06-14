#!/usr/bin/env bash
#
# Сборка OpenVINO под ARM Cortex-A75 (aarch64, armv8.2-a + FP16 + DotProd).
# Подробности и обоснование флагов: BUILD_CORTEX_A75_ASTRALINUX.md
#
# Использование:
#   export CC=/opt/gcc-11/bin/gcc-11 CXX=/opt/gcc-11/bin/g++-11   # рекомендуется GCC>=11
#   ./build_cortex_a75.sh                  # сборка с KleidiAI (нужен GCC>=11)
#   ./build_cortex_a75.sh --build-tbb      # доп. собрать свой oneTBB (нужно на
#                                          # старом glibc, напр. AstraLinux)
#   ./build_cortex_a75.sh --no-kleidiai    # запасной путь для GCC 8.3 (KleidiAI выключен)
#
# Прочее: --lto, --no-python; env: TBBROOT (готовый oneTBB), TBB_VERSION,
#         BINUTILS_BIN (каталог нового 'as'), JOBS, BUILD_DIR, INSTALL_DIR.
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
BUILD_TBB=0
TBB_VERSION="${TBB_VERSION:-v2021.13.0}"

for arg in "$@"; do
    case "$arg" in
        --no-kleidiai) ENABLE_KLEIDIAI=OFF ;;
        --lto)         ENABLE_LTO=ON ;;
        --no-python)   ENABLE_PYTHON=OFF ;;
        --build-tbb)   BUILD_TBB=1 ;;
        *) echo "Неизвестный аргумент: $arg"; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------
# Компилятор
CC="${CC:-gcc}"
CXX="${CXX:-g++}"
GCC_VER="$("$CC" -dumpfullversion -dumpversion 2>/dev/null | cut -d. -f1)"

# ВАЖНО: ComputeLibrary (ACL) собирается через scons с build=native и при этом
# ИГНОРИРУЕТ compiler_prefix — вызывает компилятор по "голому" имени (gcc-11/
# g++-11), ища его в $PATH. Если GCC стоит в /opt/gcc-11/bin и этого пути нет в
# PATH, ACL падает с "Compiler ' g++-11' not found". Поэтому кладём каталог
# компилятора в PATH.
CC_DIR="$(cd "$(dirname "$CC")" 2>/dev/null && pwd || true)"
if [[ -n "$CC_DIR" && ":$PATH:" != *":$CC_DIR:"* ]]; then
    export PATH="$CC_DIR:$PATH"
    echo "   PATH += $CC_DIR (нужно для scons-сборки ACL)"
fi

# ВАЖНО (часть 2): если GCC/binutils собраны из исходников в /opt, то их
# собственные C++-инструменты (ld.gold, g++) слинкованы со СВЕЖИМ libstdc++ и
# при запуске требуют его (GLIBCXX_3.4.29 и т.п.). Системный libstdc++ от gcc
# 8.3 их не содержит -> 'ld.gold: ... GLIBCXX_3.4.29 not found'. Кладём родной
# каталог libstdc++ нового GCC в LD_LIBRARY_PATH на время сборки.
GCC_LIBSTDCPP="$("$CXX" -print-file-name=libstdc++.so.6 2>/dev/null || true)"
if [[ "$GCC_LIBSTDCPP" == /* ]]; then
    GCC_LIBDIR="$(cd "$(dirname "$GCC_LIBSTDCPP")" && pwd)"
    if [[ ":${LD_LIBRARY_PATH:-}:" != *":$GCC_LIBDIR:"* ]]; then
        export LD_LIBRARY_PATH="$GCC_LIBDIR:${LD_LIBRARY_PATH:-}"
        echo "   LD_LIBRARY_PATH += $GCC_LIBDIR (для ld.gold/g++ нового GCC)"
    fi
fi

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
# TBB. Готовый arm64-бинарник oneTBB, который качает OpenVINO, собран против
# НОВОГО glibc (символы @GLIBC_2.32/2.34) и не линкуется на AstraLinux
# (glibc ~2.28): 'undefined reference to pthread_create@GLIBC_2.34'.
# Решение — свой oneTBB из исходников (env TBBROOT отменяет загрузку prebuilt).
if [[ "$BUILD_TBB" == "1" ]]; then
    TBB_SRC="${ROOT_DIR}/_onetbb_src"
    TBB_INSTALL="${TBBROOT:-${ROOT_DIR}/_onetbb_install}"
    echo "--- Сборка oneTBB ${TBB_VERSION} из исходников -> ${TBB_INSTALL} ---"
    if [[ ! -d "$TBB_SRC/.git" ]]; then
        git clone --depth 1 --branch "$TBB_VERSION" \
            https://github.com/uxlfoundation/oneTBB.git "$TBB_SRC"
    fi
    cmake -B "$TBB_SRC/build" -S "$TBB_SRC" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_C_FLAGS="-mtune=cortex-a75 ${BPREFIX}" \
        -DCMAKE_CXX_FLAGS="-mtune=cortex-a75 ${BPREFIX} -static-libstdc++ -static-libgcc" \
        -DTBB_TEST=OFF -DTBB_STRICT=OFF \
        -DCMAKE_INSTALL_PREFIX="$TBB_INSTALL"
    cmake --build "$TBB_SRC/build" --parallel "$JOBS"
    cmake --install "$TBB_SRC/build"
    export TBBROOT="$TBB_INSTALL"
    echo "--- oneTBB готов, TBBROOT=$TBBROOT ---"
fi

if [[ -n "${TBBROOT:-}" ]]; then
    echo "   TBBROOT=$TBBROOT (свой oneTBB; prebuilt качаться не будет)"
else
    echo ""
    echo "ВНИМАНИЕ: TBBROOT не задан. OpenVINO скачает готовый arm64 oneTBB,"
    echo "который собран против нового glibc и на AstraLinux (glibc ~2.28) даёт"
    echo "ошибку линковки 'pthread_create@GLIBC_2.34'. Если так и вышло —"
    echo "соберите свой oneTBB:  ./build_cortex_a75.sh --build-tbb   (раздел 3.4)"
    echo ""
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
