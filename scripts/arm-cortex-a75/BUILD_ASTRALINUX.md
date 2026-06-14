# Сборка OpenVINO 2026.1 под ARM Cortex-A75 на Astra Linux (aarch64)

Пошаговая инструкция по нативной сборке максимально производительной версии
OpenVINO 2026.1 на машине с процессором **ARM Cortex-A75** (48 ядер, armv8.2-a)
под управлением **Astra Linux 4.7 arm** и запуску пайплайна Stable Diffusion XL
в `benchmark_app`.

Скрипты, на которые ссылается инструкция, лежат рядом:
- `build_openvino_cortex_a75.sh` — конфигурация и сборка;
- `benchmark_sdxl.sh` — прогон всех моделей SDXL в `benchmark_app`.

---

## 0. Подходит ли gcc 8.3.0?

**Да, штатного gcc 8.3.0 на Astra Linux достаточно** для нашей конфигурации.
Обновлять компилятор не обязательно. Обоснование — по фактическим проверкам в
коде OpenVINO 2026.1 и ARM Compute Library:

| Требование | Минимум | gcc 8.3.0 | Где проверяется |
|---|---|---|---|
| Сборка OpenVINO Runtime | GCC **7.5** | ✅ | `docs/dev/build_linux.md` (в списке тестовых даже RHEL 8.2 / gcc 8.5) |
| ARM Compute Library, код armv8.2-a | GCC **6.2.1** | ✅ | `ComputeLibrary/SConstruct` |
| Флаг `-mcpu=cortex-a75` | GCC **8** | ✅ | поддержка с gcc 8.1 |
| `std::filesystem` (в рантайме) | для gcc < 9.1 линкуется `libstdc++fs` | ✅ автоматически | `src/cmake/openvino.cmake` |
| Multi-ISA ACL (SVE/SME, `arm_sve.h`) | GCC **10.2** | ⛔ — **и не нужно** | у нас `OV_CPU_AARCH64_USE_MULTI_ISA=OFF` |

Последняя строка — ключевая. Единственное, что недоступно на gcc 8.3, это сборка
ACL в режиме multi-ISA с рантайм-диспетчеризацией SVE/SVE2/SME-ядер. Но Cortex-A75
**не умеет SVE** (см. флаги `lscpu`: есть `fphp asimdhp asimddp`, нет `sve`), поэтому
этот режим для него бесполезен и в нашей конфигурации выключен. Вместо него ACL
собирается напрямую под `arm64-v8.2-a`, и fp16/dotprod-ядра «зашиваются» в бинарник
через `-march=armv8.2-a` — это работает на gcc 8.3 без всяких SVE-заголовков.

> Если позже захочется выжать ещё немного (более новый автовекторизатор и
> планировщик инструкций) или вы столкнётесь с внутренней ошибкой компилятора
> (ICE) на каком-то файле — переходите на gcc 10/11. Инструкция по обновлению —
> в разделе [6](#6-обновление-gcc-если-понадобится).

**Что на Astra Linux 4.7 (база Debian 10 «buster») действительно нужно поправить —
это CMake.** OpenVINO 2026.1 требует **CMake ≥ 3.20**, а в системе по умолчанию
3.13. Установка новой версии — в шаге 1.

---

## 1. Установка зависимостей

```bash
# Компилятор, базовые инструменты, git, python
sudo apt update
sudo apt install -y build-essential git pkg-config python3 python3-pip \
                    ninja-build libpython3-dev

# scons нужен для сборки ARM Compute Library.
# Версия из репозитория Astra подойдёт; если её нет — ставим через pip:
sudo apt install -y scons || pip3 install --user scons

# CMake >= 3.20 (системный 3.13 СЛИШКОМ СТАРЫЙ).
# Самый простой способ, не трогающий систему:
pip3 install --user "cmake>=3.27"
# убедитесь, что ~/.local/bin в PATH:
export PATH="$HOME/.local/bin:$PATH"
cmake --version   # должно быть >= 3.20
```

Проверьте компилятор:

```bash
gcc --version      # ожидаем 8.3.0 (или новее, если обновляли)
gcc -mcpu=cortex-a75 -Q --help=target 2>/dev/null | grep -E '\-(march|mcpu)='
# должно показать: -march= armv8.2-a   и   -mcpu= cortex-a75
```

---

## 2. Получение исходников OpenVINO 2026.1

```bash
git clone --branch 2026.1.2 --depth 1 \
    https://github.com/openvinotoolkit/openvino.git
cd openvino

# Подмодули. Для сборки CPU-плагина под ARM достаточно набора ниже,
# но проще инициализировать все:
git submodule update --init --recursive
```

Минимально необходимый набор подмодулей (если хотите сэкономить трафик/время
вместо `--recursive`):

```bash
git submodule update --init \
  src/plugins/intel_cpu/thirdparty/ComputeLibrary \
  src/plugins/intel_cpu/thirdparty/kleidiai \
  src/plugins/intel_cpu/thirdparty/mlas \
  src/plugins/intel_cpu/thirdparty/onednn \
  thirdparty/gflags/gflags thirdparty/pugixml thirdparty/json/nlohmann_json \
  thirdparty/zlib/zlib thirdparty/telemetry thirdparty/snappy thirdparty/ittapi/ittapi
```

Положите рядом скрипты сборки и бенчмарка (этот каталог
`scripts/arm-cortex-a75/`), если их ещё нет в дереве.

---

## 3. Сборка под Cortex-A75

Самый простой путь — скрипт. На самой aarch64-машине он соберёт **нативно**
(определяется автоматически по `uname -m`):

```bash
chmod +x scripts/arm-cortex-a75/build_openvino_cortex_a75.sh
./scripts/arm-cortex-a75/build_openvino_cortex_a75.sh
```

Что включает скрипт (и почему это оптимально для A75):

- **`-mcpu=cortex-a75`** для C и C++ — генерация под armv8.2-a c fp16, dotprod
  (asimddp), rcpc и планировщиком инструкций именно под A75.
- **`OV_CPU_ARM_TARGET_ARCH=arm64-v8.2-a` + `OV_CPU_AARCH64_USE_MULTI_ISA=OFF`** —
  ARM Compute Library компилируется напрямую под armv8.2-a, fp16/dotprod-ядра
  встроены, без балласта SVE/SVE2/SME, который A75 не исполняет. Заодно это
  снимает требование gcc ≥ 10.2.
- **`ENABLE_LTO=ON`**, `CMAKE_BUILD_TYPE=Release`, **`THREADING=TBB`** (лучшее
  масштабирование на 48 ядрах), `ENABLE_PROFILING_ITT=OFF`.
- Только CPU-плагин и IR-фронтенд (для `benchmark_app` с IR-моделями этого
  достаточно); GPU/NPU, тесты и лишние фронтенды выключены.

Артефакты:
- библиотеки и сэмплы — `bin/aarch64/Release/`;
- установленное дерево — `install-cortex-a75/`.

### Полезные переменные окружения

```bash
JOBS=48 \
EXTRA_FRONTENDS=ON \   # доп. фронтенды ONNX/TF/TFLite/PyTorch (по умолчанию OFF)
./scripts/arm-cortex-a75/build_openvino_cortex_a75.sh
```

### Эквивалентная ручная команда CMake

Если хотите без скрипта (например, чтобы встроить в свой пайплайн):

```bash
cmake -S . -B build-cortex-a75 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=install-cortex-a75 \
  -DCMAKE_C_FLAGS="-mcpu=cortex-a75" \
  -DCMAKE_CXX_FLAGS="-mcpu=cortex-a75" \
  -DENABLE_LTO=ON \
  -DTHREADING=TBB \
  -DENABLE_PROFILING_ITT=OFF \
  -DOV_CPU_AARCH64_USE_MULTI_ISA=OFF \
  -DOV_CPU_ARM_TARGET_ARCH=arm64-v8.2-a \
  -DENABLE_INTEL_GPU=OFF -DENABLE_INTEL_NPU=OFF \
  -DENABLE_SAMPLES=ON -DENABLE_TESTS=OFF \
  -DENABLE_PYTHON=OFF -DENABLE_WHEEL=OFF \
  -DENABLE_OV_IR_FRONTEND=ON \
  -DENABLE_OV_ONNX_FRONTEND=OFF -DENABLE_OV_TF_FRONTEND=OFF \
  -DENABLE_OV_TF_LITE_FRONTEND=OFF -DENABLE_OV_PYTORCH_FRONTEND=OFF \
  -DENABLE_OV_JAX_FRONTEND=OFF -DENABLE_OV_PADDLE_FRONTEND=OFF
cmake --build build-cortex-a75 --parallel "$(nproc)"
cmake --install build-cortex-a75
```

---

## 4. Настройка окружения и проверка

```bash
source install-cortex-a75/setupvars.sh   # или вручную добавить bin/aarch64/Release в LD_LIBRARY_PATH

# Проверка, что плагин видит CPU и нужные возможности (FP16/INT8):
./bin/aarch64/Release/hello_query_device | grep -A2 OPTIMIZATION_CAPABILITIES
# ожидаем: FP32 FP16 INT8 ...
```

---

## 5. Бенчмарк пайплайна Stable Diffusion XL

Скрипт `benchmark_sdxl.sh` скачивает готовые OpenVINO IR-модели SDXL с Hugging Face
и прогоняет `benchmark_app -d CPU` по всем моделям пайплайна с реальными формами
шага генерации 1024×1024 (UNet — батч 2 из-за classifier-free guidance):
`text_encoder`, `text_encoder_2`, `unet`, `vae_encoder`, `vae_decoder`.

```bash
chmod +x scripts/arm-cortex-a75/benchmark_sdxl.sh
./scripts/arm-cortex-a75/benchmark_sdxl.sh bin/aarch64/Release/benchmark_app
```

Полезные переменные:

```bash
HINT=throughput \           # latency (по умолч.) | throughput
INFER_PRECISION=f16 \       # f16 (по умолч., нативно для A75) | f32
RESOLUTION=1024 \           # размер картинки; латент = RESOLUTION/8
ONLY="unet vae_decoder" \   # подмножество моделей
MODEL_REPO=OpenVINO/stable-diffusion-xl-base-1.0-int8-ov \  # int8-вариант — быстрее на A75
./scripts/arm-cortex-a75/benchmark_sdxl.sh bin/aarch64/Release/benchmark_app
```

Для максимальной скорости на Cortex-A75:
- держите `INFER_PRECISION=f16` (A75 исполняет fp16 ASIMD нативно);
- используйте **int8**-пайплайн — GEMM считаются на dotprod-ядрах (`asimddp`)
  ACL/KleidiAI: `MODEL_REPO=OpenVINO/stable-diffusion-xl-base-1.0-int8-ov`;
- `-hint latency` минимизирует время одной картинки (все 48 ядер на запрос),
  `-hint throughput` максимизирует картинок/с при нескольких параллельных запросах.

---

## 6. Обновление gcc (если понадобится)

Нужно **только** если поймали ICE компилятора на gcc 8.3 или хотите выжать
максимум из более нового автовекторизатора. Astra Linux 4.7 arm основана на
Debian 10 «buster», поэтому подходят debian-овские способы. Рекомендуемая
цель — **gcc 10** (полный C++17 + при желании можно включить multi-ISA ACL,
хотя для A75 это не даёт выигрыша).

### Вариант A. Из репозитория (если доступен)

```bash
sudo apt update
sudo apt install -y gcc-10 g++-10     # либо gcc-11 g++-11, если есть в репо Astra

# Переключение через alternatives:
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 \
                         --slave   /usr/bin/g++ g++ /usr/bin/g++-10
gcc --version   # 10.x
```

Если пакетов `gcc-10` нет в репозиториях Astra, не подмешивайте чужие
debian-репозитории в боевую систему (риск рассинхронизации glibc/политик Astra) —
используйте вариант B или C.

### Вариант B. Указать компилятор только для сборки (без смены системного)

Если новый gcc установлен в нестандартный префикс (например, собран в
`/opt/gcc-10`), не меняйте системный компилятор — просто передайте его сборке:

```bash
CC=/opt/gcc-10/bin/gcc CXX=/opt/gcc-10/bin/g++ \
./scripts/arm-cortex-a75/build_openvino_cortex_a75.sh
```

(Скрипт уважает переменные `CC`/`CXX` при нативной сборке.)

### Вариант C. Сборка gcc 10 из исходников (надёжно на закрытой Astra)

Подходит для оффлайн/закрытого контура, где нельзя менять репозитории.

```bash
GCC_VER=10.5.0
wget https://ftp.gnu.org/gnu/gcc/gcc-${GCC_VER}/gcc-${GCC_VER}.tar.xz
tar xf gcc-${GCC_VER}.tar.xz && cd gcc-${GCC_VER}
./contrib/download_prerequisites          # gmp, mpfr, mpc (нужен интернет/зеркало)
mkdir build && cd build
../configure --prefix=/opt/gcc-${GCC_VER} \
             --enable-languages=c,c++ \
             --disable-multilib \
             --with-arch=armv8.2-a --with-cpu=cortex-a75
make -j"$(nproc)"        # долго: на 48 ядрах ~30–60 мин
sudo make install
```

Затем собирайте OpenVINO с этим компилятором (см. вариант B):

```bash
CC=/opt/gcc-10.5.0/bin/gcc CXX=/opt/gcc-10.5.0/bin/g++ \
LD_LIBRARY_PATH=/opt/gcc-10.5.0/lib64:$LD_LIBRARY_PATH \
./scripts/arm-cortex-a75/build_openvino_cortex_a75.sh
```

> На gcc ≥ 10.2 при желании можно вернуть `OV_CPU_AARCH64_USE_MULTI_ISA=ON`, но
> **для Cortex-A75 это не нужно** (нет SVE) и лишь раздувает бинарник.

---

## 7. Типичные проблемы

| Симптом | Причина / решение |
|---|---|
| `CMake 3.20 or higher is required` | Системный CMake 3.13. Поставьте `pip3 install --user "cmake>=3.27"` и проверьте `PATH`. |
| `scons: command not found` при сборке ACL | `pip3 install --user scons` (и `~/.local/bin` в `PATH`). |
| `undefined reference to std::filesystem...` | Очень старый компилятор/нестандартная линковка. На gcc 8.3 OpenVINO сам линкует `stdc++fs`; убедитесь, что собираете именно деревом 2026.1.2 без правок флагов линковки. |
| Внутренняя ошибка компилятора (ICE) на gcc 8.3 | Обновите gcc до 10/11 (раздел 6) или соберите проблемный таргет отдельно. |
| Не хватает ОЗУ при LTO/линковке | LTO-линковка `libopenvino.so` и компиляция UNet прожорливы. Уменьшите параллелизм линковки или временно отключите `-DENABLE_LTO=OFF`. |
| `benchmark_app` не находит библиотеки | `source install-cortex-a75/setupvars.sh` либо добавьте `bin/aarch64/Release` (и каталог с `libtbb.so*` из `temp/`) в `LD_LIBRARY_PATH`. |
