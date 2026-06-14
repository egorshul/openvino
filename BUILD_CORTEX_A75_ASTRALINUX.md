# Сборка OpenVINO 2026.1 под ARM Cortex-A75 (AstraLinux 4.7 / aarch64)

Инструкция по сборке максимально производительной нативной версии OpenVINO
runtime для процессора **ARM Cortex-A75** (aarch64) на **AstraLinux 4.7 arm**.

---

## 1. Анализ целевого CPU

Из `lscpu` / `/proc/cpuinfo`:

| Параметр              | Значение                                                         |
|-----------------------|------------------------------------------------------------------|
| Архитектура           | `aarch64`, `armv8.2-a`                                            |
| Модель                | Cortex-A75 (`CPU part 0xd0a`), implementer ARM (`0x41`)           |
| Ядра                  | 48 (1 сокет, 1 NUMA node, 1 thread/core)                         |
| Частота               | 1.8–2.0 GHz                                                       |
| Кэш                   | L1 64K, L2 512K, L3 2048K, L4 32768K                             |
| Флаги (ISA)           | `fp asimd fphp asimdhp` (**FP16**), `asimddp` (**DotProd, i8 dp**), `crc32 atomics lrcpc dcpop` |

**Что это значит для сборки.** Cortex-A75 — это `armv8.2-a` со следующими
полезными для нейросетей расширениями:

* `asimdhp` / `fphp` → **FP16** арифметика (ускоряет fp16-инференс);
* `asimddp` → **DotProd** (`SDOT`/`UDOT`) → быстрый INT8 matmul/conv.

Чего **НЕТ** у Cortex-A75 (важно!):

* **нет `i8mm`** (матричное INT8, появилось в Armv8.6) — именно поэтому
  падает сборка KleidiAI с `invalid feature modifier in '-march=armv8.2-a+fp16+i8mm'`;
* **нет `bf16`**;
* **нет `sve` / `sve2` / `sme`** (нет векторов переменной длины).

Оптимальная целевая микроархитектура ACL — `arm64-v8.2-a` (даёт FP16 + DotProd
ядра), тюнинг планировщика — `-mtune=cortex-a75`. SVE/SME/i8mm/bf16 ядра
для этого CPU бесполезны (в рантайме не выберутся) и только удлиняют сборку.

---

## 2. Главная проблема: компилятор

На AstraLinux 4.7 системный компилятор — **GCC 8.3.0**. Этого **недостаточно**:

1. **KleidiAI** требует **GCC ≥ 11**. На 8.3.0 вы видите:
   ```
   Using non-supported GCC version. Expected 11 or newer, received 8.3.0
   ```
   и затем фатальную ошибку компиляции микроядер:
   ```
   cc1: error: invalid feature modifier in '-march=armv8.2-a+fp16+i8mm'
   ```
   Причина: GCC 8.3 не знает модификатор `+i8mm` (поддержка добавлена в GCC 9/10),
   а KleidiAI собирает все микроядра, включая i8mm-варианты, с такими `-march`.
   В OpenVINO KleidiAI включается на aarch64 **безусловно**
   (`ENABLE_KLEIDIAI_FOR_CPU=ON` по умолчанию, без проверки версии GCC —
   см. `src/plugins/intel_cpu/CMakeLists.txt:130`).

2. **Multi-ISA ACL** (FP16 + SVE ядра одним билдом) требует **GCC ≥ 10.2**
   (нужен заголовок `arm_sve.h`, см. `src/plugins/intel_cpu/CMakeLists.txt:75`).
   На 8.3 он автоматически выключается, и теряется часть FP16-оптимизаций.

> **Вывод.** Для **максимальной производительности** нужен **GCC 11** (или 12).
> Тогда соберутся и KleidiAI (INT4/INT8 микроядра), и FP16-ядра ACL.
> См. раздел 3 (рекомендуемый путь) — обновление GCC.
>
> Если обновлять GCC нельзя — см. раздел 7 (запасной путь на GCC 8.3 с
> отключённым KleidiAI; работает, но медленнее на INT8/INT4 LLM).

---

## 3. Рекомендуемый путь: обновить GCC до 11 и собрать с KleidiAI

### 3.1. Сборка GCC 11 из исходников (нативно на устройстве)

Самый надёжный способ на AstraLinux (репозитории Buster-уровня не содержат
GCC 11). Собираем GCC 11 в `/opt/gcc-11`, **не трогая системный GCC 8.3**.
Так как GCC собирается против системного glibc устройства — бинарники
OpenVINO останутся совместимы с AstraLinux (никаких проблем с `GLIBC_*`).

```bash
# Зависимости для сборки GCC
sudo apt-get update
sudo apt-get install -y build-essential wget flex bison \
    libgmp-dev libmpfr-dev libmpc-dev libisl-dev zlib1g-dev

# Исходники GCC 11.4 (совпадает с тем, что у вас на dev-машине)
cd /tmp
wget https://ftp.gnu.org/gnu/gcc/gcc-11.4.0/gcc-11.4.0.tar.xz
tar xf gcc-11.4.0.tar.xz
cd gcc-11.4.0
./contrib/download_prerequisites          # подтянет gmp/mpfr/mpc/isl (нужна сеть)

mkdir build && cd build
../configure --prefix=/opt/gcc-11 \
             --enable-languages=c,c++ \
             --disable-multilib \
             --program-suffix=-11 \
             --enable-checking=release
make -j"$(nproc)"          # на 48 ядрах ~30-60 мин
sudo make install
```

Проверка:
```bash
/opt/gcc-11/bin/gcc-11 --version    # gcc (GCC) 11.4.0
```

> **Альтернативы обновлению GCC:**
> * Если в вашем дистрибутиве доступен пакет `gcc-11` через `apt`/бэкпорты —
>   поставьте его (`sudo apt-get install gcc-11 g++-11`) и пропустите сборку из исходников.
> * Кросс-компиляция с современного хоста (Ubuntu 22.04, GCC 11/12) возможна,
>   но рискует несовместимостью `GLIBC`/`libstdc++` с AstraLinux. Нативная
>   сборка с GCC 11, собранным на самом устройстве, — самый безопасный вариант.

### 3.2. Важно про libstdc++ в рантайме

OpenVINO, собранный GCC 11, использует более новый `libstdc++`, чем системный
(от GCC 8.3). Чтобы при запуске не ловить `GLIBCXX_3.4.29 not found`, выберите
**один** из вариантов:

* **(рекомендуется) статически слинковать** libstdc++/libgcc — это уже зашито
  в `build_cortex_a75.sh` через флаги `-static-libstdc++ -static-libgcc`;
  бинарники самодостаточны и не зависят от `/opt/gcc-11`.
* либо при запуске указывать `export LD_LIBRARY_PATH=/opt/gcc-11/lib64:$LD_LIBRARY_PATH`.

---

## 4. Получение исходников OpenVINO 2026.1

```bash
git clone --branch 2026.1.0 --depth 1 https://github.com/openvinotoolkit/openvino.git
# либо ветка релиза:
# git clone --branch releases/2026/1 https://github.com/openvinotoolkit/openvino.git

cd openvino
git submodule update --init --recursive    # ОБЯЗАТЕЛЬНО: ComputeLibrary, KleidiAI, oneDNN, TBB ...
```

Системные зависимости сборки:
```bash
sudo -E ./install_build_dependencies.sh
# scons нужен для сборки ComputeLibrary:
sudo apt-get install -y scons ccache
```

---

## 5. Сборка (рекомендуемая, GCC 11 + KleidiAI + FP16)

Удобнее всего через приложенный скрипт `build_cortex_a75.sh`
(см. раздел 6 и сам файл в корне репозитория):

```bash
export CC=/opt/gcc-11/bin/gcc-11
export CXX=/opt/gcc-11/bin/g++-11
./build_cortex_a75.sh
```

Либо вручную:

```bash
export CC=/opt/gcc-11/bin/gcc-11
export CXX=/opt/gcc-11/bin/g++-11

cmake -B build -S . -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  \
  `# --- ТЮНИНГ ПОД CORTEX-A75 ---` \
  -DCMAKE_C_FLAGS="-mtune=cortex-a75 -static-libgcc" \
  -DCMAKE_CXX_FLAGS="-mtune=cortex-a75 -static-libstdc++ -static-libgcc" \
  -DOV_CPU_AARCH64_USE_MULTI_ISA=OFF \
  -DOV_CPU_ARM_TARGET_ARCH=arm64-v8.2-a \
  \
  `# --- ОПТИМИЗАЦИИ CPU-плагина ---` \
  -DENABLE_KLEIDIAI_FOR_CPU=ON \
  -DTHREADING=TBB \
  \
  `# --- УБИРАЕМ ЛИШНЕЕ (быстрее сборка, меньше зависимостей) ---` \
  -DENABLE_INTEL_GPU=OFF \
  -DENABLE_INTEL_NPU=OFF \
  -DENABLE_OV_TF_FRONTEND=ON \
  -DENABLE_OV_ONNX_FRONTEND=ON \
  -DENABLE_OV_PADDLE_FRONTEND=OFF \
  -DENABLE_OV_PYTORCH_FRONTEND=ON \
  -DENABLE_OV_JAX_FRONTEND=OFF \
  -DENABLE_SAMPLES=ON \
  -DENABLE_TESTS=OFF \
  \
  `# --- PYTHON API (если нужен; у вас есть .venv) ---` \
  -DENABLE_PYTHON=ON \
  -DPython3_EXECUTABLE="$(which python3)"

cmake --build build --parallel "$(nproc)"
cmake --install build --prefix "$PWD/install"
```

### Почему именно эти флаги

| Флаг | Зачем |
|------|-------|
| `-mtune=cortex-a75` | Тюнинг планировщика инструкций под A75. Только tune, не меняет ISA — безопасно сочетается с `-march`, который выставляют ACL/KleidiAI пер-файлово. |
| `OV_CPU_ARM_TARGET_ARCH=arm64-v8.2-a` | Собирает ACL-ядра с FP16+DotProd ровно под A75. |
| `OV_CPU_AARCH64_USE_MULTI_ISA=OFF` | Отключает SVE/SME multi-ISA ядра — на A75 их нет, незачем тратить время сборки. (FP16-ядра при этом сохраняются за счёт `arm64-v8.2-a`.) |
| `ENABLE_KLEIDIAI_FOR_CPU=ON` | Быстрые INT4/INT8 микроядра matmul (особенно для LLM/квантованных моделей). Требует GCC ≥ 11. |
| `THREADING=TBB` | Лучшая масштабируемость на 48 ядрах. |
| `-static-libstdc++ -static-libgcc` | Самодостаточные бинарники, не зависят от свежего `libstdc++` из `/opt/gcc-11`. |
| Отключение GPU/NPU/части фронтендов | Эти плагины для ARM CPU не нужны — ускоряет сборку. Оставьте фронтенды под ваш формат модели (TF/ONNX/PyTorch). |

> **Опционально, ещё +производительность (с осторожностью):**
> `-DENABLE_LTO=ON` — межпроцедурная оптимизация (LTO). Даёт небольшой прирост,
> но заметно увеличивает время и потребление памяти при линковке; включайте,
> если сборочная машина это потянет.

---

## 6. Скрипт сборки

В корне репозитория лежит `build_cortex_a75.sh` — он инкапсулирует команды
из раздела 5, автоматически определяет компилятор и параметры. Использование:

```bash
# с GCC 11 (рекомендуется):
export CC=/opt/gcc-11/bin/gcc-11 CXX=/opt/gcc-11/bin/g++-11
./build_cortex_a75.sh

# запасной путь на системном GCC 8.3 (KleidiAI выключится автоматически):
./build_cortex_a75.sh --no-kleidiai
```

---

## 7. Запасной путь: GCC 8.3 без обновления (KleidiAI OFF)

Если обновить компилятор сейчас нельзя — собираем на системном GCC 8.3.0,
**отключив KleidiAI** (именно он генерит `+i8mm` и падает). Multi-ISA на 8.3
и так выключен, поэтому ошибок с `arm_sve.h` не будет.

```bash
cmake -B build -S . -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-mtune=cortex-a75" \
  -DCMAKE_CXX_FLAGS="-mtune=cortex-a75" \
  -DOV_CPU_AARCH64_USE_MULTI_ISA=OFF \
  -DOV_CPU_ARM_TARGET_ARCH=arm64-v8.2-a \
  -DENABLE_KLEIDIAI_FOR_CPU=OFF \
  -DTHREADING=TBB \
  -DENABLE_INTEL_GPU=OFF -DENABLE_INTEL_NPU=OFF \
  -DENABLE_TESTS=OFF

cmake --build build --parallel "$(nproc)"
```

**Что вы теряете:** оптимизированные KleidiAI INT4/INT8 matmul-микроядра.
DotProd-ускорение INT8 через ACL/oneDNN остаётся, FP16-ядра ACL остаются.
Для обычных CNN разница невелика; для квантованных LLM (INT4/INT8 weight-only)
KleidiAI заметно быстрее — поэтому для них предпочтителен путь из раздела 3.

> **Важно:** не передавайте вручную `-march=...+i8mm`/`+bf16`/`+sve` — Cortex-A75
> их не поддерживает, и GCC (любой версии) выдаст `invalid feature modifier`.
> Максимум для A75: `-march=armv8.2-a+fp16+dotprod` (но это уже покрыто
> `OV_CPU_ARM_TARGET_ARCH=arm64-v8.2-a` и `-mtune=cortex-a75`).

---

## 8. Проверка и запуск

```bash
source install/setupvars.sh

# Проверка, что плагин CPU виден и определяет ISA:
python3 -c "import openvino as ov; c=ov.Core(); print(c.available_devices); \
print(c.get_property('CPU','FULL_DEVICE_NAME'))"

# Бенчмарк (throughput на 48 ядер):
./install/samples/.../benchmark_app -m model.xml -d CPU -hint throughput
```

Рекомендации по рантайму на 48-ядерном A75:
* для пропускной способности: `-hint throughput` (или `ov::hint::PerformanceMode::THROUGHPUT`);
* для задержки одиночного запроса: `-hint latency`;
* TBB сам раскладывает потоки по 48 ядрам (1 NUMA node — без NUMA-нюансов).

---

## 9. Краткая шпаргалка (TL;DR)

1. Cortex-A75 = `armv8.2-a` + FP16 + DotProd, **без i8mm/bf16/sve**.
2. Ошибка `invalid feature modifier '...+i8mm'` = KleidiAI + старый GCC 8.3.
3. **Максимальная производительность** → собрать **GCC 11** (раздел 3),
   затем сборка с `KleidiAI=ON`, `arm64-v8.2-a`, `-mtune=cortex-a75`,
   статический libstdc++.
4. **Без обновления GCC** → собрать с `-DENABLE_KLEIDIAI_FOR_CPU=OFF`
   (раздел 7) — работает на GCC 8.3, но медленнее на квантованных моделях.
