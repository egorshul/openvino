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

## 2. Главная проблема: устаревший toolchain (GCC **и** binutils)

На AstraLinux 4.7 системный toolchain — **GCC 8.3.0 + binutils ~2.31**.
Для KleidiAI этого недостаточно сразу по двум причинам, и важно понимать, что
это **две разные проблемы**:

### 2a. GCC слишком старый (ошибка компилятора)

На GCC 8.3.0 вы видите:
```
Using non-supported GCC version. Expected 11 or newer, received 8.3.0
cc1: error: invalid feature modifier in '-march=armv8.2-a+fp16+i8mm'
```
Это ошибка **компилятора** (`cc1`): GCC 8.3 вообще не знает модификатор `+i8mm`
(добавлен в GCC 9/10). Лечится обновлением GCC до **≥ 11** (раздел 3.1).

### 2b. binutils (ассемблер) слишком старый (ошибка ассемблера)

После обновления GCC до 11 ошибка **меняется** на другую:
```
Assembler messages:
Error: unknown architectural extension `i8mm'
Error: unrecognized option -march=armv8.2-a+fp16+i8mm
...
Error: unknown architectural extension `bf16'
Error: unrecognized option -march=armv8.2-a+bf16
```
Теперь это уже ошибка **ассемблера** (`as` из binutils), а не компилятора.
GCC 11 модификаторы `+i8mm`/`+bf16` понимает и передаёт их в `as`, но
системный `as` из binutils 2.31 их не знает. Поддержка `i8mm`/`bf16` в GNU
`as` появилась только в **binutils 2.34**. Когда вы собрали GCC 11 из
исходников, он подхватил **системный `/usr/bin/as`** — отсюда ошибка.

> **KleidiAI собирает i8mm- и bf16-микроядра безусловно** — даже на Cortex-A75,
> где этих расширений нет. В рантайме они просто не выберутся, но
> **скомпилироваться обязаны**. Поэтому для KleidiAI нужны ОБА условия:
> **GCC ≥ 11** И **binutils ≥ 2.34** (рекомендуется 2.40).
> В OpenVINO KleidiAI включён на aarch64 безусловно
> (`src/plugins/intel_cpu/CMakeLists.txt:130`, без проверки версии toolchain).

### 2c. Multi-ISA ACL

**Multi-ISA ACL** (FP16 + SVE ядра одним билдом) требует **GCC ≥ 10.2**
(нужен `arm_sve.h`, см. `src/plugins/intel_cpu/CMakeLists.txt:75`). На A75 SVE
нет, поэтому в нашей конфигурации multi-ISA выключен намеренно
(`OV_CPU_AARCH64_USE_MULTI_ISA=OFF`), а FP16-ядра сохраняются за счёт
`arm64-v8.2-a`.

> **Вывод.** Для **максимальной производительности** обновите **GCC до 11**
> (раздел 3.1) **и binutils до ≥ 2.40** (раздел 3.2). Тогда соберётся весь
> KleidiAI (включая dotprod-микроядра, которые A75 реально использует) и
> FP16-ядра ACL.
>
> Если обновлять toolchain нельзя — см. раздел 7 (запасной путь с отключённым
> KleidiAI; работает на старом GCC 8.3, но медленнее на INT8/INT4-моделях).

---

## 3. Рекомендуемый путь: обновить toolchain (GCC 11 + binutils 2.40) и собрать с KleidiAI

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

### 3.2. Обновить binutils до ≥ 2.40 (КРИТИЧНО для KleidiAI)

Даже со свежим GCC 11 ассемблер из системного binutils 2.31 не знает
`i8mm`/`bf16`. Соберём binutils 2.40 и установим **в тот же prefix**
`/opt/gcc-11`, чтобы GCC брал новый `as`:

```bash
cd /tmp
wget https://ftp.gnu.org/gnu/binutils/binutils-2.40.tar.xz
tar xf binutils-2.40.tar.xz
cd binutils-2.40
mkdir build && cd build
../configure --prefix=/opt/gcc-11 --enable-gold --enable-ld=default
make -j"$(nproc)"
sudo make install
```

Теперь самое важное — **заставить уже собранный GCC 11 использовать новый `as`**.
Два способа:

* **(быстро, без пересборки GCC)** передавать gcc флаг `-B/opt/gcc-11/bin` —
  он указывает, откуда брать `as`/`ld`. В скрипте `build_cortex_a75.sh` это
  делается автоматически, если задать:
  ```bash
  export BINUTILS_BIN=/opt/gcc-11/bin
  ```
* **(чисто, навсегда)** пересобрать GCC 11 (раздел 3.1) **после** установки
  binutils в `/opt/gcc-11` — тогда GCC «запомнит» новый ассемблер и `-B` не нужен.

**Проверка, что ассемблер понимает i8mm/bf16** (должно вывести `OK`):
```bash
echo 'int f(void){return 0;}' | \
  /opt/gcc-11/bin/gcc-11 -B/opt/gcc-11/bin -march=armv8.2-a+i8mm -x c - -c -o /tmp/t.o \
  && echo OK
echo 'int f(void){return 0;}' | \
  /opt/gcc-11/bin/gcc-11 -B/opt/gcc-11/bin -march=armv8.2-a+bf16 -x c - -c -o /tmp/t.o \
  && echo OK
```
Если оба `OK` — KleidiAI соберётся.

> Примечание: `as` версии 2.40 распознаёт `i8mm`/`bf16` как архитектурные
> расширения и для Cortex-A75 (`armv8.2-a`). Это нормально: код этих микроядер
> попадёт в библиотеку, но в рантайме на A75 не будет выбран.

### 3.3. Важно про libstdc++ — и при сборке, и в рантайме

GCC 11 и binutils 2.40, собранные в `/opt/gcc-11`, используют более новый
`libstdc++`, чем системный (от GCC 8.3). Это даёт **две разные** проблемы с
одной и той же ошибкой `GLIBCXX_3.4.29 not found`:

**(а) Во время СБОРКИ** — собственные C++-инструменты toolchain (`ld.gold`,
`g++`) слинкованы со свежим `libstdc++` и при запуске требуют его:
```
/opt/gcc-11/bin/ld.gold: /lib/aarch64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /opt/gcc-11/bin/ld.gold)
```
(OpenVINO подхватывает `ld.gold` через `-fuse-ld=gold`, т.к. `/opt/gcc-11/bin`
теперь в `PATH`.) Лечится добавлением каталога с новым `libstdc++` в
`LD_LIBRARY_PATH` **на время сборки**:
```bash
export LD_LIBRARY_PATH=/opt/gcc-11/lib64:$LD_LIBRARY_PATH
```
> `-static-libstdc++` тут не спасает — он влияет на итоговые библиотеки
> OpenVINO, а не на сам линкер (это отдельный процесс).
> Скрипт `build_cortex_a75.sh` определяет нужный каталог автоматически
> (`g++ -print-file-name=libstdc++.so.6`) и выставляет `LD_LIBRARY_PATH` сам.

**(б) В РАНТАЙМЕ** готовых бинарников OpenVINO — выберите **один** вариант:

* **(рекомендуется) статически слинковать** libstdc++/libgcc — это уже зашито
  в `build_cortex_a75.sh` через флаги `-static-libstdc++ -static-libgcc`;
  бинарники самодостаточны и не зависят от `/opt/gcc-11`.
* либо при запуске указывать `export LD_LIBRARY_PATH=/opt/gcc-11/lib64:$LD_LIBRARY_PATH`.

### 3.4. Собрать oneTBB из исходников (нужно на старом glibc)

OpenVINO по умолчанию **скачивает готовый** arm64-бинарник oneTBB
(`oneapi-tbb-2021.13.1-lin-arm64-release.tgz`, см. `cmake/dependencies.cmake:143`).
Несмотря на комментарий «glibc 2.17» в коде, этот бинарник фактически собран
против **нового glibc** и на AstraLinux (glibc ~2.28) даёт ошибку линковки:
```
libtbb.so.12: undefined reference to `pthread_create@GLIBC_2.34'
libtbb.so.12: undefined reference to `dlopen@GLIBC_2.34'
libtbb.so.12: undefined reference to `pthread_getattr_np@GLIBC_2.32'
...
```
(В glibc 2.34 функции `pthread_*`/`dl*` переехали в сам `libc`; на старом glibc
их там нет — символы не разрешаются.)

**Решение — собрать oneTBB своим GCC 11 против системного glibc.** Если задана
переменная `TBBROOT`, OpenVINO **не качает** prebuilt, а использует ваш TBB
(`cmake/.../dependency_solver.cmake:8`).

```bash
git clone --depth 1 --branch v2021.13.0 https://github.com/uxlfoundation/oneTBB.git
cd oneTBB
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTBB_TEST=OFF -DTBB_STRICT=OFF \
  -DCMAKE_C_COMPILER=/opt/gcc-11/bin/gcc-11 \
  -DCMAKE_CXX_COMPILER=/opt/gcc-11/bin/g++-11 \
  -DCMAKE_C_FLAGS="-mtune=cortex-a75 -B/opt/gcc-11/bin" \
  -DCMAKE_CXX_FLAGS="-mtune=cortex-a75 -B/opt/gcc-11/bin -static-libstdc++ -static-libgcc" \
  -DCMAKE_INSTALL_PREFIX=/opt/onetbb
cmake --build build -j"$(nproc)"
sudo cmake --install build
cd ..

export TBBROOT=/opt/onetbb     # теперь OpenVINO возьмёт ваш TBB, без загрузки
```

> `-static-libstdc++` для libtbb.so делает её самодостаточной в рантайме (не
> тянет свежий libstdc++ из `/opt/gcc-11`).
> Скрипт `build_cortex_a75.sh --build-tbb` делает всё это автоматически и сам
> выставляет `TBBROOT`.

**Как этот TBB используется дальше:**
* **Сборка.** Заданный `TBBROOT` отменяет загрузку prebuilt. Если вы уже
  конфигурировали сборку со «сломанным» TBB — сначала удалите каталог сборки
  (`rm -rf _build`), иначе путь к старому TBB останется в кэше CMake. Проверить
  можно так: `grep -i 'TBB_DIR\|TBBROOT' _build/CMakeCache.txt` (должен быть ваш
  `/opt/onetbb/...`, а не `.../temp/Linux_aarch64/tbb`).
* **Установка.** Для кастомного TBB OpenVINO **сам копирует** `libtbb.so*` в
  `install/runtime/3rdparty/tbb/lib` (`src/cmake/install_tbb.cmake`), а
  `setupvars.sh` добавляет этот путь в `LD_LIBRARY_PATH`.
* **Запуск.** Достаточно `source install/setupvars.sh`. Проверка:
  `ldd install/runtime/lib/aarch64/libopenvino.so | grep tbb` — путь должен
  вести в `3rdparty/tbb/lib` и без `not found`. Без `setupvars.sh` укажите путь
  вручную: `export LD_LIBRARY_PATH=/opt/onetbb/lib:$LD_LIBRARY_PATH`.
>
> **Альтернатива без TBB:** `-DTHREADING=OMP` (использовать OpenMP вместо TBB,
> libgomp из GCC 11). Проще (никакого внешнего TBB), но на многоядерном A75
> TBB обычно даёт лучшую масштабируемость — для максимума предпочтителен oneTBB.

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
export BINUTILS_BIN=/opt/gcc-11/bin                 # каталог со свежим 'as' (binutils 2.40)
export PATH=/opt/gcc-11/bin:$PATH                   # ACL/scons ищет gcc-11 в PATH
export LD_LIBRARY_PATH=/opt/gcc-11/lib64:$LD_LIBRARY_PATH  # ld.gold/g++ нового GCC
export TBBROOT=/opt/onetbb                          # свой oneTBB (раздел 3.4)
./build_cortex_a75.sh
# Если oneTBB ещё не собран — соберите его этим же скриптом:
#   ./build_cortex_a75.sh --build-tbb
```

> **Почему нужен `PATH`.** ComputeLibrary (ACL) собирается отдельным процессом
> `scons` с `build=native` и при этом **игнорирует `compiler_prefix`** — вызывает
> компилятор по «голому» имени `g++-11`/`gcc-11`, ища его в `$PATH`
> (см. `SConstruct`: при `build==native` `compiler_prefix=""`, далее
> `env['CXX'] = compiler_cache + " " + compiler_prefix + cpp_compiler`). Если
> `/opt/gcc-11/bin` нет в `PATH`, ACL падает с:
> ```
> ERROR: Compiler ' g++-11' not found
> ```
> Скрипт `build_cortex_a75.sh` сам добавляет каталог компилятора в `PATH`; при
> ручной сборке это нужно сделать самостоятельно (строка `export PATH=...` выше).

Либо вручную (обратите внимание на `-B${BINUTILS_BIN}` — он направляет gcc к
новому ассемблеру; `export PATH` из блока выше тоже обязателен):

```bash
export CC=/opt/gcc-11/bin/gcc-11
export CXX=/opt/gcc-11/bin/g++-11
export BINUTILS_BIN=/opt/gcc-11/bin

cmake -B build -S . -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  \
  `# --- ТЮНИНГ ПОД CORTEX-A75 (+ -B на новый binutils) ---` \
  -DCMAKE_C_FLAGS="-mtune=cortex-a75 -B${BINUTILS_BIN} -static-libgcc" \
  -DCMAKE_CXX_FLAGS="-mtune=cortex-a75 -B${BINUTILS_BIN} -static-libstdc++ -static-libgcc" \
  -DCMAKE_ASM_FLAGS="-B${BINUTILS_BIN}" \
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
| `ENABLE_TBBBIND_2_5=OFF` | Отключает NUMA/hybrid-pinning (TBBBind). На этом CPU 1 NUMA-узел и 48 одинаковых ядер — биндить нечего, прироста нет. Заодно убирает warning «prebuilt TBBBIND_2_5 is not available». |
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
# с GCC 11 + binutils 2.40 (рекомендуется):
export CC=/opt/gcc-11/bin/gcc-11 CXX=/opt/gcc-11/bin/g++-11
export BINUTILS_BIN=/opt/gcc-11/bin
./build_cortex_a75.sh

# запасной путь на системном GCC 8.3 (KleidiAI выключается флагом):
./build_cortex_a75.sh --no-kleidiai
```

Скрипт перед сборкой:
* добавляет каталог компилятора (`dirname $CC`) в `PATH` — чтобы scons-сборка
  ACL нашла `gcc-11`/`g++-11` по имени (иначе `Compiler ' g++-11' not found`);
* добавляет каталог `libstdc++` нового GCC в `LD_LIBRARY_PATH` — чтобы
  `ld.gold`/`g++` нового toolchain запускались (иначе `ld.gold: ... GLIBCXX_3.4.29
  not found`);
* с `--build-tbb` собирает свой oneTBB и выставляет `TBBROOT`; иначе
  предупреждает, что готовый arm64 TBB не слинкуется на старом glibc;
* проверяет, что и GCC (≥11), и ассемблер понимают `i8mm`/`bf16`, и при проблеме
  подсказывает, что обновить.

---

## 7. Запасной путь: системный toolchain без обновления (KleidiAI OFF)

Если обновлять GCC/binutils сейчас нельзя — собираем на системном GCC 8.3.0,
**отключив KleidiAI** (именно он генерит `+i8mm`/`+bf16` и падает — как на
старом GCC, так и на старом ассемблере). Multi-ISA на 8.3 и так выключен,
поэтому ошибок с `arm_sve.h` не будет.

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
2. Две разные ошибки одного корня — старый toolchain:
   * `cc1: error: invalid feature modifier '...+i8mm'` → старый **GCC** (8.3);
   * `Assembler ... unknown architectural extension i8mm/bf16` → старый
     **binutils** (`as` 2.31), даже если GCC уже 11.
   KleidiAI безусловно собирает i8mm/bf16-микроядра, поэтому нужны **оба**:
   **GCC ≥ 11** И **binutils ≥ 2.34** (рекомендуется 2.40).
3. На старом glibc (AstraLinux ~2.28) ещё две ловушки окружения:
   * `ld.gold: ... GLIBCXX_3.4.29 not found` → инструменты нового toolchain
     требуют свежий libstdc++ → `LD_LIBRARY_PATH=/opt/gcc-11/lib64`;
   * `libtbb.so.12: undefined reference to pthread_create@GLIBC_2.34` →
     готовый arm64 oneTBB собран против нового glibc → собрать свой oneTBB
     (3.4, `--build-tbb`) и задать `TBBROOT`.
4. **Максимальная производительность** → собрать **GCC 11** (3.1) **и
   binutils 2.40** (3.2) **и свой oneTBB** (3.4); выставить `PATH`,
   `LD_LIBRARY_PATH`, `TBBROOT`; сборка с `KleidiAI=ON`, `arm64-v8.2-a`,
   `-mtune=cortex-a75`, `-B$BINUTILS_BIN`, статический libstdc++.
5. **Без обновления toolchain** → собрать с `-DENABLE_KLEIDIAI_FOR_CPU=OFF`
   (раздел 7) — работает на GCC 8.3, но медленнее на квантованных моделях.
