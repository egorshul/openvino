# RISC-V cross-compilation of the CPU plugin with TBB and QEMU validation

This note documents an end-to-end recipe to cross-compile **OpenVINO
`releases/2026/1`** for 64-bit RISC-V on an x86-64 Ubuntu host, with:

* the **Intel CPU plugin enabled** (it is the only inference device on RISC-V),
* **TBB** threading (not OMP),
* a **ResNet-50** model executed under **QEMU** emulating RVV 1.0.

It complements [`build_riscv64.md`](./build_riscv64.md), which describes the
general toolchain options. A turnkey script lives at
[`scripts/riscv64_cross_build_and_run.sh`](../../scripts/riscv64_cross_build_and_run.sh).

## 1. Host prerequisites

Validated on Ubuntu 24.04 (x86-64). Install the distro GNU RISC-V cross
toolchain, binutils, QEMU user-mode emulator and the host build tooling:

```sh
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config python3 \
    gcc-riscv64-linux-gnu g++-riscv64-linux-gnu binutils-riscv64-linux-gnu \
    qemu-user-static qemu-user
```

This provides `riscv64-linux-gnu-{gcc,g++,ld,...}` (GCC 13.3.0) and
`qemu-riscv64-static` (8.2.2), together with the RISC-V sysroot under
`/usr/riscv64-linux-gnu`.

## 2. Configure

```sh
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv64.linux.toolchain.cmake \
  -DTHREADING=TBB \
  -DENABLE_TBBBIND_2_5=OFF \
  -DENABLE_INTEL_CPU=ON \
  -DENABLE_INTEL_GPU=OFF -DENABLE_INTEL_NPU=OFF \
  -DENABLE_SAMPLES=ON -DENABLE_PYTHON=OFF -DENABLE_TESTS=OFF \
  -DENABLE_OV_ONNX_FRONTEND=OFF -DENABLE_OV_PADDLE_FRONTEND=OFF \
  -DENABLE_OV_TF_FRONTEND=OFF -DENABLE_OV_TF_LITE_FRONTEND=OFF \
  -DENABLE_OV_PYTORCH_FRONTEND=OFF -DENABLE_OV_JAX_FRONTEND=OFF \
  -DCMAKE_INSTALL_PREFIX=install_riscv64 \
  -S . -B build_riscv64
```

Notes on the requirement-critical flags:

* **CPU plugin** — `ENABLE_INTEL_CPU` defaults to `ON` for `RISCV64`
  (`cmake/features.cmake`); it is pinned here so the build fails loudly if that
  ever regresses. The configure log prints `ENABLE_INTEL_CPU = ON` and
  `XBYAK_RISCV_V=ON`, i.e. the RVV 1.0 JIT emitters are compiled in.
* **TBB** — `THREADING=TBB` makes CMake fetch the prebuilt
  `oneapi-tbb-2022.3.0-lin-riscv-release.tgz` package (see the `RISCV64` branch
  in `cmake/dependencies.cmake`). The log prints
  `THREADING = TBB` and `TBB (2022.3.0) is found at .../Linux_riscv64/tbb`.
  `ENABLE_TBBBIND_2_5=OFF` avoids a missing prebuilt `tbbbind` package warning;
  there is no NUMA binding library for RISC-V.
* The non-IR frontends are turned off because the ResNet-50 reproducer is fed an
  OpenVINO IR; this keeps the submodule/footprint minimal.

## 3. Build and install

```sh
cmake --build build_riscv64 --parallel "$(nproc)"
cmake --install build_riscv64
```

The CPU plugin (`libopenvino_intel_cpu.so`) and `benchmark_app` are produced as
RISC-V ELF objects:

```sh
$ file build_riscv64/bin/riscv64/Release/benchmark_app
... ELF 64-bit LSB ... UCB RISC-V ... dynamically linked ...
```

## 4. Run ResNet-50 under QEMU (RVV 1.0)

Any ResNet-50 OpenVINO IR works. A FP32 IR can be produced on the host with
torchvision:

```python
import torch, torchvision, openvino as ov
m = torchvision.models.resnet50(weights="IMAGENET1K_V2").eval()
ov.save_model(ov.convert_model(m, example_input=torch.randn(1,3,224,224),
                               input=[1,3,224,224]),
              "resnet50.xml", compress_to_fp16=False)
```

Run it through the freshly built RISC-V `benchmark_app`, letting QEMU emulate a
CPU with the vector extension so the CPU plugin's RVV JIT kernels execute:

```sh
CPU='rv64,v=true,vext_spec=v1.0'
CPU+=',xtheadba=true,xtheadbb=true,xtheadbs=true,xtheadcondmov=true'
CPU+=',xtheadmac=true,xtheadmemidx=true,xtheadmempair=true,xtheadsync=true'
CPU+=',xtheadcmo=true,xtheadfmemidx=true,xtheadfmv=true'

QEMU_LD_PREFIX=/usr/riscv64-linux-gnu \
LD_LIBRARY_PATH=install_riscv64/runtime/lib/riscv64:install_riscv64/runtime/3rdparty/tbb/lib \
qemu-riscv64-static -cpu "$CPU" \
    bin/riscv64/Release/benchmark_app \
    -m resnet50.xml -d CPU -niter 4 -nstreams 1 -hint none
```

> **Why the XThead flags?** The prebuilt `oneapi-tbb-2022.3.0-lin-riscv-release`
> package is compiled for the T-Head Xuantie boards OpenVINO validates on
> (e.g. the Lichee Pi 4A / TH1520 C910), so `libtbb.so` contains XThead custom
> instructions (opcode `0x0B`). A plain `-cpu rv64,v=true` model raises
> `SIGILL` during TBB's static initialization. Emulating the XThead extensions
> alongside RVV 1.0 lets both the CPU plugin JIT and TBB run. On real T-Head
> hardware no such flag is needed.

`benchmark_app` reports the loaded device as `CPU`, `EXECUTION_DEVICES: CPU`,
`TBB_PARTITIONER: STATIC` (confirming TBB threading), the OpenVINO build version
(`2026.1`), and the achieved latency/throughput — confirming the cross-built CPU
plugin runs the model end-to-end under emulation.

## 5. Targeting a RISC-V CPU *without* XThead (still using TBB)

The XThead requirement above is **only** an artefact of the prebuilt oneTBB
package — the CPU plugin itself never emits XThead instructions. For a generic
`rv64gc` core (with or without RVV) that lacks the T-Head custom extensions,
replace the prebuilt TBB with one built for the baseline ISA. OpenVINO skips the
prebuilt download whenever `TBB_DIR`/`TBBROOT` or `ENABLE_SYSTEM_TBB` is set
(see `src/cmake/ov_parallel.cmake`).

**Option A — build oneTBB from source (most portable).** The distro cross
toolchain defaults to `rv64gc`, so the resulting `libtbb.so` carries no XThead
instructions:

```sh
git clone --branch v2022.3.0 https://github.com/uxlfoundation/oneTBB.git
cmake -G Ninja -S oneTBB -B oneTBB/build \
    -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv64.linux.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release -DTBB_TEST=OFF -DTBB_STRICT=OFF \
    -DCMAKE_INSTALL_PREFIX=$PWD/onetbb-rv
cmake --build oneTBB/build --parallel "$(nproc)" --target install
```

Then point the OpenVINO configure step at it (instead of relying on the
download) and rebuild:

```sh
cmake ... -DTHREADING=TBB -DTBB_DIR=$PWD/onetbb-rv/lib/cmake/TBB ...
```

Because oneTBB keeps a stable `libtbb.so.12` ABI, an already-built tree can even
be validated without recompiling OpenVINO by putting the clean library first on
`LD_LIBRARY_PATH`. ResNet-50 then runs under a plain model with no XThead flags:

```sh
QEMU_LD_PREFIX=/usr/riscv64-linux-gnu \
LD_LIBRARY_PATH=onetbb-rv/lib:install_riscv64/runtime/lib/riscv64 \
qemu-riscv64-static -cpu rv64,v=true,vext_spec=v1.0 \
    bin/riscv64/Release/benchmark_app -m resnet50.xml -d CPU -niter 1
```

**Option B — system TBB.** Install the distro `libtbb-dev` for the riscv64
architecture (Debian/Ubuntu build it for baseline `rv64gc`) and configure with
`-DENABLE_SYSTEM_TBB=ON`. On the host this needs multiarch:

```sh
dpkg --add-architecture riscv64 && apt-get update
apt-get install -y libtbb-dev:riscv64
cmake ... -DTHREADING=TBB -DENABLE_SYSTEM_TBB=ON ...
```

Either option keeps **TBB** as the threading backend while producing binaries
that run on a plain `rv64gc` device — no XThead, no `SIGILL`.
