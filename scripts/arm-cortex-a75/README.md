# OpenVINO 2026.1 — maximum-performance build for ARM Cortex-A75

Scripts to build OpenVINO 2026.1 tuned for a 48-core aarch64 Cortex-A75 host
(`CPU part 0xd0a`, armv8.2-a) and to benchmark the full Stable Diffusion XL
pipeline with `benchmark_app`.

> **Сборка на Astra Linux 4.7 arm (gcc 8.3.0):** подробная пошаговая инструкция
> на русском — в [`BUILD_ASTRALINUX.md`](BUILD_ASTRALINUX.md). Там же — разбор,
> почему штатного gcc 8.3.0 достаточно, и как при необходимости обновить gcc.

## Target CPU capabilities

```
Flags: fp asimd evtstrm crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp
```

| Feature | Available | Used by |
|---|---|---|
| NEON/ASIMD fp32 | yes | ACL/oneDNN fp32 kernels |
| fp16 arithmetic (`fphp asimdhp`) | yes | ACL fp16 kernels — default inference precision on ARM |
| int8 dot product (`asimddp`) | yes | ACL/KleidiAI int8 GEMM — fastest path for int8 models |
| `asimdrdm`, `lrcpc`, `crc32`, `atomics` | yes | general codegen via `-mcpu=cortex-a75` |
| SVE / SVE2 / SME, bf16, i8mm | **no** | — excluded from the build |

## What the build script does

`build_openvino_cortex_a75.sh`:

* `-mcpu=cortex-a75` globally — armv8.2-a + fp16 + dotprod + rcpc codegen and
  Cortex-A75 instruction scheduling for all of OpenVINO (gcc ≥ 8; the target's
  gcc 11.4 and cross gcc 13 both support it).
* `OV_CPU_ARM_TARGET_ARCH=arm64-v8.2-a` + `OV_CPU_AARCH64_USE_MULTI_ISA=OFF` —
  ARM Compute Library is compiled directly for armv8.2-a with fp16/dotprod
  kernels baked in, instead of the default multi-ISA build that also carries
  SVE/SVE2/SME kernels the A75 can never execute (smaller binary, no runtime
  dispatch overhead, faster build).
* `ENABLE_LTO=ON`, `CMAKE_BUILD_TYPE=Release`, `THREADING=TBB` (best scaling on
  48 cores), `ENABLE_PROFILING_ITT=OFF`.
* Only the CPU plugin and the IR frontend are built (enough for `benchmark_app`
  with OpenVINO IR models); GPU/NPU plugins, tests and extra frontends are off.
  Set `EXTRA_FRONTENDS=ON` if you also need ONNX/TF/TFLite/PyTorch import.
* Works natively on the aarch64 host and cross-compiles from x86_64
  (auto-detected; cross mode uses `cmake/arm64.toolchain.cmake` and requires
  `gcc-aarch64-linux-gnu g++-aarch64-linux-gnu`).

### Build (on the 48-core target)

```bash
sudo apt install cmake ninja-build scons  # or pip install scons
git clone --branch 2026.1.2 https://github.com/openvinotoolkit/openvino.git
cd openvino
git submodule update --init --recursive
./scripts/arm-cortex-a75/build_openvino_cortex_a75.sh
```

Binaries: `bin/aarch64/Release/`, install tree: `install-cortex-a75/`.

## Benchmarking the SDXL pipeline

`benchmark_sdxl.sh` downloads pre-converted OpenVINO IRs of
[`OpenVINO/stable-diffusion-xl-base-1.0-fp16-ov`](https://huggingface.co/OpenVINO/stable-diffusion-xl-base-1.0-fp16-ov)
and runs `benchmark_app -d CPU` on every inference model of the pipeline with
the static shapes of a real 1024×1024 generation step (UNet batch 2 for
classifier-free guidance):

| Model | Shape |
|---|---|
| text_encoder | `input_ids[1,77]` |
| text_encoder_2 | `input_ids[1,77]` |
| unet | `sample[2,4,128,128]`, `timestep[1]`, `encoder_hidden_states[2,77,2048]`, `text_embeds[2,1280]`, `time_ids[2,6]` |
| vae_encoder | `sample[1,3,1024,1024]` |
| vae_decoder | `latent_sample[1,4,128,128]` |

(`tokenizer`/`tokenizer_2` are not benchmarked — they require the
openvino_tokenizers extension and take a negligible share of pipeline time.)

```bash
./scripts/arm-cortex-a75/benchmark_sdxl.sh bin/aarch64/Release/benchmark_app
```

Useful knobs (env vars): `HINT=latency|throughput`, `INFER_PRECISION=f16|f32`,
`RESOLUTION=1024`, `ONLY="unet vae_decoder"`, `NITER_OVERRIDE=N`,
`MODEL_REPO=...`.

### Getting the most performance on Cortex-A75

* Keep `INFER_PRECISION=f16` (default): A75 executes fp16 ASIMD natively,
  roughly doubling GEMM/conv throughput vs fp32.
* For the fastest SDXL, use the int8-quantized pipeline — int8 GEMMs run on the
  `asimddp` dot-product kernels:
  `MODEL_REPO=OpenVINO/stable-diffusion-xl-base-1.0-int8-ov ./benchmark_sdxl.sh ...`
* `-hint latency` minimizes single-image time using all 48 cores per request;
  `-hint throughput` maximizes images/sec with several parallel requests.
