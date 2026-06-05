#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Export Stable Diffusion XL (base 1.0) to a fully static OpenVINO 2026.1 FP16 IR.

SDXL is a diffusion pipeline (text encoders + UNet + VAE) and has no beam
search. "No runtime dynamism" here simply means every sub-model has fixed
shapes: a fixed batch, a fixed image resolution, and a fixed number of images
per prompt. With classifier-free guidance the UNet batch is internally doubled
(2 * batch), which `pipe.reshape(...)` handles for you.

Unlike causal LMs, SDXL sub-models CAN be baked fully static, so the resulting
IRs contain no dynamic dimensions and run as-is on NPU.
"""

import argparse
from pathlib import Path

import common


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--output", type=Path, default=Path("models/sdxl-base-1.0-ov-fp16-static"))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--num-images-per-prompt", type=int, default=1)
    p.add_argument("--revision", default=None,
                   help="Pin an exact checkpoint commit/revision for the SDXL pipeline.")
    p.add_argument("--mlperf", action="store_true",
                   help="Align to the MLPerf Inference reference: 1024x1024 and print the exact "
                        "runtime diffusion settings (scheduler/steps/guidance) to use.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    preset = None
    if args.mlperf:
        from mlperf_presets import MLPERF
        preset = MLPERF["sdxl"]
        args.model_id = preset["model_id"]
        args.revision = args.revision or preset["revision"]
        rt = preset["runtime"]
        args.height, args.width = rt["height"], rt["width"]
        if not args.revision:
            common.log("MLPerf mode: NOTE set --revision to your round's mandated SDXL commit "
                       "(README has no fixed hash; the reference snapshots the HF pipeline).")

    common.check_openvino_version()
    common.ensure_clean_outdir(args.output, args.overwrite)

    from optimum.intel import OVStableDiffusionXLPipeline

    common.log(f"exporting {args.model_id} (this downloads ~7 GB and converts each sub-model)")
    kwargs = {"revision": args.revision} if args.revision else {}
    pipe = OVStableDiffusionXLPipeline.from_pretrained(args.model_id, export=True, **kwargs)

    common.log(
        f"reshaping to static: batch={args.batch_size}, {args.height}x{args.width}, "
        f"num_images_per_prompt={args.num_images_per_prompt}"
    )
    pipe.reshape(
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        num_images_per_prompt=args.num_images_per_prompt,
    )
    pipe.half()  # FP16 weights
    pipe.save_pretrained(args.output)

    common.log(f"saved -> {args.output} ({common.archive_dir(args.output)} MiB)")
    _report_static(args.output)

    if preset:
        rt = preset["runtime"]
        common.log("MLPerf (Closed) runtime settings to match the reference exactly:")
        common.log(f"    scheduler={rt['scheduler']}, num_inference_steps={rt['num_inference_steps']}, "
                   f"guidance_scale={rt['guidance_scale']}, {rt['height']}x{rt['width']}")
        common.log(f"    negative_prompt={rt['negative_prompt']!r}")
        common.log(f"    latents: {rt['latents']}")
        b = preset["accuracy_bounds"]
        common.log(f"    accuracy bounds (confirm vs round): FID {b['FID']}, CLIP {b['CLIP']}")


def _report_static(out: Path) -> None:
    any_dynamic = False
    for xml in sorted(out.rglob("*.xml")):
        if xml.name in common.TOKENIZER_FILES:
            continue
        _, dynamic = common.report_shapes(xml)
        rel = xml.relative_to(out)
        if dynamic:
            any_dynamic = True
            common.log(f"  {rel}: DYNAMIC inputs -> {dynamic}")
        else:
            common.log(f"  {rel}: all inputs static")
    common.log("RESULT: " + ("some sub-models still dynamic (see above)" if any_dynamic
                             else "all SDXL sub-models are fully static."))


if __name__ == "__main__":
    main()
