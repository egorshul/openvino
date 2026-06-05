#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Export a causal LLM (default: Llama 3.1 8B Instruct) to an OpenVINO 2026.1
FP16 IR suitable for devices without runtime dynamism (Intel NPU).

Key design points
-----------------
* The model is exported **stateful** (the default). A stateful causal LM keeps
  a `beam_idx` input which is exactly the "beam-search input" the model must
  accept: at each step the runtime feeds `beam_idx` to reorder the internal KV
  cache for the selected beams, so beam search needs no graph re-compilation.

* optimum-intel intentionally refuses to bake fully static shapes into a causal
  LM IR ("Static shapes are not supported for causal language model"), because
  prefill (seq=N) and decode (seq=1) need different shapes. The elimination of
  runtime dynamism is therefore done by the OpenVINO GenAI **NPU** pipeline,
  which compiles two static sub-graphs (prefill / decode) from this same IR.
  See the README for the exact `openvino_genai.LLMPipeline(..., "NPU")` call.

So the IR produced here is the artifact you ship; "no runtime dynamism" is
realized at load time by the GenAI NPU pipeline, and the `beam_idx` input makes
the model accept beam-search inputs.
"""

import argparse
from pathlib import Path

import common


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--output", type=Path, default=Path("models/llama-3.1-8b-instruct-ov-fp16"))
    p.add_argument("--weight-format", default="fp16", choices=["fp16", "fp32", "int8", "int4"],
                   help="FP16 per request; int4/int8 are common for NPU LLMs (smaller, faster).")
    p.add_argument("--stateless", action="store_true",
                   help="Add --disable-stateful: emit explicit past/present KV I/O instead of a "
                        "stateful model. Use only if your runtime cannot consume stateful models. "
                        "Note: this DROPS the beam_idx input.")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    common.check_openvino_version()
    common.require_hf_token_for_gated(args.model_id)
    common.ensure_clean_outdir(args.output, args.overwrite)

    export_args = [
        "-m", args.model_id,
        "--task", "text-generation-with-past",
        "--weight-format", args.weight_format,
    ]
    if args.stateless:
        export_args.append("--disable-stateful")
    if args.trust_remote_code:
        export_args.append("--trust-remote-code")
    export_args.append(str(args.output))

    common.run_optimum_export(export_args)

    xml = args.output / "openvino_model.xml"
    descs, dynamic = common.report_shapes(xml)
    common.log(f"exported {args.model_id} -> {args.output} ({common.archive_dir(args.output)} MiB)")
    common.log("model inputs:")
    for d in descs:
        common.log("    " + d)

    if not args.stateless:
        if common.has_input(xml, "beam_idx"):
            common.log("OK: 'beam_idx' input present -> model accepts beam-search inputs.")
        else:
            common.log("WARNING: 'beam_idx' input missing on a stateful export (unexpected).")
    common.log(
        "Reminder: run this IR on NPU via openvino_genai.LLMPipeline(path, 'NPU', "
        "MAX_PROMPT_LEN=..., MIN_RESPONSE_LEN=...) for static (dynamism-free) compilation."
    )


if __name__ == "__main__":
    main()
