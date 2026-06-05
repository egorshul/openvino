#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Export Whisper large v3 to an OpenVINO 2026.1 FP16 IR for static / NPU use.

The export produces an encoder plus a stateful decoder. As with text LLMs, the
stateful decoder exposes a `beam_idx` input, so the model accepts beam-search
inputs (the runtime reorders the decoder KV cache per beam without recompiling).

Two layers of "no runtime dynamism":
  1. The encoder input (log-mel features) is fixed: [batch, 128, 3000] for
     large-v3, so the encoder is naturally static.
  2. Decoder static batch is fixed to the number of beams via an optional
     reshape step (--num-beams). At runtime use openvino_genai.WhisperPipeline
     on NPU, which statically compiles encoder + decoder.
"""

import argparse
from pathlib import Path

import common


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="openai/whisper-large-v3")
    p.add_argument("--output", type=Path, default=Path("models/whisper-large-v3-ov-fp16"))
    p.add_argument("--weight-format", default="fp16", choices=["fp16", "fp32", "int8"])
    p.add_argument("--num-beams", type=int, default=0,
                   help="If >0, statically reshape the decoder batch to this many beams "
                        "(fixes the beam dimension so no runtime dynamism is needed).")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    common.check_openvino_version()
    common.ensure_clean_outdir(args.output, args.overwrite)

    common.run_optimum_export([
        "-m", args.model_id,
        "--task", "automatic-speech-recognition",
        "--weight-format", args.weight_format,
        str(args.output),
    ])

    dec_xml = args.output / "openvino_decoder_model.xml"
    if not dec_xml.exists():  # stateful single-decoder layout
        dec_xml = args.output / "openvino_decoder_with_past_model.xml"

    if args.num_beams > 0:
        _reshape_static(args.output, args.num_beams)

    common.log(f"exported {args.model_id} -> {args.output} ({common.archive_dir(args.output)} MiB)")
    if dec_xml.exists() and common.has_input(dec_xml, "beam_idx"):
        common.log("OK: decoder 'beam_idx' input present -> model accepts beam-search inputs.")
    common.log(
        "Reminder: run on NPU via openvino_genai.WhisperPipeline(path, 'NPU'); "
        "set num_beams in WhisperGenerationConfig for beam search."
    )


def _reshape_static(out: Path, num_beams: int) -> None:
    """Use the optimum pipeline API to fix the decoder batch to num_beams."""
    from optimum.intel import OVModelForSpeechSeq2Seq

    common.log(f"reshaping decoder to static batch = num_beams = {num_beams}")
    m = OVModelForSpeechSeq2Seq.from_pretrained(out)
    # sequence_length=-1 keeps the (already fixed) mel/seq handling; only the
    # batch (beam) dimension is pinned. Adjust if your runtime needs a fixed seq.
    m.reshape(batch_size=num_beams, sequence_length=-1)
    m.half()
    m.save_pretrained(out)
    common.log("static reshape done.")


if __name__ == "__main__":
    main()
