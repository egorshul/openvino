# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""MLPerf Inference reference presets (Closed division, Datacenter).

Values below are taken from the MLCommons inference reference implementations
and are used by the export scripts (with --mlperf) to align arguments:
checkpoint commits, GREEDY decoding (no beam search), and static-shape budgets.

Sources (verified):
  * Llama 3.1 8B : https://github.com/mlcommons/inference/tree/master/language/llama3.1-8b
  * Whisper v3   : https://github.com/mlcommons/inference/tree/master/speech2text
  * SDXL         : https://github.com/mlcommons/inference/tree/master/text_to_image
                   https://github.com/mlcommons/inference/blob/master/text_to_image/backend_pytorch.py
  * Rules        : https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc

NOTE: MLPerf rounds change targets/commits. Re-check the rules for the round you
submit to. A valid Closed submission also requires the LoadGen harness, the exact
datasets, and the official accuracy scripts — this toolkit only covers the model.
"""

MLPERF = {
    "llama3.1-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        # Pinned reference checkpoint commit.
        "revision": "be673f326cab4cd22ccfef76109faf68e41aa5f1",
        "reference_precision": "bfloat16",
        "dataset": "CNN/DailyMail (datacenter: 13,368 samples)",
        # GREEDY: no beam search. beam_idx stays in the IR but num_beams must be 1.
        "decoding": {"do_sample": False, "num_beams": 1, "temperature": 0.0},
        # Inputs are padded to 1024 tokens in the reference dataset loader.
        "max_prompt_len": 1024,
        "scenarios": ["Offline", "Server"],
        "accuracy_targets_99pct": {
            "rouge1": 38.7792, "rouge2": 15.9075,
            "rougeL": 24.4957, "rougeLsum": 35.793, "gen_len_pct": 90,
        },
    },
    "whisper-large-v3": {
        "model_id": "openai/whisper-large-v3",
        "revision": "06f233fe06e710322aca913c1bc4249a0d71fce1",
        "dataset": "LibriSpeech dev-clean + dev-other (~10h)",
        # GREEDY (temperature=0). Fixed 30s audio is mandatory.
        "decoding": {"do_sample": False, "num_beams": 1, "temperature": 0.0,
                     "max_new_tokens": 200},
        "audio": {"sample_rate": 16000, "n_mels": 128, "chunk_seconds": 30,
                  "mel_frames": 3000, "max_model_len": 448},
        "scenarios": ["Offline", "Server"],
        "accuracy_target": {"wer_pct": 2.0671, "min_fraction_of_reference": 0.99},
    },
    "sdxl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        # MLPerf snapshots the HF pipeline; pin the round's revision here.
        "revision": None,  # set to the commit your MLPerf round mandates
        "reference_precision": "fp32 (fp16/bf16 permitted)",
        "dataset": "COCO-2014 (5,000 captions/images)",
        # Diffusion pipeline runtime settings (NOT baked into the IR):
        "runtime": {
            "scheduler": "EulerDiscreteScheduler",
            "num_inference_steps": 20,
            "guidance_scale": 8.0,
            "height": 1024,
            "width": 1024,
            "negative_prompt": "normal quality, low quality, worst quality, "
                               "low res, blurry, nsfw, nude",
            "latents": "externally seeded (passed into the pipeline)",
        },
        # Reference FID/CLIP bounds (confirm against the round's rules).
        "accuracy_bounds": {
            "FID": [23.01085758, 23.95007626],
            "CLIP": [31.68631873, 31.81331801],
        },
    },
}
