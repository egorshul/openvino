# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the OpenVINO 2026.1 static / NPU model-preparation toolkit."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import openvino as ov

# Registering the tokenizers extension lets ov.Core().read_model() open the
# openvino_tokenizer/detokenizer IRs that optimum emits next to LLM/Whisper.
try:
    import openvino_tokenizers  # noqa: F401
except Exception:  # pragma: no cover - extension is optional for SDXL
    pass

OV_REQUIRED = "2026.1"

# Sub-model files that are dynamic *by design* (string ops) and must not be
# treated as a static-shape violation.
TOKENIZER_FILES = ("openvino_tokenizer.xml", "openvino_detokenizer.xml")


def log(msg: str) -> None:
    print(f"[prep] {msg}", flush=True)


def check_openvino_version() -> None:
    ver = ov.get_version()
    if OV_REQUIRED not in ver:
        log(
            f"WARNING: openvino runtime is '{ver}', this toolkit targets {OV_REQUIRED}. "
            "Install the pinned requirements.txt to reproduce exactly."
        )
    else:
        log(f"OpenVINO runtime: {ver}")


def require_hf_token_for_gated(model_id: str) -> None:
    """Llama 3.1 is gated; fail early with a clear message if no token is present."""
    gated_prefixes = ("meta-llama/",)
    if model_id.startswith(gated_prefixes):
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            log(
                f"ERROR: '{model_id}' is a gated model. Accept the license on its "
                "HuggingFace page and export HF_TOKEN=<your token> before running."
            )
            sys.exit(2)


def run_optimum_export(args: list[str]) -> None:
    """Invoke `optimum-cli export openvino` and stream output."""
    cmd = ["optimum-cli", "export", "openvino", *args]
    log("running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def resolve_model_source(model_id: str, revision: str | None) -> str:
    """Return a -m argument for optimum-cli.

    optimum-cli has no --revision flag, so to pin an exact MLPerf checkpoint
    commit we snapshot-download that revision locally and export from the path.
    Without a revision the HF id is passed through (resolves to the latest main).
    """
    if not revision:
        return model_id
    from huggingface_hub import snapshot_download

    log(f"pinning {model_id} @ {revision[:12]} (snapshot download)")
    local = snapshot_download(repo_id=model_id, revision=revision)
    return local


def report_shapes(xml_path: Path) -> tuple[list[str], list[str]]:
    """Return (input descriptions, names of inputs that still have dynamic dims)."""
    model = ov.Core().read_model(xml_path)
    descs, dynamic = [], []
    for inp in model.inputs:
        name = inp.get_any_name()
        ps = inp.get_partial_shape()
        descs.append(f"{name}: {ps}")
        if ps.is_dynamic:
            dynamic.append(name)
    return descs, dynamic


def has_input(xml_path: Path, name: str) -> bool:
    model = ov.Core().read_model(xml_path)
    return any(i.get_any_name() == name for i in model.inputs)


def archive_dir(model_dir: Path) -> int:
    """Report the on-disk size of a produced model directory in MiB."""
    total = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    return total // (1024 * 1024)


def ensure_clean_outdir(out: Path, overwrite: bool) -> None:
    if out.exists():
        if overwrite:
            log(f"removing existing {out}")
            shutil.rmtree(out)
        else:
            log(f"ERROR: {out} already exists (use --overwrite to replace).")
            sys.exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
