#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Inspect produced IRs: list input shapes, flag remaining dynamic dimensions,
and confirm `beam_idx` (the beam-search input) is present where expected.

Usage:
    python verify_static.py models/llama-3.1-8b-instruct-ov-fp16
    python verify_static.py models/*-ov-*
"""

import sys
from pathlib import Path

import common


def inspect_model_dir(d: Path) -> None:
    print(f"\n=== {d} ===")
    xmls = sorted(d.rglob("*.xml"))
    if not xmls:
        print("  (no .xml files)")
        return
    for xml in xmls:
        rel = xml.relative_to(d)
        if xml.name in common.TOKENIZER_FILES:
            print(f"  {rel}  ->  tokenizer (dynamic by design, skipped)")
            continue
        descs, dynamic = common.report_shapes(xml)
        beam = "  [has beam_idx]" if any("beam_idx" in x for x in descs) else ""
        flag = "DYNAMIC" if dynamic else "static"
        print(f"  {rel}  ->  {flag}{beam}")
        for desc in descs:
            print(f"        {desc}")
        if dynamic:
            print(f"        ^ dynamic inputs: {dynamic}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    common.check_openvino_version()
    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.is_dir():
            inspect_model_dir(path)
        else:
            print(f"skip (not a dir): {path}")


if __name__ == "__main__":
    main()
