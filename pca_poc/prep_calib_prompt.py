#!/usr/bin/env python3
"""Trivial helper: dump first N chars of calibration_datav5.txt to a prompt file."""
import sys
from pathlib import Path

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"C:/Users/MukkeshGanesh/Downloads/kv_pca_standalone/calibration_datav5.txt")
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    r"F:/Repos/llama.cpp/kl_dumps/calib_dump/prompt.txt")
nchars = int(sys.argv[3]) if len(sys.argv) > 3 else 16384

dst.parent.mkdir(parents=True, exist_ok=True)
text = src.read_text(encoding="utf-8")[:nchars]
dst.write_text(text, encoding="utf-8")
print(f"wrote {len(text)} chars to {dst}")
