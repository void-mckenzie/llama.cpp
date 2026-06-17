#!/usr/bin/env python3
"""Quick inspection: what's actually in the first 16K chars of calibration_datav5.txt?"""
import re
from pathlib import Path

text = Path(r"C:/Users/MukkeshGanesh/Downloads/kv_pca_standalone/calibration_datav5.txt").read_text(encoding="utf-8")[:16384]

paragraphs = [p for p in text.split("\n\n") if p.strip()]
print(f"total chars: {len(text)}")
print(f"paragraphs: {len(paragraphs)}")
print()

code_re = re.compile(r"def |class |import |\{\}|=<|=>|::")
math_re = re.compile(r"\\\\\[|\\\\\(|\\\\frac|\\\\text\{")

print(f"first 150 chars of each paragraph:")
for i, p in enumerate(paragraphs):
    snippet = p[:150].replace("\n", " ").strip()
    is_code = bool(code_re.search(p))
    is_math = bool(math_re.search(p))
    tag = "[CODE]" if is_code else "[MATH]" if is_math else "[TEXT]"
    print(f"  [{i}] {tag} {snippet!r}")
