#!/usr/bin/env python3
"""Read summary_*.csv under RESULT_ROOT and print mean ± std per column."""

import csv
from pathlib import Path

import numpy as np

RESULT_ROOT = Path("/home/yuliangyan/Code/Trust-App-AI-Lab/molecular_glue_design/DeepTernary/results/MGD_test")

csvs = sorted(RESULT_ROOT.glob("**/summary_*.csv"))
if not csvs:
    print("No summary_*.csv under", RESULT_ROOT)
    raise SystemExit(1)

skip = {"seed", "complex_pred"}
with csvs[0].open() as f:
    cols = [c for c in csv.DictReader(f).fieldnames if c not in skip]

per_complex = {c: [] for c in cols}
for p in csvs:
    with p.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        continue
    for c in cols:
        xs = []
        for r in rows:
            t = (r.get(c) or "").strip()
            if not t:
                continue
            try:
                xs.append(float(t))
            except ValueError:
                pass
        per_complex[c].append(float(np.mean(xs)) if xs else np.nan)

print(len(csvs), "complexes,", RESULT_ROOT, "\n")
for c in cols:
    x = np.array(per_complex[c], dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        print(f"{c}: n/a")
        continue
    s = x.std(ddof=1) if len(x) > 1 else 0.0
    print(f"{c}: mean={x.mean():.4f}  std={s:.4f}  (n={len(x)})")
