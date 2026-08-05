#!/usr/bin/env python3
"""
Aggregate per-complex best DockQ (from detailed_results) and ligand metrics from
ligand_eval_results, then write a merged JSON (+ optional CSV).

DockQ: max over trajectories (higher is better). Vina energies (``vina_dock``, ``vina_min``,
``vina_score``) are kcal/mol-style affinities: **lower is better** (more negative = stronger).
We take the **minimum** of each over all trajectories; ``traj_idx_best_vina`` / ``qed_at_best_vina``
/ etc. come from the trajectory with the **lowest** ``vina_dock`` (tie: smaller ``traj_idx``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def best_dockq_from_sample(sample: dict[str, Any]) -> float | None:
    dockqs = sample.get("all_trajectories_dockq")
    if dockqs:
        return max(float(x) for x in dockqs)
    bt = sample.get("best_trajectory") or {}
    if "dockq" in bt:
        return float(bt["dockq"])
    return None


def load_detailed_dockq(path: Path) -> dict[str, float]:
    with path.open() as f:
        data = json.load(f)
    out: dict[str, float] = {}
    for sample in data.get("sample_results", []):
        name = sample.get("name")
        if not name:
            continue
        b = best_dockq_from_sample(sample)
        if b is not None:
            out[str(name)] = b
    return out


def _finite_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _traj_idx_int(r: dict[str, Any]) -> int:
    t = r.get("traj_idx")
    try:
        return int(t)
    except (TypeError, ValueError):
        return 10**9


def load_ligand_best(path: Path) -> dict[str, dict[str, Any]]:
    """Per complex: min ``vina_dock`` / ``vina_min`` / ``vina_score`` (lower = better); anchor = min ``vina_dock`` row."""
    with path.open() as f:
        data = json.load(f)
    rows = data.get("results", [])

    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        n = r.get("name")
        if n:
            by_name[str(n)].append(r)

    best_vina: dict[str, dict[str, Any]] = {}
    best_qed: dict[str, tuple[float, dict[str, Any]]] = {}

    for name, group in by_name.items():
        vmins = [v for r in group if (v := _finite_float(r.get("vina_min"))) is not None]
        vscores = [v for r in group if (v := _finite_float(r.get("vina_score"))) is not None]
        min_vina_min = min(vmins) if vmins else None
        min_vina_score = min(vscores) if vscores else None

        best_row: dict[str, Any] | None = None
        best_dock: float | None = None
        best_tid = 10**9
        for r in group:
            vd = _finite_float(r.get("vina_dock"))
            if vd is None:
                continue
            tid_i = _traj_idx_int(r)
            if best_row is None or vd < best_dock or (vd == best_dock and tid_i < best_tid):
                best_row = r
                best_dock = vd
                best_tid = tid_i

        if best_row is None or best_dock is None:
            continue

        best_vina[name] = {
            "vina_dock": best_dock,
            "vina_min": min_vina_min,
            "vina_score": min_vina_score,
            "high_affinity": best_row.get("high_affinity"),
            "traj_idx_best_vina": best_row.get("traj_idx"),
            "sample_idx": best_row.get("sample_idx"),
            "qed_at_best_vina": _finite_float(best_row.get("qed")),
            "sa_at_best_vina": _finite_float(best_row.get("sa")),
            "smiles_at_best_vina": best_row.get("smiles"),
        }

    for r in rows:
        name = r.get("name")
        if not name:
            continue
        name = str(name)
        q = r.get("qed")
        if q is not None:
            q = float(q)
            if name not in best_qed or q > best_qed[name][0]:
                best_qed[name] = (q, r)

    out: dict[str, dict[str, Any]] = {}
    for name, v in best_vina.items():
        entry = dict(v)
        if name in best_qed:
            qmax, rmax = best_qed[name]
            entry["best_qed"] = qmax
            entry["traj_idx_best_qed"] = rmax.get("traj_idx")
            entry["vina_dock_at_best_qed"] = _finite_float(rmax.get("vina_dock"))
        else:
            entry["best_qed"] = None
            entry["traj_idx_best_qed"] = None
            entry["vina_dock_at_best_qed"] = None
        out[name] = entry
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--detailed",
        type=Path,
        default=Path("evaluation_results_0422/detailed_results.json"),
        help="Path to detailed_results.json (DockQ source).",
    )
    p.add_argument(
        "--ligand",
        type=Path,
        default=Path("evaluation_results_0422/ligand_eval_results.json"),
        help="Path to ligand_eval_results.json (Vina / QED source).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("evaluation_results_0422/per_complex_best_metrics.json"),
        help="Output JSON path.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="If set, also write a flat CSV with the same rows.",
    )
    args = p.parse_args()

    dockq_by = load_detailed_dockq(args.detailed)
    ligand_by = load_ligand_best(args.ligand)

    all_names = sorted(set(dockq_by) | set(ligand_by))
    merged: list[dict[str, Any]] = []
    for name in all_names:
        row: dict[str, Any] = {"name": name, "best_dockq": dockq_by.get(name)}
        lv = ligand_by.get(name)
        if lv:
            row.update(
                {
                    "best_vina_dock": lv["vina_dock"],
                    "best_vina_min": lv["vina_min"],
                    "best_vina_score": lv["vina_score"],
                    "high_affinity_at_best_vina": lv["high_affinity"],
                    "traj_idx_best_vina": lv["traj_idx_best_vina"],
                    "qed_at_best_vina": lv["qed_at_best_vina"],
                    "sa_at_best_vina": lv["sa_at_best_vina"],
                    "smiles_at_best_vina": lv.get("smiles_at_best_vina"),
                    "best_qed": lv["best_qed"],
                    "traj_idx_best_qed": lv["traj_idx_best_qed"],
                    "vina_dock_at_best_qed": lv["vina_dock_at_best_qed"],
                }
            )
        else:
            row.update(
                {
                    "best_vina_dock": None,
                    "best_vina_min": None,
                    "best_vina_score": None,
                    "high_affinity_at_best_vina": None,
                    "traj_idx_best_vina": None,
                    "qed_at_best_vina": None,
                    "sa_at_best_vina": None,
                    "best_qed": None,
                    "traj_idx_best_qed": None,
                    "vina_dock_at_best_qed": None,
                    "smiles_at_best_vina": None,
                }
            )
        merged.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sources": {
            "detailed_results": str(args.detailed.resolve()),
            "ligand_eval_results": str(args.ligand.resolve()),
        },
        "n_complexes": len(merged),
        "n_with_dockq": sum(1 for r in merged if r["best_dockq"] is not None),
        "n_with_ligand": sum(1 for r in merged if r["best_vina_dock"] is not None),
        "results": merged,
    }
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.output} ({len(merged)} complexes).")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(merged[0].keys()) if merged else []
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(merged)
        print(f"Wrote {args.csv}.")


if __name__ == "__main__":
    main()
