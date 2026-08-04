import argparse
import contextlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from utils import reconstruct
from utils.evaluation import scoring_func


def evaluate_ligand_predictions(pred_pos_list, pred_v_list, gt_pos, gt_v):
    """Per-sample ligand RMSD and atom-type accuracy (aligned with TargetDiff evaluation)."""
    gt_pos_np = gt_pos.detach().cpu().numpy().astype(np.float64)
    gt_v_np = gt_v.detach().cpu().numpy().astype(np.int64)

    rmsd_list, seq_acc_list = [], []
    for pred_pos, pred_v in zip(pred_pos_list, pred_v_list):
        pred_pos = np.asarray(pred_pos, dtype=np.float64)
        pred_v = np.asarray(pred_v, dtype=np.int64)

        if pred_pos.shape != gt_pos_np.shape or pred_v.shape != gt_v_np.shape:
            rmsd_list.append(float("nan"))
            seq_acc_list.append(float("nan"))
            continue

        diff = pred_pos - gt_pos_np
        rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=-1)))
        seq_acc = np.mean((pred_v == gt_v_np).astype(np.float64))
        rmsd_list.append(float(rmsd))
        seq_acc_list.append(float(seq_acc))

    valid_rmsd = [x for x in rmsd_list if np.isfinite(x)]
    valid_seq_acc = [x for x in seq_acc_list if np.isfinite(x)]
    summary = {
        "num_samples": len(rmsd_list),
        "num_valid_metrics": len(valid_rmsd),
        "ligand_rmsd_mean": float(np.mean(valid_rmsd)) if valid_rmsd else None,
        "ligand_rmsd_std": float(np.std(valid_rmsd)) if valid_rmsd else None,
        "ligand_rmsd_min": float(np.min(valid_rmsd)) if valid_rmsd else None,
        "ligand_rmsd_max": float(np.max(valid_rmsd)) if valid_rmsd else None,
        "seq_acc_mean": float(np.mean(valid_seq_acc)) if valid_seq_acc else None,
        "seq_acc_std": float(np.std(valid_seq_acc)) if valid_seq_acc else None,
    }
    return rmsd_list, seq_acc_list, summary


def summarize_valid_values(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "median": None, "num_valid": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "num_valid": int(arr.size),
    }


def run_vina_metrics(
    rdmol,
    protein_pdb,
    *,
    debug=False,
    sample_name="",
    high_affinity_threshold=-7.0,
    dock_exhaustiveness=8,
    dock_n_poses=20,
):
    prefix = f"[Vina:{sample_name}] " if sample_name else "[Vina] "
    if rdmol is None:
        return None, None, None, None
    if protein_pdb is None or (not os.path.isfile(protein_pdb)):
        if debug:
            print(f"{prefix}skip: protein pdb not found at {protein_pdb}")
        return None, None, None, None
    try:
        from meeko import MoleculePreparation
        from openbabel import pybel
        from vina import Vina
        import AutoDockTools
    except ImportError as e:
        if debug:
            print(f"{prefix}import error: {e}")
        return None, None, None, None

    root = os.environ.get("VINA_TMP_DIR", os.path.join(os.getcwd(), ".vina_tmp"))
    os.makedirs(root, exist_ok=True)
    work = tempfile.mkdtemp(prefix="vina_", dir=root)
    try:
        rec = os.path.join(work, "rec.pdb")
        shutil.copy(protein_pdb, rec)
        mol_h = Chem.AddHs(Chem.Mol(rdmol), addCoords=True)
        lig_sdf = os.path.join(work, "lig.sdf")
        writer = Chem.SDWriter(lig_sdf)
        writer.write(mol_h)
        writer.close()

        lig_pdbqt = os.path.join(work, "lig.pdbqt")
        ob_mol = next(pybel.readfile("sdf", lig_sdf))
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                mprep = MoleculePreparation()
                mprep.prepare(ob_mol.OBMol)
                mprep.write_pdbqt_file(lig_pdbqt)

        rec_pqr = os.path.join(work, "rec.pqr")
        rec_pdbqt = os.path.join(work, "rec.pdbqt")
        pqr_proc = subprocess.run(
            ["pdb2pqr30", "--ff=AMBER", rec, rec_pqr],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        prep_rec = os.path.join(AutoDockTools.__path__[0], "Utilities24", "prepare_receptor4.py")
        prep_proc = subprocess.run(
            ["python3", prep_rec, "-r", rec_pqr, "-o", rec_pdbqt],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if debug and (pqr_proc.returncode != 0 or prep_proc.returncode != 0):
            print(
                f"{prefix}prep warning: pdb2pqr30 rc={pqr_proc.returncode}, "
                f"prepare_receptor4 rc={prep_proc.returncode}"
            )
        if not os.path.isfile(lig_pdbqt) or not os.path.isfile(rec_pdbqt):
            if debug:
                print(f"{prefix}skip: missing pdbqt files")
            return None, None, None, None

        pos = mol_h.GetConformer(0).GetPositions()
        center = (pos.max(axis=0) + pos.min(axis=0)) / 2.0
        box = (pos.max(axis=0) - pos.min(axis=0)) + 5.0

        vina = Vina(sf_name="vina", seed=0, verbosity=0)
        vina.set_receptor(rec_pdbqt)
        vina.set_ligand_from_file(lig_pdbqt)
        vina.compute_vina_maps(center=center.tolist(), box_size=box.tolist())

        vina_score = float(vina.score()[0])
        vina_min = None
        try:
            vina_min = float(vina.optimize()[0])
        except Exception as e:
            if debug:
                print(f"{prefix}optimize failed: {type(e).__name__}: {e}")

        vina_dock = None
        try:
            vina.dock(exhaustiveness=int(dock_exhaustiveness), n_poses=int(dock_n_poses))
            pose_scores = vina.energies(n_poses=1)
            if pose_scores is not None and len(pose_scores) > 0:
                vina_dock = float(pose_scores[0][0])
        except Exception as e:
            if debug:
                print(f"{prefix}dock failed: {type(e).__name__}: {e}")

        high_affinity = None if vina_dock is None else int(vina_dock <= high_affinity_threshold)
        return vina_score, vina_min, vina_dock, high_affinity
    except Exception as e:
        if debug:
            print(f"{prefix}runtime error: {type(e).__name__}: {e}")
        return None, None, None, None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def resolve_complex_name(result_file: Path, data_obj):
    ligand_filename = getattr(data_obj, "ligand_filename", None)
    if isinstance(ligand_filename, str) and ligand_filename:
        parent = Path(ligand_filename).parent
        if str(parent) not in (".", ""):
            return parent.name
    stem = result_file.stem
    if stem.startswith("result_"):
        return stem[len("result_"):]
    return stem


def resolve_protein_pdb(data_obj, complex_name: str, dataset_root: str | None, mgd_test_dir: str | None):
    # Priority-1: keep docking protein consistent with TargetDiff/PocketXMol on MGD_test.
    if mgd_test_dir:
        mgd_candidate = Path(mgd_test_dir) / complex_name / "protein1.pdb"
        if mgd_candidate.is_file():
            return str(mgd_candidate)

    protein_filename = getattr(data_obj, "protein_filename", None)
    if not isinstance(protein_filename, str) or protein_filename == "":
        return None
    p = Path(protein_filename)
    if p.is_file():
        return str(p)
    if dataset_root:
        p2 = Path(dataset_root) / protein_filename
        if p2.is_file():
            return str(p2)
        # If protein filename is relative, also try inside MGD complex dir.
        p2b = Path(dataset_root) / complex_name / Path(protein_filename).name
        if p2b.is_file():
            return str(p2b)
    p3 = Path.cwd() / protein_filename
    if p3.is_file():
        return str(p3)
    return None


def resolve_complex_name_from_data(data_obj, data_id_fallback) -> str:
    ligand_filename = getattr(data_obj, "ligand_filename", None)
    if isinstance(ligand_filename, str) and ligand_filename:
        parent = Path(ligand_filename).parent
        if str(parent) not in (".", ""):
            return parent.name
    return str(data_id_fallback)


def build_per_sample_rows(
    pred_pos,
    pred_v,
    complex_name: str,
    protein_pdb,
    args,
    *,
    ligand_rmsd_list=None,
    seq_acc_list=None,
):
    """Reconstruct molecules and compute QED / SA / Vina per generated sample (TargetDiff-style inline metrics)."""
    per_sample_rows = []
    pool = {"succ": 0, "incomp": 0, "bad": 0}
    for i, (p_pos, p_v) in enumerate(zip(pred_pos, pred_v)):
        row = {
            "complex": complex_name,
            "sample_idx": i,
            "tag": "bad",
            "smiles": "",
            "qed": None,
            "sa": None,
            "logp": None,
            "lipinski": None,
            "tpsa": None,
            "hba": None,
            "hbd": None,
            "fsp3": None,
            "rotb": None,
            "vina_score": None,
            "vina_min": None,
            "vina_dock": None,
            "high_affinity": None,
            "reconstruct_error": None,
            "protein_pdb": protein_pdb,
            "ligand_atom_types_pred": np.asarray(p_v).astype(np.int64).tolist(),
            "ligand_coords_pred": np.asarray(p_pos).astype(np.float64).tolist() if args.save_ligand_coords else None,
        }
        if ligand_rmsd_list is not None and seq_acc_list is not None:
            rmsd_i = ligand_rmsd_list[i]
            sa_i = seq_acc_list[i]
            row["ligand_rmsd"] = float(rmsd_i) if np.isfinite(rmsd_i) else float("nan")
            row["seq_acc"] = float(sa_i) if np.isfinite(sa_i) else float("nan")
        try:
            pred_v_tensor = torch.as_tensor(np.asarray(p_v), dtype=torch.long)
            pred_atom_type = trans.get_atomic_number_from_index(pred_v_tensor, mode=args.atom_enc_mode)
            pred_aromatic = trans.is_aromatic_from_index(pred_v_tensor, mode=args.atom_enc_mode)
            mol = reconstruct.reconstruct_from_generated(
                np.asarray(p_pos, dtype=np.float64),
                pred_atom_type,
                pred_aromatic,
                basic_mode=(args.atom_enc_mode == "basic"),
            )
            smiles = Chem.MolToSmiles(mol)
            row["smiles"] = smiles
            if "." in smiles:
                row["tag"] = "incomp"
                pool["incomp"] += 1
            else:
                row["tag"] = "succ"
                pool["succ"] += 1

            chem = scoring_func.get_chem(mol)
            for k in ["qed", "sa", "logp", "lipinski", "tpsa", "hba", "hbd", "fsp3", "rotb"]:
                v = chem.get(k)
                row[k] = float(v) if v is not None else None

            vina_score, vina_min, vina_dock, high_affinity = run_vina_metrics(
                mol,
                protein_pdb,
                debug=args.vina_debug,
                sample_name=f"{complex_name}_{i}",
                high_affinity_threshold=args.high_affinity_threshold,
                dock_exhaustiveness=args.dock_exhaustiveness,
                dock_n_poses=args.dock_n_poses,
            )
            row["vina_score"] = vina_score
            row["vina_min"] = vina_min
            row["vina_dock"] = vina_dock
            row["high_affinity"] = high_affinity
        except Exception as e:
            row["reconstruct_error"] = f"{type(e).__name__}: {e}"
            pool["bad"] += 1
        per_sample_rows.append(row)
    return per_sample_rows, pool


def summarize_complex(complex_name: str, per_sample_rows, pool, ligand_rmsd_list=None, seq_acc_list=None):
    def stat(metric):
        return summarize_valid_values([r.get(metric) for r in per_sample_rows])

    qed_stats = stat("qed")
    sa_stats = stat("sa")
    vina_score_stats = stat("vina_score")
    vina_min_stats = stat("vina_min")
    vina_dock_stats = stat("vina_dock")
    high_aff_stats = stat("high_affinity")

    complex_summary = {
        "complex": complex_name,
        "num_generated": len(per_sample_rows),
        "succ": pool["succ"],
        "incomp": pool["incomp"],
        "bad": pool["bad"],
        "qed_mean": qed_stats["mean"],
        "qed_std": qed_stats["std"],
        "qed_median": qed_stats["median"],
        "num_valid_qed": qed_stats["num_valid"],
        "sa_mean": sa_stats["mean"],
        "sa_std": sa_stats["std"],
        "sa_median": sa_stats["median"],
        "num_valid_sa": sa_stats["num_valid"],
        "vina_score_mean": vina_score_stats["mean"],
        "vina_score_std": vina_score_stats["std"],
        "vina_score_median": vina_score_stats["median"],
        "num_valid_vina_score": vina_score_stats["num_valid"],
        "vina_min_mean": vina_min_stats["mean"],
        "vina_min_std": vina_min_stats["std"],
        "vina_min_median": vina_min_stats["median"],
        "num_valid_vina_min": vina_min_stats["num_valid"],
        "vina_dock_mean": vina_dock_stats["mean"],
        "vina_dock_std": vina_dock_stats["std"],
        "vina_dock_median": vina_dock_stats["median"],
        "num_valid_vina_dock": vina_dock_stats["num_valid"],
        "high_affinity_rate": high_aff_stats["mean"],
        "high_affinity_std": high_aff_stats["std"],
        "high_affinity_median": high_aff_stats["median"],
        "num_valid_high_affinity": high_aff_stats["num_valid"],
    }
    if ligand_rmsd_list is not None and seq_acc_list is not None:
        valid_lr = [x for x in ligand_rmsd_list if np.isfinite(x)]
        valid_seq = [x for x in seq_acc_list if np.isfinite(x)]
        complex_summary["ligand_rmsd_mean"] = float(np.mean(valid_lr)) if valid_lr else None
        complex_summary["ligand_rmsd_std"] = float(np.std(valid_lr)) if valid_lr else None
        complex_summary["seq_acc_mean"] = float(np.mean(valid_seq)) if valid_seq else None
        complex_summary["seq_acc_std"] = float(np.std(valid_seq)) if valid_seq else None
        complex_summary["num_valid_ligand_metrics"] = len(valid_lr)
    return complex_summary


def write_complex_eval_outputs(
    complex_name: str,
    per_sample_rows,
    complex_summary,
    args,
    *,
    source_result_file: str | None = None,
    eval_summary_gt=None,
):
    complex_out_dir = Path(args.output_dir) / complex_name
    complex_out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = complex_out_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for row in per_sample_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics_path = complex_out_dir / "metrics.json"
    metrics_payload = {
        "complex_summary": complex_summary,
        "num_rows": len(per_sample_rows),
    }
    if source_result_file is not None:
        metrics_payload["source_result_file"] = source_result_file
    if eval_summary_gt is not None:
        metrics_payload["eval_summary_gt"] = eval_summary_gt
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(samples_path), str(metrics_path)


def process_one_sefmol_data(
    model,
    sample_config,
    data,
    data_id: int,
    eval_args,
    *,
    sample_mode: str,
    timesteps: int,
    batch_size: int,
    device: str,
):
    """
    Sample ligands for one complex, then immediately compute metrics (same control flow as TargetDiff `process_one_data`).
    """
    from sample import sample_diffusion_ligand

    pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
        model,
        data,
        sample_config.sample.num_samples,
        sample_mode,
        batch_size=batch_size,
        device=device,
        num_steps=timesteps,
        center_pos_mode=sample_config.sample.center_pos_mode,
        sample_num_atoms=sample_config.sample.sample_num_atoms,
    )

    if eval_args.max_samples_per_complex > 0:
        k = eval_args.max_samples_per_complex
        pred_pos = pred_pos[:k]
        pred_v = pred_v[:k]
        pred_pos_traj = pred_pos_traj[:k]
        pred_v_traj = pred_v_traj[:k]
        pred_v0_traj = pred_v0_traj[:k]
        pred_vt_traj = pred_vt_traj[:k]

    complex_name = resolve_complex_name_from_data(data, data_id)
    protein_pdb = resolve_protein_pdb(data, complex_name, eval_args.dataset_root, eval_args.mgd_test_dir)

    ligand_rmsd = seq_acc = eval_summary_gt = None
    if hasattr(data, "ligand_pos") and hasattr(data, "ligand_atom_feature_full"):
        ligand_rmsd, seq_acc, eval_summary_gt = evaluate_ligand_predictions(
            pred_pos, pred_v, data.ligand_pos, data.ligand_atom_feature_full
        )

    per_sample_rows, pool = build_per_sample_rows(
        pred_pos, pred_v, complex_name, protein_pdb, eval_args, ligand_rmsd_list=ligand_rmsd, seq_acc_list=seq_acc
    )
    complex_summary = summarize_complex(complex_name, per_sample_rows, pool, ligand_rmsd, seq_acc)

    result_path = Path(eval_args.result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    result_pt = result_path / f"result_{data_id}.pt"
    torch_payload = {
        "data": data,
        "pred_ligand_pos": pred_pos,
        "pred_ligand_v": pred_v,
        "pred_ligand_pos_traj": pred_pos_traj,
        "pred_ligand_v_traj": pred_v_traj,
        "time": time_list,
        "per_sample_rows": per_sample_rows,
        "complex_summary": complex_summary,
    }
    if ligand_rmsd is not None:
        torch_payload["ligand_rmsd"] = ligand_rmsd
        torch_payload["seq_acc"] = seq_acc
        torch_payload["eval_summary_gt"] = eval_summary_gt
    torch.save(torch_payload, result_pt)

    metrics_sidecar = result_path / f"result_{data_id}_metrics.json"
    sidecar = {"complex_summary": complex_summary, "eval_summary_gt": eval_summary_gt}
    metrics_sidecar.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    samples_jsonl, metrics_json = write_complex_eval_outputs(
        complex_name,
        per_sample_rows,
        complex_summary,
        eval_args,
        source_result_file=str(result_pt),
        eval_summary_gt=eval_summary_gt,
    )

    return {
        "name": complex_name,
        "source_result_file": str(result_pt),
        "summary": complex_summary,
        "samples_jsonl": samples_jsonl,
        "metrics_json": metrics_json,
        "result_metrics_sidecar": str(metrics_sidecar),
    }


def evaluate_result_file(result_file: Path, args):
    payload = torch.load(result_file, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{result_file} does not contain a dict payload")
    if "data" not in payload:
        raise ValueError(f"{result_file} missing key: data")
    if "pred_ligand_pos" not in payload or "pred_ligand_v" not in payload:
        raise ValueError(f"{result_file} missing pred_ligand_pos/pred_ligand_v")

    data = payload["data"]
    pred_pos = list(payload["pred_ligand_pos"])
    pred_v = list(payload["pred_ligand_v"])
    if len(pred_pos) != len(pred_v):
        raise ValueError(f"{result_file} has mismatched pred_ligand_pos/pred_ligand_v lengths")
    if args.max_samples_per_complex > 0:
        pred_pos = pred_pos[: args.max_samples_per_complex]
        pred_v = pred_v[: args.max_samples_per_complex]

    complex_name = resolve_complex_name(result_file, data)
    protein_pdb = resolve_protein_pdb(data, complex_name, args.dataset_root, args.mgd_test_dir)

    ligand_rmsd = seq_acc = None
    if hasattr(data, "ligand_pos") and hasattr(data, "ligand_atom_feature_full"):
        ligand_rmsd, seq_acc, _eval_gt = evaluate_ligand_predictions(pred_pos, pred_v, data.ligand_pos, data.ligand_atom_feature_full)

    per_sample_rows, pool = build_per_sample_rows(
        pred_pos, pred_v, complex_name, protein_pdb, args, ligand_rmsd_list=ligand_rmsd, seq_acc_list=seq_acc
    )
    complex_summary = summarize_complex(complex_name, per_sample_rows, pool, ligand_rmsd, seq_acc)

    samples_path, metrics_path = write_complex_eval_outputs(
        complex_name, per_sample_rows, complex_summary, args, source_result_file=str(result_file)
    )

    return {
        "name": complex_name,
        "source_result_file": str(result_file),
        "summary": complex_summary,
        "samples_jsonl": samples_path,
        "metrics_json": metrics_path,
    }


def save_aggregate_summaries(output_dir: Path, all_results: list, logger) -> None:
    dataset_summary_path = output_dir / "dataset_summary.json"
    dataset_summary_path.write_text(
        json.dumps({"num_items": len(all_results), "items": all_results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    per_complex_summary = []
    for item in all_results:
        if item.get("ok") and item.get("summary") is not None:
            per_complex_summary.append(item["summary"])
        else:
            per_complex_summary.append({"complex": item.get("name", ""), "error": item.get("error", "")})

    per_complex_path = output_dir / "per_complex_summary.json"
    per_complex_path.write_text(json.dumps(per_complex_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    ok_summary = [s for s in per_complex_summary if "succ" in s]
    global_summary = {"complexes_total": len(per_complex_summary), "complexes_ok": len(ok_summary)}
    if ok_summary:
        mean_qed = [s["qed_mean"] for s in ok_summary if s.get("qed_mean") is not None]
        mean_sa = [s["sa_mean"] for s in ok_summary if s.get("sa_mean") is not None]
        mean_vina_score = [s["vina_score_mean"] for s in ok_summary if s.get("vina_score_mean") is not None]
        mean_vina_min = [s["vina_min_mean"] for s in ok_summary if s.get("vina_min_mean") is not None]
        mean_vina_dock = [s["vina_dock_mean"] for s in ok_summary if s.get("vina_dock_mean") is not None]
        mean_high_aff = [s["high_affinity_rate"] for s in ok_summary if s.get("high_affinity_rate") is not None]
        global_summary.update(
            {
                "mean_of_per_complex_qed_mean": float(np.mean(mean_qed)) if mean_qed else None,
                "mean_of_per_complex_sa_mean": float(np.mean(mean_sa)) if mean_sa else None,
                "mean_of_per_complex_vina_score_mean": float(np.mean(mean_vina_score)) if mean_vina_score else None,
                "mean_of_per_complex_vina_min_mean": float(np.mean(mean_vina_min)) if mean_vina_min else None,
                "mean_of_per_complex_vina_dock_mean": float(np.mean(mean_vina_dock)) if mean_vina_dock else None,
                "mean_of_per_complex_high_affinity_rate": float(np.mean(mean_high_aff)) if mean_high_aff else None,
            }
        )
        mean_lr = [s["ligand_rmsd_mean"] for s in ok_summary if s.get("ligand_rmsd_mean") is not None]
        mean_seq = [s["seq_acc_mean"] for s in ok_summary if s.get("seq_acc_mean") is not None]
        if mean_lr:
            global_summary["mean_of_per_complex_ligand_rmsd_mean"] = float(np.mean(mean_lr))
        if mean_seq:
            global_summary["mean_of_per_complex_seq_acc_mean"] = float(np.mean(mean_seq))

    global_summary_path = output_dir / "global_summary.json"
    global_summary_path.write_text(json.dumps(global_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    logger.info(f"Saved dataset summary to {dataset_summary_path}")
    logger.info(f"Saved per-complex summary to {per_complex_path}")
    logger.info(f"Saved global summary to {global_summary_path}")


def run_inline_sampling_pipeline(args, logger) -> None:
    """Sample each complex and compute metrics immediately (TargetDiff `evaluation.py` workflow)."""
    if not args.sampling_config or not args.load_ckpt_path:
        raise ValueError("--inline_sampling requires --sampling_config and --load_ckpt_path")

    from datasets import get_dataset
    from models.sefmol_sfrl import SeFMol, initialize_diffusion_params

    sample_config = misc.load_config(args.sampling_config)
    misc.seed_all(sample_config.sample.seed)
    ckpt = torch.load(args.load_ckpt_path, map_location=args.device, weights_only=False)

    args.atom_enc_mode = ckpt["config"].data.transform.ligand_atom_mode

    default_props = [1, 1, 1, 50, 3.0, 2.0, 0.5, 2]
    props = list(args.condition_properties) if args.condition_properties is not None else default_props

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(ckpt["config"].data.transform.ligand_atom_mode, properties=props)
    transform = Compose(
        [
            protein_featurizer,
            ligand_featurizer,
            trans.FeaturizeLigandBond(),
        ]
    )

    if args.sample_mode == "rigid_sample" and args.timesteps < 1000:
        initialize_diffusion_params(ckpt["model"], ckpt["config"].model, args.timesteps)
    model = SeFMol(
        ckpt["config"].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        n_timesteps=args.timesteps,
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _, subsets = get_dataset(config=ckpt["config"].data, transform=transform)
    _, test_set = subsets["train"], subsets["test"]

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.sampling_config, result_dir / "sample.yml")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for data_id in tqdm(range(args.start_id, args.end_id + 1), desc="SeFMol sample+eval"):
        try:
            data = test_set[data_id]
            out = process_one_sefmol_data(
                model,
                sample_config,
                data,
                data_id,
                args,
                sample_mode=args.sample_mode,
                timesteps=args.timesteps,
                batch_size=args.batch_size,
                device=args.device,
            )
            all_results.append({"ok": True, **out})
        except Exception as e:
            logger.warning(f"Failed data_id={data_id}: {e}")
            all_results.append({"ok": False, "name": str(data_id), "error": str(e)})

    save_aggregate_summaries(output_dir, all_results, logger)


def main():
    parser = argparse.ArgumentParser(
        description="SeFMol: sample+evaluate inline (TargetDiff-style) or post-process result_*.pt files."
    )
    parser.add_argument(
        "--inline_sampling",
        action="store_true",
        help="Run diffusion sampling and compute metrics per complex in one pass (no separate metric pass).",
    )
    parser.add_argument("--sampling_config", type=str, default=None, help="YAML config for dataset/sample (inline mode).")
    parser.add_argument("--load_ckpt_path", type=str, default=None, help="Checkpoint path (inline mode).")
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=99)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="sefmol_sample",
        choices=["rigid_sample", "sefmol_sample"],
    )
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument(
        "--condition_properties",
        type=float,
        nargs="*",
        default=None,
        help="Optional property conditioning vector; default matches sample.py.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Where result_*.pt are stored (inline: written here; post-hoc: read recursively).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation outputs.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Optional dataset root for resolving relative data.protein_filename.",
    )
    parser.add_argument(
        "--mgd_test_dir",
        type=str,
        default="./data/TernaryDB/MGD_test",
        help="MGD_test root dir. If {mgd_test_dir}/{complex}/protein1.pdb exists, it is used with highest priority.",
    )
    parser.add_argument(
        "--atom_enc_mode",
        type=str,
        default="add_aromatic",
        choices=["basic", "add_aromatic", "full"],
        help="Atom encoding mode to decode pred_ligand_v (overridden from checkpoint in inline mode).",
    )
    parser.add_argument("--max_samples_per_complex", type=int, default=-1)
    parser.add_argument("--no_save_ligand_coords", action="store_true")
    parser.add_argument("--vina_debug", action="store_true")
    parser.add_argument("--high_affinity_threshold", type=float, default=-7.0)
    parser.add_argument("--dock_exhaustiveness", type=int, default=8)
    parser.add_argument("--dock_n_poses", type=int, default=20)
    args = parser.parse_args()
    args.save_ligand_coords = not args.no_save_ligand_coords

    logger = misc.get_logger("sefmol_evaluation")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.inline_sampling:
        run_inline_sampling_pipeline(args, logger)
        return

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        raise ValueError(f"result_dir not found: {result_dir}")

    result_files = sorted(result_dir.rglob("result_*.pt"))
    logger.info(f"Found {len(result_files)} result files in {result_dir}")
    if len(result_files) == 0:
        raise ValueError(f"No result_*.pt found under {result_dir}")

    all_results = []
    for result_file in tqdm(result_files, desc="Evaluating SeFMol results"):
        try:
            out = evaluate_result_file(result_file, args)
            all_results.append({"ok": True, **out})
        except Exception as e:
            logger.warning(f"Failed on {result_file}: {e}")
            all_results.append({"ok": False, "name": result_file.stem, "error": str(e)})

    save_aggregate_summaries(output_dir, all_results, logger)


if __name__ == "__main__":
    main()
