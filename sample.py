#!/usr/bin/env python3
"""
End-to-end sampling pipeline:
1) sample multiple trajectories from flow model
2) choose protein2 pose by best DockQ
3) choose ligand trajectory by lowest RMSD, optionally try top-k for SeFMol refine
4) save final predicted complex pdb
"""

import argparse
import json
import os
import pickle
import shutil
import tempfile
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from biopandas.pdb import PandasPdb
from datasets import load_from_disk
from rdkit import Chem
from rdkit.Geometry import Point3D

from configs.config import DictToObject
from DeepTernary.DockQ.dockq_util import cal_dockq
from eval_ligand import (
    SeFMolRefiner,
    build_molecule,
    build_processed_protein1_pdb,
    prepare_ligand_arrays,
)
from flow_model import TernaryFlowModel
from utils.constants import AA, max_num_heavyatoms, restype_to_heavyatom_names
from utils.rigid_utils import rotate_and_translate
from utils.training_utils import collate_fn, compute_rmsd, load_config_from_yaml, merge_pdbs


def _atom_xyz_from_gt_matching_template(
    template_atom_df,
    gt_atom_df,
) -> np.ndarray:
    """Pick xyz from gt_complex ATOM rows matching protein2 template atom ordering."""
    gt = gt_atom_df
    if "alt_loc" in gt.columns:
        al = gt["alt_loc"].fillna(" ").astype(str).str.strip()
        gt = gt.loc[al.isin(["", "A"])].copy()

    xyz_out = np.zeros((len(template_atom_df), 3), dtype=np.float32)
    ch_gt_raw = gt["chain_id"].astype(str).str.strip()
    res_gt_raw = gt["residue_number"].astype(int)
    atm_gt_raw = gt["atom_name"].astype(str).str.strip()
    resnm_gt_raw = gt["residue_name"].astype(str).str.strip()

    ins_gt = None
    if "insertion" in gt.columns:
        ins_gt = gt["insertion"].fillna(" ").astype(str).str.strip()

    tpl_has_ins = "insertion" in template_atom_df.columns

    for i, (_, row_t) in enumerate(template_atom_df.iterrows()):
        ch = str(row_t["chain_id"]).strip()
        resi = int(row_t["residue_number"])
        atm = str(row_t["atom_name"]).strip()
        resnm = str(row_t["residue_name"]).strip()

        mask = (ch_gt_raw == ch) & (res_gt_raw == resi) & (atm_gt_raw == atm) & (resnm_gt_raw == resnm)
        if ins_gt is not None and tpl_has_ins:
            v = row_t["insertion"]
            try:
                vnan = isinstance(v, float) and np.isnan(v)
            except (TypeError, ValueError):
                vnan = False
            if v is None or vnan:
                ins_t = " "
            else:
                ins_t = str(v).strip() or " "
            mask = mask & (ins_gt == ins_t)

        sub = gt.loc[mask]
        if len(sub) == 0:
            mask_loose = (ch_gt_raw == ch) & (res_gt_raw == resi) & (atm_gt_raw == atm)
            sub = gt.loc[mask_loose]
        if len(sub) != 1:
            raise ValueError(
                f"GT atom lookup failed for chain={ch!r} resi={resi} atom={atm!r} "
                f"(matches={len(sub)})."
            )
        xyz_out[i] = sub.iloc[0][["x_coord", "y_coord", "z_coord"]].values.astype(np.float32)
    return xyz_out


def _icode_to_str(v) -> str:
    if v is None:
        return " "
    try:
        if isinstance(v, float) and np.isnan(v):
            return " "
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s if s else " "


def _heavy_atom_lookup_from_item(protein_item: Dict, pos_l_a_3: np.ndarray) -> Dict[Tuple[str, int, str, str], np.ndarray]:
    """(chain_id, residue_number, insertion, atom_name) -> xyz from dataset heavy-atom tensor."""
    aa = np.asarray(protein_item["aa"])
    mask = np.asarray(protein_item["mask_heavyatom"], dtype=bool)
    pos_l_a_3 = np.asarray(pos_l_a_3, dtype=np.float32)
    chain_id = protein_item["chain_id"]
    resseq = protein_item["resseq"]
    icode = protein_item["icode"]
    lookup: Dict[Tuple[str, int, str, str], np.ndarray] = {}
    for ri in range(len(aa)):
        restype = AA(int(aa[ri]))
        names = restype_to_heavyatom_names[restype]
        ch = str(chain_id[ri]).strip()
        resi = int(resseq[ri])
        ins = _icode_to_str(icode[ri])
        for ai in range(max_num_heavyatoms):
            if not mask[ri, ai]:
                continue
            an = str(names[ai]).strip()
            if not an:
                continue
            lookup[(ch, resi, ins, an)] = pos_l_a_3[ri, ai].copy()
    return lookup


def _assign_protein_coords_to_atom_df(atom_df, protein_item: Dict, pos_l_a_3: np.ndarray) -> None:
    """
    Update ATOM row coordinates from ``pos_l_a_3`` [L, A, 3] by matching chain / resseq / insertion / atom_name.
    PDB rows without a match (e.g. hydrogens, alt conformers not in the tensor) keep their existing xyz.
    """
    lookup = _heavy_atom_lookup_from_item(protein_item, pos_l_a_3)
    tpl_has_ins = "insertion" in atom_df.columns
    n = len(atom_df)
    out = atom_df[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32).copy()
    for i in range(n):
        row_t = atom_df.iloc[i]
        ch = str(row_t["chain_id"]).strip()
        resi = int(row_t["residue_number"])
        atm = str(row_t["atom_name"]).strip()
        if tpl_has_ins:
            ins_t = _icode_to_str(row_t["insertion"])
        else:
            ins_t = " "
        key = (ch, resi, ins_t, atm)
        if key in lookup:
            out[i] = lookup[key]
        else:
            bk = (ch, resi, " ", atm)
            if bk in lookup:
                out[i] = lookup[bk]
    atom_df.loc[:, ["x_coord", "y_coord", "z_coord"]] = out


def save_compare_visual_complex(
    item: Dict,
    traj: Dict,
    ligand_mol: Chem.Mol,
    pdb_base_dir: Path,
    gt_complex_path: Path,
    out_dir: Path,
) -> Path:
    """
    Like ``save_final_complex``, but protein2 xyz come from ``gt_complex.pdb`` (experimental)
    matched to ``protein2.pdb`` atom ordering, then the sampled rigid transform is applied.

    Used only for pred-vs-GT figures so cartoon geometry matches crystal protein2 while pose
    reflects model rotation/translation.
    """
    name = item["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    p1_path = pdb_base_dir / name / "protein1.pdb"
    p2_path = pdb_base_dir / name / "protein2.pdb"

    p1_ppdb = PandasPdb().read_pdb(str(p1_path))
    p2_ppdb = PandasPdb().read_pdb(str(p2_path))
    gt_ppdb = PandasPdb().read_pdb(str(gt_complex_path))

    p1_coords = np.asarray(item["p1"]["pos_heavyatom"], dtype=np.float32)
    _assign_protein_coords_to_atom_df(p1_ppdb.df["ATOM"], item["p1"], p1_coords)

    xyz_gt_p2 = _atom_xyz_from_gt_matching_template(p2_ppdb.df["ATOM"], gt_ppdb.df["ATOM"])
    rot = traj["rot_pred"].detach().cpu().float()
    trans = traj["trans_pred"].detach().cpu().float().unsqueeze(0)
    p2_flat = torch.as_tensor(xyz_gt_p2.reshape(-1, 3), dtype=torch.float32)
    p2_vis = rotate_and_translate(p2_flat, rot, trans).numpy()

    p2_ppdb.df["ATOM"][["x_coord", "y_coord", "z_coord"]] = p2_vis

    p1_out = out_dir / f"{name}_protein1_pred_compareviz.pdb"
    p2_out = out_dir / f"{name}_protein2_pred_compareviz.pdb"
    lig_out = out_dir / f"{name}_ligand_pred_compareviz.pdb"
    complex_out = out_dir / f"{name}_complex_pred_compareviz.pdb"

    p1_ppdb.to_pdb(str(p1_out), records=["ATOM"], gz=False)
    p2_ppdb.to_pdb(str(p2_out), records=["ATOM"], gz=False)
    Chem.MolToPDBFile(ligand_mol, str(lig_out))
    merge_pdbs([str(p1_out), str(p2_out), str(lig_out)], str(complex_out))
    return complex_out


def _read_names_from_file(path: str) -> List[str]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"--names_file not found: {path}")
    names: List[str] = []
    text = p.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line)
    return names


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end ternary complex sampling.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--pdb_base_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="sample_outputs")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=None, help="Number of dataset items to sample.")
    parser.add_argument("--num_trajectories", type=int, default=8, help="Sampled trajectories per item.")
    parser.add_argument("--trajectory_batch_size", type=int, default=None, help="Micro-batch for model.sample().")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Sample only this complex (mutually exclusive with --sample_names / --names_file).",
    )
    parser.add_argument(
        "--names_file",
        type=str,
        default=None,
        help="Batch mode: text file with one complex name per line (# comments allowed).",
    )
    parser.add_argument("--sample_names", nargs="+", default=None, help="Optional subset of complex names.")
    parser.add_argument(
        "--render_compare",
        action="store_true",
        help="After each successful sample, render prediction vs gt_complex side-by-side (requires PyMOL + Pillow).",
    )
    parser.add_argument(
        "--compare_save_split_pngs",
        action="store_true",
        help="With --render_compare, also save separate pred-only and GT-only panel PNGs.",
    )
    parser.add_argument("--ligand_candidate_topk", type=int, default=3, help="Try top-k lowest RMSD ligands for SeFMol.")

    parser.add_argument("--sefmol_refine", action="store_true")
    parser.add_argument("--sefmol_ckpt", type=str, default=None)
    parser.add_argument("--sefmol_device", type=str, default="cuda:0")
    parser.add_argument("--sefmol_timesteps", type=int, default=50)
    parser.add_argument("--sefmol_sample_mode", type=str, default="sefmol_sample", choices=["rigid_sample", "sefmol_sample"])
    parser.add_argument("--sefmol_center_pos_mode", type=str, default="protein", choices=["none", "protein", "value_func_protein"])
    parser.add_argument("--sefmol_condition_properties", nargs="+", type=float, default=None)
    return parser.parse_args()


def load_model(args) -> Tuple[TernaryFlowModel, torch.device]:
    config = load_config_from_yaml(args.config)
    model_cfg = DictToObject(
        {
            "node_embed_size": config.model.encoder.node_embed_size,
            "edge_embed_size": config.model.encoder.edge_embed_size,
            "ipa": config.model.encoder.ipa,
            "interface_model": DictToObject(
                {
                    "path": getattr(config.model.interface_model, "path", None),
                    "trainable": getattr(config.model.interface_model, "trainable", False),
                    "feat_dim": config.model.interface_model.feat_dim,
                    "depth": getattr(config.model.interface_model, "depth", 4),
                    "num_nearest_neighbors": getattr(config.model.interface_model, "num_nearest_neighbors", 16),
                    "topk_k": getattr(config.model.interface_model, "topk_k", 50),
                }
            ),
        }
    )
    if hasattr(config.model.interpolant, "sampling"):
        sampling_cfg = config.model.interpolant.sampling
        if hasattr(sampling_cfg, "num_timesteps"):
            sampling_cfg = DictToObject({"num_steps": sampling_cfg.num_timesteps})
        else:
            sampling_cfg = DictToObject({"num_steps": getattr(sampling_cfg, "num_steps", 100)})
    else:
        sampling_cfg = DictToObject({"num_steps": 100})
    full_cfg = DictToObject({"model": model_cfg, "interpolant": config.model.interpolant, "sampling": sampling_cfg})

    device = torch.device(args.device)
    model = TernaryFlowModel(full_cfg).to(device)
    try:
        ckpt = torch.load(args.checkpoint, map_location=device)
    except pickle.UnpicklingError as e:
        # PyTorch 2.6 defaults torch.load(..., weights_only=True), which can fail
        # for older checkpoints that contain non-tensor python objects.
        print(
            f"Warning: weights_only checkpoint load failed ({e}). "
            "Retrying with weights_only=False for trusted checkpoint..."
        )
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, device


def _move_batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
        elif isinstance(batch[k], dict):
            for sk in batch[k]:
                if isinstance(batch[k][sk], torch.Tensor):
                    batch[k][sk] = batch[k][sk].to(device)


def sample_trajectories(model, item: Dict, device: torch.device, args) -> List[Dict]:
    n = int(args.num_trajectories)
    chunk = n if args.trajectory_batch_size is None else max(1, min(int(args.trajectory_batch_size), n))
    name = item.get("name", "sample")
    seed_off = zlib.crc32(str(name).encode()) & 0x7FFFFFFF

    rows = []
    done = 0
    while done < n:
        bsz = min(chunk, n - done)
        cur_seed = args.seed + seed_off + done * 100003

        while True:
            batch = collate_fn([item] * bsz)
            _move_batch_to_device(batch, device)
            torch.manual_seed(cur_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(cur_seed)
            try:
                with torch.no_grad():
                    out = model.sample(batch)[-1]
                break
            except RuntimeError as e:
                msg = str(e).lower()
                is_oom = "out of memory" in msg or "cuda error: out of memory" in msg
                if not is_oom:
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if bsz == 1:
                    raise RuntimeError(
                        "CUDA OOM even with trajectory_batch_size=1. "
                        "Try reducing --num_trajectories, disabling SeFMol refine, "
                        "or using a larger GPU."
                    ) from e
                new_bsz = max(1, bsz // 2)
                print(f"[{name}] OOM at micro-batch={bsz}, retrying with {new_bsz}...")
                bsz = new_bsz

        for i in range(bsz):
            lig_coords_pred = out["lig_coords"][i]
            lig_seq_pred = out["lig_seq"][i]
            mol_mask = batch["mol_mask"][i]
            rmsd = compute_rmsd(lig_coords_pred.unsqueeze(0), batch["lig_coords_1"][i].unsqueeze(0), mol_mask.unsqueeze(0)).item()
            rows.append(
                {
                    "traj_idx": done + i,
                    "lig_coords_pred": lig_coords_pred.detach().cpu().numpy(),
                    "lig_seq_pred": lig_seq_pred.detach().cpu().numpy(),
                    "mol_mask": mol_mask.detach().cpu().numpy(),
                    "rot_pred": out["rotmats"][i].detach().cpu(),
                    "trans_pred": out["trans"][i].detach().cpu().squeeze(0),
                    "rmsd": float(rmsd),
                }
            )
        done += bsz
    return rows


def build_complex_and_dockq(item: Dict, traj: Dict, pdb_base_dir: Path, gt_complex_path: Path) -> float:
    name = item["name"]
    p1_path = pdb_base_dir / name / "protein1.pdb"
    p2_path = pdb_base_dir / name / "protein2.pdb"
    lig_path = pdb_base_dir / name / "ligand.pdb"

    p2_coords = np.asarray(item["p2"]["pos_heavyatom"], dtype=np.float32)
    l2, a = p2_coords.shape[:2]
    p2_flat = torch.as_tensor(p2_coords.reshape(l2 * a, 3), dtype=torch.float32)
    p2_restored = rotate_and_translate(p2_flat, traj["rot_pred"], traj["trans_pred"].unsqueeze(0)).reshape(l2, a, 3).numpy()

    p1_coords = np.asarray(item["p1"]["pos_heavyatom"], dtype=np.float32)
    p1_ppdb = PandasPdb().read_pdb(str(p1_path))
    p2_ppdb = PandasPdb().read_pdb(str(p2_path))
    _assign_protein_coords_to_atom_df(p1_ppdb.df["ATOM"], item["p1"], p1_coords)
    _assign_protein_coords_to_atom_df(p2_ppdb.df["ATOM"], item["p2"], p2_restored)

    lig_coords = np.asarray(traj["lig_coords_pred"], dtype=np.float32)
    mol_mask = np.asarray(traj["mol_mask"], dtype=bool)
    lig_valid = lig_coords[mol_mask]
    lig_ppdb = PandasPdb().read_pdb(str(lig_path))
    lig_ppdb.df["HETATM"][["x_coord", "y_coord", "z_coord"]] = lig_valid

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"dockq_{name}_"))
    p1_tmp, p2_tmp, lig_tmp = tmp_dir / "p1.pdb", tmp_dir / "p2.pdb", tmp_dir / "lig.pdb"
    p1_ppdb.to_pdb(str(p1_tmp), records=["ATOM"], gz=False)
    p2_ppdb.to_pdb(str(p2_tmp), records=["ATOM"], gz=False)
    lig_ppdb.to_pdb(str(lig_tmp), records=["HETATM"], gz=False)
    complex_path = tmp_dir / "complex.pdb"
    merge_pdbs([str(p1_tmp), str(p2_tmp), str(lig_tmp)], str(complex_path))

    _fnat, _irms, _Lrms, dockq = cal_dockq(str(complex_path), str(gt_complex_path))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return float(dockq)


def refine_ligand(item: Dict, candidate_trajs: List[Dict], sefmol_refiner: Optional[SeFMolRefiner], pdb_base_dir: Path) -> Tuple[Optional[Chem.Mol], Optional[int], Optional[str]]:
    def translate_mol_to_target_centroid(mol: Chem.Mol, target_xyz: np.ndarray) -> Chem.Mol:
        """Apply translation-only rigid transform so mol centroid matches target centroid."""
        if mol is None or mol.GetNumConformers() == 0 or target_xyz.size == 0:
            return mol

        conf = mol.GetConformer()
        cur_coords = np.asarray(conf.GetPositions(), dtype=np.float32)
        if cur_coords.size == 0:
            return mol

        target_center = np.asarray(target_xyz, dtype=np.float32).mean(axis=0)
        cur_center = cur_coords.mean(axis=0)
        shift = target_center - cur_center

        for atom_idx in range(mol.GetNumAtoms()):
            x, y, z = cur_coords[atom_idx] + shift
            conf.SetAtomPosition(atom_idx, Point3D(float(x), float(y), float(z)))
        return mol

    name = item["name"]
    for traj in candidate_trajs:
        seq = traj["lig_seq_pred"].tolist()
        coords = traj["lig_coords_pred"].tolist()
        try:
            cls, xyz = prepare_ligand_arrays(seq, coords, drop_hydrogen=False)
            target_xyz = np.asarray(xyz, dtype=np.float32)
            if sefmol_refiner is not None:
                p1_tmp = build_processed_protein1_pdb(name, pdb_base_dir, item, debug=False)
                p1_for_sefmol = p1_tmp if p1_tmp is not None else (pdb_base_dir / name / "protein1.pdb")
                mol, err = sefmol_refiner.refine(cls, xyz, dataset_item=item, protein1_pdb=p1_for_sefmol)
                if p1_tmp is not None:
                    shutil.rmtree(p1_tmp.parent, ignore_errors=True)
                if mol is not None:
                    mol = translate_mol_to_target_centroid(mol, target_xyz)
                    return mol, traj["traj_idx"], None
                if err is not None:
                    continue
            mol, _smiles = build_molecule(cls, xyz)
            return mol, traj["traj_idx"], None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            continue
    return None, None, last_err if "last_err" in locals() else "all_candidates_failed"


def save_final_complex(
    item: Dict,
    best_dockq_traj: Dict,
    ligand_mol: Chem.Mol,
    out_dir: Path,
    pdb_base_dir: Path,
) -> Path:
    name = item["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    p1_path = pdb_base_dir / name / "protein1.pdb"
    p2_path = pdb_base_dir / name / "protein2.pdb"
    p1_ppdb = PandasPdb().read_pdb(str(p1_path))
    p2_ppdb = PandasPdb().read_pdb(str(p2_path))

    p1_coords = np.asarray(item["p1"]["pos_heavyatom"], dtype=np.float32)
    _assign_protein_coords_to_atom_df(p1_ppdb.df["ATOM"], item["p1"], p1_coords)

    p2_coords = np.asarray(item["p2"]["pos_heavyatom"], dtype=np.float32)
    l2, a = p2_coords.shape[:2]
    p2_flat = torch.as_tensor(p2_coords.reshape(l2 * a, 3), dtype=torch.float32)
    p2_restored = rotate_and_translate(
        p2_flat, best_dockq_traj["rot_pred"], best_dockq_traj["trans_pred"].unsqueeze(0)
    ).reshape(l2, a, 3).numpy()
    _assign_protein_coords_to_atom_df(p2_ppdb.df["ATOM"], item["p2"], p2_restored)

    p1_out = out_dir / f"{name}_protein1_pred.pdb"
    p2_out = out_dir / f"{name}_protein2_pred.pdb"
    lig_out = out_dir / f"{name}_ligand_refined.pdb"
    complex_out = out_dir / f"{name}_complex_pred.pdb"
    p1_ppdb.to_pdb(str(p1_out), records=["ATOM"], gz=False)
    p2_ppdb.to_pdb(str(p2_out), records=["ATOM"], gz=False)
    Chem.MolToPDBFile(ligand_mol, str(lig_out))
    merge_pdbs([str(p1_out), str(p2_out), str(lig_out)], str(complex_out))
    return complex_out


def main():
    args = parse_args()
    if args.name is not None and args.sample_names:
        raise ValueError("Use either --name or --sample_names, not both.")
    if args.name is not None and args.names_file:
        raise ValueError("Use either --name or --names_file, not both.")
    if args.names_file and args.sample_names:
        raise ValueError("Use either --names_file or --sample_names, not both.")
    if args.sefmol_refine and not args.sefmol_ckpt:
        raise ValueError("--sefmol_ckpt is required when --sefmol_refine is set.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, device = load_model(args)
    dataset = load_from_disk(args.dataset_path)
    dataset_by_name = {item["name"]: item for item in dataset}
    if args.name is not None:
        if args.name not in dataset_by_name:
            raise ValueError(
                f"Complex {args.name!r} not found in dataset {args.dataset_path}. "
                "Check spelling or regenerate the dataset slice."
            )
        names = [args.name]
    elif args.names_file:
        raw = _read_names_from_file(args.names_file)
        names = [n for n in raw if n in dataset_by_name]
        missing = set(raw) - set(names)
        if missing:
            print(f"Warning: names not in dataset skipped ({len(missing)}): {sorted(missing)[:20]}{'...' if len(missing) > 20 else ''}")
        if not names:
            raise ValueError(f"No valid complex names from --names_file after filtering by dataset.")
        if args.num_samples is not None:
            names = names[: args.num_samples]
    else:
        names = list(dataset_by_name.keys())
        if args.sample_names:
            names = [n for n in args.sample_names if n in dataset_by_name]
            missing = set(args.sample_names) - set(names)
            if missing:
                print(f"Warning: unknown names skipped: {sorted(missing)}")
        if args.num_samples is not None:
            names = names[: args.num_samples]

    sefmol_refiner = None
    if args.sefmol_refine:
        sefmol_refiner = SeFMolRefiner(
            ckpt_path=args.sefmol_ckpt,
            device=args.sefmol_device,
            timesteps=args.sefmol_timesteps,
            sample_mode=args.sefmol_sample_mode,
            center_pos_mode=args.sefmol_center_pos_mode,
            condition_properties=args.sefmol_condition_properties,
        )

    pdb_base_dir = Path(args.pdb_base_dir)
    results = []
    out_dir = Path(args.output_dir)
    for i, name in enumerate(names):
        compare_png = out_dir / f"{name}_pred_vs_gt.png"
        if compare_png.is_file():
            print(f"[{i+1}/{len(names)}] {name} skipped (compare image exists): {compare_png}")
            continue
        item = dataset_by_name[name]
        trajs = sample_trajectories(model, item, device, args)
        gt_complex = pdb_base_dir / name / "gt_complex.pdb"
        dockq_rows = []
        for traj in trajs:
            try:
                dockq = build_complex_and_dockq(item, traj, pdb_base_dir, gt_complex)
            except Exception:
                dockq = -1.0
            dockq_rows.append((dockq, traj))
        dockq_rows.sort(key=lambda x: x[0], reverse=True)
        best_dockq, best_dockq_traj = dockq_rows[0]

        trajs_by_rmsd = sorted(trajs, key=lambda x: x["rmsd"])
        topk = max(1, min(args.ligand_candidate_topk, len(trajs_by_rmsd)))
        ligand_mol, chosen_lig_traj_idx, lig_err = refine_ligand(
            item,
            trajs_by_rmsd[:topk],
            sefmol_refiner,
            pdb_base_dir,
        )
        if ligand_mol is None:
            print(f"[{name}] ligand refine/reconstruct failed: {lig_err}")
            continue

        out_complex = save_final_complex(item, best_dockq_traj, ligand_mol, Path(args.output_dir), pdb_base_dir)
        row = {
            "name": name,
            "best_dockq": float(best_dockq),
            "best_dockq_traj_idx": int(best_dockq_traj["traj_idx"]),
            "best_ligand_rmsd": float(trajs_by_rmsd[0]["rmsd"]),
            "chosen_ligand_traj_idx": int(chosen_lig_traj_idx) if chosen_lig_traj_idx is not None else None,
            "output_complex_pdb": str(out_complex),
            "compare_png": None,
            "compare_viz_pdb": None,
        }
        if args.render_compare:
            gt_pdb = pdb_base_dir / name / "gt_complex.pdb"
            if not gt_pdb.is_file():
                print(f"[{name}] skipped compare render: missing {gt_pdb}")
            else:
                try:
                    from visualization import render_prediction_vs_gt
                except ImportError as exc:
                    print(f"[{name}] compare render skipped (import failed): {exc}")
                else:
                    cmp_out = Path(args.output_dir) / f"{name}_pred_vs_gt.png"
                    try:
                        viz_complex = save_compare_visual_complex(
                            item,
                            best_dockq_traj,
                            ligand_mol,
                            pdb_base_dir,
                            gt_pdb,
                            Path(args.output_dir),
                        )
                        pred_for_compare = str(viz_complex)
                        row["compare_viz_pdb"] = str(viz_complex.resolve())
                    except Exception as exc:
                        print(
                            f"[{name}] compare-viz PDB (GT protein2 + rigid) failed ({exc}); "
                            f"falling back to standard pred complex."
                        )
                        pred_for_compare = str(out_complex)
                    render_prediction_vs_gt(
                        pred_for_compare,
                        str(gt_pdb),
                        str(cmp_out),
                        save_individual_pngs=args.compare_save_split_pngs,
                    )
                    row["compare_png"] = str(cmp_out.resolve())
        results.append(row)
        print(f"[{i+1}/{len(names)}] {name} dockq={row['best_dockq']:.4f} -> {out_complex}")

    out_summary = Path(args.output_dir) / "sample_results.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary, "w") as f:
        json.dump({"num_samples": len(results), "results": results}, f, indent=2)
    print(f"Saved summary to {out_summary}")


if __name__ == "__main__":
    main()
