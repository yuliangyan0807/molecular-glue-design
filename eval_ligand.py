import argparse
import contextlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit.Chem.QED import qed as qed_fn
from rdkit import Chem
from biopandas.pdb import PandasPdb
from datasets import load_from_disk

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from utils.constants import MAP_ATOM_TYPE_FULL_TO_INDEX
from targetdiff.utils.reconstruct import reconstruct_from_generated as td_reconstruct_from_generated

_IDX_TO_DESC = {idx: desc for desc, idx in MAP_ATOM_TYPE_FULL_TO_INDEX.items()}
_IDX_TO_Z = {idx: desc[0] for idx, desc in _IDX_TO_DESC.items()}

_SA_SCORER = None
_SEFMOL_REFINER = None


def _get_vina_tmp_root() -> Optional[Path]:
    """
    Optional root for Vina temp files.
    - If VINA_TMP_DIR is set, temp dirs are created there.
    - Otherwise use system temp dir (no persistent .vina_tmp in repo).
    """
    tmp_env = os.environ.get("VINA_TMP_DIR")
    if not tmp_env:
        return None
    p = Path(tmp_env).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _mk_tmpdir(prefix: str) -> Path:
    root = _get_vina_tmp_root()
    if root is None:
        return Path(tempfile.mkdtemp(prefix=prefix))
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(root)))


def _compute_sa_score_normalized(rdmol: Chem.Mol) -> float:
    """Normalized SA score (higher = easier to synthesize); same as targetdiff compute_sa_score."""
    global _SA_SCORER
    if _SA_SCORER is None:
        path = _REPO / "targetdiff/utils/evaluation/sascorer.py"
        spec = importlib.util.spec_from_file_location("mgd_td_sascorer", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _SA_SCORER = mod.compute_sa_score
    return float(_SA_SCORER(rdmol))


def compute_qed_sa(rdmol: Optional[Chem.Mol]) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute QED and normalized SA for a reconstructed RDKit mol; returns (None, None) on failure.
    QED in (0,1), higher is more drug-like; SA higher is easier to synthesize (targetdiff convention).
    """
    if rdmol is None:
        return None, None
    try:
        q = float(qed_fn(rdmol))
    except Exception:
        q = None
    try:
        s = _compute_sa_score_normalized(rdmol)
    except Exception:
        s = None
    return q, s


def run_vina_score_only(
    rdmol: Optional[Chem.Mol],
    protein1_pdb: Path,
    debug: bool = False,
    sample_name: str = "",
) -> Optional[float]:
    """
    Same idea as SeFMol VinaDockingTask + score_only: ligand RDKit mol + protein1.pdb -> Vina affinity (kcal/mol, lower is better).
    Needs: vina, meeko, openbabel, AutoDockTools, pdb2pqr30. All temp files under one tmp dir (does not write next to your PDB).
    """
    prefix = f"[Vina:{sample_name}] " if sample_name else "[Vina] "
    if rdmol is None:
        if debug:
            print(f"{prefix}skip: reconstructed ligand is None")
        return None
    if not protein1_pdb.is_file():
        if debug:
            print(f"{prefix}skip: protein1.pdb not found at {protein1_pdb}")
        return None
    try:
        from meeko import MoleculePreparation
        from openbabel import pybel
        from vina import Vina
        import AutoDockTools
    except ImportError as e:
        if debug:
            print(f"{prefix}import error: {e}")
        return None

    work = _mk_tmpdir(prefix="vina_")
    try:
        rec = work / "rec.pdb"
        shutil.copy(protein1_pdb, rec)
        mol_h = Chem.AddHs(Chem.Mol(rdmol), addCoords=True)
        lig_sdf = work / "lig.sdf"
        w = Chem.SDWriter(str(lig_sdf))
        w.write(mol_h)
        w.close()

        lig_pdbqt = work / "lig.pdbqt"
        ob_mol = next(pybel.readfile("sdf", str(lig_sdf)))
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                mp = MoleculePreparation()
                mp.prepare(ob_mol.OBMol)
                mp.write_pdbqt_file(str(lig_pdbqt))

        rec_pqr = work / "rec.pqr"
        rec_pdbqt = work / "rec.pdbqt"
        pqr_proc = subprocess.run(
            ["pdb2pqr30", "--ff=AMBER", str(rec), str(rec_pqr)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        prep_rec = os.path.join(
            AutoDockTools.__path__[0], "Utilities24", "prepare_receptor4.py"
        )
        prep_proc = subprocess.run(
            ["python3", prep_rec, "-r", str(rec_pqr), "-o", str(rec_pdbqt)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if debug and (pqr_proc.returncode != 0 or prep_proc.returncode != 0):
            print(
                f"{prefix}prep warning: pdb2pqr30 rc={pqr_proc.returncode}, "
                f"prepare_receptor4 rc={prep_proc.returncode}"
            )
        if not lig_pdbqt.is_file() or not rec_pdbqt.is_file():
            if debug:
                print(
                    f"{prefix}skip: missing pdbqt files "
                    f"(lig={lig_pdbqt.is_file()}, rec={rec_pdbqt.is_file()})"
                )
            return None

        pos = mol_h.GetConformer(0).GetPositions()
        c = (pos.max(axis=0) + pos.min(axis=0)) / 2.0
        box = (pos.max(axis=0) - pos.min(axis=0)) + 5.0

        v = Vina(sf_name="vina", seed=0, verbosity=0)
        v.set_receptor(str(rec_pdbqt))
        v.set_ligand_from_file(str(lig_pdbqt))
        v.compute_vina_maps(center=c.tolist(), box_size=box.tolist())
        return float(v.score()[0])
    except Exception as e:
        if debug:
            print(f"{prefix}runtime error: {type(e).__name__}: {e}")
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_vina_metrics(
    rdmol: Optional[Chem.Mol],
    protein1_pdb: Path,
    debug: bool = False,
    sample_name: str = "",
    high_affinity_threshold: float = -7.0,
    dock_exhaustiveness: int = 8,
    dock_n_poses: int = 20,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[int]]:
    """
    Return:
      - vina_score: score_only at current pose
      - vina_min: after local optimize()
      - vina_dock: best score after dock()
      - high_affinity: 1 if vina_dock <= threshold else 0, None if vina_dock unavailable
    """
    prefix = f"[Vina:{sample_name}] " if sample_name else "[Vina] "
    if rdmol is None:
        if debug:
            print(f"{prefix}skip: reconstructed ligand is None")
        return None, None, None, None
    if not protein1_pdb.is_file():
        if debug:
            print(f"{prefix}skip: protein1.pdb not found at {protein1_pdb}")
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

    work = _mk_tmpdir(prefix="vina_")
    try:
        rec = work / "rec.pdb"
        shutil.copy(protein1_pdb, rec)
        mol_h = Chem.AddHs(Chem.Mol(rdmol), addCoords=True)
        lig_sdf = work / "lig.sdf"
        w = Chem.SDWriter(str(lig_sdf))
        w.write(mol_h)
        w.close()

        lig_pdbqt = work / "lig.pdbqt"
        ob_mol = next(pybel.readfile("sdf", str(lig_sdf)))
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                mp = MoleculePreparation()
                mp.prepare(ob_mol.OBMol)
                mp.write_pdbqt_file(str(lig_pdbqt))

        rec_pqr = work / "rec.pqr"
        rec_pdbqt = work / "rec.pdbqt"
        pqr_proc = subprocess.run(
            ["pdb2pqr30", "--ff=AMBER", str(rec), str(rec_pqr)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        prep_rec = os.path.join(
            AutoDockTools.__path__[0], "Utilities24", "prepare_receptor4.py"
        )
        prep_proc = subprocess.run(
            ["python3", prep_rec, "-r", str(rec_pqr), "-o", str(rec_pdbqt)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if debug and (pqr_proc.returncode != 0 or prep_proc.returncode != 0):
            print(
                f"{prefix}prep warning: pdb2pqr30 rc={pqr_proc.returncode}, "
                f"prepare_receptor4 rc={prep_proc.returncode}"
            )
        if not lig_pdbqt.is_file() or not rec_pdbqt.is_file():
            if debug:
                print(
                    f"{prefix}skip: missing pdbqt files "
                    f"(lig={lig_pdbqt.is_file()}, rec={rec_pdbqt.is_file()})"
                )
            return None, None, None, None

        pos = mol_h.GetConformer(0).GetPositions()
        c = (pos.max(axis=0) + pos.min(axis=0)) / 2.0
        box = (pos.max(axis=0) - pos.min(axis=0)) + 5.0

        v = Vina(sf_name="vina", seed=0, verbosity=0)
        v.set_receptor(str(rec_pdbqt))
        v.set_ligand_from_file(str(lig_pdbqt))
        v.compute_vina_maps(center=c.tolist(), box_size=box.tolist())

        vina_score = float(v.score()[0])

        vina_min = None
        try:
            vina_min = float(v.optimize()[0])
        except Exception as e:
            if debug:
                print(f"{prefix}optimize failed: {type(e).__name__}: {e}")

        vina_dock = None
        try:
            v.dock(exhaustiveness=int(dock_exhaustiveness), n_poses=int(dock_n_poses))
            pose_scores = v.energies(n_poses=1)
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


def decode_full_indices(type_indices: List[int]) -> Tuple[List[int], List[bool]]:
    atomic_nums: List[int] = []
    aromatic_flags: List[bool] = []
    for idx in type_indices:
        key = int(idx)
        if key not in _IDX_TO_DESC:
            raise ValueError(f"Unknown atom type index: {idx}")
        z, _hyb, aromatic = _IDX_TO_DESC[key]
        atomic_nums.append(int(z))
        aromatic_flags.append(bool(aromatic))
    return atomic_nums, aromatic_flags


def build_processed_protein1_pdb(
    sample_name: str,
    pdb_base_dir: Path,
    dataset_item: dict,
    debug: bool = False,
) -> Optional[Path]:
    """
    Build temporary protein1.pdb with processed p1 coordinates from dataset item,
    matching evaluation.py behavior (replace template ATOM coords with processed coords).
    """
    prefix = f"[Vina:{sample_name}] "
    p1_template = pdb_base_dir / sample_name / "protein1.pdb"
    if not p1_template.is_file():
        if debug:
            print(f"{prefix}processed p1 skip: template protein1.pdb not found at {p1_template}")
        return None
    if "p1" not in dataset_item:
        if debug:
            print(f"{prefix}processed p1 skip: dataset item has no 'p1'")
        return None

    p1_coords = np.asarray(dataset_item["p1"]["pos_heavyatom"], dtype=np.float64)  # (L, A, 3)
    p1_mask = np.asarray(dataset_item["p1"]["mask_heavyatom"], dtype=bool)          # (L, A)
    if p1_coords.ndim != 3 or p1_coords.shape[-1] != 3 or p1_mask.shape != p1_coords.shape[:2]:
        if debug:
            print(f"{prefix}processed p1 skip: invalid p1 shapes coords={p1_coords.shape}, mask={p1_mask.shape}")
        return None

    l, a = p1_coords.shape[:2]
    valid_coords = p1_coords.reshape(l * a, 3)[p1_mask.reshape(l * a)]

    ppdb = PandasPdb().read_pdb(str(p1_template))
    if "ATOM" not in ppdb.df or ppdb.df["ATOM"].empty:
        if debug:
            print(f"{prefix}processed p1 skip: no ATOM records in template")
        return None

    atom_df = ppdb.df["ATOM"].copy()
    if len(atom_df) != len(valid_coords):
        if debug:
            print(
                f"{prefix}processed p1 skip: template ATOM count ({len(atom_df)}) "
                f"!= processed coords ({len(valid_coords)})"
            )
        return None

    atom_df.loc[:, ["x_coord", "y_coord", "z_coord"]] = valid_coords
    ppdb.df["ATOM"] = atom_df

    tmp_dir = _mk_tmpdir(prefix=f"p1proc_{sample_name}_")
    out_pdb = tmp_dir / "protein1_processed.pdb"
    ppdb.to_pdb(str(out_pdb), records=["ATOM"], gz=False)
    return out_pdb


def prepare_ligand_arrays(
    lig_seq_pred: List,
    lig_coords_pred: List,
    mol_mask: Optional[List] = None,
    drop_hydrogen: bool = True,
) -> Tuple[List[int], List[List[float]]]:
    """
    Remove padding using seq rule: first index with token 0 is padding start.
    Then align coords to the effective seq length; optionally drop hydrogens.
    Returns (class_indices, coords) with matching lengths (heavy atoms if drop_hydrogen).
    """
    cls_raw = np.asarray(lig_seq_pred, dtype=np.int64)
    xyz_raw = np.asarray(lig_coords_pred, dtype=np.float64)

    # In current data format, seq token 0 marks padding start.
    pad_pos = np.where(cls_raw == 0)[0]
    eff_len = int(pad_pos[0]) if len(pad_pos) > 0 else int(len(cls_raw))

    cls = cls_raw[:eff_len]
    if len(xyz_raw) < eff_len:
        raise ValueError(
            "coords shorter than non-padding seq length: "
            f"len(seq_eff)={eff_len}, len(coords)={len(xyz_raw)}"
        )
    xyz = xyz_raw[:eff_len]

    if drop_hydrogen:
        zv = np.array([_IDX_TO_Z[int(c)] for c in cls.tolist()], dtype=np.int64)
        keep = zv != 1
        cls = cls[keep]
        xyz = xyz[keep]

    if len(cls) == 0:
        raise ValueError("no atoms left after mol_mask / drop_hydrogen")

    return cls.tolist(), xyz.tolist()


def build_molecule(
    type_indices: List[int],
    coords: List[List[float]],
) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """
    TargetDiff-style reconstruction from atom type indices + coordinates.
    """
    if len(type_indices) != len(coords):
        raise ValueError("type_indices and coords must have the same length")
    atomic_nums, aromatic_flags = decode_full_indices(type_indices)
    rdmol = td_reconstruct_from_generated(
        coords,
        atomic_nums,
        aromatic=aromatic_flags,
        basic_mode=False,
    )

    smiles = Chem.MolToSmiles(rdmol, canonical=True, isomericSmiles=True)
    return rdmol, smiles


def _cfg_get(cfg, path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if isinstance(cur, dict):
            if p not in cur:
                return default
            cur = cur[p]
        else:
            if not hasattr(cur, p):
                return default
            cur = getattr(cur, p)
    return cur


def _build_sefmol_protein_features(protein_dict: dict) -> np.ndarray:
    atomic_numbers = np.array([1, 6, 7, 8, 16, 34], dtype=np.int64)
    element = np.asarray(protein_dict["element"], dtype=np.int64)
    aa = np.asarray(protein_dict["atom_to_aa_type"], dtype=np.int64)
    is_backbone = np.asarray(protein_dict["is_backbone"], dtype=np.int64)
    elem_oh = (element[:, None] == atomic_numbers[None, :]).astype(np.int64)
    aa = np.clip(aa, 0, 19)
    aa_oh = np.eye(20, dtype=np.int64)[aa]
    bb = is_backbone.reshape(-1, 1).astype(np.int64)
    return np.concatenate([elem_oh, aa_oh, bb], axis=1).astype(np.float32)


class SeFMolRefiner:
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda:0",
        timesteps: int = 50,
        sample_mode: str = "sefmol_sample",
        center_pos_mode: str = "protein",
        condition_properties: Optional[List[float]] = None,
    ):
        global _SEFMOL_REFINER
        if _SEFMOL_REFINER is not None:
            self.__dict__ = _SEFMOL_REFINER.__dict__
            return

        sefmol_root = _REPO / "SeFMol"
        if not sefmol_root.is_dir():
            raise FileNotFoundError(f"SeFMol directory not found: {sefmol_root}")

        # SeFMol code uses top-level `utils.*` / `models.*` imports.
        # Import inside a temporary sys.path window so these names resolve to SeFMol,
        # not to this repo's own top-level packages with the same names.
        orig_sys_path = list(sys.path)
        saved_modules = {}
        try:
            sys.path = [str(sefmol_root)] + [p for p in sys.path if Path(p).resolve() != _REPO]
            # Clear possibly-cached top-level modules from this repo that shadow SeFMol.
            for name in list(sys.modules.keys()):
                if (
                    name == "utils" or name.startswith("utils.")
                    or name == "models" or name.startswith("models.")
                    or name == "datasets" or name.startswith("datasets.")
                ):
                    saved_modules[name] = sys.modules.pop(name)
            sefmol_trans = importlib.import_module("utils.transforms")  # type: ignore[import-not-found]
            PDBProtein = importlib.import_module("utils.data").PDBProtein  # type: ignore[import-not-found]
            sefmol_reconstruct_from_generated = importlib.import_module(
                "utils.reconstruct"
            ).reconstruct_from_generated  # type: ignore[import-not-found]
            sefmol_sfrl = importlib.import_module("models.sefmol_sfrl")  # type: ignore[import-not-found]
            SeFMol = sefmol_sfrl.SeFMol
            initialize_diffusion_params = sefmol_sfrl.initialize_diffusion_params
        finally:
            sys.path = orig_sys_path
            for name in list(sys.modules.keys()):
                if (
                    name == "utils" or name.startswith("utils.")
                    or name == "models" or name.startswith("models.")
                    or name == "datasets" or name.startswith("datasets.")
                ):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_modules)

        self.device = device
        self.timesteps = int(timesteps)
        self.sample_mode = sample_mode
        self.center_pos_mode = center_pos_mode
        try:
            # PyTorch >=2.6 defaults to weights_only=True, which may reject checkpoints
            # that store config objects (e.g., EasyDict). Prefer safe mode first.
            self.ckpt = torch.load(ckpt_path, map_location=device)
        except Exception as e:
            msg = str(e)
            if "Weights only load failed" in msg or "Unsupported global" in msg:
                # This checkpoint is local/user-provided for SeFMol and needs full unpickle.
                # Keep fallback narrow to this known compatibility case.
                self.ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            else:
                raise
        self.ckpt_cfg = self.ckpt["config"]
        self.model_cfg = self.ckpt["config"].model
        self.ligand_atom_mode = _cfg_get(
            self.ckpt_cfg, "data.transform.ligand_atom_mode", "full"
        )
        protein_feat_dim = sefmol_trans.FeaturizeProteinAtom().feature_dim
        ligand_feat_dim = sefmol_trans.FeaturizeLigandAtom(
            self.ligand_atom_mode, properties=condition_properties
        ).feature_dim
        if sample_mode == "rigid_sample" and self.timesteps < 1000:
            initialize_diffusion_params(self.ckpt["model"], self.model_cfg, self.timesteps)
        self.model = SeFMol(
            self.model_cfg,
            protein_atom_feature_dim=protein_feat_dim,
            ligand_atom_feature_dim=ligand_feat_dim,
            n_timesteps=self.timesteps,
        ).to(device)
        self.model.load_state_dict(self.ckpt["model"])
        self.model.eval()

        self._sefmol_trans = sefmol_trans
        self._sefmol_pdb_protein_cls = PDBProtein
        self._sefmol_reconstruct_from_generated = sefmol_reconstruct_from_generated
        cond_dim = int(_cfg_get(self.model_cfg, "condition_dim", 0) or 0)
        default_props = [1, 1, 1, 50, 3.0, 2.0, 0.5, 2]
        src = default_props if condition_properties is None else condition_properties
        src = [float(x) for x in src]
        if cond_dim > 0:
            if len(src) >= cond_dim:
                src = src[:cond_dim]
            else:
                src = src + [0.0] * (cond_dim - len(src))
            self.cond_props = torch.tensor(src, dtype=torch.float32, device=device).view(1, -1)
        else:
            self.cond_props = torch.zeros((1, 0), dtype=torch.float32, device=device)

        _SEFMOL_REFINER = self

    def _map_target_idx_to_sefmol_idx(self, cls_idx: int) -> int:
        z, hyb, aromatic = _IDX_TO_DESC[int(cls_idx)]
        trans = self._sefmol_trans
        if self.ligand_atom_mode == "basic":
            return trans.MAP_ATOM_TYPE_ONLY_TO_INDEX.get(int(z), 0)
        if self.ligand_atom_mode == "add_aromatic":
            return trans.MAP_ATOM_TYPE_AROMATIC_TO_INDEX.get((int(z), bool(aromatic)), 0)
        return trans.MAP_ATOM_TYPE_FULL_TO_INDEX.get((int(z), str(hyb), bool(aromatic)), 0)

    def _prepare_protein_for_sefmol(
        self,
        dataset_item: Optional[dict],
        protein1_pdb: Optional[Path],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (protein_pos, protein_feat) or (None, None) if receptor cannot be built."""
        protein_pos = None
        protein_feat = None

        # Preferred path: use processed dataset coordinates directly (same frame as generated ligands).
        if dataset_item is not None and "p1" in dataset_item:
            try:
                p1 = dataset_item["p1"]
                pos = np.asarray(p1["pos_heavyatom"], dtype=np.float32)      # (L, A, 3)
                mask = np.asarray(p1["mask_heavyatom"], dtype=bool)          # (L, A)
                aa = np.asarray(p1["aa"], dtype=np.int64)                    # (L,)
                l, a = mask.shape
                flat_mask = mask.reshape(l * a)
                flat_pos = pos.reshape(l * a, 3)[flat_mask]
                if flat_pos.shape[0] > 0:
                    residue_idx = np.repeat(np.arange(l), a)[flat_mask]
                    aa_atom = np.clip(aa[residue_idx], 0, 19)
                    elem_oh = np.zeros((flat_pos.shape[0], 6), dtype=np.float32)
                    elem_oh[:, 1] = 1.0  # carbon channel
                    aa_oh = np.eye(20, dtype=np.float32)[aa_atom]
                    atom_slot = np.tile(np.arange(a), l)[flat_mask]
                    is_bb = np.isin(atom_slot, np.array([0, 1, 2, 3], dtype=np.int64)).astype(np.float32).reshape(-1, 1)
                    feat = np.concatenate([elem_oh, aa_oh, is_bb], axis=1).astype(np.float32)
                    protein_pos = torch.as_tensor(flat_pos, dtype=torch.float32, device=self.device)
                    protein_feat = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
            except Exception:
                protein_pos = None
                protein_feat = None

        if protein_pos is None or protein_feat is None:
            if protein1_pdb is None or (not protein1_pdb.is_file()):
                return None, None
            p = self._sefmol_pdb_protein_cls(str(protein1_pdb))
            pd = p.to_dict_atom()
            protein_pos = torch.as_tensor(pd["pos"], dtype=torch.float32, device=self.device)
            protein_feat = torch.as_tensor(
                _build_sefmol_protein_features(pd), dtype=torch.float32, device=self.device
            )
        return protein_pos, protein_feat

    def _run_sefmol_sample_diffusion(
        self,
        protein_pos: torch.Tensor,
        protein_feat: torch.Tensor,
        init_pos: torch.Tensor,
        init_v: torch.Tensor,
    ):
        batch_protein = torch.zeros(protein_pos.size(0), dtype=torch.long, device=self.device)
        batch_ligand = torch.zeros(init_pos.size(0), dtype=torch.long, device=self.device)
        out = None
        sample_errors: List[str] = []
        center_modes = [self.center_pos_mode]
        if self.center_pos_mode == "none":
            center_modes.append("protein")
        for center_mode in center_modes:
            try:
                with torch.no_grad():
                    out = self.model.sample_diffusion(
                        protein_pos=protein_pos,
                        protein_v=protein_feat,
                        batch_protein=batch_protein,
                        init_ligand_pos=init_pos,
                        init_ligand_v=init_v,
                        batch_ligand=batch_ligand,
                        properties=self.cond_props,
                        num_steps=self.timesteps,
                        center_pos_mode=center_mode,
                        log_prob_mode=self.sample_mode,
                    )
                break
            except Exception as e:
                sample_errors.append(f"{center_mode}:{type(e).__name__}:{e}")
        return out, sample_errors

    def _decode_sefmol_output_to_mol(self, out) -> Tuple[Optional[Chem.Mol], Optional[str]]:
        pos = out["pos"].detach().cpu().numpy().astype(np.float64).tolist()
        v = out["v"].detach().cpu().numpy().astype(np.int64).tolist()
        atomic_nums = self._sefmol_trans.get_atomic_number_from_index(
            torch.as_tensor(v, dtype=torch.long), self.ligand_atom_mode
        )
        aromatic_flags = self._sefmol_trans.is_aromatic_from_index(
            torch.as_tensor(v, dtype=torch.long), self.ligand_atom_mode
        )
        if aromatic_flags is None:
            aromatic_flags = [False] * len(atomic_nums)
        try:
            mol = self._sefmol_reconstruct_from_generated(pos, atomic_nums, aromatic_flags)
            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            return mol, smiles
        except Exception as e:
            return None, f"reconstruct_failed:{type(e).__name__}:{e}"

    def refine(
        self,
        cls: List[int],
        coords: List[List[float]],
        dataset_item: Optional[dict] = None,
        protein1_pdb: Optional[Path] = None,
    ) -> Tuple[Optional[Chem.Mol], Optional[str]]:
        if len(cls) != len(coords) or len(cls) == 0:
            return None, "invalid_init_ligand"

        protein_pos, protein_feat = self._prepare_protein_for_sefmol(dataset_item, protein1_pdb)
        if protein_pos is None or protein_feat is None:
            return None, "protein1_not_found"

        init_pos = torch.as_tensor(coords, dtype=torch.float32, device=self.device)
        init_v = torch.as_tensor(
            [self._map_target_idx_to_sefmol_idx(x) for x in cls],
            dtype=torch.long,
            device=self.device,
        )

        out, sample_errors = self._run_sefmol_sample_diffusion(
            protein_pos, protein_feat, init_pos, init_v
        )
        if out is None:
            return None, "sefmol_sample_failed:" + " | ".join(sample_errors)
        mol, smiles_or_err = self._decode_sefmol_output_to_mol(out)
        if mol is None:
            return None, smiles_or_err
        return mol, smiles_or_err

    def refine_from_random_noise(
        self,
        num_atoms: int,
        dataset_item: Optional[dict] = None,
        protein1_pdb: Optional[Path] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Optional[Chem.Mol], Optional[str]]:
        """
        Ablation: SeFMol de novo / unconditional-on-flow init — same as SeFMol/sample.py:
        ligand positions = protein centroid + Gaussian noise; atom types from uniform Gumbel-Softmax sample.
        """
        if num_atoms < 1:
            return None, "invalid_num_atoms"

        protein_pos, protein_feat = self._prepare_protein_for_sefmol(dataset_item, protein1_pdb)
        if protein_pos is None or protein_feat is None:
            return None, "protein1_not_found"

        if seed is not None:
            s = int(seed)
            torch.manual_seed(s)
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

        # Single-graph: all protein atoms belong to graph 0
        center_pos = protein_pos.mean(dim=0)
        init_pos = center_pos.unsqueeze(0).expand(num_atoms, -1).contiguous()
        init_pos = init_pos + torch.randn_like(init_pos)
        uniform_logits = torch.zeros(
            num_atoms, int(self.model.num_classes), dtype=torch.float32, device=self.device
        )
        init_v = self._log_sample_categorical(uniform_logits)

        out, sample_errors = self._run_sefmol_sample_diffusion(
            protein_pos, protein_feat, init_pos, init_v
        )
        if out is None:
            return None, "sefmol_sample_failed:" + " | ".join(sample_errors)
        mol, smiles_or_err = self._decode_sefmol_output_to_mol(out)
        if mol is None:
            return None, smiles_or_err
        return mol, smiles_or_err

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct ligands from detailed_results.json; optional Vina vs protein1."
    )
    parser.add_argument(
        "--detailed_json",
        type=str,
        default="evaluation_results_0415/detailed_results.json",
        help="Path to detailed_results.json",
    )
    parser.add_argument(
        "--pdb_base_dir",
        type=str,
        default=None,
        help="Base directory with per-complex folders (each has protein1.pdb). Enables Vina.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/Moloctite/TernaryDataset_test",
        help="Processed dataset path used by evaluation.py; provides moved protein1 coordinates for Vina.",
    )
    parser.add_argument(
        "--vina_debug",
        action="store_true",
        help="Print detailed diagnostics when Vina returns None.",
    )
    parser.add_argument(
        "--high_affinity_threshold",
        type=float,
        default=-7.0,
        help="Threshold on vina_dock (kcal/mol). high_affinity=1 if vina_dock <= threshold.",
    )
    parser.add_argument(
        "--dock_exhaustiveness",
        type=int,
        default=8,
        help="Vina dock exhaustiveness.",
    )
    parser.add_argument(
        "--dock_n_poses",
        type=int,
        default=20,
        help="Number of poses in Vina dock.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save final evaluation results (JSON).",
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=50,
        help="Evaluate only the first N trajectories per sample (set <=0 to use all).",
    )
    parser.add_argument(
        "--sefmol_refine",
        action="store_true",
        help="Refine reconstructed ligands with SeFMol using generated seq/coords as initialization.",
    )
    parser.add_argument(
        "--sefmol_ckpt",
        type=str,
        default=None,
        help="Path to SeFMol checkpoint (.pt). Required when --sefmol_refine is set.",
    )
    parser.add_argument(
        "--sefmol_device",
        type=str,
        default="cuda:0",
        help="Device for SeFMol refinement (e.g. cuda:0 or cpu).",
    )
    parser.add_argument(
        "--sefmol_timesteps",
        type=int,
        default=50,
        help="Sampling timesteps used in SeFMol refine.",
    )
    parser.add_argument(
        "--sefmol_sample_mode",
        type=str,
        default="sefmol_sample",
        choices=["rigid_sample", "sefmol_sample"],
        help="SeFMol sampling mode.",
    )
    parser.add_argument(
        "--sefmol_center_pos_mode",
        type=str,
        default="protein",
        choices=["none", "protein", "value_func_protein"],
        help="Centering mode passed to SeFMol sample_diffusion.",
    )
    parser.add_argument(
        "--sefmol_condition_properties",
        nargs="+",
        type=float,
        default=None,
        help="Optional condition properties vector for SeFMol (auto adjusted to checkpoint condition_dim).",
    )
    args = parser.parse_args()
    if args.sefmol_refine and not args.sefmol_ckpt:
        parser.error("--sefmol_ckpt is required when --sefmol_refine is set")
    if args.sefmol_refine and not Path(args.sefmol_ckpt).is_file():
        parser.error(f"--sefmol_ckpt not found: {args.sefmol_ckpt}")
    pdb_base = args.pdb_base_dir or os.environ.get("PDB_BASE_DIR")
    dataset_by_name = {}
    if pdb_base or args.sefmol_refine:
        ds = load_from_disk(args.dataset_path)
        dataset_by_name = {item["name"]: item for item in ds}

    sefmol_refiner = None
    if args.sefmol_refine:
        try:
            sefmol_refiner = SeFMolRefiner(
                ckpt_path=args.sefmol_ckpt,
                device=args.sefmol_device,
                timesteps=args.sefmol_timesteps,
                sample_mode=args.sefmol_sample_mode,
                center_pos_mode=args.sefmol_center_pos_mode,
                condition_properties=args.sefmol_condition_properties,
            )
        except Exception as e:
            print(
                f"[SeFMol] init failed; fallback to original reconstruction only: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            sefmol_refiner = None

    with open(args.detailed_json, "r") as f:
        data = json.load(f)

    if args.output_json is not None:
        output_path = Path(args.output_json)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.detailed_json).resolve().parent / f"ligand_eval_results_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "detailed_json": args.detailed_json,
        "pdb_base_dir": pdb_base,
        "dataset_path": args.dataset_path,
        "high_affinity_threshold": args.high_affinity_threshold,
        "dock_exhaustiveness": args.dock_exhaustiveness,
        "dock_n_poses": args.dock_n_poses,
        "sefmol_refine": args.sefmol_refine,
        "sefmol_ckpt": args.sefmol_ckpt if args.sefmol_refine else None,
        "sefmol_device": args.sefmol_device if args.sefmol_refine else None,
        "sefmol_timesteps": args.sefmol_timesteps if args.sefmol_refine else None,
        "sefmol_sample_mode": args.sefmol_sample_mode if args.sefmol_refine else None,
        "sefmol_center_pos_mode": (
            args.sefmol_center_pos_mode if args.sefmol_refine else None
        ),
        "sefmol_condition_properties": (
            args.sefmol_condition_properties if args.sefmol_refine else None
        ),
    }

    vina_score_vals, vina_min_vals, vina_dock_vals, high_affinity_vals = [], [], [], []

    per_sample_results = []
    total_samples = len(data["sample_results"])

    def _avg_med(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
        if not vals:
            return None, None
        arr = np.asarray(vals, dtype=np.float64)
        return float(np.mean(arr)), float(np.median(arr))

    def _build_summary() -> Dict[str, Optional[float]]:
        score_avg, score_med = _avg_med(vina_score_vals)
        min_avg, min_med = _avg_med(vina_min_vals)
        dock_avg, dock_med = _avg_med(vina_dock_vals)
        ha_avg, ha_med = _avg_med(high_affinity_vals)
        return {
            "vina_score_avg": score_avg,
            "vina_score_median": score_med,
            "vina_min_avg": min_avg,
            "vina_min_median": min_med,
            "vina_dock_avg": dock_avg,
            "vina_dock_median": dock_med,
            "high_affinity_avg": ha_avg,
            "high_affinity_median": ha_med,
        }

    def _save_progress(processed_samples: int, is_final: bool = False) -> None:
        payload = {
            "config": config_payload,
            "summary": _build_summary(),
            "num_results": len(per_sample_results),
            "processed_samples": processed_samples,
            "total_samples": total_samples,
            "is_final": is_final,
            "results": per_sample_results,
        }
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        tmp_path.replace(output_path)

    for sample_idx, item in enumerate(data["sample_results"], start=1):
        lig_seq_preds = item["all_trajectories_lig_seq_pred_atoms"]
        lig_coords_all = item["all_trajectories_lig_coords_pred"]

        if len(lig_seq_preds) != len(lig_coords_all):
            raise ValueError(
                f"Trajectory mismatch for {item.get('name', '')}: "
                f"seq={len(lig_seq_preds)} coords={len(lig_coords_all)}"
            )

        if args.max_trajectories is not None and int(args.max_trajectories) > 0:
            max_traj = int(args.max_trajectories)
            lig_seq_preds = lig_seq_preds[:max_traj]
            lig_coords_all = lig_coords_all[:max_traj]
        for traj_idx, (sample_seq, traj_coords) in enumerate(zip(lig_seq_preds, lig_coords_all)):
            rdmol = None
            smiles = None
            qv, sv = None, None
            reconstruct_error = None
            sefmol_refine_error = None

            # Use SefMol to refine our ligand with our generated atom seq and coords

            try:
                cls, coords = prepare_ligand_arrays(
                    sample_seq,
                    traj_coords,
                    drop_hydrogen=False,
                )
                # Try SeFMol refinement first; if it fails, fall back to direct reconstruction.
                if args.sefmol_refine and sefmol_refiner is not None:
                    ds_item = dataset_by_name.get(item["name"])
                    p1_for_sefmol = Path(pdb_base) / item["name"] / "protein1.pdb" if pdb_base else None
                    p1_tmp = None
                    try:
                        if pdb_base:
                            if ds_item is not None:
                                p1_tmp = build_processed_protein1_pdb(
                                    item["name"], Path(pdb_base), ds_item, debug=args.vina_debug
                                )
                                if p1_tmp is not None:
                                    p1_for_sefmol = p1_tmp
                        refined_mol, sefmol_refine_error = sefmol_refiner.refine(
                            cls,
                            coords,
                            dataset_item=ds_item,
                            protein1_pdb=p1_for_sefmol,
                        )
                        # print(refined_mol)
                        if refined_mol is not None:
                            rdmol = refined_mol
                            smiles = Chem.MolToSmiles(
                                rdmol, canonical=True, isomericSmiles=True
                            )
                            # print(smiles)
                            # print(rdmol)
                    finally:
                        if p1_tmp is not None:
                            shutil.rmtree(p1_tmp.parent, ignore_errors=True)
                if rdmol is None:
                    rdmol, smiles = build_molecule(cls, coords)
                qv, sv = compute_qed_sa(rdmol)
            except Exception as e:
                reconstruct_error = f"{type(e).__name__}: {e}"

            vina_score = None
            vina_min = None
            vina_dock = None
            high_affinity = None
            if pdb_base and rdmol is not None:
                p1_for_vina = Path(pdb_base) / item["name"] / "protein1.pdb"
                p1_tmp = None
                ds_item = dataset_by_name.get(item["name"])
                if ds_item is not None:
                    p1_tmp = build_processed_protein1_pdb(
                        item["name"], Path(pdb_base), ds_item, debug=args.vina_debug
                    )
                    if p1_tmp is not None:
                        p1_for_vina = p1_tmp
                elif args.vina_debug:
                    print(f"[Vina:{item['name']}] processed p1 skip: sample not found in dataset")

                try:
                    vina_score, vina_min, vina_dock, high_affinity = run_vina_metrics(
                        rdmol,
                        p1_for_vina,
                        debug=args.vina_debug,
                        sample_name=f"{item['name']}_{traj_idx}",
                        high_affinity_threshold=args.high_affinity_threshold,
                        dock_exhaustiveness=args.dock_exhaustiveness,
                        dock_n_poses=args.dock_n_poses,
                    )
                finally:
                    if p1_tmp is not None:
                        shutil.rmtree(p1_tmp.parent, ignore_errors=True)
            if vina_score is not None:
                vina_score_vals.append(vina_score)
            if vina_min is not None:
                vina_min_vals.append(vina_min)
            if vina_dock is not None:
                vina_dock_vals.append(vina_dock)
            if high_affinity is not None:
                high_affinity_vals.append(high_affinity)

            row = {
                "name": item["name"],
                "sample_idx": item.get("sample_idx"),
                "traj_idx": traj_idx,
                "smiles": smiles,
                "qed": qv,
                "sa": sv,
                "vina_score": vina_score,
                "vina_min": vina_min,
                "vina_dock": vina_dock,
                "high_affinity": high_affinity,
                "reconstruct_error": reconstruct_error,
                "sefmol_refine_error": sefmol_refine_error,
            }
            per_sample_results.append(row)

            print_args = [
                item["name"],
                f"traj_{traj_idx}",
                smiles,
                "QED",
                qv,
                "SA",
                sv,
                "VinaScore",
                vina_score,
                "VinaMin",
                vina_min,
                "VinaDock",
                vina_dock,
                "HighAffinity",
                high_affinity,
                "SeFMolErr",
                sefmol_refine_error,
                "ReconErr",
                reconstruct_error,
            ]
            print(*print_args, flush=True)

        # Save incrementally after each sample, so progress survives interruptions.
        _save_progress(processed_samples=sample_idx, is_final=False)
        print(
            f"[Progress] Saved {sample_idx}/{total_samples} samples to: {output_path}",
            flush=True,
        )

    score_avg, score_med = _avg_med(vina_score_vals)
    min_avg, min_med = _avg_med(vina_min_vals)
    dock_avg, dock_med = _avg_med(vina_dock_vals)
    ha_avg, ha_med = _avg_med(high_affinity_vals)
    print(
        "\nSummary:",
        "VinaScore(avg/med)=", score_avg, score_med,
        "VinaMin(avg/med)=", min_avg, min_med,
        "VinaDock(avg/med)=", dock_avg, dock_med,
        "HighAffinity(avg/med)=", ha_avg, ha_med,
    )

    summary = {
        "vina_score_avg": score_avg,
        "vina_score_median": score_med,
        "vina_min_avg": min_avg,
        "vina_min_median": min_med,
        "vina_dock_avg": dock_avg,
        "vina_dock_median": dock_med,
        "high_affinity_avg": ha_avg,
        "high_affinity_median": ha_med,
    }
    _save_progress(processed_samples=total_samples, is_final=True)
    print(f"Saved final results to: {output_path}")