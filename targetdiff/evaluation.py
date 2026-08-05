import argparse
import contextlib
import json
import os
import tempfile
import subprocess
import shutil
import time
import multiprocessing as mp
from queue import Empty

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from utils import reconstruct
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH, ProteinLigandData, torchify_dict
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from utils.data import PDBProtein, parse_sdf_file
from utils.evaluation import atom_num, scoring_func


def evaluate_ligand_predictions(pred_pos_list, pred_v_list, gt_pos, gt_v):
    """
    Compute per-sample ligand RMSD and atom-type(sequence) accuracy.

    Notes:
      - Metrics are only valid when predicted and GT atom counts match.
      - Returns NaN for invalid samples (e.g. mismatched atom counts).
    """
    gt_pos_np = gt_pos.detach().cpu().numpy().astype(np.float64)
    gt_v_np = gt_v.detach().cpu().numpy().astype(np.int64)

    rmsd_list, seq_acc_list = [], []
    for pred_pos, pred_v in zip(pred_pos_list, pred_v_list):
        pred_pos = np.asarray(pred_pos, dtype=np.float64)
        pred_v = np.asarray(pred_v, dtype=np.int64)

        if pred_pos.shape != gt_pos_np.shape or pred_v.shape != gt_v_np.shape:
            rmsd_list.append(float('nan'))
            seq_acc_list.append(float('nan'))
            continue

        diff = pred_pos - gt_pos_np
        rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=-1)))
        seq_acc = np.mean((pred_v == gt_v_np).astype(np.float64))
        rmsd_list.append(float(rmsd))
        seq_acc_list.append(float(seq_acc))

    valid_rmsd = [x for x in rmsd_list if np.isfinite(x)]
    valid_seq_acc = [x for x in seq_acc_list if np.isfinite(x)]
    summary = {
        'num_samples': len(rmsd_list),
        'num_valid_metrics': len(valid_rmsd),
        'ligand_rmsd_mean': float(np.mean(valid_rmsd)) if valid_rmsd else None,
        'ligand_rmsd_std': float(np.std(valid_rmsd)) if valid_rmsd else None,
        'ligand_rmsd_min': float(np.min(valid_rmsd)) if valid_rmsd else None,
        'ligand_rmsd_max': float(np.max(valid_rmsd)) if valid_rmsd else None,
        'seq_acc_mean': float(np.mean(valid_seq_acc)) if valid_seq_acc else None,
        'seq_acc_std': float(np.std(valid_seq_acc)) if valid_seq_acc else None,
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
    if not os.path.isfile(protein_pdb):
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


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def build_data_from_pdb_sdf(pdb_path, sdf_path, transform, pocket_radius=10):
    """Build one ProteinLigandData sample from raw pdb/sdf files with pocket extraction."""
    protein = PDBProtein(pdb_path)
    ligand_dict = parse_sdf_file(sdf_path)

    # Extract pocket residues around ligand (same idea as scripts/data_preparation/extract_pockets.py).
    pocket_block = protein.residues_to_pdb_block(
        protein.query_residues_ligand(ligand_dict, pocket_radius)
    )
    protein_dict = PDBProtein(pocket_block).to_dict_atom()

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(protein_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data.protein_filename = pdb_path
    data.ligand_filename = sdf_path

    if transform is not None:
        data = transform(data)
    return data


def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior'):
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)

        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            else:
                raise ValueError

            # init ligand pos
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

            # init ligand v
            if pos_only:
                init_ligand_v = batch.ligand_atom_feature_full
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)

            r = model.sample_diffusion(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,

                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode
            )
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            # unbatch pos
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]  # num_samples * [num_atoms_i, 3]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list


def collect_mgd_pairs(mgd_test_dir):
    """Collect (name, protein1.pdb, ligand_rcsb.sdf) under MGD_test."""
    pairs = []
    for name in sorted(os.listdir(mgd_test_dir)):
        sub = os.path.join(mgd_test_dir, name)
        if not os.path.isdir(sub):
            continue
        pdb_path = os.path.join(sub, 'protein1.pdb')
        sdf_path = os.path.join(sub, 'ligand_rcsb.sdf')
        if os.path.exists(pdb_path) and os.path.exists(sdf_path):
            pairs.append((name, pdb_path, sdf_path))
    return pairs


def process_one_data(model, config, data, data_tag, result_path, args, atom_enc_mode):
    pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
        model, data, config.sample.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms
    )

    ligand_rmsd, seq_acc, eval_summary = evaluate_ligand_predictions(
        pred_pos_list=pred_pos,
        pred_v_list=pred_v,
        gt_pos=data.ligand_pos,
        gt_v=data.ligand_atom_feature_full
    )

    per_sample_rows = []
    pool = {'succ': 0, 'incomp': 0, 'bad': 0}
    for i, (p_pos, p_v, rmsd_i, seq_acc_i) in enumerate(zip(pred_pos, pred_v, ligand_rmsd, seq_acc)):
        row = {
            'complex': str(data_tag),
            'index': i,
            'tag': 'bad',
            'smiles': '',
            'ligand_rmsd': float(rmsd_i) if np.isfinite(rmsd_i) else float('nan'),
            'seq_acc': float(seq_acc_i) if np.isfinite(seq_acc_i) else float('nan'),
            'qed': None,
            'sa': None,
            'vina_score': None,
            'vina_min': None,
            'vina_dock': None,
            'high_affinity': None,
            'reconstruct_error': None,
            'ligand_atom_types_pred': np.asarray(p_v).astype(np.int64).tolist(),
            'ligand_coords_pred': np.asarray(p_pos).astype(np.float64).tolist() if args.save_ligand_coords else None,
        }
        try:
            pred_v_tensor = torch.as_tensor(np.asarray(p_v), dtype=torch.long)
            pred_atom_type = trans.get_atomic_number_from_index(pred_v_tensor, mode=atom_enc_mode)
            pred_aromatic = trans.is_aromatic_from_index(pred_v_tensor, mode=atom_enc_mode)
            mol = reconstruct.reconstruct_from_generated(p_pos, pred_atom_type, pred_aromatic)
            smiles = Chem.MolToSmiles(mol)
            row['smiles'] = smiles
            if '.' in smiles:
                row['tag'] = 'incomp'
                pool['incomp'] += 1
            else:
                row['tag'] = 'succ'
                pool['succ'] += 1
            chem = scoring_func.get_chem(mol)
            row['qed'] = float(chem['qed']) if chem.get('qed') is not None else None
            row['sa'] = float(chem['sa']) if chem.get('sa') is not None else None
            vina_score, vina_min, vina_dock, high_affinity = run_vina_metrics(
                mol,
                data.protein_filename,
                debug=args.vina_debug,
                sample_name=f"{data_tag}_{i}",
                high_affinity_threshold=args.high_affinity_threshold,
                dock_exhaustiveness=args.dock_exhaustiveness,
                dock_n_poses=args.dock_n_poses,
            )
            row['vina_score'] = vina_score
            row['vina_min'] = vina_min
            row['vina_dock'] = vina_dock
            row['high_affinity'] = high_affinity
        except Exception as e:
            pool['bad'] += 1
            row['reconstruct_error'] = f'{type(e).__name__}: {e}'
        per_sample_rows.append(row)

    qed_stats = summarize_valid_values([r.get('qed') for r in per_sample_rows])
    sa_stats = summarize_valid_values([r.get('sa') for r in per_sample_rows])
    vina_score_stats = summarize_valid_values([r.get('vina_score') for r in per_sample_rows])
    vina_min_stats = summarize_valid_values([r.get('vina_min') for r in per_sample_rows])
    vina_dock_stats = summarize_valid_values([r.get('vina_dock') for r in per_sample_rows])
    high_aff_stats = summarize_valid_values([r.get('high_affinity') for r in per_sample_rows])
    valid_lr = [x for x in ligand_rmsd if np.isfinite(x)]
    valid_seq = [x for x in seq_acc if np.isfinite(x)]
    complex_summary = {
        'complex': str(data_tag),
        'num_generated': len(per_sample_rows),
        'succ': pool['succ'],
        'incomp': pool['incomp'],
        'bad': pool['bad'],
        'ligand_rmsd_mean': float(np.mean(valid_lr)) if valid_lr else None,
        'ligand_rmsd_std': float(np.std(valid_lr)) if valid_lr else None,
        'seq_acc_mean': float(np.mean(valid_seq)) if valid_seq else None,
        'seq_acc_std': float(np.std(valid_seq)) if valid_seq else None,
        'num_valid_ligand_metrics': len(valid_lr),
        'qed_mean': qed_stats['mean'],
        'qed_std': qed_stats['std'],
        'qed_median': qed_stats['median'],
        'num_valid_qed': qed_stats['num_valid'],
        'sa_mean': sa_stats['mean'],
        'sa_std': sa_stats['std'],
        'sa_median': sa_stats['median'],
        'num_valid_sa': sa_stats['num_valid'],
        'vina_score_mean': vina_score_stats['mean'],
        'vina_score_std': vina_score_stats['std'],
        'vina_score_median': vina_score_stats['median'],
        'num_valid_vina_score': vina_score_stats['num_valid'],
        'vina_min_mean': vina_min_stats['mean'],
        'vina_min_std': vina_min_stats['std'],
        'vina_min_median': vina_min_stats['median'],
        'num_valid_vina_min': vina_min_stats['num_valid'],
        'vina_dock_mean': vina_dock_stats['mean'],
        'vina_dock_std': vina_dock_stats['std'],
        'vina_dock_median': vina_dock_stats['median'],
        'num_valid_vina_dock': vina_dock_stats['num_valid'],
        'high_affinity_rate': high_aff_stats['mean'],
        'high_affinity_std': high_aff_stats['std'],
        'high_affinity_median': high_aff_stats['median'],
        'num_valid_high_affinity': high_aff_stats['num_valid'],
    }

    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj,
        'time': time_list,
        'ligand_rmsd': ligand_rmsd,
        'seq_acc': seq_acc,
        'eval_summary': eval_summary,
        'per_sample_rows': per_sample_rows,
        'complex_summary': complex_summary,
    }

    os.makedirs(result_path, exist_ok=True)
    result_pt = os.path.join(result_path, f'result_{data_tag}.pt')
    torch.save(result, result_pt)
    metrics_json = os.path.join(result_path, f'result_{data_tag}_metrics.json')
    with open(metrics_json, 'w') as f:
        json.dump(
            {
                'eval_summary': eval_summary,
                'ligand_rmsd': ligand_rmsd,
                'seq_acc': seq_acc,
                'complex_summary': complex_summary,
            },
            f,
            indent=2
        )
    samples_jsonl = os.path.join(result_path, 'samples.jsonl')
    with open(samples_jsonl, 'w') as f:
        for row in per_sample_rows:
            f.write(json.dumps(row) + '\n')
    return {
        'name': str(data_tag),
        'result_pt': result_pt,
        'metrics_json': metrics_json,
        'samples_jsonl': samples_jsonl,
        'eval_summary': eval_summary,
        'summary': complex_summary,
    }


def run_mgd_worker(worker_idx, gpu_id, pair_shard, args_dict, out_queue):
    class Args:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    args = Args(args_dict)
    args.device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)

    worker_results = []
    try:
        logger = misc.get_logger(f'sampling_worker_{worker_idx}')
        config = misc.load_config(args.config)
        misc.seed_all(config.sample.seed + worker_idx)
        ckpt = torch.load(config.model.checkpoint, map_location=args.device, weights_only=False)

        protein_featurizer = trans.FeaturizeProteinAtom()
        ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
        atom_enc_mode = args.atom_enc_mode or ligand_atom_mode
        ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
        transform = Compose([
            protein_featurizer,
            ligand_featurizer,
            trans.FeaturizeLigandBond(),
        ])

        model = ScorePosNet3D(
            ckpt['config'].model,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=ligand_featurizer.feature_dim
        ).to(args.device)
        model.load_state_dict(ckpt['model'])
        model.eval()

        for name, pdb_path, sdf_path in pair_shard:
            try:
                data = build_data_from_pdb_sdf(pdb_path, sdf_path, transform, pocket_radius=args.pocket_radius)
                item_result_dir = os.path.join(args.result_path, name)
                out = process_one_data(model, config, data, name, item_result_dir, args, atom_enc_mode)
                worker_results.append(out)
                out_queue.put({'type': 'progress', 'name': name})
            except Exception as e:
                out_queue.put({'type': 'error', 'name': name, 'error': str(e)})
    except Exception as e:
        out_queue.put({'type': 'worker_error', 'worker': worker_idx, 'error': str(e)})
    out_queue.put({'type': 'done', 'results': worker_results})


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('-i', '--data_id', type=int)
    parser.add_argument('--pdb_path', type=str, default=None,
                        help='Path to input pocket PDB file. If set together with --sdf_path, dataset loading is skipped.')
    parser.add_argument('--sdf_path', type=str, default=None,
                        help='Path to input ligand SDF file. If set together with --pdb_path, dataset loading is skipped.')
    parser.add_argument('--pocket_radius', type=int, default=10,
                        help='Pocket extraction radius in Angstrom when using --pdb_path/--sdf_path.')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./outputs')
    parser.add_argument('--mgd_test_dir', type=str,
                        default='/home/yuliangyan/Code/Trust-App-AI-Lab/molecular_glue_design/data/TernaryDB/MGD_test',
                        help='Root dir containing complex folders with protein1.pdb and ligand_rcsb.sdf')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='Comma-separated GPU ids for directory traversal mode, e.g. "0,1,2,3"')
    parser.add_argument('--atom_enc_mode', type=str, default=None,
                        choices=['basic', 'add_aromatic', 'full'],
                        help='Atom encoding mode for reconstructing molecules; default uses model checkpoint mode.')
    parser.add_argument('--no_save_ligand_coords', action='store_true',
                        help='Disable saving sampled ligand coordinates to JSONL.')
    parser.add_argument('--vina_debug', action='store_true', help='Print detailed Vina diagnostics')
    parser.add_argument('--high_affinity_threshold', type=float, default=-7.0)
    parser.add_argument('--dock_exhaustiveness', type=int, default=8)
    parser.add_argument('--dock_n_poses', type=int, default=20)
    args = parser.parse_args()
    args.save_ligand_coords = not args.no_save_ligand_coords

    logger = misc.get_logger('sampling')

    # Directory traversal mode for MGD_test: when neither custom pair nor data_id is provided.
    use_custom_pair = (args.pdb_path is not None) or (args.sdf_path is not None)
    if (not use_custom_pair) and (args.data_id is None):
        # Directory traversal mode for MGD_test.
        pairs = collect_mgd_pairs(args.mgd_test_dir)
        logger.info(f'Found {len(pairs)} valid complexes under {args.mgd_test_dir}')
        if len(pairs) == 0:
            raise ValueError(f'No valid pairs found in {args.mgd_test_dir}')

        os.makedirs(args.result_path, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(args.result_path, 'sample.yml'))

        if args.gpu_ids is None:
            # Build runtime components only for single-process traversal.
            config = misc.load_config(args.config)
            misc.seed_all(config.sample.seed)
            ckpt = torch.load(config.model.checkpoint, map_location=args.device, weights_only=False)
            protein_featurizer = trans.FeaturizeProteinAtom()
            ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
            atom_enc_mode = args.atom_enc_mode or ligand_atom_mode
            ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
            transform = Compose([
                protein_featurizer,
                ligand_featurizer,
                trans.FeaturizeLigandBond(),
            ])
            model = ScorePosNet3D(
                ckpt['config'].model,
                protein_atom_feature_dim=protein_featurizer.feature_dim,
                ligand_atom_feature_dim=ligand_featurizer.feature_dim
            ).to(args.device)
            model.load_state_dict(ckpt['model'])
            model.eval()

            # Single-process traversal fallback.
            logger.info('Running single-process traversal (set --gpu_ids for multi-GPU)')
            all_results = []
            for name, pdb_path, sdf_path in tqdm(pairs, desc='MGD traversal'):
                try:
                    data = build_data_from_pdb_sdf(pdb_path, sdf_path, transform, pocket_radius=args.pocket_radius)
                    item_result_dir = os.path.join(args.result_path, name)
                    out = process_one_data(model, config, data, name, item_result_dir, args, atom_enc_mode)
                    all_results.append({'ok': True, **out})
                except Exception as e:
                    logger.warning(f'Failed on {name}: {e}')
                    all_results.append({'ok': False, 'name': name, 'error': str(e)})
        else:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',') if x.strip() != '']
            if len(gpu_ids) == 0:
                raise ValueError('Empty --gpu_ids')
            shards = [pairs[i::len(gpu_ids)] for i in range(len(gpu_ids))]
            manager = mp.Manager()
            q = manager.Queue()
            args_dict = vars(args).copy()
            args_dict.pop('device', None)

            procs = []
            for wi, (gid, shard) in enumerate(zip(gpu_ids, shards)):
                if len(shard) == 0:
                    continue
                p = mp.Process(target=run_mgd_worker, args=(wi, gid, shard, args_dict, q))
                p.start()
                procs.append(p)

            done_count, progress_count = 0, 0
            all_results = []
            while done_count < len(procs):
                try:
                    msg = q.get(timeout=5)
                except Empty:
                    alive = [p for p in procs if p.is_alive()]
                    if len(alive) == 0:
                        break
                    continue
                if msg['type'] == 'progress':
                    progress_count += 1
                    if progress_count % 10 == 0 or progress_count == len(pairs):
                        logger.info(f'Progress: {progress_count}/{len(pairs)}')
                elif msg['type'] == 'error':
                    logger.warning(f"Failed on {msg['name']}: {msg['error']}")
                    all_results.append({'ok': False, 'name': msg['name'], 'error': msg['error']})
                elif msg['type'] == 'worker_error':
                    logger.warning(f"Worker-{msg['worker']} failed before sampling: {msg['error']}")
                elif msg['type'] == 'done':
                    done_count += 1
                    all_results.extend([{'ok': True, **r} for r in msg['results']])

            for p in procs:
                p.join()
                if p.exitcode != 0:
                    raise RuntimeError(f'Worker failed with exit code {p.exitcode}')

        dataset_summary_path = os.path.join(args.result_path, 'dataset_summary.json')
        with open(dataset_summary_path, 'w') as f:
            json.dump({'num_items': len(all_results), 'items': all_results}, f, indent=2)

        per_complex_summary = []
        for item in all_results:
            if item.get('ok') and item.get('summary') is not None:
                per_complex_summary.append(item['summary'])
            else:
                per_complex_summary.append({'complex': item.get('name', ''), 'error': item.get('error', '')})
        per_complex_path = os.path.join(args.result_path, 'per_complex_summary.json')
        with open(per_complex_path, 'w') as f:
            json.dump(per_complex_summary, f, indent=2)

        ok_summary = [s for s in per_complex_summary if 'succ' in s]
        global_summary = {'complexes_total': len(per_complex_summary), 'complexes_ok': len(ok_summary)}
        if ok_summary:
            mean_lr = [s['ligand_rmsd_mean'] for s in ok_summary if s.get('ligand_rmsd_mean') is not None]
            mean_seq = [s['seq_acc_mean'] for s in ok_summary if s.get('seq_acc_mean') is not None]
            mean_qed = [s['qed_mean'] for s in ok_summary if s.get('qed_mean') is not None]
            mean_sa = [s['sa_mean'] for s in ok_summary if s.get('sa_mean') is not None]
            mean_vina_score = [s['vina_score_mean'] for s in ok_summary if s.get('vina_score_mean') is not None]
            mean_vina_min = [s['vina_min_mean'] for s in ok_summary if s.get('vina_min_mean') is not None]
            mean_vina_dock = [s['vina_dock_mean'] for s in ok_summary if s.get('vina_dock_mean') is not None]
            mean_high_aff = [s['high_affinity_rate'] for s in ok_summary if s.get('high_affinity_rate') is not None]
            global_summary.update({
                'mean_of_per_complex_ligand_rmsd_mean': float(np.mean(mean_lr)) if mean_lr else None,
                'mean_of_per_complex_seq_acc_mean': float(np.mean(mean_seq)) if mean_seq else None,
                'mean_of_per_complex_qed_mean': float(np.mean(mean_qed)) if mean_qed else None,
                'mean_of_per_complex_sa_mean': float(np.mean(mean_sa)) if mean_sa else None,
                'mean_of_per_complex_vina_score_mean': float(np.mean(mean_vina_score)) if mean_vina_score else None,
                'mean_of_per_complex_vina_min_mean': float(np.mean(mean_vina_min)) if mean_vina_min else None,
                'mean_of_per_complex_vina_dock_mean': float(np.mean(mean_vina_dock)) if mean_vina_dock else None,
                'mean_of_per_complex_high_affinity_rate': float(np.mean(mean_high_aff)) if mean_high_aff else None,
            })
        global_summary_path = os.path.join(args.result_path, 'global_summary.json')
        with open(global_summary_path, 'w') as f:
            json.dump(global_summary, f, indent=2)

        logger.info(f'Saved dataset summary to {dataset_summary_path}')
        logger.info(f'Saved per-complex summary to {per_complex_path}')
        logger.info(f'Saved global summary to {global_summary_path}')
        raise SystemExit(0)

    # Single-item mode runtime setup.
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)
    ckpt = torch.load(config.model.checkpoint, map_location=args.device, weights_only=False)
    logger.info(f"Training Config: {ckpt['config']}")
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    atom_enc_mode = args.atom_enc_mode or ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    # Build the single target sample.
    use_custom_pair = (args.pdb_path is not None) or (args.sdf_path is not None)
    if use_custom_pair:
        if not (args.pdb_path and args.sdf_path):
            raise ValueError('Both --pdb_path and --sdf_path must be provided together.')
        data = build_data_from_pdb_sdf(
            args.pdb_path, args.sdf_path, transform, pocket_radius=args.pocket_radius
        )
        logger.info(
            f'Using custom pair with extracted pocket: '
            f'pdb={args.pdb_path}, sdf={args.sdf_path}, radius={args.pocket_radius}A'
        )
        data_tag = os.path.splitext(os.path.basename(args.sdf_path))[0]
    else:
        dataset, subsets = get_dataset(
            config=ckpt['config'].data,
            transform=transform
        )
        _, test_set = subsets['train'], subsets['test']
        logger.info(f'Successfully load the dataset (size: {len(test_set)})!')
        data = test_set[args.data_id]
        data_tag = str(args.data_id)

    out = process_one_data(model, config, data, data_tag, args.result_path, args, atom_enc_mode)
    logger.info('Sample done!')
    logger.info(
        f"Metrics: valid={out['eval_summary']['num_valid_metrics']}/{out['eval_summary']['num_samples']}, "
        f"ligand_rmsd_mean={out['eval_summary']['ligand_rmsd_mean']}, "
        f"seq_acc_mean={out['eval_summary']['seq_acc_mean']}"
    )
    os.makedirs(args.result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(args.result_path, 'sample.yml'))