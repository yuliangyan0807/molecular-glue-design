"""
Evaluation script for Ternary Flow Matching Model.
Evaluates the model on a dataset and computes various metrics.
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from datasets import load_from_disk
import multiprocessing as mp

from datasets import Dataset, load_from_disk
from utils.rigid_utils import parse_pdb, get_torsion_angle, parse_pdb_ligand
from utils.constants import BBHeavyAtom, MAP_ATOM_TYPE_FULL_TO_INDEX
# from utils.reconstruct import MolReconsError, reconstruct_from_generated
from rdkit import Chem

from utils.training_utils import (
    collate_fn,
    compute_rmsd,
    compute_translation_error,
    compute_rotation_error,
    compute_sequence_accuracy,
    load_config_from_yaml,
    merge_pdbs,
    set_new_coords,
    write_pdb,
    set_new_chain,
)
from utils.rigid_utils import rotate_and_translate
from flow_model import TernaryFlowModel
from configs.config import DictToObject
import tempfile
import shutil
import zlib

# from DockQ import DockQ
from DeepTernary.DockQ.dockq_util import cal_dockq
from biopandas.pdb import PandasPdb


INDEX_TO_ATOMIC_NUM = {idx: atom_desc[0] for atom_desc, idx in MAP_ATOM_TYPE_FULL_TO_INDEX.items()}
PXM_ATOMIC_NUMBERS = [6, 7, 8, 9, 15, 16, 17, 5, 35, 53, 34]


def decode_lig_seq_to_atomic_nums(lig_seq, mol_mask):
    """Decode valid ligand atom type indices to atomic numbers."""
    valid_seq = lig_seq[mol_mask].detach().cpu().numpy().astype(np.int64)
    atomic_nums = []
    for atom_idx in valid_seq:
        if atom_idx not in INDEX_TO_ATOMIC_NUM:
            raise ValueError(f"Unknown ligand atom type index: {atom_idx}")
        atomic_nums.append(int(INDEX_TO_ATOMIC_NUM[atom_idx]))
    return atomic_nums


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Ternary Flow Matching Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to evaluate (None = all)')
    parser.add_argument('--num_trajectories_per_sample', type=int, default=1, 
                       help='Total number of trajectories to sample per input sample')
    parser.add_argument('--trajectory_batch_size', type=int, default=None,
                       help='Max trajectories per model.sample() call (GPU micro-batch). '
                            'None means one batch of size num_trajectories_per_sample (legacy behavior).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_predictions', action='store_true', 
                       help='Save individual predictions to files')
    parser.add_argument('--pdb_base_dir', type=str, required=True,
                       help='Base directory containing PDB files (required for DockQ calculation)')
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='Comma-separated GPU IDs to use (e.g., "0,1,2,3"). If not specified, uses all available GPUs.')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of worker processes (default: number of GPUs)')
    parser.add_argument('--use_reconstruct', action='store_true',
                       help='Enable ligand reconstruction from predicted atom types and coordinates')
    return parser.parse_args()


def load_model_and_config(args):
    """Load model from checkpoint and config."""
    verbose = getattr(args, "verbose", True)
    if verbose:
        print(f"Loading configuration from {args.config}...")
    config = load_config_from_yaml(args.config)
    
    # Build model config structure that TernaryFlowModel expects
    model_config = DictToObject({
        'node_embed_size': config.model.encoder.node_embed_size,
        'edge_embed_size': config.model.encoder.edge_embed_size,
        'ipa': config.model.encoder.ipa,
        'interface_model': DictToObject({
            'path': getattr(config.model.interface_model, 'path', None),
            'trainable': getattr(config.model.interface_model, 'trainable', False),
            'feat_dim': config.model.interface_model.feat_dim,
            'depth': getattr(config.model.interface_model, 'depth', 4),
            'num_nearest_neighbors': getattr(config.model.interface_model, 'num_nearest_neighbors', 16),
            'topk_k': getattr(config.model.interface_model, 'topk_k', 50),
        })
    })
    
    # Get sampling config
    if hasattr(config.model.interpolant, 'sampling'):
        sampling_config = config.model.interpolant.sampling
        if hasattr(sampling_config, 'num_timesteps'):
            sampling_config = DictToObject({'num_steps': sampling_config.num_timesteps})
        else:
            sampling_config = DictToObject({'num_steps': getattr(sampling_config, 'num_steps', 100)})
    else:
        sampling_config = DictToObject({'num_steps': 100})
    
    full_config = DictToObject({
        'model': model_config,
        'interpolant': config.model.interpolant,
        'sampling': sampling_config
    })
    
    # Initialize model
    if verbose:
        print("Initializing model...")
    device = torch.device(args.device)
    model = TernaryFlowModel(full_config)
    model = model.to(device)
    
    # Load checkpoint
    if verbose:
        print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    if verbose:
        print("✓ Model loaded successfully")
    
    return model, config, device
    
def _move_batch_to_device(batch, device):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        elif isinstance(batch[key], dict):
            for sub_key in batch[key]:
                if isinstance(batch[key][sub_key], torch.Tensor):
                    batch[key][sub_key] = batch[key][sub_key].to(device)


def evaluate_sample(model, item, device, num_trajectories_per_sample=1, args=None, sample_idx=None):
    """
    Evaluate a single sample, optionally sampling multiple trajectories.

    Trajectories are accumulated up to num_trajectories_per_sample. If trajectory_batch_size
    is set (via args), each model.sample() uses at most that many copies per forward pass
    to cap GPU memory.
    """
    if args is None:
        raise ValueError("evaluate_sample requires args (e.g. pdb_base_dir, seed).")

    chunk = getattr(args, 'trajectory_batch_size', None)
    if chunk is None:
        chunk = num_trajectories_per_sample
    chunk = max(1, min(int(chunk), num_trajectories_per_sample))
    seed = args.seed

    name = item.get('name', 'sample')
    sample_seed_off = zlib.crc32(str(name).encode()) & 0x7FFFFFFF
    if sample_idx is None:
        sample_idx = -1

    pdb_dir = os.path.join(args.pdb_base_dir, name)
    p1_path = os.path.join(pdb_dir, 'protein1.pdb')
    p2_path = os.path.join(pdb_dir, 'protein2.pdb')
    lig_path = os.path.join(pdb_dir, 'ligand.pdb')
    gt_complex_path = os.path.join(pdb_dir, 'gt_complex.pdb')

    trajectories_metrics = []
    global_t = 0
    remaining = num_trajectories_per_sample

    with tempfile.TemporaryDirectory() as tmp_dir:
        while remaining > 0:
            bsz = min(chunk, remaining)
            batch = collate_fn([item] * bsz)
            _move_batch_to_device(batch, device)

            chunk_seed = seed + sample_seed_off + global_t * 100003
            torch.manual_seed(chunk_seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(chunk_seed)

            with torch.no_grad():
                traj = model.sample(batch)
                final_sample = traj[-1]

            p2_transformed = batch['p2']['pos_heavyatom']
            p2_mask = batch['p2']['mask_heavyatom']
            p1_coords = batch['p1']['pos_heavyatom']
            p1_mask = batch['p1']['mask_heavyatom']

            for local_i in range(bsz):
                traj_idx = local_i
                gi = global_t + local_i

                lig_coords_pred = final_sample['lig_coords'][traj_idx]
                lig_coords_gt = batch['lig_coords_1'][traj_idx]
                mol_mask = batch['mol_mask'][traj_idx]

                trans_pred = final_sample['trans'][traj_idx].squeeze(0)
                trans_gt = batch['t_inv_1'][traj_idx]

                rot_pred = final_sample['rotmats'][traj_idx]
                rot_gt = batch['R_inv_1'][traj_idx]

                lig_seq_pred = final_sample['lig_seq'][traj_idx]
                lig_seq_gt = batch['lig_seq_1'][traj_idx]

                rmsd = compute_rmsd(
                    lig_coords_pred.unsqueeze(0),
                    lig_coords_gt.unsqueeze(0),
                    mol_mask.unsqueeze(0),
                )
                trans_error = compute_translation_error(
                    trans_pred.unsqueeze(0),
                    trans_gt.unsqueeze(0),
                )
                rot_error = compute_rotation_error(
                    rot_pred.unsqueeze(0),
                    rot_gt.unsqueeze(0),
                )
                seq_acc = compute_sequence_accuracy(
                    lig_seq_pred.unsqueeze(0),
                    lig_seq_gt.unsqueeze(0),
                    mol_mask.unsqueeze(0),
                    align_to_pocketxmol=True,
                    pocketxmol_atomic_numbers=PXM_ATOMIC_NUMBERS,
                )

                mol_mask_np = mol_mask.detach().cpu().numpy().astype(bool)
                lig_seq_pred_np = lig_seq_pred.detach().cpu().numpy()
                # Per-atom predicted type indices for valid ligand slots (same order as masked coords).
                lig_seq_pred_atoms = lig_seq_pred_np[mol_mask_np].astype(np.int64)

                metric_dict = {
                    'rmsd': rmsd.item(),
                    'trans_error': trans_error.item(),
                    'rot_error': rot_error.item(),
                    'seq_acc': seq_acc.item(),
                    'lig_coords_pred': lig_coords_pred.cpu().numpy(),
                    'lig_coords_gt': lig_coords_gt.cpu().numpy(),
                    'trans_pred': trans_pred.cpu().numpy(),
                    'trans_gt': trans_gt.cpu().numpy(),
                    'rot_pred': rot_pred.cpu().numpy(),
                    'rot_gt': rot_gt.cpu().numpy(),
                    'lig_seq_pred': lig_seq_pred.cpu().numpy(),
                    'lig_seq_gt': lig_seq_gt.cpu().numpy(),
                    'lig_seq_pred_atoms': lig_seq_pred_atoms,
                    'mol_mask': mol_mask.cpu().numpy(),
                }

                # Reconstruct ligand only when explicitly enabled (CPU-heavy path).
                if getattr(args, "use_reconstruct", False):
                    try:
                        valid_coords = lig_coords_pred.cpu().numpy()[mol_mask.cpu().numpy()]
                        atomic_nums = decode_lig_seq_to_atomic_nums(lig_seq_pred, mol_mask)
                        reconstructed_mol = reconstruct_from_generated(valid_coords, atomic_nums)
                        metric_dict.update({
                            'reconstruct_success': True,
                            'reconstruct_smiles': Chem.MolToSmiles(reconstructed_mol),
                            'reconstruct_error': '',
                        })
                    except (MolReconsError, ValueError, Exception) as e:
                        metric_dict.update({
                            'reconstruct_success': False,
                            'reconstruct_smiles': '',
                            'reconstruct_error': str(e),
                        })

                p2_transformed_traj = p2_transformed[traj_idx].cpu()
                p2_mask_traj = p2_mask[traj_idx].cpu()

                L2, A = p2_transformed_traj.shape[:2]
                p2_coords_flat = p2_transformed_traj.reshape(L2 * A, 3)

                rot_pred_cpu = rot_pred.cpu()
                trans_pred_cpu = trans_pred.cpu()
                trans_tensor = trans_pred_cpu.unsqueeze(0)

                p2_restored_flat = rotate_and_translate(
                    p2_coords_flat,
                    rot_pred_cpu,
                    trans_tensor,
                )
                p2_restored = p2_restored_flat.reshape(L2, A, 3).numpy()

                p1_pdb = PandasPdb().read_pdb(p1_path)
                p2_pdb = PandasPdb().read_pdb(p2_path)

                p1_coords_traj = p1_coords[traj_idx].cpu().numpy()
                p1_mask_traj = p1_mask[traj_idx].cpu().numpy()
                L1 = p1_coords_traj.shape[0]

                p1_coords_flat = p1_coords_traj.reshape(L1 * A, 3)
                p1_mask_flat = p1_mask_traj.reshape(L1 * A)
                p1_valid_coords = p1_coords_flat[p1_mask_flat]

                p1_pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = p1_valid_coords

                p2_coords_flat = p2_restored.reshape(L2 * A, 3)
                p2_mask_flat = p2_mask_traj.numpy().reshape(L2 * A)
                p2_valid_coords = p2_coords_flat[p2_mask_flat]

                p2_pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = p2_valid_coords

                p1_pred_file = os.path.join(tmp_dir, f'p1_pred_{gi}.pdb')
                p2_pred_file = os.path.join(tmp_dir, f'p2_pred_{gi}.pdb')
                lig_pred_path = os.path.join(tmp_dir, f'lig_pred_{gi}.pdb')

                p1_pdb.to_pdb(p1_pred_file, records=['ATOM'], gz=False)
                p2_pdb.to_pdb(p2_pred_file, records=['ATOM'], gz=False)

                valid_coords = lig_coords_pred.cpu().numpy()[mol_mask.cpu().numpy()]
                pred_lig = set_new_coords(lig_path, valid_coords, is_ligand=True)
                pred_lig = set_new_chain(pred_lig, 'A', is_ligand=True)
                write_pdb(pred_lig, lig_pred_path)

                complex_path = os.path.join(tmp_dir, f'complex_pred_{gi}.pdb')
                merge_pdbs([p1_pred_file, p2_pred_file, lig_pred_path], path=complex_path)

                try:
                    fnat, irms, Lrms, dockq = cal_dockq(complex_path, gt_complex_path)
                    metric_dict.update({
                        'fnat': float(fnat),
                        'irms': float(irms),
                        'Lrms': float(Lrms),
                        'dockq': float(dockq),
                    })
                except Exception as e:
                    if getattr(args, "log_worker_warnings", False):
                        print(f"Warning: DockQ calculation failed for {name} trajectory {gi}: {e}")
                    metric_dict.update({
                        'fnat': -1.0,
                        'irms': -1.0,
                        'Lrms': -1.0,
                        'dockq': -1.0,
                    })

                trajectories_metrics.append(metric_dict)

            global_t += bsz
            remaining -= bsz

    return trajectories_metrics


def _evaluate_subset(model, dataset, device, args, subset_indices, progress_queue=None, desc="Evaluating"):
    all_rmsd, all_trans_error, all_rot_error, all_seq_acc = [], [], [], []
    all_seq_acc_all_trajectories = []
    all_fnat, all_irms, all_Lrms, all_dockq = [], [], [], []
    sample_results = []

    for idx in tqdm(subset_indices, desc=desc, disable=(progress_queue is not None)):
        try:
            item = dataset[idx]
            trajectories_metrics = evaluate_sample(
                model, item, device, args.num_trajectories_per_sample, args, sample_idx=idx
            )

            best_idx = np.argmin([m['rmsd'] for m in trajectories_metrics])
            best_metrics = trajectories_metrics[best_idx]

            all_rmsd.append(best_metrics['rmsd'])
            all_trans_error.append(best_metrics['trans_error'])
            all_rot_error.append(best_metrics['rot_error'])
            all_seq_acc.append(best_metrics['seq_acc'])
            all_seq_acc_all_trajectories.extend([m['seq_acc'] for m in trajectories_metrics])

            if 'dockq' in best_metrics:
                all_fnat.append(best_metrics['fnat'])
                all_irms.append(best_metrics['irms'])
                all_Lrms.append(best_metrics['Lrms'])
                all_dockq.append(best_metrics['dockq'])

            sample_result = {
                'sample_idx': idx,
                'name': item.get('name', f'sample_{idx}'),
                'best_trajectory': best_metrics,
                'num_trajectories': len(trajectories_metrics),
                'all_trajectories_seq_acc': [m['seq_acc'] for m in trajectories_metrics],
                'all_trajectories_lig_coords_pred': [m['lig_coords_pred'] for m in trajectories_metrics],
            }

            if args.num_trajectories_per_sample > 1:
                sample_result['average_metrics'] = {
                    'rmsd': np.mean([m['rmsd'] for m in trajectories_metrics]),
                    'trans_error': np.mean([m['trans_error'] for m in trajectories_metrics]),
                    'rot_error': np.mean([m['rot_error'] for m in trajectories_metrics]),
                    'seq_acc': np.mean([m['seq_acc'] for m in trajectories_metrics]),
                }
                sample_result['std_metrics'] = {
                    'rmsd': np.std([m['rmsd'] for m in trajectories_metrics]),
                    'trans_error': np.std([m['trans_error'] for m in trajectories_metrics]),
                    'rot_error': np.std([m['rot_error'] for m in trajectories_metrics]),
                    'seq_acc': np.std([m['seq_acc'] for m in trajectories_metrics]),
                }
                sample_result['all_trajectories_rmsd'] = [m['rmsd'] for m in trajectories_metrics]
                sample_result['all_trajectories_lig_seq_pred_atoms'] = [
                    m['lig_seq_pred_atoms'] for m in trajectories_metrics
                ]
                if 'dockq' in trajectories_metrics[0]:
                    sample_result['all_trajectories_dockq'] = [m['dockq'] for m in trajectories_metrics]
                    sample_result['all_trajectories_fnat'] = [m['fnat'] for m in trajectories_metrics]
                    sample_result['all_trajectories_irms'] = [m['irms'] for m in trajectories_metrics]
                    sample_result['all_trajectories_Lrms'] = [m['Lrms'] for m in trajectories_metrics]

            sample_results.append(sample_result)
        except Exception as e:
            if getattr(args, "log_worker_warnings", False):
                print(f"\nWarning: Error evaluating sample {idx}: {e}")
                import traceback
                traceback.print_exc()
        finally:
            if progress_queue is not None:
                progress_queue.put(1)

    return {
        'all_rmsd': all_rmsd,
        'all_trans_error': all_trans_error,
        'all_rot_error': all_rot_error,
        'all_seq_acc': all_seq_acc,
        'all_seq_acc_all_trajectories': all_seq_acc_all_trajectories,
        'all_fnat': all_fnat,
        'all_irms': all_irms,
        'all_Lrms': all_Lrms,
        'all_dockq': all_dockq,
        'sample_results': sample_results,
    }


def _merge_partial_results(results_list):
    merged = {
        'all_rmsd': [],
        'all_trans_error': [],
        'all_rot_error': [],
        'all_seq_acc': [],
        'all_seq_acc_all_trajectories': [],
        'all_fnat': [],
        'all_irms': [],
        'all_Lrms': [],
        'all_dockq': [],
        'sample_results': [],
    }
    for result in results_list:
        merged['all_rmsd'].extend(result['all_rmsd'])
        merged['all_trans_error'].extend(result['all_trans_error'])
        merged['all_rot_error'].extend(result['all_rot_error'])
        merged['all_seq_acc'].extend(result['all_seq_acc'])
        merged['all_seq_acc_all_trajectories'].extend(result['all_seq_acc_all_trajectories'])
        merged['all_fnat'].extend(result['all_fnat'])
        merged['all_irms'].extend(result['all_irms'])
        merged['all_Lrms'].extend(result['all_Lrms'])
        merged['all_dockq'].extend(result['all_dockq'])
        merged['sample_results'].extend(result['sample_results'])
    merged['sample_results'].sort(key=lambda x: x['sample_idx'])
    return merged


def evaluate_dataset_single_gpu(args_tuple):
    gpu_id, subset_indices, dataset_path, args_dict, progress_queue = args_tuple

    class Args:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    args = Args(args_dict)
    args.device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model, _, device = load_model_and_config(args)
    dataset = load_from_disk(dataset_path)
    return _evaluate_subset(
        model, dataset, device, args, subset_indices, progress_queue=progress_queue, desc=f"GPU {gpu_id}"
    )


def evaluate_dataset_simple_multi_gpu(args):
    """
    Simple multi-GPU evaluation:
    - shard sample indices across GPU worker processes
    - each worker iterates samples one-by-one
    - trajectories per sample are parallelized by evaluate_sample() batch replication
    """
    if args.gpu_ids is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    elif torch.cuda.is_available():
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = []

    if len(gpu_ids) <= 1:
        raise ValueError("simple_multi_gpu requires at least 2 GPUs. Set --gpu_ids with multiple devices.")

    dataset = load_from_disk(args.dataset_path)
    if args.num_samples is not None and args.num_samples < len(dataset):
        import random
        random.seed(args.seed)
        indices = random.sample(range(len(dataset)), args.num_samples)
    else:
        indices = list(range(len(dataset)))

    print(f"\n[Simple Multi-GPU] Evaluating {len(indices)} samples on GPUs: {gpu_ids}")
    print(f"[Simple Multi-GPU] Trajectories per sample: {args.num_trajectories_per_sample}")
    if getattr(args, 'trajectory_batch_size', None) is not None:
        print(f"[Simple Multi-GPU] Trajectory batch size (per forward): {args.trajectory_batch_size}")

    args_dict = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'dataset_path': args.dataset_path,
        'pdb_base_dir': args.pdb_base_dir,
        'output_dir': args.output_dir,
        'num_trajectories_per_sample': args.num_trajectories_per_sample,
        'trajectory_batch_size': getattr(args, 'trajectory_batch_size', None),
        'use_reconstruct': getattr(args, 'use_reconstruct', False),
        'seed': args.seed,
        'verbose': False,
        'log_worker_warnings': False,
    }

    shards = [indices[i::len(gpu_ids)] for i in range(len(gpu_ids))]
    manager = mp.Manager()
    progress_queue = manager.Queue()

    tasks = []
    for worker_rank, gpu_id in enumerate(gpu_ids):
        if len(shards[worker_rank]) == 0:
            continue
        tasks.append((gpu_id, shards[worker_rank], args.dataset_path, args_dict, progress_queue))

    total_samples = sum(len(t[1]) for t in tasks)
    pbar = tqdm(total=total_samples, desc="Evaluating")

    with mp.Pool(processes=len(tasks)) as pool:
        async_results = [pool.apply_async(evaluate_dataset_single_gpu, (task,)) for task in tasks]
        completed = 0
        while completed < total_samples:
            progress_queue.get()
            completed += 1
            pbar.update(1)
        partial_results = [r.get() for r in async_results]

    pbar.close()
    merged = _merge_partial_results(partial_results)

    all_rmsd = merged['all_rmsd']
    all_trans_error = merged['all_trans_error']
    all_rot_error = merged['all_rot_error']
    all_seq_acc = merged['all_seq_acc']
    all_seq_acc_all_trajectories = merged['all_seq_acc_all_trajectories']
    all_fnat = merged['all_fnat']
    all_irms = merged['all_irms']
    all_Lrms = merged['all_Lrms']
    all_dockq = merged['all_dockq']
    sample_results = merged['sample_results']

    overall_metrics = {
        'rmsd': {
            'mean': np.mean(all_rmsd) if all_rmsd else 0.0,
            'std': np.std(all_rmsd) if all_rmsd else 0.0,
            'median': np.median(all_rmsd) if all_rmsd else 0.0,
        },
        'trans_error': {
            'mean': np.mean(all_trans_error) if all_trans_error else 0.0,
            'std': np.std(all_trans_error) if all_trans_error else 0.0,
            'median': np.median(all_trans_error) if all_trans_error else 0.0,
        },
        'rot_error': {
            'mean': np.mean(all_rot_error) if all_rot_error else 0.0,
            'std': np.std(all_rot_error) if all_rot_error else 0.0,
            'median': np.median(all_rot_error) if all_rot_error else 0.0,
        },
        'seq_acc': {
            'mean': np.mean(all_seq_acc) if all_seq_acc else 0.0,
            'std': np.std(all_seq_acc) if all_seq_acc else 0.0,
            'median': np.median(all_seq_acc) if all_seq_acc else 0.0,
        },
        'seq_acc_all_trajectories': {
            'mean': np.mean(all_seq_acc_all_trajectories) if all_seq_acc_all_trajectories else 0.0,
            'std': np.std(all_seq_acc_all_trajectories) if all_seq_acc_all_trajectories else 0.0,
            'median': np.median(all_seq_acc_all_trajectories) if all_seq_acc_all_trajectories else 0.0,
        },
    }
    if all_dockq:
        overall_metrics['fnat'] = {'mean': np.mean(all_fnat), 'std': np.std(all_fnat), 'median': np.median(all_fnat)}
        overall_metrics['irms'] = {'mean': np.mean(all_irms), 'std': np.std(all_irms), 'median': np.median(all_irms)}
        overall_metrics['Lrms'] = {'mean': np.mean(all_Lrms), 'std': np.std(all_Lrms), 'median': np.median(all_Lrms)}
        overall_metrics['dockq'] = {'mean': np.mean(all_dockq), 'std': np.std(all_dockq), 'median': np.median(all_dockq)}

    return {
        'overall_metrics': overall_metrics,
        'sample_results': sample_results,
        'num_evaluated': len(sample_results),
        'num_dockq_computed': len(all_dockq),
    }


def save_results(results, output_dir, args):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary metrics
    summary_path = os.path.join(output_dir, 'summary_metrics.json')
    summary = {
        'overall_metrics': results['overall_metrics'],
        'num_evaluated': results['num_evaluated'],
        'num_dockq_computed': results.get('num_dockq_computed', 0),
        'num_trajectories_per_sample': args.num_trajectories_per_sample,
        'trajectory_batch_size': getattr(args, 'trajectory_batch_size', None),
        'use_reconstruct': getattr(args, 'use_reconstruct', False),
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary metrics saved to {summary_path}")
    
    # Save detailed results
    detailed_path = os.path.join(output_dir, 'detailed_results.json')
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(detailed_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"✓ Detailed results saved to {detailed_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Number of samples evaluated: {results['num_evaluated']}")
    print(f"Trajectories per sample: {args.num_trajectories_per_sample}")
    print(f"Ligand reconstruct enabled: {getattr(args, 'use_reconstruct', False)}")
    if getattr(args, 'trajectory_batch_size', None) is not None:
        print(f"Trajectory batch size (per forward): {args.trajectory_batch_size}")
    if results.get('num_dockq_computed', 0) > 0:
        print(f"DockQ computed for: {results['num_dockq_computed']} samples")
    print("\nOverall Metrics (Best Trajectory):")
    print(f"  RMSD: {results['overall_metrics']['rmsd']['mean']:.4f} ± {results['overall_metrics']['rmsd']['std']:.4f} Å")
    print(f"  Translation Error: {results['overall_metrics']['trans_error']['mean']:.4f} ± {results['overall_metrics']['trans_error']['std']:.4f} Å")
    print(f"  Rotation Error: {results['overall_metrics']['rot_error']['mean']:.4f} ± {results['overall_metrics']['rot_error']['std']:.4f}")
    print(f"  Sequence Accuracy: {results['overall_metrics']['seq_acc']['mean']:.4f} ± {results['overall_metrics']['seq_acc']['std']:.4f} ({results['overall_metrics']['seq_acc']['mean']*100:.2f}%)")
    print("\nOverall Metrics (All Trajectories):")
    print(f"  Sequence Accuracy: {results['overall_metrics']['seq_acc_all_trajectories']['mean']:.4f} ± {results['overall_metrics']['seq_acc_all_trajectories']['std']:.4f} ({results['overall_metrics']['seq_acc_all_trajectories']['mean']*100:.2f}%)")
    
    if 'dockq' in results['overall_metrics']:
        print("\nDockQ Metrics:")
        print(f"  DockQ: {results['overall_metrics']['dockq']['mean']:.4f} ± {results['overall_metrics']['dockq']['std']:.4f}")
        print(f"  Fnat: {results['overall_metrics']['fnat']['mean']:.4f} ± {results['overall_metrics']['fnat']['std']:.4f}")
        print(f"  iRMS: {results['overall_metrics']['irms']['mean']:.4f} ± {results['overall_metrics']['irms']['std']:.4f} Å")
        print(f"  LRMS: {results['overall_metrics']['Lrms']['mean']:.4f} ± {results['overall_metrics']['Lrms']['std']:.4f} Å")
    
    print("="*80)


def main():
    args = parse_args()
    
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This must be done before creating any multiprocessing pools
    # CUDA cannot be re-initialized in forked subprocesses
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set, ignore
        pass
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    results = evaluate_dataset_simple_multi_gpu(args)
    
    # Save results
    save_results(results, args.output_dir, args)
    
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()