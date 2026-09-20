"""Training utility functions for flow matching model training"""

import os
import math
import torch
import numpy as np
from datasets import Dataset, load_from_disk
from utils.so3_utils import geodesic_dist
from configs.config import DictToObject
import yaml
from tqdm import tqdm
import random
from typing import List, Optional, Sequence
from datetime import datetime
import wandb
from biopandas.pdb import PandasPdb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, DistributedSampler

from utils.constants import MAP_ATOM_TYPE_FULL_TO_INDEX, PAD_RESIDUE_INDEX
from torch.utils.data._utils.collate import default_collate
from flow_model import TernaryFlowModel

DEFAULT_PAD_VALUES = {
    'aa': PAD_RESIDUE_INDEX, # 21
    'chain_id': ' ', 
    'icode': ' ',
}

_INDEX_TO_ATOMIC_NUM = {
    idx: atom_desc[0] for atom_desc, idx in MAP_ATOM_TYPE_FULL_TO_INDEX.items()
}


def move_to_device(value, device):
    """Recursively move tensors in a nested batch to ``device``."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value

def _pad_last(x, n, value=0):
    """Pad tensor or array to length n along the first dimension"""
    if isinstance(x, torch.Tensor):
        assert x.size(0) <= n
        if x.size(0) == n:
            return x
        pad_size = [n - x.size(0)] + list(x.shape[1:])
        pad = torch.full(pad_size, fill_value=value, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=0)
    elif isinstance(x, np.ndarray):
        x_tensor = torch.from_numpy(x)
        padded = _pad_last(x_tensor, n, value=value)
        return padded.numpy()
    elif isinstance(x, list):
        pad = [value] * (n - len(x))
        return x + pad
    else:
        return x

def _get_pad_mask(actual_len, padded_len):
    """Create a mask where True indicates valid positions and False indicates padding"""
    return torch.cat([
        torch.ones([actual_len], dtype=torch.bool),
        torch.zeros([padded_len - actual_len], dtype=torch.bool)
    ], dim=0)

def _get_pad_value(key, pad_values):
    """Get padding value for a given key"""
    if key not in pad_values:
        return 0
    return pad_values[key]

def _get_common_keys(list_of_dict):
    """Get common keys across all dictionaries"""
    keys = set(list_of_dict[0].keys())
    for d in list_of_dict[1:]:
        keys = keys.intersection(d.keys())
    return keys

def collate_fn(batch, eight=True):
    """
    Collate function for batching, inspired by PaddingCollate design.
    Handles data structure with p1/p2 dictionaries containing 'aa' and 'pos_heavyatom'.
    Automatically pads all fields in p1/p2 that have the same first dimension length as 'aa'.
    
    Args:
        batch: List of data items, each containing:
            - 'name': string
            - 'p1': dict with 'aa' (L1,) and other fields like 'pos_heavyatom' (L1, A, 3), 
                    'mask_heavyatom' (L1, A), 'chain_id', 'resseq', 'icode', etc.
            - 'p2': dict with 'aa' (L2,) and other fields
            - 'lig_seq': (L3,) array of atom types
            - 'lig_coords': (L3, 3) array of ligand coordinates
            - 'lig_bond_index': (2, E) directed chemical-bond indices (optional
              for backward compatibility with datasets built before bonds)
            - 'lig_bond_type': (E,) bond ids: 1=single, 2=double, 3=triple,
              4=aromatic, 5=other; 0 is reserved for no bond/padding
            - 'R_inv': (3, 3) rotation matrix
            - 't_inv': (3,) translation vector
            - 'interface_flag': bool
        eight: If True, pad to multiples of 8 (for efficient GPU operations)
    
    Returns:
        batched: Dictionary with padded p1, p2 dictionaries and other fields
    """
    
    # Extract lengths
    p1_lengths = [len(item['p1']['aa']) if isinstance(item['p1']['aa'], torch.Tensor) else len(item['p1']['aa']) for item in batch]
    p2_lengths = [len(item['p2']['aa']) if isinstance(item['p2']['aa'], torch.Tensor) else len(item['p2']['aa']) for item in batch]
    lig_lengths = [len(item['lig_seq']) if isinstance(item['lig_seq'], (torch.Tensor, np.ndarray)) else len(item['lig_seq']) for item in batch]
    
    max_p1_len = max(p1_lengths)
    max_p2_len = max(p2_lengths)
    max_lig_len = max(lig_lengths)
    
    # Pad to multiples of 8 if requested
    if eight:
        max_p1_len = math.ceil(max_p1_len / 8) * 8
        max_p2_len = math.ceil(max_p2_len / 8) * 8
        max_lig_len = math.ceil(max_lig_len / 8) * 8
    
    p1_keys = _get_common_keys([item['p1'] for item in batch])
    p2_keys = _get_common_keys([item['p2'] for item in batch])
    
    # Process p1: pad all fields that have the same length as 'aa'
    p1_padded_list = []
    for i, item in enumerate(batch):
        p1 = item['p1']
        p1_padded = {}
        
        for key in p1_keys:
            value = p1[key]
            
            # Check if this field has the same first dimension length as 'aa'
            if isinstance(value, torch.Tensor):
                first_dim = value.size(0)
            elif isinstance(value, (list, np.ndarray)):
                first_dim = len(value)
            else:
                # Scalar or other types - don't pad
                p1_padded[key] = value
                continue
            
            # Only pad if first dimension matches 'aa' length
            if first_dim == p1_lengths[i]:
                # Convert to tensor if needed
                if not isinstance(value, torch.Tensor):
                    if isinstance(value, np.ndarray):
                        value = torch.from_numpy(value)
                    elif isinstance(value, list):
                        # Handle list of strings (like chain_id, icode)
                        if len(value) > 0 and isinstance(value[0], str):
                            pad_val = _get_pad_value(key, DEFAULT_PAD_VALUES)
                            padded = _pad_last(value, max_p1_len, value=pad_val)
                            p1_padded[key] = padded
                            continue
                        else:
                            value = torch.tensor(value)
                    else:
                        value = torch.tensor(value)
                
                # Get padding value
                pad_val = _get_pad_value(key, DEFAULT_PAD_VALUES)
                if key == 'aa':
                    pad_val = DEFAULT_PAD_VALUES['aa']
                elif isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                    pad_val = 0.0
                
                padded = _pad_last(value, max_p1_len, value=pad_val)
                p1_padded[key] = padded
            else:
                # Different length - don't pad, just convert if needed
                if not isinstance(value, torch.Tensor) and isinstance(value, np.ndarray):
                    p1_padded[key] = torch.from_numpy(value)
                else:
                    p1_padded[key] = value
        
        # Add mask
        p1_padded['res_mask'] = _get_pad_mask(p1_lengths[i], max_p1_len)
        p1_padded_list.append(p1_padded)
    
    # Process p2: same as p1
    p2_padded_list = []
    for i, item in enumerate(batch):
        p2 = item['p2']
        p2_padded = {}
        
        for key in p2_keys:
            value = p2[key]
            
            # Check if this field has the same first dimension length as 'aa'
            if isinstance(value, torch.Tensor):
                first_dim = value.size(0)
            elif isinstance(value, (list, np.ndarray)):
                first_dim = len(value)
            else:
                p2_padded[key] = value
                continue
            
            # Only pad if first dimension matches 'aa' length
            if first_dim == p2_lengths[i]:
                # Convert to tensor if needed
                if not isinstance(value, torch.Tensor):
                    if isinstance(value, np.ndarray):
                        value = torch.from_numpy(value)
                    elif isinstance(value, list):
                        if len(value) > 0 and isinstance(value[0], str):
                            pad_val = _get_pad_value(key, DEFAULT_PAD_VALUES)
                            padded = _pad_last(value, max_p2_len, value=pad_val)
                            p2_padded[key] = padded
                            continue
                        else:
                            value = torch.tensor(value)
                    else:
                        value = torch.tensor(value)
                
                # Get padding value
                pad_val = _get_pad_value(key, DEFAULT_PAD_VALUES)
                if key == 'aa':
                    pad_val = DEFAULT_PAD_VALUES['aa']
                elif isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                    pad_val = 0.0
                
                padded = _pad_last(value, max_p2_len, value=pad_val)
                p2_padded[key] = padded
            else:
                if not isinstance(value, torch.Tensor) and isinstance(value, np.ndarray):
                    p2_padded[key] = torch.from_numpy(value)
                else:
                    p2_padded[key] = value
        
        # Add mask
        p2_padded['res_mask'] = _get_pad_mask(p2_lengths[i], max_p2_len)
        p2_padded_list.append(p2_padded)
    
    # Use default_collate to stack p1 and p2 (like PaddingCollate does)
    batched = {}
    batched['p1'] = default_collate(p1_padded_list)
    batched['p2'] = default_collate(p2_padded_list)
    
    # Handle ligand sequence
    lig_seq_list = []
    for item in batch:
        lig_seq = item['lig_seq']
        if not isinstance(lig_seq, torch.Tensor):
            lig_seq = torch.tensor(lig_seq, dtype=torch.long)
        lig_seq_padded = _pad_last(lig_seq, max_lig_len, value=0)
        lig_seq_list.append(lig_seq_padded)
    batched['lig_seq'] = torch.stack(lig_seq_list)
    
    # Pad ligand coordinates (same for both structures)
    lig_coords_list = []
    lig_coords_gt_list = []
    mol_mask_list = []
    for i, item in enumerate(batch):
        lig_coords = item['lig_coords']
        # lig_coords_gt = item['lig_coords_gt']
        if not isinstance(lig_coords, torch.Tensor):
            lig_coords = torch.tensor(lig_coords, dtype=torch.float32)
        # if not isinstance(lig_coords_gt, torch.Tensor):
        #     lig_coords_gt = torch.tensor(lig_coords_gt, dtype=torch.float32)
        
        lig_coords_padded = _pad_last(lig_coords, max_lig_len, value=0.0)
        # lig_coords_gt_padded = _pad_last(lig_coords_gt, max_lig_len, value=0.0)
        mol_mask = _get_pad_mask(lig_lengths[i], max_lig_len)
        
        lig_coords_list.append(lig_coords_padded)
        # lig_coords_gt_list.append(lig_coords_gt_padded)
        mol_mask_list.append(mol_mask)
    
    batched['lig_coords'] = torch.stack(lig_coords_list)
    # batched['lig_coords_gt'] = torch.stack(lig_coords_gt_list)
    batched['mol_mask'] = torch.stack(mol_mask_list)

    # Convert the variable-size sparse ligand graph into a dense padded bond
    # matrix. This is directly usable as an EGNN edge feature after embedding.
    # Old datasets without bond fields remain loadable and produce all-zero
    # matrices, with lig_bond_available=False making that state explicit.
    lig_bond_type_matrix = torch.zeros(
        (len(batch), max_lig_len, max_lig_len), dtype=torch.long
    )
    lig_bond_available = torch.zeros(len(batch), dtype=torch.bool)
    for i, item in enumerate(batch):
        has_index = 'lig_bond_index' in item
        has_type = 'lig_bond_type' in item
        if has_index != has_type:
            raise ValueError(
                f"Sample {item.get('name', i)!r} has only one of "
                "lig_bond_index and lig_bond_type."
            )
        if not has_index:
            continue

        edge_index = torch.as_tensor(item['lig_bond_index'], dtype=torch.long)
        edge_type = torch.as_tensor(item['lig_bond_type'], dtype=torch.long)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"Sample {item.get('name', i)!r} has invalid lig_bond_index "
                f"shape {tuple(edge_index.shape)}; expected (2, E)."
            )
        if edge_type.ndim != 1 or edge_type.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"Sample {item.get('name', i)!r} has incompatible bond "
                f"shapes: index={tuple(edge_index.shape)}, "
                f"type={tuple(edge_type.shape)}."
            )
        if edge_index.numel() > 0:
            if edge_index.min().item() < 0 or edge_index.max().item() >= lig_lengths[i]:
                raise ValueError(
                    f"Sample {item.get('name', i)!r} contains a ligand bond "
                    f"index outside [0, {lig_lengths[i]})."
                )
            if edge_type.min().item() < 1 or edge_type.max().item() > 5:
                raise ValueError(
                    f"Sample {item.get('name', i)!r} contains a ligand bond "
                    "type outside the supported range [1, 5]."
                )
            lig_bond_type_matrix[
                i, edge_index[0], edge_index[1]
            ] = edge_type
        lig_bond_available[i] = True

    batched['lig_bond_type_matrix'] = lig_bond_type_matrix
    batched['lig_bond_mask'] = lig_bond_type_matrix.ne(0)
    batched['lig_bond_available'] = lig_bond_available
    
    # Handle R_inv and t_inv
    batched['R_inv'] = torch.stack([torch.tensor(item['R_inv'], dtype=torch.float32) if not isinstance(item['R_inv'], torch.Tensor) else item['R_inv'].float() for item in batch])
    batched['t_inv'] = torch.stack([torch.tensor(item['t_inv'], dtype=torch.float32) if not isinstance(item['t_inv'], torch.Tensor) else item['t_inv'].float() for item in batch])
    
    # Handle interface_flag
    batched['interface_flag'] = torch.stack([
        torch.tensor(item['interface_flag'], dtype=torch.bool) if not isinstance(item['interface_flag'], torch.Tensor) else item['interface_flag'].bool()
        for item in batch
    ])
    
    # Add ground truth data for loss calculation (detach to save memory)
    batched['lig_seq_1'] = batched['lig_seq'].detach().clone()
    batched['lig_coords_1'] = batched['lig_coords'].detach().clone()
    batched['lig_bond_type_matrix_1'] = (
        batched['lig_bond_type_matrix'].detach().clone()
    )
    batched['R_inv_1'] = batched['R_inv'].detach().clone()
    batched['t_inv_1'] = batched['t_inv'].detach().clone()
    
    return batched


def load_config_from_yaml(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to DictToObject recursively
    def dict_to_object(d):
        if isinstance(d, dict):
            return DictToObject({k: dict_to_object(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_object(item) for item in d]
        else:
            return d
    
    return dict_to_object(config_dict)


def apply_args_to_config(config, args, is_main_process=True):
    """Apply command line arguments to config"""
    if is_main_process:
        overrides = []
    
    # Dataset overrides
    if args.dataset is not None:
        config.dataset.path = args.dataset
        if is_main_process:
            overrides.append(f"dataset.path = {args.dataset}")
    
    if args.train_split is not None:
        config.dataset.train_split = args.train_split
        config.dataset.val_split = 1.0 - args.train_split
        if is_main_process:
            overrides.append(f"train split = {args.train_split}")
    
    # Training parameter overrides
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
        if is_main_process:
            overrides.append(f"batch_size = {args.batch_size}")
    
    if args.lr is not None:
        config.train.optimizer.lr = args.lr
        if is_main_process:
            overrides.append(f"learning_rate = {args.lr}")
    
    if args.max_epochs is not None:
        config.train.max_epochs = args.max_epochs
        if is_main_process:
            overrides.append(f"max_epochs = {args.max_epochs}")
    
    if args.val_freq is not None:
        config.train.val_freq = args.val_freq
        if is_main_process:
            overrides.append(f"val_freq = {args.val_freq}")
    
    if hasattr(args, 'save_freq') and args.save_freq is not None:
        config.train.save_freq = args.save_freq
        if is_main_process:
            overrides.append(f"save_freq = {args.save_freq}")
    
    if args.seed is not None:
        config.train.seed = args.seed
        if is_main_process:
            overrides.append(f"seed = {args.seed}")
    
    if is_main_process and overrides:
        print("\n Config overrides:")
        for override in overrides:
            print(f"   - {override}")
        print()
    
    return config


def check_for_nan(value, name, global_step, checkpoint_dir=None, save_debug=False, batch=None, model=None):
    """Check if a tensor or scalar value contains NaN or Inf, and optionally save debug info"""
    if isinstance(value, torch.Tensor):
        has_nan = torch.isnan(value).any().item()
        has_inf = torch.isinf(value).any().item()
        value_item = value.item() if value.numel() == 1 else None
    else:
        has_nan = (value != value) or np.isnan(value)  # NaN check for scalars
        has_inf = np.isinf(value) if isinstance(value, (float, np.number)) else False
        value_item = value
    
    if has_nan or has_inf:
        status = []
        if has_nan:
            status.append("NaN")
        if has_inf:
            status.append("Inf")
        
        print(f"\n{'='*80}")
        print(f"⚠️  WARNING: {name} contains {', '.join(status)} at global_step {global_step}")
        print(f"{'='*80}")
        
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                print(f"  Value: {value.item()}")
            else:
                print(f"  Shape: {value.shape}")
                print(f"  NaN count: {torch.isnan(value).sum().item()}")
                print(f"  Inf count: {torch.isinf(value).sum().item()}")
                print(f"  Min: {value[~torch.isnan(value) & ~torch.isinf(value)].min().item() if (~torch.isnan(value) & ~torch.isinf(value)).any() else 'N/A'}")
                print(f"  Max: {value[~torch.isnan(value) & ~torch.isinf(value)].max().item() if (~torch.isnan(value) & ~torch.isinf(value)).any() else 'N/A'}")
        else:
            print(f"  Value: {value_item}")
        
        if save_debug and checkpoint_dir is not None:
            debug_path = os.path.join(checkpoint_dir, f'nan_debug_step_{global_step}.pt')
            debug_data = {
                'global_step': global_step,
                'name': name,
                'value': value,
                'has_nan': has_nan,
                'has_inf': has_inf,
            }
            if batch is not None:
                debug_data['batch'] = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
            if model is not None:
                # Save model state (unwrapped if DDP)
                actual_model = model.module if hasattr(model, 'module') else model
                debug_data['model_state'] = actual_model.state_dict()
            
            torch.save(debug_data, debug_path)
            print(f"Debug info saved to: {debug_path}")
        
        print(f"{'='*80}\n")
        
        return True

    return False


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss_dict, checkpoint_dir, is_ddp=False, scaler=None, filename=None):
    """Save training checkpoint
    
    Args:
        filename: Optional filename. If None, saves as checkpoint_epoch_{epoch}.pt
                  If provided, saves with that filename instead of epoch-based name.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Handle DDP model
    if is_ddp:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss_dict': loss_dict,
    }
    
    # Save scaler state if using mixed precision
    if scaler is not None:
        checkpoint['scaler_state'] = scaler.state_dict()
    
    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'latest.pt')
    torch.save(checkpoint, latest_path)
    
    # Save checkpoint with specified filename or default epoch-based name
    if filename is not None:
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
    else:
        # Save epoch checkpoint
        epoch_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, epoch_path)
        print(f"✓ Checkpoint saved: {epoch_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load training checkpoint
    
    Returns:
        dict: Dictionary containing 'epoch', 'global_step', and optionally 'scaler_state'
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return {'epoch': 0, 'global_step': 0}
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Checkpoints store the unwrapped module state, while resumed multi-GPU
    # training passes a DistributedDataParallel wrapper here.
    model_to_load = model.module if isinstance(model, DDP) else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    result = {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'loss_dict': checkpoint.get('loss_dict'),
    }
    
    # Return scaler state if available
    if 'scaler_state' in checkpoint:
        result['scaler_state'] = checkpoint['scaler_state']
    
    print(f"✓ Checkpoint loaded: epoch={result['epoch']}, global_step={result['global_step']}")
    
    return result


def create_optimizer(model, config):
    """Create optimizer based on config"""
    # Exclude frozen interface-encoder tensors. This also makes optimizer
    # checkpoints smaller when only the virtual-keypoint heads are fine-tuned.
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if config.optimizer.type == 'adam':
        optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=(config.optimizer.beta1, config.optimizer.beta2)
        )
    elif config.optimizer.type == 'adamw':
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer.type}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler based on config"""
    if config.scheduler.type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler.factor,
            patience=config.scheduler.patience,
            min_lr=config.scheduler.min_lr,
            verbose=True
        )
    elif config.scheduler.type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.max_epochs
        )
    elif config.scheduler.type == 'linear':
        # Linear decay from initial_lr to end_lr over max_epochs
        # Use LambdaLR to implement linear decay per epoch
        initial_lr = config.optimizer.lr
        end_lr = getattr(config.scheduler, 'end_lr', 0.0)
        total_epochs = config.max_epochs
        
        def lr_lambda(epoch):
            # LambdaLR: lr = initial_lr * lr_lambda(epoch)
            # We want: lr(0) = initial_lr, lr(total_epochs) = end_lr
            # Linear interpolation: lr = initial_lr + (end_lr - initial_lr) * (epoch / total_epochs)
            # Factor: lr / initial_lr = 1 + (end_lr/initial_lr - 1) * (epoch / total_epochs)
            if total_epochs == 0 or initial_lr == 0:
                return 1.0
            if epoch >= total_epochs:
                return end_lr / initial_lr
            progress = epoch / total_epochs
            # Factor should go from 1.0 (at epoch=0) to end_lr/initial_lr (at epoch=total_epochs)
            factor = 1.0 + (end_lr / initial_lr - 1.0) * progress
            return max(factor, end_lr / initial_lr)  # Ensure we don't go below end_lr
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif config.scheduler.type == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler.type}")
    
    return scheduler

"""
# Original validate function (commented out)
def validate_original(model, val_dataset, device, config, is_main_process=True, is_ddp=False):
    '''
    Validate the model by sampling and comparing with ground truth.

    Args:
        model: The model to validate (may be wrapped in DDP)
        val_dataset: Validation dataset (not DataLoader)
        device: Device to run on
        config: Configuration object
        is_main_process: Whether this is the main process (for distributed training)
        is_ddp: Whether model is wrapped in DistributedDataParallel

    Returns:
        metrics_dict: Dictionary containing validation metrics
    '''
    model.eval()

    # Get the actual model (unwrap DDP if needed)
    actual_model = model.module if is_ddp else model

    # Number of samples to validate on
    num_val_samples = getattr(config.train, 'num_val_samples', 50)
    num_val_samples = min(num_val_samples, len(val_dataset))

    # Number of trajectories to sample per sample
    num_trajectories_per_sample = getattr(config.train, 'num_trajectories_per_sample', 1)

    # Randomly select samples from validation set
    val_indices = random.sample(range(len(val_dataset)), num_val_samples)

    # Initialize metrics accumulators
    all_rmsd = []
    all_trans_error = []
    all_rot_error = []
    all_seq_acc = []

    if is_main_process:
        print(f"Validating on {num_val_samples} samples...")
        if num_trajectories_per_sample > 1:
            print(f"Sampling {num_trajectories_per_sample} trajectories per sample")

    with torch.no_grad():
        for sample_idx, idx in enumerate(tqdm(val_indices, desc="Validating", disable=not is_main_process)):
            try:
                # Get sample from dataset
                item = val_dataset[idx]

                # Create batch by replicating the same item multiple times
                batch_items = [item] * num_trajectories_per_sample
                batch = collate_fn(batch_items)

                batch = move_to_device(batch, device)

                # Sample from model (use actual_model to handle DDP wrapping)
                traj = actual_model.sample(batch)
                final_sample = traj[-1]  # Get final sample

                # Process all trajectories in the batch
                batch_size = batch['lig_coords_1'].shape[0]
                sample_rmsds = []
                sample_trans_errors = []
                sample_rot_errors = []
                sample_seq_accs = []

                for traj_idx in range(batch_size):
                    # Extract predictions and ground truth
                    lig_coords_pred = final_sample['lig_coords'][traj_idx]  # (L, 3)
                    lig_coords_gt = batch['lig_coords_1'][traj_idx]  # (L, 3)
                    mol_mask = batch['mol_mask'][traj_idx]  # (L,)

                    trans_pred = final_sample['trans'][traj_idx].squeeze(0)  # (3,)
                    trans_gt = batch['t_inv_1'][traj_idx]  # (3,)

                    rot_pred = final_sample['rotmats'][traj_idx]  # (3, 3)
                    rot_gt = batch['R_inv_1'][traj_idx]  # (3, 3)

                    lig_seq_pred = final_sample['lig_seq'][traj_idx]  # (L,)
                    lig_seq_gt = batch['lig_seq_1'][traj_idx]  # (L,)

                    # Compute metrics for this trajectory
                    rmsd = compute_rmsd(lig_coords_pred.unsqueeze(0), lig_coords_gt.unsqueeze(0), mol_mask.unsqueeze(0))
                    trans_error = compute_translation_error(trans_pred.unsqueeze(0), trans_gt.unsqueeze(0))
                    rot_error = compute_rotation_error(rot_pred.unsqueeze(0), rot_gt.unsqueeze(0))
                    seq_acc = compute_sequence_accuracy(lig_seq_pred.unsqueeze(0), lig_seq_gt.unsqueeze(0), mol_mask.unsqueeze(0))

                    sample_rmsds.append(rmsd.item())
                    sample_trans_errors.append(trans_error.item())
                    sample_rot_errors.append(rot_error.item())
                    sample_seq_accs.append(seq_acc.item())

                # Use the best trajectory (lowest RMSD) for reporting
                best_idx = np.argmin(sample_rmsds)
                best_rmsd = sample_rmsds[best_idx]
                best_trans_error = sample_trans_errors[best_idx]
                best_rot_error = sample_rot_errors[best_idx]
                best_seq_acc = sample_seq_accs[best_idx]

                # Store metrics (using best trajectory)
                all_rmsd.append(best_rmsd)
                all_trans_error.append(best_trans_error)
                all_rot_error.append(best_rot_error)
                all_seq_acc.append(best_seq_acc)

                # Extract best trajectory predictions for display
                lig_coords_pred = final_sample['lig_coords'][best_idx]
                trans_pred = final_sample['trans'][best_idx].squeeze(0)
                rot_pred = final_sample['rotmats'][best_idx]
                lig_seq_pred = final_sample['lig_seq'][best_idx]
                mol_mask = batch['mol_mask'][best_idx]
                lig_coords_gt = batch['lig_coords_1'][best_idx]
                trans_gt = batch['t_inv_1'][best_idx]
                rot_gt = batch['R_inv_1'][best_idx]
                lig_seq_gt = batch['lig_seq_1'][best_idx]

                # Print comparison for this sample
                if is_main_process:
                    print(f"\n{'='*80}")
                    print(f"Sample {sample_idx + 1}/{num_val_samples} (Dataset index: {idx})")
                    if num_trajectories_per_sample > 1:
                        print(f"Sampled {num_trajectories_per_sample} trajectories")
                    print(f"{'='*80}")
                    
                    if num_trajectories_per_sample > 1:
                        print(f"\nBest Trajectory Metrics (out of {num_trajectories_per_sample}):")
                        print(f"  RMSD: {best_rmsd:.4f} Å")
                        print(f"  Translation error: {best_trans_error:.4f} Å")
                        print(f"  Rotation error: {best_rot_error:.4f}")
                        print(f"  Sequence accuracy: {best_seq_acc:.4f} ({best_seq_acc*100:.2f}%)")
                        
                        print(f"\nAverage Metrics across {num_trajectories_per_sample} trajectories:")
                        print(f"  RMSD: {np.mean(sample_rmsds):.4f} Å (std: {np.std(sample_rmsds):.4f})")
                        print(f"  Translation error: {np.mean(sample_trans_errors):.4f} Å (std: {np.std(sample_trans_errors):.4f})")
                        print(f"  Rotation error: {np.mean(sample_rot_errors):.4f} (std: {np.std(sample_rot_errors):.4f})")
                        print(f"  Sequence accuracy: {np.mean(sample_seq_accs):.4f} ({np.mean(sample_seq_accs)*100:.2f}%) (std: {np.std(sample_seq_accs):.4f})")
                        
                        rmsd_strs = [f'{r:.4f}' for r in sample_rmsds]
                        print(f"\nAll Trajectory RMSDs: {rmsd_strs}")
                    else:
                        print(f"\nMetrics:")
                        print(f"  RMSD: {best_rmsd:.4f} Å")
                        print(f"  Translation error: {best_trans_error:.4f} Å")
                        print(f"  Rotation error: {best_rot_error:.4f}")
                        print(f"  Sequence accuracy: {best_seq_acc:.4f} ({best_seq_acc*100:.2f}%)")

                    print(f"\nSequence Comparison (Best Trajectory):")
                    print(f"  Predicted: {format_sequence(lig_seq_pred, mol_mask)}")
                    print(f"  Ground Truth: {format_sequence(lig_seq_gt, mol_mask)}")

                    print(f"\nTranslation Comparison (Best Trajectory):")
                    print(f"  Predicted: {trans_pred}")
                    print(f"  Ground Truth: {trans_gt}")
                    print(f"  Error: {best_trans_error:.4f} Å")

                    print(f"\nRotation Matrix Comparison (Best Trajectory):")
                    print(f"  Predicted:\n{format_rotation(rot_pred)}")
                    print(f"  Ground Truth:\n{format_rotation(rot_gt)}")
                    print(f"  Rotation error: {best_rot_error:.4f}")

                    print(f"\nCoordinates Comparison (Best Trajectory, first few atoms):")
                    print(f"  Predicted:\n{format_coords(lig_coords_pred, mol_mask)}")
                    print(f"  Ground Truth:\n{format_coords(lig_coords_gt, mol_mask)}")
                    print(f"  RMSD: {best_rmsd:.4f} Å")
                    print(f"{'='*80}\n")

            except Exception as e:
                if is_main_process:
                    print(f"Warning: Error validating sample {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                continue

    # Compute average metrics
    metrics = {
        'rmsd': np.mean(all_rmsd) if all_rmsd else 0.0,
        'trans_error': np.mean(all_trans_error) if all_trans_error else 0.0,
        'rot_error': np.mean(all_rot_error) if all_rot_error else 0.0,
        'seq_acc': np.mean(all_seq_acc) if all_seq_acc else 0.0,
    }
    
    return metrics
"""

def validate(model, val_dataset, device, config, is_main_process=True, is_ddp=False):
    """
    Validate the model by sampling and comparing with ground truth.
    Supports sampling multiple trajectories per sample by replicating the item in batch.

    Args:
        model: The model to validate (may be wrapped in DDP)
        val_dataset: Validation dataset (not DataLoader)
        device: Device to run on
        config: Configuration object
        is_main_process: Whether this is the main process (for distributed training)
        is_ddp: Whether model is wrapped in DistributedDataParallel

    Returns:
        metrics_dict: Dictionary containing validation metrics
    """
    model.eval()

    # Get the actual model (unwrap DDP if needed)
    actual_model = model.module if is_ddp else model

    # Number of samples to validate on
    num_val_samples = getattr(config.train, 'num_val_samples', 50)
    num_val_samples = min(num_val_samples, len(val_dataset))

    # Number of trajectories to sample per sample
    num_trajectories_per_sample = getattr(config.train, 'num_trajectories_per_sample', 1)

    # Randomly select samples from validation set
    val_indices = random.sample(range(len(val_dataset)), num_val_samples)

    # Initialize metrics accumulators
    all_rmsd = []
    all_trans_error = []
    all_rot_error = []
    all_seq_acc = []

    if is_main_process:
        print(f"Validating on {num_val_samples} samples...")
        if num_trajectories_per_sample > 1:
            print(f"Sampling {num_trajectories_per_sample} trajectories per sample")

    with torch.no_grad():
        for sample_idx, idx in enumerate(tqdm(val_indices, desc="Validating", disable=not is_main_process)):
            try:
                # Get sample from dataset
                item = val_dataset[idx]

                # Create batch by replicating the same item multiple times
                batch_items = [item] * num_trajectories_per_sample
                batch = collate_fn(batch_items)

                batch = move_to_device(batch, device)

                # Sample from model (use actual_model to handle DDP wrapping)
                traj = actual_model.sample(batch)
                final_sample = traj[-1]  # Get final sample

                # Process all trajectories in the batch
                batch_size = batch['lig_coords_1'].shape[0]
                sample_rmsds = []
                sample_trans_errors = []
                sample_rot_errors = []
                sample_seq_accs = []

                for traj_idx in range(batch_size):
                    # Extract predictions and ground truth
                    lig_coords_pred = final_sample['lig_coords'][traj_idx]  # (L, 3)
                    lig_coords_gt = batch['lig_coords_1'][traj_idx]  # (L, 3)
                    mol_mask = batch['mol_mask'][traj_idx]  # (L,)

                    trans_pred = final_sample['trans'][traj_idx].squeeze(0)  # (3,)
                    trans_gt = batch['t_inv_1'][traj_idx]  # (3,)

                    rot_pred = final_sample['rotmats'][traj_idx]  # (3, 3)
                    rot_gt = batch['R_inv_1'][traj_idx]  # (3, 3)

                    lig_seq_pred = final_sample['lig_seq'][traj_idx]  # (L,)
                    lig_seq_gt = batch['lig_seq_1'][traj_idx]  # (L,)

                    # Compute metrics for this trajectory
                    rmsd = compute_rmsd(lig_coords_pred.unsqueeze(0), lig_coords_gt.unsqueeze(0), mol_mask.unsqueeze(0))
                    trans_error = compute_translation_error(trans_pred.unsqueeze(0), trans_gt.unsqueeze(0))
                    rot_error = compute_rotation_error(rot_pred.unsqueeze(0), rot_gt.unsqueeze(0))
                    seq_acc = compute_sequence_accuracy(lig_seq_pred.unsqueeze(0), lig_seq_gt.unsqueeze(0), mol_mask.unsqueeze(0))

                    sample_rmsds.append(rmsd.item())
                    sample_trans_errors.append(trans_error.item())
                    sample_rot_errors.append(rot_error.item())
                    sample_seq_accs.append(seq_acc.item())

                # Use the best trajectory (lowest RMSD) for reporting
                best_idx = np.argmin(sample_rmsds)
                best_rmsd = sample_rmsds[best_idx]
                best_trans_error = sample_trans_errors[best_idx]
                best_rot_error = sample_rot_errors[best_idx]
                best_seq_acc = sample_seq_accs[best_idx]

                # Store metrics (using best trajectory)
                all_rmsd.append(best_rmsd)
                all_trans_error.append(best_trans_error)
                all_rot_error.append(best_rot_error)
                all_seq_acc.append(best_seq_acc)

                # Extract best trajectory predictions for display
                lig_coords_pred = final_sample['lig_coords'][best_idx]
                trans_pred = final_sample['trans'][best_idx].squeeze(0)
                rot_pred = final_sample['rotmats'][best_idx]
                lig_seq_pred = final_sample['lig_seq'][best_idx]
                mol_mask = batch['mol_mask'][best_idx]
                lig_coords_gt = batch['lig_coords_1'][best_idx]
                trans_gt = batch['t_inv_1'][best_idx]
                rot_gt = batch['R_inv_1'][best_idx]
                lig_seq_gt = batch['lig_seq_1'][best_idx]

                # Print comparison for this sample
                if is_main_process:
                    print(f"\n{'='*80}")
                    print(f"Sample {sample_idx + 1}/{num_val_samples} (Dataset index: {idx})")
                    if num_trajectories_per_sample > 1:
                        print(f"Sampled {num_trajectories_per_sample} trajectories")
                    print(f"{'='*80}")
                    
                    if num_trajectories_per_sample > 1:
                        print(f"\nBest Trajectory Metrics (out of {num_trajectories_per_sample}):")
                        print(f"  RMSD: {best_rmsd:.4f} Å")
                        print(f"  Translation error: {best_trans_error:.4f} Å")
                        print(f"  Rotation error: {best_rot_error:.4f}")
                        print(f"  Sequence accuracy: {best_seq_acc:.4f} ({best_seq_acc*100:.2f}%)")
                        
                        print(f"\nAverage Metrics across {num_trajectories_per_sample} trajectories:")
                        print(f"  RMSD: {np.mean(sample_rmsds):.4f} Å (std: {np.std(sample_rmsds):.4f})")
                        print(f"  Translation error: {np.mean(sample_trans_errors):.4f} Å (std: {np.std(sample_trans_errors):.4f})")
                        print(f"  Rotation error: {np.mean(sample_rot_errors):.4f} (std: {np.std(sample_rot_errors):.4f})")
                        print(f"  Sequence accuracy: {np.mean(sample_seq_accs):.4f} ({np.mean(sample_seq_accs)*100:.2f}%) (std: {np.std(sample_seq_accs):.4f})")
                        
                        rmsd_strs = [f'{r:.4f}' for r in sample_rmsds]
                        print(f"\nAll Trajectory RMSDs: {rmsd_strs}")
                    else:
                        print(f"\nMetrics:")
                        print(f"  RMSD: {best_rmsd:.4f} Å")
                        print(f"  Translation error: {best_trans_error:.4f} Å")
                        print(f"  Rotation error: {best_rot_error:.4f}")
                        print(f"  Sequence accuracy: {best_seq_acc:.4f} ({best_seq_acc*100:.2f}%)")

                    print(f"\nSequence Comparison (Best Trajectory):")
                    print(f"  Predicted: {format_sequence(lig_seq_pred, mol_mask)}")
                    print(f"  Ground Truth: {format_sequence(lig_seq_gt, mol_mask)}")

                    print(f"\nTranslation Comparison (Best Trajectory):")
                    print(f"  Predicted: {trans_pred}")
                    print(f"  Ground Truth: {trans_gt}")
                    print(f"  Error: {best_trans_error:.4f} Å")

                    print(f"\nRotation Matrix Comparison (Best Trajectory):")
                    print(f"  Predicted:\n{format_rotation(rot_pred)}")
                    print(f"  Ground Truth:\n{format_rotation(rot_gt)}")
                    print(f"  Rotation error: {best_rot_error:.4f}")

                    print(f"\nCoordinates Comparison (Best Trajectory, first few atoms):")
                    print(f"  Predicted:\n{format_coords(lig_coords_pred, mol_mask)}")
                    print(f"  Ground Truth:\n{format_coords(lig_coords_gt, mol_mask)}")
                    print(f"  RMSD: {best_rmsd:.4f} Å")
                    print(f"{'='*80}\n")

            except Exception as e:
                if is_main_process:
                    print(f"Warning: Error validating sample {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                continue

    # Compute average metrics
    metrics = {
        'rmsd': np.mean(all_rmsd) if all_rmsd else 0.0,
        'trans_error': np.mean(all_trans_error) if all_trans_error else 0.0,
        'rot_error': np.mean(all_rot_error) if all_rot_error else 0.0,
        'seq_acc': np.mean(all_seq_acc) if all_seq_acc else 0.0,
    }
    
    return metrics


def setup_ddp_environment():
    """Initialize the distributed training environment."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        is_main_process = (rank == 0)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_main_process = True
        
    return device, rank, local_rank, world_size, is_main_process

def setup_wandb_logging(config, args, world_size, is_main_process):
    """Initialize WandB logging (only on the main process)."""
    if is_main_process and not args.no_wandb:
        timestamp = datetime.now().strftime("%m%d")
        wandb.init(
            project=args.wandb_project or "ternary_flow",
            name=args.wandb_name or f"{timestamp}",
            config={
                'batch_size': config.train.batch_size,
                'lr': config.train.optimizer.lr,
                'max_epochs': config.train.max_epochs,
                'world_size': world_size,
                'use_amp': args.use_amp or getattr(config.train, 'use_amp', False),
                # Add other critical hyperparameters here if needed
            }
        )

def get_dataloaders_and_sampler(config, world_size, rank, is_main_process):
    """Load dataset, split it, and build DataLoaders with optional DistributedSampler."""
    if is_main_process: 
        print(f"Loading dataset from {config.dataset.path}...")
    
    # Assuming 'load_from_disk' is defined elsewhere
    dataset = load_from_disk(config.dataset.path)
    
    # Split dataset into train and validation
    train_size = int(config.dataset.train_split * len(dataset))
    val_size = len(dataset) - train_size
    # Every DDP rank must construct the exact same Subset indices. The model
    # noise seed may be rank-specific, but the dataset split must not be.
    split_generator = torch.Generator().manual_seed(int(config.train.seed))
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=split_generator
    )

    
    # Create DistributedSampler if using multiple GPUs
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    # Create DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn, # Assuming 'collate_fn' is imported
        num_workers=4,
        pin_memory=True
    )
    
    if is_main_process: 
        print(f"✓ Data ready: {len(train_ds)} train, {len(val_ds)} val")
    
    return train_loader, val_ds, train_sampler

def build_model_system(config, device, local_rank, world_size, is_main_process):
    """Build the model, optimizer, scheduler, and wrap with DDP if necessary."""
    if is_main_process: 
        print("Initializing model...")
    
    # Build model config structure that TernaryFlowModel expects
    # TernaryFlowModel needs: config.model and config.interpolant
    from configs.config import DictToObject
    
    model_config = DictToObject({
        'node_embed_size': config.model.encoder.node_embed_size,
        'edge_embed_size': config.model.encoder.edge_embed_size,
        'ipa': config.model.encoder.ipa,
        'unified_egnn': getattr(config.model.encoder, 'unified_egnn', None),
        'interface_model': DictToObject({
            'path': getattr(config.model.interface_model, 'path', None),
            'trainable': getattr(config.model.interface_model, 'trainable', False),
            'finetune_heads_only': getattr(config.model.interface_model, 'finetune_heads_only', True),
            'feat_dim': config.model.interface_model.feat_dim,
            'depth': getattr(config.model.interface_model, 'depth', 4),
            'num_nearest_neighbors': getattr(config.model.interface_model, 'num_nearest_neighbors', 16),
            'num_att_heads': getattr(config.model.interface_model, 'num_att_heads', 50),
            'topk_k': getattr(config.model.interface_model, 'topk_k', 50),
        })
    })
    
    # Get sampling config from interpolant or create default
    if hasattr(config.model.interpolant, 'sampling'):
        sampling_config = config.model.interpolant.sampling
        # Convert num_timesteps to num_steps if needed
        if hasattr(sampling_config, 'num_timesteps'):
            sampling_config = DictToObject({'num_steps': sampling_config.num_timesteps})
        else:
            sampling_config = DictToObject({'num_steps': getattr(sampling_config, 'num_steps', 100)})
    else:
        sampling_config = DictToObject({'num_steps': 100})
    
    full_config = DictToObject({
        'model': model_config,
        'interpolant': config.model.interpolant,
        'sampling': sampling_config,
    })
    
    model = TernaryFlowModel(full_config)
    model = model.to(device)
    
    if is_main_process:
        stats = count_parameters(model) # Assuming 'count_parameters' is imported
        print(f"✓ Model loaded: {stats['trainable']:,} trainable params")

    # Wrap model with DDP for distributed training
    if world_size > 1:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank, 
            # The unified EGNN, atom head and RT head all participate in every
            # training forward; unused-parameter traversal is unnecessary.
            find_unused_parameters=False
        )
        
    optimizer = create_optimizer(model, config.train)
    scheduler = create_scheduler(optimizer, config.train)
    
    return model, optimizer, scheduler

def log_metrics(epoch, global_step, train_metrics, val_metrics, optimizer, is_main_process):
    """Unified function to log metrics to WandB."""
    if not is_main_process or wandb.run is None:
        return
        
    log_dict = {
        'epoch': epoch + 1,
        'global_step': global_step,
        'train/lr': optimizer.param_groups[0]['lr'],
        # Unpack raw/total training metrics, but do not duplicate every loss
        # with its weighted counterpart on WandB. Weighted metrics remain in
        # train_metrics for console display and checkpoint diagnostics.
        **{
            f'train/{k.removeprefix("avg_")}': v
            for k, v in train_metrics.items()
            if not k.removeprefix("avg_").startswith("weighted_")
        },
    }
    
    if val_metrics:
        # Unpack validation metrics
        log_dict.update({f'val/{k}': v for k, v in val_metrics.items()})
        
    wandb.log(log_dict)

def save_checkpoint_wrapper(model, optimizer, scheduler, epoch, global_step, train_metrics, val_metrics, config, checkpoint_dir, world_size, scaler, is_main_process, is_best=False):
    """Wrapper for saving checkpoints to reduce clutter in main loop."""
    if not is_main_process:
        return

    metrics_dict = {'train_metrics': train_metrics}
    if val_metrics:
        metrics_dict['val_metrics'] = val_metrics

    # Determine filename
    if is_best:
        filename = "best.pt"
    else:
        filename = f"checkpoint_epoch_{epoch}.pt"

    save_checkpoint(
        model, optimizer, scheduler,
        epoch, global_step,
        metrics_dict,
        checkpoint_dir,
        is_ddp=(world_size > 1),
        scaler=scaler,
        filename=filename if is_best else None  # Only use custom filename for best model
    )
    if is_best:
        path = os.path.join(checkpoint_dir, filename)
        print(f"⭐ New best model saved to {path}")


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def format_number(num):
    """Format number with appropriate unit (K, M, B)"""
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return str(num)


def compute_rmsd(coords_pred, coords_gt, mask):
    """
    Compute RMSD between predicted and ground truth coordinates.
    
    Args:
        coords_pred: (B, L, 3) predicted coordinates
        coords_gt: (B, L, 3) ground truth coordinates
        mask: (B, L) boolean mask for valid positions
    
    Returns:
        rmsd: scalar RMSD value
    """
    # Ensure all tensors are on the same device
    device = coords_pred.device
    coords_gt = coords_gt.to(device)
    mask = mask.to(device)
    
    # Apply mask to get valid coordinates
    # mask: (B, L) -> (B, L, 1) for broadcasting
    mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
    
    # Get valid coordinates
    coords_pred_masked = coords_pred * mask_expanded  # (B, L, 3)
    coords_gt_masked = coords_gt * mask_expanded  # (B, L, 3)
    
    # Flatten to (N, 3) where N is number of valid positions
    coords_pred_flat = coords_pred_masked[mask]  # (N, 3)
    coords_gt_flat = coords_gt_masked[mask]  # (N, 3)
    
    if len(coords_pred_flat) == 0:
        return torch.tensor(0.0, device=device)
    
    # Compute RMSD
    squared_diff = torch.sum((coords_pred_flat - coords_gt_flat) ** 2, dim=-1)
    rmsd = torch.sqrt(torch.mean(squared_diff))
    return rmsd


def compute_translation_error(trans_pred, trans_gt):
    """
    Compute translation error (L2 distance).
    
    Args:
        trans_pred: (B, 3) predicted translation
        trans_gt: (B, 3) ground truth translation
    
    Returns:
        error: scalar translation error
    """
    # Ensure all tensors are on the same device
    device = trans_pred.device
    trans_gt = trans_gt.to(device)
    
    error = torch.sqrt(torch.sum((trans_pred - trans_gt) ** 2, dim=-1))
    return torch.mean(error)


def compute_rotation_error(rot_pred, rot_gt):
    """
    Compute rotation error using geodesic distance.
    
    Args:
        rot_pred: (B, 3, 3) predicted rotation matrices
        rot_gt: (B, 3, 3) ground truth rotation matrices
    
    Returns:
        error: scalar rotation error
    """
    # Ensure all tensors are on the same device
    device = rot_pred.device
    rot_gt = rot_gt.to(device)
    
    # Compute geodesic distance for each sample
    errors = geodesic_dist(rot_pred, rot_gt)  # (B,)
    return torch.mean(errors)


def compute_sequence_accuracy(
    seq_pred,
    seq_gt,
    mask,
    align_to_pocketxmol: bool = False,
    pocketxmol_atomic_numbers: Optional[Sequence[int]] = None,
):
    """
    Compute sequence accuracy (amino acid recovery).
    
    Args:
        seq_pred: (B, L) predicted sequences
        seq_gt: (B, L) ground truth sequences
        mask: (B, L) boolean mask for valid positions
        align_to_pocketxmol: If True, align our 25-way atom-type vocabulary to
            PocketXMol element vocabulary before computing accuracy.
        pocketxmol_atomic_numbers: PocketXMol element list in nodetype order.
            Default: [6, 7, 8, 9, 15, 16, 17, 5, 35, 53, 34].
    
    Returns:
        accuracy: scalar accuracy value (0-1)
    """
    # Ensure all tensors are on the same device
    device = seq_pred.device
    seq_gt = seq_gt.to(device)
    mask = mask.to(device)
    
    if align_to_pocketxmol:
        if pocketxmol_atomic_numbers is None:
            pocketxmol_atomic_numbers = [6, 7, 8, 9, 15, 16, 17, 5, 35, 53, 34]

        # Map our atom-type indices -> atomic number -> PocketXMol nodetype index.
        pxm_ele_to_idx = {int(ele): i for i, ele in enumerate(pocketxmol_atomic_numbers)}
        seq_pred_aligned = torch.full_like(seq_pred, -1)
        seq_gt_aligned = torch.full_like(seq_gt, -1)
        for idx, atomic_num in _INDEX_TO_ATOMIC_NUM.items():
            if atomic_num in pxm_ele_to_idx:
                pxm_idx = pxm_ele_to_idx[atomic_num]
                seq_pred_aligned[seq_pred == idx] = pxm_idx
                seq_gt_aligned[seq_gt == idx] = pxm_idx

        # Only positions mappable to PocketXMol dictionary are counted.
        align_mask = mask & (seq_pred_aligned >= 0) & (seq_gt_aligned >= 0)
        correct = (seq_pred_aligned == seq_gt_aligned) & align_mask
        total = align_mask.sum()
    else:
        # Only consider positions where mask is True
        correct = (seq_pred == seq_gt) & mask
        total = mask.sum()
    
    if total == 0:
        return torch.tensor(0.0, device=device)
    
    accuracy = correct.sum().float() / total.float()
    return accuracy


def format_sequence(seq_tensor, mask, max_display=50):
    """Format sequence tensor to string representation"""
    seq_list = seq_tensor.cpu().numpy().tolist()
    mask_list = mask.cpu().numpy().tolist()
    
    # Only show valid positions
    valid_seq = [str(s) for s, m in zip(seq_list, mask_list) if m]
    
    if len(valid_seq) > max_display:
        return ' '.join(valid_seq[:max_display]) + f' ... (total {len(valid_seq)} atoms)'
    else:
        return ' '.join(valid_seq)


def format_coords(coords_tensor, mask, max_display=50):
    """Format coordinates tensor to string representation"""
    coords = coords_tensor.cpu().numpy()
    mask_list = mask.cpu().numpy().tolist()
    
    # Only show valid positions
    valid_coords = [coords[i] for i, m in enumerate(mask_list) if m]
    
    if len(valid_coords) > max_display:
        lines = []
        for i in range(max_display):
            lines.append(f"  Atom {i}: [{valid_coords[i][0]:.3f}, {valid_coords[i][1]:.3f}, {valid_coords[i][2]:.3f}]")
        lines.append(f"  ... (total {len(valid_coords)} atoms)")
        return '\n'.join(lines)
    else:
        lines = []
        for i, coord in enumerate(valid_coords):
            lines.append(f"  Atom {i}: [{coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f}]")
        return '\n'.join(lines)


def format_translation(trans_tensor):
    """Format translation vector to string"""
    trans_tensor = trans_tensor[0]
    trans = trans_tensor.cpu().numpy()
    return f"[{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]"

def format_rotation(rot_tensor):
    """Format rotation matrix to string"""
    rot = rot_tensor.cpu().numpy()
    lines = []
    for row in rot:
        lines.append(f"  [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}]")
    return '\n'.join(lines)

# Utils for evaluation
def read_pdb(pdb_path):
    return PandasPdb().read_pdb(pdb_path)

def get_pdb_coords(pdb, is_ligand) -> np.ndarray:
    """Get the coordinates of the atoms in a pdb file"""
    if isinstance(pdb, str):
        pdb = read_pdb(pdb)
    if not isinstance(pdb, PandasPdb):
        raise ValueError(
            "pdb must be a path to a pdb file or a PandasPdb object")
    category = "HETATM" if is_ligand else "ATOM"
    coords = pdb.df[category][["x_coord", "y_coord", "z_coord"]].to_numpy()
    return coords

def merge_pdbs(pdb_files: List[str], path: str):
    complex_text = []
    for fname in pdb_files:
        with open(fname, 'r') as f:
            complex_text.extend(f.read().strip().split('\n'))
    complex_text = [i for i in complex_text if i[:4] in ('ATOM', 'HETA', 'CONE')]
    with open(path, 'w') as f:
        f.write('\n'.join(complex_text))

def write_pdb(pdb, path):
    if not isinstance(pdb, PandasPdb):
        raise ValueError("pdb must be a PandasPdb object")
    pdb.to_pdb(path)

def set_new_coords(pdb, coords, is_ligand):
    if isinstance(pdb, str):
        pdb = read_pdb(pdb)
    if not isinstance(pdb, PandasPdb):
        raise ValueError(
            "pdb must be a path to a pdb file or a PandasPdb object")
    original_coords = get_pdb_coords(pdb, is_ligand)
    assert original_coords.shape == coords.shape, f"Original coords shape: {original_coords.shape}, new coords shape: {coords.shape}"
    category = "HETATM" if is_ligand else "ATOM"
    pdb.df[category][["x_coord", "y_coord", "z_coord"]] = coords
    return pdb


def set_new_chain(pdb, chain_id, is_ligand):
    if isinstance(pdb, str):
        pdb = read_pdb(pdb)
    if not isinstance(pdb, PandasPdb):
        raise ValueError(
            "pdb must be a path to a pdb file or a PandasPdb object")
    category = "HETATM" if is_ligand else "ATOM"
    assert len(set(pdb.df[category]["chain_id"])) == 1, f"Only one chain is allowed, but got {set(pdb.df[category]['chain_id'])}"
    pdb.df[category]["chain_id"] = chain_id
    return pdb
