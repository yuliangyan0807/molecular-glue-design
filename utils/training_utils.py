"""Training utility functions for flow matching model training"""

import os
import torch
import torch.nn as nn
import numpy as np
from utils.so3_utils import geodesic_dist


def collate_fn(batch):
    """Collate function for batching"""
    # Get the maximum lengths for padding
    max_p1_len = max([len(item['p1_residue']) for item in batch])
    max_p2_len = max([len(item['p2_residue']) for item in batch])
    max_lig_len = max([len(item['lig_seq']) for item in batch])
    
    # Pad sequences and coordinates
    batched = {}
    
    # Pad sequences
    for key in ['p1_residue', 'p2_residue', 'lig_seq']:
        batched[key] = []
        for item in batch:
            seq = torch.tensor(item[key], dtype=torch.long)
            if key == 'p1_residue':
                padded = torch.nn.functional.pad(seq, (0, max_p1_len - len(seq)), value=20)
            elif key == 'p2_residue':
                padded = torch.nn.functional.pad(seq, (0, max_p2_len - len(seq)), value=20)
            else:  # lig_seq
                padded = torch.nn.functional.pad(seq, (0, max_lig_len - len(seq)), value=0)
            batched[key].append(padded)
        batched[key] = torch.stack(batched[key])
    
    # Pad coordinates
    for key in ['p1_coords', 'p2_coords', 'p1_n_coords', 'p2_n_coords', 'p1_c_coords', 'p2_c_coords']:
        batched[key] = []
        for item in batch:
            coords = torch.tensor(item[key], dtype=torch.float32)
            if 'p1' in key:
                padded = torch.nn.functional.pad(coords, (0, 0, 0, max_p1_len - coords.shape[0]), value=0)
            else:  # p2
                padded = torch.nn.functional.pad(coords, (0, 0, 0, max_p2_len - coords.shape[0]), value=0)
            batched[key].append(padded)
        batched[key] = torch.stack(batched[key])
    
    # Pad ligand coordinates
    batched['lig_coords'] = []
    for item in batch:
        coords = torch.tensor(item['lig_coords'], dtype=torch.float32)
        padded = torch.nn.functional.pad(coords, (0, 0, 0, max_lig_len - coords.shape[0]), value=0)
        batched['lig_coords'].append(padded)
    batched['lig_coords'] = torch.stack(batched['lig_coords'])
    
    # Pad ligand ground truth coordinates
    batched['lig_coords_gt'] = []
    for item in batch:
        coords = torch.tensor(item['lig_coords_gt'], dtype=torch.float32)
        padded = torch.nn.functional.pad(coords, (0, 0, 0, max_lig_len - coords.shape[0]), value=0)
        batched['lig_coords_gt'].append(padded)
    batched['lig_coords_gt'] = torch.stack(batched['lig_coords_gt'])
    
    # Handle R_inv and t_inv
    batched['R_inv'] = torch.stack([torch.tensor(item['R_inv'], dtype=torch.float32) for item in batch])
    batched['t_inv'] = torch.stack([torch.tensor(item['t_inv'], dtype=torch.float32) for item in batch])
    
    # Handle interface_flag (for classifier-free guidance)
    batched['interface_flag'] = torch.stack([
        torch.tensor(item['interface_flag'], dtype=torch.bool) for item in batch
    ])
    
    # Create masks
    batched['p1_mask'] = torch.ones(batched['p1_residue'].shape, dtype=torch.bool)
    batched['p2_mask'] = torch.ones(batched['p2_residue'].shape, dtype=torch.bool)
    batched['mol_mask'] = torch.ones(batched['lig_seq'].shape, dtype=torch.bool)
    
    # Set padding positions to False in masks
    for i, item in enumerate(batch):
        batched['p1_mask'][i, len(item['p1_residue']):] = False
        batched['p2_mask'][i, len(item['p2_residue']):] = False
        batched['mol_mask'][i, len(item['lig_seq']):] = False
    
    # Add ground truth data for loss calculation (detach to save memory)
    batched['lig_seq_1'] = batched['lig_seq'].detach().clone()
    batched['lig_coords_1'] = batched['lig_coords'].detach().clone()
    batched['R_inv_1'] = batched['R_inv'].detach().clone()
    batched['t_inv_1'] = batched['t_inv'].detach().clone()
    
    return batched


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss_dict, checkpoint_dir, is_ddp=False):
    """Save training checkpoint"""
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
    
    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'latest.pt')
    torch.save(checkpoint, latest_path)
    
    # Save epoch checkpoint
    epoch_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, epoch_path)
    
    print(f"✓ Checkpoint saved: {epoch_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load training checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0, 0
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    
    print(f"✓ Checkpoint loaded: epoch={epoch}, global_step={global_step}")
    
    return epoch, global_step


def create_optimizer(model, config):
    """Create optimizer based on config"""
    if config.optimizer.type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=(config.optimizer.beta1, config.optimizer.beta2)
        )
    elif config.optimizer.type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
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
            # Linear interpolation: lr = initial_lr * (1 - epoch/total_epochs) + end_lr * (epoch/total_epochs)
            # Simplified: lr = initial_lr - (initial_lr - end_lr) * (epoch / total_epochs)
            if total_epochs == 0:
                return 1.0
            progress = epoch / total_epochs
            factor = 1.0 - progress + (end_lr / initial_lr) * progress if initial_lr > 0 else 1.0
            return max(factor, end_lr / initial_lr if initial_lr > 0 else 0.0)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif config.scheduler.type == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler.type}")
    
    return scheduler


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


def compute_sequence_accuracy(seq_pred, seq_gt, mask):
    """
    Compute sequence accuracy (amino acid recovery).
    
    Args:
        seq_pred: (B, L) predicted sequences
        seq_gt: (B, L) ground truth sequences
        mask: (B, L) boolean mask for valid positions
    
    Returns:
        accuracy: scalar accuracy value (0-1)
    """
    # Ensure all tensors are on the same device
    device = seq_pred.device
    seq_gt = seq_gt.to(device)
    mask = mask.to(device)
    
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


def format_coords(coords_tensor, mask, max_display=5):
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
    trans = trans_tensor.cpu().numpy()
    return f"[{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]"


def format_rotation(rot_tensor):
    """Format rotation matrix to string"""
    rot = rot_tensor.cpu().numpy()
    lines = []
    for row in rot:
        lines.append(f"  [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}]")
    return '\n'.join(lines)