#!/usr/bin/env python3
"""
Full training script for TernaryFlowModel using flow_matching_dataset_v2
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datasets import load_from_disk
import numpy as np
import yaml
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import json
import wandb

from flow_model import TernaryFlowModel
from configs.config import DictToObject


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
    
    # Create masks
    batched['p1_mask'] = torch.ones(batched['p1_residue'].shape, dtype=torch.bool)
    batched['p2_mask'] = torch.ones(batched['p2_residue'].shape, dtype=torch.bool)
    batched['mol_mask'] = torch.ones(batched['lig_seq'].shape, dtype=torch.bool)
    
    # Set padding positions to False in masks
    for i, item in enumerate(batch):
        batched['p1_mask'][i, len(item['p1_residue']):] = False
        batched['p2_mask'][i, len(item['p2_residue']):] = False
        batched['mol_mask'][i, len(item['lig_seq']):] = False
    
    # Add ground truth data for loss calculation
    batched['lig_seq_1'] = batched['lig_seq'].clone()
    batched['lig_coords_1'] = batched['lig_coords_gt'].clone()
    batched['R_inv_1'] = batched['R_inv'].clone()
    batched['t_inv_1'] = batched['t_inv'].clone()
    
    return batched


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss_dict, checkpoint_dir):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
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
        return 0
    
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
            T_max=config.max_iters
        )
    elif config.scheduler.type == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler.type}")
    
    return scheduler


def train_epoch(model, dataloader, optimizer, scheduler, device, config, global_step):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    loss_weights = config.train.loss_weights
    
    progress_bar = tqdm(dataloader, desc=f"Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Forward pass
        loss_dict = model(batch)
        
        # Compute weighted total loss
        total_batch_loss = (
            loss_weights.trans_loss * loss_dict['trans_loss'] +
            loss_weights.rot_loss * loss_dict['rot_loss'] +
            loss_weights.seqs_loss * loss_dict['seqs_loss'] +
            loss_weights.coords_loss * loss_dict['coords_loss']
        )
        
        # Backward pass with gradient accumulation
        (total_batch_loss / config.train.accum_grad).backward()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{total_batch_loss.item():.4f}",
            'trans': f"{loss_dict['trans_loss'].item():.4f}",
            'rot': f"{loss_dict['rot_loss'].item():.4f}",
            'seq': f"{loss_dict['seqs_loss'].item():.4f}",
            'coord': f"{loss_dict['coords_loss'].item():.4f}",
        })
        
        total_loss += total_batch_loss.item()
        global_step += 1
        
        # Optimizer step (with gradient accumulation)
        if (batch_idx + 1) % config.train.accum_grad == 0:
            # Gradient clipping
            if hasattr(config.train, 'max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Update learning rate scheduler
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(total_loss / (batch_idx + 1))
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                'train/loss': total_batch_loss.item(),
                'train/trans_loss': loss_dict['trans_loss'].item(),
                'train/rot_loss': loss_dict['rot_loss'].item(),
                'train/seqs_loss': loss_dict['seqs_loss'].item(),
                'train/coords_loss': loss_dict['coords_loss'].item(),
                'train/lr': optimizer.param_groups[0]['lr'],
                'global_step': global_step,
            })
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, global_step


def validate(model, dataloader, device, config):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    loss_weights = config.train.loss_weights
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            loss_dict = model(batch)
            
            # Compute weighted total loss
            total_batch_loss = (
                loss_weights.trans_loss * loss_dict['trans_loss'] +
                loss_weights.rot_loss * loss_dict['rot_loss'] +
                loss_weights.seqs_loss * loss_dict['seqs_loss'] +
                loss_weights.coords_loss * loss_dict['coords_loss']
            )
            
            total_loss += total_batch_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    # Setup
    config_path = "configs/flow_matching_config.yaml"
    
    # Load configuration
    print("Loading configuration...")
    config = load_config_from_yaml(config_path)
    print(f"✓ Configuration loaded from {config_path}")
    
    # Set random seed
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = f"checkpoints_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="ternary_flow",
        name=f"train_{timestamp}",
        config={
            'batch_size': config.train.batch_size,
            'lr': config.train.optimizer.lr,
            'max_grad_norm': config.train.max_grad_norm,
        }
    )
    
    # Load dataset
    print("Loading dataset...")
    try:
        dataset = load_from_disk("flow_matching_dataset_v2")
        print(f"✓ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model
    print("Initializing model...")
    model_config = DictToObject({
        'node_embed_size': config.model.encoder.node_embed_size,
        'edge_embed_size': config.model.encoder.edge_embed_size,
        'ipa': config.model.encoder.ipa
    })
    
    full_config = DictToObject({
        'model': model_config,
        'interpolant': config.model.interpolant
    })
    
    model = TernaryFlowModel(full_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"✓ Model initialized on {device}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config.train)
    scheduler = create_scheduler(optimizer, config.train)
    
    # Load checkpoint if exists
    start_epoch = 0
    global_step = 0
    if os.path.exists("checkpoints_latest"):
        checkpoint_dir_old = "checkpoints_latest"
        if os.path.exists(os.path.join(checkpoint_dir_old, "latest.pt")):
            start_epoch, global_step = load_checkpoint(
                model, optimizer, scheduler,
                os.path.join(checkpoint_dir_old, "latest.pt"),
                device
            )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.train.max_iters):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}")
        print(f"{'='*60}")
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, config.train, global_step
        )
        
        print(f"Train loss: {train_loss:.6f}")
        
        # Validate
        if (epoch + 1) % config.train.val_freq == 0:
            val_loss = validate(model, val_loader, device, config.train)
            print(f"Val loss: {val_loss:.6f}")
            
            # Save checkpoint
            save_checkpoint(
                model, optimizer, scheduler,
                epoch + 1, global_step,
                {'train_loss': train_loss, 'val_loss': val_loss},
                checkpoint_dir
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(checkpoint_dir, 'best.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'epoch': epoch + 1,
                }, best_path)
                print(f"✓ Best model saved with val_loss={val_loss:.6f}")
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    'val/loss': val_loss,
                    'epoch': epoch + 1,
                })
    
    print("\n✓ Training completed!")
    wandb.finish()


if __name__ == "__main__":
    main()

