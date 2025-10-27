#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split, DistributedSampler
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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Flow Matching Model')
    
    # Config file
    parser.add_argument('--config', type=str, default='configs/flow_matching_config.yaml',
                        help='Path to config file')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to dataset (overrides config)')
    parser.add_argument('--train_split', type=float, default=None,
                        help='Training split ratio (overrides config)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size per GPU (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--max_iters', type=int, default=None,
                        help='Maximum iterations (overrides config)')
    parser.add_argument('--val_freq', type=int, default=None,
                        help='Validation frequency (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    
    # Checkpoint
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint directory (overrides auto-generated name)')
    
    # Wandb
    parser.add_argument('--wandb_project', type=str, default='MGD',
                        help='Wandb project name (overrides config)')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Wandb run name (overrides config)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    
    return parser.parse_args()


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
    
    if args.max_iters is not None:
        config.train.max_iters = args.max_iters
        if is_main_process:
            overrides.append(f"max_iters = {args.max_iters}")
    
    if args.val_freq is not None:
        config.train.val_freq = args.val_freq
        if is_main_process:
            overrides.append(f"val_freq = {args.val_freq}")
    
    if args.seed is not None:
        config.train.seed = args.seed
        if is_main_process:
            overrides.append(f"seed = {args.seed}")
    
    if is_main_process and overrides:
        print("\n📝 Config overrides:")
        for override in overrides:
            print(f"   - {override}")
        print()
    
    return config


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


def train_epoch(model, dataloader, optimizer, scheduler, device, config, global_step, is_main_process=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    loss_weights = config.train.loss_weights
    
    progress_bar = tqdm(dataloader, desc=f"Training", disable=not is_main_process)
    
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


def validate(model, dataloader, device, config, is_main_process=True):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    loss_weights = config.train.loss_weights
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", disable=not is_main_process):
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
    # Parse command line arguments (only on main process)
    args = parse_args()
    
    # Setup distributed training
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
    
    # Load configuration
    if is_main_process:
        print("Loading configuration...")
    config = load_config_from_yaml(args.config)
    if is_main_process:
        print(f"✓ Configuration loaded from {args.config}")
    
    # Apply command line arguments to config
    config = apply_args_to_config(config, args, is_main_process)
    
    # Set random seed
    torch.manual_seed(config.train.seed + rank)
    np.random.seed(config.train.seed + rank)
    
    # Create checkpoint directory
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = f"checkpoints_{timestamp}"
    
    if is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize wandb (only on main process)
    if is_main_process and not args.no_wandb:
        wandb_project = args.wandb_project or "ternary_flow"
        wandb_name = args.wandb_name or f"train_{timestamp}"
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                'batch_size': config.train.batch_size,
                'lr': config.train.optimizer.lr,
                'max_grad_norm': config.train.max_grad_norm,
                'world_size': world_size,
            }
        )
    
    # Load dataset
    if is_main_process:
        print("Loading dataset...")
    try:
        dataset_path = config.dataset.path
        if is_main_process:
            print(f"  Loading from: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        if is_main_process:
            print(f"✓ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        if is_main_process:
            print(f"✗ Failed to load dataset: {e}")
        return
    
    # Split dataset
    train_split = config.dataset.train_split
    val_split = config.dataset.val_split
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    if is_main_process:
        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create distributed samplers if using multiple GPUs
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_shuffle = False  # DistributedSampler handles shuffling
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model
    if is_main_process:
        print("Initializing model...")
    model_config = DictToObject({
        'node_embed_size': config.model.encoder.node_embed_size,
        'edge_embed_size': config.model.encoder.edge_embed_size,
        'ipa': config.model.encoder.ipa,
        'interface_model': DictToObject({
            'feat_dim': config.model.interface_model.feat_dim
        })
    })
    
    full_config = DictToObject({
        'model': model_config,
        'interpolant': config.model.interpolant
    })
    
    model = TernaryFlowModel(full_config)
    model = model.to(device)
    
    # Count and print model parameters before DDP wrapping
    if is_main_process:
        param_stats = count_parameters(model)
        print(f"\n{'='*60}")
        print("Model Parameters:")
        print(f"{'='*60}")
        print(f"Total parameters:  {format_number(param_stats['total']):>10} ({param_stats['total']:,})")
        print(f"Trainable:        {format_number(param_stats['trainable']):>10} ({param_stats['trainable']:,})")
        print(f"Non-trainable:    {format_number(param_stats['non_trainable']):>10} ({param_stats['non_trainable']:,})")
        print(f"{'='*60}\n")
    
    # Wrap model with DDP if using multiple GPUs
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=True  # Enable to handle unused parameters
        )
    
    if is_main_process:
        print(f"✓ Model initialized on {device} (world_size={world_size})")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config.train)
    scheduler = create_scheduler(optimizer, config.train)
    
    # Load checkpoint if exists
    start_epoch = 0
    global_step = 0
    
    if args.resume is not None:
        # Resume from specified checkpoint
        checkpoint_path = args.resume
    else:
        # Try to resume from checkpoints_latest
        checkpoint_dir_old = "checkpoints_latest"
        if os.path.exists(checkpoint_dir_old) and os.path.exists(os.path.join(checkpoint_dir_old, "latest.pt")):
            checkpoint_path = os.path.join(checkpoint_dir_old, "latest.pt")
        else:
            checkpoint_path = None
    
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        if is_main_process:
            print(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch, global_step = load_checkpoint(
            model, optimizer, scheduler,
            checkpoint_path,
            device
        )
    
    # Training loop
    if is_main_process:
        print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.train.max_iters):
        # Set epoch for distributed sampler
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        if is_main_process:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}")
            print(f"{'='*60}")
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, config, global_step, is_main_process
        )
        
        if is_main_process:
            print(f"Train loss: {train_loss:.6f}")
        
        # Validate
        if (epoch + 1) % config.train.val_freq == 0:
            val_loss = validate(model, val_loader, device, config, is_main_process)
            
            if is_main_process:
                print(f"Val loss: {val_loss:.6f}")
                
                # Save checkpoint (only on main process)
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch + 1, global_step,
                    {'train_loss': train_loss, 'val_loss': val_loss},
                    checkpoint_dir,
                    is_ddp=(world_size > 1)
                )
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(checkpoint_dir, 'best.pt')
                    # For DDP models, use module.state_dict()
                    model_state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
                    torch.save({
                        'model_state_dict': model_state_dict,
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
    
    if is_main_process:
        print("\n✓ Training completed!")
        if wandb.run is not None:
            wandb.finish()
    
    # Clean up distributed training
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()