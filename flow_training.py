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
import random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import json
import wandb

from flow_model import TernaryFlowModel
from configs.config import DictToObject
from utils.training_utils import (
    collate_fn,
    save_checkpoint,
    load_checkpoint,
    create_optimizer,
    create_scheduler,
    count_parameters,
    format_number,
    compute_rmsd,
    compute_translation_error,
    compute_rotation_error,
    compute_sequence_accuracy,
    format_sequence,
    format_coords,
    format_translation,
    format_rotation,
)
from sampling import *

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
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Maximum epochs (overrides config)')
    parser.add_argument('--val_freq', type=int, default=None,
                        help='Validation frequency in epochs (overrides config)')
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
        print("\n📝 Config overrides:")
        for override in overrides:
            print(f"   - {override}")
        print()
    
    return config


def train_epoch(model, dataloader, optimizer, scheduler, device, config, global_step, is_main_process=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_trans_loss = 0.0
    total_rot_loss = 0.0
    total_seqs_loss = 0.0
    total_coords_loss = 0.0
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
        
        # Accumulate losses for epoch average
        total_loss += total_batch_loss.item()
        total_trans_loss += loss_dict['trans_loss'].item()
        total_rot_loss += loss_dict['rot_loss'].item()
        total_seqs_loss += loss_dict['seqs_loss'].item()
        total_coords_loss += loss_dict['coords_loss'].item()
        
        global_step += 1
        
        # Optimizer step (with gradient accumulation)
        if (batch_idx + 1) % config.train.accum_grad == 0:
            # Gradient clipping
            if hasattr(config.train, 'max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Update learning rate scheduler (for non-plateau schedulers, step is called per epoch)
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(total_loss / (batch_idx + 1))
    
    # Compute epoch averages
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_trans_loss = total_trans_loss / num_batches
    avg_rot_loss = total_rot_loss / num_batches
    avg_seqs_loss = total_seqs_loss / num_batches
    avg_coords_loss = total_coords_loss / num_batches
    
    return {
        'avg_loss': avg_loss,
        'avg_trans_loss': avg_trans_loss,
        'avg_rot_loss': avg_rot_loss,
        'avg_seqs_loss': avg_seqs_loss,
        'avg_coords_loss': avg_coords_loss,
    }, global_step


def validate(model, val_dataset, device, config, is_main_process=True, is_ddp=False):
    """
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
    """
    model.eval()
    
    # Get the actual model (unwrap DDP if needed)
    actual_model = model.module if is_ddp else model
    
    # Number of samples to validate on
    num_val_samples = getattr(config.train, 'num_val_samples', 50)
    num_val_samples = min(num_val_samples, len(val_dataset))
    
    # Randomly select samples from validation set
    val_indices = random.sample(range(len(val_dataset)), num_val_samples)
    
    # Initialize metrics accumulators
    all_rmsd = []
    all_trans_error = []
    all_rot_error = []
    all_seq_acc = []
    
    if is_main_process:
        print(f"Validating on {num_val_samples} samples...")
    
    with torch.no_grad():
        for sample_idx, idx in enumerate(tqdm(val_indices, desc="Validating", disable=not is_main_process)):
            try:
                # Get sample from dataset
                item = val_dataset[idx]
                
                # Create batch (single sample)
                batch_items = [item]
                batch = collate_fn(batch_items)
                
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # Sample from model (use actual_model to handle DDP wrapping)
                traj = actual_model.sample(batch)
                final_sample = traj[-1]  # Get final sample
                
                # Extract predictions and ground truth (batch size = 1, so index 0)
                lig_coords_pred = final_sample['lig_coords'][0]  # (L, 3)
                lig_coords_gt = batch['lig_coords_1'][0]  # (L, 3)
                mol_mask = batch['mol_mask'][0]  # (L,)
                
                trans_pred = final_sample['trans'][0]  # (3,)
                trans_gt = batch['t_inv_1'][0]  # (3,)
                
                rot_pred = final_sample['rotmats'][0]  # (3, 3)
                rot_gt = batch['R_inv_1'][0]  # (3, 3)
                
                lig_seq_pred = final_sample['lig_seq'][0]  # (L,)
                lig_seq_gt = batch['lig_seq_1'][0]  # (L,)
                
                # Compute metrics for this sample
                # RMSD for coordinates
                rmsd = compute_rmsd(lig_coords_pred.unsqueeze(0), lig_coords_gt.unsqueeze(0), mol_mask.unsqueeze(0))
                all_rmsd.append(rmsd.item())
                
                # Translation error
                trans_error = compute_translation_error(trans_pred.unsqueeze(0), trans_gt.unsqueeze(0))
                all_trans_error.append(trans_error.item())
                
                # Rotation error
                rot_error = compute_rotation_error(rot_pred.unsqueeze(0), rot_gt.unsqueeze(0))
                all_rot_error.append(rot_error.item())
                
                # Sequence accuracy
                seq_acc = compute_sequence_accuracy(lig_seq_pred.unsqueeze(0), lig_seq_gt.unsqueeze(0), mol_mask.unsqueeze(0))
                all_seq_acc.append(seq_acc.item())
                
                # Print comparison for this sample
                if is_main_process:
                    print(f"\n{'='*80}")
                    print(f"Sample {sample_idx + 1}/{num_val_samples} (Dataset index: {idx})")
                    print(f"{'='*80}")
                    print(f"\n📊 Metrics:")
                    print(f"  RMSD: {rmsd.item():.4f} Å")
                    print(f"  Translation error: {trans_error.item():.4f} Å")
                    print(f"  Rotation error: {rot_error.item():.4f}")
                    print(f"  Sequence accuracy: {seq_acc.item():.4f} ({seq_acc.item()*100:.2f}%)")
                    
                    print(f"\n🧬 Sequence Comparison:")
                    print(f"  Predicted: {format_sequence(lig_seq_pred, mol_mask)}")
                    print(f"  Ground Truth: {format_sequence(lig_seq_gt, mol_mask)}")
                    
                    print(f"\n📍 Translation Comparison:")
                    print(f"  Predicted: {format_translation(trans_pred)}")
                    print(f"  Ground Truth: {format_translation(trans_gt)}")
                    print(f"  Error: {trans_error.item():.4f} Å")
                    
                    print(f"\n🔄 Rotation Matrix Comparison:")
                    print(f"  Predicted:\n{format_rotation(rot_pred)}")
                    print(f"  Ground Truth:\n{format_rotation(rot_gt)}")
                    print(f"  Rotation error: {rot_error.item():.4f}")
                    
                    print(f"\n📐 Coordinates Comparison (first few atoms):")
                    print(f"  Predicted:\n{format_coords(lig_coords_pred, mol_mask)}")
                    print(f"  Ground Truth:\n{format_coords(lig_coords_gt, mol_mask)}")
                    print(f"  RMSD: {rmsd.item():.4f} Å")
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
                'max_epochs': config.train.max_epochs,
                'val_freq': config.train.val_freq,
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
        'sampling': sampling_config
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
    
    # Get training configuration
    max_epochs = config.train.max_epochs
    dataset_size = len(train_dataset)
    batch_size = config.train.batch_size
    iterations_per_epoch = (dataset_size + batch_size - 1) // batch_size  # ceil division
    
    if is_main_process:
        print(f"Dataset size: {dataset_size} samples")
        print(f"Batch size: {batch_size}")
        print(f"Iterations per epoch: {iterations_per_epoch}")
        print(f"Total epochs: {max_epochs}")
        print(f"Validation frequency: every {config.train.val_freq} epoch(s)")
    
    best_val_rmsd = float('inf')  # Track best RMSD (lower is better)
    
    for epoch in range(start_epoch, max_epochs):
        # Set epoch for distributed sampler
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        if is_main_process:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}")
            print(f"{'='*60}")
        
        # Train
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, config, global_step, is_main_process
        )
        
        if is_main_process:
            print(f"Train loss: {train_metrics['avg_loss']:.6f}")
            
            # Log to wandb at epoch level
            if wandb.run is not None:
                wandb.log({
                    'train/loss': train_metrics['avg_loss'],
                    'train/trans_loss': train_metrics['avg_trans_loss'],
                    'train/rot_loss': train_metrics['avg_rot_loss'],
                    'train/seqs_loss': train_metrics['avg_seqs_loss'],
                    'train/coords_loss': train_metrics['avg_coords_loss'],
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'epoch': epoch + 1,
                    'global_step': global_step,
                })
        
        # Update learning rate scheduler (for non-plateau schedulers)
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        
        # Validate
        val_metrics = None
        if (epoch + 1) % config.train.val_freq == 0:
            val_metrics = validate(model, val_dataset, device, config, is_main_process, is_ddp=(world_size > 1))
            
            if is_main_process:
                print(f"Validation metrics:")
                print(f"  RMSD: {val_metrics['rmsd']:.4f} Å")
                print(f"  Translation error: {val_metrics['trans_error']:.4f} Å")
                print(f"  Rotation error: {val_metrics['rot_error']:.4f}")
                print(f"  Sequence accuracy: {val_metrics['seq_acc']:.4f}")
                
                # Use RMSD as the main metric for model selection
                val_rmsd = val_metrics['rmsd']
                
                # Save best model (based on RMSD - lower is better)
                if val_rmsd < best_val_rmsd:
                    best_val_rmsd = val_rmsd
                    best_path = os.path.join(checkpoint_dir, 'best.pt')
                    # For DDP models, use module.state_dict()
                    model_state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
                    torch.save({
                        'model_state_dict': model_state_dict,
                        'val_metrics': val_metrics,
                        'epoch': epoch + 1,
                    }, best_path)
                    print(f"✓ Best model saved with RMSD={val_rmsd:.4f} Å")
                
                # Log to wandb
                if wandb.run is not None:
                    wandb.log({
                        'val/rmsd': val_metrics['rmsd'],
                        'val/trans_error': val_metrics['trans_error'],
                        'val/rot_error': val_metrics['rot_error'],
                        'val/seq_acc': val_metrics['seq_acc'],
                        'epoch': epoch + 1,
                    })
        
        # Save checkpoint at fixed intervals or at last epoch
        save_freq = getattr(config.train, 'save_freq', 1)
        is_last_epoch = (epoch + 1 == max_epochs)
        should_save = (epoch + 1) % save_freq == 0 or is_last_epoch
        
        if is_main_process and should_save:
            # Prepare metrics dict
            metrics_dict = {'train_metrics': train_metrics}
            if val_metrics is not None:
                metrics_dict['val_metrics'] = val_metrics
            
            # Save checkpoint
            save_checkpoint(
                model, optimizer, scheduler,
                epoch + 1, global_step,
                metrics_dict,
                checkpoint_dir,
                is_ddp=(world_size > 1)
            )
            
            if is_last_epoch:
                print(f"✓ Final epoch checkpoint saved")
    
    if is_main_process:
        print("\n✓ Training completed!")
        if wandb.run is not None:
            wandb.finish()
    
    # Clean up distributed training
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()