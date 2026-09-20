#!/usr/bin/env python3

import os
import argparse
import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
import numpy as np
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import wandb

from utils.training_utils import (
    setup_ddp_environment,
    setup_wandb_logging,
    get_dataloaders_and_sampler,
    build_model_system,
    save_checkpoint_wrapper,
    load_checkpoint,
    load_config_from_yaml,
    apply_args_to_config,
    move_to_device,
    validate,
    log_metrics
)

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
    
    # Mixed precision training
    parser.add_argument('--use_amp', action='store_true',
                        help='Use Automatic Mixed Precision (AMP) for training to reduce memory usage')
    
    return parser.parse_args()


def get_grad_norm_and_nonfinite_names(model):
    """Return the unscaled global grad norm and names with NaN/Inf gradients."""
    grad_norms = []
    nonfinite_names = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if not torch.isfinite(grad).all():
            nonfinite_names.append(name)
        grad_norms.append(torch.linalg.vector_norm(grad))

    if not grad_norms:
        return torch.zeros((), device=next(model.parameters()).device), nonfinite_names
    total_norm = torch.linalg.vector_norm(torch.stack(grad_norms))
    return total_norm, nonfinite_names


def all_ranks_true(local_condition, device):
    """Return True only when every DDP rank reports a true condition."""
    if not dist.is_available() or not dist.is_initialized():
        return bool(local_condition)

    condition = torch.tensor(
        1 if local_condition else 0,
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(condition, op=dist.ReduceOp.MIN)
    return bool(condition.item())


def get_amp_dtype(config):
    """Resolve the configured autocast dtype."""
    name = str(getattr(config.train, 'amp_dtype', 'float16')).lower()
    if name in {'bfloat16', 'bf16'}:
        return torch.bfloat16
    if name in {'float16', 'fp16'}:
        return torch.float16
    raise ValueError(f"Unsupported train.amp_dtype: {name}")

def train_epoch(model, dataloader, optimizer, scheduler, device, config, global_step, is_main_process=True, use_amp=False, scaler=None):
    """Train for one epoch (Simplified & Enhanced Logging)"""
    model.train()
    
    # Use a dictionary to automatically track cumulative values for all losses
    epoch_metrics = defaultdict(float)
    num_batches = 0
    consecutive_nonfinite_batches = 0
    max_consecutive_nonfinite_batches = int(
        getattr(config.train, 'max_consecutive_nonfinite_batches', 8)
    )
    amp_dtype = get_amp_dtype(config)
    
    # Progress bar config: dynamic_ncols automatically adjusts width
    pbar = tqdm(dataloader, desc="Training", disable=not is_main_process, dynamic_ncols=True)
    
    for batch in pbar:
        # 1. Move batch to device
        batch = move_to_device(batch, device)
        
        # 2. Forward pass
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with autocast(
            device_type=device_type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            loss_dict = model(batch)
        
            # Build the objective from losses that are both returned by the
            # model and enabled in the config.  This keeps optional objectives
            # (for example pose_coord_loss) from breaking logging/training and
            # avoids 0 * NaN contaminating an otherwise finite total.
            weights = config.train.loss_weights
            weighted_loss_terms = {}
            for loss_name, loss_value in loss_dict.items():
                weight = float(getattr(weights, loss_name, 0.0))
                if weight != 0.0:
                    weighted_loss_terms[loss_name] = weight * loss_value
            if not weighted_loss_terms:
                raise ValueError("No enabled loss terms were found in loss_weights")
            total_loss = sum(weighted_loss_terms.values())
        
        # Every rank must make the same skip/backward decision. A rank-local
        # ``continue`` makes one rank enter the next DDP forward (BROADCAST)
        # while its peers remain in backward (ALLREDUCE), causing an NCCL hang.
        local_loss_is_finite = bool(torch.isfinite(total_loss).all().item())
        all_losses_are_finite = all_ranks_true(local_loss_is_finite, device)
        if not all_losses_are_finite:
            if not local_loss_is_finite:
                rank = dist.get_rank() if dist.is_initialized() else 0
                print(f"\n{'='*80}")
                print(f"⚠️ [Rank {rank}, Step {global_step}] NaN/Inf Loss detected!")
                print(f"{'='*80}")
                
                # Check each loss component
                print("\n📊 Loss Components:")
                for loss_name, loss_value in loss_dict.items():
                    if isinstance(loss_value, torch.Tensor):
                        has_nan = torch.isnan(loss_value).any().item()
                        has_inf = torch.isinf(loss_value).any().item()
                        if has_nan or has_inf:
                            status = []
                            if has_nan:
                                status.append("NaN")
                            if has_inf:
                                status.append("Inf")
                            print(f"  ❌ {loss_name}: {', '.join(status)}")
                            if loss_value.numel() == 1:
                                print(f"     Value: {loss_value.item()}")
                            else:
                                print(f"     Shape: {loss_value.shape}")
                                print(f"     NaN count: {torch.isnan(loss_value).sum().item()}")
                                print(f"     Inf count: {torch.isinf(loss_value).sum().item()}")
                        else:
                            print(f"  ✓ {loss_name}: {loss_value.item():.6f}")
                    else:
                        print(f"  {loss_name}: {loss_value}")
                
                # Check total loss
                print(f"\n📈 Total Loss: {total_loss.item()}")
                
                # Check model parameters for NaN
                print("\n🔍 Checking model parameters for NaN/Inf...")
                nan_params = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            nan_count = torch.isnan(param.grad).sum().item()
                            inf_count = torch.isinf(param.grad).sum().item()
                            nan_params.append((name, nan_count, inf_count, "grad"))
                    if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                        nan_count = torch.isnan(param.data).sum().item()
                        inf_count = torch.isinf(param.data).sum().item()
                        nan_params.append((name, nan_count, inf_count, "data"))
                
                if nan_params:
                    print("  ❌ Found NaN/Inf in parameters:")
                    for name, nan_count, inf_count, param_type in nan_params:
                        print(f"     {name} ({param_type}): NaN={nan_count}, Inf={inf_count}")
                else:
                    print("  ✓ No NaN/Inf found in parameters")
                
                print(f"{'='*80}\n")
            elif is_main_process:
                print(
                    f"\n⚠️ [Step {global_step}] Another rank reported a "
                    "NaN/Inf loss. Skipping this batch on every rank."
                )

            optimizer.zero_grad(set_to_none=True)
            # This batch was consumed even though it produced no update. Keep
            # global-step/accumulation schedules aligned across ranks.
            global_step += 1
            consecutive_nonfinite_batches += 1
            if (
                max_consecutive_nonfinite_batches > 0
                and consecutive_nonfinite_batches
                >= max_consecutive_nonfinite_batches
            ):
                raise FloatingPointError(
                    "Stopping after "
                    f"{consecutive_nonfinite_batches} consecutive non-finite "
                    "batches. Run debug_flow_nan.py on the latest finite "
                    "checkpoint before restarting."
                )
            continue

        consecutive_nonfinite_batches = 0
        
        # 4. Backward
        scaled_loss = total_loss / config.train.accum_grad
        if use_amp and scaler:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # 5. Optimizer Step (Gradient Accumulation)
        grad_norm = 0.0
        if (global_step + 1) % config.train.accum_grad == 0:
            # Unscale & Clip
            if use_amp and scaler:
                scaler.unscale_(optimizer)

            # Check before clipping. Clipping Inf gradients multiplies them by
            # zero and turns otherwise finite gradients into NaN, obscuring the
            # real AMP overflow source.
            grad_norm, nonfinite_grad_names = get_grad_norm_and_nonfinite_names(model)
            local_grads_are_finite = (
                not nonfinite_grad_names and bool(torch.isfinite(grad_norm).item())
            )
            grads_are_finite = all_ranks_true(local_grads_are_finite, device)
            amp_scale_before = scaler.get_scale() if use_amp and scaler else None

            if grads_are_finite:
                if hasattr(config.train, 'max_grad_norm'):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.train.max_grad_norm
                    )
                if use_amp and scaler:
                    scaler.step(optimizer)
                else:
                    optimizer.step()
            elif is_main_process:
                print(
                    f"\n⚠️ [Step {global_step}] Skipping optimizer step: "
                    "at least one rank has non-finite gradients."
                )

            if not local_grads_are_finite:
                rank = dist.get_rank() if dist.is_initialized() else 0
                print(
                    f"  [Rank {rank}] Non-finite gradients in "
                    f"{len(nonfinite_grad_names)} parameters."
                )
                for name in nonfinite_grad_names[:10]:
                    print(f"    - {name}")

            if use_amp and scaler:
                if grads_are_finite:
                    scaler.update()
                else:
                    # Do not call scaler.step() on a locally-finite rank: it
                    # would update only that rank's parameters. Apply the same
                    # backoff scale everywhere and skip the optimizer globally.
                    new_scale = amp_scale_before * scaler.get_backoff_factor()
                    scaler.update(new_scale=new_scale)
                if not grads_are_finite and is_main_process:
                    print(
                        f"  AMP scale: {amp_scale_before:g} -> "
                        f"{scaler.get_scale():g}"
                    )
            
            optimizer.zero_grad(set_to_none=True)
            
            # Scheduler Step (if updating at batch level)
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                 # Note: Plateau is usually recommended to update per epoch; keeping your original logic here
                scheduler.step(total_loss)

        global_step += 1
        num_batches += 1

        # 6. Logging (Simplified output)
        # Accumulate metrics
        epoch_metrics['total'] += total_loss.item()
        for k, v in loss_dict.items():
            epoch_metrics[k] += v.item()
        for k, v in weighted_loss_terms.items():
            epoch_metrics[f'weighted_{k}'] += v.item()

        # Update progress bar
        logs = {
            'L_tot': f"{total_loss.item():.3f}",
            # 'R_crd': f"{loss_dict['coords_loss'].item():.2f}",
            # 'R_seq': f"{loss_dict['seqs_loss'].item():.2f}",
            # 'R_tra': f"{loss_dict['trans_loss'].item():.1f}",
            # 'R_rot': f"{loss_dict['rot_loss'].item():.2f}",
            'crd': f"{weighted_loss_terms.get('coords_loss', torch.zeros((), device=device)).item():.2f}",
            'seq': f"{weighted_loss_terms.get('seqs_loss', torch.zeros((), device=device)).item():.2f}",
            'bnd': f"{weighted_loss_terms.get('bond_loss', torch.zeros((), device=device)).item():.2f}",
            't': f"{weighted_loss_terms.get('trans_loss', torch.zeros((), device=device)).item():.2f}",
            'R': f"{weighted_loss_terms.get('rot_loss', torch.zeros((), device=device)).item():.2f}",
            'p2': f"{weighted_loss_terms.get('pose_coord_loss', torch.zeros((), device=device)).item():.2f}",
            'G_nrm': f"{grad_norm:.2f}" if isinstance(grad_norm, float) else f"{grad_norm.item():.2f}"
        }
        # if 'pose_coord_loss' in loss_dict:
        #     logs['R_pose'] = f"{loss_dict['pose_coord_loss'].item():.2f}"
        pbar.set_postfix(logs)

    # 7. End of Epoch Processing
    # Handle remaining gradients (Tail batch handling)
    if len(dataloader) % config.train.accum_grad != 0:
        if use_amp and scaler:
            scaler.unscale_(optimizer)
        grad_norm, nonfinite_grad_names = get_grad_norm_and_nonfinite_names(model)
        local_grads_are_finite = (
            not nonfinite_grad_names and bool(torch.isfinite(grad_norm).item())
        )
        grads_are_finite = all_ranks_true(local_grads_are_finite, device)
        if grads_are_finite:
            if hasattr(config.train, 'max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            if use_amp and scaler:
                scaler.step(optimizer)
            else:
                optimizer.step()
        elif is_main_process:
            print("\n⚠️ Skipping final partial optimizer step: non-finite gradients.")
        if use_amp and scaler:
            if grads_are_finite:
                scaler.update()
            else:
                new_scale = scaler.get_scale() * scaler.get_backoff_factor()
                scaler.update(new_scale=new_scale)
        optimizer.zero_grad(set_to_none=True)
    
    # Compute averages
    avg_metrics = {f"avg_{k}": v / num_batches for k, v in epoch_metrics.items()}
    
    return avg_metrics, global_step

def main():
    # 1. Environment and Configuration Setup
    args = parse_args()
    device, rank, local_rank, world_size, is_main = setup_ddp_environment()
    
    if is_main: 
        print("Loading configuration...")
    config = load_config_from_yaml(args.config)
    config = apply_args_to_config(config, args, is_main)
    
    # Mixed Precision Setup
    use_amp = args.use_amp or getattr(config.train, 'use_amp', False)
    amp_dtype = get_amp_dtype(config)
    # BF16 has FP32-like exponent range and does not require loss scaling.
    # Keep GradScaler only for the optional FP16 mode.
    scaler = (
        GradScaler(
            'cuda',
            enabled=use_amp and amp_dtype == torch.float16,
            init_scale=float(getattr(config.train, 'amp_init_scale', 4.0)),
            growth_interval=int(getattr(config.train, 'amp_growth_interval', 2000)),
        )
        if device.type == 'cuda' and use_amp and amp_dtype == torch.float16
        else None
    )
    if is_main:
        precision = (
            f"AMP ({str(amp_dtype).removeprefix('torch.').upper()})"
            if use_amp
            else 'FP32'
        )
        print(f"Training precision: {precision}")
    
    # 2. Initialize Logging (WandB)
    setup_wandb_logging(config, args, world_size, is_main)
    
    # 3. Random Seed and Checkpoint Directory
    # Assuming set_random_seed exists or:
    torch.manual_seed(config.train.seed + rank)
    np.random.seed(config.train.seed + rank)
    
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoint_dir = f"checkpoints_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    if is_main:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 4. Data and Model Initialization
    train_loader, val_dataset, train_sampler = get_dataloaders_and_sampler(config, world_size, rank, is_main)
    model, optimizer, scheduler = build_model_system(config, device, local_rank, world_size, is_main)
    
    # 5. Resume from Checkpoint (if applicable)
    start_epoch, global_step = 0, 0
    resume_state = None
    if args.resume:
        resume_state = load_checkpoint(
            model, optimizer, scheduler, args.resume, device
        )
        start_epoch = int(resume_state.get('epoch', 0))
        global_step = int(resume_state.get('global_step', 0))
        if scaler is not None and resume_state.get('scaler_state') is not None:
            scaler.load_state_dict(resume_state['scaler_state'])
        if world_size > 1:
            dist.barrier()
        if is_main:
            print(
                f"✓ Resuming at epoch {start_epoch + 1}, "
                f"global step {global_step}"
            )

    # 6. ======= Main Training Loop =======
    if is_main: 
        print(f"\nStarting training for {config.train.max_epochs} epochs...")
    
    best_val_rmsd = float('inf')
    if resume_state is not None:
        previous_losses = resume_state.get('loss_dict') or {}
        previous_val = previous_losses.get('val_metrics') or {}
        best_val_rmsd = float(previous_val.get('rmsd', best_val_rmsd))

    for epoch in range(start_epoch, config.train.max_epochs):
        # Set epoch for DistributedSampler to ensure proper shuffling
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        # DEBUG
        # val_metrics = validate(model, val_dataset, device, config, is_main, is_ddp=(world_size > 1))
        
        # --- A. Training Step ---
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, config, global_step, is_main, 
            use_amp=use_amp, scaler=scaler
        )
        
        # --- B. Validation Step ---
        val_metrics = None
        should_validate = (epoch + 1) % config.train.val_freq == 0
        
        if should_validate:
            # Sampling validation does not need DDP collectives. Run it once
            # on rank 0 while the other ranks wait, instead of duplicating the
            # same expensive trajectories on every GPU.
            if is_main:
                val_metrics = validate(
                    model, val_dataset, device, config,
                    is_main_process=True,
                    is_ddp=(world_size > 1),
                )
            if world_size > 1:
                dist.barrier()
            
            # Save Best Model
            if is_main and val_metrics['rmsd'] < best_val_rmsd:
                best_val_rmsd = val_metrics['rmsd']
                save_checkpoint_wrapper(
                    model, optimizer, scheduler, epoch + 1, global_step,
                    train_metrics, val_metrics, config, checkpoint_dir,
                    world_size, scaler, is_main, is_best=True
                )

        # --- C. Logging (Unified) ---
        log_metrics(epoch, global_step, train_metrics, val_metrics, optimizer, is_main)
        
        # --- D. Console Output ---
        if is_main:
            print(f"Epoch {epoch+1} | Loss: {train_metrics.get('avg_total', 0.0):.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            if val_metrics:
                print(f"Validation | RMSD: {val_metrics.get('rmsd', 0.0):.4f} | Seq Acc: {val_metrics.get('seq_acc', 0.0):.2%}")
        
        # --- E. Scheduler Step ---
        # Step the scheduler if it's not ReduceLROnPlateau (which is stepped in train_epoch usually)
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
            
        # --- F. Regular Checkpoint Saving ---
        if is_main and ((epoch + 1) % getattr(config.train, 'save_freq', 1) == 0):
            save_checkpoint_wrapper(
                model, optimizer, scheduler, epoch + 1, global_step,
                train_metrics, val_metrics, config, checkpoint_dir,
                world_size, scaler, is_main, is_best=False
            )

    # 7. Cleanup
    if is_main: 
        print("✓ Training completed!")
        if wandb.run: 
            wandb.finish()
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
