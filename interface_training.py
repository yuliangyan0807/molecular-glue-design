import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm

from datasets import load_from_disk
from models.interface_model import InterfaceModel


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_and_stack(sequences, pad_value=20.0):
    """Pad sequences to same length and stack into batch."""
    max_len = max(seq.shape[0] for seq in sequences)
    batch = sequences[0].new_full((len(sequences), max_len) + sequences[0].shape[1:], pad_value)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.bool, device=sequences[0].device)
    for i, seq in enumerate(sequences):
        n = seq.shape[0]
        batch[i, :n] = seq
        mask[i, :n] = True
    return batch, mask


def collate_fn(batch):
    """Custom collate function for variable length sequences."""
    # Convert to tensors
    p1_residue = [torch.tensor(sample['p1_residue'], dtype=torch.long) for sample in batch]
    p2_residue = [torch.tensor(sample['p2_residue'], dtype=torch.long) for sample in batch]
    p1_coords = [torch.tensor(sample['p1_coords'], dtype=torch.float32) for sample in batch]
    p2_coords = [torch.tensor(sample['p2_coords'], dtype=torch.float32) for sample in batch]

    # Pad and stack
    p1_residue, p1_mask = pad_and_stack(p1_residue, pad_value=20.0)
    p2_residue, p2_mask = pad_and_stack(p2_residue, pad_value=20.0)
    p1_coords, _ = pad_and_stack(p1_coords, pad_value=0.0)
    p2_coords, _ = pad_and_stack(p2_coords, pad_value=0.0)

    mu = torch.stack([torch.tensor(sample['mu'], dtype=torch.float32) for sample in batch], dim=0)  # (batch_size, 3)
    sigma = torch.stack([torch.tensor(sample['sigma'], dtype=torch.float32) for sample in batch], dim=0)  # (batch_size, 3, 3) or (batch_size, 9)

    return {
        'p1_residue': p1_residue,
        'p1_coords': p1_coords,
        'p2_residue': p2_residue,
        'p2_coords': p2_coords,
        'p1_mask': p1_mask,
        'p2_mask': p2_mask,
        'mu': mu,
        'sigma': sigma
    }

def ellipsoid_loss(pred_params, batch, use_kl=False, mu_scale=1.0, sigma_scale=1.0, eps=1e-6):
    """
    Ellipsoid loss with fixed per-quantity scaling (stateless).
    - mu is scaled by mu_scale (divide before computing loss).
    - sigma is scaled by sigma_scale (divide before computing loss).
    No batch statistics; use fixed constants from config/hyperparams.

    Args:
        pred_params: (B, 12) [mu(3), sigma(9)]
        batch: dict with 'mu': (B, 3), 'sigma': (B, 3, 3)
        use_kl: if True, use Gaussian KL for sigma; else Frobenius MSE
        mu_scale: scalar to scale mu terms (divide in loss)
        sigma_scale: scalar to scale sigma terms (divide in loss)
        eps: numerical jitter for PD stability
    """
    B = pred_params.shape[0]

    # Split predictions
    mu_pred = pred_params[:, :3]   # (B, 3)
    sigma_pred = pred_params[:, 3:].reshape(B, 3, 3) # (B, 3, 3)

    mu_true = batch["mu"]  # (B, 3)
    sigma_true = batch["sigma"]  # (B, 3, 3)

    # Symmetrize and stabilize covariances
    sigma_pred = 0.5 * (sigma_pred + sigma_pred.transpose(-1, -2))
    sigma_true = 0.5 * (sigma_true + sigma_true.transpose(-1, -2))

    # Scale-normalize before computing losses
    mu_pred_n = mu_pred
    mu_true_n = mu_true / mu_scale

    sigma_pred_n = sigma_pred
    sigma_true_n = sigma_true / sigma_scale

    # Mean loss (MSE in normalized space)
    mu_loss = F.mse_loss(mu_pred_n, mu_true_n)

    # Sigma loss (normalized space)
    if use_kl:
        # KL between Gaussians, covariance-only part (mean error already covered above)
        I = torch.eye(3, device=sigma_true.device, dtype=sigma_true.dtype)
        sigma_pred_n = sigma_pred_n + eps * I
        sigma_true_n = sigma_true_n + eps * I

        k = mu_pred_n.shape[1]  # 3
        sigma_true_inv = torch.linalg.inv(sigma_true_n)
        trace_term = torch.einsum("bij,bjk->bik", sigma_true_inv, sigma_pred_n).diagonal(dim1=-2, dim2=-1).sum(-1)
        logdet_term = torch.logdet(sigma_true_n) - torch.logdet(sigma_pred_n)
        cov_kl = 0.5 * (trace_term - k + logdet_term)
        sigma_loss = cov_kl.mean()
    else:
        sigma_loss = F.mse_loss(sigma_pred_n, sigma_true_n)

    return mu_loss + sigma_loss


def train_epoch(model, dataloader, optimizer, ellipsoid_loss, device, epoch, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=not args.local_rank == 0)
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v 
                for k, v in batch.items()}

        # Forward pass
        outputs = model(
            batch['p1_residue'], batch['p1_coords'],
            batch['p2_residue'], batch['p2_coords'],
            p1_mask=batch['p1_mask'], p2_mask=batch['p2_mask']
        )

        # print(batch['p1_coords'].shape)
        # print(batch['p2_coords'].shape)
        # print(batch['p1_mask'].shape)
        # print(batch['p2_mask'].shape)
        # print(batch['p1_mask'])
        # print(batch['p2_mask'])
        # print(batch['p1_residue'].shape)
        # print(batch['p2_residue'].shape)
        # print(outputs['p1_feats'].shape)
        # print(outputs['p2_feats'].shape)
        # print(outputs['joint_representation'].shape)
        # print(outputs['ellipsoid_params'].shape)

        # Compute losses using the ellipsoid loss function (apply fixed scaling)
        loss = ellipsoid_loss(
            outputs['ellipsoid_params'],
            batch,
            use_kl=args.use_kl,
            mu_scale=args.mu_scale,
            sigma_scale=args.sigma_scale
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # Accumulate stats
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        if args.local_rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

    return {
        'loss': total_loss / num_batches,
    }

def print_sample_predictions(model, dataloader, device, args, num_samples=3):
    """Randomly select samples and print model predictions vs ground truth ellipsoid parameters"""
    model.eval()
    
    # Randomly select a batch
    batch = next(iter(dataloader))
    batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v 
             for k, v in batch.items()}
    
    with torch.no_grad():
        # Model prediction
        outputs = model(
            batch['p1_residue'], batch['p1_coords'],
            batch['p2_residue'], batch['p2_coords'],
            p1_mask=batch['p1_mask'], p2_mask=batch['p2_mask']
        )
        
        pred_params = outputs['ellipsoid_params']  # (batch_size, 12)
        # Denormalize predictions to original units for display
        mu_pred = pred_params[:, :3] * args.mu_scale         # (batch_size, 3)
        sigma_pred = pred_params[:, 3:].reshape(-1, 3, 3) * args.sigma_scale  # (batch_size, 3, 3)
        
        # Ground truth values
        mu_true = batch['mu']  # (batch_size, 3) - true center
        sigma_true = batch['sigma'] # (batch_size, 3, 3) - true covariance
        
        # Randomly select samples from the batch
        batch_size = pred_params.shape[0]
        sample_indices = torch.randperm(batch_size)[:num_samples]
        
        print(f"\n{'='*60}")
        print(f"Sample Predictions (Random {num_samples} samples from batch)")
        print(f"{'='*60}")
        
        for i, idx in enumerate(sample_indices):
            print(f"\nSample {i+1} (Batch Index {idx.item()}):")
            print("-" * 40)
            
            # Mu (ellipsoid center)
            print("Mu (Center):")
            print(f"  True:  [{mu_true[idx, 0]:8.3f}, {mu_true[idx, 1]:8.3f}, {mu_true[idx, 2]:8.3f}]")
            print(f"  Pred:  [{mu_pred[idx, 0]:8.3f}, {mu_pred[idx, 1]:8.3f}, {mu_pred[idx, 2]:8.3f}]")
            print(f"  Error: [{abs(mu_true[idx, 0] - mu_pred[idx, 0]):8.3f}, {abs(mu_true[idx, 1] - mu_pred[idx, 1]):8.3f}, {abs(mu_true[idx, 2] - mu_pred[idx, 2]):8.3f}]")
            
            # Sigma (covariance matrix)
            print("\nSigma (Covariance Matrix):")
            print("  True:")
            for j in range(3):
                print(f"    [{sigma_true[idx, j, 0]:8.3f}, {sigma_true[idx, j, 1]:8.3f}, {sigma_true[idx, j, 2]:8.3f}]")
            
            print("  Pred:")
            for j in range(3):
                print(f"    [{sigma_pred[idx, j, 0]:8.3f}, {sigma_pred[idx, j, 1]:8.3f}, {sigma_pred[idx, j, 2]:8.3f}]")
            
            print("  Error:")
            for j in range(3):
                print(f"    [{abs(sigma_true[idx, j, 0] - sigma_pred[idx, j, 0]):8.3f}, {abs(sigma_true[idx, j, 1] - sigma_pred[idx, j, 1]):8.3f}, {abs(sigma_true[idx, j, 2] - sigma_pred[idx, j, 2]):8.3f}]")
            
            # Calculate some statistics
            mu_mse = torch.mean((mu_true[idx] - mu_pred[idx]) ** 2).item()
            sigma_mse = torch.mean((sigma_true[idx] - sigma_pred[idx]) ** 2).item()
            
            print(f"\n  MSE - Mu: {mu_mse:.6f}, Sigma: {sigma_mse:.6f}")
            
            # Ellipsoid volume (approximate)
            try:
                # True ellipsoid volume (sqrt(det(sigma)))
                true_volume = torch.sqrt(torch.det(sigma_true[idx] + 1e-6 * torch.eye(3, device=device))).item()
                pred_volume = torch.sqrt(torch.det(sigma_pred[idx] + 1e-6 * torch.eye(3, device=device))).item()
                print(f"  Volume - True: {true_volume:.3f}, Pred: {pred_volume:.3f}, Error: {abs(true_volume - pred_volume):.3f}")
            except:
                print("  Volume calculation failed (singular matrix)")
        
        print(f"\n{'='*60}")
    
    model.train()


def analyze_ellipsoid_properties(mu, sigma):
    """Analyze main properties of the ellipsoid"""
    try:
        # Calculate eigenvalues and eigenvectors
        eigenvals, eigenvecs = torch.linalg.eigh(sigma)
        
        # Principal axis lengths (square root of eigenvalues)
        axis_lengths = torch.sqrt(torch.clamp(eigenvals, min=1e-6))
        
        # Ellipsoid volume
        volume = torch.sqrt(torch.det(sigma + 1e-6 * torch.eye(3, device=sigma.device)))
        
        # Ellipsoid shape metrics
        # Sphericity (min eigenvalue / max eigenvalue)
        sphericity = torch.min(eigenvals) / torch.max(eigenvals)
        
        return {
            'center': mu,
            'axis_lengths': axis_lengths,
            'volume': volume,
            'sphericity': sphericity,
            'eigenvals': eigenvals,
            'eigenvecs': eigenvecs
        }
    except:
        return None

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, args):
    """Save model checkpoint."""
    if args.local_rank == 0:
        try:
            model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': metrics,
                'args': vars(args)
            }
            
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            latest_path = os.path.join(args.checkpoint_dir, 'latest.pt')
            
            torch.save(checkpoint, checkpoint_path)
            torch.save(checkpoint, latest_path)
            
            print(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            print(f"Error saving checkpoint: {e}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load model checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model_state_dict = checkpoint['model_state_dict']
        
        if hasattr(model, 'module'):
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded successfully from {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['metrics']
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting training from scratch...")
        return 0, {}


def main():
    parser = argparse.ArgumentParser(description='Interface Prediction Training')
    
    # Data arguments
    parser.add_argument('--dataset_dir', type=str, default='./interface_modeling_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--model_dim', type=int, default=128, help='Model dimension')
    parser.add_argument('--model_depth', type=int, default=4, help='Model depth')
    parser.add_argument('--num_tokens', type=int, default=21, help='Number of token types')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    
    # Logging and checkpointing
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--save_freq', type=int, default=50, help='Save frequency (epochs)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Distributed training
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    
    # Wandb
    parser.add_argument('--project', type=str, default='interface_model', help='Wandb project name')
    parser.add_argument('--name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--tags', type=str, nargs='+', default=[], help='Wandb tags')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    # Fixed scaling for ellipsoid loss and prediction
    parser.add_argument('--mu_scale', type=float, default=64.0, help='Fixed scale for mu (center) used in loss and denormalization')
    parser.add_argument('--sigma_scale', type=float, default=128.0, help='Fixed scale for sigma (covariance) used in loss and denormalization')

    # Loss options
    parser.add_argument('--use_kl', action='store_true', default=False, help='Use KL divergence for sigma term (default: False). If not set, use Frobenius MSE')
    
    args = parser.parse_args()
    
    # Set up distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
    
    # Set device
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}')
    
    # Initialize distributed training
    if args.world_size > 1:
        dist.init_process_group(backend='nccl')
        dist.barrier()
    
    # Set seed
    set_seed(args.seed + args.rank)
    
    # Set up logging
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        if not args.debug:
            wandb.init(
                project=args.project,
                name=args.name,
                tags=args.tags,
                config=vars(args)
            )
    
    # Load dataset
    dataset = load_from_disk(args.dataset_dir)
    
    # Use the entire dataset for training (no validation split)
    train_dataset = dataset
    
    # Create data loader
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if args.world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    # Create model
    model = InterfaceModel(
        num_tokens=args.num_tokens,
        dim=args.model_dim,
        depth=args.model_depth
    ).to(device)
    
    if args.world_size > 1:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
    
    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume from checkpoint
    start_epoch = 0
    
    if args.resume:
        start_epoch, metrics = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        if args.local_rank == 0:
            print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_metrics = train_epoch(model, train_loader, optimizer, ellipsoid_loss, device, epoch, args)
        
        # Log metrics and print sample predictions
        if args.local_rank == 0:
            if not args.debug:
                wandb.log(train_metrics, step=epoch)
            print(f"Epoch {epoch}: {train_metrics}")
            
            # Print sample predictions to monitor training progress
            print_sample_predictions(model, train_loader, device, args, num_samples=3)
        
        # Save checkpoint periodically
        if epoch % args.save_freq == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_metrics, args)
        
        # Update learning rate scheduler
        scheduler.step()
    
    # Cleanup
    if args.world_size > 1:
        dist.destroy_process_group()
    
    if args.local_rank == 0 and not args.debug:
        wandb.finish()


if __name__ == '__main__':
    main()