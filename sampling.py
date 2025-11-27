#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import json
from copy import deepcopy

from datasets import load_from_disk
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
    """Collate function for batching (same as in flow_training.py)"""
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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Sample from Flow Matching Model')
    
    # Config file
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (optional, uses random init if not provided)')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to dataset (overrides config)')
    
    # Sampling parameters
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples per input')
    parser.add_argument('--num_steps', type=int, default=None,
                        help='Number of sampling steps (overrides config)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for sampling')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to generate (None = all)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./samples',
                        help='Output directory for samples')
    parser.add_argument('--save_trajectory', action='store_true',
                        help='Save full sampling trajectory')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    # Seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def tensor_to_list(tensor_or_array):
    """Convert tensor or numpy array to list for JSON serialization"""
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().numpy().tolist()
    elif hasattr(tensor_or_array, 'numpy'):
        return tensor_or_array.numpy().tolist()
    elif hasattr(tensor_or_array, 'tolist'):
        return tensor_or_array.tolist()
    else:
        return tensor_or_array


def save_sample(sample_data, output_dir, sample_idx, name=None):
    """Save a single sample to disk as log file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save final sample
    sample_name = name if name else f'sample_{sample_idx:06d}'
    sample_path = os.path.join(output_dir, f'{sample_name}.log')
    
    # Prepare data to save (convert tensors to lists for JSON serialization)
    save_dict = {
        'lig_seq': tensor_to_list(sample_data['lig_seq']),
        'lig_coords': tensor_to_list(sample_data['lig_coords']),
        'rotmats': tensor_to_list(sample_data['rotmats']),
        'trans': tensor_to_list(sample_data['trans']),
    }
    
    # Add ground truth if available
    if 'lig_seq_1' in sample_data:
        save_dict['lig_seq_gt'] = tensor_to_list(sample_data['lig_seq_1'])
    if 'lig_coords_1' in sample_data:
        save_dict['lig_coords_gt'] = tensor_to_list(sample_data['lig_coords_1'])
    if 'rotmats_1' in sample_data:
        save_dict['rotmats_gt'] = tensor_to_list(sample_data['rotmats_1'])
    if 'trans_1' in sample_data:
        save_dict['trans_gt'] = tensor_to_list(sample_data['trans_1'])
    
    # Save as JSON log file
    with open(sample_path, 'w') as f:
        json.dump(save_dict, f, indent=2)
    
    return sample_path


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Determine device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load configuration
    print("Loading configuration...")
    config = load_config_from_yaml(args.config)
    print(f"✓ Configuration loaded from {args.config}")
    
    # Override dataset path if provided
    if args.dataset is not None:
        config.dataset.path = args.dataset
    
    # Override num_steps if provided
    if args.num_steps is not None:
        if not hasattr(config, 'sampling'):
            config.sampling = DictToObject({})
        config.sampling.num_steps = args.num_steps
    
    # Load dataset
    print("Loading dataset...")
    try:
        dataset_path = config.dataset.path
        print(f"  Loading from: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        print(f"✓ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return
    
    # Limit dataset size if specified
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"  Limited to {len(dataset)} samples")
    
    # Create model
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
        'interpolant': config.model.interpolant,
        'sampling': getattr(config, 'sampling', DictToObject({'num_steps': 100}))
    })
    
    model = TernaryFlowModel(full_config)
    model = model.to(device)
    
    # Load checkpoint if provided
    if args.checkpoint is not None:
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        print("✓ Model loaded from checkpoint and set to eval mode")
    else:
        print("⚠ No checkpoint provided - using randomly initialized model")
        print("  (This is useful for testing, but results will be random)")
    
    model.eval()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Sampling loop
    print(f"\nStarting sampling...")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Samples per input: {args.num_samples}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output directory: {args.output_dir}")
    
    # Randomly select indices if max_samples is set
    if args.max_samples is not None and args.max_samples < len(dataset):
        import random
        indices = random.sample(range(len(dataset)), args.max_samples)
        print(f"  Randomly selected {args.max_samples} samples from dataset")
    else:
        indices = list(range(min(len(dataset), args.max_samples or len(dataset))))
    
    total_samples = 0
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Sampling"):
            # Get single item from dataset
            item = dataset[idx]
            
            # Create batch by replicating the item
            batch_items = [item] * args.num_samples
            batch = collate_fn(batch_items)
            
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Sample
            try:
                traj = model.sample(batch)
                
                # Get final sample (last element of trajectory)
                final_sample = traj[-1]
                
                # Save samples
                for i in range(args.num_samples):
                    sample_data = {
                        'lig_seq': final_sample['lig_seq'][i],
                        'lig_coords': final_sample['lig_coords'][i],
                        'rotmats': final_sample['rotmats'][i],
                        'trans': final_sample['trans'][i],
                    }
                    
                    # Add ground truth if available
                    if 'lig_seq_1' in final_sample:
                        sample_data['lig_seq_1'] = final_sample['lig_seq_1'][i]
                    if 'lig_coords_1' in final_sample:
                        sample_data['lig_coords_1'] = final_sample['lig_coords_1'][i]
                    if 'rotmats_1' in final_sample:
                        sample_data['rotmats_1'] = final_sample['rotmats_1'][i]
                    if 'trans_1' in final_sample:
                        sample_data['trans_1'] = final_sample['trans_1'][i]
                    
                    # Get name from dataset if available
                    name = None
                    if 'name' in item:
                        name = f"{item['name']}_sample_{i}"
                    
                    save_sample(sample_data, args.output_dir, total_samples, name)
                    total_samples += 1
                
                # Save full trajectory if requested
                if args.save_trajectory:
                    traj_path = os.path.join(args.output_dir, f'trajectory_{idx:06d}.log')
                    # Convert trajectory to JSON-serializable format
                    traj_dict = {}
                    for step_idx, step in enumerate(traj):
                        traj_dict[f'step_{step_idx}'] = {
                            'lig_seq': tensor_to_list(step['lig_seq']),
                            'lig_coords': tensor_to_list(step['lig_coords']),
                            'rotmats': tensor_to_list(step['rotmats']),
                            'trans': tensor_to_list(step['trans']),
                        }
                    with open(traj_path, 'w') as f:
                        json.dump(traj_dict, f, indent=2)
                
            except Exception as e:
                print(f"\n✗ Error sampling from item {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Clear cache periodically
            if (idx + 1) % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n✓ Sampling completed!")
    print(f"  Total samples saved: {total_samples}")
    print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
    