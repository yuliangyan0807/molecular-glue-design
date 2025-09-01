from Bio.PDB import PDBParser
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import torch


def get_pocket_and_mask(p1_coords: torch.Tensor, p2_coords: torch.Tensor, cutoff=8.):
    """
    get pocket coords and mask of p1 and p2.
    The pocket coords is the coords of p1.
    Return:
        pocket_coords: (N, 3), float
        mask_p1: (N, 1), bool
        mask_p2: (M, 1), bool
    """
    # get distance matrix
    dist_mat = torch.cdist(p1_coords, p2_coords)
    # get mask
    mask_p1 = dist_mat.min(dim=1).values < cutoff
    mask_p2 = dist_mat.min(dim=0).values < cutoff
    # get pocket coords
    pocket_coords = p1_coords[mask_p1]
    return pocket_coords, mask_p1, mask_p2

def get_interface_from_graphs(protein1_coords, protein2_coords, cutoff=8.0):
    """
    Extract interface residues and coordinates from two protein coordinate tensors.
    
    Args:
        protein1_coords: First protein coordinates tensor (N, 3)
        protein2_coords: Second protein coordinates tensor (M, 3)
        cutoff: Distance cutoff for interface detection (default: 8.0Å)
    
    Returns:
        interface_coords: Coordinates of interface atoms from both proteins
        p1_interface_mask: Boolean mask for p1 interface atoms
        p2_interface_mask: Boolean mask for p2 interface atoms
        p1_interface_residues: Residue indices for p1 interface residues
        p2_interface_residues: Residue indices for p2 interface residues
    """
    # Calculate distance matrix between all atoms
    dist_mat = torch.cdist(protein1_coords, protein2_coords)  # Shape: (N, M)
    
    # Find atoms within cutoff distance
    p1_interface_mask = dist_mat.min(dim=1).values < cutoff  # Shape: (N,)
    p2_interface_mask = dist_mat.min(dim=0).values < cutoff  # Shape: (M,)
    
    # Get interface coordinates from both proteins
    p1_interface_coords = protein1_coords[p1_interface_mask]  # Shape: (K, 3) where K <= N
    p2_interface_coords = protein2_coords[p2_interface_mask]  # Shape: (L, 3) where L <= M
    
    # Combine interface coordinates from both proteins
    if len(p1_interface_coords) > 0 and len(p2_interface_coords) > 0:
        interface_coords = torch.cat([p1_interface_coords, p2_interface_coords], dim=0)
    elif len(p1_interface_coords) > 0:
        interface_coords = p1_interface_coords
    elif len(p2_interface_coords) > 0:
        interface_coords = p2_interface_coords
    else:
        interface_coords = torch.empty((0, 3), dtype=protein1_coords.dtype, device=protein1_coords.device)
    
    # Get residue indices for interface atoms
    # Since we don't have explicit residue information, we'll use atom indices as proxy
    # In practice, you might want to map these to actual residue IDs
    p1_interface_residues = torch.where(p1_interface_mask)[0]  # Atom indices that form interface
    p2_interface_residues = torch.where(p2_interface_mask)[0]  # Atom indices that form interface
    
    return interface_coords, p1_interface_mask, p2_interface_mask, p1_interface_residues, p2_interface_residues

def get_elilipsoid_for_interface(coords: torch.Tensor):
    """
    Fit an ellipsoid (Gaussian distribution) to interface coordinates.
    
    Args:
        coords: Interface coordinates tensor (N, 3) where N is the number of interface atoms
    
    Returns:
        mean: Mean vector (3,) representing the center of the ellipsoid
        covariance: Covariance matrix (3, 3) representing the shape and orientation of the ellipsoid
    """
    # Calculate mean (center of the ellipsoid)
    mean = torch.mean(coords, dim=0)  # Shape: (3,)
    
    # Center the coordinates
    centered_coords = coords - mean.unsqueeze(0)  # Shape: (N, 3)
    
    # Calculate covariance matrix
    # For numerical stability, we use the unbiased estimator
    if len(coords) > 1:
        # Unbiased estimator: divide by (N-1) instead of N
        covariance = torch.matmul(centered_coords.T, centered_coords) / (len(coords) - 1)  # Shape: (3, 3)
    else:
        # Single point case, use identity matrix
        covariance = torch.eye(3, dtype=coords.dtype, device=coords.device)
    
    # Ensure covariance matrix is symmetric and positive semi-definite
    # Make it symmetric
    covariance = (covariance + covariance.T) / 2
    
    # Add small regularization to ensure positive definiteness
    epsilon = 1e-6
    covariance = covariance + epsilon * torch.eye(3, dtype=coords.dtype, device=coords.device)
    
    return mean, covariance

if __name__ == "__main__":
    # Define the MGD_Train directory path
    mgd_train_dir = "./data/TernaryDB/MGD_Train"