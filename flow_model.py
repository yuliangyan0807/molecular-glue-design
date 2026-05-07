# Copyright (c) 2024. This code is built upon https://github.com/Ced3-han/PepFlowww
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.so3_utils import geodesic_t, uniform_so3, calc_rot_vf, geodesic_dist, project_to_so3
from models.vf_model import VFModel
from models.interface_model import InterfaceModel, MultiHeadInterfaceModel

from utils.constants import BBHeavyAtom, TRANS_MEAN, TRANS_STD
from utils.rigid_utils import get_conditioned_coords, get_rigid_transform, kabsch_align

# Helper functions
def clampped_one_hot(x, num_classes):
    mask = (x >= 0) & (x < num_classes) # (N, L)
    x = x.clamp(min=0, max=num_classes-1)
    y = F.one_hot(x, num_classes) * mask[...,None]  # (N, L, C)
    return y

def sample_from(c):
    """sample from c"""
    N,L,K = c.size()
    c = c.view(N * L, K) + 1e-8
    x = torch.multinomial(c, 1).view(N, L)
    return x

def move_dict_to_device(d, device):
    """Move all tensors in a dictionary to the specified device"""
    if isinstance(d, dict):
        return {k: move_dict_to_device(v, device) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        return d.to(device)
    else:
        return d

def center_coords(protein_coords, ligand_coords, protein_mask, ligand_mask, mode='protein'):
    """
    Center coordinates to reduce numerical instability, similar to TargetDiff.
    
    Args:
        protein_coords: (B, N, 3) protein CA coordinates
        ligand_coords: (B, L, 3) ligand coordinates
        protein_mask: (B, N) protein residue mask
        ligand_mask: (B, L) ligand atom mask
        mode: 'protein' - center to protein centroid, 'none' - no centering
    
    Returns:
        centered_protein_coords: (B, N, 3)
        centered_ligand_coords: (B, L, 3)
        offset: (B, 3) offset that was subtracted
    """
    if mode == 'none':
        return protein_coords, ligand_coords, torch.zeros(protein_coords.shape[0], 3, device=protein_coords.device)
    
    elif mode == 'protein':
        # Compute protein centroid for each sample in batch
        # Use mask to exclude padding
        protein_mask_expanded = protein_mask.unsqueeze(-1).float()  # (B, N, 1)
        protein_sum = (protein_coords * protein_mask_expanded).sum(dim=1)  # (B, 3)
        protein_count = protein_mask.sum(dim=1, keepdim=True).float()  # (B, 1)
        offset = protein_sum / (protein_count + 1e-8)  # (B, 3)
        
        # Center both protein and ligand coordinates
        centered_protein_coords = protein_coords - offset.unsqueeze(1)  # (B, N, 3)
        centered_ligand_coords = ligand_coords - offset.unsqueeze(1)  # (B, L, 3)
        
        # Apply masks to zero out padding positions
        centered_protein_coords = centered_protein_coords * protein_mask_expanded
        centered_ligand_coords = centered_ligand_coords * ligand_mask.unsqueeze(-1).float()
        
        return centered_protein_coords, centered_ligand_coords, offset
    
    else:
        raise ValueError(f"Unknown center mode: {mode}")

class TernaryFlowModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self._cfg = cfg

        self._interpolant_cfg = cfg.interpolant

        num_classes = self._interpolant_cfg.seqs.num_classes
        self.vf_model = VFModel(cfg.model, num_classes=num_classes)
        
        # Create interface model with config parameters
        interface_model = MultiHeadInterfaceModel(
            num_tokens=22,  # 20 amino acids + 2 special tokens
            dim=getattr(cfg.model.interface_model, 'feat_dim', 128),
            depth=getattr(cfg.model.interface_model, 'depth', 4),
            num_nearest_neighbors=getattr(cfg.model.interface_model, 'num_nearest_neighbors', 16),
            num_att_heads=getattr(cfg.model.interface_model, 'num_att_heads', 50),  # K: number of keypoints, default 50 to match checkpoint
        )
        
        # Load trained weights if path is provided
        if hasattr(cfg.model.interface_model, 'path') and cfg.model.interface_model.path:
            checkpoint_path = cfg.model.interface_model.path
            print(f"Loading interface model from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    interface_model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model' in checkpoint:
                    interface_model.load_state_dict(checkpoint['model'])
                else:
                    # Assume the checkpoint is the state_dict itself
                    interface_model.load_state_dict(checkpoint)
            else:
                # If checkpoint is the model itself (not recommended but possible)
                interface_model = checkpoint
            
            print("✓ Interface model loaded successfully")
            # Set to eval mode for inference
            interface_model.eval()
        else:
            print("⚠ No interface model path provided - using randomly initialized model")
        
        # Check if interface model should be trainable
        self.interface_trainable = getattr(cfg.model.interface_model, 'trainable', False)
        
        # Freeze interface model parameters if not trainable
        if not self.interface_trainable:
            for param in interface_model.parameters():
                param.requires_grad = False
            print("✓ Interface model parameters frozen (not trainable)")
        else:
            print("✓ Interface model parameters are trainable")
        
        self.interface_model = interface_model

        self.K = self._interpolant_cfg.seqs.num_classes
        self.k = self._interpolant_cfg.seqs.simplex_value
        
        # Top-k value for selecting interface residues based on attention
        self.topk_k = getattr(cfg.model.interface_model, 'topk_k', 50)
        
        # Learnable scale for t_star (translation from Kabsch alignment)
        self.alpha = nn.Parameter(torch.ones(1))
    
    def seq_to_simplex(self,seqs):
        return clampped_one_hot(seqs, self.K).float() * self.k * 2 - self.k # (B,L,K)

    def forward(self, batch):
        
        # Ground truth at time step 1.
        lig_seq_1, lig_coords_1, rotmats_1, trans_1 = batch['lig_seq_1'], batch['lig_coords_1'], batch['R_inv_1'], batch['t_inv_1']
        lig_seq_1_simplex = self.seq_to_simplex(lig_seq_1)
        lig_seq_1_prob = F.softmax(lig_seq_1_simplex,dim=-1)

        rotmats_1 = rotmats_1.unsqueeze(1) # (B, 1, 3, 3)
        trans_1 = trans_1.unsqueeze(1) # (B, 1, 3)
        # Normalize translation to mean 0 and std 1
        # trans_1 = (trans_1 - TRANS_MEAN.to(trans_1.device)) / TRANS_STD.to(trans_1.device)

        num_batch, num_lig = lig_seq_1.shape[0], lig_seq_1.shape[1]

        with torch.no_grad():
            t = torch.rand((num_batch, 1), device=batch['lig_seq_1'].device)
            t = t * (1 - 2 * self._interpolant_cfg.min_t) + self._interpolant_cfg.min_t # avoid 0

            # Randomly sample a translation vector from a normal distribution.
            lig_coords_0 = torch.randn((num_batch, num_lig, 3), device=batch['lig_seq_1'].device)
            lig_coords_t = (1 - t[...,None]) * lig_coords_0 + t[...,None] * lig_coords_1

            # Randomly sample a rotation matrix from the uniform distribution on SO(3).
            rotmats_0 = uniform_so3(num_batch, 1, device=batch['lig_seq_1'].device) # (B, 1, 3, 3)
            # Obtein the rotation matrix at time t with exponential map.
            rotmats_t = geodesic_t(t[..., None], rotmats_1, rotmats_0)

            trans_0 = torch.randn((num_batch, 1, 3), device=batch['lig_seq_1'].device) * self._interpolant_cfg.trans.sigma # scale with sigma?
            # Optimal transport.
            # trans_0 = TRANS_STD.to(trans_0.device) * trans_0 + TRANS_MEAN.to(trans_0.device)
            trans_t = (1 - t[...,None]) * trans_0 + t[...,None] * trans_1

            # Randomly sample a sequence from the uniform distribution.
            lig_seq_0_simplex = self.k * torch.randn_like(lig_seq_1_simplex) # (B,L,K)
            lig_seq_0_prob = F.softmax(lig_seq_0_simplex, dim=-1) # (B,L,K)
            lig_seq_t_simplex = ((1 - t[..., None]) * lig_seq_0_simplex) + (t[..., None] * lig_seq_1_simplex) # (B,L,K)
            lig_seq_t_prob = F.softmax(lig_seq_t_simplex, dim=-1) # (B,L,K)
            lig_seq_t = sample_from(lig_seq_t_prob) # (B,L)

        p1, p2 = batch['p1'], batch['p2']
        p1_mask = p1['res_mask']
        p2_mask = p2['res_mask']
        p1_residue = p1['aa']
        p2_residue = p2['aa']
        p1_ca_coords = p1['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        p2_ca_coords = p2['pos_heavyatom'][:, :, BBHeavyAtom.CA]

        mol_mask = batch['mol_mask']
        
        # Use no_grad if interface model is not trainable
        if self.interface_trainable:
            output = self.interface_model(
                p1_residue=p1_residue,
                p1_coords=p1_ca_coords,
                p2_residue=p2_residue,
                p2_coords=p2_ca_coords,
                p1_mask=p1_mask,
                p2_mask=p2_mask
            )
        else:
            with torch.no_grad():
                output = self.interface_model(
                    p1_residue=p1_residue,
                    p1_coords=p1_ca_coords,
                    p2_residue=p2_residue,
                    p2_coords=p2_ca_coords,
                    p1_mask=p1_mask,
                    p2_mask=p2_mask
                )
        Y1, Y2 = output['Y1'], output['Y2'] # (B, K, 3)
        mu1, mu2 = output['mu1'], output['mu2'] # (B, 3)
        sigma_1, sigma_2 = output['cov1'], output['cov2'] # (B, 3, 3)

        # Differentiable Kabsch Alignment from EquiDock
        # Y1: Receptor (p1), Y2: Target (p2) interface
        # Center the coordinates
        R_star, t_star = kabsch_align(Y1, Y2)
        p2_coords_moved = p2_ca_coords @ R_star.transpose(1, 2) + (self.alpha * t_star).unsqueeze(1)  # (B, N2, 3)
        # p2_coords_moved = p2_ca_coords + mu2.unsqueeze(1)
        
        # Denoise
        # Compute the vector field and predict the raw data at time step 1.
        pred_lig_seq_1_prob, pred_lig_coords_1, pred_rotmats_1, pred_trans_1 = self.vf_model(
            p1=p1, p2=p2, t=t, lig_coords_t=lig_coords_t, rotmats_t=rotmats_t, trans_t=trans_t, lig_seq_t=lig_seq_t,
            mol_mask=mol_mask, p2_coords_moved=p2_coords_moved,
            sigma_1=sigma_1, sigma_2=sigma_2,
            R_star=R_star, t_star=t_star,
            Y1=Y1, Y2=Y2
        )
        pred_lig_seq_1 = sample_from(F.softmax(pred_lig_seq_1_prob, dim=-1))

        norm_scale = 1 / (1 - torch.min(t[...,None], torch.tensor(self._interpolant_cfg.t_normalization_clip))) # yim etal.trick, 1/1-t

        # Translation Flow Matching loss, Euclidean Flow.
        trans_loss = torch.sum((pred_trans_1.unsqueeze(1) - trans_1) ** 2, dim=-1) # (B, )
        trans_loss = torch.mean(trans_loss)
        # trans_loss = F.mse_loss(pred_trans_1.unsqueeze(1), trans_1)
        # trans_loss = F.huber_loss(pred_trans_1.unsqueeze(1), trans_1, delta=10.0)

        # Rotation Flow Matching loss, Riemannian Flow on SO(3).
        # Rotation loss: SO(3) projection + geodesic on manifold + orth/det regularization.
        # pred_rotmats_1: (B, 3, 3), rotmats_1: (B, 1, 3, 3)
        gt_rot_vf = calc_rot_vf(rotmats_t, rotmats_1)
        pred_rot_vf = calc_rot_vf(rotmats_t, pred_rotmats_1)
        rot_loss = torch.sum(((gt_rot_vf - pred_rot_vf) * norm_scale) ** 2, dim=-1) # (B, )
        rot_loss = torch.mean(rot_loss)
        # rot_loss = F.mse_loss(pred_rotmats_1.unsqueeze(1), rotmats_1)

        # Simplex Flow Matching loss, Simplex Flow on (K - 1)-simplex.
        # pred_lig_seq_1_prob: (B, L, K)
        # Flatten for cross_entropy
        pred_flat = pred_lig_seq_1_prob.view(-1, pred_lig_seq_1_prob.size(-1))  # (B * L, K)
        target_flat = lig_seq_1.view(-1)  # (B * L,)
        mask_flat = mol_mask.view(-1)  # (B * L,)
        
        # Calculate per-position loss (reduction='none' to get per-position losses)
        seqs_loss_per_pos = F.cross_entropy(
            pred_flat, target_flat, reduction='none'  # (B * L,)
        )
        # Apply mask to ignore padding positions
        seqs_loss_per_pos = seqs_loss_per_pos * mask_flat.float()  # (B * L,)
        # Reshape back to (B, L)
        seqs_loss_per_pos = seqs_loss_per_pos.view(pred_lig_seq_1_prob.shape[0], -1)  # (B, L)
        # Average over positions for each molecule
        seqs_loss_per_mol = seqs_loss_per_pos.sum(dim=-1) / mol_mask.sum(dim=-1).float()  # (B,)
        # Average over batch
        seqs_loss = torch.mean(seqs_loss_per_mol)

        # 3D Coordinates Flow Matching loss, Euclidean Flow.
        # Calculate per-atom squared errors: (B, L)
        atom_errors = torch.sum((pred_lig_coords_1 - lig_coords_1) ** 2, dim=-1)  # (B, L)
        # Apply mask to ignore padding positions
        atom_errors = atom_errors * mol_mask.float()  # (B, L)
        # Average over atoms for each molecule
        coords_loss_per_mol = atom_errors.sum(dim=-1) / mol_mask.sum(dim=-1).float()  # (B,)
        # Average over batch
        coords_loss = torch.mean(coords_loss_per_mol)

        return {
            'trans_loss': trans_loss,
            'rot_loss': rot_loss,
            'seqs_loss': seqs_loss,
            'coords_loss': coords_loss,
        }

    @torch.no_grad()
    def sample(self, batch):
        # Setup
        num_batch = batch['lig_seq_1'].shape[0]
        num_lig = batch['lig_seq_1'].shape[1]

        # Ground truth at time step 1 (used as targets predicted by the network)
        lig_seq_1, lig_coords_1, rotmats_1, trans_1 = batch['lig_seq_1'], batch['lig_coords_1'], batch['R_inv_1'], batch['t_inv_1']
        lig_seq_1_simplex = self.seq_to_simplex(lig_seq_1)
        rotmats_1 = rotmats_1.unsqueeze(1)  # (B, 1, 3, 3)
        trans_1 = trans_1.unsqueeze(1)      # (B, 1, 3)

        mol_mask = batch['mol_mask']
        p1, p2 = batch['p1'], batch['p2']
        
        # Ensure p1 and p2 are on the correct device
        device = lig_seq_1.device
        p1 = move_dict_to_device(p1, device)
        p2 = move_dict_to_device(p2, device)
        mol_mask = mol_mask.to(device)
        
        # Compute interface masks (same as in forward)
        p1_mask = p1['res_mask']
        p2_mask = p2['res_mask']
        p1_residue = p1['aa']
        p2_residue = p2['aa']
        p1_ca_coords = p1['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        p2_ca_coords = p2['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        
        # Use no_grad if interface model is not trainable
        if self.interface_trainable:
            output = self.interface_model(
                p1_residue=p1_residue,
                p1_coords=p1_ca_coords,
                p2_residue=p2_residue,
                p2_coords=p2_ca_coords,
                p1_mask=p1_mask,
                p2_mask=p2_mask
            )
        else:
            with torch.no_grad():
                output = self.interface_model(
                    p1_residue=p1_residue,
                    p1_coords=p1_ca_coords,
                    p2_residue=p2_residue,
                    p2_coords=p2_ca_coords,
                    p1_mask=p1_mask,
                    p2_mask=p2_mask
                )
        Y1, Y2 = output['Y1'], output['Y2'] # (B, K, 3)
        mu1, mu2 = output['mu1'], output['mu2'] # (B, 3)
        sigma_1, sigma_2 = output['cov1'], output['cov2'] # (B, 3, 3)

        # Differentiable Kabsch Alignment from EquiDock
        # Y1: Receptor (p1), Y2: Target (p2) interface
        R_star, t_star = kabsch_align(Y1, Y2)

        # TDOO
        # R_star = uniform_so3(num_batch, 1, device=p2_ca_coords.device).squeeze(1)
        # t_star = torch.randn(num_batch, 3, device=p2_ca_coords.device, dtype=p2_ca_coords.dtype) * 1

        # Interface-Corrected coordinates for p2
        # p2_ca_coords: (B, N2, 3), R_star: (B, 3, 3), t_star: (B, 3)
        # Apply rotation: (B, N2, 3) @ (B, 3, 3)^T = (B, N2, 3) @ (B, 3, 3) = (B, N2, 3)
        p2_coords_moved = p2_ca_coords @ R_star.transpose(1, 2) + (self.alpha * t_star).unsqueeze(1)  # (B, N2, 3)
        # p2_coords_moved = p2_ca_coords
        
        # Time schedule
        num_steps = getattr(self._cfg.sampling, 'num_steps', 100)
        ts = torch.linspace(1.0e-2, 1.0, num_steps, device=lig_seq_1.device)
        t_1 = ts[0]

        # Initial noise at t ~ 0
        rotmats_0 = uniform_so3(num_batch, 1, device=lig_seq_1.device)   # (B, 1, 3, 3)
        trans_0 = torch.randn((num_batch, 1, 3), device=lig_seq_1.device) * self._interpolant_cfg.trans.sigma # (B, 1, 3)
        # Optimal transport.
        # trans_0 = TRANS_STD.to(trans_0.device) * trans_0 + TRANS_MEAN.to(trans_0.device)
        lig_coords_0 = torch.randn((num_batch, num_lig, 3), device=lig_seq_1.device)  # (B, L, 3)
        lig_seq_0_simplex = self.k * torch.randn_like(lig_seq_1_simplex)  # (B, L, K)
        lig_seq_0_prob = F.softmax(lig_seq_0_simplex, dim=-1)
        lig_seq_0 = sample_from(lig_seq_0_prob)  # (B, L)

        # States at current time step t_1
        rotmats_t_1 = rotmats_0.squeeze(1) # (B, 3, 3)
        trans_t_1 = trans_0 # (B, 1, 3)
        lig_coords_t_1 = lig_coords_0
        lig_seq_t_1 = lig_seq_0
        lig_seq_t_1_simplex = lig_seq_0_simplex

        clean_traj = []

        # Denoise loop
        for t_2 in ts[1:]:
            t = torch.ones((num_batch, 1), device=lig_seq_1.device) * t_1

            # Predict raw data at time step 1 given the current noisy state at time t
            pred_lig_seq_1_prob, pred_lig_coords_1, pred_rotmats_1, pred_trans_1 = self.vf_model(
                t=t,
                lig_coords_t=lig_coords_t_1,
                rotmats_t=rotmats_t_1,
                trans_t=trans_t_1,
                lig_seq_t=lig_seq_t_1,
                p1=p1,
                p2=p2,
                sigma_1=sigma_1, sigma_2=sigma_2,
                mol_mask=mol_mask, p2_coords_moved=p2_coords_moved,
                R_star=R_star, t_star=t_star,
                Y1=Y1, Y2=Y2
            )
            pred_rotmats_1 = project_to_so3(pred_rotmats_1)

            pred_lig_seq_1 = sample_from(F.softmax(pred_lig_seq_1_prob, dim=-1))
            pred_lig_seq_1_simplex = self.seq_to_simplex(pred_lig_seq_1)

            # Record trajectory (CPU for memory safety downstream)
            clean_traj.append({
                'rotmats': pred_rotmats_1.detach().cpu(),
                'trans': pred_trans_1.detach().cpu(),
                'lig_coords': pred_lig_coords_1.detach().cpu(),
                'lig_seq': pred_lig_seq_1.detach().cpu(),
                'lig_seq_simplex': pred_lig_seq_1_simplex.detach().cpu(),
                'rotmats_1': rotmats_1.detach().cpu(),
                'trans_1': trans_1.detach().cpu(),
                'lig_coords_1': lig_coords_1.detach().cpu(),
                'lig_seq_1': lig_seq_1.detach().cpu(),
            })

            # Euler step along the flow from t_1 to t_2
            d_t = (t_2 - t_1) * torch.ones((num_batch, 1), device=lig_seq_1.device)  # (B, 1)

            # Translation and coordinates (Euclidean)
            # trans_t_1: (B, 1, 3), pred_trans_1: (B, 3), trans_0: (B, 1, 3), d_t: (B, 1)
            trans_t_2 = trans_t_1 + (pred_trans_1.unsqueeze(1) - trans_0) * d_t[..., None]  # (B, 1, 3)

            lig_coords_t_2 = lig_coords_t_1 + (pred_lig_coords_1 - lig_coords_0) * d_t[..., None]  # (B, L, 3) * (B, 1, 1) -> (B, L, 3)
            lig_coords_t_2 = lig_coords_t_2 * mol_mask.unsqueeze(-1)  # Mask out padding positions

            # Rotation (SO(3) geodesic step)
            # geodesic_t expects: t [B,1,1], mat [B,L,3,3], base_mat [B,L,3,3]
            rotmats_t_2 = geodesic_t(d_t[..., None] * 10, pred_rotmats_1.unsqueeze(1), rotmats_t_1.unsqueeze(1))
            rotmats_t_2 = rotmats_t_2.squeeze(1)  # (B, 1, 3, 3) -> (B, 3, 3) to match rotmats_t_1 shape

            # Sequences (simplex)
            lig_seq_t_2_simplex = lig_seq_t_1_simplex + (pred_lig_seq_1_simplex - lig_seq_0_simplex) * d_t[..., None]
            lig_seq_t_2_simplex = lig_seq_t_2_simplex * mol_mask.unsqueeze(-1)  # Mask out padding positions
            lig_seq_t_2 = sample_from(F.softmax(lig_seq_t_2_simplex, dim=-1))

            # Move to next step
            rotmats_t_1, trans_t_1, lig_coords_t_1, lig_seq_t_1, lig_seq_t_1_simplex = (
                rotmats_t_2, trans_t_2, lig_coords_t_2, lig_seq_t_2, lig_seq_t_2_simplex
            )
            t_1 = t_2

        # Final step at t=1
        t = torch.ones((num_batch, 1), device=lig_seq_1.device) * ts[-1]
        pred_lig_seq_1_prob, pred_lig_coords_1, pred_rotmats_1, pred_trans_1 = self.vf_model(
            t=t,
            lig_coords_t=lig_coords_t_1,
            rotmats_t=rotmats_t_1,
            trans_t=trans_t_1,
            lig_seq_t=lig_seq_t_1,
            p1=p1,
            p2=p2,
            sigma_1=sigma_1, sigma_2=sigma_2,
            mol_mask=mol_mask, p2_coords_moved=p2_coords_moved,
            R_star=R_star, t_star=t_star,
            Y1=Y1, Y2=Y2
        )
        pred_rotmats_1 = project_to_so3(pred_rotmats_1)

        # pred_lig_seq_1 = sample_from(F.softmax(pred_lig_seq_1_prob, dim=-1))
        pred_lig_seq_1 = pred_lig_seq_1_prob.argmax(dim=-1)
        pred_lig_seq_1_simplex = self.seq_to_simplex(pred_lig_seq_1)

        # pred_trans_1 = pred_trans_1 * TRANS_STD.to(pred_trans_1.device) + TRANS_MEAN.to(pred_trans_1.device)

        clean_traj.append({
            'rotmats': pred_rotmats_1.detach().cpu(),
            'trans': pred_trans_1.detach().cpu(),
            'lig_coords': pred_lig_coords_1.detach().cpu(),
            'lig_seq': pred_lig_seq_1.detach().cpu(),
            'lig_seq_simplex': pred_lig_seq_1_simplex.detach().cpu(),
            'rotmats_1': rotmats_1.detach().cpu(),
            'trans_1': trans_1.detach().cpu(),
            'lig_coords_1': lig_coords_1.detach().cpu(),
            'lig_seq_1': lig_seq_1.detach().cpu(),
        })

        return clean_traj