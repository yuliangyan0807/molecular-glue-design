# Copyright (c) 2024. This code is built upon https://github.com/Ced3-han/PepFlowww
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.so3_utils import geodesic_t, uniform_so3, calc_rot_vf
from models.vf_model import VFModel
from models.interface_model import InterfaceModel

# collate_fn = PaddingCollate(eight=False)

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

class TernaryFlowModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self._cfg = cfg

        self._interpolant_cfg = cfg.interpolant

        self.vf_model = VFModel(cfg.model)
        self.interface_model = InterfaceModel()

        self.K = self._interpolant_cfg.seqs.num_classes
        self.k = self._interpolant_cfg.seqs.simplex_value
    
    def seq_to_simplex(self,seqs):
        return clampped_one_hot(seqs, self.K).float() * self.k * 2 - self.k # (B,L,K)

    def forward(self, batch):
        
        # Ground truth at time step 1.
        lig_seq_1, lig_coords_1, rotmats_1, trans_1 = batch['lig_seq_1'], batch['lig_coords_1'], batch['R_inv_1'], batch['t_inv_1']
        lig_seq_1_simplex = self.seq_to_simplex(lig_seq_1)
        lig_seq_1_prob = F.softmax(lig_seq_1_simplex,dim=-1)

        rotmats_1 = rotmats_1.unsqueeze(1) # (B, 1, 3, 3)
        trans_1 = trans_1.unsqueeze(1) # (B, 1, 3)

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
            trans_t = (1 - t[...,None]) * trans_0 + t[...,None] * trans_1

            # Randomly sample a sequence from the uniform distribution.
            lig_seq_0_simplex = self.k * torch.randn_like(lig_seq_1_simplex) # (B,L,K)
            lig_seq_0_prob = F.softmax(lig_seq_0_simplex, dim=-1) # (B,L,K)
            lig_seq_t_simplex = ((1 - t[..., None]) * lig_seq_0_simplex) + (t[..., None] * lig_seq_1_simplex) # (B,L,K)
            lig_seq_t_prob = F.softmax(lig_seq_t_simplex, dim=-1) # (B,L,K)
            lig_seq_t = sample_from(lig_seq_t_prob) # (B,L)
        
        # TODO Padding

        # Obtain the interface guidance.
        p1_residue, p1_coords, p2_residue, p2_coords = batch['p1_residue'], batch['p1_coords'], batch['p2_residue'], batch['p2_coords']
        i1_repr, i2_repr = self.interface_model(p1_residue, p1_coords, p2_residue, p2_coords)
        mol_mask = batch['mol_mask']
        
        # Denoise
        p1_coords, p1_seq, p2_coords, p2_seq = batch['p1_coords'], batch['p1_residue'], batch['p2_coords'], batch['p2_residue']
        p1_c_coords, p1_n_coords = batch['p1_c_coords'], batch['p1_n_coords']
        p2_c_coords, p2_n_coords = batch['p2_c_coords'], batch['p2_n_coords']
        p1_mask, p2_mask = batch['p1_mask'], batch['p2_mask']
        
        # Create molecular glue mask (assuming all ligand atoms are valid)
        # mol_mask = torch.ones(lig_coords_t.shape[:2], device=lig_coords_t.device, dtype=torch.bool)
        
        # Compute the vector field and predict the raw data at time step 1.
        pred_lig_seq_1_prob, pred_lig_coords_1, pred_rotmats_1, pred_trans_1 = self.vf_model(
            t=t, lig_coords_t=lig_coords_t, rotmats_t=rotmats_t, trans_t=trans_t, lig_seq_t=lig_seq_t,
            p1_coords=p1_coords, p1_c_coords=p1_c_coords, p1_n_coords=p1_n_coords, p1_seq=p1_seq, p1_mask=p1_mask,
            p2_coords=p2_coords, p2_c_coords=p2_c_coords, p2_n_coords=p2_n_coords, p2_seq=p2_seq, p2_mask=p2_mask,
            mol_mask=mol_mask, i1_repr=i1_repr, i2_repr=i2_repr
        )
        pred_lig_seq_1 = sample_from(F.softmax(pred_lig_seq_1_prob, dim=-1))

        norm_scale = 1 / (1 - torch.min(t[...,None], torch.tensor(self._interpolant_cfg.t_normalization_clip))) # yim etal.trick, 1/1-t

        # Translation Flow Matching loss, Euclidean Flow.
        trans_loss = torch.sum((pred_trans_1 - trans_1) ** 2, dim=-1) # (B, )
        trans_loss = torch.mean(trans_loss)

        # Rotation Flow Matching loss, Riemannian Flow on SO(3).
        gt_rot_vf = calc_rot_vf(rotmats_t, rotmats_1)
        pred_rot_vf = calc_rot_vf(rotmats_t, pred_rotmats_1)
        rot_loss = torch.sum(((gt_rot_vf - pred_rot_vf) * norm_scale) ** 2, dim=-1) # (B, )
        rot_loss = torch.mean(rot_loss)

        # Simplex Flow Matching loss, Simplex Flow on (K - 1)-simplex.
        # pred_lig_seq_1_prob: (B, L, K)
        seqs_loss = F.cross_entropy(
            pred_lig_seq_1_prob.view(-1, pred_lig_seq_1_prob.size(-1)),  # (B * L, K)
            lig_seq_1.view(-1)                                           # (B * L,)
        )
        seqs_loss = torch.mean(seqs_loss)

        # 3D Coordinates Flow Matching loss, Euclidean Flow.
        coords_loss = torch.sum((pred_lig_coords_1 - lig_coords_1) ** 2, dim=-1) # (B, )
        coords_loss = torch.mean(coords_loss)

        return {
            'trans_loss': trans_loss,
            'rot_loss': rot_loss,
            'seqs_loss': seqs_loss,
            'coords_loss': coords_loss,
        }

    @torch.no_grad()
    def sample(self, batch):
        pass