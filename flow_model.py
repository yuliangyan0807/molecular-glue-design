# Copyright (c) 2024. This code is built upon https://github.com/Ced3-han/PepFlowww
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from utils.so3_utils import (
    geodesic_t,
    uniform_so3,
    calc_rot_vf,
    geodesic_dist,
    project_to_so3,
    vector_to_skew_matrix,
)
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


def ligand_pair_mask(mol_mask, upper_only=False):
    """Mask valid ligand atom pairs, excluding self-pairs."""
    num_atoms = mol_mask.shape[1]
    pair_mask = mol_mask.bool().unsqueeze(2) & mol_mask.bool().unsqueeze(1)
    diagonal = torch.eye(
        num_atoms, dtype=torch.bool, device=mol_mask.device
    ).unsqueeze(0)
    pair_mask = pair_mask & ~diagonal
    if upper_only:
        pair_mask = pair_mask & torch.triu(
            torch.ones(
                num_atoms,
                num_atoms,
                dtype=torch.bool,
                device=mol_mask.device,
            ),
            diagonal=1,
        ).unsqueeze(0)
    return pair_mask


def sample_symmetric_pair_types(probabilities, mol_mask):
    """Sample one categorical state per undirected ligand atom pair."""
    if probabilities.ndim != 4:
        raise ValueError(
            "Pair probabilities must have shape (B, L, L, C), got "
            f"{tuple(probabilities.shape)}"
        )
    upper_mask = ligand_pair_mask(mol_mask, upper_only=True)
    sampled = torch.zeros(
        probabilities.shape[:3],
        dtype=torch.long,
        device=probabilities.device,
    )
    if upper_mask.any():
        upper_probabilities = probabilities[upper_mask] + 1e-8
        upper_types = torch.multinomial(upper_probabilities, 1).squeeze(-1)
        sampled[upper_mask] = upper_types
        sampled = sampled + sampled.transpose(1, 2)
    return sampled


def symmetric_pair_simplex_noise(reference, mol_mask, scale):
    """Independent Gaussian simplex noise for each undirected atom pair."""
    upper_mask = ligand_pair_mask(mol_mask, upper_only=True)
    noise = torch.zeros_like(reference)
    if upper_mask.any():
        noise[upper_mask] = scale * torch.randn_like(reference[upper_mask])
        noise = noise + noise.transpose(1, 2)
    return noise


def balanced_bond_cross_entropy(logits, targets, mol_mask):
    """Balance bonded and non-bonded pairs so class 0 cannot dominate."""
    upper_mask = ligand_pair_mask(mol_mask, upper_only=True)
    per_molecule = []
    for batch_idx in range(logits.shape[0]):
        valid = upper_mask[batch_idx]
        if not valid.any():
            continue
        pair_targets = targets[batch_idx][valid]
        pair_loss = F.cross_entropy(
            logits[batch_idx][valid], pair_targets, reduction="none"
        )
        bonded = pair_targets.ne(0)
        terms = []
        if bonded.any():
            terms.append(pair_loss[bonded].mean())
        if (~bonded).any():
            terms.append(pair_loss[~bonded].mean())
        per_molecule.append(torch.stack(terms).mean())
    if not per_molecule:
        return logits.sum() * 0.0
    return torch.stack(per_molecule).mean()


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


def coordinate_endpoint_pose_loss(
    p2_coords,
    p2_mask,
    rotmats_t,
    trans_t,
    rotmats_1,
    trans_1,
    pred_rot_vf,
    pred_trans_vf,
    t,
    translation_scale=1.0,
    beta=2.0,
):
    """Coordinate-space loss for the final rigidly docked P2.

    Extrapolate the predicted SE(3) vector field from the current pose at time
    ``t`` to time 1, transform the centered P2 rigidly, and compare that final
    docking result directly with the ground-truth docked P2.
    """
    rot_t = rotmats_t[:, 0] if rotmats_t.ndim == 4 else rotmats_t
    rot_1 = rotmats_1[:, 0] if rotmats_1.ndim == 4 else rotmats_1
    tr_t = trans_t[:, 0] if trans_t.ndim == 3 else trans_t
    tr_1 = trans_1[:, 0] if trans_1.ndim == 3 else trans_1

    remaining = (1.0 - t[:, 0]).clamp(min=0.0, max=1.0)
    pred_delta_R = torch.matrix_exp(
        vector_to_skew_matrix(remaining.unsqueeze(-1) * pred_rot_vf)
    )
    pred_rot_endpoint = rot_t @ pred_delta_R
    # The RT head predicts a dimensionless normalized translation field,
    # while poses and protein coordinates are represented in Angstrom.
    pred_trans_vf_angstrom = pred_trans_vf * float(translation_scale)
    pred_trans_endpoint = (
        tr_t + remaining.unsqueeze(-1) * pred_trans_vf_angstrom
    )

    pred_coords_endpoint = (
        p2_coords @ pred_rot_endpoint.transpose(1, 2)
        + pred_trans_endpoint.unsqueeze(1)
    )
    gt_coords_endpoint = (
        p2_coords @ rot_1.transpose(1, 2)
        + tr_1.unsqueeze(1)
    )
    per_coord = F.smooth_l1_loss(
        pred_coords_endpoint,
        gt_coords_endpoint,
        beta=beta,
        reduction="none",
    ).sum(dim=-1)
    mask = p2_mask.to(dtype=per_coord.dtype)
    return (
        (per_coord * mask).sum(dim=-1)
        / mask.sum(dim=-1).clamp_min(1.0)
    ).mean()

class TernaryFlowModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self._cfg = cfg

        self._interpolant_cfg = cfg.interpolant

        # Use one isotropic scale so translation normalization preserves SE(3)
        # equivariance. Pose states remain in Angstrom; only the learned vector
        # field and its regression target are normalized.
        self.translation_scale = float(
            getattr(self._interpolant_cfg.trans, 'normalization_scale', 50.0)
        )
        if self.translation_scale <= 0.0:
            raise ValueError(
                'interpolant.trans.normalization_scale must be positive, got '
                f'{self.translation_scale}'
            )

        num_classes = self._interpolant_cfg.seqs.num_classes
        self.num_bond_classes = int(self._interpolant_cfg.bonds.num_classes)
        self.bond_simplex_value = float(
            self._interpolant_cfg.bonds.simplex_value
        )
        self.vf_model = VFModel(
            cfg.model,
            num_classes=num_classes,
            num_bond_classes=self.num_bond_classes,
            translation_scale=self.translation_scale,
        )
        
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
        self.interface_finetune_heads_only = getattr(
            cfg.model.interface_model, 'finetune_heads_only', True
        )
        
        # Freeze interface model parameters if not trainable.  When pose
        # supervision is enabled, only adapt the virtual-keypoint heads by
        # default: updating the full pretrained geometric encoder from a weak
        # Kabsch signal is both expensive and prone to catastrophic forgetting.
        if not self.interface_trainable:
            for param in interface_model.parameters():
                param.requires_grad = False
            print("✓ Interface model parameters frozen (not trainable)")
        elif self.interface_finetune_heads_only:
            for param in interface_model.parameters():
                param.requires_grad = False
            keypoint_heads = (
                interface_model.att_mlp_key_ROT,
                interface_model.att_mlp_query_ROT,
                interface_model.mlp_h_mean_ROT,
            )
            for module in keypoint_heads:
                for param in module.parameters():
                    param.requires_grad = True
            # FIXME(RT-DEBUG): if the Kabsch validation error plateaus, compare
            # heads-only fine-tuning with full fine-tuning instead of silently
            # unfreezing the encoder.
            print("✓ Interface virtual-keypoint heads are trainable; encoder frozen")
        else:
            print("✓ Interface model parameters are trainable")
        
        self.interface_model = interface_model

        self.K = self._interpolant_cfg.seqs.num_classes
        self.k = self._interpolant_cfg.seqs.simplex_value
        
        # Top-k value for selecting interface residues based on attention
        self.topk_k = getattr(cfg.model.interface_model, 'topk_k', 50)
        
        # FIXME(RT-DEBUG): the old learnable alpha scaled only Kabsch translation
        # before applying a second absolute transform. It is intentionally
        # removed now that RT owns one unambiguous absolute pose.

    def train(self, mode=True):
        super().train(mode)
        if not self.interface_trainable:
            self.interface_model.eval()
        return self

    def _compute_interface_condition(
        self,
        p1_residue,
        p1_coords,
        p2_residue,
        p2_coords,
        p1_mask,
        p2_mask,
    ):
        """Run the required pretrained interface conditioner in FP32."""
        interface_grad_context = (
            nullcontext() if self.interface_trainable else torch.no_grad()
        )
        with interface_grad_context:
            with torch.autocast(
                device_type=p1_coords.device.type,
                enabled=False,
            ):
                return self.interface_model(
                    p1_residue=p1_residue,
                    p1_coords=p1_coords.float(),
                    p2_residue=p2_residue,
                    p2_coords=p2_coords.float(),
                    p1_mask=p1_mask,
                    p2_mask=p2_mask,
                )
    
    def seq_to_simplex(self,seqs):
        return clampped_one_hot(seqs, self.K).float() * self.k * 2 - self.k # (B,L,K)

    def bond_to_simplex(self, bonds):
        return (
            clampped_one_hot(bonds, self.num_bond_classes).float()
            * self.bond_simplex_value
            * 2
            - self.bond_simplex_value
        )

    def forward(self, batch):
        
        # Ground truth at time step 1.
        lig_seq_1, lig_coords_1, rotmats_1, trans_1 = batch['lig_seq_1'], batch['lig_coords_1'], batch['R_inv_1'], batch['t_inv_1']
        lig_bond_1 = batch['lig_bond_type_matrix_1']
        if not bool(batch['lig_bond_available'].all().item()):
            raise ValueError(
                "Bond flow requires a dataset rebuilt with lig_bond_index and "
                "lig_bond_type for every sample."
            )
        lig_seq_1_simplex = self.seq_to_simplex(lig_seq_1)
        lig_seq_1_prob = F.softmax(lig_seq_1_simplex,dim=-1)
        lig_bond_1_simplex = self.bond_to_simplex(lig_bond_1)
        mol_mask = batch['mol_mask']

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

            # Bond types follow the same linear Gaussian-to-simplex path as
            # atom types. One noise vector and one categorical sample are used
            # per undirected atom pair, then mirrored to keep the graph exact.
            pair_mask = ligand_pair_mask(mol_mask, upper_only=False)
            lig_bond_0_simplex = symmetric_pair_simplex_noise(
                lig_bond_1_simplex,
                mol_mask,
                self.bond_simplex_value,
            )
            lig_bond_t_simplex = (
                (1 - t[..., None, None]) * lig_bond_0_simplex
                + t[..., None, None] * lig_bond_1_simplex
            )
            lig_bond_t_simplex = (
                lig_bond_t_simplex * pair_mask.unsqueeze(-1)
            )
            lig_bond_t = sample_symmetric_pair_types(
                F.softmax(lig_bond_t_simplex, dim=-1), mol_mask
            )

        p1, p2 = batch['p1'], batch['p2']
        p1_mask = p1['res_mask']
        p2_mask = p2['res_mask']
        p1_residue = p1['aa']
        p2_residue = p2['aa']
        p1_ca_coords = p1['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        p2_ca_coords = p2['pos_heavyatom'][:, :, BBHeavyAtom.CA]

        mol_mask = batch['mol_mask']
        
        # The pretrained interface model is a geometric conditioner. Keep its
        # EGNN/keypoint geometry in FP32 even when the flow network uses AMP;
        # FP16 coordinate differences can overflow before GradScaler is involved.
        # When frozen, no_grad also prevents the RT objective from moving its
        # own reference keypoints while simultaneously trying to fit that frame.
        output = self._compute_interface_condition(
            p1_residue=p1_residue,
            p1_coords=p1_ca_coords,
            p2_residue=p2_residue,
            p2_coords=p2_ca_coords,
            p1_mask=p1_mask,
            p2_mask=p2_mask,
        )
        Y1, Y2 = output['Y1'], output['Y2'] # (B, K, 3)
        mu1, mu2 = output['mu1'], output['mu2'] # (B, 3)
        sigma_1, sigma_2 = output['cov1'], output['cov2'] # (B, 3, 3)

        # Kabsch is a geometric initializer, not a learned layer. Backpropagating
        # through its SVD is undefined when singular values coincide. With the
        # pretrained interface frozen, its keypoints remain a stable condition
        # for PhiRT instead of becoming another target that can drift.
        with torch.no_grad():
            R_star, t_star = kabsch_align(Y1, Y2)

        # Monitor Kabsch pose quality. These values are detached metrics because
        # SVD backward is unstable for degenerate keypoint configurations.
        kabsch_rot_vf = calc_rot_vf(
            R_star.unsqueeze(1), rotmats_1
        )
        kabsch_rot_loss = torch.mean(
            torch.sum(kabsch_rot_vf.square(), dim=-1)
        )
        kabsch_trans_loss = F.smooth_l1_loss(
            t_star,
            trans_1[:, 0],
            beta=5.0,
        )
        Y2_in_gt_pose = (
            Y2 @ rotmats_1[:, 0].transpose(1, 2)
            + trans_1[:, 0].unsqueeze(1)
        )
        keypoint_match_loss = F.smooth_l1_loss(
            Y2_in_gt_pose,
            Y1,
            beta=2.0,
        )

        # Keep P2 in its original randomized/local frame. PhiRT evaluates the
        # vector field at (rotmats_t, trans_t); Kabsch is only a prior feature.
        p2_coords_input = p2_ca_coords
        
        # Denoise
        # Compute the vector field and predict the raw data at time step 1.
        (
            pred_lig_seq_1_prob,
            pred_lig_bond_1_logits,
            pred_lig_coords_1,
            pred_rotmats_1,
            pred_trans_1,
            pred_rot_vf,
            pred_trans_vf,
        ) = self.vf_model(
            p1=p1, p2=p2, t=t, lig_coords_t=lig_coords_t, rotmats_t=rotmats_t, trans_t=trans_t, lig_seq_t=lig_seq_t,
            lig_bond_t=lig_bond_t,
            mol_mask=mol_mask, p2_coords_input=p2_coords_input,
            sigma_1=sigma_1, sigma_2=sigma_2,
            R_star=R_star, t_star=t_star,
            Y1=Y1, Y2=Y2
        )
        pred_lig_seq_1 = sample_from(F.softmax(pred_lig_seq_1_prob, dim=-1))

        # True conditional vector fields at the current state.  Unlike endpoint
        # regression, PhiRT predicts these tangents directly.
        remaining = (1.0 - t[:, 0]).clamp_min(1e-3)
        gt_trans_vf_angstrom = (
            trans_1[:, 0] - trans_t[:, 0]
        ) / remaining.unsqueeze(-1)
        gt_trans_vf = gt_trans_vf_angstrom / self.translation_scale
        gt_rot_vf = (
            calc_rot_vf(rotmats_t, rotmats_1)[:, 0]
            / remaining.unsqueeze(-1)
        )
        if pred_trans_vf.shape != gt_trans_vf.shape:
            raise RuntimeError(
                "RT translation field shape mismatch: "
                f"pred={pred_trans_vf.shape}, gt={gt_trans_vf.shape}"
            )
        if pred_rot_vf.shape != gt_rot_vf.shape:
            raise RuntimeError(
                "RT rotation field shape mismatch: "
                f"pred={pred_rot_vf.shape}, gt={gt_rot_vf.shape}"
            )
        trans_loss = torch.mean(
            torch.sum((pred_trans_vf - gt_trans_vf).square(), dim=-1)
        )
        rot_loss = torch.mean(
            torch.sum((pred_rot_vf - gt_rot_vf).square(), dim=-1)
        )

        # Directly supervise the final rigid docking result. The helper
        # converts the normalized translation field back to Angstrom.
        pose_coord_loss = coordinate_endpoint_pose_loss(
            p2_coords=p2_ca_coords,
            p2_mask=p2_mask,
            rotmats_t=rotmats_t,
            trans_t=trans_t,
            rotmats_1=rotmats_1,
            trans_1=trans_1,
            pred_rot_vf=pred_rot_vf,
            pred_trans_vf=pred_trans_vf,
            t=t,
            translation_scale=self.translation_scale,
        )

        # Simplex Flow Matching loss, Simplex Flow on (K - 1)-simplex.
        # pred_lig_seq_1_prob: (B, L, K)
        # Flatten for cross_entropy
        pred_flat = pred_lig_seq_1_prob.view(-1, pred_lig_seq_1_prob.size(-1))  # (B * L, K)
        target_flat = lig_seq_1.view(-1)  # (B * L,)
        mask_flat = mol_mask.view(-1)  # (B * L,)
        
        # Calculate per-position loss (reduction='none' to get per-position losses)
        seqs_loss_per_pos = F.cross_entropy(
            pred_flat,
            target_flat,
            reduction='none',  # (B * L,)
        )
        # Apply mask to ignore padding positions
        seqs_loss_per_pos = seqs_loss_per_pos * mask_flat.float()  # (B * L,)
        # Reshape back to (B, L)
        seqs_loss_per_pos = seqs_loss_per_pos.view(pred_lig_seq_1_prob.shape[0], -1)  # (B, L)
        # Average over positions for each molecule
        seqs_loss_per_mol = seqs_loss_per_pos.sum(dim=-1) / mol_mask.sum(dim=-1).float()  # (B,)
        # Average over batch
        seqs_loss = torch.mean(seqs_loss_per_mol)

        # Six-class simplex endpoint supervision over unique undirected pairs.
        # Bonded and no-bond pairs contribute equally within each molecule.
        bond_loss = balanced_bond_cross_entropy(
            pred_lig_bond_1_logits,
            lig_bond_1,
            mol_mask,
        )

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
            'bond_loss': bond_loss,
            'coords_loss': coords_loss,
            # 'kabsch_rot_loss': kabsch_rot_loss,
            # 'kabsch_trans_loss': kabsch_trans_loss,
            # 'keypoint_match_loss': keypoint_match_loss,
            'pose_coord_loss': pose_coord_loss,
        }

    @torch.no_grad()
    def sample(self, batch):
        # Setup
        num_batch = batch['lig_seq_1'].shape[0]
        num_lig = batch['lig_seq_1'].shape[1]

        # Ground truth at time step 1 (used as targets predicted by the network)
        lig_seq_1, lig_coords_1, rotmats_1, trans_1 = batch['lig_seq_1'], batch['lig_coords_1'], batch['R_inv_1'], batch['t_inv_1']
        lig_bond_1 = batch['lig_bond_type_matrix_1']
        if not bool(batch['lig_bond_available'].all().item()):
            raise ValueError(
                "Bond sampling requires a dataset rebuilt with ligand bonds."
            )
        lig_seq_1_simplex = self.seq_to_simplex(lig_seq_1)
        lig_bond_1_simplex = self.bond_to_simplex(lig_bond_1)
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
        
        output = self._compute_interface_condition(
            p1_residue=p1_residue,
            p1_coords=p1_ca_coords,
            p2_residue=p2_residue,
            p2_coords=p2_ca_coords,
            p1_mask=p1_mask,
            p2_mask=p2_mask,
        )
        Y1, Y2 = output['Y1'], output['Y2'] # (B, K, 3)
        mu1, mu2 = output['mu1'], output['mu2']
        sigma_1, sigma_2 = output['cov1'], output['cov2']

        # Kabsch is a detached geometric prior in both training and sampling.
        with torch.no_grad():
            R_star, t_star = kabsch_align(Y1, Y2)

        # TDOO
        # R_star = uniform_so3(num_batch, 1, device=p2_ca_coords.device).squeeze(1)
        # t_star = torch.randn(num_batch, 3, device=p2_ca_coords.device, dtype=p2_ca_coords.dtype) * 1

        # FIXME(RT-DEBUG): pass the original P2 for the same absolute-transform
        # convention used during training and when exporting sampled complexes.
        p2_coords_input = p2_ca_coords
        
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
        pair_mask = ligand_pair_mask(mol_mask, upper_only=False)
        lig_bond_0_simplex = symmetric_pair_simplex_noise(
            lig_bond_1_simplex,
            mol_mask,
            self.bond_simplex_value,
        )
        lig_bond_0_simplex = (
            lig_bond_0_simplex * pair_mask.unsqueeze(-1)
        )
        lig_bond_0 = sample_symmetric_pair_types(
            F.softmax(lig_bond_0_simplex, dim=-1), mol_mask
        )

        # States at current time step t_1
        rotmats_t_1 = rotmats_0.squeeze(1) # (B, 3, 3)
        trans_t_1 = trans_0 # (B, 1, 3)
        lig_coords_t_1 = lig_coords_0
        lig_seq_t_1 = lig_seq_0
        lig_seq_t_1_simplex = lig_seq_0_simplex
        lig_bond_t_1 = lig_bond_0
        lig_bond_t_1_simplex = lig_bond_0_simplex

        clean_traj = []
        # Denoise loop
        for step_index, t_2 in enumerate(ts[1:], start=1):
            t = torch.ones((num_batch, 1), device=lig_seq_1.device) * t_1

            # Predict raw data at time step 1 given the current noisy state at time t
            (
                pred_lig_seq_1_prob,
                pred_lig_bond_1_logits,
                pred_lig_coords_1,
                pred_rotmats_1,
                pred_trans_1,
                pred_rot_vf,
                pred_trans_vf,
            ) = self.vf_model(
                t=t,
                lig_coords_t=lig_coords_t_1,
                rotmats_t=rotmats_t_1,
                trans_t=trans_t_1,
                lig_seq_t=lig_seq_t_1,
                lig_bond_t=lig_bond_t_1,
                p1=p1,
                p2=p2,
                sigma_1=sigma_1, sigma_2=sigma_2,
                mol_mask=mol_mask, p2_coords_input=p2_coords_input,
                R_star=R_star, t_star=t_star,
                Y1=Y1, Y2=Y2
            )
            pred_rotmats_1 = project_to_so3(pred_rotmats_1)

            pred_lig_seq_1 = sample_from(F.softmax(pred_lig_seq_1_prob, dim=-1))
            pred_lig_seq_1_simplex = self.seq_to_simplex(pred_lig_seq_1)
            pred_lig_bond_1 = sample_symmetric_pair_types(
                F.softmax(pred_lig_bond_1_logits, dim=-1), mol_mask
            )
            pred_lig_bond_1_simplex = self.bond_to_simplex(
                pred_lig_bond_1
            ) * pair_mask.unsqueeze(-1)

            # Record trajectory (CPU for memory safety downstream)
            clean_traj.append({
                'rotmats': pred_rotmats_1.detach().cpu(),
                'trans': pred_trans_1.detach().cpu(),
                'lig_coords': pred_lig_coords_1.detach().cpu(),
                'lig_seq': pred_lig_seq_1.detach().cpu(),
                'lig_seq_simplex': pred_lig_seq_1_simplex.detach().cpu(),
                'lig_bond': pred_lig_bond_1.detach().cpu(),
                'lig_bond_simplex': pred_lig_bond_1_simplex.detach().cpu(),
                'rotmats_1': rotmats_1.detach().cpu(),
                'trans_1': trans_1.detach().cpu(),
                'lig_coords_1': lig_coords_1.detach().cpu(),
                'lig_seq_1': lig_seq_1.detach().cpu(),
                'lig_bond_1': lig_bond_1.detach().cpu(),
            })

            # Euler integration of the learned SE(3) field from the current
            # state. Rotation is a right/body-frame update, matching calc_rot_vf.
            d_t = t_2 - t_1
            # Pose states are physical Angstrom values; the network field is
            # normalized by the same scalar used for the training target.
            trans_t_2 = trans_t_1 + (
                d_t * self.translation_scale * pred_trans_vf.unsqueeze(1)
            )
            delta_R = torch.matrix_exp(
                vector_to_skew_matrix(d_t * pred_rot_vf)
            )
            rotmats_t_2 = project_to_so3(rotmats_t_1 @ delta_R)

            endpoint_fraction = torch.clamp(
                d_t / torch.clamp(1.0 - t_1, min=1e-6),
                min=0.0,
                max=1.0,
            )

            lig_coords_t_2 = lig_coords_t_1 + endpoint_fraction * (
                pred_lig_coords_1 - lig_coords_t_1
            )
            lig_coords_t_2 = lig_coords_t_2 * mol_mask.unsqueeze(-1)  # Mask out padding positions

            # Sequences (simplex)
            lig_seq_t_2_simplex = lig_seq_t_1_simplex + endpoint_fraction * (
                pred_lig_seq_1_simplex - lig_seq_t_1_simplex
            )
            lig_seq_t_2_simplex = lig_seq_t_2_simplex * mol_mask.unsqueeze(-1)  # Mask out padding positions
            lig_seq_t_2 = sample_from(F.softmax(lig_seq_t_2_simplex, dim=-1))

            # Bonds (simplex), with one state per undirected atom pair.
            lig_bond_t_2_simplex = lig_bond_t_1_simplex + endpoint_fraction * (
                pred_lig_bond_1_simplex - lig_bond_t_1_simplex
            )
            lig_bond_t_2_simplex = (
                lig_bond_t_2_simplex * pair_mask.unsqueeze(-1)
            )
            lig_bond_t_2 = sample_symmetric_pair_types(
                F.softmax(lig_bond_t_2_simplex, dim=-1), mol_mask
            )

            # Move to next step
            rotmats_t_1, trans_t_1, lig_coords_t_1, lig_seq_t_1, lig_seq_t_1_simplex, lig_bond_t_1, lig_bond_t_1_simplex = (
                rotmats_t_2, trans_t_2, lig_coords_t_2, lig_seq_t_2,
                lig_seq_t_2_simplex, lig_bond_t_2, lig_bond_t_2_simplex
            )
            t_1 = t_2

        # Final step at t=1
        t = torch.ones((num_batch, 1), device=lig_seq_1.device) * ts[-1]
        (
            pred_lig_seq_1_prob,
            pred_lig_bond_1_logits,
            pred_lig_coords_1,
            pred_rotmats_1,
            pred_trans_1,
            pred_rot_vf,
            pred_trans_vf,
        ) = self.vf_model(
            t=t,
            lig_coords_t=lig_coords_t_1,
            rotmats_t=rotmats_t_1,
            trans_t=trans_t_1,
            lig_seq_t=lig_seq_t_1,
            lig_bond_t=lig_bond_t_1,
            p1=p1,
            p2=p2,
            sigma_1=sigma_1, sigma_2=sigma_2,
            mol_mask=mol_mask, p2_coords_input=p2_coords_input,
            R_star=R_star, t_star=t_star,
            Y1=Y1, Y2=Y2
        )
        pred_rotmats_1 = project_to_so3(pred_rotmats_1)

        # pred_lig_seq_1 = sample_from(F.softmax(pred_lig_seq_1_prob, dim=-1))
        pred_lig_seq_1 = pred_lig_seq_1_prob.argmax(dim=-1)
        pred_lig_seq_1_simplex = self.seq_to_simplex(pred_lig_seq_1)
        pred_lig_bond_1 = pred_lig_bond_1_logits.argmax(dim=-1)
        pred_lig_bond_1 = torch.where(
            pair_mask,
            pred_lig_bond_1,
            torch.zeros_like(pred_lig_bond_1),
        )
        # Logits are symmetric, but keep this invariant explicit in case the
        # head implementation changes later.
        upper_mask = ligand_pair_mask(mol_mask, upper_only=True)
        pred_lig_bond_1 = torch.where(
            upper_mask, pred_lig_bond_1, torch.zeros_like(pred_lig_bond_1)
        )
        pred_lig_bond_1 = pred_lig_bond_1 + pred_lig_bond_1.transpose(1, 2)
        pred_lig_bond_1_simplex = self.bond_to_simplex(
            pred_lig_bond_1
        ) * pair_mask.unsqueeze(-1)

        # pred_trans_1 = pred_trans_1 * TRANS_STD.to(pred_trans_1.device) + TRANS_MEAN.to(pred_trans_1.device)

        clean_traj.append({
            'rotmats': pred_rotmats_1.detach().cpu(),
            'trans': pred_trans_1.detach().cpu(),
            'lig_coords': pred_lig_coords_1.detach().cpu(),
            'lig_seq': pred_lig_seq_1.detach().cpu(),
            'lig_seq_simplex': pred_lig_seq_1_simplex.detach().cpu(),
            'lig_bond': pred_lig_bond_1.detach().cpu(),
            'lig_bond_simplex': pred_lig_bond_1_simplex.detach().cpu(),
            'rotmats_1': rotmats_1.detach().cpu(),
            'trans_1': trans_1.detach().cpu(),
            'lig_coords_1': lig_coords_1.detach().cpu(),
            'lig_seq_1': lig_seq_1.detach().cpu(),
            'lig_bond_1': lig_bond_1.detach().cpu(),
        })

        return clean_traj
