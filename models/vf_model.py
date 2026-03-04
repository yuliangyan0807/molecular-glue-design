import math
import numpy as np
from scipy.stats import truncnorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, List, Sequence
from einops import einsum

from utils.rigid_utils import construct_3d_basis, global_to_local
from openfold.utils import rigid_utils as ru
from openfold.utils.rigid_utils import Rigid

from utils.so3_utils import rotvec_to_rotmat
from utils.constants import AA, BBHeavyAtom, max_num_heavyatoms, MAP_ATOM_TYPE_FULL_TO_INDEX, TRANS_MEAN, TRANS_STD
from utils.rigid_utils import get_backbone_dihedral_angles, pairwise_dihedrals, kabsch_align

from .egnn import EGNN_Network

def create_rigid(rots, trans):
    rots = ru.Rotation(rot_mats=rots)

    return Rigid(rots=rots, trans=trans)

def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f

def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")

def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    
    return emb

def angstrom_to_nm(x):
    return x / 10

class AngularEncoding(nn.Module):

    def __init__(self, num_funcs=3):
        super().__init__()
        self.num_funcs = num_funcs
        self.register_buffer('freq_bands', torch.FloatTensor(
            [i+1 for i in range(num_funcs)] + [1./(i+1) for i in range(num_funcs)]
        ))

    def get_out_dim(self, in_dim):
        return in_dim * (1 + 2 * 2 * self.num_funcs)

    def forward(self, x):
        """
        Args:
            x:  (..., d).
        """
        shape = list(x.shape[:-1]) + [-1]
        x = x.unsqueeze(-1) # (..., d, 1)
        code = torch.cat([x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1)   # (..., d, 2f+1)
        code = code.reshape(shape)
        return code

class NodeEmbedder(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.feat_dim = feat_dim
        self.aatype_embed = nn.Embedding(self.max_aa_types, feat_dim)
        self.dihed_embed = AngularEncoding()
        
        # Only use amino acid features, no coordinate features
        infeat_dim = feat_dim +  (self.max_aa_types * max_num_atoms * 3) + self.dihed_embed.get_out_dim(3)
        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, structure_mask=None, sequence_mask=None):
        """
        Args:
            aa:         (N, L) - amino acid types.
            res_nb:     (N, L) - residue numbers.
            chain_nb:   (N, L) - chain numbers.
            pos_atoms:  (N, L, A, 3) - atom coordinates.
            mask_atoms: (N, L, A) - atom masks.
            structure_mask: (N, L), mask out unknown structures to generate.
            sequence_mask:  (N, L), mask out unknown amino acids to generate.
        """
        N, L = aa.size()
        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA] # (N, L)

        # Remove other atoms
        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]

        # Amino acid identity features
        aa_feat = self.aatype_embed(aa) # (N, L, feat)

        # Coordinate features.
        R = construct_3d_basis(
            pos_atoms[:, :, BBHeavyAtom.CA], 
            pos_atoms[:, :, BBHeavyAtom.C],
            pos_atoms[:, :, BBHeavyAtom.N]
        )
        t = pos_atoms[:, :, BBHeavyAtom.CA]
        crd = global_to_local(R, t, pos_atoms)  # (N, L, A, 3)
        crd_mask = mask_atoms[:, :, :, None].expand_as(crd) # (N, L, A, 3)
        crd = torch.where(crd_mask, crd, torch.zeros_like(crd))

        aa_expand = aa[:, :, None, None, None].expand(N, L, self.max_aa_types, self.max_num_atoms, 3)
        rng_expand = torch.arange(0, self.max_aa_types)[None, None, :, None, None].expand(N, L, self.max_aa_types, self.max_num_atoms, 3).to(aa_expand)
        place_mask = (aa_expand == rng_expand)
        crd_expand = crd[:, :, None, :, :].expand(N, L, self.max_aa_types, self.max_num_atoms, 3)
        crd_expand = torch.where(place_mask, crd_expand, torch.zeros_like(crd_expand))
        crd_feat = crd_expand.reshape(N, L, self.max_aa_types * self.max_num_atoms * 3)
        if structure_mask is not None:
            # Avoid data leakage at training time
            crd_feat = crd_feat * structure_mask[:, :, None]
        
        # Backbone dihedral features.
        bb_dihedral, mask_bb_dihed = get_backbone_dihedral_angles(
            pos_atoms, chain_nb=chain_nb, res_nb=res_nb, mask=mask_residue
        )
        dihed_feat = self.dihed_embed(
            bb_dihedral[:, :, :, None] * mask_bb_dihed[:, :, :, None]
        ) # (N, L, 3, dihed/3)
        dihed_feat = dihed_feat.reshape(N, L, -1)
        if structure_mask is not None:
            # Avoid data leakage at training time
            dihed_mask = torch.logical_and(
                structure_mask,
                torch.logical_and(
                    torch.roll(structure_mask, shifts=+1, dims=1), 
                    torch.roll(structure_mask, shifts=-1, dims=1)
                ),
            )   # Avoid slight data leakage via dihedral angles of anchor residues
            dihed_feat = dihed_feat * dihed_mask[:, :, None]
            
        out_feat = self.mlp(
            torch.cat([aa_feat, crd_feat, dihed_feat], dim=-1)
        ) # (N, L, F)
        out_feat = out_feat * mask_residue[:, :, None]

        return out_feat

class EdgeEmbedder(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22, max_relpos=32, num_bins=16):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos
        self.num_bins = num_bins
        self.aa_pair_embed = nn.Embedding(self.max_aa_types * self.max_aa_types, feat_dim)
        self.relpos_embed = nn.Embedding(2 * max_relpos + 1, feat_dim)

        self.aapair_to_distcoef = nn.Embedding(self.max_aa_types * self.max_aa_types, max_num_atoms * max_num_atoms)
        nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.distance_embed = nn.Sequential(
            nn.Linear(max_num_atoms * max_num_atoms, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
        )

        self.dihedral_embed = AngularEncoding()
        feat_dihed_dim = self.dihedral_embed.get_out_dim(2) # Phi and Psi

        infeat_dim = feat_dim + feat_dim + feat_dim + feat_dihed_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, structure_mask=None, sequence_mask=None):
        """
        Args:
            aa: (N, L).
            res_nb: (N, L).
            chain_nb: (N, L).
            pos_atoms:  (N, L, A, 3)
            mask_atoms: (N, L, A)
            trans, sc_trans: (N,L,3)
            structure_mask: (N, L)
            sequence_mask:  (N, L), mask out unknown amino acids to generate.

        Returns:
            (N, L, L, feat_dim)
        """
        N, L = aa.size()

        # Remove other atoms
        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]

        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA] # (N, L)
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]
        pair_structure_mask = structure_mask[:, :, None] * structure_mask[:, None, :] if structure_mask is not None else None

        # Pair identities
        if sequence_mask is not None:
            # Avoid data leakage at training time
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))
        aa_pair = aa[:,:,None] * self.max_aa_types + aa[:,None,:]    # (N, L, L)
        feat_aapair = self.aa_pair_embed(aa_pair)
    
        # Relative sequential positions
        same_chain = (chain_nb[:, :, None] == chain_nb[:, None, :])
        relpos = torch.clamp(
            res_nb[:,:,None] - res_nb[:,None,:], 
            min=-self.max_relpos, max=self.max_relpos,
        )   # (N, L, L)
        feat_relpos = self.relpos_embed(relpos + self.max_relpos) * same_chain[:,:,:,None]

        # Distances
        d = angstrom_to_nm(torch.linalg.norm(
            pos_atoms[:,:,None,:,None] - pos_atoms[:,None,:,None,:],
            dim = -1, ord = 2,
        )).reshape(N, L, L, -1) # (N, L, L, A*A)
        c = F.softplus(self.aapair_to_distcoef(aa_pair))    # (N, L, L, A*A)
        d_gauss = torch.exp(-1 * c * d**2)
        mask_atom_pair = (mask_atoms[:,:,None,:,None] * mask_atoms[:,None,:,None,:]).reshape(N, L, L, -1)
        feat_dist = self.distance_embed(d_gauss * mask_atom_pair)
        if pair_structure_mask is not None:
            # Avoid data leakage at training time
            feat_dist = feat_dist * pair_structure_mask[:, :, :, None]

        # Orientations
        dihed = pairwise_dihedrals(pos_atoms)   # (N, L, L, 2)
        feat_dihed = self.dihedral_embed(dihed)
        if pair_structure_mask is not None:
            # Avoid data leakage at training time
            feat_dihed = feat_dihed * pair_structure_mask[:, :, :, None]

        feat_all = torch.cat([feat_aapair, feat_relpos, feat_dist, feat_dihed], dim=-1)
        feat_all = self.out_mlp(feat_all)   # (N, L, L, F)
        feat_all = feat_all * mask_pair[:, :, :, None]

        return feat_all
    
class LigandNodeEmbedder(nn.Module):
    """
    Embedder for ligand node features based on atom types.
    Generates single features (N, L, feat_dim) from atom type indices.
    """
    
    def __init__(self, feat_dim, num_atom_types=None):
        """
        Args:
            feat_dim: Output feature dimension
            num_atom_types: Number of atom types. If None, uses len(MAP_ATOM_TYPE_FULL_TO_INDEX)
        """
        super().__init__()
        self.feat_dim = feat_dim
        
        # Determine number of atom types
        if num_atom_types is None:
            num_atom_types = len(MAP_ATOM_TYPE_FULL_TO_INDEX)
        self.num_atom_types = num_atom_types
        
        # Atom type embedding
        self.atom_type_embed = nn.Embedding(num_atom_types, feat_dim)
        
        # Optional MLP to further process embeddings
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )
    
    def forward(self, lig_seq, mol_mask):
        """
        Args:
            lig_seq: (N, L) - Atom type indices (integer indices for atom types)
            mol_mask: (N, L) - Mask for valid atoms
        
        Returns:
            (N, L, feat_dim) - Node features for each atom
        """
        # Embed atom types
        feat = self.atom_type_embed(lig_seq)  # (N, L, feat_dim)
        
        # Apply MLP
        feat = self.mlp(feat)  # (N, L, feat_dim)
        
        # Apply mask
        feat = feat * mol_mask[:, :, None]
        
        return feat

class LigandEdgeEmbedder(nn.Module):
    """
    Embedder for ligand pair features based on atomic coordinates.
    Generates pair features (N, L, L, feat_dim) from atomic coordinates.
    """
    
    def __init__(self, feat_dim, num_distance_bins=32, distance_max=20.0):
        """
        Args:
            feat_dim: Output feature dimension
            num_distance_bins: Number of bins for distance encoding
            distance_max: Maximum distance in Angstrom for binning
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.num_distance_bins = num_distance_bins
        self.distance_max = distance_max
        
        # Distance embedding using Gaussian basis functions
        self.distance_embed = nn.Sequential(
            nn.Linear(num_distance_bins, feat_dim), 
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), 
            nn.ReLU(),
        )
        
        # Relative position vector embedding
        self.relpos_embed = nn.Sequential(
            nn.Linear(3, feat_dim), 
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        
        # Distance-based Gaussian encoding
        # Create learnable Gaussian centers and widths
        self.register_buffer('gaussian_centers', torch.linspace(0, distance_max, num_distance_bins))
        self.gaussian_widths = nn.Parameter(torch.ones(num_distance_bins) * (distance_max / num_distance_bins))
        
        # Output MLP to combine features
        infeat_dim = feat_dim + feat_dim  # distance + relative position
        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), 
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), 
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )
    
    def forward(self, lig_coords_t, mol_mask):
        """
        Args:
            lig_coords_t: (N, L, 3) - Ligand atomic coordinates
            mol_mask: (N, L) - Mask for valid atoms
        
        Returns:
            (N, L, L, feat_dim) - Pair features
        """
        N, L = lig_coords_t.shape[:2]
        
        # Create pair mask
        mask_pair = mol_mask[:, :, None] * mol_mask[:, None, :]  # (N, L, L)
        
        # Compute pairwise distances
        # lig_coords_t: (N, L, 3)
        # Expand to (N, L, L, 3) for pairwise computation
        coords_i = lig_coords_t[:, :, None, :]  # (N, L, 1, 3)
        coords_j = lig_coords_t[:, None, :, :]  # (N, 1, L, 3)
        
        # Distance: (N, L, L)
        dist = torch.linalg.norm(coords_i - coords_j, dim=-1)
        
        # Distance encoding using Gaussian basis functions
        # Expand distances: (N, L, L, 1)
        dist_expanded = dist.unsqueeze(-1)  # (N, L, L, 1)
        # Expand centers: (1, 1, 1, num_bins)
        centers = self.gaussian_centers.view(1, 1, 1, -1)  # (1, 1, 1, num_bins)
        # Expand widths: (1, 1, 1, num_bins)
        widths = self.gaussian_widths.view(1, 1, 1, -1)  # (1, 1, 1, num_bins)
        
        # Compute Gaussian activations: (N, L, L, num_bins)
        dist_diff = dist_expanded - centers  # (N, L, L, num_bins)
        gaussian_activations = torch.exp(-0.5 * (dist_diff / (widths + 1e-6)) ** 2)
        
        # Apply mask
        gaussian_activations = gaussian_activations * mask_pair[:, :, :, None]
        
        # Embed distances
        feat_dist = self.distance_embed(gaussian_activations)  # (N, L, L, feat_dim)
        
        # Relative position vector encoding
        rel_pos = coords_i - coords_j  # (N, L, L, 3)
        # Normalize relative position by distance (avoid division by zero)
        dist_safe = dist.unsqueeze(-1) + 1e-6  # (N, L, L, 1)
        rel_pos_normalized = rel_pos / dist_safe  # (N, L, L, 3)
        
        # Embed relative positions
        feat_relpos = self.relpos_embed(rel_pos_normalized)  # (N, L, L, feat_dim)
        feat_relpos = feat_relpos * mask_pair[:, :, :, None]
        
        # Combine features
        feat_all = torch.cat([feat_dist, feat_relpos], dim=-1)  # (N, L, L, 2*feat_dim)
        feat_all = self.out_mlp(feat_all)  # (N, L, L, feat_dim)
        
        # Apply final mask
        feat_all = feat_all * mask_pair[:, :, :, None]
        
        return feat_all

class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")

class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        ipa_conf,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(InvariantPointAttention, self).__init__()
        self._ipa_conf = ipa_conf

        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.c_hidden = ipa_conf.c_hidden
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((ipa_conf.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: torch.Tensor,
        mask: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
        i_repr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])
        
        if(_offload_inference):
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        # BUG
        # square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = mask.float().unsqueeze(-1) * mask.float().unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        
        a = a + pt_att 
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v] 
        o_pt = torch.sum(
            (
                a[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if(_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z[0])
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            )
        )
        
        return s

class Phix(nn.Module):
    """
    Denoising model for molecular glue coordinates, similar to targetdiff's ScorePosNet3D.
    Handles two proteins (p1, p2) and a ligand, predicting denoised coordinates and atom types.
    """
    def __init__(self, c_s, c_t, hidden_dim, num_layers=3, use_egnn=True, 
                 use_phia=True, num_classes=25, **egnn_kwargs):
        """
        Args:
            c_s: Feature dimension for protein/ligand features
            c_t: Time embedding dimension
            hidden_dim: Hidden dimension for the network
            num_layers: Number of EGNN layers
            use_egnn: Whether to use EGNN_Network (True) or custom EGNN (False)
            use_phia: Whether to use PhiA for atom type prediction (default: True)
            num_classes: Number of atom type classes for PhiA (default: 25)
            **egnn_kwargs: Additional arguments for EGNN_Network
        """
        super().__init__()
        self.c_s = c_s
        self.c_t = c_t
        self.hidden_dim = hidden_dim
        self.use_egnn = use_egnn
        self.use_phia = use_phia
        
        # Project protein features to hidden dimension
        self.protein1_emb = nn.Linear(c_s, hidden_dim)
        self.protein2_emb = nn.Linear(c_s, hidden_dim)
        
        # Project ligand features (with time embedding) to hidden dimension
        # Input: ligand features (c_s) + time embedding (c_t)
        self.ligand_emb = nn.Linear(c_s + c_t, hidden_dim)

        self.refine_net = EGNN_Network(
            depth=3,
            dim=c_s,
            num_nearest_neighbors=16,
            norm_coors=True,
            coor_weights_clamp_value=2.0
        )
        
        # PhiA for atom type prediction using refined features
        if use_phia:
            # Project refined features (hidden_dim) back to c_s for PhiA compatibility
            # Or we can modify PhiA to accept hidden_dim directly
            self.refined_to_cs = nn.Linear(hidden_dim, c_s)
            self.phia = PhiA(c_s=c_s, c_t=c_t, hidden_dim=hidden_dim, num_classes=num_classes)
    
    def forward(self, s1, p1_coords, s2, p2_coords, sl, X_tilde, t_emb, p1_mask, p2_mask, mol_mask):
        """
        Forward pass to predict denoised coordinates and optionally atom types.
        
        Args:
            s1: (B, N1, c_s) - protein 1 features
            s2: (B, N2, c_s) - protein 2 features  
            p1_coords: (B, N1, 3) - protein 1 coordinates
            p2_coords: (B, N2, 3) - protein 2 coordinates
            sl: (B, N, c_s) - ligand features (initial features)
            X_tilde: (B, N, 3) - noised ligand coordinates at timestep t
            t_emb: (B, N, c_t) or (B, c_t) - time embedding
            p1_mask: (B, N1) - mask for protein 1 residues
            p2_mask: (B, N2) - mask for protein 2 residues
            mol_mask: (B, N) - mask for ligand atoms
            
        Returns:
            pred_coords: (B, N, 3) - predicted denoised coordinates
            phi_probs: (B, N, num_classes) - predicted atom type probabilities (if use_phia=True)
        """
        B, N = X_tilde.shape[:2]
        N1 = s1.shape[1]
        N2 = s2.shape[1]
        
        # Handle time embedding: expand if needed
        if t_emb.dim() == 2:  # (B, c_t)
            t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, c_t)
        
        # Embed protein features
        h_p1 = self.protein1_emb(s1)  # (B, N1, hidden_dim)
        h_p2 = self.protein2_emb(s2)  # (B, N2, hidden_dim)
        
        # Embed ligand features with time embedding
        # Concatenate ligand features with time embedding
        sl_with_time = torch.cat([sl, t_emb], dim=-1)  # (B, N, c_s + c_t)
        h_ligand = self.ligand_emb(sl_with_time)  # (B, N, hidden_dim)
        
        # Combine all features and coordinates
        # Order: p1, p2, ligand (similar to targetdiff's compose_context)
        h_all = torch.cat([h_p1, h_p2, h_ligand], dim=1)  # (B, N1+N2+N, hidden_dim)
        coords_all = torch.cat([p1_coords, p2_coords, X_tilde], dim=1)  # (B, N1+N2+N, 3)
        
        # Create overall mask (valid nodes)
        # This mask indicates which nodes are valid (not padding)
        mask_all = torch.cat([p1_mask, p2_mask, mol_mask], dim=1)  # (B, N1+N2+N)
        
        # Use EGNN_Network to refine features and coordinates
        # EGNN_Network processes all nodes but we only care about ligand coordinates
        # Note: In targetdiff, protein coordinates remain fixed during refinement.
        # Here, EGNN_Network may update all coordinates, but we only extract ligand coordinates.
        refined_feats, refined_coords = self.refine_net(
            feats=h_all,
            coors=coords_all,
            mask=mask_all,
            return_coor_changes=False
        )
        
        # Extract only ligand coordinates (last N coordinates)
        # This gives us the denoised ligand coordinates
        pred_coords = refined_coords[:, N1+N2:, :]  # (B, N, 3)
        
        # Extract refined features for all components (p1, p2, ligand)
        # Order in refined_feats: [p1, p2, ligand]
        refined_p1_feats = refined_feats[:, :N1, :]  # (B, N1, hidden_dim)
        refined_p2_feats = refined_feats[:, N1:N1+N2, :]  # (B, N2, hidden_dim)
        refined_ligand_feats = refined_feats[:, N1+N2:, :]  # (B, N, hidden_dim)
        
        # Use PhiA to predict atom type probabilities from refined features
        if self.use_phia:
            # Project refined features back to c_s dimension for PhiA compatibility
            refined_p1_feats_cs = self.refined_to_cs(refined_p1_feats)  # (B, N1, c_s)
            refined_p2_feats_cs = self.refined_to_cs(refined_p2_feats)  # (B, N2, c_s)
            refined_ligand_feats_cs = self.refined_to_cs(refined_ligand_feats)  # (B, N, c_s)
            
            # Use PhiA to predict atom type probabilities with refined features
            phi_probs = self.phia(
                s1=refined_p1_feats_cs,  # Use refined p1 features
                s2=refined_p2_feats_cs,  # Use refined p2 features
                sl=refined_ligand_feats_cs,  # Use refined ligand features
                t_emb=t_emb,
                p1_mask=p1_mask,
                p2_mask=p2_mask,
                mol_mask=mol_mask
            )  # (B, N, num_classes)
            
            return pred_coords, phi_probs
        else:
            return pred_coords

class PhiX(nn.Module):
    """
    Implements the EGNN-style coordinate update:
    X_i^pred = X_i^t + sum_{j≠i} (X_i^t - X_j^t) * φ_X(h_i, h_j, ||X_i^t - X_j^t||², t)
    
    Supports multi-layer updates where atom features are updated iteratively.
    """
    def __init__(self, c_s, c_t, hidden_dim, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.c_s = c_s
        self.c_t = c_t
        
        # Project atom features to hidden dimension
        self.atom_proj = nn.Linear(c_s, hidden_dim)
        
        # Project protein features for cross-attention
        self.proj_s1 = nn.Linear(c_s, hidden_dim)
        self.proj_s2 = nn.Linear(c_s, hidden_dim)
        
        # Cross-attention modules to incorporate protein context
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # Multi-layer feature update modules (EGNN-style)
        self.feature_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.feature_layers.append(nn.Sequential(
                nn.Linear(hidden_dim * 2 + 1 + c_t, hidden_dim),  # h_i, h_j, dist_sq, t
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ))
        
        # Final MLP to compute φ_X weights
        # Input: h_i, h_j, ||X_i^t - X_j^t||², t
        self.phi_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + c_t, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.phi_mlp_p1 = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + c_t, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.phi_mlp_p2 = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + c_t, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # self.seq_embedder = nn.Embedding(len(MAP_ATOM_TYPE_FULL_TO_INDEX), c_s)

    def forward(self, s1, p1_coords, i1_mask, s2, p2_coords, i2_mask, sl, X_tilde, t_emb, p1_mask, p2_mask, mol_mask):
        """
        Args:
            s1: (B, N1, c_s) - protein 1 features
            s2: (B, N2, c_s) - protein 2 features
            p1_coords: (B, N1, 3) - protein 1 coordinates
            p2_coords: (B, N2, 3) - protein 2 coordinates
            i1_mask: (B, N1) - interface mask for protein 1
            i2_mask: (B, N2) - interface mask for protein 2
            sl: (B, N, c_s) - ligand features
            X_tilde: (B, N, 3) - noised coordinates at timestep t
            t_emb: (B, N, c_t) - time embedding
            p1_mask: (B, N1) - mask for protein 1 residues
            p2_mask: (B, N2) - mask for protein 2 residues
            mol_mask: (B, N) - mask for ligand atoms
        Returns:
            Updated coordinates: (B, N, 3)
        """
        B, N = X_tilde.shape[:2]

        X_i = X_tilde.unsqueeze(2)  # [B, N, 1, 3]
        X_j = X_tilde.unsqueeze(1)  # [B, 1, N, 3]
        coord_diffs = X_i - X_j     # [B, N, N, 3]

        # Mask p1 and p2 coordinates: only keep positions where interface masks are True
        X_p1_tilde = torch.masked_fill(p1_coords.clone(), ~i1_mask.unsqueeze(-1), 0.0)  # (B, N1, 3)
        X_p2_tilde = torch.masked_fill(p2_coords.clone(), ~i2_mask.unsqueeze(-1), 0.0)  # (B, N2, 3)
        
        # Project atom features to hidden dimension
        h = self.atom_proj(sl)
        
        # Incorporate protein context via cross-attention
        s1_proj = self.proj_s1(s1)  # (B, N1, hidden_dim)
        s2_proj = self.proj_s2(s2)  # (B, N2, hidden_dim)
        
        # Concatenate s1 and s2 for cross-attention
        s_proj_concat = torch.cat([s1_proj, s2_proj], dim=1)  # (B, N1+N2, hidden_dim)
        p_mask_concat = torch.cat([p1_mask, p2_mask], dim=1)  # (B, N1+N2)
        
        # Cross-attend atom features to protein features (both p1 and p2)
        h_attn, _ = self.cross_attn(h, s_proj_concat, s_proj_concat, key_padding_mask=~p_mask_concat)
        
        h = h_attn * mol_mask.unsqueeze(-1)  # Apply mask
        
        # Filter s1_proj and s2_proj to only include interface residues
        s1_tilde = s1_proj * i1_mask.unsqueeze(-1)  # (B, N1, hidden_dim)
        s2_tilde = s2_proj * i2_mask.unsqueeze(-1)  # (B, N2, hidden_dim)
        
        # Compute squared distances: ||X_i^t - X_j^t||²
        distances_sq = torch.sum(coord_diffs ** 2, dim=-1, keepdim=True)  # [B, N, N, 1]
        # Clamp initial distances_sq to prevent numerical issues
        distances_sq = torch.clamp(distances_sq, min=0.0, max=1e6)  # Max distance ~1000 Angstroms
        
        # Get number of p1 and p2 interface residues
        N1 = X_p1_tilde.shape[1]
        N2 = X_p2_tilde.shape[1]
        
        # Multi-layer feature updates (EGNN-style)
        for layer in self.feature_layers:
            # Part 1: Ligand internal edges (N x N, fully connected)
            h_i_ligand = h.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, hidden_dim)
            h_j_ligand = h.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, hidden_dim)
            t_emb_ligand = t_emb.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, c_t)
            
            # Part 2: Ligand-p1 interface edges (N x N1, fully connected)
            h_i_p1 = h.unsqueeze(2).expand(-1, -1, N1, -1)  # (B, N, N1, hidden_dim)
            s1_tilde_expanded = s1_tilde.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N1, hidden_dim)
            t_emb_p1 = t_emb.unsqueeze(2).expand(-1, -1, N1, -1)  # (B, N, N1, c_t)
            
            # Part 3: Ligand-p2 interface edges (N x N2, fully connected)
            h_i_p2 = h.unsqueeze(2).expand(-1, -1, N2, -1)  # (B, N, N2, hidden_dim)
            s2_tilde_expanded = s2_tilde.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N2, hidden_dim)
            t_emb_p2 = t_emb.unsqueeze(2).expand(-1, -1, N2, -1)  # (B, N, N2, c_t)
            
            # Compute coordinate differences and distances for all parts
            # Ligand internal: X_i^t - X_j^t
            coord_diffs_ligand = coord_diffs  # (B, N, N, 3) - already computed
            distances_sq_ligand = distances_sq  # (B, N, N, 1) - already computed
            
            # Ligand-p1: X_i^t - X_k (for k in I1)
            X_i_expanded = X_tilde.unsqueeze(2)  # (B, N, 1, 3)
            X_p1_expanded = X_p1_tilde.unsqueeze(1)  # (B, 1, N1, 3)
            coord_diffs_p1 = X_i_expanded - X_p1_expanded  # (B, N, N1, 3)
            distances_sq_p1 = torch.sum(coord_diffs_p1 ** 2, dim=-1, keepdim=True)  # (B, N, N1, 1)
            distances_sq_p1 = torch.clamp(distances_sq_p1, min=0.0, max=1e6)
            
            # Ligand-p2: X_i^t - X_k (for k in I2)
            X_p2_expanded = X_p2_tilde.unsqueeze(1)  # (B, 1, N2, 3)
            coord_diffs_p2 = X_i_expanded - X_p2_expanded  # (B, N, N2, 3)
            distances_sq_p2 = torch.sum(coord_diffs_p2 ** 2, dim=-1, keepdim=True)  # (B, N, N2, 1)
            distances_sq_p2 = torch.clamp(distances_sq_p2, min=0.0, max=1e6)
            
            # Combine edge features for feature update
            # Ligand internal edges: [h_i, h_j, dist_sq, t]
            layer_input_ligand = torch.cat([
                h_i_ligand,  # (B, N, N, hidden_dim)
                h_j_ligand,  # (B, N, N, hidden_dim)
                distances_sq_ligand,  # (B, N, N, 1)
                t_emb_ligand  # (B, N, N, c_t)
            ], dim=-1)  # (B, N, N, 2*hidden_dim + 1 + c_t)
            
            # Ligand-p1 edges: [h_i, s_k^(1), dist_sq, t]
            layer_input_p1 = torch.cat([
                h_i_p1,  # (B, N, N1, hidden_dim)
                s1_tilde_expanded,  # (B, N, N1, hidden_dim)
                distances_sq_p1,  # (B, N, N1, 1)
                t_emb_p1  # (B, N, N1, c_t)
            ], dim=-1)  # (B, N, N1, 2*hidden_dim + 1 + c_t)
            
            # Ligand-p2 edges: [h_i, s_k^(2), dist_sq, t]
            layer_input_p2 = torch.cat([
                h_i_p2,  # (B, N, N2, hidden_dim)
                s2_tilde_expanded,  # (B, N, N2, hidden_dim)
                distances_sq_p2,  # (B, N, N2, 1)
                t_emb_p2  # (B, N, N2, c_t)
            ], dim=-1)  # (B, N, N2, 2*hidden_dim + 1 + c_t)
            
            # Compute messages for all edge types
            messages_ligand = layer(layer_input_ligand)  # (B, N, N, hidden_dim)
            messages_p1 = layer(layer_input_p1)  # (B, N, N1, hidden_dim)
            messages_p2 = layer(layer_input_p2)  # (B, N, N2, hidden_dim)
            
            # Aggregate messages (sum over neighbors, excluding self for ligand internal)
            # Ligand internal: exclude self-interactions
            self_mask = torch.eye(N, device=h.device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
            mol_mask_2d = mol_mask.unsqueeze(2) & mol_mask.unsqueeze(1)  # (B, N, N)
            valid_mask_ligand = (~self_mask) & mol_mask_2d.unsqueeze(-1)  # (B, N, N, 1)
            messages_ligand = messages_ligand * valid_mask_ligand.float()
            h_update_ligand = torch.sum(messages_ligand, dim=2)  # (B, N, hidden_dim)
            
            # Ligand-p1: mask invalid positions
            i1_mask_expanded = i1_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N1, 1)
            mol_mask_p1 = mol_mask.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
            valid_mask_p1 = i1_mask_expanded & mol_mask_p1  # (B, N, N1, 1)
            messages_p1 = messages_p1 * valid_mask_p1.float()
            h_update_p1 = torch.sum(messages_p1, dim=2)  # (B, N, hidden_dim)
            
            # Ligand-p2: mask invalid positions
            i2_mask_expanded = i2_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N2, 1)
            mol_mask_p2 = mol_mask.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
            valid_mask_p2 = i2_mask_expanded & mol_mask_p2  # (B, N, N2, 1)
            messages_p2 = messages_p2 * valid_mask_p2.float()
            h_update_p2 = torch.sum(messages_p2, dim=2)  # (B, N, hidden_dim)
            
            # Combine updates from all edge types
            h_update = h_update_ligand + h_update_p1 + h_update_p2  # (B, N, hidden_dim)
            
            # Residual connection
            h = h + h_update
            h = h * mol_mask.unsqueeze(-1)
        
            # Compute φ_X weights for coordinate updates using updated h
            # Re-expand features with updated h
            h_i_ligand_updated = h.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, hidden_dim)
            h_j_ligand_updated = h.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, hidden_dim)
            h_i_p1_updated = h.unsqueeze(2).expand(-1, -1, N1, -1)  # (B, N, N1, hidden_dim)
            h_i_p2_updated = h.unsqueeze(2).expand(-1, -1, N2, -1)  # (B, N, N2, hidden_dim)
            
            # Ligand internal: [h_i, h_j, ||X_i^t - X_j^t||², t]
            phi_input_ligand = torch.cat([
                h_i_ligand_updated,  # (B, N, N, hidden_dim)
                h_j_ligand_updated,  # (B, N, N, hidden_dim)
                distances_sq_ligand,  # (B, N, N, 1)
                t_emb_ligand  # (B, N, N, c_t)
            ], dim=-1)  # (B, N, N, 2*hidden_dim + 1 + c_t)
            
            # Ligand-p1: [h_i, s_k^(1), ||X_i^t - X_k||², t]
            phi_input_p1 = torch.cat([
                h_i_p1_updated,  # (B, N, N1, hidden_dim)
                s1_tilde_expanded,  # (B, N, N1, hidden_dim)
                distances_sq_p1,  # (B, N, N1, 1)
                t_emb_p1  # (B, N, N1, c_t)
            ], dim=-1)  # (B, N, N1, 2*hidden_dim + 1 + c_t)
            
            # Ligand-p2: [h_i, s_k^(2), ||X_i^t - X_k||², t]
            phi_input_p2 = torch.cat([
                h_i_p2_updated,  # (B, N, N2, hidden_dim)
                s2_tilde_expanded,  # (B, N, N2, hidden_dim)
                distances_sq_p2,  # (B, N, N2, 1)
                t_emb_p2  # (B, N, N2, c_t)
            ], dim=-1)  # (B, N, N2, 2*hidden_dim + 1 + c_t)
        
            # Compute φ_X weights
            phi_weights_ligand = self.phi_mlp(phi_input_ligand)  # (B, N, N, 1)
            phi_weights_p1 = self.phi_mlp_p1(phi_input_p1)  # (B, N, N1, 1)
            phi_weights_p2 = self.phi_mlp_p2(phi_input_p2)  # (B, N, N2, 1)
            
            # Apply masks
            invalid_mask_ligand = self_mask | ~mol_mask_2d.unsqueeze(-1)
            phi_weights_ligand = phi_weights_ligand.masked_fill(invalid_mask_ligand, 0.0)
            phi_weights_p1 = phi_weights_p1.masked_fill(~valid_mask_p1, 0.0)
            phi_weights_p2 = phi_weights_p2.masked_fill(~valid_mask_p2, 0.0)
            
            # Clamp weights
            phi_weights_ligand = torch.clamp(phi_weights_ligand, min=-10.0, max=10.0)
            phi_weights_p1 = torch.clamp(phi_weights_p1, min=-10.0, max=10.0)
            phi_weights_p2 = torch.clamp(phi_weights_p2, min=-10.0, max=10.0)
            
            # Compute coordinate updates
            # Ligand internal
            coord_update_ligand = torch.sum(coord_diffs_ligand * phi_weights_ligand, dim=2)  # (B, N, 3)
            
            # Ligand-p1: normalize direction vector for stability
            coord_diffs_p1_norm = torch.sqrt(distances_sq_p1 + 1e-8) + 1.0  # (B, N, N1, 1)
            coord_update_p1 = torch.sum(
                (coord_diffs_p1 / coord_diffs_p1_norm) * phi_weights_p1, dim=2
            )  # (B, N, 3)
            
            # Ligand-p2: normalize direction vector for stability
            coord_diffs_p2_norm = torch.sqrt(distances_sq_p2 + 1e-8) + 1.0  # (B, N, N2, 1)
            coord_update_p2 = torch.sum(
                (coord_diffs_p2 / coord_diffs_p2_norm) * phi_weights_p2, dim=2
            )  # (B, N, 3)
            
            # Combine all updates
            coord_update = coord_update_ligand + coord_update_p1 + coord_update_p2  # (B, N, 3)
            
            # Only update ligand coordinates (like egnn.py: x = x + delta_x * mask_ligand)
            coord_update = coord_update * mol_mask.unsqueeze(-1)  # (B, N, 3)
            
            # Limit the magnitude of coordinate updates per step (prevent explosion)
            coord_update_norm = torch.norm(coord_update, dim=-1, keepdim=True)  # (B, N, 1)
            max_update_norm = 3.0  # Maximum update per step in Angstroms
            coord_update = coord_update * torch.clamp(max_update_norm / (coord_update_norm + 1e-8), max=1.0)
            
            X_tilde = X_tilde + coord_update

            # Recompute coord_diffs and distances_sq using updated coordinates
            X_i = X_tilde.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, 3]
            X_j = X_tilde.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, 3]
            coord_diffs = X_i - X_j     # [B, N, N, 3]
            distances_sq = torch.sum(coord_diffs ** 2, dim=-1, keepdim=True)  # [B, N, N, 1]
            
            # Clamp distances_sq to prevent numerical issues (very large distances)
            distances_sq = torch.clamp(distances_sq, min=0.0, max=1e6)  # Max distance ~1000 Angstroms

        return X_tilde

class PhiA(nn.Module):
    """
    It cross-attends protein features to ligand atoms and outputs sequence prediction probabilities φ_A.
    Following the equation: a_i^pred = Σ_j ||X_i^pred - X_j^pred||² φ_A(s̃^(1), s̃^(2), ã_j^t, t)
    """
    def __init__(self, c_s, c_t, hidden_dim, num_classes=25):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.proj_s1 = nn.Linear(c_s, hidden_dim)
        self.proj_s2 = nn.Linear(c_s, hidden_dim)
        
        # Projection for sequence embeddings ã_j^t
        self.proj_a = nn.Linear(c_s, hidden_dim)
        
        self.cross_attn_1 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.cross_attn_2 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # MLP outputs probability distribution over num_classes atom types
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 + c_t, hidden_dim),  # 3H for s1, s2, a + c_t for time
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)  # Output probabilities for atom types
        )

    def forward(self, s1, s2, sl, t_emb, p1_mask, p2_mask, mol_mask):
        """
        Args:
            s1: (B, N1, c_s) - protein 1 features s̃^(1)
            s2: (B, N2, c_s) - protein 2 features s̃^(2)
            a_tilde: (B, N, c_s) - sequence embeddings ã_j^t at timestep t
            t_emb: (B, N, c_t) - time embedding
            p1_mask, p2_mask, mol_mask: boolean masks
        Returns:
            φ_A probabilities: (B, N, num_classes) - probability distribution over atom types
        """
        # B, N, c_s = a_tilde.shape
        # c_t = t_emb.shape[-1]

        # Linear projections for protein features
        s1_proj = self.proj_s1(s1)  # [B, N1, H]
        s2_proj = self.proj_s2(s2)  # [B, N2, H]
        
        # Projection for sequence embeddings and apply mask to exclude padding
        # a_proj = self.proj_a(a_tilde)  # [B, N, H]
        a_proj = self.proj_a(sl)
        a_proj = a_proj * mol_mask.unsqueeze(-1)  # Zero out padding positions

        # Cross-attend ligand sequence features to protein 1 and 2
        attn_out_1, _ = self.cross_attn_1(a_proj, s1_proj, s1_proj,
                                          key_padding_mask=~p1_mask)
        attn_out_2, _ = self.cross_attn_2(a_proj, s2_proj, s2_proj,
                                          key_padding_mask=~p2_mask)

        # Apply mask to time embedding as well
        t_emb_masked = t_emb * mol_mask.unsqueeze(-1)
        
        # Combine attended features, sequence features, and time embedding
        combined = torch.cat([attn_out_1, attn_out_2, a_proj, t_emb_masked], dim=-1)  # [B, N, 3H + c_t]

        # Output probability distribution over atom types
        phi_probs = self.mlp(combined)  # [B, N, num_classes]
        
        # Numerical stability: clamp inf/nan values
        phi_probs = torch.clamp(phi_probs, min=-1e6, max=1e6)  # Prevent extreme values
        phi_probs = torch.where(torch.isnan(phi_probs), torch.zeros_like(phi_probs), phi_probs)  # Replace NaN with 0
        phi_probs = torch.where(torch.isinf(phi_probs), torch.zeros_like(phi_probs), phi_probs)  # Replace Inf with 0
        
        # Apply mask to invalid positions (final safeguard)
        phi_probs = phi_probs * mol_mask.unsqueeze(-1)

        return phi_probs

# class PhiRT(nn.Module):
#     """
#     Predicts rotation matrix R and translation vector t using features from proteins.
#     Formula: 
#         Δω = φ_R(s1, s2, p1_coords, p2_coords, R*, t*, Σ_1, Σ_2, t_emb)
#         Δt = φ_T(s1, s2, p1_coords, p2_coords, R*, t*, Σ_1, Σ_2, t_emb)
#         R̂₁ = exp([Δω]×) @ R*
#         t̂₁ = Δt + t*
#     """
#     def __init__(self, c_s, c_t, hidden_dim, num_heads=4):
#         super().__init__()
#         self.c_s = c_s
#         self.c_t = c_t
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
        
#         # Project protein features to hidden dimension
#         self.proj_s1 = nn.Linear(c_s, hidden_dim)
#         self.proj_s2 = nn.Linear(c_s, hidden_dim)
        
#         # Project rotation matrix R* (flatten 3x3 to 9-dim, then to hidden_dim)
#         self.proj_R = nn.Linear(9, hidden_dim)
        
#         # Project translation vector t* to hidden dimension
#         self.proj_t = nn.Linear(3, hidden_dim)
        
#         # Project noised rotation matrix R_tilde (flatten 3x3 to 9-dim, then to hidden_dim)
#         self.proj_R_tilde = nn.Linear(9, hidden_dim)
        
#         # Project noised translation vector t_tilde to hidden dimension
#         self.proj_t_tilde = nn.Linear(3, hidden_dim)
        
#         # Time embedding projection
#         self.proj_time = nn.Linear(c_t, hidden_dim)
        
#         # MLP for delta_omega prediction (rotation vector)
#         # Input: 2*hidden_dim (aggregated s1, s2) + 6 (weighted coords1, coords2)
#         # Total: 2*hidden_dim + 6
#         self.mlp_R_delta = nn.Sequential(
#             nn.Linear(hidden_dim * 2 + 6, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 3)  # Output rotation vector Δω in R^3
#         )
        
#         # Keypoint generation layers (similar to interface_model.py)
#         # Generate 2 keypoints for each protein
#         self.num_keypoints = 20  # K = 20
        
#         # Feature transformation before computing mean (φ in the paper)
#         self.mlp_h_mean_ROT = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#         )
        
#         # Key projection: transforms features to keys for attention
#         self.att_mlp_key_ROT = nn.Linear(hidden_dim, self.num_keypoints * hidden_dim, bias=False)
        
#         # Query projection: transforms mean features to queries for attention
#         self.att_mlp_query_ROT = nn.Linear(hidden_dim, self.num_keypoints * hidden_dim, bias=False)

#     def _generate_keypoints(self, feats1, coors1, feats2, coors2, mask1=None, mask2=None):
#         """
#         Generate K keypoints for each protein using cross-attention.
#         Similar to interface_model.py _generate_keypoints method.
        
#         Args:
#             feats1: (B, N1, d) - features for protein 1
#             coors1: (B, N1, 3) - coordinates for protein 1
#             feats2: (B, N2, d) - features for protein 2
#             coors2: (B, N2, 3) - coordinates for protein 2
#             mask1: (B, N1) - mask for protein 1 (optional)
#             mask2: (B, N2) - mask for protein 2 (optional)
        
#         Returns:
#             Y1: (B, K, 3) - keypoints for protein 1
#             Y2: (B, K, 3) - keypoints for protein 2
#             attn1: (B, K, N1) - attention weights for protein 1
#             attn2: (B, K, N2) - attention weights for protein 2
#         """
#         B, N1, d = feats1.shape
#         B, N2, d = feats2.shape
#         K = self.num_keypoints
        
#         # Compute mean features μ(φ(H₁)) and μ(φ(H₂))
#         # Apply transformation φ first
#         feats1_transformed = self.mlp_h_mean_ROT(feats1)  # (B, N1, d)
#         feats2_transformed = self.mlp_h_mean_ROT(feats2)  # (B, N2, d)
        
#         # Compute mean
#         if mask1 is not None:
#             mask1_expanded = mask1.unsqueeze(-1)  # (B, N1, 1)
#             feats1_masked = feats1_transformed * mask1_expanded
#             H1_mean = feats1_masked.sum(dim=1, keepdim=True) / (mask1.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6)  # (B, 1, d)
#         else:
#             H1_mean = feats1_transformed.mean(dim=1, keepdim=True)  # (B, 1, d)
        
#         if mask2 is not None:
#             mask2_expanded = mask2.unsqueeze(-1)  # (B, N2, 1)
#             feats2_masked = feats2_transformed * mask2_expanded
#             H2_mean = feats2_masked.sum(dim=1, keepdim=True) / (mask2.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6)  # (B, 1, d)
#         else:
#             H2_mean = feats2_transformed.mean(dim=1, keepdim=True)  # (B, 1, d)
        
#         # Generate Y2 (keypoints for protein 2)
#         # Query: μ(φ(H₁)) (mean of protein 1 features)
#         # Keys: h₂ⱼ (each feature from protein 2)
#         # Formula: βⱼᵏ = softmaxⱼ (¹/√ᵈ h₂ⱼᵀ W μ(φ(H₁)))
#         keys_2 = self.att_mlp_key_ROT(feats2)  # (B, N2, K*d)
#         keys_2 = keys_2.view(B, N2, K, d)  # (B, N2, K, d)
#         keys_2 = keys_2.transpose(1, 2)  # (B, K, N2, d)
        
#         query_2 = self.att_mlp_query_ROT(H1_mean)  # (B, 1, K*d)
#         query_2 = query_2.view(B, 1, K, d)  # (B, 1, K, d)
#         query_2 = query_2.transpose(1, 2)  # (B, K, 1, d)
#         query_2 = query_2.transpose(2, 3)  # (B, K, d, 1)
        
#         # Attention scores: (B, K, N2, 1)
#         att_scores_2 = (keys_2 @ query_2) / math.sqrt(d)  # (B, K, N2, 1)
        
#         # Apply mask if provided
#         if mask2 is not None:
#             mask2_expanded = mask2.unsqueeze(1).unsqueeze(-1)  # (B, 1, N2, 1)
#             mask2_expanded = mask2_expanded.expand(-1, K, -1, -1)  # (B, K, N2, 1)
#             att_scores_2 = att_scores_2.masked_fill(~mask2_expanded.bool(), -1e9)
        
#         attn2 = F.softmax(att_scores_2.squeeze(-1), dim=-1)  # (B, K, N2)
        
#         # Compute Y2: weighted sum of coordinates
#         # y₂ₖ := Σⱼ₌₁ᵐ βⱼᵏ z₂ⱼ
#         Y2 = einsum(attn2, coors2, 'b k n2, b n2 d -> b k d')  # (B, K, 3)
        
#         # Generate Y1 (keypoints for protein 1)
#         # Query: μ(φ(H₂)) (mean of protein 2 features)
#         # Keys: h₁ᵢ (each feature from protein 1)
#         # Formula: αᵢᵏ = softmaxᵢ (¹/√ᵈ h₁ᵢᵀ W μ(φ(H₂)))
#         keys_1 = self.att_mlp_key_ROT(feats1)  # (B, N1, K*d)
#         keys_1 = keys_1.view(B, N1, K, d)  # (B, N1, K, d)
#         keys_1 = keys_1.transpose(1, 2)  # (B, K, N1, d)
        
#         query_1 = self.att_mlp_query_ROT(H2_mean)  # (B, 1, K*d)
#         query_1 = query_1.view(B, 1, K, d)  # (B, 1, K, d)
#         query_1 = query_1.transpose(1, 2)  # (B, K, 1, d)
#         query_1 = query_1.transpose(2, 3)  # (B, K, d, 1)
        
#         # Attention scores: (B, K, N1, 1)
#         att_scores_1 = (keys_1 @ query_1) / math.sqrt(d)  # (B, K, N1, 1)
        
#         # Apply mask if provided
#         if mask1 is not None:
#             mask1_expanded = mask1.unsqueeze(1).unsqueeze(-1)  # (B, 1, N1, 1)
#             mask1_expanded = mask1_expanded.expand(-1, K, -1, -1)  # (B, K, N1, 1)
#             att_scores_1 = att_scores_1.masked_fill(~mask1_expanded.bool(), -1e9)
        
#         attn1 = F.softmax(att_scores_1.squeeze(-1), dim=-1)  # (B, K, N1)
        
#         # Compute Y1: weighted sum of coordinates
#         # y₁ₖ := Σᵢ₌₁ⁿ αᵢᵏ z₁ᵢ
#         Y1 = einsum(attn1, coors1, 'b k n1, b n1 d -> b k d')  # (B, K, 3)
        
#         return Y1, Y2, attn1, attn2

#     def forward(self, s1, p1_coords, s2, p2_coords, R_star, t_star, R_tilde, t_tilde, t_emb, p1_mask, p2_mask, sigma_1, sigma_2):
#         """
#         Args:
#             s1: (B, N1, c_s) - protein 1 features
#             s2: (B, N2, c_s) - protein 2 features
#             p1_coords: (B, N1, 3) - protein 1 coordinates
#             p2_coords: (B, N2, 3) - protein 2 coordinates (moved)
#             R_star: (B, 3, 3) - initial rotation matrix from Kabsch
#             t_star: (B, 3) - initial translation vector from Kabsch
#             R_tilde: (B, 3, 3) - noised rotation matrix at timestep t
#             t_tilde: (B, 3) - noised translation vector at timestep t
#             t_emb: (B, c_t) or (B, 1, c_t) - time embedding
#             p1_mask: (B, N1) - mask for protein 1 residues
#             p2_mask: (B, N2) - mask for protein 2 residues
#             sigma_1: (B, 3, 3) - optional interface representation for protein 1
#             sigma_2: (B, 3, 3) - optional interface representation for protein 2
        
#         Returns:
#             R_pred: (B, 3, 3) - predicted rotation matrix R̂₁
#             t_pred: (B, 3) - predicted translation vector t̂₁
#         """
#         B, N1, _ = s1.shape
#         N2 = s2.shape[1]
        
#         # Fuse R_star, t_star, R_tilde, t_tilde, time into features
#         ctx = (self.proj_R(R_star.view(B, -1)[:, :9]) + 
#                self.proj_t(t_star.view(B, -1)[:, :3]) + 
#                self.proj_R_tilde(R_tilde.view(B, -1)[:, :9]) + 
#                self.proj_t_tilde(t_tilde.view(B, -1)[:, :3]) + 
#                self.proj_time(t_emb.view(B, -1)[:, :self.c_t]))  # (B, hidden_dim)
#         s1 = s1 + ctx.unsqueeze(1)  # (B, N1, hidden_dim)
#         s2 = s2 + ctx.unsqueeze(1)  # (B, N2, hidden_dim)
        
#         # Generate keypoints using cross-attention
#         # Y1: (B, K, 3) - keypoints for protein 1
#         # Y2: (B, K, 3) - keypoints for protein 2
#         Y1, Y2, attn1, attn2 = self._generate_keypoints(
#             s1, p1_coords,
#             s2, p2_coords,
#             p1_mask, p2_mask
#         )
#         _, K, _ = Y2.shape
        
#         # Aggregate features and predict delta_omega (rotation vector)
#         s1_mean = s1.mean(dim=1)  # (B, hidden_dim)
#         s2_mean = s2.mean(dim=1)  # (B, hidden_dim)
#         Y1_mean = Y1.mean(dim=1)  # (B, 3)
#         Y2_mean = Y2.mean(dim=1)  # (B, 3)
#         delta_omega = self.mlp_R_delta(torch.cat([s1_mean, s2_mean, Y1_mean, Y2_mean], dim=-1))  # (B, 3)
        
#         # Compute ΔR = exp([Δω]×) using exponential map
#         R_delta = rotvec_to_rotmat(delta_omega)  # (B, 3, 3)
        
#         # Compute delta t
#         delta_t = Y2_mean  # (B, 3)
        
#         # Compute final rotation: R̂₁ = ΔR @ R_tilde
#         R_pred = R_delta @ R_tilde  # (B, 3, 3) @ (B, 3, 3) = (B, 3, 3)
        
#         # Compute final translation: t̂₁ = Δt + t_tilde
#         t_pred = delta_t + t_tilde  # (B, 3) + (B, 3) = (B, 3)
        
#         return R_pred, t_pred

# class PhiRT(nn.Module):
#     """
#     Predicts rotation matrix R and translation vector t using features from proteins.
#     Formula: 
#         Δω = φ_R(s1, s2, p1_coords, p2_coords, R*, t*, Σ_1, Σ_2, t_emb)
#         Δt = φ_T(s1, s2, p1_coords, p2_coords, R*, t*, Σ_1, Σ_2, t_emb)
#         R̂₁ = exp([Δω]×) @ R*
#         t̂₁ = Δt + t*
#     """
#     def __init__(self, c_s, c_t, hidden_dim, num_heads=4):
#         super().__init__()
#         self.c_s = c_s
#         self.c_t = c_t
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
        
#         # MLP for delta_omega prediction (rotation vector)
#         # Input: 2*hidden_dim (aggregated s1, s2)
#         self.mlp_R_delta = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim * 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 3)  # Output rotation vector Δω in R^3
#         )
        
#         # Keypoint generation layers (similar to interface_model.py)
#         # Generate 2 keypoints for each protein
#         self.num_keypoints = 20  # K = 20
        
#         # Feature transformation before computing mean (φ in the paper)
#         self.mlp_h_mean_ROT = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#         )
        
#         # Key projection: transforms features to keys for attention
#         self.att_mlp_key_ROT = nn.Linear(hidden_dim, self.num_keypoints * hidden_dim, bias=False)
        
#         # Query projection: transforms mean features to queries for attention
#         self.att_mlp_query_ROT = nn.Linear(hidden_dim, self.num_keypoints * hidden_dim, bias=False)

#     def _generate_keypoints(self, feats1, coors1, feats2, coors2, mask1=None, mask2=None):
#         """
#         Generate K keypoints for each protein using cross-attention.
#         Similar to interface_model.py _generate_keypoints method.
        
#         Args:
#             feats1: (B, N1, d) - features for protein 1
#             coors1: (B, N1, 3) - coordinates for protein 1
#             feats2: (B, N2, d) - features for protein 2
#             coors2: (B, N2, 3) - coordinates for protein 2
#             mask1: (B, N1) - mask for protein 1 (optional)
#             mask2: (B, N2) - mask for protein 2 (optional)
        
#         Returns:
#             Y1: (B, K, 3) - keypoints for protein 1
#             Y2: (B, K, 3) - keypoints for protein 2
#             attn1: (B, K, N1) - attention weights for protein 1
#             attn2: (B, K, N2) - attention weights for protein 2
#         """
#         B, N1, d = feats1.shape
#         B, N2, d = feats2.shape
#         K = self.num_keypoints
        
#         # Compute mean features μ(φ(H₁)) and μ(φ(H₂))
#         # Apply transformation φ first
#         feats1_transformed = self.mlp_h_mean_ROT(feats1)  # (B, N1, d)
#         feats2_transformed = self.mlp_h_mean_ROT(feats2)  # (B, N2, d)
        
#         # Compute mean
#         if mask1 is not None:
#             mask1_expanded = mask1.unsqueeze(-1)  # (B, N1, 1)
#             feats1_masked = feats1_transformed * mask1_expanded
#             H1_mean = feats1_masked.sum(dim=1, keepdim=True) / (mask1.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6)  # (B, 1, d)
#         else:
#             H1_mean = feats1_transformed.mean(dim=1, keepdim=True)  # (B, 1, d)
        
#         if mask2 is not None:
#             mask2_expanded = mask2.unsqueeze(-1)  # (B, N2, 1)
#             feats2_masked = feats2_transformed * mask2_expanded
#             H2_mean = feats2_masked.sum(dim=1, keepdim=True) / (mask2.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6)  # (B, 1, d)
#         else:
#             H2_mean = feats2_transformed.mean(dim=1, keepdim=True)  # (B, 1, d)
        
#         # Generate Y2 (keypoints for protein 2)
#         # Query: μ(φ(H₁)) (mean of protein 1 features)
#         # Keys: h₂ⱼ (each feature from protein 2)
#         # Formula: βⱼᵏ = softmaxⱼ (¹/√ᵈ h₂ⱼᵀ W μ(φ(H₁)))
#         keys_2 = self.att_mlp_key_ROT(feats2)  # (B, N2, K*d)
#         keys_2 = keys_2.view(B, N2, K, d)  # (B, N2, K, d)
#         keys_2 = keys_2.transpose(1, 2)  # (B, K, N2, d)
        
#         query_2 = self.att_mlp_query_ROT(H1_mean)  # (B, 1, K*d)
#         query_2 = query_2.view(B, 1, K, d)  # (B, 1, K, d)
#         query_2 = query_2.transpose(1, 2)  # (B, K, 1, d)
#         query_2 = query_2.transpose(2, 3)  # (B, K, d, 1)
        
#         # Attention scores: (B, K, N2, 1)
#         att_scores_2 = (keys_2 @ query_2) / math.sqrt(d)  # (B, K, N2, 1)
        
#         # Apply mask if provided
#         if mask2 is not None:
#             mask2_expanded = mask2.unsqueeze(1).unsqueeze(-1)  # (B, 1, N2, 1)
#             mask2_expanded = mask2_expanded.expand(-1, K, -1, -1)  # (B, K, N2, 1)
#             att_scores_2 = att_scores_2.masked_fill(~mask2_expanded.bool(), -1e9)
        
#         attn2 = F.softmax(att_scores_2.squeeze(-1), dim=-1)  # (B, K, N2)
        
#         # Compute Y2: weighted sum of coordinates
#         # y₂ₖ := Σⱼ₌₁ᵐ βⱼᵏ z₂ⱼ
#         Y2 = einsum(attn2, coors2, 'b k n2, b n2 d -> b k d')  # (B, K, 3)
        
#         # Generate Y1 (keypoints for protein 1)
#         # Query: μ(φ(H₂)) (mean of protein 2 features)
#         # Keys: h₁ᵢ (each feature from protein 1)
#         # Formula: αᵢᵏ = softmaxᵢ (¹/√ᵈ h₁ᵢᵀ W μ(φ(H₂)))
#         keys_1 = self.att_mlp_key_ROT(feats1)  # (B, N1, K*d)
#         keys_1 = keys_1.view(B, N1, K, d)  # (B, N1, K, d)
#         keys_1 = keys_1.transpose(1, 2)  # (B, K, N1, d)
        
#         query_1 = self.att_mlp_query_ROT(H2_mean)  # (B, 1, K*d)
#         query_1 = query_1.view(B, 1, K, d)  # (B, 1, K, d)
#         query_1 = query_1.transpose(1, 2)  # (B, K, 1, d)
#         query_1 = query_1.transpose(2, 3)  # (B, K, d, 1)
        
#         # Attention scores: (B, K, N1, 1)
#         att_scores_1 = (keys_1 @ query_1) / math.sqrt(d)  # (B, K, N1, 1)
        
#         # Apply mask if provided
#         if mask1 is not None:
#             mask1_expanded = mask1.unsqueeze(1).unsqueeze(-1)  # (B, 1, N1, 1)
#             mask1_expanded = mask1_expanded.expand(-1, K, -1, -1)  # (B, K, N1, 1)
#             att_scores_1 = att_scores_1.masked_fill(~mask1_expanded.bool(), -1e9)
        
#         attn1 = F.softmax(att_scores_1.squeeze(-1), dim=-1)  # (B, K, N1)
        
#         # Compute Y1: weighted sum of coordinates
#         # y₁ₖ := Σᵢ₌₁ⁿ αᵢᵏ z₁ᵢ
#         Y1 = einsum(attn1, coors1, 'b k n1, b n1 d -> b k d')  # (B, K, 3)
        
#         return Y1, Y2, attn1, attn2

#     def forward(self, s1, p1_coords, s2, p2_coords, R_star, t_star, R_tilde, t_tilde, t_emb, p1_mask, p2_mask, sigma_1, sigma_2):
#         """
#         Args:
#             s1: (B, N1, c_s) - protein 1 features
#             s2: (B, N2, c_s) - protein 2 features
#             p1_coords: (B, N1, 3) - protein 1 coordinates
#             p2_coords: (B, N2, 3) - protein 2 coordinates (moved)
#             R_star: (B, 3, 3) - initial rotation matrix from Kabsch
#             t_star: (B, 3) - initial translation vector from Kabsch
#             R_tilde: (B, 3, 3) - noised rotation matrix at timestep t
#             t_tilde: (B, 3) - noised translation vector at timestep t
#             t_emb: (B, c_t) or (B, 1, c_t) - time embedding
#             p1_mask: (B, N1) - mask for protein 1 residues
#             p2_mask: (B, N2) - mask for protein 2 residues
#             sigma_1: (B, 3, 3) - optional interface representation for protein 1
#             sigma_2: (B, 3, 3) - optional interface representation for protein 2
        
#         Returns:
#             R_pred: (B, 3, 3) - predicted rotation matrix R̂₁
#             t_pred: (B, 3) - predicted translation vector t̂₁
#         """
#         B, N1, _ = s1.shape
#         N2 = s2.shape[1]
        
#         # Generate keypoints using cross-attention
#         # Y1: (B, K, 3) - keypoints for protein 1
#         # Y2: (B, K, 3) - keypoints for protein 2
#         Y1, Y2, attn1, attn2 = self._generate_keypoints(
#             s1, p1_coords,
#             s2, p2_coords,
#             p1_mask, p2_mask
#         )
#         _, K, _ = Y2.shape
        
#         t_pred = Y1.mean(dim=1) + Y2.mean(dim=1)  # (B, 3)

#         # Aggregate s1, s2 and predict delta_omega
#         s1_mean = s1.mean(dim=1)  # (B, c_s)
#         s2_mean = s2.mean(dim=1)  # (B, c_s)
#         delta_omega = self.mlp_R_delta(torch.cat([s1_mean, s2_mean], dim=-1))  # (B, 3)
#         # Compute ΔR = exp([Δω]×) using exponential map
#         R_pred = rotvec_to_rotmat(delta_omega)  # (B, 3, 3)
        
#         return R_pred, t_pred

class PhiRT(nn.Module):
    """
    Predicts rotation matrix R and translation vector t using features from proteins.
    Formula: 
        Δω = φ_R(s1, s2, p1_coords, p2_coords, R*, t*, Σ_1, Σ_2, t_emb)
        Δt = φ_T(s1, s2, p1_coords, p2_coords, R*, t*, Σ_1, Σ_2, t_emb)
        R̂₁ = exp([Δω]×) @ R*
        t̂₁ = Δt + t*
    """
    def __init__(self, c_s, c_t, hidden_dim, num_heads=4):
        super().__init__()
        self.c_s = c_s
        self.c_t = c_t
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_refine_steps = 10
        self.feat_dim = c_s
        self.num_keypoints = 50  # Keypoint numbers
        
        # Project s1/s2 then compute cross-attention context.
        self.proj_s1 = nn.Linear(c_s, hidden_dim)
        self.proj_s2 = nn.Linear(c_s, hidden_dim)
        self.cross_attn_rt = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Use cross-attention context to predict keypoint weights for Y1/Y2.
        self.y1_weight_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_keypoints)
        )
        self.y2_weight_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_keypoints)
        )

        # Iterative residual head for [rot6d, delta_t].
        # Input: relative weighted coordinate (y1_weighted - y2_weighted), 3D
        self.mlp_rt_delta = nn.Sequential(
            nn.Linear(3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 9)  # [rot6d(6), delta_t(3)]
        )
        
        # Feature transformation before computing mean (φ in the paper)
        self.mlp_h_mean_ROT = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
        )
        
        # Key projection: transforms features to keys for attention
        self.att_mlp_key_ROT = nn.Linear(self.feat_dim, self.num_keypoints * self.feat_dim, bias=False)
        
        # Query projection: transforms mean features to queries for attention
        self.att_mlp_query_ROT = nn.Linear(self.feat_dim, self.num_keypoints * self.feat_dim, bias=False)

    def _masked_mean(self, x, mask):
        """Masked mean over sequence dimension."""
        if mask is None:
            return x.mean(dim=1)
        mask = mask.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return (x * mask).sum(dim=1) / denom

    def _rot6d_to_rotmat(self, rot_6d):
        """Convert 6D rotation representation to rotation matrix."""
        # rot_6d: (B, 6)
        a1 = rot_6d[:, 0:3]
        a2 = rot_6d[:, 3:6]

        b1 = F.normalize(a1, dim=-1)
        # Remove b1 component from a2, then normalize (Gram-Schmidt).
        proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = F.normalize(a2 - proj, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)

        # Assemble columns to get (B, 3, 3).
        return torch.stack([b1, b2, b3], dim=-1)

    def _generate_keypoints(self, feats1, coors1, feats2, coors2, mask1=None, mask2=None):
        """
        Generate K keypoints for each protein using cross-attention.
        Similar to interface_model.py _generate_keypoints method.
        
        Args:
            feats1: (B, N1, d) - features for protein 1
            coors1: (B, N1, 3) - coordinates for protein 1
            feats2: (B, N2, d) - features for protein 2
            coors2: (B, N2, 3) - coordinates for protein 2
            mask1: (B, N1) - mask for protein 1 (optional)
            mask2: (B, N2) - mask for protein 2 (optional)
        
        Returns:
            Y1: (B, K, 3) - keypoints for protein 1
            Y2: (B, K, 3) - keypoints for protein 2
            attn1: (B, K, N1) - attention weights for protein 1
            attn2: (B, K, N2) - attention weights for protein 2
        """
        B, N1, d = feats1.shape
        B, N2, d = feats2.shape
        K = self.num_keypoints
        
        # Compute mean features μ(φ(H₁)) and μ(φ(H₂))
        # Apply transformation φ first
        feats1_transformed = self.mlp_h_mean_ROT(feats1)  # (B, N1, d)
        feats2_transformed = self.mlp_h_mean_ROT(feats2)  # (B, N2, d)
        
        # Compute mean
        if mask1 is not None:
            mask1_expanded = mask1.unsqueeze(-1)  # (B, N1, 1)
            feats1_masked = feats1_transformed * mask1_expanded
            H1_mean = feats1_masked.sum(dim=1, keepdim=True) / (mask1.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6)  # (B, 1, d)
        else:
            H1_mean = feats1_transformed.mean(dim=1, keepdim=True)  # (B, 1, d)
        
        if mask2 is not None:
            mask2_expanded = mask2.unsqueeze(-1)  # (B, N2, 1)
            feats2_masked = feats2_transformed * mask2_expanded
            H2_mean = feats2_masked.sum(dim=1, keepdim=True) / (mask2.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6)  # (B, 1, d)
        else:
            H2_mean = feats2_transformed.mean(dim=1, keepdim=True)  # (B, 1, d)
        
        # Generate Y2 (keypoints for protein 2)
        # Query: μ(φ(H₁)) (mean of protein 1 features)
        # Keys: h₂ⱼ (each feature from protein 2)
        # Formula: βⱼᵏ = softmaxⱼ (¹/√ᵈ h₂ⱼᵀ W μ(φ(H₁)))
        keys_2 = self.att_mlp_key_ROT(feats2)  # (B, N2, K*d)
        keys_2 = keys_2.view(B, N2, K, d)  # (B, N2, K, d)
        keys_2 = keys_2.transpose(1, 2)  # (B, K, N2, d)
        
        query_2 = self.att_mlp_query_ROT(H1_mean)  # (B, 1, K*d)
        query_2 = query_2.view(B, 1, K, d)  # (B, 1, K, d)
        query_2 = query_2.transpose(1, 2)  # (B, K, 1, d)
        query_2 = query_2.transpose(2, 3)  # (B, K, d, 1)
        
        # Attention scores: (B, K, N2, 1)
        att_scores_2 = (keys_2 @ query_2) / math.sqrt(d)  # (B, K, N2, 1)
        
        # Apply mask if provided
        if mask2 is not None:
            mask2_expanded = mask2.unsqueeze(1).unsqueeze(-1)  # (B, 1, N2, 1)
            mask2_expanded = mask2_expanded.expand(-1, K, -1, -1)  # (B, K, N2, 1)
            att_scores_2 = att_scores_2.masked_fill(~mask2_expanded.bool(), -1e9)
        
        attn2 = F.softmax(att_scores_2.squeeze(-1), dim=-1)  # (B, K, N2)
        
        # Compute Y2: weighted sum of coordinates
        # y₂ₖ := Σⱼ₌₁ᵐ βⱼᵏ z₂ⱼ
        Y2 = einsum(attn2, coors2, 'b k n2, b n2 d -> b k d')  # (B, K, 3)
        
        # Generate Y1 (keypoints for protein 1)
        # Query: μ(φ(H₂)) (mean of protein 2 features)
        # Keys: h₁ᵢ (each feature from protein 1)
        # Formula: αᵢᵏ = softmaxᵢ (¹/√ᵈ h₁ᵢᵀ W μ(φ(H₂)))
        keys_1 = self.att_mlp_key_ROT(feats1)  # (B, N1, K*d)
        keys_1 = keys_1.view(B, N1, K, d)  # (B, N1, K, d)
        keys_1 = keys_1.transpose(1, 2)  # (B, K, N1, d)
        
        query_1 = self.att_mlp_query_ROT(H2_mean)  # (B, 1, K*d)
        query_1 = query_1.view(B, 1, K, d)  # (B, 1, K, d)
        query_1 = query_1.transpose(1, 2)  # (B, K, 1, d)
        query_1 = query_1.transpose(2, 3)  # (B, K, d, 1)
        
        # Attention scores: (B, K, N1, 1)
        att_scores_1 = (keys_1 @ query_1) / math.sqrt(d)  # (B, K, N1, 1)
        
        # Apply mask if provided
        if mask1 is not None:
            mask1_expanded = mask1.unsqueeze(1).unsqueeze(-1)  # (B, 1, N1, 1)
            mask1_expanded = mask1_expanded.expand(-1, K, -1, -1)  # (B, K, N1, 1)
            att_scores_1 = att_scores_1.masked_fill(~mask1_expanded.bool(), -1e9)
        
        attn1 = F.softmax(att_scores_1.squeeze(-1), dim=-1)  # (B, K, N1)
        
        # Compute Y1: weighted sum of coordinates
        # y₁ₖ := Σᵢ₌₁ⁿ αᵢᵏ z₁ᵢ
        Y1 = einsum(attn1, coors1, 'b k n1, b n1 d -> b k d')  # (B, K, 3)
        
        return Y1, Y2, attn1, attn2

    def forward(self, Y1, Y2, s1, p1_coords, s2, p2_coords, R_star, t_star, R_tilde, t_tilde, t_emb, p1_mask, p2_mask, sigma_1, sigma_2):
        """
        Args:
            s1: (B, N1, c_s) - protein 1 features
            s2: (B, N2, c_s) - protein 2 features
            p1_coords: (B, N1, 3) - protein 1 coordinates
            p2_coords: (B, N2, 3) - protein 2 coordinates (moved)
            R_star: (B, 3, 3) - initial rotation matrix from Kabsch
            t_star: (B, 3) - initial translation vector from Kabsch
            R_tilde: (B, 3, 3) - noised rotation matrix at timestep t
            t_tilde: (B, 3) - noised translation vector at timestep t
            t_emb: (B, c_t) or (B, 1, c_t) - time embedding
            p1_mask: (B, N1) - mask for protein 1 residues
            p2_mask: (B, N2) - mask for protein 2 residues
            sigma_1: (B, 3, 3) - optional interface representation for protein 1
            sigma_2: (B, 3, 3) - optional interface representation for protein 2
        
        Returns:
            R_pred: (B, 3, 3) - predicted rotation matrix R̂₁
            t_pred: (B, 3) - predicted translation vector t̂₁
        """
        B = s1.shape[0]
        # Keep R_tilde / t_tilde / t_emb in the signature for compatibility,
        # but the iterative PhiRT no longer depends on them internally.
        t_curr = t_star
        R_curr = R_star
        
        # Build cross-attention context from projected s1/s2 features.
        s1_proj = self.proj_s1(s1)  # (B, N1, hidden_dim)
        s2_proj = self.proj_s2(s2)  # (B, N2, hidden_dim)
        p1_pad_mask = ~p1_mask.bool() if p1_mask is not None else None
        p2_pad_mask = ~p2_mask.bool() if p2_mask is not None else None

        s1_cross, _ = self.cross_attn_rt(
            query=s1_proj, key=s2_proj, value=s2_proj, key_padding_mask=p2_pad_mask
        )  # (B, N1, hidden_dim)
        s2_cross, _ = self.cross_attn_rt(
            query=s2_proj, key=s1_proj, value=s1_proj, key_padding_mask=p1_pad_mask
        )  # (B, N2, hidden_dim)

        s1_ctx = self._masked_mean(s1_cross, p1_mask)  # (B, hidden_dim)
        s2_ctx = self._masked_mean(s2_cross, p2_mask)  # (B, hidden_dim)
        p2_coords_curr = p2_coords

        for _ in range(self.num_refine_steps):
            # Predict attention weights over virtual keypoints from cross-attention context.
            y1_logits = self.y1_weight_head(s1_ctx)  # (B, K)
            y2_logits = self.y2_weight_head(s2_ctx)  # (B, K)
            y1_weights = F.softmax(y1_logits, dim=-1)
            y2_weights = F.softmax(y2_logits, dim=-1)
            y1_weighted = torch.einsum('bk,bkd->bd', y1_weights, Y1)  # (B, 3)
            y2_weighted = torch.einsum('bk,bkd->bd', y2_weights, Y2)  # (B, 3)

            rt_input = y1_weighted + y2_weighted  # (B, 3)
            delta_rt = self.mlp_rt_delta(rt_input)
            rot_6d = delta_rt[:, :6]
            delta_t = delta_rt[:, 6:]

            R_delta = self._rot6d_to_rotmat(rot_6d)
            R_curr = R_delta @ R_curr
            # Keep (R, t) composition consistent with x' = x @ R^T + t.
            t_curr = (t_curr.unsqueeze(1) @ R_delta.transpose(1, 2)).squeeze(1) + delta_t

            # Alternate update: move protein-2 coordinates and regenerate virtual interface nodes.
            p2_coords_curr = p2_coords_curr @ R_delta.transpose(1, 2) + delta_t.unsqueeze(1)
            Y1, Y2, _, _ = self._generate_keypoints(
                s1, p1_coords,
                s2, p2_coords_curr,
                p1_mask, p2_mask
            )

        return R_curr, t_curr

# class PhiR(nn.Module):
#     """
#     Predicts rotation matrix R_pred using features from proteins and ligand.
#     Formula: Δω = φ_R(s̃^(1), s̃^(2), s̃^(l), Σ_1, Σ_2, t), ΔR = exp([Δω]_×), R^pred = ΔR * R̃^t
#     """
#     def __init__(self, c_s, c_t, hidden_dim):
#         super().__init__()
#         self.c_s = c_s
#         self.hidden_dim = hidden_dim
        
#         # Project features to hidden dimension
#         self.proj_s1 = nn.Linear(c_s, hidden_dim)
#         self.proj_s2 = nn.Linear(c_s, hidden_dim)
#         self.proj_sl = nn.Linear(c_s, hidden_dim)
        
#         # Project sigma matrices (3x3) to feature space
#         # Flatten 3x3 matrix to 9-dim vector, then project to hidden_dim
#         self.proj_sigma1 = nn.Linear(9, hidden_dim)
#         self.proj_sigma2 = nn.Linear(9, hidden_dim)
        
#         # MLP input: 3*hidden_dim (s1, s2, sl) + 2*hidden_dim (sigma1, sigma2) + c_t (time)
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_dim * 5 + c_t, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 3)  # Output rotation vector in R^3
#         )

#     def forward(self, s1_tilde, s2_tilde, s_l_tilde, R_tilde, t_emb, p1_mask, p2_mask, mol_mask, sigma_1, sigma_2):
#         """
#         Args:
#             s1_tilde: (B, N1, c_s) - protein 1 features
#             s2_tilde: (B, N2, c_s) - protein 2 features
#             s_l_tilde: (B, N, c_s) - ligand features
#             R_tilde: (B, 3, 3) - current rotation
#             t_emb: (B, N, c_t) - time embedding for molecular glue
#             p1_mask: (B, N1) - mask for protein 1 residues
#             p2_mask: (B, N2) - mask for protein 2 residues
#             mol_mask: (B, N) - mask for ligand atoms
#             sigma_1: (B, 3, 3) - interface representation for protein 1
#             sigma_2: (B, 3, 3) - interface representation for protein 2
#         """
#         B = s1_tilde.shape[0]
        
#         # Aggregate protein features: masked mean pooling
#         p1_mask_expanded = p1_mask.unsqueeze(-1)  # (B, N1, 1)
#         s1_mean = (s1_tilde * p1_mask_expanded).sum(dim=1) / (p1_mask.sum(dim=1, keepdim=True) + 1e-8)  # (B, c_s)
        
#         p2_mask_expanded = p2_mask.unsqueeze(-1)  # (B, N2, 1)
#         s2_mean = (s2_tilde * p2_mask_expanded).sum(dim=1) / (p2_mask.sum(dim=1, keepdim=True) + 1e-8)  # (B, c_s)
        
#         # Aggregate ligand features: masked mean pooling
#         mol_mask_expanded = mol_mask.unsqueeze(-1)  # (B, N, 1)
#         s_l_mean = (s_l_tilde * mol_mask_expanded).sum(dim=1) / (mol_mask.sum(dim=1, keepdim=True) + 1e-8)  # (B, c_s)
        
#         # Project features to hidden dimension
#         s1_proj = self.proj_s1(s1_mean).unsqueeze(1)  # (B, 1, hidden_dim)
#         s2_proj = self.proj_s2(s2_mean).unsqueeze(1)  # (B, 1, hidden_dim)
#         sl_proj = self.proj_sl(s_l_mean).unsqueeze(1)  # (B, 1, hidden_dim)
        
#         # Flatten sigma matrices: (B, 3, 3) -> (B, 9)
#         sigma_1_flat = sigma_1.reshape(-1, 9)  # (B, 9)
#         sigma_2_flat = sigma_2.reshape(-1, 9)  # (B, 9)
        
#         # Project sigma matrices to hidden dimension
#         sigma_1_proj = self.proj_sigma1(sigma_1_flat).unsqueeze(1)  # (B, 1, hidden_dim)
#         sigma_2_proj = self.proj_sigma2(sigma_2_flat).unsqueeze(1)  # (B, 1, hidden_dim)

#         # Compute masked mean of time embedding
#         t_emb_mean = (t_emb * mol_mask_expanded).sum(1, keepdim=True) / (mol_mask.sum(1, keepdim=True).unsqueeze(-1) + 1e-8)  # (B, 1, c_t)
        
#         # Combine all features: s1, s2, sl, sigma_1, sigma_2, and time embedding
#         combined = torch.cat([s1_proj, s2_proj, sl_proj, sigma_1_proj, sigma_2_proj, t_emb_mean], dim=-1)  # (B, 1, 5*hidden_dim + c_t)
        
#         # Compute Δω = φ_R(s̃^(1), s̃^(2), s̃^(l), Σ_1, Σ_2, t)
#         delta_omega = self.mlp(combined).view(-1, 3)  # (B, 3)

#         # Compute ΔR = exp([Δω]_×)
#         R_delta = rotvec_to_rotmat(delta_omega)  # (B, 3, 3)

#         # Compute R^pred = ΔR * R̃^t
#         R_pred = R_delta @ R_tilde  # (B, 3, 3)

#         return R_pred

# class PhiT(nn.Module):
#     def __init__(self, c_s, c_t, hidden_dim):
#         super().__init__()
#         self.c_s = c_s
#         self.c_t = c_t
#         self.hidden_dim = hidden_dim
        
#         # Sequence embedder for ligand atoms
#         # from utils.constants import MAP_ATOM_TYPE_FULL_TO_INDEX
#         # self.seq_embedder = nn.Embedding(len(MAP_ATOM_TYPE_FULL_TO_INDEX), c_s)
        
#         # Time embedding projection for p1, p2, and ligand
#         self.t_emb_proj_p1 = nn.Linear(c_t, c_s)
#         self.t_emb_proj_p2 = nn.Linear(c_t, c_s)
#         self.t_emb_proj_lig = nn.Linear(c_t, c_s)
        
#         # EGNN layer for equivariant feature updates
#         self.egnn_block = EGNN_Network(
#             depth=3,
#             dim=c_s,
#             num_nearest_neighbors=16,
#             norm_coors=True,
#             coor_weights_clamp_value=2.0
#         )
        
#         # Weight projection for computing weighted mean (equivariant)
#         self.weight_proj = nn.Sequential(
#             nn.Linear(c_s, hidden_dim * 2),
#             nn.SiLU(),
#             nn.Linear(hidden_dim * 2, hidden_dim * 2),
#             nn.SiLU(),
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, 1)
#         )

#         self.distance_embedding = nn.Sequential(
#             nn.Linear(3, hidden_dim * 2),
#             nn.SiLU(),
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, c_s)
#         )
        
#         # Final translation prediction MLP
#         self.trans_pred_mlp = nn.Sequential(
#             nn.Linear(c_s * 3, hidden_dim),  # p1_mu, p2_mu, lig_mu features
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, 3)
#         )

#     def forward(self, s1_tilde, s2_tilde, p1_coords, p2_coords, X_tilde, sl, t_tilde, 
#             p1_mask, p2_mask, mol_mask, t_emb):
#         """
#         Predict translation vector t_pred by computing weighted means of p1, p2, and ligand coordinates.
        
#         Args:
#             s1_tilde: (B, N1, c_s) - protein 1 features
#             s2_tilde: (B, N2, c_s) - protein 2 features
#             p1_coords: (B, N1, 3) - protein 1 coordinates
#             p2_coords: (B, N2, 3) - protein 2 coordinates
#             X_tilde: (B, N, 3) - ligand coordinates
#             sl: (B, N, c_s) - ligand features
#             t_tilde: (B, 3) - noised translation vector
#             p1_mask: (B, N1) - protein 1 mask
#             p2_mask: (B, N2) - protein 2 mask
#             mol_mask: (B, N) - ligand mask
#             t_emb: (B, N, c_t) - time embedding for ligand atoms
        
#         Returns:
#             t_pred: (B, 3) - predicted translation vector
#             p1_mu: (B, 3) - weighted mean of p1 coordinates
#             p2_mu: (B, 3) - weighted mean of p2 coordinates
#         """
#         B = s1_tilde.shape[0]
#         N1, N2, N_lig = s1_tilde.shape[1], s2_tilde.shape[1], sl.shape[1]
        
#         # Get ligand atom features from sequence embedding
#         # lig_feats = self.seq_embedder(seq_tilde)  # (B, N_lig, c_s)
#         lig_feats = sl

#         condition_feats = self.distance_embedding(t_tilde.squeeze(1))  # (B, c_s)
        
#         # Project time embeddings
#         t_emb_lig = self.t_emb_proj_lig(t_emb)  # (B, N_lig, c_s)
#         t_emb_agg = (t_emb * mol_mask.unsqueeze(-1)).sum(1, keepdim=True) / (mol_mask.sum(1, keepdim=True).unsqueeze(-1) + 1e-8)  # (B, 1, c_t)
#         t_emb_p1 = self.t_emb_proj_p1(t_emb_agg).expand(-1, N1, -1)  # (B, N1, c_s)
#         t_emb_p2 = self.t_emb_proj_p2(t_emb_agg).expand(-1, N2, -1)  # (B, N2, c_s)
        
#         # Expand condition features to match node dimensions
#         condition_p1 = condition_feats.unsqueeze(1).expand(-1, N1, -1)  # (B, N1, c_s)
#         condition_p2 = condition_feats.unsqueeze(1).expand(-1, N2, -1)  # (B, N2, c_s)
#         condition_lig = condition_feats.unsqueeze(1).expand(-1, N_lig, -1)  # (B, N_lig, c_s)
        
#         # Combine features with time embeddings and condition features
#         s1_combined = s1_tilde + t_emb_p1 + condition_p1  # (B, N1, c_s)
#         s2_combined = s2_tilde + t_emb_p2 + condition_p2  # (B, N2, c_s)
#         lig_combined = lig_feats + t_emb_lig + condition_lig  # (B, N_lig, c_s)
        
#         # Concatenate all features and coordinates for EGNN
#         all_feats = torch.cat([s1_combined, s2_combined, lig_combined], dim=1)  # (B, N1+N2+N_lig, c_s)
#         all_coords = torch.cat([p1_coords, p2_coords, X_tilde], dim=1)  # (B, N1+N2+N_lig, 3)
#         all_mask = torch.cat([p1_mask, p2_mask, mol_mask], dim=1)  # (B, N1+N2+N_lig)
        
#         # Update features using EGNN (equivariant to rotations and translations)
#         all_feats_updated, _ = self.egnn_block(all_feats, all_coords, mask=all_mask)  # (B, N1+N2+N_lig, c_s)
        
#         # Split back into p1, p2, and ligand features
#         s1_updated = all_feats_updated[:, :N1, :]  # (B, N1, c_s)
#         s2_updated = all_feats_updated[:, N1:N1+N2, :]  # (B, N2, c_s)
#         lig_updated = all_feats_updated[:, N1+N2:, :]  # (B, N_lig, c_s)
        
#         # Compute weighted means using updated features (equivariant)
#         # p1 weighted mean
#         weights_p1 = self.weight_proj(s1_updated).squeeze(-1)  # (B, N1)
#         weights_p1 = weights_p1.masked_fill(~p1_mask, -1e4)
#         attn_p1 = F.softmax(weights_p1, dim=-1)  # (B, N1)
#         p1_mu = torch.einsum("bn,bnd->bd", attn_p1, p1_coords)  # (B, 3)
        
#         # p2 weighted mean
#         weights_p2 = self.weight_proj(s2_updated).squeeze(-1)  # (B, N2)
#         weights_p2 = weights_p2.masked_fill(~p2_mask, -1e4)
#         attn_p2 = F.softmax(weights_p2, dim=-1)  # (B, N2)
#         p2_mu = torch.einsum("bn,bnd->bd", attn_p2, p2_coords)  # (B, 3)
        
#         # ligand weighted mean
#         weights_lig = self.weight_proj(lig_updated).squeeze(-1)  # (B, N_lig)
#         weights_lig = weights_lig.masked_fill(~mol_mask, -1e4)
#         attn_lig = F.softmax(weights_lig, dim=-1)  # (B, N_lig)
#         lig_mu = torch.einsum("bn,bnd->bd", attn_lig, X_tilde)  # (B, 3)
        
#         # Get weighted feature representations for translation prediction
#         # p1_feat_mu = torch.einsum("bn,bnd->bd", attn_p1, s1_updated)  # (B, c_s)
#         # p2_feat_mu = torch.einsum("bn,bnd->bd", attn_p2, s2_updated)  # (B, c_s)
#         # lig_feat_mu = torch.einsum("bn,bnd->bd", attn_lig, lig_updated)  # (B, c_s)

#         t_pred_delta = p1_mu + p2_mu + lig_mu
        
#         # Predict translation from combined features
#         # combined_feats = torch.cat([p1_feat_mu, p2_feat_mu, lig_feat_mu], dim=-1)  # (B, 3*c_s)
#         # t_pred_delta = self.trans_pred_mlp(combined_feats)  # (B, 3)
        
#         # Add residual connection with t_tilde
#         t_pred = t_tilde.squeeze(1) + t_pred_delta  # (B, 3)
        
#         # return t_pred, p1_mu, p2_mu
#         return t_pred

class TernaryDenoiseBlock(nn.Module):
    def __init__(self, ipa_conf, num_classes=25):
        super().__init__()
        self._ipa_conf = ipa_conf 
        
        self.feat_dim = self._ipa_conf.c_s

        # Ternary Denoise Block components according to Algorithm 2
        
        # Shared IPA module for feature processing
        self.ipa = InvariantPointAttention(self._ipa_conf)  # Shared for both entities
        self.ipa_ln = nn.LayerNorm(self._ipa_conf.c_s)

        self.lig_ipa = InvariantPointAttention(self._ipa_conf)  # Shared for both entities
        self.lig_ipa_ln = nn.LayerNorm(self._ipa_conf.c_s)

        # Cross attention block to update the coordinate of the molecular glue.
        # self.phi_X = PhiX(
        #     c_s=self._ipa_conf.c_s,
        #     c_t=self._ipa_conf.c_s,
        #     hidden_dim=self._ipa_conf.c_s
        # )
        
        # Cross attention block to update the sequence of the molecular glue.
        # self.phi_A = PhiA(
        #     c_s=self._ipa_conf.c_s,
        #     c_t=self._ipa_conf.c_s,
        #     hidden_dim=self._ipa_conf.c_s,
        #     num_classes=num_classes
        # )
        
        # Rotation prediction module
        # self.phi_R = PhiR(
        #     c_s=self._ipa_conf.c_s,
        #     c_t=self._ipa_conf.c_s,
        #     hidden_dim=self._ipa_conf.c_s
        # )
        
        # Translation prediction module
        # self.phi_T = PhiT(
        #     c_s=self._ipa_conf.c_s,
        #     c_t=self._ipa_conf.c_s,
        #     hidden_dim=self._ipa_conf.c_s
        # )

        self.phix = Phix(
            c_s=self._ipa_conf.c_s,
            c_t=self._ipa_conf.c_s,
            hidden_dim=self._ipa_conf.c_s
        )

        self.phiRT = PhiRT(
            c_s=self._ipa_conf.c_s,
            c_t=self._ipa_conf.c_s,
            hidden_dim=self._ipa_conf.c_s
        )
        
        # self.seq_embedder = nn.Embedding(len(MAP_ATOM_TYPE_FULL_TO_INDEX), self._ipa_conf.c_s)
    
    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)

        return timestep_emb

    def forward(self, s1, s2, z1, z2, T1, T2, 
                s_l, z_l, Tl,
                seq_tilde, X_tilde, R_tilde, t_tilde, t, 
                p1, p2, 
                mol_mask,
                p2_coords_moved,
                sigma_1, sigma_2,
                R_star, t_star,
                Y1, Y2
            ):
        """
        Ternary Denoise Block forward pass according to Algorithm 2
        
        Args:
            s1, s2: Single features for two proteins
            z1, z2: Pair features for the two proteins  
            I1, I2: Interface representations for the two proteins
            T1, T2: Residue frames for the two proteins
            seq_tilde: Noised molecular glue sequence at timestep t
            X_tilde: Noised molecular glue coordinates at timestep t, (B, N, 3)
            R_tilde: Noised rotation matrix at timestep t, (B, 3, 3)
            t_tilde: Noised translation vector at timestep t, (B, 3)
            t: Current timestep
        """
        p1_mask = p1['res_mask']
        p2_mask = p2['res_mask']
        p1_coords = p1['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        p2_coords = p2['pos_heavyatom'][:, :, BBHeavyAtom.CA]

        # Obtain the single representation of the two proteins and the molecular glue with IPA block.
        s1_tilde = self.ipa(s=s1, z=z1, r=T1, mask=p1_mask, i_repr=None) # (B, N1, c_s)
        s1_tilde = self.ipa_ln(s1_tilde)
        
        s2_tilde = self.ipa(s=s2, z=z2, r=T2, mask=p2_mask, i_repr=None) # (B, N2, c_s)
        s2_tilde = self.ipa_ln(s2_tilde)

        s_l_tilde = self.lig_ipa(s=s_l, z=z_l, r=Tl, mask=mol_mask, i_repr=None) # (B, N_l, c_s)
        s_l_tilde = self.lig_ipa_ln(s_l_tilde)
        
        # Compute pairwise coordinate differences between molecular glue atoms
        B, N, _ = X_tilde.shape
        X_i = X_tilde.unsqueeze(2)  # [B, N, 1, 3]
        X_j = X_tilde.unsqueeze(1)  # [B, 1, N, 3]
        coord_diffs = X_i - X_j     # [B, N, N, 3]

        #########################################################
        # Molecular glue coordinate prediction.
        #########################################################
        # Create combined mask: mask out self-interactions and padding atoms
        self_mask = torch.eye(N, device=X_tilde.device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)  # [1, N, N, 1]
        mol_mask_2d = mol_mask.unsqueeze(2) & mol_mask.unsqueeze(1)  # [B, N, N] - valid pairs
        combined_mask = self_mask | ~mol_mask_2d.unsqueeze(-1)  # [B, N, N, 1]
        
        # Mask out invalid coord_diffs (apply to all 3 dimensions)
        coord_diffs = coord_diffs * (~combined_mask).float()  # [B, N, N, 3]

        # Time embedding
        t_emb = self.embed_t(t, mol_mask)  # [B, N, c_t]

        # Compute updated coordinates via PhiX (with multi-layer coordinate updates)
        # Formula: X_i^pred = X_i^t + sum_{j≠i} (X_i^t - X_j^t) * φ_X(h_i, h_j, ||X_i^t - X_j^t||², t)
        # PhiX now performs multi-layer updates internally and returns the final coordinates
        # X_pred = self.phi_X(s1_tilde, p1_coords, i1_mask, s2_tilde, p2_coords, i2_mask, s_l_tilde, X_tilde, t_emb, p1_mask, p2_mask, mol_mask)  # [B, N, 3]

        #########################################################
        # Molecular glue sequence prediction.
        #########################################################
        # Sequence prediction using PhiA: a_i^pred = φ_A(s̃^(1), s̃^(2), ã_i^t, t)
        
        # Get probability distributions from PhiA for each atom
        # phi_a_probs = self.phi_A(s1_tilde, s2_tilde, s_l_tilde, t_emb, p1_mask, p2_mask, mol_mask)  # [B, N, num_classes]
        
        X_pred, phi_a_probs = self.phix(s1_tilde, p1_coords, s2_tilde, p2_coords_moved, s_l_tilde, X_tilde, t_emb, p1_mask, p2_mask, mol_mask)
        # Apply final mask
        seq_pred = phi_a_probs * mol_mask.unsqueeze(-1)

        #########################################################
        # Translation vector prediction to move the protein 2 to the final ternary complex.
        #########################################################
        # Formula: t^pred = t~^t + Σ_{i ∈ I_2} Σ_j (X~_i - X~_j^pred) φ_t(s~_i, h~_j, ||X~_i - X~_j^pred||_2^2, t)
        # Use X_pred (predicted ligand coordinates) for X~_j^pred in the formula
        # t_pred = self.phi_T(s2_tilde, p2_coords, i2_mask, X_pred, seq_tilde, t_emb, t_tilde, p2_mask, mol_mask)
        # t_pred = self.phi_T(s2_tilde, p2_coords, i2_mask, X_tilde, seq_tilde, t_emb, t_tilde, p2_mask, mol_mask)
        # X_tilde or X_pred?
        # t_pred, p1_mu, p2_mu = self.phi_T(s1_tilde, s2_tilde, p1_coords, p2_coords_moved, X_tilde, seq_tilde, t_tilde, 
        #                     p1_mask, p2_mask, mol_mask, t_emb)
        # t_pred = self.phi_T(s1_tilde, s2_tilde, p1_coords, p2_coords_moved, X_tilde, s_l_tilde, t_tilde, 
        #                     p1_mask, p2_mask, mol_mask, t_emb)

        
        # Apply the rotation to the translation residual: t_pred = t_tilde + R_pred * Δt
        # t_pred = t_tilde + torch.matmul(R_pred, delta_t.unsqueeze(-1)).squeeze(-1)  # [B, 3]

        #########################################################
        # Rotation matrix prediction to move the protein 2 to the final ternary complex.
        #########################################################
        # R_pred = self.phi_R(s1_tilde, s2_tilde, R_tilde, t_emb,
        #                     p1_coords, p2_coords,
        #                     p1_mask, p2_mask, mol_mask, p1_mu, p2_mu)
        # R_pred = self.phi_R(s1_tilde, s2_tilde, s_l_tilde, R_tilde, t_emb, p1_mask, p2_mask, mol_mask, sigma_1, sigma_2)
        
        R_pred, t_pred = self.phiRT(Y1, Y2, s1_tilde, p1_coords, s2_tilde, p2_coords_moved, R_star, t_star, R_tilde, t_tilde, t_emb, p1_mask, p2_mask, sigma_1, sigma_2)

        return X_pred, seq_pred, R_pred, t_pred

class VFModel(nn.Module):
    def __init__(self, cfg, num_classes=25):
        super().__init__()
        
        self.node_embedder = NodeEmbedder(cfg.node_embed_size, max_num_heavyatoms)
        self.edge_embedder = EdgeEmbedder(cfg.edge_embed_size, max_num_heavyatoms)

        self.lig_node_embedder = LigandNodeEmbedder(cfg.node_embed_size, num_atom_types=len(MAP_ATOM_TYPE_FULL_TO_INDEX))
        self.lig_edge_embedder = LigandEdgeEmbedder(cfg.edge_embed_size, num_distance_bins=32, distance_max=20.0)
        
        # Initialize TernaryDenoiseBlock with IPA configuration
        self.ternary_denoise_block = TernaryDenoiseBlock(cfg.ipa, num_classes=num_classes)

    def encode(self, p1, p2, lig_coords_t, lig_seq_t, mol_mask):
        """
        Encode protein and ligand features using node and edge embedders
        
        Args:
            p1, p2: protein dictionaries
                - p1_coords, p2_coords: CA coordinates (B, N, 3)
                - p1_c_coords, p2_c_coords: C coordinates (B, N, 3) 
                - p1_n_coords, p2_n_coords: N coordinates (B, N, 3)
                - p1_residue, p2_residue: amino acid sequences (B, N)
                - p1_mask, p2_mask: residue masks (B, N)
            lig_coords_t: noised molecular glue coordinates (B, N, 3)
            lig_seq_t: noised molecular glue sequence (B, N)
            mol_mask: molecular glue mask (B, N)
        Returns:
            s1, s2: protein 1 and 2 features (B, N, c_s)
            z1, z2: protein 1 and 2 pair features (B, N, N, c_z)
            T1, T2: protein 1 and 2 rigid transformations (B, N, 3, 3)
            s_l: ligand features (B, N, c_s)
            z_l: ligand pair features (B, N, N, c_z)
        """
        # Construct rigid transformations for both proteins
        rotmats_p1 = construct_3d_basis(
            p1['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            p1['pos_heavyatom'][:, :, BBHeavyAtom.C], 
            p1['pos_heavyatom'][:, :, BBHeavyAtom.N]
        )
        rotmats_p2 = construct_3d_basis(
            p2['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            p2['pos_heavyatom'][:, :, BBHeavyAtom.C],
            p2['pos_heavyatom'][:, :, BBHeavyAtom.N]
        )
        trans_p1 = p1['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        trans_p2 = p2['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        
        # Create rigid objects for both proteins
        T1 = create_rigid(rotmats_p1, trans_p1)
        T2 = create_rigid(rotmats_p2, trans_p2)
        
        # Create rigid object for ligand
        # Since ligand only has coordinates (no basis atoms), we use identity rotation
        # and the coordinates themselves as translation
        B, N_l = lig_coords_t.shape[:2]
        
        # Create identity rotation matrices for each ligand atom: (B, N_l, 3, 3)
        rotmats_l = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, N_l, 3, 3).to(lig_coords_t)
        trans_l = lig_coords_t  # (B, N_l, 3)
        Tl = create_rigid(rotmats_l, trans_l)
        
        # Encode node features (single representations)
        s1 = self.node_embedder(
            p1['aa'],
            p1['res_nb'],
            p1['chain_nb'],
            p1['pos_heavyatom'],
            p1['mask_heavyatom'],
        )   # (B, N1, c_s)
        s2 = self.node_embedder(
            p2['aa'],
            p2['res_nb'],
            p2['chain_nb'],
            p2['pos_heavyatom'],
            p2['mask_heavyatom'],
        )  # (B, N2, c_s)
        
        # Encode edge features (pair representations)  
        z1 = self.edge_embedder(
            p1['aa'],
            p1['res_nb'],
            p1['chain_nb'],
            p1['pos_heavyatom'],
            p1['mask_heavyatom'],
        )  # (B, N1, N1, c_z)
        z2 = self.edge_embedder(
            p2['aa'],
            p2['res_nb'],
            p2['chain_nb'],
            p2['pos_heavyatom'],
            p2['mask_heavyatom'],
        )  # (B, N2, N2, c_z)

        s_l = self.lig_node_embedder(lig_seq_t, mol_mask)
        z_l = self.lig_edge_embedder(lig_coords_t, mol_mask)
        
        return s1, s2, s_l, z1, z2, z_l, T1, T2, Tl
        

    def forward(self, 
                p1, 
                p2, 
                t, 
                lig_coords_t, 
                rotmats_t, 
                trans_t, 
                lig_seq_t, 
                mol_mask, 
                # i1_mask, 
                # i2_mask,
                p2_coords_moved,
                sigma_1,
                sigma_2,
                R_star,
                t_star,
                Y1,
                Y2
        ):
        """
        Forward pass using TernaryDenoiseBlock
        
        Args:
            t: Current timestep (B, 1)
            lig_coords_t: Noised molecular glue coordinates (B, N, 3)
            rotmats_t: Noised rotation matrix (B, 1, 3, 3)
            trans_t: Noised translation vector (B, 1, 3)
            lig_seq_t: Noised molecular glue sequence (B, N)
            p1, p2: Protein dictionaries
            mol_mask: Molecular glue mask (B, N)
            i1_mask, i2_mask: Interface masks (B, N1) and (B, N2)
        
        Returns:
            Tuple of predictions: (seq_pred, coords_pred, rot_pred, trans_pred)
        """
        s1, s2, s_l, z1, z2, z_l, T1, T2, Tl = self.encode(p1, p2, lig_coords_t, lig_seq_t, mol_mask)
        
        # Use TernaryDenoiseBlock for denoising
        coords_pred, seq_pred, rot_pred, trans_pred = self.ternary_denoise_block(
            s1=s1, s2=s2, s_l=s_l, z1=z1, z2=z2, z_l=z_l, T1=T1, T2=T2, Tl=Tl,
            seq_tilde=lig_seq_t, X_tilde=lig_coords_t, R_tilde=rotmats_t, t_tilde=trans_t, t=t,
            mol_mask=mol_mask, p1=p1, p2=p2,
            p2_coords_moved=p2_coords_moved,
            sigma_1=sigma_1, sigma_2=sigma_2,
            R_star=R_star, t_star=t_star,
            Y1=Y1, Y2=Y2
        )
        
        return seq_pred, coords_pred, rot_pred, trans_pred