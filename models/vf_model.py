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

from utils.constants import AA, BBHeavyAtom, max_num_heavyatoms, MAP_ATOM_TYPE_FULL_TO_INDEX
from utils.rigid_utils import get_backbone_dihedral_angles, pairwise_dihedrals, kabsch_align
from utils.so3_utils import calc_rot_vf, vector_to_skew_matrix

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

class UnifiedTernaryEGNN(nn.Module):
    """Joint equivariant encoder for protein 1, current protein 2 and ligand.

    All three components are placed in the same graph and processed by one
    EGNN trunk.  The ligand coordinate and atom heads, and the downstream RT
    head, therefore consume the same coupled representation.
    """

    def __init__(
        self,
        c_s,
        c_t,
        hidden_dim,
        num_layers=4,
        num_nearest_neighbors=32,
        message_dim=64,
        num_classes=25,
        coor_weights_clamp_value=2.0,
        coordinate_scale=10.0,
        cross_entity_neighbors=8,
        num_bond_classes=6,
        bond_edge_dim=8,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_t = c_t
        self.hidden_dim = hidden_dim
        self.coordinate_scale = float(coordinate_scale)
        if self.coordinate_scale <= 0:
            raise ValueError("coordinate_scale must be positive")
        self.num_nearest_neighbors = int(num_nearest_neighbors)
        self.cross_entity_neighbors = int(cross_entity_neighbors)
        self.num_bond_classes = int(num_bond_classes)
        self.bond_edge_dim = int(bond_edge_dim)
        if self.num_bond_classes < 2:
            raise ValueError("num_bond_classes must include no-bond and a bond class")
        if self.bond_edge_dim <= 0:
            raise ValueError("bond_edge_dim must be positive")
        if self.cross_entity_neighbors < 0:
            raise ValueError("cross_entity_neighbors must be non-negative")
        # Each receiver has forced edges to both other entity types. Keep room
        # for its self/local geometric neighbors in the shared KNN budget.
        if 2 * self.cross_entity_neighbors >= self.num_nearest_neighbors:
            raise ValueError(
                "2 * cross_entity_neighbors must be smaller than "
                "num_nearest_neighbors, got "
                f"{self.cross_entity_neighbors} and {self.num_nearest_neighbors}"
            )

        self.protein1_emb = nn.Linear(c_s, hidden_dim)
        self.protein2_emb = nn.Linear(c_s, hidden_dim)
        self.ligand_emb = nn.Linear(c_s, hidden_dim)
        self.time_emb = nn.Linear(c_t, hidden_dim, bias=False)
        self.entity_emb = nn.Embedding(3, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.refine_net = EGNN_Network(
            depth=num_layers,
            dim=hidden_dim,
            num_edge_tokens=self.num_bond_classes,
            edge_dim=self.bond_edge_dim,
            num_nearest_neighbors=num_nearest_neighbors,
            m_dim=message_dim,
            norm_coors=True,
            coor_weights_clamp_value=coor_weights_clamp_value,
            m_pool_method="mean",
        )

        # Atom logits are read directly from the ligand nodes of the shared
        # graph.  A second protein-ligand attention stack would re-introduce the
        # decoupling that this module is intended to remove.
        self.atom_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        # Symmetric pair features make the i-j and j-i bond predictions
        # identical by construction. The current noisy bond state is included
        # directly as well as through EGNN message passing.
        self.bond_head = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim + 1 + self.bond_edge_dim),
            nn.Linear(
                2 * hidden_dim + 1 + self.bond_edge_dim,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_bond_classes),
        )

    @staticmethod
    def _add_directed_cross_edges(
        adjacency,
        coords,
        source_slice,
        target_slice,
        source_mask,
        target_mask,
        num_neighbors,
    ):
        """Force nearest cross-entity targets for every valid source node."""
        source_coords = coords[:, source_slice]
        target_coords = coords[:, target_slice]
        target_size = target_coords.shape[1]
        if num_neighbors <= 0 or target_size == 0:
            return

        distances = torch.sum(
            (source_coords.unsqueeze(2) - target_coords.unsqueeze(1)).square(),
            dim=-1,
        )
        valid_pairs = source_mask.unsqueeze(-1) & target_mask.unsqueeze(1)
        distances = distances.masked_fill(~valid_pairs, float("inf"))
        k = min(int(num_neighbors), target_size)
        neighbor_indices = distances.topk(k, dim=-1, largest=False).indices

        edge_block = adjacency[:, source_slice, target_slice]
        edge_block.scatter_(2, neighbor_indices, True)
        # topk returns arbitrary padded targets if fewer than k targets are
        # valid; remove those edges explicitly after scattering.
        edge_block &= valid_pairs

    def _build_cross_entity_adjacency(
        self,
        coords,
        p1_mask,
        p2_mask,
        mol_mask,
    ):
        """Build directed P1<->P2, P1<->ligand and P2<->ligand edges."""
        batch_size = coords.shape[0]
        n1, n2, nl = p1_mask.shape[1], p2_mask.shape[1], mol_mask.shape[1]
        total_nodes = n1 + n2 + nl
        adjacency = torch.zeros(
            batch_size,
            total_nodes,
            total_nodes,
            dtype=torch.bool,
            device=coords.device,
        )
        entity_slices = (
            slice(0, n1),
            slice(n1, n1 + n2),
            slice(n1 + n2, total_nodes),
        )
        entity_masks = (p1_mask.bool(), p2_mask.bool(), mol_mask.bool())
        for source_idx in range(3):
            for target_idx in range(3):
                if source_idx == target_idx:
                    continue
                self._add_directed_cross_edges(
                    adjacency=adjacency,
                    coords=coords,
                    source_slice=entity_slices[source_idx],
                    target_slice=entity_slices[target_idx],
                    source_mask=entity_masks[source_idx],
                    target_mask=entity_masks[target_idx],
                    num_neighbors=self.cross_entity_neighbors,
                )
        return adjacency

    def forward(
        self,
        s1,
        p1_coords,
        s2,
        p2_coords_current,
        sl,
        ligand_coords,
        t_emb,
        p1_mask,
        p2_mask,
        mol_mask,
        lig_bond_t,
    ):
        B, N = ligand_coords.shape[:2]
        N1 = s1.shape[1]
        N2 = s2.shape[1]

        if t_emb.dim() == 3:
            time_context = t_emb[:, 0]
        elif t_emb.dim() == 2:
            time_context = t_emb
        else:
            raise ValueError(f"Unexpected time embedding shape: {t_emb.shape}")
        time_context = self.time_emb(time_context).unsqueeze(1)

        entity = self.entity_emb.weight
        h_p1 = self.protein1_emb(s1) + time_context + entity[0]
        h_p2 = self.protein2_emb(s2) + time_context + entity[1]
        h_ligand = self.ligand_emb(sl) + time_context + entity[2]
        h_all = self.input_norm(torch.cat([h_p1, h_p2, h_ligand], dim=1))
        coords_all = torch.cat(
            [p1_coords, p2_coords_current, ligand_coords], dim=1
        )
        mask_all = torch.cat(
            [p1_mask.bool(), p2_mask.bool(), mol_mask.bool()], dim=1
        )
        # Protein geometry is conditioning information and must remain rigid.
        # All nodes still exchange messages and update features, but only valid
        # ligand nodes are allowed to receive EGNN coordinate updates.
        coor_update_mask = torch.cat(
            [
                torch.zeros_like(p1_mask, dtype=torch.bool),
                torch.zeros_like(p2_mask, dtype=torch.bool),
                mol_mask.bool(),
            ],
            dim=1,
        )

        # Centering and scaling are SE(3)-compatible and keep squared-distance
        # inputs well-conditioned for complexes whose pose translation is
        # hundreds of Angstroms.
        coord_weights = mask_all.to(coords_all.dtype).unsqueeze(-1)
        coord_center = (
            (coords_all * coord_weights).sum(dim=1, keepdim=True)
            / coord_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        coords_normalized = (
            coords_all - coord_center
        ) / self.coordinate_scale
        cross_entity_adjacency = self._build_cross_entity_adjacency(
            coords=coords_normalized,
            p1_mask=p1_mask,
            p2_mask=p2_mask,
            mol_mask=mol_mask,
        )

        if lig_bond_t.shape != (B, N, N):
            raise ValueError(
                "lig_bond_t must have shape (B, L, L), got "
                f"{tuple(lig_bond_t.shape)} for B={B}, L={N}"
            )
        ligand_pair_mask = mol_mask.unsqueeze(2) & mol_mask.unsqueeze(1)
        lig_bond_t = torch.where(
            ligand_pair_mask,
            lig_bond_t.long(),
            torch.zeros_like(lig_bond_t, dtype=torch.long),
        )
        if lig_bond_t.numel() and (
            lig_bond_t.min().item() < 0
            or lig_bond_t.max().item() >= self.num_bond_classes
        ):
            raise ValueError(
                f"lig_bond_t values must be in [0, {self.num_bond_classes})"
            )

        # Edge token 0 means no chemical bond (and is also used for all
        # protein/cross-entity edges). Only the ligand-ligand block carries the
        # current noisy bond state.
        total_nodes = coords_all.shape[1]
        edge_tokens = torch.zeros(
            B,
            total_nodes,
            total_nodes,
            dtype=torch.long,
            device=coords_all.device,
        )
        ligand_start = N1 + N2
        edge_tokens[
            :, ligand_start:ligand_start + N, ligand_start:ligand_start + N
        ] = lig_bond_t

        refined_feats, refined_coords = self.refine_net(
            feats=h_all,
            coors=coords_normalized,
            adj_mat=cross_entity_adjacency,
            edges=edge_tokens,
            mask=mask_all,
            return_coor_changes=False,
            coor_update_mask=coor_update_mask,
        )
        refined_coords = (
            refined_coords * self.coordinate_scale + coord_center
        )

        p1_feats = refined_feats[:, :N1]
        p2_feats = refined_feats[:, N1:N1 + N2]
        lig_feats = refined_feats[:, N1 + N2:]
        # Return the exact input protein coordinates (rather than a numerically
        # round-tripped centered/scaled copy) so downstream RT geometry cannot
        # accidentally interpret protein coordinate refinement as motion.
        p1_coords_out = p1_coords
        p2_coords_out = p2_coords_current
        lig_coords_out = refined_coords[:, N1 + N2:]

        # Preserve padded coordinates exactly; downstream geometric reductions
        # additionally receive the masks.
        lig_coords_out = torch.where(
            mol_mask.unsqueeze(-1), lig_coords_out, ligand_coords
        )
        atom_logits = self.atom_head(lig_feats) * mol_mask.unsqueeze(-1)

        lig_i = lig_feats.unsqueeze(2)
        lig_j = lig_feats.unsqueeze(1)
        pair_sum = lig_i + lig_j
        pair_abs_diff = torch.abs(lig_i - lig_j)
        pair_dist_sq = torch.sum(
            (lig_coords_out.unsqueeze(2) - lig_coords_out.unsqueeze(1)).square(),
            dim=-1,
            keepdim=True,
        ) / (self.coordinate_scale ** 2)
        current_bond_emb = self.refine_net.edge_emb(lig_bond_t).to(
            lig_feats.dtype
        )
        bond_pair_features = torch.cat(
            [pair_sum, pair_abs_diff, pair_dist_sq, current_bond_emb], dim=-1
        )
        bond_logits = self.bond_head(bond_pair_features)
        bond_logits = 0.5 * (
            bond_logits + bond_logits.transpose(1, 2)
        )
        off_diagonal = ~torch.eye(
            N, dtype=torch.bool, device=mol_mask.device
        ).unsqueeze(0)
        bond_pair_mask = ligand_pair_mask & off_diagonal
        bond_logits = bond_logits * bond_pair_mask.unsqueeze(-1)

        return (
            lig_coords_out,
            atom_logits,
            bond_logits,
            p1_feats,
            p2_feats,
            lig_feats,
            p1_coords_out,
            p2_coords_out,
        )

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

class PhiRT(nn.Module):
    """
    Ligand-conditioned vector field on SO(3) x R^3.

    The current noisy pose ``(R_tilde, t_tilde)`` is the base point of the
    field.  The network builds invariant scalar weights from all three
    components and combines them with equivariant geometric vector bases to
    predict:

      * ``rot_vf``: a body-frame angular velocity (right-trivialized tangent),
      * ``trans_vf``: a world-frame translational velocity.

    ``R_star``/``t_star`` are only geometric prior features.  They are never
    used as the state from which an update starts, so repeated denoising calls
    do not reset the trajectory to the same Kabsch solution.
    """

    def __init__(
        self,
        c_s,
        c_t,
        hidden_dim,
        num_heads: int = 4,
        translation_scale: float = 50.0,
        force_neighbors: int = 8,
        force_num_heads: int = 8,
    ):
        # num_heads is retained for configuration compatibility.
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = 32
        self.cross_dim = 32
        self.num_vector_bases = 6
        self.translation_scale = float(translation_scale)
        self.max_rotation_speed = math.pi
        self.force_neighbors = int(force_neighbors)
        self.force_num_heads = int(force_num_heads)
        self.interface_ablation = "none"
        if self.force_neighbors <= 0:
            raise ValueError("force_neighbors must be positive")
        if self.force_num_heads <= 0:
            raise ValueError("force_num_heads must be positive")

        # IPA/node outputs are invariant scalar features.  Ligand features are
        # included explicitly instead of allowing RT to see only the proteins.
        self.ternary_context_proj = nn.Sequential(
            nn.Linear(3 * c_s, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.context_dim),
        )
        self.time_context_proj = nn.Sequential(
            nn.Linear(c_t, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.context_dim),
        )

        self.p1_cross_proj = nn.Linear(c_s, self.cross_dim, bias=False)
        self.p2_cross_proj = nn.Linear(c_s, self.cross_dim, bias=False)
        self.lig_cross_proj = nn.Linear(c_s, self.cross_dim, bias=False)

        num_rbf = 16
        self.register_buffer("rbf_centers", torch.linspace(0.0, 30.0, num_rbf))
        self.rbf_width = 30.0 / (num_rbf - 1)
        self.p1_lig_distance_bias = nn.Sequential(
            nn.Linear(num_rbf, self.cross_dim),
            nn.SiLU(),
            nn.Linear(self.cross_dim, 1),
        )
        self.p2_lig_distance_bias = nn.Sequential(
            nn.Linear(num_rbf, self.cross_dim),
            nn.SiLU(),
            nn.Linear(self.cross_dim, 1),
        )

        # Learned local SE(3) field on directed P2->P1 and P2->ligand edges.
        # The MLP emits invariant scalar coefficients; multiplying them by
        # relative directions and moment arms produces equivariant force and
        # torque vectors. Entity embeddings distinguish protein and ligand
        # targets while sharing the edge encoder.
        self.force_entity_dim = 8
        self.force_entity_emb = nn.Embedding(2, self.force_entity_dim)
        force_edge_dim = (
            2 * self.cross_dim
            + num_rbf
            + self.context_dim
            + self.force_entity_dim
        )
        self.force_torque_edge_mlp = nn.Sequential(
            nn.Linear(force_edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * self.force_num_heads),
        )

        # Twelve invariant geometric statistics plus ternary/time contexts.
        num_global_invariants = 12
        self.field_weight_head = nn.Sequential(
            nn.Linear(
                num_global_invariants + 2 * self.context_dim,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * self.num_vector_bases),
        )

        # A zero field is a safe initial condition under AMP.  The final layer
        # learns first; gradients then flow into the cross-component encoders.
        nn.init.zeros_(self.field_weight_head[-1].weight)
        nn.init.zeros_(self.field_weight_head[-1].bias)
        nn.init.zeros_(self.force_torque_edge_mlp[-1].weight)
        nn.init.zeros_(self.force_torque_edge_mlp[-1].bias)

    @staticmethod
    def _remove_singleton_rigid_axis(
        rotation: torch.Tensor,
        translation: torch.Tensor,
    ):
        if rotation.ndim == 4:
            if rotation.shape[1] != 1:
                raise ValueError(
                    f"Expected a singleton rigid axis, got rotation {rotation.shape}"
                )
            rotation = rotation[:, 0]
        if translation.ndim == 3:
            if translation.shape[1] != 1:
                raise ValueError(
                    f"Expected a singleton rigid axis, got translation {translation.shape}"
                )
            translation = translation[:, 0]
        return rotation, translation

    @staticmethod
    def _masked_mean(features: torch.Tensor, mask: torch.Tensor):
        weights = mask.to(dtype=features.dtype).unsqueeze(-1)
        return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _dot(lhs: torch.Tensor, rhs: torch.Tensor):
        return torch.sum(lhs * rhs, dim=-1)

    @staticmethod
    def _signed_log1p(value: torch.Tensor):
        """Compress scalar ranges without changing their SO(3) invariance."""
        return torch.sign(value) * torch.log1p(torch.abs(value))

    @staticmethod
    def _safe_unit(vector: torch.Tensor, eps: float = 1e-8):
        norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        return vector / norm.clamp_min(eps)

    @staticmethod
    def _clip_vector_norm(vector: torch.Tensor, max_norm: float):
        """Equivariant radial clipping; unlike component clipping, this preserves rotations."""
        norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        scale = torch.clamp(max_norm / norm.clamp_min(1e-8), max=1.0)
        return vector * scale

    def _rbf(self, distances: torch.Tensor):
        return torch.exp(
            -0.5
            * ((distances.unsqueeze(-1) - self.rbf_centers) / self.rbf_width).square()
        )

    @staticmethod
    def _gather_neighbors(values: torch.Tensor, indices: torch.Tensor):
        """Gather [B, M, ...] values into [B, N, K, ...]."""
        batch_indices = torch.arange(
            values.shape[0], device=values.device
        )[:, None, None]
        return values[batch_indices, indices]

    def _force_torque_from_target(
        self,
        p2_features,
        p2_coords,
        p2_mask,
        p2_center,
        p2_radius,
        target_features,
        target_coords,
        target_mask,
        time_context,
        entity_index,
    ):
        """Aggregate multi-head force/torque from one target entity type."""
        target_size = target_coords.shape[1]
        if target_size == 0:
            zeros = p2_coords.new_zeros(
                p2_coords.shape[0], self.force_num_heads, 3
            )
            return zeros, zeros

        pair_distances = torch.cdist(p2_coords, target_coords)
        pair_mask = p2_mask.unsqueeze(-1) & target_mask.unsqueeze(1)
        pair_distances = pair_distances.masked_fill(~pair_mask, float("inf"))
        k = min(self.force_neighbors, target_size)
        neighbor_distances, neighbor_indices = pair_distances.topk(
            k, dim=-1, largest=False
        )

        target_features_k = self._gather_neighbors(
            target_features, neighbor_indices
        )
        target_coords_k = self._gather_neighbors(
            target_coords, neighbor_indices
        )
        target_mask_k = self._gather_neighbors(
            target_mask, neighbor_indices
        )
        edge_mask = p2_mask.unsqueeze(-1) & target_mask_k

        p2_features_k = p2_features.unsqueeze(2).expand(
            -1, -1, k, -1
        )
        time_k = time_context[:, None, None].expand(
            -1, p2_coords.shape[1], k, -1
        )
        entity_ids = torch.full(
            neighbor_indices.shape,
            int(entity_index),
            dtype=torch.long,
            device=p2_coords.device,
        )
        edge_features = torch.cat(
            [
                p2_features_k,
                target_features_k,
                self._rbf(neighbor_distances.clamp_max(1.0e4)),
                time_k,
                self.force_entity_emb(entity_ids),
            ],
            dim=-1,
        )
        edge_coefficients = self.force_torque_edge_mlp(edge_features)
        trans_coefficients, rot_coefficients = edge_coefficients.chunk(2, dim=-1)
        valid = edge_mask.to(dtype=edge_coefficients.dtype).unsqueeze(-1)
        trans_coefficients = trans_coefficients * valid
        rot_coefficients = rot_coefficients * valid

        relative_direction = self._safe_unit(
            target_coords_k - p2_coords.unsqueeze(2)
        )
        # Normalize the moment arm so equally rotated proteins of different
        # physical sizes produce comparable torque magnitudes.
        moment_arm = (
            p2_coords - p2_center.unsqueeze(1)
        ) / p2_radius[:, None, None]
        torque_direction = torch.linalg.cross(
            moment_arm.unsqueeze(2), relative_direction, dim=-1
        )

        force_heads = torch.einsum(
            "bikh,bikd->bhd", trans_coefficients, relative_direction
        )
        torque_heads = torch.einsum(
            "bikh,bikd->bhd", rot_coefficients, torque_direction
        )
        # Mean aggregation prevents large proteins from receiving a larger
        # field solely because they contain more residues/edges.
        edge_count = edge_mask.sum(dim=(1, 2)).to(
            dtype=force_heads.dtype
        ).clamp_min(1.0)
        return (
            force_heads / edge_count[:, None, None],
            torque_heads / edge_count[:, None, None],
        )

    @staticmethod
    def _global_pair_weights(
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        distances: torch.Tensor,
        pair_mask: torch.Tensor,
        distance_bias: nn.Module,
    ):
        """Return one normalized, masked contact distribution per complex."""
        logits = torch.einsum(
            "bid,bjd->bij", source_features, target_features
        ) / math.sqrt(source_features.shape[-1])
        logits = logits + distance_bias(distances).squeeze(-1)
        logits = logits.masked_fill(~pair_mask, -1.0e4)
        weights = torch.softmax(logits.flatten(1), dim=-1).view_as(logits)
        weights = weights * pair_mask.to(dtype=weights.dtype)
        return weights / weights.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _weighted_pair_points(
        weights: torch.Tensor,
        source_coords: torch.Tensor,
        target_coords: torch.Tensor,
    ):
        source_point = torch.einsum("bij,bid->bd", weights, source_coords)
        target_point = torch.einsum("bij,bjd->bd", weights, target_coords)
        return source_point, target_point

    # The RT head performs small, numerically sensitive geometric operations
    # (cross products and matrix exponentials). Run this head in FP32 while the
    # large encoder/IPA activations remain under AMP.
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(
        self,
        Y1,
        Y2,
        R_star,
        t_star,
        R_tilde,
        t_tilde,
        t,
        t_emb,
        s1,
        s2,
        s_l,
        p1_mask,
        p2_mask,
        mol_mask,
        p1_coords,
        p2_coords,
        lig_coords,
        sigma_1,
        sigma_2,
        p2_coords_current=None,
    ):
        """
        Args:
            Y1, Y2: (B, K, 3) interface keypoints for protein 1 and 2
            R_star: (B, 3, 3) initial rotation from Kabsch on (Y1, Y2)
            t_star: (B, 3) initial translation from Kabsch
            R_tilde: (B, 3, 3) or (B, 1, 3, 3), current noisy rotation state
            t_tilde: (B, 3) or (B, 1, 3), current noisy translation state
            t_emb: (B, N_lig, c_t), per-token timestep embedding
            s_l: (B, N_lig, c_s), current ligand atom representation
            lig_coords: (B, N_lig, 3), current ligand coordinates
        
        Returns:
            R_endpoint: endpoint estimate derived from the current state/field
            t_endpoint: endpoint translation estimate
            rot_vf: body-frame angular velocity at the current state
            trans_vf: world-frame translation velocity at the current state
        """
        B = R_star.shape[0]
        R_tilde, t_tilde = self._remove_singleton_rigid_axis(
            R_tilde, t_tilde
        )

        p1_mask = p1_mask.bool()
        p2_mask = p2_mask.bool()
        mol_mask = mol_mask.bool()
        pair_mask_p1_lig = p1_mask.unsqueeze(2) & mol_mask.unsqueeze(1)
        pair_mask_p2_lig = p2_mask.unsqueeze(2) & mol_mask.unsqueeze(1)

        # Interface keypoints always follow the exact rigid flow state.  The
        # shared EGNN may additionally provide an equivariantly refined current
        # P2 representation for learned contact geometry.
        Y2_current = Y2 @ R_tilde.transpose(1, 2) + t_tilde.unsqueeze(1)
        if p2_coords_current is None:
            p2_current = (
                p2_coords @ R_tilde.transpose(1, 2) + t_tilde.unsqueeze(1)
            )
        else:
            p2_current = p2_coords_current

        mu1 = Y1.mean(dim=1)
        mu2_current = Y2_current.mean(dim=1)
        p1_center = self._masked_mean(p1_coords, p1_mask)
        p2_center = self._masked_mean(p2_coords, p2_mask)
        p2_center_current = self._masked_mean(p2_current, p2_mask)
        lig_center = self._masked_mean(lig_coords, mol_mask)

        ternary_context = self.ternary_context_proj(
            torch.cat(
                [
                    self._masked_mean(s1, p1_mask),
                    self._masked_mean(s2, p2_mask),
                    self._masked_mean(s_l, mol_mask),
                ],
                dim=-1,
            )
        )
        time_context = self.time_context_proj(
            self._masked_mean(t_emb, mol_mask)
        )

        # Learned, chemistry-aware contact distributions for P1-ligand and
        # current-P2-ligand.  They supply the RT field with the ternary geometry
        # that the old protein-only head could not observe.
        p1_lig_offsets = lig_coords.unsqueeze(1) - p1_coords.unsqueeze(2)
        p2_lig_offsets = lig_coords.unsqueeze(1) - p2_current.unsqueeze(2)
        p1_lig_dist = torch.linalg.vector_norm(p1_lig_offsets, dim=-1)
        p2_lig_dist = torch.linalg.vector_norm(p2_lig_offsets, dim=-1)

        lig_cross = self.lig_cross_proj(s_l)
        p1_lig_weights = self._global_pair_weights(
            self.p1_cross_proj(s1),
            lig_cross,
            self._rbf(p1_lig_dist),
            pair_mask_p1_lig,
            self.p1_lig_distance_bias,
        )
        p2_lig_weights = self._global_pair_weights(
            self.p2_cross_proj(s2),
            lig_cross,
            self._rbf(p2_lig_dist),
            pair_mask_p2_lig,
            self.p2_lig_distance_bias,
        )

        p1_contact, lig_contact_from_p1 = self._weighted_pair_points(
            p1_lig_weights, p1_coords, lig_coords
        )
        p2_contact, lig_contact_from_p2 = self._weighted_pair_points(
            p2_lig_weights, p2_current, lig_coords
        )

        key_displacement = Y1 - Y2_current
        mean_key_displacement = key_displacement.mean(dim=1)
        r2_key = Y2_current - mu2_current.unsqueeze(1)
        key_torque = torch.linalg.cross(
            r2_key, key_displacement, dim=-1
        ).mean(dim=1)

        p2_lig_torque = torch.einsum(
            "bij,bijd->bd",
            p2_lig_weights,
            torch.linalg.cross(
                p2_current.unsqueeze(2) - p2_center_current[:, None, None],
                p2_lig_offsets,
                dim=-1,
            ),
        )

        # Kabsch contributes a direction/prior, never an absolute reset.
        p2_center_kabsch = (
            torch.bmm(
                p2_center.unsqueeze(1), R_star.transpose(1, 2)
            ).squeeze(1)
            + t_star
        )
        kabsch_center_displacement = p2_center_kabsch - p2_center_current
        kabsch_rot_body = calc_rot_vf(R_tilde, R_star)
        kabsch_rot_world = torch.bmm(
            R_tilde, kabsch_rot_body.unsqueeze(-1)
        ).squeeze(-1)

        interface_from_p1 = mu1 - p1_center
        interface_from_p2 = mu2_current - p2_center_current
        ligand_to_p2 = lig_center - p2_center_current
        ligand_contact_displacement = lig_contact_from_p2 - p2_contact
        neosurface_displacement = lig_contact_from_p1 - p2_contact

        sigma_2_current = R_tilde @ sigma_2 @ R_tilde.transpose(1, 2)
        covariance_error_sq = torch.sum(
            (sigma_1 - sigma_2_current).square(), dim=(-1, -2)
        )
        expected_p1_lig_dist = torch.sum(
            p1_lig_weights * p1_lig_dist, dim=(1, 2)
        )
        expected_p2_lig_dist = torch.sum(
            p2_lig_weights * p2_lig_dist, dim=(1, 2)
        )

        ablation = self.interface_ablation
        valid_ablations = {
            "none",
            "no_pose_prior",
            "no_virtual_interface",
            "no_interface",
        }
        if ablation not in valid_ablations:
            raise ValueError(
                f"Unknown interface ablation {ablation!r}; expected one of "
                f"{sorted(valid_ablations)}"
            )

        # Counterfactual inference ablations. Only interface-derived quantities
        # are removed; ternary representations, ligand contacts, and the local
        # force--torque field remain identical to the full model.
        if ablation in {"no_virtual_interface", "no_interface"}:
            mean_key_displacement = torch.zeros_like(mean_key_displacement)
            key_torque = torch.zeros_like(key_torque)
            interface_from_p1 = torch.zeros_like(interface_from_p1)
            interface_from_p2 = torch.zeros_like(interface_from_p2)
            covariance_error_sq = torch.zeros_like(covariance_error_sq)
        if ablation in {"no_pose_prior", "no_interface"}:
            kabsch_center_displacement = torch.zeros_like(
                kabsch_center_displacement
            )
            kabsch_rot_body = torch.zeros_like(kabsch_rot_body)
            kabsch_rot_world = torch.zeros_like(kabsch_rot_world)

        global_invariants = torch.stack(
            [
                self._dot(mean_key_displacement, mean_key_displacement),
                self._dot(ligand_contact_displacement, ligand_contact_displacement),
                self._dot(neosurface_displacement, neosurface_displacement),
                self._dot(ligand_to_p2, ligand_to_p2),
                self._dot(kabsch_center_displacement, kabsch_center_displacement),
                self._dot(kabsch_rot_body, kabsch_rot_body),
                expected_p1_lig_dist,
                expected_p2_lig_dist,
                covariance_error_sq,
                self._dot(mean_key_displacement, ligand_contact_displacement),
                self._dot(ligand_contact_displacement, kabsch_center_displacement),
                self._dot(interface_from_p1, interface_from_p2),
            ],
            dim=-1,
        )
        global_invariants = self._signed_log1p(global_invariants)
        field_weights = self.field_weight_head(
            torch.cat(
                [global_invariants, ternary_context, time_context], dim=-1
            )
        )

        translation_bases = torch.stack(
            [
                mean_key_displacement,
                ligand_contact_displacement,
                neosurface_displacement,
                ligand_to_p2,
                kabsch_center_displacement,
                p1_contact - p2_contact,
            ],
            dim=1,
        )
        rotation_bases_world = torch.stack(
            [
                key_torque,
                p2_lig_torque,
                torch.linalg.cross(
                    p2_contact - p2_center_current,
                    ligand_contact_displacement,
                    dim=-1,
                ),
                kabsch_rot_world,
                torch.linalg.cross(
                    interface_from_p2, interface_from_p1, dim=-1
                ),
                torch.linalg.cross(
                    p2_contact - p2_center_current,
                    neosurface_displacement,
                    dim=-1,
                ),
            ],
            dim=1,
        )

        trans_vf = torch.sum(
            field_weights[:, : self.num_vector_bases].unsqueeze(-1)
            * self._safe_unit(translation_bases),
            dim=1,
        )
        rot_vf_world = torch.sum(
            field_weights[:, self.num_vector_bases :].unsqueeze(-1)
            * self._safe_unit(rotation_bases_world),
            dim=1,
        )

        # Local learned residual field. Unlike the six global bases above,
        # this retains residue/atom-level P2->P1 and P2->ligand interactions.
        p2_force_features = self.p2_cross_proj(s2)
        p2_radius = torch.sqrt(
            self._masked_mean(
                torch.sum(
                    (p2_current - p2_center_current.unsqueeze(1)).square(),
                    dim=-1,
                    keepdim=True,
                ),
                p2_mask,
            )[:, 0]
        ).clamp_min(1.0)
        p1_force_heads, p1_torque_heads = self._force_torque_from_target(
            p2_features=p2_force_features,
            p2_coords=p2_current,
            p2_mask=p2_mask,
            p2_center=p2_center_current,
            p2_radius=p2_radius,
            target_features=self.p1_cross_proj(s1),
            target_coords=p1_coords,
            target_mask=p1_mask,
            time_context=time_context,
            entity_index=0,
        )
        lig_force_heads, lig_torque_heads = self._force_torque_from_target(
            p2_features=p2_force_features,
            p2_coords=p2_current,
            p2_mask=p2_mask,
            p2_center=p2_center_current,
            p2_radius=p2_radius,
            target_features=lig_cross,
            target_coords=lig_coords,
            target_mask=mol_mask,
            time_context=time_context,
            entity_index=1,
        )
        head_scale = math.sqrt(float(self.force_num_heads))
        learned_force_world = (
            p1_force_heads + lig_force_heads
        ).sum(dim=1) / head_scale
        learned_torque_world = (
            p1_torque_heads + lig_torque_heads
        ).sum(dim=1) / head_scale
        trans_vf = trans_vf + learned_force_world
        rot_vf_world = rot_vf_world + learned_torque_world
        # calc_rot_vf/geodesic_t use a right-trivialized (body-frame) tangent.
        rot_vf = torch.bmm(
            R_tilde.transpose(1, 2), rot_vf_world.unsqueeze(-1)
        ).squeeze(-1)
        rot_vf = self._clip_vector_norm(rot_vf, self.max_rotation_speed)

        # Keep an endpoint estimate for the ligand coordinate head and existing
        # output/visualization code.  Flow integration itself uses rot_vf/trans_vf.
        remaining = (1.0 - t[:, 0]).clamp(min=0.0, max=1.0)
        delta_R_to_endpoint = torch.matrix_exp(
            vector_to_skew_matrix(remaining.unsqueeze(-1) * rot_vf)
        )
        R_endpoint = R_tilde @ delta_R_to_endpoint
        t_endpoint = t_tilde + (
            remaining.unsqueeze(-1) * self.translation_scale * trans_vf
        )

        return R_endpoint, t_endpoint, rot_vf, trans_vf

class BackboneUpdateLocal(nn.Module):
    """
    Lightweight backbone update head: maps single representation to a 6D rigid update
    (3D quaternion update vector + 3D translation update), to be applied via
    Rigid.compose_q_update_vec.
    """

    def __init__(self, c_s: int):
        super().__init__()
        self.linear = Linear(c_s, 6, init="final")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: (B, N, C_s) -> update: (B, N, 6)
        return self.linear(s)


class EdgeTransition(nn.Module):
    """
    Updates edge (pair) representation from node embeddings. Same interface as
    PepFlowww.models_con.ipa_pytorch.EdgeTransition; used in StackedIPABlocks.
    """

    def __init__(
        self,
        *,
        node_embed_size: int,
        edge_embed_in: int,
        edge_embed_out: int,
        num_layers: int = 2,
        node_dilation: int = 2,
    ):
        super().__init__()
        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed: torch.Tensor, edge_embed: torch.Tensor) -> torch.Tensor:
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat([
            node_embed.unsqueeze(2).expand(-1, -1, num_res, -1),
            node_embed.unsqueeze(1).expand(-1, num_res, -1, -1),
        ], dim=-1)
        edge_embed = torch.cat([edge_embed, edge_bias], dim=-1).reshape(
            batch_size * num_res * num_res, -1
        )
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(batch_size, num_res, num_res, -1)
        return edge_embed


class StackedIPABlocks(nn.Module):
    """
    GA-style stacked IPA blocks operating on single representations + rigids.

    Per block:
        node_embed -> IPA -> LN(residual)
                    -> Transformer over nodes -> Linear back to c_s
                    -> node transition MLP
                    -> backbone rigid update (compose_q_update_vec)
    """

    def __init__(self, ipa_conf, num_blocks: int):
        super().__init__()
        self._ipa_conf = ipa_conf
        self.num_blocks = num_blocks

        self.trunk = nn.ModuleDict()

        # Defaults if not present in ipa_conf
        n_heads = getattr(self._ipa_conf, "seq_tfmr_num_heads", 4)
        n_layers = getattr(self._ipa_conf, "seq_tfmr_num_layers", 1)

        for b in range(self.num_blocks):
            # IPA + layer norm
            self.trunk[f"ipa_{b}"] = InvariantPointAttention(self._ipa_conf)
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(self._ipa_conf.c_s)

            # Transformer over node features
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=n_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False,
            )
            self.trunk[f"seq_tfmr_{b}"] = torch.nn.TransformerEncoder(
                tfmr_layer,
                n_layers,
                enable_nested_tensor=False,
            )
            self.trunk[f"post_tfmr_{b}"] = Linear(
                tfmr_in, self._ipa_conf.c_s, init="final"
            )

            # Node transition
            self.trunk[f"node_transition_{b}"] = nn.Sequential(
                nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
                nn.ReLU(),
                nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
            )

            # Backbone update
            self.trunk[f"bb_update_{b}"] = BackboneUpdateLocal(self._ipa_conf.c_s)

            # Edge transition (no edge update on the last block, same as ga.py)
            if b < self.num_blocks - 1:
                self.trunk[f"edge_transition_{b}"] = EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=self._ipa_conf.c_z,
                    edge_embed_out=self._ipa_conf.c_z,
                )

    def forward(
        self,
        node_embed: torch.Tensor,
        edge_embed: torch.Tensor,
        rigids: Rigid,
        node_mask: torch.Tensor,
    ):
        """
        Args:
            node_embed: (B, N, c_s)
            edge_embed: (B, N, N, c_z) or None (only passed to IPA)
            rigids:     Rigid object with batch & residue dims matching node_embed
            node_mask:  (B, N) 0/1 mask
        Returns:
            node_embed: updated single representations
            edge_embed: updated pair representations (same as input if num_blocks==1)
            rigids:     updated rigids after all blocks
        """
        node_mask = node_mask.float()
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        curr_rigids = rigids

        for b in range(self.num_blocks):
            ipa_out = self.trunk[f"ipa_{b}"](
                s=node_embed,
                z=edge_embed,
                r=curr_rigids,
                mask=node_mask,
                i_repr=None,
            )
            ipa_out = ipa_out * node_mask.unsqueeze(-1)
            node_embed = self.trunk[f"ipa_ln_{b}"](node_embed + ipa_out)

            # Transformer over nodes
            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](
                node_embed, src_key_padding_mask=(1 - node_mask).bool()
            )
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)

            # Node transition
            node_embed = self.trunk[f"node_transition_{b}"](node_embed)
            node_embed = node_embed * node_mask.unsqueeze(-1)

            # Backbone update via compose_q_update_vec
            rigid_update = self.trunk[f"bb_update_{b}"](
                node_embed * node_mask.unsqueeze(-1)
            )
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask.unsqueeze(-1)
            )

            # Edge transition (no edge update on the last block, same as ga.py)
            if b < self.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed = edge_embed * edge_mask.unsqueeze(-1)

        return node_embed, edge_embed, curr_rigids


class TernaryDenoiseBlock(nn.Module):
    def __init__(
        self,
        ipa_conf,
        num_classes=25,
        num_bond_classes=6,
        unified_egnn_conf=None,
        translation_scale=50.0,
    ):
        super().__init__()
        self._ipa_conf = ipa_conf 
        self.feat_dim = self._ipa_conf.c_s

        unified_egnn_conf = unified_egnn_conf or {}
        self.unified_egnn = UnifiedTernaryEGNN(
            c_s=self._ipa_conf.c_s,
            c_t=self._ipa_conf.c_s,
            hidden_dim=int(getattr(
                unified_egnn_conf, "hidden_dim", self._ipa_conf.c_s
            )),
            num_layers=int(getattr(unified_egnn_conf, "num_layers", 4)),
            num_nearest_neighbors=int(getattr(
                unified_egnn_conf, "num_nearest_neighbors", 32
            )),
            message_dim=int(getattr(
                unified_egnn_conf, "message_dim", 64
            )),
            coor_weights_clamp_value=float(getattr(
                unified_egnn_conf, "coor_weights_clamp_value", 2.0
            )),
            coordinate_scale=float(getattr(
                unified_egnn_conf, "coordinate_scale", 10.0
            )),
            cross_entity_neighbors=int(getattr(
                unified_egnn_conf, "cross_entity_neighbors", 8
            )),
            num_bond_classes=num_bond_classes,
            bond_edge_dim=int(getattr(
                unified_egnn_conf, "bond_edge_dim", 8
            )),
            num_classes=num_classes,
        )

        self.phiRT = PhiRT(
            c_s=int(getattr(
                unified_egnn_conf, "hidden_dim", self._ipa_conf.c_s
            )),
            c_t=self._ipa_conf.c_s,
            hidden_dim=self._ipa_conf.c_s,
            translation_scale=translation_scale,
            force_neighbors=int(getattr(
                unified_egnn_conf, "force_neighbors", 8
            )),
            force_num_heads=int(getattr(
                unified_egnn_conf, "force_num_heads", 8
            )),
        )
        
        # self.seq_embedder = nn.Embedding(len(MAP_ATOM_TYPE_FULL_TO_INDEX), self._ipa_conf.c_s)
    
    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)

        return timestep_emb

    def forward(self, s1, s2, s_l,
                seq_tilde, X_tilde, R_tilde, t_tilde, t,
                p1, p2, 
                mol_mask,
                bond_tilde,
                p2_coords_input,
                sigma_1, sigma_2,
                R_star, t_star,
                Y1, Y2
            ):
        """Coupled ligand/pose denoising from one ternary EGNN state."""
        p1_mask = p1['res_mask']
        p2_mask = p2['res_mask']
        p1_coords = p1['pos_heavyatom'][:, :, BBHeavyAtom.CA]

        # All components must be expressed in one current coordinate frame
        # before joint EGNN message passing.
        R_current, t_current = PhiRT._remove_singleton_rigid_axis(
            R_tilde, t_tilde
        )
        p2_coords_current = (
            p2_coords_input @ R_current.transpose(1, 2)
            + t_current.unsqueeze(1)
        )

        t_emb = self.embed_t(t, mol_mask)  # [B, N, c_t]
        (
            X_pred,
            seq_pred,
            bond_pred,
            s1_joint,
            s2_joint,
            s_l_joint,
            p1_coords_joint,
            p2_coords_joint,
        ) = self.unified_egnn(
            s1=s1,
            p1_coords=p1_coords,
            s2=s2,
            p2_coords_current=p2_coords_current,
            sl=s_l,
            ligand_coords=X_tilde,
            t_emb=t_emb,
            p1_mask=p1_mask,
            p2_mask=p2_mask,
            mol_mask=mol_mask,
            lig_bond_t=bond_tilde,
        )

        # RT is predicted only after the shared EGNN update.  It sees the same
        # coupled features/coordinates as the atom and ligand-coordinate heads,
        # together with the pretrained interface geometry.
        R_pred, t_pred, rot_vf, trans_vf = self.phiRT(
            Y1=Y1,
            Y2=Y2,
            R_star=R_star,
            t_star=t_star,
            R_tilde=R_tilde,
            t_tilde=t_tilde,
            t=t,
            t_emb=t_emb,
            s1=s1_joint,
            s2=s2_joint,
            s_l=s_l_joint,
            p1_mask=p1_mask,
            p2_mask=p2_mask,
            mol_mask=mol_mask,
            p1_coords=p1_coords_joint,
            p2_coords=p2_coords_input,
            p2_coords_current=p2_coords_joint,
            lig_coords=X_pred,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
        )

        return X_pred, seq_pred, bond_pred, R_pred, t_pred, rot_vf, trans_vf

class VFModel(nn.Module):
    def __init__(self, cfg, num_classes=25, num_bond_classes=6,
                 translation_scale=50.0):
        super().__init__()
        
        self.node_embedder = NodeEmbedder(cfg.node_embed_size, max_num_heavyatoms)
        self.lig_node_embedder = LigandNodeEmbedder(cfg.node_embed_size, num_atom_types=len(MAP_ATOM_TYPE_FULL_TO_INDEX))
        
        self.ternary_denoise_block = TernaryDenoiseBlock(
            cfg.ipa,
            num_classes=num_classes,
            num_bond_classes=num_bond_classes,
            unified_egnn_conf=getattr(cfg, "unified_egnn", None),
            translation_scale=translation_scale,
        )

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
            Initial protein 1, protein 2 and ligand node features.  Pair
            encoders are intentionally omitted: cross-component geometry is
            represented by the shared EGNN rather than three independent IPA
            streams.
        """
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
        
        s_l = self.lig_node_embedder(lig_seq_t, mol_mask)

        return s1, s2, s_l
        

    def forward(self, p1, p2, t, lig_coords_t, rotmats_t, trans_t,
                lig_seq_t, lig_bond_t, mol_mask, p2_coords_input, sigma_1,
                sigma_2, R_star, t_star, Y1, Y2):
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
        s1, s2, s_l = self.encode(
            p1, p2, lig_coords_t, lig_seq_t, mol_mask
        )
        
        # Use TernaryDenoiseBlock for denoising
        (
            coords_pred,
            seq_pred,
            bond_pred,
            rot_pred,
            trans_pred,
            rot_vf,
            trans_vf,
        ) = self.ternary_denoise_block(
            s1=s1, s2=s2, s_l=s_l,
            seq_tilde=lig_seq_t, bond_tilde=lig_bond_t,
            X_tilde=lig_coords_t, R_tilde=rotmats_t, t_tilde=trans_t, t=t,
            mol_mask=mol_mask, p1=p1, p2=p2,
            p2_coords_input=p2_coords_input,
            sigma_1=sigma_1, sigma_2=sigma_2,
            R_star=R_star, t_star=t_star,
            Y1=Y1, Y2=Y2
        )
        
        return (
            seq_pred,
            bond_pred,
            coords_pred,
            rot_pred,
            trans_pred,
            rot_vf,
            trans_vf,
        )
