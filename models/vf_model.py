import math
import numpy as np
from scipy.stats import truncnorm
import torch
import torch.nn as nn
from typing import Optional, Callable, List, Sequence

from utils.rigid_utils import construct_3d_basis, global_to_local
from openfold.utils import rigid_utils as ru, Rigid

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

class NodeEmbedder(nn.Module):

    def __init__(self, feat_dim, max_aa_types=21):
        super().__init__()
        # self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.feat_dim = feat_dim
        self.aatype_embed = nn.Embedding(self.max_aa_types, feat_dim)
        
        # Only use amino acid features, no coordinate features
        infeat_dim = feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, aa, p_mask):
        """
        Args:
            aa:         (N, L) - amino acid types.
            p_coords:   (N, L, 3) - c alpha coordinates (not used).
            p_n_coords: (N, L, 3) - n coordinates (not used).
            p_c_coords: (N, L, 3) - c coordinates (not used).
            p_mask:     (N, L) - residue masks.
        """

        # Amino acid identity features
        aa_feat = self.aatype_embed(aa) # (N, L, feat)

        # Only use amino acid features (no coordinate features)
        out_feat = self.mlp(aa_feat) # (N, L, F)
        out_feat = out_feat * p_mask[:, :, None]

        return out_feat

class EdgeEmbedder(nn.Module):

    def __init__(self, feat_dim, max_aa_types=21):
        super().__init__()
        self.max_aa_types = max_aa_types
        self.feat_dim = feat_dim
        self.aa_pair_embed = nn.Embedding(self.max_aa_types * self.max_aa_types, feat_dim)

        # Simplified distance features for CA, C, N atoms only (3x3=9 distances)
        self.aapair_to_distcoef = nn.Embedding(self.max_aa_types * self.max_aa_types, 9)
        nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.distance_embed = nn.Sequential(
            nn.Linear(9, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
        )

        # Only use amino acid pairs and simplified distances (no relative positions)
        infeat_dim = feat_dim + feat_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, aa, p_coords, p_n_coords, p_c_coords, p_mask):
        """
        Args:
            aa: (N, L) - amino acid types
            p_coords: (N, L, 3) - CA coordinates
            p_n_coords: (N, L, 3) - N coordinates  
            p_c_coords: (N, L, 3) - C coordinates
            p_mask: (N, L) - residue masks

        Returns:
            (N, L, L, feat_dim)
        """
        N, L = aa.size()
        
        # Stack coordinates: CA, C, N (assuming CA=0, C=1, N=2)
        pos_atoms = torch.stack([p_coords, p_c_coords, p_n_coords], dim=2)  # (N, L, 3, 3)
        mask_atoms = p_mask[:, :, None].expand(N, L, 3)  # (N, L, 3)
        
        mask_pair = p_mask[:, :, None] * p_mask[:, None, :]  # (N, L, L)

        # Pair identities
        aa_pair = aa[:,:,None] * self.max_aa_types + aa[:,None,:]    # (N, L, L)
        feat_aapair = self.aa_pair_embed(aa_pair)

        # Simplified distances between CA, C, N atoms (3x3=9 distances)
        d = torch.linalg.norm(
            pos_atoms[:,:,None,:,None] - pos_atoms[:,None,:,None,:],
            dim = -1, ord = 2,
        ).reshape(N, L, L, 9) # (N, L, L, 9)
        
        c = torch.nn.functional.softplus(self.aapair_to_distcoef(aa_pair))    # (N, L, L, 9)
        d_gauss = torch.exp(-1 * c * d ** 2)
        mask_atom_pair = (mask_atoms[:,:,None,:,None] * mask_atoms[:,None,:,None,:]).reshape(N, L, L, 9)
        feat_dist = self.distance_embed(d_gauss * mask_atom_pair)

        # Combine features (only amino acid pairs and distances)
        feat_all = torch.cat([feat_aapair, feat_dist], dim=-1)
        feat_all = self.out_mlp(feat_all)   # (N, L, L, F)
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
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
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

class PhiX(nn.Module):
    """
    It cross-attends protein features to ligand atoms and outputs pairwise weights φ_X(i,j).
    """
    def __init__(self, c_s, c_t, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_s1 = nn.Linear(c_s, hidden_dim)
        self.proj_s2 = nn.Linear(c_s, hidden_dim)
        
        # Extract invariant features from coord_diffs to generate queries
        # Input: [distance, |dx|, |dy|, |dz|] = 4 invariant features
        self.coord_to_query = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),  # Process distance + 3D displacement magnitudes
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        self.cross_attn_1 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.cross_attn_2 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # MLP now takes coord_diffs features instead of just distances_sq
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3 + c_t, hidden_dim),  # 3 for coord_diffs invariants
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, s1, s2, coord_diffs, t_emb, p1_mask, p2_mask, mol_mask):
        """
        Args:
            s1: (B, N1, c_s) - protein 1 features
            s2: (B, N2, c_s) - protein 2 features
            coord_diffs: (B, N, N, 3) - pairwise coordinate differences
            t_emb: (B, N, c_t) - time embedding
            p1_mask, p2_mask, mol_mask: boolean masks
        Returns:
            φ_X weights: (B, N, N, 1)
        """
        B, N, _, _ = coord_diffs.shape
        c_t = t_emb.shape[-1]

        # Linear projections for protein features
        s1_proj = self.proj_s1(s1)  # [B, N1, H]
        s2_proj = self.proj_s2(s2)  # [B, N2, H]

        # Extract invariant features from coord_diffs for query generation
        distances_sq = torch.sum(coord_diffs ** 2, dim=-1, keepdim=True)  # [B, N, N, 1]
        # Also use absolute values of each coordinate difference (rotation variant but informative)
        coord_abs = torch.abs(coord_diffs)  # [B, N, N, 3]
        # Combine into invariant features: [distance, |dx|, |dy|, |dz|]
        coord_invariants = torch.cat([distances_sq, coord_abs], dim=-1)  # [B, N, N, 4]
        
        # Aggregate over pairwise features for each atom
        coord_features = torch.mean(coord_invariants, dim=2)  # [B, N, 4]
        # Use all 4 invariant features: [distance, |dx|, |dy|, |dz|]
        ligand_queries = self.coord_to_query(coord_features)  # [B, N, H]

        # Cross-attend ligand to protein 1 and 2
        attn_out_1, _ = self.cross_attn_1(ligand_queries, s1_proj, s1_proj,
                                          key_padding_mask=~p1_mask)
        attn_out_2, _ = self.cross_attn_2(ligand_queries, s2_proj, s2_proj,
                                          key_padding_mask=~p2_mask)

        # Combine attended features and time embedding
        combined = torch.cat([attn_out_1, attn_out_2, t_emb], dim=-1)  # [B, N, 2H + c_t]

        # Pairwise expansion and fusion with coord_diffs
        combined_i = combined.unsqueeze(2).expand(-1, N, N, -1)  # [B, N, N, 2H + c_t]
        # Use coord_diffs directly (model will learn invariant combinations)
        phi_input = torch.cat([combined_i, coord_diffs], dim=-1)  # [B, N, N, 2H + c_t + 3]

        phi_weights = self.mlp(phi_input)  # [B, N, N, 1]

        return phi_weights

class TernaryDenoiseBlock(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf 
        
        self.feat_dim = self._ipa_conf.c_s

        # Ternary Denoise Block components according to Algorithm 2
        
        # Shared IIPA module for feature processing
        self.iipa = InvariantPointAttention(self._ipa_conf)  # Shared for both entities
        self.iipa_ln = nn.LayerNorm(self._ipa_conf.c_s)

        # Cross attention block to update the coordinate of the molecular glue.
        self.phi_X = PhiX(
            c_s=self._ipa_conf.c_s,
            c_t=self._ipa_conf.c_s,
            hidden_dim=self._ipa_conf.c_s
        )
        
        # TODO: Specific the heavyatom types.
        self.seq_embedder = nn.Embedding(20, self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + 2 * self._ipa_conf.c_s + self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )
        
        # Angular residual function φ_R
        self.phi_R = nn.Sequential(
            nn.Linear(2 * self._ipa_conf.c_s + 3 + 9 + 9 + 1, self._ipa_conf.c_s),  # s̃^(1), s̃^(2), X̃, Σ₁, Σ₂, timestep
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 3)  # Output: angular residual Δω in R³
        )
        
        # Translation residual function φ_t
        self.phi_t = nn.Sequential(
            nn.Linear(2 * self._ipa_conf.c_s + 3 + 1 + 1, self._ipa_conf.c_s),  # s̃^(1), s̃^(2), X̃, ||μ₁-μ₂||², timestep
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 3)  # Output: translation residual Δt
        )
    
    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)

        return timestep_emb

    def forward(self, s1, s2, z1, z2, I1, I2, T1, T2, seq_tilde, X_tilde, R_tilde, t_tilde, t, p1_mask, p2_mask, mol_mask):
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
        
        # Obtain the single representation of the two proteins with IIPA block.
        # TODO: add the interface representation.
        s1_tilde = self.iipa(s1, z1, T1, p1_mask) # (B, N1, c_s)
        s1_tilde = self.iipa_ln(s1_tilde)
        
        s2_tilde = self.iipa(s2, z2, T2, p2_mask) # (B, N2, c_s)
        s2_tilde = self.iipa_ln(s2_tilde)
        
        # Compute pairwise coordinate differences between molecular glue atoms
        B, N, _ = X_tilde.shape
        X_i = X_tilde.unsqueeze(2)  # [B, N, 1, 3]
        X_j = X_tilde.unsqueeze(1)  # [B, 1, N, 3]
        coord_diffs = X_i - X_j     # [B, N, N, 3]

        # Create combined mask: mask out self-interactions and padding atoms
        self_mask = torch.eye(N, device=X_tilde.device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)  # [1, N, N, 1]
        mol_mask_2d = mol_mask.unsqueeze(2) & mol_mask.unsqueeze(1)  # [B, N, N] - valid pairs
        combined_mask = self_mask | ~mol_mask_2d.unsqueeze(-1)  # [B, N, N, 1]
        
        # Mask out invalid coord_diffs (apply to all 3 dimensions)
        coord_diffs = coord_diffs * (~combined_mask).float()  # [B, N, N, 3]

        # Time embedding
        t_emb = self.embed_t(t, mol_mask)  # [B, N, c_t]

        # Compute pairwise interaction weights via PhiX using coord_diffs
        phi_weights = self.phi_X(s1_tilde, s2_tilde, coord_diffs, t_emb, p1_mask, p2_mask, mol_mask)  # [B, N, N, 1]
        phi_weights = phi_weights.masked_fill(combined_mask, 0.0)  # [B, N, N, 1]

        # Predict coordinates via weighted aggregation of pairwise displacements
        X_pred = torch.sum(coord_diffs * phi_weights, dim=2)  # [B, N, 3]
        X_pred = X_pred * mol_mask.unsqueeze(-1)  # Apply final mask for safety

        # Sequence prediction: aggregate protein context and ligand features
        seq_embed = self.seq_embedder(seq_tilde)  # [B, N, c_s]
        
        # Pool protein features and broadcast to each ligand atom
        s1_pooled = s1_tilde.mean(dim=1, keepdim=True)  # [B, 1, c_s]
        s2_pooled = s2_tilde.mean(dim=1, keepdim=True)  # [B, 1, c_s]
        
        phi_a_input = torch.cat([
            s1_pooled.expand(-1, N, -1),
            s2_pooled.expand(-1, N, -1),
            t_emb, 
            seq_embed
        ], dim=-1)  # [B, N, 4*c_s]
        
        seq_pred = self.seq_net(phi_a_input)  # [B, N, 20]

        return X_pred, seq_pred

class VFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.node_embedder = NodeEmbedder(cfg.node_embed_size)
        self.edge_embedder = EdgeEmbedder(cfg.edge_embed_size)

    def encode(self, batch):
        rotmats_p1 = construct_3d_basis(batch['p1_coords'], batch['p1_c_coords'], batch['p1_n_coords'])
        rotmats_p2 = construct_3d_basis(batch['p2_coords'], batch['p2_c_coords'], batch['p2_n_coords'])
        trans_p1 = batch['p1_coords']
        trans_p2 = batch['p2_coords']
        

    def forward(self,):
        pass