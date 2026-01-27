"""
IEGMN (Interaction Equivariant Graph Matching Network) implementation.
Adapted from EquiDock's rigid_docking_model.py to match EGNN input format.

Based on the IEGMN algorithm described in the paper, which processes two sets
of nodes (V1 and V2) with both intra-set and cross-set message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
import math


def normalize_vector(v, dim, eps=1e-6):
    """Normalize vector along specified dimension."""
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def construct_local_basis_from_atoms(coors_N, coors_CA, coors_C, mask=None):
    """
    Construct local basis vectors (n_i, u_i, v_i) from N, CA, C atom coordinates.
    Based on the standard protein backbone local coordinate system:
    - u_i: direction from CA to N (C-N bond direction)
    - t_i: direction from CA to C (C-C bond direction, to carbonyl carbon)
    - n_i: cross(u_i, t_i) / ||cross(u_i, t_i)||
    - v_i: cross(n_i, u_i)
    
    Args:
        coors_N: (B, N, 3) - N atom coordinates
        coors_CA: (B, N, 3) - CA (alpha-carbon) coordinates
        coors_C: (B, N, 3) - C (carbonyl carbon) coordinates
        mask: (B, N) - optional mask for valid residues
    
    Returns:
        basis: (B, N, 3, 3) - local basis matrix where:
               basis[:, :, 0] = n_i (first basis vector)
               basis[:, :, 1] = u_i (second basis vector, points to N)
               basis[:, :, 2] = v_i (third basis vector)
    """
    B, N, _ = coors_CA.shape
    device = coors_CA.device
    eps = 1e-6
    
    # u_i: direction from CA to N (normalized)
    u_i = coors_N - coors_CA  # (B, N, 3)
    u_i = normalize_vector(u_i, dim=-1, eps=eps)
    
    # t_i: direction from CA to C (normalized)
    t_i = coors_C - coors_CA  # (B, N, 3)
    t_i = normalize_vector(t_i, dim=-1, eps=eps)
    
    # n_i: cross(u_i, t_i) / ||cross(u_i, t_i)||
    n_i = torch.cross(u_i, t_i, dim=-1)  # (B, N, 3)
    n_i_norm = torch.linalg.norm(n_i, ord=2, dim=-1, keepdim=True) + eps
    n_i = n_i / n_i_norm
    
    # Handle degenerate case: if u_i and t_i are parallel, use a default direction
    degenerate_mask = n_i_norm.squeeze(-1) < 1e-4  # (B, N)
    if degenerate_mask.any():
        # Use a default n_i perpendicular to u_i
        # Choose a reference direction that's not parallel to u_i
        ref_dir = torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1)
        n_i_alt = torch.cross(u_i, ref_dir, dim=-1)
        n_i_alt_norm = torch.linalg.norm(n_i_alt, ord=2, dim=-1, keepdim=True) + eps
        
        # If still degenerate, try another direction
        still_degen = n_i_alt_norm.squeeze(-1) < 1e-4
        if still_degen.any():
            ref_dir2 = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1)
            n_i_alt2 = torch.cross(u_i, ref_dir2, dim=-1)
            n_i_alt2_norm = torch.linalg.norm(n_i_alt2, ord=2, dim=-1, keepdim=True) + eps
            n_i_alt2 = n_i_alt2 / n_i_alt2_norm
            n_i_alt = torch.where(still_degen.unsqueeze(-1).expand_as(n_i_alt), n_i_alt2, n_i_alt / n_i_alt_norm)
        else:
            n_i_alt = n_i_alt / n_i_alt_norm
        
        # Replace degenerate n_i with alternative
        n_i = torch.where(degenerate_mask.unsqueeze(-1).expand_as(n_i), n_i_alt, n_i)
    
    # v_i: cross(n_i, u_i) to complete the orthonormal basis
    v_i = torch.cross(n_i, u_i, dim=-1)  # (B, N, 3)
    v_i = normalize_vector(v_i, dim=-1, eps=eps)
    
    # Stack to form basis matrix: (B, N, 3, 3)
    # Each basis vector is a column: [n_i, u_i, v_i]
    basis = torch.stack([n_i, u_i, v_i], dim=-1)  # (B, N, 3, 3)
    
    # Apply mask if provided
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
        basis = basis * mask_expanded
    
    return basis
    

def compute_edge_features(coors_i, coors_j,
                         coors_N_i, coors_CA_i, coors_C_i,
                         coors_N_j, coors_CA_j, coors_C_j,
                         mask_i=None, mask_j=None, 
                         num_dist_basis=15, dist_sigma_base=1.5):
    """
    Compute three types of edge features:
    1. Relative Position: p_{j→i} = [n_i^T; u_i^T; v_i^T] * (x_j - x_i)
    2. Relative Orientation: q_{j→i}, k_{j→i}, t_{j→i} = [n_i^T; u_i^T; v_i^T] * [n_j, u_j, v_j]
    3. Distance-Based: f_{j→i,r} = exp(-||x_j - x_i||^2 / (2σ_r^2))
    
    Args:
        coors_i: (B, N_i, 3) - CA coordinates for nodes i
        coors_j: (B, N_j, 3) - CA coordinates for nodes j
        coors_N_i: (B, N_i, 3) - N atom coordinates for nodes i (required)
        coors_CA_i: (B, N_i, 3) - CA atom coordinates for nodes i (required)
        coors_C_i: (B, N_i, 3) - C atom coordinates for nodes i (required)
        coors_N_j: (B, N_j, 3) - N atom coordinates for nodes j (required)
        coors_CA_j: (B, N_j, 3) - CA atom coordinates for nodes j (required)
        coors_C_j: (B, N_j, 3) - C atom coordinates for nodes j (required)
        mask_i: (B, N_i) - optional mask for nodes i
        mask_j: (B, N_j) - optional mask for nodes j
        num_dist_basis: number of distance basis functions
        dist_sigma_base: base for distance sigma values
    
    Returns:
        edge_feats: (B, N_i, N_j, feat_dim) - concatenated edge features
    """
    B, N_i, _ = coors_i.shape
    B, N_j, _ = coors_j.shape
    
    # Compute relative position: x_j - x_i (using CA positions)
    rel_pos = rearrange(coors_j, 'b j d -> b () j d') - rearrange(coors_i, 'b i d -> b i () d')  # (B, N_i, N_j, 3)
    
    # Construct local basis from N, CA, C coordinates
    basis_i = construct_local_basis_from_atoms(coors_N_i, coors_CA_i, coors_C_i, mask_i)
    basis_j = construct_local_basis_from_atoms(coors_N_j, coors_CA_j, coors_C_j, mask_j)
    
    # Extract basis vectors from j for relative orientation features
    n_j = basis_j[:, :, 0, :]  # (B, N_j, 3)
    u_j = basis_j[:, :, 1, :]  # (B, N_j, 3)
    v_j = basis_j[:, :, 2, :]  # (B, N_j, 3)
    
    # 1. Relative Position Edge Features: p_{j→i} = [n_i^T; u_i^T; v_i^T] * (x_j - x_i)
    # Stack basis vectors: (B, N_i, 3, 3) where rows are n_i^T, u_i^T, v_i^T
    basis_matrix_i = basis_i.transpose(-1, -2)  # (B, N_i, 3, 3) - transpose so rows are basis vectors
    
    # Compute p_{j→i}: (B, N_i, N_j, 3)
    # For each i, we have basis_matrix_i[i] which is (3, 3), and rel_pos[i, j] which is (3,)
    # We want: basis_matrix_i[i] @ rel_pos[i, j] for all j
    p_ji = einsum(basis_matrix_i, rel_pos, 'b i d1 d2, b i j d2 -> b i j d1')  # (B, N_i, N_j, 3)
    
    # 2. Relative Orientation Edge Features: q_{j→i}, k_{j→i}, t_{j→i}
    # q_{j→i} = [n_i^T; u_i^T; v_i^T] * n_j
    # k_{j→i} = [n_i^T; u_i^T; v_i^T] * u_j
    # t_{j→i} = [n_i^T; u_i^T; v_i^T] * v_j
    q_ji = einsum(basis_matrix_i, n_j, 'b i d1 d2, b j d2 -> b i j d1')  # (B, N_i, N_j, 3)
    k_ji = einsum(basis_matrix_i, u_j, 'b i d1 d2, b j d2 -> b i j d1')  # (B, N_i, N_j, 3)
    t_ji = einsum(basis_matrix_i, v_j, 'b i d1 d2, b j d2 -> b i j d1')  # (B, N_i, N_j, 3)
    
    # 3. Distance-Based Edge Features: f_{j→i,r} = exp(-||x_j - x_i||^2 / (2σ_r^2))
    rel_dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)  # (B, N_i, N_j, 1)
    
    # Compute RBF features for different sigma values
    sigmas = [dist_sigma_base ** x for x in range(num_dist_basis)]
    dist_feats = []
    for sigma in sigmas:
        sigma_sq = 2 * (sigma ** 2)
        feat = torch.exp(-rel_dist_sq / sigma_sq)  # (B, N_i, N_j, 1)
        dist_feats.append(feat)
    
    dist_feats = torch.cat(dist_feats, dim=-1)  # (B, N_i, N_j, num_dist_basis)
    
    # Concatenate all edge features
    edge_feats = torch.cat([
        p_ji,      # (B, N_i, N_j, 3) - relative position
        q_ji,      # (B, N_i, N_j, 3) - relative orientation (n)
        k_ji,      # (B, N_i, N_j, 3) - relative orientation (u)
        t_ji,      # (B, N_i, N_j, 3) - relative orientation (v)
        dist_feats # (B, N_i, N_j, num_dist_basis) - distance-based
    ], dim=-1)  # (B, N_i, N_j, 3+3+3+3+num_dist_basis)
    
    return edge_feats


def exists(val):
    return val is not None


def get_non_lin(type, negative_slope=0.1):
    if type == 'swish' or type == 'silu':
        return nn.SiLU()
    else:
        assert type == 'lkyrelu' or type == 'leakyrelu'
        return nn.LeakyReLU(negative_slope=negative_slope)


def get_layer_norm(layer_norm_type, dim):
    if layer_norm_type == 'BN':
        return nn.BatchNorm1d(dim)
    elif layer_norm_type == 'LN':
        return nn.LayerNorm(dim)
    else:
        return nn.Identity()


def compute_cross_attention(queries, keys, values, mask, cross_msgs=True):
    """
    Compute cross attention between two sets.
    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    attention_x = sum_j a_{i->j} y_j
    
    Args:
        queries: (B, N1, D) float tensor --> queries from set 1
        keys: (B, N2, D) float tensor --> keys from set 2
        values: (B, N2, d) float tensor --> values from set 2
        mask: (B, N1, N2) boolean tensor --> mask for valid pairs
        cross_msgs: bool --> whether to compute cross messages
    
    Returns:
        attention_x: (B, N1, d) float tensor.
    """
    if not cross_msgs:
        return queries * 0.
    
    # Compute attention scores: (B, N1, N2)
    a = einsum(queries, keys, 'b i d, b j d -> b i j')
    
    # Apply mask: set invalid pairs to -inf
    a = a.masked_fill(~mask, -1e9)
    
    # Softmax over j dimension
    a_x = F.softmax(a, dim=-1)  # (B, N1, N2)
    
    # Weighted sum of values
    attention_x = einsum(a_x, values, 'b i j, b j d -> b i d')  # (B, N1, d)
    
    return attention_x


class IEGMN_Layer(nn.Module):
    """
    Single IEGMN layer implementing equations (5)-(10) from the paper.
    
    Processes two sets of nodes (V1 and V2) with:
    - Intra-set message passing (within V1 and V2)
    - Cross-set attention (between V1 and V2)
    - Coordinate and feature updates
    """
    
    def __init__(
        self,
        h_feats_dim,  # input dimension of h
        out_feats_dim,  # output dimension of h
        input_edge_feats_dim=0,  # dimension of input edge features
        dropout=0.0,
        nonlin='silu',
        cross_msgs=True,
        layer_norm='LN',
        layer_norm_coors='0',
        final_h_layer_norm='LN',
        use_dist_in_layers=True,
        skip_weight_h=0.75,
        x_connection_init=0.5,
        leakyrelu_neg_slope=0.1,
        num_dist_basis=15,
        dist_sigma_base=1.5,
    ):
        super().__init__()
        
        self.h_feats_dim = h_feats_dim
        self.out_feats_dim = out_feats_dim
        self.cross_msgs = cross_msgs
        self.use_dist_in_layers = use_dist_in_layers
        self.skip_weight_h = skip_weight_h
        self.x_connection_init = x_connection_init
        self.num_dist_basis = num_dist_basis
        self.dist_sigma_base = dist_sigma_base
        
        # Distance basis functions: exp(-||x_i - x_j||^2 / sigma)
        self.all_sigmas_dist = [dist_sigma_base ** x for x in range(num_dist_basis)]
        
        # Edge features dimensions:
        # - Relative position: 3
        # - Relative orientation (q, k, t): 3 + 3 + 3 = 9
        # - Distance-based: num_dist_basis
        self.edge_feat_dim = 3 + 9 + num_dist_basis  # 3 + 9 + num_dist_basis
        
        # EDGES: Equation (5) - m_{j→i} = φ^e(h_i, h_j, exp(-||x_i - x_j||^2/σ), f_{j→i})
        # edge_input_dim = h_i + h_j + edge_features + input_edge_feats
        edge_input_dim = (h_feats_dim * 2) + self.edge_feat_dim + input_edge_feats_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, out_feats_dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            get_layer_norm(layer_norm, out_feats_dim),
            nn.Linear(out_feats_dim, out_feats_dim),
        )
        
        # NODES: Equation (6) - Cross attention μ_{j→i}
        self.att_mlp_Q = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_K = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_V = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
        )
        
        # NODES: Equation (10) - h_i^{(l+1)} = (1-β)h_i^{(l)} + βφ^h(h_i^{(l)}, m_i, μ_i, f_i)
        self.node_norm = get_layer_norm(layer_norm, h_feats_dim)
        
        # Original features are concatenated for skip connection
        orig_h_feats_dim = h_feats_dim  # Assuming same as input
        self.node_mlp = nn.Sequential(
            nn.Linear(orig_h_feats_dim + 2 * h_feats_dim + out_feats_dim, h_feats_dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            get_layer_norm(layer_norm, h_feats_dim),
            nn.Linear(h_feats_dim, out_feats_dim),
        )
        
        self.final_h_layernorm_layer = get_layer_norm(final_h_layer_norm, out_feats_dim)
        
        # COORDINATES: Equation (9) - φ^x(m_{j→i}) for coordinate update
        self.coors_mlp = nn.Sequential(
            nn.Linear(out_feats_dim, out_feats_dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            get_layer_norm(layer_norm_coors, out_feats_dim),
            nn.Linear(out_feats_dim, 1)
        )
    
    def forward(
        self,
        feats1, coors1, orig_feats1, orig_coors1,
        feats2, coors2, orig_feats2, orig_coors2,
        coors_N1, coors_CA1, coors_C1,
        coors_N2, coors_CA2, coors_C2,
        edges1=None, mask1=None, edges2=None, mask2=None,
    ):
        """
        Forward pass of IEGMN layer.
        
        Args:
            feats1: (B, N1, h_feats_dim) - node features for set 1
            coors1: (B, N1, 3) - node coordinates for set 1
            orig_feats1: (B, N1, orig_h_feats_dim) - original node features for set 1 (for skip connection)
            orig_coors1: (B, N1, 3) - original node coordinates for set 1 (for skip connection)
            edges1: (B, N1, N1, edge_dim) - edge features for set 1 (optional)
            mask1: (B, N1) - mask for set 1 (optional)
            
            feats2: (B, N2, h_feats_dim) - node features for set 2
            coors2: (B, N2, 3) - node coordinates for set 2
            orig_feats2: (B, N2, orig_h_feats_dim) - original node features for set 2 (for skip connection)
            orig_coors2: (B, N2, 3) - original node coordinates for set 2 (for skip connection)
            edges2: (B, N2, N2, edge_dim) - edge features for set 2 (optional)
            mask2: (B, N2) - mask for set 2 (optional)
        
        Returns:
            feats1_out: (B, N1, out_feats_dim) - updated features for set 1
            coors1_out: (B, N1, 3) - updated coordinates for set 1
            feats2_out: (B, N2, out_feats_dim) - updated features for set 2
            coors2_out: (B, N2, 3) - updated coordinates for set 2
        """
        B, N1, _ = feats1.shape
        B, N2, _ = feats2.shape
        device = feats1.device
        
        # ========== INTRA-SET MESSAGE PASSING ==========
        # Equation (5): m_{j→i} = φ^e(h_i, h_j, exp(-||x_i - x_j||^2/σ), f_{j→i})
        
        # Compute edge features for set 1 (intra-set)
        edge_feats1 = compute_edge_features(
            coors1, coors1,
            coors_N_i=coors_N1, coors_CA_i=coors_CA1, coors_C_i=coors_C1,
            coors_N_j=coors_N1, coors_CA_j=coors_CA1, coors_C_j=coors_C1,
            mask_i=mask1, mask_j=mask1,
            num_dist_basis=self.num_dist_basis,
            dist_sigma_base=self.dist_sigma_base
        )  # (B, N1, N1, edge_feat_dim)
        
        if not self.use_dist_in_layers:
            # Zero out distance-based features if not using them
            edge_feats1[:, :, :, -len(self.all_sigmas_dist):] = 0.
        
        # Prepare edge input for set 1
        feats_i1 = rearrange(feats1, 'b i d -> b i () d')  # (B, N1, 1, d)
        feats_j1 = rearrange(feats1, 'b j d -> b () j d')  # (B, 1, N1, d)
        feats_i1, feats_j1 = torch.broadcast_tensors(feats_i1, feats_j1)  # (B, N1, N1, d)
        
        edge_input1 = torch.cat([feats_i1, feats_j1, edge_feats1], dim=-1)  # (B, N1, N1, 2*d + edge_feat_dim)
        if edges1 is not None:
            edge_input1 = torch.cat([edge_input1, edges1], dim=-1)
        
        # Compute messages for set 1
        msg1 = self.edge_mlp(edge_input1)  # (B, N1, N1, out_feats_dim)
        
        # Apply mask if provided
        if mask1 is not None:
            mask1_2d = mask1.unsqueeze(2) & mask1.unsqueeze(1)  # (B, N1, N1)
            msg1 = msg1.masked_fill(~mask1_2d.unsqueeze(-1), 0.)
        
        # Same for set 2
        edge_feats2 = compute_edge_features(
            coors2, coors2,
            coors_N_i=coors_N2, coors_CA_i=coors_CA2, coors_C_i=coors_C2,
            coors_N_j=coors_N2, coors_CA_j=coors_CA2, coors_C_j=coors_C2,
            mask_i=mask2, mask_j=mask2,
            num_dist_basis=self.num_dist_basis,
            dist_sigma_base=self.dist_sigma_base
        )  # (B, N2, N2, edge_feat_dim)
        
        if not self.use_dist_in_layers:
            # Zero out distance-based features if not using them
            edge_feats2[:, :, :, -len(self.all_sigmas_dist):] = 0.
        
        feats_i2 = rearrange(feats2, 'b i d -> b i () d')
        feats_j2 = rearrange(feats2, 'b j d -> b () j d')
        feats_i2, feats_j2 = torch.broadcast_tensors(feats_i2, feats_j2)
        
        edge_input2 = torch.cat([feats_i2, feats_j2, edge_feats2], dim=-1)
        if edges2 is not None:
            edge_input2 = torch.cat([edge_input2, edges2], dim=-1)
        
        msg2 = self.edge_mlp(edge_input2)
        
        if mask2 is not None:
            mask2_2d = mask2.unsqueeze(2) & mask2.unsqueeze(1)
            msg2 = msg2.masked_fill(~mask2_2d.unsqueeze(-1), 0.)
        
        # ========== CROSS-SET ATTENTION ==========
        # Equation (6): μ_{j→i} = a_{j→i} W h_j^{(l)}
        # Equation (8): μ_i = sum_{j∈V2} μ_{j→i} for i∈V1, and vice versa
        
        # Create cross-attention mask
        if mask1 is not None and mask2 is not None:
            cross_mask = mask1.unsqueeze(2) & mask2.unsqueeze(1)  # (B, N1, N2)
        else:
            cross_mask = torch.ones(B, N1, N2, dtype=torch.bool, device=device)
        
        # Cross attention: set 1 attends to set 2
        aggr_cross_msg1 = compute_cross_attention(
            self.att_mlp_Q(feats1),  # queries from set 1
            self.att_mlp_K(feats2),  # keys from set 2
            self.att_mlp_V(feats2),  # values from set 2
            cross_mask,
            self.cross_msgs
        )  # (B, N1, h_feats_dim)
        
        # Cross attention: set 2 attends to set 1
        aggr_cross_msg2 = compute_cross_attention(
            self.att_mlp_Q(feats2),  # queries from set 2
            self.att_mlp_K(feats1),  # keys from set 1
            self.att_mlp_V(feats1),  # values from set 1
            cross_mask.transpose(1, 2),  # (B, N2, N1)
            self.cross_msgs
        )  # (B, N2, h_feats_dim)
        
        # ========== COORDINATE UPDATE ==========
        # Equation (9): x_i^{(l+1)} = ηx_i^{(0)} + (1-η)x_i^{(l)} + sum_j (x_i^{(l)} - x_j^{(l)}) φ^x(m_{j→i})
        
        # Compute coordinate update weights for set 1
        edge_coef1 = self.coors_mlp(msg1)  # (B, N1, N1, 1) - φ^x(m_{j→i})
        # Compute relative coordinates for coordinate update
        rel_coors1 = rearrange(coors1, 'b i d -> b i () d') - rearrange(coors1, 'b j d -> b () j d')  # (B, N1, N1, 3)
        x_moment1 = rel_coors1 * edge_coef1  # (B, N1, N1, 3) - (x_i - x_j) * φ^x(m_{j→i})
        
        # Aggregate coordinate updates: mean over neighbors
        if mask1 is not None:
            mask1_2d = mask1.unsqueeze(2) & mask1.unsqueeze(1)
            x_moment1 = x_moment1.masked_fill(~mask1_2d.unsqueeze(-1), 0.)
            num_neighbors1 = mask1_2d.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, N1, 1)
            x_update1 = x_moment1.sum(dim=-2) / num_neighbors1  # (B, N1, 3)
        else:
            x_update1 = x_moment1.mean(dim=-2)  # (B, N1, 3)
        
        # Apply coordinate update with skip connection
        coors1_out = (
            self.x_connection_init * orig_coors1 +
            (1. - self.x_connection_init) * coors1 +
            x_update1
        )
        
        # Same for set 2
        edge_coef2 = self.coors_mlp(msg2)
        # Compute relative coordinates for coordinate update
        rel_coors2 = rearrange(coors2, 'b i d -> b i () d') - rearrange(coors2, 'b j d -> b () j d')  # (B, N2, N2, 3)
        x_moment2 = rel_coors2 * edge_coef2
        
        if mask2 is not None:
            mask2_2d = mask2.unsqueeze(2) & mask2.unsqueeze(1)
            x_moment2 = x_moment2.masked_fill(~mask2_2d.unsqueeze(-1), 0.)
            num_neighbors2 = mask2_2d.sum(dim=-1, keepdim=True).clamp(min=1)
            x_update2 = x_moment2.sum(dim=-2) / num_neighbors2
        else:
            x_update2 = x_moment2.mean(dim=-2)
        
        coors2_out = (
            self.x_connection_init * orig_coors2 +
            (1. - self.x_connection_init) * coors2 +
            x_update2
        )
        
        # ========== FEATURE UPDATE ==========
        # Equation (7): m_i = (1/|N(i)|) sum_{j∈N(i)} m_{j→i}
        # Equation (10): h_i^{(l+1)} = (1-β)h_i^{(l)} + βφ^h(h_i^{(l)}, m_i, μ_i, f_i)
        
        # Aggregate intra-set messages for set 1
        if mask1 is not None:
            mask1_2d = mask1.unsqueeze(2) & mask1.unsqueeze(1)
            msg1 = msg1.masked_fill(~mask1_2d.unsqueeze(-1), 0.)
            num_neighbors1 = mask1_2d.sum(dim=-1, keepdim=True).clamp(min=1)
            aggr_msg1 = msg1.sum(dim=-2) / num_neighbors1  # (B, N1, out_feats_dim)
        else:
            aggr_msg1 = msg1.mean(dim=-2)  # (B, N1, out_feats_dim)
        
        # Prepare node update input for set 1
        input_node_upd1 = torch.cat([
            self.node_norm(feats1),  # h_i^{(l)}
            aggr_msg1,  # m_i
            aggr_cross_msg1,  # μ_i
            orig_feats1,  # original features for skip connection
        ], dim=-1)  # (B, N1, orig_h_feats_dim + 2*h_feats_dim + out_feats_dim)
        
        # Update features with skip connection
        if self.h_feats_dim == self.out_feats_dim:
            node_upd1 = (
                self.skip_weight_h * self.node_mlp(input_node_upd1) +
                (1. - self.skip_weight_h) * feats1
            )
        else:
            node_upd1 = self.node_mlp(input_node_upd1)
        
        # Apply final layer norm
        feats1_out = self.final_h_layernorm_layer(node_upd1)
        
        # Same for set 2
        if mask2 is not None:
            mask2_2d = mask2.unsqueeze(2) & mask2.unsqueeze(1)
            msg2 = msg2.masked_fill(~mask2_2d.unsqueeze(-1), 0.)
            num_neighbors2 = mask2_2d.sum(dim=-1, keepdim=True).clamp(min=1)
            aggr_msg2 = msg2.sum(dim=-2) / num_neighbors2
        else:
            aggr_msg2 = msg2.mean(dim=-2)
        
        input_node_upd2 = torch.cat([
            self.node_norm(feats2),
            aggr_msg2,
            aggr_cross_msg2,
            orig_feats2,
        ], dim=-1)
        
        if self.h_feats_dim == self.out_feats_dim:
            node_upd2 = (
                self.skip_weight_h * self.node_mlp(input_node_upd2) +
                (1. - self.skip_weight_h) * feats2
            )
        else:
            node_upd2 = self.node_mlp(input_node_upd2)
        
        feats2_out = self.final_h_layernorm_layer(node_upd2)
        
        return feats1_out, coors1_out, feats2_out, coors2_out


class IEGMN(nn.Module):
    """
    IEGMN network with multiple layers.
    
    Input format matches EGNN:
    - feats1, coors1, mask1: features, coordinates, and mask for set 1
    - feats2, coors2, mask2: features, coordinates, and mask for set 2
    - edges1, edges2: optional edge features
    """
    
    def __init__(
        self,
        dim,  # feature dimension
        depth=4,  # number of layers
        edge_dim=0,  # edge feature dimension
        dropout=0.0,
        nonlin='silu',
        cross_msgs=True,
        layer_norm='LN',
        layer_norm_coors='0',
        final_h_layer_norm='LN',
        use_dist_in_layers=True,
        skip_weight_h=0.75,
        x_connection_init=0.5,
        leakyrelu_neg_slope=0.1,
        num_dist_basis=15,
        dist_sigma_base=1.5,
        shared_layers=False,
        num_att_heads=4,  # number of attention heads for keypoint generation
    ):
        super().__init__()
        
        self.dim = dim
        self.depth = depth
        self.num_att_heads = num_att_heads
        
        self.layers = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.layers.append(
            IEGMN_Layer(
                h_feats_dim=dim,
                out_feats_dim=dim,
                input_edge_feats_dim=edge_dim,
                dropout=dropout,
                nonlin=nonlin,
                cross_msgs=cross_msgs,
                layer_norm=layer_norm,
                layer_norm_coors=layer_norm_coors,
                final_h_layer_norm=final_h_layer_norm,
                use_dist_in_layers=use_dist_in_layers,
                skip_weight_h=skip_weight_h,
                x_connection_init=x_connection_init,
                leakyrelu_neg_slope=leakyrelu_neg_slope,
                num_dist_basis=num_dist_basis,
                dist_sigma_base=dist_sigma_base,
            )
        )
        
        # Remaining layers
        if shared_layers:
            interm_layer = IEGMN_Layer(
                h_feats_dim=dim,
                out_feats_dim=dim,
                input_edge_feats_dim=edge_dim,
                dropout=dropout,
                nonlin=nonlin,
                cross_msgs=cross_msgs,
                layer_norm=layer_norm,
                layer_norm_coors=layer_norm_coors,
                final_h_layer_norm=final_h_layer_norm,
                use_dist_in_layers=use_dist_in_layers,
                skip_weight_h=skip_weight_h,
                x_connection_init=x_connection_init,
                leakyrelu_neg_slope=leakyrelu_neg_slope,
                num_dist_basis=num_dist_basis,
                dist_sigma_base=dist_sigma_base,
            )
            for _ in range(1, depth):
                self.layers.append(interm_layer)
        else:
            for _ in range(1, depth):
                self.layers.append(
                    IEGMN_Layer(
                        h_feats_dim=dim,
                        out_feats_dim=dim,
                        input_edge_feats_dim=edge_dim,
                        dropout=dropout,
                        nonlin=nonlin,
                        cross_msgs=cross_msgs,
                        layer_norm=layer_norm,
                        layer_norm_coors=layer_norm_coors,
                        final_h_layer_norm=final_h_layer_norm,
                        use_dist_in_layers=use_dist_in_layers,
                        skip_weight_h=skip_weight_h,
                        x_connection_init=x_connection_init,
                        leakyrelu_neg_slope=leakyrelu_neg_slope,
                        num_dist_basis=num_dist_basis,
                        dist_sigma_base=dist_sigma_base,
                    )
                )
        
        # Initialize attention layers for keypoint generation (for R and t prediction)
        self.att_mlp_key_ROT = nn.Linear(self.dim, self.num_att_heads * self.dim, bias=False)
        self.att_mlp_query_ROT = nn.Linear(self.dim, self.num_att_heads * self.dim, bias=False)
        
        self.mlp_h_mean_ROT = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
    
    def forward(
        self,
        feats1, coors1,
        feats2, coors2,
        coors_N1, coors_CA1, coors_C1,
        coors_N2, coors_CA2, coors_C2,
        mask1=None, edges1=None, mask2=None, edges2=None,
    ):
        """
        Forward pass through IEGMN network.
        
        Args:
            feats1: (B, N1, dim) - node features for set 1
            coors1: (B, N1, 3) - node coordinates for set 1
            mask1: (B, N1) - mask for set 1 (optional)
            edges1: (B, N1, N1, edge_dim) - edge features for set 1 (optional)
            
            feats2: (B, N2, dim) - node features for set 2
            coors2: (B, N2, 3) - node coordinates for set 2
            mask2: (B, N2) - mask for set 2 (optional)
            edges2: (B, N2, N2, edge_dim) - edge features for set 2 (optional)
        
        Returns:
            feats1_out: (B, N1, dim) - updated features for set 1
            coors1_out: (B, N1, 3) - updated coordinates for set 1
            feats2_out: (B, N2, dim) - updated features for set 2
            coors2_out: (B, N2, 3) - updated coordinates for set 2
            R: (B, 3, 3) - rotation matrices
            t: (B, 3) - translation vectors
            Y1: (B, K, 3) - keypoints for set 1
            Y2: (B, K, 3) - keypoints for set 2
        """
        # Store original features and coordinates for skip connections
        orig_feats1 = feats1
        orig_coors1 = coors1
        orig_feats2 = feats2
        orig_coors2 = coors2
        
        # Pass through all layers
        for layer in self.layers:
            feats1, coors1, feats2, coors2 = layer(
                feats1, coors1, orig_feats1, orig_coors1,
                feats2, coors2, orig_feats2, orig_coors2,
                coors_N1, coors_CA1, coors_C1,
                coors_N2, coors_CA2, coors_C2,
                edges1, mask1, edges2, mask2,
            )
        
        # Predict rigid transformation (R and t) using Kabsch algorithm
        B, N1, d = feats1.shape
        B, N2, d = feats2.shape
        K = self.num_att_heads
        device = feats1.device
        
        # Compute mean features for attention queries
        # Apply mask if provided
        if mask1 is not None:
            mask1_expanded = mask1.unsqueeze(-1)  # (B, N1, 1)
            feats1_masked = feats1 * mask1_expanded
            H1_mean = self.mlp_h_mean_ROT(feats1_masked)  # (B, N1, d)
            H1_mean = H1_mean.sum(dim=1, keepdim=True) / (mask1.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6)  # (B, 1, d)
        else:
            H1_mean = torch.mean(self.mlp_h_mean_ROT(feats1), dim=1, keepdim=True)  # (B, 1, d)
        
        if mask2 is not None:
            mask2_expanded = mask2.unsqueeze(-1)  # (B, N2, 1)
            feats2_masked = feats2 * mask2_expanded
            H2_mean = self.mlp_h_mean_ROT(feats2_masked)  # (B, N2, d)
            H2_mean = H2_mean.sum(dim=1, keepdim=True) / (mask2.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6)  # (B, 1, d)
        else:
            H2_mean = torch.mean(self.mlp_h_mean_ROT(feats2), dim=1, keepdim=True)  # (B, 1, d)
        
        # Generate keypoints Y1 and Y2 using multi-head attention
        # Y1: attention weights computed using H2_mean as query, H1 as keys
        # Y2: attention weights computed using H1_mean as query, H2 as keys
        
        # Compute attention weights for Y2 (receptor keypoints)
        # Query: H1_mean (average of set 1 features), Keys: H2 (set 2 features)
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
        
        att_weights_2 = F.softmax(att_scores_2.squeeze(-1), dim=-1)  # (B, K, N2)
        
        # Compute Y2: weighted sum of coordinates
        Y2 = einsum(att_weights_2, coors2, 'b k n2, b n2 d -> b k d')  # (B, K, 3)
        
        # Compute attention weights for Y1 (ligand keypoints)
        # Query: H2_mean (average of set 2 features), Keys: H1 (set 1 features)
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
        
        att_weights_1 = F.softmax(att_scores_1.squeeze(-1), dim=-1)  # (B, K, N1)
        
        # Compute Y1: weighted sum of coordinates
        Y1 = einsum(att_weights_1, coors1, 'b k n1, b n1 d -> b k d')  # (B, K, 3)
        
        # Apply Kabsch algorithm to compute R and t
        # Center keypoints
        Y1_mean = Y1.mean(dim=1, keepdim=True)  # (B, 1, 3)
        Y2_mean = Y2.mean(dim=1, keepdim=True)  # (B, 1, 3)
        
        Y1_centered = Y1 - Y1_mean  # (B, K, 3)
        Y2_centered = Y2 - Y2_mean  # (B, K, 3)
        
        # Compute covariance matrix A = Y2^T @ Y1
        A = einsum(Y2_centered, Y1_centered, 'b k d2, b k d1 -> b d2 d1')  # (B, 3, 3)
        
        # SVD: A = U @ S @ V^T
        # Handle numerical instability for each sample in batch
        R_list = []
        for b in range(B):
            A_b = A[b]  # (3, 3)
            U_b, S_b, Vt_b = torch.linalg.svd(A_b)
            
            # Check for numerical instability
            num_it = 0
            while torch.min(S_b) < 1e-3:
                # Add small random perturbation to diagonal
                A_b = A_b + torch.rand(3, 3, device=device) * 1e-6 * torch.eye(3, device=device)
                U_b, S_b, Vt_b = torch.linalg.svd(A_b)
                num_it += 1
                if num_it > 10:
                    # If still unstable, use identity rotation
                    U_b = torch.eye(3, device=device)
                    Vt_b = torch.eye(3, device=device)
                    break
            
            # Compute correction factor: d = sign(det(U @ V^T))
            det_U_Vt = torch.det(U_b @ Vt_b)
            d = torch.sign(det_U_Vt)
            
            # Construct correction matrix: diag([1, 1, d])
            corr_mat_b = torch.eye(3, device=device)
            corr_mat_b[2, 2] = d
            
            # Compute rotation matrix: R = U @ corr_mat @ V^T
            R_b = U_b @ corr_mat_b @ Vt_b  # (3, 3)
            R_list.append(R_b)
        
        R = torch.stack(R_list, dim=0)  # (B, 3, 3)
        
        # Compute translation: t = mu(Y2) - R @ mu(Y1)
        Y1_mean_squeezed = Y1_mean.squeeze(1)  # (B, 3)
        R_Y1_mean = einsum(R, Y1_mean_squeezed, 'b d1 d2, b d2 -> b d1')  # (B, 3)
        t = Y2_mean.squeeze(1) - R_Y1_mean  # (B, 3)
        
        return feats1, coors1, feats2, coors2, R, t, Y1, Y2

