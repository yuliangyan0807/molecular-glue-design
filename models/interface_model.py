from .egnn import EGNN_Network
from .iegmn import IEGMN
from einops import einsum

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InterfaceModel(nn.Module):
    def __init__(
        self,
        num_tokens: int = 22,
        dim: int = 128,
        depth: int = 4,
        num_nearest_neighbors: int = 16,
        norm_coors: bool = True,
        coor_weights_clamp_value: float = 2.0,
    ):
        super().__init__()
        # Share the parameters.
        self.egnn_p1 = EGNN_Network(
            num_tokens=num_tokens,
            dim=dim,
            depth=depth,
            num_nearest_neighbors=num_nearest_neighbors,
            norm_coors=norm_coors,
            coor_weights_clamp_value=coor_weights_clamp_value
        )
        self.egnn_p2 = EGNN_Network(
            num_tokens=num_tokens,
            dim=dim,
            depth=depth,
            num_nearest_neighbors=num_nearest_neighbors,
            norm_coors=norm_coors,
            coor_weights_clamp_value=coor_weights_clamp_value
        )

        # self.weight_proj = nn.Linear(dim, 1)
        self.weight_proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )
    
    def _ellipsoid_params(self, feats, coords, mask=None):
        """
        feats: (B, N, d) invariant
        coords: (B, N, 3) equivariant
        mask: (B, N) optional
        """

        # Obtein the invariant weights
        weights = self.weight_proj(feats).squeeze(-1)  # (B, N)
        if mask is not None:
            # Use -1e4 instead of -1e9 for AMP compatibility (float16 range: ~-65504 to 65504)
            weights = weights.masked_fill(mask == 0, -1e4)
        attn = F.softmax(weights, dim=-1)  # (B, N)

        # Calculate the weighted mean mu (equivariant)
        # coords = coords / torch.sqrt(torch.sum(coords ** 2, dim=-1, keepdim=True) + 1e-6)
        mu = torch.einsum("bn,bnd->bd", attn, coords)  # (B, 3)

        # Calculate the weighted covariance Σ (equivariant SPD)
        diff = coords - mu.unsqueeze(1)  # (B, N, 3)
        cov = torch.einsum("bn,bni,bnj->bij", attn, diff, diff)  # (B, 3, 3)
        cov = cov + 1e-5 * torch.eye(3, device=cov.device).unsqueeze(0)  # numerical stability

        return mu, cov, attn

    def forward(self, p1_residue, p1_coords, p2_residue, p2_coords, p1_mask=None, p2_mask=None):

        B, _, _ = p1_coords.shape
        
        # feats are invariant to the Rotation and Translation.
        p1_feats_out, _ = self.egnn_p1(p1_residue, p1_coords, mask=p1_mask)
        p2_feats_out, _ = self.egnn_p2(p2_residue, p2_coords, mask=p2_mask)


        mu1, cov1, attn1 = self._ellipsoid_params(p1_feats_out, p1_coords, p1_mask)
        mu2, cov2, attn2 = self._ellipsoid_params(p2_feats_out, p2_coords, p2_mask)

        # Attention-weighted feature representations (interface representations)
        i1_repr = torch.einsum("bn,bnd->bd", attn1, p1_feats_out)  # (B, dim)
        i2_repr = torch.einsum("bn,bnd->bd", attn2, p2_feats_out)  # (B, dim)

        i1_params = torch.cat([mu1, cov1.reshape(B, -1)], dim=-1) # (B, 12)
        i2_params = torch.cat([mu2, cov2.reshape(B, -1)], dim=-1) # (B, 12)

        return {
            'i1_params': i1_params,
            'i2_params': i2_params,
            'p1_feats': p1_feats_out,
            'p2_feats': p2_feats_out,
            'i1_repr': i1_repr,
            'i2_repr': i2_repr,
            'attn1': attn1,
            'attn2': attn2,
        }

class MultiHeadInterfaceModel(nn.Module):
    """
    Multi-head interface model that generates K keypoints for each protein using cross-attention.
    
    Based on the paper's algorithm:
    - y₁ₖ := Σᵢ₌₁ⁿ αᵢᵏ z₁ᵢ  (keypoints for protein 1)
    - y₂ₖ := Σⱼ₌₁ᵐ βⱼᵏ z₂ⱼ  (keypoints for protein 2)
    - αᵢᵏ = softmaxᵢ (¹/√ᵈ h₁ᵢᵀ W μ(φ(H₂)))  (attention scores for protein 1)
    - βⱼᵏ = softmaxⱼ (¹/√ᵈ h₂ⱼᵀ W μ(φ(H₁)))  (attention scores for protein 2)
    
    Where:
    - z₁ᵢ, z₂ⱼ are coordinates (equivariant)
    - h₁ᵢ, h₂ⱼ are features (invariant)
    - μ(φ(H₁)) is the mean of transformed features from protein 1
    - μ(φ(H₂)) is the mean of transformed features from protein 2
    """
    def __init__(
        self,
        num_tokens: int = 22,
        dim: int = 128,
        depth: int = 4,
        num_nearest_neighbors: int = 16,
        norm_coors: bool = True,
        coor_weights_clamp_value: float = 2.0,
        num_att_heads: int = 4,  # K: number of keypoints
        dropout: float = 0.0,
        nonlin: str = 'leakyrelu',
        leakyrelu_neg_slope: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_att_heads = num_att_heads
        
        # Share the parameters.
        self.egnn_p1 = EGNN_Network(
            num_tokens=num_tokens,
            dim=dim,
            depth=depth,
            num_nearest_neighbors=num_nearest_neighbors,
            norm_coors=norm_coors,
            coor_weights_clamp_value=coor_weights_clamp_value
        )
        self.egnn_p2 = EGNN_Network(
            num_tokens=num_tokens,
            dim=dim,
            depth=depth,
            num_nearest_neighbors=num_nearest_neighbors,
            norm_coors=norm_coors,
            coor_weights_clamp_value=coor_weights_clamp_value
        )

        # Multi-head attention layers for keypoint generation (matching EquiDock's implementation)
        # Key projection: transforms features to keys for attention
        self.att_mlp_key_ROT = nn.Linear(dim, num_att_heads * dim, bias=False)
        
        # Query projection: transforms mean features to queries for attention
        self.att_mlp_query_ROT = nn.Linear(dim, num_att_heads * dim, bias=False)
        
        # Feature transformation before computing mean (φ in the paper)
        # This is applied before computing μ(φ(H₁)) and μ(φ(H₂))
        if nonlin == 'leakyrelu':
            nonlin_layer = nn.LeakyReLU(negative_slope=leakyrelu_neg_slope)
        elif nonlin == 'relu':
            nonlin_layer = nn.ReLU()
        elif nonlin == 'silu':
            nonlin_layer = nn.SiLU()
        else:
            raise ValueError(f"Unknown nonlin: {nonlin}")
        
        self.mlp_h_mean_ROT = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nonlin_layer,
        )
    
    def _generate_keypoints(self, feats1, coors1, feats2, coors2, mask1=None, mask2=None):
        """
        Generate K keypoints for each protein using cross-attention.
        
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
        K = self.num_att_heads
        
        # Compute mean features μ(φ(H₁)) and μ(φ(H₂))
        # Apply transformation φ first
        feats1_transformed = self.mlp_h_mean_ROT(feats1)  # (B, N1, d)
        feats2_transformed = self.mlp_h_mean_ROT(feats2)  # (B, N2, d)
        
        # Compute mean
        mask1_expanded = mask1.unsqueeze(-1)  # (B, N1, 1)
        feats1_masked = feats1_transformed * mask1_expanded
        H1_mean = feats1_masked.sum(dim=1, keepdim=True) / (mask1.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6)  # (B, 1, d)
        
        mask2_expanded = mask2.unsqueeze(-1)  # (B, N2, 1)
        feats2_masked = feats2_transformed * mask2_expanded
        H2_mean = feats2_masked.sum(dim=1, keepdim=True) / (mask2.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6)  # (B, 1, d)
        
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
    
    def _ellipsoid_params(self, Y):
        """
        Compute ellipsoid parameters (mu and covariance) from keypoints.
        
        Args:
            Y: (B, K, 3) - keypoints for one protein
        
        Returns:
            mu: (B, 3) - mean (center) of keypoints
            cov: (B, 3, 3) - covariance matrix of keypoints
        """
        B, K, _ = Y.shape
        mu = Y.mean(dim=1)  # (B, 3) - mean of keypoints
        
        # Center keypoints: subtract mean from each keypoint
        diff = Y - mu.unsqueeze(1)  # (B, K, 3)
        
        # Compute covariance matrix: cov = (1/K) * sum_k (y_k - mu) @ (y_k - mu)^T
        # Using einsum: sum over k dimension, outer product of diff vectors
        cov = einsum(diff, diff, 'b k i, b k j -> b i j') / K  # (B, 3, 3)
        
        # Add small regularization for numerical stability (ensure positive definite)
        cov = cov + 1e-5 * torch.eye(3, device=cov.device).unsqueeze(0)  # (B, 3, 3)

        return mu, cov

    def forward(self, p1_residue, p1_coords, p2_residue, p2_coords, p1_mask=None, p2_mask=None):
        """
        Forward pass.
        
        Args:
            p1_residue: (B, N1) - residue type indices for protein 1
            p1_coords: (B, N1, 3) - coordinates for protein 1
            p2_residue: (B, N2) - residue type indices for protein 2
            p2_coords: (B, N2, 3) - coordinates for protein 2
            p1_mask: (B, N1) - mask for protein 1 (optional)
            p2_mask: (B, N2) - mask for protein 2 (optional)
        
        Returns:
            Dictionary containing:
            - Y1: (B, K, 3) - keypoints for protein 1
            - Y2: (B, K, 3) - keypoints for protein 2
            - attn1: (B, K, N1) - attention weights for protein 1
            - attn2: (B, K, N2) - attention weights for protein 2
            - p1_feats: (B, N1, dim) - features for protein 1
            - p2_feats: (B, N2, dim) - features for protein 2
        """
        B, _, _ = p1_coords.shape
        
        # feats are invariant to the Rotation and Translation.
        p1_feats_out, _ = self.egnn_p1(p1_residue, p1_coords, mask=p1_mask)
        p2_feats_out, _ = self.egnn_p2(p2_residue, p2_coords, mask=p2_mask)
        
        # Generate K keypoints for each protein using cross-attention
        Y1, Y2, attn1, attn2 = self._generate_keypoints(
            p1_feats_out, p1_coords,
            p2_feats_out, p2_coords,
            p1_mask, p2_mask
        )

        mu1, cov1 = self._ellipsoid_params(Y1)
        mu2, cov2 = self._ellipsoid_params(Y2)

        i1_params = torch.cat([mu1, cov1.reshape(B, -1)], dim=-1) # (B, 12)
        i2_params = torch.cat([mu2, cov2.reshape(B, -1)], dim=-1) # (B, 12)

        return {
            'Y1': Y1,  # (B, K, 3)
            'Y2': Y2,  # (B, K, 3)
            'attn1': attn1,  # (B, K, N1)
            'attn2': attn2,  # (B, K, N2)
            'p1_feats': p1_feats_out,
            'p2_feats': p2_feats_out,
            'mu1': mu1,
            'cov1': cov1,
            'mu2': mu2,
            'cov2': cov2,
            'i1_params': i1_params,
            'i2_params': i2_params,
        }

class InterfaceNet(nn.Module):
    def __init__(
        self,
        num_tokens: int = 22,
        dim: int = 128,
        depth: int = 4,
        dropout: float = 0.0,
        nonlin: str = 'silu',
        cross_msgs: bool = True,
        layer_norm: str = 'LN',
        layer_norm_coors: str = '0',
        final_h_layer_norm: str = 'LN',
        use_dist_in_layers: bool = True,
        skip_weight_h: float = 0.75,
        x_connection_init: float = 0.5,
        leakyrelu_neg_slope: float = 0.1,
        num_dist_basis: int = 15,
        dist_sigma_base: float = 1.5,
        shared_layers: bool = False,
        num_nearest_neighbors: int = 10,  # number of nearest neighbors for KNN graph
        cutoff: float = 20.0,  # distance cutoff for neighbor selection
    ):
        super().__init__()
        
        # Residue embedding layer
        self.residue_emb = nn.Embedding(num_tokens, dim)
        
        # IEGMN network for processing two proteins with interaction
        self.iegmn = IEGMN(
            dim=dim,
            depth=depth,
            edge_dim=0,
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
            shared_layers=shared_layers,
            num_nearest_neighbors=num_nearest_neighbors,
            cutoff=cutoff,
        )

    def forward(
        self,
        p1_residue, p1_coords,
        p2_residue, p2_coords,
        p1_coords_N, p1_coords_CA, p1_coords_C,
        p2_coords_N, p2_coords_CA, p2_coords_C,
        p1_mask=None, p2_mask=None,
    ):
        """
        Forward pass of InterfaceNet.
        
        Args:
            p1_residue: (B, N1) - residue type indices for protein 1
            p1_coords: (B, N1, 3) - CA coordinates for protein 1
            p2_residue: (B, N2) - residue type indices for protein 2
            p2_coords: (B, N2, 3) - CA coordinates for protein 2
            p1_coords_N: (B, N1, 3) - N atom coordinates for protein 1
            p1_coords_CA: (B, N1, 3) - CA atom coordinates for protein 1
            p1_coords_C: (B, N1, 3) - C atom coordinates for protein 1
            p2_coords_N: (B, N2, 3) - N atom coordinates for protein 2
            p2_coords_CA: (B, N2, 3) - CA atom coordinates for protein 2
            p2_coords_C: (B, N2, 3) - C atom coordinates for protein 2
            p1_mask: (B, N1) - mask for protein 1 (optional)
            p2_mask: (B, N2) - mask for protein 2 (optional)
        
        Returns:
            Dictionary containing:
            - p1_feats: (B, N1, dim) - updated features for protein 1
            - p2_feats: (B, N2, dim) - updated features for protein 2
            - p1_coords: (B, N1, 3) - updated coordinates for protein 1
            - p2_coords: (B, N2, 3) - updated coordinates for protein 2
            - R: (B, 3, 3) - rotation matrices
            - t: (B, 3) - translation vectors
            - Y1: (B, K, 3) - keypoints for protein 1
            - Y2: (B, K, 3) - keypoints for protein 2
        """
        # Embed residue types to features
        p1_feats = self.residue_emb(p1_residue)  # (B, N1, dim)
        p2_feats = self.residue_emb(p2_residue)  # (B, N2, dim)
        
        # Pass through IEGMN network
        p1_feats_out, p1_coords_out, p2_feats_out, p2_coords_out, R, t, Y1, Y2 = self.iegmn(
            feats1=p1_feats,
            coors1=p1_coords,
            feats2=p2_feats,
            coors2=p2_coords,
            coors_N1=p1_coords_N,
            coors_CA1=p1_coords_CA,
            coors_C1=p1_coords_C,
            coors_N2=p2_coords_N,
            coors_CA2=p2_coords_CA,
            coors_C2=p2_coords_C,
            mask1=p1_mask,
            mask2=p2_mask,
        )
        
        # Compute ellipsoid parameters directly from keypoints Y1 and Y2
        # mu is the mean of keypoints (center of ellipsoid)
        i1_mu_pred = Y1.mean(dim=1)  # (B, 3)
        i2_mu_pred = Y2.mean(dim=1)  # (B, 3)
        
        # sigma is the covariance matrix of keypoints
        B, K, _ = Y1.shape
        eps = 1e-5
        
        # Center keypoints
        Y1_centered = Y1 - i1_mu_pred.unsqueeze(1)  # (B, K, 3)
        Y2_centered = Y2 - i2_mu_pred.unsqueeze(1)  # (B, K, 3)
        
        # Compute covariance matrix: sigma = (1/K) * Y_centered^T @ Y_centered
        i1_sigma_pred = einsum(Y1_centered, Y1_centered, 'b k d1, b k d2 -> b d1 d2') / K  # (B, 3, 3)
        i2_sigma_pred = einsum(Y2_centered, Y2_centered, 'b k d1, b k d2 -> b d1 d2') / K  # (B, 3, 3)
        
        # Add small regularization for numerical stability
        I = torch.eye(3, device=Y1.device).unsqueeze(0)  # (1, 3, 3)
        i1_sigma_pred = i1_sigma_pred + eps * I
        i2_sigma_pred = i2_sigma_pred + eps * I

        return {
            'p1_feats': p1_feats_out,
            'p2_feats': p2_feats_out,
            'p1_coords': p1_coords_out,
            'p2_coords': p2_coords_out,
            'R': R,
            't': t,
            'Y1': Y1,
            'Y2': Y2,
            'i1_mu_pred': i1_mu_pred,
            'i1_sigma_pred': i1_sigma_pred,
            'i2_mu_pred': i2_mu_pred,
            'i2_sigma_pred': i2_sigma_pred,
        }