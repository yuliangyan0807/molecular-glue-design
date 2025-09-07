from .egnn import EGNN_Network

import torch
import torch.nn as nn

class InterfaceModel(nn.Module):
    def __init__(
        self,
        num_tokens: int = 21,
        dim: int = 128,
        depth: int = 4,
        num_nearest_neighbors: int = 16,
        norm_coors: bool = True,
        coor_weights_clamp_value: float = 2.0,
        out_dim: int = 512 # hidden dimension of the interface representation.
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

        # Optional joint readout if representation is needed downstream
        self.readout = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, out_dim)
        )

        # Decoder for predicting mu and sigma of the ellipsoid
        # mu: (batch_size, 3), sigma: (batch_size, 3, 3).
        self.mu_sigma_decoder = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 12)  # 3 for mu, 9 for sigma (3x3 covariance matrix)
        )

    def forward(self, p1_residue, p1_coords, p2_residue, p2_coords, p1_mask=None, p2_mask=None):
        p1_feats_out, _ = self.egnn_p1(p1_residue, p1_coords, mask=p1_mask)
        p2_feats_out, _ = self.egnn_p2(p2_residue, p2_coords, mask=p2_mask)

        # A simple joint representation via mean pooling then concatenation
        p1_repr = (p1_feats_out if p1_mask is None else (p1_feats_out * p1_mask.unsqueeze(-1))).mean(dim=1)
        p2_repr = (p2_feats_out if p2_mask is None else (p2_feats_out * p2_mask.unsqueeze(-1))).mean(dim=1)
        combined = torch.cat([p1_repr, p2_repr], dim=-1)
        joint_representation = self.readout(combined) # (batch_size, out_dim)

        ellipsoid_params = self.mu_sigma_decoder(joint_representation) # (batch_size, 12)

        return {
            'ellipsoid_params': ellipsoid_params,
            'p1_feats': p1_feats_out,
            'p2_feats': p2_feats_out,
            'joint_representation': joint_representation,
        }
