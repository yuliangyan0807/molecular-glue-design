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
        out_dim: int = 512, # hidden dimension of the interface representation.
        attn_num_layers: int = 2,
        attn_num_heads: int = 8,
        attn_ffn_dim: int | None = None,
        attn_dropout: float = 0.1,
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

        # Separate readouts for p1 and p2 representations
        self.readout_p1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, out_dim)
        )
        self.readout_p2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, out_dim)
        )

        # Pairwise attention over the two interface representations (sequence length = 2)
        ffn_dim = attn_ffn_dim if attn_ffn_dim is not None else out_dim * 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=attn_num_heads,
            dim_feedforward=ffn_dim,
            dropout=attn_dropout,
            batch_first=True,
            activation='relu',
            norm_first=True,
        )
        self.pair_attention = nn.TransformerEncoder(encoder_layer, num_layers=attn_num_layers)

        # Decoders for predicting mu and sigma of the ellipsoid for p1 and p2
        # Each outputs 12 dims: 3 for mu, 9 for sigma (3x3 covariance matrix)
        self.mu_sigma_decoder_p1 = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 12)
        )
        self.mu_sigma_decoder_p2 = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 12)
        )

    def forward(self, p1_residue, p1_coords, p2_residue, p2_coords, p1_mask=None, p2_mask=None):
        p1_feats_out, _ = self.egnn_p1(p1_residue, p1_coords, mask=p1_mask)
        p2_feats_out, _ = self.egnn_p2(p2_residue, p2_coords, mask=p2_mask)

        # Mean pooling per chain to get chain-wise representations
        p1_repr = (p1_feats_out if p1_mask is None else (p1_feats_out * p1_mask.unsqueeze(-1))).mean(dim=1)
        p2_repr = (p2_feats_out if p2_mask is None else (p2_feats_out * p2_mask.unsqueeze(-1))).mean(dim=1)

        # Separate readouts to produce initial interface-level representations
        p1_representation = self.readout_p1(p1_repr)  # (batch_size, out_dim)
        p2_representation = self.readout_p2(p2_repr)  # (batch_size, out_dim)

        # Apply attention over the pair [p1_representation, p2_representation]
        pair_reprs = torch.stack([p1_representation, p2_representation], dim=1)  # (B, 2, out_dim)
        pair_reprs = self.pair_attention(pair_reprs)  # (B, 2, out_dim)
        p1_representation, p2_representation = pair_reprs[:, 0, :], pair_reprs[:, 1, :]

        i1_params = self.mu_sigma_decoder_p1(p1_representation) # (batch_size, 12)
        i2_params = self.mu_sigma_decoder_p2(p2_representation) # (batch_size, 12)

        return {
            'i1_params': i1_params,
            'i2_params': i2_params,
            'p1_feats': p1_feats_out,
            'p2_feats': p2_feats_out,
            'p1_representation': p1_representation,
            'p2_representation': p2_representation,
        }
