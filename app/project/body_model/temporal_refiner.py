from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .regions import REGION_NAMES
from .temporal_dataset import TEMPORAL_FEATURE_DIM


class TemporalRegionRefiner(nn.Module):
    """Causal transformer that denoises sequential body-region observations."""

    def __init__(
        self,
        *,
        num_regions: int = len(REGION_NAMES),
        input_dim: int = TEMPORAL_FEATURE_DIM,
        hidden_dim: int = 192,
        layers: int = 3,
        heads: int = 6,
        max_length: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.num_regions = int(num_regions)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_length = max(int(max_length), 2)
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.position = nn.Parameter(torch.zeros(1, self.max_length, self.hidden_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(heads),
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=max(int(layers), 1),
            norm=nn.LayerNorm(self.hidden_dim),
        )
        output_dim = self.num_regions
        self.center_2d_absolute = nn.Linear(self.hidden_dim, output_dim * 2)
        self.center_2d_delta = nn.Linear(self.hidden_dim, output_dim * 2)
        self.center_3d_absolute = nn.Linear(self.hidden_dim, output_dim * 3)
        self.center_3d_delta = nn.Linear(self.hidden_dim, output_dim * 3)
        self.presence = nn.Linear(self.hidden_dim, output_dim)
        # Preserve the frame-level observation at initialization. The temporal
        # model must earn every correction instead of degrading a calibrated
        # RGB-D baseline during early training.
        nn.init.zeros_(self.center_2d_delta.weight)
        nn.init.zeros_(self.center_2d_delta.bias)
        nn.init.zeros_(self.center_3d_delta.weight)
        nn.init.zeros_(self.center_3d_delta.bias)

    def _causal_mask(self, length: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.full((length, length), float("-inf"), device=device), diagonal=1
        )

    def forward(
        self,
        features: Tensor,
        observed_center_2d: Tensor,
        observed_valid_2d: Tensor,
        observed_center_3d: Tensor,
        observed_valid_3d: Tensor,
        *,
        causal: bool = True,
    ) -> dict[str, Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"features must have shape [B, T, {self.input_dim}]"
            )
        batch, length, _ = features.shape
        if length > self.max_length:
            raise ValueError(f"sequence length {length} exceeds {self.max_length}")
        hidden = self.input_projection(features) + self.position[:, :length]
        mask = self._causal_mask(length, features.device) if causal else None
        hidden = self.encoder(hidden, mask=mask)
        center_2d_absolute = torch.sigmoid(
            self.center_2d_absolute(hidden).view(batch, length, self.num_regions, 2)
        )
        center_2d_delta = 0.25 * torch.tanh(
            self.center_2d_delta(hidden).view(batch, length, self.num_regions, 2)
        )
        center_2d_residual = (
            observed_center_2d + center_2d_delta
        ).clamp(-0.5, 1.5)
        center_2d = torch.where(
            observed_valid_2d.unsqueeze(-1) > 0.5,
            center_2d_residual,
            center_2d_absolute,
        )
        center_3d_absolute = self.center_3d_absolute(
            hidden
        ).view(batch, length, self.num_regions, 3)
        center_3d_delta = 0.15 * torch.tanh(
            self.center_3d_delta(hidden).view(batch, length, self.num_regions, 3)
        )
        center_3d_residual = observed_center_3d + center_3d_delta
        center_3d = torch.where(
            observed_valid_3d.unsqueeze(-1) > 0.5,
            center_3d_residual,
            center_3d_absolute,
        )
        return {
            "center_2d": center_2d,
            "center_3d": center_3d,
            "presence_logits": self.presence(hidden),
            "hidden": hidden,
        }


def _masked_smooth_l1(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    per_item = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    weights = mask.to(per_item.dtype)
    return (per_item * weights).sum() / weights.sum().clamp_min(1.0)


def _masked_velocity_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    if prediction.shape[1] < 2:
        return prediction.new_zeros(())
    prediction_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    pair_mask = mask[:, 1:] * mask[:, :-1]
    return _masked_smooth_l1(prediction_delta, target_delta, pair_mask)


def temporal_region_loss(
    prediction: dict[str, Tensor],
    target: dict[str, Tensor],
) -> dict[str, Tensor]:
    visible = target["target_visible"].float()
    valid_3d = target["target_3d_valid"].float()
    center_2d = _masked_smooth_l1(
        prediction["center_2d"], target["target_center_2d"], visible
    )
    center_3d = _masked_smooth_l1(
        prediction["center_3d"], target["target_center_3d"], valid_3d
    )
    presence = F.binary_cross_entropy_with_logits(
        prediction["presence_logits"], visible
    )
    velocity_2d = _masked_velocity_loss(
        prediction["center_2d"], target["target_center_2d"], visible
    )
    velocity_3d = _masked_velocity_loss(
        prediction["center_3d"], target["target_center_3d"], valid_3d
    )
    total = center_2d + 0.5 * center_3d + 0.25 * presence
    total = total + 0.25 * velocity_2d + 0.10 * velocity_3d
    return {
        "total": total,
        "center_2d": center_2d,
        "center_3d": center_3d,
        "presence": presence,
        "velocity_2d": velocity_2d,
        "velocity_3d": velocity_3d,
    }
