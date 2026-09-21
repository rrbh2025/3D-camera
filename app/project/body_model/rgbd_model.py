from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .regions import REGION_KEYPOINT_GROUPS, REGION_NAMES


class RGBDBodyRegionNet(nn.Module):
    """Trainable RGB-D/keypoint refinement head for the body-region schema.

    The model is intentionally small enough for the existing L20 environment.
    It uses a shared RGB-D encoder and learned region queries, so the output
    remains fixed and directly aligned with ``REGION_NAMES``.
    """

    def __init__(
        self,
        num_regions: int = len(REGION_NAMES),
        keypoints: int = 17,
        width: int = 32,
        use_metric_keypoints: bool = False,
        use_metric_geometry: bool = False,
        use_prior_residual: bool = False,
        use_prior_residual_depth_only: bool = False,
        use_metric_region_prior: bool = False,
    ):
        super().__init__()
        self.num_regions = int(num_regions)
        self.use_metric_keypoints = bool(use_metric_keypoints)
        self.use_metric_geometry = bool(use_metric_geometry)
        self.use_prior_residual = bool(use_prior_residual)
        self.use_prior_residual_depth_only = bool(use_prior_residual_depth_only)
        self.use_metric_region_prior = bool(use_metric_region_prior)
        if self.num_regions != len(REGION_NAMES):
            raise ValueError("the body-region schema has a fixed number of regions")
        self.image_encoder = nn.Sequential(
            # RGB-D plus normalized x/y coordinates. The coordinate channels
            # disambiguate anatomically adjacent regions with similar texture.
            nn.Conv2d(6, width, 3, stride=2, padding=1),
            nn.GroupNorm(max(1, width // 8), width),
            nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.GroupNorm(max(1, (width * 2) // 8), width * 2),
            nn.SiLU(),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),
            nn.GroupNorm(max(1, (width * 4) // 8), width * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.keypoint_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(keypoints * 3, width * 4),
            nn.SiLU(),
            nn.Linear(width * 4, width * 4),
        )
        if self.use_metric_keypoints:
            self.metric_keypoint_encoder = nn.Sequential(
                nn.Linear(keypoints * 4 + 4, width * 4),
                nn.SiLU(),
                nn.Linear(width * 4, width * 4),
            )
        feature_dim = width * 8
        self.image_feature_dim = width * 4
        self.region_queries = nn.Parameter(torch.randn(self.num_regions, feature_dim) * 0.02)
        self.region_embedding = nn.Parameter(torch.randn(self.num_regions, width * 2) * 0.02)
        self.local_projection = nn.Linear(self.image_feature_dim, width * 2)
        self.pose_projection = nn.Linear(keypoints * 3, width * 2)
        self.depth_projection = nn.Linear(2, width * 2)
        self.modality_projection = nn.Linear(2, width * 2)
        if self.use_metric_geometry:
            # Robust local camera-space statistics around each region prior.
            # RGB-only records keep this branch zero-valued.
            self.geometry_projection = nn.Sequential(
                nn.Linear(8, width * 2),
                nn.LayerNorm(width * 2),
                nn.SiLU(),
                nn.Linear(width * 2, width * 2),
            )
        if self.use_metric_region_prior:
            self.metric_region_projection = nn.Sequential(
                nn.Linear(4, width * 2),
                nn.LayerNorm(width * 2),
                nn.SiLU(),
                nn.Linear(width * 2, width * 2),
            )
        fusion_input_dim = feature_dim * 2 + width * 10
        if self.use_metric_geometry:
            fusion_input_dim += width * 2
        if self.use_metric_region_prior:
            fusion_input_dim += width * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
        )
        attention_heads = 4 if feature_dim % 4 == 0 else 1
        attention_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=attention_heads,
            dim_feedforward=feature_dim * 2,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.region_attention = nn.TransformerEncoder(
            attention_layer,
            num_layers=2,
            norm=nn.LayerNorm(feature_dim),
        )
        self.center_2d = nn.Linear(feature_dim, 2)
        self.center_delta = None
        if self.use_prior_residual:
            # Start from the deterministic keypoint prior and learn only a
            # bounded visual/depth correction for domain adaptation.
            self.center_delta = nn.Linear(feature_dim, 2)
            nn.init.zeros_(self.center_delta.weight)
            nn.init.zeros_(self.center_delta.bias)
        # Predict bbox center and positive width/height, then decode to xyxy.
        # This keeps the public target schema unchanged while guaranteeing
        # valid boxes during the first unstable training iterations.
        self.bbox_center = nn.Linear(feature_dim, 2)
        self.bbox_size = nn.Linear(feature_dim, 2)
        self.visible = nn.Linear(feature_dim, 1)
        self.metric_center_delta = None
        self.metric_center_gate = None
        if self.use_metric_region_prior:
            self.metric_center_delta = nn.Linear(feature_dim, 3)
            self.metric_center_gate = nn.Linear(feature_dim, 1)
            nn.init.zeros_(self.metric_center_delta.weight)
            nn.init.zeros_(self.metric_center_delta.bias)
            nn.init.zeros_(self.metric_center_gate.weight)
            nn.init.constant_(self.metric_center_gate.bias, 2.0)
            self.register_buffer(
                "metric_center_delta_scale",
                torch.tensor([0.20, 0.20, 0.30], dtype=torch.float32),
            )
        self.confidence = nn.Linear(feature_dim, 1)
        self.center_3d = nn.Linear(feature_dim, 3)
        self.region_presence = nn.Linear(feature_dim, 1)
        self.heatmap_projection = nn.Conv2d(self.image_feature_dim, feature_dim, 1)
        self.region_heatmap_head = nn.Sequential(
            nn.Conv2d(self.image_feature_dim, width * 2, 3, padding=1),
            nn.GroupNorm(max(1, (width * 2) // 8), width * 2),
            nn.SiLU(),
            nn.Conv2d(width * 2, self.num_regions, 1),
        )
        self.heatmap_prior_strength = nn.Parameter(torch.tensor(0.25))

    @staticmethod
    def _sample_local(features: Tensor, centers: Tensor) -> Tensor:
        """Sample one local feature vector per region with differentiable bilinear sampling."""
        # centers are [B, R, 2] in x/y normalized coordinates.
        grid = centers.mul(2.0).sub(1.0).unsqueeze(2)
        sampled = F.grid_sample(features, grid, mode="bilinear", align_corners=True)
        return sampled.squeeze(-1).transpose(1, 2)

    @staticmethod
    def _sample_local_depth(
        depth: Tensor,
        centers: Tensor,
        *,
        radius_px: float = 4.0,
        samples_per_axis: int = 5,
        min_depth_m: float = 0.2,
        max_depth_m: float = 10.0,
    ) -> tuple[Tensor, Tensor]:
        """Sample a robust median depth around normalized region centers."""
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError("depth must have shape [B, 1, H, W]")
        if centers.ndim != 3 or centers.shape[-1] != 2:
            raise ValueError("centers must have shape [B, R, 2]")
        _, _, height, width = depth.shape
        samples = max(int(samples_per_axis), 1)
        offsets_px = torch.linspace(
            -float(radius_px),
            float(radius_px),
            samples,
            device=depth.device,
            dtype=depth.dtype,
        )
        offset_y, offset_x = torch.meshgrid(
            offsets_px, offsets_px, indexing="ij"
        )
        offsets = torch.stack(
            [
                offset_x.reshape(-1) * 2.0 / max(width - 1, 1),
                offset_y.reshape(-1) * 2.0 / max(height - 1, 1),
            ],
            dim=-1,
        )
        grid = centers.mul(2.0).sub(1.0).unsqueeze(2) + offsets.view(
            1, 1, -1, 2
        )
        sampled = F.grid_sample(
            depth,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0]
        valid = (
            torch.isfinite(sampled)
            & (sampled >= float(min_depth_m))
            & (sampled <= float(max_depth_m))
        )
        count = valid.sum(dim=-1)
        ordered = torch.where(
            valid,
            sampled,
            torch.full_like(sampled, torch.inf),
        ).sort(dim=-1).values
        median_index = ((count - 1).clamp_min(0) // 2).unsqueeze(-1)
        median = ordered.gather(-1, median_index).squeeze(-1)
        median = torch.where(count > 0, median, torch.zeros_like(median))
        return median, count > 0

    @staticmethod
    def _sample_local_geometry(
        depth: Tensor,
        centers: Tensor,
        camera_intrinsics: Tensor,
        *,
        radius_px: float = 5.0,
        samples_per_axis: int = 5,
        min_depth_m: float = 0.2,
        max_depth_m: float = 10.0,
        max_background_delta_m: float = 0.50,
    ) -> tuple[Tensor, Tensor]:
        """Sample robust local camera-space geometry around region priors."""
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError("depth must have shape [B, 1, H, W]")
        if centers.ndim != 3 or centers.shape[-1] != 2:
            raise ValueError("centers must have shape [B, R, 2]")
        if camera_intrinsics.ndim != 2 or camera_intrinsics.shape[1] != 5:
            raise ValueError("camera_intrinsics must have shape [B, 5]")
        batch, _, height, width = depth.shape
        samples = max(int(samples_per_axis), 1)
        offsets_px = torch.linspace(
            -float(radius_px),
            float(radius_px),
            samples,
            device=depth.device,
            dtype=depth.dtype,
        )
        offset_y, offset_x = torch.meshgrid(
            offsets_px, offsets_px, indexing="ij"
        )
        offsets = torch.stack(
            [
                offset_x.reshape(-1) * 2.0 / max(width - 1, 1),
                offset_y.reshape(-1) * 2.0 / max(height - 1, 1),
            ],
            dim=-1,
        )
        grid = centers.mul(2.0).sub(1.0).unsqueeze(2) + offsets.view(
            1, 1, -1, 2
        )
        sampled_z = F.grid_sample(
            depth,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0]
        calibrated = camera_intrinsics[:, 4].reshape(batch, 1, 1) > 0.5
        valid = (
            torch.isfinite(sampled_z)
            & (sampled_z >= float(min_depth_m))
            & (sampled_z <= float(max_depth_m))
            & calibrated
        )
        count = valid.sum(dim=-1)
        ordered = torch.where(
            valid,
            sampled_z,
            torch.full_like(sampled_z, torch.inf),
        ).sort(dim=-1).values
        median_index = ((count - 1).clamp_min(0) // 2).unsqueeze(-1)
        median_z = ordered.gather(-1, median_index).squeeze(-1)
        median_z = torch.where(count > 0, median_z, torch.zeros_like(median_z))

        # Reject a nearby foreground/background mixture around the prior.
        close = valid & (
            (sampled_z - median_z.unsqueeze(-1)).abs()
            <= float(max_background_delta_m)
        )
        close_count = close.sum(dim=-1)
        use_mask = torch.where(
            (close_count > 0).unsqueeze(-1), close, valid
        )
        weights = use_mask.to(depth.dtype)
        denominator = weights.sum(dim=-1).clamp_min(1.0)
        u = centers[..., 0].unsqueeze(-1) + offsets[:, 0].view(1, 1, -1) * 0.5
        v = centers[..., 1].unsqueeze(-1) + offsets[:, 1].view(1, 1, -1) * 0.5
        fx = camera_intrinsics[:, 0].reshape(batch, 1, 1).clamp_min(1e-6)
        fy = camera_intrinsics[:, 1].reshape(batch, 1, 1).clamp_min(1e-6)
        cx = camera_intrinsics[:, 2].reshape(batch, 1, 1)
        cy = camera_intrinsics[:, 3].reshape(batch, 1, 1)
        x = (u - cx) * sampled_z / fx
        y = (v - cy) * sampled_z / fy

        def _masked_mean(value: Tensor) -> Tensor:
            return (value * weights).sum(dim=-1) / denominator

        mean_x = _masked_mean(x)
        mean_y = _masked_mean(y)
        mean_z = _masked_mean(sampled_z)
        spread_x = torch.sqrt(
            _masked_mean((x - mean_x.unsqueeze(-1)).square()) + 1e-6
        )
        spread_y = torch.sqrt(
            _masked_mean((y - mean_y.unsqueeze(-1)).square()) + 1e-6
        )
        spread_z = torch.sqrt(
            _masked_mean((sampled_z - mean_z.unsqueeze(-1)).square()) + 1e-6
        )
        valid_fraction = count.to(depth.dtype) / float(max(samples * samples, 1))
        geometry = torch.stack(
            [
                mean_x,
                mean_y,
                mean_z,
                torch.log1p(mean_z.clamp_min(0.0)),
                valid_fraction,
                spread_x,
                spread_y,
                spread_z,
            ],
            dim=-1,
        )
        geometry = torch.nan_to_num(geometry, nan=0.0, posinf=10.0, neginf=-10.0)
        geometry = geometry.clamp(-10.0, 10.0)
        return geometry, count > 0

    @staticmethod
    def _gaussian_priors(centers: Tensor, height: int, width: int) -> Tensor:
        ys = (torch.arange(height, device=centers.device, dtype=centers.dtype) + 0.5) / height
        xs = (torch.arange(width, device=centers.device, dtype=centers.dtype) + 0.5) / width
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        dx = xx.view(1, 1, height, width) - centers[..., 0, None, None]
        dy = yy.view(1, 1, height, width) - centers[..., 1, None, None]
        return torch.exp(-(dx.square() + dy.square()) / (2.0 * 0.12**2))

    @staticmethod
    def _soft_argmax_2d(logits: Tensor) -> Tensor:
        batch, regions, height, width = logits.shape
        probability = torch.softmax(logits.flatten(2), dim=-1)
        ys = (torch.arange(height, device=logits.device, dtype=logits.dtype) + 0.5) / height
        xs = (torch.arange(width, device=logits.device, dtype=logits.dtype) + 0.5) / width
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coordinates = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        return torch.matmul(probability, coordinates)

    @staticmethod
    def _camera_project(
        centers: Tensor,
        depth_z: Tensor,
        camera_intrinsics: Tensor,
    ) -> Tensor:
        """Back-project normalized image centers using normalized intrinsics."""
        fx = camera_intrinsics[:, 0, None].clamp_min(1e-6)
        fy = camera_intrinsics[:, 1, None].clamp_min(1e-6)
        cx = camera_intrinsics[:, 2, None]
        cy = camera_intrinsics[:, 3, None]
        x = (centers[..., 0] - cx) * depth_z / fx
        y = (centers[..., 1] - cy) * depth_z / fy
        return torch.stack([x, y, depth_z], dim=-1)

    def forward(
        self,
        rgbd: Tensor,
        keypoints: Tensor,
        modality_mask: Tensor | None = None,
        camera_intrinsics: Tensor | None = None,
        metric_keypoints: Tensor | None = None,
        metric_context: Tensor | None = None,
        metric_available: Tensor | None = None,
        metric_region_centers: Tensor | None = None,
        metric_region_available: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if rgbd.ndim != 4 or rgbd.shape[1] != 4:
            raise ValueError("rgbd must have shape [B, 4, H, W] (RGB plus depth)")
        if keypoints.ndim != 3 or keypoints.shape[2] != 3:
            raise ValueError("keypoints must have shape [B, 17, 3]")
        batch, _, height, width = rgbd.shape
        if modality_mask is None:
            depth_present = (rgbd[:, 3].abs().flatten(1).sum(dim=1) > 0).to(rgbd.dtype)
            modality_mask = torch.stack([torch.ones_like(depth_present), depth_present], dim=1)
        if modality_mask.shape != (batch, 2):
            raise ValueError("modality_mask must have shape [B, 2] for RGB/depth")
        modality_mask = modality_mask.to(device=rgbd.device, dtype=rgbd.dtype)
        if camera_intrinsics is not None:
            if camera_intrinsics.shape != (batch, 5):
                raise ValueError("camera_intrinsics must have shape [B, 5]")
            camera_intrinsics = camera_intrinsics.to(
                device=rgbd.device, dtype=rgbd.dtype
            )
        x_coords = torch.linspace(-1.0, 1.0, width, device=rgbd.device, dtype=rgbd.dtype)
        y_coords = torch.linspace(-1.0, 1.0, height, device=rgbd.device, dtype=rgbd.dtype)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
        coords = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
        image_map = self.image_encoder[:-1](torch.cat([rgbd, coords], dim=1))
        image_features = self.image_encoder[-1](image_map).flatten(1)
        pose_features = self.keypoint_encoder(keypoints)
        if self.use_metric_keypoints:
            if metric_keypoints is None:
                metric_keypoints = torch.zeros(
                    (batch, keypoints.shape[1], 4),
                    device=rgbd.device,
                    dtype=rgbd.dtype,
                )
            if metric_context is None:
                metric_context = torch.zeros(
                    (batch, 4), device=rgbd.device, dtype=rgbd.dtype
                )
            if metric_available is None:
                metric_available = torch.zeros(
                    (batch,), device=rgbd.device, dtype=rgbd.dtype
                )
            if metric_keypoints.shape != (batch, keypoints.shape[1], 4):
                raise ValueError("metric_keypoints must have shape [B, 17, 4]")
            if metric_context.shape != (batch, 4):
                raise ValueError("metric_context must have shape [B, 4]")
            metric_available = metric_available.to(
                device=rgbd.device, dtype=rgbd.dtype
            ).reshape(batch, 1)
            metric_features = self.metric_keypoint_encoder(
                torch.cat(
                    [
                        metric_keypoints.to(device=rgbd.device, dtype=rgbd.dtype).flatten(1),
                        metric_context.to(device=rgbd.device, dtype=rgbd.dtype),
                    ],
                    dim=1,
                )
            )
            pose_features = pose_features + metric_features * metric_available
        metric_region_centers_input = torch.zeros(
            (batch, self.num_regions, 3),
            device=rgbd.device,
            dtype=rgbd.dtype,
        )
        metric_region_available_input = torch.zeros(
            (batch, self.num_regions),
            device=rgbd.device,
            dtype=rgbd.dtype,
        )
        if self.use_metric_region_prior:
            if metric_region_centers is not None:
                metric_region_centers_input = metric_region_centers.to(
                    device=rgbd.device, dtype=rgbd.dtype
                )
            if metric_region_available is not None:
                metric_region_available_input = metric_region_available.to(
                    device=rgbd.device, dtype=rgbd.dtype
                )
            if metric_region_centers_input.shape != (batch, self.num_regions, 3):
                raise ValueError(
                    "metric_region_centers must have shape [B, R, 3]"
                )
            if metric_region_available_input.shape != (batch, self.num_regions):
                raise ValueError(
                    "metric_region_available must have shape [B, R]"
                )
            finite_prior = torch.isfinite(metric_region_centers_input).all(dim=-1)
            metric_region_available_input = (
                metric_region_available_input.clamp(0.0, 1.0)
                * finite_prior.to(rgbd.dtype)
            )
            metric_region_centers_input = torch.nan_to_num(
                metric_region_centers_input, nan=0.0, posinf=0.0, neginf=0.0
            )
        pose_flat = keypoints.flatten(1)
        prior_centers = keypoints.new_zeros((keypoints.shape[0], self.num_regions, 2))
        for index, name in enumerate(REGION_NAMES):
            indices = REGION_KEYPOINT_GROUPS[name]
            xy = keypoints[:, indices, :2]
            weights = keypoints[:, indices, 2:3].clamp(0.0, 1.0) + 1e-4
            prior_centers[:, index] = (xy * weights).sum(dim=1) / weights.sum(dim=1)
        prior_centers = prior_centers.clamp(0.0, 1.0)
        region_geometry = keypoints.new_zeros((batch, self.num_regions, 8))
        region_geometry_valid = torch.zeros(
            (batch, self.num_regions),
            device=rgbd.device,
            dtype=torch.bool,
        )
        geometry_features = None
        if self.use_metric_geometry:
            geometry_intrinsics = (
                camera_intrinsics
                if camera_intrinsics is not None
                else keypoints.new_zeros((batch, 5))
            )
            region_geometry, region_geometry_valid = self._sample_local_geometry(
                rgbd[:, 3:4],
                prior_centers,
                geometry_intrinsics,
            )
            geometry_features = self.geometry_projection(region_geometry)
            # Do not let the projection bias turn missing calibration/depth
            # into a learned geometry signal on RGB-only records.
            geometry_features = geometry_features * region_geometry_valid.unsqueeze(-1).to(
                geometry_features.dtype
            )
        local = self._sample_local(image_map, prior_centers)
        local = self.local_projection(local)
        sampled_depth = self._sample_local(rgbd[:, 3:4], prior_centers)
        sampled_depth_valid = (sampled_depth > 0).to(sampled_depth.dtype)
        depth_features = self.depth_projection(
            torch.cat([torch.log1p(sampled_depth.clamp_min(0.0)), sampled_depth_valid], dim=-1)
        )
        pose_global = self.pose_projection(pose_flat).unsqueeze(1).expand(-1, self.num_regions, -1)
        modality_features = self.modality_projection(modality_mask)
        modality_features = modality_features.unsqueeze(1).expand(-1, self.num_regions, -1)
        queries = self.region_queries.unsqueeze(0).expand(rgbd.shape[0], -1, -1)
        region_embedding = self.region_embedding.unsqueeze(0).expand(rgbd.shape[0], -1, -1)
        global_features = torch.cat([image_features, pose_features], dim=1)
        global_features = global_features.unsqueeze(1).expand(-1, self.num_regions, -1)
        fusion_inputs = [
            queries,
            global_features,
            local,
            pose_global,
            region_embedding,
            depth_features,
            modality_features,
        ]
        if geometry_features is not None:
            fusion_inputs.append(geometry_features)
        if self.use_metric_region_prior:
            metric_region_features = self.metric_region_projection(
                torch.cat(
                    [
                        metric_region_centers_input,
                        metric_region_available_input.unsqueeze(-1),
                    ],
                    dim=-1,
                )
            )
            metric_region_features = (
                metric_region_features
                * metric_region_available_input.unsqueeze(-1)
            )
            fusion_inputs.append(metric_region_features)
        fused = self.fusion(torch.cat(fusion_inputs, dim=2))
        fused = self.region_attention(fused)
        dense_features = self.heatmap_projection(image_map)
        query_heatmaps = torch.einsum("brd,bdhw->brhw", fused, dense_features)
        query_heatmaps = query_heatmaps / (dense_features.shape[1] ** 0.5)
        region_heatmap_logits = self.region_heatmap_head(image_map) + query_heatmaps
        prior_heatmaps = self._gaussian_priors(
            prior_centers,
            region_heatmap_logits.shape[-2],
            region_heatmap_logits.shape[-1],
        )
        prior_logits = torch.logit(prior_heatmaps.clamp(1e-4, 1.0 - 1e-4))
        region_heatmap_logits = (
            region_heatmap_logits + self.heatmap_prior_strength * prior_logits
        )
        bbox_center = torch.sigmoid(self.bbox_center(fused))
        bbox_size = torch.sigmoid(self.bbox_size(fused))
        bbox_half_size = bbox_size * 0.5
        bbox_2d = torch.cat(
            [bbox_center - bbox_half_size, bbox_center + bbox_half_size],
            dim=-1,
        ).clamp(0.0, 1.0)
        absolute_center = torch.sigmoid(self.center_2d(fused))
        if self.use_prior_residual:
            if self.center_delta is None:
                raise RuntimeError("prior residual head was not initialized")
            residual_center = (
                prior_centers + 0.25 * torch.tanh(self.center_delta(fused))
            ).clamp(0.0, 1.0)
            if self.use_prior_residual_depth_only:
                depth_available = (modality_mask[:, 1] > 0.5).view(batch, 1, 1)
                center_2d = torch.where(
                    depth_available, residual_center, absolute_center
                )
            else:
                center_2d = residual_center
        else:
            center_2d = absolute_center
        center_3d_raw = self.center_3d(fused)
        center_3d = center_3d_raw
        metric_region_valid = metric_region_available_input > 0.5
        if self.use_metric_region_prior:
            if self.metric_center_delta is None or self.metric_center_gate is None:
                raise RuntimeError("metric region prior heads were not initialized")
            metric_delta = torch.tanh(self.metric_center_delta(fused))
            metric_delta = metric_delta * self.metric_center_delta_scale.to(
                device=rgbd.device, dtype=rgbd.dtype
            )
            metric_prior_prediction = metric_region_centers_input + metric_delta
            metric_prior_weight = torch.sigmoid(self.metric_center_gate(fused))
            metric_prior_prediction = (
                metric_prior_weight * metric_prior_prediction
                + (1.0 - metric_prior_weight) * center_3d_raw
            )
            center_3d = torch.where(
                metric_region_valid.unsqueeze(-1),
                metric_prior_prediction,
                center_3d,
            )
        center_depth_m, center_depth_valid = self._sample_local_depth(
            rgbd[:, 3:4], center_2d
        )
        center_depth_used = torch.zeros_like(center_depth_valid)
        if camera_intrinsics is not None:
            network_projected = self._camera_project(
                center_2d,
                center_3d_raw[..., 2],
                camera_intrinsics,
            )
            calibrated = camera_intrinsics[:, 4].view(batch, 1, 1) > 0.5
            calibrated_prediction = torch.where(
                metric_region_valid.unsqueeze(-1),
                center_3d,
                network_projected,
            )
            center_3d = torch.where(
                calibrated, calibrated_prediction, center_3d
            )
            center_depth_used = (
                center_depth_valid
                & (modality_mask[:, 1:2] > 0.5)
                & calibrated.squeeze(-1)
                & ~metric_region_valid
            )
            measured_projected = self._camera_project(
                center_2d,
                center_depth_m,
                camera_intrinsics,
            )
            center_3d = torch.where(
                center_depth_used.unsqueeze(-1), measured_projected, center_3d
            )
        return {
            "center_2d": center_2d,
            "bbox_2d": bbox_2d,
            "visible_logits": self.visible(fused).squeeze(-1),
            "confidence_logits": self.confidence(fused).squeeze(-1),
            "center_3d": center_3d,
            "center_3d_raw": center_3d_raw,
            "metric_region_prior": metric_region_centers_input,
            "metric_region_prior_valid": metric_region_valid,
            "center_depth_m": center_depth_m,
            "center_depth_valid": center_depth_valid,
            "center_depth_used": center_depth_used,
            "region_presence_logits": self.region_presence(fused).squeeze(-1),
            "region_priors": prior_centers,
            "region_geometry": region_geometry,
            "region_geometry_valid": region_geometry_valid,
            "region_heatmap_logits": region_heatmap_logits,
            "heatmap_center_2d": self._soft_argmax_2d(region_heatmap_logits),
        }


def rgbd_region_loss(prediction: dict[str, Tensor], target: dict[str, Tensor]) -> dict[str, Tensor]:
    """Compute masked multi-task losses for a training batch."""
    visible = target["visible"].float()
    center_loss = torch.nn.functional.smooth_l1_loss(prediction["center_2d"], target["center_2d"], reduction="none").mean(-1)
    bbox_loss = torch.nn.functional.smooth_l1_loss(prediction["bbox_2d"], target["bbox_2d"], reduction="none").mean(-1)
    visible_loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction["visible_logits"], visible)
    confidence_target = target.get("confidence", visible).float().clamp(0.0, 1.0)
    confidence_loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction["confidence_logits"], confidence_target)
    has_3d = target.get("has_3d", torch.zeros_like(visible)).float()
    center_3d_loss = torch.nn.functional.smooth_l1_loss(prediction["center_3d"], target["center_3d"], reduction="none").mean(-1)
    center_loss = (center_loss * visible).sum() / visible.sum().clamp_min(1.0)
    bbox_loss = (bbox_loss * visible).sum() / visible.sum().clamp_min(1.0)
    center_3d_loss = (center_3d_loss * has_3d).sum() / has_3d.sum().clamp_min(1.0)
    prior_loss = torch.zeros_like(center_loss)
    if "region_priors" in prediction and "center_2d" in target:
        prior_per_region = F.smooth_l1_loss(
            prediction["region_priors"], target["center_2d"], reduction="none"
        ).mean(-1)
        prior_loss = (prior_per_region * visible).sum() / visible.sum().clamp_min(1.0)
    presence_loss = F.binary_cross_entropy_with_logits(
        prediction["region_presence_logits"], visible
    )
    heatmap_focal_loss = torch.zeros_like(center_loss)
    heatmap_dice_loss = torch.zeros_like(center_loss)
    heatmap_center_loss = torch.zeros_like(center_loss)
    if "region_heatmaps" in target and "region_heatmap_logits" in prediction:
        heatmap_target = target["region_heatmaps"].float()
        heatmap_logits = prediction["region_heatmap_logits"]
        if heatmap_target.shape[-2:] != heatmap_logits.shape[-2:]:
            heatmap_target = F.interpolate(
                heatmap_target,
                size=heatmap_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        probability = torch.sigmoid(heatmap_logits)
        heatmap_ce = F.binary_cross_entropy_with_logits(
            heatmap_logits, heatmap_target, reduction="none"
        )
        heatmap_focal_loss = (
            heatmap_ce
            * (probability - heatmap_target).abs().square()
            * (1.0 + 4.0 * heatmap_target)
        ).mean()
        intersection = (probability * heatmap_target).sum(dim=(-2, -1))
        denominator = probability.sum(dim=(-2, -1)) + heatmap_target.sum(dim=(-2, -1))
        dice_per_region = 1.0 - (2.0 * intersection + 1e-5) / (denominator + 1e-5)
        heatmap_dice_loss = (dice_per_region * visible).sum() / visible.sum().clamp_min(1.0)
        heatmap_center_per_region = F.smooth_l1_loss(
            prediction["heatmap_center_2d"],
            target["center_2d"],
            reduction="none",
        ).mean(-1)
        heatmap_center_loss = (
            heatmap_center_per_region * visible
        ).sum() / visible.sum().clamp_min(1.0)
    total = (
        center_loss
        + bbox_loss
        + visible_loss
        + 0.5 * confidence_loss
        + 0.5 * center_3d_loss
        + 0.25 * presence_loss
        + 0.1 * prior_loss
        + 0.5 * heatmap_focal_loss
        + 0.25 * heatmap_dice_loss
        + 0.25 * heatmap_center_loss
    )
    return {
        "total": total,
        "center_2d": center_loss,
        "bbox_2d": bbox_loss,
        "visible": visible_loss,
        "confidence": confidence_loss,
        "center_3d": center_3d_loss,
        "presence": presence_loss,
        "prior": prior_loss,
        "heatmap_focal": heatmap_focal_loss,
        "heatmap_dice": heatmap_dice_loss,
        "heatmap_center": heatmap_center_loss,
    }
