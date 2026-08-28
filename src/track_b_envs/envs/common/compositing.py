"""GPU-side calibrated foreground composition over the real background."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from .constants import (
    BACKGROUND_IMAGE,
    CAMERA_INTRINSIC_MATRIX,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    POLICY_IMAGE_SIZE,
)


class RealBackgroundCompositor:
    """Warp Isaac camera output to measured intrinsics and composite it."""

    def __init__(self, device: str | torch.device) -> None:
        self.device = torch.device(device)
        background = np.asarray(Image.open(BACKGROUND_IMAGE).convert("RGB"), dtype=np.uint8).copy()
        if background.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
            raise RuntimeError(f"unexpected background shape: {background.shape}")
        self.background = torch.from_numpy(background).to(self.device)
        self.calibrated_intrinsic = torch.tensor(
            CAMERA_INTRINSIC_MATRIX, dtype=torch.float32, device=self.device
        ).reshape(3, 3)
        y, x = torch.meshgrid(
            torch.arange(IMAGE_HEIGHT, dtype=torch.float32, device=self.device),
            torch.arange(IMAGE_WIDTH, dtype=torch.float32, device=self.device),
            indexing="ij",
        )
        self._output_pixels = torch.stack(
            (x.reshape(-1), y.reshape(-1), torch.ones(x.numel(), device=self.device))
        )
    def _warp_grid(self, render_intrinsic: torch.Tensor, batch_size: int) -> torch.Tensor:
        render_intrinsic = torch.as_tensor(render_intrinsic, dtype=torch.float32, device=self.device)
        if render_intrinsic.ndim == 2:
            render_intrinsic = render_intrinsic.unsqueeze(0)
        if render_intrinsic.shape[0] == 1:
            render_intrinsic = render_intrinsic.expand(batch_size, -1, -1)
        if render_intrinsic.shape != (batch_size, 3, 3):
            raise ValueError(f"unexpected camera intrinsics: {tuple(render_intrinsic.shape)}")
        output_to_source = torch.linalg.inv(
            self.calibrated_intrinsic.expand(batch_size, -1, -1)
            @ torch.linalg.inv(render_intrinsic)
        )
        source = torch.bmm(output_to_source, self._output_pixels.expand(batch_size, -1, -1))
        source_xy = source[:, :2] / source[:, 2:3].clamp_min(1.0e-8)
        source_x = 2.0 * source_xy[:, 0] / (IMAGE_WIDTH - 1.0) - 1.0
        source_y = 2.0 * source_xy[:, 1] / (IMAGE_HEIGHT - 1.0) - 1.0
        return torch.stack((source_x, source_y), dim=-1).reshape(
            batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 2
        )

    def compose(self, rgb: torch.Tensor, semantic: torch.Tensor, render_intrinsic: torch.Tensor) -> torch.Tensor:
        """Return BHWC uint8 composite images at calibrated resolution."""

        if rgb.ndim != 4 or rgb.shape[1:3] != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError(f"expected BHWC RGB at 1280x720, got {tuple(rgb.shape)}")
        rgb = rgb[..., :3].to(self.device)
        if semantic.ndim == 4:
            semantic = semantic[..., 0]
        if semantic.shape != rgb.shape[:3]:
            raise ValueError(f"unexpected semantic shape: {tuple(semantic.shape)}")
        grid = self._warp_grid(render_intrinsic, rgb.shape[0])
        warped_rgb = functional.grid_sample(
            rgb.permute(0, 3, 1, 2).float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).permute(0, 2, 3, 1)
        warped_mask = functional.grid_sample(
            (semantic != 0)[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=True
        )[:, 0] > 0.5
        background = self.background.unsqueeze(0).expand(rgb.shape[0], -1, -1, -1)
        return torch.where(
            warped_mask[..., None], warped_rgb.round().clamp(0, 255).to(torch.uint8), background
        )

    @staticmethod
    def policy_view(composite: torch.Tensor) -> torch.Tensor:
        """Center-crop 16:9 RGB to square and return normalized BCHW."""

        left = (IMAGE_WIDTH - IMAGE_HEIGHT) // 2
        square = composite[:, :, left : left + IMAGE_HEIGHT, :]
        return functional.interpolate(
            square.permute(0, 3, 1, 2).float() / 255.0,
            size=POLICY_IMAGE_SIZE,
            mode="bilinear",
            align_corners=False,
        )
