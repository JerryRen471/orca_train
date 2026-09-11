from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .agent import Actor, Encoder


@dataclass(frozen=True)
class PixelStudentConfig:
    image_size: int = 84
    frame_stack: int = 3
    feature_dim: int = 64
    hidden_dim: int = 256
    action_dim: int = 16
    camera_name: str = "closeup"
    camera_fovy: float = 30.0

    def __post_init__(self):
        if self.image_size < 32 or min(
            self.frame_stack, self.feature_dim, self.hidden_dim, self.action_dim
        ) < 1:
            raise ValueError("Invalid student dimensions")
        if not np.isfinite(self.camera_fovy) or not 0 < self.camera_fovy < 180:
            raise ValueError("camera_fovy must be in (0, 180)")

    @property
    def pixel_shape(self):
        return (3 * self.frame_stack, self.image_size, self.image_size)


class PixelStudent(nn.Module):
    """Image-only actor; no joint state, goal coordinates, clock, or teacher at inference."""

    def __init__(self, config: PixelStudentConfig | None = None):
        super().__init__()
        self.config = config or PixelStudentConfig()
        self.encoder = Encoder(self.config.pixel_shape, self.config.feature_dim)
        self.actor = Actor(self.config.feature_dim, self.config.hidden_dim, self.config.action_dim)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != self.config.pixel_shape:
            raise ValueError(f"Expected batched pixels with shape {self.config.pixel_shape}")
        return self.actor(self.encoder(pixels))

    @torch.no_grad()
    def act(self, observation, step=0, eval_mode=True):
        if set(observation) != {"pixels"}:
            raise ValueError("Pixel students accept only the pixels observation")
        pixels = np.asarray(observation["pixels"])
        if pixels.dtype != np.uint8 or pixels.shape != self.config.pixel_shape:
            raise ValueError("Expected uint8 RGB frame stack matching the saved config")
        device = next(self.parameters()).device
        return self(torch.as_tensor(pixels, device=device).unsqueeze(0))[0].cpu().numpy()

    def save(self, path: str | Path, *, metadata: dict):
        # Inference bundles deliberately omit teachers, critics, and optimizer state.
        torch.save({"format": "orca-pixel-student-v1", "config": asdict(self.config),
                    "policy": self.state_dict(), "metadata": metadata}, path)

    @classmethod
    def load(cls, path: str | Path, device="cpu"):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("format") != "orca-pixel-student-v1":
            raise ValueError("Unsupported pixel student format")
        model = cls(PixelStudentConfig(**saved["config"]))
        model.load_state_dict(saved["policy"])
        return model.to(device).eval(), saved["metadata"]
