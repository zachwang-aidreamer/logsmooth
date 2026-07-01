#
# LogSmooth — configuration subclass.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quark.torch.quantization.config.config import OnlineRotationConfig, RotationConfig


@dataclass
class LogSmoothRotationConfig(RotationConfig):
    """
    RotationConfig extended with LogSmooth power-of-two smoothing options.

    Adds a static weight-side per-(block-)channel scale (``r1_weight_log_smooth``)
    fused into rotated weights and carried on the wrapper as ``post_rotation_scale``,
    and a dynamic per-channel activation sidecar (``r1_activation_log_smooth``).

    Uses a distinct ``name`` so Quark's algorithm/config dispatch routes to the
    LogSmooth processor without shadowing stock ``"rotation"``.
    """

    name: str = "logsmooth_rotation"

    r1_weight_log_smooth: bool = False
    r1_weight_log_smooth_percentile: float = 99.9
    r1_weight_log_smooth_s_max: float = 8.0
    r1_activation_log_smooth: bool = False
    r1_activation_log_smooth_percentile: float = 99.9
    r1_activation_log_smooth_s_max: float = 16.0
    r1_log_smooth_block_size: int = 32

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.r1_weight_log_smooth or self.r1_activation_log_smooth:
            if not self.r1 or not self.online_r1_rotation:
                raise ValueError(
                    "r1_weight_log_smooth=True or r1_activation_log_smooth=True requires "
                    "r1=True and online_r1_rotation=True."
                )
            if self.trainable:
                raise ValueError(
                    "r1_weight_log_smooth=True or r1_activation_log_smooth=True is currently supported only for "
                    "trainable=False."
                )
            if self.r1_log_smooth_block_size <= 0:
                raise ValueError(f"r1_log_smooth_block_size must be > 0, got {self.r1_log_smooth_block_size}.")
        if self.r1_weight_log_smooth:
            if not 0 < self.r1_weight_log_smooth_percentile <= 100:
                raise ValueError(
                    f"r1_weight_log_smooth_percentile must be in (0, 100], got {self.r1_weight_log_smooth_percentile}."
                )
            if self.r1_weight_log_smooth_s_max < 1:
                raise ValueError(f"r1_weight_log_smooth_s_max must be >= 1, got {self.r1_weight_log_smooth_s_max}.")
        if self.r1_activation_log_smooth:
            if not 0 < self.r1_activation_log_smooth_percentile <= 100:
                raise ValueError(
                    "r1_activation_log_smooth_percentile must be in (0, 100], got "
                    f"{self.r1_activation_log_smooth_percentile}."
                )
            if self.r1_activation_log_smooth_s_max < 1:
                raise ValueError(
                    f"r1_activation_log_smooth_s_max must be >= 1, got {self.r1_activation_log_smooth_s_max}."
                )

    @classmethod
    def from_dict(cls, rotation_dict: dict[str, Any]) -> "LogSmoothRotationConfig":
        rotation_dict = dict(rotation_dict)
        if "online_config" in rotation_dict and isinstance(rotation_dict["online_config"], dict):
            rotation_dict["online_config"] = OnlineRotationConfig(**rotation_dict["online_config"])
        return cls(**rotation_dict)
