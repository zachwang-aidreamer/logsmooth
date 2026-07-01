#
# LogSmooth — input-rotation wrappers with post-rotation scale + activation sidecar.
# SPDX-License-Identifier: MIT
#
# Subclasses of Quark's rotation wrappers. The base __init__ (rotation matrix
# setup, transform, buffers) is reused via super().__init__; LogSmooth only adds
# the post_rotation_scale buffer, the activation-smooth attributes, and the
# forward branches that apply them.
#
from typing import Any

import torch

from quark.torch.algorithm.rotation.rotation_utils import (
    InputRotationWrapperHadamard as _BaseHadamard,
    InputRotationWrapperOrthogonal as _BaseOrthogonal,
)

from .functions import forward_with_activation_exponent_sidecar, get_channel_log_smooth_values

_SIDECAR_QUANT_ATTRS = ("get_quant_input", "get_quant_weight", "get_quant_bias", "get_quant_output")


def _logsmooth_forward(self: Any, x: torch.Tensor) -> Any:
    x = self.transform(x)
    if hasattr(self, "post_rotation_scale"):
        scale = self.post_rotation_scale
        if scale.device != x.device:
            scale = scale.to(x.device)
        x = x * scale.to(dtype=x.dtype)

    if hasattr(self, "activation_log_smooth_percentile"):
        activation_scale = get_channel_log_smooth_values(x, s_max=self.activation_log_smooth_s_max)
        return _forward_with_activation_exponent_sidecar(self, x, activation_scale)

    # quantization happens here, in between (since it happens before a nn.Linear layer)
    x = self.original_module(x)
    return x


def _forward_with_activation_exponent_sidecar(self: Any, x: torch.Tensor, activation_scale: torch.Tensor) -> Any:
    if all(hasattr(self.original_module, attr) for attr in _SIDECAR_QUANT_ATTRS):
        return forward_with_activation_exponent_sidecar(
            x=x,
            activation_scale=activation_scale,
            weight=self.original_module.weight,
            bias=self.original_module.bias,
            quant_input=self.original_module.get_quant_input,
            quant_weight=self.original_module.get_quant_weight,
            quant_bias=self.original_module.get_quant_bias,
            quant_output=self.original_module.get_quant_output,
        )
    return self.original_module(x)


def _attach_log_smooth(
    self: Any,
    post_rotation_scale: torch.Tensor | None,
    activation_log_smooth_percentile: float | None,
    activation_log_smooth_s_max: float,
    log_smooth_block_size: int,
) -> None:
    if post_rotation_scale is not None:
        self.register_buffer("post_rotation_scale", post_rotation_scale.reshape(1, -1))
    if activation_log_smooth_percentile is not None:
        self.activation_log_smooth_percentile = activation_log_smooth_percentile
        self.activation_log_smooth_s_max = activation_log_smooth_s_max
        self.log_smooth_block_size = log_smooth_block_size


class InputRotationWrapperHadamard(_BaseHadamard):
    def __init__(
        self,
        original_module: torch.nn.Linear,
        rotation_size: int | None = None,
        hadamard_K: torch.Tensor | None = None,
        K: int | None = None,
        post_rotation_scale: torch.Tensor | None = None,
        activation_log_smooth_percentile: float | None = None,
        activation_log_smooth_s_max: float = 16.0,
        log_smooth_block_size: int = 32,
    ):
        super().__init__(original_module, rotation_size=rotation_size, hadamard_K=hadamard_K, K=K)
        _attach_log_smooth(
            self,
            post_rotation_scale,
            activation_log_smooth_percentile,
            activation_log_smooth_s_max,
            log_smooth_block_size,
        )

    forward = _logsmooth_forward
    _forward_with_activation_exponent_sidecar = _forward_with_activation_exponent_sidecar


class InputRotationWrapperOrthogonal(_BaseOrthogonal):
    def __init__(
        self,
        original_module: torch.nn.Linear,
        rotation_matrix: torch.Tensor,
        post_rotation_scale: torch.Tensor | None = None,
        activation_log_smooth_percentile: float | None = None,
        activation_log_smooth_s_max: float = 16.0,
        log_smooth_block_size: int = 32,
    ):
        super().__init__(original_module, rotation_matrix=rotation_matrix)
        _attach_log_smooth(
            self,
            post_rotation_scale,
            activation_log_smooth_percentile,
            activation_log_smooth_s_max,
            log_smooth_block_size,
        )

    forward = _logsmooth_forward
    _forward_with_activation_exponent_sidecar = _forward_with_activation_exponent_sidecar
