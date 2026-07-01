#
# LogSmooth — power-of-two log-smoothing values and activation sidecar.
# SPDX-License-Identifier: MIT
#
# These are LogSmooth's own math primitives. They are pure (depend only on
# torch / torch.nn.functional) and are therefore copied into the standalone
# package rather than imported from Quark.
#
from collections.abc import Callable
from typing import Any

import torch
from torch.nn import functional as F


def get_block_log_smooth_values(
    x: torch.Tensor, percentile: float, s_max: float, block_size: int = 32
) -> torch.Tensor:
    """
    Compute power-of-two log smoothing values shared by each MX block.

    The returned tensor has shape ``(1, hidden_size)`` with one value repeated
    across every ``block_size`` channels, making the scale compatible with MXFP4
    block-scale granularity.
    """
    hidden_size = x.shape[-1]
    if hidden_size % block_size != 0:
        raise ValueError(
            f"Log smooth block scaling expects hidden_size to be divisible by block_size, "
            f"got hidden_size={hidden_size}, block_size={block_size}."
        )

    x_abs_blocks = x.detach().abs().float().reshape(-1, hidden_size // block_size, block_size)
    if percentile >= 100:
        amax = x_abs_blocks.amax(dim=(0, 2))
    else:
        block_values = x_abs_blocks.permute(1, 0, 2).reshape(hidden_size // block_size, -1)
        amax = torch.quantile(block_values, percentile / 100.0, dim=1)

    smooth_values = amax / torch.log2(2.0 + amax)
    smooth_values = torch.nan_to_num(smooth_values, nan=1.0, posinf=s_max, neginf=1.0)
    smooth_values = torch.clamp(smooth_values, min=1.0, max=s_max)
    smooth_values = torch.pow(2.0, torch.round(torch.log2(smooth_values)))
    smooth_values = smooth_values.repeat_interleave(block_size).reshape(1, hidden_size)
    return smooth_values.to(device=x.device, dtype=x.dtype)


def get_channel_log_smooth_values(x: torch.Tensor, s_max: float) -> torch.Tensor:
    """
    Compute per-channel power-of-two log smoothing values for dynamic activation compensation.

    The returned tensor has shape ``(1, hidden_size)``. It uses the per-channel
    maximum instead of a percentile so the online forward path avoids the cost of
    ``torch.quantile``.
    """
    hidden_size = x.shape[-1]
    x_abs = x.detach().abs().float().reshape(-1, hidden_size)
    amax = x_abs.amax(dim=0)

    smooth_values = amax / torch.log2(2.0 + amax)
    smooth_values = torch.nan_to_num(smooth_values, nan=1.0, posinf=s_max, neginf=1.0)
    smooth_values = torch.clamp(smooth_values, min=1.0, max=s_max)
    smooth_values = torch.pow(2.0, torch.round(torch.log2(smooth_values)))
    smooth_values = smooth_values.reshape(1, hidden_size)
    return smooth_values.to(device=x.device, dtype=x.dtype)


def forward_with_activation_exponent_sidecar(
    x: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    quant_input: Callable[[torch.Tensor], torch.Tensor],
    quant_weight: Callable[[torch.Tensor], torch.Tensor],
    quant_bias: Callable[[Any], Any],
    quant_output: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Simulate per-channel exponent sidecar compensation in the fake/dequantized linear path.

    A real sidecar kernel would apply the per-channel ``2^g_i`` factor inside the
    K-lane product or accumulation. This validation helper materializes the same
    compensation on the fake-quantized activation tensor while keeping K unchanged.
    """
    activation_scale = activation_scale.to(device=x.device, dtype=x.dtype)
    x_scaled = x * torch.reciprocal(activation_scale)

    quant_input_tensor = quant_input(x_scaled)
    quant_input_tensor = quant_input_tensor * activation_scale.to(
        device=quant_input_tensor.device, dtype=quant_input_tensor.dtype
    )

    quant_weight_tensor = quant_weight(weight)
    quant_bias_tensor = quant_bias(bias)

    if quant_weight_tensor.dtype != quant_input_tensor.dtype:
        quant_weight_tensor = quant_weight_tensor.to(quant_input_tensor.dtype)
    if quant_weight_tensor.device != quant_input_tensor.device:
        quant_weight_tensor = quant_weight_tensor.to(quant_input_tensor.device)

    if quant_bias_tensor is not None:
        if quant_bias_tensor.dtype != quant_input_tensor.dtype:
            quant_bias_tensor = quant_bias_tensor.to(quant_input_tensor.dtype)
        if quant_bias_tensor.device != quant_input_tensor.device:
            quant_bias_tensor = quant_bias_tensor.to(quant_input_tensor.device)

    output = F.linear(quant_input_tensor, quant_weight_tensor, bias=quant_bias_tensor)
    return quant_output(output)
