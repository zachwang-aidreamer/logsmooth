#
# LogSmooth — export / reload integration.
# SPDX-License-Identifier: MIT
#
# Provides:
#   * QParamsLinearWithRotation subclass carrying the LogSmooth post_rotation_scale
#     buffer + activation-smooth attributes (real_quantized path), and
#   * a replacement _convert_quantized_model that rebuilds LogSmooth wrapper
#     subclasses for fake_quantized reload (reads the log-smooth attrs off the
#     QuantLinear and forwards them to the wrappers).
#
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from quark.torch.export.nn.modules.qparamslinear import QParamsLinearWithRotation as _BaseQParamsRot
from quark.torch.quantization.config.config import AlgoConfig, QLayerConfig, RotationConfig
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.utils import setattr_recursive

from .functions import forward_with_activation_exponent_sidecar, get_channel_log_smooth_values
from .wrappers import InputRotationWrapperHadamard, InputRotationWrapperOrthogonal


def _rotation_config_from(algo_config: list[AlgoConfig] | None) -> RotationConfig | None:
    if algo_config is None:
        return None
    for algo_conf in algo_config:
        if isinstance(algo_conf, RotationConfig):
            return algo_conf
    return None


class QParamsLinearWithRotation(_BaseQParamsRot):
    """QParamsLinearWithRotation with the LogSmooth post-rotation scale + activation sidecar."""

    def __init__(
        self,
        linear: nn.Linear,
        custom_mode: str,
        pack_method: str | None = "reorder",
        quant_config: QLayerConfig | None = None,
        algo_config: list[AlgoConfig] | None = None,
    ):
        super().__init__(
            linear=linear,
            custom_mode=custom_mode,
            pack_method=pack_method,
            quant_config=quant_config,
            algo_config=algo_config,
        )

        rotation_config = _rotation_config_from(algo_config)
        # The analytic `r1_weight_log_smooth` serializes a static per-(block-)channel
        # `post_rotation_scale` applied after the online rotation.
        self.uses_post_rotation_scale = bool(getattr(rotation_config, "r1_weight_log_smooth", False))
        if self.uses_post_rotation_scale:
            post_rotation_scale = torch.ones(
                (1, linear.in_features), device=linear.weight.device, dtype=linear.weight.dtype
            )
            self.register_buffer("post_rotation_scale", post_rotation_scale)
        self.uses_activation_log_smooth = bool(getattr(rotation_config, "r1_activation_log_smooth", False))
        self.activation_log_smooth_percentile = getattr(rotation_config, "r1_activation_log_smooth_percentile", 99.9)
        self.activation_log_smooth_s_max = getattr(rotation_config, "r1_activation_log_smooth_s_max", 16.0)
        self.log_smooth_block_size = getattr(rotation_config, "r1_log_smooth_block_size", 32)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        assert len(args) == 1
        inp = args[0]

        inp = self.transform(inp)
        if self.uses_post_rotation_scale:
            inp = inp * self.post_rotation_scale.to(device=inp.device, dtype=inp.dtype)

        if self.uses_activation_log_smooth:
            activation_scale = get_channel_log_smooth_values(inp, s_max=self.activation_log_smooth_s_max)
            return forward_with_activation_exponent_sidecar(
                x=inp,
                activation_scale=activation_scale,
                weight=self.weight,
                bias=self.bias,
                quant_input=self._get_qinput,
                quant_weight=self._get_qweight,
                quant_bias=self._get_qbias,
                quant_output=self._get_qoutput,
            ).to(inp.dtype)

        # Skip the base QParamsLinearWithRotation.forward (it re-applies transform);
        # call the grandparent QParamsLinear.forward with the already-transformed input.
        return _BaseQParamsRot.__bases__[0].forward(self, inp)


def convert_quantized_model(model: nn.Module, model_config: Any) -> nn.Module:
    """
    LogSmooth replacement for quark.torch.export.utils._convert_quantized_model.

    Same control flow as upstream 0.11, but the fake_quantized branch rebuilds
    LogSmooth wrapper subclasses and carries the post_rotation_scale buffer +
    activation-smooth attributes recorded on the QuantLinear during reload prep.
    """
    from quark.torch.export.utils import _convert_e4m3fn_to_e4m3fnuz, logger

    if model_config.quantization_config is None:
        return model

    custom_mode = model_config.quantization_config["quant_method"]
    assert custom_mode in ["fp8", "awq", "quark"], f"Unsupported quantization method: {custom_mode}"

    is_real_quantized_mode = model_config.weight_format != "fake_quantized"
    if custom_mode == "fp8" and is_real_quantized_mode and torch.version.hip is not None:
        logger.info("In-place fp8 e4m3fn to e4m3fnuz conversion start.")
        _convert_e4m3fn_to_e4m3fnuz(model)

    for name, submodule in model.named_modules():
        # `weight_format="real_quantized"` case.
        if isinstance(submodule, _BaseQParamsRot):
            submodule.post_process_after_loading()

        # `weight_format="fake_quantized"` case.
        if isinstance(submodule, QuantLinear) and hasattr(submodule, "input_rotation"):
            post_rotation_scale = getattr(submodule, "post_rotation_scale", None)
            activation_log_smooth_percentile = getattr(submodule, "activation_log_smooth_percentile", None)
            activation_log_smooth_s_max = getattr(submodule, "activation_log_smooth_s_max", 16.0)
            log_smooth_block_size = getattr(submodule, "log_smooth_block_size", 32)
            if submodule.input_rotation.dtype == torch.float64:
                layer_with_input_rotation: nn.Module = InputRotationWrapperOrthogonal(
                    submodule,
                    rotation_matrix=submodule.input_rotation,
                    post_rotation_scale=post_rotation_scale,
                    activation_log_smooth_percentile=activation_log_smooth_percentile,
                    activation_log_smooth_s_max=activation_log_smooth_s_max,
                    log_smooth_block_size=log_smooth_block_size,
                )
            elif submodule.input_rotation.dtype == torch.int8:
                rotation_size = submodule.input_rotation.shape[0]
                layer_with_input_rotation = InputRotationWrapperHadamard(
                    submodule,
                    rotation_size=rotation_size,
                    post_rotation_scale=post_rotation_scale,
                    activation_log_smooth_percentile=activation_log_smooth_percentile,
                    activation_log_smooth_s_max=activation_log_smooth_s_max,
                    log_smooth_block_size=log_smooth_block_size,
                )
            else:
                raise ValueError("Wrong input_rotation dtype.")

            setattr_recursive(model, name, layer_with_input_rotation)

    return model
