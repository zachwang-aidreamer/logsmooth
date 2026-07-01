#
# LogSmooth — rotation processor subclass.
# SPDX-License-Identifier: MIT
#
# Overrides the three RotationProcessor methods where LogSmooth logic is
# inlined into the loop bodies (apply_online_r1, r4, prepare_model_for_reloading_fake)
# plus the static get_weight_log_smooth_values helper. Each override is the
# amd-quark release/0.11 method body with the LogSmooth branches spliced in, and
# uses LogSmooth's wrapper subclasses instead of the stock wrappers.
#
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Sequence

import torch
import torch.nn as nn
from tqdm import tqdm

from quark.torch.algorithm.rotation.hadamard import _get_hadamard_K, matmul_hadU
from quark.torch.algorithm.rotation.rotation import RotationLinear, RotationProcessor
from quark.torch.algorithm.rotation.rotation_utils import get_rotation_matrix, rotate_in_channels_
from quark.torch.algorithm.utils.utils import clear_memory
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.utils import getattr_recursive, resolve_star, setattr_recursive

from .functions import get_block_log_smooth_values
from .wrappers import InputRotationWrapperHadamard

if TYPE_CHECKING:
    from quark.torch.quantization.config.config import QConfig


class LogSmoothRotationProcessor(RotationProcessor):
    """RotationProcessor with LogSmooth weight-side D and activation sidecar wiring."""

    @staticmethod
    def get_weight_log_smooth_values(
        weights: Sequence[torch.Tensor], percentile: float, s_max: float, block_size: int
    ) -> torch.Tensor:
        """
        Compute a shared D for the group-wise weight-side transform W @ R1 @ D.

        The scale is based on already-rotated PyTorch weight tensors with shape
        [out_features, in_features]. For each input-channel column, the statistic is
        computed over all rows from all weights in the current R1 scaling group.
        """
        if len(weights) == 0:
            raise ValueError("Expected at least one weight tensor to compute weight log smooth values.")

        reference_weight = weights[0]
        in_features = reference_weight.shape[-1]
        device = reference_weight.device
        for weight in weights:
            if weight.shape[-1] != in_features:
                raise ValueError(
                    "Group-wise r1_weight_log_smooth expects all target modules in a group to have the same "
                    f"in_features, got {in_features} and {weight.shape[-1]}."
                )
            if weight.device != device:
                raise ValueError(
                    "Group-wise r1_weight_log_smooth expects all target modules in a group to be on the same device, "
                    f"got {device} and {weight.device}."
                )

        weight_group = torch.cat([weight.detach() for weight in weights], dim=0)
        return (
            get_block_log_smooth_values(weight_group, percentile=percentile, s_max=s_max, block_size=block_size)
            .reshape(-1)
            .to(device=reference_weight.device, dtype=reference_weight.dtype)
        )

    def apply_online_r1(
        self,
        layers_pattern: dict[str, Any],
        target_modules: list[nn.Module],
        rotation_size: int,
    ) -> None:
        """
        Inserts InputRotationWrapper modules in a decoder layer, running activation rotations online.
        """
        wrapped_layers: list[tuple[nn.Module, str, torch.Tensor, int, nn.Module]] = []

        for i, layer in enumerate(target_modules):
            full_layer_name = layers_pattern["target_modules"][i]

            # e.g. `"model.layers.10.mlp"`.
            next_module_parent_name = ".".join(full_layer_name.split(".")[:-1])

            if next_module_parent_name == "":
                next_module_parent = self.model
            else:
                next_module_parent = getattr_recursive(self.model, next_module_parent_name)
            relative_layer_name = full_layer_name.split(".")[-1]

            dtype = layer.weight.data.dtype
            in_features = layer.weight.shape[-1]

            # The buffer `rotation_buffer` avoids recomputing the same hadamard matrices multiple times.
            if in_features not in self.rotation_buffer:
                hadamard_K, K = _get_hadamard_K(rotation_size)
                hadamard_K = hadamard_K.to(layer.weight.device)
                self.rotation_buffer[in_features] = (hadamard_K, K)
            else:
                hadamard_K, K = self.rotation_buffer[in_features]

            if rotation_size == layer.weight.data.shape[1]:
                # `inverse=True` is not required here as nn.Linear already transpose the weight.
                layer.weight.data = matmul_hadU(layer.weight.data, hadamard_K=hadamard_K, K=K).to(dtype)
            else:
                if hadamard_K.shape[0] != rotation_size:
                    hadamard_1, _ = _get_hadamard_K(rotation_size // K)

                    hadamard_1 = hadamard_1.to(layer.weight.device)
                    hadamard_K = hadamard_K.to(layer.weight.device)

                    hadamard_K = torch.kron(hadamard_K, hadamard_1)
                    K = rotation_size

                assert hadamard_K.shape[0] == rotation_size
                rotate_in_channels_(layer, rotation=hadamard_K.to(torch.float64) / math.sqrt(rotation_size))

            wrapped_layers.append((next_module_parent, relative_layer_name, hadamard_K, K, layer))

        # Weight-side log-smooth scale, computed per `in_features` partition. A shared
        # D is only valid across modules that consume the SAME input activation, which
        # in turn requires the same in_features. A configured R1 group may mix dims
        # (e.g. self_attn q/k/v have in_features=hidden, but o_proj reads the attention
        # output with a different in_features), so we group by in_features and compute
        # one D per partition. Modules in the same partition (e.g. q/k/v) share a D and
        # therefore a single rotated+scaled activation downstream.
        post_rotation_scale_by_in_features: dict[int, torch.Tensor] = {}
        if self.rotation_config.r1_weight_log_smooth:
            partitions: dict[int, list[nn.Module]] = {}
            for _, _, _, _, layer in wrapped_layers:
                partitions.setdefault(layer.weight.shape[-1], []).append(layer)

            for in_features, layers_in_partition in partitions.items():
                smooth_values = LogSmoothRotationProcessor.get_weight_log_smooth_values(
                    [layer.weight.data for layer in layers_in_partition],
                    percentile=self.rotation_config.r1_weight_log_smooth_percentile,
                    s_max=self.rotation_config.r1_weight_log_smooth_s_max,
                    block_size=self.rotation_config.r1_log_smooth_block_size,
                )
                post_rotation_scale_by_in_features[in_features] = smooth_values.detach()
                for layer in layers_in_partition:
                    layer.weight.data = layer.weight.data * torch.reciprocal(smooth_values)[None, :]

        for next_module_parent, relative_layer_name, hadamard_K, K, layer in wrapped_layers:
            layer_post_rotation_scale = post_rotation_scale_by_in_features.get(layer.weight.shape[-1])
            layer_with_input_rotation = InputRotationWrapperHadamard(
                layer,
                hadamard_K=hadamard_K,
                K=K,
                rotation_size=rotation_size,
                post_rotation_scale=layer_post_rotation_scale.clone() if layer_post_rotation_scale is not None else None,
                activation_log_smooth_percentile=(
                    self.rotation_config.r1_activation_log_smooth_percentile
                    if self.rotation_config.r1_activation_log_smooth
                    else None
                ),
                activation_log_smooth_s_max=self.rotation_config.r1_activation_log_smooth_s_max,
                log_smooth_block_size=self.rotation_config.r1_log_smooth_block_size,
            )

            setattr(next_module_parent, relative_layer_name, layer_with_input_rotation)

    def r4(self) -> None:
        if self.rotation_size is not None:
            custom_rotation_size = True
            r4_rotation_size = self.rotation_size
        else:
            custom_rotation_size = False
            r4_rotation_size = getattr(self.model.config, "moe_intermediate_size", self.model.config.intermediate_size)

        for layer in tqdm(self.layers, desc="R4 Rotation"):
            # We allow `rotation_config.mlp="mlp.experts.*"` in the case of MOE models.
            mlp_names = resolve_star([self.rotation_config.mlp], layer)

            mlp = getattr_recursive(layer, mlp_names[0])
            if isinstance(mlp.down_proj, RotationLinear):
                dtype = mlp.down_proj.linear.weight.dtype
                device = mlp.down_proj.linear.weight.device
            else:
                dtype = mlp.down_proj.weight.dtype
                device = mlp.down_proj.weight.device

            if self.trainable and self.shared_parallel:
                rotation4 = get_rotation_matrix(r4_rotation_size, random=False, device=device)
                rotation4 = torch.nn.Parameter(rotation4)

            # MOE layers may have several MLP.
            for mlp_name in mlp_names:
                mlp = getattr_recursive(layer, mlp_name)

                if not self.trainable:
                    if custom_rotation_size:
                        rotation4 = get_rotation_matrix(r4_rotation_size, random=False, device=device)  # type: ignore[arg-type]

                        rotate_in_channels_(mlp.down_proj, rotation=rotation4)
                    else:
                        # `inverse=True` is not required here as nn.Linear already transpose the weight.
                        mlp.down_proj.weight.data = matmul_hadU(mlp.down_proj.weight.data).to(dtype)

                    # Weight-side log-smooth on the rotated down_proj weight. R4 down_proj
                    # has no preceding norm (its input is the SwiGLU product), so unlike R1
                    # there is no group to share a D with: each down_proj gets its own D over
                    # its in_features. The reciprocal scale is fused into the weight here and
                    # the forward scale is carried by `post_rotation_scale` on the wrapper,
                    # so the linear's output is mathematically unchanged (s cancels per
                    # in-channel) while the rotated activation/weight quantize better.
                    down_proj_post_rotation_scale = None
                    if self.rotation_config.r1_weight_log_smooth:
                        smooth_values = LogSmoothRotationProcessor.get_weight_log_smooth_values(
                            [mlp.down_proj.weight.data],
                            percentile=self.rotation_config.r1_weight_log_smooth_percentile,
                            s_max=self.rotation_config.r1_weight_log_smooth_s_max,
                            block_size=self.rotation_config.r1_log_smooth_block_size,
                        )
                        down_proj_post_rotation_scale = smooth_values.detach()
                        mlp.down_proj.weight.data = mlp.down_proj.weight.data * torch.reciprocal(smooth_values)[None, :]

                    mlp.down_proj = InputRotationWrapperHadamard(
                        mlp.down_proj,
                        r4_rotation_size,
                        post_rotation_scale=(
                            down_proj_post_rotation_scale.clone()
                            if down_proj_post_rotation_scale is not None
                            else None
                        ),
                        activation_log_smooth_percentile=(
                            self.rotation_config.r1_activation_log_smooth_percentile
                            if self.rotation_config.r1_activation_log_smooth
                            else None
                        ),
                        activation_log_smooth_s_max=self.rotation_config.r1_activation_log_smooth_s_max,
                        log_smooth_block_size=self.rotation_config.r1_log_smooth_block_size,
                    )
                else:
                    if not self.shared_parallel:
                        rotation4 = get_rotation_matrix(r4_rotation_size, random=False, device=device)
                        rotation4 = torch.nn.Parameter(rotation4)

                    if isinstance(mlp.down_proj, RotationLinear):
                        assert mlp.down_proj.rotation_in is None
                        mlp.down_proj.rotation_in = rotation4
                        mlp.down_proj.hint_in = "r4"
                        mlp.down_proj.rotate_activation = True
                    else:
                        rotation_linear_down_proj = RotationLinear(
                            mlp.down_proj, rotation_in=rotation4, hint_in="r4", rotate_activation=True
                        )

                        down_proj_name = f"{mlp_name}.down_proj"

                        setattr_recursive(layer, down_proj_name, rotation_linear_down_proj)

            clear_memory()

    @staticmethod
    def prepare_model_for_reloading_fake(model: nn.Module, quantization_config: "QConfig") -> None:
        """
        Prepares a model using ``weight_format="fake_quantized"`` to reload online rotations.

        The case ``weight_format="real_quantized"`` is handled directly in ``QParamsLinearWithRotation``.
        """
        rotation_config = quantization_config.get_rotation_config()

        if rotation_config is None:
            raise RuntimeError(
                "rotation_config is None in `prepare_model_for_reloading`, which is unexpected. Please open an issue."
            )

        layers_online_rotation = LogSmoothRotationProcessor.get_online_rotation_layers(rotation_config, model)

        if rotation_config.r3:
            raise NotImplementedError(
                "Reloading a model quantization using rotation algorithm with r3=True is not supported at the moment. Please open an issue."
            )

        for name, module in model.named_modules():
            if isinstance(module, QuantLinear) and name in layers_online_rotation:
                rotation_size = rotation_config.rotation_size  # type: ignore[union-attr]
                trainable = rotation_config.trainable  # type: ignore[union-attr]

                if rotation_size is None:
                    rotation_size = module.in_features

                if trainable:
                    rotation_dtype = torch.float64
                else:
                    rotation_dtype = torch.int8

                input_rotation = torch.zeros(
                    (rotation_size, rotation_size), device=module.weight.device, dtype=rotation_dtype
                )
                module.register_buffer("input_rotation", input_rotation)
                if getattr(rotation_config, "r1_weight_log_smooth", False):
                    post_rotation_scale = torch.ones(
                        (1, module.in_features), device=module.weight.device, dtype=module.weight.dtype
                    )
                    module.register_buffer("post_rotation_scale", post_rotation_scale)
                if getattr(rotation_config, "r1_activation_log_smooth", False):
                    module.activation_log_smooth_percentile = rotation_config.r1_activation_log_smooth_percentile
                    module.activation_log_smooth_s_max = rotation_config.r1_activation_log_smooth_s_max
                    module.log_smooth_block_size = rotation_config.r1_log_smooth_block_size
