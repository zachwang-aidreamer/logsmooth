#
# LogSmooth — install(): wire LogSmooth into Quark's name-gated dispatch seams.
# SPDX-License-Identifier: MIT
#
# Quark hard-references its rotation classes/functions by name at a few points
# that plain subclassing cannot reach. install() registers the LogSmooth
# processor/config and patches those seams. It is idempotent.
#
from __future__ import annotations

from typing import Any

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from .config import LogSmoothRotationConfig
    from .export import QParamsLinearWithRotation as LogSmoothQParamsLinearWithRotation
    from .export import convert_quantized_model as logsmooth_convert
    from .processor import LogSmoothRotationProcessor

    # 1) Algorithm dispatch: PROCESSOR_MAP[name] -> processor class.
    import quark.torch.algorithm.api as algo_api

    algo_api.PROCESSOR_MAP["logsmooth_rotation"] = LogSmoothRotationProcessor

    # 2) Config deserialization: name -> config class (else raises ValueError).
    import quark.torch.quantization.config.config as qconfig

    _orig_load_algo = qconfig._load_quant_algo_config_from_dict

    def _patched_load_algo(algo_config_dict: dict[str, Any]) -> Any:
        if algo_config_dict.get("name") == "logsmooth_rotation":
            return LogSmoothRotationConfig.from_dict(algo_config_dict)
        return _orig_load_algo(algo_config_dict)

    qconfig._load_quant_algo_config_from_dict = _patched_load_algo

    # 3a) Fake-quantized reload prep: export/utils.py hard-calls
    #     RotationProcessor.prepare_model_for_reloading_fake. Redirect the base
    #     static method to LogSmooth's when the config is a LogSmoothRotationConfig.
    from quark.torch.algorithm.rotation.rotation import RotationProcessor

    # Accessing a staticmethod through the class yields the plain underlying function.
    _orig_prepare = RotationProcessor.prepare_model_for_reloading_fake

    def _patched_prepare(model: Any, quantization_config: Any) -> None:
        rotation_config = quantization_config.get_rotation_config()
        if isinstance(rotation_config, LogSmoothRotationConfig):
            LogSmoothRotationProcessor.prepare_model_for_reloading_fake(model, quantization_config)
        else:
            _orig_prepare(model, quantization_config)

    RotationProcessor.prepare_model_for_reloading_fake = staticmethod(_patched_prepare)

    # 3b) Model conversion (real + fake quantized). Replace the function both in
    #     its defining module AND where export/api.py imported it by value.
    import quark.torch.export.utils as export_utils

    export_utils._convert_quantized_model = logsmooth_convert
    try:
        import quark.torch.export.api as export_api

        if hasattr(export_api, "_convert_quantized_model"):
            export_api._convert_quantized_model = logsmooth_convert
    except Exception:
        pass

    # 3c) Real-quantized class selection in export/api.py picks
    #     QParamsLinearWithRotation for online-rotation layers. Swap the bound
    #     reference so LogSmooth's subclass (with post_rotation_scale/sidecar) is used.
    try:
        import quark.torch.export.api as export_api

        export_api.QParamsLinearWithRotation = LogSmoothQParamsLinearWithRotation
    except Exception:
        pass

    _INSTALLED = True
