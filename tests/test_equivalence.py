"""Config round-trip + numeric parity checks."""

import pytest
import torch


def test_config_roundtrip_through_quark_loader():
    """logsmooth_rotation must deserialize back to LogSmoothRotationConfig via Quark's loader."""
    import logsmooth  # noqa: F401  (registers the loader patch)
    from quark.torch.quantization.config.config import _load_quant_algo_config_from_dict

    from logsmooth import LogSmoothRotationConfig

    d = {
        "name": "logsmooth_rotation",
        "scaling_layers": {},
        "r1": True,
        "online_r1_rotation": True,
        "r1_weight_log_smooth": True,
        "r1_weight_log_smooth_percentile": 99.5,
        "r1_weight_log_smooth_s_max": 8.0,
        "r1_activation_log_smooth": True,
        "r1_activation_log_smooth_s_max": 16.0,
        "r1_log_smooth_block_size": 32,
    }
    cfg = _load_quant_algo_config_from_dict(dict(d))
    assert isinstance(cfg, LogSmoothRotationConfig)
    assert cfg.r1_weight_log_smooth is True
    assert cfg.r1_weight_log_smooth_percentile == 99.5
    assert cfg.r1_log_smooth_block_size == 32


def test_unknown_name_still_raises():
    import logsmooth  # noqa: F401
    from quark.torch.quantization.config.config import _load_quant_algo_config_from_dict

    with pytest.raises(ValueError):
        _load_quant_algo_config_from_dict({"name": "definitely_not_a_real_algo"})


def test_stock_rotation_still_loads():
    """The loader patch must delegate non-logsmooth names to the original."""
    import logsmooth  # noqa: F401
    from quark.torch.quantization.config.config import RotationConfig, _load_quant_algo_config_from_dict

    cfg = _load_quant_algo_config_from_dict({"name": "rotation", "scaling_layers": {}})
    assert isinstance(cfg, RotationConfig)
    assert cfg.name == "rotation"


def test_weight_side_scale_is_output_preserving():
    """W @ diag(1/D) applied with activation * D leaves the linear output unchanged."""
    from logsmooth import get_block_log_smooth_values

    torch.manual_seed(0)
    in_features, out_features = 64, 32
    W = torch.randn(out_features, in_features)
    x = torch.randn(4, in_features)

    D = get_block_log_smooth_values(W, percentile=99.9, s_max=8.0, block_size=32).reshape(-1)

    y_ref = x @ W.T
    W_scaled = W * torch.reciprocal(D)[None, :]
    x_scaled = x * D[None, :]
    y_scaled = x_scaled @ W_scaled.T

    assert torch.allclose(y_ref, y_scaled, atol=1e-4, rtol=1e-4)
