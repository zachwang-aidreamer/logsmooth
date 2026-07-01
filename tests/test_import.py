"""Import + registration smoke tests: proves LogSmooth loads against clean Quark."""


def test_import_and_install():
    import logsmooth

    # install() ran on import and is idempotent.
    logsmooth.install()

    import quark.torch.algorithm.api as algo_api

    assert algo_api.PROCESSOR_MAP["logsmooth_rotation"] is logsmooth.LogSmoothRotationProcessor


def test_config_is_rotation_config_subclass():
    from quark.torch.quantization.config.config import RotationConfig

    from logsmooth import LogSmoothRotationConfig

    assert issubclass(LogSmoothRotationConfig, RotationConfig)
    cfg = LogSmoothRotationConfig(scaling_layers={})
    assert cfg.name == "logsmooth_rotation"
    # get_rotation_config uses isinstance(RotationConfig) -> subclass must satisfy it.
    assert isinstance(cfg, RotationConfig)


def test_pure_functions_shapes():
    import torch

    from logsmooth import get_block_log_smooth_values, get_channel_log_smooth_values

    x = torch.randn(8, 64)
    block = get_block_log_smooth_values(x, percentile=99.9, s_max=8.0, block_size=32)
    assert block.shape == (1, 64)
    assert torch.all(block >= 1.0) and torch.all(block <= 8.0)

    chan = get_channel_log_smooth_values(x, s_max=16.0)
    assert chan.shape == (1, 64)
    # power-of-two values only
    assert torch.allclose(chan, torch.pow(2.0, torch.round(torch.log2(chan))))
