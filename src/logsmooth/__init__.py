#
# LogSmooth — power-of-two log-smoothing for AMD Quark rotation PTQ.
# SPDX-License-Identifier: MIT
#
from .config import LogSmoothRotationConfig
from .functions import (
    forward_with_activation_exponent_sidecar,
    get_block_log_smooth_values,
    get_channel_log_smooth_values,
)
from .integration import install
from .processor import LogSmoothRotationProcessor
from .wrappers import InputRotationWrapperHadamard, InputRotationWrapperOrthogonal

# Wire LogSmooth into Quark's dispatch seams on import. Idempotent.
install()

__all__ = [
    "LogSmoothRotationConfig",
    "LogSmoothRotationProcessor",
    "InputRotationWrapperHadamard",
    "InputRotationWrapperOrthogonal",
    "get_block_log_smooth_values",
    "get_channel_log_smooth_values",
    "forward_with_activation_exponent_sidecar",
    "install",
]
