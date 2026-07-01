# LogSmooth

Standalone package implementing **LogSmooth** — power-of-two log-smoothing for
rotation-based post-training quantization (PTQ) — on top of [AMD Quark](https://github.com/amd/quark).

LogSmooth adds two smoothing transforms to Quark's R1/R4 rotation flow:

- **Weight-side static scale** (`r1_weight_log_smooth`): a per-(block-)channel
  power-of-two scale `D`, computed on the rotated weights and fused into them
  (`W @ R1 @ D`), carried on the wrapper as `post_rotation_scale`. Mathematically
  output-preserving; improves how the rotated weight/activation quantize.
- **Activation-side dynamic sidecar** (`r1_activation_log_smooth`): a per-channel
  power-of-two factor computed online from the rotated activation and compensated
  in the (fake-quant) linear path.

## How it relates to Quark

Quark provides all rotation, quantization, and export infrastructure. LogSmooth
**imports** that infrastructure and only **copies** the small amount of logic that
Quark inlines into its own method bodies (where there is no extension hook):

- `functions.py` — the pure log-smooth math.
- `config.py` — `LogSmoothRotationConfig(RotationConfig)`, `name="logsmooth_rotation"`.
- `wrappers.py` — wrapper subclasses adding `post_rotation_scale` + activation sidecar.
- `processor.py` — `LogSmoothRotationProcessor(RotationProcessor)` overriding
  `apply_online_r1`, `r4`, `prepare_model_for_reloading_fake`.
- `export.py` — `QParamsLinearWithRotation` subclass + a `_convert_quantized_model` replacement.
- `integration.py` — `install()`, wiring the above into Quark's name-gated dispatch.

Importing `logsmooth` runs `install()` automatically (idempotent).

## Install

Targets AMD Quark `release/0.11` (pinned in `pyproject.toml`). PyPI `amd-quark`
(0.6.0) is too old and lacks the required rotation internals.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # pulls amd-quark @ release/0.11 from GitHub
python -c "import logsmooth; print('ok')"
```

## Usage

```python
import logsmooth  # registers logsmooth_rotation with Quark
from logsmooth import LogSmoothRotationConfig

algo_config = LogSmoothRotationConfig(
    scaling_layers=...,          # same schema as Quark RotationConfig
    r1=True, online_r1_rotation=True,
    r1_weight_log_smooth=True,
    r1_activation_log_smooth=True,
)
# feed algo_config into Quark's PTQ pipeline as usual
```
