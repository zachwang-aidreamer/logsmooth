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

## The math

### Why power-of-two, and why it composes with MXFP4 for free

MXFP4 stores each block of `B` values (default `B = 32`) as `B` 4-bit FP4 mantissas
plus **one shared block scale in E8M0 format** — i.e. the block scale is itself a pure
power of two `2^e`. A value is reconstructed as `x ≈ 2^e · fp4(x / 2^e)`.

A general smoothing scale `s` would have to be applied by *dequantizing* the packed FP4
nibbles, multiplying, and re-quantizing. LogSmooth instead constrains every scale to a
**power of two** `2^g`. Multiplying an MXFP4 tensor by `2^g` then touches **only the E8M0
exponent** (`e → e + g`) and leaves all FP4 mantissa nibbles bit-identical:

```
2^g · (2^e · fp4(m))  =  2^(e+g) · fp4(m)
```

So the scale is applied as an **integer add on the block exponent** — no unpacking, no
dequant, no re-quantization of the weights. This is the whole reason the scales are
snapped to powers of two.

### Computing the scale `D`

Both scales come from the same log-smoothing rule. Given the per-channel (or per-block)
absolute-max statistic `a` of the rotated tensor:

```
s = a / log2(2 + a)                 # compress large-magnitude channels more than small ones
s = clamp(s, 1, s_max)              # never shrink (>= 1); cap outlier growth at s_max
D = 2^round(log2(s))                # snap to the nearest power of two  →  E8M0-compatible
```

`a = amax` (or a high percentile, `r1_*_percentile`) taken over the reduction axis. The
`a / log2(2 + a)` shape gives outlier channels a larger scale while leaving near-unit
channels at `D ≈ 1`, which is what redistributes dynamic range into the FP4-friendly
region. `s_max` (`r1_weight_log_smooth_s_max`, `r1_activation_log_smooth_s_max`) bounds how
aggressive the redistribution can get.

### Weight side (static `D_group`) — fully fused, output-preserving

`D` is computed once at quantization time over each R1 scaling group and **fused into the
already-rotated weight**, while the reciprocal rides the wrapper as a per-input-channel
`post_rotation_scale`:

```
W' = (W @ R1) · diag(1/D)           # baked into the checkpoint weight
x' = (x @ R1) · D                   # applied to the activation before the GEMM
```

The transform is exact: `x' · W'ᵀ = (x @ R1) · Wᵀ`, because `D` cancels per input channel.
Since `D` is power-of-two and per-input-channel, the activation multiply `· D` is again an
E8M0 exponent shift on the MXFP4 activation blocks — a plain, dequant-free multiply before
the fused MXFP4 GEMM. `W'` is quantized and packed **once**; nothing is unpacked at runtime.

### Activation side (dynamic `D_act`) — per-forward sidecar, no weight dequant

The activation scale is data-dependent, so it is recomputed each forward from the rotated
(and weight-side-scaled) activation `Z` using the per-channel `amax` variant (cheaper than a
percentile). Let `g = log2(D_act)` (integer, per input channel):

```
D_act = 2^g              # per-channel power-of-two, from amax of Z
U     = Z / D_act        # shrink outliers into FP4 range, then MXFP4-quantize U
Z·Wᵀ  = Σ_k 2^{g_k} · U_k · W_k     # exact reconstruction of the un-shrunk product
```

The compensation factor `2^{g_k}` is injected into the GEMM's K-reduction as a **per-channel
exponent bias on the dequantized activation lane** — the packed MXFP4 **weight stays packed**
and `K` is unchanged. Because `g_k` is an integer exponent, this is a near-free exponent add
inside the accumulator, not a materialized rescale. The in-Python reference path
(`forward_with_activation_exponent_sidecar`) reproduces the same result on fake-quantized
tensors for validation, while a fused kernel applies `2^{g_k}` directly in the MXFP4 GEMM.

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

Targets AMD Quark `release/0.11` (pinned in `pyproject.toml`).

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
