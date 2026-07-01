# LogSmooth example: Qwen3.5-35B-A13B

One-line rotation + LogSmooth + MXFP4 quantization of **Qwen3.5-35B-A13B**.

## Run

```bash
bash examples/run_qwen35_35b_logsmooth.sh
```

That's it. The script:

1. loads `Qwen/Qwen3.5-35B-A13B-FP8`,
2. applies **online R1 Hadamard rotation** + **LogSmooth** (weight-side static
   `D_group` + activation-side dynamic per-channel sidecar),
3. quantizes **attention** (`self_attn` + `linear_attn`) and the **MoE**
   (`mlp.shared_expert` *and* routed `mlp.experts.*`) to **MXFP4**, and
4. exports a Quark checkpoint to `examples/outputs/qwen35_35b_a13b_logsmooth/`.

`import logsmooth` in the driver registers the `logsmooth_rotation` algorithm with
Quark, so the config JSON's `"name": "logsmooth_rotation"` routes to the LogSmooth
processor automatically.

## Configuration

Everything is overridable via environment variables:

| Env var | Default | Meaning |
|---|---|---|
| `MODEL` | `Qwen/Qwen3.5-35B-A13B-FP8` | HF id or local path |
| `OUTPUT_DIR` | `examples/outputs/qwen35_35b_a13b_logsmooth` | export directory |
| `ROTATION_CONFIG` | `examples/qwen35_35b_a13b_logsmooth_rotation.json` | rotation + log-smooth config |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `DATASET` | `pileval` | calibration set (`pileval`/`cnn_dailymail`/`wikitext`, or `synthetic`) |
| `NUM_CALIB_DATA` | `128` | calibration samples |
| `SEQ_LEN` | `512` | calibration sequence length |
| `WEIGHT_FORMAT` | `real_quantized` | `real_quantized` (packed MXFP4) or `fake_quantized` (QDQ, for eval) |

Examples:

```bash
# Local checkpoint, custom output:
MODEL=/data/qwen35-35b-a13b OUTPUT_DIR=/data/out bash examples/run_qwen35_35b_logsmooth.sh

# No dataset download (random-token calibration — smoke run only):
DATASET=synthetic bash examples/run_qwen35_35b_logsmooth.sh
```

## Files

- `run_qwen35_35b_logsmooth.sh` — the one-line entry point.
- `quantize.py` — minimal driver (load → calibrate → quantize → export).
- `qwen35_35b_a13b_logsmooth_rotation.json` — rotation config with LogSmooth fields
  and the full attention + MoE scaling targets.

## Toggling LogSmooth

The log-smooth behavior lives entirely in the JSON. Set both to `false` for a plain
rotation baseline (identical to stock Quark `rotation`):

```json
"r1_weight_log_smooth": true,       // weight-side static D_group
"r1_activation_log_smooth": true,   // activation-side dynamic sidecar
```

## Notes

- Requires a GPU with enough memory for a 35B-A13B model in the chosen dtype for
  `DEVICE=cuda`. Use a smaller local model + `DEVICE=cpu` to try the flow.
- If your `datasets` package version can't fetch the built-in calibration sets, use
  `DATASET=synthetic` to run the pipeline end-to-end without any download.
