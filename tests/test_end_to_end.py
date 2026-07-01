"""End-to-end PTQ run through Quark's pipeline with logsmooth_rotation.

Proves the processor overrides, wrappers, and config dispatch actually execute
against unmodified amd-quark. Requires network access to fetch a tiny random
model; skipped if unavailable.
"""

import pytest
import torch

import logsmooth  # noqa: F401  (registers logsmooth_rotation)
from logsmooth import LogSmoothRotationConfig
from logsmooth.wrappers import InputRotationWrapperHadamard

pytestmark = pytest.mark.filterwarnings("ignore")

MODEL_ID = "llamafactory/tiny-random-Llama-3"
# tiny model hidden_size is small; block_size must divide it.
LOG_SMOOTH_BLOCK_SIZE = 16

SCALING_LAYERS_LLAMA = {
    "first_layer": [
        {
            "prev_modules": ["model.embed_tokens"],
            "norm_module": "model.layers.layer_id.input_layernorm",
            "next_modules": [
                "model.layers.layer_id.self_attn.q_proj",
                "model.layers.layer_id.self_attn.k_proj",
                "model.layers.layer_id.self_attn.v_proj",
            ],
        },
        {
            "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
            "norm_module": "model.layers.layer_id.post_attention_layernorm",
            "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"],
        },
    ],
    "middle_layers": [
        {
            "prev_modules": ["model.layers.pre_layer_id.mlp.down_proj"],
            "norm_module": "model.layers.layer_id.input_layernorm",
            "next_modules": [
                "model.layers.layer_id.self_attn.q_proj",
                "model.layers.layer_id.self_attn.k_proj",
                "model.layers.layer_id.self_attn.v_proj",
            ],
        },
        {
            "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
            "norm_module": "model.layers.layer_id.post_attention_layernorm",
            "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"],
        },
    ],
    "last_layer": [
        {
            "prev_modules": ["model.layers.layer_id.mlp.down_proj"],
            "norm_module": "model.norm",
            "next_modules": ["lm_head"],
        }
    ],
}


def _load_model():
    try:
        from transformers import AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"transformers unavailable: {exc}")
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, attn_implementation="sdpa")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"could not fetch {MODEL_ID}: {exc}")
    return model.eval().to(torch.float32)


def _build_config(weight_ls: bool, act_ls: bool):
    from quark.torch.quantization import OCP_MXFP4Spec, QConfig, QLayerConfig

    algo = LogSmoothRotationConfig(
        scaling_layers=SCALING_LAYERS_LLAMA,
        r1=True,
        r2=False,
        r4=False,
        online_r1_rotation=True,
        r1_weight_log_smooth=weight_ls,
        r1_activation_log_smooth=act_ls,
        r1_log_smooth_block_size=LOG_SMOOTH_BLOCK_SIZE,
    )
    spec = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=True).to_quantization_spec()
    layer_cfg = QLayerConfig(weight=spec, input_tensors=spec)
    return QConfig(global_quant_config=layer_cfg, algo_config=[algo])


def _count_wrappers(model):
    n_total = n_with_scale = n_with_act = 0
    for module in model.modules():
        if isinstance(module, InputRotationWrapperHadamard):
            n_total += 1
            if hasattr(module, "post_rotation_scale"):
                n_with_scale += 1
            if hasattr(module, "activation_log_smooth_percentile"):
                n_with_act += 1
    return n_total, n_with_scale, n_with_act


def test_ptq_logsmooth_weight_and_activation():
    from quark.torch import ModelQuantizer

    model = _load_model()
    inp = torch.randint(0, model.config.vocab_size, (1, 16))

    with torch.no_grad():
        ref = model(inp).logits

    quant_config = _build_config(weight_ls=True, act_ls=True)
    model = ModelQuantizer(quant_config).quantize_model(model)

    n_total, n_with_scale, n_with_act = _count_wrappers(model)
    # The processor override ran and inserted LogSmooth wrappers with both the
    # static weight scale buffer and the activation-smooth attributes.
    assert n_total > 0, "no LogSmooth rotation wrappers were inserted"
    assert n_with_scale > 0, "post_rotation_scale buffer missing (weight-side log-smooth did not run)"
    assert n_with_act > 0, "activation_log_smooth attributes missing (activation sidecar not wired)"

    with torch.no_grad():
        out = model(inp).logits
    assert out.shape == ref.shape
    assert torch.isfinite(out).all()


def test_ptq_logsmooth_flags_off_is_plain_rotation():
    """With both flags off, no post_rotation_scale / activation attrs should appear."""
    from quark.torch import ModelQuantizer

    model = _load_model()
    quant_config = _build_config(weight_ls=False, act_ls=False)
    model = ModelQuantizer(quant_config).quantize_model(model)

    n_total, n_with_scale, n_with_act = _count_wrappers(model)
    assert n_total > 0
    assert n_with_scale == 0
    assert n_with_act == 0
