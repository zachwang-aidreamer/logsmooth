#!/usr/bin/env python
#
# Minimal LogSmooth PTQ driver: rotation + log-smooth + MXFP4, then export.
# SPDX-License-Identifier: MIT
#
# Importing `logsmooth` registers the `logsmooth_rotation` algorithm with Quark,
# so a rotation config JSON with "name": "logsmooth_rotation" is routed to the
# LogSmooth processor automatically.
#
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import logsmooth  # noqa: F401  (registers logsmooth_rotation with Quark)

from quark.torch import ModelQuantizer, export_safetensors
from quark.torch.quantization import OCP_MXFP4Spec, QConfig, QLayerConfig
from quark.torch.quantization.config.config import load_quant_algo_config_from_file
from quark.torch.utils.llm.data_preparation import get_calib_dataloader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LogSmooth rotation + MXFP4 PTQ example.")
    p.add_argument("--model", required=True, help="HF model id or local path.")
    p.add_argument("--rotation_config", required=True, help="Path to logsmooth_rotation JSON.")
    p.add_argument("--output_dir", required=True, help="Where to write the quantized checkpoint.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--dataset",
        default="pileval",
        help="Calibration dataset name, or 'synthetic' for random tokens (no dataset download).",
    )
    p.add_argument("--num_calib_data", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument(
        "--weight_format",
        default="real_quantized",
        choices=["real_quantized", "fake_quantized"],
        help="real_quantized packs MXFP4 weights; fake_quantized keeps QDQ for eval.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa"
    ).eval()
    if args.device == "cuda":
        model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dataset == "synthetic":
        # Random-token calibration — avoids any HF datasets download / version issues.
        # Fine for a smoke run; use a real dataset for quality-sensitive quantization.
        from torch.utils.data import DataLoader

        text_config = getattr(model.config, "text_config", model.config)
        vocab_size = getattr(text_config, "vocab_size", model.config.vocab_size)
        tokens = torch.randint(0, vocab_size, (args.num_calib_data, args.seq_len))
        if args.device == "cuda":
            tokens = tokens.to("cuda")
        calib_dataloader = DataLoader(tokens, batch_size=1)
    else:
        calib_dataloader = get_calib_dataloader(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            num_calib_data=args.num_calib_data,
            seqlen=args.seq_len,
            device=args.device,
        )

    # Weight + activation MXFP4 (per-group, dynamic) — the target format LogSmooth
    # is designed to improve.
    mxfp4 = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=True).to_quantization_spec()
    global_quant_config = QLayerConfig(weight=mxfp4, input_tensors=mxfp4)

    # "name": "logsmooth_rotation" in the JSON -> LogSmoothRotationConfig (via the
    # loader patch installed on `import logsmooth`).
    algo_config = load_quant_algo_config_from_file(args.rotation_config)

    quant_config = QConfig(global_quant_config=global_quant_config, algo_config=[algo_config])

    quantizer = ModelQuantizer(quant_config)
    model = quantizer.quantize_model(model, calib_dataloader)

    export_safetensors(model, args.output_dir, weight_format=args.weight_format)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[logsmooth] quantized checkpoint written to: {args.output_dir}")


if __name__ == "__main__":
    main()
