#!/usr/bin/env python3
"""
Verify models_tflite/llio128.tflite against the source PyTorch checkpoint.

LLIO-Net uses GELU (ONNX Erf op) throughout, which onnx2tf's default TFLite
lowering sends to tf.math.erf -- a Flex/SELECT_TF_OPS op with no builtin TFLite
kernel, so the converted model fails to load in a plain tflite-runtime
interpreter (the kind deployed on the Raspberry Pi) with:
    RuntimeError: ... FlexErf failed to prepare.
models_tflite/llio128.tflite was produced with `onnx2tf -rtpo Erf`
(see tools/convert_onnx_to_tflite.py), which replaces Erf with a builtin-op
pseudo-approximation before conversion. This script checks that swap did not
change the model's outputs beyond float32 rounding.

Run on a laptop/desktop with both PyTorch and TensorFlow available:
    python tools/verify_llio128_tflite.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def add_source_to_path(base_dir):
    source_dir = base_dir / "source"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=".")
    parser.add_argument(
        "--checkpoint", default="output/train_llio128/checkpoints/checkpoint_99.pt"
    )
    parser.add_argument("--tflite", default="models_tflite/llio128.tflite")
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    add_source_to_path(base_dir)
    from model_llio1d import LLIO1D

    import tensorflow as tf

    model = LLIO1D(in_channels=6, num_outputs=2, dropout=0.2, feature_dim=args.feature_dim)
    checkpoint = torch.load(base_dir / args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    interpreter = tf.lite.Interpreter(model_path=str(base_dir / args.tflite))
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    print(f"tflite input:  {in_detail['shape']} {in_detail['dtype']}")
    print(f"tflite output: {out_detail['shape']} {out_detail['dtype']}")

    rng = np.random.default_rng(args.seed)
    max_diffs = []
    for i in range(args.num_samples):
        sample = rng.normal(size=(1, 6, 200)).astype(np.float32)
        with torch.no_grad():
            torch_out = model(torch.from_numpy(sample)).numpy()

        # onnx2tf's channel-last convention: (1, 6, 200) NCW -> (1, 200, 6) NWC.
        tflite_in = np.transpose(sample, (0, 2, 1)).astype(np.float32)
        interpreter.set_tensor(in_detail["index"], tflite_in)
        interpreter.invoke()
        tflite_out = interpreter.get_tensor(out_detail["index"])

        diff = float(np.max(np.abs(torch_out - tflite_out)))
        max_diffs.append(diff)
        print(f"sample {i}: torch={torch_out} tflite={tflite_out} diff={diff:.8f}")

    overall = max(max_diffs)
    print(f"\noverall max_abs_diff: {overall:.8f}")
    if overall > 1e-3:
        raise SystemExit(f"FAILED: max_abs_diff {overall} exceeds 1e-3 tolerance")
    print("PASSED")


if __name__ == "__main__":
    main()
