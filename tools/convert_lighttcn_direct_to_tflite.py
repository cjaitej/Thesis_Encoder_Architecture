#!/usr/bin/env python3
"""
Directly convert the LightTCN PyTorch checkpoint to TFLite.

This bypasses ONNX/onnx2tf because LightTCN's causal TCN chomp/residual graph is
mis-converted by onnx2tf on this project. Run on the laptop/desktop where both
PyTorch and TensorFlow are available. No files under source/ are modified.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch


def add_source_to_path(base_dir):
    source_dir = base_dir / "source"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))


def load_torch_lighttcn(base_dir, checkpoint_path):
    add_source_to_path(base_dir)
    from model_lighttcn1d import LightTCN1D

    model = LightTCN1D(in_channels=6, num_outputs=2, dropout=0.2)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, 6, 200))
    return model


def torch_conv_to_keras(conv_layer, keras_conv):
    weight = conv_layer.weight.detach().cpu().numpy()  # out, in, kernel
    kernel = np.transpose(weight, (2, 1, 0))  # kernel, in, out
    if conv_layer.bias is None:
        bias = np.zeros((weight.shape[0],), dtype=np.float32)
    else:
        bias = conv_layer.bias.detach().cpu().numpy()
    keras_conv.set_weights([kernel.astype(np.float32), bias.astype(np.float32)])


def torch_dense_to_keras(linear_layer, keras_dense):
    weight = linear_layer.weight.detach().cpu().numpy()  # out, in
    bias = linear_layer.bias.detach().cpu().numpy()
    keras_dense.set_weights([weight.T.astype(np.float32), bias.astype(np.float32)])


def prelu_const(x, alpha):
    alpha = tf.constant(float(alpha), dtype=tf.float32)
    return tf.maximum(x, 0.0) + alpha * tf.minimum(x, 0.0)


def build_keras_lighttcn(torch_model):
    inputs = tf.keras.Input(shape=(6, 200), batch_size=1, name="input")
    x = tf.keras.layers.Permute((2, 1), name="input_ncw_to_nwc")(inputs)

    keras_layers = {"blocks": []}
    for idx, block in enumerate(torch_model.tcn.network):
        dilation = int(block.conv1.dilation[0])

        conv1 = tf.keras.layers.Conv1D(
            filters=block.conv1.out_channels,
            kernel_size=block.conv1.kernel_size[0],
            dilation_rate=dilation,
            padding="causal",
            use_bias=True,
            name=f"tcn_{idx}_conv1",
        )
        conv2 = tf.keras.layers.Conv1D(
            filters=block.conv2.out_channels,
            kernel_size=block.conv2.kernel_size[0],
            dilation_rate=dilation,
            padding="causal",
            use_bias=True,
            name=f"tcn_{idx}_conv2",
        )

        y = conv1(x)
        alpha1 = block.relu1.weight.detach().cpu().numpy().reshape(-1)[0]
        y = tf.keras.layers.Lambda(lambda t, a=alpha1: prelu_const(t, a), name=f"tcn_{idx}_prelu1")(y)
        y = conv2(y)
        alpha2 = block.relu2.weight.detach().cpu().numpy().reshape(-1)[0]
        y = tf.keras.layers.Lambda(lambda t, a=alpha2: prelu_const(t, a), name=f"tcn_{idx}_prelu2")(y)

        downsample = None
        if block.downsample is not None:
            downsample = tf.keras.layers.Conv1D(
                filters=block.downsample.out_channels,
                kernel_size=1,
                padding="same",
                use_bias=True,
                name=f"tcn_{idx}_downsample",
            )
            residual = downsample(x)
        else:
            residual = x

        x = tf.keras.layers.Add(name=f"tcn_{idx}_add")([y, residual])
        alpha = block.relu.weight.detach().cpu().numpy().reshape(-1)[0]
        x = tf.keras.layers.Lambda(lambda t, a=alpha: prelu_const(t, a), name=f"tcn_{idx}_prelu_out")(x)

        keras_layers["blocks"].append(
            {
                "torch_block": block,
                "conv1": conv1,
                "conv2": conv2,
                "downsample": downsample,
            }
        )

    x = tf.keras.layers.GlobalAveragePooling1D(name="pool")(x)
    dense1 = tf.keras.layers.Dense(24, activation="relu", name="head_dense1")
    dense2 = tf.keras.layers.Dense(2, name="head_dense2")
    x = dense1(x)
    outputs = dense2(x)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lighttcn_direct")

    for item in keras_layers["blocks"]:
        block = item["torch_block"]
        torch_conv_to_keras(block.conv1, item["conv1"])
        torch_conv_to_keras(block.conv2, item["conv2"])
        if item["downsample"] is not None:
            torch_conv_to_keras(block.downsample, item["downsample"])

    torch_dense_to_keras(torch_model.head[0], dense1)
    torch_dense_to_keras(torch_model.head[3], dense2)
    return keras_model


def validate(torch_model, keras_model):
    rng = np.random.default_rng(7)
    sample = rng.normal(size=(1, 6, 200)).astype(np.float32)
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(sample)).detach().cpu().numpy()
    keras_out = keras_model(sample, training=False).numpy()
    max_abs_diff = float(np.max(np.abs(torch_out - keras_out)))
    return torch_out.tolist(), keras_out.tolist(), max_abs_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=".")
    parser.add_argument(
        "--checkpoint",
        default="output/train_lighttcn/checkpoints/lighttcn_checkpoint_42.pt",
    )
    parser.add_argument("--out_dir", default="models_tflite")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    checkpoint_path = (base_dir / args.checkpoint).resolve()
    out_dir = (base_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_model = load_torch_lighttcn(base_dir, checkpoint_path)
    keras_model = build_keras_lighttcn(torch_model)
    torch_out, keras_out, max_abs_diff = validate(torch_model, keras_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    tflite_path = out_dir / "lighttcn.tflite"
    tflite_path.write_bytes(tflite_model)

    metadata = {
        "name": "lighttcn",
        "method": "direct_pytorch_to_keras_to_tflite",
        "checkpoint": {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
        },
        "tflite": {
            "path": str(tflite_path),
            "size_bytes": tflite_path.stat().st_size,
            "size_mb": tflite_path.stat().st_size / (1024 * 1024),
        },
        "input_shape": [1, 6, 200],
        "output_shape": [1, 2],
        "validation": {
            "torch_output": torch_out,
            "keras_output": keras_out,
            "max_abs_diff": max_abs_diff,
        },
        "notes": "Bypasses ONNX because onnx2tf mis-converts LightTCN causal chomp/residual axes.",
    }
    metadata_path = out_dir / "lighttcn_tflite_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {tflite_path}")
    print(f"Wrote {metadata_path}")
    print(f"PyTorch vs Keras max_abs_diff: {max_abs_diff:.8f}")


if __name__ == "__main__":
    main()
