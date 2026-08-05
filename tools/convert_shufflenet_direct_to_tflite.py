#!/usr/bin/env python3
"""
Directly convert the ShuffleNetV2-1D PyTorch checkpoint to TFLite.

This bypasses ONNX/onnx2tf because ShuffleNet's channel_shuffle reshape path can
produce zero-sized dynamic dimensions during ONNX-to-TF tracing. Run on the
laptop/desktop where PyTorch and TensorFlow are available. No files under
source/ are modified.
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


def load_torch_shufflenet(base_dir, checkpoint_path):
    add_source_to_path(base_dir)
    from ronin_yolo26_baseline_plain import get_model

    model = get_model("shufflenetv2", model_dropout=0.2, use_attention=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, 6, 200))
    return model


def first_existing(base_dir, candidates):
    for candidate in candidates:
        path = base_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError("Could not find ShuffleNet checkpoint")


def conv_weights(conv):
    weight = conv.weight.detach().cpu().numpy()  # out, in/group, kernel
    kernel = np.transpose(weight, (2, 1, 0)).astype(np.float32)
    if conv.bias is None:
        return [kernel]
    return [kernel, conv.bias.detach().cpu().numpy().astype(np.float32)]


def set_conv(conv, layer):
    layer.set_weights(conv_weights(conv))


def set_bn(bn, layer):
    layer.set_weights(
        [
            bn.weight.detach().cpu().numpy().astype(np.float32),
            bn.bias.detach().cpu().numpy().astype(np.float32),
            bn.running_mean.detach().cpu().numpy().astype(np.float32),
            bn.running_var.detach().cpu().numpy().astype(np.float32),
        ]
    )


def set_dense(linear, layer):
    layer.set_weights(
        [
            linear.weight.detach().cpu().numpy().T.astype(np.float32),
            linear.bias.detach().cpu().numpy().astype(np.float32),
        ]
    )


def pytorch_max_pool1d_nwc(x, kernel_size=3, stride=2, padding=1):
    if padding:
        x = tf.pad(
            x,
            [[0, 0], [padding, padding], [0, 0]],
            constant_values=tf.constant(-3.4028234663852886e38, dtype=tf.float32),
        )
    return tf.nn.max_pool1d(x, ksize=kernel_size, strides=stride, padding="VALID")


def channel_shuffle_nwc(x, groups=2):
    shape = tf.shape(x)
    b = shape[0]
    t = shape[1]
    c = int(x.shape[-1])
    x = tf.reshape(x, [b, t, groups, c // groups])
    x = tf.transpose(x, [0, 1, 3, 2])
    return tf.reshape(x, [b, t, c])


class ConvBNAct(tf.keras.layers.Layer):
    def __init__(self, torch_block, name):
        super().__init__(name=name)
        self.torch_block = torch_block
        conv = torch_block.conv
        self.pad = int(conv.padding[0])
        self.conv = tf.keras.layers.Conv1D(
            filters=conv.out_channels,
            kernel_size=conv.kernel_size[0],
            strides=conv.stride[0],
            padding="valid",
            use_bias=conv.bias is not None,
            name=f"{name}_conv",
        )
        self.bn = tf.keras.layers.BatchNormalization(
            epsilon=torch_block.bn.eps,
            momentum=1.0 - torch_block.bn.momentum,
            name=f"{name}_bn",
        )
        self.use_relu = not torch_block.act.__class__.__name__ == "Identity"

    def call(self, x, training=False):
        if self.pad:
            x = tf.pad(x, [[0, 0], [self.pad, self.pad], [0, 0]])
        x = self.conv(x)
        x = self.bn(x, training=False)
        if self.use_relu:
            x = tf.nn.relu(x)
        return x

    def copy_weights(self):
        set_conv(self.torch_block.conv, self.conv)
        set_bn(self.torch_block.bn, self.bn)


class DWConvBN(tf.keras.layers.Layer):
    def __init__(self, torch_block, name):
        super().__init__(name=name)
        self.torch_block = torch_block
        conv = torch_block.conv
        self.pad = int(conv.padding[0])
        self.conv = tf.keras.layers.Conv1D(
            filters=conv.out_channels,
            kernel_size=conv.kernel_size[0],
            strides=conv.stride[0],
            padding="valid",
            groups=conv.groups,
            use_bias=conv.bias is not None,
            name=f"{name}_conv",
        )
        self.bn = tf.keras.layers.BatchNormalization(
            epsilon=torch_block.bn.eps,
            momentum=1.0 - torch_block.bn.momentum,
            name=f"{name}_bn",
        )

    def call(self, x, training=False):
        if self.pad:
            x = tf.pad(x, [[0, 0], [self.pad, self.pad], [0, 0]])
        x = self.conv(x)
        return self.bn(x, training=False)

    def copy_weights(self):
        set_conv(self.torch_block.conv, self.conv)
        set_bn(self.torch_block.bn, self.bn)


class SequentialList(tf.keras.layers.Layer):
    def __init__(self, layers, name):
        super().__init__(name=name)
        self.layers_list = layers

    def call(self, x, training=False):
        for layer in self.layers_list:
            x = layer(x, training=False)
        return x

    def copy_weights(self):
        for layer in self.layers_list:
            layer.copy_weights()


def convert_torch_sequence(torch_seq, name):
    layers = []
    for idx, item in enumerate(torch_seq):
        class_name = item.__class__.__name__
        if class_name == "ConvBNAct1D":
            layers.append(ConvBNAct(item, f"{name}_{idx}_convbnact"))
        elif class_name == "DWConvBN1D":
            layers.append(DWConvBN(item, f"{name}_{idx}_dwconvbn"))
        else:
            raise ValueError(f"Unsupported layer in ShuffleNet branch: {class_name}")
    return SequentialList(layers, name)


class ShuffleBlock(tf.keras.layers.Layer):
    def __init__(self, torch_block, name):
        super().__init__(name=name)
        self.torch_block = torch_block
        self.stride = torch_block.stride
        self.branch2 = convert_torch_sequence(torch_block.branch2, f"{name}_branch2")
        self.branch1 = None
        if self.stride != 1:
            self.branch1 = convert_torch_sequence(torch_block.branch1, f"{name}_branch1")

    def call(self, x, training=False):
        if self.stride == 1:
            c = int(x.shape[-1])
            x1 = x[:, :, : c // 2]
            x2 = x[:, :, c // 2 :]
            out = tf.concat([x1, self.branch2(x2, training=False)], axis=-1)
        else:
            out = tf.concat(
                [
                    self.branch1(x, training=False),
                    self.branch2(x, training=False),
                ],
                axis=-1,
            )
        return channel_shuffle_nwc(out, groups=2)

    def copy_weights(self):
        self.branch2.copy_weights()
        if self.branch1 is not None:
            self.branch1.copy_weights()


class ShuffleNetKeras(tf.keras.Model):
    def __init__(self, torch_model):
        super().__init__(name="shufflenet_direct")
        self.torch_model = torch_model
        self.stem = ConvBNAct(torch_model.stem, "stem")
        self.stages = []
        for stage_idx, stage in enumerate(torch_model.stages):
            stage_layers = []
            for block_idx, block in enumerate(stage):
                stage_layers.append(ShuffleBlock(block, f"stage{stage_idx}_block{block_idx}"))
            self.stages.append(stage_layers)
        self.tail = ConvBNAct(torch_model.tail, "tail")
        self.dense1 = tf.keras.layers.Dense(256, activation="relu", name="head_dense1")
        self.dense2 = tf.keras.layers.Dense(2, name="head_dense2")

    def call(self, inputs, training=False):
        x = tf.transpose(inputs, [0, 2, 1])  # NCW -> NWC
        x = self.stem(x, training=False)
        x = pytorch_max_pool1d_nwc(x, kernel_size=3, stride=2, padding=1)
        for stage in self.stages:
            for block in stage:
                x = block(x, training=False)
        x = self.tail(x, training=False)
        x = tf.reduce_mean(x, axis=1)
        return self.dense2(self.dense1(x))

    def copy_weights(self):
        self.stem.copy_weights()
        for stage in self.stages:
            for block in stage:
                block.copy_weights()
        self.tail.copy_weights()
        set_dense(self.torch_model.head[0], self.dense1)
        set_dense(self.torch_model.head[3], self.dense2)


def validate(torch_model, keras_model):
    rng = np.random.default_rng(13)
    sample = rng.normal(size=(1, 6, 200)).astype(np.float32)
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(sample)).detach().cpu().numpy()
    keras_out = keras_model(sample, training=False).numpy()
    max_abs_diff = float(np.max(np.abs(torch_out - keras_out)))
    return torch_out.tolist(), keras_out.tolist(), max_abs_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=".")
    parser.add_argument("--out_dir", default="models_tflite")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = (base_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = first_existing(
        base_dir,
        [
            "models/shufflenet_checkpoint_63.pt",
            "output/train_shufflenetv2/checkpoints/shufflenet_checkpoint_63.pt",
        ],
    )

    torch_model = load_torch_shufflenet(base_dir, checkpoint_path)
    keras_model = ShuffleNetKeras(torch_model)
    _ = keras_model(np.zeros((1, 6, 200), dtype=np.float32), training=False)
    keras_model.copy_weights()
    torch_out, keras_out, max_abs_diff = validate(torch_model, keras_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    tflite_path = out_dir / "shufflenet.tflite"
    tflite_path.write_bytes(tflite_model)

    metadata = {
        "name": "shufflenet",
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
        "notes": "Bypasses ONNX because onnx2tf produces zero-sized dimensions for ShuffleNet channel shuffle.",
    }
    metadata_path = out_dir / "shufflenet_tflite_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {tflite_path}")
    print(f"Wrote {metadata_path}")
    print(f"PyTorch vs Keras max_abs_diff: {max_abs_diff:.8f}")


if __name__ == "__main__":
    main()
