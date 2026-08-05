#!/usr/bin/env python3
"""
Directly convert the YOLO26 PyTorch checkpoint to TFLite.

This bypasses ONNX/onnx2tf because the YOLO26 neck/concat path is prone to
NCW/NWC layout errors during ONNX-to-TF conversion. Run on the laptop/desktop
where PyTorch and TensorFlow are available. No files under source/ are modified.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch


def add_source_to_path(base_dir):
    source_dir = base_dir / "source"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))


def load_torch_yolo26(base_dir, checkpoint_path):
    add_source_to_path(base_dir)
    from ronin_yolo26_baseline_plain import get_model

    model = get_model("yolo26", model_dropout=0.2, use_attention=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, 6, 200))
    return model


def conv_weights(conv):
    weight = conv.weight.detach().cpu().numpy()
    kernel = np.transpose(weight, (2, 1, 0)).astype(np.float32)
    if conv.bias is None:
        return [kernel]
    else:
        bias = conv.bias.detach().cpu().numpy().astype(np.float32)
        return [kernel, bias]


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


class CBS(tf.keras.layers.Layer):
    def __init__(self, torch_cbs, name):
        super().__init__(name=name)
        self.torch_cbs = torch_cbs
        conv = torch_cbs.conv
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
            epsilon=torch_cbs.bn.eps,
            momentum=1.0 - torch_cbs.bn.momentum,
            name=f"{name}_bn",
        )

    def call(self, x, training=False):
        if self.pad:
            x = tf.pad(x, [[0, 0], [self.pad, self.pad], [0, 0]])
        x = self.conv(x)
        x = self.bn(x, training=False)
        return tf.nn.silu(x)

    def copy_weights(self):
        set_conv(self.torch_cbs.conv, self.conv)
        set_bn(self.torch_cbs.bn, self.bn)


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, torch_block, name):
        super().__init__(name=name)
        self.torch_block = torch_block
        self.cv1 = CBS(torch_block.cv1, f"{name}_cv1")
        self.cv2 = CBS(torch_block.cv2, f"{name}_cv2")
        self.cv3 = CBS(torch_block.cv3, f"{name}_cv3")

    def call(self, x, training=False):
        out = self.cv3(self.cv2(self.cv1(x, training=False), training=False), training=False)
        if self.torch_block.use_shortcut:
            return x + out
        return out

    def copy_weights(self):
        self.cv1.copy_weights()
        self.cv2.copy_weights()
        self.cv3.copy_weights()


class C3k2(tf.keras.layers.Layer):
    def __init__(self, torch_block, name):
        super().__init__(name=name)
        self.torch_block = torch_block
        self.ds = None if torch_block.ds.__class__.__name__ == "Identity" else CBS(torch_block.ds, f"{name}_ds")
        self.cv1 = CBS(torch_block.cv1, f"{name}_cv1")
        self.cv2 = CBS(torch_block.cv2, f"{name}_cv2")
        self.bottlenecks = [
            Bottleneck(block, f"{name}_bottleneck_{idx}")
            for idx, block in enumerate(torch_block.bottlenecks)
        ]
        self.cv_out = CBS(torch_block.cv_out, f"{name}_cv_out")

    def call(self, x, training=False):
        if self.ds is not None:
            x = self.ds(x, training=False)
        main = self.cv1(x, training=False)
        for block in self.bottlenecks:
            main = block(main, training=False)
        skip = self.cv2(x, training=False)
        return self.cv_out(tf.concat([main, skip], axis=-1), training=False)

    def copy_weights(self):
        if self.ds is not None:
            self.ds.copy_weights()
        self.cv1.copy_weights()
        self.cv2.copy_weights()
        for block in self.bottlenecks:
            block.copy_weights()
        self.cv_out.copy_weights()


class PSA(tf.keras.layers.Layer):
    def __init__(self, torch_psa, name):
        super().__init__(name=name)
        self.torch_psa = torch_psa
        self.embed_dim = int(torch_psa.attn.embed_dim)
        self.num_heads = int(torch_psa.attn.num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.q_dense = tf.keras.layers.Dense(self.embed_dim, name=f"{name}_q")
        self.k_dense = tf.keras.layers.Dense(self.embed_dim, name=f"{name}_k")
        self.v_dense = tf.keras.layers.Dense(self.embed_dim, name=f"{name}_v")
        self.out_dense = tf.keras.layers.Dense(self.embed_dim, name=f"{name}_out")
        self.norm = tf.keras.layers.LayerNormalization(epsilon=torch_psa.norm.eps, name=f"{name}_norm")
        self.ffn1 = tf.keras.layers.Dense(self.embed_dim * 2, activation=tf.nn.gelu, name=f"{name}_ffn1")
        self.ffn2 = tf.keras.layers.Dense(self.embed_dim, name=f"{name}_ffn2")

    def split_heads(self, x):
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        x = tf.reshape(x, [b, t, self.num_heads, self.head_dim])
        return tf.transpose(x, [0, 2, 1, 3])

    def merge_heads(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        return tf.reshape(x, [b, t, self.embed_dim])

    def call(self, x, training=False):
        # x is NWC. PyTorch PSA receives NCW, permutes to NWC, then returns NCW.
        q = self.split_heads(self.q_dense(x))
        k = self.split_heads(self.k_dense(x))
        v = self.split_heads(self.v_dense(x))
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.head_dim, tf.float32))
        attn = tf.nn.softmax(scores, axis=-1)
        a = self.out_dense(self.merge_heads(tf.matmul(attn, v)))
        x = self.norm(x + a)
        return x + self.ffn2(self.ffn1(x))

    def copy_weights(self):
        in_w = self.torch_psa.attn.in_proj_weight.detach().cpu().numpy()
        in_b = self.torch_psa.attn.in_proj_bias.detach().cpu().numpy()
        q_w, k_w, v_w = np.split(in_w, 3, axis=0)
        q_b, k_b, v_b = np.split(in_b, 3, axis=0)
        self.q_dense.set_weights([q_w.T.astype(np.float32), q_b.astype(np.float32)])
        self.k_dense.set_weights([k_w.T.astype(np.float32), k_b.astype(np.float32)])
        self.v_dense.set_weights([v_w.T.astype(np.float32), v_b.astype(np.float32)])
        self.out_dense.set_weights(
            [
                self.torch_psa.attn.out_proj.weight.detach().cpu().numpy().T.astype(np.float32),
                self.torch_psa.attn.out_proj.bias.detach().cpu().numpy().astype(np.float32),
            ]
        )
        self.norm.set_weights(
            [
                self.torch_psa.norm.weight.detach().cpu().numpy().astype(np.float32),
                self.torch_psa.norm.bias.detach().cpu().numpy().astype(np.float32),
            ]
        )
        set_dense(self.torch_psa.ffn[0], self.ffn1)
        set_dense(self.torch_psa.ffn[2], self.ffn2)


class Yolo26Keras(tf.keras.Model):
    def __init__(self, torch_model):
        super().__init__(name="yolo26_direct")
        self.torch_model = torch_model
        self.stem0 = CBS(torch_model.stem[0], "stem0")
        self.stage1 = C3k2(torch_model.stage1, "stage1")
        self.stage2 = C3k2(torch_model.stage2, "stage2")
        self.stage3 = C3k2(torch_model.stage3, "stage3")
        self.psa = PSA(torch_model.psa, "psa")
        self.lat1 = CBS(torch_model.neck.lat1, "neck_lat1")
        self.lat2 = CBS(torch_model.neck.lat2, "neck_lat2")
        self.fuse = C3k2(torch_model.neck.fuse, "neck_fuse")
        self.dense1 = tf.keras.layers.Dense(128, activation="relu", name="head_dense1")
        self.dense2 = tf.keras.layers.Dense(2, name="head_dense2")

    def call(self, inputs, training=False):
        x = tf.transpose(inputs, [0, 2, 1])  # NCW -> NWC
        x = pytorch_max_pool1d_nwc(self.stem0(x, training=False), kernel_size=3, stride=2, padding=1)
        f1 = self.stage1(x, training=False)
        f2 = self.stage2(f1, training=False)
        f3 = self.stage3(f2, training=False)
        f3 = self.psa(f3, training=False)
        target_len = int(f3.shape[1])
        f1_up = fixed_adaptive_avg_pool_time(f1, target_len)
        f2_up = fixed_adaptive_avg_pool_time(f2, target_len)
        f1_up = self.lat1(f1_up, training=False)
        f2_up = self.lat2(f2_up, training=False)
        x = self.fuse(tf.concat([f1_up, f2_up, f3], axis=-1), training=False)
        x = tf.reduce_mean(x, axis=1)
        return self.dense2(self.dense1(x))

    def copy_weights(self):
        self.stem0.copy_weights()
        self.stage1.copy_weights()
        self.stage2.copy_weights()
        self.stage3.copy_weights()
        self.psa.copy_weights()
        self.lat1.copy_weights()
        self.lat2.copy_weights()
        self.fuse.copy_weights()
        set_dense(self.torch_model.head[0], self.dense1)
        set_dense(self.torch_model.head[3], self.dense2)


def fixed_adaptive_avg_pool_time(x, output_size):
    input_size = int(x.shape[1])
    bins = []
    for i in range(output_size):
        start = int(math.floor(i * input_size / output_size))
        end = int(math.ceil((i + 1) * input_size / output_size))
        bins.append(tf.reduce_mean(x[:, start:end, :], axis=1, keepdims=True))
    return tf.concat(bins, axis=1)


def pytorch_max_pool1d_nwc(x, kernel_size=3, stride=2, padding=1):
    if padding:
        x = tf.pad(
            x,
            [[0, 0], [padding, padding], [0, 0]],
            constant_values=tf.constant(-3.4028234663852886e38, dtype=tf.float32),
        )
    return tf.nn.max_pool1d(x, ksize=kernel_size, strides=stride, padding="VALID")


def validate(torch_model, keras_model):
    rng = np.random.default_rng(11)
    sample = rng.normal(size=(1, 6, 200)).astype(np.float32)
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(sample)).detach().cpu().numpy()
    keras_out = keras_model(sample, training=False).numpy()
    max_abs_diff = float(np.max(np.abs(torch_out - keras_out)))
    return torch_out.tolist(), keras_out.tolist(), max_abs_diff


def first_existing(base_dir, candidates):
    for candidate in candidates:
        path = base_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError("Could not find YOLO26 checkpoint")


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
            "models/yolov26_checkpoint_35.pt",
            "output/yolo26_adam_attn_t2_r3/checkpoints/yolov26_checkpoint_35.pt",
            "output/yolo26_adam_attn_t2_r3/checkpoints/checkpoint_35.pt",
        ],
    )

    torch_model = load_torch_yolo26(base_dir, checkpoint_path)
    keras_model = Yolo26Keras(torch_model)
    _ = keras_model(np.zeros((1, 6, 200), dtype=np.float32), training=False)
    keras_model.copy_weights()
    torch_out, keras_out, max_abs_diff = validate(torch_model, keras_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    tflite_path = out_dir / "yolo26.tflite"
    tflite_path.write_bytes(tflite_model)

    metadata = {
        "name": "yolo26",
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
        "notes": "Bypasses ONNX because onnx2tf mis-converts YOLO26 neck channel/time axes.",
    }
    metadata_path = out_dir / "yolo26_tflite_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {tflite_path}")
    print(f"Wrote {metadata_path}")
    print(f"PyTorch vs Keras max_abs_diff: {max_abs_diff:.8f}")


if __name__ == "__main__":
    main()
