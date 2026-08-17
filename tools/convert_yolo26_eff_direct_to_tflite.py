#!/usr/bin/env python3
"""
Directly convert the Efficient YOLOv26-1D PyTorch checkpoint to TFLite.

Same approach and rationale as convert_yolo26_direct_to_tflite.py: build a Keras
mirror of the PyTorch graph and copy weights across, bypassing ONNX/onnx2tf,
which mis-converts the neck's channel/time axes. Run on the laptop/desktop where
PyTorch and TensorFlow are available. No files under source/ are modified.

The backbone (stem + 3x C3k2 stages) is identical to YOLO26_1D_Regressor, so the
CBS / Bottleneck / C3k2 Keras layers are imported from the yolo26 converter
rather than duplicated -- they mirror the same torch classes and would otherwise
drift. Only the two modules that differ need new code here:

  EfficientPSA1D -> EfficientPSA : depthwise local conv + REDUCED-dim attention
                                   (256 -> 96 -> attend -> 256) + slim FFN
  LiteNeck1D     -> three 1x1 laterals to 96 ch before concat (288), fused to 192
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

# Shared Keras mirrors of the blocks both backbones have in common.
from convert_yolo26_direct_to_tflite import (
    CBS,
    C3k2,
    add_source_to_path,
    fixed_adaptive_avg_pool_time,
    pytorch_max_pool1d_nwc,
    set_dense,
    validate,
)


def load_torch_yolo26_eff(base_dir, checkpoint_path):
    add_source_to_path(base_dir)
    from ronin_yolo26_baseline_plain import get_model

    # 'yolo26_eff' always builds EfficientPSA1D; use_attention is ignored for it.
    model = get_model("yolo26_eff", model_dropout=0.5, use_attention=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, 6, 200))
    return model


def set_layernorm(norm, layer):
    layer.set_weights(
        [
            norm.weight.detach().cpu().numpy().astype(np.float32),
            norm.bias.detach().cpu().numpy().astype(np.float32),
        ]
    )


class EfficientPSA(tf.keras.layers.Layer):
    """Keras mirror of source/model_yolo26_1d.py::EfficientPSA1D.

    PyTorch forward (input NCW):
        x = x + local(x)                 # depthwise conv over time
        q = x.permute(0, 2, 1)           # -> NWC
        z = down(norm1(q))               # 256 -> attn_dim
        q = q + up(attn(z, z, z))        # attn_dim -> 256
        q = q + ffn(norm2(q))
        return q.permute(0, 2, 1)

    Here the tensor is already NWC, so the two permutes disappear. Note both
    residuals add onto the UN-normalised stream (pre-norm blocks).
    """

    def __init__(self, torch_psa, name):
        super().__init__(name=name)
        self.torch_psa = torch_psa

        self.channels = int(torch_psa.norm1.normalized_shape[0])
        self.attn_dim = int(torch_psa.attn.embed_dim)
        self.num_heads = int(torch_psa.attn.num_heads)
        self.head_dim = self.attn_dim // self.num_heads

        local = torch_psa.local
        self.local_pad = int(local.padding[0])
        self.local = tf.keras.layers.DepthwiseConv1D(
            kernel_size=local.kernel_size[0],
            strides=local.stride[0],
            padding="valid",
            depth_multiplier=1,
            use_bias=local.bias is not None,
            name=f"{name}_local",
        )

        self.norm1 = tf.keras.layers.LayerNormalization(
            epsilon=torch_psa.norm1.eps, name=f"{name}_norm1"
        )
        self.down = tf.keras.layers.Dense(self.attn_dim, name=f"{name}_down")

        self.q_dense = tf.keras.layers.Dense(self.attn_dim, name=f"{name}_q")
        self.k_dense = tf.keras.layers.Dense(self.attn_dim, name=f"{name}_k")
        self.v_dense = tf.keras.layers.Dense(self.attn_dim, name=f"{name}_v")
        self.out_dense = tf.keras.layers.Dense(self.attn_dim, name=f"{name}_out")

        self.up = tf.keras.layers.Dense(self.channels, name=f"{name}_up")

        self.norm2 = tf.keras.layers.LayerNormalization(
            epsilon=torch_psa.norm2.eps, name=f"{name}_norm2"
        )
        hidden = int(torch_psa.ffn[0].out_features)
        self.ffn1 = tf.keras.layers.Dense(hidden, activation=tf.nn.gelu, name=f"{name}_ffn1")
        self.ffn2 = tf.keras.layers.Dense(self.channels, name=f"{name}_ffn2")

    def split_heads(self, x):
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        x = tf.reshape(x, [b, t, self.num_heads, self.head_dim])
        return tf.transpose(x, [0, 2, 1, 3])

    def merge_heads(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        return tf.reshape(x, [b, t, self.attn_dim])

    def call(self, x, training=False):
        # x is NWC.
        local = x
        if self.local_pad:
            local = tf.pad(local, [[0, 0], [self.local_pad, self.local_pad], [0, 0]])
        x = x + self.local(local)

        z = self.down(self.norm1(x))
        q = self.split_heads(self.q_dense(z))
        k = self.split_heads(self.k_dense(z))
        v = self.split_heads(self.v_dense(z))
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.head_dim, tf.float32))
        attn = tf.nn.softmax(scores, axis=-1)
        a = self.out_dense(self.merge_heads(tf.matmul(attn, v)))
        x = x + self.up(a)

        return x + self.ffn2(self.ffn1(self.norm2(x)))

    def copy_weights(self):
        # torch depthwise Conv1d weight is (ch, 1, k); Keras wants (k, ch, 1).
        local_w = self.torch_psa.local.weight.detach().cpu().numpy()
        self.local.set_weights([np.transpose(local_w, (2, 0, 1)).astype(np.float32)])

        set_layernorm(self.torch_psa.norm1, self.norm1)
        set_dense(self.torch_psa.down, self.down)

        # nn.MultiheadAttention packs q, k, v into one in_proj matrix.
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

        set_dense(self.torch_psa.up, self.up)
        set_layernorm(self.torch_psa.norm2, self.norm2)
        set_dense(self.torch_psa.ffn[0], self.ffn1)
        set_dense(self.torch_psa.ffn[2], self.ffn2)


class Yolo26EffKeras(tf.keras.Model):
    """Keras mirror of YOLO26_1D_Efficient. Input (B, 6, 200) NCW -> (B, 2)."""

    def __init__(self, torch_model):
        super().__init__(name="yolo26_eff_direct")
        self.torch_model = torch_model

        self.stem0 = CBS(torch_model.stem[0], "stem0")
        pool = torch_model.stem[1]
        self.pool_kernel = int(pool.kernel_size)
        self.pool_stride = int(pool.stride)
        self.pool_pad = int(pool.padding)

        self.stage1 = C3k2(torch_model.stage1, "stage1")
        self.stage2 = C3k2(torch_model.stage2, "stage2")
        self.stage3 = C3k2(torch_model.stage3, "stage3")

        self.psa = EfficientPSA(torch_model.psa, "psa")

        # LiteNeck1D: unlike ELANNeck1D there is a third lateral (f3 is projected
        # down to fuse_dim too, instead of being concatenated at full width).
        self.lat1 = CBS(torch_model.neck.lat1, "neck_lat1")
        self.lat2 = CBS(torch_model.neck.lat2, "neck_lat2")
        self.lat3 = CBS(torch_model.neck.lat3, "neck_lat3")
        self.fuse = C3k2(torch_model.neck.fuse, "neck_fuse")

        self.dense1 = tf.keras.layers.Dense(
            int(torch_model.head[0].out_features), activation="relu", name="head_dense1"
        )
        self.dense2 = tf.keras.layers.Dense(
            int(torch_model.head[3].out_features), name="head_dense2"
        )

    def call(self, inputs, training=False):
        x = tf.transpose(inputs, [0, 2, 1])  # NCW -> NWC
        x = pytorch_max_pool1d_nwc(
            self.stem0(x, training=False),
            kernel_size=self.pool_kernel,
            stride=self.pool_stride,
            padding=self.pool_pad,
        )
        f1 = self.stage1(x, training=False)
        f2 = self.stage2(f1, training=False)
        f3 = self.stage3(f2, training=False)
        f3 = self.psa(f3, training=False)

        target_len = int(f3.shape[1])
        f1_up = self.lat1(fixed_adaptive_avg_pool_time(f1, target_len), training=False)
        f2_up = self.lat2(fixed_adaptive_avg_pool_time(f2, target_len), training=False)
        f3_p = self.lat3(f3, training=False)

        x = self.fuse(tf.concat([f1_up, f2_up, f3_p], axis=-1), training=False)
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
        self.lat3.copy_weights()
        self.fuse.copy_weights()
        set_dense(self.torch_model.head[0], self.dense1)
        set_dense(self.torch_model.head[3], self.dense2)


def first_existing(base_dir, candidates):
    for candidate in candidates:
        path = base_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find an Efficient YOLOv26-1D checkpoint; pass --checkpoint explicitly"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=".")
    parser.add_argument("--out_dir", default="models_tflite")
    parser.add_argument("--checkpoint", default=None,
                        help="path to the .pt checkpoint (relative to --base_dir)")
    parser.add_argument("--max_abs_diff", type=float, default=1e-4,
                        help="fail if PyTorch vs Keras disagree by more than this")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = (base_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        checkpoint_path = (base_dir / args.checkpoint).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_path = first_existing(
            base_dir,
            [
                "output/yolo26_eff_adam_v1/checkpoints/checkpoint_59.pt",
                "output/yolo26_eff_adam_v1/checkpoints/checkpoint_latest.pt",
            ],
        )

    torch_model = load_torch_yolo26_eff(base_dir, checkpoint_path)
    n_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
    print(f"Loaded {checkpoint_path} ({n_params:,} params)")

    keras_model = Yolo26EffKeras(torch_model)
    _ = keras_model(np.zeros((1, 6, 200), dtype=np.float32), training=False)
    keras_model.copy_weights()

    torch_out, keras_out, max_abs_diff = validate(torch_model, keras_model)
    print(f"PyTorch vs Keras max_abs_diff: {max_abs_diff:.8f}")
    if max_abs_diff > args.max_abs_diff:
        raise SystemExit(
            f"Weight transfer mismatch: {max_abs_diff:.8f} > {args.max_abs_diff:g}. "
            "Refusing to write a TFLite model that does not match the checkpoint."
        )

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    tflite_path = out_dir / "yolo26_eff.tflite"
    tflite_path.write_bytes(tflite_model)

    metadata = {
        "name": "yolo26_eff",
        "method": "direct_pytorch_to_keras_to_tflite",
        "neural_params": n_params,
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
        "notes": (
            "Bypasses ONNX because onnx2tf mis-converts the YOLO26 neck channel/time axes. "
            "EfficientPSA1D uses reduced-dimension attention (256->96->256) plus a depthwise "
            "temporal conv; LiteNeck1D projects all three scales to 96 ch before fusing."
        ),
    }
    metadata_path = out_dir / "yolo26_eff_tflite_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {tflite_path} ({tflite_path.stat().st_size / (1024 * 1024):.2f} MiB)")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
