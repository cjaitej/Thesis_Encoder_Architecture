#!/usr/bin/env python3
"""
Export RoNIN PyTorch checkpoints to ONNX.

Run this on a laptop/desktop where PyTorch is already installed. This script
does not modify any files under source/.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_SPECS = {
    "resnet": {
        "script": "ronin_resnet_baseline_plain",
        "checkpoints": ["ronin_resnet/checkpoint_gsn_latest.pt"],
        "backbone": None,
        "use_attention": False,
        "dropout": 0.2,
    },
    "yolo26": {
        "script": "ronin_yolo26_baseline_plain",
        "checkpoints": [
            "models/yolov26_checkpoint_35.pt",
            "output/yolo26_adam_attn_t2_r3/checkpoints/yolov26_checkpoint_35.pt",
            "output/yolo26_adam_attn_t2_r3/checkpoints/checkpoint_35.pt",
        ],
        "backbone": "yolo26",
        "use_attention": True,
        "dropout": 0.2,
    },
    "mobilenet": {
        "script": "ronin_yolo26_baseline_plain",
        "checkpoints": [
            "models/mobilenet_checkpoint_44.pt",
            "output/train_mobilenetv2/checkpoints/mobilenet_checkpoint_44.pt",
        ],
        "backbone": "mobilenetv2",
        "use_attention": False,
        "dropout": 0.2,
    },
    "shufflenet": {
        "script": "ronin_yolo26_baseline_plain",
        "checkpoints": [
            "models/shufflenet_checkpoint_63.pt",
            "output/train_shufflenetv2/checkpoints/shufflenet_checkpoint_63.pt",
        ],
        "backbone": "shufflenetv2",
        "use_attention": False,
        "dropout": 0.2,
    },
    "tinycnn": {
        "script": "ronin_yolo26_baseline_plain",
        "checkpoints": [
            "models/tinycnn_checkpoint_83.pt",
            "output/train_tinycnn/checkpoints/tinycnn_checkpoint_83.pt",
        ],
        "backbone": "tinycnn",
        "use_attention": False,
        "dropout": 0.2,
    },
    "lighttcn": {
        "script": "ronin_yolo26_baseline_plain",
        "checkpoints": [
            "models/lighttcn_checkpoint_42.pt",
            "output/train_lighttcn/checkpoints/lighttcn_checkpoint_42.pt",
        ],
        "backbone": "lighttcn",
        "use_attention": False,
        "dropout": 0.2,
    },
}


def load_module(base_dir, module_name):
    source_dir = base_dir / "source"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    return __import__(module_name)


def build_model(base_dir, name, spec, window_size):
    module = load_module(base_dir, spec["script"])

    if hasattr(module, "_fc_config"):
        module._fc_config["in_dim"] = window_size // 32 + 1

    if spec["backbone"] is None:
        model = module.get_model("resnet18")
    else:
        model = module.get_model(
            spec["backbone"],
            model_dropout=spec["dropout"],
            use_attention=spec["use_attention"],
        )

    checkpoint_path = None
    for candidate in spec["checkpoints"]:
        path = base_dir / candidate
        if path.exists():
            checkpoint_path = path
            break
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint found for {name}. Tried: "
            + ", ".join(str(base_dir / candidate) for candidate in spec["checkpoints"])
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint_path


class FixedYoloNeckExport(nn.Module):
    def __init__(self, neck, target_len):
        super().__init__()
        self.lat1 = neck.lat1
        self.lat2 = neck.lat2
        self.fuse = neck.fuse
        self.target_len = int(target_len)

    @staticmethod
    def fixed_adaptive_avg_pool1d(x, output_size):
        input_size = int(x.shape[-1])
        bins = []
        for i in range(output_size):
            start = int(math.floor(i * input_size / output_size))
            end = int(math.ceil((i + 1) * input_size / output_size))
            bins.append(x[..., start:end].mean(dim=-1, keepdim=True))
        return torch.cat(bins, dim=-1)

    def forward(self, f1, f2, f3):
        f1_up = self.fixed_adaptive_avg_pool1d(f1, self.target_len)
        f2_up = self.fixed_adaptive_avg_pool1d(f2, self.target_len)
        f1_up = self.lat1(f1_up)
        f2_up = self.lat2(f2_up)
        return self.fuse(torch.cat([f1_up, f2_up, f3], dim=1))


def prepare_for_fixed_export(model, name, dummy):
    """Apply export-only fixes. Does not modify source files."""
    if name != "yolo26" or not hasattr(model, "neck"):
        return None

    with torch.no_grad():
        x = model.stem(dummy)
        f1 = model.stage1(x)
        f2 = model.stage2(f1)
        f3 = model.stage3(f2)
        target_len = f3.shape[-1]

    original_neck = model.neck
    model.neck = FixedYoloNeckExport(model.neck, target_len)
    return original_neck


class ExportForwardWrapper(nn.Module):
    """Export-only forwards that avoid AdaptiveAvgPool1d(1).squeeze(-1)."""

    def __init__(self, model, name):
        super().__init__()
        self.model = model
        self.name = name

    @staticmethod
    def causal_conv1d(x, conv):
        pad = conv.padding[0]
        if pad:
            x = F.pad(x, (pad, 0))
        return F.conv1d(
            x,
            conv.weight,
            conv.bias,
            stride=conv.stride,
            padding=0,
            dilation=conv.dilation,
            groups=conv.groups,
        )

    def lighttcn_block(self, x, block):
        out = self.causal_conv1d(x, block.conv1)
        out = block.relu1(out)
        out = self.causal_conv1d(out, block.conv2)
        out = block.relu2(out)
        res = x if block.downsample is None else block.downsample(x)
        return block.relu(out + res)

    @staticmethod
    def fixed_channel_shuffle(x, groups=2):
        # Export-only fixed-shape version of source/model_shufflenet1d.py:channel_shuffle.
        b = int(x.shape[0])
        c = int(x.shape[1])
        t = int(x.shape[2])
        x = x.reshape(b, groups, c // groups, t)
        x = x.transpose(1, 2).contiguous()
        return x.reshape(b, c, t)

    def shufflenet_block(self, x, block):
        if block.stride == 1:
            x1, x2 = torch.chunk(x, 2, dim=1)
            out = torch.cat([x1, block.branch2(x2)], dim=1)
        else:
            out = torch.cat([block.branch1(x), block.branch2(x)], dim=1)
        return self.fixed_channel_shuffle(out, 2)

    def forward(self, x):
        m = self.model
        if self.name == "yolo26":
            x = m.stem(x)
            f1 = m.stage1(x)
            f2 = m.stage2(f1)
            f3 = m.stage3(f2)
            f3 = m.psa(f3)
            x = m.neck(f1, f2, f3)
            x = x.mean(dim=-1)
            return m.head(x)
        if self.name == "mobilenet":
            x = m.stem(x)
            x = m.features(x)
            x = m.tail(x)
            x = x.mean(dim=-1)
            return m.head(x)
        if self.name == "shufflenet":
            x = m.stem(x)
            x = m.maxpool(x)
            for stage in m.stages:
                for block in stage:
                    x = self.shufflenet_block(x, block)
            x = m.tail(x)
            x = x.mean(dim=-1)
            return m.head(x)
        if self.name == "tinycnn":
            x = m.stem(x)
            x = m.blocks(x)
            x = x.mean(dim=-1)
            return m.head(x)
        if self.name == "lighttcn":
            for block in m.tcn.network:
                x = self.lighttcn_block(x, block)
            x = x.mean(dim=-1)
            return m.head(x)
        return m(x)


def file_info(path):
    size_bytes = path.stat().st_size if path.exists() else 0
    return {
        "path": str(path),
        "size_bytes": size_bytes,
        "size_mb": size_bytes / (1024 * 1024),
    }


def pytorch_param_details(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffer_values = sum(b.numel() for b in model.buffers())
    return {
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "buffer_values": int(buffer_values),
        "fp32_parameter_size_bytes": int(total_params * 4),
        "fp32_parameter_size_mb": (total_params * 4) / (1024 * 1024),
    }


def shape_from_onnx_value(value):
    dims = []
    tensor_type = value.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def onnx_model_details(onnx_path):
    import onnx

    model = onnx.load(str(onnx_path))
    graph = model.graph
    initializer_count = 0
    initializer_values = 0
    for initializer in graph.initializer:
        initializer_count += 1
        values = 1
        for dim in initializer.dims:
            values *= int(dim)
        initializer_values += values

    return {
        "opset_imports": [
            {"domain": opset.domain or "ai.onnx", "version": int(opset.version)}
            for opset in model.opset_import
        ],
        "graph_name": graph.name,
        "inputs": [
            {"name": value.name, "shape": shape_from_onnx_value(value)}
            for value in graph.input
        ],
        "outputs": [
            {"name": value.name, "shape": shape_from_onnx_value(value)}
            for value in graph.output
        ],
        "initializer_tensors": initializer_count,
        "initializer_values": int(initializer_values),
        "fp32_initializer_size_bytes": int(initializer_values * 4),
        "fp32_initializer_size_mb": (initializer_values * 4) / (1024 * 1024),
        "node_count": len(graph.node),
    }


def export_one(base_dir, out_dir, name, spec, window_size, opset):
    model, checkpoint_path = build_model(base_dir, name, spec, window_size)
    dummy = torch.randn(1, 6, window_size, dtype=torch.float32)
    out_path = out_dir / f"{name}.onnx"
    pytorch_details = pytorch_param_details(model)

    with torch.no_grad():
        torch_output = model(dummy)

    original_neck = prepare_for_fixed_export(model, name, dummy)
    export_model = ExportForwardWrapper(model, name)
    torch.onnx.export(
        export_model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
    )
    if original_neck is not None:
        model.neck = original_neck

    metadata = {
        "name": name,
        "backbone": spec["backbone"] or "resnet18",
        "script": spec["script"],
        "checkpoint": file_info(checkpoint_path),
        "onnx": file_info(out_path),
        "export": {
            "opset": opset,
            "window_size": window_size,
            "input_shape_used_for_export": [1, 6, window_size],
            "torch_output_shape": list(torch_output.shape),
            "fixed_shape_export": True,
            "notes": (
                "YOLO26 uses an export-only fixed pooling wrapper. "
                "No source model files are modified."
                if name == "yolo26"
                else "No export-only model wrapper was required."
            ),
        },
        "pytorch_model": pytorch_details,
        "onnx_model": onnx_model_details(out_path),
    }

    metadata_path = out_dir / f"{name}_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported {name}: {checkpoint_path} -> {out_path}")
    print(f"Wrote metadata: {metadata_path}")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=".", help="Project root containing source/, models/, ronin_resnet/")
    parser.add_argument("--out_dir", default="models_onnx", help="Output directory for ONNX files")
    parser.add_argument("--which", default="all", help="all or one model name")
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = (base_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = list(MODEL_SPECS) if args.which == "all" else [args.which]
    unknown = [name for name in selected if name not in MODEL_SPECS]
    if unknown:
        raise SystemExit(f"Unknown model(s): {', '.join(unknown)}")

    manifest = []
    for name in selected:
        manifest.append(export_one(base_dir, out_dir, name, MODEL_SPECS[name], args.window_size, args.opset))

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
