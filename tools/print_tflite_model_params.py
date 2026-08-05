#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_MODELS = [
    "resnet",
    "yolo26",
    "mobilenet",
    "shufflenet",
    "tinycnn",
    "lighttcn",
]


def size_mb(num_bytes):
    return num_bytes / (1024 * 1024)


def load_metadata(tflite_path):
    metadata_path = tflite_path.with_name(f"{tflite_path.stem}_tflite_metadata.json")
    if not metadata_path.exists():
        return None
    with metadata_path.open() as f:
        return json.load(f)


def tensor_shape_numel(tensor):
    shape = [int(tensor.Shape(i)) for i in range(tensor.ShapeLength())]
    if not shape:
        return 1
    return int(np.prod(shape))


def dtype_name(schema, dtype_code):
    for name in dir(schema.TensorType):
        if name.startswith("_"):
            continue
        if getattr(schema.TensorType, name) == dtype_code:
            return name
    return str(dtype_code)


def parse_tflite_parameters(tflite_path):
    try:
        from tensorflow.lite.python import schema_py_generated as schema
    except Exception:
        return None

    data = tflite_path.read_bytes()
    model = schema.Model.GetRootAsModel(data, 0)
    if model.SubgraphsLength() == 0:
        return None

    subgraph = model.Subgraphs(0)
    input_tensors = {subgraph.Inputs(i) for i in range(subgraph.InputsLength())}
    output_tensors = {subgraph.Outputs(i) for i in range(subgraph.OutputsLength())}

    param_count = 0
    param_bytes = 0
    param_tensors = 0
    dtype_counts = {}

    for tensor_idx in range(subgraph.TensorsLength()):
        if tensor_idx in input_tensors or tensor_idx in output_tensors:
            continue

        tensor = subgraph.Tensors(tensor_idx)
        buffer_idx = tensor.Buffer()
        if buffer_idx < 0 or buffer_idx >= model.BuffersLength():
            continue

        buffer = model.Buffers(buffer_idx)
        buffer_bytes = buffer.DataLength()
        if buffer_bytes <= 0:
            continue

        numel = tensor_shape_numel(tensor)
        dtype = dtype_name(schema, tensor.Type())

        param_count += numel
        param_bytes += buffer_bytes
        param_tensors += 1
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + numel

    return {
        "parameter_count": param_count,
        "parameter_tensors": param_tensors,
        "parameter_bytes": param_bytes,
        "dtype_counts": dtype_counts,
    }


def metadata_param_count(metadata):
    if not metadata:
        return None

    direct_count = metadata.get("parameter_count")
    if direct_count is not None:
        return direct_count

    onnx_meta = metadata.get("onnx_export_metadata") or {}
    for key in ("onnx_initializer_count", "pytorch_parameter_count"):
        value = onnx_meta.get(key)
        if value is not None:
            return value
    return None


def shape_from_metadata(metadata, key):
    if not metadata:
        return ""
    value = metadata.get(key)
    if value is not None:
        return str(value)
    onnx_meta = metadata.get("onnx_export_metadata") or {}
    value = onnx_meta.get(key)
    return str(value) if value is not None else ""


def inspect_model(tflite_path):
    metadata = load_metadata(tflite_path)
    parsed = parse_tflite_parameters(tflite_path)

    if parsed is None:
        parameter_count = metadata_param_count(metadata)
        parameter_bytes = parameter_count * 4 if parameter_count is not None else None
        parameter_tensors = ""
        dtype_counts = {}
    else:
        parameter_count = parsed["parameter_count"]
        parameter_bytes = parsed["parameter_bytes"]
        parameter_tensors = parsed["parameter_tensors"]
        dtype_counts = parsed["dtype_counts"]

    return {
        "name": tflite_path.stem,
        "path": str(tflite_path),
        "size_bytes": tflite_path.stat().st_size,
        "size_mb": size_mb(tflite_path.stat().st_size),
        "parameter_count": parameter_count,
        "parameter_tensors": parameter_tensors,
        "parameter_bytes": parameter_bytes,
        "parameter_mb": size_mb(parameter_bytes) if parameter_bytes is not None else None,
        "input_shape": shape_from_metadata(metadata, "input_shape"),
        "output_shape": shape_from_metadata(metadata, "output_shape"),
        "validation_max_abs_diff": (metadata or {}).get("validation", {}).get("max_abs_diff"),
        "dtype_counts": dtype_counts,
    }


def format_int(value):
    if value is None or value == "":
        return "n/a"
    return f"{int(value):,}"


def format_float(value, digits=3):
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.{digits}f}"


def print_table(rows):
    headers = [
        "model",
        "params",
        "param_mb",
        "tflite_mb",
        "param_tensors",
        "input",
        "output",
        "max_abs_diff",
    ]
    formatted = []
    for row in rows:
        formatted.append(
            [
                row["name"],
                format_int(row["parameter_count"]),
                format_float(row["parameter_mb"]),
                format_float(row["size_mb"]),
                str(row["parameter_tensors"]) if row["parameter_tensors"] != "" else "n/a",
                row["input_shape"],
                row["output_shape"],
                format_float(row["validation_max_abs_diff"], digits=8),
            ]
        )

    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in formatted))
        for idx in range(len(headers))
    ]
    print("  ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("  ".join("-" * widths[idx] for idx in range(len(headers))))
    for row in formatted:
        print("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite_dir", default="models_tflite")
    parser.add_argument("--which", default="all", help="all or one model name, e.g. yolo26")
    parser.add_argument("--json", action="store_true", help="Print full JSON instead of table")
    args = parser.parse_args()

    tflite_dir = Path(args.tflite_dir).resolve()
    if args.which == "all":
        paths = [tflite_dir / f"{name}.tflite" for name in DEFAULT_MODELS]
        paths += sorted(
            path for path in tflite_dir.glob("*.tflite")
            if path.stem not in DEFAULT_MODELS
        )
    else:
        paths = [tflite_dir / f"{args.which}.tflite"]

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit("Missing TFLite file(s): " + ", ".join(missing))

    rows = [inspect_model(path) for path in paths]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_table(rows)


if __name__ == "__main__":
    main()
