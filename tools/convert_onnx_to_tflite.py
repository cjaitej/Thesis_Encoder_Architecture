#!/usr/bin/env python3
"""
Convert exported ONNX models to TFLite.

Run this on a laptop/desktop, not on the Raspberry Pi.

Recommended environment:
    python -m pip install onnx onnx2tf tensorflow

The script calls the onnx2tf command-line converter, then copies/renames the
produced .tflite file into models_tflite/.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("Running:", " ".join(str(x) for x in cmd))
    env = os.environ.copy()
    env.pop("TF_USE_LEGACY_KERAS", None)
    subprocess.run(cmd, check=True, env=env)


def find_tflite(path):
    matches = sorted(path.rglob("*.tflite"))
    return matches[0] if matches else None


def file_info(path):
    size_bytes = path.stat().st_size if path.exists() else 0
    return {
        "path": str(path),
        "size_bytes": size_bytes,
        "size_mb": size_bytes / (1024 * 1024),
    }


def load_onnx_metadata(onnx_path):
    metadata_path = onnx_path.with_name(f"{onnx_path.stem}_metadata.json")
    if not metadata_path.exists():
        return None
    with metadata_path.open() as f:
        return json.load(f)


def write_tflite_metadata(onnx_path, final_path, out_dir, converter_work_dir):
    metadata = {
        "name": onnx_path.stem,
        "onnx": file_info(onnx_path),
        "tflite": file_info(final_path),
        "converter": {
            "tool": "onnx2tf",
            "work_dir": str(converter_work_dir),
            "tf_use_legacy_keras_unset_for_subprocess": True,
        },
        "onnx_export_metadata": load_onnx_metadata(onnx_path),
    }
    metadata_path = out_dir / f"{onnx_path.stem}_tflite_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata: {metadata_path}")


def convert_one(onnx_path, work_dir, out_dir):
    name = onnx_path.stem
    model_work_dir = work_dir / name
    if model_work_dir.exists():
        shutil.rmtree(model_work_dir)
    model_work_dir.mkdir(parents=True, exist_ok=True)

    # onnx2tf normally creates a SavedModel and may also emit a .tflite file.
    # Try the plain conversion first. Some rewritten export graphs convert
    # better when onnx2tf is allowed to choose its own internal layout.
    attempts = [
        [],
        ["-kat", "input"],
        ["-k", "input"],
        ["-ois", "input", "1,6,200"],
    ]
    last_error = None
    for extra_args in attempts:
        if model_work_dir.exists():
            shutil.rmtree(model_work_dir)
        model_work_dir.mkdir(parents=True, exist_ok=True)
        try:
            run(["onnx2tf", "-i", str(onnx_path), "-o", str(model_work_dir), *extra_args])
            last_error = None
            break
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"Conversion attempt failed with extra args: {extra_args}")
    if last_error is not None:
        raise last_error

    tflite_path = find_tflite(model_work_dir)
    if tflite_path is None:
        saved_model = model_work_dir / "saved_model"
        if not saved_model.exists():
            saved_model = model_work_dir
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise SystemExit("TensorFlow is required when onnx2tf does not directly emit .tflite") from exc
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
        tflite_model = converter.convert()
        tflite_path = model_work_dir / f"{name}.tflite"
        tflite_path.write_bytes(tflite_model)

    final_path = out_dir / f"{name}.tflite"
    shutil.copy2(tflite_path, final_path)
    print(f"Wrote {final_path}")
    write_tflite_metadata(onnx_path, final_path, out_dir, model_work_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", default="models_onnx")
    parser.add_argument("--out_dir", default="models_tflite")
    parser.add_argument("--work_dir", default="models_tflite_work")
    parser.add_argument("--which", default="all", help="all or one ONNX stem, e.g. tinycnn")
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    direct_only = {"lighttcn", "shufflenet", "yolo26"}
    if args.which == "all":
        onnx_files = [path for path in sorted(onnx_dir.glob("*.onnx")) if path.stem not in direct_only]
        print(
            "Skipping lighttcn.onnx, shufflenet.onnx, and yolo26.onnx in --which all; "
            "use their direct *_direct_to_tflite.py converters."
        )
    else:
        onnx_files = [onnx_dir / f"{args.which}.onnx"]

    missing = [str(path) for path in onnx_files if not path.exists()]
    if missing:
        raise SystemExit("Missing ONNX file(s): " + ", ".join(missing))

    for onnx_path in onnx_files:
        convert_one(onnx_path, work_dir, out_dir)

    print("\nCopy this folder to Raspberry Pi:")
    print(f"  {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Conversion command failed with exit code {exc.returncode}", file=sys.stderr)
        raise
