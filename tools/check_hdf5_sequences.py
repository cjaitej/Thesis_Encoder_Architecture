#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py


def check_file(path):
    try:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
        return True, keys, ""
    except Exception as exc:
        return False, [], str(exc).splitlines()[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data", help="Dataset root containing *_set folders")
    parser.add_argument(
        "--split",
        default="seen_subjects_test_set",
        help="Split folder to scan, or 'all'",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if args.split == "all":
        split_dirs = sorted(path for path in root.glob("*_set") if path.is_dir())
    else:
        split_dirs = [root / args.split]

    any_ok = False
    for split_dir in split_dirs:
        print(f"\n== {split_dir.name} ==")
        if not split_dir.exists():
            print(f"Missing: {split_dir}")
            continue

        for data_path in sorted(split_dir.glob("*/data.hdf5")):
            ok, keys, error = check_file(data_path)
            status = "OK " if ok else "BAD"
            seq_name = data_path.parent.name
            size = data_path.stat().st_size
            if ok:
                any_ok = True
                print(f"{status}  {seq_name:<10}  {size:>12,} bytes  keys={keys}")
            else:
                print(f"{status}  {seq_name:<10}  {size:>12,} bytes  {error}")

    if not any_ok:
        raise SystemExit("\nNo readable HDF5 sequence found in the selected split.")


if __name__ == "__main__":
    main()
