#!/usr/bin/env python3
"""
Run the trajectory evaluation for the stage-2 variants and collate ATE/RTE.

Drives source/ronin_yolo26_baseline_plain.py once per configuration (the bare
backbone plus each stage-2 corrector), on the seen and/or unseen test set, then
parses the final "avg ATE:... avg RTE:..." line from each run and prints one
Model / ATE / RTE table per split.

Nothing here is new evaluation logic - every number comes from the same eval
path the paper already uses. This script only sequences the runs and gathers
the results.

Typical use, after training the correctors:

  python tools/eval_stage2_variants.py \
      --model_path best_model/checkpoint_43.pt \
      --stage2_dir output/stage2_models_ckpt43

Re-print the table later without re-running anything:

  python tools/eval_stage2_variants.py --collate_only

Check the commands before committing an hour of compute:

  python tools/eval_stage2_variants.py --dry_run
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

VARIANTS = ["A_rf", "B_ridge", "C_ema", "D_mlp", "E_tcn"]

LABELS = {
    "base": "YOLOv26-1D (no stage 2)",
    "A_rf": "+ Random Forest",
    "B_ridge": "+ Ridge",
    "C_ema": "+ Causal EMA",
    "D_mlp": "+ MLP",
    "E_tcn": "+ Causal TCN",
}

RESULT_RE = re.compile(r"avg ATE:\s*([\d.eE+-]+),\s*avg RTE:\s*([\d.eE+-]+)")


def build_command(args, split, variant):
    """Exactly the invocation used for the published numbers, plus --stage2_model_path."""
    cmd = [
        sys.executable, str(Path("source") / "ronin_yolo26_baseline_plain.py"),
        "--mode", "test",
        "--backbone", args.backbone,
        "--dataset", "ronin",
        "--root_dir", str(Path(args.data_dir) / ("%s_subjects_test_set" % split)),
        "--test_list", str(Path(args.list_dir) / ("list_test_%s.txt" % split)),
        "--window_size", str(args.window_size),
        "--step_size", str(args.step_size),
        "--model_path", args.model_path,
        "--out_dir", str(Path(args.out_dir) / ("eval_%s_%s" % (split, variant))),
    ]
    if args.use_attention:
        cmd.append("--use_attention")
    if args.cache_path:
        cmd += ["--cache_path", args.cache_path]
    if variant != "base":
        cmd += ["--stage2_model_path",
                str(Path(args.stage2_dir) / ("%s_corrector.joblib" % variant))]
    return cmd


def parse_result(text):
    """Last ATE/RTE line wins - the eval prints per-sequence lines before the summary."""
    hits = RESULT_RE.findall(text)
    if not hits:
        return None
    ate, rte = hits[-1]
    return float(ate), float(rte)


def run_one(cmd, log_path, dry_run):
    if dry_run:
        print("  " + " ".join(cmd))
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace")
    log_path.write_text(proc.stdout, errors="replace")
    if proc.returncode != 0:
        print("  FAILED (exit %d) - see %s" % (proc.returncode, log_path))
        tail = [l for l in proc.stdout.strip().splitlines()[-3:]]
        for l in tail:
            print("    | " + l)
        return None
    return parse_result(proc.stdout)


def print_table(split, results, order, deltas, latex):
    present = [k for k in order if k in results]
    if not present:
        return
    print()
    print("%s test set" % split.capitalize())
    print("-" * 46)
    if deltas and "base" in results:
        ba, br = results["base"]
        print("%-26s %8s %8s %9s %9s" % ("Model", "ATE", "RTE", "dATE", "dRTE"))
        for k in present:
            a, r = results[k]
            if k == "base":
                print("%-26s %8.4f %8.4f %9s %9s" % (LABELS[k], a, r, "-", "-"))
            else:
                print("%-26s %8.4f %8.4f %8.2f%% %8.2f%%" % (
                    LABELS[k], a, r, 100 * (a - ba) / ba, 100 * (r - br) / br))
    else:
        print("%-26s %8s %8s" % ("Model", "ATE", "RTE"))
        for k in present:
            a, r = results[k]
            print("%-26s %8.4f %8.4f" % (LABELS[k], a, r))

    if latex:
        print()
        print("  %% %s test set" % split)
        for k in present:
            a, r = results[k]
            print("  %s & %.4f & %.4f \\\\" % (LABELS[k], a, r))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", default="best_model/checkpoint_43.pt",
                   help="stage-1 checkpoint")
    p.add_argument("--stage2_dir", default="output/stage2_models_ckpt43",
                   help="directory holding <variant>_corrector.joblib")
    p.add_argument("--variants", nargs="+", default=VARIANTS)
    p.add_argument("--splits", nargs="+", default=["seen", "unseen"])
    p.add_argument("--no_baseline", action="store_true",
                   help="skip the stage-1-only row")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--list_dir", default="lists")
    p.add_argument("--backbone", default="yolo26")
    p.add_argument("--use_attention", action="store_true", default=True)
    p.add_argument("--no_attention", dest="use_attention", action="store_false")
    p.add_argument("--window_size", type=int, default=200)
    p.add_argument("--step_size", type=int, default=10)
    p.add_argument("--cache_path", default="/tmp/ronin_cache_good")
    p.add_argument("--out_dir", default="output/stage2_eval")
    p.add_argument("--log_dir", default="output/stage2_eval/logs")
    p.add_argument("--csv", default="output/stage2_eval/ate_rte.csv")
    p.add_argument("--collate_only", action="store_true",
                   help="parse existing logs instead of running anything")
    p.add_argument("--dry_run", action="store_true",
                   help="print the commands and exit")
    p.add_argument("--deltas", action="store_true",
                   help="add percent change against the stage-1 baseline")
    p.add_argument("--latex", action="store_true",
                   help="also emit booktabs rows")
    args = p.parse_args()

    order = ([] if args.no_baseline else ["base"]) + list(args.variants)
    log_dir = Path(args.log_dir)
    all_results = {}

    for split in args.splits:
        print("=" * 60)
        print("split: %s" % split)
        results = {}
        for variant in order:
            log_path = log_dir / ("%s_%s.log" % (split, variant))

            if args.collate_only:
                if log_path.exists():
                    got = parse_result(log_path.read_text(errors="replace"))
                    if got:
                        results[variant] = got
                        print("  %-10s ATE %.4f  RTE %.4f  (from log)" % (variant, *got))
                    else:
                        print("  %-10s no result line in %s" % (variant, log_path))
                else:
                    print("  %-10s missing %s" % (variant, log_path))
                continue

            if not args.dry_run:
                print("  running %s ..." % variant, flush=True)
            got = run_one(build_command(args, split, variant), log_path, args.dry_run)
            if got:
                results[variant] = got
                print("    ATE %.4f  RTE %.4f" % got)

        if results:
            all_results[split] = results

    if args.dry_run:
        return

    for split in args.splits:
        if split in all_results:
            print_table(split, all_results[split], order, args.deltas, args.latex)

    if all_results:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "variant", "model", "ate", "rte"])
            for split, res in all_results.items():
                for k in order:
                    if k in res:
                        w.writerow([split, k, LABELS[k], "%.4f" % res[k][0],
                                    "%.4f" % res[k][1]])
        print("\nwrote %s" % csv_path)
    else:
        print("\nNo results collected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
