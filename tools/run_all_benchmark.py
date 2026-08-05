#!/usr/bin/env python3
"""
Run commands multiple times and extract per-run timing metrics.
Writes CSV with columns: run,sample_time_ms,seq_time_ms,returncode

This version is adjusted for the Raspberry Pi folder layout:

    ~/sohel/
      models/
      ronin_resnet/
      seen_subjects_test_set/
      source/
      tools/

Default commands are run from the repository `source` directory.
"""
import argparse
import shlex
import subprocess
import re
import csv
import statistics
import time
from pathlib import Path


# Default commands for Raspberry Pi (run from `source` directory)
COMMANDS = {
    'resnet': (
        "python ronin_resnet_baseline_plain.py --mode test --dataset ronin "
        "--model_path ../ronin_resnet/checkpoint_gsn_latest.pt "
        "--root_dir ../seen_subjects_test_set "
        "--test_path ../seen_subjects_test_set/a001_2 "
        "--out_dir ../output/timing_resnet_a001_2 --cpu"
    ),
    'yolo26': (
        "python ronin_yolo26_baseline_plain.py --mode test --dataset ronin "
        "--backbone yolo26 "
        "--model_path ../models/yolov26_checkpoint_35.pt "
        "--root_dir ../seen_subjects_test_set "
        "--test_path ../seen_subjects_test_set/a001_2 "
        "--out_dir ../output/timing_yolo26_a001_2 "
        "--use_attention --model_dropout 0.2 --cpu"
    ),
    'mobilenet': (
        "python ronin_yolo26_baseline_plain.py --mode test --dataset ronin "
        "--backbone mobilenetv2 "
        "--model_path ../models/mobilenet_checkpoint_44.pt "
        "--root_dir ../seen_subjects_test_set "
        "--test_path ../seen_subjects_test_set/a001_2 "
        "--out_dir ../output/timing_mobilenet_a001_2 "
        "--model_dropout 0.2 --cpu"
    ),
    'shufflenet': (
        "python ronin_yolo26_baseline_plain.py --mode test --dataset ronin "
        "--backbone shufflenetv2 "
        "--model_path ../models/shufflenet_checkpoint_63.pt "
        "--root_dir ../seen_subjects_test_set "
        "--test_path ../seen_subjects_test_set/a001_2 "
        "--out_dir ../output/timing_shufflenet_a001_2 "
        "--model_dropout 0.2 --cpu"
    ),
    'tinycnn': (
        "python ronin_yolo26_baseline_plain.py --mode test --dataset ronin "
        "--backbone tinycnn "
        "--model_path ../models/tinycnn_checkpoint_83.pt "
        "--root_dir ../seen_subjects_test_set "
        "--test_path ../seen_subjects_test_set/a001_2 "
        "--out_dir ../output/timing_tinycnn_a001_2 "
        "--model_dropout 0.2 --cpu"
    ),
    'lighttcn': (
        "python ronin_yolo26_baseline_plain.py --mode test --dataset ronin "
        "--backbone lighttcn "
        "--model_path ../models/lighttcn_checkpoint_42.pt "
        "--root_dir ../seen_subjects_test_set "
        "--test_path ../seen_subjects_test_set/a001_2 "
        "--out_dir ../output/timing_lighttcn_a001_2 "
        "--model_dropout 0.2 --cpu"
    ),
    'yolo26_rf': (
        "python ronin_yolo26_baseline_plain.py --mode test --dataset ronin "
        "--backbone yolo26 "
        "--model_path ../models/yolov26_checkpoint_35.pt "
        "--root_dir ../seen_subjects_test_set "
        "--test_path ../seen_subjects_test_set/a001_2 "
        "--out_dir ../output/timing_yolo26_rf_a001_2 "
        "--use_attention --model_dropout 0.2 "
        "--use_rf_postprocess --rf_model_path ../rf_yolo26/rf_corrector.joblib "
        "--rf_alpha 1.0 --rf_clip 0.75 --rf_hist_window 50 --cpu"
    ),
    'mobilenet_rf': (
        "python ronin_yolo26_baseline_plain.py --mode test --dataset ronin "
        "--backbone mobilenetv2 "
        "--model_path ../models/mobilenet_checkpoint_44.pt "
        "--root_dir ../seen_subjects_test_set "
        "--test_path ../seen_subjects_test_set/a001_2 "
        "--out_dir ../output/timing_mobilenet_rf_a001_2 "
        "--model_dropout 0.2 "
        "--use_rf_postprocess --rf_model_path ../rf_yolo26/rf_corrector.joblib "
        "--rf_alpha 1.0 --rf_clip 0.75 --rf_hist_window 50 --cpu"
    ),
    'shufflenet_rf': (
        "python ronin_yolo26_baseline_plain.py --mode test --dataset ronin "
        "--backbone shufflenetv2 "
        "--model_path ../models/shufflenet_checkpoint_63.pt "
        "--root_dir ../seen_subjects_test_set "
        "--test_path ../seen_subjects_test_set/a001_2 "
        "--out_dir ../output/timing_shufflenet_rf_a001_2 "
        "--model_dropout 0.2 "
        "--use_rf_postprocess --rf_model_path ../rf_yolo26/rf_corrector.joblib "
        "--rf_alpha 1.0 --rf_clip 0.75 --rf_hist_window 50 --cpu"
    ),
    'tinycnn_rf': (
        "python ronin_yolo26_baseline_plain.py --mode test --dataset ronin "
        "--backbone tinycnn "
        "--model_path ../models/tinycnn_checkpoint_83.pt "
        "--root_dir ../seen_subjects_test_set "
        "--test_path ../seen_subjects_test_set/a001_2 "
        "--out_dir ../output/timing_tinycnn_rf_a001_2 "
        "--model_dropout 0.2 "
        "--use_rf_postprocess --rf_model_path ../rf_yolo26/rf_corrector.joblib "  
        "--rf_alpha 1.0 --rf_clip 0.75 --rf_hist_window 50 --cpu"
    ),
    'lighttcn_rf': (
        "python ronin_yolo26_baseline_plain.py --mode test --dataset ronin "
        "--backbone lighttcn "
        "--model_path ../models/lighttcn_checkpoint_42.pt "
        "--root_dir ../seen_subjects_test_set "
        "--test_path ../seen_subjects_test_set/a001_2 "
        "--out_dir ../output/timing_lighttcn_rf_a001_2 "
        "--model_dropout 0.2 "
        "--use_rf_postprocess --rf_model_path ../rf_yolo26/rf_corrector.joblib "
        "--rf_alpha 1.0 --rf_clip 0.75 --rf_hist_window 50 --cpu"
    ),
}


RE_MEAN = re.compile(r"Mean time per sample:\s*([0-9]+\.?[0-9]*)\s*ms", re.IGNORECASE)
RE_MEAN_TOTAL = re.compile(r"Mean time per sample\s*\(total\):\s*([0-9]+\.?[0-9]*)\s*ms", re.IGNORECASE)
RE_TIME_PER_SAMPLE = re.compile(r"time_per_sample=([0-9]+\.?[0-9]*)ms", re.IGNORECASE)
RE_SEQ_TIME = re.compile(r"seq_time=([0-9]+\.?[0-9]*)ms", re.IGNORECASE)
RE_RF = re.compile(r"neural=([0-9]+\.?[0-9]*)ms,\s*rf=([0-9]+\.?[0-9]*)ms,\s*total=([0-9]+\.?[0-9]*)ms", re.IGNORECASE)


def extract_times(stdout_text):
    """Return tuple (sample_time_ms_or_None, seq_time_ms_or_None)"""
    # Prefer total timing when RF postprocessing is enabled.
    m = RE_MEAN_TOTAL.search(stdout_text)
    if m:
        try:
            return float(m.group(1)), None
        except Exception:
            pass
    # Try final summary first
    m = RE_MEAN.search(stdout_text)
    if m:
        try:
            return float(m.group(1)), None
        except Exception:
            pass
    # Try explicit time_per_sample=...ms
    m = RE_TIME_PER_SAMPLE.search(stdout_text)
    if m:
        try:
            return float(m.group(1)), None
        except Exception:
            pass
    # Try RF combined pattern
    m = RE_RF.search(stdout_text)
    if m:
        try:
            total_ms = float(m.group(3))
            return None, total_ms
        except Exception:
            pass
    # Try seq_time=...ms
    m = RE_SEQ_TIME.search(stdout_text)
    if m:
        try:
            return None, float(m.group(1))
        except Exception:
            pass
    return None, None


def run_repeat(cmd_list, cwd, runs, delay, log_dir=None, name='run'):
    results = []
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, runs + 1):
        print(f"Run {i}/{runs}: executing...", flush=True)
        start = time.time()
        p = subprocess.run(cmd_list, cwd=cwd, capture_output=True, text=True)
        elapsed = (time.time() - start) * 1000.0
        stdout = (p.stdout or "") + "\n" + (p.stderr or "")

        if log_dir is not None:
            run_log_path = log_dir / f"{name}_run_{i:03d}.log"
            with open(run_log_path, 'w', newline='') as f:
                f.write("Command: " + " ".join(cmd_list) + "\n")
                f.write(f"Return code: {p.returncode}\n")
                f.write(f"Wall time: {elapsed:.4f} ms\n")
                f.write("\n===== STDOUT =====\n")
                f.write(p.stdout or "")
                f.write("\n===== STDERR =====\n")
                f.write(p.stderr or "")

        sample_time_ms, seq_time_ms = extract_times(stdout)
        # fallback: if neither parsed, as a last resort use wall-clock elapsed for sequence
        if sample_time_ms is None and seq_time_ms is None:
            seq_time_ms = elapsed
        results.append({
            'run': i,
            'sample_time_ms': sample_time_ms,
            'seq_time_ms': seq_time_ms,
            'returncode': p.returncode,
        })
        print(f"  parsed: sample_time_ms={sample_time_ms}, seq_time_ms={seq_time_ms}, rc={p.returncode}")
        if p.returncode != 0:
            if log_dir is not None:
                print(f"  full log: {run_log_path}")
            print("  last output lines:")
            for line in stdout.strip().splitlines()[-12:]:
                print(f"    {line}")
        if delay and i < runs:
            time.sleep(delay)
    return results


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run', 'sample_time_ms', 'seq_time_ms', 'returncode'])
        for r in rows:
            writer.writerow([r['run'], r['sample_time_ms'] if r['sample_time_ms'] is not None else '', r['seq_time_ms'] if r['seq_time_ms'] is not None else '', r['returncode']])


def summarize(rows):
    sample_vals = [r['sample_time_ms'] for r in rows if r['sample_time_ms'] is not None and r['returncode'] == 0]
    seq_vals = [r['seq_time_ms'] for r in rows if r['seq_time_ms'] is not None and r['returncode'] == 0]

    def stats(v):
        if not v:
            return None
        return {'count': len(v), 'mean': statistics.mean(v), 'median': statistics.median(v), 'stdev': statistics.stdev(v) if len(v) > 1 else 0.0}

    return {'sample': stats(sample_vals), 'sequence': stats(seq_vals)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmd', type=str, default=None, help='Single command to run (overrides --which)')
    parser.add_argument('--which', type=str, default='all', help="Which predefined command to run: 'all', 'resnet', 'yolo26', 'mobilenet', 'shufflenet', 'tinycnn', 'lighttcn', 'yolo26_rf', 'mobilenet_rf', 'shufflenet_rf', 'tinycnn_rf', 'lighttcn_rf'")
    parser.add_argument('--cwd', type=str, default='source', help='Working directory to run the command in')
    parser.add_argument('--runs', type=int, default=50, help='Number of iterations')
    parser.add_argument('--out', type=str, default='benchmark_results.csv', help='CSV output path')
    parser.add_argument('--delay', type=float, default=0.5, help='Seconds to wait between runs')
    args = parser.parse_args()

    cwd = Path(args.cwd)
    if not cwd.exists():
        print(f"Error: cwd {cwd} does not exist")
        return

    # choose commands to run
    to_run = []
    if args.cmd is not None:
        to_run = [('custom', args.cmd)]
    else:
        if args.which == 'all':
            to_run = list(COMMANDS.items())
        elif args.which in COMMANDS:
            to_run = [(args.which, COMMANDS[args.which])]
        else:
            print(f"Unknown --which value: {args.which}")
            return

    results_dir = Path('bench_results')
    results_dir.mkdir(exist_ok=True)
    log_dir = results_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    overall = {}
    for name, cmd in to_run:
        print(f"\n=== Running benchmark for: {name} ===")
        cmd_list = shlex.split(cmd)
        print(f"Command: {cmd_list}")
        rows = run_repeat(cmd_list, cwd=str(cwd), runs=args.runs, delay=args.delay, log_dir=log_dir, name=name)
        out_path = results_dir / f"{name}_{args.out}"
        write_csv(out_path, rows)
        print(f"Wrote raw results to {out_path}")
        summary = summarize(rows)
        overall[name] = summary
        print('\nSummary for', name)
        print(summary)

    # write combined summary
    summary_path = results_dir / 'benchmark_summary.txt'
    with open(summary_path, 'w') as f:
        for name, s in overall.items():
            f.write(f"== {name} ==\n")
            f.write(str(s) + '\n')
    print(f"Wrote summary to {summary_path}")


if __name__ == '__main__':
    main()
