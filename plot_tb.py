"""
Extract scalars from a TensorBoard events file into a CSV + a train/val loss PNG,
so the loss curves can be viewed/shared without a browser.

Usage:
    python plot_tb.py [logdir_or_events_file]
        (default logdir: output/yolo26_eff_v1)

Outputs (written to the current directory):
    loss_dump.csv   - every scalar tag, one column per tag, one row per step
    loss_curve.png  - best-effort Train vs Val total-loss curve

If EventAccumulator is missing:  pip install tensorboard
"""
import sys
import csv
from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "output/yolo26_eff_v1"
    print(f"reading events from: {path}")

    ea = EventAccumulator(path)
    ea.Reload()
    tags = ea.Tags()["scalars"]
    if not tags:
        print("No scalar tags found. Point this at the directory that contains "
              "the events.out.tfevents.* file (e.g. your --out_dir).")
        return
    print("scalar tags:", tags)

    # collect all scalars keyed by step
    data = defaultdict(dict)
    for t in tags:
        for s in ea.Scalars(t):
            data[s.step][t] = s.value

    # dump every tag to CSV
    with open("loss_dump.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + tags)
        for step in sorted(data):
            w.writerow([step] + [data[step].get(t, "") for t in tags])
    print("wrote loss_dump.csv")

    # best-effort Train vs Val total loss = mean of the per-component tags
    def total(prefix):
        xs, ys = [], []
        for step in sorted(data):
            comps = [v for k, v in data[step].items() if k.startswith(prefix)]
            if comps:
                xs.append(step)
                ys.append(sum(comps) / len(comps))
        return xs, ys

    plotted = False
    for prefix, label in [("train_loss", "Train"), ("val_loss", "Val")]:
        xs, ys = total(prefix)
        if xs:
            plt.plot(xs, ys, label=label)
            plotted = True

    if plotted:
        plt.xlabel("step / epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title("YOLO26-eff loss")
        plt.grid(True, alpha=0.3)
        plt.savefig("loss_curve.png", dpi=120, bbox_inches="tight")
        print("wrote loss_curve.png")
    else:
        print("Could not find train_loss/ or val_loss/ tags to plot; "
              "check loss_dump.csv for the available tags.")

    # also print a compact table to stdout so it can be pasted directly
    print("\nstep, " + ", ".join(tags))
    for step in sorted(data):
        print(step, ", ".join(f"{data[step].get(t, ''):.4f}" if isinstance(data[step].get(t), float)
                              else str(data[step].get(t, "")) for t in tags))


if __name__ == "__main__":
    main()
