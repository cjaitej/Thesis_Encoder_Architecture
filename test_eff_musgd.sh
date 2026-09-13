#!/usr/bin/env bash
# =============================================================================
# Test YOLOv26-1D-Eff (wd 1e-4, dropout 0.2 retrain) on seen + unseen subjects,
# backbone only (no RF). Same flags as step [3] of retrain_eff_musgd.sh.
#
# Run:   GPU=3 bash test_eff_musgd.sh
# Other checkpoint:   CKPT=/path/to/checkpoint_N.pt GPU=3 bash test_eff_musgd.sh
#
# Output:  output/test_yolo26_eff_musgd/{seen,unseen}/  (test.log, losses.csv, *_gsn.npy/png)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

GPU="${GPU:-0}"
RUN_NAME="${RUN_NAME:-yolo26_eff_musgd}"
TRAIN_DIR="output/train_${RUN_NAME}"
TEST_DIR="output/test_${RUN_NAME}"
CKPT="${CKPT:-${TRAIN_DIR}/checkpoints/checkpoint_56.pt}"

if [ ! -f "${CKPT}" ]; then echo "ERROR: checkpoint not found: ${CKPT}" >&2; exit 1; fi
for p in data/seen_subjects_test_set data/unseen_subjects_test_set \
         lists/list_test_seen.txt lists/list_test_unseen.txt; do
    if [ ! -e "${p}" ]; then echo "ERROR: missing ${p}" >&2; exit 1; fi
done

# checkpoint_N.pt is only written on a new best validation loss, so the highest N
# is the best-val model. stage2_eff_musgd.sh picks that one automatically -- warn
# if we are testing a different checkpoint, or the two results won't correspond.
highest=-1
for f in "${TRAIN_DIR}"/checkpoints/checkpoint_[0-9]*.pt; do
    [ -e "${f}" ] || continue
    n="${f##*/checkpoint_}"; n="${n%.pt}"
    [ "${n}" -gt "${highest}" ] && highest="${n}"
done
this="${CKPT##*/checkpoint_}"; this="${this%.pt}"
echo "checkpoint: ${CKPT}  (highest best-val checkpoint in ${TRAIN_DIR}: ${highest})"
if [ "${this}" != "${highest}" ]; then
    echo "WARNING: this is NOT the best-validation checkpoint (${highest}); stage2_eff_musgd.sh would use ${highest}."
fi
if [ -f "${TRAIN_DIR}/epoch_metrics.csv" ]; then
    python - "${TRAIN_DIR}/epoch_metrics.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
best = min(rows, key=lambda r: float(r['val_avg_mse']))
print("best val MSE: %.5f at epoch %s  | epochs run: %d" % (float(best['val_avg_mse']), best['epoch'], len(rows)))
PY
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
for split in seen unseen; do
    if [ "${split}" = "seen" ]; then
        ROOT="data/seen_subjects_test_set";   LIST="lists/list_test_seen.txt"
    else
        ROOT="data/unseen_subjects_test_set"; LIST="lists/list_test_unseen.txt"
    fi
    OUT="${TEST_DIR}/${split}"
    mkdir -p "${OUT}"
    echo
    echo "=== testing on ${split} subjects -> ${OUT}"
    PYTHONUNBUFFERED=1 python source/ronin_yolo26_baseline_plain.py \
        --mode          test \
        --backbone      yolo26_eff \
        --model_dropout 0.2 \
        --root_dir      "${ROOT}" \
        --test_list     "${LIST}" \
        --model_path    "${CKPT}" \
        --out_dir       "${OUT}" \
        --dataset       ronin \
        --window_size   200 \
        --step_size     10 \
        --no_tqdm \
        2>&1 | tee "${OUT}/test.log"
done

echo
echo "=== RESULTS: YOLOv26-1D-Eff retrain, ${CKPT##*/}"
for split in seen unseen; do
    printf "  %-7s " "${split}"
    grep -oE "avg ATE:[0-9.]+, avg RTE:[0-9.]+" "${TEST_DIR}/${split}/test.log" | tail -1
done
echo "  old Adam model (wd 0, dropout 0.5):  seen 4.1649 / 3.1488   unseen 5.8977 / 4.9368"
echo "  deleted retrain (epoch 37):          seen 3.9595 / 2.9618   unseen 5.6974 / 4.7883"
