#!/usr/bin/env bash
# =============================================================================
# Retrain YOLOv26-1D-Eff with the SAME optimizer and regularization as
# RoNIN-ResNet and the competing backbones, so the paper can state that every
# network was trained with one protocol.
#
# Old Eff run (output/yolo26_eff_adam_v1):  adam,  weight_decay 0,    dropout 0.5
# This run:                                 musgd, weight_decay 1e-4, dropout 0.2
# Unchanged: 100 epochs, lr 1e-4, batch 128, window 200, stride 10, max_ori_error 20
#
# SELECTION RULE -- decide BEFORE looking at test results:
#   use this MuSGD model for the paper regardless of its score (consistency).
#   Choosing between Adam and MuSGD by TEST accuracy would be test-set selection.
#
# Runtime: ~270 s/epoch on the original hardware -> ~7.5 h for 100 epochs.
# Can run on a different GPU than the MobileNetV2 retrain, e.g.:
#   GPU=2 nohup bash retrain_eff_musgd.sh > retrain_eff.out 2>&1 &
#
# Steps: [1] train  [2] pick best-val checkpoint  [3] test backbone (no RF)
#        [4] export TFLite to a SEPARATE folder (does not overwrite the Adam model)
# The Stage-2 correctors (RF etc.) must be refit on this backbone's residuals --
# NOT included here; see the note at the bottom.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-100}"
DROPOUT="${DROPOUT:-0.2}"
RUN_NAME="${RUN_NAME:-yolo26_eff_musgd}"
EXPORT_TFLITE="${EXPORT_TFLITE:-1}"
OUT_DIR="output/train_${RUN_NAME}"
TEST_DIR="output/test_${RUN_NAME}"
CACHE_DIR="cache/${RUN_NAME}"
TFLITE_DIR="models_tflite/${RUN_NAME}"

# ---- pre-flight -------------------------------------------------------------
# Training root: the competing backbones used data/seen_subjects_train_set, the
# old Eff run used data/train_dataset (same lists). Use whichever exists.
if [ -z "${TRAIN_ROOT:-}" ]; then
    for d in data/seen_subjects_train_set data/train_dataset; do
        if [ -d "${d}" ]; then TRAIN_ROOT="${d}"; break; fi
    done
fi
for p in "${TRAIN_ROOT:-<none>}" data/seen_subjects_test_set data/unseen_subjects_test_set \
         lists/list_train.txt lists/list_val.txt lists/list_test_seen.txt lists/list_test_unseen.txt; do
    if [ ! -e "${p}" ]; then echo "ERROR: missing ${p} (set TRAIN_ROOT=... if the data lives elsewhere)" >&2; exit 1; fi
done
# Refuse to mix runs: best-checkpoint selection takes the highest checkpoint_N.pt,
# so leftovers from an earlier or aborted run would be picked up silently.
if ls "${OUT_DIR}"/checkpoints/checkpoint_[0-9]*.pt >/dev/null 2>&1; then
    echo "ERROR: ${OUT_DIR}/checkpoints already has checkpoints. Remove the folder or set RUN_NAME=..." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
mkdir -p "${OUT_DIR}/checkpoints" "${CACHE_DIR}"
echo "training root: ${TRAIN_ROOT} | GPU ${GPU} | out: ${OUT_DIR}"

# ---- [1] train ----------------------------------------------------------------
echo "=== [1/4] training YOLOv26-1D-Eff with MuSGD for ${EPOCHS} epochs"
python source/ronin_yolo26_baseline_plain.py \
    --mode          train \
    --backbone      yolo26_eff \
    --model_dropout "${DROPOUT}" \
    --root_dir      "${TRAIN_ROOT}" \
    --train_list    lists/list_train.txt \
    --val_list      lists/list_val.txt \
    --cache_path    "${CACHE_DIR}" \
    --out_dir       "${OUT_DIR}" \
    --dataset       ronin \
    --window_size   200 \
    --step_size     10 \
    --max_ori_error 20.0 \
    --batch_size    128 \
    --epochs        "${EPOCHS}" \
    --lr            1e-4 \
    --optim         musgd \
    --momentum      0.9 \
    --ns_steps      5 \
    --weight_decay  1e-4 \
    --no_tqdm \
    2>&1 | tee "${OUT_DIR}/train.log"

# ---- [2] best-validation checkpoint --------------------------------------------
echo
echo "=== [2/4] selecting the best-validation checkpoint"
best_epoch=-1
BEST_CKPT=""
for f in "${OUT_DIR}"/checkpoints/checkpoint_[0-9]*.pt; do
    [ -e "${f}" ] || continue
    n="${f##*/checkpoint_}"; n="${n%.pt}"
    if [ "${n}" -gt "${best_epoch}" ]; then best_epoch="${n}"; BEST_CKPT="${f}"; fi
done
if [ -z "${BEST_CKPT}" ]; then
    echo "ERROR: no best-validation checkpoint found in ${OUT_DIR}/checkpoints" >&2
    exit 1
fi
echo "best checkpoint: ${BEST_CKPT} (epoch ${best_epoch} of ${EPOCHS})"
if [ "${best_epoch}" -ge $((EPOCHS - 10)) ]; then
    echo "WARNING: best epoch is within 10 epochs of the cap -- the model may STILL be improving."
fi
if [ -f "${OUT_DIR}/epoch_metrics.csv" ]; then
    python - "${OUT_DIR}/epoch_metrics.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
best = min(rows, key=lambda r: float(r['val_avg_mse']))
print("best val MSE: %.5f at epoch %s  (old Adam run: 0.03332 at epoch 59)"
      % (float(best['val_avg_mse']), best['epoch']))
PY
fi

# ---- [3] test backbone (no RF) -------------------------------------------------
echo
echo "=== [3/4] evaluating the backbone (no RF) on seen and unseen subjects"
for split in seen unseen; do
    if [ "${split}" = "seen" ]; then
        ROOT="data/seen_subjects_test_set";   LIST="lists/list_test_seen.txt"
    else
        ROOT="data/unseen_subjects_test_set"; LIST="lists/list_test_unseen.txt"
    fi
    OUT="${TEST_DIR}/${split}"
    mkdir -p "${OUT}"
    python source/ronin_yolo26_baseline_plain.py \
        --mode          test \
        --backbone      yolo26_eff \
        --model_dropout "${DROPOUT}" \
        --root_dir      "${ROOT}" \
        --test_list     "${LIST}" \
        --model_path    "${BEST_CKPT}" \
        --out_dir       "${OUT}" \
        --dataset       ronin \
        --window_size   200 \
        --step_size     10 \
        --no_tqdm \
        2>&1 | tee "${OUT}/test.log"
done

echo
echo "=== RESULTS: YOLOv26-1D-Eff, MuSGD (best epoch ${best_epoch})"
for split in seen unseen; do
    printf "  %-7s " "${split}"
    grep -oE "avg ATE:[0-9.]+, avg RTE:[0-9.]+" "${TEST_DIR}/${split}/test.log" | tail -1
done
echo "  old Adam model (no RF):  seen 4.1649 / 3.1488   unseen 5.8977 / 4.9368"

# ---- [4] TFLite export (separate folder) ---------------------------------------
if [ "${EXPORT_TFLITE}" = "1" ]; then
    echo
    echo "=== [4/4] exporting TFLite -> ${TFLITE_DIR}/yolo26_eff.tflite"
    if python tools/convert_yolo26_eff_direct_to_tflite.py \
            --checkpoint "${BEST_CKPT}" --out_dir "${TFLITE_DIR}" \
            2>&1 | tee "${OUT_DIR}/tflite_export.log"; then
        echo "exported. For the Pi benchmark, copy ${TFLITE_DIR}/yolo26_eff.tflite over"
        echo "models_tflite/yolo26_eff.tflite on the Pi (latency depends on the architecture, not the weights)."
    else
        echo "WARNING: TFLite export failed (training and test results above are still valid)."
    fi
fi

# -----------------------------------------------------------------------------
# NEXT: the Stage-2 correctors (A_rf ... E_tcn) were fitted to the OLD Adam
# backbone's residuals and must be refit on this model before any "+RF" number
# is valid. That refit is also the zero-cost moment to drop the non-causal
# time_progress feature.
# -----------------------------------------------------------------------------
