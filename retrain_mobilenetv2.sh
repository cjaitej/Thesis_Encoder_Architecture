#!/usr/bin/env bash
# =============================================================================
# Retrain MobileNetV2-1D to convergence and re-evaluate it (backbone only).
#
# Why: the original run (output/train_mobilenetv2) was capped at 50 epochs and
# was still improving when it stopped -- best val at epoch 44, and val loss fell
# ~5% over its last 10 epochs. Every other backbone plateaued well before its cap.
#
# Everything is identical to output/train_mobilenetv2/config.json EXCEPT:
#   --epochs 100      (was 50; matches RoNIN-ResNet, ShuffleNetV2, YOLOv26-1D-Eff)
#   --out_dir         new directory, so the original run and its results are kept
#   --cache_path      speeds up data loading only; does not change results
#   --no_tqdm         cleaner logs only; does not change results
#
# Runtime: ~255 s/epoch on the original hardware -> ~7 h for 100 epochs.
# Run it detached, e.g.:   GPU=1 nohup bash retrain_mobilenetv2.sh > retrain_mnv2.out 2>&1 &
#
# The +RF evaluation is intentionally NOT included -- see the note at the bottom.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"                        # original run used repo-root-relative paths

GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-100}"
RUN_NAME="${RUN_NAME:-mobilenetv2_e100}"
OUT_DIR="output/train_${RUN_NAME}"
TEST_DIR="output/test_${RUN_NAME}"
CACHE_DIR="cache/${RUN_NAME}"

# ---- pre-flight -------------------------------------------------------------
# Training root: the original run used data/seen_subjects_train_set; the server
# currently has data/train_dataset (same split lists). Use whichever exists.
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

echo "=== [1/3] training MobileNetV2-1D for ${EPOCHS} epochs on GPU ${GPU} -> ${OUT_DIR}"
python source/ronin_yolo26_baseline_plain.py \
    --mode          train \
    --backbone      mobilenetv2 \
    --model_dropout 0.2 \
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

echo
echo "=== [2/3] selecting the best-validation checkpoint"
# checkpoint_<N>.pt is written only when val loss reaches a new best, so the
# highest N is the best-validation model (checkpoint_latest.pt is excluded).
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
    echo "         Check ${OUT_DIR}/epoch_metrics.csv before using these results."
fi

echo
echo "=== [3/3] evaluating the backbone (no RF) on seen and unseen subjects"
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
        --backbone      mobilenetv2 \
        --model_dropout 0.2 \
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
echo "=== RESULTS: MobileNetV2-1D, ${EPOCHS}-epoch retrain (best epoch ${best_epoch})"
for split in seen unseen; do
    printf "  %-7s " "${split}"
    grep -oE "avg ATE:[0-9.]+, avg RTE:[0-9.]+" "${TEST_DIR}/${split}/test.log" | tail -1
done
echo "  paper currently reports (50-epoch run): seen 3.5579 / 2.7352, unseen 5.5656 / 4.4741"

# -----------------------------------------------------------------------------
# NOTE on +RF: the original MobileNetV2 "+RF" results did not use a corrector
# fitted to MobileNetV2 -- they loaded output/rf_yolo26/rf_corrector.joblib, the
# Random Forest trained on the full YOLOv26-1D's residuals (the same file was used
# for TinyCNN, LightTCN and ShuffleNetV2). That RF also includes the non-causal
# time_progress feature. Decide the RF protocol before regenerating +RF numbers.
# -----------------------------------------------------------------------------
