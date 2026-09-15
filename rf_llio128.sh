#!/usr/bin/env bash
# =============================================================================
# Apply ResidualNav (RF corrector) to the trained LLIO-Net ResMLP128, producing
# the "+RF" seen/unseen results, using exactly the same protocol as the other
# transferred backbones (rf_mobilenetv2.sh / tinycnn / lighttcn / shufflenetv2):
#
#   the corrector is NOT refitted on LLIO. output/rf_yolo26/rf_corrector.joblib
#   -- the Random Forest fitted on the 1.17M-parameter YOLOv26-1D reference
#   backbone, and already shared by TinyCNN, LightTCN, ShuffleNetV2 and
#   MobileNetV2 -- is applied unchanged. It also still contains the non-causal
#   time_progress feature, exactly as it does for those other "+RF" rows, so
#   the LLIO "+RF" number is on the same footing as theirs, not a special case.
#
# alpha/clip/hist are the code defaults (1.0 / 0.75 / 50), matching every other
# "+RF" run in this repo.
#
# Run AFTER train_llio128.sh. Takes a few minutes, not hours:
#   GPU=1 bash rf_llio128.sh 2>&1 | tee rf_llio128.out
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

GPU="${GPU:-1}"
FEATURE_DIM="${FEATURE_DIM:-128}"
DROPOUT="${DROPOUT:-0.2}"
RUN_NAME="${RUN_NAME:-llio${FEATURE_DIM}}"
OUT_DIR="output/train_${RUN_NAME}"
TEST_DIR="output/test_${RUN_NAME}"
RF_MODEL="${RF_MODEL:-output/rf_yolo26/rf_corrector.joblib}"
RF_ALPHA="${RF_ALPHA:-1.0}"
RF_CLIP="${RF_CLIP:-0.75}"
RF_HIST="${RF_HIST:-50}"

PYTHON="${PYTHON:-/mnt/data/suryansh/miniconda3/envs/ronin/bin/python}"

# ---- pre-flight -------------------------------------------------------------
if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: ${PYTHON} is not executable (set PYTHON=/path/to/python)" >&2; exit 1
fi
for p in "${RF_MODEL}" data/seen_subjects_test_set data/unseen_subjects_test_set \
         lists/list_test_seen.txt lists/list_test_unseen.txt; do
    if [ ! -e "${p}" ]; then echo "ERROR: missing ${p}" >&2; exit 1; fi
done

best_epoch=-1
BEST_CKPT=""
for f in "${OUT_DIR}"/checkpoints/checkpoint_[0-9]*.pt; do
    [ -e "${f}" ] || continue
    n="${f##*/checkpoint_}"; n="${n%.pt}"
    if [ "${n}" -gt "${best_epoch}" ]; then best_epoch="${n}"; BEST_CKPT="${f}"; fi
done
if [ -z "${BEST_CKPT}" ]; then
    echo "ERROR: no checkpoint in ${OUT_DIR}/checkpoints -- run train_llio128.sh first" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
echo "checkpoint : ${BEST_CKPT} (epoch ${best_epoch})"
echo "RF model   : ${RF_MODEL}  (alpha=${RF_ALPHA} clip=${RF_CLIP} hist=${RF_HIST})"

# ---- evaluate with RF correction ---------------------------------------------
for split in seen unseen; do
    if [ "${split}" = "seen" ]; then
        ROOT="data/seen_subjects_test_set";   LIST="lists/list_test_seen.txt"
    else
        ROOT="data/unseen_subjects_test_set"; LIST="lists/list_test_unseen.txt"
    fi
    OUT="${TEST_DIR}/${split}_rf"
    mkdir -p "${OUT}"
    echo
    echo "=== ${split} + RF -> ${OUT}"
    "${PYTHON}" source/ronin_yolo26_baseline_plain.py \
        --mode               test \
        --backbone           llio \
        --llio_feature_dim   "${FEATURE_DIM}" \
        --model_dropout      "${DROPOUT}" \
        --root_dir           "${ROOT}" \
        --test_list          "${LIST}" \
        --model_path         "${BEST_CKPT}" \
        --out_dir            "${OUT}" \
        --dataset            ronin \
        --window_size        200 \
        --step_size          10 \
        --use_rf_postprocess \
        --rf_model_path      "${RF_MODEL}" \
        --rf_alpha           "${RF_ALPHA}" \
        --rf_clip            "${RF_CLIP}" \
        --rf_hist_window     "${RF_HIST}" \
        --no_tqdm \
        2>&1 | tee "${OUT}/test.log"
done

# ---- summary -----------------------------------------------------------------
echo
echo "=== LLIO-Net ResMLP${FEATURE_DIM}, 100-epoch MuSGD run (best epoch ${best_epoch})"
for split in seen unseen; do
    for suf in "" "_rf"; do
        d="${TEST_DIR}/${split}${suf}"
        if [ -f "${d}/test.log" ]; then
            printf "  %-12s " "${split}${suf}"
            grep -oE "avg ATE:[0-9.]+, avg RTE:[0-9.]+" "${d}/test.log" | tail -1
        fi
    done
done
