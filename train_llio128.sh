#!/usr/bin/env bash
# =============================================================================
# Train the LLIO-Net ResMLP128 backbone -- the LLIO paper's LIGHTWEIGHT variant
# -- under THIS repo's protocol, and evaluate it on seen/unseen subjects
# (backbone only, no RF).
#
# Why ResMLP128 rather than the headline ResMLP512: by the paper's own Table II,
# ResMLP512 has MORE FLOPs than the ResNet it is compared against (25.78M vs
# 21.15M) and is actually SLOWER on the plain JIT runtime (14.5 ms vs 9.9 ms on
# a Pixel 3); its speed-up needs PyTorch's mobile optimiser. ResMLP128 is the
# variant carrying their efficiency claim: 2.08M FLOPs (10x fewer than ResNet)
# and 9.2-12x faster inference, at 0.119 m vs 0.108 m distance error. At 489,570
# parameters it also sits just below YOLOv26-1D-Eff (598,530), so it compares
# like-for-like against this paper's edge-budget backbones instead of being a
# 7.27M-parameter outlier larger than the RoNIN-ResNet baseline.
#
# Why: the paper compares only against RoNIN-ResNet, so a reviewer asked for at
# least one modern learned-IO baseline. LLIO is the only one of TLIO/CTIN/LLIO
# whose architecture is actually obtainable: TLIO ships training code but needs
# its own proprietary dataset and is EKF-coupled; CTIN's code is not released.
#
# What this is: LLIO's *architecture* (source/model_llio1d.py), at the config
# published in their Section IV-A.4 / Table III (6 ResMLP blocks, expansion 2,
# feature_dim 128 for ResMLP128, patch 25 -> 50 to hold 0.25 s/patch at our
# 200 Hz, mean-pool + linear + GELU head, dropout 0.2), trained with OUR RoNIN
# subset. It is NOT a reproduction of their published numbers: their repo ships
# the architecture only -- no training code, no loss, no dataset loader, no
# weights -- and their 3-D displacement + covariance head (SCEKF measurement +
# NLL loss) is replaced by this pipeline's 2-D planar-velocity MSE head.
#
# Protocol is copied from the other competing backbones' config.json
# (tinycnn / lighttcn / shufflenetv2 / mobilenetv2_e100):
#   musgd, lr 1e-4, weight_decay 1e-4, momentum 0.9, ns_steps 5,
#   batch 128, window 200, stride 10, max_ori_error 20, dropout 0.2.
# NOTE: the LLIO paper itself trains its ResMLP series with Adam at lr 5e-4.
# Protocol consistency across backbones is chosen here deliberately, so the
# comparison is controlled; report that choice wherever these numbers are used.
#
# Runtime: 489,570 params, MLP-only ops -- faster per step than ResMLP512.
# Run it detached, e.g.:
#   GPU=1 nohup bash train_llio128.sh > train_llio128.out 2>&1 &
#
# The +RF evaluation is intentionally NOT included -- see the note at the bottom.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"                        # scripts here use repo-root-relative paths

GPU="${GPU:-1}"
EPOCHS="${EPOCHS:-100}"
DROPOUT="${DROPOUT:-0.2}"
FEATURE_DIM="${FEATURE_DIM:-128}"          # 512 / 256 / 128 = ResMLP512 / 256 / 128
NUM_WORKERS="${NUM_WORKERS:-16}"           # box has 64 CPUs idle; 0 = old synchronous behavior
RUN_NAME="${RUN_NAME:-llio${FEATURE_DIM}}"
OUT_DIR="output/train_${RUN_NAME}"
TEST_DIR="output/test_${RUN_NAME}"
CACHE_DIR="cache/${RUN_NAME}"

# The system `python` has no torch; use the env that does unless told otherwise.
PYTHON="${PYTHON:-/mnt/data/suryansh/miniconda3/envs/ronin/bin/python}"

# ---- pre-flight -------------------------------------------------------------
if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: ${PYTHON} is not executable (set PYTHON=/path/to/python)" >&2; exit 1
fi
if ! "${PYTHON}" -c "import torch" 2>/dev/null; then
    echo "ERROR: ${PYTHON} cannot import torch (set PYTHON=... to the ronin env)" >&2; exit 1
fi

# Training root: older runs used data/seen_subjects_train_set, newer ones
# data/train_dataset (same split lists). Use whichever exists.
if [ -z "${TRAIN_ROOT:-}" ]; then
    for d in data/seen_subjects_train_set data/train_dataset; do
        if [ -d "${d}" ]; then TRAIN_ROOT="${d}"; break; fi
    done
fi
for p in "${TRAIN_ROOT:-<none>}" data/seen_subjects_test_set data/unseen_subjects_test_set \
         lists/list_train.txt lists/list_val.txt lists/list_test_seen.txt lists/list_test_unseen.txt \
         source/model_llio1d.py; do
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
echo "python: ${PYTHON}"
echo "training root: ${TRAIN_ROOT} | GPU ${GPU} | epochs ${EPOCHS} | ResMLP${FEATURE_DIM} | out: ${OUT_DIR}"

# Parameter count of the selected variant, printed before training starts so it
# sits at the top of the log instead of scrolling past mid-run. The trainer also
# prints it itself ("[llio] Parameters: ...").
PARAM_LINE="$("${PYTHON}" tools/llio_param_count.py "${FEATURE_DIM}")"
PARAMS="${PARAM_LINE%% *}"
PARAM_MIB="${PARAM_LINE##* }"
echo "model: LLIO-Net ResMLP${FEATURE_DIM} | parameters: ${PARAMS} | FP32 size: ${PARAM_MIB} MiB"
echo "       reference: YOLOv26-1D-Eff 598,530 (2.28 MiB) | RoNIN ResNet 4,634,882 (17.68 MiB)"

# ---- [1] train ----------------------------------------------------------------
echo "=== [1/3] training LLIO-Net (ResMLP${FEATURE_DIM}) for ${EPOCHS} epochs on GPU ${GPU}"
"${PYTHON}" source/ronin_yolo26_baseline_plain.py \
    --mode          train \
    --backbone      llio \
    --llio_feature_dim "${FEATURE_DIM}" \
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
    --num_workers   "${NUM_WORKERS}" \
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
if [ -f "${OUT_DIR}/epoch_metrics.csv" ]; then
    "${PYTHON}" - "${OUT_DIR}/epoch_metrics.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
key = 'val_avg_mse' if rows and 'val_avg_mse' in rows[0] else None
if key:
    best = min(rows, key=lambda r: float(r[key]))
    print("best val MSE: %.5f at epoch %s" % (float(best[key]), best['epoch']))
PY
fi

# ---- [3] test backbone (no RF) -------------------------------------------------
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
    "${PYTHON}" source/ronin_yolo26_baseline_plain.py \
        --mode          test \
        --backbone      llio \
        --llio_feature_dim "${FEATURE_DIM}" \
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
echo "=== RESULTS: LLIO-Net ResMLP${FEATURE_DIM}, ${EPOCHS}-epoch MuSGD run (best epoch ${best_epoch})"
for split in seen unseen; do
    printf "  %-7s " "${split}"
    grep -oE "avg ATE:[0-9.]+, avg RTE:[0-9.]+" "${TEST_DIR}/${split}/test.log" | tail -1
done
echo
echo "  for context, the paper's current backbone-only numbers (ATE / RTE):"
echo "    YOLOv26-1D-Eff (ours)    seen 4.1649 / 3.1488   unseen 5.8977 / 4.9368   0.60M params"
echo "    RoNIN ResNet (baseline)  seen 3.7114 / 2.7614   unseen 5.1400 / 4.3770   4.63M params"
echo "    MobileNetV2-1D           seen 3.9705 / 2.7708   unseen 5.5274 / 4.5530   2.64M params"
echo "    ShuffleNetV2-1D          seen 3.9425 / 2.8061   unseen 5.7805 / 4.6046   0.85M params"
echo "    TinyCNN-1D               seen 4.7783 / 3.1978   unseen 6.5321 / 5.0162   0.12M params"
echo "    LightTCN-1D              seen 5.6082 / 3.3909   unseen 8.6503 / 4.9925   0.08M params"
echo "    LLIO-Net ResMLP${FEATURE_DIM} (this run)                                  ${PARAMS} params"

# -----------------------------------------------------------------------------
# NOTE on +RF: do NOT reuse output/rf_yolo26/rf_corrector.joblib for a "+RF" row
# here. That is the RF fitted to the OLD full YOLOv26-1D's residuals, and it
# still contains the non-causal time_progress feature that the paper's feature
# list no longer claims. The other transferred backbones' +RF numbers came from
# that same file, so decide the RF protocol (refit causally, or keep as-is and
# say so) before generating any +RF number for LLIO.
#
# NOTE on the Pi row: Table V needs on-device latency/power/energy from the
# Raspberry Pi 3 B+ with the INA219 rig. That cannot be produced here; export
# the best checkpoint to TFLite and run the same benchmark on the device, or
# leave the LLIO row's Pi columns blank.
# -----------------------------------------------------------------------------
