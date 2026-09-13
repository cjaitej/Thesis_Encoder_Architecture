#!/usr/bin/env bash
# =============================================================================
# Retrain the RoNIN ResNet-18 model (Herath, Yan & Furukawa, ICRA 2020)
#   Paper: https://arxiv.org/abs/1905.12853
#
# Hyperparameters below reproduce the paper's "Implementation Details":
#   input        200 x 6 tensor (1 s @ 200 Hz, 3-axis accel + 3-axis gyro)
#   output       2D velocity in the heading-agnostic coordinate frame (HACF)
#   loss         MSE on strided velocity  (net(i)  vs.  P_i - P_{i-200})
#   optimizer    Adam, initial lr 1e-4
#   scheduler    ReduceLROnPlateau, factor 0.1, patience 10  (on val loss)
#   batch size   128
#   epochs       ~100 (paper: "converges after 100 epochs, ~10 hours")
#   augmentation random HACF rotation about gravity, applied per step
#
# NOTE: use ronin_resnet_baseline_plain.py, NOT ronin_resnet.py.
#       ronin_resnet.py in this repo has been modified for the YOLO26 thesis
#       experiments -- its get_model() returns a YOLO26_1D_Regressor.
#       baseline_plain.py is the untouched ResNet-18 recipe.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${REPO_ROOT}/data"
LISTS="${REPO_ROOT}/lists"

RUN_NAME="${RUN_NAME:-ronin_resnet_repro}"
OUT_DIR="${REPO_ROOT}/output/${RUN_NAME}"
CACHE_DIR="${REPO_ROOT}/cache/${RUN_NAME}"     # speeds up epoch 1 a lot

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WINDOW_SIZE="${WINDOW_SIZE:-200}"
STEP_SIZE="${STEP_SIZE:-10}"
ARCH="${ARCH:-resnet18}"

mkdir -p "${OUT_DIR}" "${CACHE_DIR}"

# The training scripts use flat imports (`from data_glob_speed import *`),
# so they must be launched from inside source/.
cd "${REPO_ROOT}/source"

python ronin_resnet_baseline_plain.py \
    --mode        train \
    --arch        "${ARCH}" \
    --root_dir    "${DATA_ROOT}/seen_subjects_train_set" \
    --train_list  "${LISTS}/list_train.txt" \
    --val_list    "${LISTS}/list_val.txt" \
    --cache_path  "${CACHE_DIR}" \
    --out_dir     "${OUT_DIR}" \
    --dataset     ronin \
    --window_size "${WINDOW_SIZE}" \
    --step_size   "${STEP_SIZE}" \
    --batch_size  "${BATCH_SIZE}" \
    --epochs      "${EPOCHS}" \
    --lr          "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --max_ori_error 20.0 \
    2>&1 | tee "${OUT_DIR}/train.log"

echo
echo "Done. Checkpoints -> ${OUT_DIR}/checkpoints/"
echo "TensorBoard      -> tensorboard --logdir ${OUT_DIR}/logs"
