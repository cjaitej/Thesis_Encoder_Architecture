#!/usr/bin/env bash
# Evaluate a retrained RoNIN ResNet checkpoint on BOTH test splits.
# Reports ATE / RTE (RTE over a 1-minute window), as in Table I of the paper.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${REPO_ROOT}/data"
LISTS="${REPO_ROOT}/lists"

RUN_NAME="${RUN_NAME:-ronin_resnet_repro}"
CKPT_DIR="${REPO_ROOT}/output/${RUN_NAME}/checkpoints"

# checkpoint_<N>.pt is written ONLY when val loss hits a new best, so the
# highest-numbered one IS the best-validation model. checkpoint_latest.pt is
# just the final epoch -- usually worse. Default to best, allow an override.
if [ -z "${MODEL_PATH:-}" ]; then
    best_epoch=-1
    for f in "${CKPT_DIR}"/checkpoint_[0-9]*.pt; do
        [ -e "${f}" ] || continue
        n="${f##*/checkpoint_}"
        n="${n%.pt}"
        if [ "${n}" -gt "${best_epoch}" ] 2>/dev/null; then
            best_epoch="${n}"
            MODEL_PATH="${f}"
        fi
    done
    if [ -z "${MODEL_PATH:-}" ]; then
        MODEL_PATH="${CKPT_DIR}/checkpoint_latest.pt"
        echo "No best-val checkpoint found; falling back to checkpoint_latest.pt"
    fi
fi

if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: checkpoint not found: ${MODEL_PATH}" >&2
    exit 1
fi
echo "Evaluating checkpoint: ${MODEL_PATH}"

cd "${REPO_ROOT}/source"

for split in seen unseen; do
    if [ "${split}" = "seen" ]; then
        ROOT="${DATA_ROOT}/seen_subjects_test_set"
        LIST="${LISTS}/list_test_seen.txt"
    else
        ROOT="${DATA_ROOT}/unseen_subjects_test_set"
        LIST="${LISTS}/list_test_unseen.txt"
    fi

    OUT="${REPO_ROOT}/output/${RUN_NAME}/${split}"
    mkdir -p "${OUT}"

    echo "=============== ${split} test set ==============="
    python ronin_resnet_baseline_plain.py \
        --mode       test \
        --arch       resnet18 \
        --root_dir   "${ROOT}" \
        --test_list  "${LIST}" \
        --model_path "${MODEL_PATH}" \
        --out_dir    "${OUT}" \
        --dataset    ronin \
        --window_size 200 \
        --step_size   10 \
        2>&1 | tee "${OUT}/test.log"
done

echo
echo "Per-sequence ATE/RTE written to output/${RUN_NAME}/{seen,unseen}/losses.csv"
