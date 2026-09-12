#!/usr/bin/env bash
# =============================================================================
# Stage-2 pipeline for the RETRAINED (MuSGD) YOLOv26-1D-Eff: build the residual
# dataset, fit all five correctors, evaluate base + 5 variants on both splits.
#
# Mirrors how the published Eff numbers were produced, in the same folder fashion:
#   output/rf_yolo26_eff_musgd/rf_train.npz, rf_val.npz        <- saved data
#   output/stage2_models_yolo26_eff_musgd/<V>_corrector.joblib <- saved models
#   output/test_yolo26_eff_musgd/<split>_<variant>/            <- saved results
#   output/test_yolo26_eff_musgd/ate_rte.csv                   <- collated table
#
# The residual dataset comes from the SPLIT LISTS (72 train + 16 val sequences),
# never from test data -- same as the original rf_yolo26_eff npz files.
# alpha/clip are tuned on validation by stage2_variants.py and stored inside each
# corrector, so the test step needs no alpha/clip flags.
# time_progress is kept (non-causal), unchanged from the published setup; add
# --drop_non_causal to STAGE2_ARGS to remove it.
#
# Run AFTER retrain_eff_musgd.sh. Roughly 30-60 min total:
#   GPU=3 bash stage2_eff_musgd.sh 2>&1 | tee stage2_eff_musgd.out
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

GPU="${GPU:-0}"
RUN_NAME="${RUN_NAME:-yolo26_eff_musgd}"
DROPOUT="${DROPOUT:-0.2}"
TRAIN_DIR="output/train_${RUN_NAME}"
RF_DIR="output/rf_${RUN_NAME}"
S2_DIR="output/stage2_models_${RUN_NAME}"
TEST_DIR="output/test_${RUN_NAME}"
STAGE2_ARGS="${STAGE2_ARGS:-}"
VARIANTS="${VARIANTS:-base A_rf B_ridge C_ema D_mlp E_tcn}"

# ---- pre-flight -------------------------------------------------------------
if [ -z "${TRAIN_ROOT:-}" ]; then
    for d in data/seen_subjects_train_set data/train_dataset; do
        if [ -d "${d}" ]; then TRAIN_ROOT="${d}"; break; fi
    done
fi
for p in "${TRAIN_ROOT:-<none>}" data/seen_subjects_test_set data/unseen_subjects_test_set \
         lists/list_train.txt lists/list_val.txt lists/list_test_seen.txt lists/list_test_unseen.txt; do
    if [ ! -e "${p}" ]; then echo "ERROR: missing ${p}" >&2; exit 1; fi
done
best_epoch=-1; BEST_CKPT=""
for f in "${TRAIN_DIR}"/checkpoints/checkpoint_[0-9]*.pt; do
    [ -e "${f}" ] || continue
    n="${f##*/checkpoint_}"; n="${n%.pt}"
    if [ "${n}" -gt "${best_epoch}" ]; then best_epoch="${n}"; BEST_CKPT="${f}"; fi
done
if [ -z "${BEST_CKPT}" ]; then
    echo "ERROR: no checkpoint in ${TRAIN_DIR}/checkpoints -- run retrain_eff_musgd.sh first" >&2; exit 1
fi
if ls "${S2_DIR}"/*_corrector.joblib >/dev/null 2>&1; then
    echo "ERROR: ${S2_DIR} already has correctors. Remove it or set RUN_NAME=..." >&2; exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
mkdir -p "${RF_DIR}" "${S2_DIR}" "${TEST_DIR}"
echo "checkpoint  : ${BEST_CKPT} (epoch ${best_epoch})"
echo "train root  : ${TRAIN_ROOT} | GPU ${GPU}"

# ---- [1/4] residual dataset from the split lists ------------------------------
for spl in train val; do
    echo
    echo "=== [1/4] building ${spl} residual dataset -> ${RF_DIR}/rf_${spl}.npz"
    python source/prepare_rf_dataset_yolo26.py \
        --arch           yolo26_eff \
        --model_dropout  "${DROPOUT}" \
        --model_path     "${BEST_CKPT}" \
        --root_dir       "${TRAIN_ROOT}" \
        --list_path      "lists/list_${spl}.txt" \
        --out_path       "${RF_DIR}/rf_${spl}.npz" \
        --dataset        ronin \
        --window_size    200 \
        --step_size      10 \
        --hist_window    50 \
        --max_ori_error  20.0 \
        2>&1 | tee "${RF_DIR}/prepare_${spl}.log"
done

# ---- [2/4] fit the five correctors -------------------------------------------
echo
echo "=== [2/4] fitting correctors -> ${S2_DIR}"
# shellcheck disable=SC2086
python source/stage2_variants.py \
    --train_npz "${RF_DIR}/rf_train.npz" \
    --val_npz   "${RF_DIR}/rf_val.npz" \
    --out_dir   "${S2_DIR}" \
    ${STAGE2_ARGS} \
    2>&1 | tee "${S2_DIR}/stage2_train.log"

# ---- [3/4] evaluate base + 5 variants on both splits --------------------------
for split in seen unseen; do
    ROOT="data/${split}_subjects_test_set"
    LIST="lists/list_test_${split}.txt"
    for variant in ${VARIANTS}; do
        OUT="${TEST_DIR}/${split}_${variant}"
        mkdir -p "${OUT}"
        echo
        echo "=== [3/4] ${split} / ${variant} -> ${OUT}"
        EXTRA=()
        if [ "${variant}" != "base" ]; then
            CORR="${S2_DIR}/${variant}_corrector.joblib"
            if [ ! -f "${CORR}" ]; then echo "  SKIP: ${CORR} not found"; continue; fi
            EXTRA=(--stage2_model_path "${CORR}")
        fi
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
            "${EXTRA[@]}" \
            2>&1 | tee "${OUT}/test.log"
    done
done

# ---- [4/4] collate ------------------------------------------------------------
echo
echo "=== [4/4] collating -> ${TEST_DIR}/ate_rte.csv"
python - "${TEST_DIR}" "${VARIANTS}" <<'PY'
import os, re, sys, csv
test_dir, variants = sys.argv[1], sys.argv[2].split()
pat = re.compile(r"avg ATE:\s*([\d.eE+-]+),\s*avg RTE:\s*([\d.eE+-]+)")
old = {  # published Adam-trained Eff, for reference
 ('seen','base'):(4.1649,3.1488), ('seen','A_rf'):(3.5848,2.6081), ('seen','B_ridge'):(3.9472,2.9097),
 ('seen','C_ema'):(4.1575,3.1220), ('seen','D_mlp'):(3.7298,2.8269), ('seen','E_tcn'):(4.8027,2.8996),
 ('unseen','base'):(5.8977,4.9368), ('unseen','A_rf'):(5.9046,4.7313), ('unseen','B_ridge'):(5.7980,4.7197),
 ('unseen','C_ema'):(5.8889,4.9136), ('unseen','D_mlp'):(5.8741,4.8625), ('unseen','E_tcn'):(6.7961,4.8896)}
rows = []
for split in ('seen','unseen'):
    for v in variants:
        p = os.path.join(test_dir, '%s_%s' % (split, v), 'test.log')
        if not os.path.exists(p): continue
        m = pat.findall(open(p, errors='replace').read())
        if not m: continue
        rows.append((split, v, float(m[-1][0]), float(m[-1][1])))
with open(os.path.join(test_dir, 'ate_rte.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['split','variant','ate','rte']); w.writerows(rows)
print("%-7s %-9s %18s %22s" % ('split','variant','MuSGD ATE / RTE','published Adam ATE / RTE'))
print('-'*62)
for split, v, ate, rte in rows:
    o = old.get((split, v))
    ref = '%.4f / %.4f' % o if o else '-'
    print("%-7s %-9s %8.4f / %.4f %14s" % (split, v, ate, rte, ref))
PY
echo
echo "saved: ${RF_DIR} (npz)  ${S2_DIR} (correctors)  ${TEST_DIR} (results + ate_rte.csv)"
