#!/usr/bin/env bash
# =============================================================================
# Select the correction gain (alpha) and clip bound on the VALIDATION split by
# TRAJECTORY error, instead of by validation residual MSE as stage2_variants.py
# does. The paper reports ATE/RTE, so tuning on ATE/RTE removes an objective
# mismatch -- E_tcn is the proof: it wins on residual MSE, then degrades ATE ~20%.
#
# Validation = lists/list_val.txt (16 sequences, never in training, never the
# test set). Nothing here touches seen/unseen test data.
#
# Writes no result files: --out_dir is omitted, ATE/RTE are parsed from stdout.
# Re-uses the training cache, so each combination takes ~30-40 s.
#
#   GPU=3 bash tune_alpha_clip_val.sh                    # A_rf, 35 combinations, ~25 min
#   VARIANTS="A_rf D_mlp" bash tune_alpha_clip_val.sh    # two correctors
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

GPU="${GPU:-0}"
RUN_NAME="${RUN_NAME:-yolo26_eff_musgd}"
DROPOUT="${DROPOUT:-0.2}"
VARIANTS="${VARIANTS:-A_rf}"
ALPHAS="${ALPHAS:-0.25 0.5 0.75 1.0 1.25 1.5 2.0}"
CLIPS="${CLIPS:-0 0.25 0.5 0.75 1.0}"          # 0 = no clipping (code needs clip > 0)
TRAIN_DIR="output/train_${RUN_NAME}"
S2_DIR="output/stage2_models_${RUN_NAME}"
OUT_DIR="output/tune_alpha_clip_${RUN_NAME}"
CACHE="${CACHE:-cache/${RUN_NAME}}"

if [ -z "${TRAIN_ROOT:-}" ]; then
    for d in data/seen_subjects_train_set data/train_dataset; do
        if [ -d "${d}" ]; then TRAIN_ROOT="${d}"; break; fi
    done
fi
[ -d "${TRAIN_ROOT:-}" ] || { echo "ERROR: no training data folder found" >&2; exit 1; }
[ -f lists/list_val.txt ] || { echo "ERROR: missing lists/list_val.txt" >&2; exit 1; }

best_epoch=-1; BEST_CKPT=""
for f in "${TRAIN_DIR}"/checkpoints/checkpoint_[0-9]*.pt; do
    [ -e "${f}" ] || continue
    n="${f##*/checkpoint_}"; n="${n%.pt}"
    if [ "${n}" -gt "${best_epoch}" ]; then best_epoch="${n}"; BEST_CKPT="${f}"; fi
done
[ -n "${BEST_CKPT}" ] || { echo "ERROR: no checkpoint in ${TRAIN_DIR}/checkpoints" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${GPU}"
mkdir -p "${OUT_DIR}"
echo "checkpoint : ${BEST_CKPT} (epoch ${best_epoch})"
echo "val list   : lists/list_val.txt   root: ${TRAIN_ROOT}   GPU ${GPU}"

run_eval () {   # $1 = extra args; echoes "ATE RTE"
    # shellcheck disable=SC2086
    python source/ronin_yolo26_baseline_plain.py \
        --mode test --backbone yolo26_eff --model_dropout "${DROPOUT}" \
        --root_dir "${TRAIN_ROOT}" --test_list lists/list_val.txt \
        --model_path "${BEST_CKPT}" --dataset ronin \
        --window_size 200 --step_size 10 --cache_path "${CACHE}" --no_tqdm $1 2>/dev/null \
    | grep -oE "avg ATE:[0-9.]+, avg RTE:[0-9.]+" | tail -1 \
    | sed -E 's/avg ATE:([0-9.]+), avg RTE:([0-9.]+)/\1 \2/'
}

echo
echo "=== uncorrected baseline on the validation split"
BASE=$(run_eval "")
[ -n "${BASE}" ] || { echo "ERROR: baseline eval produced no result" >&2; exit 1; }
BASE_ATE=$(echo "${BASE}" | awk '{print $1}')
BASE_RTE=$(echo "${BASE}" | awk '{print $2}')
printf "  base   ATE %s   RTE %s\n" "${BASE_ATE}" "${BASE_RTE}"

for variant in ${VARIANTS}; do
    CORR="${S2_DIR}/${variant}_corrector.joblib"
    if [ ! -f "${CORR}" ]; then echo "SKIP ${variant}: ${CORR} not found"; continue; fi
    CSV="${OUT_DIR}/${variant}.csv"
    echo "alpha,clip,ate,rte" > "${CSV}"
    echo
    echo "=== ${variant}: sweeping alpha x clip on validation -> ${CSV}"
    for a in ${ALPHAS}; do
        for c in ${CLIPS}; do
            r=$(run_eval "--stage2_model_path ${CORR} --stage2_alpha ${a} --stage2_clip ${c}")
            if [ -z "${r}" ]; then echo "  a=${a} c=${c}  FAILED"; continue; fi
            ate=$(echo "${r}" | awk '{print $1}'); rte=$(echo "${r}" | awk '{print $2}')
            echo "${a},${c},${ate},${rte}" >> "${CSV}"
            printf "  a=%-5s clip=%-5s ATE %-9s RTE %-9s\n" "${a}" "${c}" "${ate}" "${rte}"
        done
    done

    python - "${CSV}" "${BASE_ATE}" "${BASE_RTE}" "${variant}" "${S2_DIR}" <<'PY'
import csv, json, os, sys
csv_path, base_ate, base_rte, variant, s2dir = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
rows = [(float(r['alpha']), float(r['clip']), float(r['ate']), float(r['rte']))
        for r in csv.DictReader(open(csv_path))]
if not rows:
    sys.exit("no rows in %s" % csv_path)
def show(tag, r):
    a, c, ate, rte = r
    print("  %-22s alpha=%-5g clip=%-5g  ATE %.4f (%+.1f%%)  RTE %.4f (%+.1f%%)"
          % (tag, a, c, ate, 100*(ate-base_ate)/base_ate, rte, 100*(rte-base_rte)/base_rte))
print("\n  validation base: ATE %.4f  RTE %.4f" % (base_ate, base_rte))
show("best RTE",  min(rows, key=lambda r: r[3]))
show("best ATE",  min(rows, key=lambda r: r[2]))
show("best balanced",  min(rows, key=lambda r: r[2]/base_ate + r[3]/base_rte))
p = os.path.join(s2dir, "%s_corrector_summary.json" % variant)
if os.path.exists(p):
    d = json.load(open(p))
    a0 = d.get("correction_alpha"); c0 = d.get("correction_clip") or 0.0
    cur = [r for r in rows if abs(r[0]-float(a0)) < 1e-9 and abs(r[1]-float(c0)) < 1e-9]
    if cur: show("current (residual-MSE)", cur[0])
    else:   print("  current (residual-MSE) alpha=%s clip=%s -- not in this grid" % (a0, c0))
PY
done
echo
echo "Pick ONE criterion before looking at any test numbers, then re-run the test"
echo "with --stage2_alpha/--stage2_clip set to the winner."
