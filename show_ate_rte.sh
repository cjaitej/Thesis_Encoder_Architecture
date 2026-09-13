#!/usr/bin/env bash
# Print ATE / RTE for every <split>_<variant> result folder.
#   bash show_ate_rte.sh                                  # default: output/test_yolo26_eff_musgd
#   bash show_ate_rte.sh output/test_yolo26_eff           # published Adam Eff results
#   bash show_ate_rte.sh output/test_*                    # everything at once
# "log" = the summary line the eval printed. "losses.csv" = mean over sequences,
# an independent check; n must be 32, otherwise sequences are missing.
set -uo pipefail

DIRS=("$@")
[ ${#DIRS[@]} -eq 0 ] && DIRS=("output/test_yolo26_eff_musgd")

for base in "${DIRS[@]}"; do
    [ -d "${base}" ] || { echo "skip (not a directory): ${base}"; continue; }
    echo "== ${base}"
    printf "   %-16s %-21s %-21s\n" "folder" "log: ATE / RTE" "losses.csv: ATE / RTE (n)"
    found=0
    for d in "${base}"/*/; do
        [ -d "${d}" ] || continue
        name=$(basename "${d}")
        from_log="-"
        if [ -f "${d}/test.log" ]; then
            v=$(grep -oE "avg ATE:[0-9.]+, avg RTE:[0-9.]+" "${d}/test.log" | tail -1 \
                | sed -E 's/avg ATE:([0-9.]+), avg RTE:([0-9.]+)/\1 \2/')
            [ -n "${v}" ] && from_log=$(echo "${v}" | awk '{printf "%.4f / %.4f", $1, $2}')
        fi
        from_csv="-"
        if [ -f "${d}/losses.csv" ]; then
            from_csv=$(awk -F, 'NR>1 && NF>=6 {a+=$5; r+=$6; n++} END {
                if (n) printf "%.4f / %.4f (%d)", a/n, r/n, n; else printf "-" }' "${d}/losses.csv")
        fi
        [ "${from_log}" = "-" ] && [ "${from_csv}" = "-" ] && continue
        printf "   %-16s %-21s %-21s\n" "${name}" "${from_log}" "${from_csv}"
        found=1
    done
    [ "${found}" -eq 0 ] && echo "   (no results found)"
    echo
done
