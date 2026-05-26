#!/usr/bin/env bash

STATS_BIN=./extract_stats
INPUT_DIR=assets/harrison_pcp_positive_100_0.0025
OUT_STAT_DIR=summary_stats_harrison_pcp_positive_100_0.0025

mkdir -p "$OUT_STAT_DIR"

files=("$INPUT_DIR"/pcp_harrison_100_*.png)
total=${#files[@]}
count=0

echo "Processing $total images..."

for img in "${files[@]}"; do
    count=$((count + 1))

    base=$(basename "$img" .png)

    # Extract correlation value
    # scatter_bw_70_-0.35_z  ->  -0.35
    corr=$(printf "%s\n" "$base" | grep -oE -- '-?[0-9]+\.[0-9]+')

    if [[ -z "$corr" ]]; then
        echo "ERROR: Could not extract correlation from $base"
        continue
    fi

    suffix=$(awk -v c="$corr" 'BEGIN {
        if (c < 0) print "n";
        else if (c > 0) print "p";
        else print "z";
    }')

    out_stats="$OUT_STAT_DIR/stats_corr_${corr}_${suffix}.csv"

    printf "\r[%3d/%3d] corr=%s" "$count" "$total" "$corr"

    "$STATS_BIN" "$img" "$out_stats" > /dev/null 2>&1

done

echo
echo "Done. Statistics saved in $OUT_STAT_DIR"
