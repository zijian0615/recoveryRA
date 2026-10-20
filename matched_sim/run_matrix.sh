#!/usr/bin/env bash
# Paired comparison: same seed (layout) and attack for every response.
# Usage: run_matrix.sh OUT_ROOT "SEEDS" [PARALLEL]
# Env: RENDER (default realtime), ARMS (default "stop continue inverse_replay ours"),
#      RL_ACTOR (checkpoint for the ours_rl arm)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$1"; SEEDS="$2"; PAR="${3:-3}"
CAL="$(ls -d "$HERE"/../runs/v05/cal_s10{0,1,2}/report_trace.json | tr '\n' ' ')"
jobs=()
for s in $SEEDS; do
  jobs+=("$s clean none none")
  for r in ${ARMS:-stop continue inverse_replay ours}; do jobs+=("$s $r open_when_held $r"); done
done
i=0
for j in "${jobs[@]}"; do
  read -r s name attack resp <<<"$j"
  echo "$i $s $name $attack $resp"
  i=$((i+1))
done | xargs -P "$PAR" -L 1 bash -c '
  i=$0; s=$1; name=$2; attack=$3; resp=$4
  d="'"$OUT"'/s${s}_${name}"
  [[ -s "$d/report.json" ]] && exit 0
  PORT=$((8900 + i % 50)) "'"$HERE"'/run.sh" "$d" -- --seed "$s" --jitter 0 --replan-interval 0 \
    --steps 600 --attack "$attack" --attack-delay 10 --attack-duration 5 --response "$resp" \
    --render '"${RENDER:-realtime}"' --rl-actor '"${RL_ACTOR:-none}"' --trace --calibration-traces '"$CAL"' >/dev/null 2>&1
  echo "done s$s $name"'
