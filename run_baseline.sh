#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SIM="$ROOT/screw_sim"
POLICY="$ROOT/repos/policy"
ACT_PY="/home/sidc-lab/lerobot/.venv/bin/python"
ISAAC_PY="/home/sidc-lab/isaacsim-venv/bin/python"
OUT="${1:-$ROOT/runs/baseline_multiscrew_seed_20}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
if [[ -e "$OUT/report.json" ]]; then
  echo "Output directory already contains a report: $OUT" >&2
  echo "Choose a fresh output directory as the first argument." >&2
  exit 2
fi
SEED="${SEED:-20}"
STEPS="${STEPS:-350}"
PORT="${PORT:-8765}"

cd "$SIM"
OMNI_KIT_ACCEPT_EULA=YES "$ACT_PY" act_policy_server.py --ckpt "$POLICY" --port "$PORT" >"$OUT/policy.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  if (echo >"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    cat "$OUT/policy.log"
    exit 1
  fi
  sleep 1
done

sim_status=0
OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_PY" eval_act_sim.py \
  --seed "$SEED" --port "$PORT" --steps "$STEPS" \
  --report "$OUT/report.json" "${@:2}" >"$OUT/sim.log" 2>&1 || sim_status=$?
if [[ "$sim_status" -ne 0 || ! -s "$OUT/report.json" ]]; then
  echo "Simulation did not produce a report. Relevant log lines:" >&2
  rg -n 'Traceback|Error|Exception|ConnectionRefused|success' "$OUT/sim.log" | tail -30 >&2 || true
  exit 1
fi
cat "$OUT/report.json"
