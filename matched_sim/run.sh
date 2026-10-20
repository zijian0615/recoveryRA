#!/usr/bin/env bash
# Usage: run.sh OUT_DIR -- <eval_v05.py args without --report/--port>
# Runs one episode with a fresh ACT server; writes OUT_DIR/report.json.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
OUT="$1"; shift; [[ "${1:-}" == "--" ]] && shift
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
PORT="${PORT:-8765}"
cd "$ROOT/screw_sim"
/home/sidc-lab/lerobot/.venv/bin/python act_policy_server.py --ckpt "$ROOT/repos/policy" --port "$PORT" >"$OUT/policy.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  (echo >"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && break
  kill -0 "$SERVER_PID" 2>/dev/null || { cat "$OUT/policy.log"; exit 1; }
  sleep 1
done
cd "$HERE"
OMNI_KIT_ACCEPT_EULA=YES /home/sidc-lab/isaacsim-venv/bin/python eval_v05.py \
  --port "$PORT" --report "$OUT/report.json" "$@" >"$OUT/sim.log" 2>&1 || true
grep -h '^\[eval\]' "$OUT/sim.log" || { grep -n 'Traceback\|Error' "$OUT/sim.log" | tail -20; exit 1; }
