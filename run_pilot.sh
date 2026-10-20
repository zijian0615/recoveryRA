#!/usr/bin/env bash
# One invocation prepares paired, same-seed normal and attack/continue runs.
# This is a pilot; it does not evaluate a restore policy.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SEED="${SEED:-20}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/runs/pilot_${STAMP}_seed_${SEED}"
mkdir -p "$OUT"

SEED="$SEED" bash "$ROOT/run_baseline.sh" "$OUT/normal"
SEED="$SEED" bash "$ROOT/run_baseline.sh" "$OUT/attack_continue" \
  --attack-when-held --attack-duration 5

python3 - "$OUT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
normal = json.loads((root / "normal/report.json").read_text())
attack = json.loads((root / "attack_continue/report.json").read_text())
summary = {
    "seed": normal["seed"],
    "normal_success": normal["success"],
    "attack_triggered": attack["actual_attack_step"] is not None,
    "attack_step": attack["actual_attack_step"],
    "grasp_checkpoint_before_attack": attack["checkpoint_before_attack"],
    "grasp_checkpoint_after_attack": attack["checkpoint_after_attack"],
    "attack_continue_success": attack["success"],
    "valid_comparison": bool(normal["success"] and attack["actual_attack_step"] is not None),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
