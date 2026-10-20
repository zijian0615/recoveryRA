"""Summarise a run_matrix.sh output directory into summary.json and a Markdown table."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ARMS = ["clean", "stop", "continue", "inverse_replay", "ours", "ours_rl_v1", "ours_rl"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main(root):
    root = Path(root)
    rows = defaultdict(dict)
    for rep in sorted(root.glob("s*_*/report.json")):
        seed, arm = rep.parent.name.split("_", 1)
        rows[int(seed[1:])][arm] = json.loads(rep.read_text())
    seeds = sorted(rows)
    # An attack trial is valid when the grasp checkpoint held before the attack and
    # the attack actually broke it (screw no longer lifted after the attack window).
    def valid(r):
        return bool(r["checkpoint_before_attack"]) and r["checkpoint_after_attack"] is False

    out = {"seeds": seeds, "arms": {}}
    for arm in [a for a in ARMS if any(a in rows[s] for s in seeds)]:
        allr = [rows[s][arm] for s in seeds if arm in rows[s]]
        rs = allr if arm == "clean" else [r for r in allr if valid(r)]
        excluded = {r["seed"]: ("no hold before attack" if not r["checkpoint_before_attack"]
                                else "screw still held after attack")
                    for r in allr if r not in rs}
        k = sum(r["success"] for r in rs)
        entry = {"n": len(rs), "excluded": excluded, "success": k, "success_rate": k / len(rs) if rs else None,
                 "success_ci95": wilson(k, len(rs)),
                 "per_seed": {r["seed"]: r["success"] for r in rs}}
        if arm == "clean":
            entry["runs_with_alarm"] = sum(r["first_alarm"] is not None for r in rs)
        else:
            det = [r for r in rs if r["first_alarm"] is not None and r["first_alarm"] >= r["attack_start"]]
            entry["detected"] = len(det)
            entry["false_alarm_before_attack"] = sum(r["alarms_before_attack"] > 0 for r in rs)
            entry["detection_delay_steps"] = [r["first_alarm"] - r["attack_start"] for r in det]
        if arm.startswith("ours"):
            entry["restore_outcomes"] = [r["restore_outcome"] for r in rs]
            ck = sum(bool(r["checkpoint_restored"]) for r in rs)
            entry["checkpoint_restored"] = ck
            entry["checkpoint_ci95"] = wilson(ck, len(rs))
            entry["recovery_latency_steps"] = [
                next((t["step"] for t in json.loads((root / f"s{r['seed']}_{arm}/report_trace.json").read_text())
                      if t.get("mode") == "policy" and t["step"] > r["response_step"]), None)
                - r["response_step"] for r in rs if r["checkpoint_restored"]]
        out["arms"][arm] = entry
    (root / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    lines = ["| arm | n | success | 95% CI |", "| --- | --- | --- | --- |"]
    for arm in out["arms"]:
        e = out["arms"][arm]
        lo, hi = e["success_ci95"]
        lines.append(f"| {arm} | {e['n']} | {e['success']} | {lo:.2f}-{hi:.2f} |")
    print("\n".join(lines))
    print(json.dumps({a: {k: v for k, v in e.items() if k not in ("per_seed",)}
                      for a, e in out["arms"].items()}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
