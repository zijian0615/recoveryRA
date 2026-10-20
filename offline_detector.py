"""Success-trajectory novelty check using the provided LeRobot v3 state/action data.

This is an offline diagnostic, not the image-conditioned detector or physical
recovery policy proposed in the research memo. It never uses simulator state.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "repos/dataset/data/chunk-000/file-000.parquet"
OUT = ROOT / "runs/offline_detector.json"


def features(rows):
    state = np.asarray([r["observation.state"] for r in rows], np.float64)
    action = np.asarray([r["action"] for r in rows], np.float64)
    # Absolute joints and the commanded next state are deployment-visible.
    return np.concatenate([state, action], axis=1)


def main():
    rows = pq.read_table(DATA, columns=["episode_index", "observation.state", "action"]).to_pylist()
    episodes = {}
    for row in rows:
        episodes.setdefault(row["episode_index"], []).append(row)
    if len(episodes) < 10:
        raise RuntimeError("Expected all ten demonstration episodes")
    train_ids, test_ids = sorted(episodes)[:8], sorted(episodes)[8:]
    train = {i: features(episodes[i]) for i in train_ids}
    test = {i: features(episodes[i]) for i in test_ids}
    pool = np.concatenate(list(train.values()))
    scale = np.maximum(pool.std(axis=0), 0.05)
    # Gripper is binary; weight state and command changes enough to matter.
    scale[[6, 13]] = 0.25

    def score(query, reference):
        ref = reference / scale
        ref_sq = np.sum(ref * ref, axis=1)
        distances = []
        for chunk in np.array_split(query / scale, max(1, (len(query) + 127) // 128)):
            d2 = np.sum(chunk * chunk, axis=1)[:, None] + ref_sq[None, :] - 2 * chunk @ ref.T
            distances.extend(np.sqrt(np.maximum(d2.min(axis=1), 0.0)))
        return np.asarray(distances)

    clean_train = np.concatenate([
        score(train[i], np.concatenate([v for j, v in train.items() if j != i]))
        for i in train_ids
    ])
    threshold = float(np.quantile(clean_train, 0.99))
    reference = np.concatenate(list(train.values()))
    results = []
    for i in test_ids:
        clean = test[i]
        clean_score = score(clean, reference)
        # Inject an attack into the command channel during a held-object interval.
        # This tests command novelty only; no physical object motion is simulated.
        attacked = clean.copy()
        closed = np.flatnonzero((clean[:, 6] > 0.5) & (clean[:, 13] > 0.5))
        if len(closed) < 5:
            raise RuntimeError(f"Episode {i} has no sustained closed-gripper interval")
        start = int(closed[len(closed) // 2])
        indices = np.arange(start, min(start + 5, len(clean)))
        attacked[indices, 13] = 0.0
        attack_score = score(attacked[indices], reference)
        results.append({
            "episode": i,
            "frames": len(clean),
            "clean_false_positive_frames": int((clean_score > threshold).sum()),
            "attack_start_frame": start,
            "attack_flagged_frames": int((attack_score > threshold).sum()),
            "attack_frames": len(indices),
            "max_attack_score": float(attack_score.max()),
        })
    report = {
        "method": "nearest-neighbor state/action novelty; 8 train, 2 held-out episodes",
        "threshold": threshold,
        "training_frames": int(sum(len(v) for v in train.values())),
        "heldout": results,
        "limitations": [
            "No images or phase labels; this is not D_t = -log p(w_t | phase, success).",
            "Synthetic command flip has no physical effect in this offline check.",
            "False-positive rate from two episodes is too small for a paper claim.",
        ],
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
