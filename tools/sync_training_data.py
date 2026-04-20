"""
Sync training data from Colab CSV → local JSON files used by the War Room UI.

Usage (run this on Colab after training, then download the updated JSONs):
    python tools/sync_training_data.py
    python tools/sync_training_data.py --csv path/to/training_log.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).parent.parent


def load_csv(path: Path) -> List[Dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def rebuild_training_curves(rows: List[Dict], config: Dict) -> Dict:
    episodes, r_invs, f1s, scheme_types = [], [], [], []
    for row in rows:
        episodes.append(int(row["episode"]))
        r_invs.append(round(float(row["r_inv"]), 4))
        f1s.append(round(float(row["f1"]), 4))
        scheme_types.append(row["scheme_type"])
    return {
        "episodes": episodes,
        "r_inv": r_invs,
        "f1": f1s,
        "scheme_types": scheme_types,
        "config": config,
    }


def rebuild_f1_history(rows: List[Dict], batch_size: int = 5) -> List[Dict]:
    history = []
    # Compute rolling mean F1 per scheme type at each batch boundary
    scheme_f1s: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        scheme = row["scheme_type"]
        scheme_f1s[scheme].append(float(row["f1"]))
        ep = int(row["episode"])
        if ep % batch_size == 0:
            entry: Dict = {"episode": ep, "mean_f1": {}}
            for s, vals in scheme_f1s.items():
                entry["mean_f1"][s] = round(sum(vals) / len(vals), 4)
            history.append(entry)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(_ROOT / "training_log.csv"))
    parser.add_argument("--out-dir", default=str(_ROOT))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run training first.")
        sys.exit(1)

    rows = load_csv(csv_path)
    if not rows:
        print("ERROR: CSV is empty.")
        sys.exit(1)

    print(f"Loaded {len(rows)} episodes from {csv_path}")

    config = {
        "model": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        "num_episodes": len(rows),
        "grpo_g": 4,
        "lr": 5e-5,
    }

    out = Path(args.out_dir)

    curves = rebuild_training_curves(rows, config)
    (out / "training_curves.json").write_text(json.dumps(curves, indent=2))
    print(f"  Wrote training_curves.json  ({len(rows)} episodes)")

    f1_hist = rebuild_f1_history(rows)
    (out / "f1_history.json").write_text(json.dumps(f1_hist, indent=2))
    print(f"  Wrote f1_history.json       ({len(f1_hist)} batch entries)")

    print("Done. Download training_curves.json and f1_history.json to your local heist/ folder.")


if __name__ == "__main__":
    main()
