"""Generate leakage-free degree labels from the training-period graph only.

The original helper generated labels from all edges. For paper experiments,
this script builds node degrees using only edges with timestamp <= the 70%
chronological split, then assigns a label to every interaction according to
the source node's training-period degree.
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd


def generate_labels(csv_path, train_quantile=0.70):
    df = pd.read_csv(csv_path)
    src = df.u.values
    dst = df.i.values
    ts = df.ts.values

    val_time = np.quantile(ts, train_quantile)
    train_mask = ts <= val_time

    degree = Counter()
    for s, d in zip(src[train_mask], dst[train_mask]):
        degree[int(s)] += 1
        degree[int(d)] += 1

    train_src_degrees = np.array([degree[int(s)] for s in src[train_mask]], dtype=np.float64)
    if len(train_src_degrees) == 0:
        raise ValueError(f"No training edges found in {csv_path}")

    threshold = np.median(train_src_degrees)
    all_src_degrees = np.array([degree[int(s)] for s in src], dtype=np.float64)
    labels = (all_src_degrees >= threshold).astype(np.int32)

    stats = {
        "num_edges": len(df),
        "num_train_edges": int(train_mask.sum()),
        "threshold": float(threshold),
        "positives": int(labels.sum()),
        "negatives": int(len(labels) - labels.sum()),
        "positive_rate": float(labels.mean()),
    }
    return labels, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["reddit", "uci", "enron"])
    parser.add_argument("--processed_dir", default="./processed")
    parser.add_argument("--suffix", default="train_degree_labels")
    parser.add_argument("--train_quantile", type=float, default=0.70)
    args = parser.parse_args()

    os.makedirs(args.processed_dir, exist_ok=True)
    rows = []

    for dataset in args.datasets:
        csv_path = os.path.join(args.processed_dir, f"ml_{dataset}.csv")
        if not os.path.exists(csv_path):
            print(f"{dataset}: missing {csv_path}, skipped")
            continue

        labels, stats = generate_labels(csv_path, train_quantile=args.train_quantile)
        out_path = os.path.join(args.processed_dir, f"ml_{dataset}_{args.suffix}.npy")
        np.save(out_path, labels)

        row = {"dataset": dataset, "output": out_path, **stats}
        rows.append(row)
        print(
            f"{dataset}: saved {out_path} | pos={stats['positives']} "
            f"neg={stats['negatives']} rate={stats['positive_rate']:.4f} "
            f"threshold={stats['threshold']:.2f}"
        )

    if rows:
        summary_path = os.path.join(args.processed_dir, f"{args.suffix}_summary.csv")
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
