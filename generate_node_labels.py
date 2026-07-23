"""
Generate synthetic node classification labels for datasets that lack them.

DGB Reddit, UCI, and Enron have all-zero edge labels — they were designed for
link prediction only. For node classification, we create structurally meaningful
labels based on node degree:

  label = 1 if the source node's total degree (over all time) >= median degree
  label = 0 otherwise

This is a standard self-supervised node classification task used in graph
SSL papers (GRACE, BGRL, etc.): predicting structural role from embeddings.
"""

import numpy as np
import pandas as pd
import os
import argparse
from collections import Counter


def generate_degree_labels(csv_path):
    """Label nodes by whether their total degree exceeds the edge-weighted median.

    The edge-weighted median ensures exactly ~50/50 split: each edge is labeled
    by its source node's total degree, and the threshold is the median of these
    edge-level degree values.
    """
    df = pd.read_csv(csv_path)
    src = df.u.values
    dst = df.i.values

    # Count total degree for each node (both as source and destination)
    degree = Counter()
    for s, d in zip(src, dst):
        degree[s] += 1
        degree[d] += 1

    # Edge-level degrees: assign each edge its source node's total degree
    edge_degrees = np.array([degree[s] for s in src], dtype=np.float64)

    # Use edge-weighted median as threshold → ~50/50 split
    threshold = np.median(edge_degrees)
    print(f'  Edge-weighted median degree: {threshold:.1f}, '
          f'max degree: {max(degree.values())}, num_nodes: {len(degree)}')

    labels = (edge_degrees >= threshold).astype(np.int32)

    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['reddit', 'uci', 'enron'])
    parser.add_argument('--output_dir', default='./processed')
    args = parser.parse_args()

    for ds in args.datasets:
        csv_path = f'{args.output_dir}/ml_{ds}.csv'
        if not os.path.exists(csv_path):
            print(f'{ds}: SKIP — {csv_path} not found')
            continue

        labels = generate_degree_labels(csv_path)

        pos = labels.sum()
        neg = len(labels) - pos
        print(f'{ds}: {pos} positive (high-degree), {neg} negative (low-degree) — '
              f'{pos/len(labels)*100:.1f}%')

        out_path = f'{args.output_dir}/ml_{ds}_node_labels.npy'
        np.save(out_path, labels)
        print(f'  Saved to {out_path}')
        print()


if __name__ == '__main__':
    main()
