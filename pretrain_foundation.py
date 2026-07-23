"""
Pre-training script for the Temporal Graph Foundation Model.

Jointly trains on Wikipedia + Reddit + MOOC with 4 self-supervised pretext tasks:
  MTLP — Masked Temporal Link Prediction
  TNP  — Temporal Neighborhood Prediction
  CDC  — Cross-Domain Contrastive (via domain discrimination)
  EFR  — Edge Feature Reconstruction (bidirectional consistency)

Usage:
  python pretrain_foundation.py --epochs 50 --shared_dim 256 --lr 0.0005

After pre-training, the model is saved to ./saved_models/foundation_pretrained.pth
and can be fine-tuned with finetune_foundation.py.
"""

import math
import time
import random
import argparse
import json
import logging
import sys
import os

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import average_precision_score, roc_auc_score

from graph import NeighborFinder
from utils import RandEdgeSampler, EarlyStopMonitor
from foundation_model import TemporalGraphFoundationModel


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser('Temporal Graph Foundation Model — Pre-training')
parser.add_argument('--datasets', type=str, nargs='+',
                    default=['wikipedia', 'reddit', 'mooc'],
                    help='datasets to pre-train on')
parser.add_argument('--epochs', type=int, default=50, help='pre-training epochs')
parser.add_argument('--bs', type=int, default=512, help='batch size per dataset per step')
parser.add_argument('--shared_dim', type=int, default=256, help='shared hidden dimension')
parser.add_argument('--time_dim', type=int, default=128, help='time encoding dimension')
parser.add_argument('--num_layers', type=int, default=2, help='temporal encoder layers')
parser.add_argument('--num_neighbors', type=int, default=20, help='historical neighbors to sample')
parser.add_argument('--n_head', type=int, default=4, help='attention heads')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout rate')
parser.add_argument('--gpu', type=int, default=0, help='GPU index (-1 for CPU)')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--prefix', type=str, default='foundation', help='save prefix')
parser.add_argument('--grad_accum', type=int, default=1, help='gradient accumulation steps')
parser.add_argument('--eval_every', type=int, default=5, help='evaluate every N epochs')
parser.add_argument('--limit_batches', type=int, default=200,
                    help='max batches per dataset per epoch (limits epoch length)')
parser.add_argument('--total_batches_per_epoch', type=int, default=0,
                    help='optional compute-matched batch budget shared equally across datasets; '
                         '0 keeps the per-dataset limit behavior')
parser.add_argument('--domain_weighting', choices=['uniform', 'compatibility', 'shuffled'],
                    default='uniform',
                    help='source-domain loss weighting strategy')
parser.add_argument('--domain_distance_csv', type=str, default='',
                    help='pairwise train-period domain-distance CSV')
parser.add_argument('--target_dataset', type=str, default='',
                    help='unlabeled target domain used to compute compatibility weights')
parser.add_argument('--compatibility_temperature', type=float, default=1.0,
                    help='softmax temperature for distance-based source weights')
parser.add_argument('--compatibility_uniform_mix', type=float, default=0.10,
                    help='fraction of uniform weighting mixed into compatibility weights')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
os.makedirs('log', exist_ok=True)
fh = logging.FileHandler(f'log/foundation_pretrain_{time.time():.0f}.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

logger.info(args)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

if args.gpu >= 0 and torch.cuda.is_available():
    device = torch.device(f'cuda:{args.gpu}')
else:
    device = torch.device('cpu')
logger.info(f'Using device: {device}')

# ---------------------------------------------------------------------------
# Load all datasets
# ---------------------------------------------------------------------------

def load_single_dataset(name):
    """Load one dataset, build NeighborFinder, return all necessary objects."""
    csv_path = f'./processed/ml_{name}.csv'
    npy_path = f'./processed/ml_{name}.npy'
    node_npy_path = f'./processed/ml_{name}_node.npy'

    df = pd.read_csv(csv_path)
    e_feat = np.load(npy_path)
    n_feat = np.load(node_npy_path)

    src_l = df.u.values
    dst_l = df.i.values
    e_idx_l = df.idx.values
    ts_l = df.ts.values
    label_l = df.label.values

    max_idx = max(src_l.max(), dst_l.max())

    # Train/val/test split by time
    val_time, test_time = list(np.quantile(ts_l, [0.70, 0.85]))

    # Build graphs
    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    full_ngh_finder = NeighborFinder(adj_list, uniform=False)

    # Training data: edges before val_time
    train_flag = ts_l <= val_time
    train_src = src_l[train_flag]
    train_dst = dst_l[train_flag]
    train_ts = ts_l[train_flag]

    # Validation data
    val_flag = (ts_l > val_time) & (ts_l <= test_time)
    val_src = src_l[val_flag]
    val_dst = dst_l[val_flag]
    val_ts = ts_l[val_flag]

    # Test data
    test_flag = ts_l > test_time
    test_src = src_l[test_flag]
    test_dst = dst_l[test_flag]
    test_ts = ts_l[test_flag]

    rand_sampler = RandEdgeSampler(train_src, train_dst)

    raw_node_dim = n_feat.shape[1]
    raw_edge_dim = e_feat.shape[1]

    config = {
        'name': name,
        'raw_node_dim': raw_node_dim,
        'raw_edge_dim': raw_edge_dim,
        'n_nodes': max_idx,
        'n_edges': len(e_feat) - 1,
    }

    data = {
        'df': df,
        'n_feat': n_feat,
        'e_feat': e_feat,
        'ngh_finder': full_ngh_finder,
        'rand_sampler': rand_sampler,
        'train_src': train_src,
        'train_dst': train_dst,
        'train_ts': train_ts,
        'val_src': val_src,
        'val_dst': val_dst,
        'val_ts': val_ts,
        'test_src': test_src,
        'test_dst': test_dst,
        'test_ts': test_ts,
        'val_time': val_time,
        'test_time': test_time,
    }

    return config, data


logger.info('Loading datasets...')
dataset_configs = []
dataset_data = {}

for ds_name in args.datasets:
    try:
        cfg, data = load_single_dataset(ds_name)
        dataset_configs.append(cfg)
        dataset_data[ds_name] = data
        logger.info(f'  {ds_name}: {cfg["n_nodes"]} nodes, {len(data["train_src"])} train edges, '
                    f'node_dim={cfg["raw_node_dim"]}, edge_dim={cfg["raw_edge_dim"]}')
    except FileNotFoundError as e:
        logger.warning(f'  {ds_name}: SKIPPED — {e}')
        logger.warning(f'  Run "python process.py" first to generate processed/ml_{ds_name}.csv')

if len(dataset_configs) == 0:
    logger.error('No datasets loaded. Run process.py first.')
    sys.exit(1)


def build_domain_weights(dataset_names):
    """Return source probabilities and mean-one loss multipliers."""
    count = len(dataset_names)
    uniform = np.full(count, 1.0 / count, dtype=np.float64)
    distances = {name: 0.0 for name in dataset_names}

    if args.domain_weighting == 'uniform':
        probabilities = uniform
    else:
        if not args.target_dataset:
            raise ValueError('--target_dataset is required for compatibility weighting')
        if not args.domain_distance_csv:
            raise ValueError('--domain_distance_csv is required for compatibility weighting')
        if args.compatibility_temperature <= 0:
            raise ValueError('--compatibility_temperature must be positive')
        if not 0.0 <= args.compatibility_uniform_mix <= 1.0:
            raise ValueError('--compatibility_uniform_mix must be in [0, 1]')

        distance_frame = pd.read_csv(args.domain_distance_csv, index_col=0)
        missing = [
            name for name in [args.target_dataset, *dataset_names]
            if name not in distance_frame.index or name not in distance_frame.columns
        ]
        if missing:
            raise ValueError(f'Datasets missing from distance matrix: {sorted(set(missing))}')

        distance_values = distance_frame.loc[args.target_dataset, dataset_names].to_numpy(
            dtype=np.float64
        )
        distances = dict(zip(dataset_names, distance_values.tolist()))
        logits = -distance_values / args.compatibility_temperature
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        probabilities = (
            (1.0 - args.compatibility_uniform_mix) * probabilities
            + args.compatibility_uniform_mix * uniform
        )
        if args.domain_weighting == 'shuffled':
            probabilities = np.random.RandomState(args.seed).permutation(probabilities)

    probability_map = dict(zip(dataset_names, probabilities.tolist()))
    # Mean-one multipliers keep the overall gradient scale comparable to uniform training.
    multiplier_map = {
        name: probability_map[name] * count for name in dataset_names
    }
    return probability_map, multiplier_map, distances


loaded_dataset_names = [cfg['name'] for cfg in dataset_configs]
domain_probabilities, domain_loss_weights, target_distances = build_domain_weights(
    loaded_dataset_names
)
logger.info(f'Domain weighting: {args.domain_weighting}')
logger.info(f'Target dataset: {args.target_dataset or "not specified"}')
for ds_name in loaded_dataset_names:
    logger.info(
        f'  {ds_name}: distance={target_distances[ds_name]:.4f} '
        f'probability={domain_probabilities[ds_name]:.4f} '
        f'loss_multiplier={domain_loss_weights[ds_name]:.4f}'
    )

# ---------------------------------------------------------------------------
# Build foundation model
# ---------------------------------------------------------------------------

logger.info('Building foundation model...')
model = TemporalGraphFoundationModel(
    dataset_configs=dataset_configs,
    shared_dim=args.shared_dim,
    time_dim=args.time_dim,
    num_layers=args.num_layers,
    n_head=args.n_head,
    drop_out=args.drop_out,
    num_neighbors=args.num_neighbors,
)
model = model.to(device)

# Load dataset raw embeddings
for ds_name in args.datasets:
    if ds_name in dataset_data:
        model.load_dataset_embeddings(ds_name, dataset_data[ds_name]['n_feat'], dataset_data[ds_name]['e_feat'])

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f'Model params: {total_params:,} total, {trainable_params:,} trainable')

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

os.makedirs('./saved_models', exist_ok=True)
os.makedirs('./saved_checkpoints', exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}_pretrained.pth'
FINAL_MODEL_SAVE_PATH = f'./saved_models/{args.prefix}_final.pth'
WEIGHT_METADATA_PATH = f'./saved_models/{args.prefix}_domain_weights.json'
with open(WEIGHT_METADATA_PATH, 'w', encoding='utf-8') as handle:
    json.dump(
        {
            'seed': args.seed,
            'source_datasets': loaded_dataset_names,
            'target_dataset': args.target_dataset,
            'domain_weighting': args.domain_weighting,
            'distance_csv': args.domain_distance_csv,
            'temperature': args.compatibility_temperature,
            'uniform_mix': args.compatibility_uniform_mix,
            'distances': target_distances,
            'probabilities': domain_probabilities,
            'loss_multipliers': domain_loss_weights,
        },
        handle,
        indent=2,
        sort_keys=True,
    )

# ---------------------------------------------------------------------------
# Pre-training loop
# ---------------------------------------------------------------------------

logger.info(f'Starting pre-training on {len(dataset_configs)} datasets for {args.epochs} epochs')
logger.info(f'Pretext tasks: MTLP + TNP + CDC + EFR + NPP')

best_val_ap = -1.0
early_stopper = EarlyStopMonitor(max_round=10, higher_better=True)

for epoch in range(args.epochs):
    model.train()
    epoch_losses = {'mtlp': 0.0, 'tnp': 0.0, 'cdc': 0.0, 'efr': 0.0, 'npp': 0.0, 'total': 0.0}
    epoch_weighted_total = 0.0
    epoch_domain_total = {name: 0.0 for name in loaded_dataset_names}
    epoch_domain_batches = {name: 0 for name in loaded_dataset_names}
    total_batches = 0

    # Iterate over datasets in shuffled order
    ds_order = list(loaded_dataset_names)
    random.shuffle(ds_order)

    for ds_name in ds_order:
        if ds_name not in dataset_data:
            continue
        data = dataset_data[ds_name]

        train_src = data['train_src']
        train_dst = data['train_dst']
        train_ts = data['train_ts']
        ngh_finder = data['ngh_finder']
        rand_sampler = data['rand_sampler']

        num_instances = len(train_src)
        per_dataset_budget = args.limit_batches
        if args.total_batches_per_epoch > 0:
            per_dataset_budget = min(
                per_dataset_budget,
                math.ceil(args.total_batches_per_epoch / max(len(loaded_dataset_names), 1)),
            )
        num_batches = min(math.ceil(num_instances / args.bs), per_dataset_budget)
        idx_list = np.arange(num_instances)
        np.random.shuffle(idx_list)

        for k in range(num_batches):
            s_idx = k * args.bs
            e_idx = min(num_instances, s_idx + args.bs)
            batch_idx = idx_list[s_idx:e_idx]
            size = len(batch_idx)
            if size == 0:
                continue

            src_l = train_src[batch_idx]
            dst_l = train_dst[batch_idx]
            ts_l = train_ts[batch_idx]
            neg_src, neg_dst = rand_sampler.sample(size)

            # Compute neighborhood stats for TNP pretext task
            ngh_stats = model.compute_neighborhood_stats(
                ds_name, ngh_finder, src_l, ts_l
            )

            # NPP uses same targets as TNP (reuse computed stats)
            node_props = ngh_stats

            # Pretext forward pass
            batch_data = {
                'src_idx': src_l,
                'dst_idx': dst_l,
                'cut_time': ts_l,
                'neg_dst_idx': neg_dst,
                'ngh_stats': ngh_stats,
                'node_props': node_props,
            }

            losses, _, _ = model.pretrain_forward(ds_name, ngh_finder, batch_data)
            total_loss = model.compute_pretrain_loss(losses)
            weighted_total_loss = total_loss * domain_loss_weights[ds_name]

            (weighted_total_loss / args.grad_accum).backward()

            if (k + 1) % args.grad_accum == 0 or k == num_batches - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            epoch_losses['total'] += total_loss.item()
            epoch_weighted_total += weighted_total_loss.item()
            epoch_domain_total[ds_name] += total_loss.item()
            epoch_domain_batches[ds_name] += 1
            total_batches += 1

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= max(total_batches, 1)
    epoch_weighted_total /= max(total_batches, 1)

    scheduler.step()

    # Log task weights
    task_weights = torch.exp(-model.task_log_vars).detach().cpu().numpy()
    logger.info(
        f'Epoch {epoch:3d} | total={epoch_losses["total"]:.4f} '
        f'weighted_total={epoch_weighted_total:.4f} '
        f'mtlp={epoch_losses["mtlp"]:.4f} tnp={epoch_losses["tnp"]:.4f} '
        f'cdc={epoch_losses["cdc"]:.4f} efr={epoch_losses["efr"]:.4f} '
        f'npp={epoch_losses["npp"]:.4f} '
        f'| w_mtlp={task_weights[0]:.3f} w_tnp={task_weights[1]:.3f} '
        f'w_cdc={task_weights[2]:.3f} w_efr={task_weights[3]:.3f} '
        f'w_npp={task_weights[4]:.3f} '
        f'| lr={scheduler.get_last_lr()[0]:.2e}'
    )
    logger.info(
        '  Domain losses: ' + ' '.join(
            f'{name}={epoch_domain_total[name] / max(epoch_domain_batches[name], 1):.4f}'
            for name in loaded_dataset_names
        )
    )

    # Validation (link prediction AP on each dataset)
    if (epoch + 1) % args.eval_every == 0:
        model.eval()
        val_aps = {}
        with torch.no_grad():
            for ds_name in loaded_dataset_names:
                if ds_name not in dataset_data:
                    continue
                data = dataset_data[ds_name]
                ngh_finder = data['ngh_finder']
                val_src = data['val_src'][:500]
                val_dst = data['val_dst'][:500]
                val_ts = data['val_ts'][:500]
                rand_sampler = data['rand_sampler']

                pos_scores, neg_scores = [], []
                for j in range(0, len(val_src), 64):
                    s = val_src[j:j+64]
                    d = val_dst[j:j+64]
                    t = val_ts[j:j+64]
                    ns, nd = rand_sampler.sample(len(s))
                    src_emb = model.encode_nodes(ds_name, ngh_finder, s, t)
                    dst_emb = model.encode_nodes(ds_name, ngh_finder, d, t)
                    neg_emb = model.encode_nodes(ds_name, ngh_finder, nd, t)
                    pos_logit = model.link_head(src_emb, dst_emb)
                    neg_logit = model.link_head(src_emb, neg_emb)
                    pos_scores.append(torch.sigmoid(pos_logit).cpu().numpy())
                    neg_scores.append(torch.sigmoid(neg_logit).cpu().numpy())

                pos = np.concatenate(pos_scores)
                neg = np.concatenate(neg_scores)
                preds = np.concatenate([pos, neg])
                labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
                ap = average_precision_score(labels, preds)
                auc = roc_auc_score(labels, preds)
                val_aps[ds_name] = ap
                logger.info(f'  Val {ds_name}: AP={ap:.4f} AUC={auc:.4f}')

        mean_val_ap = sum(
            domain_probabilities[name] * val_aps[name] for name in val_aps
        ) if val_aps else 0.0
        logger.info(f'  Weighted mean val AP: {mean_val_ap:.4f}')

        # Save best checkpoints
        if mean_val_ap > best_val_ap + 1e-4:
            best_val_ap = mean_val_ap
            model.save_pretrained(MODEL_SAVE_PATH)
            logger.info(f'  [BEST] Model saved to {MODEL_SAVE_PATH} (AP={best_val_ap:.4f})')

        ckpt_path = f'./saved_checkpoints/{args.prefix}_epoch{epoch}.pth'
        model.save_pretrained(ckpt_path)

        if early_stopper.early_stop_check(mean_val_ap):
            logger.info(f'Early stopping at epoch {epoch}')
            break

# ---------------------------------------------------------------------------
# Final save
# ---------------------------------------------------------------------------

model.save_pretrained(FINAL_MODEL_SAVE_PATH)
if best_val_ap < 0:
    model.save_pretrained(MODEL_SAVE_PATH)
    logger.info(f'No validation checkpoint was selected; saved final model to {MODEL_SAVE_PATH}')
logger.info(f'Pre-training complete. Final model saved to {FINAL_MODEL_SAVE_PATH}')
logger.info(f'Best model retained at {MODEL_SAVE_PATH}')
logger.info(f'Best validation AP: {best_val_ap:.4f}')
logger.info('Next step: run finetune_foundation.py for downstream task transfer.')
