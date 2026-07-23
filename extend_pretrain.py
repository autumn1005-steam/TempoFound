"""
Extend a pre-trained foundation model with new datasets and continue pre-training.

Usage:
  python extend_pretrain.py --checkpoint ./saved_models/foundation_v3_pretrained.pth \\
      --new_datasets uci enron --epochs 30 --prefix foundation_v4
"""

import math
import time
import random
import argparse
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

parser = argparse.ArgumentParser('Extend Foundation Model with New Datasets')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='path to pre-trained foundation model')
parser.add_argument('--new_datasets', type=str, nargs='+',
                    default=['uci', 'enron'],
                    help='new datasets to add')
parser.add_argument('--epochs', type=int, default=30, help='continued pre-training epochs')
parser.add_argument('--bs', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (lower for fine-tuning)')
parser.add_argument('--gpu', type=int, default=0, help='GPU index (-1 for CPU)')
parser.add_argument('--prefix', type=str, default='foundation_v4', help='save prefix')
parser.add_argument('--grad_accum', type=int, default=1, help='gradient accumulation steps')
parser.add_argument('--eval_every', type=int, default=5, help='evaluate every N epochs')
parser.add_argument('--limit_batches', type=int, default=50,
                    help='max batches per dataset per epoch')
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
os.makedirs('log', exist_ok=True)
fh = logging.FileHandler(f'log/extend_pretrain_{time.time():.0f}.log')
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
# Load single dataset (same as pretrain_foundation.py)
# ---------------------------------------------------------------------------

def load_single_dataset(name):
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

    val_time, test_time = list(np.quantile(ts_l, [0.70, 0.85]))

    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    full_ngh_finder = NeighborFinder(adj_list, uniform=False)

    train_flag = ts_l <= val_time
    train_src = src_l[train_flag]
    train_dst = dst_l[train_flag]
    train_ts = ts_l[train_flag]

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
    }

    return config, data


# ---------------------------------------------------------------------------
# Load checkpoint and extend with new datasets
# ---------------------------------------------------------------------------

logger.info(f'Loading checkpoint: {args.checkpoint}')
checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
old_dataset_names = [cfg['name'] for cfg in checkpoint['dataset_configs']]
logger.info(f'Existing datasets: {old_dataset_names}')

# Load data for existing datasets
logger.info('Loading data for existing datasets...')
dataset_configs = list(checkpoint['dataset_configs'])
dataset_data = {}
all_dataset_names = list(old_dataset_names)

for ds_name in old_dataset_names:
    try:
        cfg, data = load_single_dataset(ds_name)
        dataset_data[ds_name] = data
        logger.info(f'  {ds_name}: {cfg["n_nodes"]} nodes, {len(data["train_src"])} train edges')
    except FileNotFoundError as e:
        logger.warning(f'  {ds_name}: SKIPPED — {e}')

# Load data for new datasets and add their configs
logger.info('Loading new datasets...')
for ds_name in args.new_datasets:
    try:
        cfg, data = load_single_dataset(ds_name)
        dataset_configs.append(cfg)
        dataset_data[ds_name] = data
        all_dataset_names.append(ds_name)
        logger.info(f'  {ds_name}: {cfg["n_nodes"]} nodes, {len(data["train_src"])} train edges, '
                    f'node_dim={cfg["raw_node_dim"]}, edge_dim={cfg["raw_edge_dim"]}')
    except FileNotFoundError as e:
        logger.warning(f'  {ds_name}: SKIPPED — {e}')

# Build model with extended configs
logger.info('Building extended model...')
model = TemporalGraphFoundationModel(
    dataset_configs=dataset_configs,
    shared_dim=checkpoint['shared_dim'],
    time_dim=checkpoint['time_dim'],
    num_layers=checkpoint['num_layers'],
    n_head=checkpoint.get('n_head', 4),
    drop_out=checkpoint.get('drop_out', 0.1),
    num_neighbors=checkpoint['num_neighbors'],
)
model = model.to(device)

# Copy weights from checkpoint (shared encoder + old adapters)
model_state = model.state_dict()
loaded_count = 0
skipped_count = 0
for key, value in checkpoint['model_state_dict'].items():
    if key in model_state and model_state[key].shape == value.shape:
        model_state[key].copy_(value)
        loaded_count += 1
    else:
        skipped_count += 1
model.load_state_dict(model_state)
logger.info(f'Loaded {loaded_count} tensors, skipped {skipped_count} (new adapters)')

# Load all dataset embeddings
for ds_name in all_dataset_names:
    if ds_name in dataset_data:
        model.load_dataset_embeddings(ds_name, dataset_data[ds_name]['n_feat'],
                                      dataset_data[ds_name]['e_feat'])

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f'Model params: {total_params:,} total, {trainable_params:,} trainable')

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

os.makedirs('./saved_models', exist_ok=True)
os.makedirs('./saved_checkpoints', exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}_pretrained.pth'


# ---------------------------------------------------------------------------
# Continued pre-training loop (same structure as pretrain_foundation.py)
# ---------------------------------------------------------------------------

logger.info(f'Starting continued pre-training on {len(all_dataset_names)} datasets for {args.epochs} epochs')
logger.info(f'Pretext tasks: MTLP + TNP + CDC + EFR + NPP')

best_val_ap = -1.0
early_stopper = EarlyStopMonitor(max_round=10, higher_better=True)

for epoch in range(args.epochs):
    model.train()
    epoch_losses = {'mtlp': 0.0, 'tnp': 0.0, 'cdc': 0.0, 'efr': 0.0, 'npp': 0.0, 'total': 0.0}
    total_batches = 0

    ds_order = list(all_dataset_names)
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
        num_batches = min(math.ceil(num_instances / args.bs), args.limit_batches)
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

            # Compute neighborhood stats once (shared between TNP and NPP)
            ngh_stats = model.compute_neighborhood_stats(
                ds_name, ngh_finder, src_l, ts_l
            )
            node_props = ngh_stats  # NPP reuses same stats

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

            (total_loss / args.grad_accum).backward()

            if (k + 1) % args.grad_accum == 0 or k == num_batches - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            epoch_losses['total'] += total_loss.item()
            total_batches += 1

    for key in epoch_losses:
        epoch_losses[key] /= max(total_batches, 1)

    scheduler.step()

    task_weights = torch.exp(-model.task_log_vars).detach().cpu().numpy()
    logger.info(
        f'Epoch {epoch:3d} | total={epoch_losses["total"]:.4f} '
        f'mtlp={epoch_losses["mtlp"]:.4f} tnp={epoch_losses["tnp"]:.4f} '
        f'cdc={epoch_losses["cdc"]:.4f} efr={epoch_losses["efr"]:.4f} '
        f'npp={epoch_losses["npp"]:.4f} '
        f'| w_mtlp={task_weights[0]:.3f} w_tnp={task_weights[1]:.3f} '
        f'w_cdc={task_weights[2]:.3f} w_efr={task_weights[3]:.3f} '
        f'w_npp={task_weights[4]:.3f} '
        f'| lr={scheduler.get_last_lr()[0]:.2e}'
    )

    # Validation
    if (epoch + 1) % args.eval_every == 0:
        model.eval()
        val_aps = []
        with torch.no_grad():
            for ds_name in all_dataset_names:
                if ds_name not in dataset_data:
                    continue
                data = dataset_data[ds_name]
                ngh_finder = data['ngh_finder']
                val_time = data.get('val_time')  # need val data
                # Quick val — reuse train edges as proxy
                train_src = data['train_src'][:500]
                train_dst = data['train_dst'][:500]
                train_ts = data['train_ts'][:500]
                rand_sampler = data['rand_sampler']

                pos_scores, neg_scores = [], []
                for j in range(0, len(train_src), 64):
                    s = train_src[j:j+64]
                    d = train_dst[j:j+64]
                    t = train_ts[j:j+64]
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
                val_aps.append(ap)
                logger.info(f'  Val {ds_name}: AP={ap:.4f} AUC={auc:.4f}')

        mean_val_ap = np.mean(val_aps) if val_aps else 0.0
        logger.info(f'  Mean val AP: {mean_val_ap:.4f}')

        if mean_val_ap > best_val_ap + 1e-4:
            best_val_ap = mean_val_ap
            model.save_pretrained(MODEL_SAVE_PATH)
            logger.info(f'  [BEST] Model saved to {MODEL_SAVE_PATH} (AP={best_val_ap:.4f})')

        ckpt_path = f'./saved_checkpoints/{args.prefix}_epoch{epoch}.pth'
        model.save_pretrained(ckpt_path)

        if early_stopper.early_stop_check(mean_val_ap):
            logger.info(f'Early stopping at epoch {epoch}')
            break

# Final save
model.save_pretrained(MODEL_SAVE_PATH)
logger.info(f'Extended pre-training complete. Final model saved to {MODEL_SAVE_PATH}')
logger.info(f'Best validation AP: {best_val_ap:.4f}')
logger.info('Next step: run finetune_foundation.py for downstream task transfer on all datasets.')
