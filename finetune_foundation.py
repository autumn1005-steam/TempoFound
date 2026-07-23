"""
Fine-tuning script for the pre-trained Temporal Graph Foundation Model.

Supports:
  - Zero-shot evaluation (no fine-tuning, just load and evaluate)
  - Few-shot fine-tuning (freeze encoder, train only task head)
  - Full fine-tuning (unfreeze encoder, train end-to-end)

Tasks:
  - link_prediction  — transductive & inductive link prediction
  - node_classification — dynamic node classification

Usage:
  # Full fine-tuning on a downstream dataset
  python finetune_foundation.py --task link_prediction --dataset wikipedia --mode full --epochs 20

  # Few-shot: freeze pre-trained encoder, only train task head
  python finetune_foundation.py --task node_classification --dataset mooc --mode few_shot --epochs 10

  # Zero-shot: evaluate pre-trained model without any fine-tuning
  python finetune_foundation.py --task link_prediction --dataset reddit --mode zero_shot
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
import torch.nn.functional as F

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

from graph import NeighborFinder
from utils import RandEdgeSampler
from foundation_model import TemporalGraphFoundationModel


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser('Temporal Graph Foundation Model — Fine-tuning')
parser.add_argument('--task', type=str, choices=['link_prediction', 'node_classification'],
                    default='link_prediction')
parser.add_argument('--dataset', type=str, default='wikipedia',
                    help='downstream dataset')
parser.add_argument('--mode', type=str, choices=['zero_shot', 'few_shot', 'full', 'head', 'fe'],
                    default='full', help='fine-tuning mode')
parser.add_argument('--pretrained_path', type=str,
                    default='./saved_models/foundation_pretrained.pth',
                    help='path to pre-trained foundation model')
parser.add_argument('--epochs', type=int, default=20, help='fine-tuning epochs')
parser.add_argument('--bs', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (full mode)')
parser.add_argument('--head_lr', type=float, default=0.001, help='learning rate for task head (few_shot mode)')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--prefix', type=str, default='finetune', help='save prefix')
parser.add_argument('--num_neighbors', type=int, default=20)
parser.add_argument('--few_shot_k', type=int, default=100,
                    help='number of training samples per class in few-shot mode')
parser.add_argument('--weighted_bce', action='store_true', default=True,
                    help='use class-weighted BCE for imbalanced node classification (default: True)')
parser.add_argument('--no_weighted_bce', action='store_true',
                    help='disable weighted BCE')
parser.add_argument('--lambda_contrastive', type=float, default=0.1,
                    help='weight for contrastive auxiliary loss during node classification fine-tuning')
parser.add_argument('--link_aux_weight', type=float, default=1.0,
                    help='weight for auxiliary link prediction loss during node clf fine-tuning (preserves LP capability)')
parser.add_argument('--tune', action='store_true',
                    help='use random val split within training period (avoids temporal distribution shift)')
parser.add_argument('--warmup_epochs', type=int, default=3,
                    help='learning rate warmup epochs (full mode only)')
parser.add_argument('--limit_batches', type=int, default=0,
                    help='max batches per epoch (0 = use all data)')
parser.add_argument('--node_label_path', type=str, default='',
                    help='path to .npy file with node classification labels '
                         '(overrides CSV label column; use for synthetic labels)')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed for training, sampling, and few-shot selection')
parser.add_argument('--results_csv', type=str, default='./results_finetune.csv',
                    help='CSV file used to append final test results')
parser.add_argument('--shared_dim', type=int, default=256,
                    help='hidden dimension used when --pretrained_path scratch')
parser.add_argument('--time_dim', type=int, default=128,
                    help='time encoding dimension used when --pretrained_path scratch')
parser.add_argument('--num_layers', type=int, default=2,
                    help='encoder layers used when --pretrained_path scratch')
parser.add_argument('--n_head', type=int, default=4,
                    help='attention heads used when --pretrained_path scratch')
parser.add_argument('--drop_out', type=float, default=0.1,
                    help='dropout used when --pretrained_path scratch')
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
fh = logging.FileHandler(f'log/finetune_{args.task}_{args.dataset}_{time.time():.0f}.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

logger.info(f'Task: {args.task} | Dataset: {args.dataset} | Mode: {args.mode}')

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

if args.gpu >= 0 and torch.cuda.is_available():
    device = torch.device(f'cuda:{args.gpu}')
else:
    device = torch.device('cpu')
logger.info(f'Device: {device}')

# ---------------------------------------------------------------------------
# Load downstream dataset
# ---------------------------------------------------------------------------

csv_path = f'./processed/ml_{args.dataset}.csv'
npy_path = f'./processed/ml_{args.dataset}.npy'
node_npy_path = f'./processed/ml_{args.dataset}_node.npy'

if not os.path.exists(csv_path):
    logger.error(f'Dataset not found: {csv_path}. Run process.py first.')
    sys.exit(1)

df = pd.read_csv(csv_path)
e_feat = np.load(npy_path)
n_feat = np.load(node_npy_path)

src_l = df.u.values
dst_l = df.i.values
e_idx_l = df.idx.values
ts_l = df.ts.values
label_l = df.label.values

# Override with external node labels if provided (for synthetic labels)
if args.node_label_path:
    if os.path.exists(args.node_label_path):
        ext_labels = np.load(args.node_label_path)
        if len(ext_labels) == len(label_l):
            label_l = ext_labels.astype(np.int32)
            logger.info(f'Loaded node labels from {args.node_label_path} '
                        f'(pos={label_l.sum()}, neg={len(label_l) - label_l.sum()})')
        else:
            logger.error(f'Label file length mismatch: {len(ext_labels)} vs {len(label_l)}')
    else:
        logger.error(f'Node label file not found: {args.node_label_path}')

max_idx = max(src_l.max(), dst_l.max())
val_time, test_time = list(np.quantile(ts_l, [0.70, 0.85]))

# Build graphs
full_adj_list = [[] for _ in range(max_idx + 1)]
train_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
    if ts <= val_time:
        train_adj_list[src].append((dst, eidx, ts))
        train_adj_list[dst].append((src, eidx, ts))

full_ngh_finder = NeighborFinder(full_adj_list, uniform=False)

# Train / val / test split
if args.tune:
    # Random split within training period — avoids temporal distribution shift
    pretest_flag = ts_l <= test_time
    pretest_idx = np.where(pretest_flag)[0]
    np.random.seed(args.seed)
    np.random.shuffle(pretest_idx)
    n_val = int(len(pretest_idx) * 0.1)
    val_idx = pretest_idx[:n_val]
    train_idx = pretest_idx[n_val:]
    test_idx = np.where(ts_l > test_time)[0]

    train_src = src_l[train_idx]
    train_dst = dst_l[train_idx]
    train_ts = ts_l[train_idx]
    train_labels = label_l[train_idx]
    train_eidx = e_idx_l[train_idx]

    val_src = src_l[val_idx]
    val_dst = dst_l[val_idx]
    val_ts = ts_l[val_idx]
    val_labels = label_l[val_idx]

    test_src = src_l[test_idx]
    test_dst = dst_l[test_idx]
    test_ts = ts_l[test_idx]
    test_labels = label_l[test_idx]
    logger.info('TUNE mode: random val split within training period')
else:
    # Chronological split
    train_flag = ts_l <= val_time
    val_flag = (ts_l > val_time) & (ts_l <= test_time)
    test_flag = ts_l > test_time

    train_src = src_l[train_flag]
    train_dst = dst_l[train_flag]
    train_ts = ts_l[train_flag]
    train_labels = label_l[train_flag]
    train_eidx = e_idx_l[train_flag]

    val_src = src_l[val_flag]
    val_dst = dst_l[val_flag]
    val_ts = ts_l[val_flag]
    val_labels = label_l[val_flag]

    test_src = src_l[test_flag]
    test_dst = dst_l[test_flag]
    test_ts = ts_l[test_flag]
    test_labels = label_l[test_flag]

rand_sampler = RandEdgeSampler(train_src, train_dst)

train_nodes = set(np.unique(train_src))

logger.info(f'Dataset: {len(train_src)} train, {len(val_src)} val, {len(test_src)} test edges')
logger.info(f'Train nodes: {len(train_nodes)}')

# ---------------------------------------------------------------------------
# Load foundation model
# ---------------------------------------------------------------------------

if not os.path.exists(args.pretrained_path):
    logger.error(f'Pre-trained model not found: {args.pretrained_path}')
    logger.error(f'Run pretrain_foundation.py first.')
    sys.exit(1)

new_dataset_config = {
    'name': args.dataset,
    'raw_node_dim': n_feat.shape[1],
    'raw_edge_dim': e_feat.shape[1],
    'n_nodes': max_idx,
    'n_edges': len(e_feat) - 1,
}

logger.info(f'Loading pre-trained foundation model from {args.pretrained_path}...')

# Check if the dataset is already in the checkpoint (e.g., v4 with 5 datasets)
checkpoint = torch.load(args.pretrained_path, map_location=device, weights_only=False)
existing_names = [cfg['name'] for cfg in checkpoint['dataset_configs']]

if args.dataset in existing_names:
    logger.info('  Dataset already in checkpoint — loading directly.')
    model = TemporalGraphFoundationModel.load_pretrained(args.pretrained_path, map_location=device)
else:
    logger.info('  Dataset NOT in checkpoint — extending with new adapter.')
    try:
        model = TemporalGraphFoundationModel.load_for_new_dataset(
            args.pretrained_path, new_dataset_config, map_location=device
        )
        logger.info('  Extended model with new dataset adapter.')
    except Exception:
        logger.warning('  load_for_new_dataset failed — building model from checkpoint metadata.')
        model = TemporalGraphFoundationModel(
            dataset_configs=checkpoint['dataset_configs'] + [new_dataset_config],
            shared_dim=checkpoint['shared_dim'],
            time_dim=checkpoint['time_dim'],
            num_layers=checkpoint['num_layers'],
            num_neighbors=checkpoint['num_neighbors'],
        )
        state = model.state_dict()
        for k, v in checkpoint['model_state_dict'].items():
            if k in state and state[k].shape == v.shape:
                state[k].copy_(v)
        model.load_state_dict(state)

model = model.to(device)
model.load_dataset_embeddings(args.dataset, n_feat, e_feat)

# Free memory: delete unused dataset embeddings (only keep the current dataset)
for ds_name in list(model.dataset_node_embed.keys()):
    if ds_name != args.dataset:
        del model.dataset_node_embed[ds_name]
        del model.dataset_edge_embed[ds_name]
logger.info('  Freed unused dataset embeddings, keeping only %s.', args.dataset)

# ---------------------------------------------------------------------------
# Fine-tuning mode setup
# ---------------------------------------------------------------------------

if args.mode == 'zero_shot':
    logger.info('ZERO-SHOT mode: evaluating pre-trained model without fine-tuning.')
    model.eval()

elif args.mode == 'few_shot':
    logger.info(f'FEW-SHOT mode: freezing encoder, training task head with {args.few_shot_k} samples.')
    model.freeze_encoder()

    # Select few-shot samples (balanced across labels if node classification)
    num_train = len(train_src)
    if args.task == 'node_classification':
        pos_idx = np.where(train_labels == 1)[0]
        neg_idx = np.where(train_labels == 0)[0]
        k_pos = min(args.few_shot_k, len(pos_idx))
        k_neg = min(args.few_shot_k, len(neg_idx))
        few_idx = np.concatenate([
            np.random.choice(pos_idx, k_pos, replace=False),
            np.random.choice(neg_idx, k_neg, replace=False),
        ])
    else:
        k_total = min(args.few_shot_k * 2, num_train)
        few_idx = np.random.choice(num_train, k_total, replace=False)

    train_src = train_src[few_idx]
    train_dst = train_dst[few_idx]
    train_ts = train_ts[few_idx]
    train_labels = train_labels[few_idx] if args.task == 'node_classification' else None
    logger.info(f'  Training on {len(train_src)} few-shot samples.')

elif args.mode == 'head':
    logger.info('HEAD mode: freezing encoder, training task head on all data.')
    model.freeze_encoder()

elif args.mode == 'fe':
    logger.info(
        'FE mode: freezing the shared encoder and source adapters; '
        'training the target adapter and task head on all target training data.'
    )
    model.freeze_encoder()
    for parameter in model.adapters[args.dataset].parameters():
        parameter.requires_grad = True

elif args.mode == 'full':
    logger.info('FULL mode: fine-tuning encoder + task head end-to-end.')
    model.unfreeze_encoder()

# ---------------------------------------------------------------------------
# Task-specific heads
# ---------------------------------------------------------------------------

if args.task == 'link_prediction':
    model.add_finetune_link_head()
    model.ft_link_head = model.ft_link_head.to(device)
elif args.task == 'node_classification':
    model.add_finetune_node_clf_head()
    model.ft_node_clf_head = model.ft_node_clf_head.to(device)

# Optimizer
if args.mode in ('few_shot', 'head'):
    # Only optimize task head
    if args.task == 'link_prediction':
        opt_params = model.ft_link_head.parameters()
    else:
        opt_params = model.ft_node_clf_head.parameters()
    lr = args.head_lr
elif args.mode == 'fe':
    if args.task == 'link_prediction':
        head_params = list(model.ft_link_head.parameters())
    else:
        head_params = list(model.ft_node_clf_head.parameters())
    adapter_params = list(model.adapters[args.dataset].parameters())
    opt_params = adapter_params + head_params
    lr = args.head_lr
else:
    opt_params = model.parameters()
    lr = args.lr

optimizer = torch.optim.AdamW(opt_params, lr=lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

if args.task == 'node_classification':
    use_weighted_bce = args.weighted_bce and not args.no_weighted_bce
    if use_weighted_bce:
        pos_count = max(float(train_labels.sum()), 1.0)
        neg_count = max(float(len(train_labels) - train_labels.sum()), 1.0)
        pos_weight = torch.tensor([neg_count / pos_count], device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info(f'Weighted BCE with pos_weight={pos_weight.item():.2f}')
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
else:
    criterion = torch.nn.BCEWithLogitsLoss()

# ---------------------------------------------------------------------------
# Zero-shot evaluation (no training)
# ---------------------------------------------------------------------------

def evaluate_link_prediction(ngh_finder, src, dst, ts, label=None):
    """Evaluate link prediction AP/AUC."""
    model.eval()
    pos_scores, neg_scores = [], []
    with torch.no_grad():
        for j in range(0, len(src), 64):
            s = src[j:j+64]
            d = dst[j:j+64]
            t = ts[j:j+64]
            ns, nd = rand_sampler.sample(len(s))
            logits = model.finetune_link_forward(args.dataset, ngh_finder, s, d, t)
            neg_logits = model.finetune_link_forward(args.dataset, ngh_finder, ns, nd, t)
            pos_scores.append(torch.sigmoid(logits).cpu().numpy())
            neg_scores.append(torch.sigmoid(neg_logits).cpu().numpy())

    pos = np.concatenate(pos_scores)
    neg = np.concatenate(neg_scores)
    preds = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    ap = average_precision_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, preds > 0.5)
    return ap, auc, acc


def _safe_metric(metric_fn, y_true, y_score, **kwargs):
    """Call a sklearn metric safely, returning NaN when only one class is present."""
    try:
        return metric_fn(y_true, y_score, **kwargs)
    except ValueError:
        return float('nan')


def evaluate_node_classification(ngh_finder, src, dst, ts, labels):
    """Evaluate node classification AUC/AP with old/new node split."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for j in range(0, len(src), args.bs):
            s = src[j:j+args.bs]
            d = dst[j:j+args.bs]
            t = ts[j:j+args.bs]
            logits = model.finetune_node_clf_forward(args.dataset, ngh_finder, s, d, t)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(all_probs)

    auc = _safe_metric(roc_auc_score, labels, probs)
    ap = _safe_metric(average_precision_score, labels, probs)
    acc = _safe_metric(accuracy_score, labels, probs > 0.5)

    def _split_metric(metric_fn, mask, fallback=float('nan')):
        if mask.sum() == 0:
            return fallback
        return _safe_metric(metric_fn, labels[mask], probs[mask])

    new_node_mask = np.array([n not in train_nodes for n in src])
    old_node_mask = np.array([n in train_nodes for n in src])

    new_node_auc = _split_metric(roc_auc_score, new_node_mask)
    new_node_ap = _split_metric(average_precision_score, new_node_mask, fallback=0.0)
    old_node_auc = _split_metric(roc_auc_score, old_node_mask)
    old_node_ap = _split_metric(average_precision_score, old_node_mask, fallback=0.0)

    return auc, ap, acc, new_node_auc, new_node_ap, old_node_auc, old_node_ap


# ---------------------------------------------------------------------------
# CSV results logging (for paper-ready tables)
# ---------------------------------------------------------------------------
RESULTS_CSV = args.results_csv
CSV_COLUMNS = ['timestamp', 'seed', 'task', 'dataset', 'mode', 'few_shot_k',
               'pretrained', 'epochs',
               'test_AUC', 'test_AP', 'test_Acc',
               'NewNode_AUC', 'NewNode_AP', 'OldNode_AUC', 'OldNode_AP']

def append_csv_row(row_dict):
    """Append a row to the results CSV, creating it with headers if needed."""
    import csv
    results_dir = os.path.dirname(RESULTS_CSV)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    file_exists = os.path.exists(RESULTS_CSV)
    # Fill missing columns with empty string
    full_row = {col: row_dict.get(col, '') for col in CSV_COLUMNS}
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(full_row)
    logger.info(f'Results appended to {RESULTS_CSV}')


if args.mode == 'zero_shot':
    logger.info('=== Zero-Shot Evaluation ===')
    if args.task == 'link_prediction':
        val_ap, val_auc, val_acc = evaluate_link_prediction(
            full_ngh_finder, val_src, val_dst, val_ts)
        test_ap, test_auc, test_acc = evaluate_link_prediction(
            full_ngh_finder, test_src, test_dst, test_ts)
        logger.info(f'Val  — AP={val_ap:.4f} AUC={val_auc:.4f} Acc={val_acc:.4f}')
        logger.info(f'Test — AP={test_ap:.4f} AUC={test_auc:.4f} Acc={test_acc:.4f}')
        csv_row = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'seed': args.seed,
            'task': args.task, 'dataset': args.dataset, 'mode': 'zero_shot',
            'pretrained': os.path.basename(args.pretrained_path), 'epochs': 0,
            'test_AP': f'{test_ap:.4f}', 'test_AUC': f'{test_auc:.4f}',
            'test_Acc': f'{test_acc:.4f}',
        }
    else:
        val_auc, val_ap, val_acc, val_new_auc, val_new_ap, val_old_auc, val_old_ap = evaluate_node_classification(
            full_ngh_finder, val_src, val_dst, val_ts, val_labels)
        test_auc, test_ap, test_acc, test_new_auc, test_new_ap, test_old_auc, test_old_ap = evaluate_node_classification(
            full_ngh_finder, test_src, test_dst, test_ts, test_labels)
        logger.info(f'Val  — AUC={val_auc:.4f} AP={val_ap:.4f} Acc={val_acc:.4f} '
                    f'NewNodeAUC={val_new_auc:.4f} OldNodeAUC={val_old_auc:.4f}')
        logger.info(f'Test — AUC={test_auc:.4f} AP={test_ap:.4f} Acc={test_acc:.4f} '
                    f'NewNodeAUC={test_new_auc:.4f} OldNodeAUC={test_old_auc:.4f}')
        csv_row = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'seed': args.seed,
            'task': args.task, 'dataset': args.dataset, 'mode': 'zero_shot',
            'pretrained': os.path.basename(args.pretrained_path), 'epochs': 0,
            'test_AUC': f'{test_auc:.4f}', 'test_AP': f'{test_ap:.4f}',
            'test_Acc': f'{test_acc:.4f}',
            'NewNode_AUC': f'{test_new_auc:.4f}', 'NewNode_AP': f'{test_new_ap:.4f}',
            'OldNode_AUC': f'{test_old_auc:.4f}', 'OldNode_AP': f'{test_old_ap:.4f}',
        }
    append_csv_row(csv_row)
    sys.exit(0)

# ---------------------------------------------------------------------------
# Fine-tuning training loop
# ---------------------------------------------------------------------------

num_instances = len(train_src)
num_batches = math.ceil(num_instances / args.bs)
if args.limit_batches > 0:
    num_batches = min(num_batches, args.limit_batches)
idx_list = np.arange(num_instances)

logger.info(f'Fine-tuning: {num_instances} samples, {num_batches} batches/epoch, {args.epochs} epochs')
if args.task == 'node_classification':
    logger.info(f'Link auxiliary loss weight: {args.link_aux_weight}')
    logger.info(f'Warmup epochs: {args.warmup_epochs}')

best_val_metric = -1.0
best_epoch = -1
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}_{args.task}_{args.dataset}.pth'

for epoch in range(args.epochs):
    model.train()
    np.random.shuffle(idx_list)
    epoch_loss = 0.0
    epoch_task_loss = 0.0
    epoch_link_aux_loss = 0.0

    # Learning rate warmup (full mode only)
    if args.mode == 'full' and epoch < args.warmup_epochs:
        warmup_factor = (epoch + 1) / max(args.warmup_epochs, 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * warmup_factor

    for k in range(num_batches):
        s_idx = k * args.bs
        e_idx = min(num_instances, s_idx + args.bs)
        batch_idx = idx_list[s_idx:e_idx]
        size = len(batch_idx)
        if size == 0:
            continue

        src_batch = train_src[batch_idx]
        dst_batch = train_dst[batch_idx]
        ts_batch = train_ts[batch_idx]

        optimizer.zero_grad()

        if args.task == 'link_prediction':
            neg_src, neg_dst = rand_sampler.sample(size)

            pos_logit = model.finetune_link_forward(
                args.dataset, full_ngh_finder, src_batch, dst_batch, ts_batch)
            neg_logit = model.finetune_link_forward(
                args.dataset, full_ngh_finder, neg_src, neg_dst, ts_batch)

            pos_label = torch.ones(size, device=device)
            neg_label = torch.zeros(size, device=device)
            loss = criterion(pos_logit, pos_label) + criterion(neg_logit, neg_label)

        else:  # node_classification — with auxiliary link prediction to preserve encoder quality
            src_emb = model.encode_nodes(args.dataset, full_ngh_finder, src_batch, ts_batch)
            dst_emb = model.encode_nodes(args.dataset, full_ngh_finder, dst_batch, ts_batch)
            node_stats = model.compute_node_stats_tensor(
                args.dataset, full_ngh_finder, src_batch, ts_batch)
            combined = torch.cat([src_emb, dst_emb, src_emb * dst_emb, node_stats], dim=-1)
            logits = model.ft_node_clf_head(combined).squeeze(-1)
            labels = torch.from_numpy(train_labels[batch_idx]).float().to(device)
            task_loss = criterion(logits, labels)

            # Auxiliary link prediction loss: preserves pre-trained link prediction
            # capability while encoder adapts to node classification.
            if args.link_aux_weight > 0:
                neg_src, neg_dst = rand_sampler.sample(size)
                neg_emb = model.encode_nodes(args.dataset, full_ngh_finder, neg_dst, ts_batch)
                pos_logit = model.link_head(src_emb, dst_emb)
                neg_logit = model.link_head(src_emb, neg_emb)
                pos_label = torch.ones(size, device=device)
                neg_label = torch.zeros(size, device=device)
                link_loss = F.binary_cross_entropy_with_logits(pos_logit, pos_label) + \
                            F.binary_cross_entropy_with_logits(neg_logit, neg_label)
                loss = task_loss + args.link_aux_weight * link_loss
                epoch_link_aux_loss += args.link_aux_weight * link_loss.item()
            else:
                loss = task_loss

            epoch_task_loss += task_loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

        # Batch-level progress (suppress buffering anxiety on CPU)
        if k % 20 == 0 or k == num_batches - 1:
            logger.info(f'  Batch {k+1}/{num_batches} | loss={loss.item():.4f}')

    epoch_loss /= max(num_batches, 1)
    if args.task == 'node_classification':
        epoch_task_loss /= max(num_batches, 1)
        epoch_link_aux_loss /= max(num_batches, 1)
    scheduler.step()

    # Validation
    model.eval()
    with torch.no_grad():
        if args.task == 'link_prediction':
            val_ap, val_auc, val_acc = evaluate_link_prediction(
                full_ngh_finder, val_src, val_dst, val_ts)
            val_metric = val_ap
            logger.info(f'Epoch {epoch:3d} | loss={epoch_loss:.4f} | '
                        f'val_AP={val_ap:.4f} val_AUC={val_auc:.4f} val_Acc={val_acc:.4f} '
                        f'| lr={scheduler.get_last_lr()[0]:.2e}')
            metric_name = 'AP'
        else:
            val_auc, val_ap, val_acc, val_new_auc, val_new_ap, val_old_auc, val_old_ap = evaluate_node_classification(
                full_ngh_finder, val_src, val_dst, val_ts, val_labels)
            # Use new_node_auc for model selection (better indicator of transfer quality)
            val_metric = val_new_auc if not np.isnan(val_new_auc) else val_auc
            logger.info(f'Epoch {epoch:3d} | loss={epoch_loss:.4f} task_loss={epoch_task_loss:.4f} '
                        f'link_aux={epoch_link_aux_loss:.4f} | '
                        f'val_AUC={val_auc:.4f} val_AP={val_ap:.4f} val_Acc={val_acc:.4f} '
                        f'NewAUC={val_new_auc:.4f} OldAUC={val_old_auc:.4f} '
                        f'| lr={scheduler.get_last_lr()[0]:.2e}')
            metric_name = 'NewNodeAUC'

    if val_metric > best_val_metric + 1e-5:
        best_val_metric = val_metric
        best_epoch = epoch
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        logger.info(f'  [BEST] val_{metric_name}={best_val_metric:.4f}')

logger.info(f'Best epoch: {best_epoch} (val_{metric_name}={best_val_metric:.4f})')

# Fallback: if no model saved (e.g. all val metrics NaN), save last epoch
if best_epoch == -1:
    logger.info('No best model selected (all val metrics NaN). Saving last epoch as fallback.')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    best_epoch = args.epochs - 1

# ---------------------------------------------------------------------------
# Final test evaluation
# ---------------------------------------------------------------------------

logger.info('Loading best model for final evaluation...')
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False))
model.eval()

with torch.no_grad():
    if args.task == 'link_prediction':
        test_ap, test_auc, test_acc = evaluate_link_prediction(
            full_ngh_finder, test_src, test_dst, test_ts)
        logger.info(f'=== Final Test Results ===')
        logger.info(f'Link Prediction — AP={test_ap:.4f} AUC={test_auc:.4f} Acc={test_acc:.4f}')
    else:
        test_auc, test_ap, test_acc, test_new_auc, test_new_ap, test_old_auc, test_old_ap = evaluate_node_classification(
            full_ngh_finder, test_src, test_dst, test_ts, test_labels)
        logger.info(f'=== Final Test Results ===')
        logger.info(f'Node Classification — AUC={test_auc:.4f} AP={test_ap:.4f} Acc={test_acc:.4f}')
        logger.info(f'  Old nodes  — AUC={test_old_auc:.4f} AP={test_old_ap:.4f}')
        logger.info(f'  New nodes  — AUC={test_new_auc:.4f} AP={test_new_ap:.4f}')

logger.info(f'Fine-tuned model saved to {MODEL_SAVE_PATH}.')

csv_row = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'seed': args.seed,
    'task': args.task,
    'dataset': args.dataset,
    'mode': args.mode,
    'few_shot_k': args.few_shot_k if args.mode == 'few_shot' else '',
    'pretrained': os.path.basename(args.pretrained_path),
    'epochs': best_epoch + 1,
}
if args.task == 'link_prediction':
    csv_row.update({
        'test_AP': f'{test_ap:.4f}', 'test_AUC': f'{test_auc:.4f}',
        'test_Acc': f'{test_acc:.4f}',
    })
else:
    csv_row.update({
        'test_AUC': f'{test_auc:.4f}', 'test_AP': f'{test_ap:.4f}',
        'test_Acc': f'{test_acc:.4f}',
        'NewNode_AUC': f'{test_new_auc:.4f}', 'NewNode_AP': f'{test_new_ap:.4f}',
        'OldNode_AUC': f'{test_old_auc:.4f}', 'OldNode_AP': f'{test_old_ap:.4f}',
    })
append_csv_row(csv_row)
