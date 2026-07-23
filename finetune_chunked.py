"""
Chunked fine-tuning: run N epochs, save, exit. Designed to work around
CPU training hangs by running in fresh processes.
Usage: python finetune_chunked.py --task link_prediction --dataset uci --epochs 5 --pretrained_path ... --resume_from ...
"""
import math, time, random, argparse, logging, sys, os, gc
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from graph import NeighborFinder
from utils import RandEdgeSampler
from foundation_model import TemporalGraphFoundationModel

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='link_prediction',
                    choices=['link_prediction', 'node_classification'])
parser.add_argument('--dataset', type=str, default='uci')
parser.add_argument('--mode', type=str, default='full')
parser.add_argument('--pretrained_path', type=str, required=True)
parser.add_argument('--resume_from', type=str, default='',
                    help='path to previously saved fine-tuned model')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--bs', type=int, default=512)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--node_label_path', type=str, default='')
parser.add_argument('--prefix', type=str, default='finetune')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

MODEL_SAVE_PATH = f'./saved_models/{args.prefix}_{args.task}_{args.dataset}.pth'
device = torch.device('cpu')

# ---- Load data ----
g_df = pd.read_csv(f'./processed/ml_{args.dataset}.csv')
e_feat = np.load(f'./processed/ml_{args.dataset}.npy')
n_feat = np.load(f'./processed/ml_{args.dataset}_node.npy')

val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
src_l = g_df.u.values; dst_l = g_df.i.values
e_idx_l = g_df.idx.values; ts_l = g_df.ts.values
label_l = g_df.label.values if 'label' in g_df.columns else np.zeros_like(src_l)

if args.node_label_path and os.path.exists(args.node_label_path):
    ext_labels = np.load(args.node_label_path)
    if len(ext_labels) == len(label_l):
        label_l = ext_labels.astype(np.int32)

max_idx = max(src_l.max(), dst_l.max())

# Build adjacency
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))

full_ngh_finder = NeighborFinder(full_adj_list, uniform=False)

# Split
train_mask = ts_l <= val_time
val_mask = (ts_l > val_time) & (ts_l <= test_time)
test_mask = ts_l > test_time

train_src = src_l[train_mask]; train_dst = dst_l[train_mask]; train_ts = ts_l[train_mask]
val_src = src_l[val_mask]; val_dst = dst_l[val_mask]; val_ts = ts_l[val_mask]
test_src = src_l[test_mask]; test_dst = dst_l[test_mask]; test_ts = ts_l[test_mask]

train_labels = label_l[train_mask] if args.task == 'node_classification' else None
val_labels = label_l[val_mask] if args.task == 'node_classification' else None
test_labels = label_l[test_mask] if args.task == 'node_classification' else None

num_train = len(train_src)
idx_list = np.arange(num_train)
num_batches = int(np.ceil(num_train / args.bs))

rand_sampler = RandEdgeSampler(train_src, train_dst)
logger.info(f'Dataset: {num_train} train, {len(val_src)} val, {len(test_src)} test')
logger.info(f'Train batches: {num_batches}, epochs: {args.epochs}')

# ---- Build model ----
new_dataset_config = {
    'name': args.dataset,
    'raw_node_dim': n_feat.shape[1],
    'raw_edge_dim': e_feat.shape[1],
    'n_nodes': max_idx,
    'n_edges': len(e_feat) - 1,
}

checkpoint = torch.load(args.pretrained_path, map_location=device, weights_only=False)
existing_names = [cfg['name'] for cfg in checkpoint['dataset_configs']]

if args.dataset in existing_names:
    model = TemporalGraphFoundationModel.load_pretrained(args.pretrained_path, map_location=device)
else:
    model = TemporalGraphFoundationModel.load_for_new_dataset(
        args.pretrained_path, new_dataset_config, map_location=device)

model = model.to(device)
model.load_dataset_embeddings(args.dataset, n_feat, e_feat)

for ds_name in list(model.dataset_node_embed.keys()):
    if ds_name != args.dataset:
        del model.dataset_node_embed[ds_name]
        del model.dataset_edge_embed[ds_name]

# Add task head
if args.task == 'link_prediction':
    model.add_finetune_link_head()
else:
    model.add_finetune_node_clf_head()

# Resume if specified
if args.resume_from and os.path.exists(args.resume_from):
    ft_state = torch.load(args.resume_from, map_location=device, weights_only=False)
    model_state = model.state_dict()
    for k, v in ft_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k].copy_(v)
    model.load_state_dict(model_state)
    logger.info(f'Resumed from {args.resume_from}')

model.unfreeze_encoder()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

# ---- Eval helpers ----
def evaluate_link_prediction(ngh_finder, src, dst, ts):
    model.eval()
    with torch.no_grad():
        pos_logits = []
        bs = args.bs
        for k in range(0, len(src), bs):
            s = src[k:k+bs]; d = dst[k:k+bs]; t = ts[k:k+bs]
            pos_logits.append(model.finetune_link_forward(args.dataset, ngh_finder, s, d, t).cpu())
        pos_logits = torch.cat(pos_logits)

        neg_src, neg_dst = rand_sampler.sample(len(src))
        neg_logits = []
        for k in range(0, len(neg_src), bs):
            s = neg_src[k:k+bs]; d = neg_dst[k:k+bs]
            neg_logits.append(model.finetune_link_forward(args.dataset, ngh_finder, s, d, ts[k:k+bs]).cpu())
        neg_logits = torch.cat(neg_logits)

        y_true = np.concatenate([np.ones(len(src)), np.zeros(len(src))])
        y_score = torch.sigmoid(torch.cat([pos_logits, neg_logits])).numpy()
    return (average_precision_score(y_true, y_score),
            roc_auc_score(y_true, y_score),
            accuracy_score(y_true, y_score > 0.5))

# ---- Training ----
best_val_metric = -1.0
best_epoch = -1

for epoch in range(args.epochs):
    model.train()
    np.random.shuffle(idx_list)
    epoch_loss = 0.0

    for k in range(num_batches):
        s_idx = k * args.bs
        e_idx = min(num_train, s_idx + args.bs)
        batch_idx = idx_list[s_idx:e_idx]
        size = len(batch_idx)
        if size == 0:
            continue

        src_batch = train_src[batch_idx]
        dst_batch = train_dst[batch_idx]
        ts_batch = train_ts[batch_idx]

        optimizer.zero_grad()

        neg_src, neg_dst = rand_sampler.sample(size)
        pos_logit = model.finetune_link_forward(args.dataset, full_ngh_finder, src_batch, dst_batch, ts_batch)
        neg_logit = model.finetune_link_forward(args.dataset, full_ngh_finder, neg_src, neg_dst, ts_batch)

        pos_label = torch.ones(size, device=device)
        neg_label = torch.zeros(size, device=device)
        loss = F.binary_cross_entropy_with_logits(pos_logit, pos_label) + \
               F.binary_cross_entropy_with_logits(neg_logit, neg_label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

        if k % 20 == 0 or k == num_batches - 1:
            logger.info(f'  Batch {k+1}/{num_batches} | loss={loss.item():.4f}')

    epoch_loss /= max(num_batches, 1)
    gc.collect()

    # Validation
    model.eval()
    with torch.no_grad():
        val_ap, val_auc, val_acc = evaluate_link_prediction(
            full_ngh_finder, val_src, val_dst, val_ts)
        logger.info(f'Epoch {epoch:3d} | loss={epoch_loss:.4f} | '
                    f'val_AP={val_ap:.4f} val_AUC={val_auc:.4f} val_Acc={val_acc:.4f}')

    if val_ap > best_val_metric + 1e-5:
        best_val_metric = val_ap
        best_epoch = epoch
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        logger.info(f'  [BEST] val_AP={best_val_metric:.4f}')

logger.info(f'Best epoch: {best_epoch} (val_AP={best_val_metric:.4f})')

# Final test evaluation on best model
if best_epoch >= 0:
    ft_state = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
    model_state = model.state_dict()
    for k, v in ft_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k].copy_(v)
    model.load_state_dict(model_state)

model.eval()
with torch.no_grad():
    pos_logits = []
    bs = args.bs
    for k in range(0, len(test_src), bs):
        s = test_src[k:k+bs]; d = test_dst[k:k+bs]; t = test_ts[k:k+bs]
        pos_logits.append(model.finetune_link_forward(args.dataset, full_ngh_finder, s, d, t).cpu())
    pos_logits = torch.cat(pos_logits)

    neg_src, neg_dst = rand_sampler.sample(len(test_src))
    neg_logits = []
    for k in range(0, len(neg_src), bs):
        s = neg_src[k:k+bs]; d = neg_dst[k:k+bs]
        neg_logits.append(model.finetune_link_forward(args.dataset, full_ngh_finder, s, d, test_ts[k:k+bs]).cpu())
    neg_logits = torch.cat(neg_logits)

    y_true = np.concatenate([np.ones(len(test_src)), np.zeros(len(test_src))])
    y_score = torch.sigmoid(torch.cat([pos_logits, neg_logits])).numpy()

test_ap = average_precision_score(y_true, y_score)
test_auc = roc_auc_score(y_true, y_score)
test_acc = accuracy_score(y_true, y_score > 0.5)

logger.info(f'=== Final Test Results ===')
logger.info(f'Link Prediction — AP={test_ap:.4f} AUC={test_auc:.4f} Acc={test_acc:.4f}')
logger.info(f'Model saved to {MODEL_SAVE_PATH}')
