"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import sys
import random
import argparse
import os

from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from module import TGAN
from graph import NeighborFinder
from model.thycrod import T_HyCroD
from thycrod_config import VARIANT_CHOICES, apply_thycrod_variant


class LR(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


def snapshot_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def load_compatible_state_dict(module, state_dict):
    current = module.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current and current[key].shape == value.shape
    }
    skipped = len(state_dict) - len(compatible)
    current.update(compatible)
    module.load_state_dict(current)
    return len(compatible), skipped


random.seed(222)
np.random.seed(222)
torch.manual_seed(222)

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on node classification')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--model_name', '--model', dest='model_name', type=str, choices=['tgat', 'thycrod'], default='tgat', help='temporal encoder to use')
parser.add_argument('--variant', type=str, choices=VARIANT_CHOICES, default='full', help='T-HyCroD ablation variant')
parser.add_argument('--bs', type=int, default=30, help='batch_size')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--n_degree', type=int, default=50, help='number of neighbors to sample')
parser.add_argument('--num_neighbors', type=int, default=None, help='alias for --n_degree')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_epoch', type=int, default=15, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--num_layers', type=int, default=None, help='alias for --n_layer')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--tune', action='store_true', help='parameters tunning mode, use train-test split on training data only.')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=None, help='Dimentions of the node embedding')
parser.add_argument('--hidden_dim', type=int, default=None, help='T-HyCroD hidden dimension')
parser.add_argument('--time_dim', type=int, default=None, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
parser.add_argument('--train_encoder', action='store_true', help='train the temporal encoder jointly with the node classifier')
parser.add_argument('--weighted_bce', action='store_true', help='use positive-class weighted BCE for imbalanced dynamic node labels')
parser.add_argument('--chronological_node_split', action='store_true', help='use strict chronological train/val/test split for node classification')
parser.add_argument('--node_use_dst', action='store_true', help='use concatenated src/dst temporal embeddings for T-HyCroD node classification')
parser.add_argument('--no_node_stats', action='store_true', help='disable T-HyCroD event-history statistics in node classification')
parser.add_argument('--node_edge_feat', action='store_true', help='enable current edge features in T-HyCroD node classification')
parser.add_argument('--no_node_edge_feat', action='store_true', help='deprecated alias kept for old commands; edge features are disabled by default')
parser.add_argument('--use_contrastive', action='store_true', help='add T-HyCroD multi-view contrastive loss during node classification')
parser.add_argument('--tau', type=float, default=0.2, help='temperature for T-HyCroD contrastive loss')
parser.add_argument('--lambda_dyn', type=float, default=1.0, help='weight for A-B dynamic-view contrastive loss')
parser.add_argument('--lambda_hyp', type=float, default=1.0, help='weight for A-C hyperdiff contrastive loss')
parser.add_argument('--curvature', type=float, default=1.0, help='Poincare ball curvature for T-HyCroD')
parser.add_argument('--use_hyperbolic', action='store_true', default=True, help='enable T-HyCroD hyperbolic mapping')
parser.add_argument('--use_time_encoding', action='store_true', default=None, help='enable T-HyCroD time encoding')
parser.add_argument('--use_view_b', action='store_true', default=True, help='enable T-HyCroD View B')
parser.add_argument('--use_historical_context', action='store_true', default=True, help='enable T-HyCroD historical context')
parser.add_argument('--edge_drop_rate', type=float, default=0.1, help='T-HyCroD View B edge dropout rate')
parser.add_argument('--feat_mask_rate', type=float, default=0.1, help='T-HyCroD View B feature mask rate')
parser.add_argument('--contrastive_batch_size', type=int, default=1024, help='max samples used for T-HyCroD contrastive loss')
parser.add_argument('--use_hyperdiff', action='store_true', help='enable T-HyCroD HyperDiff View C')

parser.add_argument('--new_node', action='store_true', help='model new node')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

if args.num_neighbors is not None:
    args.n_degree = args.num_neighbors
if args.num_layers is not None:
    args.n_layer = args.num_layers
if args.hidden_dim is not None:
    args.node_dim = args.hidden_dim
if args.model_name == 'thycrod':
    args = apply_thycrod_variant(args)
if args.use_time_encoding is not None:
    args.time = 'time' if args.use_time_encoding else 'empty'

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
MODEL = args.model_name
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)
logger.info('Current model: {}'.format(MODEL))
if MODEL == 'thycrod':
    logger.info('T-HyCroD variant: {}'.format(args.variant))
    logger.info('T-HyCroD config: curvature={}, lambda_dyn={}, lambda_hyp={}, tau={}, use_hyperdiff={}, use_hyperbolic={}, use_time_encoding={}, use_view_b={}, use_historical_context={}, edge_drop_rate={}, feat_mask_rate={}, num_neighbors={}, num_layers={}, hidden_dim={}'.format(
        args.curvature, args.lambda_dyn, args.lambda_hyp, args.tau, args.use_hyperdiff,
        args.use_hyperbolic, USE_TIME != 'empty', args.use_view_b, args.use_historical_context,
        args.edge_drop_rate, args.feat_mask_rate,
        NUM_NEIGHBORS, NUM_LAYER, NODE_DIM
    ))

### Load data and train val test split
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))

val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values

max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())

total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))

valid_test_flag = ts_l > test_time

if args.chronological_node_split:
    valid_train_flag = ts_l <= val_time
    valid_val_flag = (ts_l > val_time) * (ts_l <= test_time)
else:
    valid_pretest_flag = ts_l <= test_time
    assignment = np.random.randint(0, 10, len(valid_pretest_flag))
    valid_train_flag = valid_pretest_flag
    valid_val_flag = valid_pretest_flag * (assignment < 2)

if args.tune:
    valid_tune_flag = valid_pretest_flag if not args.chronological_node_split else (ts_l <= test_time)
    assignment = np.random.randint(0, 10, len(valid_tune_flag))
    valid_train_flag = valid_tune_flag * (assignment >= 2)
    valid_val_flag = valid_tune_flag * (assignment < 2)
    valid_test_flag = valid_val_flag

    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    train_label_l = label_l[valid_train_flag]

    val_src_l = src_l[valid_val_flag]
    val_dst_l = dst_l[valid_val_flag]
    val_ts_l = ts_l[valid_val_flag]
    val_e_idx_l = e_idx_l[valid_val_flag]
    val_label_l = label_l[valid_val_flag]

    # use the validation as test dataset in tuning mode
    test_src_l = src_l[valid_val_flag]
    test_dst_l = dst_l[valid_val_flag]
    test_ts_l = ts_l[valid_val_flag]
    test_e_idx_l = e_idx_l[valid_val_flag]
    test_label_l = label_l[valid_val_flag]
else:
    logger.info('Training with TGAT node-classification split for event-level dynamic labels')
    if args.chronological_node_split:
        logger.info('Using strict chronological train/val/test split')
    else:
        logger.info('Using original TGAT node split: train on all events before test_time')
    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    train_label_l = label_l[valid_train_flag]

    val_src_l = src_l[valid_val_flag]
    val_dst_l = dst_l[valid_val_flag]
    val_ts_l = ts_l[valid_val_flag]
    val_e_idx_l = e_idx_l[valid_val_flag]
    val_label_l = label_l[valid_val_flag]

    # use the true test dataset
    test_src_l = src_l[valid_test_flag]
    test_dst_l = dst_l[valid_test_flag]
    test_ts_l = ts_l[valid_test_flag]
    test_e_idx_l = e_idx_l[valid_test_flag]
    test_label_l = label_l[valid_test_flag]


### Initialize the data structure for graph and edge sampling
adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

train_nodes = set(np.unique(train_src_l))

### Model initialize
if GPU >= 0 and torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(GPU))
else:
    device = torch.device('cpu')
    if GPU >= 0:
        logger.warning('CUDA not available, falling back to CPU')
if MODEL == 'tgat':
    tgan = TGAN(train_ngh_finder, n_feat, e_feat,
                num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
                seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
else:
    tgan = T_HyCroD(train_ngh_finder, n_feat, e_feat,
                    num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD,
                    drop_out=DROP_OUT, node_dim=NODE_DIM, hidden_dim=args.hidden_dim, time_dim=TIME_DIM,
                    curvature=args.curvature, use_hyperbolic=args.use_hyperbolic,
                    tau=args.tau, lambda_dyn=args.lambda_dyn, lambda_hyp=args.lambda_hyp,
                    use_contrastive=args.use_contrastive, use_hyperdiff=args.use_hyperdiff,
                    max_contrastive_batch_size=args.contrastive_batch_size,
                    view_b_edge_dropout=args.edge_drop_rate, view_b_feature_mask=args.feat_mask_rate,
                    use_view_b=args.use_view_b, use_historical_context=args.use_historical_context)
tgan = tgan.to(device)


num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)
logger.debug('num of training instances: {}'.format(num_instance))
logger.debug('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)
np.random.shuffle(idx_list) 

display_model_name = 'TGAN' if MODEL == 'tgat' else 'T-HyCroD'
logger.info(f'loading saved {display_model_name} model if available')
if MODEL == 'tgat':
    model_path = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{DATA}.pth'
else:
    model_path = f'./saved_models/{args.prefix}-{args.model_name}-{args.agg_method}-{args.attn_mode}-{DATA}.pth'
if MODEL == 'tgat':
    tgan.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    logger.info('TGAN model loaded')
elif os.path.exists(model_path):
    loaded, skipped = load_compatible_state_dict(tgan, torch.load(model_path, map_location=device, weights_only=False))
    logger.info(f'T-HyCroD checkpoint loaded with compatible weights: {loaded}, skipped incompatible/missing: {skipped}')
else:
    logger.info(f'No T-HyCroD checkpoint found at {model_path}; training from current initialization')
tgan.eval()
logger.info('Start training node classification task')

base_clf_dim = tgan.feat_dim if hasattr(tgan, 'feat_dim') else tgan.n_feat_dim
node_clf_dim = base_clf_dim * 2 if MODEL == 'thycrod' and args.node_use_dst else base_clf_dim
if MODEL == 'thycrod' and not args.no_node_stats:
    node_clf_dim += 5
use_node_edge_feat = MODEL == 'thycrod' and args.node_edge_feat and not args.no_node_edge_feat
if use_node_edge_feat:
    node_clf_dim += tgan.raw_edge_dim
lr_model = LR(node_clf_dim)
train_encoder = args.train_encoder
optim_params = list(lr_model.parameters())
if train_encoder:
    optim_params += list(tgan.parameters())
lr_optimizer = torch.optim.Adam(optim_params, lr=args.lr)
lr_model = lr_model.to(device)
tgan.ngh_finder = full_ngh_finder
idx_list = np.arange(len(train_src_l))
if args.weighted_bce:
    pos_count = max(float(train_label_l.sum()), 1.0)
    neg_count = max(float(len(train_label_l) - train_label_l.sum()), 1.0)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float, device=device)
    logger.info(f'Using weighted BCE with pos_weight={pos_weight.item()}')
    lr_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    lr_criterion = torch.nn.BCEWithLogitsLoss()
lr_criterion_eval = torch.nn.BCEWithLogitsLoss()

def node_task_embedding(src_l_cut, dst_l_cut, ts_l_cut, e_idx_l_cut=None, training=False):
    if MODEL == 'thycrod' and args.node_use_dst:
        src_embed, dst_embed = tgan.get_temporal_embedding(
            src_l_cut, dst_l_cut, ts_l_cut, num_neighbors=NUM_NEIGHBORS
        )
        node_embed = torch.cat([src_embed, dst_embed], dim=1)
    else:
        node_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER)
    if MODEL == 'thycrod' and not args.no_node_stats:
        stats = tgan.temporal_node_stats(src_l_cut, ts_l_cut, NUM_NEIGHBORS)
        node_embed = torch.cat([node_embed, stats], dim=1)
    if use_node_edge_feat and e_idx_l_cut is not None:
        edge_idx_th = torch.from_numpy(e_idx_l_cut.astype(np.int64)).long().to(device)
        edge_feat = tgan.edge_raw_embed(edge_idx_th)
        node_embed = torch.cat([node_embed, edge_feat], dim=1)
    return node_embed

def eval_epoch(src_l, dst_l, ts_l, e_idx_l, label_l, batch_size, lr_model, tgan, num_layer=NODE_LAYER):
    pred_prob = np.zeros(len(src_l))
    loss = 0
    num_instance = len(src_l)
    num_batch = math.ceil(num_instance / batch_size)
    with torch.no_grad():
        lr_model.eval()
        tgan.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)
            src_l_cut = src_l[s_idx:e_idx]
            dst_l_cut = dst_l[s_idx:e_idx]
            ts_l_cut = ts_l[s_idx:e_idx]
            e_idx_l_cut = e_idx_l[s_idx:e_idx]
            label_l_cut = label_l[s_idx:e_idx]
            size = len(src_l_cut)
            if size == 0:
                continue
            src_embed = node_task_embedding(src_l_cut, dst_l_cut, ts_l_cut, e_idx_l_cut)
            src_label = torch.from_numpy(label_l_cut).float().to(device)
            lr_logits = lr_model(src_embed)
            lr_prob = lr_logits.sigmoid()
            loss += lr_criterion_eval(lr_logits, src_label).item()
            pred_prob[s_idx:e_idx] = lr_prob.cpu().numpy()

    auc_roc = roc_auc_score(label_l, pred_prob)
    ap = average_precision_score(label_l, pred_prob)
    acc = accuracy_score(label_l, pred_prob > 0.5)

    new_node_mask = np.array([n not in train_nodes for n in src_l])
    new_node_auc = roc_auc_score(label_l[new_node_mask], pred_prob[new_node_mask]) if new_node_mask.sum() > 0 else float('nan')
    new_node_ap = average_precision_score(label_l[new_node_mask], pred_prob[new_node_mask]) if new_node_mask.sum() > 0 else 0.0
    old_node_mask = np.array([n in train_nodes for n in src_l])
    old_node_auc = roc_auc_score(label_l[old_node_mask], pred_prob[old_node_mask]) if old_node_mask.sum() > 0 else float('nan')

    return auc_roc, ap, acc, loss / num_instance, new_node_auc, new_node_ap, old_node_auc



best_val_auc = float('-inf')
best_epoch = -1
best_lr_state = None
best_tgan_state = None
use_best_state = args.tune or args.chronological_node_split

for epoch in tqdm(range(args.n_epoch)):
    lr_pred_prob = np.zeros(len(train_src_l))
    np.random.shuffle(idx_list)
    tgan = tgan.train() if train_encoder else tgan.eval()
    lr_model = lr_model.train()
    #num_batch
    for k in range(num_batch):
        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance, s_idx + BATCH_SIZE)
        src_l_cut = train_src_l[s_idx:e_idx]
        dst_l_cut = train_dst_l[s_idx:e_idx]
        ts_l_cut = train_ts_l[s_idx:e_idx]
        e_idx_l_cut = train_e_idx_l[s_idx:e_idx]
        label_l_cut = train_label_l[s_idx:e_idx]
        
        size = len(src_l_cut)
        if size == 0:
            continue
        
        lr_optimizer.zero_grad()
        if train_encoder:
            src_embed = node_task_embedding(src_l_cut, dst_l_cut, ts_l_cut, e_idx_l_cut, training=True)
        else:
            with torch.no_grad():
                src_embed = node_task_embedding(src_l_cut, dst_l_cut, ts_l_cut, e_idx_l_cut)
        
        src_label = torch.from_numpy(label_l_cut).float().to(device)
        lr_logits = lr_model(src_embed)
        task_loss = lr_criterion(lr_logits, src_label)
        loss = task_loss
        contrast_dict = {}
        if MODEL == 'thycrod' and args.use_contrastive:
            contrast_dict = tgan.contrastive_loss_for_batch(src_l_cut, ts_l_cut, NUM_NEIGHBORS, NODE_LAYER)
            loss = loss + contrast_dict['loss_contrast']
        loss.backward()
        lr_optimizer.step()

    train_auc, train_ap, train_acc, train_loss, _, _, _ = eval_epoch(train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l, BATCH_SIZE, lr_model, tgan)
    val_auc, val_ap, val_acc, val_loss, val_new_auc, val_new_ap, val_old_auc = eval_epoch(val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l, BATCH_SIZE, lr_model, tgan)
    test_auc, test_ap, test_acc, test_loss, test_new_auc, test_new_ap, test_old_auc = eval_epoch(test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l, BATCH_SIZE, lr_model, tgan)
    #torch.save(lr_model.state_dict(), './saved_models/edge_{}_wkiki_node_class.pth'.format(DATA))
    logger.info(f'Epoch {epoch:3d} | train auc: {train_auc:.4f} ap: {train_ap:.4f} acc: {train_acc:.4f} loss: {train_loss:.6f}')
    logger.info(f'         | val auc: {val_auc:.4f} new_node_auc: {val_new_auc:.4f} old_node_auc: {val_old_auc:.4f}')
    logger.info(f'         | test auc: {test_auc:.4f} new_node_auc: {test_new_auc:.4f} old_node_auc: {test_old_auc:.4f}')
    if MODEL == 'thycrod' and args.use_contrastive and contrast_dict:
        logger.info(tgan.format_loss_log(task_loss.detach(), contrast_dict))
    val_metric = val_new_auc if not np.isnan(val_new_auc) else val_auc
    if use_best_state and val_metric > best_val_auc:
        best_val_auc = val_metric
        best_epoch = epoch
        best_lr_state = snapshot_state(lr_model)
        if train_encoder:
            best_tgan_state = snapshot_state(tgan)

if best_lr_state is not None:
    lr_model.load_state_dict(best_lr_state)
    if train_encoder and best_tgan_state is not None:
        tgan.load_state_dict(best_tgan_state)
    logger.info(f'Loaded best node-classification state from epoch {best_epoch} with val new_node_auc {best_val_auc:.4f}')

train_auc, train_ap, train_acc, train_loss, _, _, _ = eval_epoch(train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l, BATCH_SIZE, lr_model, tgan)
test_auc, test_ap, test_acc, test_loss, test_new_auc, test_new_ap, test_old_auc = eval_epoch(test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l, BATCH_SIZE, lr_model, tgan)
#torch.save(lr_model.state_dict(), './saved_models/edge_{}_wkiki_node_class.pth'.format(DATA))
logger.info(f'Final train statistics -- auc: {train_auc:.4f} ap: {train_ap:.4f} acc: {train_acc:.4f} loss: {train_loss:.6f}')
logger.info(f'=== Final Test Results ===')
logger.info(f'Test statistics: Old nodes -- auc: {test_old_auc:.4f} ap: {test_new_ap:.4f}')
logger.info(f'Test statistics: New nodes -- auc: {test_new_auc:.4f} ap: {test_new_ap:.4f}')




 




