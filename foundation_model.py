"""
Temporal Graph Foundation Model — cross-dataset pre-training for dynamic graphs.

Architecture:
  Per-Dataset Adapter → Shared Temporal Encoder → Multi-Task Heads

Pretext tasks (self-supervised, no labels needed):
  MTLP — Masked Temporal Link Prediction
  TNP  — Temporal Neighborhood Prediction
  CDC  — Cross-Domain Contrastive
  EFR  — Edge Feature Reconstruction
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from module import TimeEncode, AttnModel, MergeLayer
from graph import NeighborFinder


# ---------------------------------------------------------------------------
# Dataset adapter — maps heterogeneous raw features → shared hidden space
# ---------------------------------------------------------------------------

class DatasetAdapter(nn.Module):
    """Per-dataset input projection + learnable dataset embedding."""

    def __init__(self, raw_node_dim, raw_edge_dim, shared_dim, dataset_id, num_datasets, drop_out=0.1):
        super().__init__()
        self.raw_node_dim = raw_node_dim
        self.raw_edge_dim = raw_edge_dim
        self.shared_dim = shared_dim

        self.node_project = nn.Sequential(
            nn.Linear(raw_node_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(shared_dim, shared_dim),
        )
        self.edge_project = nn.Sequential(
            nn.Linear(raw_edge_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(shared_dim, shared_dim),
        )

        self.dataset_embed = nn.Embedding(num_datasets, shared_dim)

    def forward(self, node_feat, edge_feat):
        """node_feat: [B, raw_node_dim], edge_feat: [B, N, raw_edge_dim]"""
        node_out = self.node_project(node_feat)
        edge_out = self.edge_project(edge_feat)
        return node_out, edge_out


# ---------------------------------------------------------------------------
# Shared temporal encoder — K-hop TGAT-style aggregation (dataset-agnostic)
# ---------------------------------------------------------------------------

class SharedTemporalEncoder(nn.Module):
    """Multi-hop temporal neighbor aggregation shared across all datasets.

    Uses a ``feature_fn`` callback to resolve raw node IDs → projected embeddings.
    This keeps the recursive temporal convolution dataset-agnostic: each dataset
    provides its own adapter pipeline through the callback.
    """

    def __init__(self, shared_dim, time_dim, num_layers=2, n_head=2, drop_out=0.1, use_time='time'):
        super().__init__()
        self.shared_dim = shared_dim
        self.time_dim = time_dim
        self.num_layers = num_layers

        if use_time == 'time':
            self.time_encoder = TimeEncode(expand_dim=time_dim)
        else:
            self.time_encoder = None

        self.time_project = nn.Linear(time_dim, shared_dim) if use_time == 'time' else None

        self.attn_layers = nn.ModuleList([
            AttnModel(shared_dim, shared_dim, shared_dim,
                      attn_mode='prod', n_head=n_head, drop_out=drop_out)
            for _ in range(num_layers)
        ])

        self.merge_layer = MergeLayer(shared_dim, shared_dim, shared_dim, shared_dim)

        self._feature_fn = None
        self._edge_feature_fn = None

    def set_feature_fn(self, fn):
        """Attach a per-dataset node-id → projected-embedding resolver.

        ``fn(node_ids: np.ndarray) -> Tensor [N, shared_dim]``
        """
        self._feature_fn = fn

    def set_edge_feature_fn(self, fn):
        """Attach a per-dataset edge-id → projected-embedding resolver.

        ``fn(edge_ids: np.ndarray) -> Tensor [N, shared_dim]``
        """
        self._edge_feature_fn = fn

    def forward(self, src_idx_l, cut_time_l, num_neighbors=20, ngh_finder=None):
        """Temporal convolution entry point — resolves base features via feature_fn.

        Returns [batch_size, shared_dim].
        """
        if self._feature_fn is None:
            raise RuntimeError('SharedTemporalEncoder.feature_fn not set — call set_feature_fn() first.')
        return self._tem_conv(
            src_idx_l, cut_time_l, ngh_finder,
            curr_layers=self.num_layers, num_neighbors=num_neighbors
        )

    def _tem_conv(self, src_idx_l, cut_time_l, ngh_finder, curr_layers, num_neighbors=20):
        src_idx_np = np.asarray(src_idx_l, dtype=np.int64)
        base_feat = self._feature_fn(src_idx_np)

        if curr_layers == 0:
            return base_feat

        device = base_feat.device
        batch_size = len(src_idx_l)

        # Recurse to previous layer
        prev_embed = self._tem_conv(
            src_idx_l, cut_time_l, ngh_finder,
            curr_layers - 1, num_neighbors
        )

        # Sample historical neighbors
        ngh_node_batch, ngh_eidx_batch, ngh_t_batch = ngh_finder.get_temporal_neighbor(
            src_idx_l, cut_time_l, num_neighbors=num_neighbors
        )

        ngh_node_flat = ngh_node_batch.flatten()
        ngh_t_flat = ngh_t_batch.flatten()

        # Get neighbor embeddings from previous layer (uses feature_fn internally)
        ngh_prev = self._tem_conv(
            ngh_node_flat, ngh_t_flat, ngh_finder,
            curr_layers - 1, num_neighbors
        )
        ngh_feat = ngh_prev.view(batch_size, num_neighbors, self.shared_dim)

        # Time encoding for neighbor time deltas
        cut_time_th = torch.from_numpy(np.asarray(cut_time_l, dtype=np.float32)).float().to(device)
        ngh_t_delta = cut_time_th[:, None] - torch.from_numpy(ngh_t_batch).float().to(device)
        ngh_t_delta_th = ngh_t_delta

        src_t_embed = self._encode_time(torch.zeros(batch_size, 1, device=device))
        ngh_t_embed = self._encode_time(ngh_t_delta_th)

        # Edge features — project through adapter if available
        if self._edge_feature_fn is not None:
            ngh_edge_feat_flat = self._edge_feature_fn(ngh_eidx_batch.flatten())
            ngh_edge_feat = ngh_edge_feat_flat.view(batch_size, num_neighbors, self.shared_dim)
        else:
            ngh_edge_feat = torch.zeros(batch_size, num_neighbors, self.shared_dim, device=device)

        # Attention mask for padding
        mask = torch.from_numpy(ngh_node_batch == 0).to(device)

        # Attention aggregation
        attn_layer = self.attn_layers[curr_layers - 1]
        local, _ = attn_layer(prev_embed, src_t_embed, ngh_feat, ngh_t_embed, ngh_edge_feat, mask)

        return self.merge_layer(local, prev_embed)

    def _encode_time(self, ts):
        if self.time_encoder is None or self.time_project is None:
            return torch.zeros(*ts.shape, self.shared_dim, device=ts.device)
        encoded = self.time_encoder(ts)
        return self.time_project(encoded)


# ---------------------------------------------------------------------------
# Pretext task heads
# ---------------------------------------------------------------------------

class LinkPredictionHead(nn.Module):
    """Masked Temporal Link Prediction — discriminate real vs. random edges."""
    def __init__(self, shared_dim, drop_out=0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(shared_dim, 1),
        )

    def forward(self, src_emb, dst_emb):
        return self.scorer(torch.cat([src_emb, dst_emb], dim=-1)).squeeze(-1)


class NeighborhoodPredictionHead(nn.Module):
    """Temporal Neighborhood Prediction — predict neighborhood statistics."""
    def __init__(self, shared_dim, num_stats=5, drop_out=0.1):
        super().__init__()
        self.num_stats = num_stats
        self.predictor = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(shared_dim, num_stats),
        )

    def forward(self, node_emb):
        return self.predictor(node_emb)


class DomainDiscriminatorHead(nn.Module):
    """Cross-Domain Discrimination — adversarial head for domain-invariant learning."""
    def __init__(self, shared_dim, num_datasets, drop_out=0.1):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(shared_dim // 2, num_datasets),
        )

    def forward(self, node_emb):
        return self.discriminator(node_emb)


class TimeIntervalPredictionHead(nn.Module):
    """Time-Shift Prediction — predict if a temporal context has been shifted."""
    def __init__(self, shared_dim, drop_out=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(shared_dim, 1),
        )

    def forward(self, src_emb, dst_emb):
        return self.classifier(torch.cat([src_emb, dst_emb], dim=-1)).squeeze(-1)


class NodePropertyHead(nn.Module):
    """Node Property Prediction — predict structural node properties from embedding.

    Self-supervised pretext task: given a node's temporal embedding, predict its
    current structural properties (log-degree, clustering coefficient proxy,
    temporal activity). This forces the encoder to capture node-level state
    that is directly useful for downstream node classification.
    """
    def __init__(self, shared_dim, num_properties=5, drop_out=0.1):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(shared_dim, shared_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(shared_dim // 2, num_properties),
        )

    def forward(self, node_emb):
        return self.predictor(node_emb)


# ---------------------------------------------------------------------------
# Top-level Foundation Model
# ---------------------------------------------------------------------------

class TemporalGraphFoundationModel(nn.Module):
    """
    Cross-dataset temporal graph foundation model.

    Pre-training: joint optimization over multiple datasets with 4 pretext tasks.
    Fine-tuning: freeze shared encoder, adapt task head to downstream dataset.
    """

    def __init__(
        self,
        dataset_configs,          # list of dicts: {name, raw_node_dim, raw_edge_dim, n_nodes}
        shared_dim=256,
        time_dim=128,
        num_layers=2,
        n_head=4,
        drop_out=0.1,
        num_neighbors=20,
        num_datasets=None,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.time_dim = time_dim
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors

        num_datasets = num_datasets or len(dataset_configs)
        self.dataset_configs = dataset_configs
        self.dataset_names = [cfg['name'] for cfg in dataset_configs]
        self.name_to_id = {name: i for i, name in enumerate(self.dataset_names)}

        # Per-dataset adapters
        self.adapters = nn.ModuleDict()
        for cfg in dataset_configs:
            self.adapters[cfg['name']] = DatasetAdapter(
                raw_node_dim=cfg['raw_node_dim'],
                raw_edge_dim=cfg['raw_edge_dim'],
                shared_dim=shared_dim,
                dataset_id=self.name_to_id[cfg['name']],
                num_datasets=num_datasets,
                drop_out=drop_out,
            )

        # Shared temporal encoder
        self.encoder = SharedTemporalEncoder(
            shared_dim=shared_dim,
            time_dim=time_dim,
            num_layers=num_layers,
            n_head=n_head,
            drop_out=drop_out,
        )

        # Per-dataset node/edge raw embedding (frozen, loaded from data)
        self.dataset_node_embed = nn.ModuleDict()
        self.dataset_edge_embed = nn.ModuleDict()
        for cfg in dataset_configs:
            self.dataset_node_embed[cfg['name']] = nn.Embedding(
                cfg['n_nodes'] + 1, cfg['raw_node_dim'], padding_idx=0
            )
            self.dataset_edge_embed[cfg['name']] = nn.Embedding(
                cfg.get('n_edges', 1) + 1, cfg['raw_edge_dim'], padding_idx=0
            )

        # Pretext task heads
        self.link_head = LinkPredictionHead(shared_dim, drop_out)
        self.neighborhood_head = NeighborhoodPredictionHead(shared_dim, num_stats=5, drop_out=drop_out)
        self.domain_head = DomainDiscriminatorHead(shared_dim, num_datasets, drop_out)
        self.time_shift_head = TimeIntervalPredictionHead(shared_dim, drop_out)
        self.node_property_head = NodePropertyHead(shared_dim, num_properties=5, drop_out=drop_out)

        # Task weights for adaptive loss balancing (5 pretext tasks)
        self.task_log_vars = nn.Parameter(torch.zeros(5))

        # Fine-tuning heads (added during fine-tuning)
        self.ft_link_head = None
        self.ft_node_clf_head = None

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def load_dataset_embeddings(self, dataset_name, n_feat, e_feat):
        """Load pre-computed node and edge features into the model."""
        device = next(self.parameters()).device
        n_tensor = torch.from_numpy(n_feat.astype(np.float32)).to(device)
        e_tensor = torch.from_numpy(e_feat.astype(np.float32)).to(device)
        self.dataset_node_embed[dataset_name].weight.data.copy_(n_tensor)
        self.dataset_edge_embed[dataset_name].weight.data.copy_(e_tensor)
        self.dataset_node_embed[dataset_name].weight.requires_grad = False
        self.dataset_edge_embed[dataset_name].weight.requires_grad = False

    def _get_raw_features(self, dataset_name, node_ids, edge_ids):
        """Look up raw features from stored embeddings."""
        device = next(self.parameters()).device
        node_th = torch.from_numpy(node_ids.astype(np.int64)).long().to(device)
        edge_th = torch.from_numpy(edge_ids.astype(np.int64)).long().to(device)
        node_feat = self.dataset_node_embed[dataset_name](node_th)
        edge_feat = self.dataset_edge_embed[dataset_name](edge_th)
        return node_feat, edge_feat

    # ------------------------------------------------------------------
    # Core encoding
    # ------------------------------------------------------------------

    def encode_nodes(self, dataset_name, ngh_finder, src_idx_l, cut_time_l):
        """
        Full encoding pipeline: raw features → adapter → shared encoder → embeddings.

        Returns [batch_size, shared_dim].
        """
        device = next(self.parameters()).device
        adapter = self.adapters[dataset_name]
        ds_id = self.name_to_id[dataset_name]
        ds_emb = adapter.dataset_embed(torch.tensor(ds_id, device=device))

        def feature_fn(node_ids):
            """Resolve raw node IDs → projected embeddings for this dataset."""
            node_ids = np.asarray(node_ids, dtype=np.int64)
            raw_feat, _ = self._get_raw_features(
                dataset_name, node_ids,
                np.zeros(len(node_ids), dtype=np.int64)
            )
            projected, _ = adapter(raw_feat.float(), raw_feat.float().unsqueeze(1)[:, :1, :])
            return projected + ds_emb.unsqueeze(0).expand(len(node_ids), -1)

        def edge_feature_fn(edge_ids):
            """Resolve edge IDs → projected edge features."""
            edge_ids = np.asarray(edge_ids, dtype=np.int64)
            _, raw_edge = self._get_raw_features(
                dataset_name,
                np.zeros(len(edge_ids), dtype=np.int64),
                edge_ids,
            )
            _, edge_proj = adapter(
                raw_edge.new_zeros(len(edge_ids), adapter.raw_node_dim),
                raw_edge.float().unsqueeze(1),
            )
            return edge_proj.squeeze(1)

        self.encoder.set_feature_fn(feature_fn)
        self.encoder.set_edge_feature_fn(edge_feature_fn)
        return self.encoder.forward(src_idx_l, cut_time_l, num_neighbors=self.num_neighbors, ngh_finder=ngh_finder)

    # ------------------------------------------------------------------
    # Pretext task forward passes
    # ------------------------------------------------------------------

    def pretrain_forward(self, dataset_name, ngh_finder, batch_data):
        """
        Single pre-training step returning all task losses.

        batch_data dict:
          src_idx, dst_idx, cut_time — for link prediction
          neg_dst_idx — negative destination nodes
          ngh_stats — neighborhood statistics (target for TNP)
        """
        src_idx = batch_data['src_idx']
        dst_idx = batch_data['dst_idx']
        cut_time = batch_data['cut_time']
        neg_dst_idx = batch_data.get('neg_dst_idx')
        size = len(src_idx)

        # Encode
        src_emb = self.encode_nodes(dataset_name, ngh_finder, src_idx, cut_time)
        dst_emb = self.encode_nodes(dataset_name, ngh_finder, dst_idx, cut_time)

        losses = {}

        # ---- MTLP: Masked Temporal Link Prediction ----
        pos_logit = self.link_head(src_emb, dst_emb)
        if neg_dst_idx is not None:
            neg_emb = self.encode_nodes(dataset_name, ngh_finder, neg_dst_idx, cut_time)
            neg_logit = self.link_head(src_emb, neg_emb)
            link_loss = F.binary_cross_entropy_with_logits(
                pos_logit, torch.ones(size, device=pos_logit.device)
            ) + F.binary_cross_entropy_with_logits(
                neg_logit, torch.zeros(size, device=neg_logit.device)
            )
        else:
            link_loss = pos_logit.new_tensor(0.0)
        losses['mtlp'] = link_loss

        # ---- TNP: Temporal Neighborhood Prediction ----
        if 'ngh_stats' in batch_data and batch_data['ngh_stats'] is not None:
            stats_target = torch.from_numpy(batch_data['ngh_stats']).float().to(src_emb.device)
            stats_pred = self.neighborhood_head(src_emb)
            ngh_loss = F.mse_loss(stats_pred, stats_target)
        else:
            ngh_loss = src_emb.new_tensor(0.0)
        losses['tnp'] = ngh_loss

        # ---- CDC: Cross-Domain Contrastive ----
        ds_id = self.name_to_id[dataset_name]
        domain_logits = self.domain_head(src_emb)
        domain_target = torch.full((size,), ds_id, dtype=torch.long, device=domain_logits.device)
        domain_loss = F.cross_entropy(domain_logits, domain_target)
        losses['cdc'] = domain_loss

        # ---- EFR: Edge Feature Reconstruction (via link head in reverse direction) ----
        # Predict whether src→dst and dst→src agree (consistency regularization)
        reverse_logit = self.link_head(dst_emb, src_emb)
        efr_loss = F.mse_loss(torch.sigmoid(pos_logit), torch.sigmoid(reverse_logit))
        losses['efr'] = efr_loss

        # ---- NPP: Node Property Prediction ----
        # Predict structural node properties from temporal embedding.
        # Forces encoder to capture node-level state useful for node classification.
        if 'node_props' in batch_data and batch_data['node_props'] is not None:
            props_target = torch.from_numpy(batch_data['node_props']).float().to(src_emb.device)
            props_pred = self.node_property_head(src_emb)
            npp_loss = F.mse_loss(props_pred, props_target)
        else:
            npp_loss = src_emb.new_tensor(0.0)
        losses['npp'] = npp_loss

        return losses, src_emb, dst_emb

    def compute_pretrain_loss(self, losses):
        """Adaptive task weighting via learned uncertainty (5 pretext tasks)."""
        precision_mtlp = torch.exp(-self.task_log_vars[0])
        precision_tnp = torch.exp(-self.task_log_vars[1])
        precision_cdc = torch.exp(-self.task_log_vars[2])
        precision_efr = torch.exp(-self.task_log_vars[3])
        precision_npp = torch.exp(-self.task_log_vars[4])

        total = (
            precision_mtlp * losses['mtlp'] + self.task_log_vars[0] +
            precision_tnp * losses['tnp'] + self.task_log_vars[1] +
            precision_cdc * losses['cdc'] + self.task_log_vars[2] +
            precision_efr * losses['efr'] + self.task_log_vars[3] +
            precision_npp * losses['npp'] + self.task_log_vars[4]
        )
        return total

    # ------------------------------------------------------------------
    # Fine-tuning interface
    # ------------------------------------------------------------------

    def add_finetune_link_head(self):
        """Add a downstream link prediction head (reuses shared encoder)."""
        self.ft_link_head = LinkPredictionHead(self.shared_dim)

    def add_finetune_node_clf_head(self):
        """Add a downstream node classification head with richer interaction features.

        Input: cat(src_emb, dst_emb, src_emb * dst_emb, node_stats)
        The Hadamard product captures interaction patterns, and node_stats
        (5 temporal statistics) provide explicit node-level features that
        link-prediction embeddings alone lack.
        """
        # shared_dim*2 (src+dst) + shared_dim (hadamard) + 5 (node stats)
        input_dim = self.shared_dim * 3 + 5
        self.ft_node_clf_head = nn.Sequential(
            nn.Linear(input_dim, self.shared_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.shared_dim, self.shared_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.shared_dim // 2, 1),
        )

    def finetune_link_forward(self, dataset_name, ngh_finder, src_idx, dst_idx, cut_time):
        """Link prediction forward for fine-tuning."""
        src_emb = self.encode_nodes(dataset_name, ngh_finder, src_idx, cut_time)
        dst_emb = self.encode_nodes(dataset_name, ngh_finder, dst_idx, cut_time)
        if self.ft_link_head is None:
            self.add_finetune_link_head()
        return self.ft_link_head(src_emb, dst_emb)

    def finetune_node_clf_forward(self, dataset_name, ngh_finder, src_idx, dst_idx, cut_time):
        """Node classification forward for fine-tuning.

        Uses richer features than link prediction:
          - src and dst temporal embeddings (same encoder)
          - Hadamard product (src * dst) capturing interaction patterns
          - 5 temporal node statistics (log-degree, time deltas, density)
        """
        src_emb = self.encode_nodes(dataset_name, ngh_finder, src_idx, cut_time)
        dst_emb = self.encode_nodes(dataset_name, ngh_finder, dst_idx, cut_time)
        node_stats = self.compute_node_stats_tensor(dataset_name, ngh_finder, src_idx, cut_time)
        if self.ft_node_clf_head is None:
            self.add_finetune_node_clf_head()
            self.ft_node_clf_head = self.ft_node_clf_head.to(src_emb.device)
        combined = torch.cat([src_emb, dst_emb, src_emb * dst_emb, node_stats], dim=-1)
        return self.ft_node_clf_head(combined).squeeze(-1)

    # ------------------------------------------------------------------
    # Freeze / unfreeze for transfer learning
    # ------------------------------------------------------------------

    def freeze_encoder(self):
        """Freeze shared encoder + adapters for few-shot fine-tuning."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        for adapter in self.adapters.values():
            for p in adapter.parameters():
                p.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze for full fine-tuning."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        for adapter in self.adapters.values():
            for p in adapter.parameters():
                p.requires_grad = True

    def compute_neighborhood_stats(self, dataset_name, ngh_finder, src_idx_l, cut_time_l):
        """Compute ground-truth neighborhood stats for TNP pretext task.

        Returns numpy array [batch_size, 5]:
          log(1+deg), log(1+min_delta), log(1+mean_delta), log(1+max_delta), density
        """
        import numpy as np
        ngh_node, _, ngh_time = ngh_finder.get_temporal_neighbor(
            src_idx_l, cut_time_l, num_neighbors=self.num_neighbors
        )
        batch_size = len(src_idx_l)
        stats = np.zeros((batch_size, 5), dtype=np.float32)
        for i in range(batch_size):
            valid = (ngh_node[i] != 0) & (ngh_time[i] < cut_time_l[i])
            count = valid.sum()
            stats[i, 0] = np.log1p(count)
            if count > 0:
                deltas = cut_time_l[i] - ngh_time[i][valid]
                stats[i, 1] = np.log1p(deltas.min())
                stats[i, 2] = np.log1p(deltas.mean())
                stats[i, 3] = np.log1p(deltas.max())
                stats[i, 4] = min(count / max(self.num_neighbors, 1), 1.0)
        return stats

    def compute_node_stats_tensor(self, dataset_name, ngh_finder, src_idx_l, cut_time_l):
        """Compute temporal node statistics as a tensor for node classification.

        Returns [batch_size, 5] tensor on the same device as model parameters.
        These are the same 5 statistics used in T-HyCroD's node classification:
          log(1+deg), log(1+min_delta), log(1+mean_delta), log(1+max_delta), density
        """
        import numpy as np
        device = next(self.parameters()).device
        ngh_node, _, ngh_time = ngh_finder.get_temporal_neighbor(
            src_idx_l, cut_time_l, num_neighbors=self.num_neighbors
        )
        batch_size = len(src_idx_l)
        stats = np.zeros((batch_size, 5), dtype=np.float32)
        for i in range(batch_size):
            valid = (ngh_node[i] != 0) & (ngh_time[i] < cut_time_l[i])
            count = valid.sum()
            stats[i, 0] = np.log1p(count)
            if count > 0:
                deltas = cut_time_l[i] - ngh_time[i][valid]
                stats[i, 1] = np.log1p(deltas.min())
                stats[i, 2] = np.log1p(deltas.mean())
                stats[i, 3] = np.log1p(deltas.max())
                stats[i, 4] = min(count / max(self.num_neighbors, 1), 1.0)
        return torch.from_numpy(stats).float().to(device)

    def compute_node_properties(self, dataset_name, ngh_finder, src_idx_l, cut_time_l):
        """Compute node property targets for the NPP pretext task.

        Returns numpy array [batch_size, 5] — same format as compute_neighborhood_stats.
        Used during pre-training to provide self-supervised node-level targets.
        """
        return self.compute_neighborhood_stats(dataset_name, ngh_finder, src_idx_l, cut_time_l)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_pretrained(self, path):
        """Save foundation model for downstream fine-tuning."""
        state = {
            'model_state_dict': self.state_dict(),
            'dataset_configs': self.dataset_configs,
            'shared_dim': self.shared_dim,
            'time_dim': self.time_dim,
            'num_layers': self.num_layers,
            'num_neighbors': self.num_neighbors,
            'dataset_names': self.dataset_names,
        }
        torch.save(state, path)

    @classmethod
    def load_pretrained(cls, path, map_location='cpu'):
        """Load a pre-trained foundation model with backward compatibility."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(
            dataset_configs=checkpoint['dataset_configs'],
            shared_dim=checkpoint['shared_dim'],
            time_dim=checkpoint['time_dim'],
            num_layers=checkpoint['num_layers'],
            num_neighbors=checkpoint['num_neighbors'],
        )
        # Handle old checkpoints that have 4 task_log_vars instead of 5
        model_state = model.state_dict()
        for key, value in checkpoint['model_state_dict'].items():
            if key in model_state and model_state[key].shape == value.shape:
                model_state[key].copy_(value)
        model.load_state_dict(model_state)
        return model

    @classmethod
    def load_for_new_dataset(cls, path, new_dataset_config, map_location='cpu'):
        """
        Load pre-trained model and extend with a new dataset adapter for transfer.
        The shared encoder weights are preserved; new adapter is randomly initialized.
        """
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        # Combine old configs with new
        extended_configs = checkpoint['dataset_configs'] + [new_dataset_config]

        model = cls(
            dataset_configs=extended_configs,
            shared_dim=checkpoint['shared_dim'],
            time_dim=checkpoint['time_dim'],
            num_layers=checkpoint['num_layers'],
            num_neighbors=checkpoint['num_neighbors'],
        )
        # Load matching weights (old adapters + shared encoder)
        model_state = model.state_dict()
        for key, value in checkpoint['model_state_dict'].items():
            if key in model_state and model_state[key].shape == value.shape:
                model_state[key].copy_(value)
        model.load_state_dict(model_state)
        return model
