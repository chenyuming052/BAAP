"""
Anatomy-Guided Temporal Pre-training Modules
=============================================

Core architectural components for anatomy-level temporal change modeling:

1. CrossImageRegionalDiffAttention (CRDA)
   Cross-attention between prior/current ROI features with gated difference.

2. AnatomyTemporalPretrainHead
   Orchestrates ROI pooling, CRDA, and computes anatomy-level losses:
   - L_anat_contrastive: InfoNCE between image-deltas and text phrase embeddings
   - L_anat_classify: 3-way classification (improved/stable/worsened) per anatomy

These modules are used by medst_module_anatomy.py for Stage 2 pre-training.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ROI Pooling (reuses logic from anatomy_temporal_finetuner.py)
# ---------------------------------------------------------------------------
VIT_PATCH_SIZE = 16
VIT_GRID_SIZE = 14  # 224 / 16


def bbox_to_patch_mask(
    bboxes: torch.Tensor,
    patch_size: int = VIT_PATCH_SIZE,
    grid_size: int = VIT_GRID_SIZE,
) -> torch.Tensor:
    """Convert bboxes in 224-space to patch-level binary masks.

    Args:
        bboxes: [N, 4] (x1, y1, x2, y2) in 224x224 coordinates.

    Returns:
        masks: [N, grid_size^2] boolean tensor.
    """
    device = bboxes.device
    col_start = torch.clamp(torch.div(bboxes[:, 0].long(), patch_size, rounding_mode="floor"), 0, grid_size - 1)
    row_start = torch.clamp(torch.div(bboxes[:, 1].long(), patch_size, rounding_mode="floor"), 0, grid_size - 1)
    col_end = torch.clamp(torch.div(bboxes[:, 2].long() - 1, patch_size, rounding_mode="floor"), 0, grid_size - 1)
    row_end = torch.clamp(torch.div(bboxes[:, 3].long() - 1, patch_size, rounding_mode="floor"), 0, grid_size - 1)
    col_end = torch.max(col_start, col_end)
    row_end = torch.max(row_start, row_end)

    rows = torch.arange(grid_size, device=device)
    cols = torch.arange(grid_size, device=device)
    row_mask = (rows[None, :] >= row_start[:, None]) & (rows[None, :] <= row_end[:, None])
    col_mask = (cols[None, :] >= col_start[:, None]) & (cols[None, :] <= col_end[:, None])
    mask_2d = row_mask.unsqueeze(2) & col_mask.unsqueeze(1)
    return mask_2d.reshape(-1, grid_size * grid_size)


def roi_pool(
    patch_features: torch.Tensor,
    bboxes: torch.Tensor,
    sample_indices: torch.Tensor,
) -> torch.Tensor:
    """Masked average pooling of ViT patch tokens within bounding boxes.

    Args:
        patch_features: [B, 196, D]
        bboxes: [N, 4] in 224-space
        sample_indices: [N] long tensor

    Returns:
        roi_features: [N, D]
    """
    gathered = patch_features[sample_indices]
    masks = bbox_to_patch_mask(bboxes).float()
    masked = gathered * masks.unsqueeze(-1)
    counts = masks.sum(dim=1, keepdim=True).clamp(min=1.0)
    return masked.sum(dim=1) / counts


# ---------------------------------------------------------------------------
# ROI Projection
# ---------------------------------------------------------------------------
class ROIProjection(nn.Module):
    """Project pooled ROI features (768-dim) to embedding space (128-dim)."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Cross-Image Regional Difference Attention (CRDA)
# ---------------------------------------------------------------------------
class CrossImageRegionalDiffAttention(nn.Module):
    """Learn expressive change representations via cross-attention + gating.

    Given ROI features from prior and current images, produces a
    change-aware representation for each anatomy region.

    Architecture:
        Q = Linear(r_current)
        K, V = Linear(r_prior), Linear(r_prior)
        cross_attn = softmax(QK^T / sqrt(d)) @ V
        gate = sigmoid(Linear([r_current; cross_attn]))
        diff = gate * (r_current - cross_attn) + (1 - gate) * r_current
        output = LayerNorm(diff + r_current)
    """

    def __init__(self, dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.gate_proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.gate_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        r_current: torch.Tensor,
        r_prior: torch.Tensor,
        sample_indices: torch.Tensor,
        num_comparisons: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gated cross-attention difference.

        Args:
            r_current: [N, D] current ROI embeddings
            r_prior:   [N, D] prior ROI embeddings
            sample_indices: [N] batch sample index for each ROI
            num_comparisons: [B] number of comparisons per sample

        Returns:
            change_emb: [N, D] change-aware embeddings
        """
        N, D = r_current.shape

        if N == 0:
            return torch.zeros_like(r_current)

        # Per-sample cross-attention: for each sample's K anatomies,
        # current queries attend to all prior regions in the same sample.
        # Since K is small (typically 2-15), we use a simple batched approach.
        B = num_comparisons.shape[0]
        max_K = num_comparisons.max().item()

        # Pad into [B, max_K, D] for batched attention
        q_padded = torch.zeros(B, max_K, D, device=r_current.device, dtype=r_current.dtype)
        k_padded = torch.zeros(B, max_K, D, device=r_current.device, dtype=r_current.dtype)
        v_padded = torch.zeros(B, max_K, D, device=r_current.device, dtype=r_current.dtype)
        attn_mask = torch.ones(B, max_K, dtype=torch.bool, device=r_current.device)  # True = masked

        offset = 0
        for b_idx in range(B):
            n_k = num_comparisons[b_idx].item()
            if n_k > 0:
                q_padded[b_idx, :n_k] = r_current[offset:offset + n_k]
                k_padded[b_idx, :n_k] = r_prior[offset:offset + n_k]
                v_padded[b_idx, :n_k] = r_prior[offset:offset + n_k]
                attn_mask[b_idx, :n_k] = False
            offset += n_k

        # Project Q, K, V
        Q = self.q_proj(q_padded)  # [B, max_K, D]
        K = self.k_proj(k_padded)  # [B, max_K, D]
        V = self.v_proj(v_padded)  # [B, max_K, D]

        # Multi-head attention scores
        scale = math.sqrt(self.head_dim)
        Q = Q.view(B, max_K, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, K, d]
        K = K.view(B, max_K, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, max_K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, H, K, K]

        # Apply mask: set masked positions to -inf
        mask_2d = attn_mask.unsqueeze(1).unsqueeze(2).expand_as(attn_scores)  # [B, H, K, K]
        attn_scores = attn_scores.masked_fill(mask_2d, float("-inf"))

        # Handle fully-masked rows (prevent NaN from softmax of all -inf)
        row_mask = attn_mask.unsqueeze(1).unsqueeze(-1).expand_as(attn_scores)  # query mask
        attn_scores = attn_scores.masked_fill(row_mask, 0.0)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_weights = attn_weights.masked_fill(row_mask, 0.0)

        cross_attn = torch.matmul(attn_weights, V)  # [B, H, K, d]
        cross_attn = cross_attn.transpose(1, 2).reshape(B, max_K, D)  # [B, K, D]

        # Gated difference
        gate = torch.sigmoid(self.gate_proj(torch.cat([q_padded, cross_attn], dim=-1)))
        diff = gate * (q_padded - cross_attn) + (1 - gate) * q_padded
        out = self.norm(diff + q_padded)

        # Extract back to flat [N, D]
        result = torch.zeros(N, D, device=r_current.device, dtype=r_current.dtype)
        offset = 0
        for b_idx in range(B):
            n_k = num_comparisons[b_idx].item()
            if n_k > 0:
                result[offset:offset + n_k] = out[b_idx, :n_k]
            offset += n_k

        return result


# ---------------------------------------------------------------------------
# Anatomy Classification Head
# ---------------------------------------------------------------------------
class AnatomyClassifier(nn.Module):
    """MLP for per-anatomy temporal change prediction (3 classes)."""

    def __init__(self, input_dim: int, num_classes: int = 3, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss (SupCon, Khosla et al. 2020)
# ---------------------------------------------------------------------------
def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """SupCon loss: same-label pairs are positives, different-label are negatives.

    This directly uses temporal labels (improved/stable/worsened) to define
    positives/negatives in the change embedding space, providing 100% temporal
    learning signal (vs ~5.6% in the phrase-matching InfoNCE).

    Args:
        features: [N, D] L2-normalized embeddings
        labels: [N] integer class labels
        temperature: scalar

    Returns:
        scalar loss
    """
    N = features.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    sim = features @ features.t() / temperature  # [N, N]

    # Mask: same label = positive (excluding self)
    label_match = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [N, N]
    self_mask = 1.0 - torch.eye(N, device=features.device)
    pos_mask = label_match * self_mask

    # Skip rows with no positives (e.g., singleton class in batch)
    has_pos = pos_mask.sum(dim=1) > 0
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    # Log-softmax over all non-self entries
    # Mask out self-similarity with large negative value before logsumexp
    logits = sim * self_mask + (1.0 - self_mask) * (-1e9)
    log_prob = sim - torch.logsumexp(logits, dim=1, keepdim=True)

    # Mean of log-prob over positive pairs per anchor
    mean_log_prob = (pos_mask * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
    loss = -mean_log_prob[has_pos].mean()
    return loss


# ---------------------------------------------------------------------------
# Anatomy Temporal Pre-training Head
# ---------------------------------------------------------------------------
class AnatomyTemporalPretrainHead(nn.Module):
    """Anatomy-level temporal pre-training head.

    Given ViT patch features from prior/current images plus anatomy bboxes,
    computes two losses:
      1. L_anat_contrastive: InfoNCE between anatomy image-deltas and
         text phrase embeddings
      2. L_anat_classify: 3-way cross-entropy per anatomy

    Components:
      - ROI pooling (bbox -> patch averaging)
      - ROI projection (768 -> 128)
      - CRDA (cross-image regional difference attention)
      - Contrastive head (projects delta to contrastive space)
      - Classification head (predicts improved/stable/worsened)
    """

    def __init__(
        self,
        backbone_dim: int = 768,
        emb_dim: int = 128,
        num_classes: int = 3,
        num_heads: int = 4,
        contrastive_temperature: float = 0.1,
        class_weights: list = None,
        label_smoothing: float = 0.1,
        use_crda: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.temperature = contrastive_temperature
        self.label_smoothing = label_smoothing
        self.use_crda = use_crda

        # ROI projection: backbone_dim -> emb_dim
        self.roi_projection = ROIProjection(
            input_dim=backbone_dim, hidden_dim=emb_dim * 4, output_dim=emb_dim
        )

        # Cross-Image Regional Difference Attention
        if use_crda:
            self.crda = CrossImageRegionalDiffAttention(
                dim=emb_dim, num_heads=num_heads, dropout=dropout
            )

        # Contrastive projection (maps change embedding to contrastive space)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # Text projection (maps phrase embedding to same contrastive space)
        self.text_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # Classification head: concat(prior_roi, current_roi, change_emb) -> 3 classes
        self.classifier = AnatomyClassifier(
            input_dim=emb_dim * 3,
            num_classes=num_classes,
            hidden_dim=256,
            dropout=dropout,
        )

        # Class weights for imbalanced labels
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(
        self,
        prior_patch_feats: torch.Tensor,
        current_patch_feats: torch.Tensor,
        prior_bboxes: torch.Tensor,
        current_bboxes: torch.Tensor,
        sample_indices: torch.Tensor,
        num_comparisons: torch.Tensor,
        labels: torch.Tensor,
        phrase_embs: torch.Tensor = None,
        phrase_ids: torch.Tensor = None,
    ) -> dict:
        """Compute anatomy-level losses.

        Args:
            prior_patch_feats: [B, 196, 768] from ViT backbone (prior images)
            current_patch_feats: [B, 196, 768] from ViT backbone (current images)
            prior_bboxes: [N, 4] in 224-space
            current_bboxes: [N, 4] in 224-space
            sample_indices: [N] mapping to batch index
            num_comparisons: [B] comparisons per sample
            labels: [N] in {0=improved, 1=no_change, 2=worsened}
            phrase_embs: [N, emb_dim] text embeddings for each anatomy phrase
                         (from text encoder). None if contrastive loss is disabled.
            phrase_ids: [N, seq_len] tokenized phrase IDs for false-negative
                        masking. Entries sharing the same phrase_ids are treated
                        as positive pairs instead of negatives.

        Returns:
            dict with loss_anat_contrastive, loss_anat_classify, anat_acc
        """
        N = labels.shape[0]
        device = labels.device

        # Edge case: no comparisons in batch
        if N == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                "loss_anat_contrastive": zero,
                "loss_anat_classify": zero,
                "loss_supcon": zero,
                "anat_acc": torch.tensor(0.0, device=device),
            }

        # 1. ROI pooling
        prior_roi_raw = roi_pool(prior_patch_feats, prior_bboxes, sample_indices)    # [N, 768]
        current_roi_raw = roi_pool(current_patch_feats, current_bboxes, sample_indices)  # [N, 768]

        # 2. ROI projection: 768 -> 128
        prior_roi = self.roi_projection(prior_roi_raw)    # [N, emb_dim]
        current_roi = self.roi_projection(current_roi_raw)  # [N, emb_dim]

        # 3. Change representation
        if self.use_crda:
            change_emb = self.crda(current_roi, prior_roi, sample_indices, num_comparisons)
        else:
            change_emb = current_roi - prior_roi

        # 4. L_anat_classify: 3-way classification
        classify_input = torch.cat([prior_roi, current_roi, change_emb], dim=-1)  # [N, 3*emb_dim]
        logits = self.classifier(classify_input)  # [N, 3]
        loss_classify = F.cross_entropy(
            logits, labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        # Per-class accuracy: improved(0), no_change(1), worsened(2)
        per_class_acc = {}
        for cls_id, cls_name in enumerate(["improved", "stable", "worsened"]):
            mask = labels == cls_id
            if mask.sum() > 0:
                per_class_acc[f"acc_{cls_name}"] = (preds[mask] == cls_id).float().mean()
                per_class_acc[f"count_{cls_name}"] = mask.sum().float()
            else:
                per_class_acc[f"acc_{cls_name}"] = torch.tensor(0.0, device=device)
                per_class_acc[f"count_{cls_name}"] = torch.tensor(0.0, device=device)

        # 5. L_anat_contrastive: InfoNCE between image-deltas and phrase embeddings
        #    With phrase-aware false-negative masking: entries sharing the same
        #    phrase text are treated as positive pairs (soft labels) instead of
        #    negatives.  This is critical because the same phrase is reused across
        #    multiple anatomy regions (~3x duplication on average).
        loss_contrastive = torch.tensor(0.0, device=device, requires_grad=True)
        if phrase_embs is not None and phrase_embs.shape[0] == N:
            img_delta_proj = F.normalize(self.contrastive_proj(change_emb), dim=-1)  # [N, emb_dim]
            txt_emb_proj = F.normalize(self.text_proj(phrase_embs), dim=-1)  # [N, emb_dim]

            sim = img_delta_proj @ txt_emb_proj.t() / self.temperature  # [N, N]

            if phrase_ids is not None and phrase_ids.shape[0] == N:
                # Detect same-phrase pairs by comparing tokenized IDs
                # same_phrase[i,j] = True iff phrase_ids[i] == phrase_ids[j]
                same_phrase = (phrase_ids.unsqueeze(0) == phrase_ids.unsqueeze(1)).all(dim=-1)  # [N, N]
                # Soft labels: uniform probability over all positive (same-phrase) entries
                pos_mask = same_phrase.float()
                pos_mask = pos_mask / pos_mask.sum(dim=1, keepdim=True).clamp(min=1)
                # Soft cross-entropy with multiple positives per row
                loss_i2t = -(pos_mask * F.log_softmax(sim, dim=1)).sum(dim=1).mean()
                loss_t2i = -(pos_mask * F.log_softmax(sim.t(), dim=1)).sum(dim=1).mean()
            else:
                # Fallback: standard InfoNCE with hard labels
                nce_labels = torch.arange(N, device=device)
                loss_i2t = F.cross_entropy(sim, nce_labels)
                loss_t2i = F.cross_entropy(sim.t(), nce_labels)
            loss_contrastive = (loss_i2t + loss_t2i) / 2.0

        # 6. L_supcon: supervised contrastive on change embeddings using labels
        #    Directly uses temporal labels (improved/stable/worsened) to define
        #    positives/negatives, providing 100% temporal discrimination signal.
        #    Reuses contrastive_proj to share the embedding space.
        change_emb_norm = F.normalize(self.contrastive_proj(change_emb), dim=-1)
        loss_supcon = supervised_contrastive_loss(change_emb_norm, labels, self.temperature)

        return {
            "loss_anat_contrastive": loss_contrastive,
            "loss_anat_classify": loss_classify,
            "loss_supcon": loss_supcon,
            "anat_acc": acc,
            **per_class_acc,
        }
