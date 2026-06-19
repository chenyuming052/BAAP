"""
BAAP Anatomy-aware Temporal Fine-tuning
Location: medst/experiments/code/anatomy_temporal_finetuner.py

Task: Given a pair of CXR images (prior, current) with per-anatomy bounding boxes,
predict the temporal change label (improved / no_change / worsened) for each
anatomical region independently.

Architecture:
  1. Shared ViT-base backbone extracts 14x14 = 196 patch tokens (768-dim) per image
  2. ROI pooling maps anatomy bounding boxes to subset of patches, then avg-pools
  3. ROI projection: 768-dim -> 128-dim per region
  4. Per-anatomy fusion: concat(prior_roi, current_roi, diff) -> 384-dim
  5. MLP classifier: 384 -> 256 -> 128 -> 3  (per-anatomy 3-way classification)

Data format (from prepare_anatomy_temporal_dataset.py):
  Each JSONL sample has a prior-current CXR pair with variable number of
  per-anatomy comparisons. Each comparison has bbox coordinates in 224x224 space,
  a temporal change label, and associated clinical findings.

Usage:
    export PYTHONPATH=$PWD:${PYTHONPATH:-}
    python medst/experiments/code/anatomy_temporal_finetuner.py \
        --pretrained_ckpt /path/to/pretrained_encoder.ckpt \
        --data_dir /path/to/chest-imagenome/temporal_finetuning_dataset \
        --max_epochs 30 \
        --gpus 1
"""

import os
import sys
import datetime
import json
import random
import math
import time
from argparse import ArgumentParser
from typing import Optional, List, Tuple, Dict, Any
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from dateutil import tz

from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from torchvision import transforms
import torchvision.transforms.functional as TF
from torchmetrics import Accuracy, F1Score
from torch.optim.lr_scheduler import LambdaLR

# ============================================================================
# Project path setup
# ============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # experiments/code/
EXPERIMENTS_DIR = os.path.dirname(CURRENT_DIR)             # experiments/
MEDST_DIR = os.path.dirname(EXPERIMENTS_DIR)               # medst/
PROJECT_ROOT = os.path.dirname(MEDST_DIR)                  # MedST/

RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")
# Handle symlink: remove broken symlinks, skip makedirs for valid symlinks (Python 3.9 compat)
if os.path.islink(RESULTS_DIR):
    if not os.path.exists(RESULTS_DIR):
        os.remove(RESULTS_DIR)
if not os.path.islink(RESULTS_DIR) and not os.path.isdir(RESULTS_DIR):
    os.makedirs(RESULTS_DIR, exist_ok=True)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from medst.models.backbones.encoder import ImageEncoder
from medst.models.medst.anatomy_modules import CrossImageRegionalDiffAttention
print("Successfully imported MedST ImageEncoder")


# ============================================================================
# Constants
# ============================================================================
LABEL_MAP = {"improved": 0, "no_change": 1, "worsened": 2}
LABEL_NAMES = ["improved", "no_change", "worsened"]
VIT_PATCH_SIZE = 16
VIT_GRID_SIZE = 14   # 224 / 16
IMG_SIZE = 224

# 38 anatomy regions from Chest ImaGenome (ordered by frequency)
ANATOMY_LIST = [
    "right lung", "left lung", "left lower lung zone", "cardiac silhouette",
    "right hilar structures", "left hilar structures", "right lower lung zone",
    "right costophrenic angle", "left costophrenic angle", "mediastinum",
    "right mid lung zone", "left mid lung zone", "upper mediastinum",
    "right upper lung zone", "right apical zone", "left upper lung zone",
    "aortic arch", "left apical zone", "right chest wall", "left hemidiaphragm",
    "right hemidiaphragm", "left chest wall", "spine", "abdomen", "neck",
    "trachea", "right clavicle", "left clavicle", "right shoulder",
    "left shoulder", "right arm", "svc", "left arm", "right atrium",
    "carina", "right breast", "left breast", "cavoatrial junction",
]
ANATOMY_TO_IDX = {name: i for i, name in enumerate(ANATOMY_LIST)}
ANATOMY_UNK_IDX = len(ANATOMY_LIST)   # 38
NUM_ANATOMIES = len(ANATOMY_LIST) + 1  # 39 (including unknown)


# ============================================================================
# Image loading utilities (matching MedST preprocessing)
# ============================================================================
def resize_img(img, scale):
    """Aspect-preserving resize + zero-padding to (scale, scale).

    Copied from medst/datasets/utils.py to avoid import issues.
    """
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)

    if max_ind == 0:
        # image is taller
        wpercent = scale / float(size[0])
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
    else:
        # image is wider
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
    resized_img = cv2.resize(
        img, desireable_size[::-1], interpolation=cv2.INTER_AREA
    )

    if max_ind == 0:
        pad_size = scale - resized_img.shape[1]
        left = int(np.floor(pad_size / 2))
        right = int(np.ceil(pad_size / 2))
        top = int(0)
        bottom = int(0)
    else:
        pad_size = scale - resized_img.shape[0]
        top = int(np.floor(pad_size / 2))
        bottom = int(np.ceil(pad_size / 2))
        left = int(0)
        right = int(0)
    resized_img = np.pad(
        resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
    )
    return resized_img


def _bbox_224_to_256crop(bbox_224: list, crop_size: int = 224, resize_scale: int = 256) -> list:
    """Convert bbox from ImaGenome resize-224+pad space to MedST resize-256+pad+CenterCrop(224) space.

    Chest ImaGenome computes bbox_224 via aspect-preserving resize to 224 + zero-pad.
    MedST loads images via resize to 256 + zero-pad + CenterCrop(224).
    The conversion is: coord_crop = coord_224 * (256/224) - 16, clamped to [0, 224].
    """
    scale = resize_scale / crop_size  # 256 / 224 ≈ 1.143
    offset = (resize_scale - crop_size) / 2  # (256 - 224) / 2 = 16
    return [max(0.0, min(float(crop_size), coord * scale - offset)) for coord in bbox_224]


def load_cxr_image(img_path, scale=256):
    """Load CXR image following MedST pipeline.

    Pipeline: grayscale cv2 -> resize_img(256) -> RGB PIL Image (256x256).
    The subsequent transform applies CenterCrop(224) to produce the final 224x224
    image. Note: bbox_224 coordinates must be converted via _bbox_224_to_256crop()
    to align with this resize-256+CenterCrop(224) pipeline.
    """
    x = cv2.imread(str(img_path), 0)
    if x is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    x = resize_img(x, scale)
    return Image.fromarray(x).convert("RGB")


# ============================================================================
# Paired Temporal Transform (synchronized spatial augmentation)
# ============================================================================
class PairedTemporalTransform:
    """Apply synchronized spatial augmentation to prior/current image pairs.

    For temporal comparison tasks, spatial augmentations (crop, flip, affine)
    must be identical for both images so that geometric differences are not
    mistaken for temporal changes.  Color/intensity augmentations are applied
    independently to each image, which improves robustness to scanner and
    acquisition differences between time-points.

    Usage:
        transform = PairedTemporalTransform(img_size=224, mode="image_level")
        prior_tensor, current_tensor = transform(prior_pil, current_pil)
    """

    def __init__(self, img_size: int = 224, mode: str = "image_level"):
        self.img_size = img_size
        self.mode = mode  # "image_level" or "roi"

        # Color augmentation (applied independently to each image)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __call__(self, prior_img, current_img):
        if self.mode == "image_level":
            prior_img, current_img = self._spatial_sync(prior_img, current_img)
        else:
            # ROI/CLS: deterministic center crop (preserves bbox alignment)
            prior_img = TF.center_crop(prior_img, [self.img_size, self.img_size])
            current_img = TF.center_crop(current_img, [self.img_size, self.img_size])

        # Independent color augmentation per image (robust to scanner differences)
        prior_img = self.color_jitter(prior_img)
        current_img = self.color_jitter(current_img)

        # Gaussian blur (independent, p=0.2)
        if random.random() < 0.2:
            prior_img = TF.gaussian_blur(prior_img, kernel_size=7, sigma=random.uniform(0.1, 2.0))
        if random.random() < 0.2:
            current_img = TF.gaussian_blur(current_img, kernel_size=7, sigma=random.uniform(0.1, 2.0))

        if self.mode == "roi":
            # RandomAutocontrast (independent, p=0.2)
            if random.random() < 0.2:
                prior_img = TF.autocontrast(prior_img)
            if random.random() < 0.2:
                current_img = TF.autocontrast(current_img)

        # To tensor + normalize
        prior_t = self.normalize(self.to_tensor(prior_img))
        current_t = self.normalize(self.to_tensor(current_img))

        # NOTE: RandomErasing intentionally omitted. In temporal comparison tasks,
        # erasing the same region in both images can destroy subtle pathological
        # change signals (e.g. lesion improvement/worsening), biasing predictions
        # toward no_change.

        return prior_t, current_t

    def _spatial_sync(self, prior_img, current_img):
        """Synchronized spatial augmentation for both images."""
        w, h = prior_img.size  # PIL: (width, height)

        # 1. RandomResizedCrop: same crop for both
        i, j, crop_h, crop_w = transforms.RandomResizedCrop.get_params(
            prior_img, scale=(0.8, 1.0), ratio=(0.75, 4.0 / 3.0))
        prior_img = TF.resized_crop(prior_img, i, j, crop_h, crop_w, [self.img_size, self.img_size])
        current_img = TF.resized_crop(current_img, i, j, crop_h, crop_w, [self.img_size, self.img_size])

        # 2. RandomHorizontalFlip: same decision for both
        if random.random() < 0.5:
            prior_img = TF.hflip(prior_img)
            current_img = TF.hflip(current_img)

        # 3. RandomAffine: same params for both (degrees=15, shear=10, translate=0.05)
        angle = random.uniform(-15, 15)
        shear_x = random.uniform(-10, 10)
        shear_y = random.uniform(-10, 10)
        max_dx = 0.05 * self.img_size
        max_dy = 0.05 * self.img_size
        tx = random.uniform(-max_dx, max_dx)
        ty = random.uniform(-max_dy, max_dy)
        prior_img = TF.affine(prior_img, angle=angle, translate=[tx, ty],
                              scale=1.0, shear=[shear_x, shear_y])
        current_img = TF.affine(current_img, angle=angle, translate=[tx, ty],
                                scale=1.0, shear=[shear_x, shear_y])

        return prior_img, current_img


# ============================================================================
# ROI Pooling from ViT Patch Tokens
# ============================================================================
def bbox_to_patch_mask(
    bboxes: torch.Tensor,
    patch_size: int = VIT_PATCH_SIZE,
    grid_size: int = VIT_GRID_SIZE,
) -> torch.Tensor:
    """Convert bounding boxes in 224-space to patch-level binary masks.

    Args:
        bboxes: [N, 4] with (x1, y1, x2, y2) in 224x224 pixel coordinates.
        patch_size: ViT patch size in pixels (default 16).
        grid_size: ViT grid size (default 14, since 224 / 16 = 14).

    Returns:
        masks: [N, grid_size * grid_size] boolean tensor.
    """
    device = bboxes.device

    # Convert pixel coords to inclusive patch grid indices
    col_start = torch.clamp(torch.div(bboxes[:, 0].long(), patch_size, rounding_mode="floor"), 0, grid_size - 1)
    row_start = torch.clamp(torch.div(bboxes[:, 1].long(), patch_size, rounding_mode="floor"), 0, grid_size - 1)
    col_end = torch.clamp(torch.div(bboxes[:, 2].long() - 1, patch_size, rounding_mode="floor"), 0, grid_size - 1)
    row_end = torch.clamp(torch.div(bboxes[:, 3].long() - 1, patch_size, rounding_mode="floor"), 0, grid_size - 1)

    # Ensure end >= start
    col_end = torch.max(col_start, col_end)
    row_end = torch.max(row_start, row_end)

    # Build [N, G*G] mask via broadcasting
    rows = torch.arange(grid_size, device=device)  # [G]
    cols = torch.arange(grid_size, device=device)  # [G]

    row_mask = (rows[None, :] >= row_start[:, None]) & (rows[None, :] <= row_end[:, None])  # [N, G]
    col_mask = (cols[None, :] >= col_start[:, None]) & (cols[None, :] <= col_end[:, None])  # [N, G]

    # Outer product -> [N, G, G] then flatten
    mask_2d = row_mask.unsqueeze(2) & col_mask.unsqueeze(1)  # [N, rows, cols]
    return mask_2d.reshape(-1, grid_size * grid_size)


def roi_pool(
    patch_features: torch.Tensor,
    bboxes: torch.Tensor,
    sample_indices: torch.Tensor,
) -> torch.Tensor:
    """Extract ROI features via masked average pooling of ViT patch tokens.

    Args:
        patch_features: [B, 196, D] patch token features from ViT.
        bboxes: [N, 4] bounding boxes (x1, y1, x2, y2) in 224-space.
        sample_indices: [N] long tensor indexing into batch dim.

    Returns:
        roi_features: [N, D] average-pooled features per ROI.
    """
    gathered = patch_features[sample_indices]                # [N, 196, D]
    masks = bbox_to_patch_mask(bboxes).float()               # [N, 196]
    masked = gathered * masks.unsqueeze(-1)                  # [N, 196, D]
    counts = masks.sum(dim=1, keepdim=True).clamp(min=1.0)   # [N, 1]
    return masked.sum(dim=1) / counts                        # [N, D]


class AttentionROIPool(nn.Module):
    """Learnable attention-weighted ROI pooling over ViT patches.

    Instead of simple average pooling within the bbox, this module learns
    attention scores over patches to focus on diagnostically relevant regions.
    Zero-initialized last layer starts as uniform attention (≈ average pooling).
    """

    def __init__(self, dim: int = 768, hidden_dim: int = 192):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Zero-init last layer → starts ≈ uniform attention ≈ average pooling
        nn.init.zeros_(self.attn[-1].weight)
        nn.init.zeros_(self.attn[-1].bias)

    def forward(
        self,
        patch_features: torch.Tensor,
        bboxes: torch.Tensor,
        sample_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            patch_features: [B, 196, 768] from backbone.
            bboxes: [N, 4] in 224-space.
            sample_indices: [N] mapping to batch dim.
        Returns:
            [N, 768] attention-weighted ROI features.
        """
        gathered = patch_features[sample_indices]              # [N, 196, 768]
        masks = bbox_to_patch_mask(bboxes).float()             # [N, 196]
        scores = self.attn(gathered).squeeze(-1)               # [N, 196]
        scores = scores.masked_fill(masks == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)                    # [N, 196]
        return (gathered * weights.unsqueeze(-1)).sum(dim=1)   # [N, 768]


# ============================================================================
# Console Logger Callback (k9s-friendly, replaces tqdm)
# ============================================================================
class ConsoleLogCallback(Callback):
    """Print clean one-line training progress to stdout every N steps.

    Designed for non-interactive terminals (k9s, docker logs) where tqdm
    progress bars produce unreadable output.
    """

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._epoch_start = 0.0
        self._train_start = 0.0

    def on_train_start(self, trainer, pl_module):
        self._train_start = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        step = batch_idx + 1
        if step % self.log_every_n_steps != 0:
            return
        total = trainer.num_training_batches
        loss = trainer.callback_metrics.get("train/loss_step", float("nan"))
        acc = trainer.callback_metrics.get("train/acc_step", float("nan"))
        elapsed = time.time() - self._epoch_start
        eta_epoch = elapsed / step * (total - step)
        print(
            f"  Epoch {trainer.current_epoch} | "
            f"Step {step}/{total} | "
            f"loss={float(loss):.4f} | "
            f"acc={float(acc):.4f} | "
            f"ETA(epoch)={eta_epoch:.0f}s"
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        val_loss = float(m.get("val/loss", float("nan")))
        val_f1 = float(m.get("val/f1_macro", float("nan")))
        val_acc = float(m.get("val/acc", float("nan")))
        elapsed_total = time.time() - self._train_start
        print(
            f"  Epoch {trainer.current_epoch} [VAL] | "
            f"loss={val_loss:.4f} | "
            f"f1_macro={val_f1:.4f} | "
            f"acc={val_acc:.4f} | "
            f"elapsed={elapsed_total / 60:.1f}min"
        )


# ============================================================================
# EMA Callback
# ============================================================================
class EMACallback(Callback):
    """Exponential Moving Average of model weights for more stable evaluation.

    Swaps to EMA weights during validation, restores originals after.
    Only tracks parameters that have requires_grad=True.
    """

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        for name, param in pl_module.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.data.clone()
            else:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def on_validation_epoch_start(self, trainer, pl_module):
        self.backup = {}
        for name, param in pl_module.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def on_validation_epoch_end(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ============================================================================
# Classifier Head
# ============================================================================
class AnatomyClassifier(nn.Module):
    """MLP classifier for per-anatomy temporal change prediction.

    Architecture (when n_hidden is not None):
      Dropout -> Linear -> BN -> GELU -> Dropout -> Linear -> BN -> GELU
      -> Dropout -> Linear(n_classes)
    """

    def __init__(
        self,
        n_input: int,
        n_classes: int = 3,
        n_hidden: Optional[int] = 256,
        p: float = 0.1,
    ):
        super().__init__()

        if n_hidden is None:
            self.block = nn.Sequential(
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes),
            )
        else:
            self.block = nn.Sequential(
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.GELU(),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_hidden // 2, bias=False),
                nn.BatchNorm1d(n_hidden // 2),
                nn.GELU(),
                nn.Dropout(p=p),
                nn.Linear(n_hidden // 2, n_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ============================================================================
# ROI Projection Head
# ============================================================================
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


# ============================================================================
# Main Fine-tuning Model
# ============================================================================
class AnatomyTemporalFineTuner(LightningModule):
    """MedST anatomy-aware temporal fine-tuning model.

    Given a prior-current CXR pair and per-anatomy bounding boxes, predicts
    3-way temporal change (improved / no_change / worsened) for each anatomy.
    """

    def __init__(
        self,
        # Model
        pretrained_ckpt: Optional[str] = None,
        img_encoder: str = "vit_base",
        emb_dim: int = 128,
        num_classes: int = 3,
        hidden_dim: Optional[int] = 256,
        fusion_type: str = "concat_diff",   # "concat_diff" or "concat"
        use_anatomy_emb: bool = False,
        roi_mode: str = "roi",  # "roi", "cls", "image_level"
        # Training strategy
        freeze_backbone: bool = True,
        unfreeze_epoch: int = -1,
        unfreeze_layers: int = -1,
        backbone_grad_clip: float = 0.0,
        # Optimizer
        learning_rate: float = 1e-4,
        backbone_lr_scale: float = 0.01,
        weight_decay: float = 1e-4,
        warmup_ratio: float = 0.1,
        # Loss
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        mixup_alpha: float = 0.0,
        # Architecture
        use_crda: bool = False,
        use_attn_roi: bool = False,
        build_task_heads: bool = True,
        use_rfa_loss: bool = False,
        rfa_loss_weight: float = 0.1,
        # Other
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- Backbone ----
        self._build_backbone()
        self._backbone_unfrozen = False  # tracks whether staged unfreeze has fired
        if freeze_backbone:
            self._freeze_backbone()

        if build_task_heads:
            # ---- ROI projection: 768 -> emb_dim ----
            self.roi_projection = ROIProjection(
                input_dim=self.backbone.feature_dim,
                hidden_dim=emb_dim * 4,
                output_dim=emb_dim,
            )

            # ---- Attention ROI Pooling (optional, replaces roi_pool) ----
            if use_attn_roi:
                self.attn_roi_pool = AttentionROIPool(
                    dim=self.backbone.feature_dim,
                    hidden_dim=self.backbone.feature_dim // 4,
                )
                print(f"[AttentionROIPool] dim={self.backbone.feature_dim}")

            # ---- CRDA (optional, replaces simple diff) ----
            if use_crda:
                self.crda = CrossImageRegionalDiffAttention(
                    dim=emb_dim, num_heads=4, dropout=dropout)
                print(f"[CRDA] dim={emb_dim}, num_heads=4, dropout={dropout}")

            # ---- Anatomy embedding (optional) ----
            if use_anatomy_emb:
                self.anatomy_embedding = nn.Embedding(NUM_ANATOMIES, emb_dim)

            # ---- Classifier ----
            if fusion_type == "concat_diff":
                classifier_input = emb_dim * 3   # prior, current, diff
            else:
                classifier_input = emb_dim * 2   # prior, current
            if use_anatomy_emb:
                classifier_input += emb_dim      # + anatomy embedding

            self.classifier = AnatomyClassifier(
                n_input=classifier_input,
                n_classes=num_classes,
                n_hidden=hidden_dim,
                p=dropout,
            )

        else:
            self.roi_projection = None
            self.classifier = None
            print("[Init] build_task_heads=False: ROI/task heads are disabled")

        # ---- RFA loss cache (populated during forward, consumed in training_step) ----
        self._rfa_cache = {}

        # ---- Class weights ----
        if class_weights is not None:
            self.register_buffer(
                "class_weights", torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None

        if use_focal_loss:
            print(f"[FocalLoss] gamma={focal_gamma}, "
                  f"class_weights={'yes' if class_weights else 'no'}, "
                  f"label_smoothing={label_smoothing}")
        if mixup_alpha > 0:
            print(f"[ManifoldMixUp] alpha={mixup_alpha} (fused embedding level, training only)")

        # ---- Metrics ----
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)

    # ------------------------------------------------------------------ init
    def _build_backbone(self):
        self.backbone = ImageEncoder(
            model_name=self.hparams.img_encoder,
            output_dim=self.hparams.emb_dim,
        )
        if self.hparams.pretrained_ckpt and os.path.exists(self.hparams.pretrained_ckpt):
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        ckpt_path = self.hparams.pretrained_ckpt
        print(f"Loading pretrained weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith("img_encoder_q."):
                encoder_state[k.replace("img_encoder_q.", "")] = v

        if not encoder_state:
            print("  Warning: No img_encoder_q weights found in checkpoint")
            return

        # Filter out size-mismatched keys (e.g. when loading from a backbone with different
        # projection head dim, like MGCA's 128-dim projection vs MedST's 768-dim). BAAP only
        # uses raw ViT patch features so projection-head mismatches are safe to drop.
        own_state = self.backbone.state_dict()
        filtered_state = {}
        skipped_size_mismatch = []
        for k, v in encoder_state.items():
            if k in own_state and own_state[k].shape != v.shape:
                skipped_size_mismatch.append(
                    f"{k} ({tuple(v.shape)} vs {tuple(own_state[k].shape)})"
                )
                continue
            filtered_state[k] = v

        missing, unexpected = self.backbone.load_state_dict(
            filtered_state, strict=False
        )
        print(
            f"  Loaded {len(filtered_state)}/{len(encoder_state)} weights. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}, "
            f"SizeSkipped: {len(skipped_size_mismatch)}"
        )
        if missing:
            print(f"  Missing keys (first 5): {missing[:5]}")
        if skipped_size_mismatch:
            print(f"  Size-mismatched keys (first 5): {skipped_size_mismatch[:5]}")

    def _freeze_backbone(self):
        for param in self.backbone.model.parameters():
            param.requires_grad = False
        print("ViT backbone frozen (projection heads remain trainable)")

    def _unfreeze_backbone(self):
        n_layers = self.hparams.unfreeze_layers
        if n_layers > 0 and hasattr(self.backbone.model, "blocks"):
            # Layer-wise: only unfreeze top N transformer blocks + norm
            for param in self.backbone.model.parameters():
                param.requires_grad = False
            # Always unfreeze the final norm
            for param in self.backbone.model.norm.parameters():
                param.requires_grad = True
            # Unfreeze top N blocks
            total_blocks = len(self.backbone.model.blocks)
            start = max(0, total_blocks - n_layers)
            for i in range(start, total_blocks):
                for param in self.backbone.model.blocks[i].parameters():
                    param.requires_grad = True
            print(f"Backbone: unfroze top {n_layers}/{total_blocks} ViT blocks + norm")
        else:
            # Original: unfreeze everything
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("Backbone fully unfrozen")

    def _add_backbone_to_optimizer(self):
        """Dynamically add backbone params to the optimizer at unfreeze time.

        This avoids wasting memory on optimizer states for frozen params during
        the frozen phase, and gives the backbone a fresh LR schedule starting
        from the unfreeze epoch.
        """
        optimizer = self.optimizers()
        # Unwrap LightningOptimizer if needed
        if hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer

        backbone_lr = self.hparams.learning_rate * self.hparams.backbone_lr_scale
        # Only add params that were unfrozen (supports layer-wise unfreezing)
        unfrozen_params = [p for p in self.backbone.parameters() if p.requires_grad]
        optimizer.add_param_group({
            "params": unfrozen_params,
            "lr": backbone_lr,
            "name": "backbone",
        })

        # Sync scheduler to cover the new param group.
        # PyTorch LambdaLR does not auto-extend base_lrs/lr_lambdas when
        # param groups are added after init, causing zip() truncation.
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            sch = schedulers.scheduler if hasattr(schedulers, "scheduler") else schedulers
            sch.base_lrs.append(backbone_lr)
            sch.lr_lambdas.append(sch.lr_lambdas[0])  # reuse same cosine+warmup lambda

        print(f"Added backbone params to optimizer (lr={backbone_lr:.2e})")

    def _backbone_trainable(self):
        if not self.hparams.freeze_backbone:
            return True
        if self.hparams.unfreeze_epoch >= 0:
            try:
                if self.current_epoch >= self.hparams.unfreeze_epoch:
                    return True
            except ReferenceError:
                # Trainer weakref expired (standalone inference) — treat as unfrozen
                return True
        return False

    # ------------------------------------------------------------ hooks
    def on_after_backward(self):
        """Apply separate gradient clipping to backbone parameters if configured."""
        clip_val = self.hparams.backbone_grad_clip
        if clip_val > 0 and self._backbone_unfrozen:
            backbone_params = [
                p for p in self.backbone.model.parameters()
                if p.grad is not None
            ]
            if backbone_params:
                torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=clip_val)

    def on_train_batch_start(self, batch, batch_idx):
        if self.hparams.freeze_backbone:
            if self.hparams.unfreeze_epoch < 0 or self.current_epoch < self.hparams.unfreeze_epoch:
                self.backbone.eval()

    def on_train_epoch_start(self):
        if (
            self.hparams.freeze_backbone
            and self.hparams.unfreeze_epoch >= 0
            and self.current_epoch >= self.hparams.unfreeze_epoch
            and not self._backbone_unfrozen
        ):
            self._unfreeze_backbone()
            self.backbone.train()
            self._add_backbone_to_optimizer()
            self._backbone_unfrozen = True

    # ------------------------------------------------------------ forward
    def _extract_features(self, imgs: torch.Tensor):
        """Run backbone and return (cls_feat [B, 768], patch_feat [B, 196, 768]).

        Note: The MedST ViT uses ``len(x) == 3`` to detect tuple inputs
        (frontal, lateral, mask).  Since ``len(tensor)`` returns the batch
        dimension, a batch of exactly 3 images triggers the wrong branch.
        We pad to 4 when this happens and slice back afterwards.
        """
        B = imgs.shape[0]
        need_grad = self._backbone_trainable()
        with torch.set_grad_enabled(need_grad and self.training):
            if B == 3:
                imgs = torch.cat([imgs, imgs[:1]], dim=0)  # pad to 4
                cls_feat, patch_feat = self.backbone(imgs, view_type="frontal")
                cls_feat, patch_feat = cls_feat[:B], patch_feat[:B]
            else:
                cls_feat, patch_feat = self.backbone(imgs, view_type="frontal")
        return cls_feat, patch_feat

    def forward(self, batch: Dict, mixup_lam: float = None,
                mixup_perm: torch.Tensor = None) -> torch.Tensor:
        """Forward pass -> logits [N_total, 3] (per-anatomy) or [B, 3] (image-level)."""
        if self.roi_projection is None or self.classifier is None:
            raise RuntimeError(
                "Task heads are disabled (build_task_heads=False). "
                "Use a subclass that overrides forward for this mode."
            )

        prior_cls, prior_patches = self._extract_features(batch["prior_imgs"])
        current_cls, current_patches = self._extract_features(batch["current_imgs"])

        _cache_rfa = self.training and getattr(self.hparams, 'use_rfa_loss', False)

        if self.hparams.roi_mode == "image_level":
            # Image-level: CLS features directly, one prediction per pair
            prior_emb_raw = self.roi_projection(prior_cls)                     # [B, 128]
            current_emb_raw = self.roi_projection(current_cls)
            prior_emb = F.normalize(prior_emb_raw, dim=-1)
            current_emb = F.normalize(current_emb_raw, dim=-1)
        elif self.hparams.roi_mode == "cls":
            # CLS per-anatomy: expand CLS token for each comparison
            idx = batch["sample_indices"]                                      # [N]
            prior_emb_raw = self.roi_projection(prior_cls[idx])               # [N, 128]
            current_emb_raw = self.roi_projection(current_cls[idx])
            prior_emb = F.normalize(prior_emb_raw, dim=-1)
            current_emb = F.normalize(current_emb_raw, dim=-1)
        else:  # "roi" (default)
            idx = batch["sample_indices"]
            if getattr(self.hparams, 'use_attn_roi', False):
                prior_roi = self.attn_roi_pool(prior_patches, batch["prior_bboxes"], idx)
                current_roi = self.attn_roi_pool(current_patches, batch["current_bboxes"], idx)
            else:
                prior_roi = roi_pool(prior_patches, batch["prior_bboxes"], idx)    # [N, 768]
                current_roi = roi_pool(current_patches, batch["current_bboxes"], idx)
            prior_emb_raw = self.roi_projection(prior_roi)                    # [N, 128]
            current_emb_raw = self.roi_projection(current_roi)
            prior_emb = F.normalize(prior_emb_raw, dim=-1)
            current_emb = F.normalize(current_emb_raw, dim=-1)

        if _cache_rfa:
            self._rfa_cache = {"prior_raw": prior_emb_raw, "current_raw": current_emb_raw}

        # Fusion
        if self.hparams.fusion_type == "concat_diff":
            if getattr(self.hparams, 'use_crda', False) and self.hparams.roi_mode == "roi":
                num_comps = batch.get("num_comparisons")
                if num_comps is None:
                    num_comps = torch.bincount(
                        batch["sample_indices"],
                        minlength=batch["prior_imgs"].shape[0])
                change_emb = self.crda(
                    current_emb, prior_emb,
                    batch["sample_indices"], num_comps)
                fused = torch.cat([prior_emb, current_emb, change_emb], dim=-1)
            else:
                fused = torch.cat([prior_emb, current_emb, current_emb - prior_emb], dim=-1)
        else:
            fused = torch.cat([prior_emb, current_emb], dim=-1)

        # Anatomy conditioning
        if self.hparams.use_anatomy_emb:
            anat_emb = self.anatomy_embedding(batch["anatomy_indices"])  # [N, 128]
            fused = torch.cat([fused, anat_emb], dim=-1)

        # Manifold MixUp: interpolate fused embeddings before classifier
        if mixup_lam is not None and mixup_perm is not None:
            fused = mixup_lam * fused + (1 - mixup_lam) * fused[mixup_perm]

        return self.classifier(fused)

    # ------------------------------------------------------------ steps
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss with optional focal modulation."""
        if getattr(self.hparams, 'use_focal_loss', False):
            # Manual per-sample CE to avoid CUDA kernel bug with
            # F.cross_entropy(reduction='none', weight=..., label_smoothing=...)
            C = logits.size(-1)
            log_probs = F.log_softmax(logits, dim=-1)               # [B, C]
            pt = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1).exp()  # [B]

            # Label-smoothed target: mix one-hot with uniform (matches PyTorch)
            ls = self.hparams.label_smoothing
            one_hot = F.one_hot(labels, C).to(log_probs.dtype)      # [B, C]
            smooth = one_hot * (1 - ls) + ls / C                    # [B, C]
            ce = -(smooth * log_probs).sum(dim=-1)                   # [B]

            # Skip class_weights: focal (1-pt)^γ already down-weights easy
            # samples (typically majority class), avoiding double-correction.
            return (((1 - pt) ** self.hparams.focal_gamma) * ce).mean()
        return F.cross_entropy(
            logits, labels, weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing)

    def _compute_rfa_loss(self, batch: Dict) -> torch.Tensor:
        """Residual Feature Alignment loss.

        Passes the pixel-level residual image |current - prior| through the
        backbone + ROI pipeline and aligns the resulting embedding with the
        feature-space difference (current_raw - prior_raw) from the main
        forward pass. Gradients flow only through the residual path.
        """
        cache = self._rfa_cache
        self._rfa_cache = {}
        if not cache:
            return torch.tensor(0.0, device=self.device)

        # Pixel-level residual, shifted to backbone input range [-1, 1]
        residual_img = torch.abs(batch["current_imgs"] - batch["prior_imgs"]) - 1.0

        residual_cls, residual_patches = self._extract_features(residual_img)

        # Temporarily switch roi_projection to eval mode so that residual
        # features (very different distribution from real CXR) do not pollute
        # BatchNorm running statistics used at inference time.
        was_training = self.roi_projection.training
        self.roi_projection.eval()
        try:
            # Match the ROI mode used in forward()
            if self.hparams.roi_mode == "image_level":
                residual_emb = self.roi_projection(residual_cls)
            elif self.hparams.roi_mode == "cls":
                residual_emb = self.roi_projection(residual_cls[batch["sample_indices"]])
            else:  # "roi"
                idx = batch["sample_indices"]
                if getattr(self.hparams, 'use_attn_roi', False):
                    residual_roi = self.attn_roi_pool(
                        residual_patches, batch["current_bboxes"], idx)
                else:
                    residual_roi = roi_pool(
                        residual_patches, batch["current_bboxes"], idx)
                residual_emb = self.roi_projection(residual_roi)
        finally:
            if was_training:
                self.roi_projection.train()

        # Consistency: feature_diff ≈ residual_emb
        target_diff = (cache["current_raw"] - cache["prior_raw"]).detach()
        return F.mse_loss(residual_emb, target_diff)

    def shared_step(self, batch: Dict):
        labels = batch["labels"]

        mixup_alpha = getattr(self.hparams, 'mixup_alpha', 0.0)
        if self.training and mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            perm = torch.randperm(labels.size(0), device=labels.device)
            logits = self(batch, mixup_lam=lam, mixup_perm=perm)
            loss = lam * self._compute_loss(logits, labels) \
                 + (1 - lam) * self._compute_loss(logits, labels[perm])
        else:
            logits = self(batch)
            loss = self._compute_loss(logits, labels)
        return loss, logits, labels

    def training_step(self, batch: Dict, batch_idx: int):
        loss, logits, labels = self.shared_step(batch)
        preds = logits.argmax(dim=-1)
        self.train_acc(preds, labels)

        log_dict = {"train/loss_cls": loss, "train/acc": self.train_acc}
        total_loss = loss

        if getattr(self.hparams, 'use_rfa_loss', False):
            rfa_loss = self._compute_rfa_loss(batch)
            total_loss = loss + self.hparams.rfa_loss_weight * rfa_loss
            log_dict["train/rfa_loss"] = rfa_loss

        log_dict["train/loss"] = total_loss
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(labels),
        )
        return total_loss

    def validation_step(self, batch: Dict, batch_idx: int):
        loss, logits, labels = self.shared_step(batch)
        preds = logits.argmax(dim=-1)

        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_f1_per_class(preds, labels)

        self.log_dict(
            {"val/loss": loss, "val/acc": self.val_acc, "val/f1_macro": self.val_f1},
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(labels),
        )
        return loss

    def on_validation_epoch_end(self):
        f1_per_class = self.val_f1_per_class.compute()
        for i, name in enumerate(LABEL_NAMES):
            self.log(f"val/f1_{name}", f1_per_class[i], sync_dist=True)
        self.val_f1_per_class.reset()

    def test_step(self, batch: Dict, batch_idx: int):
        loss, logits, labels = self.shared_step(batch)
        preds = logits.argmax(dim=-1)

        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.test_f1_per_class(preds, labels)

        self.log_dict(
            {"test/loss": loss, "test/acc": self.test_acc, "test/f1_macro": self.test_f1},
            on_epoch=True,
            sync_dist=True,
            batch_size=len(labels),
        )
        return loss

    def on_test_epoch_end(self):
        f1_per_class = self.test_f1_per_class.compute()
        for i, name in enumerate(LABEL_NAMES):
            self.log(f"test/f1_{name}", f1_per_class[i], sync_dist=True)
        self.test_f1_per_class.reset()

    # ------------------------------------------------------------ optimizer
    def configure_optimizers(self):
        if self.roi_projection is None or self.classifier is None:
            raise RuntimeError(
                "Task heads are disabled (build_task_heads=False). "
                "Base configure_optimizers expects ROI/classifier heads."
            )

        new_params = list(self.roi_projection.parameters()) + list(self.classifier.parameters())
        if self.hparams.use_anatomy_emb:
            new_params += list(self.anatomy_embedding.parameters())
        if getattr(self.hparams, 'use_crda', False):
            new_params += list(self.crda.parameters())
        if getattr(self.hparams, 'use_attn_roi', False):
            new_params += list(self.attn_roi_pool.parameters())

        if self._backbone_trainable() and self.hparams.unfreeze_epoch < 0:
            # Backbone already trainable from the start (no staged unfreeze)
            param_groups = [
                {
                    "params": list(self.backbone.parameters()),
                    "lr": self.hparams.learning_rate * self.hparams.backbone_lr_scale,
                    "name": "backbone",
                },
                {
                    "params": new_params,
                    "lr": self.hparams.learning_rate,
                    "name": "new_heads",
                },
            ]
        else:
            # Frozen (or freeze-then-unfreeze): start with head params only.
            # Backbone params will be added dynamically via _add_backbone_to_optimizer()
            param_groups = [
                {"params": new_params, "lr": self.hparams.learning_rate, "name": "new_heads"}
            ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )

        # Total training steps
        if hasattr(self.trainer, "estimated_stepping_batches"):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            try:
                if self.trainer.datamodule:
                    train_loader_len = len(self.trainer.datamodule.train_dataloader())
                else:
                    train_loader_len = 100
            except Exception:
                train_loader_len = 100
            accumulate = getattr(self.trainer, "accumulate_grad_batches", 1)
            if not isinstance(accumulate, int):
                accumulate = 1
            total_steps = (train_loader_len // accumulate) * self.trainer.max_epochs

        total_steps = max(1, int(total_steps))
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)
        print(f"Optimizer: total_steps={total_steps}, warmup_steps={warmup_steps}")

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=_get_cosine_with_warmup_lambda(warmup_steps, total_steps),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


# ============================================================================
# Dataset
# ============================================================================
class AnatomyTemporalDataset(Dataset):
    """Dataset for anatomy-aware temporal change classification.

    Each sample is a prior-current CXR pair with variable-length per-anatomy
    comparisons, bounding boxes, and temporal change labels (improved / no_change
    / worsened).
    """

    def __init__(self, data_file: str, transform=None, split: str = "train",
                 image_root_remap: Optional[str] = None,
                 exclude_swapped: bool = False):
        self.transform = transform
        self.split = split
        self.image_root_remap = image_root_remap  # "OLD_PREFIX:NEW_PREFIX" or None
        self.exclude_swapped = exclude_swapped
        self.data = self._load_data(data_file)
        print(f"[{split}] Loaded {len(self.data)} samples"
              f"{' (excluded is_swapped=True)' if exclude_swapped else ''}")

    def _load_data(self, data_file: str) -> List[Dict]:
        data = []
        with open(data_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                # Filter out swapped samples if requested
                if self.exclude_swapped and item.get("is_swapped", False):
                    continue

                # Keep only comparisons with valid bboxes and labels
                valid_comps = []
                for c in item.get("comparisons", []):
                    if (
                        c.get("current_bbox_224") is not None
                        and c.get("prior_bbox_224") is not None
                        and c.get("label") in LABEL_MAP
                    ):
                        valid_comps.append(c)

                if valid_comps:
                    item["comparisons"] = valid_comps
                    item["num_comparisons"] = len(valid_comps)
                    data.append(item)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        # Load images (256x256 PIL RGB, matching MedST preprocessing)
        prior_path = item["prior_image_path"]
        current_path = item["current_image_path"]
        if self.image_root_remap:
            old_prefix, new_prefix = self.image_root_remap.split(":", 1)
            prior_path = prior_path.replace(old_prefix, new_prefix, 1)
            current_path = current_path.replace(old_prefix, new_prefix, 1)
        prior_img = load_cxr_image(prior_path, scale=256)
        current_img = load_cxr_image(current_path, scale=256)

        if isinstance(self.transform, PairedTemporalTransform):
            # Synchronized spatial augmentation for temporal pairs
            prior_img, current_img = self.transform(prior_img, current_img)
        elif self.transform:
            prior_img = self.transform(prior_img)
            current_img = self.transform(current_img)

        # Collect comparisons
        comparisons = []
        for c in item["comparisons"]:
            comparisons.append(
                {
                    "prior_bbox": torch.tensor(_bbox_224_to_256crop(c["prior_bbox_224"]), dtype=torch.float32),
                    "current_bbox": torch.tensor(_bbox_224_to_256crop(c["current_bbox_224"]), dtype=torch.float32),
                    "label": LABEL_MAP[c["label"]],
                    "anatomy": c["anatomy"],
                }
            )

        return {
            "prior_img": prior_img,
            "current_img": current_img,
            "comparisons": comparisons,
            "sample_id": item.get("sample_id", ""),
            "patient_id": str(item.get("patient_id", "")),
        }


def anatomy_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate that flattens variable-length comparisons with sample indices."""
    prior_imgs = torch.stack([b["prior_img"] for b in batch])
    current_imgs = torch.stack([b["current_img"] for b in batch])

    all_prior_bboxes = []
    all_current_bboxes = []
    all_labels = []
    all_sample_indices = []
    all_anatomies = []
    all_anatomy_indices = []
    num_comparisons = []

    for i, b in enumerate(batch):
        comps = b["comparisons"]
        num_comparisons.append(len(comps))
        for c in comps:
            all_prior_bboxes.append(c["prior_bbox"])
            all_current_bboxes.append(c["current_bbox"])
            all_labels.append(c["label"])
            all_sample_indices.append(i)
            all_anatomies.append(c["anatomy"])
            all_anatomy_indices.append(
                ANATOMY_TO_IDX.get(c["anatomy"], ANATOMY_UNK_IDX)
            )

    return {
        "prior_imgs": prior_imgs,                                           # [B, 3, 224, 224]
        "current_imgs": current_imgs,                                       # [B, 3, 224, 224]
        "prior_bboxes": torch.stack(all_prior_bboxes),                      # [N, 4]
        "current_bboxes": torch.stack(all_current_bboxes),                  # [N, 4]
        "labels": torch.tensor(all_labels, dtype=torch.long),               # [N]
        "sample_indices": torch.tensor(all_sample_indices, dtype=torch.long),  # [N]
        "anatomy_indices": torch.tensor(all_anatomy_indices, dtype=torch.long),  # [N]
        "num_comparisons": torch.tensor(num_comparisons, dtype=torch.long), # [B]
        "anatomies": all_anatomies,                                         # list[str]
    }


def _majority_label(comparisons: List[Dict]) -> int:
    """Majority vote of per-anatomy labels for image-level classification."""
    from collections import Counter
    labels = [c["label"] for c in comparisons]
    return Counter(labels).most_common(1)[0][0]


def image_level_collate_fn(batch: List[Dict]) -> Dict:
    """Collate for image-level mode: one label per image pair (majority vote)."""
    return {
        "prior_imgs": torch.stack([b["prior_img"] for b in batch]),
        "current_imgs": torch.stack([b["current_img"] for b in batch]),
        "labels": torch.tensor(
            [_majority_label(b["comparisons"]) for b in batch],
            dtype=torch.long,
        ),
    }


# ============================================================================
# Data Module
# ============================================================================
class AnatomyTemporalDataModule(LightningDataModule):
    """Lightning DataModule for the anatomy-aware temporal dataset.

    Expects a directory with train.jsonl, valid.jsonl, test.jsonl produced by
    prepare_anatomy_temporal_dataset.py.
    """

    def __init__(
        self,
        data_dir: str,
        train_file: str = "train.jsonl",
        val_file: str = "valid.jsonl",
        test_file: str = "test.jsonl",
        batch_size: int = 16,
        num_workers: int = 8,
        img_size: int = 224,
        image_root_remap: Optional[str] = None,
        roi_mode: str = "roi",
        exclude_swapped: bool = False,
        strong_augmentation: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.image_root_remap = image_root_remap
        self.roi_mode = roi_mode
        self.exclude_swapped = exclude_swapped
        self.train_path = self._resolve_data_path(os.path.join(data_dir, train_file))
        self.val_path = self._resolve_data_path(os.path.join(data_dir, val_file))
        self.test_path = self._resolve_data_path(os.path.join(data_dir, test_file))

        # Select training transform based on augmentation mode
        if strong_augmentation:
            # Use PairedTemporalTransform for synchronized spatial augmentation
            self.train_transform = PairedTemporalTransform(
                img_size=img_size, mode=roi_mode if roi_mode != "cls" else "roi")
            mode_label = "image_level (synced spatial)" if roi_mode == "image_level" else "ROI/CLS (bbox-safe)"
            print(f"[DataModule] Using PairedTemporalTransform ({mode_label})")
        else:
            self.train_transform = self._get_train_transform(img_size)
        self.val_transform = self._get_val_transform(img_size)

    @staticmethod
    def _resolve_data_path(path: str) -> str:
        """If *path* doesn't exist, try the ``_clean`` variant (e.g. train.jsonl -> train_clean.jsonl)."""
        if os.path.exists(path):
            return path
        base, ext = os.path.splitext(path)
        clean_path = f"{base}_clean{ext}"
        if os.path.exists(clean_path):
            print(f"[INFO] {path} not found, using {clean_path}")
            return clean_path
        return path  # let it fail later with a clear error

    @staticmethod
    def _get_train_transform(img_size: int):
        """Training transform: CenterCrop (to preserve bbox coords) + mild color augmentation."""
        return transforms.Compose(
            [
                transforms.CenterCrop(img_size),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    @staticmethod
    def _get_train_transform_image_level(img_size: int):
        """Strong spatial augmentation for image_level mode (no bbox dependency).

        Inspired by BioViL-T: random crop, horizontal flip, affine transforms,
        color jitter, Gaussian blur, and random erasing.
        """
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, shear=10, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
            transforms.RandomApply([transforms.GaussianBlur(7, sigma=(0.1, 2.0))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])

    @staticmethod
    def _get_train_transform_roi(img_size: int):
        """Bbox-safe augmentation for ROI/CLS modes (preserves spatial coords).

        Only non-spatial augmentations: stronger color jitter, Gaussian blur,
        auto-contrast, and random erasing.
        """
        return transforms.Compose([
            transforms.CenterCrop(img_size),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
            transforms.RandomApply([transforms.GaussianBlur(7, sigma=(0.1, 2.0))], p=0.2),
            transforms.RandomAutocontrast(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])

    @staticmethod
    def _get_val_transform(img_size: int):
        return transforms.Compose(
            [
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = AnatomyTemporalDataset(
                self.train_path, self.train_transform, split="train",
                image_root_remap=self.image_root_remap,
                exclude_swapped=self.exclude_swapped,
            )
            self.val_dataset = AnatomyTemporalDataset(
                self.val_path, self.val_transform, split="valid",
                image_root_remap=self.image_root_remap,
            )
        if stage == "test" and os.path.exists(self.test_path):
            self.test_dataset = AnatomyTemporalDataset(
                self.test_path, self.val_transform, split="test",
                image_root_remap=self.image_root_remap,
            )

    @staticmethod
    def _worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
        random.seed(torch.initial_seed() % 2**32)

    def _collate_fn(self):
        """Select collate function based on roi_mode."""
        if self.roi_mode == "image_level":
            return image_level_collate_fn
        return anatomy_collate_fn

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self._collate_fn(),
            worker_init_fn=self._worker_init_fn,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self._collate_fn(),
            worker_init_fn=self._worker_init_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        if hasattr(self, "test_dataset"):
            return DataLoader(
                self.test_dataset,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.hparams.num_workers,
                collate_fn=self._collate_fn(),
                worker_init_fn=self._worker_init_fn,
                pin_memory=True,
            )
        return None


# ============================================================================
# Utility functions
# ============================================================================
def compute_class_weights(data_file: str, num_classes: int = 3,
                          exclude_swapped: bool = False) -> List[float]:
    """Compute inverse-frequency class weights from the training JSONL."""
    counts = Counter()
    with open(data_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if exclude_swapped and item.get("is_swapped", False):
                continue
            for c in item.get("comparisons", []):
                label = c.get("label")
                if label in LABEL_MAP:
                    counts[LABEL_MAP[label]] += 1

    total = sum(counts.values())
    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        weights.append(total / (num_classes * c))

    dist = [counts.get(i, 0) for i in range(num_classes)]
    print(f"Class distribution ({LABEL_NAMES}): {dist}")
    print(f"Class weights: {[f'{w:.3f}' for w in weights]}")
    return weights


def _get_cosine_with_warmup_lambda(warmup_steps, total_steps, eta_min_ratio=1e-8):
    """Cosine schedule with linear warmup."""

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(eta_min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return lr_lambda


# ============================================================================
# Main
# ============================================================================
def main():
    parser = ArgumentParser(description="MedST Anatomy-Aware Temporal Fine-tuning")
    parser = Trainer.add_argparse_args(parser)

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with train.jsonl, valid.jsonl, test.jsonl",
    )
    parser.add_argument("--train_file", type=str, default="train.jsonl")
    parser.add_argument("--val_file", type=str, default="valid.jsonl")
    parser.add_argument("--test_file", type=str, default="test.jsonl")
    parser.add_argument("--image_root_remap", type=str, default=None,
                        help="Remap image path prefix, format 'OLD:NEW' (e.g. '/old/data:/new/data')")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--exclude_swapped", action="store_true",
                        help="Exclude is_swapped=True samples from training data")
    parser.add_argument("--strong_augmentation", action="store_true",
                        help="Use stronger data augmentation (spatial for image_level, bbox-safe for ROI/CLS)")

    # Model
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    parser.add_argument("--img_encoder", type=str, default="vit_base")
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument(
        "--fusion_type",
        type=str,
        default="concat_diff",
        choices=["concat_diff", "concat"],
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--use_anatomy_emb",
        action="store_true",
        help="Add learnable anatomy identity embedding to fusion features",
    )
    parser.add_argument(
        "--roi_mode", type=str, default="roi",
        choices=["roi", "cls", "image_level"],
        help="Feature mode: roi=ROI pooling, cls=CLS token per-anatomy, image_level=CLS image-level",
    )

    # Training strategy
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument(
        "--no_freeze_backbone", action="store_false", dest="freeze_backbone"
    )
    parser.add_argument("--unfreeze_epoch", type=int, default=-1)
    parser.add_argument(
        "--unfreeze_layers", type=int, default=-1,
        help="Number of top ViT blocks to unfreeze (-1 = all blocks, 2 = top 2, etc.)",
    )
    parser.add_argument(
        "--backbone_grad_clip", type=float, default=0.0,
        help="Max gradient norm for backbone params (0 = disabled)",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--backbone_lr_scale", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--use_focal_loss", action="store_true",
                        help="Use focal loss instead of cross-entropy")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma (focusing parameter)")
    parser.add_argument("--mixup_alpha", type=float, default=0.0,
                        help="Manifold MixUp alpha (0=off, >0=Beta(a,a) fused embedding mixing)")
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--use_crda", action="store_true",
                        help="Use CrossImageRegionalDiffAttention (ROI mode only)")
    parser.add_argument("--use_attn_roi", action="store_true",
                        help="Use learnable attention ROI pooling (ROI mode only)")
    parser.add_argument("--use_rfa_loss", action="store_true",
                        help="Add Residual Feature Alignment auxiliary loss")
    parser.add_argument("--rfa_loss_weight", type=float, default=0.1,
                        help="Weight for RFA loss (default: 0.1)")
    parser.add_argument("--use_ema", action="store_true", help="Enable EMA for validation")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")

    # Experiment
    parser.add_argument("--experiment_name", type=str, default="anatomy_temporal",
                        help="Experiment subdirectory name (default: anatomy_temporal)")
    parser.add_argument("--results_dir", type=str, default="",
                        help="Root directory for saving results (default: MedST/medst/experiments/results)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--disable_early_stopping", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_tag", type=str, default="")

    args = parser.parse_args()

    if args.max_epochs is None:
        args.max_epochs = 20

    seed_everything(args.seed, workers=True)

    # ---- Class weights ----
    class_weights = None
    if args.use_class_weights:
        train_path = AnatomyTemporalDataModule._resolve_data_path(
            os.path.join(args.data_dir, args.train_file)
        )
        if os.path.exists(train_path):
            class_weights = compute_class_weights(
                train_path, args.num_classes,
                exclude_swapped=args.exclude_swapped,
            )
        else:
            print(f"WARNING: --use_class_weights set but train file not found: {train_path}")

    # ---- Data ----
    datamodule = AnatomyTemporalDataModule(
        data_dir=args.data_dir,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        image_root_remap=args.image_root_remap,
        roi_mode=args.roi_mode,
        exclude_swapped=args.exclude_swapped,
        strong_augmentation=args.strong_augmentation,
    )

    # ---- Model ----
    model = AnatomyTemporalFineTuner(
        pretrained_ckpt=args.pretrained_ckpt,
        img_encoder=args.img_encoder,
        emb_dim=args.emb_dim,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        fusion_type=args.fusion_type,
        freeze_backbone=args.freeze_backbone,
        unfreeze_epoch=args.unfreeze_epoch,
        unfreeze_layers=args.unfreeze_layers,
        backbone_grad_clip=args.backbone_grad_clip,
        learning_rate=args.learning_rate,
        backbone_lr_scale=args.backbone_lr_scale,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        mixup_alpha=args.mixup_alpha,
        use_crda=args.use_crda,
        use_attn_roi=args.use_attn_roi,
        use_rfa_loss=args.use_rfa_loss,
        rfa_loss_weight=args.rfa_loss_weight,
        dropout=args.dropout,
        use_anatomy_emb=args.use_anatomy_emb,
        roi_mode=args.roi_mode,
    )

    # ---- Experiment directory ----
    # In DDP, each rank runs main() independently with slightly different
    # timestamps. Rank 0 generates the directory and signals other ranks
    # via a sync file to ensure all ranks use the same exp_dir.
    results_dir = args.results_dir if args.results_dir else RESULTS_DIR
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    sync_file = os.path.join(results_dir, args.experiment_name, ".ddp_exp_dir")

    if local_rank == 0:
        # Clean up stale sync file from previous runs
        if os.path.exists(sync_file):
            os.remove(sync_file)

        now = datetime.datetime.now(tz.tzlocal())
        timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
        run_name = f"{args.run_tag}_{timestamp}" if args.run_tag else timestamp
        exp_dir = os.path.join(results_dir, args.experiment_name, run_name)
        os.makedirs(exp_dir, exist_ok=True)

        with open(sync_file, "w") as f:
            f.write(exp_dir)
    else:
        # Wait for rank 0 to create exp_dir (120 retries × 0.5s = 60s max)
        for _ in range(120):
            if os.path.exists(sync_file):
                with open(sync_file, "r") as f:
                    exp_dir = f.read().strip()
                if exp_dir and os.path.isdir(exp_dir):
                    break
            time.sleep(0.5)
        else:
            raise RuntimeError(f"Rank {local_rank}: timed out waiting for exp_dir from rank 0")

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- Logger ----
    if args.use_wandb:
        logger = WandbLogger(
            project="MedST-AnatomyTemporal", name=os.path.basename(exp_dir), save_dir=exp_dir
        )
    else:
        logger = TensorBoardLogger(save_dir=exp_dir, name="logs")

    # ---- Callbacks ----
    callbacks = [
        ModelCheckpoint(
            dirpath=exp_dir,
            filename="{epoch:02d}-{val/f1_macro:.4f}",
            monitor="val/f1_macro",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        ConsoleLogCallback(log_every_n_steps=50),
    ]
    if not args.disable_early_stopping:
        callbacks.append(
            EarlyStopping(monitor="val/loss", patience=args.patience, mode="min", verbose=True)
        )
    if args.use_ema:
        callbacks.append(EMACallback(decay=args.ema_decay))
        print(f"  EMA enabled (decay={args.ema_decay})")

    # ---- Trainer ----
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        default_root_dir=exp_dir,
    )

    # ---- Summary ----
    print("=" * 60)
    print("  Anatomy-Aware Temporal Fine-Tuning (Plan B)")
    print("=" * 60)
    print(f"  Experiment   : {os.path.basename(exp_dir)}")
    print(f"  Results dir  : {exp_dir}")
    print(f"  Data dir     : {args.data_dir}")
    print(f"  Fusion       : {args.fusion_type}")
    print(f"  Anatomy emb  : {args.use_anatomy_emb}")
    print(f"  Freeze       : {args.freeze_backbone}")
    print(f"  Class weights: {class_weights is not None}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  LR           : {args.learning_rate}")
    print(f"  Patience     : {args.patience}")
    print(f"  Max epochs   : {args.max_epochs}")
    if args.unfreeze_layers > 0:
        print(f"  Unfreeze     : top {args.unfreeze_layers} ViT blocks at epoch {args.unfreeze_epoch}")
    if args.backbone_grad_clip > 0:
        print(f"  BB grad clip : {args.backbone_grad_clip}")
    if args.use_ema:
        print(f"  EMA          : decay={args.ema_decay}")
    print(f"  GPUs         : {args.gpus}")
    print("=" * 60)

    # ---- Train ----
    trainer.fit(model, datamodule=datamodule)

    # ---- Test ----
    test_path = AnatomyTemporalDataModule._resolve_data_path(
        os.path.join(args.data_dir, args.test_file)
    )
    if os.path.exists(test_path):
        test_ckpt = None if args.fast_dev_run else "best"
        trainer.test(model, datamodule=datamodule, ckpt_path=test_ckpt)

    # ---- Save final ----
    trainer.save_checkpoint(os.path.join(exp_dir, "final.ckpt"))
    print(f"\nResults saved to: {exp_dir}")


if __name__ == "__main__":
    main()
