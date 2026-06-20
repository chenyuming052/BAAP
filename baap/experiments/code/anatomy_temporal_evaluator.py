"""
BAAP Anatomy-Temporal Fine-tuner Evaluation Script
Location: baap/experiments/code/anatomy_temporal_evaluator.py

Evaluates the finetuned AnatomyTemporalFineTuner model on:
  1. Chest ImaGenome test set — native per-anatomy evaluation
  2. MS-CXR-T benchmark (SVM) — global features + linear SVM (comparable to temporal_test.py)
  3. MS-CXR-T benchmark (ROI) — full anatomy pipeline with aggregated predictions
  4. Chest ImaGenome Gold Dataset — expert-annotated temporal comparisons

Usage:
    export PYTHONPATH=$PWD:${PYTHONPATH:-}
    python baap/experiments/code/anatomy_temporal_evaluator.py \
        --ckpt_path /path/to/finetuned.ckpt \
        --eval_mode mscxrt \
        --data_dir /path/to/chest-imagenome/temporal_finetuning_dataset \
        --scene_graph_dir /path/to/chest-imagenome/scene_graph
"""

import os
import sys
import json
import time
import hashlib
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from pytorch_lightning import Trainer, seed_everything

# ============================================================================
# Project path setup
# ============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # experiments/code/
EXPERIMENTS_DIR = os.path.dirname(CURRENT_DIR)             # experiments/
BAAP_DIR = os.path.dirname(EXPERIMENTS_DIR)               # baap/
PROJECT_ROOT = os.path.dirname(BAAP_DIR)                  # BAAP/
DEFAULT_DATA_ROOT = os.environ.get(
    "BAAP_DATA_DIR", os.path.join(PROJECT_ROOT, "data")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from anatomy_temporal_finetuner import (
    AnatomyTemporalFineTuner,
    AnatomyTemporalDataModule,
    AnatomyTemporalDataset,
    anatomy_collate_fn,
    load_cxr_image,
    resize_img,
    bbox_to_patch_mask,
    roi_pool,
    LABEL_MAP,
    LABEL_NAMES,
    ANATOMY_LIST,
    ANATOMY_TO_IDX,
    ANATOMY_UNK_IDX,
    IMG_SIZE,
    _bbox_224_to_256crop,
)

# Support loading Stage 2 anatomy pre-training checkpoints
try:
    from baap.models.medst.medst_module_anatomy import MedSTAnatomy
except ImportError:
    MedSTAnatomy = None

# Support loading base MedST (Stage 1) checkpoints
try:
    from baap.models.medst.medst_module import MedST as MedSTBase
except ImportError:
    MedSTBase = None


def load_model(ckpt_path: str, model_type: str = "finetuner"):
    """Load model checkpoint with support for multiple model types.

    Args:
        ckpt_path: Path to the checkpoint file.
        model_type: One of 'finetuner' (AnatomyTemporalFineTuner),
                    'anatomy_pretrain' (MedSTAnatomy Stage 2).

    Returns:
        model: Loaded model on CUDA.
        backbone: ImageEncoder reference (for feature extraction).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "finetuner":
        model = AnatomyTemporalFineTuner.load_from_checkpoint(
            ckpt_path, strict=True, pretrained_ckpt=None,
        )
        model = model.to(device).eval()
        backbone = model.backbone
    elif model_type == "anatomy_pretrain":
        if MedSTAnatomy is None:
            raise ImportError("MedSTAnatomy not available. Check import path.")
        model = MedSTAnatomy.load_from_checkpoint(ckpt_path, strict=False)
        model = model.to(device).eval()
        backbone = model.img_encoder_q
    elif model_type == "base_medst":
        if MedSTBase is None:
            raise ImportError("MedST base model not available. Check import path.")
        model = MedSTBase.load_from_checkpoint(ckpt_path, strict=False)
        model = model.to(device).eval()
        backbone = model.img_encoder_q
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"  Model loaded ({model_type}) on {device}", flush=True)
    return model, backbone


def _extract_patch_features(backbone, imgs: torch.Tensor) -> torch.Tensor:
    """Extract ViT patch tokens [B, 196, D] with batch-size==3 guard."""
    batch_size = imgs.shape[0]
    if batch_size == 3:
        imgs = torch.cat([imgs, imgs[:1]], dim=0)
        _, patch_feats = backbone(imgs, view_type="frontal")
        return patch_feats[:batch_size]
    _, patch_feats = backbone(imgs, view_type="frontal")
    return patch_feats


def infer_temporal_logits(model, batch: Dict, model_type: str) -> torch.Tensor:
    """Unified per-anatomy 3-way logits inference for both checkpoint types."""
    if model_type == "finetuner":
        roi_mode = getattr(model.hparams, "roi_mode", "roi")
        if roi_mode == "image_level":
            # image_level model outputs [B, 3] but ROI eval needs [N_anat, 3].
            # Force CLS-based per-anatomy prediction: expand CLS for each anatomy.
            cls_p, _ = model._extract_features(batch["prior_imgs"])
            cls_c, _ = model._extract_features(batch["current_imgs"])
            idx = batch["sample_indices"]
            prior_emb = F.normalize(model.roi_projection(cls_p[idx]), dim=-1)
            current_emb = F.normalize(model.roi_projection(cls_c[idx]), dim=-1)
            if model.hparams.fusion_type == "concat_diff":
                fused = torch.cat([prior_emb, current_emb, current_emb - prior_emb], dim=-1)
            else:
                fused = torch.cat([prior_emb, current_emb], dim=-1)
            if model.hparams.use_anatomy_emb:
                anat_emb = model.anatomy_embedding(batch["anatomy_indices"])
                fused = torch.cat([fused, anat_emb], dim=-1)
            return model.classifier(fused)  # [N_anat, 3]
        return model(batch)

    if model_type != "anatomy_pretrain":
        raise ValueError(f"Unsupported model_type for inference: {model_type}")

    sample_indices = batch["sample_indices"]
    prior_patch = _extract_patch_features(model.img_encoder_q, batch["prior_imgs"])
    current_patch = _extract_patch_features(model.img_encoder_q, batch["current_imgs"])

    prior_roi_raw = roi_pool(prior_patch, batch["prior_bboxes"], sample_indices)
    current_roi_raw = roi_pool(current_patch, batch["current_bboxes"], sample_indices)

    prior_roi = model.anatomy_head.roi_projection(prior_roi_raw)
    current_roi = model.anatomy_head.roi_projection(current_roi_raw)

    num_comparisons = batch.get("num_comparisons")
    if num_comparisons is None:
        batch_size = batch["prior_imgs"].shape[0]
        if sample_indices.numel() == 0:
            num_comparisons = torch.zeros(
                batch_size, device=sample_indices.device, dtype=torch.long
            )
        else:
            num_comparisons = torch.bincount(sample_indices, minlength=batch_size)

    if getattr(model.anatomy_head, "use_crda", False):
        change_emb = model.anatomy_head.crda(
            current_roi, prior_roi, sample_indices, num_comparisons
        )
    else:
        change_emb = current_roi - prior_roi

    cls_input = torch.cat([prior_roi, current_roi, change_emb], dim=-1)
    logits = model.anatomy_head.classifier(cls_input)
    return logits

# ============================================================================
# Constants
# ============================================================================
DISEASES = ["consolidation", "edema", "pleural_effusion", "pneumonia", "pneumothorax"]

DISEASE_ANATOMY_MAP = {
    "consolidation": [
        "right lung", "left lung",
        "right lower lung zone", "left lower lung zone",
        "right mid lung zone", "left mid lung zone",
        "right upper lung zone", "left upper lung zone",
    ],
    "edema": [
        "right lung", "left lung",
        "right lower lung zone", "left lower lung zone",
        "right mid lung zone", "left mid lung zone",
        "right costophrenic angle", "left costophrenic angle",
    ],
    "pleural_effusion": [
        "right costophrenic angle", "left costophrenic angle",
        "right lower lung zone", "left lower lung zone",
        "right lung", "left lung",
    ],
    "pneumonia": [
        "right lung", "left lung",
        "right lower lung zone", "left lower lung zone",
        "right mid lung zone", "left mid lung zone",
        "right upper lung zone", "left upper lung zone",
    ],
    "pneumothorax": [
        "right apical zone", "left apical zone",
        "right upper lung zone", "left upper lung zone",
        "right lung", "left lung",
    ],
}

# MS-CXR-T label mapping: improving=0, stable=1, worsening=2
MSCXRT_LABEL_MAP = {"improving": 0, "stable": 1, "worsening": 2}

# Anatomy finetuner label → MS-CXR-T label mapping
# improved(0)→improving(0), no_change(1)→stable(1), worsened(2)→worsening(2)
# (labels happen to align numerically)


# ============================================================================
# Evaluation 1: Chest ImaGenome Test Set
# ============================================================================
def eval_imagenome(
    model,
    model_type: str,
    data_dir: str,
    test_file: str,
    batch_size: int,
    num_workers: int,
) -> Dict:
    """Evaluate finetuned model on ImaGenome test set with detailed per-anatomy metrics."""
    print("\n" + "=" * 60)
    print("  Evaluation 1: Chest ImaGenome Test Set")
    print("=" * 60)

    # Set up data module for test only
    datamodule = AnatomyTemporalDataModule(
        data_dir=data_dir,
        test_file=test_file,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    datamodule.setup(stage="test")

    if not hasattr(datamodule, "test_dataset"):
        print("  WARNING: No test dataset found. Skipping ImaGenome evaluation.")
        return {}

    test_loader = datamodule.test_dataloader()
    if test_loader is None:
        print("  WARNING: test_dataloader() returned None. Skipping.")
        return {}

    # Run built-in trainer.test() only for finetuner checkpoints.
    # anatomy_pretrain checkpoints do not implement Lightning test hooks.
    builtin_metrics = {}
    if model_type == "finetuner":
        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False,
            enable_progress_bar=True,
        )
        test_results = trainer.test(model, dataloaders=test_loader)
        builtin_metrics = test_results[0] if test_results else {}
    else:
        print("  Skip trainer.test(): anatomy_pretrain checkpoint has no test_step.")

    # Detailed per-anatomy analysis
    model.eval()
    device = next(model.parameters()).device

    all_preds = []
    all_labels = []
    all_anatomies = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="ImaGenome detailed eval"):
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = infer_temporal_logits(model, batch_dev, model_type=model_type)
            preds = logits.argmax(dim=-1).cpu()
            labels = batch["labels"]
            anatomies = batch["anatomies"]

            all_preds.append(preds)
            all_labels.append(labels)
            all_anatomies.extend(anatomies)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Overall metrics
    overall_acc = (all_preds == all_labels).mean()

    # Per-class metrics
    per_class_f1 = {}
    per_class_precision = {}
    per_class_recall = {}
    for cls_idx, cls_name in enumerate(LABEL_NAMES):
        tp = ((all_preds == cls_idx) & (all_labels == cls_idx)).sum()
        fp = ((all_preds == cls_idx) & (all_labels != cls_idx)).sum()
        fn = ((all_preds != cls_idx) & (all_labels == cls_idx)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        per_class_f1[cls_name] = round(float(f1), 4)
        per_class_precision[cls_name] = round(float(precision), 4)
        per_class_recall[cls_name] = round(float(recall), 4)

    f1_macro = np.mean(list(per_class_f1.values()))

    # Confusion matrix (rows=true, cols=pred)
    n_classes = len(LABEL_NAMES)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        confusion[t, p] += 1

    # Per-anatomy breakdown
    per_anatomy = {}
    anatomy_arr = np.array(all_anatomies)
    unique_anatomies = np.unique(anatomy_arr)
    for anat in unique_anatomies:
        mask = anatomy_arr == anat
        anat_preds = all_preds[mask]
        anat_labels = all_labels[mask]
        anat_acc = (anat_preds == anat_labels).mean()
        # Per-anatomy macro F1
        f1s = []
        for cls_idx in range(n_classes):
            tp = ((anat_preds == cls_idx) & (anat_labels == cls_idx)).sum()
            fp = ((anat_preds == cls_idx) & (anat_labels != cls_idx)).sum()
            fn = ((anat_preds != cls_idx) & (anat_labels == cls_idx)).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            f1s.append(float(f1))
        per_anatomy[anat] = {
            "acc": round(float(anat_acc), 4),
            "f1_macro": round(float(np.mean(f1s)), 4),
            "count": int(mask.sum()),
        }

    # Sort by count descending
    per_anatomy = dict(sorted(per_anatomy.items(), key=lambda x: -x[1]["count"]))

    result = {
        "test_acc": round(float(overall_acc), 4),
        "test_f1_macro": round(float(f1_macro), 4),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "confusion_matrix": confusion.tolist(),
        "confusion_labels": LABEL_NAMES,
        "per_anatomy": per_anatomy,
        "total_comparisons": len(all_preds),
        "builtin_metrics": {k: round(float(v), 4) for k, v in builtin_metrics.items()},
    }

    print(f"\n  Overall accuracy: {overall_acc:.4f}")
    print(f"  Macro F1:         {f1_macro:.4f}")
    print(f"  Per-class F1:     {per_class_f1}")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"    Labels: {LABEL_NAMES}")
    for row in confusion:
        print(f"    {row.tolist()}")
    print(f"  Top-5 anatomies by count:")
    for anat, info in list(per_anatomy.items())[:5]:
        print(f"    {anat}: acc={info['acc']:.4f}, f1={info['f1_macro']:.4f}, n={info['count']}")

    return result


# ============================================================================
# Evaluation 2a: MS-CXR-T Global Features + SVM
# ============================================================================
def eval_mscxrt_svm(
    model,
    mscxrt_csv: str,
    mimic_cxr_dir: str,
    batch_size: int,
    num_workers: int,
    svm_seeds: List[int] = None,
    no_shuffle_folds: bool = False,
    backbone=None,
    svm_feature_mode: str = "original",
) -> Dict:
    """Evaluate global CLS→global_embed features with SVM 10-fold CV.

    Args:
        svm_feature_mode: 'original' = concat(prior, current) [2*D];
                          'enhanced' = concat(prior, current, diff) [3*D].
    """
    print("\n" + "=" * 60)
    print(f"  Evaluation 2a: MS-CXR-T Global Features + SVM 10-fold CV (mode={svm_feature_mode})")
    print("=" * 60)

    img_df = pd.read_csv(mscxrt_csv)

    # Validation transform matching BAAP eval pipeline
    val_transform = transforms.Compose([
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    model.eval()
    device = next(model.parameters()).device
    if backbone is None:
        backbone = getattr(model, "backbone", getattr(model, "img_encoder_q", None))

    def extract_global_features(img_paths: List[str]) -> np.ndarray:
        """Extract global_embed features for a list of images, return [N, emb_dim] numpy."""
        all_feats = []
        n_batches = (len(img_paths) + batch_size - 1) // batch_size
        for bi, i in enumerate(range(0, len(img_paths), batch_size)):
            if bi % 10 == 0:
                print(f"    Extracting features: batch {bi+1}/{n_batches}", end="\r", flush=True)
            batch_paths = img_paths[i : i + batch_size]
            imgs = []
            for p in batch_paths:
                img = load_cxr_image(p, scale=256)
                img = val_transform(img)
                imgs.append(img)
            imgs_tensor = torch.stack(imgs).to(device)

            with torch.no_grad():
                B = imgs_tensor.shape[0]
                # Handle the batch_size==3 edge case
                if B == 3:
                    imgs_padded = torch.cat([imgs_tensor, imgs_tensor[:1]], dim=0)
                    cls_token, _ = backbone(imgs_padded, view_type="frontal")
                    cls_token = cls_token[:B]
                else:
                    cls_token, _ = backbone(imgs_tensor, view_type="frontal")

                emb = backbone.global_embed(cls_token)
                emb = F.normalize(emb, dim=-1)
            all_feats.append(emb.cpu().numpy())
        print(f"    Extracting features: done ({len(img_paths)} images)     ", flush=True)
        return np.concatenate(all_feats, axis=0)

    def run_svm_for_disease(disease: str, df: pd.DataFrame, cv: int, seed: int) -> float:
        """Run SVM cross-validation for one disease, return mean accuracy * 100."""
        label_col = f"{disease}_progression"
        sub_df = df[["dicom_id", "previous_dicom_id", "study_id", "subject_id", label_col]].dropna(
            subset=[label_col]
        )
        if len(sub_df) == 0:
            return 0.0

        # Labels: improving=0, stable=1, else=2
        def classify_value(x):
            if x == "improving":
                return 0
            elif x == "stable":
                return 1
            else:
                return 2

        y = sub_df[label_col].apply(classify_value).values

        # Collect image paths (current, previous interleaved)
        all_paths = []
        for _, row in sub_df.iterrows():
            all_paths.append(os.path.join(mimic_cxr_dir, row["dicom_id"] + ".jpg"))
            all_paths.append(os.path.join(mimic_cxr_dir, row["previous_dicom_id"] + ".jpg"))

        # Extract features
        feats = extract_global_features(all_paths)  # [2N, emb_dim]
        emb_dim = feats.shape[1]
        if svm_feature_mode == "enhanced":
            # Paths are interleaved as [current_0, previous_0, current_1, previous_1, ...]
            current_feats = feats[0::2]   # [N, emb_dim]
            previous_feats = feats[1::2]  # [N, emb_dim]
            diff_feats = current_feats - previous_feats  # temporal change direction
            feats = np.concatenate([current_feats, previous_feats, diff_feats], axis=1)  # [N, 3*emb_dim]
        else:
            feats = feats.reshape(-1, 2, emb_dim).reshape(-1, 2 * emb_dim)  # [N, 2*emb_dim]

        classifier = SVC(kernel="linear", random_state=seed)
        scores = cross_val_score(classifier, feats, y, cv=cv)
        mean_acc = round(scores.mean(), 4) * 100
        print(f"    {disease} (n={len(y)}, cv={cv}): {mean_acc:.2f}%")
        return mean_acc

    # Multi-seed evaluation (matching temporal_test.py)
    random_seeds = svm_seeds or [50, 52, 100]
    all_cv10 = []
    print(f"  Seeds: {random_seeds}, shuffle_folds: {not no_shuffle_folds}")

    for seed in random_seeds:
        print(f"\n  Seed {seed}:")
        shuffled_df = img_df.sample(frac=1, random_state=seed).reset_index(drop=True)

        cv = 10
        results = []
        for disease in DISEASES:
            acc = run_svm_for_disease(disease, shuffled_df, cv, seed)
            results.append(acc)
        avg = np.mean(results)
        results.append(avg)
        print(f"    SVM 10-fold CV avg: {avg:.2f}%")
        all_cv10.append(results)

    # Aggregate across seeds
    disease_keys = DISEASES + ["avg"]
    avg_cv10 = np.mean(all_cv10, axis=0)

    result = {
        "svm_10fold_cv_accuracy": {
            k: round(float(v), 2) for k, v in zip(disease_keys, avg_cv10)
        },
    }

    print(f"\n  SVM 10-fold CV accuracy: {result['svm_10fold_cv_accuracy']}")

    return result


# ============================================================================
# Evaluation 2a+: MS-CXR-T Direct Classification (global_classifier)
# ============================================================================
def eval_mscxrt_direct(
    model,
    mscxrt_csv: str,
    mimic_cxr_dir: str,
    batch_size: int,
    backbone=None,
    svm_seeds: List[int] = None,
    no_shuffle_folds: bool = False,
) -> Dict:
    """Evaluate image-level classifier logits using SVM 10-fold CV."""
    print("\n" + "=" * 60)
    print("  Evaluation: MS-CXR-T Image-level Logits + SVM 10-fold CV")
    print("=" * 60)

    img_df = pd.read_csv(mscxrt_csv)

    val_transform = transforms.Compose([
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    model.eval()
    device = next(model.parameters()).device
    if backbone is None:
        backbone = getattr(model, "backbone", None)

    global_cls = getattr(model, "global_classifier", None)
    if global_cls is None:
        print("  ERROR: model has no global_classifier. Skipping.")
        return {}

    def _extract_cls(imgs_tensor: torch.Tensor) -> torch.Tensor:
        B = imgs_tensor.shape[0]
        if B == 3:
            imgs_tensor = torch.cat([imgs_tensor, imgs_tensor[:1]], dim=0)
            cls_feat, _ = backbone(imgs_tensor, view_type="frontal")
            return cls_feat[:B]
        cls_feat, _ = backbone(imgs_tensor, view_type="frontal")
        return cls_feat

    # ---- Step 1: Collect logits per disease ----
    print("\n  Collecting image-level logits for each disease...")
    disease_data = {}  # disease -> {"logits": [N,3], "labels": [N]}

    for disease in DISEASES:
        label_col = f"{disease}_progression"
        sub_df = img_df[["dicom_id", "previous_dicom_id", label_col]].dropna(subset=[label_col])
        if len(sub_df) == 0:
            disease_data[disease] = {"logits": np.array([]), "labels": np.array([])}
            continue

        labels = sub_df[label_col].map(MSCXRT_LABEL_MAP).values
        all_logits = []

        for i in range(0, len(sub_df), batch_size):
            batch_df = sub_df.iloc[i : i + batch_size]
            prior_imgs = []
            current_imgs = []
            for _, row in batch_df.iterrows():
                prior_path = os.path.join(mimic_cxr_dir, row["previous_dicom_id"] + ".jpg")
                current_path = os.path.join(mimic_cxr_dir, row["dicom_id"] + ".jpg")
                prior_imgs.append(val_transform(load_cxr_image(prior_path, scale=256)))
                current_imgs.append(val_transform(load_cxr_image(current_path, scale=256)))

            prior_tensor = torch.stack(prior_imgs).to(device)
            current_tensor = torch.stack(current_imgs).to(device)

            with torch.no_grad():
                prior_cls = _extract_cls(prior_tensor)
                current_cls = _extract_cls(current_tensor)
                fused = torch.cat([prior_cls, current_cls], dim=-1)
                logits = global_cls(fused)
                all_logits.append(logits.cpu().numpy())

        all_logits = np.concatenate(all_logits)
        disease_data[disease] = {"logits": all_logits, "labels": labels}
        print(f"    {disease}: collected {len(labels)} pairs")

    # ---- Step 2: SVM 10-fold CV on logits ----
    random_seeds = svm_seeds or [50, 52, 100]
    all_cv10 = []
    print(f"\n  Running SVM 10-fold CV on logits (seeds={random_seeds})...")

    for seed in random_seeds:
        print(f"\n  Seed {seed}:")
        results = []
        cv = 10
        for disease in DISEASES:
            data = disease_data[disease]
            if len(data["labels"]) < cv:
                results.append(0.0)
                continue
            classifier = SVC(kernel="linear", random_state=seed)
            if no_shuffle_folds:
                skf = StratifiedKFold(n_splits=cv, shuffle=False)
            else:
                skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
            scores = cross_val_score(classifier, data["logits"], data["labels"], cv=skf)
            acc = round(scores.mean(), 4) * 100
            results.append(acc)
            print(f"    {disease} SVM 10-fold CV: {acc:.2f}%")
        avg = np.mean(results)
        results.append(avg)
        print(f"    SVM 10-fold CV avg: {avg:.2f}%")
        all_cv10.append(results)

    disease_keys = DISEASES + ["avg"]
    result = {
        "svm_10fold_cv_accuracy": {
            k: round(float(v), 2)
            for k, v in zip(disease_keys, np.mean(all_cv10, axis=0))
        },
    }

    print(f"\n  SVM 10-fold CV accuracy: {result['svm_10fold_cv_accuracy']}")

    return result


# ============================================================================
# Evaluation 2b: MS-CXR-T ROI-based Prediction
# ============================================================================
def load_scene_graph(scene_graph_dir: str, image_id: str) -> Optional[Dict]:
    """Load a Chest ImaGenome scene graph for a given dicom image_id.

    The image_id is the 44-char DICOM UID (e.g., 3bea0373-0d10dd77-...).
    Scene graph file: {image_id}_SceneGraph.json
    """
    sg_path = os.path.join(scene_graph_dir, f"{image_id}_SceneGraph.json")
    if not os.path.exists(sg_path):
        return None
    with open(sg_path, "r") as f:
        return json.load(f)


def extract_anatomy_bboxes(scene_graph: Dict, coord_mode: str = "crop224") -> Dict[str, List[float]]:
    """Extract anatomy name → bbox coords from a scene graph.

    Scene graph bboxes are in ImaGenome 224-space.  By default, we convert
    them to the BAAP resize-256 + CenterCrop(224) coordinate space via
    ``_bbox_224_to_256crop`` so that ROI pooling is consistent with training.
    Use ``coord_mode="raw224"`` to reproduce older paper-protocol ROI
    evaluation that used the scene graph coordinates directly.

    Returns dict mapping anatomy name (lowercase) to [x1, y1, x2, y2] in
    the selected coordinate space.
    """
    if coord_mode not in {"crop224", "raw224"}:
        raise ValueError(f"Unsupported bbox coord_mode: {coord_mode}")

    bboxes = {}
    for obj in scene_graph.get("objects", []):
        anat_name = obj.get("bbox_name", "").lower().strip()
        if not anat_name:
            continue
        x1, y1, x2, y2 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]
        bbox = [x1, y1, x2, y2]
        if coord_mode == "crop224":
            bbox = _bbox_224_to_256crop(bbox)
        bboxes[anat_name] = bbox
    return bboxes


def _make_image_noise_seed(base_seed: int, image_id: str) -> int:
    """Deterministic per-image seed derived from a stable md5 hash of the image id."""
    h = hashlib.md5(image_id.encode("utf-8")).hexdigest()
    return (int(base_seed) + int(h[:8], 16)) & 0x7FFFFFFF


def perturb_bbox_dict(
    bboxes: Dict[str, List[float]],
    noise_frac: float,
    rng_seed: int,
    img_bound: float = 224.0,
    min_size: float = 2.0,
) -> Dict[str, List[float]]:
    """Apply center + scale noise to each bbox to simulate a noisy detector.

    For each [x1, y1, x2, y2]:
      - Center is jittered by Gaussian noise with std = ``noise_frac * box_size``.
      - Width / height are scaled by independent uniform factors in
        ``[1 - noise_frac, 1 + noise_frac]``.
      - Resulting coordinates are clipped to ``[0, img_bound]``.
      - Degenerate boxes are widened to ``min_size``.

    A no-op when ``noise_frac <= 0``.
    """
    if noise_frac <= 0.0 or not bboxes:
        return bboxes

    rng = np.random.RandomState(rng_seed)
    out: Dict[str, List[float]] = {}
    for name, box in bboxes.items():
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        cx += rng.normal(0.0, noise_frac * w)
        cy += rng.normal(0.0, noise_frac * h)
        sw = rng.uniform(1.0 - noise_frac, 1.0 + noise_frac)
        sh = rng.uniform(1.0 - noise_frac, 1.0 + noise_frac)
        w_new = min(img_bound, max(min_size, w * sw))
        h_new = min(img_bound, max(min_size, h * sh))
        cx = float(np.clip(cx, w_new / 2.0, img_bound - w_new / 2.0))
        cy = float(np.clip(cy, h_new / 2.0, img_bound - h_new / 2.0))
        nx1 = cx - w_new / 2.0
        ny1 = cy - h_new / 2.0
        nx2 = cx + w_new / 2.0
        ny2 = cy + h_new / 2.0
        out[name] = [nx1, ny1, nx2, ny2]
    return out


def get_dicom_id_from_path(rel_path: str) -> str:
    """Extract the 44-char DICOM UID from a relative path like p10/p100/s557/3bea0373-...."""
    basename = os.path.basename(rel_path)
    # Remove .jpg extension if present
    if basename.endswith(".jpg"):
        basename = basename[:-4]
    return basename


def eval_mscxrt_roi(
    model,
    model_type: str,
    mscxrt_csv: str,
    mimic_cxr_dir: str,
    scene_graph_dir: str,
    batch_size: int,
    num_workers: int,
    svm_seeds: List[int] = None,
    no_shuffle_folds: bool = False,
    roi_aggregation: str = "majority",
    bbox_coord_mode: str = "crop224",
    bbox_noise_frac: float = 0.0,
    bbox_noise_seed: int = 42,
) -> Dict:
    """Evaluate using full ROI pipeline on MS-CXR-T, aggregating per-anatomy predictions.

    Args:
        roi_aggregation: 'majority' = hard vote; 'softmax' = softmax-weighted mean.
    """
    print("\n" + "=" * 60)
    print(f"  Evaluation 2b: MS-CXR-T ROI-based Prediction (aggregation={roi_aggregation})")
    print(f"  BBox coordinate mode: {bbox_coord_mode}")
    if bbox_noise_frac > 0.0:
        print(f"  [E5] BBox noise injection ENABLED: frac={bbox_noise_frac}, seed={bbox_noise_seed}")
    print("=" * 60)

    img_df = pd.read_csv(mscxrt_csv)

    val_transform = transforms.Compose([
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    model.eval()
    device = next(model.parameters()).device

    # Pre-check scene graph availability
    all_current_ids = img_df["dicom_id"].apply(get_dicom_id_from_path).unique()
    all_previous_ids = img_df["previous_dicom_id"].apply(get_dicom_id_from_path).unique()
    all_ids = set(all_current_ids) | set(all_previous_ids)
    found = sum(
        1 for uid in all_ids
        if os.path.exists(os.path.join(scene_graph_dir, f"{uid}_SceneGraph.json"))
    )
    print(f"  Scene graph coverage: {found}/{len(all_ids)} ({100 * found / len(all_ids):.1f}%)")

    def predict_anatomies_for_pair(
        current_path: str,
        previous_path: str,
        disease: str,
    ) -> Optional[Dict]:
        """Return aggregated ROI logits for one image pair and disease."""
        current_id = get_dicom_id_from_path(current_path)
        previous_id = get_dicom_id_from_path(previous_path)

        sg_current = load_scene_graph(scene_graph_dir, current_id)
        sg_previous = load_scene_graph(scene_graph_dir, previous_id)

        if sg_current is None or sg_previous is None:
            return None

        bboxes_current = extract_anatomy_bboxes(sg_current, coord_mode=bbox_coord_mode)
        bboxes_previous = extract_anatomy_bboxes(sg_previous, coord_mode=bbox_coord_mode)

        if bbox_noise_frac > 0.0:
            bboxes_current = perturb_bbox_dict(
                bboxes_current,
                bbox_noise_frac,
                _make_image_noise_seed(bbox_noise_seed, current_id),
            )
            bboxes_previous = perturb_bbox_dict(
                bboxes_previous,
                bbox_noise_frac,
                _make_image_noise_seed(bbox_noise_seed, previous_id),
            )

        # Find relevant anatomies that exist in both scene graphs
        relevant = DISEASE_ANATOMY_MAP.get(disease, [])
        common = [a for a in relevant if a in bboxes_current and a in bboxes_previous]

        if not common:
            return None

        # Load images
        try:
            current_img = load_cxr_image(
                os.path.join(mimic_cxr_dir, current_path + ".jpg"), scale=256
            )
            previous_img = load_cxr_image(
                os.path.join(mimic_cxr_dir, previous_path + ".jpg"), scale=256
            )
        except FileNotFoundError:
            return None

        current_img = val_transform(current_img)
        previous_img = val_transform(previous_img)

        # Build batch dict for model forward
        prior_bboxes = []
        current_bboxes_list = []
        anatomy_indices = []

        for anat in common:
            prior_bboxes.append(torch.tensor(bboxes_previous[anat], dtype=torch.float32))
            current_bboxes_list.append(torch.tensor(bboxes_current[anat], dtype=torch.float32))
            anatomy_indices.append(ANATOMY_TO_IDX.get(anat, ANATOMY_UNK_IDX))

        batch = {
            "prior_imgs": previous_img.unsqueeze(0).to(device),       # [1, 3, 224, 224]
            "current_imgs": current_img.unsqueeze(0).to(device),      # [1, 3, 224, 224]
            "prior_bboxes": torch.stack(prior_bboxes).to(device),     # [N_anat, 4]
            "current_bboxes": torch.stack(current_bboxes_list).to(device),
            "sample_indices": torch.zeros(len(common), dtype=torch.long).to(device),
            "anatomy_indices": torch.tensor(anatomy_indices, dtype=torch.long).to(device),
        }

        with torch.no_grad():
            logits = infer_temporal_logits(model, batch, model_type=model_type)  # [N_anat, 3]

        logits_np = logits.cpu().numpy()

        # Mean logits for SVM-style evaluation
        aggregated_logits = logits_np.mean(axis=0).tolist()

        return {
            "aggregated_logits": aggregated_logits,
        }

    def evaluate_disease_roi(
        disease: str,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect aggregated ROI logits and labels for one disease."""
        label_col = f"{disease}_progression"
        sub_df = df[["dicom_id", "previous_dicom_id", label_col]].dropna(subset=[label_col])

        if len(sub_df) == 0:
            return np.array([]), np.array([])

        def classify_value(x):
            if x == "improving":
                return 0
            elif x == "stable":
                return 1
            else:
                return 2

        all_logits = []
        all_labels = []
        skipped = 0

        for _, row in tqdm(sub_df.iterrows(), total=len(sub_df), desc=f"  ROI {disease}"):
            label = classify_value(row[label_col])
            result = predict_anatomies_for_pair(
                row["dicom_id"], row["previous_dicom_id"], disease
            )
            if result is None:
                skipped += 1
                continue
            all_logits.append(result["aggregated_logits"])
            all_labels.append(label)

        if not all_labels:
            print(f"    {disease}: no valid pairs (skipped {skipped})")
            return np.array([]), np.array([])

        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)

        print(f"    {disease}: collected {len(all_labels)} pairs (skipped={skipped})")

        return all_logits, all_labels

    # Run evaluation with multi-seed SVM 10-fold CV on aggregated logits.
    random_seeds = svm_seeds or [50, 52, 100]
    all_cv10 = []
    print(f"  Seeds: {random_seeds}, shuffle_folds: {not no_shuffle_folds}")

    # Run ROI predictions once (seed-independent), then vary SVM seed
    print("\n  Collecting ROI logits for each disease...")
    disease_data = {}
    for disease in DISEASES:
        logits, labels = evaluate_disease_roi(disease, img_df)
        disease_data[disease] = {
            "logits": logits,
            "labels": labels,
        }

    # SVM on aggregated logits
    print("\n  Running SVM 10-fold CV on aggregated logits...")
    for seed in random_seeds:
        print(f"\n  Seed {seed}:")
        results = []
        cv = 10
        for disease in DISEASES:
            data = disease_data[disease]
            if len(data["labels"]) < cv:
                results.append(0.0)
                continue
            classifier = SVC(kernel="linear", random_state=seed)
            if no_shuffle_folds:
                skf = StratifiedKFold(n_splits=cv, shuffle=False)
            else:
                skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
            scores = cross_val_score(classifier, data["logits"], data["labels"], cv=skf)
            acc = round(scores.mean(), 4) * 100
            results.append(acc)
            print(f"    {disease} SVM 10-fold CV: {acc:.2f}%")
        avg = np.mean(results)
        results.append(avg)
        print(f"    SVM 10-fold CV avg: {avg:.2f}%")
        all_cv10.append(results)

    disease_keys = DISEASES + ["avg"]
    avg_cv10 = np.mean(all_cv10, axis=0)
    result = {
        "svm_10fold_cv_accuracy": {
            k: round(float(v), 2) for k, v in zip(disease_keys, avg_cv10)
        },
    }

    print(f"\n  SVM 10-fold CV accuracy: {result['svm_10fold_cv_accuracy']}")

    return result


# ============================================================================
# Evaluation 3: MS-CXR-T Ensemble (backbone global_embed + ROI logits → SVM)
# ============================================================================
def eval_mscxrt_ensemble(
    model,
    model_type: str,
    mscxrt_csv: str,
    mimic_cxr_dir: str,
    scene_graph_dir: str,
    batch_size: int,
    num_workers: int,
    svm_seeds: List[int] = None,
    no_shuffle_folds: bool = False,
    roi_aggregation: str = "softmax",
    backbone=None,
) -> Dict:
    """Ensemble evaluation: concat backbone global_embed features with ROI logits, then SVM.

    For each image pair:
      - backbone global_embed: concat(current, prior, diff) → [3 * emb_dim]
      - ROI pipeline: aggregated logits → [3]
      - Ensemble feature: concat both → [3 * emb_dim + 3]
      - Feed to linear SVM with cross-validation
    """
    print("\n" + "=" * 60)
    print("  Evaluation 3: MS-CXR-T Ensemble (global_embed + ROI logits)")
    print("=" * 60)

    img_df = pd.read_csv(mscxrt_csv)

    val_transform = transforms.Compose([
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    model.eval()
    device = next(model.parameters()).device
    if backbone is None:
        backbone = getattr(model, "backbone", getattr(model, "img_encoder_q", None))

    # Check scene graph coverage
    all_current_ids = img_df["dicom_id"].apply(get_dicom_id_from_path).unique()
    all_previous_ids = img_df["previous_dicom_id"].apply(get_dicom_id_from_path).unique()
    all_ids = set(all_current_ids) | set(all_previous_ids)
    found = sum(
        1 for uid in all_ids
        if os.path.exists(os.path.join(scene_graph_dir, f"{uid}_SceneGraph.json"))
    )
    print(f"  Scene graph coverage: {found}/{len(all_ids)} ({100 * found / len(all_ids):.1f}%)")

    def extract_ensemble_features_for_disease(disease: str, df: pd.DataFrame):
        """Extract ensemble features (global_embed + ROI logits) for one disease.

        Returns:
            X: [N_valid, 3*emb_dim + 3] ensemble features
            y: [N_valid] labels
        """
        label_col = f"{disease}_progression"
        sub_df = df[["dicom_id", "previous_dicom_id", label_col]].dropna(subset=[label_col])

        if len(sub_df) == 0:
            return np.array([]), np.array([])

        def classify_value(x):
            if x == "improving":
                return 0
            elif x == "stable":
                return 1
            else:
                return 2

        all_features = []
        all_labels = []
        skipped = 0

        relevant_anatomies = DISEASE_ANATOMY_MAP.get(disease, [])

        for _, row in tqdm(sub_df.iterrows(), total=len(sub_df), desc=f"  Ensemble {disease}"):
            label = classify_value(row[label_col])

            current_path = os.path.join(mimic_cxr_dir, row["dicom_id"] + ".jpg")
            previous_path = os.path.join(mimic_cxr_dir, row["previous_dicom_id"] + ".jpg")

            # Load scene graphs for ROI
            current_id = get_dicom_id_from_path(row["dicom_id"])
            previous_id = get_dicom_id_from_path(row["previous_dicom_id"])
            sg_current = load_scene_graph(scene_graph_dir, current_id)
            sg_previous = load_scene_graph(scene_graph_dir, previous_id)

            if sg_current is None or sg_previous is None:
                skipped += 1
                continue

            bboxes_current = extract_anatomy_bboxes(sg_current)
            bboxes_previous = extract_anatomy_bboxes(sg_previous)
            common = [a for a in relevant_anatomies if a in bboxes_current and a in bboxes_previous]

            if not common:
                skipped += 1
                continue

            # Load images
            try:
                current_img = load_cxr_image(current_path, scale=256)
                previous_img = load_cxr_image(previous_path, scale=256)
            except FileNotFoundError:
                skipped += 1
                continue

            current_img_t = val_transform(current_img)
            previous_img_t = val_transform(previous_img)

            with torch.no_grad():
                # --- Part 1: backbone global_embed features ---
                imgs = torch.stack([current_img_t, previous_img_t]).to(device)  # [2, 3, 224, 224]
                cls_tokens, _ = backbone(imgs, view_type="frontal")  # [2, 768]
                global_embs = backbone.global_embed(cls_tokens)  # [2, emb_dim]
                global_embs = F.normalize(global_embs, dim=-1).cpu().numpy()
                current_emb = global_embs[0]   # [emb_dim]
                previous_emb = global_embs[1]  # [emb_dim]
                diff_emb = current_emb - previous_emb
                global_feat = np.concatenate([current_emb, previous_emb, diff_emb])  # [3*emb_dim]

                # --- Part 2: ROI logits ---
                prior_bboxes = []
                current_bboxes_list = []
                anatomy_indices = []
                for anat in common:
                    prior_bboxes.append(torch.tensor(bboxes_previous[anat], dtype=torch.float32))
                    current_bboxes_list.append(torch.tensor(bboxes_current[anat], dtype=torch.float32))
                    anatomy_indices.append(ANATOMY_TO_IDX.get(anat, ANATOMY_UNK_IDX))

                batch = {
                    "prior_imgs": previous_img_t.unsqueeze(0).to(device),
                    "current_imgs": current_img_t.unsqueeze(0).to(device),
                    "prior_bboxes": torch.stack(prior_bboxes).to(device),
                    "current_bboxes": torch.stack(current_bboxes_list).to(device),
                    "sample_indices": torch.zeros(len(common), dtype=torch.long).to(device),
                    "anatomy_indices": torch.tensor(anatomy_indices, dtype=torch.long).to(device),
                }
                logits = infer_temporal_logits(model, batch, model_type=model_type)  # [N_anat, 3]

                # Aggregate ROI logits
                if roi_aggregation == "softmax":
                    probs = F.softmax(logits, dim=-1).cpu().numpy()
                    aggregated_logits = probs.mean(axis=0)  # [3]
                else:
                    aggregated_logits = logits.cpu().numpy().mean(axis=0)  # [3]

            # --- Concat ensemble feature ---
            ensemble_feat = np.concatenate([global_feat, aggregated_logits])  # [3*emb_dim + 3]
            all_features.append(ensemble_feat)
            all_labels.append(label)

        if not all_labels:
            print(f"    {disease}: no valid pairs (skipped {skipped})")
            return np.array([]), np.array([])

        X = np.array(all_features)
        y = np.array(all_labels)
        print(f"    {disease}: {len(y)} valid pairs (skipped {skipped}), feat_dim={X.shape[1]}")
        return X, y

    # Extract features for all diseases
    print("\n  Extracting ensemble features for each disease...")
    disease_data = {}
    for disease in DISEASES:
        X, y = extract_ensemble_features_for_disease(disease, img_df)
        disease_data[disease] = {"X": X, "y": y}

    # Run SVM cross-validation
    random_seeds = svm_seeds or [50, 52, 100]
    all_cv10 = []
    print(f"\n  Running SVM 10-fold CV on ensemble features...")
    print(f"  Seeds: {random_seeds}, shuffle_folds: {not no_shuffle_folds}")

    for seed in random_seeds:
        print(f"\n  Seed {seed}:")
        cv = 10
        results = []
        for disease in DISEASES:
            data = disease_data[disease]
            if len(data["y"]) < cv:
                results.append(0.0)
                continue
            classifier = SVC(kernel="linear", random_state=seed)
            if no_shuffle_folds:
                skf = StratifiedKFold(n_splits=cv, shuffle=False)
            else:
                skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
            scores = cross_val_score(classifier, data["X"], data["y"], cv=skf)
            acc = round(scores.mean(), 4) * 100
            results.append(acc)
            print(f"    {disease} SVM 10-fold CV: {acc:.2f}%")
        avg = np.mean(results)
        results.append(avg)
        print(f"    SVM 10-fold CV avg: {avg:.2f}%")
        all_cv10.append(results)

    disease_keys = DISEASES + ["avg"]
    avg_cv10 = np.mean(all_cv10, axis=0)
    result = {
        "svm_10fold_cv_accuracy": {
            k: round(float(v), 2) for k, v in zip(disease_keys, avg_cv10)
        },
    }

    print(f"\n  Ensemble SVM 10-fold CV accuracy: {result['svm_10fold_cv_accuracy']}")

    return result


# ============================================================================
# Evaluation 4: Chest ImaGenome Gold Dataset — Expert-Annotated Temporal
# ============================================================================

def _build_dicom_path_lookup(mimic_metadata_csv: str, mimic_cxr_dir: str) -> Dict[str, str]:
    """Build dicom_id → full JPEG path mapping from MIMIC-CXR metadata.

    Args:
        mimic_metadata_csv: Path to mimic-cxr-2.0.0-metadata.csv.
        mimic_cxr_dir: Root of MIMIC-CXR-JPG files/ directory.

    Returns:
        Dict mapping 44-char dicom_id → absolute JPEG path.
    """
    meta = pd.read_csv(mimic_metadata_csv, usecols=["dicom_id", "subject_id", "study_id"])
    lookup = {}
    for _, row in meta.iterrows():
        sid = int(row["subject_id"])
        did = str(row["dicom_id"])
        study = int(row["study_id"])
        path = os.path.join(
            mimic_cxr_dir,
            f"p{sid // 1000000}/p{sid}/s{study}/{did}.jpg",
        )
        lookup[did] = path
    return lookup


def _gold_label_name_to_disease(label_name: str) -> str:
    """Map Chest ImaGenome gold `label_name` to coarse disease bucket.

    The mapping is keyword-based and conservative, used only for supplementary
    disease-level analysis on the gold dataset.
    """
    name = str(label_name).lower().strip()
    if not name:
        return ""
    if "pneumothorax" in name:
        return "pneumothorax"
    if "pneumonia" in name:
        return "pneumonia"
    if "effusion" in name:
        return "pleural_effusion"
    if "edema" in name:
        return "edema"
    if "consolidation" in name:
        return "consolidation"
    return ""


def eval_gold_temporal(
    model,
    model_type: str,
    gold_comparison_file: str,
    mimic_cxr_dir: str,
    mimic_metadata_csv: str,
    train_data_dir: Optional[str] = None,
    svm_seeds: Optional[List[int]] = None,
) -> Dict:
    """Evaluate on Chest ImaGenome Gold Dataset expert-annotated temporal comparisons.

    The gold dataset provides per-anatomy temporal change labels (improved / no change /
    worsened) with bounding box coordinates, manually annotated by radiologists.

    Args:
        gold_comparison_file: Path to gold_object_comparison_with_coordinates.txt.
        mimic_cxr_dir: MIMIC-CXR-JPG files root.
        mimic_metadata_csv: Path to mimic-cxr-2.0.0-metadata.csv.
        train_data_dir: Optional path to training data dir (for leakage check).

    Returns:
        Dict with overall accuracy, macro F1, per-anatomy breakdown, confusion matrix.
    """
    import ast
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_recall_fscore_support,
        confusion_matrix,
    )

    print("\n" + "=" * 60)
    print("  Evaluation 4: Chest ImaGenome Gold Dataset (Expert-Annotated)")
    print("=" * 60)

    # ---- Load gold comparison data ----
    df = pd.read_csv(gold_comparison_file, sep="\t")
    total_before = len(df)

    # Filter ambiguous labels (contain ";;")
    df = df[~df["comparison"].str.contains(";;", na=False)]
    # Also drop the header-like row if comparison == "comparison"
    df = df[df["comparison"].isin(["improved", "no change", "worsened"])]
    print(f"  Loaded {total_before} comparisons, kept {len(df)} after filtering ambiguous")

    # Map labels: improved→0, no_change→1, worsened→2
    gold_label_map = {"improved": 0, "no change": 1, "worsened": 2}
    df["label"] = df["comparison"].map(gold_label_map)
    df["bbox_norm"] = df["bbox"].astype(str).str.lower().str.strip()
    df["pair_key"] = df["current_image_id"].astype(str) + "||" + df["previous_image_id"].astype(str)
    df["disease_tag"] = df["label_name"].fillna("").astype(str).map(_gold_label_name_to_disease)

    label_counts = df["comparison"].value_counts()
    for lbl in ["improved", "no change", "worsened"]:
        cnt = label_counts.get(lbl, 0)
        print(f"    {lbl}: {cnt} ({100 * cnt / len(df):.1f}%)")

    # Collapse duplicate attribute rows into one sample per (pair, anatomy bbox).
    # If a (pair, bbox) has conflicting temporal labels, drop it as ambiguous.
    before_dedup = len(df)
    dedup_rows = []
    dropped_conflict_groups = 0
    mixed_disease_groups = 0
    for _, g in df.groupby(["pair_key", "bbox_norm"], sort=False):
        if g["label"].nunique() != 1:
            dropped_conflict_groups += 1
            continue
        row = g.iloc[0].copy()
        row["bbox"] = row["bbox_norm"]
        disease_set = sorted({d for d in g["disease_tag"].tolist() if d})
        if len(disease_set) == 1:
            row["disease_tag"] = disease_set[0]
        else:
            row["disease_tag"] = ""
            if len(disease_set) > 1:
                mixed_disease_groups += 1
        dedup_rows.append(row)

    if dedup_rows:
        df = pd.DataFrame(dedup_rows).reset_index(drop=True)
    else:
        df = df.iloc[:0].copy()
    print(f"  Dedup (pair+bbox): {before_dedup} -> {len(df)} samples")
    if dropped_conflict_groups > 0:
        print(f"    Dropped conflicting (pair+bbox) groups: {dropped_conflict_groups}")
    if mixed_disease_groups > 0:
        print(f"    Mixed disease-tag groups (excluded from disease metrics): {mixed_disease_groups}")

    # ---- Data leakage check ----
    if train_data_dir:
        train_file = os.path.join(train_data_dir, "train.jsonl")
        if os.path.exists(train_file):
            train_pids = set()
            with open(train_file) as f:
                for line in f:
                    item = json.loads(line)
                    train_pids.add(str(item.get("patient_id", "")))
            gold_pids = set(df["patient_id"].astype(str).unique())
            overlap = gold_pids & train_pids
            print(f"  [Leakage] Gold patients: {len(gold_pids)}, "
                  f"Train patients: {len(train_pids)}, "
                  f"Overlap: {len(overlap)} ({100 * len(overlap) / max(len(gold_pids), 1):.1f}%)")

    # ---- Build dicom_id → path mapping ----
    print("  Building DICOM path lookup...", flush=True)
    dicom_lookup = _build_dicom_path_lookup(mimic_metadata_csv, mimic_cxr_dir)
    print(f"  Lookup built: {len(dicom_lookup)} entries")

    # ---- Group by image pair ----
    pairs = df.groupby("pair_key")
    n_pairs = len(pairs)
    print(f"  Unique image pairs: {n_pairs}")

    # ---- Inference ----
    val_transform = transforms.Compose([
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    model.eval()
    device = next(model.parameters()).device

    # For base_medst: collect ROI features for SVM (no classification head)
    # For finetuner / anatomy_pretrain: direct classification only (no SVM)
    use_svm = (model_type == "base_medst")
    if use_svm:
        backbone = model.img_encoder_q
    else:
        backbone = None

    all_preds = []
    all_logits = []  # for logits-based SVM (finetuner)
    all_labels = []
    all_anatomies = []
    all_diseases = []
    all_features = []  # only used for base_medst SVM
    all_pair_groups = []  # pair index per sample, for GroupKFold
    skipped_pairs = 0
    pair_idx = 0

    with torch.no_grad():
        for pair_key, group in tqdm(pairs, total=n_pairs, desc="  Gold eval"):
            current_id = group["current_image_id"].iloc[0]
            previous_id = group["previous_image_id"].iloc[0]

            # Resolve image paths
            current_path = dicom_lookup.get(current_id)
            previous_path = dicom_lookup.get(previous_id)
            if not current_path or not previous_path:
                skipped_pairs += 1
                continue
            if not os.path.exists(current_path) or not os.path.exists(previous_path):
                skipped_pairs += 1
                continue

            # Load images
            try:
                current_img = load_cxr_image(current_path, scale=256)
                previous_img = load_cxr_image(previous_path, scale=256)
            except Exception:
                skipped_pairs += 1
                continue

            current_tensor = val_transform(current_img)
            previous_tensor = val_transform(previous_img)

            # Build per-anatomy batch
            prior_bboxes = []
            current_bboxes = []
            anatomy_indices = []
            labels = []
            anatomies = []
            diseases = []

            for _, row in group.iterrows():
                try:
                    bbox_cur = ast.literal_eval(row["bbox_coord_224_subject"])
                    bbox_pri = ast.literal_eval(row["bbox_coord_224_object"])
                except (ValueError, SyntaxError):
                    continue

                # Convert from ImaGenome 224-space to BAAP 256-crop space
                bbox_cur = _bbox_224_to_256crop(bbox_cur)
                bbox_pri = _bbox_224_to_256crop(bbox_pri)

                anat_name = str(row["bbox"]).lower().strip()
                anat_idx = ANATOMY_TO_IDX.get(anat_name, ANATOMY_UNK_IDX)

                current_bboxes.append(torch.tensor(bbox_cur, dtype=torch.float32))
                prior_bboxes.append(torch.tensor(bbox_pri, dtype=torch.float32))
                anatomy_indices.append(anat_idx)
                labels.append(int(row["label"]))
                anatomies.append(anat_name)
                diseases.append(str(row.get("disease_tag", "")))

            if len(labels) == 0:
                skipped_pairs += 1
                continue

            sample_indices = torch.zeros(len(labels), dtype=torch.long, device=device)
            prior_bboxes_t = torch.stack(prior_bboxes).to(device)
            current_bboxes_t = torch.stack(current_bboxes).to(device)

            if use_svm:
                # base_medst: extract patch features and ROI pool for SVM
                prior_patch = _extract_patch_features(
                    backbone, previous_tensor.unsqueeze(0).to(device))
                current_patch = _extract_patch_features(
                    backbone, current_tensor.unsqueeze(0).to(device))
                prior_roi = roi_pool(prior_patch, prior_bboxes_t, sample_indices)   # [N_anat, D]
                current_roi = roi_pool(current_patch, current_bboxes_t, sample_indices)
                diff_roi = current_roi - prior_roi
                feats = torch.cat([prior_roi, current_roi, diff_roi], dim=-1)  # [N_anat, 3*D]
                all_features.append(feats.cpu().numpy())
            else:
                # finetuner / anatomy_pretrain: direct classification
                batch = {
                    "prior_imgs": previous_tensor.unsqueeze(0).to(device),
                    "current_imgs": current_tensor.unsqueeze(0).to(device),
                    "prior_bboxes": prior_bboxes_t,
                    "current_bboxes": current_bboxes_t,
                    "sample_indices": sample_indices,
                    "anatomy_indices": torch.tensor(
                        anatomy_indices, dtype=torch.long, device=device),
                }
                logits = infer_temporal_logits(model, batch, model_type=model_type)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_logits.append(logits.cpu().numpy())

            all_labels.extend(labels)
            all_anatomies.extend(anatomies)
            all_diseases.extend(diseases)
            all_pair_groups.extend([pair_idx] * len(labels))
            pair_idx += 1

    if skipped_pairs > 0:
        print(f"  Skipped {skipped_pairs}/{n_pairs} pairs (missing images)")

    all_labels = np.array(all_labels)
    n_evaluated = len(all_labels)
    print(f"  Evaluated {n_evaluated} comparisons across {n_pairs - skipped_pairs} pairs")

    if n_evaluated == 0:
        print("  WARNING: No valid comparisons found. Returning empty result.")
        return {"n_comparisons": 0, "n_pairs": 0, "error": "no valid comparisons"}

    # ---- SVM path (base_medst only) ----
    svm_result = None
    if use_svm:
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score, cross_val_predict, GroupKFold

        if svm_seeds is None:
            svm_seeds = [50, 52, 100]

        X = np.concatenate(all_features, axis=0)  # [N_total, 3*D]
        y = all_labels
        groups = np.array(all_pair_groups)
        n_group_pairs = int(len(np.unique(groups)))
        print(f"  SVM feature matrix: {X.shape}, unique groups (pairs): {n_group_pairs}")

        if n_group_pairs < 2:
            print("  WARNING: Need at least 2 valid pairs for GroupKFold. "
                  "Skipping SVM and returning SVM-only result.")
            return {
                "n_comparisons": n_evaluated,
                "n_pairs": n_pairs - skipped_pairs,
                "error": "too few pairs for GroupKFold",
            }

        # Use GroupKFold to prevent same image pair leaking across train/test
        cv10_splits = min(10, n_group_pairs)
        gkf10 = GroupKFold(n_splits=cv10_splits)

        all_cv10 = []
        for seed in svm_seeds:
            clf10 = SVC(kernel="linear", random_state=seed)
            scores10 = cross_val_score(clf10, X, y, cv=gkf10, groups=groups)
            all_cv10.append(scores10.mean() * 100)
            print(f"    Seed {seed}: SVM 10-fold CV={scores10.mean()*100:.2f}%")

        svm_result = {
            "method": "svm_roi_group_cv",
            "feature_dim": int(X.shape[1]),
            "cv10_splits": int(cv10_splits),
            "cv10_mean": round(float(np.mean(all_cv10)), 2),
            "cv10_std": round(float(np.std(all_cv10)), 2),
        }
        print(f"\n  SVM Group 10-fold CV: {svm_result['cv10_mean']:.2f} +/- {svm_result['cv10_std']:.2f}%")

        # Use SVM out-of-fold predictions for downstream metrics
        clf_oof = SVC(kernel="linear", random_state=42)
        all_preds = cross_val_predict(clf_oof, X, y, cv=gkf10, groups=groups)
    else:
        all_preds = np.array(all_preds)

    # ---- Logits-based SVM (finetuner / anatomy_pretrain) ----
    logits_svm_result = None
    if not use_svm and len(all_logits) > 0:
        from sklearn.svm import SVC as _SVC
        from sklearn.model_selection import cross_val_score as _cv_score, GroupKFold as _GKF

        if svm_seeds is None:
            svm_seeds = [50, 52, 100]

        X_logits = np.concatenate(all_logits, axis=0)  # [N_total, 3]
        y_logits = all_labels
        groups_logits = np.array(all_pair_groups)
        n_group_pairs_l = int(len(np.unique(groups_logits)))
        print(f"\n  Logits-SVM feature matrix: {X_logits.shape}, unique groups: {n_group_pairs_l}")

        if n_group_pairs_l >= 5:
            cv10_sp = min(10, n_group_pairs_l)
            gkf10_l = _GKF(n_splits=cv10_sp)

            lcv10 = []
            for seed in svm_seeds:
                clf10 = _SVC(kernel="linear", random_state=seed)
                s10 = _cv_score(clf10, X_logits, y_logits, cv=gkf10_l, groups=groups_logits)
                lcv10.append(s10.mean() * 100)
                print(f"    Logits-SVM seed {seed}: SVM 10-fold CV={s10.mean()*100:.2f}%")

            logits_svm_result = {
                "method": "logits_svm_group_cv",
                "feature_dim": int(X_logits.shape[1]),
                "cv10_mean": round(float(np.mean(lcv10)), 2),
                "cv10_std": round(float(np.std(lcv10)), 2),
            }
            print(f"\n  Logits-SVM Group 10-fold CV: {logits_svm_result['cv10_mean']:.2f} +/- {logits_svm_result['cv10_std']:.2f}%")
        else:
            print(f"  WARNING: Too few groups ({n_group_pairs_l}) for logits-SVM GroupKFold, skipping.")

    # ---- Compute metrics ----
    overall_acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    prec, rec, f1_per, sup = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1, 2], zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

    label_names = ["improved", "no_change", "worsened"]
    print(f"\n  Overall accuracy: {100 * overall_acc:.2f}%")
    print(f"  Macro F1:         {100 * macro_f1:.2f}%")
    print(f"  Weighted F1:      {100 * weighted_f1:.2f}%")
    print(f"\n  Per-class metrics:")
    for i, name in enumerate(label_names):
        print(f"    {name:12s}: P={prec[i]:.3f}  R={rec[i]:.3f}  F1={f1_per[i]:.3f}  n={sup[i]}")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(f"    {'':12s}  {'improved':>10s}  {'no_change':>10s}  {'worsened':>10s}")
    for i, name in enumerate(label_names):
        print(f"    {name:12s}  {cm[i][0]:>10d}  {cm[i][1]:>10d}  {cm[i][2]:>10d}")

    # ---- Per-anatomy breakdown ----
    print(f"\n  Per-anatomy breakdown:")
    anatomy_results = {}
    for anat in sorted(set(all_anatomies)):
        mask = np.array([a == anat for a in all_anatomies])
        if mask.sum() == 0:
            continue
        anat_acc = accuracy_score(all_labels[mask], all_preds[mask])
        anat_f1 = f1_score(all_labels[mask], all_preds[mask], average="macro", zero_division=0)
        anatomy_results[anat] = {
            "accuracy": round(100 * anat_acc, 2),
            "macro_f1": round(100 * anat_f1, 2),
            "n": int(mask.sum()),
        }
        print(f"    {anat:30s}: acc={100 * anat_acc:5.1f}%  F1={100 * anat_f1:5.1f}%  n={mask.sum()}")

    # ---- Disease-level breakdown (supplementary; derived from label_name keyword mapping) ----
    print(f"\n  Per-disease breakdown (supplementary, keyword-derived):")
    disease_results = {}
    disease_arr = np.array(all_diseases, dtype=object)
    disease_order = ["consolidation", "edema", "pleural_effusion", "pneumonia", "pneumothorax"]
    mapped_count = int((disease_arr != "").sum())
    for disease in disease_order:
        mask = disease_arr == disease
        if mask.sum() < 5:
            continue
        dis_acc = accuracy_score(all_labels[mask], all_preds[mask])
        dis_f1 = f1_score(all_labels[mask], all_preds[mask], average="macro", zero_division=0)
        disease_results[disease] = {
            "accuracy": round(100 * dis_acc, 2),
            "macro_f1": round(100 * dis_f1, 2),
            "n": int(mask.sum()),
        }
        print(f"    {disease:30s}: acc={100 * dis_acc:5.1f}%  F1={100 * dis_f1:5.1f}%  n={mask.sum()}")
    if not disease_results:
        print("    (No disease bucket has n>=5 after conservative mapping)")
    print(f"    Disease mapping coverage: {mapped_count}/{len(disease_arr)}")

    result = {}
    if svm_result is not None:
        result["svm"] = svm_result
    if logits_svm_result is not None:
        result["logits_svm"] = logits_svm_result
    result.update({
        "n_comparisons": n_evaluated,
        "n_pairs": n_pairs - skipped_pairs,
        "dropped_conflict_pair_bbox": int(dropped_conflict_groups),
        "disease_mapping_coverage": {
            "mapped": mapped_count,
            "total": int(len(disease_arr)),
        },
        "overall_accuracy": round(100 * overall_acc, 2),
        "macro_f1": round(100 * macro_f1, 2),
        "weighted_f1": round(100 * weighted_f1, 2),
        "per_class": {
            name: {"precision": round(float(prec[i]), 4),
                   "recall": round(float(rec[i]), 4),
                   "f1": round(float(f1_per[i]), 4),
                   "support": int(sup[i])}
            for i, name in enumerate(label_names)
        },
        "confusion_matrix": cm.tolist(),
        "per_anatomy": anatomy_results,
        "per_disease": disease_results,
    })
    return result


# ============================================================================
# Main
# ============================================================================
def main():
    parser = ArgumentParser(description="Anatomy-Temporal Fine-tuner Evaluation")

    parser.add_argument(
        "--ckpt_path", type=str, required=True,
        help="Path to finetuned AnatomyTemporalFineTuner checkpoint",
    )
    parser.add_argument(
        "--eval_mode", type=str, nargs="+", default=["mscxrt"],
        help="Which evaluation(s) to run. One or more of: "
             "all, imagenome, mscxrt, mscxrt_svm, mscxrt_direct, mscxrt_roi, mscxrt_ensemble, gold_temporal",
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=os.path.join(
            DEFAULT_DATA_ROOT,
            "chest-imagenome-dataset-1.0.0",
            "temporal_finetuning_dataset",
        ),
        help="Chest ImaGenome temporal_finetuning_dataset dir (for imagenome eval)",
    )
    parser.add_argument("--test_file", type=str, default="test.jsonl")
    parser.add_argument(
        "--mscxrt_csv", type=str,
        default=os.path.join(
            DEFAULT_DATA_ROOT,
            "ms-cxr-t",
            "MS_CXR_T_temporal_image_classification_v1.0.0.csv",
        ),
        help="MS-CXR-T image classification CSV path",
    )
    parser.add_argument(
        "--mimic_cxr_dir", type=str,
        default=os.path.join(DEFAULT_DATA_ROOT, "mimic-cxr-jpg-2.1.0", "files"),
        help="MIMIC-CXR-JPG files root",
    )
    parser.add_argument(
        "--scene_graph_dir", type=str,
        default=os.path.join(
            DEFAULT_DATA_ROOT,
            "chest-imagenome-dataset-1.0.0",
            "silver_dataset",
            "scene_graph",
        ),
        help="Chest ImaGenome scene_graph dir (for ROI on MS-CXR-T)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--svm_seeds", type=int, nargs="+", default=[50, 52, 100],
        help="Random seeds for SVM cross-validation (default: 50 52 100)",
    )
    parser.add_argument(
        "--no_shuffle_folds", action="store_true",
        help="Use StratifiedKFold without shuffle (deterministic fold split, ignores svm_seeds for fold assignment)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="",
        help="Directory to save evaluation results JSON (default: same dir as ckpt)",
    )
    parser.add_argument(
        "--model_type", type=str, default="finetuner",
        choices=["finetuner", "anatomy_pretrain", "base_medst"],
        help="Type of checkpoint: 'finetuner' (AnatomyTemporalFineTuner), "
             "'anatomy_pretrain' (MedSTAnatomy Stage 2), or "
             "'base_medst' (original MedST Stage 1)",
    )
    parser.add_argument(
        "--svm_feature_mode", type=str, default="original",
        choices=["original", "enhanced"],
        help="SVM feature mode: 'original' = concat(prior, current); "
             "'enhanced' = concat(prior, current, current-prior)",
    )
    parser.add_argument(
        "--roi_aggregation", type=str, default="majority",
        choices=["majority", "softmax"],
        help="ROI aggregation: 'majority' = hard vote; "
             "'softmax' = softmax-weighted mean of logits",
    )
    parser.add_argument(
        "--bbox_coord_mode", type=str, default="crop224",
        choices=["crop224", "raw224"],
        help="BBox coordinate interpretation for MS-CXR-T ROI eval: "
             "'crop224' converts Chest ImaGenome 224-space bboxes to the "
             "resize-256 + center-crop coordinate space; 'raw224' uses scene "
             "graph coordinates directly to reproduce older paper-protocol runs.",
    )
    parser.add_argument(
        "--bbox_noise_frac", type=float, default=0.0,
        help="[E5 auto-bbox robustness] Fractional bbox noise applied at eval time "
             "(0.0 = use ground-truth Chest ImaGenome bboxes). Each bbox center is "
             "jittered with Gaussian std = frac*size, and width/height are scaled "
             "uniformly by [1-frac, 1+frac]. Affects mscxrt_roi only.",
    )
    parser.add_argument(
        "--bbox_noise_seed", type=int, default=42,
        help="[E5 auto-bbox robustness] Base RNG seed for bbox noise (per-image "
             "seed is derived from md5 hash, so noise is reproducible).",
    )
    parser.add_argument(
        "--gold_comparison_file", type=str,
        default=os.path.join(
            DEFAULT_DATA_ROOT,
            "chest-imagenome-dataset-1.0.0",
            "gold_dataset",
            "gold_object_comparison_with_coordinates.txt",
        ),
        help="Gold dataset temporal comparison TSV file",
    )
    parser.add_argument(
        "--mimic_metadata_csv", type=str,
        default=os.path.join(
            DEFAULT_DATA_ROOT,
            "mimic-cxr-jpg-2.1.0",
            "mimic-cxr-2.0.0-metadata.csv",
        ),
        help="MIMIC-CXR metadata CSV (for dicom_id → path mapping)",
    )

    args = parser.parse_args()
    seed_everything(args.seed, workers=True)

    t_total = time.time()

    # Load model
    print(f"[Step 1/3] Loading checkpoint: {args.ckpt_path}", flush=True)
    print(f"  Model type: {args.model_type}", flush=True)
    t0 = time.time()
    model, backbone = load_model(args.ckpt_path, model_type=args.model_type)
    print(f"  Model loaded in {time.time() - t0:.1f}s", flush=True)
    # Normalise eval_mode list into a set of concrete modes
    _ALL_MODES = {"imagenome", "mscxrt_svm", "mscxrt_direct", "mscxrt_roi", "mscxrt_ensemble", "gold_temporal"}
    eval_modes = set()
    for m in args.eval_mode:
        if m == "all":
            eval_modes = _ALL_MODES.copy()
            break
        elif m == "mscxrt":
            eval_modes |= {"mscxrt_svm", "mscxrt_roi"}
        else:
            eval_modes.add(m)

    report = {
        "checkpoint": args.ckpt_path,
        "svm_feature_mode": args.svm_feature_mode,
        "roi_aggregation": args.roi_aggregation,
        "bbox_coord_mode": args.bbox_coord_mode,
        "bbox_noise_frac": args.bbox_noise_frac,
        "bbox_noise_seed": args.bbox_noise_seed,
    }

    # Filter unsupported modes by model type
    # base_medst: no classifier heads at all
    _BASE_MEDST_UNSUPPORTED = {"imagenome", "mscxrt_roi", "mscxrt_ensemble", "mscxrt_direct"}
    if args.model_type == "base_medst":
        unsupported = eval_modes & _BASE_MEDST_UNSUPPORTED
        if unsupported:
            print(f"  [WARN] base_medst does not support eval modes: {unsupported} "
                  f"(no temporal classification head). Skipping them.")
            eval_modes -= unsupported
        if not eval_modes:
            print(f"  ERROR: No supported eval modes remain for base_medst. "
                  f"Supported modes: mscxrt_svm, gold_temporal")
            return

    print(f"  Eval modes: {sorted(eval_modes)}", flush=True)

    # ---- Eval 1: ImaGenome ----
    if "imagenome" in eval_modes and args.model_type != "base_medst":
        report["imagenome"] = eval_imagenome(
            model=model,
            model_type=args.model_type,
            data_dir=args.data_dir,
            test_file=args.test_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    # ---- Eval 2a: MS-CXR-T SVM ----
    if "mscxrt_svm" in eval_modes:
        print(f"\n[Step 2/3] Starting MS-CXR-T SVM evaluation...", flush=True)
        t0 = time.time()
        report["mscxrt_svm"] = eval_mscxrt_svm(
            model=model,
            mscxrt_csv=args.mscxrt_csv,
            mimic_cxr_dir=args.mimic_cxr_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            svm_seeds=args.svm_seeds,
            no_shuffle_folds=args.no_shuffle_folds,
            backbone=backbone,
            svm_feature_mode=args.svm_feature_mode,
        )
        print(f"  SVM evaluation done in {time.time() - t0:.1f}s", flush=True)

    # ---- Eval 2a+: MS-CXR-T Direct Classification ----
    if "mscxrt_direct" in eval_modes:
        print(f"\n[Direct] Starting MS-CXR-T direct classification evaluation...", flush=True)
        t0 = time.time()
        report["mscxrt_direct"] = eval_mscxrt_direct(
            model=model,
            mscxrt_csv=args.mscxrt_csv,
            mimic_cxr_dir=args.mimic_cxr_dir,
            batch_size=args.batch_size,
            backbone=backbone,
            svm_seeds=args.svm_seeds,
            no_shuffle_folds=args.no_shuffle_folds,
        )
        print(f"  Direct classification done in {time.time() - t0:.1f}s", flush=True)

    # ---- Eval 2b: MS-CXR-T ROI ----
    if "mscxrt_roi" in eval_modes and args.model_type != "base_medst":
        print(f"\n[Step 3/3] Starting MS-CXR-T ROI evaluation...", flush=True)
        t0 = time.time()
        report["mscxrt_roi"] = eval_mscxrt_roi(
            model=model,
            model_type=args.model_type,
            mscxrt_csv=args.mscxrt_csv,
            mimic_cxr_dir=args.mimic_cxr_dir,
            scene_graph_dir=args.scene_graph_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            svm_seeds=args.svm_seeds,
            no_shuffle_folds=args.no_shuffle_folds,
            roi_aggregation=args.roi_aggregation,
            bbox_coord_mode=args.bbox_coord_mode,
            bbox_noise_frac=args.bbox_noise_frac,
            bbox_noise_seed=args.bbox_noise_seed,
        )
        print(f"  ROI evaluation done in {time.time() - t0:.1f}s", flush=True)

    # ---- Eval 3: MS-CXR-T Ensemble ----
    if "mscxrt_ensemble" in eval_modes and args.model_type != "base_medst":
        print(f"\n[Ensemble] Starting MS-CXR-T Ensemble evaluation...", flush=True)
        t0 = time.time()
        report["mscxrt_ensemble"] = eval_mscxrt_ensemble(
            model=model,
            model_type=args.model_type,
            mscxrt_csv=args.mscxrt_csv,
            mimic_cxr_dir=args.mimic_cxr_dir,
            scene_graph_dir=args.scene_graph_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            svm_seeds=args.svm_seeds,
            no_shuffle_folds=args.no_shuffle_folds,
            roi_aggregation=args.roi_aggregation,
            backbone=backbone,
        )
        print(f"  Ensemble evaluation done in {time.time() - t0:.1f}s", flush=True)

    # ---- Eval 4: Gold Dataset Temporal ----
    if "gold_temporal" in eval_modes:
        print(f"\n[Gold] Starting Gold Dataset temporal evaluation...", flush=True)
        t0 = time.time()
        report["gold_temporal"] = eval_gold_temporal(
            model=model,
            model_type=args.model_type,
            gold_comparison_file=args.gold_comparison_file,
            mimic_cxr_dir=args.mimic_cxr_dir,
            mimic_metadata_csv=args.mimic_metadata_csv,
            train_data_dir=args.data_dir,
            svm_seeds=args.svm_seeds,
        )
        print(f"  Gold evaluation done in {time.time() - t0:.1f}s", flush=True)

    # ---- Save report ----
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.ckpt_path)
    os.makedirs(output_dir, exist_ok=True)

    ckpt_name = os.path.splitext(os.path.basename(args.ckpt_path))[0]
    report_path = os.path.join(output_dir, f"eval_report_{ckpt_name}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    if "imagenome" in report and report["imagenome"]:
        ig = report["imagenome"]
        print(f"  ImaGenome: acc={ig['test_acc']}, f1_macro={ig['test_f1_macro']}")
    if "mscxrt_svm" in report and report["mscxrt_svm"]:
        svm = report["mscxrt_svm"]
        print(f"  MS-CXR-T SVM 10-fold CV: {svm.get('svm_10fold_cv_accuracy', {})}")
    if "mscxrt_direct" in report and report["mscxrt_direct"]:
        dc = report["mscxrt_direct"]
        print(f"  MS-CXR-T Image-level SVM 10-fold CV: {dc.get('svm_10fold_cv_accuracy', {})}")
    if "mscxrt_roi" in report and report["mscxrt_roi"]:
        roi = report["mscxrt_roi"]
        print(f"  MS-CXR-T ROI SVM 10-fold CV: {roi.get('svm_10fold_cv_accuracy', {})}")
    if "mscxrt_ensemble" in report and report["mscxrt_ensemble"]:
        ens = report["mscxrt_ensemble"]
        print(f"  MS-CXR-T Ensemble SVM 10-fold CV: {ens.get('svm_10fold_cv_accuracy', {})}")
    if "gold_temporal" in report and report["gold_temporal"]:
        gold = report["gold_temporal"]
        if "overall_accuracy" in gold and "macro_f1" in gold:
            print(f"  Gold Dataset: acc={gold['overall_accuracy']}%, "
                  f"macro_F1={gold['macro_f1']}%, "
                  f"n={gold['n_comparisons']}")
        else:
            print(f"  Gold Dataset: n={gold.get('n_comparisons', 0)}, "
                  f"error={gold.get('error', 'unknown')}")
    print(f"  Total time: {time.time() - t_total:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
