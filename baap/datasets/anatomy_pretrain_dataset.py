"""
Anatomy-Aware Temporal Pre-training Dataset
============================================

Loads Chest ImaGenome temporal pairs (decontaminated) for Stage 2 pre-training.
Each sample provides:
  - Prior and current CXR images (224x224 after CenterCrop)
  - Tokenized summary text (for ITA + local alignment losses)
  - Per-anatomy bounding boxes, labels, and phrase texts (for anatomy losses)

Data source: temporal_finetuning_dataset_clean/ produced by decontaminate_imagenome.py

Usage:
    from baap.datasets.anatomy_pretrain_dataset import (
        AnatomyPretrainDataset, anatomy_pretrain_collate_fn
    )
    dataset = AnatomyPretrainDataset("train", data_dir="path/to/clean")
"""

import json
import os
import re
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from transformers import BertTokenizer

# ---------------------------------------------------------------------------
# Constants (shared with anatomy_temporal_finetuner.py)
# ---------------------------------------------------------------------------
LABEL_MAP = {"improved": 0, "no_change": 1, "worsened": 2}

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
ANATOMY_UNK_IDX = len(ANATOMY_LIST)

# Image paths in ImaGenome JSONL may use a different base path than the
# project data directory.  We resolve this at load time.
_OLD_IMAGE_BASE = os.environ.get("BAAP_OLD_MIMIC_CXR_ROOT", "")
_NEW_IMAGE_BASE = None  # set by dataset __init__

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Image loading (matching BAAP preprocessing)
# ---------------------------------------------------------------------------
def _resize_img(img, scale):
    """Aspect-preserving resize + zero-padding to (scale, scale)."""
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)
    if max_ind == 0:
        wpercent = scale / float(size[0])
        hsize = int(float(size[1]) * float(wpercent))
        desireable_size = (scale, hsize)
    else:
        hpercent = scale / float(size[1])
        wsize = int(float(size[0]) * float(hpercent))
        desireable_size = (wsize, scale)
    resized_img = cv2.resize(img, desireable_size[::-1], interpolation=cv2.INTER_AREA)
    if max_ind == 0:
        pad_size = scale - resized_img.shape[1]
        left = int(np.floor(pad_size / 2))
        right = int(np.ceil(pad_size / 2))
        top, bottom = 0, 0
    else:
        pad_size = scale - resized_img.shape[0]
        top = int(np.floor(pad_size / 2))
        bottom = int(np.ceil(pad_size / 2))
        left, right = 0, 0
    return np.pad(resized_img, [(top, bottom), (left, right)], "constant", constant_values=0)


def _load_cxr(img_path: str, scale: int = 256) -> Image.Image:
    """Load CXR: grayscale cv2 -> resize(256) -> RGB PIL."""
    x = cv2.imread(str(img_path), 0)
    if x is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    x = _resize_img(x, scale)
    return Image.fromarray(x).convert("RGB")


def _remap_path(path: str, new_base: Optional[str]) -> str:
    """Remap image paths from ImaGenome JSONL to the project data directory."""
    if new_base is None or not _OLD_IMAGE_BASE or not path.startswith(_OLD_IMAGE_BASE):
        return path
    return path.replace(_OLD_IMAGE_BASE, new_base, 1)


# ---------------------------------------------------------------------------
# Bbox coordinate conversion
# ---------------------------------------------------------------------------
def _bbox_224_to_256crop(bbox_224: list, crop_size: int = 224, resize_scale: int = 256) -> list:
    """Convert bbox from ImaGenome resize-224+pad space to BAAP resize-256+pad+CenterCrop(224) space.

    Chest ImaGenome computes bbox_224 via aspect-preserving resize to 224 + zero-pad.
    BAAP loads images via resize to 256 + zero-pad + CenterCrop(224).
    The conversion is: coord_crop = coord_224 * (256/224) - 16, clamped to [0, 224].
    """
    scale = resize_scale / crop_size  # 256 / 224 ≈ 1.143
    offset = (resize_scale - crop_size) / 2  # (256 - 224) / 2 = 16
    return [max(0.0, min(float(crop_size), coord * scale - offset)) for coord in bbox_224]


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------
def _clean_text(text: str) -> str:
    """Minimal cleaning for radiology text."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AnatomyPretrainDataset(data.Dataset):
    """Dataset for anatomy-aware temporal pre-training (Stage 2).

    Each sample yields a prior-current CXR pair with:
    - Tokenized summary text for global ITA + local alignment
    - Per-anatomy bounding boxes and temporal change labels
    - Per-anatomy phrase text for contrastive alignment
    """

    def __init__(
        self,
        split: str = "train",
        data_dir: str = "",
        image_base: Optional[str] = None,
        transform=None,
        max_words: int = 112,
        max_comparisons: int = 15,
    ):
        super().__init__()
        self.split = split
        self.transform = transform
        self.max_words = max_words
        self.max_comparisons = max_comparisons

        # Resolve image base path
        if image_base is None:
            # Default: project data dir
            project_root = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
            self.image_base = os.path.join(project_root, "data", "mimic-cxr-jpg-2.1.0")
        else:
            self.image_base = image_base

        # Load decontaminated JSONL
        file_map = {"train": "train_clean.jsonl", "valid": "valid_clean.jsonl", "test": "test_clean.jsonl"}
        data_file = os.path.join(data_dir, file_map[split])
        self.data = self._load_jsonl(data_file)
        print(f"[AnatomyPretrainDataset-{split}] Loaded {len(self.data)} samples")

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    def _load_jsonl(self, path: str) -> List[Dict]:
        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # Filter to valid comparisons
                valid = [
                    c for c in item.get("comparisons", [])
                    if c.get("current_bbox_224") and c.get("prior_bbox_224") and c.get("label") in LABEL_MAP
                ]
                if valid:
                    item["comparisons"] = valid[:self.max_comparisons]
                    item["num_comparisons"] = len(item["comparisons"])
                    records.append(item)
        return records

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        # --- Images ---
        prior_path = _remap_path(item["prior_image_path"], self.image_base)
        current_path = _remap_path(item["current_image_path"], self.image_base)
        prior_img = _load_cxr(prior_path)
        current_img = _load_cxr(current_path)
        if self.transform:
            prior_img = self.transform(prior_img)
            current_img = self.transform(current_img)

        # --- Summary text (for ITA + local alignment) ---
        summary = _clean_text(item.get("summary_text", ""))
        if not summary:
            # Fallback: concatenate per-anatomy phrases
            phrases = [c.get("phrase", "") for c in item["comparisons"] if c.get("phrase")]
            summary = " ".join(phrases)
        if not summary:
            summary = "no significant change"

        tokens = self.tokenizer(
            summary,
            max_length=self.max_words,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # --- Per-anatomy comparisons ---
        comparisons = []
        for c in item["comparisons"]:
            phrase = _clean_text(c.get("phrase", ""))
            if not phrase:
                phrase = f"{c['anatomy']} shows {c['label']}"

            phrase_tokens = self.tokenizer(
                phrase,
                max_length=64,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Convert bbox from ImaGenome 224-space to BAAP 256+crop space
            prior_bbox_crop = _bbox_224_to_256crop(c["prior_bbox_224"])
            current_bbox_crop = _bbox_224_to_256crop(c["current_bbox_224"])

            comparisons.append({
                "prior_bbox": torch.tensor(prior_bbox_crop, dtype=torch.float32),
                "current_bbox": torch.tensor(current_bbox_crop, dtype=torch.float32),
                "label": LABEL_MAP[c["label"]],
                "anatomy_idx": ANATOMY_TO_IDX.get(c["anatomy"], ANATOMY_UNK_IDX),
                "phrase_ids": phrase_tokens["input_ids"].squeeze(0),
                "phrase_attention_mask": phrase_tokens["attention_mask"].squeeze(0),
            })

        return {
            "prior_img": prior_img,
            "current_img": current_img,
            # Summary text tokens (for ITA + local)
            "caption_ids": tokens["input_ids"].squeeze(0),
            "token_type_ids": tokens["token_type_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            # Anatomy comparisons
            "comparisons": comparisons,
            "num_comparisons": len(comparisons),
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def anatomy_pretrain_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate that handles variable-length anatomy comparisons.

    Returns a batch dict with:
    - Image tensors: [B, 3, 224, 224] for prior and current
    - Text tensors: [B, max_words] for summary captions
    - Flattened anatomy tensors: [N_total, ...] with sample_indices for mapping
    """
    B = len(batch)

    # --- Images ---
    prior_imgs = torch.stack([b["prior_img"] for b in batch])
    current_imgs = torch.stack([b["current_img"] for b in batch])

    # --- Summary text ---
    caption_ids = torch.stack([b["caption_ids"] for b in batch])
    token_type_ids = torch.stack([b["token_type_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])

    # --- Anatomy comparisons (flatten across batch) ---
    all_prior_bboxes = []
    all_current_bboxes = []
    all_labels = []
    all_sample_indices = []
    all_anatomy_indices = []
    all_phrase_ids = []
    all_phrase_masks = []
    num_comparisons = []

    for i, b in enumerate(batch):
        n = b["num_comparisons"]
        num_comparisons.append(n)
        for c in b["comparisons"]:
            all_prior_bboxes.append(c["prior_bbox"])
            all_current_bboxes.append(c["current_bbox"])
            all_labels.append(c["label"])
            all_sample_indices.append(i)
            all_anatomy_indices.append(c["anatomy_idx"])
            all_phrase_ids.append(c["phrase_ids"])
            all_phrase_masks.append(c["phrase_attention_mask"])

    # Handle edge case: empty batch (no comparisons at all)
    if not all_labels:
        N = 0
        dummy_bbox = torch.zeros(0, 4)
        return {
            "prior_imgs": prior_imgs,
            "current_imgs": current_imgs,
            "caption_ids": caption_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "prior_bboxes": dummy_bbox,
            "current_bboxes": dummy_bbox,
            "labels": torch.zeros(0, dtype=torch.long),
            "sample_indices": torch.zeros(0, dtype=torch.long),
            "anatomy_indices": torch.zeros(0, dtype=torch.long),
            "phrase_ids": torch.zeros(0, 64, dtype=torch.long),
            "phrase_attention_mask": torch.zeros(0, 64, dtype=torch.long),
            "num_comparisons": torch.zeros(B, dtype=torch.long),
        }

    return {
        # Images [B, 3, 224, 224]
        "prior_imgs": prior_imgs,
        "current_imgs": current_imgs,
        # Summary text [B, max_words]
        "caption_ids": caption_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        # Anatomy comparisons [N_total, ...]
        "prior_bboxes": torch.stack(all_prior_bboxes),
        "current_bboxes": torch.stack(all_current_bboxes),
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "sample_indices": torch.tensor(all_sample_indices, dtype=torch.long),
        "anatomy_indices": torch.tensor(all_anatomy_indices, dtype=torch.long),
        "phrase_ids": torch.stack(all_phrase_ids),
        "phrase_attention_mask": torch.stack(all_phrase_masks),
        "num_comparisons": torch.tensor(num_comparisons, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torchvision.transforms as T

    transform = T.Compose([
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    ds = AnatomyPretrainDataset(
        split="train",
        data_dir=os.environ.get(
            "BAAP_ANATOMY_PRETRAIN_DIR",
            "data/chest-imagenome-dataset-1.0.0/temporal_finetuning_dataset_clean",
        ),
        transform=transform,
    )
    print(f"Dataset size: {len(ds)}")

    sample = ds[0]
    print(f"Prior image shape: {sample['prior_img'].shape}")
    print(f"Current image shape: {sample['current_img'].shape}")
    print(f"Caption IDs shape: {sample['caption_ids'].shape}")
    print(f"Num comparisons: {sample['num_comparisons']}")
    for i, c in enumerate(sample["comparisons"]):
        print(f"  Comp {i}: bbox={c['current_bbox'].tolist()}, label={c['label']}, anat={c['anatomy_idx']}")

    # Test collate
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, collate_fn=anatomy_pretrain_collate_fn, num_workers=0)
    batch = next(iter(loader))
    print(f"\nBatch prior_imgs: {batch['prior_imgs'].shape}")
    print(f"Batch caption_ids: {batch['caption_ids'].shape}")
    print(f"Batch labels: {batch['labels'].shape}")
    print(f"Batch num_comparisons: {batch['num_comparisons']}")
