"""
Gold Dataset evaluation using CLS global embedding + SVM
(same approach as temporal_test.py but applied to the Gold Dataset)

Usage:
    export PYTHONPATH=$PWD:${PYTHONPATH:-}
    python -m medst.experiments.code.gold_cls_svm_eval \
        --ckpt_path /path/to/pretrained_encoder.ckpt
"""

import os
import sys
import ast
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

from medst.models.medst.medst_module import MedST
from medst.datasets.transforms import DataTransforms
from medst.datasets.utils import get_imgs

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DEFAULT_DATA_ROOT = os.environ.get("BAAP_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))


def load_model(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedST.load_from_checkpoint(ckpt_path, strict=False).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, device


def extract_cls_embedding(model, img_path, transform, device, imsize=256):
    """Extract CLS global embedding (128-dim, normalized) for a single image."""
    img, _ = get_imgs(img_path, imsize, transform, multiscale=False, return_size=True)
    with torch.no_grad():
        cls_feat, _ = model.img_encoder_q(img.unsqueeze(0).to(device))
        emb = model.img_encoder_q.global_embed(cls_feat)
        emb = F.normalize(emb, dim=-1)
    return emb.squeeze(0).cpu().numpy()  # (128,)


def build_dicom_path_lookup(metadata_csv, mimic_cxr_dir):
    meta = pd.read_csv(metadata_csv, usecols=["dicom_id", "subject_id", "study_id"])
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


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--gold_comparison_file", type=str,
                        default=os.path.join(
                            DEFAULT_DATA_ROOT,
                            "chest-imagenome-dataset-1.0.0",
                            "gold_dataset",
                            "gold_object_comparison_with_coordinates.txt",
                        ))
    parser.add_argument("--mimic_cxr_dir", type=str,
                        default=os.path.join(DEFAULT_DATA_ROOT, "mimic-cxr-jpg-2.1.0", "files"))
    parser.add_argument("--mimic_metadata_csv", type=str,
                        default=os.path.join(
                            DEFAULT_DATA_ROOT,
                            "mimic-cxr-jpg-2.1.0",
                            "mimic-cxr-2.0.0-metadata.csv",
                        ))
    parser.add_argument("--output_dir", type=str, default="medst/experiments/results/final_evl_results/base_medst_gold_cls_svm")
    parser.add_argument("--svm_seeds", type=int, nargs="+", default=[42, 100, 666])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.ckpt_path}")
    model, device = load_model(args.ckpt_path)
    transform = DataTransforms(is_train=False)

    # Load gold dataset
    print("Loading gold comparison data...")
    df = pd.read_csv(args.gold_comparison_file, sep="\t")
    total_before = len(df)
    df = df[~df["comparison"].str.contains(";;", na=False)]
    df = df[df["comparison"].isin(["improved", "no change", "worsened"])]
    print(f"  Loaded {total_before} rows, kept {len(df)} after filtering")

    gold_label_map = {"improved": 0, "no change": 1, "worsened": 2}
    df["label"] = df["comparison"].map(gold_label_map)
    df["pair_key"] = df["current_image_id"].astype(str) + "||" + df["previous_image_id"].astype(str)

    # Deduplicate: one label per (pair_key) — majority vote at pair level
    # Since we use CLS (image-level), we aggregate labels per image pair
    pair_labels = []
    for pair_key, group in df.groupby("pair_key", sort=False):
        label_counts = group["label"].value_counts()
        majority_label = label_counts.idxmax()
        pair_labels.append({
            "pair_key": pair_key,
            "current_image_id": group["current_image_id"].iloc[0],
            "previous_image_id": group["previous_image_id"].iloc[0],
            "label": majority_label,
        })
    df_pairs = pd.DataFrame(pair_labels)
    print(f"  Unique image pairs: {len(df_pairs)}")

    for lbl_name, lbl_val in gold_label_map.items():
        cnt = (df_pairs["label"] == lbl_val).sum()
        print(f"    {lbl_name}: {cnt} ({100 * cnt / len(df_pairs):.1f}%)")

    # Build DICOM path lookup
    print("Building DICOM path lookup...")
    dicom_lookup = build_dicom_path_lookup(args.mimic_metadata_csv, args.mimic_cxr_dir)
    print(f"  Lookup: {len(dicom_lookup)} entries")

    # Extract CLS embeddings for each pair
    print("Extracting CLS embeddings...")
    features = []
    labels = []
    skipped = 0

    for _, row in tqdm(df_pairs.iterrows(), total=len(df_pairs), desc="  Pairs"):
        cur_path = dicom_lookup.get(row["current_image_id"])
        pri_path = dicom_lookup.get(row["previous_image_id"])
        if not cur_path or not pri_path:
            skipped += 1
            continue
        if not os.path.exists(cur_path) or not os.path.exists(pri_path):
            skipped += 1
            continue

        try:
            cur_emb = extract_cls_embedding(model, cur_path, transform, device)
            pri_emb = extract_cls_embedding(model, pri_path, transform, device)
        except Exception as e:
            skipped += 1
            continue

        # concat(current, prior) → 256-dim, same as temporal_test.py
        feat = np.concatenate([cur_emb, pri_emb])
        features.append(feat)
        labels.append(int(row["label"]))

    if skipped > 0:
        print(f"  Skipped {skipped}/{len(df_pairs)} pairs")

    X = np.array(features)
    y = np.array(labels)
    print(f"  Feature matrix: {X.shape}")

    # SVM with CV-5 and CV-10
    # Shuffle data with different seeds (same as temporal_test.py) to get variance
    print("\n" + "=" * 60)
    print("  SVM (CLS embedding, no ROI)")
    print("=" * 60)

    all_cv5, all_cv10 = [], []
    for seed in args.svm_seeds:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(y))
        X_shuf = X[perm]
        y_shuf = y[perm]
        clf5 = SVC(kernel="linear", random_state=seed)
        scores5 = cross_val_score(clf5, X_shuf, y_shuf, cv=5)
        clf10 = SVC(kernel="linear", random_state=seed)
        scores10 = cross_val_score(clf10, X_shuf, y_shuf, cv=10)
        all_cv5.append(scores5.mean() * 100)
        all_cv10.append(scores10.mean() * 100)
        print(f"  Seed {seed}: CV-5={scores5.mean()*100:.2f}%  CV-10={scores10.mean()*100:.2f}%")

    print(f"\n  CV-5  mean: {np.mean(all_cv5):.2f} +/- {np.std(all_cv5):.2f}%")
    print(f"  CV-10 mean: {np.mean(all_cv10):.2f} +/- {np.std(all_cv10):.2f}%")

    # Out-of-fold predictions for confusion matrix
    clf_oof = SVC(kernel="linear", random_state=42)
    preds = cross_val_predict(clf_oof, X, y, cv=10)

    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro")
    prec, rec, f1_per, sup = precision_recall_fscore_support(y, preds, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y, preds, labels=[0, 1, 2])

    label_names = ["improved", "no_change", "worsened"]
    print(f"\n  OOF accuracy:  {100 * acc:.2f}%")
    print(f"  OOF macro F1:  {100 * macro_f1:.2f}%")
    for i, name in enumerate(label_names):
        print(f"    {name:12s}: P={prec[i]:.3f}  R={rec[i]:.3f}  F1={f1_per[i]:.3f}  n={sup[i]}")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(f"    {'':12s}  {'improved':>10s}  {'no_change':>10s}  {'worsened':>10s}")
    for i, name in enumerate(label_names):
        print(f"    {name:12s}  {cm[i][0]:>10d}  {cm[i][1]:>10d}  {cm[i][2]:>10d}")

    # Save results
    result = {
        "checkpoint": args.ckpt_path,
        "method": "cls_embedding_svm",
        "feature_dim": int(X.shape[1]),
        "n_pairs": int(X.shape[0]),
        "skipped_pairs": skipped,
        "cv5_mean": round(float(np.mean(all_cv5)), 2),
        "cv5_std": round(float(np.std(all_cv5)), 2),
        "cv10_mean": round(float(np.mean(all_cv10)), 2),
        "cv10_std": round(float(np.std(all_cv10)), 2),
        "oof_accuracy": round(100 * acc, 2),
        "oof_macro_f1": round(100 * macro_f1, 2),
        "per_class": {
            name: {"precision": round(float(prec[i]), 4),
                   "recall": round(float(rec[i]), 4),
                   "f1": round(float(f1_per[i]), 4),
                   "support": int(sup[i])}
            for i, name in enumerate(label_names)
        },
        "confusion_matrix": cm.tolist(),
    }

    out_path = os.path.join(args.output_dir, "gold_cls_svm_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
