"""
Decontaminate Chest ImaGenome temporal finetuning dataset.

Problem: 78.1% of MS-CXR-T evaluation subjects appear in the ImaGenome
training set. Both datasets derive from MIMIC-CXR. Training on these
subjects would invalidate all evaluation results.

Solution:
1. Extract all subject IDs from MS-CXR-T evaluation benchmark
2. Remove ALL temporal pairs from ImaGenome where patient_id overlaps
3. Re-split remaining data using MIMIC-CXR official patient-level split
4. Output clean files: train_clean.jsonl, valid_clean.jsonl, test_clean.jsonl
5. Verify zero overlap with MS-CXR-T

Usage:
    python medst/preprocess/decontaminate_imagenome.py \
        --ms_cxr_t_csv data/ms-cxr-t/MS_CXR_T_temporal_image_classification_v1.0.0.csv \
        --imagenome_dir data/chest-imagenome-dataset-1.0.0/temporal_finetuning_dataset \
        --mimic_split_csv data/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-split.csv \
        --output_dir data/chest-imagenome-dataset-1.0.0/temporal_finetuning_dataset_clean
"""

import argparse
import json
import os
from collections import Counter

import pandas as pd


def load_ms_cxr_t_subjects(csv_path: str) -> set:
    """Extract all unique subject IDs from MS-CXR-T evaluation benchmark."""
    df = pd.read_csv(csv_path)
    subjects = set(df["subject_id"].astype(int).unique())
    print(f"[MS-CXR-T] Loaded {len(subjects)} unique evaluation subjects")
    return subjects


def load_mimic_split(csv_path: str) -> dict:
    """Load MIMIC-CXR official patient-level split mapping.

    Returns dict: subject_id -> split ('train' | 'validate' | 'test')
    """
    df = pd.read_csv(csv_path)
    # MIMIC-CXR assigns all images of a patient to the same split
    subject_split = df.groupby("subject_id")["split"].first().to_dict()
    split_counts = Counter(subject_split.values())
    print(f"[MIMIC-CXR split] {len(subject_split)} subjects: {dict(split_counts)}")
    return subject_split


def load_jsonl(path: str) -> list:
    """Load all records from a JSONL file."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list, path: str):
    """Save records to a JSONL file."""
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def decontaminate(
    ms_cxr_t_csv: str,
    imagenome_dir: str,
    mimic_split_csv: str,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load MS-CXR-T evaluation subjects (the "forbidden" set)
    forbidden_subjects = load_ms_cxr_t_subjects(ms_cxr_t_csv)

    # Step 2: Load MIMIC-CXR official split
    subject_split = load_mimic_split(mimic_split_csv)

    # Step 3: Load ALL ImaGenome temporal pairs from all splits
    all_records = []
    for split_name in ["train", "valid", "test"]:
        path = os.path.join(imagenome_dir, f"{split_name}.jsonl")
        if os.path.exists(path):
            records = load_jsonl(path)
            print(f"[ImaGenome] Loaded {len(records)} records from {split_name}.jsonl")
            all_records.extend(records)
        else:
            print(f"[ImaGenome] WARNING: {path} not found, skipping")

    print(f"[ImaGenome] Total records before decontamination: {len(all_records)}")

    # Step 4: Remove records with forbidden subjects
    clean_records = []
    removed_count = 0
    for record in all_records:
        patient_id = int(record["patient_id"])
        if patient_id in forbidden_subjects:
            removed_count += 1
        else:
            clean_records.append(record)

    print(f"[Decontamination] Removed {removed_count} records "
          f"({removed_count / len(all_records) * 100:.1f}%) due to MS-CXR-T overlap")
    print(f"[Decontamination] Remaining: {len(clean_records)} records")

    # Step 5: Re-split using MIMIC-CXR official split
    # Map ImaGenome 'validate' to MIMIC-CXR 'validate' and so on
    split_map = {"train": [], "validate": [], "test": []}
    unknown_split = []

    for record in clean_records:
        patient_id = int(record["patient_id"])
        split = subject_split.get(patient_id, None)
        if split is not None:
            split_map[split].append(record)
        else:
            # Patient not in MIMIC-CXR split file - assign to train by default
            unknown_split.append(record)
            split_map["train"].append(record)

    if unknown_split:
        print(f"[WARNING] {len(unknown_split)} records have patients not in "
              f"MIMIC-CXR split file, assigned to train")

    for split_name, records in split_map.items():
        out_name = "valid" if split_name == "validate" else split_name
        out_path = os.path.join(output_dir, f"{out_name}_clean.jsonl")
        save_jsonl(records, out_path)
        unique_patients = len(set(r["patient_id"] for r in records))
        print(f"[Output] {out_name}_clean.jsonl: {len(records)} records, "
              f"{unique_patients} unique patients")

    # Step 6: Verification - assert zero overlap
    all_clean_patients = set()
    for records in split_map.values():
        for r in records:
            all_clean_patients.add(int(r["patient_id"]))

    overlap = all_clean_patients & forbidden_subjects
    assert len(overlap) == 0, (
        f"DATA LEAKAGE DETECTED! {len(overlap)} subjects overlap with MS-CXR-T: "
        f"{sorted(list(overlap))[:10]}..."
    )
    print(f"\n[VERIFICATION PASSED] Zero overlap between "
          f"{len(all_clean_patients)} training patients and "
          f"{len(forbidden_subjects)} MS-CXR-T evaluation subjects")

    # Step 7: Save statistics
    stats = {
        "ms_cxr_t_subjects": len(forbidden_subjects),
        "original_total_records": len(all_records),
        "removed_records": removed_count,
        "clean_total_records": len(clean_records),
        "train_records": len(split_map["train"]),
        "valid_records": len(split_map["validate"]),
        "test_records": len(split_map["test"]),
        "train_patients": len(set(r["patient_id"] for r in split_map["train"])),
        "valid_patients": len(set(r["patient_id"] for r in split_map["validate"])),
        "test_patients": len(set(r["patient_id"] for r in split_map["test"])),
        "overlap_with_ms_cxr_t": 0,
    }
    stats_path = os.path.join(output_dir, "decontamination_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[Stats] Saved to {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decontaminate ImaGenome dataset")
    parser.add_argument(
        "--ms_cxr_t_csv",
        default="data/ms-cxr-t/MS_CXR_T_temporal_image_classification_v1.0.0.csv",
    )
    parser.add_argument(
        "--imagenome_dir",
        default="data/chest-imagenome-dataset-1.0.0/temporal_finetuning_dataset",
    )
    parser.add_argument(
        "--mimic_split_csv",
        default="data/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-split.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="data/chest-imagenome-dataset-1.0.0/temporal_finetuning_dataset_clean",
    )
    args = parser.parse_args()
    decontaminate(
        ms_cxr_t_csv=args.ms_cxr_t_csv,
        imagenome_dir=args.imagenome_dir,
        mimic_split_csv=args.mimic_split_csv,
        output_dir=args.output_dir,
    )
