"""
Prepare ICG-CXR data for Stage 3 fine-tuning.

Scans the ICG-CXR directory structure, parses JSON metadata for each pair,
and produces a JSONL file compatible with the AnatomyPretrainDataset format.

Also applies MS-CXR-T decontamination (removes overlapping subjects).

Usage:
    python baap/preprocess/prepare_icg_cxr.py \
        --icg_dir data/icg-cxr-full/mimic_cxr \
        --ms_cxr_t_csv data/ms-cxr-t/MS_CXR_T_temporal_image_classification_v1.0.0.csv \
        --mimic_split_csv data/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-split.csv \
        --output_dir data/icg-cxr-full/temporal_finetuning_dataset_clean
"""

import argparse
import glob
import json
import os
from collections import Counter

import pandas as pd


def parse_icg_json(json_path: str, pair_dir: str) -> dict:
    """Parse a single ICG-CXR JSON metadata file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Find registered image files
    ref_img = os.path.join(pair_dir, os.path.basename(pair_dir) + "-ref-reg.png")
    flu_img = os.path.join(pair_dir, os.path.basename(pair_dir) + "-flu-reg.png")

    if not os.path.exists(ref_img) or not os.path.exists(flu_img):
        return None

    subject_id = data.get("subject-id", "")
    try:
        subject_id = int(subject_id)
    except (ValueError, TypeError):
        subject_id = 0

    record = {
        "patient_id": subject_id,
        "prior_image_path": ref_img,
        "current_image_path": flu_img,
        "prior_study_id": data.get("reference-study-id", ""),
        "current_study_id": data.get("followup-study-id", ""),
        "changes_of_findings": data.get("changes-of-findings", ""),
        "progression_description": data.get("progression-description", ""),
        "time_interval": data.get("time-interval", ""),
        "summary_text": data.get("progression-description", "")
                        or data.get("changes-of-findings", ""),
    }
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--icg_dir", default="data/icg-cxr-full/mimic_cxr")
    parser.add_argument("--ms_cxr_t_csv",
                        default="data/ms-cxr-t/MS_CXR_T_temporal_image_classification_v1.0.0.csv")
    parser.add_argument("--mimic_split_csv",
                        default="data/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-split.csv")
    parser.add_argument("--output_dir",
                        default="data/icg-cxr-full/temporal_finetuning_dataset_clean")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load forbidden subjects
    ms_df = pd.read_csv(args.ms_cxr_t_csv)
    forbidden = set(ms_df["subject_id"].astype(int).unique())
    print(f"MS-CXR-T forbidden subjects: {len(forbidden)}")

    # Load MIMIC-CXR split
    split_df = pd.read_csv(args.mimic_split_csv)
    subject_split = split_df.groupby("subject_id")["split"].first().to_dict()

    # Scan ICG-CXR JSON files
    json_files = glob.glob(os.path.join(args.icg_dir, "**", "*.json"), recursive=True)
    print(f"Found {len(json_files)} JSON metadata files")

    records = []
    skipped_forbidden = 0
    skipped_invalid = 0

    for jf in json_files:
        pair_dir = os.path.dirname(jf)
        record = parse_icg_json(jf, pair_dir)
        if record is None:
            skipped_invalid += 1
            continue
        if record["patient_id"] in forbidden:
            skipped_forbidden += 1
            continue
        records.append(record)

    print(f"Valid records: {len(records)}")
    print(f"Skipped (forbidden): {skipped_forbidden}")
    print(f"Skipped (invalid): {skipped_invalid}")

    # Split by MIMIC-CXR official split
    splits = {"train": [], "validate": [], "test": []}
    for r in records:
        split = subject_split.get(r["patient_id"], "train")
        splits[split].append(r)

    # Save
    for split_name, recs in splits.items():
        out_name = "valid" if split_name == "validate" else split_name
        out_path = os.path.join(args.output_dir, f"{out_name}_clean.jsonl")
        with open(out_path, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {out_name}_clean.jsonl: {len(recs)} records")

    # Stats
    stats = {
        "total_json": len(json_files),
        "valid_records": len(records),
        "forbidden_removed": skipped_forbidden,
        "train": len(splits["train"]),
        "valid": len(splits["validate"]),
        "test": len(splits["test"]),
    }
    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
