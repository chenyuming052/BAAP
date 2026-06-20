# BAAP

Official code for **BAAP: Bidirectional Anatomy-Aware Progression Perception**.

BAAP learns and evaluates temporal chest X-ray representations from prior-current
image pairs with anatomy-aware regional features. This repository contains the
public training, preprocessing, and evaluation code used for the paper
experiments.

## Repository

The Python package is named `baap`. Legacy `MedST` names are kept only where
the code refers to the upstream MedST model/backbone for checkpoint
compatibility and attribution.

Public release scope:

- anatomy-aware temporal fine-tuning
- MS-CXR-T ROI evaluation and SVM CV-10 evaluation
- Chest ImaGenome temporal data utilities
- downstream dataset preprocessing utilities
- BAAP v1 checkpoint download helper

Not included:

- local batch launch scripts
- private machine paths
- generated results and checkpoints
- draft documents
- large datasets

## Model Checkpoint

The BAAP v1 paper-best checkpoint is distributed through GitHub Releases, not
stored in the Git repository.

Release tag:

```text
baap-v1.0
```

Release assets are intentionally minimal:

```text
baap-v1-paperbest.ckpt
SHA256SUMS
```

Direct checkpoint URL:

```text
https://github.com/chenyuming052/BAAP/releases/download/baap-v1.0/baap-v1-paperbest.ckpt
```

Download with checksum verification:

```bash
bash scripts/download_baap_v1.sh
```

See `MODEL_CARD.md` and `MODEL_LICENSE` for intended use and model weight
terms.

## Installation

This codebase was developed with Python 3.9 and PyTorch 1.12.

```bash
git clone https://github.com/chenyuming052/BAAP.git
cd BAAP

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

When running modules from source, add the repository root to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

## Data

Download the required public datasets from their official sources:

- MIMIC-CXR-JPG
- Chest ImaGenome
- MS-CXR-T
- RSNA Pneumonia Detection Challenge
- COVIDx CXR

By default, BAAP resolves data under `./data`. To use another location:

```bash
export BAAP_DATA_DIR=/path/to/data
```

A typical layout is:

```text
$BAAP_DATA_DIR/
  mimic-cxr-jpg-2.1.0/
    files/
    mimic-cxr-2.0.0-metadata.csv
  chest-imagenome-dataset-1.0.0/
    silver_dataset/
      scene_graph/
      temporal_finetuning_dataset/
        train.jsonl
        valid.jsonl
        test.jsonl
    gold_dataset/
      gold_object_comparison_with_coordinates.txt
  ms-cxr-t/
    MS_CXR_T_temporal_image_classification_v1.0.0.csv
    MS_CXR_T_temporal_sentence_similarity_v1.0.0.csv
```

### Image Path Remapping

The temporal JSONL files may contain absolute image paths generated on another
machine. If those paths no longer match your local data location, use
`--image_root_remap OLD:NEW`.

Example:

```bash
--image_root_remap /old/project/data/mimic-cxr-jpg-2.1.0:/data/mimic-cxr-jpg-2.1.0
```

This only rewrites image path prefixes at runtime; it does not modify labels,
bounding boxes, or the JSONL files.

## Training

### Full Fine-Tuning

```bash
python -m baap.experiments.code.anatomy_temporal_finetuner \
  --data_dir "$BAAP_DATA_DIR/chest-imagenome-dataset-1.0.0/silver_dataset/temporal_finetuning_dataset" \
  --pretrained_ckpt /path/to/pretrained_encoder.ckpt \
  --results_dir ./outputs \
  --experiment_name baap_finetune \
  --use_anatomy_emb \
  --roi_mode roi \
  --fusion_type concat_diff \
  --batch_size 32 \
  --gpus 1
```

Use `--image_root_remap OLD:NEW` if the JSONL image paths point to a different
root than your local MIMIC-CXR-JPG directory.

## Evaluation

### MS-CXR-T ROI and SVM CV-10

```bash
python -m baap.experiments.code.anatomy_temporal_evaluator \
  --ckpt_path /path/to/checkpoint.ckpt \
  --model_type finetuner \
  --eval_mode mscxrt_roi \
  --mscxrt_csv "$BAAP_DATA_DIR/ms-cxr-t/MS_CXR_T_temporal_image_classification_v1.0.0.csv" \
  --mimic_cxr_dir "$BAAP_DATA_DIR/mimic-cxr-jpg-2.1.0/files" \
  --scene_graph_dir "$BAAP_DATA_DIR/chest-imagenome-dataset-1.0.0/silver_dataset/scene_graph" \
  --mimic_metadata_csv "$BAAP_DATA_DIR/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-metadata.csv" \
  --roi_aggregation majority \
  --bbox_coord_mode crop224 \
  --svm_seeds 50 52 100 \
  --output_dir ./outputs/eval
```

`--bbox_coord_mode crop224` converts Chest ImaGenome 224-space bounding boxes to
the resize-256 plus center-crop-224 evaluation coordinate frame. Use
`--bbox_coord_mode raw224` only when reproducing older raw-coordinate protocol
runs.

The paper main table uses SVM CV-10 evaluation. The evaluator also reports
direct ROI accuracy, direct macro-F1, CV-5, and per-disease metrics.

### Gold Temporal Evaluation

```bash
python -m baap.experiments.code.anatomy_temporal_evaluator \
  --ckpt_path /path/to/checkpoint.ckpt \
  --model_type finetuner \
  --eval_mode gold_temporal \
  --gold_comparison_file "$BAAP_DATA_DIR/chest-imagenome-dataset-1.0.0/gold_dataset/gold_object_comparison_with_coordinates.txt" \
  --mimic_cxr_dir "$BAAP_DATA_DIR/mimic-cxr-jpg-2.1.0/files" \
  --mimic_metadata_csv "$BAAP_DATA_DIR/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-metadata.csv" \
  --output_dir ./outputs/eval
```

## Smoke Test

Use `--fast_dev_run True` to verify that the training pipeline can load data,
run one train/validation/test batch, and save outputs:

```bash
python -m baap.experiments.code.anatomy_temporal_finetuner \
  --data_dir "$BAAP_DATA_DIR/chest-imagenome-dataset-1.0.0/silver_dataset/temporal_finetuning_dataset" \
  --pretrained_ckpt /path/to/pretrained_encoder.ckpt \
  --results_dir /tmp/baap_smoke \
  --experiment_name smoke \
  --use_anatomy_emb \
  --roi_mode roi \
  --fusion_type concat_diff \
  --batch_size 2 \
  --num_workers 0 \
  --gpus 1 \
  --fast_dev_run True
```

## Checkpoints

Large pretrained and fine-tuned checkpoints are not stored in git. Put local
checkpoints outside the repository, or under an ignored output directory such as
`./outputs`.

## Acknowledgements

BAAP builds on open-source chest X-ray representation learning code and models,
including MedST and MGCA. Please cite the relevant upstream work when using this
repository.

## License

Code license pending. BAAP v1 model weights are released under `MODEL_LICENSE`.
