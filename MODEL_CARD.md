# BAAP v1 Model Card

## Model

**BAAP v1** is the public checkpoint for **Bidirectional Anatomy-Aware
Progression Perception**. The model is provided to support research
reproducibility and non-clinical experimentation.

Release asset:

- `baap-v1-paperbest.ckpt`

The checkpoint is an inference-oriented PyTorch Lightning checkpoint. Training
optimizer states, scheduler states, callback states, and local machine paths
are not included.

## Intended Use

This model is released for research on chest X-ray temporal progression
perception, representation learning, and reproducibility of BAAP experiments.

It is not intended for clinical diagnosis, clinical workflow automation,
triage, treatment decisions, or direct patient care.

## Use

Download the checkpoint from the `baap-v1.0` GitHub Release and load it with
the BAAP inference/evaluation code using `--model_type finetuner`.

Benchmark details and experimental protocols are described in the accompanying
paper.

## Limitations

BAAP v1 is a research model trained and evaluated on public chest X-ray
research datasets. Performance may not generalize across institutions,
acquisition protocols, patient populations, scanner vendors, or report-labeling
procedures.

The model can make incorrect temporal progression predictions. It should not be
used as a standalone medical device or decision support system.

## Data

The release does not include chest X-ray images, reports, labels, scene graphs,
or other source datasets. Users are responsible for obtaining datasets from
their official sources and following the applicable dataset licenses, data use
agreements, and institutional requirements.

## License

The BAAP v1 model weights are released under `MODEL_LICENSE`.
