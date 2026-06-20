# BAAP v1.0

This release provides the BAAP v1 public checkpoint for research
reproducibility.

## Assets

- `baap-v1-paperbest.ckpt`
- `SHA256SUMS`

## Checkpoint

- Model: BAAP
- Model type: `finetuner`
- Checkpoint file: `baap-v1-paperbest.ckpt`
The checkpoint is an inference-oriented PyTorch Lightning checkpoint. Training
optimizer states, scheduler states, callback states, and local machine paths
are not included.

## Verification

After downloading all assets:

```bash
sha256sum -c SHA256SUMS
```

Expected checksum for the checkpoint:

```text
761f6781401e866f53561aaac3b704a27e97c531e2bd9ce089678b96098c8f69  baap-v1-paperbest.ckpt
```

## Use

See `MODEL_CARD.md` for intended use and limitations. Model weights are
released under `MODEL_LICENSE`.
