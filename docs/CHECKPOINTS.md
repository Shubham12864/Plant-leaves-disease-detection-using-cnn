# Checkpoint policy

## Why this file exists

`.gitignore` excludes every `*.pt`, so a fresh clone contains no weights. That is the right default for a public repository, but it means a reviewer cannot reproduce a single number without being handed binaries. This file states where the binaries live, how to verify them, and what each one is.

## The rule

1. **Binaries are not committed.** Keeps the repository small and avoids redistributing model artifacts alongside dataset terms that would need checking.
2. **Checksums are committed.** `checkpoints/SHA256SUMS` is tracked and lists every artifact the project depends on.
3. **Provenance is committed.** Every checkpoint has a recorded description of how it was produced, or an explicit statement that the provenance is partial.
4. **A number in a document must name the checkpoint it came from**, by filename plus checksum prefix.

To verify a copy you were given:

```bash
cd checkpoints && sha256sum -c SHA256SUMS
```

## Inventory

| File | Size | Architecture | Source | Provenance |
|---|---|---|---|---|
| `efficientnet_b0/efficientnet_b0_best.pt` | 15.8 MB | EfficientNet-B0, 38 classes | Kaggle, 2× T4, pilot script | `efficientnet_b0/provenance.json` — contract verified; no `dataset_fingerprint` because the pilot script did not write one |
| `resnet18/resnet18_best.pt` | 44.9 MB | ResNet18, 38 classes | Kaggle model-comparison run | partial — training script was never saved (see `docs/KNOWN_ISSUES.md` D12) |
| `resnet18/resnet18_last.pt` | 134.5 MB | ResNet18, resumable | same run | carries `resume_config` and `history`; its recorded `manifest_sha256` is not reproducible (D20) |
| `smallcnn/smallcnn_best.pt` | 1.6 MB | SmallCNN, 38 classes | Phase 2 baseline | `reports/phase2_baseline/experiment_config.json` |
| `source/model_comparison_artifacts.zip` | 167 MB | bundle | Kaggle model-comparison output | contains `resnet18/` and `smallcnn/` only; metadata says `completed_models: [smallcnn, resnet18]` |

## How to hand these to someone

Ship a single archive with a README that states the checksums. Kaggle datasets work well for this because the notebooks accept a dataset attachment; the training notebook already expects the PlantVillage dataset at a configurable path and the evaluation notebook expects the checkpoint as a second attachment.

## Open decision

Whether to commit the checkpoints through Git LFS instead of shipping an archive. LFS is cleaner for reproduction, but it adds a second delivery channel the university does not need. Not needed to finish the project; recorded so the choice is deliberate rather than accidental.
