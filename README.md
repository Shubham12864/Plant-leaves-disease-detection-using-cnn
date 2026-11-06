# 🌿 Plant Leaf Disease Detection Using CNN: From Lab Accuracy to Field Reliability

A research project and deployed edge-AI system addressing the **lab-to-field generalization gap** in agricultural computer vision. Models trained solely on controlled laboratory datasets (PlantVillage) reach ~99% accuracy on lab test splits but collapse to ~20–26% on real-world agricultural field photography (PlantDoc).

This project presents a single, unified, end-to-end differentiable neural architecture—**Unified Gated ConvNeXt-V2 with Global Response Normalization (GRN)**—that closes this gap, achieving **85.59% Field Disease Accuracy**, **91.04% Crop Family Accuracy**, and **93.83% Biological Consistency** on 6,195 field images without external bounding-box object detectors.

---

## 🚀 Live Deployments & Pre-Trained Weights

- 🌐 **Live Interactive Edge Web App (PWA)**: [**huggingface.co/spaces/Shubham12864/plant-leaf-disease-detection-using-cnn**](https://huggingface.co/spaces/Shubham12864/plant-leaf-disease-detection-using-cnn)
  - *Runs 100% in-browser via WebAssembly / ONNX Runtime Web. Zero server latency, mobile camera support, and offline-capable.*
- 🤗 **Hugging Face Model Hub**: [**huggingface.co/Shubham12864/plant-leaf-disease-detection-using-cnn**](https://huggingface.co/Shubham12864/plant-leaf-disease-detection-using-cnn)
  - *Weights available: PyTorch FP32 (`.pt`), ONNX FP32 (`.onnx`), and Edge-Optimized Dynamic INT8 (`.onnx` — 27.7 MB, 74.3% compression).*

---

## 🔬 Key Research Benchmarks

| Metric / Evaluation Dimension | Lab Baseline (PlantVillage, 10.7k test) | Naive Field Baseline (PlantDoc, 236 test) | **Unified Gated ConvNeXt-V2 (Extended Field, 6,195 test)** |
| :--- | :---: | :---: | :---: |
| **Field Disease Top-1 Accuracy** | 99.28% (lab) | 20.85%–26.27% (field) | **85.59%** (field) |
| **Field Botanical Crop Family Accuracy** | — | 68.42% | **91.04%** ($k=14$) |
| **Biological Taxonomic Consistency** | — | 64.18% | **93.83%** |
| **Non-Foliar Background Clutter Suppression** | 0.0% | 0.0% | **~82%** ($M(x) = 0.182$) |
| **Conformal Coverage Guarantee ($1 - \alpha = 95\%$)** | — | — | **96.61%** empirical coverage |
| **Edge INT8 Model Footprint** | 108 MB | 108 MB | **27.7 MB** (74.3% storage reduction) |

*Note on Edge Execution: INT8 dynamic quantization yields a 74.3% storage reduction (27.7 MB) ideal for mobile WASM and memory-constrained IoT flash storage; FP32 provides lowest latency on general non-VNNI desktop CPUs (95.7 ms).*

---

## 🧠 Architectural Innovations

```
Input Image (224x224x3)
         │
         ▼
[ConvNeXt-V2 Backbone with GRN]
Stage 0 (96-d) ──► Stage 1 (192-d) ──► Stage 2 (384-d, 14x14)
                                              │
                                              ▼
                         [Differentiable Spatial Saliency Gating]
                         M(x) in [0, 1] via GroupNorm(1, C) + Conv2D
                         Controlled Attenuation: F_gated = F * (0.1 + 0.9 * M)
                         (Autonomous 90% non-foliar clutter suppression)
                                              │
                                              ▼
                                       Stage 3 (768-d)
                                              │
                                              ▼
                         [Anatomical Hierarchical Dual Head]
                         ├── Head 1: Crop Family (k=14, 91.04% Acc)
                         └── Head 2: Conditioned Pathology (k=38, 85.59% Acc)
                                              │
                                              ▼
                         [Free Helmholtz Energy Safety Head]
                         E(x) = -T * logsumexp(logits / T)
```

1. **Autonomous Spatial Saliency Gating Module**:
   Learns to distinguish foliar lesions from non-foliar clutter (soil, hands, pots, sky) directly in latent feature space:
   $$F_{\text{gated}} = F \odot \Big(\alpha + (1 - \alpha) \cdot M(x)\Big), \quad \alpha = 0.1$$
   Attenuates background noise by ~90% without requiring two-stage detector/cropper glue code.
2. **Hierarchical Taxonomic Conditioning**:
   Conditions pathology predictions on soft botanical family distributions, eliminating cross-species hallucinations (e.g. diagnosing apple scab on a tomato leaf).
3. **Split Conformal Prediction Safety Guard**:
   Provides distribution-free prediction sets $\mathcal{C}(X)$ guaranteeing $\ge 95\%$ true disease coverage with non-conformity threshold $\hat{q}_{0.95} = 0.9822$.
4. **Helmholtz Free Energy Score**:
   Measures out-of-distribution (OOD) non-crop inputs without requiring external OOD training sets ($E(x) = -T \cdot \ln \sum e^{z_i / T}$).

---

## Initial Lab-to-Field Generalization Sweep (Phase 1 Baseline)

Before developing the Unified Gated architecture, the baseline laboratory-trained models (SmallCNN, ResNet18, EfficientNet-B0, ConvNeXt-Tiny) were evaluated on real field photographs to measure the magnitude of the generalization failure. The hypotheses and their falsifiers were preregistered in [`docs/RESEARCH_CLAIM.md`](docs/RESEARCH_CLAIM.md).

| | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Field accuracy is far below lab accuracy | **Confirmed** | 0.9928 to 0.2085 over 5 seeds; all 27 mapped classes are worse |
| H2 | The gap is partly background dependence | **Partially supported** | Swapping the leaf moves the prediction 50.3% of the time, swapping the background 21.8%, against 2.6% by chance: the model uses both, with the leaf dominant |
| H3 | Background augmentation recovers part of the gap | **Not supported** | Six paired runs: field macro-F1 +1.24 pp, 95% CI [-3.32, +5.80]; in-domain cost -0.87 pp |
| H4 | Nominal accuracy does not bound deployable accuracy | **Confirmed** | Calibration error 0.0026 to 0.5808; abstaining on low-confidence images gains only 0.08 pp at 90% coverage |

Negative results are reported as results. Details: [field gap](reports/field_gap/FIELD_GAP_RESULTS.md), [mechanism](reports/mechanism/MECHANISM_RESULTS.md), [augmentation](reports/mechanism/S19_BACKGROUND_AUGMENTATION_RESULTS.md), [usefulness and calibration](reports/field_gap/USEFULNESS_RESULTS.md).

## Method

- **Data.** PlantVillage (38 classes; frozen split of 36,762 train, 6,544 validation and 10,709 test images) and PlantDoc (field photographs, mapped to PlantVillage classes). Splits are frozen and hash-checked; the test split is scored for reporting only.
- **Models.** SmallCNN, ResNet18, EfficientNet-B0 and ConvNeXt-Tiny, trained through one engine under one frozen protocol ([`configs/`](configs/)), 3 to 5 seeds each: fourteen runs, no failures. Validation macro-F1: ConvNeXt-Tiny 0.98951, EfficientNet-B0 0.98946, ResNet18 0.98460, SmallCNN 0.91599.
- **Uncertainty.** Bootstrap intervals on every reported number; paired bootstrap for comparisons.
- **Traceability.** Every reported number is registered with its artifact in [`results/registry.json`](results/registry.json), and `python scripts/build_tables.py --verify` fails if a table disagrees with it.

See [`reports/model_comparison/SWEEP_V2_RESULTS.md`](reports/model_comparison/SWEEP_V2_RESULTS.md) for the comparison.

## Current status

| Area | State |
|---|---|
| Evidence, split freeze, label mapping | Done |
| Fair four-model comparison (14 runs) | Done |
| Field gap (H1) | Done, confirmed |
| Mechanism: occlusion, leaf/background swap, Grad-CAM (H2) | Done, partially supported |
| Background augmentation (H3) | Done, not supported |
| Calibration and abstention (H4) | Done, confirmed |
| Reproducibility gates, 48 automated tests, registry check | Done |
| Unified Gated ConvNeXt-V2 Architecture | Done (85.59% Field Acc, 91.04% Crop Acc) |
| Edge ONNX INT8 Quantization & Benchmarks | Done (27.7 MB, 74.3% compression) |
| Split Conformal Risk Control (95% Safety Guard) | Done (96.61% empirical coverage) |
| Free Helmholtz Energy OOD Meter | Done |
| Live Inference Service & PWA Web App | Done — Deployed to [Hugging Face Spaces](https://huggingface.co/spaces/Shubham12864/plant-leaf-disease-detection-using-cnn) |
| Pre-Trained Checkpoints & ONNX Models | Done — Published on [Hugging Face Model Hub](https://huggingface.co/Shubham12864/plant-leaf-disease-detection-using-cnn) |
| Dissertation & Research Synthesis Report | Done |

---

## 💻 Quickstart: Edge Inference CLI & ONNX Runtime

### 1. Run Diagnostic CLI Locally
```bash
# Diagnose any field leaf photo with autonomous saliency gating & conformal prediction
python scripts/diagnose_leaf.py data/raw/PlantDoc/test/Tomato\ Early\ blight\ leaf/sample.jpg
```

### 2. Launch Gradio Diagnostic Web UI (Local & Shareable)
```bash
# Launch interactive web interface locally with public gradio.live tunnel
python app.py --share
```

### 3. Run ONNX INT8 Edge Model (WASM / Python)
```python
import numpy as np
from PIL import Image
import onnxruntime as ort

# Load lightweight 27.7 MB edge INT8 quantized model
session = ort.InferenceSession("deploy/hf_static_space/unified_gated_plant_cnn_int8.onnx")

# Preprocess image (ImageNet normalized 224x224 NCHW)
img = Image.open("leaf_sample.jpg").convert("RGB").resize((224, 224))
arr = (np.array(img, dtype=np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
tensor = np.transpose(arr, (2, 0, 1))[np.newaxis, :].astype(np.float32)

# Single forward pass: returns disease logits, crop family logits, and saliency mask
disease_logits, crop_logits, saliency_mask = session.run(None, {"input": tensor})
print(f"Predicted Disease Index: {np.argmax(disease_logits)}, Crop Index: {np.argmax(crop_logits)}")
print(f"Mean Saliency Gating M(x): {saliency_mask.mean():.3f}")
```

---

## Repository layout

| Path | Contents |
|---|---|
| `src/plant_disease/` | Unified Gated CNN, ConvNeXt-V2 with GRN, diagnostic engine, data pipeline, metrics, conformal calibration |
| `scripts/` | Training engine, ONNX export, Hugging Face deployment, diagnostic CLI, table builders, XAI visualization |
| `deploy/` | Hugging Face Model Hub README, Gradio application, and Static WebAssembly PWA suite |
| `notebooks/` | Interactive Jupyter notebooks covering all phases (Spatial Isolation to Extended Pre-Training) |
| `tests/` | Automated test suite (48 unit & integration tests) |
| `configs/` | Frozen protocol and data contract |
| `data/manifests/` | Frozen splits and the PlantDoc label mapping (images are not in git) |
| `results/registry.json` | Reported numbers with their artifacts |
| `reports/` | Empirical benchmarks, confusion matrices, XAI saliency panels, edge latency benchmarks |
| `docs/` | Hypotheses, defect register, scientific summaries |
| `app/static/` | Static front end ([`app/README.md`](app/README.md)) |
| `thesis/` | Dissertation LaTeX source |

Datasets, model weights and built documents are not stored in git. See [`data/README.md`](data/README.md) and [`configs/data_contract.yaml`](configs/data_contract.yaml) for how the data is arranged, and check each dataset's licence before redistributing images.

## Reproduce the checks

```
pip install -r requirements.lock
python -m pytest tests -q
python scripts/build_tables.py --verify
```

## Licence

Code is released under the [MIT licence](LICENSE). PlantVillage and PlantDoc are separate works under their own licences.
