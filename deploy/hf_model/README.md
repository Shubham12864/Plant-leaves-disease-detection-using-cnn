---
license: mit
library_name: onnx
tags:
- image-classification
- agricultural-ai
- plant-leaf-disease
- pure-2d-cnn
- convnext-v2
- conformal-prediction
- onnx
- edge-ai
datasets:
- plantvillage
- plantdoc
metrics:
- accuracy
- f1
model-index:
- name: plant-leaf-disease-detection-using-cnn
  results:
  - task:
      type: image-classification
      name: Agricultural Leaf Pathology Diagnosis
    dataset:
      name: PlantDoc Field Benchmark
      type: plantdoc
    metrics:
    - type: accuracy
      value: 85.59
      name: Field Disease Accuracy (10-Epoch Extended)
    - type: accuracy
      value: 91.04
      name: Field Crop Family Accuracy
    - type: consistency
      value: 93.83
      name: Biological Taxonomic Consistency
---

# 🌿 Plant Leaf Disease Detection Using CNN

**Official Capstone Project & Benchmark:** `plant-leaf-disease-detection-using-cnn`  
**Author:** Shubham Kumar (`Shubham12864`)  
**Deployment Formats:** PyTorch (`.pt`), ONNX FP32 (`.onnx`), and Edge-Optimized Dynamic INT8 (`.onnx`)

---

## 🔬 Architectural Innovations

Eliminates the limitations of discrete multi-tool pipelines (YOLO cropping + classifier glue code) by creating a **single, unified, end-to-end differentiable neural network**:

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

1. **Autonomous Spatial Gating Module:**
   Learns to distinguish healthy/diseased leaf foliage from non-foliar background (soil, hands, pots, sky) directly in latent feature space:
   $$F_{\text{gated}} = F \odot \Big(\alpha + (1 - \alpha) \cdot M(x)\Big), \quad \alpha = 0.1$$
   Attenuates background clutter by **$90\%$** without requiring external bounding-box object detectors.
2. **Hierarchical Taxonomic Conditioning:**
   Conditions disease diagnostic logits on soft crop family probabilities, achieving **$93.83\%$ biological consistency** and preventing cross-species hallucinations.
3. **Split Conformal Prediction Safety Guard:**
   Provides mathematically guaranteed prediction sets $\mathcal{C}(X)$ achieving **$96.61\%$ empirical field coverage** at $1 - \alpha = 95.0\%$ confidence.

---

## 📊 Edge Benchmarking (PyTorch vs. ONNX vs. INT8)

Evaluated on standard CPU runtime across 50 iterations:

| Runtime Engine | Precision | Model Size | Size Reduction | Mean Latency | P95 Latency | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch (Base)** | FP32 | 107.8 MB | 0.0% | 73.7 ms | 83.0 ms | 13.6 FPS |
| **ONNX Runtime** | FP32 | 107.9 MB | 0.0% | 95.7 ms | 145.6 ms | 10.5 FPS |
| **ONNX Runtime (Edge)** | **INT8** | **27.7 MB** | **74.3%** | 293.7 ms | 403.9 ms | 3.4 FPS |

*Numerical Parity Verification: Maximum absolute error between PyTorch FP32 and ONNX FP32 is $< 3.58 \times 10^{-6}$.*

---

## 💻 Quickstart: Running Inference with ONNX Runtime

```python
import numpy as np
from PIL import Image
import onnxruntime as ort

# Load Edge INT8 or FP32 model
session = ort.InferenceSession("unified_gated_plant_cnn_int8.onnx", providers=["CPUExecutionProvider"])

# Preprocess image
img = Image.open("leaf_sample.jpg").convert("RGB").resize((224, 224))
arr = np.array(img, dtype=np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
tensor = np.transpose((arr - mean) / std, (2, 0, 1))[np.newaxis, :]

# Run single forward pass
disease_logits, crop_logits, saliency_mask = session.run(None, {"input": tensor})

# Results
pred_disease_idx = np.argmax(disease_logits, axis=-1)[0]
pred_crop_idx = np.argmax(crop_logits, axis=-1)[0]
print(f"Predicted Disease Index: {pred_disease_idx}, Crop Index: {pred_crop_idx}")
```

---

## 📜 Citation
```bibtex
@article{kumar2026plantcnn,
  title={Plant Leaf Disease Detection using Pure 2D CNNs with Differentiable Spatial Gating and Conformal Risk Control},
  author={Kumar, Shubham},
  year={2026}
}
```
