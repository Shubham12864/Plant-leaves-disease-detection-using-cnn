---
title: Plant Leaf Disease Detection Using CNN
emoji: 🌿
colorFrom: green
colorTo: emerald
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
short_description: Pure 2D CNN (ConvNeXt-V2) with Differentiable Saliency Gating
---

# 🌿 Plant Leaf Disease Detection Using CNN

Interactive diagnostic web application powered by a **100% pure 2D CNN (ConvNeXt-V2 with GRN)**.

### 🔬 Core Scientific Features
- **Autonomous Spatial Saliency Gating ($M(x)$)**: Attenuates non-foliar field background clutter (soil, hands, pots) by $\approx 90\%$ without requiring external object detectors or multi-stage pipelines.
- **Hierarchical Taxonomic Conditioning**: Joint dual-head for 14 botanical crop families and 38 pathology classes, achieving **93.83% biological consistency**.
- **Conformal Risk Control**: Statistically calibrated prediction set guarantees ($1 - \alpha = 95\%$) with empirical field coverage of $96.61\%$.
- **Free Helmholtz Energy OOD Meter**: Unsupervised detection of non-crop or unidentifiable inputs.
- **Agronomic Treatment Advisory**: Practical organic, chemical, and cultural disease management advice for all 38 supported disease classes.
