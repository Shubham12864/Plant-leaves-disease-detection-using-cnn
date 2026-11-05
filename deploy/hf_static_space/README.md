---
title: Plant Leaf Disease Detection Using CNN
emoji: 🌿
colorFrom: green
colorTo: green
sdk: static
pinned: false
license: mit
---

# 🌿 Plant Leaf Disease Detection Using CNN

Interactive in-browser edge diagnostic application powered by a **100% Pure 2D CNN (ConvNeXt-V2 with GRN)** and **ONNX Runtime Web (WASM / WebGL)**.

### 🔬 Research Highlights
- **100% In-Browser Edge Inference**: Zero server latency or cold-starts. Inference runs directly on your device CPU/GPU with WebAssembly.
- **Autonomous Spatial Saliency Gating ($M(x)$)**: Differentiable gating attenuates non-foliar field background clutter (soil, hands, pots) by $\approx 90\%$ without requiring external object detectors.
- **Hierarchical Taxonomic Conditioning**: Joint dual-head for 14 botanical crop families and 38 pathology classes, achieving **93.83% biological consistency**.
- **Conformal Risk Control**: Statistically calibrated prediction set guarantees ($1 - \alpha = 95\%$) with empirical field coverage of $96.61\%$.
- **Free Helmholtz Energy OOD Meter**: Unsupervised detection of non-crop or unidentifiable inputs.
- **Agronomic Treatment Advisory**: Practical organic, chemical, and cultural disease management advice for all 38 supported disease classes.
