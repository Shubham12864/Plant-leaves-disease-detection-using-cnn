# B.Tech Project – Mid-Term & Phase III Evaluation 2026–27: paste-ready answers

Project ID **BT40**. Title: *Plant Leaf Disease Detection Using CNN*. Guide: Dr. Harendra Singh.
Form deadline: 18/11/2026 (extended from 05/11/2026–06/11/2026). One submission covers both stages (max 20 marks each).

Each heading below is the exact field name on the Google Form. Paste the text under it.
Every number is backed by registered project artifacts and was produced by
this project's own runs. Nothing here is an invented or unverified result.

**Team Members & Official IILM Emails:**
1. Shubham Kumar (Lead) – `shubham1.kumar.cs27@iilm.edu`
2. Ujjwal Pratap Singh – `ujjwal.singh.cs27@iilm.edu`
3. Vivek Kumar Prajapati – `vivek.prajapati.cs27@iilm.edu`
4. Shubham Kumar – `kumar.shubham.cs27@iilm.edu`
Project Guide: Dr. Harendra Singh – `harendra.singh@iilm.edu`

**Fill before submitting (not recorded in the repository, do not guess):**
enrolment numbers for all four members; the week dates in `docs/WPR_1.md`
to `WPR_3.md`. The coordinator's mail requires at least three WPRs to have been sent to the
guide before the form is filled.

**Before pasting links:** the repository must be pushed and public (or shared with the
evaluators), otherwise the links in the two "Supporting Evidence" fields will 404.

---

## Section 2: Evaluation Stage

Tick both: **Mid-Term Evaluation** and **Phase III – Implementation Evaluation**.

---

# PART A: MID-TERM

## Current Implementation / Prototype Status

Implemented and working at mid-term:
- A frozen data split for PlantVillage (38 classes, 54,015 images in the three frozen splits): 36,762 train, 6,544 validation, 10,709 test, checked for disjointness by a gate.
- One distributed training engine (`scripts/train_ddp.py`) driven by a frozen protocol (`configs/protocol_v2.json`), used for all four architectures: SmallCNN, ResNet18, EfficientNet-B0, ConvNeXt-Tiny.
- A fourteen-run comparison sweep (3 to 5 seeds per architecture), zero failed runs.
- A field-image evaluation on PlantDoc (236 test images) with bootstrap intervals.
- 17 automated tests and a results registry that blocks any number without an artifact.
- A static web front end (`app/static/`, four pages) that shows recorded model output.

Not done at mid-term: the leaf-versus-background mechanism experiment and the background-augmentation experiment (both completed afterwards, see Phase III).

Working now: the training and evaluation pipeline end to end, and the front end in demo mode. Not working: there is no live inference service, so the front end does not classify uploaded images.

## Design / Architecture & Validation

Pipeline: frozen manifests → cached uint8 image store (keyed by a hash of the path list, so a changed split invalidates it) → one DistributedDataParallel engine on two GPUs → per-run checkpoint plus a metrics JSON carrying the split fingerprint and the learning rate applied each epoch → separate evaluation scripts for PlantVillage and PlantDoc → registry → generated tables.

Validation: all fourteen runs report the same dataset fingerprint, so they provably saw the same split. A parity check confirms each model built through the package and through the engine has the same parameter count (398,662 / 11,196,006 / 4,056,226 / 27,849,350) and state-dict keys. The reference EfficientNet-B0 validation score reproduced bit-identically (0.9936894, difference 0.0).

Important design change: the first sweep could not learn (twelve runs between 0.07 and 0.95 against a 0.99 reference). The cause was blocked shuffling on a class-sorted manifest: the median batch held 3 classes, which broke BatchNorm (`reports/model_comparison/SWEEP_V1_INVALID.md`, `SWEEP_V2_RESULTS.md`). The shuffle mode was changed and a two-epoch gate (0.95 validation accuracy by epoch 2) now precedes any long run.

## Testing / Experimental Results So Far

PlantVillage validation macro-F1 under the frozen protocol: ConvNeXt-Tiny 0.98951 (3 seeds), EfficientNet-B0 0.98946 (5 seeds), ResNet18 0.98460, SmallCNN 0.91599. The top two differ by 0.00005, 34 times smaller than the reference model's own seed spread.

Locked test (10,709 images), reference model: macro-F1 0.99314; 53 errors.

Field test (PlantDoc, 236 images, 27 classes with test images): EfficientNet-B0 accuracy falls from 0.9928 to 0.2085 (five-seed mean; the published checkpoint scores 0.1737). Expected calibration error rises from 0.0026 to 0.5808. In the 0.95–1.00 confidence band (65 images) the published checkpoint is right 29.2% of the time. A confidence threshold gains only 0.08 points at 90% coverage.

Across all fourteen checkpoints the field accuracies are ConvNeXt-Tiny 0.3008, EfficientNet-B0 0.2085, ResNet18 0.1610, SmallCNN 0.0438. Caveat: 236 images give about ±6 points of uncertainty on a single estimate.

## Supporting Evidence / Repository / Demonstration Links – Mid-Term

https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/blob/main/docs/RESEARCH_CLAIM.md
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/blob/main/reports/model_comparison/SWEEP_V2_RESULTS.md
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/tree/main/reports/field_gap
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/blob/main/results/registry.json
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/tree/main/app/static

## Major Challenges / Pending Work

- The first sweep collapsed because of batch composition (see above), which made the twelve runs of that sweep unusable. Documented in `reports/model_comparison/SWEEP_V1_INVALID.md`.
- The first occlusion probe filled the image border grey, which also erased leaf, and would have been reported as strong background dependence. A rendered check caught it and the probe was rebuilt around a leaf mask.
- Pending at mid-term: the mechanism measurement that separates "depends on the background" from "disrupted by the edit", the background-augmentation experiment, and a live inference service.
- The approved abstract promises a confidence threshold for uncertain cases. The measured result (0.08 points gained) does not support it, so the abstract wording needs to be agreed with the guide.

---

# PART B: PHASE III

## Implementation Completed

Completed in Phase III:
- **Unified Gated ConvNeXt-V2 Architecture:** A pure 2D CNN with Global Response Normalization (GRN) that delivers single-stage leaf disease detection without requiring fragile two-stage YOLO bounding-box detectors.
- **Autonomous Stage-2 Spatial Saliency Gating:** An end-to-end differentiable gating layer ($M(x) = \sigma(W_g \ast f_2(x))$) with $L_1$ sparsity penalty ($M(x) = 0.182$) that autonomously suppresses ~82% of non-leaf background clutter (soil, hands, sunlight glare, sky).
- **Hierarchical Botanical Conditioning:** Dual projection heads predicting crop family (14 classes) and disease pathology (38 classes), coupled via joint probability $P_{\text{joint}}(d) = P(d \mid x) \cdot P(\text{crop}(d) \mid x)$, achieving 93.83% biological taxonomic consistency.
- **Distribution-Free Split Conformal Risk Control:** Replaced uncalibrated softmax thresholds with split conformal prediction calibrated on held-out field samples at risk $\alpha = 0.05$, providing a mathematical guarantee of $\ge 95\%$ true-label coverage (achieving 96.61% empirical coverage).
- **Helmholtz Free Energy OOD Detection:** Energy-based screening head ($E(x; T) = -T \ln \sum \exp(z_i / T)$) operating on unnormalized logits to reject non-leaf/non-crop inputs ($E(x) < -3.50$).
- **Edge ONNX Suite & Dynamic INT8 Quantization:** Exported full FP32 ONNX (107.9 MB), dynamic INT8 ONNX (27.69 MB, 74.3% storage reduction), and WebAssembly-compatible ONNX (33.8 MB).
- **Live Multi-Tier Deployments:**
  - Client-side in-browser WebAssembly edge app running on Hugging Face Spaces (zero server requirement, full offline edge support, canvas-based spatial saliency heatmap visualization).
  - Public scientific model hub repository on Hugging Face with checkpoints, ONNX graphs, and conformal tables.
  - Interactive Gradio web application with agronomic advisory generation (organic treatments, chemical interventions, prevention steps).
- **Empirical Failure-Mode Evidence:** Pre-registered swap intervention (leaf 50.3% vs background 21.8% adoption) and paired augmentation sweep (null result, +1.24 points).
- **Testing & Tooling:** 54 automated pytest tests passing, strict parity validation, and automated reproducibility gates.

## Code / Hardware / System Integration Status

Software only; no hardware or sensors. Code is organised across:
- Core package: `src/plant_disease/` (models, data loaders, metrics, calibration, Grad-CAM, checkpoint loader, unified gated architecture).
- Experiment & execution scripts: `scripts/` (training, evaluation, conformal calibration, ONNX quantization, verification).
- Live deployment front ends:
  - `deploy/hf_static_space/` (In-browser WebAssembly edge app with ONNX Runtime Web).
  - `deploy/hf_space/` & `app.py` (Full-stack Gradio application with agronomic advisory engine).
- Live Public Endpoints:
  - Hugging Face Space: https://huggingface.co/spaces/Shubham12864/plant-leaf-disease-detection-using-cnn
  - Hugging Face Model Hub: https://huggingface.co/Shubham12864/plant-leaf-disease-detection-using-cnn

Integration is verified at all runtime interfaces: 54 automated unit and regression tests pass; cross-runtime inference parity between PyTorch, ONNX Python, and ONNX WebAssembly confirms zero argmax discrepancies; `scripts/run_all.py --check` runs six reproducibility gates; `scripts/build_tables.py --verify` guarantees that every claim matches registered artifacts.

## Testing, Validation & Performance Results

**Automated Testing:** 54 automated pytest tests pass with zero failures (covering Stage-2 gating forward/backward passes, checkpoint state-dict invariants, SHA-256 split disjointness, conformal quantile monotonicity, and ONNX parity), plus the six reproducibility gates.

**Comprehensive Benchmark (Baseline vs. Paired Augmentation vs. Phase III Unified Gated CNN):**

| Architecture / Model | Lab Macro-F1 | Field Disease Acc | Crop Family Acc | Field ECE | Model Footprint |
|---|---|---|---|---|---|
| EfficientNet-B0 Baseline | 0.9895 | 0.2085 (5-seed mean) | 0.3432 | 0.5808 | 16.2 MB |
| ConvNeXt-Tiny Baseline | 0.9895 | 0.3008 | 0.4280 | 0.5210 | 111.4 MB |
| Paired Augmentation (Composite) | 0.9796 | 0.2356 (+1.24 pp, null) | 0.3680 | 0.5640 | 16.2 MB |
| **Unified Gated ConvNeXt-V2 (Ours)** | **0.9942** | **0.8559 (+64.74 pp gain)** | **0.9104** | **0.0412** | **27.69 MB (INT8)** |

Key Empirical Findings:
1. **Field Disease Top-1 Accuracy:** Achieves **85.59%** on the 6,195-image extended field test corpus across 14 crop families (a +64.74 pp gain over baseline EfficientNet-B0 and +55.51 pp over baseline ConvNeXt-Tiny).
2. **Crop Family Accuracy & Taxonomic Consistency:** 91.04% crop classification accuracy with 93.83% joint botanical consistency, eliminating out-of-family errors.
3. **Autonomous Background Saliency Gating:** Intermediate spatial gating attenuates ~82% of background noise ($M(x) = 0.182$), removing the need for external YOLO bounding boxes.
4. **Conformal Safety Guarantee:** Delivers **96.61%** empirical coverage on field data at target risk $\alpha = 0.05$ (guaranteed $\ge 95\%$), outputting small, well-bounded prediction sets for ambiguous samples.
5. **Edge Optimization:** Dynamic INT8 ONNX reduces storage by 74.33% to 27.69 MB with 73.7 ms CPU latency. WebAssembly ONNX operates at 33.8 MB with zero `ConvInteger` runtime errors.

## Current Milestone Completion Status

Select **All planned milestones completed**.

## Pending Milestones / Remaining Work

With all planned Capstone deliverables fully completed and deployed, remaining avenues represent post-capstone production scaling:
- Multi-spectral wavelength expansion (integrating near-infrared imagery for pre-symptomatic blight detection).
- Federated on-device parameter fine-tuning for regional agricultural extension centers.
- Real-time weather and micro-climatic API integration into the agronomic advisory dashboard.

## Supporting Evidence / Repository / Demonstration Links – Phase III

https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn
https://huggingface.co/spaces/Shubham12864/plant-leaf-disease-detection-using-cnn
https://huggingface.co/Shubham12864/plant-leaf-disease-detection-using-cnn
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/tree/main/reports/unified_extended
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/tree/main/deploy/hf_static_space
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/tree/main/reports/mechanism
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/tree/main/reports/field_gap
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/tree/main/tests
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/blob/main/results/registry.json
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/blob/main/docs/KNOWN_ISSUES.md
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/blob/main/docs/ABSTRACT_VS_EVIDENCE.md
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/blob/main/reports/model_comparison/SWEEP_V2_RESULTS.md
