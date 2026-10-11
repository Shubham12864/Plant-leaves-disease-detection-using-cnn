---
title: Updated Literature Survey — Plant Leaves Disease Detection Using CNN
project: Plant leaves disease detection using CNN
type: literature-review
survey_count_reported: 21
status: updated from supplied review PDF; citation audit in progress
tags:
  - capstone
  - plant-disease
  - literature-review
---

# Updated Literature Survey

## Purpose

The Phase I records say that the team reviewed 21 papers. The supplied review PDF now provides notes for all 21 entries. This file keeps those entries and checks the details that can be checked from original or publisher sources.

The supplied PDF does not contain a full reference list. Some entries have only an author name, journal name, year, or model label. I mark those entries as **citation incomplete**. I do not guess their full title, author list, venue, DOI, dataset, or result.

This is an evidence summary. It is not a promise that the project will outperform every paper. Published scores use different datasets, splits, classes, and test conditions. Compare scores only when the evaluation conditions match.

## Main findings

1. **Controlled images can produce very high scores.** PlantVillage uses centered leaves and plain backgrounds. Mohanty et al. reported 99.35% accuracy on their held-out PlantVillage test set. Their separate test on images from other sources reached 31.4%. These are different test conditions, so they show a domain gap rather than a direct like-for-like comparison. [Mohanty et al. (2016)](https://doi.org/10.3389/fpls.2016.01419)
2. **Field testing matters.** Kamal et al. reported 98.34% for Reduced MobileNet on a PlantVillage subset, but 36.03% when MobileNet was tested on images taken in different conditions. PlantDoc was created to support detection in more varied visual scenes. [Kamal et al. (2019)](https://doi.org/10.1016/j.compag.2019.104948), [PlantDoc dataset paper](https://arxiv.org/abs/1911.10317)
3. **Complex models do not remove evaluation risk.** Attention, transformers, ensembles, and generated images can raise reported scores. The score still depends on how data was split and whether the test images represent the intended setting.
4. **Efficiency and interpretability have practical value.** Depthwise separable CNNs target smaller devices. Grad-CAM and other explanation methods can show which image regions influenced a prediction. These methods need their own evaluation. A heatmap does not prove that a prediction is correct.
5. **Class-level performance matters.** A single accuracy score can hide weak classes. Macro-F1, balanced accuracy, per-class precision and recall, and a confusion matrix give a fuller view.

## Review of the 21 entries

The “review note” column reports what the supplied PDF says. The “audit” column distinguishes verified publication facts from claims that still need the original paper or a complete citation.

| # | Study named in the supplied PDF | Review note | Audit and project relevance |
|---:|---|---|---|
| 1 | Ester (2024), review of deep-learning methods | Reviews CNNs and transformers. Notes overfitting and narrow datasets. | **Citation incomplete.** The PDF gives no title, venue, or stable link. Do not cite as a verified paper until the exact reference is supplied. |
| 2 | Mandlik & Birgale (2025), *Current Agriculture Research Journal* | Custom batch-normalized CNN for ten maize, potato, and soybean conditions. PDF reports 95% accuracy and class F1 from 0.84 to 0.96. | **Publication verified; metric details need review.** The publisher abstract reports 95%. Check the paper's result tables for the exact overall metric, classwise values, and environmental test protocol before using the score in a comparison. The article notes limited dataset size and environmental variety. [Publisher article](https://doi.org/10.12944/CARJ.13.3.15) |
| 3 | Johari, Chavan, Kshirsagar & Parkhi (2021), IJCRT | Uses public leaf images and examines learned features such as lesion color and texture. | **Citation incomplete.** The supplied PDF gives no paper title, volume, pages, or DOI. Treat the interpretation as a review note, not verified evidence. |
| 4 | Wagh, Gavali, Patil & Sharma (2025), IJRAR | Adapts a pretrained backbone to five crops and uses heavy augmentation. The PDF says the training and validation curves were stable. | **Citation incomplete.** No title, volume, pages, or stable URL appears in the supplied PDF. |
| 5 | Guo (2023), *Highlights in Science, Engineering and Technology* | Standard convolution and pooling stack. The PDF reports 92.23%. | **Citation incomplete.** Dataset, split, metric protocol, and full citation are missing. Do not compare this number with our test score yet. |
| 6 | Ghosh (2024), JETIR | Uses VGG16 and ResNet34 feature extraction across 38 conditions and 14 species. | **Citation incomplete.** Full paper title, dataset version, and test method are missing. |
| 7 | Patil, Vishwakarma Institute of Technology | Applies a similar network to pepper, tomato, and potato PlantVillage classes. The note says data quality limits model performance. | **Citation incomplete.** The authors, year, title, and result are not fully identified. |
| 8 | Shelar, Shinde, Sawant, Dhumal & Fakir, RAIT Navi Mumbai | Separates healthy and infected tissue. The PDF reports average precision of 96.3% and class results from 91% to 98%. | **Citation incomplete.** The title, publication year, dataset, and precision definition are missing. |
| 9 | Murugavalli & Gopi, *Scientific Reports* | PLA-ViT uses a vision transformer, bilateral filtering, and a proposed sensor connection. | **Publication verified.** The paper is “Plant leaf disease detection using vision transformers for precision agriculture,” *Scientific Reports* 15, 22361 (2025), DOI 10.1038/s41598-025-05102-0. The supplied PDF's claim that it beats CNN baselines on accuracy, localization, and speed needs the paper's exact table and protocol before reuse. [Publisher article](https://doi.org/10.1038/s41598-025-05102-0) |
| 10 | Pandian et al. (2022), *Applied Sciences* | 14-layer CNN; the PDF says 147,500 images, 59 classes, and 99.9655%. It describes GAN and style-transfer augmentation. | **Publication verified; important mismatch.** The original article describes 147,500 images for 42 diseases across 16 plants and reports 99.9655% on the training dataset. The PDF's 59-class count is inconsistent with the primary paper, and a training score is not a held-out test score. [Publisher article](https://doi.org/10.3390/app12146982) |
| 11 | Hassan, Maji, Jasiński et al. (2021), *Electronics* | Compares transfer-learning CNNs, including InceptionV3, InceptionResNetV2, MobileNetV2, and EfficientNetB0. | **Publication verified.** The full title is “Identification of Plant-Leaf Diseases Using CNN and Transfer-Learning Approach,” *Electronics* 10(12), 1388. The PDF's per-model scores need direct checking against the paper before citation. [Publisher article](https://doi.org/10.3390/electronics10121388) |
| 12 | Hingolikar, Gadkar & Mahajan (2023), *Kronika Journal* | Uses image processing with a neural model for paddy leaves. The PDF reports up to 99.6% on tomato comparisons. | **Citation incomplete and scope unclear.** The source note does not identify the title, dataset, or why a paddy study includes tomato results. Verify before use. |
| 13 | Sun, Li, Qu et al. (2022), *Cognitive Robotics* | FL-EfficientNet combines architecture search, attention, and focal loss. The PDF reports 99.72% and faster convergence. | **Publication verified.** The paper tests ten disease categories across five crops on the New Plant Diseases Dataset. The reported 99.72% is dataset-specific and does not establish field performance. [Publisher article](https://doi.org/10.1016/j.cogr.2022.07.001) |
| 14 | Flavian & Ramachandran (2026), *Artificial Intelligence Review* | Review of 137 papers on detection and severity grading. It identifies the lab-to-field gap. | **Publication located; full audit pending.** The supplied PDF omits the title and DOI. A matching 2026 review record exists under DOI 10.1007/s10462-026-11659-7. Confirm that this is the intended review and check its methods before citing the 137-paper count. [Publisher DOI record](https://doi.org/10.1007/s10462-026-11659-7) |
| 15 | Vinitha et al. (2024), IJSR | PyTorch CNN on 39 PlantVillage classes, with a reported score near 91%. | **Publication located.** The title is “Plant Leaf Disease Detection Using Convolutional Neural Network,” *International Journal of Science and Research*, 13(4), 2024. The supplied PDF gives only one author surname; confirm the complete author list and exact result from the article. [Journal PDF](https://www.ijsr.net/archive/v13i4/SR24422151104.pdf) |
| 16 | Mirza & Talupula, IJERT, Krishna University | Adapts MobileNetV2 with small-device use as a goal. | **Citation incomplete.** The year, title, issue, dataset, and measured model size or latency are absent. |
| 17 | Sharif, Khalil, Sajid et al. (2026), *Frontiers in Plant Science* | EnviroSpect-guided ConvMixer-ViT for potato blight. The PDF reports 99% on PlantVillage and 98% on field images. | **Publication verified; exact metrics need review.** The published title is “EnviroSpect-guided ConvMixer-ViT framework for environment-robust potato leaf disease detection,” DOI 10.3389/fpls.2026.1886452. Check the original tables for the stated accuracy, F1, and image domains before comparison. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/42698553/) |
| 18 | Das, Prethija & Sudha (2026), *Frontiers in Artificial Intelligence* | LeafFusionNet combines CNN, ViT, LeafTAM attention, and Gabor features. The PDF reports 99.33% on PlantVillage. | **Publication verified.** The model reports 99.33% accuracy on PlantVillage. The article's use of “unseen data” must not be read as proof of external field generalization without a separate field test. [Publisher article](https://doi.org/10.3389/frai.2026.1773329) |
| 19 | Sujatha, Krishnan & Chatterjee (2025), *Scientific Reports* | Compares CNN feature extraction plus classical classifiers across banana, custard apple, fig, and potato datasets. | **Publication verified; summary correction.** The paper title is “Advancing plant leaf disease detection integrating machine learning and deep learning,” *Scientific Reports* 15, 11552. The supplied PDF's claim of 98% CRNN-RCNN banana results does not match the abstract's reported Banana result of 91.9% for InceptionV3+SVM. Check the full paper before quoting. [Publisher article](https://doi.org/10.1038/s41598-024-72197-2) |
| 20 | Hoffmann, Mai & Buettner (2026), *PLOS ONE* | Creates a 2,801-image, five-class radish dataset and evaluates Gaussian filtering with five-fold validation. The PDF reports 99.86%. | **Publication verified.** The DOI is 10.1371/journal.pone.0353374. Interpret its score as a result on that radish dataset under five-fold validation, not a cross-crop result. [Publisher article](https://doi.org/10.1371/journal.pone.0353374) |
| 21 | Srinivasu, Anirudh et al. (2026), *Frontiers in Plant Science* | The PDF describes an ensemble, heatmap explanations, and a YOLOv12 detector. It reports 99.94% accuracy, 99.5% mAP, and 99.18% combined accuracy. | **Citation not verified.** The PDF does not provide the article title or DOI, and searches did not identify a matching primary paper. Do not cite these figures until the exact source is supplied. |

## What this evidence means for our Phase 2 work

| Literature concern | Phase 2 choice | Current status |
|---|---|---|
| Very high benchmark scores can hide data overlap | Keep source test split separate. Remove 290 PlantVillage training rows that match the published test by group ID or exact hash. Save the frozen split manifests and audit. | Done. No exact-hash or group overlap remains in the saved audit. |
| Accuracy hides weak classes | Report accuracy with balanced accuracy, macro-F1, per-class scores, and a confusion matrix. | Done for the PlantVillage test set. |
| Complex methods are hard to compare | Start with a clear SmallCNN baseline before adding attention or transformer branches. | Done. Four convolution stages use 32, 64, 128, and 256 channels. |
| Repeated validation can bias the test | Use validation macro-F1 to select the checkpoint. Evaluate the locked test set once after selection. | Done for the recorded run. |
| Controlled data does not prove field performance | Keep PlantDoc separate. Map its labels carefully before any external evaluation. | Dataset retained; model has not been evaluated on PlantDoc. |
| Reproducibility is often incomplete | Save seed, split manifests, configuration, training history, checkpoint, and evaluation outputs. | Saved in the repository and Kaggle result bundle. |

### Phase 2 experiment at a glance

- **Primary dataset:** PlantVillage, 38 classes.
- **Frozen split:** 36,762 train images, 6,544 validation images, and 10,709 test images. The split removes 290 source-train rows that match the test by group or exact hash.
- **Model:** SmallCNN with channels 32, 64, 128, and 256. It is a baseline, not the final model.
- **Training:** 224 × 224 input, AdamW, cross-entropy loss, batch size 64, up to 12 epochs, mixed precision, and two Tesla T4 GPUs.
- **Training transforms:** horizontal flip, rotation, and modest color jitter. Validation and test images are not augmented.
- **Checkpoint rule:** highest validation macro-F1. The recorded best checkpoint is epoch 12.
- **Held-out PlantVillage test:** accuracy 96.95%, balanced accuracy 96.02%, macro-F1 96.07% on 10,709 images.
- **Limit:** These scores measure the PlantVillage test split. They do not establish field performance or show that the model beats the cited papers.

The result bundle contains the per-epoch training history, training curves, class report, confusion matrix, experiment configuration, and selected checkpoint. See the [Phase 2 results summary](../phase2_baseline/PHASE2_RESULTS_SUMMARY.md) and the [public code repository](https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn).

## Research gap and next work

The main gap is reliable performance on images from new farms, phones, crops, and lighting conditions. The current baseline provides a reproducible PlantVillage reference. It does not solve that field-generalization problem.

1. Review error patterns in the confusion matrix and per-class report. Keep the PlantVillage test scores fixed.
2. Write and review a PlantVillage-to-PlantDoc class map. Report only confirmed shared classes. Keep unmatched classes visible in the report.
3. Evaluate PlantDoc as an external test. Do not use its test images to select a model or tune thresholds.
4. Compare a small set of baselines under the same split and metrics. Include a pretrained lightweight backbone if compute allows.
5. Add calibration and confidence reporting. Test whether the model can abstain when the image is out of scope.
6. Add Grad-CAM as an explanation aid and review whether the heatmaps focus on symptoms. Do not treat heatmaps as proof of correctness.
7. Consider severity grading, leaf localization, and a simple web demo after the external evaluation.
8. Consider verified agricultural advice only after agronomists or trusted extension sources review the content. Do not let the model give pesticide directions from an unverified label.

## Source and integrity notes

- The attached **Review of Literature: Plant leaf disease detection using CNN — notes on twenty-one papers** is the source for the 21-entry notes.
- Publisher or primary links appear beside entries where the identity could be matched.
- “Reported” means the value appears in the supplied survey or paper. It does not mean this team reproduced it.
- Citation-incomplete entries remain in the register so the survey count stays visible. They are not treated as verified evidence.
- Keep dataset, split, class count, metric, and test conditions attached to every reported number.
- Do not claim that this work exceeds all 21 papers. A fair comparison needs the same data, classes, split rules, test domain, and metrics.
