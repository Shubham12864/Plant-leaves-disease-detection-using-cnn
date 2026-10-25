# Frontend plan - leaf scan demo

Status: **static frontend built, backend not started.** Written 2026-10-25.

What exists now (verified in a browser): four pages served as plain files with no server.

| Page | Does |
|---|---|
| `app/static/index.html` | landing page, one recorded case inline, links into the app |
| `app/static/scan.html` | one photo: file drop, camera, staged loading, result, Grad-CAM compare slider, session strip |
| `app/static/upload.html` | batch: queue of twelve, per image progress, summary tiles, results table with expandable heat maps, CSV export |
| `app/static/inference.html` | one case in full: all 38 class scores, per stage timings, checkpoint hash and normalisation, and the exact JSON the backend must return |

`scripts/make_demo_assets.py` also records the full 38 component probability vector, the preprocess,
forward and Grad-CAM timings per case, and the checkpoint SHA-256, which is what makes the inference
page possible without a server. Shared behaviour lives in `js/ui.js` (nav, reveals, tagline, bars,
model card) and `js/api.js` (the inference client, empty until a backend exists). Every number on
screen comes from that generated file, which is regenerated from the checkpoint, never hand written.
Section 3 below describes the backend API that is still to be built; nothing on that endpoint exists
yet, and `app/README.md` documents the contract the pages already expect.
Design system: `landing-page-design` skill (elayadesign/ai-design-skills, MIT).

---

## 1. Verified facts this plan is built on

| Fact | Evidence |
|---|---|
| Checkpoint loads and is self describing | `checkpoints/efficientnet_b0/efficientnet_b0_best.pt` keys: `architecture`, `model_state_dict`, `classes`, `image_size`, `normalization_mean`, `normalization_std`, `epoch`, `val_macro_f1`, `seed` |
| Architecture / size | `efficientnet_b0`, `image_size=224`, 38 classes, class order stored in the file |
| Recorded score | `val_macro_f1 = 0.99369` (validation, single seed) |
| **Locked test already scored** (reporting pass, 2026-10-25 11:44 UTC) | `reports/model_comparison/locked_test_efficientnet_b0/test/metrics.json` - 10,709 images, macro F1 0.99314, 95% CI [0.99104, 0.99507], ECE 0.0022 |
| Grad-CAM already implemented | `src/plant_disease/gradcam.py` - `GradCAM(model, target_layer)`, `target_layer_for(model, arch)`, `overlay(cam, image_rgb, alpha, colormap)` |
| Model factory | `src/plant_disease/models/build_model(name, num_classes, pretrained=False)` |
| Preprocessing that must be reused | `Resize((image_size, image_size))` -> `ToTensor()` -> `Normalize(checkpoint normalization_mean/std)`; `data.py` uses ImageNet mean/std for every arch except SmallCNN |
| Nothing web-shaped exists yet | no `app/`, `web/`, `frontend/` directory |

The class order and the normalisation values are read **from the checkpoint**, never hard coded in the
frontend. That is why the demo cannot disagree with the evaluation scripts.

---

## 2. Architecture decision

Three honest options:

| Option | How | Verdict |
|---|---|---|
| A. Gradio / Streamlit | one Python file, framework draws the UI | fast, but near zero control over the visual system, and it looks like a notebook attachment, not a project deliverable |
| B. **FastAPI backend + static HTML/CSS/JS** | Python serves one page and one JSON endpoint; `src/plant_disease` is imported as a library | **chosen.** Real API contract, full control of the design rules, no build step, runs on a laptop CPU |
| C. Browser only (ONNX + onnxruntime-web) | model exported to ONNX, runs in the tab, no server | needs an export + numeric parity check, and Grad-CAM would have to be rewritten in JS. More failure modes, no benefit for a capstone demo |

**Chosen: B.** One process, one port, no Node toolchain, and every number on screen comes from the
same checkpoint the report already cites.

### File layout

```
app/
  main.py          FastAPI: GET /  GET /api/health  GET /api/model  POST /api/predict
  inference.py     checkpoint loading (once, at startup), preprocessing, predict, gradcam
  schemas.py       pydantic response models (the JSON contract below)
  static/
    index.html
    css/app.css
    js/app.js
  requirements.txt fastapi, uvicorn, pillow, torch, torchvision, numpy
run_app.ps1        one line launcher, documents the exact command
```

`src/plant_disease` stays the single source of truth for model building and Grad-CAM. `app/inference.py`
only adds: read one image file, apply the identical transform, run it, hand back JSON.

---

## 3. How the frontend connects to the model and Grad-CAM

Plain version, no jargon:

1. **The browser holds no model.** The page has a drop zone and a canvas. It sends the image file to
   our Python server.
2. **The server loaded the model once, at startup** - `torch.load` the checkpoint, `build_model(...)`,
   `load_state_dict(strict=True)`, `eval()`, and keep it in memory. So a prediction is one forward pass,
   not a cold start.
3. **The image is preprocessed exactly like training** - resize to 224 x 224, to tensor, normalise with
   the mean and std stored inside the checkpoint.
4. **Forward pass gives 38 numbers** (logits). Softmax turns them into probabilities. Top 3 go to the UI.
5. **Grad-CAM needs a second, backward pass.** The same image tensor is pushed through again with
   gradients enabled, we take the score of the winning class, and backpropagate *that single number*.
   Two hooks installed on the last convolution layer recorded the feature map and the gradient. The
   result is one 224 x 224 heat map, coloured and blended onto the original photo at alpha 0.45.
6. **The browser paints the answer**: class, crop, condition, probability bar, top 3, the overlay image,
   inference time, plus any warning (low confidence, tiny image, non plant input).

Because the heat map is a PNG produced by the server, the frontend is just an `<img>` tag. No model, no
GPU, no WebGL.

### JSON contract

```json
POST /api/predict   (multipart/form-data, field "file")

{
  "ok": true,
  "prediction": {
    "label": "Tomato___Late_blight",
    "crop": "Tomato",
    "condition": "Late blight",
    "probability": 0.9412,
    "confident": true
  },
  "top3": [
    {"label": "Tomato___Late_blight", "probability": 0.9412},
    {"label": "Tomato___Early_blight", "probability": 0.0413},
    {"label": "Tomato___Leaf_Mold", "probability": 0.0087}
  ],
  "gradcam": {"png_base64": "iVBORw0...", "target_layer": "features.8", "alpha": 0.45},
  "timing_ms": {"preprocess": 9, "forward": 47, "gradcam": 33, "total": 89},
  "warnings": [],
  "model": {"architecture": "efficientnet_b0", "image_size": 224, "classes": 38,
            "val_macro_f1": 0.99369, "checkpoint_sha256": "90e003cf..."}
}
```

`GET /api/model` returns the same `model` block plus the honesty text (dataset, split sizes, single
seed, field images not covered) so the UI footer and the "where this works" section are generated from
the checkpoint, not typed by hand.

### Uncertainty handling

- `confident = probability >= THRESHOLD` (start at 0.70, expose in `GET /api/model`).
- Below it the UI must say **"The model is not confident. Treat this as inconclusive."** and still show
  the top 3 and the heat map.
- Reject before inference: file over 10 MB, unreadable image, image under 64 px on the short side.
  These fail with a specific message, never a generic error.

---

## 4. Grad-CAM, explained simply

**The analogy.** A doctor says "this looks like late blight". You ask "what made you say that?" and they
point at the brown patches on the scan. **Grad-CAM is the model pointing.** It does not prove the model
reasoned like a plant pathologist, it shows which pixels pushed the answer up.

**The mechanics, in three sentences.**

1. The last convolution layer is the last place the network still has a picture shaped map (7 x 7
   instead of one number per class), so we watch it.
2. We record two things there: the map itself during the forward pass, and the gradient of the winning
   class's score during the backward pass. A channel with a large gradient is a channel that mattered.
3. Weight each channel by its average gradient, add them up, keep the positive part, stretch it to
   224 x 224, colour it, blend it over the photo.

**How to read one.**

| What you see | What it means |
|---|---|
| Heat on the lesion, veins, discoloured area | consistent with the model using the symptom |
| Heat on the pot, table, hand, background, image border | a shortcut - the model learned the setting, not the disease |
| Heat everywhere, flat | the model is not concentrating; treat the answer as weak |

That third row is not hypothetical for this project: the audit found `Soybean___healthy` predicted as
`Pepper,_bell___healthy` in 20 cases on the SmallCNN test split, which is a two-crop confusion and a
shortcut suspicion. The demo page should carry that finding next to the heat map, in one sentence.

---

## 5. The page (per the design skill)

**One offer, one audience, one action.** The offer is an instant first read on one leaf photo. The
audience is a farmer, student or reviewer holding a leaf. The action is *upload a photo*.

**Layout: C (minimal conversion page).** The product is understood the moment one image returns a
result; there is nothing to argue about before the drop zone.

### Outline

1. Nav pill (fluid island per B7): logo, "How it works", "Where it fails", "Model card"
2. Hero: headline, subheadline, primary CTA that scrolls to the drop zone, one proof line
3. **Scanner** (the product): drop zone / file picker / camera, result card, top 3 bars, Grad-CAM
   slider (raw photo <-> heat map), timing
4. **How it works** - the 3 steps from section 3, one sentence each
5. **Tagline reveal** (mandatory per B11): "It shows you the pixels behind the answer" over two lines,
   words activating on scroll
6. **Where this works, and where it does not** - the honest section: PlantVillage lab images vs field
   photos, 38 classes, one seed, not a diagnosis, test split pending
7. FAQ (8 questions)
8. Footer: model card numbers pulled from `/api/model`, links, no dead links

### Hero copy

- **Headline** (formula "{outcome} without {pain}"), max 680px, gradient text per B5:
  `A first read on a leaf, before you walk the field`
- **Subheadline:** `EfficientNet-B0 trained on 36,762 leaf photos across 38 classes. Upload one photo
  and get the prediction, the two alternatives it considered, and the pixels that decided it.`
- **CTA:** `Upload a leaf photo`
- **Proof line:** `38 classes · 43,306 development images · validation macro F1 0.9937`

### FAQ (draft, answer in the skill's voice)

1. What does the score mean? - A probability over 38 classes, not a diagnosis.
2. Why show three answers? - A confident top 1 with a close runner up is a different situation from a
   dominant one, and the gap is information.
3. Why a heat map? - So you can see whether the model looked at the leaf or at the background.
4. Does it work on my phone photo? - Partially. It was trained on lab images; field photos are harder,
   and this is stated on the page rather than hidden.
5. What is it not good at? - Unseen diseases, more than one condition in frame, blur, extreme lighting.
6. Where do the numbers come from? - The checkpoint and the frozen manifests in this repo.
7. Can I run it offline? - Yes, one Python process, no network calls after the dependencies install.
8. What should I do with the result? - Use it to decide what to photograph or ask about next; confirm
   with an extension service or agronomist.

### Visual rules I will hold to (Part B)

- Type: **Geist** only, Geist Mono for numbers and timings. No italics, no 900 weight.
- Sizes snapped to the scale: 12/14/16/18/20/24/30/36/48/60 px with paired line heights.
- Spacing only from the table: 2/4/8/12/16/24/32/40/48/64/80/96.
- Radii: Tailwind steps, inner = outer - gap when the gap is under 32px.
- Flat backgrounds only (`#000000`, `#181818`, `#1F1F1F`, `#272727`, `#313131`), the single gradient is
  the hero heading text (`#FFFFFF` -> `#9B9B9B`).
- Every transition `cubic-bezier(0.32, 0.72, 0, 1)` at 700ms; scroll reveals via `IntersectionObserver`,
  never a scroll listener.
- All six states on the drop zone and the button: hover, active, focus ring, loading skeleton, empty,
  error.
- Icons from Phosphor. No Material.
- Copy: no hyphens in sentences, no orphan words, sentence case, no "seamless" or "unleash".

---

## 6. Milestones (vertical slices, each ends runnable and verified)

| # | Slice | Done when | Cost |
|---|---|---|---|
| M1 | `app/inference.py` loads the checkpoint on CPU | `/api/health` returns 38 classes, sha256, and a forward pass runs | ~30 min |
| M2 | `POST /api/predict` returns top 3 | a test image from `data/raw/PlantVillage/test` returns the correct label | ~45 min |
| M3 | Grad-CAM wired into the response | overlay PNG for that image shows heat on the lesion | ~45 min |
| M4 | The page, section by section per A6 | hero -> drop zone -> result card, all states present | ~2 to 3 h |
| M5 | Honest section + FAQ + model card footer fed by `/api/model` | no number on screen that is not in the repo | ~1 h |
| M6 | Evidence + docs | 3 fixture images behave as promised, README section, screenshots | ~1 h |

### Test fixtures the demo has to survive

| Fixture | Expected |
|---|---|
| one PlantVillage test image | correct label, high probability, heat on the leaf |
| one blurry or badly cropped photo | lower probability, `confident: false` path renders |
| a non leaf image (desk, wall) | low probability plus the uncertainty banner, no crash |
| 12 MB file, 20 px image, corrupt bytes | specific inline error, no traceback in the UI |

---

## 7. What this demo will not claim

- Not a diagnosis. The page says so twice: once under the result, once in the FAQ.
- Not field validated. PlantVillage is controlled imagery; the PlantDoc mapping is still incomplete, so
  no field accuracy exists to quote.
- A single seed behind every number. The locked test split **is** scored now (macro F1 0.99314,
  10,709 images) and it may be quoted, but only as a reporting pass with the single seed stated. Never
  present it as a blind holdout or as a second independent estimate.
- No advisory or treatment text in v1. Retrieval grounded advisories are a later step, and they would
  need sources attached before they can appear next to a prediction.
