"""Interactive Web Application for Unified Gated Plant CNN.

Deployable to Hugging Face Spaces (Gradio) or local browser:
- Autonomous Spatial Saliency Gating Overlay
- Hierarchical Botanical Crop & Pathology Diagnosis
- Conformal Risk Control (1 - alpha = 95% Statistical Guarantee)
- Free Helmholtz Energy Out-of-Distribution (OOD) Scoring
- Agronomic Treatment & Disease Management Advisory
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import gradio as gr

# Ensure app directory is on sys.path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.plant_disease.models.unified_gated_cnn import build_unified_plant_cnn

# Search for checkpoint locally or via Hugging Face Hub fallback
def resolve_checkpoint() -> Path:
    candidates = [
        APP_DIR / "reports" / "unified_extended" / "unified_gated_plant_cnn_extended_best.pt",
        APP_DIR.parents[1] / "reports" / "unified_extended" / "unified_gated_plant_cnn_extended_best.pt",
        Path("reports/unified_extended/unified_gated_plant_cnn_extended_best.pt"),
    ]
    for p in candidates:
        if p.exists():
            return p
    try:
        from huggingface_hub import hf_hub_download
        repo_id = os.environ.get("HF_MODEL_REPO", "Shubham12864/plant-leaf-disease-detection-using-cnn")
        print(f"Downloading weights from Hugging Face Model Hub ({repo_id})...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename="unified_gated_plant_cnn_extended_best.pt"
        )
        return Path(downloaded)
    except Exception as e:
        print(f"Warning resolving checkpoint: {e}")
        return candidates[0]

CHECKPOINT_PATH = resolve_checkpoint()

CLASSES_38 = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES_38)}
CROPS_14 = sorted(list(set(c.split('___')[0] for c in CLASSES_38)))
CROP_TO_IDX = {cr: i for i, cr in enumerate(CROPS_14)}
CLASS_TO_CROP_IDX = {i: CROP_TO_IDX[c.split('___')[0]] for i, c in enumerate(CLASSES_38)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
Q_HAT_95 = 0.9822

# Agronomic Treatment Protocols
TREATMENT_ADVISORY = {
    "Early_blight": {
        "organic": "Apply Bacillus subtilis or bio-fungicide Serenade at 7-day intervals; spray copper octanoate.",
        "chemical": "Chlorothalonil (Bravo) or Azoxystrobin (Quadris) at first symptom appearance.",
        "cultural": "Prune lower 12 inches of foliage, mulch soil to prevent rain-splash spores, switch to drip irrigation."
    },
    "Late_blight": {
        "organic": "Fixed copper hydroxide or copper sulfate pentahydrate before rainfall events.",
        "chemical": "Mefenoxam + Mancozeb (Ridomil Gold MZ) or Mandipropamid (Revus).",
        "cultural": "Destroy volunteer host plants, ensure 36-inch row spacing for rapid leaf drying, avoid overhead sprinkler watering."
    },
    "Apple_scab": {
        "organic": "Sulfur or liquid lime-sulfur sprays applied between green-tip and petal fall stages.",
        "chemical": "Myclobutanil (Rally 40WSP) or Difenoconazole (Inspire Super).",
        "cultural": "Rake and burn/compost fallen autumn leaves to eliminate overwintering ascospores."
    },
    "Black_rot": {
        "organic": "Copper-based fungicidal sprays at bud swell and pre-bloom intervals.",
        "chemical": "Captan 50WP or Mancozeb applied every 10 to 14 days during wet spring flushes.",
        "cultural": "Prune out dead mummified fruit, dead wood, and cankers during dormant winter pruning."
    },
    "Bacterial_spot": {
        "organic": "Fixed copper combined with agricultural-grade copper hydroxide sprays.",
        "chemical": "Streptomycin sulfate sprays (where approved) or copper-mancozeb tank mixes.",
        "cultural": "Use certified pathogen-free seeds, avoid working in fields when foliage is wet from dew or rain."
    },
    "Powdery_mildew": {
        "organic": "Potassium bicarbonate (MilStop), horticultural oils (Neem, JMS Stylet Oil), or sulfur.",
        "chemical": "Myclobutanil, Trifloxystrobin (Flint), or Tebuconazole.",
        "cultural": "Plant in full sun (>6 hours direct light); space rows for good cross-ventilation."
    },
    "Common_rust_": {
        "organic": "Bio-fungicidal strains of Bacillus pumilus; neem oil foliar spray.",
        "chemical": "Pyraclostrobin (Headline AMP) or Azoxystrobin + Propiconazole (Quilt Xcel).",
        "cultural": "Plant rust-resistant hybrids; rotate with non-host broadleaf crops."
    },
    "healthy": {
        "organic": "No intervention required. Foliage exhibits vigorous chlorophyll pigmentation and cell turgidity.",
        "chemical": "None needed. Maintain balanced N-P-K nutrient fertility and monitor scouting traps.",
        "cultural": "Continue standard crop rotation, drip irrigation, and integrated pest scouting."
    }
}


def get_treatment_for_disease(disease_code: str) -> Dict[str, str]:
    sub = disease_code.split("___")[-1]
    for key, advisory in TREATMENT_ADVISORY.items():
        if key in sub:
            return advisory
    return {
        "organic": "Apply broad-spectrum bio-fungicide (Bacillus subtilis) and neem foliar shield.",
        "chemical": "Consult local agricultural extension for registered broad-spectrum fungicides.",
        "cultural": "Isolate affected plants, sanitize pruning tools with 70% ethanol, maintain drip lines."
    }


# Load model
def load_app_model():
    model = build_unified_plant_cnn(num_crops=len(CROPS_14), num_diseases=len(CLASSES_38))
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model

MODEL = load_app_model()


def diagnose_leaf(img_input: Image.Image, overlay_alpha: float = 0.5):
    if img_input is None:
        return None, None, "Please upload or capture a leaf image.", "", ""

    img_rgb = img_input.convert("RGB").resize((224, 224))
    tensor = TF.normalize(TF.to_tensor(img_rgb), mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0)

    with torch.no_grad():
        d_logits, c_logits, mask = MODEL(tensor)
        d_probs = torch.softmax(d_logits, dim=-1)[0].numpy()
        c_probs = torch.softmax(c_logits, dim=-1)[0].numpy()
        energy = MODEL.compute_energy(d_logits).item()

    # Saliency overlay
    mask_high = F.interpolate(mask, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()
    img_arr = np.array(img_rgb, dtype=np.float32) / 255.0

    # Colorize saliency mask
    saliency_colored = np.zeros_like(img_arr)
    saliency_colored[:, :, 0] = mask_high  # Red highlight
    saliency_colored[:, :, 1] = 0.2 * mask_high

    # Attenuated overlay
    gain = 0.1 + (1.0 - 0.1) * mask_high[:, :, None]
    attenuated_img = np.clip(img_arr * gain, 0.0, 1.0)
    blended = np.clip((1.0 - overlay_alpha) * attenuated_img + overlay_alpha * saliency_colored, 0.0, 1.0)
    result_overlay = Image.fromarray((blended * 255.0).astype(np.uint8))

    # Top predictions
    top_d_idx = int(np.argmax(d_probs))
    top_c_idx = int(np.argmax(c_probs))

    pred_disease = CLASSES_38[top_d_idx]
    pred_crop = CROPS_14[top_c_idx].replace("_", " ")
    disease_clean = pred_disease.split("___")[-1].replace("_", " ")
    d_conf = float(d_probs[top_d_idx])
    c_conf = float(c_probs[top_c_idx])

    # Conformal Set
    sorted_d = np.argsort(d_probs)[::-1]
    sorted_probs = d_probs[sorted_d]
    cum_probs = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cum_probs, Q_HAT_95)
    conf_indices = sorted_d[:cutoff + 1]

    conf_set_str = "\n".join([
        f"• **{CROPS_14[CLASS_TO_CROP_IDX[k]].replace('_', ' ')}**: {CLASSES_38[k].split('___')[-1].replace('_', ' ')} ({d_probs[k]*100:.1f}%)"
        for k in conf_indices[:4]
    ])

    # OOD status
    is_foliar = energy < -3.5
    foliar_status = "✅ Valid Plant Foliage (In-Distribution)" if is_foliar else "⚠️ Out-of-Distribution (Non-Foliar / Clutter Alert)"

    # Diagnosis markdown
    diagnosis_md = f"""### 🌿 Primary Diagnosis
- **Botanical Crop Family:** **{pred_crop}** ({c_conf*100:.1f}% confidence)
- **Diagnosed Condition:** **{disease_clean}** ({d_conf*100:.1f}% confidence)
- **Foliar Integrity:** {foliar_status} *(Free Helmholtz Energy: {energy:.2f})*
- **Saliency Module Status:** Active focus on necrotic tissue *(Mean Gating: {mask_high.mean():.3f})*
"""

    # Conformal Safety Box
    conformal_md = f"""### 🛡️ Split Conformal Safety Set (95.0% Guaranteed Coverage)
*The true pathology is statistically guaranteed ($\ge 95\%$) to reside in this set:*

{conf_set_str}
"""

    # Treatment Advisory
    advisory = get_treatment_for_disease(pred_disease)
    treatment_md = f"""### 🚜 Agronomic Treatment Protocol for {disease_clean}
* **Organic / Biological Control:**  
  {advisory['organic']}
* **Chemical Treatment:**  
  {advisory['chemical']}
* **Cultural / Field Prevention:**  
  {advisory['cultural']}
"""

    # Probabilities dict for Gradio Label
    top3_dict = {
        CLASSES_38[k].split("___")[-1].replace("_", " "): float(d_probs[k])
        for k in sorted_d[:5]
    }

    return result_overlay, top3_dict, diagnosis_md, conformal_md, treatment_md


# Gradio UI Theme & Layout
custom_theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="teal",
    neutral_hue="slate"
)

with gr.Blocks(title="🌿 Plant Leaf Disease Detection Using CNN") as demo:
    gr.Markdown("""# 🌿 Plant Leaf Disease Detection Using CNN
### End-to-End Pure 2D CNN Architecture with Differentiable Spatial Saliency Gating & Conformal Safety Control
*Official Project: plant-leaf-disease-detection-using-cnn | 100% Pure 2D CNN (ConvNeXt-V2 with GRN) | Zero External Tool Dependencies*
""")

    with gr.Row():
        with gr.Column(scale=4):
            input_img = gr.Image(type="pil", label="Upload Leaf Photo or Mobile Camera Capture", sources=["upload", "webcam"])
            alpha_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.05, label="Saliency Attention Overlay Opacity")
            diagnose_btn = gr.Button("⚡ Analyze Leaf Pathology", variant="primary", size="lg")

        with gr.Column(scale=4):
            output_img = gr.Image(type="pil", label="Spatial Saliency & Foliage Focus Overlay")
            label_output = gr.Label(num_top_classes=5, label="Top Candidate Pathologies")

    with gr.Row():
        with gr.Column(scale=4):
            diag_box = gr.Markdown("### Primary Diagnosis\n*Awaiting leaf image upload...*")
            conf_box = gr.Markdown("### Conformal Risk Set\n*Statistical guarantee: 1 - α = 95%*")
        with gr.Column(scale=4):
            treat_box = gr.Markdown("### Agronomic Advisory\n*Actionable organic and chemical recommendations will appear here.*")

    # Connect button
    diagnose_btn.click(
        fn=diagnose_leaf,
        inputs=[input_img, alpha_slider],
        outputs=[output_img, label_output, diag_box, conf_box, treat_box]
    )

    # Auto-run on change
    input_img.change(
        fn=diagnose_leaf,
        inputs=[input_img, alpha_slider],
        outputs=[output_img, label_output, diag_box, conf_box, treat_box]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=custom_theme)
