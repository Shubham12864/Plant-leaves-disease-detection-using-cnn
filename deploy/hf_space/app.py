#!/usr/bin/env python3
"""Interactive Web Application for Unified Gated Plant CNN.

Deployable locally or on Hugging Face Spaces:
- Unified inference via PlantDiagnosticEngine (zero logic duplication)
- Aspect-ratio preserving leaf preprocessing
- Autonomous Spatial Saliency Gating Overlay
- Hierarchical Botanical Crop & Pathology Diagnosis
- Calibrated Split Conformal Risk Control (q_hat = 0.9822)
- Helmholtz Free Energy Out-of-Distribution (OOD) Scoring
- Pathogen-Accurate Agronomic Treatment Advisory (fungal, bacterial, viral, mite)
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import gradio as gr

# Robust Path & Package Resolution
FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR.parents[1] if FILE_DIR.name == "hf_space" else FILE_DIR

for path_candidate in (FILE_DIR, FILE_DIR / "src", REPO_ROOT, REPO_ROOT / "src"):
    str_p = str(path_candidate)
    if str_p not in sys.path:
        sys.path.insert(0, str_p)

try:
    from plant_disease.diagnostic_engine import PlantDiagnosticEngine
except ImportError:
    from src.plant_disease.diagnostic_engine import PlantDiagnosticEngine


def resolve_checkpoint() -> Path:
    """Resolve local checkpoint or download from Hugging Face Hub if absent."""
    candidates = [
        REPO_ROOT / "reports" / "unified_extended" / "unified_gated_plant_cnn_extended_best.pt",
        FILE_DIR / "reports" / "unified_extended" / "unified_gated_plant_cnn_extended_best.pt",
        FILE_DIR / "unified_gated_plant_cnn_extended_best.pt",
        Path("reports/unified_extended/unified_gated_plant_cnn_extended_best.pt"),
    ]
    for p in candidates:
        if p.exists():
            return p
    from huggingface_hub import hf_hub_download

    repo_id = os.environ.get("HF_MODEL_REPO", "Shubham12864/plant-leaf-disease-detection-using-cnn")
    print(f"Downloading weights from Hugging Face Model Hub ({repo_id})...")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename="unified_gated_plant_cnn_extended_best.pt",
    )
    return Path(downloaded)


def load_engine() -> PlantDiagnosticEngine:
    ckpt_path = resolve_checkpoint()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    print(f"Loading Unified Gated ConvNeXt-V2 Engine from {ckpt_path}...")
    return PlantDiagnosticEngine(checkpoint_path=ckpt_path, q_hat=0.9822, energy_threshold=-3.5)


ENGINE = load_engine()


def diagnose_leaf(img_input: Image.Image):
    if img_input is None:
        return None, None, "### 📸 Please upload or photograph a leaf to begin diagnosis.", "", ""

    result = ENGINE.diagnose_image(img_input)
    overlay_alpha = 0.45
    img_arr = np.array(result["preprocessed_image"], dtype=np.float32) / 255.0
    mask_high = result["saliency_mask"]

    # Colorize saliency mask: vibrant amber/emerald lesion highlight
    saliency_colored = np.zeros_like(img_arr)
    saliency_colored[:, :, 0] = np.clip(mask_high * 1.2, 0.0, 1.0)
    saliency_colored[:, :, 1] = np.clip(mask_high * 0.9, 0.0, 1.0)
    saliency_colored[:, :, 2] = np.clip(mask_high * 0.1, 0.0, 1.0)

    blended = np.clip((1.0 - overlay_alpha) * img_arr + overlay_alpha * saliency_colored, 0.0, 1.0)
    result_overlay = Image.fromarray((blended * 255.0).astype(np.uint8))

    foliar_status = "✅ Valid Plant Foliage" if result["is_foliar_leaf"] else "⚠️ Non-Foliar / Clutter Alert"
    bg_suppression = (1.0 - result["saliency_mean"]) * 100.0

    diagnosis_md = f"""### 🌿 Primary Diagnosis
- **Botanical Crop Family:** **{result['predicted_crop']}** ({result['crop_confidence']*100:.1f}% confidence)
- **Diagnosed Condition:** **{result['disease_clean_name']}** ({result['disease_confidence']*100:.1f}% confidence)
- **Foliar Check:** {foliar_status} *(Suppressed ~{bg_suppression:.0f}% background noise | {result['inference_ms']:.1f} ms)*
"""

    conf_set_str = "\n".join([
        f"• **{item['crop']}**: {item['clean_name']} ({item['confidence']*100:.1f}%)"
        for item in result["conformal_prediction_set"]
    ])
    conformal_md = (
        f"*Calibrated Split Conformal Safe Set ($\\hat{{q}} = {ENGINE.q_hat:.4f}$, "
        f"total set size = {result['conformal_set_size']} classes):*\n\n{conf_set_str}"
    )

    advisory = result["treatment_advisory"]
    treatment_md = f"""### 🚜 Treatment Advisory for {result['disease_clean_name']}
- **🌱 Organic / Biological Control:**  
  {advisory['organic']}
- **💊 Chemical / Vector Treatment:**  
  {advisory['chemical']}
- **✂️ Cultural / Field Prevention:**  
  {advisory['cultural']}
"""

    top3_dict = {
        item["disease"].replace("___", " → ").replace("_", " "): float(item["confidence"])
        for item in result["top3_candidates"]
    }

    return result_overlay, top3_dict, diagnosis_md, conformal_md, treatment_md


custom_theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="teal",
    neutral_hue="slate",
)

example_dir = FILE_DIR / "examples"
if not example_dir.exists():
    example_dir = REPO_ROOT / "examples"

example_files = sorted([[str(p)] for p in example_dir.glob("*.jpg")]) if example_dir.exists() else []

with gr.Blocks(title="🌿 Plant Leaf Disease Detection") as demo:
    gr.Markdown("""# 🌿 Plant Leaf Disease Detection & Treatment Guide
Upload or take a photo of a plant leaf to get an instant diagnosis, lesion focus heatmap, and pathogen-specific treatment recommendations.
""")

    with gr.Row():
        with gr.Column(scale=5):
            input_img = gr.Image(type="pil", label="Upload Leaf Photo or Camera Snapshot", sources=["upload", "webcam"])
            diagnose_btn = gr.Button("🔍 Diagnose Leaf", variant="primary", size="lg")

            if example_files:
                gr.Markdown("#### 🧪 Test With Sample Leaves:")
                gr.Examples(
                    examples=example_files,
                    inputs=[input_img],
                    outputs=[],
                    fn=diagnose_leaf,
                    cache_examples=False,
                )

        with gr.Column(scale=5):
            diag_box = gr.Markdown("### 🌿 Diagnosis Results\n*Upload a photo or click a sample to see results.*")
            label_output = gr.Label(num_top_classes=3, label="Top Diagnoses & Confidence")
            output_img = gr.Image(type="pil", label="Lesion Focus Saliency Heatmap")
            treat_box = gr.Markdown("### 🚜 Treatment Advisory\n*Recommended agronomic protocols will appear here.*")

            with gr.Accordion("🛡️ Statistical Safety Set (Split Conformal Guarantee)", open=False):
                conf_box = gr.Markdown("*Statistical guarantees will appear here upon diagnosis.*")

    diagnose_btn.click(
        fn=diagnose_leaf,
        inputs=[input_img],
        outputs=[output_img, label_output, diag_box, conf_box, treat_box],
    )
    input_img.change(
        fn=diagnose_leaf,
        inputs=[input_img],
        outputs=[output_img, label_output, diag_box, conf_box, treat_box],
    )


def main():
    parser = argparse.ArgumentParser(description="Run Plant Leaf Disease Gradio Interface")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", help="Generate public gradio.live tunnel URL")
    args = parser.parse_args()

    print("\n=======================================================")
    print("🌿 LAUNCHING PLANT CNN GRADIO WEB INTERFACE")
    print("=======================================================")
    print(f"Local URL:   http://localhost:{args.port}")
    if args.share:
        print("Share Mode:  Enabled (generating public https://*.gradio.live URL)")
    print("=======================================================\n")

    demo.launch(server_name=args.host, server_port=args.port, share=args.share, theme=custom_theme)


if __name__ == "__main__":
    main()
