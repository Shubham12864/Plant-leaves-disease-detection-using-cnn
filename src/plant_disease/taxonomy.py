"""Canonical plant pathology taxonomy, preprocessing, and pathogen-accurate treatment advisories.

Single source of truth shared across training, evaluation, CLI, Gradio, and model export
to prevent class-list or mapping drift across modules.
"""
from typing import Dict, Tuple
from PIL import Image
import torchvision.transforms.functional as TF

CLASSES_38 = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES_38)}
CROPS_14 = sorted({c.split("___")[0] for c in CLASSES_38})
CROP_TO_IDX: Dict[str, int] = {cr: i for i, cr in enumerate(CROPS_14)}
CLASS_TO_CROP_IDX: Dict[int, int] = {
    i: CROP_TO_IDX[c.split("___")[0]] for i, c in enumerate(CLASSES_38)
}

PLANTDOC_TO_PV_MAP: Dict[str, str] = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Potato leaf": "Potato___healthy",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def preprocess_pil_image(img_pil: Image.Image, target_size: int = 224, center_crop: bool = True) -> Image.Image:
    """Resize and optionally center-crop a PIL image while preserving aspect ratio.

    Prevents geometric distortion when non-square field smartphone photos are passed
    to a 224x224 CNN input.
    """
    img_rgb = img_pil.convert("RGB")
    if not center_crop:
        return img_rgb.resize((target_size, target_size), Image.Resampling.BILINEAR)
    w, h = img_rgb.size
    if w == target_size and h == target_size:
        return img_rgb
    # Scale shortest edge to int(target_size * 256 / 224) then center-crop
    short_target = int(round(target_size * 256 / 224))
    scale = short_target / min(w, h)
    new_w, new_h = max(target_size, int(round(w * scale))), max(target_size, int(round(h * scale)))
    resized = img_rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return TF.center_crop(resized, [target_size, target_size])


# Ordered from most specific pathogen class (viruses, mites, bacteria) to fungal classes.
TREATMENT_ADVISORY: Tuple[Tuple[str, Dict[str, str]], ...] = (
    ("healthy", {
        "organic": "Foliage appears healthy. Maintain compost tea soil health and beneficial microbial inoculants.",
        "chemical": "No fungicide or pesticide required. Maintain balanced micronutrient (Ca/B/Zn) foliar feeding.",
        "cultural": "Continue weekly field scouting, clean irrigation scheduling, and balanced nitrogen fertility.",
    }),
    ("virus", {
        "organic": "Viruses cannot be cured chemically; remove and destroy infected plants immediately to halt spread. Control insect vectors (whiteflies/aphids) with neem oil or insecticidal soap.",
        "chemical": "Fungicides are ineffective against viruses. Apply vector-targeted control (e.g., imidacloprid or pymetrozine per local guidelines) only if whitefly/aphid pressure is high.",
        "cultural": "Use certified virus-free transplants, install reflective mulches or fine insect netting, and wash hands/tools thoroughly (especially for mosaic virus).",
    }),
    ("spider_mites", {
        "organic": "Spray horticultural oil, neem oil, or potassium salts of fatty acids thoroughly on leaf undersides; release predatory mites (Phytoseiulus persimilis).",
        "chemical": "Do not apply fungicides. Use a labeled miticide/acaricide (e.g., abamectin or bifenazate) rotating IRAC modes of action.",
        "cultural": "Reduce dust on foliage and avoid water stress, which accelerates two-spotted spider mite outbreaks in hot, dry weather.",
    }),
    ("haunglongbing", {
        "organic": "No chemical cure exists for Citrus Greening (HLB). Control Asian citrus psyllid vectors with kaolin clay or horticultural oil sprays.",
        "chemical": "Manage psyllid vectors per regional citrus quarantine programs alongside targeted foliar micronutrient therapy (Zn, Mn, Fe) to sustain vigor.",
        "cultural": "Use certified disease-free nursery stock, scout flush growth for psyllids, and remove severely declined trees to protect nearby blocks.",
    }),
    ("bacterial", {
        "organic": "Apply fixed copper hydroxide combined with Bacillus amyloliquefaciens bio-bactericide every 7–10 days during wet weather.",
        "chemical": "Copper-mancozeb tank mixes or registered agricultural antibiotics where permitted to suppress bacterial lesion spread.",
        "cultural": "Use hot-water-treated pathogen-free seed, avoid overhead irrigation, and never work fields while foliage is wet.",
    }),
    ("scab", {
        "organic": "Apply elemental sulfur or copper octanoate sprays from green tip through petal fall.",
        "chemical": "Rotate DMI (myclobutanil) or QoI (trifloxystrobin) fungicides with multi-site protectants (captan/mancozeb).",
        "cultural": "Rake, shred, or compost fallen autumn leaves to reduce overwintering ascospores; prune for airflow.",
    }),
    ("blight", {
        "organic": "Apply Bacillus subtilis bio-fungicide or fixed copper soap before humid/rainy periods; prune heavily infected lower foliage.",
        "chemical": "Protectant chlorothalonil or mancozeb; for active late blight, systemic mandipropamid or cyazofamid.",
        "cultural": "Maintain 3-year non-solanaceous crop rotation, widen row spacing, and irrigate at the root zone.",
    }),
    ("rust", {
        "organic": "Apply sulfur or neem oil foliar sprays at the earliest pustule appearance.",
        "chemical": "Triazole (tebuconazole) or strobilurin (azoxystrobin) fungicide when rust severity crosses action thresholds.",
        "cultural": "Select rust-resistant cultivars and orient rows with prevailing winds to speed morning dew evaporation.",
    }),
    ("mildew", {
        "organic": "Apply potassium bicarbonate (3 g/L), wettable sulfur, or horticultural oil thoroughly across upper and lower leaf surfaces.",
        "chemical": "Systemic myclobutanil or trifloxystrobin rotated across FRAC groups to prevent resistance.",
        "cultural": "Provide full sunlight, avoid excess nitrogen flush, and prune overcrowded shoots for ventilation.",
    }),
    ("esca", {
        "organic": "Apply Trichoderma-based wound protectants immediately after dormant pruning.",
        "chemical": "No foliar fungicide reverses vascular Esca wood rot; protect fresh pruning cuts and mark symptomatic vines.",
        "cultural": "Perform late-winter dry-weather pruning, remove dead cordons/wood well below discolored vascular tissue, and burn infected prunings.",
    }),
    ("black_rot", {
        "organic": "Remove mummified berries/fruit and apply wettable sulfur + fixed copper sprays at early shoot growth.",
        "chemical": "Captan or mancozeb at 10–14 day intervals during warm wet springs through fruit set.",
        "cultural": "Eliminate all mummies, tendrils, and infected canes during winter dormant pruning.",
    }),
    ("mold", {
        "organic": "Reduce greenhouse/canopy humidity below 85% and apply copper or Bacillus subtilis bio-fungicide.",
        "chemical": "Protectant chlorothalonil or mancozeb as label directs under sustained high-humidity pressure.",
        "cultural": "Increase ventilation, space plants generously, and water early in the day.",
    }),
    ("scorch", {
        "organic": "Apply fixed copper sprays at early leaf expansion after renovation.",
        "chemical": "Registered captan or broad-spectrum protectant fungicides following harvest renovation.",
        "cultural": "Remove heavily infected leaves post-harvest, avoid dense matted beds, and switch to drip irrigation.",
    }),
    ("spot", {
        "organic": "Apply Bacillus subtilis or copper-based organic sprays; mulch soil to block rain-splash inoculum.",
        "chemical": "Chlorothalonil or mancozeb protectant sprays on a 7–10 day schedule in wet conditions.",
        "cultural": "Remove lower splashed leaves, rotate away from susceptible hosts, and sanitize stakes/tools.",
    }),
)


def get_treatment_for_disease(disease_code: str) -> Dict[str, str]:
    """Return pathogen-appropriate organic, chemical, and cultural advisories."""
    sub = disease_code.split("___")[-1].lower()
    for key, advisory in TREATMENT_ADVISORY:
        if key in sub:
            return dict(advisory)
    return {
        "organic": "Remove symptomatic leaves manually, sanitize shears with 70% alcohol, and apply broad-spectrum bio-fungicide only if fungal symptoms are confirmed.",
        "chemical": "Consult local agricultural extension services for crop-specific registered treatment before spraying.",
        "cultural": "Isolate severely affected plants, avoid overhead watering, and improve canopy airflow.",
    }
