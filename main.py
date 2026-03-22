"""
FastAPI Backend for Crop Stress Detection
Loads the trained PyTorch ResNet50 model and serves predictions via REST API.
"""

import os
import sys
import json
import time
import io
import logging
from typing import Optional
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import CropStressResNet50

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_stress_model.pth")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")

# ─── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crop-stress-api")
# Also log to a file for debugging
_fh = logging.FileHandler(os.path.join(BASE_DIR, "predict_debug.log"), mode="w")
_fh.setLevel(logging.INFO)
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_fh)


# ─── Global model state ────────────────────────────────────────────────────────
model = None
class_names = None
device = None
transform = None

# ─── Recommendation mappings ────────────────────────────────────────────────────
RECOMMENDATIONS = {
    "Healthy": (
        "Your crop appears healthy with no visible signs of biotic or abiotic stress. "
        "Continue your current agricultural practices including regular irrigation schedules, "
        "balanced NPK fertilization, and integrated pest management (IPM). Monitor the crop "
        "canopy weekly for early symptoms such as yellowing, wilting, or lesion formation. "
        "Maintain field hygiene by removing weeds and crop debris. Consider applying preventive "
        "foliar sprays of micronutrients (Zinc, Boron) during critical growth stages to "
        "strengthen plant immunity and ensure sustained healthy growth throughout the season."
    ),
    "Chilli_Disease": (
        "Biotic stress detected in your chilli crop. The symptoms suggest possible fungal or "
        "bacterial infection such as Anthracnose (Colletotrichum capsici), Bacterial Leaf Spot, "
        "or Cercospora leaf spot. Immediate action: spray Mancozeb 75% WP at 2.5 g/L or "
        "Copper Oxychloride 50% WP at 3 g/L at 10-day intervals. Remove and burn severely "
        "infected leaves and fruits to prevent spore dispersal. Improve air circulation by "
        "maintaining 45-60 cm plant spacing. Avoid overhead irrigation as leaf wetness "
        "promotes fungal growth. Apply Trichoderma viride as a soil drench for root disease "
        "prevention. Rotate with non-solanaceous crops for 2-3 seasons."
    ),
    "Cotton_Disease": (
        "Biotic stress detected in your cotton crop. Symptoms indicate possible Bacterial Blight "
        "(Xanthomonas citri pv. malvacearum), Alternaria Leaf Spot, or Grey Mildew. Immediate "
        "action: spray Streptomycin Sulphate 500 ppm + Copper Oxychloride 3 g/L for bacterial "
        "infections, or Propiconazole 25% EC at 1 mL/L for fungal diseases. Remove infected "
        "plant parts and destroy by burning. Ensure proper field drainage to prevent waterlogging "
        "which increases disease incidence. Maintain recommended spacing of 60×30 cm for adequate "
        "ventilation. Apply neem cake at 250 kg/ha as a soil amendment to suppress soil-borne "
        "pathogens. Use certified Bt cotton varieties with built-in resistance for next season."
    ),
    "Groundnut_Disease": (
        "Biotic stress detected in your groundnut crop. Symptoms suggest Early Leaf Spot "
        "(Cercospora arachidicola), Late Leaf Spot (Phaeoisariopsis personata), or Rust "
        "(Puccinia arachidis). Immediate action: spray Chlorothalonil 75% WP at 2 g/L or "
        "Carbendazim 50% WP at 1 g/L, starting at 30 days after sowing (DAS) and repeating "
        "every 15 days. Treat seeds with Thiram at 3 g/kg before sowing to prevent Collar Rot. "
        "Implement 3-year crop rotation with cereals like sorghum or maize. Apply gypsum at "
        "500 kg/ha at pegging stage to strengthen pods and reduce Aspergillus infection. "
        "Remove and destroy crop residue after harvest. Use resistant varieties like GPBD-4 "
        "or TAG-24 for future plantings to minimize disease pressure."
    ),
    "Maize_Disease": (
        "Biotic stress detected in your maize crop. Symptoms indicate possible Northern Corn "
        "Leaf Blight (Exserohilum turcicum), Common Rust (Puccinia sorghi), or Gray Leaf Spot. "
        "Immediate action: apply foliar spray of Propiconazole 25% EC at 1 mL/L or Mancozeb "
        "75% WP at 2.5 g/L at first symptom appearance. Ensure balanced nitrogen fertilization "
        "as excess nitrogen (above 120 kg/ha) increases leaf blight susceptibility. Remove "
        "lower infected leaves to reduce spore inoculum. Maintain optimal plant density of "
        "75,000-80,000 plants/ha for proper air circulation. Use resistant hybrids such as "
        "HQPM-1 or DHM-117 for next season. Practice deep ploughing after harvest to bury "
        "infected crop residue and break the disease cycle effectively."
    ),
    "Sorghum_Disease": (
        "Biotic stress detected in your sorghum crop. Symptoms suggest Anthracnose "
        "(Colletotrichum sublineolum), Grain Mold, or Downy Mildew (Peronosclerospora sorghi). "
        "Immediate action: spray Mancozeb 75% WP at 2.5 g/L or Metalaxyl 35% WS as seed "
        "treatment at 6 g/kg for downy mildew prevention. Remove and destroy infected "
        "panicles showing grain mold symptoms to prevent mycotoxin contamination. Ensure "
        "early sowing to avoid grain mold during the rainy period at harvest. Apply Carbendazim "
        "50% WP at 1 g/L during boot leaf stage for anthracnose management. Practice field "
        "sanitation by removing infected plant debris post-harvest. Rotate with leguminous "
        "crops like groundnut or soybean. Use resistant varieties such as CSV-17 or CSV-22."
    ),
    "Heat_Stress": (
        "Abiotic heat stress detected in your crop. Elevated temperatures are causing leaf "
        "scorching, wilting, and reduced photosynthetic activity. Immediate action: increase "
        "irrigation frequency using drip or sprinkler systems, providing 5-7 cm water every "
        "3-4 days to cool the crop canopy and maintain soil moisture. Apply organic mulch "
        "(paddy straw or sugarcane trash) at 5 tonnes/ha to reduce soil temperature by 4-6°C. "
        "Spray 0.5% Potassium Chloride (KCl) solution to improve stomatal regulation and heat "
        "tolerance. Foliar application of Salicylic Acid at 100 ppm enhances thermotolerance. "
        "Consider installing shade nets (35-50% shade) during flowering and grain-filling stages. "
        "For future seasons, shift sowing dates 2-3 weeks earlier to avoid peak heat periods."
    ),
}

SEVERITY_MAP = {
    "Healthy": None,
    "Chilli_Disease": "Moderate",
    "Cotton_Disease": "Moderate",
    "Groundnut_Disease": "Moderate",
    "Maize_Disease": "Moderate",
    "Sorghum_Disease": "Moderate",
    "Heat_Stress": "Moderate",
}

# ─── Common diseases per crop (for enhanced display) ───────────────────────────
COMMON_DISEASES = {
    "Chilli_Disease": [
        "Anthracnose (Colletotrichum capsici)",
        "Bacterial Leaf Spot (Xanthomonas campestris)",
        "Powdery Mildew (Leveillula taurica)",
        "Cercospora Leaf Spot",
        "Phytophthora Blight",
        "Mosaic Virus (CMV / TMV)",
    ],
    "Cotton_Disease": [
        "Bacterial Blight (Xanthomonas citri)",
        "Fusarium Wilt (Fusarium oxysporum)",
        "Verticillium Wilt",
        "Alternaria Leaf Spot",
        "Grey Mildew (Ramularia areola)",
        "Cotton Leaf Curl Virus (CLCuV)",
    ],
    "Groundnut_Disease": [
        "Early Leaf Spot (Cercospora arachidicola)",
        "Late Leaf Spot (Phaeoisariopsis personata)",
        "Rust (Puccinia arachidis)",
        "Collar Rot (Aspergillus niger)",
        "Stem Rot (Sclerotium rolfsii)",
        "Bud Necrosis Virus (GBNV)",
    ],
    "Maize_Disease": [
        "Northern Corn Leaf Blight (Exserohilum turcicum)",
        "Southern Corn Leaf Blight",
        "Common Rust (Puccinia sorghi)",
        "Gray Leaf Spot (Cercospora zeae-maydis)",
        "Maize Dwarf Mosaic Virus",
        "Stalk Rot (Fusarium spp.)",
    ],
    "Sorghum_Disease": [
        "Anthracnose (Colletotrichum sublineolum)",
        "Grain Mold Complex",
        "Downy Mildew (Peronosclerospora sorghi)",
        "Rust (Puccinia purpurea)",
        "Charcoal Rot (Macrophomina phaseolina)",
        "Ergot (Claviceps sorghi)",
    ],
    "Heat_Stress": [
        "Thermal canopy stress",
        "Soil moisture depletion",
        "Pollen sterility from high temperature",
        "Leaf scorching / tip burn",
    ],
}


# ─── Response models ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    stress_type: str
    disease_name: Optional[str] = None
    stress_cause: Optional[str] = None
    severity: Optional[str] = None
    confidence: float
    recommendation: Optional[str] = None
    predicted_class: str
    processing_time_ms: float
    possible_diseases: Optional[list[str]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    num_classes: int
    class_names: list


# ─── Transforms (same as training validation) ──────────────────────────────────
def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ─── Model loading ─────────────────────────────────────────────────────────────
def load_model_and_classes():
    global model, class_names, device, transform

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load class names
    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    logger.info(f"Loaded {len(class_names)} classes: {class_names}")

    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    num_classes = len(class_names)
    model = CropStressResNet50(num_classes)

    # Load the checkpoint
    state_dict = torch.load(MODEL_PATH, map_location=device)


    # The saved checkpoint may have raw ResNet keys (e.g. "conv1.weight")
    # but CropStressResNet50 wraps ResNet inside self.model, expecting
    # keys like "model.conv1.weight". Detect and fix this mismatch.
    model_keys = set(model.state_dict().keys())
    saved_keys = set(state_dict.keys())

    if model_keys != saved_keys:
        # Check if adding "model." prefix fixes it
        prefixed = {f"model.{k}": v for k, v in state_dict.items()}
        if set(prefixed.keys()) == model_keys:
            logger.info("Auto-prefixing state_dict keys with 'model.'")
            state_dict = prefixed
        else:
            # Try removing "model." prefix
            unprefixed = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
            if set(unprefixed.keys()) == model_keys:
                state_dict = unprefixed

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    logger.info("Model loaded successfully!")

    # Build transforms
    transform = get_transforms()


# ─── App lifecycle ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    logger.info("[START] Starting Crop Stress Detection API ...")
    try:
        load_model_and_classes()
        logger.info("[OK] Model ready for inference")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load model: {e}")
        raise
    yield
    logger.info("[STOP] Shutting down API")


# ─── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Crop Stress Detection API",
    description="AI-powered crop stress detection using multimodal ResNet50",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend (dev + production)
# CORS configuration
_cors_env = os.environ.get("CORS_ORIGINS", "")
_extra_origins = [
    o.strip() for o in _cors_env.split(",") if o.strip()
] if _cors_env else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "https://frontend-project-npg.vercel.app",
    ] + _extra_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ─── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API and model are running."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device),
        num_classes=len(class_names) if class_names else 0,
        class_names=class_names or [],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    crop_type: str = Form(default=""),
    growth_stage: str = Form(default=""),
    pm25: float = Form(default=0),
    pm10: float = Form(default=0),
    no2: float = Form(default=0),
    so2: float = Form(default=0),
    o3: float = Form(default=0),
):
    """
    Perform crop stress detection on an uploaded image.

    Parameters
    ----------
    image : uploaded image file (JPG / PNG)
    crop_type : selected crop name
    growth_stage : current growth stage
    pm25, pm10, no2, so2, o3 : air pollution readings
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    # Read and validate the image
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Pre-process
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # ── Map user-selected crop name → model class name ──────────────────────
    CROP_TO_CLASS = {
        "cotton":     "Cotton_Disease",
        "soybean":    "Sorghum_Disease",   # no dedicated class; closest proxy
        "sorghum":    "Sorghum_Disease",
        "pigeon pea": "Sorghum_Disease",   # no dedicated class; closest proxy
        "groundnut":  "Groundnut_Disease",
        "sunflower":  "Sorghum_Disease",   # no dedicated class; closest proxy
        "maize":      "Maize_Disease",
        "chili":      "Chilli_Disease",
        "chilli":     "Chilli_Disease",
    }
    crop_key = crop_type.strip().lower()
    is_known_crop = crop_key in CROP_TO_CLASS

    # ── Inference ───────────────────────────────────────────────────────────
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)

        # Log raw probabilities for ALL classes (for debugging)
        raw_probs = {class_names[i]: round(probabilities[0, i].item(), 4) for i in range(len(class_names))}
        logger.info(f"Raw probabilities: {raw_probs}")

        # Get the global best prediction (unrestricted)
        global_confidence, global_idx = torch.max(probabilities, 1)
        global_class = class_names[global_idx.item()]
        logger.info(f"Global best: {global_class} ({global_confidence.item():.4f})")

        if is_known_crop:
            # ALWAYS constrain prediction to the user's selected crop.
            # Never show a different crop's disease (e.g., don't show Groundnut
            # when user selected Chili).
            expected_disease_class = CROP_TO_CLASS[crop_key]
            allowed_classes = {expected_disease_class, "Healthy", "Heat_Stress"}
            allowed_indices = [
                i for i, name in enumerate(class_names) if name in allowed_classes
            ]

            # Get probabilities only for allowed classes
            disease_idx = class_names.index(expected_disease_class)
            healthy_idx = class_names.index("Healthy")
            heat_idx = class_names.index("Heat_Stress")

            p_disease = probabilities[0, disease_idx].item()
            p_healthy = probabilities[0, healthy_idx].item()
            p_heat    = probabilities[0, heat_idx].item()

            # Check if the GLOBAL best prediction is a disease class
            # (any disease, not just the user's crop)
            global_is_disease = global_class.endswith("_Disease")
            global_is_heat = global_class == "Heat_Stress"

            logger.info(
                f"Allowed: {expected_disease_class}={p_disease:.4f}, "
                f"Healthy={p_healthy:.4f}, Heat_Stress={p_heat:.4f} | "
                f"Global best: {global_class} ({global_confidence.item():.4f})"
            )

            # Decision logic:
            # If the model sees disease globally (any crop disease), AND
            # Healthy probability is very low, then the plant IS diseased.
            # Attribute it to the user's selected crop disease class.
            if global_is_disease and p_healthy < 0.20:
                # Plant is clearly diseased — assign to user's crop disease
                predicted_idx = torch.tensor([disease_idx])
                # Use the higher of: crop-specific prob or global confidence
                confidence = torch.tensor([max(p_disease, global_confidence.item() * 0.85)])
                logger.info(f"Disease detected globally → mapped to {expected_disease_class}")
            elif global_is_heat:
                predicted_idx = torch.tensor([heat_idx])
                confidence = torch.tensor([p_heat if p_heat > 0.01 else global_confidence.item()])
                logger.info(f"Heat stress detected")
            else:
                # Use masked prediction among allowed classes
                masked_probs = probabilities.clone()
                mask = torch.ones_like(masked_probs, dtype=torch.bool)
                for idx in allowed_indices:
                    mask[0, idx] = False
                masked_probs[mask] = 0.0

                # Re-normalise for picking, but use raw prob as confidence
                confidence, predicted_idx = torch.max(masked_probs, 1)
                picked = class_names[predicted_idx.item()]
                logger.info(f"Standard masked pick: {picked} (raw conf: {confidence.item():.4f})")
        else:
            # No crop selected or "Other Crops" — use full distribution
            confidence = global_confidence
            predicted_idx = global_idx

    predicted_class = class_names[predicted_idx.item()]
    confidence_score = round(confidence.item(), 4)

    processing_time_ms = round((time.time() - start_time) * 1000, 1)

    # Build structured response
    response = PredictionResponse(
        stress_type="Healthy",
        disease_name=None,
        stress_cause=None,
        severity=None,
        confidence=confidence_score,
        recommendation=None,
        predicted_class=predicted_class,
        processing_time_ms=processing_time_ms,
    )

    if predicted_class == "Healthy":
        response.stress_type = "Healthy"
        response.recommendation = RECOMMENDATIONS["Healthy"]

    elif predicted_class.endswith("_Disease"):
        response.stress_type = "Biotic"

        if is_known_crop:
            # Known crop → show crop-specific disease info
            crop_name = predicted_class.replace("_Disease", "").replace("_", " ")
            response.disease_name = f"{crop_name} Disease (Biotic Stress)"
            response.possible_diseases = COMMON_DISEASES.get(predicted_class, [])
            response.recommendation = RECOMMENDATIONS.get(predicted_class,
                "Monitor crop closely and consult an agricultural expert.")
        else:
            # Unknown / "Other" crop → show generalized biotic stress
            user_crop_label = crop_type.strip() if crop_type.strip() else "Crop"
            response.disease_name = f"Disease Detected on {user_crop_label} (Biotic Stress)"
            response.possible_diseases = [
                "Fungal infection (leaf spot, blight, rust, mildew)",
                "Bacterial infection (wilt, canker, soft rot)",
                "Viral infection (mosaic, leaf curl, yellowing)",
                "Nutrient deficiency mimicking disease symptoms",
            ]
            response.recommendation = (
                f"Biotic stress (disease) detected on your {user_crop_label}. "
                "The AI model identified disease-like symptoms in the image. "
                "Since this crop is outside the model's primary training set, "
                "please consult a local agricultural expert or extension officer "
                "for specific disease identification. Consider applying a broad-spectrum "
                "fungicide as a precaution and isolate affected plants to prevent spread."
            )

        # Adjust severity based on confidence
        if confidence_score >= 0.85:
            response.severity = "Severe"
        elif confidence_score >= 0.60:
            response.severity = "Moderate"
        else:
            response.severity = "Mild"

    elif predicted_class == "Heat_Stress":
        response.stress_type = "Abiotic"
        response.stress_cause = "Heat Stress"
        response.possible_diseases = COMMON_DISEASES.get(predicted_class, [])
        response.recommendation = RECOMMENDATIONS.get("Heat_Stress",
            "Monitor crop closely and consult an agricultural expert.")
        if confidence_score >= 0.85:
            response.severity = "Severe"
        elif confidence_score >= 0.60:
            response.severity = "Moderate"
        else:
            response.severity = "Mild"

    else:
        # Fallback for any unknown class
        response.stress_type = "Abiotic"
        response.stress_cause = predicted_class.replace("_", " ")
        response.severity = "Moderate"
        response.recommendation = "Monitor crop closely and consult an agricultural expert."

    logger.info(
        f"Prediction: {predicted_class} | Confidence: {confidence_score:.2%} | "
        f"Stress: {response.stress_type} | Known crop: {is_known_crop} | "
        f"Time: {processing_time_ms}ms"
    )

    return response

@app.get("/")
async def root():
    return {"message": "Crop Stress Detection API is running 🚀"}

# ─── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
