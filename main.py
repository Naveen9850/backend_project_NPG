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

# ─── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crop-stress-api")

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_stress_model.pth")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")

# ─── Global model state ────────────────────────────────────────────────────────
model = None
class_names = None
device = None
transform = None

# ─── Recommendation mappings ────────────────────────────────────────────────────
RECOMMENDATIONS = {
    "Healthy": (
        "Your crop appears healthy! Continue current agricultural practices. "
        "Monitor regularly for early signs of stress. Maintain proper irrigation "
        "and nutrient management schedules."
    ),
    "Chilli_Disease": (
        "Chilli disease detected. Apply appropriate fungicide or bactericide treatment. "
        "Ensure proper spacing between plants for air circulation. Remove and destroy "
        "severely infected plant parts. Monitor humidity levels and avoid overhead irrigation."
    ),
    "Cotton_Disease": (
        "Cotton disease detected. Apply targeted fungicide treatment immediately. "
        "Improve field drainage and reduce plant density if needed. Consider crop rotation "
        "in subsequent seasons. Scout regularly to track spread and adjust treatment."
    ),
    "Groundnut_Disease": (
        "Groundnut disease detected. Apply recommended fungicide spray at appropriate intervals. "
        "Ensure proper seed treatment before planting. Implement crop rotation with cereals. "
        "Remove crop debris after harvest to reduce inoculum carryover."
    ),
    "Maize_Disease": (
        "Maize disease detected. Apply foliar fungicide if infection is in early stages. "
        "Ensure balanced nitrogen fertilization (excess N increases susceptibility). "
        "Consider resistant varieties for future planting. Remove and destroy crop residue."
    ),
    "Sorghum_Disease": (
        "Sorghum disease detected. Use certified disease-free seeds for planting. "
        "Apply recommended fungicides at early symptom onset. Practice field sanitation "
        "by removing infected plant debris. Rotate with non-host crops."
    ),
    "Heat_Stress": (
        "Abiotic heat stress detected. Increase irrigation frequency using drip or "
        "sprinkler systems to cool the crop canopy. Apply mulching to conserve soil moisture "
        "and reduce soil temperature. Consider shade nets for sensitive growth stages. "
        "Shift planting dates to avoid peak heat periods in future seasons."
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
    logger.info("🚀  Starting Crop Stress Detection API ...")
    try:
        load_model_and_classes()
        logger.info("✅  Model ready for inference")
    except Exception as e:
        logger.error(f"❌  Failed to load model: {e}")
        raise
    yield
    logger.info("🛑  Shutting down API")


# ─── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Crop Stress Detection API",
    description="AI-powered crop stress detection using multimodal ResNet50",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend (dev + production)
_cors_env = os.environ.get("CORS_ORIGINS", "")
_extra_origins = [o.strip() for o in _cors_env.split(",") if o.strip()] if _cors_env else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow everything
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
),
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

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = class_names[predicted_idx.item()]
    confidence_score = round(confidence.item(), 4)

    processing_time_ms = round((time.time() - start_time) * 1000, 1)

    # Determine if the user's crop matches one of the trained crops
    TRAINED_CROPS = {"cotton", "soybean", "sorghum", "pigeon pea", "groundnut", "sunflower", "maize", "chili", "chilli"}
    is_known_crop = crop_type.strip().lower() in TRAINED_CROPS

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
