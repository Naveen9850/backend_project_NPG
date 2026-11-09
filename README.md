# 🌾 Crop Stress Detection API

A production-ready **FastAPI** backend that leverages a fine-tuned **ResNet-50** deep learning model to detect and classify crop stress from plant leaf images. The API distinguishes between **biotic stress** (disease-driven) and **abiotic stress** (environmental factors such as heat), and delivers actionable agronomic recommendations for each detected condition.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Supported Stress Classes](#supported-stress-classes)
- [API Reference](#api-reference)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
  - [Docker Deployment](#docker-deployment)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Model Details](#model-details)

---

## Overview

This service powers the backend of an AI-driven precision agriculture platform. By uploading a crop leaf image — with an optional crop type hint — users receive:

- **Stress classification** (Healthy / Biotic / Abiotic)
- **Disease or stress name** with possible causes
- **Severity level** (Mild / Moderate / Severe)
- **Confidence score** from the model
- **Tailored agronomic recommendations** specific to the detected condition and crop

The model was trained on a curated dataset of **25,000+ labeled crop images** across 7 classes and achieves robust real-world inference performance.

---

## Architecture

```
Client (Mobile / Web App)
        │
        ▼
  FastAPI REST API  (main.py)
        │
        ├─── Image Preprocessing  (torchvision transforms)
        │
        ├─── CropStressResNet50   (model.py)
        │         └── Fine-tuned ResNet-50 backbone
        │
        └─── Structured Response
               ├── stress_type
               ├── disease_name / stress_cause
               ├── severity
               ├── confidence
               ├── possible_diseases
               └── recommendation
```

---

## Supported Stress Classes

| Class | Stress Type | Description |
|---|---|---|
| `Healthy` | — | No visible stress detected |
| `Chilli_Disease` | Biotic | Fungal / bacterial infections in chilli |
| `Cotton_Disease` | Biotic | Diseases affecting cotton crops |
| `Groundnut_Disease` | Biotic | Diseases affecting groundnut crops |
| `Maize_Disease` | Biotic | Diseases affecting maize / corn |
| `Sorghum_Disease` | Biotic | Diseases affecting sorghum crops |
| `Heat_Stress` | Abiotic | Environmental heat-related stress |

> **Note:** The API also handles images of crops outside the training set by providing generalized biotic or abiotic stress guidance.

---

## API Reference

### `POST /predict`

Accepts a crop leaf image and returns a structured stress analysis.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `UploadFile` | ✅ | Crop leaf image (JPEG / PNG) |
| `crop_type` | `string` | ❌ | Crop hint (e.g., `"Maize"`, `"Cotton"`) for masked prediction |

**Response** — `application/json`

```json
{
  "stress_type": "Biotic",
  "disease_name": "Maize Disease (Biotic Stress)",
  "stress_cause": null,
  "severity": "Moderate",
  "confidence": 0.8731,
  "recommendation": "...",
  "possible_diseases": ["Northern Leaf Blight", "Gray Leaf Spot", "..."],
  "predicted_class": "Maize_Disease",
  "processing_time_ms": 143.5
}
```

---

### `GET /health`

Returns the current API and model health status.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_classes": 7,
  "device": "cpu"
}
```

---

### `GET /`

Root endpoint — confirms the API is running.

```json
{
  "message": "Crop Stress Detection API is running 🚀"
}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | [FastAPI](https://fastapi.tiangolo.com/) |
| **ASGI Server** | [Uvicorn](https://www.uvicorn.org/) |
| **Deep Learning** | [PyTorch 2.1.2](https://pytorch.org/) (CPU build) |
| **Model Backbone** | ResNet-50 (torchvision) |
| **Image Processing** | [Pillow](https://pillow.readthedocs.io/) + torchvision transforms |
| **Containerization** | Docker |
| **Deployment** | Render / Railway (PaaS) |
| **Language** | Python 3.10 |

---

## Getting Started

### Prerequisites

- Python **3.10+**
- pip
- (Optional) Docker

---

### Local Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Naveen9850/backend_project_NPG.git
   cd backend_project_NPG
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your values
   ```

5. **Run the development server**

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Open the interactive API docs**

   Navigate to [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Docker Deployment

1. **Build the image**

   ```bash
   docker build -t crop-stress-api .
   ```

2. **Run the container**

   ```bash
   docker run -p 10000:10000 crop-stress-api
   ```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Port the server listens on |
| `CORS_ORIGINS` | _(empty)_ | Comma-separated list of allowed frontend origins (e.g., `https://your-app.vercel.app`) |

Copy `.env.example` to `.env` and populate the values before running locally.

---

## Deployment

### Render

The repository includes a `render.yaml` for one-click deployment on [Render](https://render.com):

```yaml
services:
  - type: web
    name: crop-stress-backend
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Railway

The `Procfile` supports deployment on [Railway](https://railway.app):

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## Project Structure

```
backend_project_NPG/
├── main.py               # FastAPI application, endpoints, prediction logic
├── model.py              # CropStressResNet50 model architecture
├── inference.py          # Standalone inference utility script
├── test_predict.py       # Unit tests for prediction pipeline
├── crop_stress_model.pth # Pre-trained model weights
├── class_names.json      # Ordered list of class labels
├── 25kclass1.txt         # Training dataset class manifest
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker container definition
├── render.yaml           # Render PaaS deployment config
├── Procfile              # Railway / Heroku process file
├── runtime.txt           # Python version pin (3.10.13)
└── .env.example          # Environment variable template
```

---

## Model Details

| Attribute | Value |
|---|---|
| **Architecture** | ResNet-50 (pretrained on ImageNet) |
| **Fine-tuning** | Final fully-connected layer replaced; backbone frozen |
| **Input size** | 224 × 224 RGB |
| **Normalization** | ImageNet mean/std (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`) |
| **Output classes** | 7 |
| **Training samples** | ~25,000 images |
| **Inference device** | CPU (GPU-compatible) |

The model uses **masked prediction** when a known crop type is provided — restricting the softmax distribution to only the relevant class and its `Healthy` counterpart — which improves precision for single-crop use cases.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built with ❤️ for precision agriculture — transforming AI into actionable field insights.</sub>
</div>

