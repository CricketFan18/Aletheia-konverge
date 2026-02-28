from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ExifTags
import io

# Import custom heatmap generator from utils.py
from utils import generate_ela_heatmap

app = FastAPI(title="Authenticity Verifier - Enterprise API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://interstrial-epithelial-anaya.ngrok-free.dev",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ENGINE INITIALIZATION (Custom Local Model)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"API using device: {device}")

classifier = models.efficientnet_b0()
num_ftrs = classifier.classifier[1].in_features
classifier.classifier[1] = nn.Linear(num_ftrs, 1)

# FIX: Point to the actual weights saved by train.py
MODEL_PATH = "base_model.pth" 
import os
if os.path.exists(MODEL_PATH):
    classifier.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Local model weights loaded successfully.")
else:
    print(f"WARNING: '{MODEL_PATH}' not found. Serving with untrained random weights. Run train.py first!")

classifier = classifier.to(device)
classifier.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# EXIF EXTRACTOR
# ==========================================
def extract_exif(img: Image.Image) -> dict:
    try:
        exif = img._getexif()
        if not exif: return {"status": "Missing", "data": "No EXIF found (Common in AI/Edited images)."}
        clean = {ExifTags.TAGS[k]: str(v) for k, v in exif.items() if k in ExifTags.TAGS}
        return {"status": "Found", "data": clean}
    except:
        return {"status": "Error", "data": "Failed to parse metadata."}

# ==========================================
# THE MASTER ENDPOINT
# ==========================================
# FIX: Removed 'async' so FastAPI runs this blocking code in a background threadpool
@app.post("/api/analyze")
def analyze_image_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file. Images only.")
    
    try:
        # FIX: Process entirely in RAM. No disk writing means no file overwrite collisions.
        image_bytes = file.file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # AI Inference
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            raw_output = classifier(input_tensor)
            prediction = torch.sigmoid(raw_output).item()
            
        label = "AUTHENTIC" if prediction > 0.5 else "AI_GENERATED"
        raw_confidence = prediction if prediction > 0.5 else (1 - prediction)
        confidence = round(raw_confidence * 100, 2)
        
        # Generate Visual Proof directly from PIL Image
        heatmap_base64 = generate_ela_heatmap(img)
        
        # Extract Digital Footprint
        exif_data = extract_exif(img)
        
        return {
            "status": "success",
            "verdict": {
                "label": label,
                "confidence": confidence
            },
            "evidence": {
                "heatmap_image": heatmap_base64,
                "metadata": exif_data
            }
        }
        
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        raise HTTPException(status_code=500, detail="Server failed to process the image.")