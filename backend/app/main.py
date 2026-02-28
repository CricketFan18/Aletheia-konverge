from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ExifTags
import io
import shutil
import os

# Import custom heatmap generator from utils.py
from utils import generate_ela_heatmap

app = FastAPI(title="Authenticity Verifier - Enterprise API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ENGINE INITIALIZATION (Custom Local Model)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"API using device: {device}")

# 1. Rebuild the exact EfficientNet-B0 architecture used in train.py
classifier = models.efficientnet_b0()
num_ftrs = classifier.classifier[1].in_features
classifier.classifier[1] = nn.Linear(num_ftrs, 1)
# 2. Load the trained weights
MODEL_PATH = "efficientnet_b0_deepfake.pth"
if os.path.exists(MODEL_PATH):
    classifier.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Local model weights loaded successfully.")
else:
    print(f"WARNING: '{MODEL_PATH}' not found. Serving with untrained random weights. Run train.py first!")

classifier = classifier.to(device)
classifier.eval()

# 3. Define the preprocessing pipeline for incoming API images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Ensure upload directory exists
os.makedirs("uploads", exist_ok=True)

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
@app.post("/api/analyze")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file. Images only.")
    
    file_location = f"uploads/{file.filename}"
    
    try:
        # 1. Save file locally (Required for ELA and OpenCV)
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
            
        # 2. Load into memory
        img = Image.open(file_location).convert("RGB")
        
        # 3. AI Inference (The Brains)
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            raw_output = classifier(input_tensor)
            # Manually apply sigmoid to get the 0 to 1 probability
            prediction = torch.sigmoid(raw_output).item()
            
        # Based on ImageFolder alphabetical mapping: 0 = Fake, 1 = Real
        label = "AUTHENTIC" if prediction > 0.5 else "AI_GENERATED"
        
        # Calculate strict confidence percentage
        raw_confidence = prediction if prediction > 0.5 else (1 - prediction)
        confidence = round(raw_confidence * 100, 2)
        
        # 4. Generate Visual Proof (The Heatmap)
        heatmap_base64 = generate_ela_heatmap(file_location)
        
        # 5. Extract Digital Footprint (EXIF)
        exif_data = extract_exif(img)
        
        # Clean up the file to save disk space
        os.remove(file_location)
        
        # 6. Return the ultimate payload to the Frontend
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
        # Clean up file on failure
        if os.path.exists(file_location):
            os.remove(file_location)
        raise HTTPException(status_code=500, detail="Server failed to process the image.")