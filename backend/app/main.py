from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ExifTags
import io
import shutil
import os
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

# ENGINE INITIALIZATION (Custom Local Model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"API using device: {device}")

classifier = models.efficientnet_b0()
num_ftrs = classifier.classifier[1].in_features
classifier.classifier[1] = nn.Linear(num_ftrs, 1)

MODEL_PATH = "base_model.pth" 

if os.path.exists(MODEL_PATH):
    classifier.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Local model weights ({MODEL_PATH}) loaded successfully.")
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

os.makedirs("uploads", exist_ok=True)

# EXIF EXTRACTOR
def extract_exif(img: Image.Image) -> dict:
    try:
        exif = img.getexif() # More stable than _getexif()
        if not exif: return {"status": "Missing", "data": "No EXIF found (Common in AI/Edited images)."}
        clean = {ExifTags.TAGS[k]: str(v) for k, v in exif.items() if k in ExifTags.TAGS}
        return {"status": "Found", "data": clean}
    except:
        return {"status": "Error", "data": "Failed to parse metadata."}

@app.post("/api/analyze")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file. Images only.")
    
    file_location = f"uploads/{file.filename}"
    
    try:
        # 1. Save file locally
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
            
        # 2. Load into memory AND force standard JPEG to strip format bias
        raw_img = Image.open(file_location).convert("RGB")
        buffer = io.BytesIO()
        raw_img.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        img = Image.open(buffer) # The standardized image
        
        # 3. AI Inference
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            raw_output = classifier(input_tensor)
            prediction = torch.sigmoid(raw_output).item()
            
        label = "AUTHENTIC" if prediction > 0.5 else "AI_GENERATED"
        
        raw_confidence = prediction if prediction > 0.5 else (1 - prediction)
        confidence = round(raw_confidence * 100, 2)
        
        # 4. Extract EXIF and generate heatmap
        exif_data = extract_exif(raw_img) # Get EXIF from the original raw image
        heatmap_base64 = generate_ela_heatmap(raw_img)
        
        # --- HACKATHON MVP HEURISTIC ADJUSTMENT ---
        # If it thinks it's real, but there's no EXIF, pull the confidence down
        if label == "AUTHENTIC" and exif_data["status"] == "Missing":
            adjusted_prediction = prediction - 0.20 # Penalize the score
            if adjusted_prediction <= 0.5:
                label = "AI_GENERATED"
                raw_confidence = 1 - adjusted_prediction
            else:
                raw_confidence = adjusted_prediction
            confidence = round(raw_confidence * 100, 2)
        # ------------------------------------------

        os.remove(file_location)
        
        return {
            "status": "success",
            "verdict": {"label": label, "confidence": confidence},
            "evidence": {"heatmap_image": heatmap_base64, "metadata": exif_data}
        }
        
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        if os.path.exists(file_location):
            os.remove(file_location)
        raise HTTPException(status_code=500, detail="Server failed to process the image.")