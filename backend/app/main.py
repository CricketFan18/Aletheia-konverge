from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ExifTags
import shutil
import os
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

# Import custom heatmap generator
from utils import generate_ela_heatmap

# Load environment variables
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

app = FastAPI(title="Authenticity Verifier - Enterprise API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ENGINE INITIALIZATION (Custom Local Model)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"API using device: {device}")

# Rebuild EfficientNet-B0 architecture
classifier = models.efficientnet_b0()
num_ftrs = classifier.classifier[1].in_features
classifier.classifier[1] = nn.Linear(num_ftrs, 1)

# Load trained weights
MODEL_PATH = "efficientnet_b0_deepfake.pth"
if os.path.exists(MODEL_PATH):
    classifier.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Local model weights loaded successfully.")
else:
    print(f"WARNING: '{MODEL_PATH}' not found. Serving with untrained random weights.")

classifier = classifier.to(device)
classifier.eval()

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Ensure upload directory exists
os.makedirs("uploads", exist_ok=True)

# ==========================================
# EXIF EXTRACTOR
# ==========================================
def extract_exif(img: Image.Image) -> dict:
    try:
        exif = img._getexif()
        if not exif:
            return {
                "status": "Missing",
                "data": "No EXIF found (Common in AI/Edited images)."
            }
        clean = {
            ExifTags.TAGS[k]: str(v)
            for k, v in exif.items()
            if k in ExifTags.TAGS
        }
        return {"status": "Found", "data": clean}
    except Exception:
        return {"status": "Error", "data": "Failed to parse metadata."}

# ==========================================
# MASTER ENDPOINT (WITH CLOUDINARY)
# ==========================================
@app.post("/api/analyze")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file. Images only.")

    file_location = f"uploads/{file.filename}"

    try:
        # 1. Save file locally (temporary)
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)

        # 2. Upload original image to Cloudinary
        cloudinary_upload = cloudinary.uploader.upload(
            file_location,
            folder="authenticity_verifier/originals"
        )
        cloudinary_url = cloudinary_upload.get("secure_url")

        # 3. Load image for AI inference
        img = Image.open(file_location).convert("RGB")

        # 4. AI Inference
        input_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            raw_output = classifier(input_tensor)
            prediction = torch.sigmoid(raw_output).item()

        # Label mapping (0 = Fake, 1 = Real)
        label = "AUTHENTIC" if prediction > 0.5 else "AI_GENERATED"

        # Confidence calculation
        raw_confidence = prediction if prediction > 0.5 else (1 - prediction)
        confidence = round(raw_confidence * 100, 2)

        # 5. Generate ELA Heatmap
        heatmap_base64 = generate_ela_heatmap(file_location)

        # 6. Extract EXIF metadata
        exif_data = extract_exif(img)

        # 7. Delete local file to save disk space
        if os.path.exists(file_location):
            os.remove(file_location)

        # 8. Final Enterprise Response
        return {
            "status": "success",
            "source": {
                "original_image_url": cloudinary_url
            },
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

        # Cleanup on failure
        if os.path.exists(file_location):
            os.remove(file_location)

        raise HTTPException(
            status_code=500,
            detail="Server failed to process the image."
        )