import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# 1. Hardware Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Evaluating on: {device}")

# 2. Data Preprocessing (Must match training exacty, minus augmentation)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the Test Dataset
TEST_DIR = 'dataset/Test'
if not os.path.exists(TEST_DIR):
    print(f"❌ Error: Test directory '{TEST_DIR}' not found!")
    exit()

test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
# Batch size 32 is safe for 4GB VRAM during inference
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Model Setup (EfficientNet-B0)
model = models.efficientnet_b0()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 1)

# Load your trained weights
MODEL_PATH = "base_model.pth"
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ Loaded weights from '{MODEL_PATH}'")
else:
    print(f"❌ Error: '{MODEL_PATH}' not found!")
    exit()

model = model.to(device)
model.eval() # Set to evaluation mode!

# 4. Run Inference
all_preds = []
all_labels = []

print(f"🚀 Starting evaluation on {len(test_dataset)} images...")
start_time = time.time()

with torch.no_grad(): # No gradients = low VRAM usage
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        # Convert logits to probabilities, then to binary predictions (0 or 1)
        probs = torch.sigmoid(outputs).squeeze()
        
        # Handle single-item batches dimension issues
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
            
        preds = (probs > 0.5).long()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"⏱️ Evaluation finished in {time.time() - start_time:.0f} seconds.\n")

# 5. Calculate and Print Metrics
class_names = test_dataset.classes # Usually ['Fake', 'Real']

print("-" * 50)
print("📊 CLASSIFICATION REPORT")
print("-" * 50)
print(classification_report(all_labels, all_preds, target_names=class_names))

# 6. Generate and Save Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('EfficientNet-B0 Evaluation: Confusion Matrix')
plt.ylabel('Actual Truth')
plt.xlabel('Model Prediction')
plt.tight_layout()

CM_FILENAME = "confusion_matrix_doc.png"
plt.savefig(CM_FILENAME, dpi=300)
print(f"🔥 Documentation asset saved: {CM_FILENAME}")