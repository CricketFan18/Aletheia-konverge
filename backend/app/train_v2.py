import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import random
import io
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training V2 (Robust) on: {device}")

# --- 1. Custom JPEG Compression Transform ---
class RandomJPEGCompression:
    def __init__(self, quality_min=40, quality_max=95, p=0.7):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        quality = random.randint(self.quality_min, self.quality_max)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

# --- 2. Robust Data Preprocessing ---
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    # REAL-WORLD ROBUSTNESS
    RandomJPEGCompression(quality_min=40, quality_max=95, p=0.7),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation shouldn't be augmented, just standardized
transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('dataset/Train', transform=transform_train)
val_dataset = datasets.ImageFolder('dataset/Validation', transform=transform_val)

# Batch size 16 for 4GB VRAM
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# --- 3. Model Setup & Partial Unfreezing ---
model = models.efficientnet_b0(weights='DEFAULT')

# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# UNFREEZE the last 2 feature blocks for deep fine-tuning
for param in model.features[-2:].parameters():
    param.requires_grad = True

# Replace head for Binary Classification
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 1)
model = model.to(device)

# --- 4. Training Essentials ---
criterion = nn.BCEWithLogitsLoss()
# Pass ONLY the parameters that require gradients to the optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
scaler = torch.amp.GradScaler('cuda')

print(f"Starting Robust Training. Classes: {train_dataset.class_to_idx}")
epochs = 3 # Hard limit to 3 epochs for time safety

for epoch in range(epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    
    # Validation Loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            predicted = (outputs > 0.0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_time = time.time() - start_time
    val_acc = 100 * correct / total
    
    print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.0f}s | "
          f"Train Loss: {running_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")

torch.save(model.state_dict(), "efficientnet_b0_robust.pth")
print("Robust Model Saved to 'efficientnet_b0_robust.pth'!")