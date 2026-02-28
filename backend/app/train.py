import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os
import time

# 1. Hardware Setup (Optimized for 4GB VRAM)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# 2. Data Preprocessing
# EfficientNet-B0 expects 224x224 inputs
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Datasets using your exact directory structure
# ImageFolder automatically maps 'Fake' to 0 and 'Real' to 1 (alphabetical)
train_dataset = datasets.ImageFolder('dataset/Train', transform=transform)
val_dataset = datasets.ImageFolder('dataset/Validation', transform=transform)

# Batch size 16 is safe for 4GB VRAM. Do not increase this.
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 3. Model Setup (EfficientNet-B0)
model = models.efficientnet_b0(weights='DEFAULT')

# Freeze the base layers for rapid hackathon training
for param in model.parameters():
    param.requires_grad = False

# Replace the head for Binary Classification (Fake vs Real)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 1)
model = model.to(device)

# 4. Memory-Efficient Training Essentials
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)

# Initialize Gradient Scaler for Mixed Precision
scaler = torch.amp.GradScaler('cuda')

print(f"Starting Training. Classes: {train_dataset.class_to_idx}")
epochs = 5

for epoch in range(epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    
    # Training Loop
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        
        # Mixed Precision Context
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

# Save the final model weights in the root directory
torch.save(model.state_dict(), "base_model.pth")
print("Model Weights Saved Successfully to 'base_model.pth'!")