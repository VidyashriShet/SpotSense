import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)

# Modify Fully Connected Layer
num_classes = len(ImageFolder("c:\SpotSense_Project\spotsense_temples_preprocessed_dataset").classes)
model.fc = nn.Sequential(
    nn.Dropout(0.5),  
    nn.Linear(model.fc.in_features, num_classes)
)

# Unfreeze Last Two Layers for Fine-Tuning
for param in model.parameters():
    param.requires_grad = False  # Freeze all layers initially

for param in list(model.layer3.parameters())[-10:]:
    param.requires_grad = True  # Unfreeze last 10 parameters in layer 3

for param in list(model.layer4.parameters()):
    param.requires_grad = True  # Fully unfreeze layer 4

model = model.to(device)

# Define Data Transformations (No Augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder("c:\SpotSense_Project\spotsense_train_dataset", transform=transform)
test_dataset = ImageFolder("c:\SpotSense_Project\spotsense_test_dataset", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define Loss Function 
criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

# Define Optimizer and Learning Rate Scheduler
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
early_stopper = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5, verbose=True)

num_epochs = 30
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    scheduler.step()

torch.save(model.state_dict(), "spotsense_updated_30_resnet50_temples_model.pth")
print("Model saved successfully!")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

acc = 100 * correct / total
print(f"Test Accuracy: {acc:.2f}%")
early_stopper.step(acc)
