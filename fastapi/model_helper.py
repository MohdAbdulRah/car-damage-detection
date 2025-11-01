from PIL import Image
import torch
import os
import requests
from torch import nn
from torchvision import models, transforms
import gdown
import os

MODEL_PATH = "saved_model.pth"
DRIVE_ID = "1H76AbNs2vGx3wEM-5EPQm1jaXbBoY6DZ"

# --- Step 1: Download model safely ---
def download_model():
    if not os.path.exists(MODEL_PATH):
      print("Downloading model from Google Drive using gdown...")
      gdown.download(f"https://drive.google.com/uc?id={DRIVE_ID}", MODEL_PATH, quiet=False)
      print("Model downloaded successfully.")
    print("Download completed:", MODEL_PATH)

download_model()

# --- Step 2: Define model ---
class_names = ['Front Breakage', 'Front Crushed', 'Front Normal',
               'Rear Breakage', 'Rear Crushed', 'Rear Normal']

class CarClassifierWithResNet(nn.Module):
    def __init__(self, num_classes=len(class_names)):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')

        # Freeze earlier layers
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# --- Step 3: Prediction function ---
trained_model = None

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    global trained_model
    if trained_model is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- Step 4: Validate downloaded file ---
        if os.path.getsize(MODEL_PATH) < 1_000_000:  # smaller than ~1MB → probably invalid
            raise ValueError("Downloaded model file is too small — likely HTML instead of .pth")

        trained_model = CarClassifierWithResNet()
        state_dict = torch.load(MODEL_PATH, map_location=device)
        trained_model.load_state_dict(state_dict)
        trained_model.to(device)
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]
