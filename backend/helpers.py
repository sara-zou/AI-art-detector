import torch
import io
import torch
import torch.nn as nn
from config import MODEL_URL, CHECKPOINT_PATH, IMG_SIZE, THRESHOLD, NORM_MEAN, NORM_STD
from torchvision import transforms, models
from PIL import Image
import gc
import urllib.request
import os

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(device):
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[Artifex] Downloading model from Hugging Face...")
        token = os.environ.get("HF_TOKEN")
        request = urllib.request.Request(
            MODEL_URL,
            headers={"Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(request) as response, open(CHECKPOINT_PATH, "wb") as f:
            f.write(response.read())
        print(f"[Artifex] Model downloaded.")
    
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device).eval()
    return model

def predict(file_bytes: bytes, model: nn.Module, device: torch.device):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        prob_ai = torch.sigmoid(output).item()
    
    label = "Human" if prob_ai >= THRESHOLD else "AI"
    confidence = prob_ai if prob_ai >= THRESHOLD else 1.0 - prob_ai
    
    gc.collect()
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "prob_ai": round(prob_ai, 4),
    }
    
    