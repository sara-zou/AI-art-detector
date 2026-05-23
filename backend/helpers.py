import torch
import io
import torch
import torch.nn as nn
from config import CHECKPOINT_PATH, IMG_SIZE, THRESHOLD, NORM_MEAN, NORM_STD
from torchvision import transforms, models
from PIL import Image

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

device = get_device()

def load_model():
    model = models.resnet18(weights=None)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device).eval()
    return model

def predict(file_bytes: bytes, model: nn.Module, device: torch.device):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        prob_ai = torch.sigmoid(output).item()
    
    label = "AI" if prob_ai >= THRESHOLD else "Human"
    confidence = prob_ai if prob_ai >= THRESHOLD else 1.0 - prob_ai
    
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "prob_ai": round(prob_ai, 4),
    }
    
    