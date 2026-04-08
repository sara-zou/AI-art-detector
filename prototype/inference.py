import torch
from torchvision import transforms, models
from PIL import Image
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 1)

model.load_state_dict(torch.load("resnet18_best.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()

    label = "AI" if prob >= 0.5 else "Human"
    confidence = prob if prob >= 0.5 else 1 - prob

    print(f"{os.path.basename(image_path)} → {label} ({confidence:.2f})")
    return label, confidence

def get_image():
    path = input("Enter image path: ")
    if not os.path.exists(path):
        print("File not found!")
        return
    predict_image(path)


def main():
    print("\nAI Art Detector Inference")
    get_image()

if __name__ == "__main__":
    main()