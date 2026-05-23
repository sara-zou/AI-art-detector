import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from threshold import THRESHOLD

CHECKPOINT  = "../resnet18_best.pth"
IMG_SIZE    = 224
CLASS_NAMES = {1: "Human", 0: "AI"}
THRESHOLD

_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(checkpoint_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict_image(image_path: str, model, device) -> dict:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"path": image_path, "error": str(e)}

    tensor = _TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob   = torch.sigmoid(output).item()  

    label      = CLASS_NAMES[int(prob >= THRESHOLD)]
    confidence = prob if prob >= THRESHOLD else 1.0 - prob

    return {
        "path":       os.path.basename(image_path),
        "label":      label,
        "confidence": confidence,
        "prob_ai":    prob,
    }


def print_result(r: dict):
    if "error" in r:
        print(f"  ERROR  {r['path']}: {r['error']}")
        return
    print(f"label: {r['label']:<6} confidence: {r['confidence']*100:5.1f} prob: {r['prob_ai']} image: {r['path']}")


def batch_predict(paths, model, device):
    for p in paths:
        r = predict_image(p, model, device)
        print_result(r)


def collect_images(path: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        files = []
        for root, _, fnames in os.walk(path):
            for f in fnames:
                if os.path.splitext(f)[1].lower() in exts:
                    files.append(os.path.join(root, f))
        return sorted(files)
    return []


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}\ncheckpoint: {CHECKPOINT}\n")

    model = load_model(CHECKPOINT, device)

    if len(sys.argv) > 1:
        paths = []
        for arg in sys.argv[1:]:
            paths.extend(collect_images(arg))
        if not paths:
            print("No valid image files found.")
            return
        batch_predict(paths, model, device)
        return

    while True:
        user_input = input("\nEnter image path or folder, 'q' to quit: ").strip()
        if user_input.lower() in {"q", "quit", "exit"}:
            break
        paths = collect_images(user_input)
        if not paths:
            print("File/folder not found or no images inside.")
            continue
        batch_predict(paths, model, device)


if __name__ == "__main__":
    main()