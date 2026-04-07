from tqdm import tqdm
import torch
from torchvision import models
from dataloading import get_data_loaders
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    batch_size = 16
    _, val_loader = get_data_loaders(batch_size=16, max_samples=1000, num_workers=0) 

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 1) 
    model = model.to(device)

    model.load_state_dict(torch.load("resnet18_best.pth", map_location=device))
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            preds = (outputs > 0).int()  

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = [int(x[0]) for x in all_labels]
    all_preds = [int(x[0]) for x in all_preds]
    
    print("Label counts:", {0: all_labels.count(0), 1: all_labels.count(1)})

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human", "AI"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Validation Confusion Matrix")
    plt.show()

    correct = sum([1 if p == l else 0 for p, l in zip(all_preds, all_labels)])
    accuracy = correct / len(all_labels)
    print(f"Validation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()