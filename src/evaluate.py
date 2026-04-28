import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_auc_score, roc_curve,
)

from dataloading import get_data_loaders


def load_model(checkpoint_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def evaluate(model, loader, device):
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs   = torch.sigmoid(outputs).cpu().squeeze(1)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.numpy())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs >= 0.5).astype(int)
    return all_labels, all_preds, all_probs


def plot_confusion_matrix(labels, preds, save_path=None):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human", "AI"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.set_title("Validation Confusion Matrix", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix : {save_path}")
    plt.show()


def plot_roc_curve(labels, probs, save_path=None):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved ROC curve : {save_path}")
    plt.show()
    return auc


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Evaluate on the full validation set
    _, val_loader = get_data_loaders(
        batch_size   = 32,
        max_samples  = None,
        num_workers  = 4,
        img_size     = 224,
        use_weighted_sampler = False,
    )

    model = load_model("resnet18_best.pth", device)

    labels, preds, probs = evaluate(model, val_loader, device)

    print("\n--- Classification Report ---")
    print(classification_report(labels, preds, target_names=["Human", "AI"]))

    auc = plot_roc_curve(labels, probs, save_path="roc_curve.png")
    print(f"ROC-AUC: {auc:.4f}")

    plot_confusion_matrix(labels, preds, save_path="confusion_matrix.png")


if __name__ == "__main__":
    main()