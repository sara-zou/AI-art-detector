import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


def get_data_loaders(batch_size=32, max_samples=None, num_workers=4, img_size=224):

    train_transform = transforms.Compose([
        transforms.Resize((int(img_size * 1.15), int(img_size * 1.15))),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder('dataset/train', transform=train_transform)
    val_dataset   = datasets.ImageFolder('dataset/val',   transform=val_transform)

    if max_samples:
        train_indices = torch.randperm(len(train_dataset))[:min(max_samples, len(train_dataset))]
        val_indices = torch.randperm(len(val_dataset))[:min(max_samples // 5, len(val_dataset))]
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    print(f"[DataLoader] Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    if hasattr(train_dataset, 'classes'):
        print(f"[DataLoader] Classes: {train_dataset.classes}")
    return train_loader, val_loader

# Visualisation helper
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def imshow(img_tensor, title=None):
    img = img_tensor.cpu() * _STD + _MEAN
    img = img.clamp(0, 1).numpy().transpose(1, 2, 0)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
