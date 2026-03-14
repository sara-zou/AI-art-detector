import torch
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

transform = transforms.Compose([
    transforms.Resize((224,224)),   # ResNet expects 224x224
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=transform
)

model = resnet18(pretrained=True)

# get data from datatset folder
def get_data(batch_size=64):
    train_datatset = datasets.ImageFolder(
        root='dataset',
        train=True,
        download=True,
        transform=ToTensor(),)
    
    valid_datatset = datasets.ImageFolder(
        root='dataset',
        train=False,
        download=True,
        transform=ToTensor(),)
    
    #data loaders
    train_loader = DataLoader(
        train_datatset,
        batch_size=batch_size,
        shuffle=True
    )
    
    valid_loader = DataLoader(
        train_datatset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return train_loader, valid_loader
    