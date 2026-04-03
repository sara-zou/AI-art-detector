import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def get_data_loaders(batch_size=8, max_samples=None, num_workers=4):
    transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),     
    transforms.Normalize(          
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder('../dataset/train', transform=transform)
    val_dataset = datasets.ImageFolder('../dataset/val', transform=transform)
    if max_samples:
        train_dataset = Subset(train_dataset, range(min(max_samples, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(max_samples, len(val_dataset))))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, 
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, 
                            num_workers=num_workers)

    return train_loader, val_loader

#visualize sample batches
def imshow(img):
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()