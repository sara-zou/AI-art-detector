import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataloading import get_data_loaders 
import torchvision.models as models

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

batch_size = 16
max_samples = 500  
train_loader, val_loader = get_data_loaders(batch_size=batch_size, max_samples=max_samples)

num_classes = 1
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=1e-4)
model = model.to(device)

num_epochs = 5
best_val_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device).float()
        labels = labels.unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss=loss.item())
        

    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float()
            labels = labels.unsqueeze(1)
            outputs = model(images)
            predicted = (outputs > 0).int() 
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1} Validation Accuracy: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "resnet18_best.pth")
        print(f"Saved new best model with accuracy: {best_val_acc:.4f}")


print(f"\nBest Validation Accuracy Achieved: {best_val_acc:.4f}")